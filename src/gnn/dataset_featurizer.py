import concurrent.futures
import multiprocessing
import pandas as pd
import torch
from torch_geometric.data import Dataset
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
from deepchem.feat.graph_data import GraphData
from model_architecture.tree_building import DGLMolTree
import numpy as np
import time, threading, hashlib

def _unique_name(prefix: str, smiles: str, index: int) -> str:
    # IMPORTANTE: l'hash del contenuto (smiles) e' parte del nome, non solo l'indice.
    # Senza l'hash, due dataset diversi con lo stesso flag test=True (es. il validation set ZINC
    # usato in train.py e il dataset TCM usato in predict.py) generano nomi file IDENTICI per lo
    # stesso indice (es. data_test_0.pt) pur contenendo molecole completamente diverse: il primo
    # dataset processato "vince" la cache, e il secondo la riusa in silenzio credendo siano gia'
    # le sue molecole. Con l'hash nel nome, il file combacia (ed e' quindi riusato) solo se
    # indice E contenuto coincidono entrambi con quanto gia' salvato.
    h = hashlib.sha1(str(smiles).encode('utf-8')).hexdigest()[:10]
    return f'{prefix}_{index:08d}_{h}.pt'

def _default_max_workers():
    """Numero di worker per la featurizzazione. Sovrascrivibile con la var d'ambiente
    MOLDATASET_MAX_WORKERS (utile su un server condiviso, per non prendersi tutti i core)."""
    env = os.environ.get("MOLDATASET_MAX_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, multiprocessing.cpu_count() // 2)

def _atomic_torch_save(obj, final_path: str, retries: int = 3):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = f"{final_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    last = None
    for i in range(retries):
        try:
            with open(tmp_path, "wb") as f:
                torch.save(obj, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)  # atomico su POSIX
            return
        except Exception as e:
            last = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass
            time.sleep(0.2 * (i + 1))
    raise RuntimeError(f"atomic save failed for {final_path}: {last}")

def crea_features_albero(smiles, raw_x, raw_edge_attr, raw_edge_index):
    if "." in smiles:
        print("Smiles code not valid")
        return None, None
    tree = DGLMolTree(smiles)

    tree.add_mol_edge_features(raw_edge_index, raw_edge_attr)
    cliques = []
    for clique_idx, clique in tree.nodes_dict.items():
        clique_ids = clique['clique']
        clique_sum = torch.mean(raw_x[clique_ids], dim=0)
        cliques.append(clique_sum)

    if not cliques:
        print(f"No clique extracted from smiles: {smiles}")

    tree_data = torch.stack(cliques, dim=0)
    tree_edge_index = tree.get_edge_index()
    tree_edge_attr = tree.get_edge_attr()

    # If there are no arcs, return empty arcs.
    if tree_edge_index.size(0) == 0 or tree_edge_index.size(1) == 0:
        tree_features = {
            'node_features': tree_data,
            'edge_index': torch.empty([2, 0]),
            'edge_attr': torch.empty([0, 11]),
        }
    else:
        tree_features = {
            'node_features': tree_data,
            'edge_index': tree_edge_index,
            'edge_attr': tree_edge_attr,
        }

    return tree_features, tree


def _build_data_and_tree(smiles, drug_label, final):
    """Costruisce il Data della molecola e il Data dell'albero per una riga. Nessuno stato condiviso: sicura da eseguire in un processo worker separato."""
    mol = Chem.MolFromSmiles(smiles)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    f = featurizer._featurize(mol)
    data = f.to_pyg_graph()
    if not final:
        data.y = torch.tensor([drug_label], dtype=torch.int64)
    data.smiles = smiles

    tt, tree = crea_features_albero(smiles, data.x, data.edge_attr, data.edge_index)

    tree_graph = ExtendedGraphData(
        node_features=np.asarray(tt['node_features']),
        edge_index=np.asarray(tt['edge_index'], dtype=int),
        edge_features=np.asarray(tt['edge_attr'], dtype=float),
        mol_tree=tree
    ).to_pyg_graph()

    return data, tree_graph


def _process_one_molecule(index, smiles, drug_label, test, final, processed_dir):
    """Funzione top-level (non un metodo) cosi' puo' essere eseguita in un ProcessPoolExecutor senza serializzare l'intero MoleculeDataset ad ogni chiamata."""
    # If the SMILES contains a dot, it is invalid
    if "." in str(smiles):
        return

    data_filename = _unique_name('data_test' if test else 'data', smiles, index)
    tree_filename = _unique_name('tree_data_test' if test else 'tree_data', smiles, index)
    data_path = os.path.join(processed_dir, data_filename)
    tree_path = os.path.join(processed_dir, tree_filename)

    # Resume: se questa molecola e' gia' stata processata in una run precedente, saltala.
    if os.path.exists(data_path) and os.path.exists(tree_path):
        return

    try:
        data, tree_graph = _build_data_and_tree(smiles, drug_label, final)
    except Exception as e:
        print(f"smiles: {smiles}")
        raise RuntimeError(f"Errore nella featurizzazione della molecola {index} ({smiles}): {e}") from e

    _atomic_torch_save(data, data_path)
    _atomic_torch_save(tree_graph, tree_path)


class ExtendedGraphData(GraphData):
    def __init__(self, node_features, edge_index, edge_features, mol_tree):
        super().__init__(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features
        )
        self.mol_tree = mol_tree  # Add the tree as a property

    def to_pyg_graph(self):
        # Convert the graph to a format that PyTorch Geometric can use.
        data = super().to_pyg_graph()
        data.mol_tree = self.mol_tree  # Include the tree in the PyG graph.
        return data


class MoleculeDataset(Dataset):
    def __init__(self, filename, test=False,final = False, transform=None, pre_transform=None, root = '../results'):
        self.test = test
        self.root= root
        self.filename = os.path.join('../', filename)
        self.final = final
        super(MoleculeDataset, self).__init__(self.root, transform, pre_transform)
        

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.filename).reset_index()
        # Niente iterrows() (lento su dataset grandi): stessa estrazione vettoriale usata in process().
        smiles_list = self.data["smiles"].tolist()
        prefix_data = 'data_test' if self.test else 'data'
        prefix_tree = 'tree_data_test' if self.test else 'tree_data'
        return [_unique_name(prefix_data, s, i) for i, s in enumerate(smiles_list)] + \
               [_unique_name(prefix_tree, s, i) for i, s in enumerate(smiles_list)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.filename).reset_index()

        # 🔽 1️⃣ Filtra le molecole con "." nei smiles
        bad_smiles_mask = self.data["smiles"].astype(str).str.contains(r"\.")
        if bad_smiles_mask.any():
            print("[MoleculeDataset] Rimuovo SMILES multi-frammento:")
            print(self.data.loc[bad_smiles_mask, ["smiles", "ZINCID"]])  # o la colonna identificativa
            # salva anche un log, se vuoi
            self.data.loc[bad_smiles_mask, ["smiles", "ZINCID"]].to_csv(
                os.path.join(self.root, "removed_multifragment_smiles.csv"), index=False
            )

        self.data = self.data[~bad_smiles_mask].reset_index(drop=True)
        print(f"[MoleculeDataset] Molecole valide: {len(self.data)}")
        
        # Parallelization with ProcessPoolExecutor: il lavoro per molecola (RDKit + costruzione
        # albero) e' CPU-bound e in gran parte puro Python, quindi con i thread (GIL) non si
        # ottiene parallelismo reale. Con i processi si usano davvero piu' core.
        # Si usa una funzione top-level (non un metodo) per non dover serializzare l'intero
        # MoleculeDataset (incluso il DataFrame) ad ogni singola chiamata.
        # IMPORTANTE: mp_context="spawn" invece del default "fork". A questo punto il processo
        # padre ha gia' importato torch/tensorflow/jax/CUDA, che aprono thread interni; fare
        # fork() di un processo multi-thread puo' lasciare un lock ereditato per sempre bloccato
        # nel figlio (deadlock silenzioso, 0% CPU per sempre). "spawn" fa partire i figli puliti.
        #
        # NON si usa self.data.iterrows(): e' uno dei modi piu' lenti di iterare un DataFrame
        # (costruisce una Series per riga, con coercizione di tipo) - su 200k+ righe puo' costare
        # da solo diversi minuti. Si estraggono invece le due colonne che servono come liste
        # Python semplici (estrazione vettoriale, praticamente istantanea anche su 200k righe).
        smiles_list = self.data["smiles"].tolist()
        drug_list = [None] * len(smiles_list) if self.final else self.data["Drug"].tolist()

        ctx = multiprocessing.get_context("spawn")
        n_items = len(smiles_list)
        n_workers = _default_max_workers()

        # IMPORTANTE: non sottomettere tutti i 204500 task in un colpo solo. La coda interna di
        # ProcessPoolExecutor accetta solo ~n_workers+1 task prima di riempirsi; se i worker sono
        # lenti a diventare pronti (devono reimportare torch/deepchem/tensorflow), il thread che
        # gestisce la coda si blocca a meta' e non riesce piu' a "drenare" la pipe di notifica
        # interna -> il thread principale si blocca per sempre dentro submit() (visto nel
        # traceback del Ctrl+C: bloccato in send_bytes). Si tiene invece una "finestra" di task in
        # volo, sottomettendone di nuovi solo man mano che i precedenti completano.
        items_iter = zip(self.data.index, smiles_list, drug_list)
        max_in_flight = n_workers * 4

        def _submit_next(executor, pending):
            try:
                idx, smiles, drug = next(items_iter)
            except StopIteration:
                return
            pending.add(executor.submit(
                _process_one_molecule, idx, smiles, drug, self.test, self.final, self.processed_dir
            ))

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            pending = set()
            for _ in range(min(max_in_flight, n_items)):
                _submit_next(executor, pending)

            with tqdm(total=n_items) as pbar:
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        fut.result()  # rilancia eventuali eccezioni dal worker
                        pbar.update(1)
                        _submit_next(executor, pending)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        # usa gli stessi nomi della fase di salvataggio
        data_filename = _unique_name('data_test' if self.test else 'data',
                                    self.data.loc[idx, "smiles"], idx)
        tree_filename = _unique_name('tree_data_test' if self.test else 'tree_data',
                                    self.data.loc[idx, "smiles"], idx)

        molecule_data = torch.load(os.path.join(self.processed_dir, data_filename))
        tree_data     = torch.load(os.path.join(self.processed_dir, tree_filename))
        return molecule_data, tree_data
