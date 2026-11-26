"""Infrastruttura comune per l'importanza post-hoc dei frammenti sul modello pubblicato.

Questo modulo NON definisce un criterio di importanza: fornisce solo i pezzi che tutti i
metodi (occlusione, Integrated Gradients, Shapley) condividono:

  * caricamento del modello pubblicato e dei suoi parametri da MLflow;
  * mappa clique -> atomi, cioe' il legame fra i nodi dell'albero e gli atomi della molecola;
  * costruzione di un batch in cui la stessa molecola e' replicata k volte, ciascuna con una
    maschera diversa, e un unico passaggio in avanti che restituisce i k logit.

Perche' replicare invece di fare k passaggi separati: il modello e' piccolo (636k parametri) e
il costo e' dominato dall'overhead per chiamata, quindi valutare tutte le varianti di una
molecola in un solo batch e' molto piu' veloce.

ATTENZIONE (vincolo sul batch): in `myModel.forward` la LayerNorm e' applicata solo se
`x.size(0) > 1`, dove x ha una riga per grafo. Un batch con un solo grafo salta quindi la
LayerNorm e produce un logit diverso da quello della pipeline pubblicata (misurato: -0.44
contro -0.62 sulla stessa molecola). Tutte le funzioni qui sotto garantiscono quindi almeno
due grafi per batch: vedi `MIN_GRAPHS_PER_BATCH`.
"""
import os
import sys

import torch

# Il modello pubblicato nel paper (run 64582bc2 dell'esperimento MLflow in results/trained_model).
DEFAULT_EXPERIMENT_ID = "418303031868899849"
DEFAULT_RUN_ID = "64582bc220f74718892ff3eb63ba31a0"

# Il forward del modello salta la LayerNorm quando il batch ha un solo grafo: mai scendere
# sotto questa soglia, altrimenti i logit non sono quelli del modello pubblicato.
MIN_GRAPHS_PER_BATCH = 2


def repo_root():
    """Radice del repository (questo file sta in src/explain/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_repo_on_path():
    """Rende importabili i moduli di src/gnn e porta il cwd in src/.

    Il codice del repo usa path relativi del tipo '../results' e import piatti
    (`from dataset_featurizer import ...`), quindi assume di essere eseguito con cwd == src/.
    Lo stesso vale per il modello: e' un oggetto pickled la cui classe sta in
    src/gnn/model_architecture/, e torch.load la cerca per nome di modulo.
    """
    src_dir = os.path.join(repo_root(), "src")
    gnn_dir = os.path.join(src_dir, "gnn")
    for path in (gnn_dir, src_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    if os.path.abspath(os.getcwd()) != os.path.abspath(src_dir):
        os.chdir(src_dir)


def checkpoint_path(experiment_id=DEFAULT_EXPERIMENT_ID, run_id=DEFAULT_RUN_ID):
    """Percorso del .pth del modello, con le stesse convenzioni di predict.py (benchmark=True)."""
    return os.path.join(repo_root(), "results", "trained_model", "mlartifacts",
                        experiment_id, run_id, "artifacts", "model", "data", "model.pth")


def load_run_params(experiment_id=DEFAULT_EXPERIMENT_ID, run_id=DEFAULT_RUN_ID):
    """Iperparametri della run, letti dalla cartella params/ di MLflow (valori come stringhe)."""
    params_dir = os.path.join(repo_root(), "results", "trained_model", "mlruns",
                              experiment_id, run_id, "params")
    params = {}
    for name in os.listdir(params_dir):
        with open(os.path.join(params_dir, name)) as handle:
            params[os.path.splitext(name)[0]] = handle.read().strip()
    return params


def load_published_model(experiment_id=DEFAULT_EXPERIMENT_ID, run_id=DEFAULT_RUN_ID,
                         device="cpu"):
    """Carica il modello pubblicato in modalita' eval, senza modificarne i pesi.

    eval() e' indispensabile: in training il modello ha dropout 0.29 dentro ogni
    TransformerConv e 0.8 sull'uscita degli encoder, quindi le predizioni sarebbero casuali.
    """
    ensure_repo_on_path()
    model = torch.load(checkpoint_path(experiment_id, run_id), map_location=device)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def subset_csv(csv_relpath, limit, out_dir=None):
    """Scrive un CSV con le prime `limit` molecole e ne restituisce il path relativo al repo.

    Serve perche' MoleculeDataset featurizza TUTTE le molecole del CSV quando viene costruito,
    prima che il chiamante possa limitarsi alle prime N: passare il file completo di TM-MC per
    poi usarne 20 significherebbe featurizzare 20.974 molecole.

    Si prende un PREFISSO, non un campione casuale: cosi' l'indice di una molecola nel
    sottoinsieme coincide con il suo indice nel dataset completo, e i risultati ottenuti su un
    sottoinsieme restano confrontabili con quelli della run completa.
    """
    import pandas as pd

    source_abs = os.path.join(repo_root(), csv_relpath)
    if out_dir is None:
        out_dir = os.path.join(repo_root(), "results", "explain", "subsets")
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_relpath))[0]
    out_abs = os.path.join(out_dir, f"{base}_first{limit}.csv")
    pd.read_csv(source_abs).head(limit).to_csv(out_abs, index=False)
    return os.path.relpath(out_abs, repo_root())


def load_dataset(csv_relpath, test=True, final=True, limit=None):
    """Costruisce il MoleculeDataset del repo per un CSV di molecole.

    `csv_relpath` e' relativo alla radice del repo (es. 'data/tm-mc2/preprocessed_TMMC_compounds.csv'),
    come si aspetta MoleculeDataset, che gli antepone '../'.
    `final=True` significa dataset senza colonna 'Drug' (solo predizioni, come i composti TCM).
    `limit` ritaglia il CSV alle prime N molecole PRIMA di featurizzare (vedi `subset_csv`).

    Nota: la featurizzazione usa un ProcessPoolExecutor con mp_context='spawn', quindi chi
    chiama questa funzione da uno script deve stare dentro `if __name__ == "__main__":`.
    """
    ensure_repo_on_path()
    from dataset_featurizer import MoleculeDataset

    if limit is not None:
        csv_relpath = subset_csv(csv_relpath, limit)
    return MoleculeDataset(filename=csv_relpath, test=test, final=final)


def clique_atoms(tree_data):
    """Lista degli atomi di ogni clique, indicizzata come i nodi dell'albero.

    L'allineamento con le feature dei nodi non e' casuale: `crea_features_albero` costruisce la
    riga i di tree_data.x come media di raw_x sugli atomi di `nodes_dict[i]['clique']`, nello
    stesso ordine di iterazione usato qui. Lo scambio radice/nodo 0 fatto da DGLMolTree agisce
    su nodes_dict stesso, quindi la corrispondenza resta valida.
    """
    nodes_dict = tree_data.mol_tree.nodes_dict
    return [list(nodes_dict[i]["clique"]) for i in range(len(nodes_dict))]


def clique_smiles(tree_data):
    """SMILES di ogni clique, cosi' come calcolato dalla decomposizione ad albero."""
    nodes_dict = tree_data.mol_tree.nodes_dict
    return [nodes_dict[i]["smiles"] for i in range(len(nodes_dict))]


def clique_degrees(tree_data):
    """Numero di vicini di ogni nodo dell'albero.

    Serve per sapere quali clique sono foglie: sono quelle che il criterio attuale
    sull'attenzione seleziona sempre, per costruzione.
    """
    n_nodes = tree_data.x.size(0)
    degrees = [0] * n_nodes
    edge_index = tree_data.edge_index
    # Gli archi dell'albero sono salvati nei due versi, quindi contare le sole sorgenti
    # da' esattamente il grado di ciascun nodo.
    for src in edge_index[0].tolist():
        degrees[src] += 1
    return degrees


def recompute_tree_features(raw_x, atoms_per_clique):
    """Ricalcola le feature dei nodi dell'albero come medie delle feature atomiche.

    E' la stessa formula della featurizzazione (`crea_features_albero`): la riga di una clique
    e' la media delle righe dei suoi atomi. Usata quando si vuole che una maschera sul grafo
    degli atomi si propaghi all'albero, invece di mascherare i due grafi in modo indipendente.
    """
    rows = [raw_x[atoms].mean(dim=0) for atoms in atoms_per_clique]
    return torch.stack(rows, dim=0)


def clique_mean_matrix(atoms_per_clique, n_atoms, device="cpu", dtype=torch.float32):
    """Matrice M tale che tree_x = M @ raw_x: la riga i media le feature degli atomi della clique i.

    E' la stessa operazione di `recompute_tree_features`, scritta una volta per molecola come
    matrice. Serve perche' ogni variante mascherata richiede di ricalcolare le feature
    dell'albero: farlo con un prodotto fra matrici invece che con un ciclo Python sulle clique
    e' cio' che rende praticabile il calcolo su tutto il dataset.
    """
    matrix = torch.zeros((len(atoms_per_clique), n_atoms), device=device, dtype=dtype)
    for clique_id, atoms in enumerate(atoms_per_clique):
        matrix[clique_id, atoms] = 1.0 / len(atoms)
    return matrix


def tree_features_from_matrix(raw_x_batched, mean_matrix):
    """Feature dell'albero per un blocco di varianti: (k, n_atoms, F) -> (k * n_cliques, F).

    Differenziabile, quindi usabile anche dentro Integrated Gradients.
    """
    stacked = torch.matmul(mean_matrix, raw_x_batched)
    return stacked.reshape(-1, stacked.size(-1))


def _replicate_graph(x, edge_index, edge_attr, n_copies):
    """Replica un grafo n_copies volte in un unico grafo disgiunto (come fa il collate di PyG).

    Gli indici degli archi di ogni copia sono traslati di j * n_nodes, cosi' le copie non si
    collegano fra loro; `batch` assegna a ogni nodo la copia a cui appartiene.
    """
    n_nodes = x.size(0)
    x_rep = x.repeat(n_copies, 1)
    edge_attr_rep = edge_attr.repeat(n_copies, 1)
    offsets = (torch.arange(n_copies, device=edge_index.device) * n_nodes).repeat_interleave(
        edge_index.size(1))
    edge_index_rep = edge_index.repeat(1, n_copies) + offsets.unsqueeze(0)
    batch = torch.arange(n_copies, device=x.device).repeat_interleave(n_nodes)
    return x_rep, edge_index_rep, edge_attr_rep, batch


def build_masked_batch(raw_x, raw_edge_index, raw_edge_attr,
                       tree_x, tree_edge_index, tree_edge_attr,
                       atom_masks, tree_masks, baseline_atom=None, baseline_tree=None):
    """Costruisce il dizionario di input del modello per k varianti mascherate della molecola.

    Args:
        raw_*: grafo degli atomi della molecola (una sola copia).
        tree_*: albero dei frammenti della molecola (una sola copia).
        atom_masks: tensore bool (k, n_atoms). True = atomo mascherato in quella variante.
        tree_masks: tensore bool (k, n_cliques). True = clique mascherata in quella variante.
        baseline_atom/baseline_tree: valore con cui si sostituiscono le feature mascherate
            (default: zeri, cioe' l'occlusione classica).

    La variante 0 va costruita dal chiamante senza maschere, cosi' il primo logit restituito e'
    la predizione intatta e tutte le differenze sono calcolate nello stesso passaggio.
    """
    n_copies = atom_masks.size(0)
    if n_copies < MIN_GRAPHS_PER_BATCH:
        raise ValueError(
            f"Il batch deve contenere almeno {MIN_GRAPHS_PER_BATCH} grafi: con uno solo il "
            "modello salta la LayerNorm e i logit non sono quelli del modello pubblicato.")

    x_rep, ei_rep, ea_rep, batch = _replicate_graph(
        raw_x, raw_edge_index, raw_edge_attr, n_copies)
    tx_rep, tei_rep, tea_rep, tree_batch = _replicate_graph(
        tree_x, tree_edge_index, tree_edge_attr, n_copies)

    if baseline_atom is None:
        baseline_atom = torch.zeros_like(raw_x[0])
    if baseline_tree is None:
        baseline_tree = torch.zeros_like(tree_x[0])

    x_rep = x_rep.clone()
    tx_rep = tx_rep.clone()
    x_rep[atom_masks.reshape(-1)] = baseline_atom
    tx_rep[tree_masks.reshape(-1)] = baseline_tree

    return {
        "mol_raws": {
            "x": x_rep.float(),
            "edge_attr": ea_rep.float(),
            "edge_index": ei_rep,
            "batch": batch,
        },
        "mol_trees": {
            "tree_node_features": tx_rep.float(),
            "tree_edge_attr": tea_rep.float(),
            "tree_edge_index": tei_rep,
            "tree_batch_index": tree_batch,
        },
    }


@torch.no_grad()
def forward_logits(model, data, device="cpu"):
    """Logit (uno per grafo del batch) del modello sul dizionario di input dato."""
    logits = model(data, device)[0]
    return logits.reshape(-1)


@torch.no_grad()
def forward_with_attention(model, data, device="cpu"):
    """Logit piu' archi e pesi di attenzione dei due encoder.

    Il modello restituisce
    (y, edge_index_raw, attention_raw, edge_index_tree, attention_tree); l'attenzione e' quella
    del primo TransformerConv di ogni encoder, mediata sulle teste, cioe' esattamente il
    punteggio che la pipeline pubblicata usa per scegliere i frammenti.
    """
    logits, raw_edge_index, raw_attention, tree_edge_index, tree_attention = model(data, device)
    return {
        "logits": logits.reshape(-1),
        "raw_edge_index": raw_edge_index,
        "raw_attention": raw_attention,
        "tree_edge_index": tree_edge_index,
        "tree_attention": tree_attention,
    }


def logits_to_probs(logits):
    """Probabilita' della classe Drug.

    Promemoria dal report: con pos_weight = 0.5 queste probabilita' non sono calibrate
    (p = 0.5 corrisponde a una probabilita' vera di circa 2/3). Per questo l'importanza va
    riportata anche come differenza di logit, che non satura.
    """
    return torch.sigmoid(logits)


@torch.no_grad()
def tree_attention_per_clique(model, data, tree_data, device="cpu"):
    """Punteggio per clique secondo il criterio PUBBLICATO, usato come termine di confronto.

    Nel codice del paper un frammento nasce da un arco dell'albero con alpha > 0.80, e il
    frammento e' l'unione delle due clique agli estremi. Qui si riporta lo stesso criterio su
    una scala per clique: il punteggio di una clique e' il massimo alpha fra gli archi che la
    toccano. Secondo il report le foglie prendono sempre 1.00, perche' la softmax su un solo
    vicino vale 1 per costruzione.

    La molecola e' replicata due volte perche' con un solo grafo il modello salta la LayerNorm
    (vedi MIN_GRAPHS_PER_BATCH); si leggono poi solo gli archi della prima copia.
    """
    n_cliques = tree_data.x.size(0)
    atom_masks = torch.zeros((MIN_GRAPHS_PER_BATCH, data.x.size(0)), dtype=torch.bool,
                             device=device)
    tree_masks = torch.zeros((MIN_GRAPHS_PER_BATCH, n_cliques), dtype=torch.bool, device=device)
    batch = build_masked_batch(
        data.x.float().to(device), data.edge_index.to(device), data.edge_attr.float().to(device),
        tree_data.x.float().to(device), tree_data.edge_index.to(device),
        tree_data.edge_attr.float().to(device), atom_masks, tree_masks)

    _, _, _, tree_edge_index, tree_alpha = model(batch, device)

    scores = torch.zeros(n_cliques, device=device)
    src = tree_edge_index[0]
    dst = tree_edge_index[1]
    # Solo gli archi della prima copia della molecola, e nessun self-loop (come fa
    # highlight_mol_per_graph, che scarta i casi start == end).
    keep = (src < n_cliques) & (dst < n_cliques) & (src != dst)
    for edge in keep.nonzero(as_tuple=True)[0].tolist():
        alpha = tree_alpha[edge].reshape(-1).mean()
        scores[src[edge]] = torch.maximum(scores[src[edge]], alpha)
        scores[dst[edge]] = torch.maximum(scores[dst[edge]], alpha)
    return scores


# ---------------------------------------------------------------------------
# Utilita' spostate qui quando il resto dei metodi di attribuzione e' stato
# archiviato: servono al percorso EdgeSHAPer e non hanno piu' una casa altrove.
# ---------------------------------------------------------------------------

def tree_pairs(tree_data):
    """Coppie di clique collegate da un arco dell'albero, ciascuna una sola volta.

    Gli archi sono salvati nei due versi e possono contenere self-loop: si tiene (min, max)
    e si scartano i casi a == b.
    """
    edge_index = tree_data.edge_index
    pairs = set()
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if src == dst:
            continue
        pairs.add((min(src, dst), max(src, dst)))
    return sorted(pairs)


def directed_index(tree_data):
    """Mappa ogni coppia non orientata (a, b) alle colonne di tree_edge_index che la formano."""
    edge_index = tree_data.edge_index
    columns = {}
    for position, (src, dst) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        if src == dst:
            continue
        columns.setdefault((min(src, dst), max(src, dst)), []).append(position)
    return columns


def batch_graphs(graphs, device="cpu"):
    """Impacchetta grafi di dimensioni DIVERSE in un unico batch disgiunto."""
    import torch
    xs, eis, eas, bs = [], [], [], []
    txs, teis, teas, tbs = [], [], [], []
    atom_offset = clique_offset = 0
    for index, g in enumerate(graphs):
        xs.append(g["raw_x"])
        eis.append(g["raw_edge_index"] + atom_offset)
        eas.append(g["raw_edge_attr"])
        bs.append(torch.full((g["raw_x"].size(0),), index, dtype=torch.long, device=device))
        txs.append(g["tree_x"])
        teis.append(g["tree_edge_index"] + clique_offset)
        teas.append(g["tree_edge_attr"])
        tbs.append(torch.full((g["tree_x"].size(0),), index, dtype=torch.long, device=device))
        atom_offset += g["raw_x"].size(0)
        clique_offset += g["tree_x"].size(0)
    return {
        "mol_raws": {"x": torch.cat(xs).float(), "edge_attr": torch.cat(eas).float(),
                     "edge_index": torch.cat(eis, dim=1), "batch": torch.cat(bs)},
        "mol_trees": {"tree_node_features": torch.cat(txs).float(),
                      "tree_edge_attr": torch.cat(teas).float(),
                      "tree_edge_index": torch.cat(teis, dim=1),
                      "tree_batch_index": torch.cat(tbs)},
    }


def export_fragments(scores, threshold):
    """Dal file dei punteggi al formato della pipeline a valle.

    Le colonne sono quelle che `crea_fingerprint` pretende: smiles, num_neigh, id_molecola,
    pred. Il filtro sulle molecole predette drug-like resta a valle, come nel resto della
    pipeline.
    """
    frame = scores[scores["delta_prob"] > threshold]
    out = frame[["smiles", "num_neigh", "mol_index", "pred"]].rename(
        columns={"mol_index": "id_molecola"})
    return out.sort_values(["id_molecola", "smiles"]).reset_index(drop=True)
