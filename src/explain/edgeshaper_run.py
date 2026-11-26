#!/usr/bin/env python3
"""Attribuzione dei frammenti con il pacchetto EdgeSHAPer (Mastropietro et al., iScience 2022).

E' il passo di spiegazione della pipeline: prende il modello addestrato e le molecole TM-MC,
assegna un valore di Shapley a ogni arco dell'albero delle clique e scrive i frammenti che
superano la soglia nel formato che `crea_fingerprint` si aspetta a valle.

Il calcolo lo fa il codice degli autori (`edgeshaper.Edgeshaper`), non una reimplementazione.
Qui c'e' solo l'adattatore che mette il nostro modello nella forma che quel codice richiede:

  - EdgeSHAPer chiama `model(x, edge_index, batch=..., edge_weight=...)` e si aspetta logit a
    due colonne, su cui applica softmax e legge `target_class`;
  - il nostro classificatore restituisce UN logit per grafo e la probabilita' e' sigmoid(logit);
  - l'adattatore restituisce [0, logit], la cui softmax vale [1 - sigmoid, sigmoid], quindi con
    target_class=1 la funzione caratteristica di EdgeSHAPer e' esattamente la nostra p(Drug).

Il grafo consegnato a EdgeSHAPer e' l'albero delle clique con UNA colonna per coppia di clique
adiacenti: i giocatori sono quindi i frammenti-coppia, l'unita' che la pipeline esporta. Quando
EdgeSHAPer trattiene una colonna, l'adattatore ripristina entrambi i versi dell'arco, perche' il
passaggio di messaggi del modello e' bidirezionale. I nodi e il grafo atomo-legame non vengono
mai toccati: togliere un arco interrompe il flusso di informazione fra due parti dell'albero
senza inventare una molecola che non esiste.

Uso:

    cd src
    python3 explain/edgeshaper_run.py --out ../results/explain/edgeshaper_edges.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import explain_core as core  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(_HERE))
_VENDOR = os.path.join(_ROOT, "vendor", "EdgeSHAPer", "src")
_PYLIBS = os.path.join(_ROOT, "vendor", "pylibs")

DEFAULT_SAMPLES = 100


def load_edgeshaper():
    """Importa la classe del pacchetto degli autori.

    Nel container EdgeSHAPer sta in /opt/EdgeSHAPer/src ed e' gia' sul PYTHONPATH (vedi
    TCM.def). Fuori dal container si usa il checkout in vendor/, se presente. `vendor/pylibs`
    contiene la sola dipendenza mancante, `rdkit_heatmaps`, e viene aggiunto IN CODA proprio
    per non poter mai oscurare i pacchetti dell'immagine.
    """
    if os.path.isdir(_PYLIBS):
        sys.path.append(_PYLIBS)
    if os.path.isdir(_VENDOR):
        sys.path.insert(0, _VENDOR)
    try:
        from edgeshaper import Edgeshaper
    except ImportError as exc:
        raise SystemExit(
            f"[edgeshaper] pacchetto non trovato ({exc}).\n"
            f"  Nel container: deve stare in /opt/EdgeSHAPer/src (vedi TCM.def).\n"
            f"  Fuori: git clone https://github.com/AndMastro/EdgeSHAPer.git "
            f"{os.path.join(_ROOT, 'vendor', 'EdgeSHAPer')}")
    return Edgeshaper


class TreeEdgeAdapter(nn.Module):
    """Espone il modello con la firma che EdgeSHAPer si aspetta.

    x ed edge_index riguardano SOLO l'albero delle clique: il grafo atomo-legame resta quello
    vero e intatto. Le colonne che EdgeSHAPer passa vengono riconosciute dai loro valori
    (src, dst) e ritradotte nelle colonne reali di `tree_edge_index`, cosi' gli attributi
    d'arco restano allineati senza doverli far transitare da `edge_weight`.
    """

    def __init__(self, model, data, tree_data, device="cpu"):
        super().__init__()
        self.model = model
        self.data = data
        self.tree_data = tree_data
        self.device = device
        self.by_pair = core.directed_index(tree_data)

    def players(self):
        """edge_index da consegnare a EdgeSHAPer: una colonna per coppia adiacente."""
        pairs = core.tree_pairs(self.tree_data)
        index = torch.tensor([[a for a, _ in pairs], [b for _, b in pairs]], dtype=torch.long)
        return index, pairs

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if edge_index.dim() == 1:                      # index_select su una sola colonna
            edge_index = edge_index.reshape(2, -1)
        positions = []
        if edge_index.numel():
            for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                positions.extend(self.by_pair.get((min(src, dst), max(src, dst)), []))
        tree_data = self.tree_data
        if positions:
            tree_edge_index = tree_data.edge_index[:, positions].to(self.device)
            tree_edge_attr = tree_data.edge_attr[positions].float().to(self.device)
        else:
            tree_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            tree_edge_attr = torch.zeros((0, tree_data.edge_attr.size(1)), device=self.device)
        graph = {
            "raw_x": self.data.x.float().to(self.device),
            "raw_edge_index": self.data.edge_index.to(self.device),
            "raw_edge_attr": self.data.edge_attr.float().to(self.device),
            "tree_x": tree_data.x.float().to(self.device),
            "tree_edge_index": tree_edge_index,
            "tree_edge_attr": tree_edge_attr,
        }
        copies = [graph] * core.MIN_GRAPHS_PER_BATCH   # con un grafo solo il modello salta LayerNorm
        logits = core.forward_logits(self.model, core.batch_graphs(copies, self.device),
                                     self.device)[:1].reshape(1, 1)
        # softmax([0, z]) = [1 - sigmoid(z), sigmoid(z)]  ->  colonna 1 = la nostra p(Drug)
        return torch.cat([torch.zeros_like(logits), logits], dim=1)


def attribute(model, dataset, indices, samples=DEFAULT_SAMPLES, device="cpu", seed=42):
    """Valori di Shapley per ogni coppia di clique adiacenti, molecola per molecola."""
    Edgeshaper = load_edgeshaper()
    rows, skipped = [], 0
    for mol_index in tqdm(indices, desc="EdgeSHAPer"):
        data, tree_data = dataset[mol_index]
        try:
            adapter = TreeEdgeAdapter(model, data, tree_data, device)
            edge_index, pairs = adapter.players()
            if len(pairs) < 2:
                continue
            explainer = Edgeshaper(adapter, tree_data.x.float(), edge_index, device=device)
            phi = explainer.explain(M=samples, target_class=1, seed=seed, progress_bar=False)
            degrees = core.clique_degrees(tree_data)
        except Exception as exc:
            print(f"  molecola {mol_index} saltata: {exc}")
            skipped += 1
            continue
        for (clique_a, clique_b), value in zip(pairs, phi):
            rows.append({
                "mol_index": mol_index,
                "clique_a": clique_a,
                "clique_b": clique_b,
                "touches_leaf": degrees[clique_a] <= 1 or degrees[clique_b] <= 1,
                "edge_shapley": float(value),
            })
    return pd.DataFrame(rows), skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/tm-mc2/preprocessed_TMMC_compounds.csv")
    parser.add_argument("--out", default="../results/explain/edgeshaper_edges.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="M di EdgeSHAPer: campioni Monte Carlo per arco")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    core.ensure_repo_on_path()
    model = core.load_published_model(device=args.device)
    dataset = core.load_dataset(args.csv, test=True, final=True)
    stop = len(dataset) if args.limit is None else min(args.start + args.limit, len(dataset))

    frame, skipped = attribute(model, dataset, range(args.start, stop),
                               samples=args.samples, device=args.device, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\n[edgeshaper] {len(frame)} archi su "
          f"{frame.mol_index.nunique() if len(frame) else 0} molecole "
          f"(saltate {skipped}) -> {args.out}")
    return frame


if __name__ == "__main__":
    main()
