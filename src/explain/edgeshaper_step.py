#!/usr/bin/env python3
"""Passo di spiegazione della pipeline: da modello addestrato a frammenti selezionati.

Mette insieme i tre pezzi del percorso EdgeSHAPer in un'unica chiamata, cosi' la pipeline
principale ne vede uno solo:

  1. `edgeshaper_run.attribute` assegna un valore di Shapley a ogni coppia di clique adiacenti,
     usando il pacchetto degli autori;
  2. a ogni arco si associa lo SMILES del frammento che forma (l'unione delle due clique), il
     numero di vicini e la predizione della molecola;
  3. i frammenti sopra soglia finiscono in un CSV con le colonne che `crea_fingerprint`
     pretende a valle: smiles, num_neigh, id_molecola, pred.

L'uscita ha lo stesso formato del file di frammenti che `predict_drug` produceva dai pesi di
attenzione, quindi il resto della pipeline (fingerprint, UMAP, HDBSCAN, medoidi) non cambia.

Uso come modulo:

    from edgeshaper_step import run_edgeshaper_step
    frags_path = run_edgeshaper_step(csv, pred_path, out_dir)

Uso da riga di comando:

    cd src
    python3 explain/edgeshaper_step.py --pred-path ../results/TMMC_molecule_predictions.csv
"""
import argparse
import os
import sys

import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import explain_core as core  # noqa: E402
import edgeshaper_run as esr  # noqa: E402

DEFAULT_THRESHOLD = 0.005


def _fragment_smiles(tree_data, clique_a, clique_b):
    """SMILES del frammento formato dalle due clique, con la funzione della pipeline."""
    from rdkit import Chem
    from model_architecture.unify_high_mol import unify_fragments
    submol, _ = unify_fragments(tree_data.mol_tree, [clique_a, clique_b], deep_search=False)
    if submol is None:
        return None
    try:
        Chem.SanitizeMol(
            submol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        return None
    return Chem.MolToSmiles(submol)


def annotate(frame, dataset, predictions):
    """Aggiunge a ogni arco lo SMILES del frammento, num_neigh e la predizione della molecola."""
    preds = dict(zip(predictions["id_molecola"], predictions["pred"])) \
        if "id_molecola" in predictions.columns else dict(enumerate(predictions["pred"]))
    smiles, neigh, pred = [], [], []
    for mol_index, group in frame.groupby("mol_index", sort=False):
        _, tree_data = dataset[int(mol_index)]
        degrees = core.clique_degrees(tree_data)
        for row in group.itertuples(index=False):
            smiles.append(_fragment_smiles(tree_data, int(row.clique_a), int(row.clique_b)))
            neigh.append(max((degrees[int(row.clique_a)] - 1)
                             + (degrees[int(row.clique_b)] - 1), 0))
            pred.append(preds.get(int(mol_index)))
    out = frame.copy()
    out["smiles"], out["num_neigh"], out["pred"] = smiles, neigh, pred
    missing = out["smiles"].isna().sum()
    if missing:
        print(f"[edgeshaper] {missing} archi senza SMILES ricostruibile: scartati")
        out = out[out["smiles"].notna()].reset_index(drop=True)
    return out


def run_edgeshaper_step(csv, pred_path, out_dir, threshold=DEFAULT_THRESHOLD,
                        samples=esr.DEFAULT_SAMPLES, device=None, seed=42, limit=None):
    """Esegue l'attribuzione e scrive i frammenti selezionati. Restituisce il path del CSV."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    explain_dir = os.path.join(out_dir, "explain")
    os.makedirs(explain_dir, exist_ok=True)

    core.ensure_repo_on_path()
    model = core.load_published_model(device=device)
    dataset = core.load_dataset(csv, test=True, final=True)
    stop = len(dataset) if limit is None else min(limit, len(dataset))

    print(f"[edgeshaper] attribuzione su {stop} molecole, M={samples}")
    edges, skipped = esr.attribute(model, dataset, range(stop), samples=samples,
                                   device=device, seed=seed)
    edges_path = os.path.join(explain_dir, "edgeshaper_edges.csv")

    predictions = pd.read_csv(pred_path)
    edges = annotate(edges, dataset, predictions)
    edges.to_csv(edges_path, index=False)
    print(f"[edgeshaper] {len(edges)} archi su {edges.mol_index.nunique()} molecole "
          f"(saltate {skipped}) -> {edges_path}")

    frags = core.export_fragments(edges.rename(columns={"edge_shapley": "delta_prob"}),
                                  threshold)
    frags_path = os.path.join(explain_dir, "edgeshaper_fragments.csv")
    frags.to_csv(frags_path, index=False)
    drug = frags[frags["pred"] == 1]
    print(f"[edgeshaper] soglia {threshold:g}: {len(frags):,} frammenti, di cui {len(drug):,} "
          f"su molecole predette drug-like ({drug['smiles'].nunique():,} tipi unici) "
          f"-> {frags_path}")
    return frags_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/tm-mc2/preprocessed_TMMC_compounds.csv")
    parser.add_argument("--pred-path", required=True,
                        help="CSV delle predizioni prodotto da predict_drug")
    parser.add_argument("--out-dir", default="../results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--samples", type=int, default=esr.DEFAULT_SAMPLES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    run_edgeshaper_step(args.csv, args.pred_path, args.out_dir, threshold=args.threshold,
                        samples=args.samples, device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
