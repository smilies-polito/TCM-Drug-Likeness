"""Costruisce la fingerprint a valle dai frammenti selezionati, e ne riporta la forma.

Usa `crea_fingerprint` del repository senza modificarla: la selezione dei frammenti cambia,
la costruzione della fingerprint no. Serve solo a eseguirla sul nuovo file di frammenti e a
stampare le grandezze che contano per il passo successivo (clustering): quante molecole,
quante colonne, quanto e' sparsa.

La dimensionalita' non e' un dettaglio: i parametri di UMAP e HDBSCAN del paper sono stati
scelti su 634 colonne, e una fingerprint molto piu' larga va ri-tarata.

Uso:

    cd src
    python3 explain/downstream_fingerprint.py \\
        --frags ../results/explain/frags_thr005_merged.csv \\
        --out ../results/explain/fingerprint_thr005.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "analysis"))

import explain_core as core  # noqa: E402


def describe(fingerprint_path):
    """Forma della fingerprint: righe, colonne, densita'."""
    frame = pd.read_csv(fingerprint_path)
    columns = [c for c in frame.columns if c != "id_molecola"]
    values = frame[columns].to_numpy()
    nonzero = int((values > 0).sum())
    cells = values.size
    counts_per_molecule = (values > 0).sum(axis=1)
    occurrences = int(values.sum())
    return {
        "molecules": len(frame),
        "fragment_columns": len(columns),
        "nonzero_cells": nonzero,
        "density": nonzero / cells if cells else 0.0,
        "occurrences": occurrences,
        "median_fragments_per_molecule": float(pd.Series(counts_per_molecule).median()),
        "columns_in_one_molecule_only": int((( values > 0).sum(axis=0) == 1).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frags", required=True, help="file dei frammenti selezionati")
    parser.add_argument("--out", required=True, help="CSV della fingerprint")
    args = parser.parse_args(argv)

    core.ensure_repo_on_path()
    from post_proc_pipeline import crea_fingerprint

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    crea_fingerprint(args.frags, args.out)

    shape = describe(args.out)
    print(f"\n--- forma della fingerprint ({os.path.basename(args.out)}) ---")
    for key, value in shape.items():
        print(f"  {key:34s} {value:,.4f}" if isinstance(value, float)
              else f"  {key:34s} {value:,}")
    print("  (riferimento pubblicato: 9.695 molecole, 634 colonne, 113.634 occorrenze)")
    return shape


if __name__ == "__main__":
    main()
