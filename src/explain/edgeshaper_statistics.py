#!/usr/bin/env python3
"""Statistiche citate nel manoscritto, ricalcolate sull'attribuzione EdgeSHAPer.

Riusa le funzioni di `paper/manuscript_statistics.py` senza modificarle: le definizioni di
classe chimica, coerenza delle famiglie e qualita' del clustering restano quelle del paper, e
cambia solo l'attribuzione da cui partono. Quelle funzioni leggono percorsi fissi relativi a una
cartella di lavoro, quindi se ne compone una con dei link simbolici ai file prodotti dalla
pipeline EdgeSHAPer: niente copie di dati e nessuna duplicazione di codice.
"""
import argparse
import json
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "paper"))

import manuscript_statistics as ms  # noqa: E402


def fragment_summary(frags_path, out_path):
    """Riepilogo per tipo di frammento, nello stesso formato del file della pipeline."""
    frame = pd.read_csv(frags_path)
    drug = frame[frame["pred"] == 1]
    n_compounds = drug["id_molecola"].nunique()
    summary = (drug.groupby("smiles")
               .agg(occurrences=("smiles", "size"), compounds=("id_molecola", "nunique"))
               .reset_index()
               .sort_values("occurrences", ascending=False)
               .reset_index(drop=True))
    summary["pct_of_compounds"] = (100 * summary["compounds"] / n_compounds).round(2)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    summary.to_csv(out_path, index=False)
    print(f"[stats] {len(summary)} tipi di frammento, {int(summary['occurrences'].sum())} "
          f"occorrenze su {n_compounds} composti -> {out_path}")
    return summary


def staging(work, summary_path, fingerprint_path, cluster_dir, dataset_path):
    """Cartella con i nomi di file che `manuscript_statistics` si aspetta."""
    os.makedirs(os.path.join(work, "final_clustering"), exist_ok=True)

    def link(src, dst):
        src, dst = os.path.abspath(src), os.path.abspath(dst)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    link(summary_path, os.path.join(work, "high_attention_fragments_summary.csv"))
    link(fingerprint_path, os.path.join(work, "new_molecule_fingerprints2.csv"))
    link(dataset_path, os.path.join(work, "dataset_with_inchikey.csv"))
    for name in ("final_UMAP_embedding.csv",
                 "final_UMAP_HDBSCAN_summary.json",
                 "final_UMAP_HDBSCAN_labels_min_samples_20_min_cluster_size_30.csv"):
        link(os.path.join(cluster_dir, name), os.path.join(work, "final_clustering", name))
    return work


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frags", default="../results/explain/edgeshaper_fragments.csv")
    parser.add_argument("--fingerprint",
                        default="../results/explain/fingerprint_edgeshaper.csv")
    parser.add_argument("--clusters", default="../results/explain/clusters_edgeshaper")
    parser.add_argument("--dataset", default="../results/dataset_with_inchikey.csv")
    parser.add_argument("--out-dir", default="../results/explain/statistics_edgeshaper")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "edgeshap_fragments_summary.csv")
    fragment_summary(args.frags, summary_path)

    work = staging(os.path.join(args.out_dir, "_staging"), summary_path, args.fingerprint,
                   args.clusters, args.dataset)

    stats = {
        "fragments": ms.fragment_statistics(work),
        "clustering": ms.clustering_statistics(work),
        "coherence": ms.family_coherence(work),
    }

    out = os.path.join(args.out_dir, "statistics.json")
    with open(out, "w") as handle:
        json.dump(stats, handle, indent=2)
    print(ms._render(stats))
    print(f"\n[stats] scritto {out}")


if __name__ == "__main__":
    main()
