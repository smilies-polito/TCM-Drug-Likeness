#!/usr/bin/env python3
"""Costruisce l'elenco dei medoidi da un clustering e lo confronta con quello precedente.

`cluster_published_params.py` scrive solo (id_molecola, id_cluster). Qui si aggiungono la
dimensione del cluster e lo SMILES, cioe' il formato che `medoids_to_tcm.py` e lo screening
della letteratura si aspettano.

Il confronto con l'elenco precedente serve a sapere quanto lavoro a valle va rifatto: un
medoide che non cambia porta con se' la sua scheda di letteratura gia' prodotta, uno nuovo va
rischermato. Il confronto e' per SMILES e non per indice, perche' la stessa molecola puo'
ricevere un id_cluster diverso quando il clustering viene rifatto: i cluster non hanno
un'identita' stabile fra esecuzioni, i composti si'.

Uso:

    cd src
    python3 explain/medoids_from_clusters.py \
        --clusters ../results/explain/clusters_edgeshaper \
        --previous ../results/explain/medoids_edgeshap.csv \
        --out ../results/explain/medoids_edgeshaper.csv
"""
import argparse
import glob
import os

import pandas as pd


def load_labels(clusters_dir):
    """Etichette HDBSCAN: il nome del file porta i parametri, quindi si cerca per pattern."""
    matches = glob.glob(os.path.join(clusters_dir, "final_UMAP_HDBSCAN_labels_*.csv"))
    if not matches:
        raise SystemExit(f"[medoidi] nessun file di etichette in {clusters_dir}")
    frame = pd.read_csv(sorted(matches)[0])
    column = "labels" if "labels" in frame.columns else frame.columns[-1]
    return frame[column].values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--fingerprint", required=True,
                        help="serve per risalire da riga della fingerprint a id_molecola")
    parser.add_argument("--compounds", default="../data/tm-mc2/preprocessed_TMMC_compounds.csv")
    parser.add_argument("--previous", default=None,
                        help="elenco precedente, per il confronto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    centroids = pd.read_csv(os.path.join(args.clusters, "centroids_id_molecola.csv"))
    labels = load_labels(args.clusters)
    sizes = pd.Series(labels).value_counts()

    catalogue = pd.read_csv(args.compounds)
    centroids["cluster_size"] = centroids["id_cluster"].map(sizes).astype("Int64")
    centroids["smiles"] = catalogue["smiles"].iloc[centroids["id_molecola"].values].values
    frame = centroids[["id_cluster", "id_molecola", "cluster_size", "smiles"]]
    frame = frame.sort_values("id_cluster").reset_index(drop=True)
    frame.to_csv(args.out, index=False)
    print(f"[medoidi] {len(frame)} medoidi -> {args.out}")
    print(f"[medoidi] dimensioni dei cluster: min {frame.cluster_size.min()}, "
          f"mediana {frame.cluster_size.median():.0f}, max {frame.cluster_size.max()}")

    if args.previous and os.path.exists(args.previous):
        old = pd.read_csv(args.previous)
        kept = set(frame["smiles"]) & set(old["smiles"])
        added = set(frame["smiles"]) - set(old["smiles"])
        dropped = set(old["smiles"]) - set(frame["smiles"])
        print(f"\n[confronto con {os.path.basename(args.previous)}]")
        print(f"  medoidi prima {len(old)}, ora {len(frame)}")
        print(f"  invariati {len(kept)} ({len(kept)/max(len(frame),1):.1%} dei nuovi): "
              f"la scheda di letteratura gia' prodotta resta valida")
        print(f"  nuovi     {len(added)}: da schermare")
        print(f"  usciti    {len(dropped)}")
        reuse = frame[frame["smiles"].isin(kept)]
        todo = frame[frame["smiles"].isin(added)]
        base = os.path.splitext(args.out)[0]
        reuse.to_csv(f"{base}_invariati.csv", index=False)
        todo.to_csv(f"{base}_da_schermare.csv", index=False)
        print(f"  -> {base}_da_schermare.csv ({len(todo)} composti)")
    return frame


if __name__ == "__main__":
    main()
