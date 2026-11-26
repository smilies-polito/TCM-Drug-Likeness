"""Clustering dei frammenti con i parametri gia' validati nel paper, senza ri-tarare.

La griglia completa del paper costa circa 200 ore di CPU: qui si riusano i parametri salvati
in results/trained_model/best_valid_clustering_params.txt, cosi' tutte le varianti di
selezione dei frammenti vengono raggruppate nello stesso modo e sono confrontabili fra loro.

Limite da tenere presente: quei parametri sono stati scelti su una fingerprint da 634 colonne.
Le fingerprint nuove hanno dimensioni diverse, quindi un risultato peggiore puo' dipendere dai
parametri e non dai frammenti. Il confronto e' onesto solo perche' TUTTE le varianti, baseline
dell'attenzione inclusa, girano con gli stessi parametri.

Silhouette e DBCV sono calcolate come in UMAP_validation.py: solo sui punti non rumore.

Il percorso delle predizioni passato a centroids_extraction e' quello in ORDINE DI INPUT, non
il file ordinato per SMILES: vedi la nota sull'indice delle molecole nel report.

Uso:

    cd src
    python3 explain/cluster_published_params.py \\
        --fingerprint ../results/explain/fingerprint_attention_baseline.csv \\
        --out-dir ../results/explain/clusters_attention
"""
import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "analysis"))

import explain_core as core  # noqa: E402

PARAMS_FILE = "../results/trained_model/best_valid_clustering_params.txt"


def published_params(path=PARAMS_FILE):
    """Legge metodo e parametri di clustering validati nel paper."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    def field(key):
        for line in text.splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
        return None

    return (field("Reduction Method"), field("Cluster Method"),
            ast.literal_eval(field("Best Parameters")))


def score_clusters(embedding, labels):
    """Silhouette e DBCV sui soli punti assegnati a un cluster, come nel repo."""
    from sklearn.metrics import silhouette_score
    from hdbscan import validity

    clustered = labels >= 0
    n_clusters = len(set(labels[clustered]))
    result = {
        "n_clusters": n_clusters,
        "n_points": int(len(labels)),
        "noise_fraction": float((~clustered).mean()),
    }
    if n_clusters > 1:
        result["silhouette"] = float(silhouette_score(embedding[clustered], labels[clustered]))
        try:
            result["dbcv"] = float(validity.validity_index(
                embedding[clustered].astype(np.float64), labels[clustered]))
        except Exception as exc:
            result["dbcv"] = float("nan")
            print(f"  DBCV non calcolabile: {exc}")
    else:
        result["silhouette"] = float("nan")
        result["dbcv"] = float("nan")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--predictions",
                        default="../results/explain/tmmc_predictions_input_order.csv",
                        help="predizioni in ORDINE DI INPUT, per la mappa id_molecola -> SMILES")
    parser.add_argument("--skip-centroids", action="store_true")
    args = parser.parse_args(argv)

    core.ensure_repo_on_path()
    from apply_clustering_method import apply_clustering, centroids_extraction

    reduction, method, params = published_params()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"fingerprint : {args.fingerprint}")
    print(f"riduzione   : {reduction}   clustering: {method}")
    print(f"parametri   : {params}")

    labels_df, clusterer, embeddings_df = apply_clustering(
        reduction, method, args.fingerprint, out_dir, params)

    embedding = embeddings_df.drop("id_molecola", axis=1).to_numpy()
    labels = np.asarray(clusterer.labels_)
    scores = score_clusters(embedding, labels)

    print("\n--- qualita' del clustering ---")
    for key, value in scores.items():
        print(f"  {key:18s} {value:.4f}" if isinstance(value, float)
              else f"  {key:18s} {value}")

    pd.DataFrame([scores]).to_csv(os.path.join(out_dir, "cluster_quality.csv"), index=False)

    if not args.skip_centroids:
        print("\n--- estrazione dei medoidi ---")
        centroid_df = centroids_extraction(args.predictions, clusterer, embeddings_df, out_dir)
        print(f"  {len(centroid_df)} medoidi -> {out_dir}")
    return scores


if __name__ == "__main__":
    main()
