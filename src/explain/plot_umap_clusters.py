"""Mappa dei cluster per ciascun criterio di selezione dei frammenti.

Non disegna niente in proprio: chiama `umap_clusters` di src/paper/make_figures.py, la stessa
funzione che produce la figura del manoscritto, una volta per variante. Quella funzione usa
umap1 e umap2 dell'embedding su cui il clustering e' stato fatto, senza ulteriori riduzioni, e
affianca un pannello ingrandito sulla regione densa: ricalcolare una UMAP a 2 componenti
spezzerebbe le famiglie (verificato nel suo docstring), e le prime due di 40 componenti da
sole, senza ingrandimento, schiacciano tutto in un angolo.

Ogni variante scrive <cartella della variante>/figures/umap_clusters.pdf, piu' una copia PNG
per chi non ha un lettore PDF a portata di mano. Il PDF resta il formato del manoscritto.

Uso:

    cd src
    python3 explain/plot_umap_clusters.py
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "paper"))

import make_figures  # noqa: E402


def saving_png_too(png_path):
    """Salva anche un PNG della figura che make_figures sta per chiudere.

    `umap_clusters` scrive solo il PDF voluto dal manoscritto e poi chiude la figura. Invece
    di duplicarne il codice o di modificarla, qui si intercetta la chiusura: la figura e'
    ancora viva in quel momento, quindi basta salvarla una seconda volta. Il contenuto e'
    identico, cambia solo il formato.
    """
    original_close = make_figures.plt.close

    def close(fig=None):
        if fig is not None:
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            print(f"[plot] salvata anche {png_path}")
        original_close(fig)

    return original_close, close

# (titolo della variante, cartella dei risultati di clustering)
VARIANTS = [
    ("attenzione (criterio pubblicato)", "../results/explain/clusters_attention"),
    ("occlusione di clique singole", "../results/explain/clusters_single"),
    ("occlusione di coppie", "../results/explain/clusters_pairs"),
    ("EdgeSHAPer, soglia 0.005", "../results/explain/clusters_edgeshaper"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zoom-pct", type=float, default=2.0,
                        help="percentile delle mediane di famiglia escluso dalla vista; "
                             "0 tiene tutte le famiglie")
    args = parser.parse_args(argv)

    for title, directory in VARIANTS:
        labels = glob.glob(os.path.join(directory, "*labels*.csv"))
        if not labels:
            print(f"[plot] salto {title}: etichette non trovate in {directory}")
            continue
        print(f"\n### {title} ###")
        png = os.path.join(directory, "figures", "umap_clusters.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        original_close, patched_close = saving_png_too(png)
        make_figures.plt.close = patched_close
        try:
            make_figures.umap_clusters(
                out_dir=directory,
                embedding_path=os.path.join(directory, "final_UMAP_embedding.csv"),
                labels_path=labels[0],
                centroids_path=os.path.join(directory, "centroids_id_molecola.csv"),
                zoom_pct=args.zoom_pct)
        finally:
            make_figures.plt.close = original_close
    return 0


if __name__ == "__main__":
    sys.exit(main())
