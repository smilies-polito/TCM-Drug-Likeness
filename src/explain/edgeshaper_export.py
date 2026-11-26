"""Esporta i frammenti selezionati da EdgeSHAPer nel formato della pipeline a valle.

Riusa `pair_occlusion.export_fragments` rinominando la colonna del punteggio: il frammento e'
sempre la coppia di clique ai capi dell'arco, quindi la selezione e l'esportazione sono le
stesse: cambia solo CHI assegna il punteggio (Shapley sull'arco invece dell'occlusione della
coppia). Il filtro su pred==1 resta a valle, in `crea_fingerprint`.

Uso:

    cd src
    python3 explain/edge_shapley_export.py --threshold 0.005 \
        --out ../results/explain/frags_edgeshap_thr0005.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import explain_core as core  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default="../results/explain/edgeshaper_edges.csv")
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    scores = pd.read_csv(args.scores).rename(columns={"edge_shapley": "delta_prob"})
    print(f"[edge_shapley_export] {len(scores):,} archi da {args.scores}")
    frags = core.export_fragments(scores, args.threshold)
    drug = frags[frags["pred"] == 1]
    print(f"  soglia {args.threshold:g}: {len(frags):,} righe, "
          f"di cui {len(drug):,} su molecole predette Drug "
          f"({drug['id_molecola'].nunique():,} molecole, "
          f"{drug['smiles'].nunique():,} tipi unici)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frags.to_csv(args.out, index=False)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
