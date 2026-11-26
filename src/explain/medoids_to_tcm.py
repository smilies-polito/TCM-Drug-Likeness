#!/usr/bin/env python3
"""Informazioni TM-MC (erbe e proteine) per i medoidi di un clustering.

Riusa le tre funzioni gia' presenti nella pipeline, nello stesso ordine in cui le chiama
`results_reproduction.py`: conversione SMILES -> InChIKey dell'intero catalogo TM-MC, mappatura
`id_molecola` -> InChIKey per i soli medoidi, interrogazione delle tabelle TM-MC.

La tabella degli InChIKey e' calcolata una volta sola e riusata fra i criteri: dipende solo dal
catalogo dei composti, non dal clustering, e convertire 20.973 molecole non e' istantaneo.

`generate_centroids_with_inchikey` risolve `id_molecola` per posizione (`smiles_df.iloc[i]`),
quindi l'indice deve riferirsi al catalogo completo e non alle sole molecole predette come
farmaco. Il controllo sugli SMILES qui sotto verifica proprio questo: se l'indice fosse quello
della fingerprint filtrata, le stringhe non coinciderebbero.
"""
import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "analysis"))

from smiles_conversion import smiles_to_inchikey                      # noqa: E402
from generate_centroids_with_inchikey import generate_centroids_with_inchikey  # noqa: E402
from Centroids_to_TMMC import centroids_to_TMMC                       # noqa: E402


def check_alignment(medoids_path, inchikey_table):
    """Confronta gli SMILES dei medoidi con quelli alla stessa posizione nel catalogo."""
    medoids = pd.read_csv(medoids_path)
    catalogue = pd.read_csv(inchikey_table)
    if "smiles" not in medoids.columns:
        print("[tcm] il file dei medoidi non riporta gli SMILES: controllo saltato")
        return
    found = catalogue["smiles"].iloc[medoids["id_molecola"].values].values
    mismatch = (found != medoids["smiles"].values).sum()
    print(f"[tcm] controllo indici: {len(medoids) - mismatch}/{len(medoids)} SMILES coincidono")
    if mismatch:
        raise SystemExit(f"[tcm] {mismatch} medoidi non corrispondono: id_molecola non indicizza "
                         f"{inchikey_table}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medoids", default="../results/explain/medoids_edgeshap.csv")
    parser.add_argument("--out-dir", default="../results/explain/tcm_edgeshap")
    parser.add_argument("--compounds", default="../data/tm-mc2/preprocessed_TMMC_compounds.csv")
    parser.add_argument("--datafolder", default="../data/tm-mc2/")
    parser.add_argument("--inchikey-table", default="../results/dataset_with_inchikey.csv")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.inchikey_table):
        print(f"[tcm] riuso la tabella InChIKey gia' presente: {args.inchikey_table}")
    else:
        print(f"[tcm] converto gli SMILES del catalogo -> {args.inchikey_table}")
        smiles_to_inchikey(args.compounds, args.inchikey_table)

    check_alignment(args.medoids, args.inchikey_table)

    centroids_ik = os.path.join(args.out_dir, "centroids_with_inchikey.csv")
    generate_centroids_with_inchikey(args.medoids, args.inchikey_table, centroids_ik)

    centroids_to_TMMC(centroids_ik, args.out_dir, args.datafolder)

    # tabella finale: dati del medoide + quanto recuperato da TM-MC
    medoids = pd.read_csv(centroids_ik)
    tcm = pd.read_csv(os.path.join(args.out_dir, "inchikey_dict.csv"))
    table = medoids.merge(tcm.drop(columns=["id_cluster", "SMILES"]), on="INCHIKEY", how="left")
    table["n_herbs"] = table["COMPOUNDS"].fillna("").apply(lambda s: len(s.split(";")) if s else 0)
    table["n_proteins"] = table["Proteins"].fillna("").apply(lambda s: len(s.split(";")) if s else 0)
    table = table.sort_values("cluster_size", ascending=False) if "cluster_size" in table else table

    out = os.path.join(args.out_dir, "medoids_tcm.csv")
    table.to_csv(out, index=False)

    matched = table["ID"].notna().sum()
    print(f"\n[tcm] {len(table)} medoidi, {matched} con un ID TM-MC")
    print(f"[tcm] {(table['n_herbs'] > 0).sum()} con almeno un'erba, "
          f"{(table['n_proteins'] > 0).sum()} con almeno una proteina")
    print(f"[tcm] erbe per medoide: mediana {table['n_herbs'].median():.0f}, "
          f"massimo {table['n_herbs'].max()}")
    print(f"[tcm] scritta {out}")


if __name__ == "__main__":
    main()
