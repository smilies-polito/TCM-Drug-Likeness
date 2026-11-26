#!/usr/bin/env python3
"""
Converte i test set nei formati di input richiesti da ciascun modello del benchmark.

  - nostro test ZINC (ZINCID,smiles,Drug)  -> zinc_test_dgcan.txt     (righe "SMILES label", per D-GCAN)
                                           -> zinc_test_generaldl.csv (smiles,label, per GeneralDL/SpecDL)
  - test D-GCAN (righe "SMILES  label")    -> dgcan_test.csv          (ZINCID,smiles,Drug, per il nostro modello)
  - test druglikeness (smiles,label,...)   -> druglikeness_test.csv   (ZINCID,smiles,Drug, per il nostro modello)

Nei due test set esterni non ci sono ID ZINC: la colonna ZINCID contiene l'indice di riga.
L'ordine delle molecole e' sempre quello del file di partenza.
"""
import argparse
import os

import pandas as pd


def read_dgcan_txt(path):
    # Il test set di D-GCAN ha fine riga CRLF, righe vuote finali e due spazi come separatore.
    rows = []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            smiles, label = fields
            rows.append((smiles, int(label)))
    return pd.DataFrame(rows, columns=["smiles", "Drug"])


def to_our_format(df_smiles_label):
    df = df_smiles_label.reset_index(drop=True)
    return pd.DataFrame({"ZINCID": range(len(df)), "smiles": df["smiles"], "Drug": df["Drug"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zinc-test", required=True)
    parser.add_argument("--dgcan-test", required=True)
    parser.add_argument("--druglikeness-test", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    zinc = pd.read_csv(args.zinc_test)
    with open(os.path.join(args.out_dir, "zinc_test_dgcan.txt"), "w") as f:
        for smiles, label in zip(zinc["smiles"], zinc["Drug"]):
            f.write(f"{smiles} {label}\n")
    zinc[["smiles", "Drug"]].rename(columns={"Drug": "label"}).to_csv(
        os.path.join(args.out_dir, "zinc_test_generaldl.csv"), index=False)

    dgcan = read_dgcan_txt(args.dgcan_test)
    to_our_format(dgcan).to_csv(os.path.join(args.out_dir, "dgcan_test.csv"), index=False)

    druglikeness = pd.read_csv(args.druglikeness_test).rename(columns={"label": "Drug"})
    to_our_format(druglikeness).to_csv(os.path.join(args.out_dir, "druglikeness_test.csv"), index=False)

    print(f"nostro test ZINC:    {len(zinc)} molecole")
    print(f"test D-GCAN:         {len(dgcan)} molecole")
    print(f"test druglikeness:   {len(druglikeness)} molecole")
    print(f"file scritti in {args.out_dir}")


if __name__ == "__main__":
    main()
