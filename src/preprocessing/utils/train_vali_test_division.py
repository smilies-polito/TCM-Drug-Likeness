"""Train/validation/test split of the balanced ZINC dataset, stratified by scaffold and class.

Molecules are grouped by Bemis-Murcko scaffold (DeepChem's ScaffoldSplitter) and the
molecules of EACH scaffold are shuffled and divided among the three sets, separately for
the two classes. Every set therefore contains the same scaffolds in similar proportions:
this is not a scaffold split in the strict sense (where each scaffold goes to one set only).

Note on the fractions: they are applied *per scaffold* and truncated with int(), so the
realised validation/test fractions come out appreciably lower than the nominal values
(a scaffold with fewer than 1/fraction molecules contributes nothing to val/test and
goes entirely to train). The nominal 0.14/0.15 defaults below are the values used to
produce the dataset splits shipped in data/zinc/splits/, which realise roughly an 80/10/10
split.

The shipped splits were generated in March 2025, before the seed was added (July 2026), and
are the reference: re-running this script reproduces their sizes and class counts exactly,
but not the assignment of individual molecules.

Usage:
    python train_vali_test_division.py <input_csv> --output-dir DIR [--seed N]
                                       [--valid-frac-drug F] [--test-frac-drug F]
                                       [--valid-frac-nondrug F] [--test-frac-nondrug F]
"""
import argparse
import os
import random

import deepchem as dc
import numpy as np
import pandas as pd
import polars as pl


def dataset_split(data_path,
                  output_dir,
                  seed=42,
                  valid_frac_drug=0.14,
                  test_frac_drug=0.14,
                  valid_frac_non_drug=0.15,
                  test_frac_non_drug=0.15):
    # Seed esplicito: senza di questo lo split non e' riproducibile (random.shuffle sotto
    # decide quali molecole finiscono in ciascun set). I conteggi non dipendono dal seed,
    # solo l'assegnazione delle singole molecole.
    random.seed(seed)

    df = pl.read_csv(data_path).to_pandas()

    data_test = df['smiles'].tolist()

    Xs = np.zeros(len(data_test))
    Ys = np.ones(len(data_test))

    # DeepChem dataset with the SMILES in the IDs field
    dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=Ys, w=np.zeros(len(data_test)), ids=data_test)
    scaffoldsplitter = dc.splits.ScaffoldSplitter()

    scaffolds_set = scaffoldsplitter.generate_scaffolds(dataset)

    print('Starting creating training, validation, and test sets')
    print(f'Number of scaffolds: {len(scaffolds_set)}')

    train_set, valid_set, test_set = [], [], []

    for scaffold in scaffolds_set:
        random.shuffle(scaffold)
        scaffold_df = df.iloc[list(map(int, scaffold))]

        drugs = scaffold_df[scaffold_df['Drug'] == 1]
        non_drugs = scaffold_df[scaffold_df['Drug'] == 0]

        # Numero di elementi per ciascun set, calcolato SEPARATAMENTE per drugs e non_drugs
        valid_drugs = int(len(drugs) * valid_frac_drug)
        test_drugs = int(len(drugs) * test_frac_drug)

        valid_non_drugs = int(len(non_drugs) * valid_frac_non_drug)
        test_non_drugs = int(len(non_drugs) * test_frac_non_drug)

        valid_set.append(drugs.iloc[:valid_drugs])
        test_set.append(drugs.iloc[valid_drugs:valid_drugs + test_drugs])
        train_set.append(drugs.iloc[valid_drugs + test_drugs:])

        valid_set.append(non_drugs.iloc[:valid_non_drugs])
        test_set.append(non_drugs.iloc[valid_non_drugs:valid_non_drugs + test_non_drugs])
        train_set.append(non_drugs.iloc[valid_non_drugs + test_non_drugs:])

    train_set = pd.concat(train_set)
    valid_set = pd.concat(valid_set)
    test_set = pd.concat(test_set)

    train_path = os.path.join(output_dir, 'new2_ZINC_final_balanced_train_set.csv')
    valid_path = os.path.join(output_dir, 'new2_ZINC_final_balanced_validation_set.csv')
    test_path = os.path.join(output_dir, 'new2_ZINC_final_balanced_test_set.csv')
    os.makedirs(output_dir, exist_ok=True)

    train_set.to_csv(train_path, index=False)
    valid_set.to_csv(valid_path, index=False)
    test_set.to_csv(test_path, index=False)

    # Riepilogo dello split effettivamente realizzato (le frazioni realizzate sono piu'
    # basse di quelle nominali per via del troncamento per scaffold, vedi docstring).
    total = len(train_set) + len(valid_set) + len(test_set)
    print('\nRealised split:')
    for name, subset, path in (('train', train_set, train_path),
                               ('validation', valid_set, valid_path),
                               ('test', test_set, test_path)):
        n_drug = int((subset['Drug'] == 1).sum())
        n_non_drug = int((subset['Drug'] == 0).sum())
        print(f'  {name:<11} {len(subset):>7} molecules ({len(subset) / total:.1%})  '
              f'Drug=1: {n_drug:>6}  Drug=0: {n_non_drug:>6}  -> {path}')

    return train_set, valid_set, test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train/validation/test split stratified by scaffold and class.')
    parser.add_argument('data_path', help='Input CSV with columns: smiles, Drug')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for the three split files. Required, so that the shipped '
                             'splits in data/zinc/splits/ are never overwritten by accident')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the per-scaffold shuffle (default: 42)')
    parser.add_argument('--valid-frac-drug', type=float, default=0.14)
    parser.add_argument('--test-frac-drug', type=float, default=0.14)
    parser.add_argument('--valid-frac-nondrug', type=float, default=0.15)
    parser.add_argument('--test-frac-nondrug', type=float, default=0.15)
    args = parser.parse_args()

    dataset_split(args.data_path,
                  output_dir=args.output_dir,
                  seed=args.seed,
                  valid_frac_drug=args.valid_frac_drug,
                  test_frac_drug=args.test_frac_drug,
                  valid_frac_non_drug=args.valid_frac_nondrug,
                  test_frac_non_drug=args.test_frac_nondrug)
