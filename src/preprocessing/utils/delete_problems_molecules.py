import pandas as pd
from rdkit import Chem
import deepchem as dc
import argparse
import os

# Function to validate and count atoms
def is_valid_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Invalid SMILES: {smiles} for error: {e}")
        return False
    if mol is None or mol.GetNumAtoms() < 2:
        return False
    return True

def can_featurize(smiles):
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    mol = Chem.MolFromSmiles(smiles)
    try:
        featurizer._featurize(mol)
        return True
    except Exception:
        return False

def filter_smiles_by_atoms(input_file, output_file, smiles_column):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Instantiates the DeepChem featurizer

    # Filter molecules with invalid SMILES or less than 2 atoms.
    df = df[df[smiles_column].apply(is_valid_molecule)]
    # Function to featurize the molecule.
    

    # Filter the DataFrame keeping only molecules that can be featurized.
    df_filtered = df[df[smiles_column].apply(can_featurize)]

    # Save the result to a new CSV file.
    df_filtered.to_csv(output_file, index=False)

    print(f"Samples with invalid SMILES or less than two atoms removed. Result saved in {output_file}")



if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Filter SMILES by atoms.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_file', type=str, help='Name of the output CSV file')
    parser.add_argument('output_dir', type=str, help='Directory where the output file will be saved')
    parser.add_argument('smiles_column', type=str, default='smiles', help='Name of the column containing SMILES')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Construct full output path
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Run the function
    filter_smiles_by_atoms(args.input_file, output_path, args.smiles_column)
