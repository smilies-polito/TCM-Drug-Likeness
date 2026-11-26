import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import inchi

def get_inchikey_from_smiles(smiles):
    if not smiles or pd.isna(smiles):
        return "SMILES non valido"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Errore: molecola non valida"
        inchi_str = inchi.MolToInchi(mol)
        inchikey = inchi.InchiToInchiKey(inchi_str)
        return inchikey
    except Exception as e:
        return f"Errore: {str(e)}"

def smiles_to_inchikey(test_data_file, output_file):
    # Carica il file CSV
    df = pd.read_csv(test_data_file)

    # Aggiungi tqdm alla progressione
    tqdm.pandas(desc="Elaborazione InChIKey da SMILES")
    df['InChIKey'] = df['smiles'].progress_apply(get_inchikey_from_smiles)

    # Salva il risultato
    df.to_csv(output_file, index=False)

    print(f"Operazione completata. Risultati salvati in '{output_file}'.")

if __name__ == "__main__":
    # Carica il file CSV
    test_data_file = "../data/tm-mc2/preprocessed_TMMC_compounds.csv"
    output_file = "../results/dataset_with_inchikey.csv"

    smiles_to_inchikey(test_data_file, output_file)



    print("Operazione completata. Risultati salvati in 'output_with_inchikey.csv'.")
