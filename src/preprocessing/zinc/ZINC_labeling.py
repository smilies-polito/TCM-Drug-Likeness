import pandas as pd
import sys

def label_data(dataset_type):
    if dataset_type not in ["in_vivo", "in_vitro"]:
        raise ValueError("Invalid dataset type. Choose 'in_vivo' or 'in_vitro'.")

    input_file = f"ZINC_{dataset_type}_nuovi_samples_smiles_output.csv"
    df = pd.read_csv(input_file)

    df["Drug"] = 1 if dataset_type == "in_vivo" else 0

    df.to_csv(input_file, index=False, header=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ZINC_labeling.py <in_vivo|in_vitro>")
        sys.exit(1)

    dataset_type = sys.argv[1]
    label_data(dataset_type)