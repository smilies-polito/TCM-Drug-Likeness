import pandas as pd
import os

def generate_centroids_with_inchikey(centroids_file, test_data_file, output_file):
    # Load centroid IDs
    centroids_df = pd.read_csv(centroids_file)

    # Load test SMILES data (which includes InChIKey or CID)
    smiles_df = pd.read_csv(test_data_file)

    # Map InChIKey to centroid IDs using the index (id_molecola)
    centroids_df['INCHIKEY'] = centroids_df['id_molecola'].apply(lambda i: smiles_df.iloc[i]['InChIKey'])

    # Save the final file
    centroids_df.to_csv(output_file, index=False)
    print(f"✅ Output file saved: {output_file}")

if __name__ == "__main__":
    centroids_file = "../results/centroids_id_molecola.csv"
    test_data_file = "../../output_with_inchikey.csv"
    output_file = "../../centroids_with_inchikey.csv"

    generate_centroids_with_inchikey(centroids_file, test_data_file, output_file)
