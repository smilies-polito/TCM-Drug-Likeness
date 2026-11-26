import pandas as pd

def find_matching_smiles(file1, file2, output_file):
    """
    This function reads two CSV files, merges them based on a common 'ZINCID' column, and writes the result to an output file.
    Args:
        file1 (str): The file path to the first CSV file containing 'ZINCID' and 'Smiles' columns.
        file2 (str): The file path to the second CSV file containing 'ZINCID' and 'Drug' columns.
        output_file (str): The file path where the merged results will be saved.
    Returns:
        None
    The function performs the following steps:
        1. Reads the input CSV files into pandas DataFrames.
        2. Merges the DataFrames on the 'ZINCID' column using an inner join.
        3. Prints the number of matching SMILES found.
        4. Writes the merged DataFrame to the specified output file.
    """
    # Read CSV files using pandas
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Perform a merge between the two DataFrames on the ZINC IDs.
    result_df = pd.merge(df2[['ZINCID', 'Drug']], df1[['ZINCID', 'Smiles']], on='ZINCID', how='inner')

    print('Numero di SMILES trovati:', len(result_df))

    # Write the results to the output file
    result_df.to_csv(output_file, index=False)

# Example of using the function
file1 = 'ZINC\\new\\ZINC_smiles_output.csv'
file2 = 'ZINC\\new2\\new_ZINC_ids_labelled.csv'
output_file = 'ZINC\\new2\\new2_ZINC_smiles_output.csv'
find_matching_smiles(file1, file2, output_file)
