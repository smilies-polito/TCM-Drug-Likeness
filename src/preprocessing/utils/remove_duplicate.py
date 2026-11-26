import pandas as pd

def remove_common_elements(file1_path, file2_path, output_path, column_name):
    # Read the two CSV files in DataFrame
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Find common elements in the specified column
    common_elements = set(df1[column_name]).intersection(set(df2[column_name]))

    # Filter the second DataFrame by removing rows with common elements.
    df2_filtered = df2[~df2[column_name].isin(common_elements)]

    # Save the result to a new CSV file.
    df2_filtered.to_csv(output_path, index=False)

    # Calculates the amount of eliminated elements
    num_elements_removed = len(df2) - len(df2_filtered)

    print(f"Common rows removed: {num_elements_removed}. Result saved in {output_path}")

# File paths
file1_path = 'ZINC\\new\\ZINC_final_balanced_test_set.csv'
file2_path = 'ZINC\\new\\ZINC_final_balanced_validation_set.csv'
output_path = 'ZINC\\new\\output.csv'
column_name = 'smiles'  # Replace with the name of the column of interest

# Run the function
remove_common_elements(file1_path, file2_path, output_path, column_name)
