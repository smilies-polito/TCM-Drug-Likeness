import pandas as pd
from sklearn.utils import shuffle
import sys

def balance_dataset(input_file, output_file):
    # Load your dataset
    df = pd.read_csv(input_file)

    # Display class distribution before balancing
    print("Distribuzione delle classi prima del bilanciamento:")
    print(df['Drug'].value_counts())

    # Separate classes
    class_0 = df[df['Drug'] == 0]
    class_1 = df[df['Drug'] == 1]

    # Determine the minimum number of samples between the two classes
    min_samples = min(len(class_0), len(class_1))

    # Undersample the classes to have the same number of samples
    class_0_under = class_0.sample(n=min_samples, random_state=42)
    class_1_under = class_1.sample(n=min_samples, random_state=42)

    # Combine the undersampled classes
    df_balanced = pd.concat([class_0_under, class_1_under])

    # Shuffle the balanced dataset
    df_balanced = shuffle(df_balanced, random_state=42)

    # Display the new class distribution
    print("\nDistribuzione delle classi dopo il bilanciamento:")
    print(df_balanced['Drug'].value_counts())

    # Save the balanced dataset to the specified output file
    df_balanced.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ZINC_balance_db.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        balance_dataset(input_file, output_file)
