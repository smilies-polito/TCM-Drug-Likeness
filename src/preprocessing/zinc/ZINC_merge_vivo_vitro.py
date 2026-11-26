import sys

def merge_files(file1, file2, output_file):
    header = "ZINC,label\n"

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as output:
        output.write(header)
        
        # Write elements from the first file with label 0
        for line in f1:
            output.write(f"{line.strip()},0\n")
        
        # Write elements from the second file with label 1
        for line in f2:
            output.write(f"{line.strip()},1\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ZINC_merge_vivo_vitro.py <file1> <file2> <output_file>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]

    merge_files(file1, file2, output_file)