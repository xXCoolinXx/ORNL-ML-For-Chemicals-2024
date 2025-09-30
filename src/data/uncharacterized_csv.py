import csv
import re

def txt_to_csv(input_file, output_file):
    """
    Converts a space-delimited text file to a CSV file, 
    handling varying numbers of spaces.

    Args:
        input_file (str): Path to the input TXT file.
        output_file (str): Path to the output CSV file.
    """

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        writer.writerow(["exclude_indices"])
        for line in infile:
            # Split the line by one or more spaces
            try:
                row = re.split(r'\s+', line.strip()) 
                # print(row)
                row = int(row[0])
                writer.writerow([row])
            except ValueError:
                continue

if __name__ == "__main__":
    input_file = '../../data/custom_qm9/raw/uncharacterized.txt'  # Replace with your input file name
    output_file = '../../data/custom_qm9/raw/uncharacterized.csv'  # Replace with desired output file name

    txt_to_csv(input_file, output_file)