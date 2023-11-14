import pandas as pd
import glob
import os

BIG_CSV_PATH = 'C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\Modeling\\query_final_results_mayo_20231026092933.csv\\query_final_results_20231026092933.csv'

""" this cleanup function is meant for large datasets with duplicate values """

def split_csv(input_file, output_prefix, chunk_size):
    # Read the large CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Calculate the number of chunks
    num_chunks = len(df) // chunk_size + 1

    # Split the DataFrame into chunks
    chunks = [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    # Save each chunk to a separate CSV file
    for i, chunk in enumerate(chunks):
        output_file = f"{output_prefix}_{i+1}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Chunk {i+1} saved to {output_file}")


def concatenate_and_remove(input_pattern, output_file):
    # Get a list of all CSV files that match the specified pattern
    input_files = glob.glob(input_pattern)

    # Initialize an empty list to store DataFrames
    dfs = []

    # Read each CSV file into a DataFrame and append it to the list
    for file in input_files:
        df = pd.read_csv(file)
        df = df.drop_duplicates()
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

    # Remove the individual CSV files
    for file in input_files:
        os.remove(file)
        print(f"Removed: {file}")
 
    
# Example usage: 
output_prefix = 'output_chunk'  # Prefix for output files
chunk_size = 5000000  # Number of rows per chunk
split_csv(BIG_CSV_PATH, output_prefix, chunk_size)

# Example usage:
input_pattern = 'C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\output_chunk_*.csv'  # Replace with the path to your CSV files
output_file = 'concatenated_output.csv'  # Replace with the desired output file name

concatenate_and_remove(input_pattern, output_file)