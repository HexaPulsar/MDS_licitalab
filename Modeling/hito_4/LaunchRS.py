import pandas as pd 
from utils.RecommenderSystem import RecommenderSystem
from unidecode import unidecode
import sys
from tqdm import tqdm
import os

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)  # Change the working directory to the script's directory
        if len(sys.argv) != 3:
                print("Usage: python script.py [csv_file_path] [csv_percentage_to_be_used: float | optional]")
                print("Using default values for arg1 and arg2.")
                percentage = 0.5
                
        else:
                csv_file_path = sys.argv[1]
            
                percentage = sys.argv[2]
            
        # Incorporate data
        print("Counting length of csv file")
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for _ in file) 

        # Create an empty list to store the DataFrames from chunks
        dfs = [] 
        # Create a tqdm wrapper for pd.read_csv
        with tqdm(total=total_lines, desc = 'Loading Dataset') as pbar:
            def update_progress(n):
                pbar.update(n)

            # Read the CSV file using pd.read_csv and provide the progress callback
            df_chunks = pd.read_csv(csv_file_path, chunksize=10000, iterator=True, encoding='utf-8',nrows=total_lines)  # Specify the encoding
            for chunk in df_chunks:
                # Process each chunk if needed
                # You can access the chunk data in the 'chunk' DataFrame
                #chunk['first_two_digits_code'] = chunk['agilebuyingscode'].apply(lambda x: x[:2])
                chunk['feature_vector'] = chunk['agilebuyingscode'].apply(lambda x: x[:2]) + ' ' + chunk['agileoffereditemsdescripcionofertada']
                chunk['feature_vector'] = chunk['feature_vector'].apply(lambda x: unidecode(str(x)).lower())
                
                dfs.append(chunk)
                update_progress(chunk.shape[0])

        # Concatenate the list of DataFrames into a single DataFrame
        df = pd.concat(dfs, ignore_index=True)
        RS = RecommenderSystem(df,save_path=  os.path.dirname(os.path.abspath(__file__)))
        return
    except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main( )