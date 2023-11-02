import pandas as pd
import os
from utils.RecommenderSystem import RecommenderSystem
from tqdm import tqdm

def initialize_recommender_system(save_path):
    csv_file_path = 'C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\Modeling\\20231007200103_query_results.csv'

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)

    # Create an empty list to store the DataFrames from chunks
    dfs = []
    total_lines = 100000
    # Create a tqdm wrapper for pd.read_csv
    with tqdm(total=total_lines, desc = 'Loading Dataset') as pbar:
        def update_progress(n):
            pbar.update(n)

        # Read the CSV file using pd.read_csv and provide the progress callback
        df_chunks = pd.read_csv(csv_file_path, chunksize=1000, iterator=True, encoding='utf-8',nrows=100000)  # Specify the encoding
        for chunk in df_chunks:
            # Process each chunk if needed
            # You can access the chunk data in the 'chunk' DataFrame
            #chunk['first_two_digits_code'] = chunk['agilebuyingscode'].apply(lambda x: x[:2])
            #chunk['feature_vector'] = chunk['first_two_digits_code'] + ' ' + chunk['agileoffereditemsdescripcionofertada']
            #chunk['feature_vector'] = chunk['feature_vector'].apply(lambda x: unidecode(str(x)).lower())
            
            dfs.append(chunk)
            update_progress(chunk.shape[0])

    # Concatenate the list of DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)
    RS = RecommenderSystem(df,save_path )
    return RS