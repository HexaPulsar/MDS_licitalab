
import pickle
import os
import pandas as pd 
from utils.UserSpaceGenerator import UserSpaceGenerator


class UserSpace(UserSpaceGenerator):
    def __init__(self,DF, save_path) -> None:
        self.files_in_directory = os.listdir(save_path)
        #print(self.files_in_directory)
        self.check_files = ['kmeans_clusters.csv',
                            'kmeans_model.pkl',
                            'model.pkl',
                            'tokenizer.pkl',
                            'vectorized_corpus.csv']
         
        is_subset = set(self.check_files).issubset(self.files_in_directory)
        
        if not is_subset:
            print("Models and Dataframes not found, initializing a Recommender System from zero.")
            super().__init__(DF,save_path=save_path)
        else:
            print('All necesary files have been found.') 
                
            try:
                with open(f"{save_path}/kmeans_model.pkl", "rb") as file:
                    self.cluster_model = pickle.load(file)
            except FileNotFoundError:
                print(f"Error: kmeans_model.pkl not found in {save_path}")
                self.cluster_model = None
            
            try:
                with open(f"{save_path}/model.pkl", "rb") as file:
                    self.model = pickle.load(file)
            except FileNotFoundError:
                print(f"Error: model.pkl not found in {save_path}")
                self.model = None
            
            try:
                with open(f"{save_path}/tokenizer.pkl", "rb") as file:
                    print('loaded tokenizer')
                    self.tokenizer = pickle.load(file)
            except FileNotFoundError:
                print(f"Error: tokenizer.pkl not found in {save_path}")
                self.tokenizer = None
            
            try:
                self.vectorized_corpus = pd.read_csv(f"{save_path}/vectorized_corpus.csv")
            except FileNotFoundError:
                print(f"Error: vectorized_corpus.csv not found in {save_path}")
                self.vectorized_corpus = None
            
            try:
                self.data_with_clusters = pd.read_csv(f"{save_path}/kmeans_clusters.csv")
            except FileNotFoundError:
                print(f"Error: kmeans_clusters.csv not found in {save_path}")
                self.data_with_clusters = None
            