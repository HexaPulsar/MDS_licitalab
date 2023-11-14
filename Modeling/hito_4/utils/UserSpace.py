
import pickle
import os
import shutil
import pandas as pd 
from utils.UserSpaceGenerator import UserSpaceGenerator


class UserSpace(UserSpaceGenerator):
    def __init__(self,
                 train,
                 test,
                 save_path) -> None:
        print("Initializing User Space")
        self.model_save_directory = os.path.join(save_path, 'userspace_data')
        self.train = train
        self.test = test
        self.save_path = save_path
        try:
            # Attempt to create the directory
            os.makedirs(self.model_save_directory, exist_ok=True)
            print(f"Directory '{self.model_save_directory}' created or already exists.")
        except Exception as e:
            print(f"Error creating directory: {e}")
        self.files_in_directory = os.listdir(self.model_save_directory)
        print(self.files_in_directory)
        self.check_files = ['BERT_model.pkl', 'BERT_tokenizer.pkl', 'kmeans_clusters.csv', 'kmeans_model.pkl', 'vectorized_corpus.csv']
        is_subset = set(self.check_files).issubset(self.files_in_directory)
        
        if not is_subset:
            print("Models and Dataframes not found, initializing a Recommender System from zero.")
            super().__init__(self.train,save_path=save_path)
        else:
            print('All necesary files have been found.') 
        try:
            with open(os.path.join(self.model_save_directory, "kmeans_model.pkl"), "rb") as file:
                self.cluster_model = pickle.load(file)
                print("Loaded cluster model")
        except FileNotFoundError:
            print(f"Error: kmeans_model.pkl not found in {self.model_save_directory}")

        try:
            with open(os.path.join(self.model_save_directory, "BERT_model.pkl"), "rb") as file:
                self.model = pickle.load(file)
                print("Loaded BERT_model")
        except FileNotFoundError:
            print(f"Error: BERT_model.pkl not found in {self.model_save_directory}")

        try:
            with open(os.path.join(self.model_save_directory, "BERT_tokenizer.pkl"), "rb") as file:
                self.tokenizer = pickle.load(file)
                print("Loaded tokenizer")
        except FileNotFoundError:
            print(f"Error: BERT_tokenizer.pkl not found in {self.model_save_directory}")

        try:
            self.vectorized_corpus = pd.read_csv(os.path.join(self.model_save_directory, "vectorized_corpus.csv"))
            print("Loaded vectorized data")
        except FileNotFoundError:
            print(f"Error: vectorized_corpus.csv not found in {self.model_save_directory}")

        try:
            self.data_with_clusters = pd.read_csv(os.path.join(self.model_save_directory, "kmeans_clusters.csv"))
            print("Loaded kmeans data")
        except FileNotFoundError:
            print(f"Error: kmeans_clusters.csv not found in {self.model_save_directory}")
            
    def regenerate_system(self):
        try:
            # Attempt to remove the directory and its contents
            shutil.rmtree(self.model_save_directory)
            print(f"Successfully removed directory: {self.model_save_directory}. Re-Initializing userspace")
            super().__init__(self.train,self.test,save_path=self.save_path)
            
        except FileNotFoundError:
            print(f"Error: Directory not found: {self.model_save_directory}")
        except Exception as e:
            print(f"An error occurred: {e}")
        