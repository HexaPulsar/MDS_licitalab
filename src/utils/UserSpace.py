import pickle
import os
import shutil
import numpy as np
import pandas as pd 
from .UserSpaceGenerator import UserSpaceGenerator

class UserSpace(UserSpaceGenerator):
    def __init__(self,
                 train:pd.DataFrame,
                 test:pd.DataFrame, 
                 save_path:str = os.getcwd(), 
                 elbow_range: np.linspace =  np.linspace(200,250,45,dtype = int)) -> None:
        
        print("Initializing User Space")
        self.save_directory = os.path.join(save_path, 'userspace_data')
        self.train = train
        self.test = test 
        
        
        
            
        try:
            # Attempt to create the directory
            os.makedirs(self.save_directory, exist_ok=True)
            print(f"Directory '{self.save_directory}' created or already exists.")
        except Exception as e:
            print(f"Error creating directory: {e}")
        self.files_in_directory = os.listdir(self.save_directory)
        print(self.files_in_directory)
        self.check_files = ['BERT_model.pkl', 'BERT_tokenizer.pkl', 'clustering_model.pkl', 'clusters.csv', 'corpus.csv', 'vectorized_corpus.csv']
        is_subset = set(self.check_files).issubset(self.files_in_directory)
        
        if not is_subset:
            print("Models and Dataframes not found, initializing a Recommender System from zero.")
            super().__init__(self.train,self.test,elbow_range= elbow_range,save_path=save_path)
        else:
            print('All necesary files have been found.') 
    
        try:
            with open(os.path.join(self.save_directory, "clustering_model.pkl"), "rb") as file:
                self.cluster_model = pickle.load(file)
                print("Loaded cluster model")
        except FileNotFoundError:
            print(f"Error: kmeans_model.pkl not found in {self.save_directory}")

        try:
            with open(os.path.join(self.save_directory, "BERT_model.pkl"), "rb") as file:
                self.model = pickle.load(file)
                print("Loaded BERT_model")
        except FileNotFoundError:
            print(f"Error: BERT_model.pkl not found in {self.save_directory}")

        try:
            with open(os.path.join(self.save_directory, "BERT_tokenizer.pkl"), "rb") as file:
                self.tokenizer = pickle.load(file)
                print("Loaded tokenizer")
        except FileNotFoundError:
            print(f"Error: BERT_tokenizer.pkl not found in {self.save_directory}")

        try:
            self.vectorized_corpus = pd.read_csv(os.path.join(self.save_directory, "vectorized_corpus.csv"))
            print("Loaded vectorized data")
        except FileNotFoundError:
            print(f"Error: vectorized_corpus.csv not found in {self.save_directory}")

        try:
            self.data_with_clusters = pd.read_csv(os.path.join(self.save_directory, "clusters.csv"))
            print("Loaded kmeans data")
        except FileNotFoundError:
            print(f"Error: kmeans_clusters.csv not found in {self.save_directory}")
        try:
            self.corpus  = pd.read_csv(os.path.join(self.save_directory, "corpus.csv"))
            gentrain = self.train[['taxnumberprovider','feature_vector','agilebuyingscode']]
            n_strings = 10
            gb = gentrain.groupby(by =['taxnumberprovider']).agg({'agilebuyingscode':'nunique'})
            gb = gb.sort_values(by = 'agilebuyingscode')
            self.qualifying_users =  gb[gb['agilebuyingscode'] >=  n_strings].index.values
            print(f'Se han removido {round((gb.shape[0] - self.qualifying_users.shape[0])/gb.shape[0] *100,2)}% de taxnumberproviders, por tener < {n_strings} licitaciones. \n El numero de usuarios para crear el corpus será {self.qualifying_users.shape[0]}.')
                #self.qualifying_users = self.generate_corpus()
            print("Loaded corpus data")
            
        except FileNotFoundError:
            print(f"Error: kmeans_clusters.csv not found in {self.save_directory}")
            
    def plot_clusters(self):
        
        pass
        
 