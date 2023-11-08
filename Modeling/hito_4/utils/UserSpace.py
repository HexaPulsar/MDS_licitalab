
import pickle
import os
import pandas as pd 


class UserSpace():
    def __init__(self, save_path) -> None:
        with open(f"{save_path}/kmeans_model.pkl", "rb") as file:
                self.cluster_model = pickle.load(file)
        with open(f"{save_path}/count_vectorizer_model.pkl", "rb") as file:
            self.vectorizer = pickle.load(file)
        self.vectorized_corpus = pd.read_csv(f"{save_path}/vectorized_corpus.csv")
        self.data_with_clusters = pd.read_csv(f"{save_path}/kmeans_clusters.csv")
         