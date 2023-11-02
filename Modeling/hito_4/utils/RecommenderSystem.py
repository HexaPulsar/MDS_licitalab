import pickle
from utils.UserSpaceGenerator import UserSpaceGenerator
from utils.UserVector import UserVector
from sklearn.metrics.pairwise import euclidean_distances
import os 
import pandas as pd

class RecommenderSystem(UserSpaceGenerator):
    
    def __init__(self, DF:pd.DataFrame, save_path:str = os.getcwd(), autoinitialize = True,autosave = True) -> None:
        print("Initializing Recommender System")
        print(f"The current directory is {os.getcwd()}")
        self.df = DF
        self.files_in_directory = os.listdir(save_path)
        self.check_files = ['kmeans_clusters.csv',
                            'vectorized_corpus.csv',
                            'count_vectorizer_model.pkl',
                            'kmeans_model.pkl']
        print(save_path)
        print(self.files_in_directory)
        is_subset = set(self.check_files).issubset(self.files_in_directory)
        
        if not is_subset:
                super().__init__(DF,save_path=save_path, autoinitialize = True, autosave=True)
        else:
            with open(f"{save_path}/kmeans_model.pkl", "rb") as file:
                self.cluster_model = pickle.load(file)
            with open(f"{save_path}/count_vectorizer_model.pkl", "rb") as file:
                self.vectorizer = pickle.load(file)
            self.vectorized_corpus = pd.read_csv(f"{save_path}/vectorized_corpus.csv")
            self.data_with_clusters = pd.read_csv(f"{save_path}/kmeans_clusters.csv")
            
        
    def predict(self, taxnumberprovider:str):
        """Predict items (recommend items) to a user based on its taxnumberprovider

        Args:
            taxnumberprovider (str): _description_
        """
        user_vector = UserVector(taxnumberprovider,self.df)
        user_vectorized = self.vectorizer.transform([' '.join(user_vector.strings)]).toarray().flatten()
        distances = euclidean_distances(user_vectorized.reshape(1,-1), self.vectorized_corpus)
        nearest_core_point_cluster = self.data_with_clusters['Cluster'][distances.argmin()]
        print(f"Unseen data point belongs to cluster {nearest_core_point_cluster}")
         
        def get_cluster_data(cluster, cluster_method):
            cluster_data = cluster_method[cluster_method['Cluster'] == cluster]
            return cluster_data

        match_df = get_cluster_data(nearest_core_point_cluster,self.data_with_clusters)
        gg = self.df.merge(match_df,on='taxnumberprovider', how='left')
        return nearest_core_point_cluster,gg
    
