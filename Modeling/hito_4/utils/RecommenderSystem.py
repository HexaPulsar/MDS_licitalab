import pickle
from utils.UserSpaceGenerator import UserSpaceGenerator
from utils.UserVector import UserVector
from utils.UserSpace import UserSpace
from sklearn.metrics.pairwise import euclidean_distances
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class RecommenderSystem(UserSpaceGenerator):
    
    def __init__(self, DF:pd.DataFrame, save_path:str = os.getcwd(), autoinitialize = True,autosave = True) -> None:
        print("Initializing Recommender System")
        print(f"The current directory is {os.getcwd()}")
        self.df = DF
        #TODO agregar log file que guarde metadata
        self.files_in_directory = os.listdir(save_path)
        self.check_files = ['kmeans_clusters.csv',
                            'vectorized_corpus.csv',
                            'count_vectorizer_model.pkl',
                            'kmeans_model.pkl']
         
        is_subset = set(self.check_files).issubset(self.files_in_directory)
        
        if not is_subset:
            print("Models and Dataframes not found, initializing a Recommender System from zero.")
            super().__init__(DF,save_path=save_path, autoinitialize = True, autosave=True)
        else:
            print('All necesary files have been found.')
            self.user_space = UserSpace(save_path)
            
        
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
    
    def evaluate(self,taxnumberprovider:str):
        """Receives a tax number and evaluates its recommendations
        
        Args:
            taxnumberprovider (str): user to be studied.
        """
        nearest_core_point_cluster,gg = self.predict(taxnumberprovider)
        
        
    def plot_kmeans(self,data):
        plt.figure(figsize=(8,8 ))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=self.user_space.kmeans_model.fit_predict(data), s=70,marker = '.',palette='hls', legend=False)
        plt.title('User Space Kmeans Clustering Results')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        return