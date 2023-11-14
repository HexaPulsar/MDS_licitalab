from utils.UserVector import UserVector
from utils.UserSpace import UserSpace
from sklearn.metrics.pairwise import euclidean_distances
import os 
import pandas as pd 
import torch 



class RecommenderSystem(UserSpace):
    
    def __init__(self, 
                 train:pd.DataFrame, 
                 test: pd.DataFrame,
                 save_path:str = os.getcwd(), 
                 autoinitialize = True,
                 autosave = True) -> None:
        self.train = train
        self.test = test
        print("Initializing Recommender System")
        print(f"The current directory is {os.getcwd()}")
        if torch.cuda.is_available():
            # Set the GPU device (assuming you have at least one GPU)
            gpu_device = 0  # You can change this to the index of the GPU you want to use
            torch.cuda.set_device(gpu_device)
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_device)}")
        else:
            # If no GPU is available, use the CPU
            self.device = torch.device("cpu")
            print("No GPU available, using CPU")
        self.df = train
        #TODO agregar log file que guarde metadata
        
        super().__init__(train,test,save_path=save_path)
        
    
    def predict(self, taxnumberprovider:str):
        """Predict items (recommend items) to a user based on its taxnumberprovider

        Args:
            taxnumberprovider (str): _description_
        """
        user_vector = UserVector(taxnumberprovider,self.test)
        user_vectorized = self.BERT_vectorize(' '.join(user_vector.strings))
        #print(self.vectorized_corpus.shape,user_vectorized.shape)
        distances = euclidean_distances(user_vectorized.reshape(1, -1), self.vectorized_corpus)
        nearest_core_point_cluster = self.data_with_clusters['Cluster'][distances.argmin()]
        print(f"({taxnumberprovider}) data point belongs to cluster {nearest_core_point_cluster}")
         
        def get_cluster_data(cluster, cluster_method):
            cluster_data = cluster_method[cluster_method['Cluster'] == cluster]
            return cluster_data

        match_train = get_cluster_data(nearest_core_point_cluster,self.data_with_clusters)
        gg = self.train.merge(match_train,on='taxnumberprovider', how='left')
        return nearest_core_point_cluster,gg
    
    def evaluate(self,taxnumberprovider:str):
        """Receives a tax number and evaluates its recommendations
        
        Args:
            taxnumberprovider (str): user to be studied.
        """
        nearest_core_point_cluster,gg = self.predict(taxnumberprovider)
        
 