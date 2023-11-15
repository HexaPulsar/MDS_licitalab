 
from sklearn.feature_extraction.text import CountVectorizer 
from utils.UserVector import UserVector
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import os 
from sklearn.cluster import KMeans, AgglomerativeClustering 
import numpy as np 
import re
from transformers import BertTokenizer, BertModel 
from sklearn.metrics import silhouette_score 
from scipy.cluster.hierarchy import dendrogram, linkage


class UserSpaceGenerator(UserVector):
    def __init__(self, 
                 train:pd.DataFrame, 
                 test:pd.DataFrame,
                 save_path:str = os.getcwd(),
                 clustering_model = KMeans,
                 n_strings:int = 10,
                 **kwargs 
                 ) -> None:
        
        try:
            # Attempt to create the directory
            os.makedirs(self.model_save_directory, exist_ok=True)
            print(f"Directory '{self.model_save_directory}' created or already exists.")
        except Exception as e:
            print(f"Error creating directory: {e}")
            
        print("Generating User Space")
        self.train = train  
        
        self.save_directory = f"{save_path}/userspace_data"
        self.qualifying_users = self.find_qualifying_users(self.train)
        self.corpus = self.generate_corpus()
        
        self.tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.model = self.model.to('cuda')
        
        self.vectorized_corpus = self.BERT_vectorize_corpus()
        
        
        self.cluster_model = clustering_model(**kwargs)
        
        
        #TODO permitir definir el numero de clusters desde la definicion de clase o desde yaml file
        self.data_with_clusters = self.reduce_and_plot()
            
    
        self.export_cluster_model_and_data()
        self.export_vectorizer()
        self.export_vectorized_corpus() 
        self.export_corpus()
    
    
    def find_qualifying_users(self,df):
        gentrain = df[['taxnumberprovider','feature_vector','agilebuyingscode']]
        n_strings = 10
        gb = gentrain.groupby(by =['taxnumberprovider']).agg({'agilebuyingscode':'nunique'})
        gb = gb.sort_values(by = 'agilebuyingscode')
        qualifying_users =  gb[gb['agilebuyingscode'] >=  n_strings].index.values
        print(f'Se han removido {round((gb.shape[0] - qualifying_users.shape[0])/gb.shape[0] *100,2)}% de taxnumberproviders, por tener < {n_strings} licitaciones. \n El numero de usuarios para crear el corpus será {qualifying_users.shape[0]}.')
        return qualifying_users
    
    def generate_corpus(self):
        """generates list of UserVector objects from a list of <n_string> taxnumberprovider. 
            
        Args:
            n_strings (int, optional): number of strings to be considered when creating the UserVector objects. Defaults to 10.
            to_csv (bool, optional): whether to export these strings to a csv. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        corpus = [' '.join(UserVector(i,self.train).strings) for i in tqdm(self.qualifying_users, desc = 'Selecting strings from each user')]
        return corpus 
        
    def reduce_and_plot(self, elbow_range:np.linspace = np.linspace(15, 45, 30, dtype=int)):
        figsize = (10,6)
        n_clusters = self.auto_elbow_method(n_clusters_range=elbow_range)
        tsne = TSNE(n_components=3)
        X_pca = tsne.fit_transform(self.vectorized_corpus)
        
        # Apply Agglomerative Clustering
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg.fit_predict(X_pca)
        
        # Apply KMeans clustering
        
        self.cluster_model.set_params(n_clusters= n_clusters)
        print(self.cluster_model)
        clustering_labels = self.cluster_model.fit_predict(X_pca)
        
        
        print(f"Used {self.cluster_model} to clusterize.")
        # Evaluate clustering using silhouette score
        print("\nSilhouette Scores:")
        print("KMeans:", silhouette_score(X_pca, clustering_labels))
        print("Agglomerative Clustering:", silhouette_score(X_pca, agg_labels))
            
        fig, axes = plt.subplots(nrows=1, ncols=2,figsize = figsize)

        # Visualize KMeans clusters
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clustering_labels, cmap='plasma', marker='.')#, s=70)
        axes[0].set_title('KMeans Clustering')

        # Visualize Agglomerative Clustering clusters
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='plasma', marker='.')#, s=70)
        axes[1].set_title('Agglomerative Clustering')
        
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=2,figsize = figsize)
        
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clustering_labels, cmap='plasma')
        axes[0].set_title('KMeans Clustering')

        # Visualize Agglomerative Clustering clusters in 3D
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=agg_labels, cmap='plasma')
        axes[1].set_title('Agglomerative Clustering')
        plt.show() 
        data_with_clusters = pd.DataFrame({'taxnumberprovider': self.qualifying_users,
                                            'feature_vector': self.corpus.squeeze() if isinstance(self.corpus,pd.DataFrame) else self.corpus,
                                            'Cluster': clustering_labels})
        
        return data_with_clusters
    
    def BERT_vectorize(self, string):
        def preprocess_text(string,
                            filter_long_numbers=True,
                            filter_any_numbers=False,
                            filter_at_sign=True,
                            filter_special_chars=True,
                            special_chars=None):
            # Define los caracteres especiales por defecto
            if special_chars is None:
                special_chars = ['/', '(', ')', '#', '$', '%', '?', '+']

            # Función para aplicar los filtros
            def filter_text(text):
                if filter_long_numbers:
                    text = re.sub(r'\b\d{5,}\b', '', text)
                if filter_any_numbers:
                    text = re.sub(r'\d+', '', text)
                if filter_at_sign:
                    text = re.sub(r'@\w+', '', text)
                if filter_special_chars:
                    special_chars_regex = '[' + re.escape(''.join(special_chars)) + ']'
                    text = re.sub(special_chars_regex, '', text)
                return text

            # Aplica la función de filtro a la columna especificada
            string = filter_text(string)
            return string

        def vectorize(text):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().squeeze().flatten()

        vectorized_data = vectorize(preprocess_text(string))

        return vectorized_data
    
    def BERT_vectorize_corpus(self):
            df = pd.DataFrame({'corpus': self.corpus})
            vectorized_array = []
            
            for text in tqdm(df['corpus'], desc='BERT Vectorization Progress', unit='texts'):
                vectorized_data = self.BERT_vectorize(text)
                vectorized_array.append(vectorized_data)
            
            vectorized_array = np.stack(vectorized_array)
            #print(vectorized_array)
            return vectorized_array
    
    def auto_elbow_method(self,n_clusters_range,_n_init = 5, plot_elbow = True):
        
        # Range of cluster numbers to try
        # Initialize an empty list to store the variance explained by each cluster
        inertia = []

        # Perform K-Means clustering for different values of k
        for n_clusters in tqdm(n_clusters_range,desc='testing clusters in elbow method',unit= 'user'):
            kmeans = KMeans(n_clusters=n_clusters,n_init=_n_init)
            kmeans.fit(self.vectorized_corpus)
            inertia.append(kmeans.inertia_)

        if plot_elbow:
            # Create the Elbow Method graph
            plt.figure()
            plt.plot(n_clusters_range, inertia, marker='o')
            plt.title('Elbow Method for Optimal K')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Variance Explained (Inertia)')
            plt.grid(True)
            plt.show()
        variation = [(inertia[i] - inertia[i+1])/ inertia[i] * 100 for i in range(len(inertia)-1)]
        n_clusters = n_clusters_range[variation.index(max(variation)) + 1]
        print(f"Optimal n_clusters is {int(n_clusters)}")
        return n_clusters
    
    def export_cluster_model_and_data(self):
        print('Exporting cluster model')
        with open(self.save_directory+'/clustering_model.pkl', 'wb') as model_file:
            pickle.dump(self.cluster_model, model_file)
        print('Exporting clusters')
        self.data_with_clusters.to_csv(self.save_directory+'/clusters.csv', index=False)

    def export_vectorizer(self):
         
        print('Exporting vectorizer model')
        # Open the file for writing
        with open(self.save_directory+'/BERT_model.pkl', 'wb') as model_file:
            pickle.dump(self.model, model_file)
            
         
        with open(self.save_directory+'/BERT_tokenizer.pkl', 'wb') as model_file:
            pickle.dump(self.tokenizer, model_file)
        print('Done')
    
    def export_corpus(self):
        print('Exporting vectorized corpus')
        self.corpus = pd.DataFrame(self.corpus)
        self.corpus.to_csv(self.save_directory+'/corpus.csv', index = False)
        
    def export_vectorized_corpus(self):
        print('Exporting vectorized corpus')
        self.vectorized_corpus = pd.DataFrame(self.vectorized_corpus)
        self.vectorized_corpus.to_csv(self.save_directory+'/vectorized_corpus.csv', index = False)
