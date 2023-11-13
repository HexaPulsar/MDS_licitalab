from sklearn.feature_extraction.text import CountVectorizer 
from utils.UserVector import UserVector
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import os 
from sklearn.cluster import KMeans, AgglomerativeClustering
import seaborn as sns
import numpy as np 
import re
from transformers import BertTokenizer, BertModel 

class UserSpaceGenerator(UserVector):
    def __init__(self, DF:pd.DataFrame, save_path:str = os.getcwd(), autoinitialize = True,autosave = True) -> None:
         
        
        self.DF = DF
        self.genDF = DF[['taxnumberprovider','feature_vector','agilebuyingscode']]
        self.vectorizer = CountVectorizer()
        self.corpus,self.qualifying_users = self.generate_corpus()
        self.tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.model = self.model.to('cuda')
        self.vectorized_corpus = self.BERT_vectorize_corpus()

        if autoinitialize:
            #self.tsne_data = self.tsne_reduction()
            #TODO permitir definir el numero de clusters desde la definicion de clase o desde yaml file
            self.kmeans_model, self.data_with_clusters = self.reduce_and_plot('reduce_clusterize')
             
        
        self.save_path = save_path
        if autosave:
            self.export_kmeans_model_and_data()
            self.export_vectorizer()
            self.export_vectorized_corpus() 
        
    def generate_corpus(self, n_strings:int = 10, to_csv:bool = False):
        """generates list of UserVector objects from a list of <n_string> taxnumberprovider. 
            
        Args:
            n_strings (int, optional): number of strings to be considered when creating the UserVector objects. Defaults to 10.
            to_csv (bool, optional): whether to export these strings to a csv. Defaults to False.

        Returns:
            _type_: _description_
        """
        gb = self.genDF.groupby(by =['taxnumberprovider']).agg({'agilebuyingscode':'nunique'})
        gb = gb.sort_values(by = 'agilebuyingscode')
        qualifying_users =  gb[gb['agilebuyingscode'] >= n_strings].index.values
        print(f'Se han removido {round((gb.shape[0] - qualifying_users.shape[0])/gb.shape[0] *100,2)}% de taxnumberproviders, por tener < {n_strings} licitaciones. \n El numero de usuarios para crear el corpus será {qualifying_users.shape[0]}.')
         
        corpus = [' '.join(UserVector(i,self.df).strings) for i in tqdm(qualifying_users, desc = 'Selecting strings from each user')]
         
        #TODO fix path for save
        if to_csv:
            pd.DataFrame(corpus).to_csv('')
        
        return corpus,qualifying_users
        
    
    
    
    def reduce_and_plot(self,method ='clusterize_reduce'):
        if method == 'clusterize_reduce':
            tsne = TSNE()
            n_clusters = self.auto_elbow_method(n_clusters_range=np.linspace(5,15,10,dtype=int))
            
            clustering_method = KMeans(n_clusters= n_clusters,random_state=42)
            cluster_labels = clustering_method.fit_predict(self.vectorized_corpus)
            X_pca =tsne.fit_transform(self.vectorized_corpus)
             
            plt.figure(figsize=(8,8 ))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='hls',s = 20,markers='.',legend=False)
            plt.title('PCA Scatter Plot with Clusters')

            #plt.title('User Space Kmeans Clustering Results')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
             
            plt.show()
            data_with_clusters = pd.DataFrame({'taxnumberprovider':self.qualifying_users,
                                           'feature_vector': self.corpus, 
                                           'Cluster': cluster_labels})
            return clustering_method, data_with_clusters
        
        elif method == 'reduce_clusterize':
                     
            tsne = TSNE()
            n_clusters = self.auto_elbow_method(n_clusters_range=np.linspace(5,45,45,dtype=int))
             
            X_pca =tsne.fit_transform(self.vectorized_corpus)
            clustering_method = KMeans(n_clusters= n_clusters)
            cluster_assignments = clustering_method.fit_predict(X_pca)
            
             
            plt.figure(figsize=(8,8 ))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_assignments, palette='hls',s = 20,markers='.',legend=False)
            plt.title('PCA Scatter Plot with Clusters')

            #plt.title('User Space Kmeans Clustering Results')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
             
            plt.show()
            data_with_clusters = pd.DataFrame({'taxnumberprovider':self.qualifying_users,
                                           'feature_vector': self.corpus, 
                                           'Cluster': cluster_assignments})
            
            return clustering_method, data_with_clusters
        
        elif method == 'both':
            tsne = TSNE()
            n_clusters = self.auto_elbow_method(n_clusters_range=np.linspace(5,45,30,dtype=int))
            print(n_clusters)
             
            X_pca =tsne.fit_transform(self.vectorized_corpus)
            clustering_method = KMeans(n_clusters= n_clusters)
            cluster_assignments = clustering_method.fit_predict(X_pca)
            
             
            plt.figure(figsize=(8,8 ))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_assignments, palette='hls',s = 20,markers='.',legend=False)
            plt.title('PCA Scatter Plot with Clusters')

            #plt.title('User Space Kmeans Clustering Results')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.show() 
            
            tsne = TSNE()
            n_clusters = self.auto_elbow_method(n_clusters_range=np.linspace(5,45,45,dtype=int))
              
            clustering_method = KMeans(n_clusters= n_clusters)
            cluster_labels = clustering_method.fit_predict(self.vectorized_corpus)
            X_pca =tsne.fit_transform(self.vectorized_corpus)
             
            plt.figure(figsize=(8,8 ))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='hls',s = 20,markers='.',legend=False)
            plt.title('PCA Scatter Plot with Clusters')

            #plt.title('User Space Kmeans Clustering Results')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
             
            plt.show()
            
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
        for n_clusters in tqdm(n_clusters_range,desc='testing clusters in elbow method'):
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
    
    def export_kmeans_model_and_data(self):
        print('Exporting Kmeans model')
        with open(self.save_path+'/kmeans_model.pkl', 'wb') as model_file:
            pickle.dump(self.kmeans_model, model_file)
        print('Exporting Kmeans clusters')
        self.data_with_clusters.to_csv(self.save_path+'/kmeans_clusters.csv', index=False)

    def export_vectorizer(self):
        file_path = os.path.join(os.getcwd(), 'model.pkl')
        print('Exporting vectorizer model')
        # Open the file for writing
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
            
        file_path = os.path.join(os.getcwd(), 'tokenizer.pkl')
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.tokenizer, model_file)
        print('Done')
         

    def export_vectorized_corpus(self):
        print('Exporting vectorized corpus')
        self.vectorized_corpus = pd.DataFrame(self.vectorized_corpus)
        self.vectorized_corpus.to_csv(self.save_path+'/vectorized_corpus.csv', index = False)