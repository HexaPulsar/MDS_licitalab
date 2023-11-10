from sklearn.feature_extraction.text import CountVectorizer
from utils.UserVector import UserVector
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import os 
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np 
import re
from transformers import BertTokenizer, BertModel


class UserSpaceGenerator(UserVector):
    def __init__(self, DF:pd.DataFrame, save_path:str = os.getcwd(), autoinitialize = True,autosave = True) -> None:
        self.DF = DF
        
        self.vectorizer = CountVectorizer()
        self.corpus,self.qualifying_users = self.generate_corpus()
        self.tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.vectorized_corpus = self.BERT_vectorize_corpus()
        print(self.vectorized_corpus.shape)
        
        if autoinitialize:
            self.tsne_data = self.tsne_reduction()
            #TODO permitir definir el numero de clusters desde la definicion de clase o desde yaml file
            self.kmeans_model, self.data_with_clusters = self.launch_kmeans(30,
                                                                            self.tsne_data,
                                                                            plot=True)
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
        gb = self.DF.groupby(by =['taxnumberprovider']).agg({'agilebuyingscode':'nunique'})
        gb = gb.sort_values(by = 'agilebuyingscode')
        qualifying_users =  gb[gb['agilebuyingscode'] >= n_strings].index.values
        print(f'Se han removido {round((gb.shape[0] - qualifying_users.shape[0])/gb.shape[0] *100,2)}% de taxnumberproviders, por tener < {n_strings} licitaciones. \n El numero de usuarios para crear el corpus será {qualifying_users.shape[0]}.')
         
        corpus = [' '.join(UserVector(i,self.df).strings) for i in tqdm(qualifying_users, desc = 'Selecting strings from each user')]
         
        #TODO fix path for save
        if to_csv:
            pd.DataFrame(corpus).to_csv('')
        
        return corpus,qualifying_users#{'corpus':corpus, 'qualifying_users':qualifying_users}
        
    
    def BERT_vectorize(self,string):
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
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy().squeeze().flatten()
        
        vectorized_data = vectorize(preprocess_text(string))

        return vectorized_data
         
    
    def BERT_vectorize_corpus(self):
        df = pd.DataFrame({'corpus':self.corpus}) 
        x = df['corpus'].apply(lambda x: self.BERT_vectorize(x))
        vectorized_array = np.stack(x)
        print(vectorized_array)
        return vectorized_array

    
    def tsne_reduction(self):
        
        tsne = TSNE(n_components=2, init = 'random',random_state=42)
        tsne_data = tsne.fit_transform(self.vectorized_corpus) 
        return tsne_data

    def auto_elbow_method(data,n_clusters_range:np.linspace,_n_init = 5, _random_state = 42, plot = False):
        """auto elbow method for kmeans

        Args:
            data (_type_): data to clusterize
            n_clusters_range (np.linspace): range of n_clusters for elbow method
            _n_init (int, optional): number of kmeans init. Defaults to 5.
            _random_state (int, optional): for replicability. Defaults to 42.
            plot (bool, optional): plots elbow method. Defaults to False.

        Returns:
            _n_clusters_ (int): returns number of clusters for kmeans 
        """
        # Range of cluster numbers to try
        # Initialize an empty list to store the variance explained by each cluster
        inertia = []

        # Perform K-Means clustering for different values of k
        for n_clusters in tqdm(n_clusters_range,desc='testing clusters in elbow method'):
            kmeans = KMeans(n_clusters=n_clusters,n_init=_n_init,random_state=_random_state)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)

        if plot:
            # Create the Elbow Method graph
            plt.figure(figsize=(20, 8))
            plt.plot(n_clusters_range, inertia, marker='o')
            plt.title('Elbow Method for Optimal K')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Variance Explained (Inertia)')
            plt.grid(True)
            plt.show()

        variation = [(inertia[i] - inertia[i+1])/ inertia[i] * 100 for i in range(len(inertia)-1)]
        n_clusters = n_clusters_range[variation.index(max(variation)) + 1]
        print(f"El número óptimo de clusters es {n_clusters}")

        return n_clusters

    def launch_kmeans(self,n_clusters:int,data, plot:bool = False):
        """Launches Kmeans

        Args:
            n_clusters (int): number of clusters
            data (_type_): data to clusterize
            corpus (_type_): original corpus
            plot (bool, optional): plot kmeans. Defaults to False.

        Returns:
            kmeans (Kmeans): Kmeans model
            data_with_clusters (DataFrame): dataframe of corpus with asociated cluster
        """
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(data)
        #cluster_assignments
        # Create a scatter plot with different colors for each cluster

        if plot:
            plt.figure(figsize=(8,8 ))
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=cluster_assignments, s=70,marker = '.',palette='hls', legend=False)
            plt.title('User Space Kmeans Clustering Results')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            
            plt.show()
            #plt.savefig('kmeans_plot.png')
        #print(corpus.shape,cluster_assignments.shape)
        # Create a DataFrame to associate original strings with clusters
        data_with_clusters = pd.DataFrame({'taxnumberprovider':self.qualifying_users,
                                           'feature_vector': self.corpus, 
                                           'Cluster': cluster_assignments})
        return kmeans,data_with_clusters
    
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