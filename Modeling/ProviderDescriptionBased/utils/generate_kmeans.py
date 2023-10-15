
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from unidecode import unidecode
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
from modeling_utils import *
import yaml
import pickle

with open('Modeling/ProviderDescriptionBased/config.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

DEBUG = False

if __name__ == "__main__":

    #TODO remove corpus generation from main. It should only load already constructed csr matrix from a csv.
    print("Loading csv")
    df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                     encoding="utf-8" , 
                     nrows=yaml_data['N_ROWS'])
    print("Cleaning csv")
    df['first_two_digits_code'] = df['agilebuyingscode'].apply(lambda x: x[:2])
    df['feature_vector']=  df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
    corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())

    with open('count_vectorizer_model.pkl', 'rb') as model_file:
        vectorizer = pickle.load(model_file)

    vectorized_corpus = pd.read_csv('vectorized_corpus.csv')
    print('Dimensionality Reduction')
    tsne_data = tsne_reduction(vectorized_corpus)
    auto_n_cluster = auto_elbow_method(tsne_data,
                                     yaml_data['ELBOW_RANGE_LINSPACE'],
                                     plot = DEBUG)  
     
    kmeans_model, kmeans_clusters = launch_kmeans(auto_n_cluster,
                                    tsne_data,
                                    corpus.unique(),
                                    plot=DEBUG)
    with open('count_vectorizer_model.pkl', 'wb') as model_file:
        pickle.dump(kmeans_model, model_file)
    
    kmeans_clusters.to_csv('kmeans_clusters.csv',index = False)

    