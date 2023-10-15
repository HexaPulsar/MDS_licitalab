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

if __name__ == "__main__":

    print("Loading csv")
    df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                     encoding="utf-8" , 
                     nrows=yaml_data['N_ROWS'])
    print("Cleaning csv")
    df['first_two_digits_code'] = df['agilebuyingscode'].apply(lambda x: x[:2])
    df['feature_vector']=  df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
    corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())
    vectorizer, vectorized_corpus = generate_vectorized_corpus(corpus)
    #print(vectorized_corpus)
    df_vectorized_corpus =  pd.DataFrame.sparse.from_spmatrix(vectorized_corpus)
    df_vectorized_corpus.to_csv('vectorized_corpus.csv', index = False)

    with open('count_vectorizer_model.pkl', 'wb') as model_file:
        pickle.dump(vectorizer, model_file)
    print('Succesfully generated corpus.')
    