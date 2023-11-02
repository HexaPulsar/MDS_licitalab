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
from Modeling.ProviderDescriptionBased.modeling_utils.modeling_utils import *
import yaml

with open('Modeling/ProviderDescriptionBased/config.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

DEBUG = yaml_data['DEBUG']

def main():

    #TODO remove corpus generation from main. It should only load already constructed csr matrix from a csv.
    print("Loading csv")
    df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                     encoding="utf-8" , 
                     nrows=yaml_data['N_ROWS'])
    print("Cleaning csv")
    df['first_two_digits_code'] = df['agilebuyingscode'].apply(lambda x: x[:2])
    df['feature_vector']=  df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
    corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())

    vectorizer, vectorized_corpus = generate_vectorized_corpus(corpus)
    print('Dimensionality Reduction')
    tsne_data = tsne_reduction(vectorized_corpus)
    auto_n_cluster = auto_elbow_method(tsne_data,
                                     yaml_data['ELBOW_RANGE_LINSPACE'],
                                     plot = DEBUG)  
     
    kmeans_clusters = launch_kmeans(auto_n_cluster,
                                    tsne_data,
                                    corpus.unique(),
                                    plot=DEBUG)
    #print("merging clusters to original dataframe")
    #merged_df = merge_cluster_data_to_df(kmeans_clusters,df)
    
    #single user lookup
    user = str(input('rut del usuario para el cual se quiere recomendar:'))
    example_user_df = df.query(f'taxnumberprovider == "{user}" and adjudicada == True' ) #no es necesario que la licitacion este adjudicada, pero aquí probamos con licitaciones adjudicadas
    if yaml_data['USE_AWARDED']:
        example_user_df = example_user_df.query(f'adjudicada == True' )
    cluster_number = predict_user_item_cluster(example_user_df,
                                               vectorizer,
                                               vectorized_corpus,
                                               items_to_consider=yaml_data['ITEMS_TO_CONSIDER'],
                                               cluster_method= kmeans_clusters)
    #matching
    #cluster_data = get_cluster_data(cluster_number, kmeans_clusters)
    gg = match_cluster_data_with_agilebuy(_user_df = example_user_df,
                                          _df = df,
                                          _vectorizer = vectorizer,
                                          _vectorized_data= vectorized_corpus,
                                          _items_to_consider=yaml_data['ITEMS_TO_CONSIDER'],
                                          _cluster_method=kmeans_clusters)
    exploration_query = gg.query(f"Cluster == {cluster_number} and taxnumberprovider != '{user}'")
    #plot_rec_categories(user,exploration_query)
    return gg

if __name__ == '__main__':
    main()