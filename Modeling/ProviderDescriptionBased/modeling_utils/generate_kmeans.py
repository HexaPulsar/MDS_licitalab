import pandas as pd 
from unidecode import unidecode 
from modeling_utils import *

import yaml
import pickle
import os 

with open('Modeling/ProviderDescriptionBased/config/config.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    for dat in yaml_data:
        print(dat)

DEBUG = True

if __name__ == "__main__":

    print("Loading csv") 
    df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                     encoding="utf-8" , 
                     nrows=yaml_data['N_ROWS'])
    print("Cleaning csv")
    df['first_two_digits_code'] = df['agileitemsmp_id'].apply(lambda x: str(x)[:2])
    df['feature_vector']= df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
    corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())

    file_path_vectorizer = os.path.join(os.getcwd(), 
                                'Modeling', 
                                'ProviderDescriptionBased', 
                                'models',
                                'count_vectorized_model.pkl')
    

    with open(file_path_vectorizer, 'rb') as model_file:
        vectorizer = pickle.load(model_file)
     

    file_path_vectorized_c = os.path.join(os.getcwd(), 
                                        'Modeling', 
                                        'ProviderDescriptionBased', 
                                        'csv',
                                        'vectorized_corpus.csv')
    vectorized_corpus = pd.read_csv(file_path_vectorized_c)

    print('Dimensionality Reduction')
    tsne_data = tsne_reduction(vectorized_corpus)
    auto_n_cluster = auto_elbow_method(tsne_data,
                                     np.linspace(yaml_data['ELBOW_MIN'],yaml_data['ELBOW_MAX'],yaml_data['ELBOW_N'],dtype = int),
                                     plot = DEBUG)  
    print('Generating Kmeans clusters')
    
    kmeans_model, kmeans_clusters = launch_kmeans(auto_n_cluster,
                                    tsne_data,
                                    corpus.unique(),
                                    plot=DEBUG)
    
    file_path_model = os.path.join(os.getcwd(), 
                                'Modeling', 
                                'ProviderDescriptionBased', 
                                'models',
                                'kmeans_model.pkl')
    print('Exporting Kmeans model')
    with open(file_path_model, 'wb') as model_file:
        pickle.dump(kmeans_model, model_file)

    file_path = os.path.join(os.getcwd(), 
                        'Modeling', 
                        'ProviderDescriptionBased', 
                        'csv',
                        'kmeans_clusters.csv')

    kmeans_clusters.to_csv(file_path, index=False)

    print('Done')