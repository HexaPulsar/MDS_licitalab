import pandas as pd 
from unidecode import unidecode
from modeling_utils import *
import yaml
import pickle
import os 
from tqdm import tqdm

with open('Modeling/ProviderDescriptionBased/config/config.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    for data in yaml_data:
        print(data,':', yaml_data[data])

if __name__ == "__main__":
    print("Loading csv")
    df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                     encoding="utf-8" , 
                     nrows=yaml_data['N_ROWS'])
    print("Cleaning csv")
    df['first_two_digits_code'] = df['agileitemsmp_id'].apply(lambda x: str(x)[:2])
    df['feature_vector']=  df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
    corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())
    print('Vectorizing corpus')
    vectorizer, vectorized_corpus = generate_vectorized_corpus(corpus)
    #TODO es necesario pasar de csr matriz a df?  otherwise remove
    df_vectorized_corpus =  pd.DataFrame.sparse.from_spmatrix(vectorized_corpus)
    
    tqdm.pandas(desc="Writing CSV")
    df_vectorized_corpus.to_csv('Modeling/ProviderDescriptionBased/csv/vectorized_corpus.csv', index = False)

    file_path = os.path.join(os.getcwd(), 
                             'Modeling', 
                             'ProviderDescriptionBased', 
                             'models',
                             'count_vectorized_model.pkl')
    print('Exporting vectorizer model')
    # Open the file for writing
    with open(file_path, 'wb') as model_file:
        pickle.dump(vectorizer, model_file)
    print('Done')