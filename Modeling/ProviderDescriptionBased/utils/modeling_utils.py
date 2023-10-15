import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from unidecode import unidecode 
from sklearn.metrics.pairwise import euclidean_distances
import scipy 

def generate_vectorized_corpus(corpus: pd.Series):
    corpus = corpus.unique()
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(corpus) 
    return vectorizer, vectorized_data

def generate_vector(str:str,vectorized_corpus: scipy.sparse._csr.csr_matrix):
    vector = vectorized_corpus.transform([str])
    # Convert the result to a dense array for inspection
    vector_array = vector.toarray().flatten()
    #print(vector_array.shape)
    return vector_array

def tsne_reduction(data):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)#.toarray())
    return tsne_data

def auto_elbow_method(data,n_clusters_range:np.linspace,_n_init = 5, _random_state = 42, plot = False):
    # Range of cluster numbers to try
    # Initialize an empty list to store the variance explained by each cluster
    inertia = []

    # Perform K-Means clustering for different values of k
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters,n_init=_n_init,random_state=_random_state)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    if plot:
        # Create the Elbow Method graph
        plt.figure(figsize=(10, 4))
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

def launch_kmeans(n_clusters:int,data,corpus, plot:bool = False):
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(data)
    #cluster_assignments
    # Create a scatter plot with different colors for each cluster

    if plot:
        fig,ax = plt.subplots(figsize=(8, 8))
        for i in range(n_clusters):
            ax.scatter(data[cluster_assignments == i, 0], data[cluster_assignments == i, 1], label=f'Cluster {i}', marker= '.')
        plt.show()
    print(corpus.shape,cluster_assignments.shape)
    # Create a DataFrame to associate original strings with clusters
    data_with_clusters = pd.DataFrame({'feature_vector': corpus, 'Cluster': cluster_assignments})
    return kmeans,data_with_clusters

def launch_dbscan(eps:float or int,min_samples:int, data, str_corpus, plot:bool = False):
    dbscan = DBSCAN(eps=6, min_samples=80)
    cluster_assignments = dbscan.fit_predict(data)
    data_with_clusters = pd.DataFrame({'feature_vector': str_corpus, 'Cluster': cluster_assignments})
    if plot:
        sns.set(style='darkgrid')
        plt.figure(figsize=(8,8 ))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=cluster_assignments, s=80,marker = '.',palette='plasma',)
        plt.title('DBSCAN Clustering Results')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
    return data_with_clusters 

def merge_cluster_data_to_df(clusters_df,str_corpus_df):
    print(str_corpus_df.columns)
    str_corpus_df['feture_vector'] = str_corpus_df['feature_vector'].apply(lambda x:  unidecode(x).lower())
    return str_corpus_df.merge(clusters_df, on='feature_vector', how='left')
            
def predict_user_item_cluster(df,_vectorizer,_vectorized_data,_items_to_consider,_cluster_method):
    #TODO hacer sampling en items to consider
    ten = df['feature_vector'].unique()[:_items_to_consider]
    ten = " ".join(ten).lower()
    vectorized_ten = generate_vector(ten,_vectorizer)
    distances = euclidean_distances(vectorized_ten.reshape(1,-1), _vectorized_data)
    nearest_core_point_cluster = _cluster_method['Cluster'][distances.argmin()]
    print(f"Unseen data point belongs to cluster {nearest_core_point_cluster}")
    return nearest_core_point_cluster

def get_cluster_data(cluster, cluster_method):
    cluster_data = cluster_method[cluster_method['Cluster'] == cluster]
    #print(cluster_data.shape)
    return cluster_data

def match_cluster_data_with_agilebuy(_user_df,_df,_vectorizer,_vectorized_data,_items_to_consider,_cluster_method):
    match_df = get_cluster_data(predict_user_item_cluster(_user_df,
                                                          _vectorizer,
                                                          _vectorized_data,
                                                          _items_to_consider,
                                                          _cluster_method),
                                                          _cluster_method)

    gg = _df.merge(match_df,on='feature_vector', how='left')
    #print(gg)
    return gg

def plot_vector(vector):
    fig,ax = plt.subplots(figsize= (4,3))
    x = np.linspace(0,len(vector),len(vector))
    ax.bar(x,vector)
    plt.show() 

def plot_rec_categories(user,exploration_query):
    fig, ax = plt.subplots()  # Create a Matplotlib figure and an AxesSubplot
    ax = exploration_query.groupby(by='first_two_digits_code').count()['organismosolicitante'].plot.bar(ax=ax, title=f'Principales categorías recomendadas para el "{user}"')
    # Now, 'fig' contains the entire figure with your bar plot
    return ax