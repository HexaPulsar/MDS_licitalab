from collections import defaultdict
import io
from utils.UserVector import UserVector
from utils.UserSpace import UserSpace
from sklearn.metrics.pairwise import euclidean_distances
import os 
import pandas as pd 
import torch  
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np


class RecommenderSystem(UserSpace):
    
    def __init__(self, 
                 train:pd.DataFrame, 
                 test: pd.DataFrame,
                 userspace_data_path:str = None,
                 save_path:str = os.getcwd(),
                 initialize_from:str = '') -> None:
        
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
        
        #TODO agregar log file que guarde metadata
        self.save_path = save_path
        if initialize_from !='':
            super().__init__(train,test,save_path=save_path, initialize_from=initialize_from)
            
        else:
            super().__init__(train,test,userspace_data_path,save_path=save_path)
            #for key, value in vars(self).items():
            #    print(f"{key}: {value}")
        self.intersection  = np.intersect1d(self.find_qualifying_users(train), self.find_qualifying_users(test))
        
    def describe(self):
        """dicc = {'Current directory':,
                'train shape':,
                'test shape':,
                'train qualifying users':,
                'test qualifying users':,
                'Using GPU':,
                }
        """
        vari = vars(self).copy()
        print(vari.keys())
    
    def save_to_directory(self,directory:str):
        directory = os.path.join(self.save_path,directory)
        try:
                # Attempt to create the directory
                os.makedirs(directory, exist_ok=True)
                print(f"Directory '{directory}' created or already exists.")
                
        except Exception as e:
            print(f"Error creating directory: {e}")
        pass
    
    def regenerate_clustering(self,clustering_model,elbow_range:np.linspace = np.linspace(15, 45, 30, dtype=int)):
 
            self.cluster_model = clustering_model
            self.reduce_and_plot(elbow_range = elbow_range)
            self.export_cluster_model_and_data() 
            
   
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
 
    
    def evaluate_users(self):
        """Receives a tax number and evaluates its recommendations
        
        Args:
            taxnumberprovider (str): user to be studied.
        """
        def participaciones_rut(n_strings  = 10):
            # Use defaultdict to simplify the logic
            indices_por_rut = defaultdict(list)
            # Assuming self.test is a pandas DataFrame
            
            df =  self.test[self.test['taxnumberprovider'].isin(self.intersection)]
            for index, row in df.iterrows():
                rut = row['taxnumberprovider']
                indices_por_rut[rut].append(index)

            # Convert the defaultdict to a regular dictionary
            diccionario_participaciones_proovedores = dict(indices_por_rut)
            return diccionario_participaciones_proovedores
        
        
        def evaluar_ruts(dict_participaciones_proovedores,limit_participations:int = 10):
            # Obtener los índices de las filas que cumplen la condición
            dict_participaciones = participaciones_rut()
            lista_ruts_participaron = list(dict_participaciones.keys())
            
            # Iteramos para todos los ruts y lo vamos llenando en una tabla  | rut_proveedor | Cantidad_de_participaciones_reales_del_usuario
            #  | Cantidad_de_sugerencias_del_cluster | similitudes | cantidad_de_similitudes | Cluster
            columnas = ["rut_proveedor", "Cantidad_de_participaciones_reales_del_usuario", "Cantidad_de_sugerencias_del_cluster","similitudes", "cantidad_de_similitudes", "Cluster"]
            #Metricas_por_usuario = pd.DataFrame()
            Metricas_por_usuario = pd.DataFrame(columns=columnas)

            
            
            for participante in tqdm(lista_ruts_participaron, desc="Processing RUTs"):
                with io.StringIO() as buf, redirect_stdout(buf):
                    tabla_rutmascluster = self.predict(participante)
                cluster = tabla_rutmascluster[0] 
                tabla = tabla_rutmascluster[1]

                # Obtener los índices de las filas que cumplen la condición 
                condicion = tabla['Cluster'] == cluster
                indices_filtrados = tabla.index[condicion].tolist()

                # Encuentra la intersección de las dos listas
                similitudes = set(diccionario_participaciones_proovedores[participante]) & set(indices_filtrados)

                # Crear un diccionario con los datos de la fila
                nueva_fila = pd.DataFrame({
                                            "rut_proveedor": [participante],
                                            "Cantidad_de_participaciones_reales_del_usuario": [len(dict_participaciones_proovedores[participante])],
                                            "Cantidad_de_sugerencias_del_cluster": [len(indices_filtrados)],
                                            "similitudes": [similitudes],
                                            "cantidad_de_similitudes": [len(similitudes)],
                                            "Cluster": [cluster]
                                        })
                # Agregar la fila al DataFrame
                Metricas_por_usuario = pd.concat([Metricas_por_usuario, nueva_fila], ignore_index=True)
                #Metricas_por_usuario = Metricas_por_usuario.append(nueva_fila, ignore_index=True)
            return Metricas_por_usuario

   
        def rut_metricas_iniciales(Tabla):

            # Metricas por persona

            Tabla['self_precision_%'] = Tabla['cantidad_de_similitudes'] / Tabla['Cantidad_de_sugerencias_del_cluster']  *100 # LO DEJAMOS EN % PARA NO TENER PROBLEMAS CON LOS DECIMALES DE PYTHON
            Tabla['self_recall_%'] = Tabla['cantidad_de_similitudes'] / Tabla['Cantidad_de_participaciones_reales_del_usuario'] * 100 # LO DEJAMOS EN % PARA NO TENER PROBLEMAS CON LOS DECIMALES DE PYTHON
            Tabla['self_f1-score_%'] =Tabla['self_f1-score_%'] = (2 * Tabla['self_precision_%'] * Tabla['self_recall_%'] / (Tabla['self_precision_%'] + Tabla['self_recall_%'] + 0.000001))  # LO DEJAMOS EN % PARA NO TENER PROBLEMAS CON LOS DECIMALES DE PYTHON

            # Hacemos una agrupación auxiliar para conocer el peso de cada persona
            agregado_aux = Tabla.groupby('Cluster')['Cantidad_de_participaciones_reales_del_usuario'].sum().reset_index()
            
            # Cambiar el nombre de la columna usando el método rename
            agregado_aux = agregado_aux.rename(columns={'Cantidad_de_participaciones_reales_del_usuario': 'Total_participaciones_cluster'})
            df_final = pd.merge(Tabla, agregado_aux, on='Cluster', how='left')

            # Calculamos el peso finalmente
            df_final['peso_persona_en_el_cluster'] = df_final['Cantidad_de_participaciones_reales_del_usuario'] / df_final['Total_participaciones_cluster']

            # Ponderamos cada métrica

            df_final['self_precision_ponderada_%'] = df_final['self_precision_%'] * df_final['peso_persona_en_el_cluster'] 
            df_final['self_recall_ponderada_%'] = df_final['self_recall_%'] * df_final['peso_persona_en_el_cluster'] 
            df_final['self_f1-score_ponderada_%'] =  df_final['self_f1-score_%'] * df_final['peso_persona_en_el_cluster'] 

            return df_final
        
        def rut_to_clusters_table(rut_table):

            result = rut_table.groupby('Cluster').agg({'self_precision_ponderada_%': 'sum', 'self_recall_ponderada_%': 'sum', 'self_f1-score_ponderada_%': 'sum', 'cantidad_de_similitudes': 'sum', 'Cantidad_de_participaciones_reales_del_usuario': 'sum', 'Cantidad_de_sugerencias_del_cluster': 'min', 'rut_proveedor' : 'count' }).reset_index()
            result = result.rename(columns={'self_precision_ponderada_%': 'self_precision_%', 'self_recall_ponderada_%': 'self_recall_%', 'self_f1-score_ponderada_%': 'self_f1-score_%', 'cantidad_de_similitudes': 'Licitaciones_calzadas', 'Cantidad_de_participaciones_reales_del_usuario': 'Total_participaciones_de_cada_rut_en_cluster', 'rut_proveedor' : 'Cantidad_de_provedores_en_cluster', 'Cantidad_de_sugerencias_del_cluster' : 'Cantidad_de_sugerencias_para_el_cluster'})
            
            #  Calculamos el peso por cantidad de proovedores para cada cluster
            
            result['Peso_cluster'] = result['Cantidad_de_provedores_en_cluster'] / result['Cantidad_de_provedores_en_cluster'].sum() 
            
            # Ponderamos para cada cluster

            result['self_precision_ponderada_%'] = result['self_precision_%']*result['Peso_cluster']
            result['self_recall_ponderada_%'] = (result['self_recall_%'] *result['Peso_cluster']) 
            result['self_f1-score_ponderada_%'] =  result['self_f1-score_%'] * result['Peso_cluster'] 
            return result
        
        diccionario_participaciones_proovedores = participaciones_rut()
        
        def cluster_to_resumen(cluster_info):
            """
            Calcula métricas globales ponderadas para un cluster.

            Args:
            - cluster_info (pandas.DataFrame): Información del cluster.

            Returns:
            - dict: Diccionario con métricas globales.
            """

            
            # Suma de métricas ponderadas para todas las instancias en el cluster
            self_precision_porcentual_global = cluster_info['self_precision_ponderada_%'].sum()
            self_recall_porcentual_global = cluster_info['self_recall_ponderada_%'].sum()
            self_f1_score_porcentual_global = cluster_info['self_f1-score_ponderada_%'].sum()

            # Suma de participaciones y licitaciones calzadas
            total_participaciones_todos = cluster_info['Total_participaciones_de_cada_rut_en_cluster'].sum()
            total_licitaciones_calzadas = cluster_info['Licitaciones_calzadas'].sum()



            # Calcula el ratio de participaciones calzadas
            ratio_par_calz = total_licitaciones_calzadas / total_participaciones_todos

            # Devuelve las métricas globales en un diccionario
            resumen = {
                'self_precision_porcentual_global': self_precision_porcentual_global,
                'self_recall_porcentual_global': self_recall_porcentual_global,
                'self_f1_score_porcentual_global': self_f1_score_porcentual_global,
                'total_participaciones_todos': total_participaciones_todos,
                'total_licitaciones_calzadas': total_licitaciones_calzadas,
                'ratio_par_calz': ratio_par_calz
            }

            # Top 3 de cluster
            # Cantidad de cluster con métricas 0

            return resumen
        dict_participaciones_proovedores = participaciones_rut()
        self.a =  evaluar_ruts(dict_participaciones_proovedores)
        self.b = rut_metricas_iniciales(self.a)
        self.c = rut_to_clusters_table(self.b)
        self.d = cluster_to_resumen(self.c)
        
        
        #return b
 