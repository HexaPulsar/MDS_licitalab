from collections import defaultdict
import io
from .UserVector import UserVector
from .UserSpace import UserSpace
from sklearn.metrics.pairwise import euclidean_distances
import os 
import pandas as pd 
import torch  
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np
from .ScorerV1 import ScorerV1

class RecommenderSystem(UserSpace):
    
    def __init__(self, 
                 train:pd.DataFrame, 
                 test: pd.DataFrame,
                 userspace_data_path:str = None,
                 save_path:str = os.getcwd(),
                 elbow_range:np.linspace =  np.linspace(150,250,15,dtype = int)) -> None:
        
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
         
        self.save_path = save_path
 
    
        super().__init__(train,test,elbow_range= elbow_range,save_path=save_path)
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
            #dict_participaciones = participaciones_rut()
            lista_ruts_participaron = list(dict_participaciones_proovedores.keys())
            
            # Iteramos para todos los ruts y lo vamos llenando en una tabla  | rut_proveedor | Cantidad_de_participaciones_reales_del_usuario
            #  | Cantidad_de_sugerencias_del_cluster | similitudes | cantidad_de_similitudes | Cluster
            columnas = ["rut_proveedor", 
                        "Cantidad_de_participaciones_reales_del_usuario", 
                        "Cantidad_de_sugerencias_del_cluster",
                        "similitudes", "cantidad_de_similitudes",
                        "Cluster"]
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


    def scores(self,train,test):
        def tabla_a_tabla_scores(df, condicion_min_xx, condicion_min_xxyy):

            # Primero nos aseguramos que esten las columnas necesarias que son
            df = df[['taxnumberprovider', 'adjudicada', 'agileitemsmp_id']]

            # Cambiar el tipo de la columna 'agileitemsmp_id' a string
            df.loc[:, 'agileitemsmp_id'] = df['agileitemsmp_id'].astype(str)

            # Crear una nueva columna 'xx' utilizando str.slice
            df.loc[:, 'xx'] = df['agileitemsmp_id'].str.slice(0, 2)
            df.loc[:, 'xxyy'] = df['agileitemsmp_id'].str.slice(0,4)

            # Ahora agrupamos para obtener el conteo y la suma de adjudicadas por tipo de compra agil
            
            # Para xx
            grupo1 = df.groupby(['taxnumberprovider','xx'], group_keys=False).apply(lambda x:x)#.agg({'adjudicada': ['sum', 'count']})
            grupo1.loc[:,'xx'] = grupo1.loc[:,'xx'].astype(int)
            result = grupo1.groupby(['taxnumberprovider', 'xx']).agg({'taxnumberprovider': 'count', 'adjudicada': 'sum'}).rename(columns={'taxnumberprovider': 'count'}).reset_index()
            result = result.rename(columns={
                'taxnumberprovider	': 'taxnumberprovider',
                'xx': 'xx',
                'count': 'participaciones_xx',
                'adjudicada': 'ganadas_xx'
                
            })
            ## Para xxyy

            grupo2 = df.groupby(['taxnumberprovider','xxyy'], group_keys=False).apply(lambda x:x)#.agg({'adjudicada': ['sum', 'count']})
            grupo2.loc[:,'xxyy'] = grupo1.loc[:,'xxyy'].astype(int)
            result2 = grupo2.groupby(['taxnumberprovider', 'xxyy']).agg({'taxnumberprovider': 'count', 'adjudicada': 'sum'}).rename(columns={'taxnumberprovider': 'count'}).reset_index()
            result2 = result2.rename(columns={
                'taxnumberprovider	': 'taxnumberprovider',
                'xxyy': 'xxyy',
                'count': 'participaciones_xxyy',
                'adjudicada': 'ganadas_xxyy'
                
            })
            tabla_score = pd.merge(result,result2, on='taxnumberprovider', how='left')

            # Creamos el score, basado en la cantidad de adjudicaciones
            tabla_score = tabla_score.loc[tabla_score['participaciones_xx'] >= condicion_min_xx]
            tabla_score = tabla_score.loc[tabla_score['participaciones_xxyy'] >= condicion_min_xxyy]
            tabla_score['%_ganado_xx'] = tabla_score['ganadas_xx'] / tabla_score['participaciones_xx'] 
            tabla_score['%_ganado_xxyy'] = tabla_score['ganadas_xxyy'] / tabla_score['participaciones_xxyy'] 
            tabla_score['%_ganar_general'] = (tabla_score['%_ganado_xx']+ tabla_score['%_ganado_xxyy']) /2

            return tabla_score


        def tabla_score_a_dummies(df_score):
            df_score['score'] = df_score['%_ganar_general']
            # Escalamos todos los scores entre 1 y 90

            # Supongamos que tienes un DataFrame llamado 'df' y la columna que deseas escalar se llama 'columna'
            x_min = df_score['score'].min()
            x_max = df_score['score'].max()

            df_score['score'] = ((df_score['score'] - x_min) * (90 - 1) / (x_max - x_min)) + 1

            # Agregamos ruido
            np.random.seed(425)
            df_score['score'] = df_score['score'] + np.random.random() # sirve el ruido especialmente para los casos donde es 0% de adjudicacion
            df_score['score'] = df_score['score'] 

            # Creamos la tabla a la que se le hara todo, 
            data2 =  df_score[['taxnumberprovider','xx','xxyy','score']]

            # Extraer las columnas que se convertirán en variables dummy

            df_dummies = pd.get_dummies(data2['taxnumberprovider'], prefix='rut')
            df_dummies2 = pd.get_dummies(data2['xx'], prefix='xx')
            df_dummies3 = pd.get_dummies(data2['xxyy'], prefix='xxyy')

            # Eliminar las columnas originales
            data2 = data2.drop(columns=['taxnumberprovider'])
            data2 = data2.drop(columns=['xx'])
            data2 = data2.drop(columns=['xxyy'])

            # Concatenar las variables dummy al DataFrame original
            data2 = pd.concat([data2, df_dummies], axis=1)
            data2 = pd.concat([data2, df_dummies2], axis=1)
            data2 = pd.concat([data2, df_dummies3], axis=1)

            return data2 



        def verificar_ruts_tt_y_clases(df1, df2):   # train es marzo, # test es abril
            # Obtener las columnas que coinciden en ambos DataFrames
            columnas_coincidentes = df1.columns.intersection(df2.columns)

            # Calcular las columnas que se van a eliminar en cada DataFrame
            columnas_a_eliminar_df1 = df1.columns.difference(columnas_coincidentes)
            columnas_a_eliminar_df2 = df2.columns.difference(columnas_coincidentes)

            # Eliminar las columnas que no coinciden en ambos DataFrames
            df1 = df1[columnas_coincidentes]
            df2 = df2[columnas_coincidentes]

            # Calcular y mostrar la cantidad de columnas eliminadas en cada DataFrame
            columnas_eliminadas_df1 = len(columnas_a_eliminar_df1)
            columnas_eliminadas_df2 = len(columnas_a_eliminar_df2)
            
            print(f"Se eliminaron {columnas_eliminadas_df1} columnas en DataFrame 1") #{', '.join(columnas_a_eliminar_df1)}.")
            print(f"Se eliminaron {columnas_eliminadas_df2} columnas en DataFrame 2") #{', '.join(columnas_a_eliminar_df2)}.")

            # Eliminar las filas cuya suma no sea igual a 3 (excluyendo la primera columna)
            df1 = df1[df1.iloc[:, 1:].sum(axis=1) == 3]
            df2 = df2[df2.iloc[:, 1:].sum(axis=1) == 3]

            return df1, df2

        def prediccion_a_dict(predicciones, datos):

            # Crear el DataFrame con la lista de predicciones
            df1 = pd.DataFrame(predicciones)

            # Asignar el nombre "score" a la primera columna
            df1.columns = ["score"] + list(df1.columns[1:])

            df2 = pd.DataFrame(datos)

            # Reiniciar índices antes de la concatenación
            #df1_reset = df1.reset_index(drop=True)
            df2_reset = df2.reset_index(drop=True)

            dummies_taxnumberprovider = df2_reset.filter(regex='^rut_')
            dummies_xx = df2_reset.filter(regex='^xx_')
            dummies_xxyy = df2_reset.filter(regex='^xxyy_')

            # Crear un nuevo DataFrame con una sola columna
            df_ruts = pd.DataFrame(columns=["rut"])

            # Iterar sobre las filas del DataFrame original
            for index, row in dummies_taxnumberprovider.iterrows():
                # Iterar sobre las columnas del DataFrame original
                for col in dummies_taxnumberprovider.columns:
                    # Si el valor en la celda es True, agregar el "rut" a la nueva columna
                    if row[col]:
                        df_ruts.loc[len(df_ruts)] = [col] 

            df_ruts["rut"] = df_ruts["rut"].apply(lambda x: x[4:])            


            #############
            # Buscamos el xx
            df_xx = pd.DataFrame(columns=["xx"])

            # Iterar sobre las filas del DataFrame original
            for index, row in dummies_xx.iterrows():
                # Iterar sobre las columnas del DataFrame original
                for col in dummies_xx.columns:
                    # Si el valor en la celda es True, agregar el "rut" a la nueva columna
                    if row[col]:
                        df_xx.loc[len(df_xx)] = [col]

            df_xx["xx"] = df_xx["xx"].apply(lambda x: x[3:])                              

            
            ######
            # Buscamos el xxyy
        
            df_xxyy = pd.DataFrame(columns=["xxyy"])

            # Iterar sobre las filas del DataFrame original
            for index, row in dummies_xxyy.iterrows():
                # Iterar sobre las columnas del DataFrame original
                for col in dummies_xxyy.columns:
                    # Si el valor en la celda es True, agregar el "rut" a la nueva columna
                    if row[col]:
                        df_xxyy.loc[len(df_xxyy)] = [col]

            df_xxyy["xxyy"] = df_xxyy["xxyy"].apply(lambda x: x[5:])



            # Por ultimo concatenamos todo al final
            tabla_predicciones = pd.concat([df1, df_ruts, df_xx, df_xxyy], axis=1)
        

            return tabla_predicciones
        
        pruebam = tabla_a_tabla_scores(train, 20, 20)
        pruebaa = tabla_a_tabla_scores(test, 20, 20)
        media_columna = pruebam['%_ganar_general'].mean()
        std_columna = pruebam['%_ganar_general'].std()

        # Imprime los resultados
        print(f'Media de la columna: {media_columna}')
        print(f'Desviación estándar de la columna: {std_columna}')
        scores_marzo = tabla_score_a_dummies(pruebam)
        scores_abril = tabla_score_a_dummies(pruebaa)
        # Imprime los resultados
        print(f'Media de la columna: {media_columna}')
        print(f'Desviación estándar de la columna: {std_columna}')
        conteo_valores = scores_marzo['score'].value_counts().sort_index(ascending=False)
        scores_marzo_2 , scores_abril_2 = verificar_ruts_tt_y_clases(scores_marzo, scores_abril)
            
        X_marzo = scores_marzo_2.drop('score', axis=1)
        X_train = X_marzo
        y_marzo = scores_marzo_2['score']
        y_train = y_marzo

        X_abril = scores_abril_2.drop('score', axis=1)
        X_test = X_abril
        y_abril = scores_abril_2['score']
        y_test = y_abril
        launch_model = ScorerV1(X_train,y_train,X_test,y_test)
        
        resulting_dict = prediccion_a_dict(launch_model.y_pred, X_test)
        return resulting_dict   