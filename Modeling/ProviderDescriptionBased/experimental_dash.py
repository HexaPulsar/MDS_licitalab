from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import yaml
from unidecode import unidecode
from utils.modeling_utils import *
from dash import Dash, html, dash_table, dcc
import pandas as pd
import pickle
import os 

with open('Modeling/ProviderDescriptionBased/config/config.yaml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

DEBUG = False

# Incorporate data
df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                encoding="utf-8" , 
                nrows=yaml_data['N_ROWS'])   

file_path_vectorized_c = os.path.join(os.getcwd(), 
                                        'Modeling', 
                                        'ProviderDescriptionBased', 
                                        'csv',
                                        'vectorized_corpus.csv')


file_path_kmeans = os.path.join(os.getcwd(), 
                        'Modeling', 
                        'ProviderDescriptionBased', 
                        'csv',
                        'kmeans_clusters.csv')

vectorized_corpus =pd.read_csv(file_path_vectorized_c)
kmeans_clusters = pd.read_csv(file_path_kmeans)


df['first_two_digits_code'] = df['agileitemsmp_id'].apply(lambda x: str(x)[:2])
df['feature_vector']=  df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())

file_path_vectorizer = os.path.join(os.getcwd(), 
                                'Modeling', 
                                'ProviderDescriptionBased', 
                                'models',
                                'count_vectorized_model.pkl')

with open(file_path_vectorizer, 'rb') as model_file:
    vectorizer = pickle.load(model_file)

app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1(children='Recomendaciones', style={'textAlign':'center'}),
    dcc.Dropdown(df.taxnumberprovider.unique(), value='76.567.318-6', id='dropdown-selection'),
        # DataTable
    dash_table.DataTable(
        id='display-table',
        columns=[],
        data=[],
        style_data={
            'white-space': 'normal',  # Enable word wrapping
            'text-align': 'left',     # Align text to the left
        },
        page_size = 30
    ),
    dcc.Graph(
        id='bar-plot'
    )
])

# Add controls to build the interaction
@callback(
    #Output(component_id='graph-content', component_property='figure'),
    Output('display-table', 'columns'),
    Output('display-table', 'data'),
    Output('bar-plot', 'figure'),
    Input(component_id='dropdown-selection', component_property='value')
)

def update_table(table):
    user = str(table)
    example_user_df = df.query(f'taxnumberprovider == "{user}"') # and adjudicada == True' ) #no es necesario que la licitacion este adjudicada, pero aquí probamos con licitaciones adjudicadas
    if yaml_data['USE_AWARDED']:
        example_user_df = example_user_df.query(f'adjudicada == True' )

    cluster_number = predict_user_item_cluster(example_user_df,
                                            vectorizer,
                                            vectorized_corpus,
                                            yaml_data['ITEMS_TO_CONSIDER'],
                                            kmeans_clusters)
    #matching 
    gg = match_cluster_data_with_agilebuy(_user_df = example_user_df,
                                        _df = df,
                                        _vectorizer = vectorizer,
                                        _vectorized_data= vectorized_corpus,
                                        _items_to_consider=yaml_data['ITEMS_TO_CONSIDER'],
                                        _cluster_method=kmeans_clusters)

    exploration_query = gg.query(f"Cluster == {cluster_number} and taxnumberprovider != '{user}'") 
    save = exploration_query.groupby(by = 'first_two_digits_code').count()['agileitemsproductcategory'].reset_index()
     
    fig = px.bar(save, x='first_two_digits_code', y='agileitemsproductcategory', title='Categorías Presentes')
    data = exploration_query.groupby(by='agilebuyingsdescription').count()['agileitemsproductcategory'].reset_index()
   
    columns = [{'name': col, 'id': col} for col in data.columns]
    data = data.to_dict('records')
    return columns, data,fig

#TODO arreglar ejes del grafico
#TODO agregar vista de solo agile offered items

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

