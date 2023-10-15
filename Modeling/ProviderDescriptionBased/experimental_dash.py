from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import yaml
from unidecode import unidecode
from utils.modeling_utils import *
from dash import Dash, html, dash_table, dcc
import pandas as pd
import pickle

with open('Modeling/ProviderDescriptionBased/config.yaml', 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
DEBUG = False

# Incorporate data
df = pd.read_csv(yaml_data['PATH_TO_CSV'],
                encoding="utf-8" , 
                nrows=yaml_data['N_ROWS'])      
vectorized_corpus = pd.read_csv('C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\vectorized_corpus.csv')    
kmeans_clusters = pd.read_csv('C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\kmeans_clusters_test.csv')


df['first_two_digits_code'] = df['agilebuyingscode'].apply(lambda x: x[:2])
df['feature_vector']=  df['first_two_digits_code'] + ' '+ df['agileoffereditemsdescripcionofertada']
corpus = df['feature_vector'].apply(lambda x: unidecode(x).lower())


with open('C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\count_vectorizer_model.pkl', 'rb') as model_file:
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
    #cluster_data = get_cluster_data(cluster_number, kmeans_clusters)
    gg = match_cluster_data_with_agilebuy(_user_df = example_user_df,
                                        _df = df,
                                        _vectorizer = vectorizer,
                                        _vectorized_data= vectorized_corpus,
                                        _items_to_consider=yaml_data['ITEMS_TO_CONSIDER'],
                                        _cluster_method=kmeans_clusters)
    exploration_query = gg.query(f"Cluster == {cluster_number} and taxnumberprovider != '{user}'")
    
    
    example_user_df = df.query(f'taxnumberprovider == "{user}"') # and adjudicada == True' ) #no es necesario que la licitacion este adjudicada, pero aquí probamos con licitaciones adjudicadas
    if yaml_data['USE_AWARDED']:
        example_user_df = example_user_df.query(f'adjudicada == True' )

    save = exploration_query.groupby(by = 'first_two_digits_code').count()['agileitemsproductcategory'].reset_index()

    
    fig = px.bar(save, x='first_two_digits_code', y='agileitemsproductcategory', title='Categorías Presentes')
    #columns = [{'name': col, 'id': col} for col in exploration_query.columns]
    data = exploration_query.groupby(by='agilebuyingsdescription').count()['agileitemsproductcategory'].reset_index()
    columns = [{'name': col, 'id': col} for col in data.columns]
    data = data.to_dict('records')
    return columns, data,fig

#TODO arreglar ejes del grafico
#TODO agregar vista de solo agile offered items

# Run the app
if __name__ == '__main__':
    app.run(debug=True)



#TODO remove corpus generation from main. It should only load already constructed csr matrix from a csv.

    