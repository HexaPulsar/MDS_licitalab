from dash import Dash, html, dcc, callback, Output, Input 
import plotly.express as px
import numpy as np
from dash import Dash, html, dash_table, dcc
import pandas as pd 
from utils.RecommenderSystem import RecommenderSystem
 
from unidecode import unidecode
 
from tqdm import tqdm
import os

DEBUG = False

# Incorporate data
csv_file_path = 'C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\Modeling\\20231007200103_query_results.csv'

with open(csv_file_path, 'r', encoding='utf-8') as file:
    total_lines = sum(1 for _ in file)

# Create an empty list to store the DataFrames from chunks
dfs = []
total_lines = 100000
# Create a tqdm wrapper for pd.read_csv
with tqdm(total=total_lines, desc = 'Loading Dataset') as pbar:
    def update_progress(n):
        pbar.update(n)

    # Read the CSV file using pd.read_csv and provide the progress callback
    df_chunks = pd.read_csv(csv_file_path, chunksize=1000, iterator=True, encoding='utf-8',nrows=100000)  # Specify the encoding
    for chunk in df_chunks:
        # Process each chunk if needed
        # You can access the chunk data in the 'chunk' DataFrame
        #chunk['first_two_digits_code'] = chunk['agilebuyingscode'].apply(lambda x: x[:2])
        chunk['feature_vector'] = chunk['agilebuyingscode'].apply(lambda x: x[:2]) + ' ' + chunk['agileoffereditemsdescripcionofertada']
        chunk['feature_vector'] = chunk['feature_vector'].apply(lambda x: unidecode(str(x)).lower())
        
        dfs.append(chunk)
        update_progress(chunk.shape[0])

# Concatenate the list of DataFrames into a single DataFrame
df = pd.concat(dfs, ignore_index=True)
RS = RecommenderSystem(df,save_path=  os.path.dirname(os.path.abspath(__file__)))
        
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
        page_size = 10
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
    cluster_number, gg = RS.predict(user)

    exploration_query = gg.query(f"Cluster == {cluster_number}")
    save = exploration_query.groupby(by = 'agilebuyingsdescription').count()['agileitemsproductcategory'].reset_index()
     
    fig = px.bar(save, x='agilebuyingsdescription', y='agileitemsproductcategory', title='Categorías Presentes en las Recomendaciones')
    data = exploration_query.groupby(by=['agilebuyingsdescription','agileitemsproductcategory']).count().reset_index()
    data['Score']= pd.DataFrame({'Score' :np.zeros(data.shape[1])})
    
    data = data[['agilebuyingsdescription','agileitemsproductcategory','Score']].sort_values(by='agileitemsproductcategory')
   
    columns = [{'name': col, 'id': col} for col in data.columns]
    data = data.to_dict('records')
    return columns, data,fig

#TODO arreglar ejes del grafico
#TODO agregar vista de solo agile offered items

# Run the app
if __name__ == '__main__':
     
    
    
    app.run(debug=True)

