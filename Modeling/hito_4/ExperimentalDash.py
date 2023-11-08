import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from dash import dash_table 
from utils.RecommenderSystem import RecommenderSystem
import os
import numpy as np
from tqdm import tqdm
from unidecode import unidecode
import plotly.express as px



app = dash.Dash(__name__)

#aqui cambiar file path
csv_file_path = 'C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\query_final_results_20231026023937.csv'
printed_text = html.Div(id='printed-text', children=[])
#with open(csv_file_path, 'r', encoding='utf-8') as file:
#    total_lines = sum(1 for _ in file)

# Create an empty list to store the DataFrames from chunks
dfs = []
total_lines = 100000
# Create a tqdm wrapper for pd.read_csv
with tqdm(total=total_lines, desc = 'Loading Dataset') as pbar:
    def update_progress(n):
        pbar.update(n)

    # Read the CSV file using pd.read_csv and provide the progress callback
    df_chunks = pd.read_csv(csv_file_path, chunksize=10000, iterator=True, encoding='utf-8',nrows=total_lines)  # Specify the encoding
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

df_empty = pd.DataFrame({'agilebuyingsdescription':[ ],'agileitemsproductcategory':[ ],'Score':[ ]})
#print(df)
RS = RecommenderSystem(df,save_path=  os.path.dirname(os.path.abspath(__file__)))
app.layout = html.Div([
    html.H1("Display DataFrame in Dash App with Dropdown and Bar Plot"),
    dcc.Dropdown(
        id='data-dropdown',
        options=[{'label': file, 'value': file} for file in df.taxnumberprovider.unique()],
        value='C:\\Users\\magda\\OneDrive\\Escritorio\\MDS_licitalab\\query_final_results_20231026023937.csv',  # Default selected file
    ),
    printed_text,  # Display printed text here
    dash_table.DataTable(
        id='data-table',
        columns=[{"name": col, "id": col} for col in df_empty.columns],
        data=df_empty.to_dict('records'),
        style_data={
            'white-space': 'normal',  # Enable word wrapping
            'text-align': 'left',     # Align text to the left
        },
        page_size = 10
    ),
    dcc.Graph(id='bar-plot'),
    
])

@app.callback( 
    Output('printed-text', 'children'),
    Output('data-table', 'data'),
    Output('bar-plot', 'figure'),
   
    Input('data-dropdown', 'value')
)
def update_data(user):
    global df_empty  # Update the global DataFrame
     
    cluster_number, gg = RS.predict(user)  # Load the selected CSV file
    exploration_query = gg.query(f"Cluster == {cluster_number}")
    
    
    printed_text_content = [
        html.P("This is a printed text section."),
        html.P(f"User: {user}"),
        html.P(f"Cluster Number: {cluster_number}"),
    ]

    # Create a bar plot using Plotly
    save = exploration_query.groupby(by = 'agilebuyingsdescription').count()['agileitemsproductcategory'].reset_index()
    
    figure = px.bar(save, x='agilebuyingsdescription', y='agileitemsproductcategory', title='Categorías Presentes en las Recomendaciones')

    data = exploration_query.groupby(by=['agilebuyingsdescription','agileitemsproductcategory']).count().reset_index()
    data['Score']= pd.DataFrame({'Score' :np.zeros(data.shape[0])})
    data = data[['agilebuyingsdescription','agileitemsproductcategory','Score']].sort_values(by='agileitemsproductcategory')

    return  printed_text_content, data.to_dict('records'), figure

if __name__ == '__main__':
    app.run_server(debug=True)