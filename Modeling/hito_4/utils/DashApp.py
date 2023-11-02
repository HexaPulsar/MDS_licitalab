from dash import html, dcc, callback, Output, Input
from dash_table import DataTable
import plotly.express as px
from utils.RecommenderServer import initialize_recommender_system

def create_dash_app(app):
    # Initialize RecommenderSystem
    RS = initialize_recommender_system(df)

    # Dash app layout
    app.layout = html.Div([
        html.H1(children='Recomendaciones', style={'textAlign': 'center'}),
        dcc.Dropdown(options=[{'label': user, 'value': user} for user in df['taxnumberprovider'].unique()],
                     value='76.567.318-6', id='dropdown-selection'),
        # DataTable
        DataTable(
            id='display-table',
            columns=[],
            data=[],
            style_data={
                'white-space': 'normal',  # Enable word wrapping
                'text-align': 'left',  # Align text to the left
            },
            page_size=10
        ),
        dcc.Graph(
            id='bar-plot'
        )
    ])

    # Add controls to build the interaction
    @callback(
        Output('display-table', 'columns'),
        Output('display-table', 'data'),
        Output('bar-plot', 'figure'),
        Input('dropdown-selection', 'value')
    )
    def update_table(table):
        user = str(table)
        cluster_number, gg = RS.predict(user)

        exploration_query = gg.query(f"Cluster == {cluster_number} and taxnumberprovider != '{user}'")
        save = exploration_query.groupby(by='first_two_digits_code').count()['agileitemsproductcategory'].reset_index()

        fig = px.bar(save, x='first_two_digits_code', y='agileitemsproductcategory',
                     title='Categorías Presentes en las Recomendaciones')
        data = exploration_query.groupby(by=['agilebuyingsdescription', 'agileitemsproductcategory']).count().reset_index()
        data['Score'] = pd.DataFrame({'Score': np.zeros(data.shape[1])})

        data = data[['agilebuyingsdescription', 'agileitemsproductcategory', 'Score']].sort_values(by='agileitemsproductcategory')

        columns = [{'name': col, 'id': col} for col in data.columns]
        data = data.to_dict('records')
        return columns, data, fig
