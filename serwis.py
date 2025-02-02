import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
from scipy.stats import t
import numpy as np
from geopy.distance import geodesic

def load_displacement_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df = df.melt(id_vars=['Date'], 
                 var_name='pid', 
                 value_name='displacement')
    df['timestamp'] = pd.to_datetime(df['Date'])
    df.drop(columns=['Date'], inplace=True)
    df['file'] = file_label
    df['pid'] = df['pid'].astype(str)
    return df

def load_anomaly_data(file_path, file_label):
    df = pd.read_csv(file_path)
    df['file'] = file_label
    df['pid'] = df['pid'].astype(str)
    return df

geo_data_lstm = pd.read_csv('wroclaw_geo.csv', delimiter=',')
geo_data_lstm['pid'] = geo_data_lstm['pid'].astype(str).str.strip()

geo_data_conv = pd.read_csv('wroclaw_geo.csv', delimiter=',')
geo_data_conv['pid'] = geo_data_conv['pid'].astype(str).str.strip()

geo_data_dense = pd.read_csv('wroclaw_geo.csv', delimiter=',')
geo_data_dense['pid'] = geo_data_dense['pid'].astype(str).str.strip()

geo_data_ml = pd.read_csv('wroclaw_geo.csv', delimiter=',')
geo_data_ml['pid'] = geo_data_ml['pid'].astype(str).str.strip()


displacement_data_1 = load_displacement_data('wro.csv', 'Ascending 175')
displacement_data_1['pid'] = displacement_data_1['pid'].astype(str).str.strip() 
all_data_lstm = pd.merge(displacement_data_1, geo_data_lstm, on='pid', how='left')

displacement_data_2 = load_displacement_data('wro.csv', 'Ascending 175')
displacement_data_2['pid'] = displacement_data_2['pid'].astype(str).str.strip() 
all_data_conv = pd.merge(displacement_data_2, geo_data_conv, on='pid', how='left')

displacement_data_3 = load_displacement_data('wro.csv', 'Ascending 175')
displacement_data_3['pid'] = displacement_data_3['pid'].astype(str).str.strip() 
all_data_dense = pd.merge(displacement_data_3, geo_data_dense, on='pid', how='left')

displacement_data_3 = load_displacement_data('wro.csv', 'Ascending 175')
displacement_data_3['pid'] = displacement_data_3['pid'].astype(str).str.strip() 
all_data_ml = pd.merge(displacement_data_3, geo_data_ml, on='pid', how='left')

prediction_data_1 = pd.read_csv('predictions_lstm.csv')
prediction_data_1['Date'] = pd.to_datetime(prediction_data_1['Date'])
prediction_data_1 = prediction_data_1.melt(id_vars='Date', var_name='pid', value_name='predicted_displacement')
prediction_data_1 = prediction_data_1.sort_values(by=['pid', 'Date'])
prediction_data_1['step'] = prediction_data_1.groupby('pid').cumcount() + 1
prediction_data_1['label'] = 'Prediction Set LSTM'

prediction_data_2 = pd.read_csv('predictions_conv.csv') 
prediction_data_2['Date'] = pd.to_datetime(prediction_data_2['Date'])
prediction_data_2 = prediction_data_2.melt(id_vars='Date', var_name='pid', value_name='predicted_displacement')
prediction_data_2 = prediction_data_2.sort_values(by=['pid', 'Date'])
prediction_data_2['step'] = prediction_data_1.groupby('pid').cumcount() + 1
prediction_data_2['label'] = 'Prediction Set Conv Autoencoder'

prediction_data_3 = pd.read_csv('predictions_dense.csv') 
prediction_data_3['Date'] = pd.to_datetime(prediction_data_3['Date'])
prediction_data_3 = prediction_data_3.melt(id_vars='Date', var_name='pid', value_name='predicted_displacement')
prediction_data_3 = prediction_data_3.sort_values(by=['pid', 'Date'])
prediction_data_3['step'] = prediction_data_3.groupby('pid').cumcount() + 1
prediction_data_3['label'] = 'Prediction Set Dense Autoencoder'

prediction_data_4 = pd.read_csv('predictions_ml.csv') 
prediction_data_4['Date'] = pd.to_datetime(prediction_data_4['Date'])
prediction_data_4 = prediction_data_4.melt(id_vars='Date', var_name='pid', value_name='predicted_displacement')
prediction_data_4 = prediction_data_4.sort_values(by=['pid', 'Date'])
prediction_data_4['step'] = prediction_data_4.groupby('pid').cumcount() + 1
prediction_data_4['label'] = 'Prediction Set ML'

anomaly_data_lstm_95  = load_anomaly_data('anomaly_lstm_95.csv', 'Anomaly Set 1 (95%)')
anomaly_data_lstm_95 = anomaly_data_lstm_95.groupby('pid').head(61)

anomaly_data_lstm_99 = load_anomaly_data('anomaly_lstm_99.csv', 'Anomaly Set 1 (99%)')
anomaly_data_lstm_99 = anomaly_data_lstm_99.groupby('pid').head(61)

anomaly_data_conv_95 = load_anomaly_data('anomaly_conv_95.csv', 'Anomaly Set 2 (95%)')
anomaly_data_conv_95 = anomaly_data_conv_95.groupby('pid').head(61)

anomaly_data_conv_99 = load_anomaly_data('anomaly_conv_99.csv', 'Anomaly Set 2 (99%)')
anomaly_data_conv_99 = anomaly_data_conv_99.groupby('pid').head(61)

anomaly_data_dense_95 = load_anomaly_data('anomaly_dense_95.csv', 'Anomaly Set 3 (95%)')
anomaly_data_dense_95 = anomaly_data_dense_95.groupby('pid').head(61)

anomaly_data_dense_99 = load_anomaly_data('anomaly_dense_99.csv', 'Anomaly Set 3 (99%)')
anomaly_data_dense_99 = anomaly_data_dense_99.groupby('pid').head(61)

anomaly_data_ml_95 = load_anomaly_data('anomaly_ml_95.csv', 'Anomaly Set 4 (95%)')
anomaly_data_ml_95 = anomaly_data_ml_95.groupby('pid').head(61)

anomaly_data_ml_99 = load_anomaly_data('anomaly_ml_99.csv', 'Anomaly Set 4 (99%)')
anomaly_data_ml_99 = anomaly_data_ml_99.groupby('pid').head(61)


all_data_lstm.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_lstm['displacement_diff'] = all_data_lstm.groupby('pid')['displacement'].diff().round(1)
all_data_lstm['time_diff'] = all_data_lstm.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_lstm['displacement_speed'] = ((all_data_lstm['displacement_diff'] / all_data_lstm['time_diff']) * 365).round(1)

mean_velocity_data_lstm = all_data_lstm.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_lstm.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_lstm = pd.merge(all_data_lstm, mean_velocity_data_lstm, on='pid', how='left')

all_data_conv.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_conv['displacement_diff'] = all_data_conv.groupby('pid')['displacement'].diff().round(1)
all_data_conv['time_diff'] = all_data_conv.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_conv['displacement_speed'] = ((all_data_conv['displacement_diff'] / all_data_conv['time_diff']) * 365).round(1)

mean_velocity_data_conv = all_data_conv.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_conv.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_conv = pd.merge(all_data_conv, mean_velocity_data_conv, on='pid', how='left')

all_data_dense.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_dense['displacement_diff'] = all_data_dense.groupby('pid')['displacement'].diff().round(1)
all_data_dense['time_diff'] = all_data_dense.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_dense['displacement_speed'] = ((all_data_dense['displacement_diff'] / all_data_dense['time_diff']) * 365).round(1)

mean_velocity_data_dense = all_data_dense.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_dense.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_dense = pd.merge(all_data_dense, mean_velocity_data_dense, on='pid', how='left')

all_data_ml.sort_values(by=['pid', 'timestamp'], inplace=True)
all_data_ml['displacement_diff'] = all_data_ml.groupby('pid')['displacement'].diff().round(1)
all_data_ml['time_diff'] = all_data_ml.groupby('pid')['timestamp'].diff().dt.days.round(1)
all_data_ml['displacement_speed'] = ((all_data_ml['displacement_diff'] / all_data_ml['time_diff']) * 365).round(1)

mean_velocity_data_ml = all_data_ml.groupby('pid')['displacement_speed'].mean().round(1).reset_index()
mean_velocity_data_ml.rename(columns={'displacement_speed': 'mean_velocity'}, inplace=True)
all_data_ml = pd.merge(all_data_ml, mean_velocity_data_ml, on='pid', how='left')

def compute_prefix_sums(data):
    data = data.sort_values(by=['pid', 'step'])
    pivot = data.pivot(index='pid', columns='step', values='predicted_displacement').fillna(0).round(1)
    pivot[0] = 0 
    pivot = pivot.sort_index(axis=1)
    for col in pivot.columns[1:]:
        pivot[col] = (pivot[col] + pivot[col-1]).round(1)
    return pivot

lstm_prefix = compute_prefix_sums(prediction_data_1)
conv_prefix = compute_prefix_sums(prediction_data_2)
dense_prefix = compute_prefix_sums(prediction_data_3)
ml_prefix = compute_prefix_sums(prediction_data_4)

prefix_data = {
    'lstm': lstm_prefix,
    'conv': conv_prefix,
    'dense': dense_prefix,
    'ml': ml_prefix,
}

MAX_LSTM = lstm_prefix.columns.max()
MAX_CONV = conv_prefix.columns.max()
MAX_DENSE = dense_prefix.columns.max()
MAX_ML = ml_prefix.columns.max()


def add_obs_step(df):
    df = df.sort_values(by=['pid', 'timestamp'])
    df['obs_step'] = df.groupby('pid').cumcount() + 1
    return df

all_data_lstm = add_obs_step(all_data_lstm)
all_data_conv = add_obs_step(all_data_conv)
all_data_dense = add_obs_step(all_data_dense)
all_data_ml = add_obs_step(all_data_ml)

def compute_prefix_sums_actual(df):
    pivot = df.pivot(index='pid', columns='obs_step', values='displacement').fillna(0).round(1)
    pivot[0] = 0
    pivot = pivot.sort_index(axis=1)
    for col in pivot.columns[1:]:
        pivot[col] = (pivot[col] + pivot[col-1]).round(1)
    return pivot

actual_lstm_prefix = compute_prefix_sums_actual(all_data_lstm)
actual_conv_prefix = compute_prefix_sums_actual(all_data_conv)
actual_dense_prefix = compute_prefix_sums_actual(all_data_dense)
actual_ml_prefix = compute_prefix_sums_actual(all_data_ml)

actual_prefix_data = {
    'lstm': actual_lstm_prefix,
    'conv': actual_conv_prefix,
    'dense': actual_dense_prefix,
    'ml': actual_ml_prefix,
}

MAX_ACTUAL_LSTM = actual_lstm_prefix.columns.max()
MAX_ACTUAL_CONV = actual_conv_prefix.columns.max()
MAX_ACTUAL_DENSE = actual_dense_prefix.columns.max()
MAX_ACTUAL_ML = actual_ml_prefix.columns.max()

orbit_geometry_info = {
    'Ascending 175': {
        'Relative orbit number': '175',
        'View angle': '348.9°',
        'Mean Incidence angle': '33.18°'  
    }}


px.set_mapbox_access_token('pk.eyJ1IjoibWFycGllayIsImEiOiJjbTBxbXBsMGQwYjgyMmxzN3RpdmlhZDVrIn0.YWJh1RM6HKfN_pbH-jtJ6A')


app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H3("Select Map and Data Visualization Options"),

    html.Div([
        html.Div([
            html.Label("Map Style"),
            dcc.Dropdown(
                id='map-style-dropdown',
                options=[
                    {'label': 'Satellite', 'value': 'satellite'},
                    {'label': 'Outdoors', 'value': 'outdoors'},
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Dark', 'value': 'dark'},
                    {'label': 'Streets', 'value': 'streets'}
                ],
                value='satellite',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Visualization Option"),
            dcc.Dropdown(
                id='color-mode-dropdown',
                options=[
                    {'label': 'Orbit Type', 'value': 'orbit'},
                    {'label': 'Displacement Mean Velocity [mm/year]', 'value': 'speed'},
                    {'label': 'Anomaly Type', 'value': 'anomaly_type'},
                    {'label': 'Prediction Velocity', 'value': 'prediction_velocity'},
                    {'label': 'Cumulative Displacement', 'value': 'actual_displacement_velocity'}
                ],
                value='orbit',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Filter by LOS Geometry"),
            dcc.Dropdown(
                id='orbit-filter-dropdown',
                options=[
                    {'label': 'Ascending 175', 'value': 'Ascending 175'},
                    {'label': 'Descending 124', 'value': 'Descending 124'}
                ],
                value='Ascending 175',
                multi=True,
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Select Area of Interest"),
            dcc.Dropdown(
                id='area-dropdown',
                options=[
                    {'label': 'Wrocław - LSTM', 'value': 'lstm'},
                    {'label': 'Wrocław - CONV', 'value': 'conv'},
                    {'label': 'Wrocław - DENSE', 'value': 'dense'},
                    {'label': 'Wrocław - ML', 'value': 'ml'}
                ],
                value='lstm',
                clearable=False,
                persistence=True,     
                persistence_type='memory',
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'}),

        html.Div([
            html.Label("Enable Distance Calculation"),
            dcc.Dropdown(
                id='distance-calc-dropdown',
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ],
                value='no',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'display': 'inline-block', 'width': '19%', 'padding': '10px'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    
    html.Div(id='distance-output', style={'font-size': '16px', 'padding': '10px', 'color': 'black'}),


    html.Div([
        html.Label("Select Observation Range"),
        html.Div(id='selected-range-dates', style={'fontSize': '14px', 'margin': '10px 0'}),

        dcc.RangeSlider(
            id='dynamic-prediction-range-slider',
            min=1,
            max=60,
            step=1, 
            marks={}, 
            value=[1, 5],
            tooltip={"placement": "bottom", "always_visible": True}, 
            allowCross=False
        )
    ], id='prediction-slider-container', style={'display': 'none', 'padding': '10px'}),

    dcc.Graph(id='map', style={'height': '80vh', 'width': '95vw'}, config={'scrollZoom': True, 'doubleClick': False}),
    dcc.Store(id='selected-points', data={'point_1': None, 'point_2': None}),

    html.Div(id='displacement-container', children=[
        html.Div([ 
            html.Label("Select Date Range", style={'font-size': '16px'}),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=all_data_lstm['timestamp'].min(),
                end_date=all_data_lstm['timestamp'].max(),
                display_format='YYYY-MM-DD',
                style={'height': '5px', 'width': '300px', 'font-family': 'Arial', 'font-size': '4px', 'display': 'inline-block', 'padding': '5px'}
            )
        ], style={'display': 'inline-block', 'padding': '10px'}),

        html.Div([ 
            html.Label("Set Y-Axis Range (mm)"),
            dcc.Input(
                id='y-axis-min',
                type='number',
                placeholder='Min',
                style={'width': '20%', 'margin-right': '10px'}
            ),
            dcc.Input(
                id='y-axis-max',
                type='number',
                placeholder='Max',
                style={'width': '20%'}
            ),
        ], style={'display': 'inline-block', 'padding': '10px'}),

        dcc.Graph(id='displacement-graph', style={'height': '50vh', 'width': '95vw'})
    ], style={'display': 'none'}),
    html.Div([
        html.Hr(style={'margin': '5px 0'}),
        html.Div(
            [
                html.P("This work was supported by the Wrocław University of Environmental and Life Sciences (Poland) "
                    "as part of the research project No. N060/0004/23.")
            ],
            style={'textAlign': 'center', 'fontSize': '14px'}
        )
    ], style={'padding': '10px'})
])

@app.callback(
    Output('selected-range-dates', 'children'),
    Input('dynamic-prediction-range-slider', 'value'),
    State('area-dropdown', 'value')
)
def display_selected_dates(range_value, selected_area):
    start_val, end_val = range_value

    data_for_area = {
        'lstm': all_data_lstm,
        'conv': all_data_conv,
        'dense': all_data_dense,
        'ml': all_data_ml
    }.get(selected_area, all_data_lstm)

    timestamps_df = (
        data_for_area
        .drop_duplicates(subset='obs_step')[['obs_step', 'timestamp']]
        .sort_values('obs_step')
    )
    
    step_to_date = dict(zip(timestamps_df['obs_step'], timestamps_df['timestamp']))

    date_start = step_to_date.get(start_val)
    date_end = step_to_date.get(end_val)

    if date_start:
        date_start_str = date_start.strftime('%Y-%m-%d')
    else:
        date_start_str = "N/A"

    if date_end:
        date_end_str = date_end.strftime('%Y-%m-%d')
    else:
        date_end_str = "N/A"

    return f"Selected date range: {date_start_str} to {date_end_str}"

@app.callback(
    Output('prediction-slider-container', 'style'),
    [Input('color-mode-dropdown', 'value')]
)
def toggle_prediction_slider_visibility(color_mode):
    if color_mode in ['prediction_velocity', 'actual_displacement_velocity']: 
        return {'display': 'block', 'padding': '10px'}
    else:
        return {'display': 'none', 'padding': '10px'}

@app.callback(
    [Output('orbit-filter-dropdown', 'options'), 
     Output('orbit-filter-dropdown', 'value'),
     Output('orbit-filter-dropdown', 'disabled')],
    [Input('area-dropdown', 'value')]
)
def update_orbit_filter(selected_area):
    if selected_area == 'lstm':
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 175', True
    elif selected_area == 'conv':
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 175', True
    elif selected_area == 'dense':
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 175', True
    else:
        return [{'label': 'Ascending 175', 'value': 'Ascending 175'}], 'Ascending 175', True

@app.callback(
    [Output('dynamic-prediction-range-slider', 'max'),
     Output('dynamic-prediction-range-slider', 'marks'),
     Output('dynamic-prediction-range-slider', 'value')],
    [Input('area-dropdown', 'value'),
     Input('color-mode-dropdown', 'value')]
)
def update_slider_max(selected_area, color_mode):

    if color_mode == 'actual_displacement_velocity':
        max_val = {
            'turow': MAX_ACTUAL_LSTM,
            'bedzin': MAX_ACTUAL_CONV,
            'grunwald': MAX_ACTUAL_DENSE
        }.get(selected_area, MAX_ACTUAL_LSTM)
    else:
        max_val = 60  
        
    data_for_area = {
        'lstm': all_data_lstm,
        'conv': all_data_conv,
        'dense': all_data_dense,
        'ml': all_data_ml
    }.get(selected_area, all_data_lstm)

    timestamps_df = data_for_area.drop_duplicates(subset='obs_step')[['obs_step', 'timestamp']].sort_values('obs_step')

    if max_val > 60:
        N = 40 
    elif max_val > 20:
        N = 15  
    else:
        N = 5  

    marks = {}
    for _, row in timestamps_df.iterrows():
        step_val = row['obs_step']
        if step_val <= max_val:
            if step_val == 1 or step_val == max_val or step_val % N == 0:
                marks[step_val] = row['timestamp'].strftime('%Y-%m-%d')  
            else:
                marks[step_val] = "" 

    default_end = min(5, max_val)
    return max_val, marks, [1, default_end]

@app.callback(
    Output('map', 'figure'),
    [
        Input('map-style-dropdown', 'value'),
        Input('color-mode-dropdown', 'value'),
        Input('orbit-filter-dropdown', 'value'),
        Input('area-dropdown', 'value'),
        Input('dynamic-prediction-range-slider', 'value')
    ]
)
def update_map(map_style, color_mode, orbit_filter, selected_area, pred_range):
    if selected_area == 'lstm':
        data = all_data_lstm.drop_duplicates(subset=['pid'])
        center_coords = {'lat': all_data_lstm['latitude'].mean(), 'lon': all_data_lstm['longitude'].mean()}
        zoom_level = 14
    elif selected_area == 'conv':
        data = all_data_conv.drop_duplicates(subset=['pid'])
        center_coords = {'lat': all_data_conv['latitude'].mean(), 'lon': all_data_conv['longitude'].mean()}
        zoom_level = 14
    elif selected_area == 'dense':
        data = all_data_dense.drop_duplicates(subset=['pid'])
        center_coords = {'lat': all_data_dense['latitude'].mean(), 'lon': all_data_dense['longitude'].mean()}
        zoom_level = 14
    else:
        data = all_data_ml.drop_duplicates(subset=['pid'])
        center_coords = {'lat': all_data_ml['latitude'].mean(), 'lon': all_data_ml['longitude'].mean()}
        zoom_level = 14

    if isinstance(orbit_filter, str):
        orbit_filter = [orbit_filter]

    filtered_data = data[data['file'].isin(orbit_filter)].copy()
    filtered_data.loc[:, 'mean_velocity'] = filtered_data['mean_velocity'].round(1)

    start_val, end_val = pred_range

    if color_mode == 'prediction_velocity':
        if selected_area == 'lstm':
            max_steps = MAX_LSTM
        elif selected_area == 'conv':
            max_steps = MAX_CONV
        elif selected_area == 'dense':
            max_steps = MAX_DENSE
        else:
            max_steps = MAX_ML

        pred_key = (
            selected_area)
        
        prefix_pivot = prefix_data[pred_key]

        end_val = min(end_val, max_steps)
        start_val = min(start_val, max_steps)

        numerator = prefix_pivot[end_val] - prefix_pivot[start_val-1]
        denominator = (end_val - start_val + 1)
        prediction_avg = numerator / denominator

        merged_data = filtered_data.set_index('pid')
        merged_data['prediction_velocity'] = prediction_avg
        merged_data.reset_index(inplace=True)

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='prediction_velocity',  
            color_continuous_scale='Jet',
            range_color=(-5, 5),
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Prediction Velocity Average')

    elif color_mode == 'actual_displacement_velocity':
        if selected_area == 'lstm':
            max_steps = MAX_ACTUAL_LSTM
            prefix_pivot = actual_prefix_data['lstm']
        elif selected_area == 'conv':
            max_steps = MAX_ACTUAL_CONV
            prefix_pivot = actual_prefix_data['conv']
        elif selected_area == 'dense':
            max_steps = MAX_ACTUAL_DENSE
            prefix_pivot = actual_prefix_data['dense']
        else:
            max_steps = MAX_ACTUAL_ML
            prefix_pivot = actual_prefix_data['ml']

        end_val = min(end_val, max_steps)
        start_val = min(start_val, max_steps)

        numerator = prefix_pivot[end_val] - prefix_pivot[start_val-1]
        denominator = (end_val - start_val + 1)
        actual_avg = numerator / denominator

        merged_data = filtered_data.set_index('pid')
        merged_data['actual_displacement_velocity'] = actual_avg
        merged_data.reset_index(inplace=True)

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='actual_displacement_velocity',  
            color_continuous_scale='Jet',
            range_color=(-5, 5),
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Actual Displacement Velocity Average')

    elif color_mode == 'anomaly_type':
        if selected_area == 'lstm':
            merged_data = filtered_data.merge(anomaly_data_lstm_99[['pid', 'is_anomaly']], on='pid', how='left')
        elif selected_area == 'conv':
            merged_data = filtered_data.merge(anomaly_data_conv_99[['pid', 'is_anomaly']], on='pid', how='left')
        elif selected_area == 'dense':
            merged_data = filtered_data.merge(anomaly_data_dense_99[['pid', 'is_anomaly']], on='pid', how='left')
        else:
            merged_data = filtered_data.merge(anomaly_data_ml_99[['pid', 'is_anomaly']], on='pid', how='left')

        merged_data['is_anomaly'] = merged_data['is_anomaly'].fillna(False).astype(bool)
        merged_data['consecutive_anomalies'] = (
            merged_data.groupby('pid')['is_anomaly']
            .rolling(3, min_periods=3).sum().reset_index(0, drop=True)
        )
        merged_data['anomaly_3plus'] = merged_data['consecutive_anomalies'] >= 3

        fig = px.scatter_mapbox(
            merged_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            color=merged_data['anomaly_3plus'].map({True: 'Anomaly', False: 'No Anomaly'}),
            color_discrete_map={'Anomaly': 'red', 'No Anomaly': 'green'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Anomaly Type')

    elif color_mode == 'orbit':
        fig = px.scatter_mapbox(
            filtered_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            color='file',
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Orbit Type')

    elif color_mode == 'speed':
        fig = px.scatter_mapbox(
            filtered_data,
            lat='latitude', lon='longitude',
            hover_name='pid',
            hover_data={'latitude': True, 'longitude': True, 'height': True},
            color='mean_velocity',
            color_continuous_scale='Jet',
            labels={'latitude': 'Latitude', 'longitude': 'Longitude', 'height': 'Height'},
            zoom=zoom_level
        )
        fig.update_layout(legend_title_text='Mean Velocity')
        
    if orbit_filter is not None:
        if isinstance(orbit_filter, str):
            orbit_filter = [orbit_filter]

        annotation_lines = ["Orbit Geometry Info:<br>"]
        for orbit in orbit_filter:
            if orbit in orbit_geometry_info:
                info = orbit_geometry_info[orbit]
                annotation_lines.append(
                    f"<b>{orbit}</b>:<br>"
                    f"Relative orbit number: {info['Relative orbit number']}<br>"
                    f"View angle: {info['View angle']}<br>"
                    f"Mean Incidence angle: {info['Mean Incidence angle']}<br><br>"
                )

        if len(annotation_lines) > 1:
            annotation_text = "".join(annotation_lines)
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=1, y=1,
                showarrow=False,
                align="left",
                bordercolor="#cccccc",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )

    fig.update_layout(
        mapbox_style=map_style,
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),
        mapbox=dict(center=center_coords),
        coloraxis_colorbar=dict(title=None), 
    )

    return fig

@app.callback(
    Output('selected-points', 'data'),
    [Input('map', 'clickData')],
    [State('selected-points', 'data')]
)
def update_selected_points(clickData, selected_points):
    if clickData is None:
        return selected_points
    
    point_id = clickData['points'][0]['hovertext']
    lat = clickData['points'][0]['lat']
    lon = clickData['points'][0]['lon']
    
    if selected_points['point_1'] is None:
        selected_points['point_1'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    elif selected_points['point_2'] is None:
        selected_points['point_2'] = {'pid': point_id, 'lat': lat, 'lon': lon}
    else:
        selected_points = {'point_1': None, 'point_2': None}

    return selected_points

@app.callback(
    Output('distance-output', 'children'),
    [Input('selected-points', 'data'),
     Input('distance-calc-dropdown', 'value')]
)
def display_distance(selected_points, distance_calc_enabled):
    if distance_calc_enabled == 'no':
        return ""

    point_1 = selected_points['point_1']
    point_2 = selected_points['point_2']
    
    if point_1 is not None and point_2 is not None:
        coords_1 = (point_1['lat'], point_1['lon'])
        coords_2 = (point_2['lat'], point_2['lon'])

        distance_km = geodesic(coords_1, coords_2).kilometers
        
        return html.Div([
            html.H4("Selected Points and Distance"),
            html.Ul([
                html.Li(f"Point 1: {point_1['pid']} (Lat: {point_1['lat']}, Lon: {point_1['lon']})"),
                html.Li(f"Point 2: {point_2['pid']} (Lat: {point_2['lat']}, Lon: {point_2['lon']})"),
                html.Li(f"Distance: {distance_km:.2f} km")
            ], style={'list-style-type': 'none', 'padding': '0', 'margin': '0'})
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'})
    else:
        return "Select two points on the map to calculate the distance."

@app.callback(
    [Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed')],
    [Input('area-dropdown', 'value')]
)
def update_date_picker(selected_area):
    if selected_area == 'lstm':
        start_date = all_data_lstm['timestamp'].min()
        end_date = all_data_lstm['timestamp'].max()
    elif selected_area == 'conv':
        start_date = all_data_conv['timestamp'].min()
        end_date = all_data_conv['timestamp'].max()
    elif selected_area == 'dense':
        start_date = all_data_dense['timestamp'].min()
        end_date = all_data_dense['timestamp'].max()
    else:
        start_date = all_data_ml['timestamp'].min()
        end_date = all_data_ml['timestamp'].max()

    return start_date, end_date, start_date, end_date

@app.callback(
    [Output('displacement-graph', 'figure'), Output('displacement-container', 'style')],
    [Input('map', 'clickData'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('y-axis-min', 'value'),
     Input('y-axis-max', 'value'),
     Input('area-dropdown', 'value')] 
)
def display_displacement(clickData, start_date, end_date, y_min, y_max, selected_area):
    if clickData is None:
        return {}, {'display': 'none'}

    point_id = clickData['points'][0]['hovertext']
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if selected_area == 'lstm':
        full_data = all_data_lstm[all_data_lstm['pid'] == point_id].copy() 
        anomaly_data_95 = anomaly_data_lstm_95
        anomaly_data_99 = anomaly_data_lstm_99
        last_n_data = full_data.tail(61)
    elif selected_area == 'conv':
        full_data = all_data_conv[all_data_conv['pid'] == point_id].copy() 
        anomaly_data_95 = anomaly_data_conv_95
        anomaly_data_99 = anomaly_data_conv_99
        last_n_data = full_data.tail(61)
    elif selected_area == 'dense':
        full_data = all_data_dense[all_data_dense['pid'] == point_id].copy() 
        anomaly_data_95 = anomaly_data_dense_95
        anomaly_data_99 = anomaly_data_dense_99
        last_n_data = full_data.tail(61)
    else:
        full_data = all_data_ml[all_data_ml['pid'] == point_id].copy() 
        anomaly_data_95 = anomaly_data_ml_95
        anomaly_data_99 = anomaly_data_ml_99
        last_n_data = full_data.tail(61)
        

    last_n_data.set_index('timestamp', inplace=True)
    filtered_data = full_data[(full_data['timestamp'] >= start_date) & (full_data['timestamp'] <= end_date)].copy()

    if not last_n_data.empty and (last_n_data.index.min() <= end_date) and (last_n_data.index.max() >= start_date):
        filtered_last_n_data = last_n_data[(last_n_data.index >= start_date) & (last_n_data.index <= end_date)].copy()
    else:
        filtered_last_n_data = pd.DataFrame()

    filtered_anomalies_95 = anomaly_data_95[anomaly_data_95['pid'] == point_id].copy()
    filtered_anomalies_99 = anomaly_data_99[anomaly_data_99['pid'] == point_id].copy()

    if not filtered_anomalies_95.empty:
        filtered_anomalies_95 = filtered_anomalies_95.tail(len(filtered_last_n_data)).copy()
        if len(filtered_anomalies_95) > 0:
            filtered_anomalies_95['timestamp'] = filtered_last_n_data.index.values[:len(filtered_anomalies_95)]
            filtered_anomalies_95.set_index('timestamp', inplace=True)
            filtered_last_n_data = filtered_last_n_data.join(
                filtered_anomalies_95[['predicted_value', 'upper_bound', 'lower_bound', 'is_anomaly']], 
                how='left'
            )

    if not filtered_anomalies_99.empty:
        filtered_anomalies_99 = filtered_anomalies_99.tail(len(filtered_last_n_data)).copy()
        if len(filtered_anomalies_99) > 0:
            filtered_anomalies_99['timestamp'] = filtered_last_n_data.index.values[:len(filtered_anomalies_99)]
            filtered_anomalies_99.set_index('timestamp', inplace=True)
            filtered_last_n_data = filtered_last_n_data.join(
                filtered_anomalies_99[['upper_bound', 'lower_bound', 'is_anomaly']], 
                how='left', rsuffix='_99'
            )

    fig = px.line(filtered_data, x='timestamp', y='displacement', 
                  title=f"Displacement LOS for point {point_id}",
                  markers=True, 
                  labels={'displacement': 'Displacement[mm]'})

    fig.add_scatter(x=filtered_data['timestamp'], y=filtered_data['displacement'], 
                    mode='lines+markers', 
                    name='InSAR measured displacement', 
                    line=dict(color='blue'))

    if not filtered_last_n_data.empty:
        if 'predicted_value' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['predicted_value'], 
                            mode='lines+markers', 
                            name='Predicted Displacement', 
                            line=dict(color='orange'))

        if 'upper_bound' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['upper_bound'], 
                            mode='lines', line=dict(color='yellow', dash='dash'),
                            name='Upper Bound p=95')

            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['lower_bound'],
                            mode='lines', line=dict(color='yellow', dash='dash'),
                            fill='tonexty', fillcolor='rgba(255, 252, 127, 0.2)',
                            name='Lower Bound p=95')

        anomalies_95 = filtered_last_n_data[filtered_last_n_data['is_anomaly'] == 1]
        if not anomalies_95.empty:
            fig.add_scatter(x=anomalies_95.index, y=anomalies_95['displacement'], 
                            mode='markers', name='Anomalies p=95', 
                            marker=dict(color='yellow', size=10))

        if 'upper_bound_99' in filtered_last_n_data.columns:
            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['upper_bound_99'], 
                            mode='lines', line=dict(color='red', dash='dash'),
                            name='Upper Bound p=99')

            fig.add_scatter(x=filtered_last_n_data.index, y=filtered_last_n_data['lower_bound_99'],
                            mode='lines', line=dict(color='red', dash='dash'),
                            fill='tonexty', fillcolor='rgba(254, 121, 104, 0.1)',
                            name='Lower Bound p=99')

            anomalies_99 = filtered_last_n_data[filtered_last_n_data['is_anomaly_99'] == 1]
            if not anomalies_99.empty:
                fig.add_scatter(x=anomalies_99.index, y=anomalies_99['displacement'], 
                                mode='markers', name='Anomalies p=99', 
                                marker=dict(color='red', size=10))

    if y_min is not None and y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Displacement LOS[mm]', 
        legend_title="Legend",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.05)
    )

    return fig, {'display': 'block'}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
