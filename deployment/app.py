# app.py
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import os
import yaml
import cv2 
from collections import defaultdict
import dash_bootstrap_components as dbc
from pathlib import Path 
import random

from .yolo_prediction import get_prediction_tab_content, register_prediction_callbacks, check_models_availability
from .exploracion import get_exploration_tab_content, register_exploration_callbacks, CLASS_NAMES_MAP as EXPLORATION_CLASS_NAMES_MAP 
from .preprocesamiento_imagenes import get_preprocessing_tab_content, register_preprocessing_callbacks 

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.DARKLY, '/assets/style.css'])
app.title = "Detección de Minería Ilegal - Dashboard"

register_prediction_callbacks(app)
register_exploration_callbacks(app)
register_preprocessing_callbacks(app) 


app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Dashboard de Detección de Minería Ilegal", 
                            className="text-center text-primary my-4 display-4"), 
                    width=12)),
    dbc.Tabs(id="tabs", active_tab="tab-prediction", className="mb-4", children=[
        dbc.Tab(label="Predicción YOLO", tab_id="tab-prediction", className="custom-tab", active_label_style={"color": "#00f0f0"}),
        dbc.Tab(label="Exploración de Datos", tab_id="tab-exploration", className="custom-tab", active_label_style={"color": "#ff69b4"}),
        dbc.Tab(label="Preprocesamiento de Imágenes", tab_id="tab-preprocessing", className="custom-tab", active_label_style={"color": "#39ff14"})
    ]),
    html.Div(id="tab-content", className="p-4 bg-dark text-light border rounded")
], fluid=True, className="bg-dark")

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab') 
)
def render_tab_content(tab_value):
    if tab_value == 'tab-prediction':
        return get_prediction_tab_content()
    elif tab_value == 'tab-exploration':
        return get_exploration_tab_content()
    elif tab_value == 'tab-preprocessing':
        return get_preprocessing_tab_content() 
    return html.Div("Selecciona una pestaña")


if __name__ == '__main__':
    print("=== Dashboard de Detección de Minería Ilegal ===")
    print("Verificando disponibilidad de modelos YOLO y datasets...")
    models_status = check_models_availability()
    
    all_complete = True
    for model, status in models_status.items():
        print(f"\n{status['name']} ({status['description']}):")
        print(f"  Pesos (.pt): {'✓ Disponible' if status['weights'] else '✗ No encontrado'}")
        print(f"  Dataset YAML: {'✓ Disponible' if status['yaml'] else '✗ No encontrado'}")
        
        if not status['complete']:
            all_complete = False
            if not status['weights']:
                print(f"    ⚠️  Archivo de pesos faltante")
            if not status['yaml']:
                print(f"    ⚠️  Archivo YAML de dataset faltante")
    
    if not all_complete:
        print("\n⚠️  Algunos archivos no están disponibles. La funcionalidad de predicción puede estar limitada.")
        print("   Verifica que todos los archivos .pt y .yaml estén en las rutas especificadas.")
    else:
        print("\n✓ Todos los modelos y datasets están disponibles")
    
    app.run_server(debug=True, port=8050)