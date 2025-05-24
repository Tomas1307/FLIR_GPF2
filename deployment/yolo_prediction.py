# yolo_prediction.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import os
import cv2
from collections import defaultdict
import dash_bootstrap_components as dbc
import time
from ultralytics import YOLO
import yaml

MODEL_CONFIGS = {
    "filtered": {
        "name": "Finetuning datos filtrados sin preprocesamiento - Mejor modelo general",
        "weights_path": "./models/filtrado/best.pt",
        "yaml_path": "./YOLO_data/modelo_yolov11_dataset_filtrado/dataset.yaml",
        "description": "filtrado_freeze0"
    },
    "preprocessed": {
        "name": "Finetuning datos completos preprocesados - Mejor modelo para minería ilegal",
        "weights_path": "./models/completo_procesado/best.pt", 
        "yaml_path": "./YOLO_data/modelo_yolov11_dataset_completo_preprocesado/dataset.yaml",
        "description": "completo_preprocesado"
    }
}

CLASS_NAMES_MAP = {
    0: 'Vehiculos',
    1: 'Bodegas', 
    2: 'Caminos',
    3: 'Rios',
    4: 'Zonas de mineria ilegal'
}

_model_cache = {}
_dataset_info_cache = {}

def load_dataset_info(model_key):
    """Carga información del dataset desde el archivo YAML"""
    if model_key not in _dataset_info_cache:
        yaml_path = MODEL_CONFIGS[model_key]["yaml_path"]
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    dataset_info = yaml.safe_load(f)
                _dataset_info_cache[model_key] = dataset_info
                print(f"Información del dataset {model_key} cargada desde {yaml_path}")
            except Exception as e:
                print(f"Error cargando YAML {yaml_path}: {e}")
                _dataset_info_cache[model_key] = None
        else:
            print(f"Archivo YAML no encontrado: {yaml_path}")
            _dataset_info_cache[model_key] = None
    return _dataset_info_cache[model_key]

def get_class_names_from_yaml(model_key):
    """Obtiene los nombres de clases desde el archivo YAML del modelo"""
    dataset_info = load_dataset_info(model_key)
    if dataset_info and 'names' in dataset_info:
        return dataset_info['names']
    return CLASS_NAMES_MAP  # Fallback a la configuración manual

def load_model(model_key):
    """Carga y cachea el modelo YOLO"""
    if model_key not in _model_cache:
        weights_path = MODEL_CONFIGS[model_key]["weights_path"]
        if os.path.exists(weights_path):
            try:
                _model_cache[model_key] = YOLO(weights_path)
                print(f"Modelo {model_key} cargado exitosamente desde {weights_path}")
            except Exception as e:
                print(f"Error cargando modelo {model_key}: {e}")
                return None
        else:
            print(f"Archivo del modelo no encontrado: {weights_path}")
            return None
    return _model_cache[model_key]

def predict_image(image_data, model_key, conf_threshold=0.25, imgsz=640):
    """
    Realiza predicción sobre una imagen usando el modelo especificado
    
    Args:
        image_data: Datos de la imagen en base64
        model_key: Clave del modelo a usar ('filtered' o 'preprocessed')
        conf_threshold: Umbral de confianza para las predicciones
        imgsz: Tamaño de imagen para inferencia
    
    Returns:
        dict con resultados de la predicción
    """
    try:
        model = load_model(model_key)
        if model is None:
            return {"error": f"No se pudo cargar el modelo {MODEL_CONFIGS[model_key]['name']}"}
    
        class_names = get_class_names_from_yaml(model_key)
        
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes))
        
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        results = model(image_cv, conf=conf_threshold, imgsz=imgsz, verbose=False)
        inference_time = time.time() - start_time
        
        result = results[0]
        
        pred_image = result.plot()
        pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        
        _, buffer = cv2.imencode('.png', pred_image_rgb)
        pred_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        detections = []
        class_counts = defaultdict(int)
        confidences = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if isinstance(class_names, dict):
                    class_name = class_names.get(class_id, f'Clase {class_id}')
                elif isinstance(class_names, list) and class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = CLASS_NAMES_MAP.get(class_id, f'Clase {class_id}')
                
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                class_counts[class_name] += 1
                confidences.append(confidence)
        
        metrics = {
            'total_detections': len(detections),
            'inference_time_ms': round(inference_time * 1000, 2),
            'avg_confidence': round(np.mean(confidences), 3) if confidences else 0,
            'max_confidence': round(np.max(confidences), 3) if confidences else 0,
            'min_confidence': round(np.min(confidences), 3) if confidences else 0,
            'class_distribution': dict(class_counts),
            'image_size': f"{image.width}x{image.height}"
        }
        
        return {
            'success': True,
            'predicted_image': f"data:image/png;base64,{pred_image_b64}",
            'detections': detections,
            'metrics': metrics,
            'model_name': MODEL_CONFIGS[model_key]['name'],
            'model_description': MODEL_CONFIGS[model_key]['description'],
            'dataset_yaml': MODEL_CONFIGS[model_key]['yaml_path']
        }
        
    except Exception as e:
        return {"error": f"Error en la predicción: {str(e)}"}

def create_model_info_card(model_key):
    """Crea una tarjeta con información detallada del modelo"""
    config = MODEL_CONFIGS[model_key]
    dataset_info = load_dataset_info(model_key)
    
    weights_exists = os.path.exists(config["weights_path"])
    yaml_exists = os.path.exists(config["yaml_path"])
    
    status_color = "success" if weights_exists and yaml_exists else "warning" if weights_exists else "danger"
    
    info_items = [
        html.P([html.Strong("Descripción: "), config["description"]], className="mb-1 text-light"),
        html.P([html.Strong("Pesos: "), 
                html.Span("✓ Disponible" if weights_exists else "✗ No encontrado", 
                         className=f"text-{'success' if weights_exists else 'danger'}")], className="mb-1"),
        html.P([html.Strong("Dataset YAML: "), 
                html.Span("✓ Disponible" if yaml_exists else "✗ No encontrado", 
                         className=f"text-{'success' if yaml_exists else 'danger'}")], className="mb-1"),
    ]
    
    if dataset_info:
        if 'nc' in dataset_info:
            info_items.append(html.P([html.Strong("Número de clases: "), str(dataset_info['nc'])], className="mb-1 text-light"))
        if 'names' in dataset_info:
            classes_text = ", ".join([f"{k}: {v}" for k, v in dataset_info['names'].items()]) if isinstance(dataset_info['names'], dict) else ", ".join(dataset_info['names'])
            info_items.append(html.P([html.Strong("Clases: "), classes_text], className="mb-1 text-light small"))
    
    return dbc.Alert(info_items, color=status_color, className="mb-3")

def create_metrics_cards(metrics, model_info=None):
    """Crea tarjetas con las métricas de predicción"""
    if not metrics:
        return html.Div()
    
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(str(metrics['total_detections']), className="text-primary mb-0"),
                html.P("Detecciones Totales", className="text-muted small mb-0")
            ])
        ], className="text-center bg-dark border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{metrics['inference_time_ms']} ms", className="text-success mb-0"),
                html.P("Tiempo de Inferencia", className="text-muted small mb-0")
            ])
        ], className="text-center bg-dark border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{metrics['avg_confidence']:.3f}", className="text-warning mb-0"),
                html.P("Confianza Promedio", className="text-muted small mb-0")
            ])
        ], className="text-center bg-dark border-0 shadow-sm"),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(metrics['image_size'], className="text-info mb-0"),
                html.P("Resolución", className="text-muted small mb-0")
            ])
        ], className="text-center bg-dark border-0 shadow-sm"),
    ]
    
    if model_info:
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(model_info.get('model_description', 'N/A'), className="text-secondary mb-0"),
                    html.P("Configuración", className="text-muted small mb-0")
                ])
            ], className="text-center bg-dark border-0 shadow-sm")
        )
    
    return dbc.Row([
        dbc.Col(card, lg=2, md=4, sm=6, className="mb-3") for card in cards
    ])

def create_class_distribution_chart(class_distribution):
    """Crea gráfico de distribución de clases detectadas"""
    if not class_distribution:
        return html.Div(dbc.Alert("No se detectaron objetos en la imagen", color="info", className="text-center"))
    
    df = pd.DataFrame(list(class_distribution.items()), columns=['Clase', 'Cantidad'])
    
    fig = px.bar(df, x='Clase', y='Cantidad',
                 title="Distribución de Objetos Detectados",
                 color='Cantidad',
                 color_continuous_scale='viridis',
                 template="plotly_dark")
    
    fig.update_layout(
        height=300,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="#2c3e50", 
        plot_bgcolor="#2c3e50",
        font=dict(color="white"),
        showlegend=False
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def create_detections_table(detections):
    """Crea tabla con detalles de las detecciones"""
    if not detections:
        return html.Div(dbc.Alert("No hay detecciones para mostrar", color="info"))
    
    table_data = []
    for i, det in enumerate(detections):
        table_data.append({
            'ID': i + 1,
            'Clase': det['class_name'],
            'Confianza': f"{det['confidence']:.3f}",
            'Coordenadas': f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})"
        })
    
    return dbc.Table.from_dataframe(
        pd.DataFrame(table_data),
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-dark table-sm"
    )

def create_prediction_layout():
    """Crea el layout para la pestaña de predicción"""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Predicción con Modelos YOLO", 
                           className="section-title text-center my-4 text-primary"), width=12)
        ]),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Configuración de Predicción", className="card-title text-primary")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Seleccionar Modelo:", className="form-label text-light"),
                            dcc.Dropdown(
                                id='model-selector',
                                options=[
                                    {'label': MODEL_CONFIGS['filtered']['name'], 'value': 'filtered'},
                                    {'label': MODEL_CONFIGS['preprocessed']['name'], 'value': 'preprocessed'}
                                ],
                                value='filtered',
                                className="mb-3"
                            ),
                            html.Div(id='model-info-display', className="mt-2")
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Umbral de Confianza:", className="form-label text-light"),
                            dcc.Slider(
                                id='confidence-slider',
                                min=0.1,
                                max=0.9,
                                step=0.05,
                                value=0.25,
                                marks={i/10: f'{i/10:.1f}' for i in range(1, 10, 2)},
                                className="mb-3"
                            )
                        ], md=6)
                    ]),
                    html.Hr(className="my-3 border-light"),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt fa-3x mb-3 text-primary"),
                            html.P("Arrastra y suelta una imagen aquí o haz clic para seleccionar", 
                                   className="text-light"),
                            html.P("Formatos soportados: JPG, PNG, JPEG", className="text-muted small")
                        ]),
                        style={
                            'width': '100%',
                            'height': '120px',
                            'lineHeight': '120px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'borderColor': '#6c757d'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', className="mt-3")
                ])
            ], className="mb-4 shadow-sm bg-dark text-light"), width=12)
        ]),
        
        html.Div(id='prediction-results', className="mb-4"),
        
        dcc.Store(id='stored-image-data')
        
    ], fluid=True, className="py-4 bg-dark")

def register_prediction_callbacks(app):
    """Registra los callbacks para la funcionalidad de predicción"""
    
    @app.callback(
        Output('model-info-display', 'children'),
        Input('model-selector', 'value')
    )
    def update_model_info(model_key):
        if model_key:
            return create_model_info_card(model_key)
        return html.Div()
    
    @app.callback(
        [Output('stored-image-data', 'data'),
         Output('upload-status', 'children')],
        Input('upload-image', 'contents'),
        State('upload-image', 'filename')
    )
    def handle_image_upload(contents, filename):
        if contents is None:
            return None, ""
        
        try:
            if filename and not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return None, dbc.Alert("Por favor selecciona un archivo de imagen válido (PNG, JPG, JPEG)", 
                                     color="danger", className="mt-2")
            
            return contents, dbc.Alert(f"Imagen '{filename}' cargada exitosamente", 
                                     color="success", className="mt-2")
        except Exception as e:
            return None, dbc.Alert(f"Error al cargar la imagen: {str(e)}", 
                                 color="danger", className="mt-2")
    
    @app.callback(
        Output('prediction-results', 'children'),
        [Input('stored-image-data', 'data'),
         Input('model-selector', 'value'),
         Input('confidence-slider', 'value')],
        prevent_initial_call=True
    )
    def update_predictions(image_data, model_key, confidence):
        if image_data is None:
            return html.Div()
        
        results = predict_image(image_data, model_key, confidence)
        
        if 'error' in results:
            return dbc.Alert(results['error'], color="danger", className="mt-3")
        
        original_image_section = html.Div([
            html.H5("Imagen Original", className="text-center text-light mb-3"),
            html.Img(src=image_data, style={'width': '100%', 'height': 'auto', 'border-radius': '5px'})
        ])
        
        predicted_image_section = html.Div([
            html.H5("Predicción", className="text-center text-light mb-3"),
            html.Img(src=results['predicted_image'], style={'width': '100%', 'height': 'auto', 'border-radius': '5px'})
        ])
        
        metrics_cards = create_metrics_cards(results['metrics'], results)
        class_chart = create_class_distribution_chart(results['metrics']['class_distribution'])
        detections_table = create_detections_table(results['detections'])
        
        return html.Div([
            dbc.Alert([
                html.Strong("Modelo utilizado: "), results['model_name'],
                html.Br(),
                html.Small(f"Dataset: {results['dataset_yaml']}", className="text-muted")
            ], color="info", className="mb-4"),
            
            dbc.Row([
                dbc.Col(original_image_section, md=6, className="mb-4"),
                dbc.Col(predicted_image_section, md=6, className="mb-4")
            ]),
            
            html.H4("Métricas de Predicción", className="text-primary mb-3"),
            metrics_cards,
            
            dbc.Row([
                dbc.Col([
                    html.H5("Distribución de Clases", className="text-light mb-3"),
                    class_chart
                ], md=6, className="mb-4"),
                dbc.Col([
                    html.H5("Detalle de Detecciones", className="text-light mb-3"),
                    detections_table
                ], md=6, className="mb-4")
            ])
        ])

def get_prediction_tab_content():
    """Función principal que retorna el contenido de la pestaña de predicción"""
    return create_prediction_layout()

def check_models_availability():
    """Verifica si los archivos de modelo y dataset YAML existen"""
    available_models = {}
    for key, config in MODEL_CONFIGS.items():
        weights_exists = os.path.exists(config['weights_path'])
        yaml_exists = os.path.exists(config['yaml_path'])
        available_models[key] = {
            'weights': weights_exists,
            'yaml': yaml_exists,
            'complete': weights_exists and yaml_exists,
            'name': config['name'],
            'description': config['description']
        }
    return available_models

if __name__ == "__main__":
    print("Verificando disponibilidad de modelos y datasets...")
    models_status = check_models_availability()
    for model, status in models_status.items():
        print(f"\n{status['name']}:")
        print(f"  Pesos (.pt): {'✓' if status['weights'] else '✗'}")
        print(f"  Dataset YAML: {'✓' if status['yaml'] else '✗'}")
        print(f"  Estado: {'✓ Completo' if status['complete'] else '⚠ Incompleto'}")