import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import base64
import os
import cv2
from collections import defaultdict
import dash_bootstrap_components as dbc

# Configuración de constantes
CLASS_NAMES_MAP = {
    0: 'Vehiculos',
    1: 'Bodegas',
    2: 'Caminos',
    3: 'Rios',
    4: 'Zonas de mineria ilegal'
}

# Professional Color Palette for charts
PROFESSIONAL_COLORS = px.colors.sequential.Plotly3

# Define colors for bounding boxes and labels
BOX_COLORS = {
    0: (255, 100, 100),  # Red for Vehiculos (BGR)
    1: (100, 255, 100),  # Green for Bodegas
    2: (100, 100, 255),  # Blue for Caminos
    3: (255, 255, 100),  # Yellow for Rios
    4: (255, 100, 255),  # Magenta for Zonas de mineria ilegal
}

TARGET_CLASS_HIGHLIGHT_COLOR = (255, 200, 0)  # Orange (BGR)

# Base directories for datasets
DATASET_BASE_PATHS = {
    "general": "YOLO_data/modelo_yolov11_dataset_filtrado",
}

# Cache for dataset analysis results
_dataset_analysis_cache = {}

def get_class_names_for_display():
    """Obtiene los nombres de clases para mostrar"""
    return [CLASS_NAMES_MAP[i] for i in sorted(CLASS_NAMES_MAP.keys())]

def get_image_and_label_paths(dataset_path):
    """Obtiene las rutas de imágenes y etiquetas del dataset"""
    paths = defaultdict(lambda: {'images': [], 'labels': []})
    splits = ['train', 'val', 'test']

    for split in splits:
        images_dir = os.path.join(dataset_path, split, 'images')
        labels_dir = os.path.join(dataset_path, split, 'labels')

        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    label_name = os.path.splitext(img_name)[0] + '.txt'
                    paths[split]['images'].append(os.path.join(images_dir, img_name))
                    paths[split]['labels'].append(os.path.join(labels_dir, label_name))
    return dict(paths)

def analyze_dataset_labels(dataset_path):
    """Analiza las etiquetas del dataset y extrae estadísticas"""
    total_images = 0
    class_counts = defaultdict(int)
    image_resolutions = []

    if dataset_path not in _dataset_analysis_cache:
        if not os.path.exists(dataset_path):
            print(f"Error: Base path for dataset '{dataset_path}' not found.")
            return {
                'total_images': 0,
                'class_counts': {},
                'resolutions': []
            }

        splits_data = get_image_and_label_paths(dataset_path)

        for split in ['train', 'val', 'test']:
            for i, img_path in enumerate(splits_data[split]['images']):
                total_images += 1
                try:
                    with Image.open(img_path) as img:
                        image_resolutions.append(img.size[0])
                except Exception as e:
                    pass

                label_path = splits_data[split]['labels'][i]
                if os.path.exists(label_path):
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    try:
                                        class_id = int(parts[0])
                                        class_counts[class_id] += 1
                                    except ValueError:
                                        pass
                    except Exception as e:
                        pass
        _dataset_analysis_cache[dataset_path] = {
            'total_images': total_images,
            'class_counts': class_counts,
            'resolutions': image_resolutions
        }
    return _dataset_analysis_cache[dataset_path]

def get_images_for_class(target_class_id, max_examples=4):
    """Encuentra imágenes que contengan la clase objetivo"""
    examples_found = []
    dataset_path = DATASET_BASE_PATHS["general"]
    splits_data = get_image_and_label_paths(dataset_path)

    for split in ['train', 'val', 'test']:
        for i, img_path in enumerate(splits_data[split]['images']):
            label_path = splits_data[split]['labels'][i]
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    class_id = int(parts[0])
                                    if class_id == target_class_id:
                                        examples_found.append((img_path, label_path))
                                        if len(examples_found) >= max_examples:
                                            return examples_found
                                        break
                                except ValueError:
                                    pass
                except Exception:
                    pass
    return examples_found

def draw_boxes_on_image(image_path, label_path, target_class_id):
    """Dibuja cajas delimitadoras en una imagen"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, box_width, box_height = map(float, parts[1:])

                        x1 = int((x_center - box_width / 2) * w)
                        y1 = int((y_center - box_height / 2) * h)
                        x2 = int((x_center + box_width / 2) * w)
                        y2 = int((y_center + box_height / 2) * h)

                        color = BOX_COLORS.get(class_id, (200, 200, 200))
                        thickness = 2

                        if class_id == target_class_id:
                            color = TARGET_CLASS_HIGHLIGHT_COLOR
                            thickness = 3

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                        class_name = CLASS_NAMES_MAP.get(class_id, f'Clase {class_id}')
                        text = f"{class_name}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        font_thickness = 1
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10

                        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5),
                                      (text_x + text_size[0] + 5, text_y + 5),
                                      color, -1)

                        cv2.putText(img, text, (text_x + 2, text_y), font, font_scale, 
                                  (255, 255, 255), font_thickness, cv2.LINE_AA)

        _, buffer = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{encoded_image}"

    except Exception as e:
        print(f"Error processing image {image_path} or label {label_path}: {e}")
        return ""

def create_overview_layout():
    """Crea el layout para la pestaña de exploración"""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Exploración del Dataset", 
                           className="section-title text-center my-4 text-primary"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Información General del Dataset", className="card-title text-primary")),
                dbc.CardBody(id="dataset-overview")
            ], className="mb-4 shadow-sm h-100 bg-dark text-light"), lg=6, md=12),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Distribución de Clases", className="card-title text-primary")),
                dbc.CardBody(dcc.Graph(id="class-distribution", config={'displayModeBar': False},
                                       figure={"layout": {"paper_bgcolor": "#222", "plot_bgcolor": "#222"}}))
            ], className="mb-4 shadow-sm h-100 bg-dark text-light"), lg=6, md=12)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Ejemplos por Clase", className="card-title text-primary")),
                dbc.CardBody([
                    dbc.Label("Selecciona una clase:", className="form-label text-light"),
                    dcc.Dropdown(
                        id='class-selector',
                        placeholder="Selecciona una clase para ver ejemplos",
                        className="mb-3 dbc"
                    ),
                    html.Div(id="class-examples", className="py-3")
                ])
            ], className="mb-4 shadow-sm bg-dark text-light"), width=12)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Estadísticas Detalladas del Dataset", className="card-title text-primary")),
                dbc.CardBody(html.Div(id="dataset-stats"))
            ], className="mb-4 shadow-sm bg-dark text-light"), width=12)
        ])
    ], fluid=True, className="py-4 bg-dark")

def register_exploration_callbacks(app):
    """Registra los callbacks para la funcionalidad de exploración"""
    
    @app.callback(
        [Output("dataset-overview", "children"),
         Output("class-distribution", "figure"),
         Output("class-selector", "options"),
         Output("dataset-stats", "children")],
        Input("tabs", "active_tab")  # FIXED: Changed from "value" to "active_tab"
    )
    def update_overview_and_stats(tab):
        if tab != 'tab-exploration':  # FIXED: Changed from 'tab-overview' to 'tab-exploration'
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        dataset_key = "general"
        analysis_results = analyze_dataset_labels(DATASET_BASE_PATHS.get(dataset_key))

        total_images = analysis_results['total_images']
        class_counts_raw = analysis_results['class_counts']
        image_resolutions = analysis_results['resolutions']
        class_names = get_class_names_for_display()

        # Dataset Overview
        class_counts_named = {CLASS_NAMES_MAP.get(int(cid), f'Clase {cid}'): count 
                            for cid, count in class_counts_raw.items()}
        full_class_counts = {name: class_counts_named.get(name, 0) for name in class_names}
        sorted_class_names = list(full_class_counts.keys())
        sorted_class_counts = list(full_class_counts.values())

        overview = html.Div([
            html.P(f"Total de imágenes analizadas: ", className="mb-1 text-muted"),
            html.H3(f"{total_images:,}", className="text-primary display-5 fw-bold mb-3"),
            html.P(f"Número de clases detectadas: {len(class_names)}", className="mb-1 text-light"),
            html.P(f"Datos de labels procesados desde archivos .txt", className="small text-muted")
        ])

        # Class Distribution Graph
        df_distribution = pd.DataFrame({'Class': sorted_class_names, 'Count': sorted_class_counts})
        fig_class_distribution = px.bar(df_distribution, x='Class', y='Count',
                     title="Conteo de Detecciones por Clase",
                     color='Count',
                     color_continuous_scale=PROFESSIONAL_COLORS,
                     labels={'Count': 'Número de Detecciones', 'Class': 'Clase'},
                     template="plotly_dark")
        fig_class_distribution.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20),
                          paper_bgcolor="#2c3e50", plot_bgcolor="#2c3e50",
                          font=dict(color="white"))
        fig_class_distribution.update_xaxes(showgrid=False, zeroline=False, color="white")
        fig_class_distribution.update_yaxes(gridcolor='#444', zeroline=False, color="white")

        # Class Selector Options
        options_class_selector = [{'label': name, 'value': i} for i, name in enumerate(class_names)]

        # Detailed Dataset Stats
        total_detections = sum(full_class_counts.values())
        class_data_for_table = []
        for class_name in class_names:
            count = full_class_counts[class_name]
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            class_data_for_table.append({
                "Clase": class_name,
                "Cantidad": f"{count:,}",
                "Porcentaje": f"{percentage:.1f}%"
            })

        # Mock train/val/test split values
        split_values = [70, 20, 10]
        split_names = ['Train', 'Validation', 'Test']

        stats_content = dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("Distribución Train/Val/Test", className="card-title text-primary")),
                dbc.CardBody(
                    dcc.Graph(
                        figure=px.pie(values=split_values,
                                      names=split_names,
                                      title="División del Dataset",
                                      hole=.3,
                                      color_discrete_sequence=px.colors.qualitative.Plotly,
                                      template="plotly_dark"),
                        config={'displayModeBar': False}
                    )
                )
            ], className="h-100 shadow-sm bg-dark text-light"), lg=4, md=6, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("Resoluciones de Imágenes", className="card-title text-primary")),
                dbc.CardBody(
                    dcc.Graph(
                        figure=px.histogram(x=image_resolutions,
                                          title="Distribución de Ancho de Imágenes",
                                          labels={'x': 'Ancho (px)', 'y': 'Frecuencia'},
                                          nbins=20 if image_resolutions else 10,
                                          color_discrete_sequence=[px.colors.sequential.Plasma[4]],
                                          template="plotly_dark"),
                        config={'displayModeBar': False}
                    )
                )
            ], className="h-100 shadow-sm bg-dark text-light"), lg=4, md=6, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5("Balance de Clases (Detecciones)", className="card-title text-primary")),
                dbc.CardBody(
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Clase", className="text-primary"),
                                    html.Th("Cantidad", className="text-primary text-end"),
                                    html.Th("Porcentaje", className="text-primary text-end")
                                ])
                            ),
                            html.Tbody(
                                [
                                    html.Tr([
                                        html.Td(row["Clase"]),
                                        html.Td(row["Cantidad"], className="text-end"),
                                        html.Td(row["Porcentaje"], className="text-end")
                                    ], className="text-light")
                                    for row in class_data_for_table
                                ]
                            )
                        ],
                        bordered=True, hover=True, responsive=True, className="table-sm table-dark"
                    )
                )
            ], className="h-100 shadow-sm bg-dark text-light"), lg=4, md=12, className="mb-4")
        ])

        return overview, fig_class_distribution, options_class_selector, stats_content

    @app.callback(
        Output("class-examples", "children"),
        Input("class-selector", "value")
    )
    def show_class_examples(selected_class_index):
        if selected_class_index is None:
            return dbc.Alert("Selecciona una clase del menú desplegable para ver ejemplos.", 
                           color="secondary", className="text-center mt-3")

        class_names = get_class_names_for_display()

        if selected_class_index < len(class_names):
            class_name = class_names[selected_class_index]
            
            examples_data = get_images_for_class(selected_class_index, max_examples=4)

            if not examples_data:
                return dbc.Alert(f"No se encontraron ejemplos para la clase '{class_name}'.", 
                               color="warning", className="text-center mt-3")

            examples_list = []
            for img_path, label_path in examples_data:
                encoded_image_with_boxes = draw_boxes_on_image(img_path, label_path, selected_class_index)

                examples_list.append(
                    dbc.Col(dbc.Card([
                        dbc.CardImg(src=encoded_image_with_boxes, top=True, className="example-card-img"),
                        dbc.CardBody([
                            html.H6(f"Clase: {class_name}", className="card-title text-light text-center"),
                        ])
                    ], className="h-100 bg-secondary text-light border-0"), md=3, xs=6, className="mb-3")
                )

            examples = html.Div([
                html.H4(f"Ejemplos de: {class_name}", className="mb-3 text-center text-primary"),
                dbc.Row(examples_list, justify="center")
            ])

            return examples
        return dbc.Alert("Clase no válida.", color="danger", className="text-center mt-3")

def get_exploration_tab_content():
    """Función principal que retorna el contenido de la pestaña de exploración"""
    return create_overview_layout()