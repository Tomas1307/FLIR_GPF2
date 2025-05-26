# exploracion.py
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

CLASS_NAMES_MAP = {
    0: 'Vehiculos',
    1: 'Bodegas',
    2: 'Caminos',
    3: 'Rios',
    4: 'Zonas de mineria ilegal'
}

# Nueva paleta de colores para gráficos, más adecuada para un tema claro
PROFESSIONAL_COLORS = px.colors.qualitative.D3 # O px.colors.qualitative.Pastel

BOX_COLORS = {
    0: (255, 100, 100),
    1: (100, 255, 100),
    2: (100, 100, 255),
    3: (255, 255, 100),
    4: (255, 100, 255),
}

TARGET_CLASS_HIGHLIGHT_COLOR = (255, 200, 0)

# Base directories for datasets
DATASET_BASE_PATHS = {
    "general": "YOLO_data/modelo_yolov11_dataset_filtrado",
}

# Cache for dataset analysis results
_dataset_analysis_cache = {}

# --- FUNCIONES DE UTILIDAD PARA CARGA DE DATOS ---

def analyze_dataset_labels(dataset_path):
    """
    Analiza el dataset en la ruta especificada para contar clases, imágenes y resoluciones.
    Carga los resultados desde la caché si están disponibles.
    AJUSTADO PARA LA ESTRUCTURA: base_path/split/images y base_path/split/labels
    """
    if dataset_path in _dataset_analysis_cache:
        return _dataset_analysis_cache[dataset_path]

    class_counts = defaultdict(int)
    image_resolutions = []
    total_images_scanned = 0
    annotated_images_count = 0
    total_detections_count = 0
    split_counts = defaultdict(int)

    # Rutas para las carpetas de splits (train, val, test)
    for split in ['train', 'val', 'test']:
        # Corregido: images y labels están DENTRO de la carpeta split
        split_images_dir = os.path.join(dataset_path, split, 'images')
        split_labels_dir = os.path.join(dataset_path, split, 'labels')

        if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
            print(f"Advertencia: Directorios '{split}' no encontrados o incompletos en {dataset_path}")
            continue

        for img_file in os.listdir(split_images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                total_images_scanned += 1
                split_counts[split] += 1
                img_path = os.path.join(split_images_dir, img_file)

                try:
                    with Image.open(img_path) as img:
                        image_resolutions.append({'width': img.width, 'height': img.height})
                except Exception as e:
                    print(f"Advertencia: No se pudo leer la imagen {img_path}: {e}")
                    continue

                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(split_labels_dir, label_file)

                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    annotated_images_count += 1
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                try:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    total_detections_count += 1
                                except ValueError:
                                    print(f"Advertencia: Formato de etiqueta inválido en {label_path}: {line}")
                                    continue
    # Convertir class_counts a un formato más manejable
    class_df = pd.DataFrame([
        {'class_id': k, 'class_name': CLASS_NAMES_MAP.get(k, f'Clase Desconocida {k}'), 'count': v}
        for k, v in class_counts.items()
    ])
    class_df = class_df.sort_values(by='count', ascending=False)

    # Calcular porcentajes para el split
    split_df = pd.DataFrame([
        {'split': k, 'count': v, 'percentage': (v / total_images_scanned) * 100 if total_images_scanned > 0 else 0}
        for k, v in split_counts.items()
    ])

    results = {
        'total_images_scanned': total_images_scanned,
        'annotated_images_count': annotated_images_count,
        'total_detections_count': total_detections_count,
        'class_distribution': class_df,
        'image_resolutions': pd.DataFrame(image_resolutions),
        'split_distribution': split_df
    }
    _dataset_analysis_cache[dataset_path] = results
    return results

def get_images_for_class(target_class_id, max_examples=4):
    """
    Busca rutas de imágenes y sus etiquetas que contengan la clase objetivo.
    AJUSTADO PARA LA ESTRUCTURA: base_path/split/images y base_path/split/labels
    """
    dataset_path = DATASET_BASE_PATHS["general"] # Usar el dataset general para exploración

    found_examples = []

    # Buscar en train, val, test
    for split in ['train', 'val', 'test']:
        # Corregido: images y labels están DENTRO de la carpeta split
        split_labels_dir = os.path.join(dataset_path, split, 'labels')
        split_images_dir = os.path.join(dataset_path, split, 'images')

        if os.path.exists(split_labels_dir) and os.path.exists(split_images_dir):
            for label_file in os.listdir(split_labels_dir):
                if label_file.lower().endswith('.txt'): # Asegurarse de que sea un archivo .txt
                    label_path = os.path.join(split_labels_dir, label_file)
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts and int(parts[0]) == target_class_id:
                                    # Intentar encontrar el archivo de imagen correspondiente
                                    img_base_name = os.path.splitext(label_file)[0]
                                    img_extensions = ['.jpg', '.jpeg', '.png', '.gif'] # Posibles extensiones de imagen
                                    img_path = None
                                    for ext in img_extensions:
                                        potential_img_path = os.path.join(split_images_dir, img_base_name + ext)
                                        if os.path.exists(potential_img_path):
                                            img_path = potential_img_path
                                            break

                                    if img_path:
                                        found_examples.append((img_path, label_path))
                                        if len(found_examples) >= max_examples:
                                            return found_examples
                                    break # Encontró la clase en esta etiqueta, pasa a la siguiente
                    except Exception as e:
                        print(f"Advertencia: Error leyendo etiqueta {label_path}: {e}")
                        continue
        else:
            print(f"Advertencia: Directorios '{split}' no encontrados en {dataset_path}")

    return found_examples

def draw_boxes_on_image(image_path, label_path, target_class_id=None):
    """
    Dibuja bounding boxes en una imagen basada en un archivo de etiquetas YOLO.
    Resalta la clase objetivo si se especifica.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) != 5: # Asegurarse de que el formato YOLO sea correcto
                        continue
                    class_id = int(parts[0])
                    # Coordenadas YOLO normalizadas a píxeles
                    x_center, y_center, bbox_width, bbox_height = parts[1:]

                    # Convertir a coordenadas de esquina (x_min, y_min, x_max, y_max)
                    x_min = int((x_center - bbox_width / 2) * w)
                    y_min = int((y_center - bbox_height / 2) * h)
                    x_max = int((x_center + bbox_width / 2) * w)
                    y_max = int((y_center + bbox_height / 2) * h)

                    color = BOX_COLORS.get(class_id, (200, 200, 200)) # Color por defecto si no está en el mapa
                    thickness = 2

                    # Resaltar la clase objetivo
                    if target_class_id is not None and class_id == target_class_id:
                        color = TARGET_CLASS_HIGHLIGHT_COLOR
                        thickness = 3 # Un poco más grueso para destacar

                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

                    class_name = CLASS_NAMES_MAP.get(class_id, f'Clase {class_id}')
                    text = f"{class_name}"

                    # Ajustar posición del texto para que no se salga de la imagen
                    # Obtener tamaño del texto
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    # Calcular posición del texto (encima del bbox, o dentro si no hay espacio arriba)
                    text_org_x = x_min
                    text_org_y = y_min - 10 if y_min - 10 > text_height + 5 else y_min + text_height + 5

                    # Dibujar un fondo para el texto para mejorar la legibilidad
                    cv2.rectangle(img, (text_org_x, text_org_y - text_height - 5),
                                  (text_org_x + text_width + 5, text_org_y + baseline + 5),
                                  color, -1) # Fondo del mismo color que el bounding box

                    cv2.putText(img, text, (text_org_x + 2, text_org_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA) # Texto en negro

        # Codificar la imagen a base64
        _, buffer = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{encoded_image}"
    except Exception as e:
        print(f"Error procesando imagen o etiqueta: {e}")
        return None # O retorna una imagen de placeholder de error


# --- LAYOUT DE LA PESTAÑA ---

def create_overview_layout():
    """
    Crea el layout para la pestaña de exploración de datos.
    """
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Exploración de Datos del Dataset", className="text-center text-dark mb-4 display-5"), width=12)
        ]),

        # Información General del Dataset - Estilo de tarjetas de métricas
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Total de Imágenes", className="card-title text-muted text-center"),
                    html.H3(id="total-images-count", className="text-primary text-center display-4 mb-0") # Número grande
                ])
            ], className="shadow-sm border-0 h-100"), md=6, lg=3, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Clases Detectadas", className="card-title text-muted text-center"),
                    html.H3(id="detected-classes-count", className="text-primary text-center display-4 mb-0") # Número grande
                ])
            ], className="shadow-sm border-0 h-100"), md=6, lg=3, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Imágenes con anotaciones", className="card-title text-muted text-center"),
                    html.H3(id="annotated-images-count", className="text-primary text-center display-4 mb-0") # Número grande
                ])
            ], className="shadow-sm border-0 h-100"), md=6, lg=3, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Total de Detecciones", className="card-title text-muted text-center"),
                    html.H3(id="total-detections-count", className="text-primary text-center display-4 mb-0") # Número grande
                ])
            ], className="shadow-sm border-0 h-100"), md=6, lg=3, className="mb-4"),

        ], className="mb-5 justify-content-center"), # Espacio extra y centrar

        # Gráfico de Distribución de Clases
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Distribución de Clases en el Dataset", className="text-dark card-title")),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='class-distribution-chart',
                                  config={'displayModeBar': False},
                                  style={'height': '400px'})
                    )
                ])
            ], className="shadow-sm border-0 h-100"), md=12, lg=6, className="mb-4"),

            # Ejemplos por Clase
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Ejemplos de Imágenes por Clase", className="text-dark card-title")),
                dbc.CardBody([
                    dbc.Label("Selecciona una Clase:", className="text-dark mb-2"),
                    dcc.Dropdown(
                        id='class-dropdown',
                        options=[{'label': name, 'value': idx} for idx, name in CLASS_NAMES_MAP.items()],
                        placeholder="Selecciona una clase",
                        className="mb-3", # Espacio debajo del dropdown
                        clearable=False,
                        value=0 # Valor por defecto
                    ),
                    dcc.Loading(
                        html.Div(id='class-examples-container', className="mt-3")
                    )
                ])
            ], className="shadow-sm border-0 h-100"), md=12, lg=6, className="mb-4")
        ], className="mb-5"), # Espacio extra

        # Estadísticas Detalladas del Dataset
        dbc.Row([
            dbc.Col(html.H3("Estadísticas Detalladas del Dataset", className="text-dark mb-4 text-center"), width=12),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Distribución de Datos (Train/Val/Test)", className="text-dark card-title")),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='split-distribution-chart',
                                  config={'displayModeBar': False},
                                  style={'height': '350px'})
                    )
                ])
            ], className="shadow-sm border-0 h-100"), md=12, lg=6, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Distribución de Anchos de Imagen", className="text-dark card-title")),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='image-width-distribution-chart',
                                  config={'displayModeBar': False},
                                  style={'height': '350px'})
                    )
                ])
            ], className="shadow-sm border-0 h-100"), md=12, lg=6, className="mb-4"),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H4("Balance de Clases (Tabla)", className="text-dark card-title")),
                dbc.CardBody([
                    dbc.Table(id='class-balance-table', bordered=True, responsive=True, className="mt-3 mb-0 table-hover"), # Quitar table-dark
                ])
            ], className="shadow-sm border-0 h-100"), md=12, className="mb-4"),

        ], className="mb-4") # Espacio extra
    ], fluid=True, className="py-4") # Espacio padding interno

# --- CALLBACKS ---

def register_exploration_callbacks(app):
    """Registra los callbacks para la funcionalidad de exploración de datos."""

    @app.callback(
        [Output('total-images-count', 'children'),
         Output('detected-classes-count', 'children'),
         Output('annotated-images-count', 'children'),
         Output('total-detections-count', 'children'),
         Output('class-distribution-chart', 'figure'),
         Output('split-distribution-chart', 'figure'),
         Output('image-width-distribution-chart', 'figure'),
         Output('class-balance-table', 'children'),
         Output('class-dropdown', 'options')], # Añadir este output para actualizar las opciones del dropdown
        [Input('tabs', 'active_tab')]
    )
    def update_overview_and_stats(tab_value):
        # Este callback debe activarse solo si la pestaña de exploración es la activa
        if tab_value != 'tab-exploration':
            # Retorna valores por defecto o no actualiza si no es la pestaña activa
            return ("N/A", "N/A", "N/A", "N/A", go.Figure(), go.Figure(), go.Figure(),
                    html.Div("Selecciona la pestaña para cargar los datos.", className="text-center text-muted"),
                    [{'label': name, 'value': idx} for idx, name in CLASS_NAMES_MAP.items()]) # Mantener opciones del dropdown

        analysis_results = analyze_dataset_labels(DATASET_BASE_PATHS["general"])

        if analysis_results is None or analysis_results['total_images_scanned'] == 0:
            # Retorna un estado de error o vacío si no se pudo analizar o no hay imágenes
            empty_figure = go.Figure().update_layout(
                paper_bgcolor='white', plot_bgcolor='#f8f9fa', font=dict(color='black'),
                title_text="No hay datos disponibles"
            )
            return ("0", "0", "0", "0", empty_figure, empty_figure, empty_figure,
                    dbc.Alert("No se pudieron cargar los datos del dataset. Verifica la ruta y el contenido.", color="danger", className="text-center mt-3"),
                    [{'label': name, 'value': idx} for idx, name in CLASS_NAMES_MAP.items()])

        total_images = analysis_results['total_images_scanned']
        annotated_images_count = analysis_results['annotated_images_count']
        total_detections_count = analysis_results['total_detections_count']
        class_df = analysis_results['class_distribution']
        image_res_df = analysis_results['image_resolutions']
        split_df = analysis_results['split_distribution']

        detected_classes_count = len(class_df) if not class_df.empty else 0

        # Gráfico de Distribución de Clases (Barras)
        fig_class_dist = px.bar(
            class_df,
            x='class_name',
            y='count',
            title='Frecuencia de Aparición de Clases',
            labels={'class_name': 'Clase', 'count': 'Número de Detecciones'},
            color='class_name',
            color_discrete_sequence=PROFESSIONAL_COLORS
        )
        fig_class_dist.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(color='black'),
            title_font_color='black',
            xaxis_title_font_color='black',
            yaxis_title_font_color='black',
            xaxis=dict(tickfont=dict(color='black'), gridcolor='#e0e0e0'),
            yaxis=dict(tickfont=dict(color='black'), gridcolor='#e0e0e0'),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Gráfico de Distribución de Split (Pie Chart)
        fig_split_dist = px.pie(
            split_df,
            values='percentage',
            names='split',
            title='Distribución Train/Val/Test',
            color_discrete_sequence=PROFESSIONAL_COLORS
        )
        fig_split_dist.update_traces(textinfo='percent+label')
        fig_split_dist.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(color='black'),
            title_font_color='black',
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Gráfico de Distribución de Anchos de Imagen (Histograma)
        fig_img_width_dist = px.histogram(
            image_res_df,
            x='width',
            nbins=20,
            title='Distribución de Anchos de Imagen',
            labels={'width': 'Ancho de Imagen (píxeles)', 'count': 'Número de Imágenes'},
            color_discrete_sequence=[PROFESSIONAL_COLORS[0]]
        )
        fig_img_width_dist.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            font=dict(color='black'),
            title_font_color='black',
            xaxis_title_font_color='black',
            yaxis_title_font_color='black',
            xaxis=dict(tickfont=dict(color='black'), gridcolor='#e0e0e0'),
            yaxis=dict(tickfont=dict(color='black'), gridcolor='#e0e0e0'),
            bargap=0.1,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Tabla de Balance de Clases
        table_header = [
            html.Thead(html.Tr([
                html.Th("Clase", className="text-center"),
                html.Th("Conteo", className="text-center"),
                html.Th("Porcentaje (%)", className="text-center")
            ], className="table-light"))
        ]

        table_body = []
        if not class_df.empty:
            total_detections_for_table = class_df['count'].sum()
            for index, row in class_df.iterrows():
                percentage = (row['count'] / total_detections_for_table * 100) if total_detections_for_table > 0 else 0
                table_body.append(
                    html.Tr([
                        html.Td(row['class_name'], className="text-dark text-center"),
                        html.Td(row['count'], className="text-dark text-center"),
                        html.Td(f"{percentage:.2f}%", className="text-dark text-center")
                    ])
                )
        else:
            table_body.append(html.Tr(html.Td("No hay datos de clases disponibles.", colSpan=3, className="text-center text-muted")))

        class_balance_table = dbc.Table(table_header + [html.Tbody(table_body)],
                                        bordered=True, responsive=True,
                                        className="mt-3 mb-0 table-hover")

        # Opciones para el Dropdown de clases
        dropdown_options = [{'label': name, 'value': idx} for idx, name in CLASS_NAMES_MAP.items()]


        return (total_images, detected_classes_count, annotated_images_count, total_detections_count,
                fig_class_dist, fig_split_dist, fig_img_width_dist, class_balance_table, dropdown_options)


    @app.callback(
        Output('class-examples-container', 'children'),
        Input('class-dropdown', 'value')
    )
    def show_class_examples(selected_class_index):
        if selected_class_index is not None and selected_class_index in CLASS_NAMES_MAP:
            class_name = CLASS_NAMES_MAP[selected_class_index]

            examples_data = get_images_for_class(selected_class_index, max_examples=4)

            if not examples_data:
                return dbc.Alert(f"No se encontraron ejemplos para la clase '{class_name}'.",
                               color="info", className="text-center mt-3 text-dark")

            examples_list = []
            for img_path, label_path in examples_data:
                encoded_image_with_boxes = draw_boxes_on_image(img_path, label_path, selected_class_index)

                if encoded_image_with_boxes: # Solo añadir si la imagen se procesó correctamente
                    examples_list.append(
                        dbc.Col(dbc.Card([
                            dbc.CardImg(src=encoded_image_with_boxes, top=True, className="example-card-img"),
                            dbc.CardBody([
                                html.H6(f"{class_name}", className="card-title text-dark text-center"),
                            ])
                        ], className="h-100 shadow-sm border-0 bg-white"), md=3, xs=6, className="mb-3")
                    )

            if not examples_list: # Si no se pudo procesar ninguna imagen a pesar de encontrar rutas
                 return dbc.Alert(f"Se encontraron rutas, pero no se pudieron cargar o procesar ejemplos para la clase '{class_name}'.",
                               color="warning", className="text-center mt-3 text-dark")


            examples = html.Div([
                html.H4(f"Ejemplos de: {class_name}", className="mb-3 text-center text-dark"),
                dbc.Row(examples_list, justify="center")
            ])

            return examples
        return dbc.Alert("Clase no válida. Por favor, selecciona una.", color="warning", className="text-center mt-3 text-dark")

def get_exploration_tab_content():
    """Función principal que retorna el contenido de la pestaña de exploración."""
    return create_overview_layout()