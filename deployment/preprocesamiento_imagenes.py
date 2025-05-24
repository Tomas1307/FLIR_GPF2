# preprocesamiento_imagenes.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import cv2
import base64
import dash_bootstrap_components as dbc
from pathlib import Path
import random

DATASET_BASE_PATH = "YOLO_data/modelo_yolov11_dataset_completo_preprocesado"

def reducir_ruido_sal_pimienta(imagen, umbral_deteccion=0.01, tamano_kernel=3):
    """
    Aplica filtro mediana solo si se detecta suficiente ruido sal y pimienta.
    
    Args:
        imagen: Imagen de entrada (BGR)
        umbral_deteccion: Porcentaje mínimo de píxeles extremos para considerar que hay ruido
        tamano_kernel: Tamaño del kernel para el filtro mediana (3 o 5 recomendado)
    
    Returns:
        Tupla (imagen_procesada, ruido_detectado)
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) > 2 else imagen.copy()

    num_pixels = gris.shape[0] * gris.shape[1]
    if num_pixels == 0: 
        return imagen.copy(), False
    
    muy_oscuros = np.sum(gris < 20) 
    muy_brillantes = np.sum(gris > 235) 

    proporcion_extremos = (muy_oscuros + muy_brillantes) / num_pixels

    if proporcion_extremos >= umbral_deteccion:
        imagen_procesada = cv2.medianBlur(imagen, tamano_kernel)
        return imagen_procesada, True
    else:
        return imagen.copy(), False

def aplicar_mejora_contraste(imagen, clip_limit=2.0, grid_size=(8, 8)):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar el contraste.
    
    Args:
        imagen: Imagen de entrada (BGR)
        clip_limit: Umbral para limitar el contraste.
        grid_size: Tamaño de la cuadrícula para el histograma adaptativo.
    
    Returns:
        Imagen con contraste mejorado.
    """
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    imagen_contraste = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return imagen_contraste

def convertir_escala_grises(imagen):
    """
    Convierte una imagen a escala de grises.
    
    Args:
        imagen: Imagen de entrada (BGR)
    
    Returns:
        Imagen en escala de grises (BGR).
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

def get_random_image_paths(num_images=3):
    """
    Obtiene rutas aleatorias de imágenes y sus correspondientes etiquetas.
    """
    dataset_base = Path(DATASET_BASE_PATH)
    all_images = []
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_base / split / "images"
        if images_dir.exists():
            split_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
            all_images.extend(split_images)
    
    if len(all_images) == 0:
        return []

    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    image_label_paths = []
    for img_path in selected_images:
        split = img_path.parent.parent.name 
        label_path = dataset_base / split / "labels" / f"{img_path.stem}.txt"
        
        if label_path.exists():
            image_label_paths.append((img_path, label_path))
        else:
            image_label_paths.append((img_path, None))
    
    return image_label_paths

def process_and_encode_image(image_path, process_type="original"):
    """
    Carga una imagen, aplica el preprocesamiento especificado y la codifica a base64.
    """
    if not image_path.exists():
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    processed_img = img.copy()
    
    if process_type == "denoised":
        processed_img, _ = reducir_ruido_sal_pimienta(img)
    elif process_type == "contrasted":
        processed_img = aplicar_mejora_contraste(img)
    elif process_type == "grayscale":
        processed_img = convertir_escala_grises(img)

    _, buffer = cv2.imencode('.png', processed_img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{encoded_image}"

def generate_preprocessing_examples(num_images=3):
    """
    Genera y muestra ejemplos de imágenes antes y después del preprocesamiento.
    """
    dataset_base = Path(DATASET_BASE_PATH)
    if not dataset_base.exists():
        return dbc.Alert(
            f"Error: La ruta del dataset no existe: {DATASET_BASE_PATH}",
            color="danger",
            className="text-center mt-3"
        )
    
    image_label_paths = get_random_image_paths(num_images)
    
    if not image_label_paths:
        checked_paths = []
        for split in ['train', 'val', 'test']:
            images_dir = dataset_base / split / "images"
            checked_paths.append(f"{split}/images: {'✓' if images_dir.exists() else '✗'}")
        
        debug_info = " | ".join(checked_paths)
        
        return dbc.Alert([
            html.P("No se encontraron imágenes en la ruta especificada para ejemplos de preprocesamiento."),
            html.P(f"Ruta base: {DATASET_BASE_PATH}", className="small"),
            html.P(f"Verificación de carpetas: {debug_info}", className="small")
        ],
            color="warning",
            className="text-center mt-3"
        )

    examples_rows = []
    for img_path, label_path in image_label_paths:
        original_img_encoded = process_and_encode_image(img_path, "original")
        
        if original_img_encoded is None:
            continue
            
        denoised_img_encoded = process_and_encode_image(img_path, "denoised")
        contrasted_img_encoded = process_and_encode_image(img_path, "contrasted")

        examples_rows.append(
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardImg(src=original_img_encoded, top=True, className="example-card-img"),
                    dbc.CardBody(html.P("Original", className="card-text text-light text-center"))
                ], className="h-100 bg-secondary text-light border-0"), md=3, xs=6, className="mb-3"),
                dbc.Col(dbc.Card([
                    dbc.CardImg(src=denoised_img_encoded, top=True, className="example-card-img"),
                    dbc.CardBody(html.P("Reducción de Ruido", className="card-text text-light text-center"))
                ], className="h-100 bg-secondary text-light border-0"), md=3, xs=6, className="mb-3"),
                dbc.Col(dbc.Card([
                    dbc.CardImg(src=contrasted_img_encoded, top=True, className="example-card-img"),
                    dbc.CardBody(html.P("Mejora de Contraste", className="card-text text-light text-center"))
                ], className="h-100 bg-secondary text-light border-0"), md=3, xs=6, className="mb-3"),
            ], justify="center", className="mb-4")
        )
    
    if not examples_rows:
        return dbc.Alert(
            "No se pudieron procesar las imágenes encontradas.",
            color="warning",
            className="text-center mt-3"
        )
    
    return examples_rows

def create_preprocessing_layout():
    """Crea el layout para la pestaña de preprocesamiento."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.Div([
                html.H2("Preprocesamiento de Imágenes", className="text-primary text-center mb-3"),
                html.P("Aquí se muestran ejemplos de las transformaciones aplicadas a las imágenes para mejorar su calidad y la detección de objetos. Las imágenes se procesan para reducir ruido, mejorar contraste y aumentar nitidez.",
                       className="text-light text-center"),
                html.Hr(className="my-4 border-light")
            ]), width=12)
        ]),
        dbc.Row(id='preprocessing-examples-container', className="my-4") 
    ], fluid=True, className="py-4 bg-dark")
    
def register_preprocessing_callbacks(app):
    """Registra los callbacks para la funcionalidad de preprocesamiento"""
    
    @app.callback(
        Output('preprocessing-examples-container', 'children'),
        Input('tabs', 'active_tab') 
    )
    def update_preprocessing_examples(tab_value):
        if tab_value == 'tab-preprocessing':
            return generate_preprocessing_examples(num_images=3)
        return html.Div([]) 

def get_preprocessing_tab_content():
    """Función principal que retorna el contenido de la pestaña de preprocesamiento"""
    return create_preprocessing_layout()