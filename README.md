# Detección de Campamentos de Minería Ilegal en Videos de Cámaras FLIR

Este proyecto propone un modelo basado en YOLOv11 para la detección de campamentos de minería ilegal en videos aéreos capturados con cámaras FLIR. El objetivo principal es maximizar el recall para la detección de zonas de minería ilegal mediante búsqueda exhaustiva de hiperparámetros y estrategias de preprocesamiento optimizadas.

## Datos

Preprocesamiento dataset: https://drive.google.com/file/d/1-9u4OphZ50NWHwjtRQN1js9J3LTLU9kP/view?usp=sharing

Finetuning resultados: https://drive.google.com/file/d/1hNvvQ8y9eZCxFTN3wtEIuTm07dI5CIev/view?usp=sharing

## Tablero

http://20.42.84.158:8050/

## Resultados Destacados

- **Recall: 80.3%** para detección de zonas de minería ilegal (Configuración Conservative)
- **Precision: 86.5%** para zonas de minería ilegal
- **mAP@50: 93.4%** promedio general en todas las clases
- **Configuración Conservative** como ganadora tras búsqueda exhaustiva de hiperparámetros
- **8 experimentos** comparativos (4 configuraciones × 2 datasets)
- **Dataset preprocesado** demostró superioridad consistente (-4.2% promedio en dataset original)
- **Fine-tuning eficiente** con 15 épocas de entrenamiento
- **Umbral de confianza optimizado** (0.15) para maximizar detección crítica

## Autores

* **Tomas Acosta**  
    Ingeniería de Sistemas y Computación  
    Universidad de los Andes, Bogotá, Colombia  
    t.acosta@uniandes.edu.co

* **Juan Andrés Carrasquilla**  
    Ingeniería de Sistemas y Computación  
    Universidad de los Andes, Bogotá, Colombia  
    j.carrasquillag@uniandes.edu.co

* **Samuel Romero**  
    Ingeniería de Sistemas y Computación  
    Universidad de los Andes, Bogotá, Colombia  
    sj.romero10@uniandes.edu.co

## Resumen

La minería ilegal en la Amazonía colombiana representa una amenaza crítica para la biodiversidad y los recursos hídricos. Este trabajo aborda este problema desarrollando un modelo de detección de objetos optimizado para identificar campamentos de minería ilegal en videos aéreos. La metodología incluye:

1. **Búsqueda exhaustiva de hiperparámetros:** Se evaluaron 4 configuraciones especializadas (Ultra Recall, High Resolution, Balanced, Conservative) en 2 versiones del dataset.
2. **Augmentación híbrida:** Combinación de augmentación offline (Albumentations) y online (YOLO nativo) para maximizar robustez.
3. **Preprocesamiento optimizado:** Implementación de filtro mediano y CLAHE para mejorar detección en condiciones adversas.
4. **Fine-tuning estratégico:** Transfer learning eficiente desde YOLOv11 pre-entrenado en COCO.

Los resultados demuestran que la **configuración Conservative** con dataset preprocesado logra el mejor balance entre recall (80.3%) y estabilidad, superando enfoques más agresivos. La estrategia de umbral de confianza bajo (0.15) se justifica por la criticidad de minimizar falsos negativos en detección de minería ilegal.

## Objetivos del Proyecto

### Objetivo Principal
Desarrollar un sistema automatizado de detección de campamentos de minería ilegal que maximice el recall para minimizar falsos negativos, implementando búsqueda sistemática de hiperparámetros para optimización del rendimiento.

### Objetivos Específicos
- Comparar sistemáticamente 4 configuraciones de entrenamiento especializadas
- Evaluar el impacto del preprocesamiento (CLAHE + filtro mediano) vs dataset original
- Optimizar hiperparámetros específicamente para la clase crítica (minería ilegal)
- Validar estrategias de augmentación híbrida para clases minoritarias
- Establecer marco metodológico replicable para detección de actividades ilegales

## Metodología

### 1. Búsqueda de Hiperparámetros

Se diseñaron **4 configuraciones especializadas**, cada una probada en 2 versiones del dataset:

**Configuraciones evaluadas:**
- **Ultra Recall:** LR=0.02, Mosaic=1.0, Mixup=0.15, sin dropout
- **High Resolution:** 832px, modelo yolo11l.pt para mayor capacidad
- **Balanced:** LR=0.01, Mosaic=0.85, Mixup=0.05, enfoque equilibrado
- **Conservative:** LR=0.005, Mosaic=0.8, sin mixup, dropout=0.1 (GANADOR)

**Total: 8 experimentos** (4 configuraciones × 2 datasets)

### 2. Datasets Comparados

| Dataset | Descripción | Preprocesamiento |
|---------|-------------|------------------|
| `modelo_yolov11_dataset_completo` | Dataset original sin procesar | Ninguno |
| `modelo_yolov11_dataset_completo_preprocesado` | Dataset optimizado | CLAHE + Filtro mediano |

### 3. Estrategia de Augmentación Híbrida

**Augmentación offline (Albumentations):**
- HorizontalFlip, MotionBlur, RandomBrightness, RandomGamma
- Expansión de clase minoritaria: 91 → 4000 imágenes de minería

**Augmentación online (YOLO nativo):**
- Mosaic (0.8): Combinación de 4 imágenes para contexto diverso
- HSV augmentation: Hue (0.015), Saturation (0.7), Value (0.4)
- Geometric transforms: Scale (0.5), Translate (0.1)
- Random Erasing (0.4): Oclusión simulada
- **Mixup desactivado (0.0):** Preserva características específicas de minería

### 4. Clases del Dataset

Se definieron **5 clases** principales para el análisis:

| Clase | Descripción | Distribución |
|-------|-------------|--------------|
| 0 | Vehículos | 22 imágenes, 23 instancias |
| 1 | Bodegas/Construcciones | 249 imágenes, 897 instancias |
| 2 | Caminos | 245 imágenes, 400 instancias |
| 3 | Ríos | 114 imágenes, 169 instancias |
| 4 | Zonas de minería ilegal | 65 imágenes, 80 instancias |

### 5. Optimización de Thresholds

**Configuración optimizada para recall:**
- **Confidence threshold: 0.15** (umbral bajo para maximizar detección)
- **IoU threshold: 0.6** (balance entre duplicados y detecciones válidas)
- **Justificación:** En minería ilegal, falsos negativos son más costosos que falsos positivos

## Resultados

### Configuración Ganadora: Conservative

Identificamos que el modelo tiene mas capacidad para aprender, consideramos que pueden ser 100 o mas epocas, pero por tiempo y restricciones computacionales no pudimos evaluar mas de 48 épocas.

**Métricas del modelo Conservative con dataset preprocesado:**

| Clase | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **Todas** | **91.4%** | **90.8%** | **93.4%** | **81.5%** |
| Vehículos | 93.1% | 100.0% | 99.5% | 98.6% |
| Bodegas | 93.5% | 87.1% | 94.9% | 70.3% |
| Caminos | 94.7% | 95.5% | 95.8% | 84.8% |
| Ríos | 89.2% | 91.1% | 92.6% | 79.8% |
| **Minería ilegal** | **86.5%** | **80.3%** | **84.4%** | **74.0%** |

### Comparación entre Configuraciones

| Configuración | Recall Minería | Diferencia vs Conservative |
|---------------|----------------|---------------------------|
| **Conservative** | **80.3%** | - |
| Ultra Recall | 77.4% | -2.9% |
| Balanced | 76.6% | -3.7% |
| High Resolution | 78.1% | -2.2% |

### Impacto del Preprocesamiento

- **Dataset preprocesado vs original:** +4.2% promedio en recall
- **CLAHE + filtro mediano** mejoró consistentemente todas las configuraciones
- **Especialmente efectivo** en condiciones de baja visibilidad y presencia de nubes

## Lecciones Aprendidas

### Hallazgos Técnicos Clave

1. **Estabilidad > Agresividad:** Learning rate conservador (0.005) superó a enfoques agresivos
2. **Mixup contraproducente:** Para clases minoritarias críticas, mixup diluye características específicas
3. **Preprocesamiento fundamental:** CLAHE demostró valor consistente en todas las configuraciones
4. **Umbral optimizado válido:** Confidence=0.15 es estrategia apropiada para aumentar el recall.
5. **Fine-tuning eficiente:** 15 épocas suficientes con transfer learning adecuado

### Metodología Validada

- **Búsqueda sistemática** más efectiva que optimización manual
- **Augmentación híbrida** exitosa para datasets pequeños (91 → 4000 imágenes)
- **Evaluación comparativa** esencial para justificar decisiones técnicas

## Estructura del Proyecto

```
FLIR_GPF2/
├── app/                        # Paquete principal (ver architecture.md)
│   ├── config.py               # Configuracion centralizada (Singleton)
│   ├── schemas/                # Modelos Pydantic
│   ├── utils/                  # Funciones utilitarias
│   ├── core/                   # Operaciones sobre el dataset
│   ├── preprocessing/          # Filtros de imagen (CLAHE, ruido)
│   ├── augmentation/           # Augmentacion por clase
│   ├── training/               # Entrenamiento, HP search, fine-tuning
│   ├── adapters/               # Wrappers para YOLO y rutas
│   ├── facades/                # Orquestadores de pipeline
│   └── visualization/          # Graficas y visualizaciones
├── scripts/                    # Entry points ejecutables
│   ├── run_data_pipeline.py    # Preparacion de datos completa
│   ├── run_preprocessing.py    # CLAHE + filtro mediano
│   ├── run_training.py         # Entrenamiento conservative
│   ├── run_hyperparameter_search.py
│   ├── run_finetuning.py       # Fine-tuning con backbone congelado
│   ├── run_evaluation.py       # Evaluacion de modelo
│   └── run_full_pipeline.py    # Pipeline end-to-end
├── architecture.md             # Documentacion de arquitectura para LLMs
├── requirements.txt
└── .gitignore
```

---

## Estructura Esperada del Dataset

El pipeline espera los datos en estructuras especificas para cada etapa.
A continuacion se detalla que estructura debe tener el directorio de datos
en cada punto del pipeline.

### Etapa 1 -- Datos Crudos (entrada de `run_data_pipeline.py`)

El directorio `data/` debe contener las imagenes y etiquetas originales
separadas en carpetas por split. Las etiquetas usan formato YOLO
(`class_id x_center y_center width height`, coordenadas normalizadas 0-1).

```
data/
├── Imagenes/                         # o "images/"
│   ├── train/
│   │   ├── video_11min_001.jpg
│   │   ├── video_11min_002.jpg
│   │   └── ...
│   ├── val/                          # acepta tambien "validation/" o "valid/"
│   │   └── *.jpg
│   └── test/
│       └── *.jpg
└── Etiquetas/                        # o "labels/"
    ├── train/
    │   ├── video_11min_001.txt       # OPCIONAL: si falta, se crea vacio (background)
    │   └── ...
    ├── val/
    │   └── *.txt
    └── test/
        └── *.txt
```

**Formato de cada archivo `.txt` de etiqueta:**

```
<class_id> <x_center> <y_center> <width> <height>
```

Ejemplo con dos objetos en una imagen:

```
1 0.512345 0.345678 0.120000 0.085000
4 0.723456 0.567890 0.200000 0.150000
```

| class_id | Clase |
|----------|-------|
| 0 | Vehiculos |
| 1 | Bodegas |
| 2 | Caminos |
| 3 | Rios |
| 4 | Mineria ilegal |

> Las imagenes sin objetos (background) deben tener un archivo `.txt` vacio
> o simplemente no tener archivo de etiqueta (se crea automaticamente).

### Etapa 2 -- Dataset Unificado (salida automatica)

`DatasetUnifier` mezcla todos los splits en una estructura plana.
**No es necesario crearlo manualmente**, se genera automaticamente.

```
data_unified/
├── images/
│   ├── video_11min_001.jpg
│   ├── video_11min_002.jpg
│   └── ... (todas las imagenes juntas)
└── labels/
    ├── video_11min_001.txt
    ├── video_11min_002.txt
    └── ... (todas las etiquetas juntas)
```

### Etapa 3 -- Dataset YOLO con Augmentacion (salida automatica)

`StrategicSplitter` redistribuye los datos y `ClassAugmentor` genera
imagenes sinteticas. Los archivos aumentados siguen la convencion
`aug_c{clase}_{stem_original}_{secuencia}.jpg`.

```
yolo_dataset/
├── train/
│   ├── images/
│   │   ├── video_11min_001.jpg           # imagen original
│   │   ├── aug_c4_video_11min_098_0000.jpg   # augmentacion clase 4
│   │   ├── aug_c4_video_11min_098_0001.jpg
│   │   ├── aug_c0_video_11min_465_0000.jpg   # augmentacion clase 0
│   │   ├── aug_c-1_video_11min_002_0000.jpg  # augmentacion background
│   │   └── ...
│   └── labels/
│       ├── video_11min_001.txt
│       ├── aug_c4_video_11min_098_0000.txt
│       ├── aug_c-1_video_11min_002_0000.txt  # vacio (background)
│       └── ...
├── val/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
└── test/
    ├── images/
    │   └── *.jpg
    └── labels/
        └── *.txt
```

**Objetivos de augmentacion por clase (configurables en `app/config.py`):**

| Clase | Nombre | Objetivo |
|-------|--------|----------|
| 4 | Mineria ilegal | 3500 |
| 0 | Vehiculos | 2000 |
| 1 | Bodegas | 3000 |
| 2 | Caminos | 3000 |
| 3 | Rios | 3000 |
| -1 | Background | 2500 |

### Etapa 4 -- Dataset Preprocesado (entrada de entrenamiento)

`DatasetPreprocessor` aplica el filtro mediano y CLAHE a cada imagen.
Las etiquetas se copian sin cambios. **Este directorio es el que se usa
para entrenamiento.**

```
preprocessed/
├── train/
│   ├── images/
│   │   └── *.jpg                    # imagenes con CLAHE + filtro mediano
│   └── labels/
│       └── *.txt                    # etiquetas sin modificar
├── val/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
├── test/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
└── dataset.yaml                     # REQUERIDO para entrenamiento
```

### Archivo `dataset.yaml` (requerido para entrenamiento y fine-tuning)

El archivo `dataset.yaml` debe existir en la raiz del dataset que se
pasa al entrenador. Formato estandar de Ultralytics:

```yaml
path: /ruta/absoluta/al/dataset      # o ruta relativa desde donde se ejecuta
train: train/images
val: val/images
test: test/images

nc: 5
names:
  0: Vehicles
  1: Warehouses
  2: Roads
  3: Rivers
  4: Illegal Mining
```

> **Importante:** El `ConservativeTrainer` y el `Finetuner` buscan
> `dataset.yaml` dentro de la carpeta del dataset. Si no existe, el
> entrenamiento fallara con un error de Ultralytics.

### Etapa 5 -- Salidas de Entrenamiento (generadas automaticamente)

```
conservative_final_<nombre>_<timestamp>/
├── conservative_final_<dataset_name>/
│   └── weights/
│       ├── best.pt                  # mejor modelo (usado para evaluacion)
│       └── last.pt                  # ultimo checkpoint
├── conservative_mining_metrics.json # metricas detalladas
└── conservative_mining_detector_recall_0.XXX.pt  # copia del mejor modelo
```

### Etapa 6 -- Salidas de Fine-tuning (generadas automaticamente)

```
finetuning_results/
└── finetuning_run/
    └── weights/
        ├── best.pt
        └── last.pt
```

### Resumen del Flujo de Datos

```
data/  (datos crudos)
  |  DatasetUnifier
  v
data_unified/  (estructura plana)
  |  StrategicSplitter + ClassAugmentor + LabelCleaner + DuplicateRemover + ClassBalancer
  v
yolo_dataset/  (YOLO format, augmentado y limpio)
  |  DatasetPreprocessor
  v
preprocessed/  (CLAHE + filtro mediano) + dataset.yaml
  |  ConservativeTrainer
  v
conservative_final_*/  (modelo entrenado)
  |  Finetuner (opcional)
  v
finetuning_results/  (modelo afinado)
```

---

## Requisitos de Instalacion

### Dependencias

```bash
pip install -r requirements.txt
```

Las dependencias principales son:

```
ultralytics, torch, pydantic, pydantic-settings, albumentations,
opencv-python, numpy, pandas, matplotlib, Pillow, scikit-learn, tqdm
```

### Requisitos de Hardware

**Utilizados en el proyecto:**
- GPU: RTX 4090 24GB VRAM
- RAM: 32GB
- Tiempo de entrenamiento: ~65 minutos por configuracion

**Minimos recomendados:**
- GPU: 8GB+ VRAM
- RAM: 16GB
- Almacenamiento: 50GB libres

## Uso

### Pipeline Completo (end-to-end)

```bash
python scripts/run_full_pipeline.py
```

### Etapas Individuales

```bash
# 1. Preparacion de datos (unificar, split, augmentar, limpiar)
python scripts/run_data_pipeline.py

# 2. Preprocesamiento (CLAHE + filtro mediano)
python scripts/run_preprocessing.py

# 3. Entrenamiento conservative
python scripts/run_training.py

# 4. Busqueda de hiperparametros
python scripts/run_hyperparameter_search.py

# 5. Fine-tuning (requiere modelo entrenado)
python scripts/run_finetuning.py --model path/to/best.pt

# 6. Evaluacion
python scripts/run_evaluation.py --model path/to/best.pt
```

### Inferencia

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict("test_image.jpg", conf=0.15, iou=0.6)
```

### Verificacion Rapida de Imports

```bash
python -c "from app.config import settings; print(settings.TARGET_CLASS)"
```

## Limitaciones y Trabajo Futuro

### Limitaciones Actuales

- **Dataset limitado:** Solo 65 imágenes originales de minería ilegal
- **Validación geográfica:** Entrenado en región específica de Colombia
- **Condiciones de captura:** Principalmente cámaras FLIR en condiciones diurnas

### Trabajo Futuro

**Mejoras técnicas:**
- Implementación de YOLOv8 y modelos más recientes
- Ensemble de múltiples configuraciones
- Optimización para inferencia en tiempo real

**Expansión de datos:**
- Incorporación de datos satelitales
- Diversificación geográfica
- Datos de diferentes sensores (SAR, multiespectrales)

**Funcionalidades:**
- API REST para integración
- Sistema de monitoreo continuo
- Dashboard de alertas en tiempo real

## Consideraciones Éticas y de Implementación

### Priorización del Recall
En detección de minería ilegal, **es preferible tolerar falsos positivos que omitir actividad real**. La configuración Conservative con umbral de confianza bajo (0.15) está justificada por el alto costo de falsos negativos en vigilancia ambiental.

### Recomendaciones de Despliegue
1. **Validación humana:** Usar como sistema de filtrado inicial
2. **Actualización continua:** Reentrenamiento con nuevos datos
3. **Monitoreo de deriva:** Validación periódica en datos reales
4. **Integración gradual:** Implementación piloto antes de despliegue completo

## Contacto

Para preguntas, sugerencias o colaboraciones:

- **Tomas Acosta:** t.acosta@uniandes.edu.co
- **Juan Andrés Carrasquilla:** j.carrasquillag@uniandes.edu.co  
- **Samuel Romero:** sj.romero10@uniandes.edu.co

**Universidad de los Andes**  
Facultad de Ingeniería  
Departamento de Ingeniería de Sistemas y Computación  
Bogotá, Colombia

---

*Este proyecto fue desarrollado como parte de una investigación académica en la Universidad de los Andes para contribuir a la lucha contra la minería ilegal en la región amazónica de Colombia.*
