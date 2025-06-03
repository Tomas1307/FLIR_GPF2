# Detección de Campamentos de Minería Ilegal en Videos de Cámaras FLIR

Este proyecto propone un modelo basado en YOLOv11 para la detección de campamentos de minería ilegal en videos aéreos capturados con cámaras FLIR. El objetivo principal es maximizar el recall para la detección de zonas de minería ilegal mediante búsqueda exhaustiva de hiperparámetros y estrategias de preprocesamiento optimizadas.

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
deteccion-mineria-ilegal/
├── data/
│   ├── modelo_yolov11_dataset_completo/           # Dataset original
│   ├── modelo_yolov11_dataset_completo_preprocesado/  # Dataset con CLAHE
│   └── augmented_data/                           # Datos con augmentación offline
├── models/
│   ├── conservative_final/                       # Modelo ganador
│   ├── hyperparameter_search/                    # Resultados de búsqueda
│   └── configs/                                  # Configuraciones YAML
├── scripts/
│   ├── final_training.py                        # Entrenamiento configuración Conservative
│   ├── hyperparameter_search.py                 # Búsqueda sistemática
│   ├── preprocess.py                            # Preprocesamiento CLAHE
│   └── augmentation.py                          # Augmentación híbrida
├── notebooks/
│   ├── hyperparameter_analysis.ipynb           # Análisis comparativo
│   ├── model_evaluation.ipynb                  # Evaluación detallada
│   └── results_visualization.ipynb             # Visualización de resultados
└── deployment/
    ├── conservative_mining_detector.pt         # Modelo final optimizado
    └── inference_app.py                        # Aplicación de inferencia
```

## Requisitos de Instalación

### Dependencias Principales

```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
albumentations>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
scikit-learn>=1.3.0
```

### Requisitos de Hardware

**Utilizados en el proyecto:**
- GPU: RTX 4090 24GB VRAM
- RAM: 32GB
- Tiempo de entrenamiento: ~65 minutos por configuración

**Mínimos recomendados:**
- GPU: 8GB+ VRAM
- RAM: 16GB
- Almacenamiento: 50GB libres

## Uso

### Entrenamiento con Configuración Conservative

```bash
python scripts/final_training.py
```

### Búsqueda de Hiperparámetros

```bash
python scripts/hyperparameter_search.py --configs 4 --datasets 2
```

### Preprocesamiento de Datos

```bash
python scripts/preprocess.py --input_dir data/original --output_dir data/preprocessed --apply_clahe --apply_median_filter
```

### Inferencia

```python
from ultralytics import YOLO

model = YOLO('deployment/conservative_mining_detector.pt')
results = model.predict('test_image.jpg', conf=0.15, iou=0.6)
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
