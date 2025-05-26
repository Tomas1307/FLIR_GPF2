# Detección de Campamentos de Minería Ilegal en Videos de Cámaras FLIR

Este proyecto propone un modelo basado en YOLO para la detección de campamentos de minería ilegal en videos aéreos capturados con cámaras FLIR. El objetivo principal es comparar el rendimiento de modelos entrenados con y sin preprocesamiento de imágenes, buscando optimizar la detección en condiciones desafiantes como la presencia de nubes.

## 📊 Resultados Destacados

- **Mejora del 5%** en métricas clave con preprocesamiento CLAHE
- **Recall: 0.926** para detección de zonas de minería ilegal (Modelo ID 18)
- **mAP@0.5: 0.959** en el mejor modelo general (Modelo ID 15)
- **Validación cruzada (K=5)** para garantizar robustez del modelo
- **Dataset expandido** a 11,658 imágenes mediante técnicas de aumentación

## 👥 Autores

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

## 📋 Resumen

La minería ilegal en la Amazonía colombiana representa una amenaza crítica para la biodiversidad y los recursos hídricos. Este trabajo aborda este problema desarrollando un modelo de detección de objetos para identificar campamentos de minería ilegal en videos aéreos. La metodología incluye:

1.  **Aumentación de datos:** Se aplicaron técnicas como `HorizontalFlip` y `MotionBlur` para expandir el dataset a 11,658 imágenes.
2.  **Preprocesamiento de imágenes:** Se implementaron filtros como el filtro mediano y CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar la calidad visual de las imágenes.
3.  **Validación cruzada ($K=5$):** Se realizaron entrenamientos comparativos en dos variantes: con imágenes crudas y con imágenes preprocesadas.

Los resultados demuestran que el preprocesamiento de imágenes, especialmente el uso de CLAHE, mejora significativamente las métricas clave (recall: 0.926, mAP@0.5: 0.959) en un 5% en comparación con los modelos sin preprocesamiento. Esto es particularmente notable en zonas nubosas, lo que optimiza la detección de patrones críticos sin generar sobreajuste.

## 🎯 Objetivos del Proyecto

### Objetivo Principal
Desarrollar un sistema automatizado de detección de campamentos de minería ilegal en imágenes aéreas FLIR que maximice el recall (sensibilidad) para minimizar falsos negativos.

### Objetivos Específicos
- Comparar el rendimiento de modelos YOLO con y sin preprocesamiento de imágenes
- Evaluar el impacto de técnicas como CLAHE en la detección de patrones críticos
- Implementar un sistema robusto para vigilancia ambiental en tiempo real
- Establecer un marco replicable para análisis ambiental basado en visión artificial

## 🔬 Metodología

El proyecto sigue una metodología rigurosa para la detección de campamentos de minería ilegal, que se puede desglosar en las siguientes etapas:

### 1. Recopilación y Aumentación de Datos

* Se utilizó un dataset de videos capturados con cámaras FLIR.
* Para aumentar la robustez y la capacidad de generalización del modelo, se aplicaron técnicas de aumentación de datos:
    * **`HorizontalFlip`**: Inversión horizontal de las imágenes.
    * **`MotionBlur`**: Simulación de desenfoque por movimiento.
    * **`RandomBrightness`**: Modificación aleatoria del nivel de brillo.
    * **`RandomGamma`**: Ajuste del contraste no lineal.
* Gracias a estas técnicas, el dataset se expandió a un total de 11,658 imágenes.

### 2. Clases del Dataset

Se definieron **6 clases** principales para el análisis:

| Clase | Descripción | Ejemplos |
|-------|-------------|----------|
| -1 | Background (Fondo) | Imágenes sin objetos de interés |
| 0 | Vehículos | Maquinaria pesada, camiones, excavadoras |
| 1 | Bodegas | Estructuras de almacenamiento |
| 2 | Caminos | Vías no pavimentadas en entorno selvático |
| 3 | Ríos | Cuerpos de agua delimitados |
| 4 | Zonas de minería ilegal | Áreas con intervención evidente por maquinaria |

### 3. Preprocesamiento de Imágenes

Se exploraron y aplicaron técnicas de preprocesamiento para mejorar la visibilidad de los campamentos en las imágenes térmicas:

* **Filtro Mediano:** Para reducir el ruido de tipo sal y pimienta en las imágenes.
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Para mejorar el contraste local de las imágenes, lo cual es crucial para resaltar características en entornos con poca visibilidad o con presencia de nubes.

### 4. Modelo de Detección de Objetos

* Se utilizó la arquitectura **YOLO (You Only Look Once)**, específicamente **YOLOv11**, un modelo de detección de objetos en tiempo real conocido por su eficiencia y precisión.
* Se realizaron entrenamientos comparativos:
    * **Modelo sin preprocesamiento:** Entrenado directamente con las imágenes crudas.
    * **Modelo con preprocesamiento:** Entrenado con las imágenes a las que se les aplicó el filtro mediano y CLAHE.

### 5. Evaluación y Validación

* Se empleó una estrategia de **validación cruzada ($K=5$)** para evaluar el rendimiento de los modelos de manera robusta y evitar el sobreajuste.
* Se entrenaron los modelos durante hasta 150 épocas para asegurar la convergencia y comparar el desempeño.
* Las métricas clave utilizadas para la evaluación fueron:
    * **Recall:** Capacidad del modelo para identificar correctamente todos los campamentos de minería ilegal presentes.
    * **mAP@0.5 (mean Average Precision at IoU 0.5):** Una métrica estándar para la detección de objetos que mide la precisión promedio en una umbral de intersección sobre unión (IoU) de 0.5.
    * **mAP@0.5:0.95:** Promedio de mAP calculado en diferentes umbrales de IoU.

## 📈 Resultados

Los resultados obtenidos demuestran el impacto positivo del preprocesamiento de imágenes:

### Mejores Modelos Seleccionados

#### Modelo ID 15 (Mejor Desempeño General)
- **Configuración:** Filtrado: Sí, Preprocesamiento: No, Finetuning (freeze=0), 15 épocas
- **Recall general:** 0.875
- **mAP@0.5 general:** 0.896
- **Recall minería ilegal:** 0.868
- **mAP@0.5 minería ilegal:** 0.956

#### Modelo ID 18 (Mejor para Minería Ilegal)
- **Configuración:** Filtrado: Sí, Preprocesamiento: Sí, Finetuning (freeze=1), 15 épocas
- **Recall general:** 0.850
- **mAP@0.5 general:** 0.870
- **Recall minería ilegal:** 0.926 ⭐
- **mAP@0.5 minería ilegal:** 0.957

### Comparación General
* **Modelos con preprocesamiento (CLAHE):**
    * **Recall:** 0.926
    * **mAP@0.5:** 0.959
* **Modelos sin preprocesamiento:**
    * Las métricas fueron aproximadamente un 5% inferiores a las obtenidas con el preprocesamiento.

Estos hallazgos confirman que las técnicas de preprocesamiento, particularmente CLAHE, optimizan la detección de patrones críticos sin generar sobreajuste, lo que es especialmente beneficioso en la identificación de campamentos de minería ilegal en zonas con cobertura nubosa.

## 🗂️ Estructura del Proyecto

```
deteccion-mineria-ilegal/
├── data/
│   ├── raw_images/                 # Imágenes originales sin procesar
│   ├── processed_images/           # Imágenes con preprocesamiento aplicado
│   ├── annotations/                # Archivos de anotaciones en formato YOLO
│   └── augmented_data/            # Datos aumentados
├── models/
│   ├── yolov11/                   # Configuraciones del modelo YOLOv11
│   ├── trained_models/            # Modelos entrenados (.pt files)
│   └── configs/                   # Archivos de configuración YAML
├── notebooks/
│   ├── data_exploration.ipynb     # Análisis exploratorio de datos
│   ├── preprocessing.ipynb        # Notebooks de preprocesamiento
│   ├── model_training.ipynb       # Entrenamiento de modelos
│   └── model_evaluation.ipynb     # Evaluación y métricas
├── scripts/
│   ├── preprocess.py              # Script de preprocesamiento
│   ├── train.py                   # Script de entrenamiento
│   ├── evaluate.py                # Script de evaluación
│   └── predict.py                 # Script de predicción
├── config/
│   ├── config.yaml                # Configuración principal
│   └── dataset.yaml               # Configuración del dataset
├── deployment/
│   ├── app.py                     # Aplicación web de demostración
│   ├── __init__.py
│   ├── exploracion.py             # Módulo de exploración de datos
│   ├── preprocessing_imagenes.py   # Módulo de preprocesamiento
│   └── yolo_prediction.py         # Módulo de predicción YOLO
├── docs/
│   └── paper.pdf                  # Documento académico del proyecto
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## ⚙️ Requisitos de Instalación

Para ejecutar este proyecto, necesitarás instalar las siguientes dependencias. Se recomienda usar un entorno virtual.

### Crear Entorno Virtual

```bash
# Crear un entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### Instalar Dependencias

```bash
# Instalar las dependencias principales
pip install -r requirements.txt
```

### Archivo requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
albumentations>=1.3.0
scikit-learn>=1.3.0
tqdm>=4.65.0
Pillow>=9.5.0
pyyaml>=6.0
tensorboard>=2.13.0
scipy>=1.10.0
```

### Requisitos de Hardware

**Mínimos:**
- GPU con al menos 8GB VRAM (NVIDIA GTX 1070 o superior)
- 16GB RAM
- 50GB espacio libre en disco

**Recomendados:**
- GPU con 12GB+ VRAM (NVIDIA RTX 3080 o superior)
- 32GB RAM
- SSD con 100GB espacio libre

## 🚀 Uso

### Preprocesamiento de Datos

Si tienes imágenes sin preprocesar, puedes usar el script de preprocesamiento:

```bash
python scripts/preprocess.py --input_dir data/raw_images --output_dir data/processed_images --apply_clahe --apply_median_filter
```

**Parámetros:**
- `--input_dir`: Directorio con imágenes originales
- `--output_dir`: Directorio de salida para imágenes procesadas
- `--apply_clahe`: Aplicar ecualización adaptativa del histograma
- `--apply_median_filter`: Aplicar filtro mediano para reducción de ruido


**Parámetros principales:**
- `--model_type`: Variante del modelo (yolov11n, yolov11s, yolov11m, yolov11l, yolov11x)
- `--data_path`: Ruta al archivo YAML de configuración del dataset
- `--epochs`: Número de épocas de entrenamiento
- `--freeze`: Número de capas a congelar (0 = ninguna, 1 = backbone)
- `--batch_size`: Tamaño del lote


### 5. Aplicación Web de Demostración

Para ejecutar la aplicación web interactiva:

```bash
# Ejecutar la aplicación
python deployment/app.py

# La aplicación estará disponible en http://localhost:5000
```

## 📊 Configuraciones Experimentales

El proyecto incluye 20 configuraciones experimentales diferentes que varían en:

| ID | Filtrado | Preprocesamiento | Método | Épocas | Descripción |
|----|----------|------------------|--------|--------|-------------|
| 15 | ✅ | ❌ | Finetuning (freeze=0) | 15 | **Mejor desempeño general** |
| 18 | ✅ | ✅ | Finetuning (freeze=1) | 15 | **Mejor para minería ilegal** |
| 5 | ❌ | ❌ | Completo | 150 | Baseline de larga duración |
| 6 | ✅ | ✅ | Completo | 150 | Mejor preprocesamiento largo |

Ver el documento completo para todas las configuraciones evaluadas.

### Archivo dataset.yaml

```yaml
# Configuración del dataset para YOLO
train: data/train/images
val: data/val/images
test: data/test/images

nc: 5  # Número de clases
names: 
  0: vehiculos
  1: bodegas
  2: caminos
  3: rios
  4: mineria_ilegal
```

## Consideraciones Importantes

### Priorización del Recall
En este contexto, **es preferible tolerar falsos positivos que omitir zonas con actividad minera ilegal**. El modelo prioriza la detección completa (recall) sobre la precisión para garantizar que ninguna área crítica pase desapercibida.

### Limitaciones Actuales
- La clase "bodegas" presenta menor precisión debido a similitudes visuales con otras estructuras
- El rendimiento puede degradarse en condiciones de iluminación extrema
- Requiere validación humana para confirmación de alertas críticas

### Recomendaciones de Implementación
1. **Validación en dos etapas:** Usar el modelo como filtro inicial seguido de revisión humana
2. **Actualización continua:** Reentrenar periódicamente con nuevos datos
3. **Ensemble de modelos:** Combinar predicciones de múltiples configuraciones para mayor robustez

## 🔬 Trabajo Futuro

### Mejoras Técnicas Planificadas
- [ ] Implementación de YOLOv8 y comparación de rendimiento
- [ ] Integración con modelos de segmentación semántica (SAM)
- [ ] Desarrollo de un sistema de ensemble para mayor precisión
- [ ] Optimización para detección en tiempo real en dispositivos edge

### Extensiones del Dataset
- [ ] Incorporación de imágenes satelitales de diferentes sensores
- [ ] Aumento de datos con más variaciones estacionales
- [ ] Inclusión de datos de radar (SAR) para condiciones adversas
- [ ] Datos de diferentes regiones geográficas para generalización

### Funcionalidades Adicionales
- [ ] API REST para integración con sistemas externos
- [ ] Dashboard web para monitoreo en tiempo real
- [ ] Sistema de alertas automáticas
- [ ] Módulo de análisis temporal para detección de cambios


## 📄 Licencia

Este proyecto está licenciado bajo la [MIT License](LICENSE).

## 🙏 Agradecimientos

- A la **Universidad de los Andes** por el apoyo y los recursos brindados para la realización de este proyecto.
- A la comunidad de **Ultralytics** por el desarrollo y mantenimiento de YOLO.
- A los desarrolladores de **OpenCV** y **Albumentations** por las herramientas de procesamiento de imágenes.
- A las autoridades ambientales por proporcionar contexto sobre la problemática de la minería ilegal.

## 📚 Referencias

1. **Boscó Arias, J. A.** (2023). "Minería ilegal y el impacto a la seguridad ambiental en la región amazónica en Colombia." En *El crimen organizado en la Amazonía: Escenario de desafíos para la seguridad regional*. Sello Editorial ESDEG.

2. **Foro Nacional Ambiental.** (2024). *Informe Nacional: Minería ilegal y contaminación por mercurio en Colombia.*

3. **Terven, J., Córdova-Esparza, D.-M., & Romero-González, J.-A.** (2023). "A comprehensive review of YOLO architectures in computer vision: From YOLOv1 to YOLOv8 and YOLO-NAS." *Machine Learning and Knowledge Extraction*, 5(4), 1680–1716.

4. **Reza, A. M.** (2004). "Realization of the contrast limited adaptive histogram equalization (CLAHE) for real-time image enhancement." *Journal of VLSI Signal Processing Systems for Signal, Image and Video Technology*, 38, 35–44.

---

## 📞 Contacto

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