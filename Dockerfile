FROM python:3.11

# Instala las dependencias necesarias para OpenCV
# Estas librerías son comunes para resolver el error 'libGL.so.1'
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/ /app/

COPY models/ /app/models/

COPY YOLO_data/modelo_yolov11_dataset_filtrado/ /app/YOLO_data/modelo_yolov11_dataset_filtrado/
COPY YOLO_data/modelo_yolov11_dataset_completo_preprocesado/ /app/YOLO_data/modelo_yolov11_dataset_completo_preprocesado/

EXPOSE 8050

# Comando por defecto
CMD ["python", "app.py"]