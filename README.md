# MODELOSS

Aplicación web basada en Streamlit para la clasificación de imágenes médicas del dataset Herlev, utilizando arquitecturas de Deep Learning (ResNet, DenseNet y Xception) con generación automatizada de reportes en PDF.

## Características
- Clasificación de imágenes utilizando múltiples modelos pre-entrenados:
  - ResNet50V2
  - ResNet101
  - DenseNet121
  - DenseNet169
  - Xception
- Interfaz interactiva con Streamlit.
- Generación de reportes detallados en formato PDF.
- Análisis de métricas de rendimiento (Accuracy, Recall, F1-score).

## Instalación
Instala las dependencias necesarias:
```bash
pip install -r requirements.txt
```

## Uso
Ejecuta la aplicación con Streamlit:
```bash
streamlit run app.py
```