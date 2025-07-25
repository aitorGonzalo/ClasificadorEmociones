# Clasificador de Emociones en Texto

Este proyecto es un clasificador de emociones basado en texto en inglés. Usa un modelo de regresión logística entrenado con características TF-IDF y procesamiento básico de texto (tokenización, eliminación de stopwords y limpieza).

## Requisitos

- Python 3.x
- pandas
- nltk
- scikit-learn

## Instalación

1. Clona este repositorio (o copia los archivos en tu máquina).

2. Instala las dependencias usando pip:

```bash
pip install -r requirements.txt

    Descarga los datasets y colócalos en la carpeta datos/raw con los nombres:

    train.txt

    val.txt

    test.txt

Cada archivo debe tener el formato:
<texto>;<emocion>
Uso

Ejecuta el script principal:

python3 main.py

El script realizará:

    Carga de datos

    Preprocesamiento (minusculas, eliminación de signos, stopwords)

    Vectorización TF-IDF

    Entrenamiento con regresión logística

    Evaluación sobre los conjuntos de validación y test, mostrando métricas (accuracy, precision, recall, f1-score).

Estructura de archivos

    main.py: código principal para entrenar y evaluar el modelo

    datos/raw/: carpeta con los archivos de datos

Notas

    Asegúrate de tener descargados los recursos de NLTK (punkt y stopwords). El script los descargará automáticamente si es necesario.

    Puedes modificar la ruta a los datasets en la función load_datasets() si tus archivos están en otra ubicación.

    El modelo actual es básico, ideal para comenzar y experimentar con técnicas más avanzadas.