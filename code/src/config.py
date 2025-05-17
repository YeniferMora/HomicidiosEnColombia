import os
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Base Directory: {BASE_DIR}")
print(f"Current Directory: {os.getcwd()}")
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'data_homicidios.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'homicidios_preprocesados.csv'

# Configuración de análisis
TARGET_VARIABLE = 'Circunstancia del Hecho'
CATEGORICAL_VARS = [
    'Sexo de la victima', 
    'Grupo de edad de la victima', 
    'Zona del Hecho',
    'Escenario del Hecho', 
    'Manera de muerte'
]