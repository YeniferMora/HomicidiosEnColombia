import pandas as pd
import numpy as np
from .config import PROCESSED_DATA_PATH, TARGET_VARIABLE, CATEGORICAL_VARS

def preprocess_data(df):
    """Realiza todas las operaciones de preprocesamiento."""
    
    # 1. Eliminación de columnas no relevantes
    columns_to_drop = ['Edad judicial', 'Ciclo Vital', 'Código Dane Municipio', 'Código Dane Departamento']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # 2. Limpieza de datos categóricos
    if "Escenario del Hecho" in df.columns:
        df["Escenario del Hecho"] = df["Escenario del Hecho"].str.strip().str.lower()
        df["Escenario del Hecho"] = df["Escenario del Hecho"].replace({
            "en vía pública": "vía pública",
            "via pública": "vía pública",
            "en la via publica": "vía pública",
            "vía publica": "vía pública"
        })
    
    # 3. Eliminar duplicados
    df = df.drop_duplicates()
    
    # 4. Manejo de valores faltantes
    fecha_cols = ['Año del hecho', 'Mes del hecho', 'Dia del hecho']
    df = df.dropna(subset=fecha_cols)
    
    for col in fecha_cols:
        df = df[~df[col].astype(str).str.strip().str.lower().eq("sin información")]
    
    # 5. Transformación de variables categóricas
    # (Aquí irían todas las transformaciones como el mapeo de circunstancias, etc.)
    
    return df

def save_processed_data(df, file_path=PROCESSED_DATA_PATH):
    """Guarda los datos preprocesados en un archivo CSV."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Datos preprocesados guardados en: {file_path}")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        raise