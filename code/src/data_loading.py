import pandas as pd
from pathlib import Path
from .config import RAW_DATA_PATH

def load_raw_data(file_path=RAW_DATA_PATH):
    """Carga los datos crudos desde el archivo CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Datos cargados correctamente. Dimensiones: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise