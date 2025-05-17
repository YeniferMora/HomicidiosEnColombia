import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from tabulate import tabulate
from .config import TARGET_VARIABLE, CATEGORICAL_VARS

def describe_columns(df):
    """Genera una descripción detallada de todas las columnas."""
    descripcion = []
    for columna in df.columns:
        tipo = df[columna].dtype
        nulos = df[columna].isnull().sum()
        unicos = df[columna].nunique()

        if pd.api.types.is_numeric_dtype(df[columna]):
            rango = (df[columna].min(), df[columna].max())
            media = df[columna].mean()
            mediana = df[columna].median()
        else:
            rango = None
            media = None
            mediana = None

        if pd.api.types.is_object_dtype(df[columna]):
            categorias = df[columna].unique()
        else:
            categorias = None

        descripcion.append({
            'Columna': columna,
            'Tipo de Dato': tipo,
            'Valores Nulos': nulos,
            'Valores Únicos': unicos,
            'Rango': rango,
            'Media': media,
            'Mediana': mediana,
            'Categorías': categorias
        })

    return pd.DataFrame(descripcion)

def plot_target_distribution(df, target=TARGET_VARIABLE):
    """Visualiza la distribución de la variable objetivo."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=target, order=df[target].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f'Distribución de la Variable Objetivo: {target}')
    plt.show()