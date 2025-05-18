import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from tabulate import tabulate
from .config import TARGET_VARIABLE, CATEGORICAL_VARS

def exploratory_analysis(df):
    # Identificación de Variables
    cat_vars = df.select_dtypes(include='object').columns
    print("Variables categóricas identificadas:")
    print(cat_vars)

    # Revisar tipos de datos y nulos básicos
    df.info()
    df.isnull().sum().sort_values(ascending=False)

    # Frecuencia absoluta en cada variable categórica
    for col in df.select_dtypes(include='object').columns:
        print(col, df[col].nunique(), "valores únicos")
        print(df[col].value_counts(), "\n")

    # Describir columnas
    description = describe_columns(df)
    print(tabulate(description, headers='keys', tablefmt='fancy_grid'))

    # Descripción sexo de la víctima
    df.groupby('Sexo de la victima').size()
    fp = df.groupby('Sexo de la victima').size() / df.shape[0]
    print(fp)

    # Descripción circunstancias del hecho
    df.groupby('Circunstancia del Hecho').size()
    fp = df.groupby('Circunstancia del Hecho').size() / df.shape[0]
    print(fp)

    # Variables numéricas (relevante para Año del Hecho)
    df.describe()

    # Frecuencia relativa para las variables categóricas más importantes
    targets = ['Sexo de la victima', 'Grupo de edad de la victima', 'Zona del Hecho',
            'Escenario del Hecho', 'Circunstancia del Hecho', 'Manera de muerte']
    for col in targets:
        print(col, df[col].value_counts(normalize=True).mul(100).round(2).head(10), '\n')

    # Gráficas de distribución de cada variable (comentadas para evitar saturación)
    # for var in cat_vars:
    #     plt.figure(figsize=(10, 4))
    #     sns.countplot(data=df, x=var, order=df[var].value_counts().index, palette='Set2')
    #     plt.title(f'Distribución de {var}')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.tight_layout()
    #     plt.show()

    # Histograma de años
    df['Año del hecho'].plot(kind='hist', bins=9)
    plt.title('Número de casos por año')
    plt.xlabel('Año del hecho')
    plt.ylabel('Frecuencia')
    plt.show()

    # Boxplot de edad judicial por sexo
    plt.figure()
    sns.boxplot(x='Sexo de la victima', y='Edad judicial', data=df)
    plt.title('Distribución de edad judicial por sexo')
    plt.show()

    # Barplot de circunstancia del hecho
    df['Circunstancia del Hecho'].value_counts().head(8).plot(kind='bar')
    plt.title('Top 8 circunstancias del hecho')
    plt.ylabel('Número de casos')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Heatmap de correlaciones numéricas (Demuestra que la única variable que aporta información relevante es el año)
    plt.figure(figsize=(6,5))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f')
    plt.title('Correlación entre variables numéricas')
    plt.show()

    # Analisis de asociación entre el sexo de la víctima y la circunstancia del hecho (chi-cuadrado).
    pd.crosstab(df['Sexo de la victima'], df['Circunstancia del Hecho'], normalize='index')
    from scipy.stats import chi2_contingency
    tabla = pd.crosstab(df['Sexo de la victima'], df['Circunstancia del Hecho'])
    chi2_stat, p, dof, expected = chi2_contingency(tabla)
    print(f"Chi-cuadrado: {chi2_stat:.2f}, p-valor: {p:.4f}")



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