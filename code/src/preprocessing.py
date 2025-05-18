from matplotlib import pyplot as plt
import prince
from prince import MCA
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import chi2
from .config import PROCESSED_DATA_PATH, TARGET_VARIABLE, CATEGORICAL_VARS

def preprocess_data(df):
    """Realiza todas las operaciones de preprocesamiento."""

    # Identificación de Outliers
    cat_vars = df.select_dtypes(include='object').columns
    for var in cat_vars:
        rare_labels = df[var].value_counts(normalize=True)[df[var].value_counts(normalize=True) < 0.01]
        print(f'Valores raros en {var}:')
        print(rare_labels)
        print('\n')

    df.info()

    
    # REDUCCIÓN DE DIMENSIONALIDAD
    
    # 1. Eliminación de columnas no relevantes
    columns_to_drop = ['Edad judicial', 'Ciclo Vital', 'Código Dane Municipio', 'Código Dane Departamento']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Conteno de valores únicos y frecuencia absoluta
    for col in df.select_dtypes(include='object').columns:
        print(col, df[col].nunique(), "valores únicos")
        print(df[col].value_counts(), "\n")
    

    # Limpieza de datos categóricos

    # 1. Corregir errores por digitación en "Escenario del Hecho"
    if "Escenario del Hecho" in df.columns:
        df["Escenario del Hecho"] = df["Escenario del Hecho"].str.strip().str.lower()
        df["Escenario del Hecho"] = df["Escenario del Hecho"].replace({
            "en vía pública": "vía pública",
            "via pública": "vía pública",
            "en la via publica": "vía pública",
            "vía publica": "vía pública"
        })
    
    # 2. Eliminar valores duplicados
    df = df.drop_duplicates()

    # Valores faltantes

    # 3. Eliminar filas con valores faltantes en edad (ya que representan 0.1% y están incompletos)
    edad_cols = ['Grupo de edad de la victima', 'Edad judicial', 'Ciclo Vital']
    #df = df.dropna(subset=edad_cols)

    # 4. Imputar con la moda en Pertenencia Grupal
    columna = "Pertenencia Grupal"

    if columna in df.columns:
        sin_info_mask = df[columna].str.strip().str.lower() == "sin información"
        moda = df.loc[~sin_info_mask, columna].mode()
        if not moda.empty:
            df.loc[sin_info_mask, columna] = moda[0]

    # 5. Eliminar filas con fechas faltantes (0.207%) en caso de que analizamos relaciones temporales
    fecha_cols = ['Año del hecho', 'Mes del hecho', 'Dia del hecho']

    # Eliminar filas con NaN
    df = df.dropna(subset=fecha_cols)

    # 6. Eliminar filas con "sin información" en cualquiera de esas columnas
    for col in fecha_cols:
        df = df[~df[col].astype(str).str.strip().str.lower().eq("sin información")]


    # TRANSFORMACIÓN DE DATOS
    # 7. One-hot encoding para 'Grupo Mayor/Menor de Edad'
    if "Grupo Mayor/Menor de Edad" in df.columns:
        df = pd.get_dummies(df, columns=["Grupo Mayor/Menor de Edad"], prefix="Grupo_Edad")

    # 8. Agrupación y One-hot en Estado Civil (ejemplo de agrupar unidos, casados, unión libre)
    estado_civil_map = {
        "Unión libre": "Unión/Relación",
        "Casado(a)": "Unión/Relación",
        "Separado(a)": "Separado/Divorciado",
        "Divorciado(a)": "Separado/Divorciado",
        "Soltero(a)": "Soltero",
        "Viudo(a)": "Viudo",
        "No registra": "Otro",
        "No Aplica": "Otro"
    }

    if "Estado Civil" in df.columns:
        df["Estado Civil"] = df["Estado Civil"].map(estado_civil_map)
        df = pd.get_dummies(df, columns=["Estado Civil"], prefix="EstadoCivil")


    # 9. Reagrupar País de nacimiento de acuerdo a la frecuencia
    threshold = 50
    country_counts = df['País de Nacimiento de la Víctima'].value_counts()
    rare_countries = country_counts[country_counts < threshold].index
    df['País de Nacimiento de la Víctima'] = df['País de Nacimiento de la Víctima'].replace(rare_countries, 'Otros')
    
    # 10. Reagrupación de Circuntancias del Hecho
    # Diccionario de mapeo para la nueva categoría
    map_circunstancias = {
        'Sin información': 'Sin información',
        'Riña': 'Violencia interpersonal',
        'Violencia de pareja': 'Violencia interpersonal',
        'Violencia entre otros familiares': 'Violencia interpersonal',
        'Celos': 'Violencia interpersonal',
        'Embriaguez (Alcohólica y no alcohólica)': 'Violencia interpersonal',
        'Agresión o ataque sexual': 'Violencia interpersonal',

        'Sicariato': 'Crimen organizado / sicariato',
        'Ajuste de cuentas': 'Crimen organizado / sicariato',
        'Venganza o ajuste de cuentas': 'Crimen organizado / sicariato',
        'Atraco callejero o intento de': 'Crimen organizado / sicariato',
        'Hurto': 'Crimen organizado / sicariato',
        'Acción bandas criminales': 'Crimen organizado / sicariato',

        'Acción grupos alzados al margen de la ley': 'Conflicto armado / terrorismo',
        'Acción militar': 'Conflicto armado / terrorismo',
        'Enfrentamiento armado': 'Conflicto armado / terrorismo',
        'Emboscada': 'Conflicto armado / terrorismo',
        'Acto terrorista': 'Conflicto armado / terrorismo',
        'Mina Antipersona - Munición sin Explotar': 'Conflicto armado / terrorismo',
        'Explosión': 'Conflicto armado / terrorismo',
        'Artefacto explosivo': 'Conflicto armado / terrorismo',
        'Artefacto explosivo improvisado': 'Conflicto armado / terrorismo',

        'Intervención Legal': 'Violencia institucional / Estado',
        'Retención legal': 'Violencia institucional / Estado',
        'Presunta responsabilidad en la prestación de servicios de salud': 'Violencia institucional / Estado',
        'Ataque a instalación de las fuerzas armadas estatales': 'Violencia institucional / Estado',
        'Marcha o protesta social': 'Violencia institucional / Estado',

        'Asesinato político': 'Violencia estructural o sociopolítica',
        'Violencia Sociopolitica': 'Violencia estructural o sociopolítica',
        'Reclutamiento de niños, niñas y adolescentes': 'Violencia estructural o sociopolítica',
        'Disturbios civiles': 'Violencia estructural o sociopolítica',

        'Violencia a niños, niñas y adolescentes': 'Violencia contra población vulnerable',
        'Violencia al adulto mayor': 'Violencia contra población vulnerable',
        'Agresión contra grupos marginales o descalificados': 'Violencia contra población vulnerable',
        'Explotación sexual y comercial': 'Violencia contra población vulnerable',

        'Tortura': 'Otros',
        'Negligencia': 'Otros',
        'Desaparición forzada': 'Otros',
        'Minería ilegal': 'Otros',
        'Aglomeración de público': 'Otros',
        'Acceso carnal violento': 'Otros',
        'Ejercicio de actividades ilícitas': 'Otros',
        'Masacre': 'Otros',
        'Linchamiento': 'Otros',
        'Otra': 'Otros',
        'Feminicidio': 'Otros',
        'Retención ilegal - secuestro': 'Otros'
    }

    # Aplicar el mapeo
    df['Circunstancia del Hecho'] = df['Circunstancia del Hecho'].map(map_circunstancias).fillna('Otros')

    # Verificar los nuevos conteos
    df['Circunstancia del Hecho'].value_counts()

    # Ver resultado final
    print("Tamaño final del conjunto de datos:", df.shape)
    df.head()
    df.info()


    # IDENTIFICACIÓN DE VARIABLES RELEVANTES PARA LA SELECCIÓN DE CARACTERÍSTICAS

    all_categorical_columns = df.select_dtypes(include=['object']).columns

    categorical_columns = [col for col in all_categorical_columns if df[col].nunique() > 1]

    # Cálculo de la matriz de Cramer's V
    def cramers_v_matrix(df, categorical_cols):
        """Cálculo de Cramer's V para todas las combinaciones de columnas categóricas."""
        # Ensure the DataFrame passed only contains the selected columns
        df_filtered = df[categorical_cols]

        cramers_v = pd.DataFrame(np.zeros((len(categorical_cols), len(categorical_cols))),
                                columns=categorical_cols,
                                index=categorical_cols)

        for col1 in categorical_cols:
            for col2 in categorical_cols:
                contingency_table = pd.crosstab(df_filtered[col1], df_filtered[col2])
                # Check if the contingency table is valid for chi-squared
                if min(contingency_table.shape) == 0: # Handle cases where one category is empty after filtering
                    cramers_v.loc[col1, col2] = 0.0
                    continue

                chi2_stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                n = df_filtered.shape[0]
                r, k = contingency_table.shape

                # Avoid division by zero if min(r-1, k-1) is 0
                min_dim_minus_1 = min(r - 1, k - 1)
                if min_dim_minus_1 == 0:
                    cramers_v.loc[col1, col2] = 0.0
                else:
                    cramers_v.loc[col1, col2] = np.sqrt(chi2_stat / (n * min_dim_minus_1))

        return cramers_v

    # Generar la matriz de Cramer's V
    cramers_v = cramers_v_matrix(df, categorical_columns)

    # Visualizar la matriz de Cramer's V
    plt.figure(figsize=(15, 10))
    sns.heatmap(cramers_v, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Matriz de Cramér\'s V')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    def entropy(series):
        """Calcula la entropía de una serie categórica."""
        freqs = series.value_counts(normalize=True)
        return -(freqs * np.log(freqs)).sum()

    def normalized_variance(series):
        """Calcula la varianza normalizada para una serie numérica."""
        if series.max() == series.min():
            return 0.0
        return series.var() / (series.max() - series.min())**2

    results = []
    for col in df.columns:
        if col == 'ID':
            continue
        if df[col].dtype == 'object' or df[col].dtype.name == 'category' or df[col].dtype == 'bool':
            score = entropy(df[col])
            metric = 'entropy'
        else:
            score = normalized_variance(df[col])
            metric = 'var_norm'
        results.append({'feature': col, 'score': score, 'metric': metric})

    ranked = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
    print(ranked)



    
    # SELECCIÓN DE VARIABLE OBJETIVO
    df_encoded = df.copy()

    # Paso 2: Codificar las variables categóricas
    label_encoders = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Paso 3: Definir todas las posibles variables objetivo (todas las columnas categóricas)
    pseudo_targets = df_encoded.columns[df_encoded.dtypes != 'int64'].tolist()

    # Paso 4: Calcular Chi-cuadrado para cada pseudo-objetivo
    results = {}
    for target in pseudo_targets:
        X = df_encoded.drop(columns=[target, 'ID'])  # Excluir la variable objetivo y columnas irrelevantes
        y = df_encoded[target]  # Variable objetivo

        # Calcular Chi-cuadrado
        chi2_scores, p_values = chi2(X, y)

        # Guardar resultados en un DataFrame
        chi2_results = pd.DataFrame({
            'Feature': X.columns,
            'Chi2_Score': chi2_scores,
            'P-Value': p_values
        }).sort_values(by='Chi2_Score', ascending=False)

        results[target] = chi2_results

    # Paso 5: Graficar resultados
    for target, chi2_results in results.items():
        top_features = chi2_results.head(10)  # Seleccionar las 10 características más relevantes

        # Crear gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Chi2_Score'], color='skyblue')
        plt.xlabel('Chi2 Score')
        plt.ylabel('Features')
        plt.title(f'Top Features para la pseudo-variable objetivo: {target}')
        plt.gca().invert_yaxis()  # Invertir el eje para que la característica más relevante esté arriba
        plt.show()
    
    df.info()


    # Selección de columnas categóricas para análisis MCA
    categorical_columns = ['Sexo de la victima', 'Grupo de edad de la victima', 'Zona del Hecho',
                        'Escenario del Hecho', 'Circunstancia del Hecho', 'Manera de muerte']

    # Asegúrate de que sean tipo string
    df[categorical_columns] = df[categorical_columns].astype(str)

    # Crear una copia con solo variables categóricas
    df_cat = df[categorical_columns]

    # Aplicar MCA
    mca = prince.MCA(n_components=2, random_state=42)
    mca_result = mca.fit(df_cat)

    # Obtener coordenadas
    mca_coords = mca.transform(df_cat)

    # Visualizar las dos primeras componentes
    plt.figure(figsize=(10, 6))
    plt.scatter(mca_coords[0], mca_coords[1], alpha=0.2)
    plt.title('MCA - Reducción de dimensionalidad (2D)')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.grid(True)
    plt.show()


        # Aplicar MCA
    mca = MCA(n_components=2, random_state=42)
    mca_result = mca.fit(df[categorical_columns])

    # Extraer coordenadas de las dimensiones principales
    principal_coords = mca.transform(df[categorical_columns])

    # Visualizar las dimensiones principales
    plt.figure(figsize=(10, 8))
    plt.scatter(principal_coords[0], principal_coords[1], alpha=0.6, color='blue')
    plt.title('Proyección de MCA (2D)')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid()
    plt.show()


    # Visualizar desbalance de la variable objetivo
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Circunstancia del Hecho', order=df['Circunstancia del Hecho'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Distribución de la Variable Objetivo: Circunstancia del hecho')
    plt.show()


    
    return df

def save_processed_data(df, file_path=PROCESSED_DATA_PATH):
    """Guarda los datos preprocesados en un archivo CSV."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Datos preprocesados guardados en: {file_path}")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        raise