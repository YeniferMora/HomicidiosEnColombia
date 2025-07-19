# Resultados del Pipeline de Clasificación de Homicidios

## Información General
- **Fecha de ejecución**: 2025-07-19 14:40:00
- **Directorio de resultados**: ../results\run_20250719_141208
- **Mejor modelo**: Gradient Boosting
- **Precisión del mejor modelo**: 0.728
- **F1-Score del mejor modelo**: 0.709

## Estructura de Archivos

### Datos Procesados
- `X_processed.csv`: Características después del preprocesamiento
- `y_processed.csv`: Variable objetivo procesada
- `X_encoded.csv`: Características después de la codificación
- `X_train.csv`, `X_test.csv`: Conjuntos de entrenamiento y prueba (características)
- `y_train.csv`, `y_test.csv`: Conjuntos de entrenamiento y prueba (objetivo)

### Modelos Entrenados
- `model_*.pkl`: Modelos entrenados guardados en formato pickle
- `optimized_model_*.pkl`: Modelo optimizado con mejores hiperparámetros

### Resultados y Métricas
- `model_results_summary.json`: Resumen de métricas de todos los modelos
- `predictions_train_*.csv`: Predicciones en conjunto de entrenamiento
- `predictions_test_*.csv`: Predicciones en conjunto de prueba
- `classification_report_*.txt`: Reporte detallado de clasificación
- `confusion_matrix_*.csv`: Matriz de confusión en formato tabular
- `metrics_by_class_*.csv`: Métricas detalladas por clase
- `feature_importance_*.csv`: Importancia de características

### Visualizaciones
- `model_comparison.png`: Comparación visual entre modelos
- `confusion_matrix_*.png`: Matriz de confusión visualizada
- `feature_importance_*.png`: Gráfico de importancia de características

### Información Técnica
- `dataset_info.json`: Información básica del dataset
- `feature_engineering_info.json`: Detalles del proceso de ingeniería de características
- `encoding_info.json`: Información sobre la codificación de variables
- `data_split_info.json`: Detalles de la división de datos
- `model_configurations.json`: Configuraciones de todos los modelos
- `hyperparameter_optimization_*.json`: Resultados de la optimización de hiperparámetros
- `pipeline_summary.json`: Resumen completo del pipeline

## Cómo Usar los Resultados

### Cargar un Modelo Entrenado
```python
import pickle
with open('model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Realizar Predicciones
```python
# Cargar datos nuevos y aplicar el mismo preprocesamiento
# predictions = model.predict(X_new)
```

### Analizar Métricas
```python
import pandas as pd
import json

# Cargar métricas
with open('model_results_summary.json', 'r') as f:
    results = json.load(f)
    
# Cargar métricas por clase
metrics_df = pd.read_csv('metrics_by_class_*.csv')
```

## Notas Importantes
- Todos los modelos fueron entrenados con validación cruzada estratificada
- Se aplicó balanceamiento de clases cuando fue necesario
- Los hiperparámetros fueron optimizados para el mejor modelo
- Las características categóricas fueron codificadas usando One-Hot Encoding
- Se mantuvieron las 10 categorías más frecuentes por variable categórica
