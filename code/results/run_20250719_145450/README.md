# Resultados del Pipeline de Clasificación de Homicidios

## Información General
- **Fecha de ejecución**: 2025-07-19 15:32:12
- **Directorio de resultados**: ../results\run_20250719_145450
- **Mejor modelo**: Gradient Boosting
- **Número de clases**: 7
- **Número de características**: 115

## Modelos Entrenados

### Modelos Tradicionales
- Random Forest
- Gradient Boosting  
- Logistic Regression
- Naive Bayes

### Redes Neuronales (sklearn)
- Neural Network Simple (1 capa oculta)
- Neural Network Deep (2 capas ocultas)
- Neural Network Complex (3 capas ocultas)

## Mejores Resultados
- **Precisión del mejor modelo**: 0.728
- **F1-Score del mejor modelo**: 0.709

## Archivos Generados

### Modelos
- `model_*.pkl`: Modelos entrenados
- `optimized_model_*.pkl`: Modelos optimizados
- `scaler.pkl`: Escalador para redes neuronales
- `label_encoder.pkl`: Codificador de etiquetas

### Datos
- `X_train.csv`, `X_test.csv`: Conjuntos de datos originales
- `X_train_scaled.csv`, `X_test_scaled.csv`: Conjuntos escalados
- `y_train.csv`, `y_test.csv`: Variables objetivo

### Resultados
- `model_results_summary.json`: Resumen de métricas
- `predictions_*.csv`: Predicciones de cada modelo
- `classification_report_*.txt`: Reportes detallados
- `confusion_matrix_*.csv`: Matrices de confusión

### Visualizaciones
- `model_comparison_all.png`: Comparación entre modelos
- `neural_networks_analysis.png`: Análisis específico de redes neuronales
- `feature_importance_*.png`: Importancia de características

## Uso de los Modelos

### Cargar Modelo
```python
import pickle
with open('model_neural_network_deep.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Hacer Predicciones
```python
# Para redes neuronales, usar datos escalados
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

### Decodificar Etiquetas
```python
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

predictions_labels = label_encoder.inverse_transform(predictions)
```
