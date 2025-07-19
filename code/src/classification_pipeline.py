import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class HomicideClassificationPipeline:
    """
    Pipeline completo para clasificación de circunstancias de homicidios en Colombia
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """
        Carga y prepara los datos para clasificación
        """
        print("🔄 Cargando y preparando datos...")
        
        # Cargar datos preprocesados
        self.df = pd.read_csv(self.data_path)
        
        # Información básica del dataset
        print(f"📊 Dimensiones del dataset: {self.df.shape}")
        print(f"📊 Distribución de la variable objetivo:")
        print(self.df['Circunstancia del Hecho'].value_counts())
        
        return self.df
    
    def feature_engineering(self):
        """
        Ingeniería de características basada en el análisis exploratorio previo
        """
        print("🔧 Realizando ingeniería de características...")
        
        # Seleccionar características relevantes identificadas en el análisis exploratorio
        features_selected = [
            'Sexo de la victima',
            'Grupo de edad de la victima', 
            'Zona del Hecho',
            'Escenario del Hecho',
            'Mes del hecho',
            'Dia del hecho',
            'Rango de Hora del Hecho X 3 Horas',
            'Departamento del hecho DANE',
            'Mecanismo Causal',
            'Diagnostico Topográfico de la Lesión',
            'Presunto Agresor',
            'Pertenencia Grupal',
            'Escolaridad',
            'Ancestro Racial'
        ]
        
        # Filtrar características disponibles
        available_features = [col for col in features_selected if col in self.df.columns]
        print(f"📋 Características disponibles: {len(available_features)}")
        
        # Crear dataset de características
        X_raw = self.df[available_features].copy()
        y_raw = self.df['Circunstancia del Hecho'].copy()
        
        # Eliminar filas con valores faltantes en la variable objetivo (Sin información)
        mask = y_raw != 'Sin información'
        print(f"📊 Filtrando datos: {mask.sum()} observaciones válidas")
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        
        # Manejar valores faltantes en características
        for col in X_raw.columns:
            if X_raw[col].dtype == 'object':
                X_raw[col] = X_raw[col].fillna('Sin información')
            else:
                X_raw[col] = X_raw[col].fillna(X_raw[col].median())
        
        print(f"📊 Dataset final: {X_raw.shape[0]} observaciones, {X_raw.shape[1]} características")
        print(f"📊 Distribución de clases:")
        print(y_raw.value_counts(normalize=True).round(3))
        
        return X_raw, y_raw
    
    def encode_features(self, X_raw):
        """
        Codifica las variables categóricas
        """
        print("🔄 Codificando variables categóricas...")
        
        # Crear copia de los datos
        X_encoded = X_raw.copy()
        
        # Aplicar One-Hot Encoding
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        # Para variables con muchas categorías, mantener solo las más frecuentes
        for col in categorical_cols:
            value_counts = X_encoded[col].value_counts()
            # Mantener top 10 categorías más frecuentes
            top_categories = value_counts.head(10).index
            X_encoded.loc[~X_encoded[col].isin(top_categories), col] = 'Otros'
        
        # Aplicar One-Hot Encoding
        X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols, drop_first=True)
        
        # Guardar nombres de características
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"📊 Características después de codificación: {X_encoded.shape[1]}")
        
        return X_encoded
    
    def split_and_balance_data(self, X, y, test_size=0.2, balance_method='none'):
        """
        Divide los datos y aplica técnicas de balanceamiento si es necesario
        """
        print("🔄 Dividiendo y balanceando datos...")
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Conjunto de entrenamiento: {X_train.shape}")
        print(f"📊 Conjunto de prueba: {X_test.shape}")
        
        # Aplicar balanceamiento si se solicita
        if balance_method == 'smote':
            print("⚖️ Aplicando SMOTE...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"📊 Después de SMOTE: {X_train.shape}")
            
        elif balance_method == 'undersample':
            print("⚖️ Aplicando submuestreo...")
            undersampler = RandomUnderSampler(random_state=42)
            X_train, y_train = undersampler.fit_resample(X_train, y_train)
            print(f"📊 Después de submuestreo: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Inicializa los modelos de clasificación con parámetros optimizados
        """
        print("🤖 Inicializando modelos...")
        
        # Calcular pesos de clase para manejar desbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            # 'SVM': SVC(
            #     class_weight='balanced',
            #     probability=True,
            #     random_state=42
            # ),
            'Naive Bayes': MultinomialNB()
        }
        
        print(f"🤖 Modelos inicializados: {list(self.models.keys())}")
        
    def train_and_evaluate_models(self):
        """
        Entrena y evalúa todos los modelos
        """
        print("🚀 Entrenando y evaluando modelos...")
        
        self.results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\n🔄 Entrenando {name}...")
            
            # Validación cruzada
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_weighted')
            
            # Entrenar en todo el conjunto de entrenamiento
            model.fit(self.X_train, self.y_train)
            
            # Predicciones
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Métricas
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            train_f1 = f1_score(self.y_train, y_pred_train, average='weighted')
            test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
            
            # Guardar resultados
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test
            }
            
            print(f"   ✅ CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            print(f"   ✅ Test Accuracy: {test_accuracy:.3f}")
            print(f"   ✅ Test F1-Score: {test_f1:.3f}")
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """
        Optimización de hiperparámetros para el mejor modelo
        """
        print(f"🔧 Optimizando hiperparámetros para {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            
        else:
            print(f"❌ Optimización no implementada para {model_name}")
            return None
        
        # Grid Search con validación cruzada
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"🏆 Mejores parámetros: {grid_search.best_params_}")
        print(f"🏆 Mejor score CV: {grid_search.best_score_:.3f}")
        
        # Evaluar modelo optimizado
        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(self.X_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        print(f"🏆 Test Accuracy optimizado: {test_accuracy:.3f}")
        print(f"🏆 Test F1-Score optimizado: {test_f1:.3f}")
        
        return best_model
    
    def analyze_feature_importance(self, model_name='Random Forest'):
        """
        Analiza la importancia de las características
        """
        print(f"📊 Analizando importancia de características para {model_name}...")
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            # Crear DataFrame con importancias
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Visualizar top 20 características
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top 20 Características Más Importantes - {model_name}')
            plt.xlabel('Importancia')
            plt.tight_layout()
            plt.show()
            
            print("🔝 Top 10 características más importantes:")
            print(importance_df.head(10))
            
            return importance_df
        else:
            print(f"❌ {model_name} no soporta importancia de características")
            return None
    
    def plot_results_comparison(self):
        """
        Visualiza la comparación de resultados entre modelos
        """
        print("📊 Generando visualización comparativa...")
        
        # Preparar datos para visualización
        models = list(self.results.keys())
        cv_means = [self.results[model]['cv_mean'] for model in models]
        test_accuracies = [self.results[model]['test_accuracy'] for model in models]
        test_f1_scores = [self.results[model]['test_f1'] for model in models]
        
        # Crear subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # CV F1-Score
        axes[0].bar(models, cv_means, alpha=0.7, color='skyblue')
        axes[0].set_title('F1-Score (Validación Cruzada)')
        axes[0].set_ylabel('F1-Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Test Accuracy
        axes[1].bar(models, test_accuracies, alpha=0.7, color='lightgreen')
        axes[1].set_title('Accuracy (Conjunto de Prueba)')
        axes[1].set_ylabel('Accuracy')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Test F1-Score
        axes[2].bar(models, test_f1_scores, alpha=0.7, color='salmon')
        axes[2].set_title('F1-Score (Conjunto de Prueba)')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Tabla resumen
        results_df = pd.DataFrame({
            'Modelo': models,
            'CV F1-Score': [f"{score:.3f}" for score in cv_means],
            'Test Accuracy': [f"{score:.3f}" for score in test_accuracies],
            'Test F1-Score': [f"{score:.3f}" for score in test_f1_scores]
        })
        
        print("\n📋 Resumen de resultados:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def generate_detailed_report(self, best_model_name):
        """
        Genera un reporte detallado del mejor modelo
        """
        print(f"📄 Generando reporte detallado para {best_model_name}...")
        
        model = self.results[best_model_name]['model']
        y_pred_test = self.results[best_model_name]['y_pred_test']
        
        # Classification Report
        print("\n📊 REPORTE DE CLASIFICACIÓN:")
        print("=" * 50)
        print(classification_report(self.y_test, y_pred_test))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Matriz de Confusión - {best_model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Métricas por clase
        report_dict = classification_report(self.y_test, y_pred_test, output_dict=True)
        
        # Crear DataFrame con métricas por clase
        metrics_df = pd.DataFrame(report_dict).transpose()
        metrics_df = metrics_df[metrics_df.index != 'accuracy'].round(3)
        
        print("\n📊 MÉTRICAS POR CLASE:")
        print("=" * 50)
        print(metrics_df)
        
        return metrics_df
    
    def run_complete_pipeline(self, balance_method='none', optimize_best=True):
        """
        Ejecuta el pipeline completo de clasificación
        """
        print("🚀 INICIANDO PIPELINE DE CLASIFICACIÓN DE HOMICIDIOS")
        print("=" * 60)
        
        # 1. Cargar y preparar datos
        self.load_and_prepare_data()
        
        # 2. Ingeniería de características
        X_raw, y_raw = self.feature_engineering()
        
        # 3. Codificar características
        X_encoded = self.encode_features(X_raw)
        
        # 4. Dividir y balancear datos
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_balance_data(
            X_encoded, y_raw, balance_method=balance_method
        )
        
        # 5. Inicializar modelos
        self.initialize_models()
        
        # 6. Entrenar y evaluar modelos
        self.train_and_evaluate_models()
        
        # 7. Comparar resultados
        results_df = self.plot_results_comparison()
        
        # 8. Identificar mejor modelo
        best_model_name = results_df.loc[results_df['Test F1-Score'].idxmax(), 'Modelo']
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        
        # 9. Análisis de importancia de características
        importance_df = self.analyze_feature_importance(best_model_name)
        
        # 10. Optimización de hiperparámetros (opcional)
        if optimize_best:
            optimized_model = self.hyperparameter_tuning(best_model_name)
        
        # 11. Reporte detallado
        metrics_df = self.generate_detailed_report(best_model_name)
        
        print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return {
            'best_model': best_model_name,
            'results_summary': results_df,
            'feature_importance': importance_df,
            'detailed_metrics': metrics_df,
            'models': self.models,
            'results': self.results
        }

# Función de uso principal
def main():
    """
    Función principal para ejecutar el análisis de clasificación
    """
    # Ruta a los datos preprocesados
    DATA_PATH = '../data/processed/homicidios_procesado.csv'
    
    # Crear instancia del pipeline
    classifier = HomicideClassificationPipeline(DATA_PATH)
    
    # Ejecutar pipeline completo
    results = classifier.run_complete_pipeline(
        balance_method='none',  # Opciones: 'none', 'smote', 'undersample'
        optimize_best=True
    )
    
    print("\n🎯 CONCLUSIONES:")
    print("=" * 40)
    print(f"✅ Mejor modelo: {results['best_model']}")
    print("✅ Características más importantes identificadas")
    print("✅ Métricas detalladas generadas")
    print("✅ Modelos listos para producción")
    
    return results

if __name__ == "__main__":
    results = main()