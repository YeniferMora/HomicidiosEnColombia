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
import os
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class HomicideClassificationPipeline:
    """
    Pipeline completo para clasificación de circunstancias de homicidios en Colombia
    Con funcionalidades de guardado de resultados y modelos
    """
    
    def __init__(self, data_path, output_dir='../results'):
        self.data_path = data_path
        self.output_dir = output_dir
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
        
        # Crear directorio de salida si no existe
        self._create_output_directories()
        
        # Crear timestamp para esta ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
    def _create_output_directories(self):
        """Crea los directorios necesarios para guardar resultados"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'models'),
            os.path.join(self.output_dir, 'plots'),
            os.path.join(self.output_dir, 'reports'),
            os.path.join(self.output_dir, 'data')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
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
        target_distribution = self.df['Circunstancia del Hecho'].value_counts()
        print(target_distribution)
        
        # Guardar información básica del dataset
        dataset_info = {
            'dimensions': self.df.shape,
            'target_distribution': target_distribution.to_dict(),
            'columns': self.df.columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
        
        self._save_json(dataset_info, 'dataset_info.json')
        
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
        class_distribution = y_raw.value_counts(normalize=True).round(3)
        print(class_distribution)
        
        # Guardar información de feature engineering
        feature_info = {
            'features_selected': features_selected,
            'available_features': available_features,
            'final_shape': X_raw.shape,
            'class_distribution': class_distribution.to_dict(),
            'missing_handling': 'categorical: Sin información, numerical: median'
        }
        
        self._save_json(feature_info, 'feature_engineering_info.json')
        
        # Guardar datasets procesados
        X_raw.to_csv(os.path.join(self.run_dir, 'X_processed.csv'), index=False)
        y_raw.to_csv(os.path.join(self.run_dir, 'y_processed.csv'), index=False)
        
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
        category_mapping = {}
        for col in categorical_cols:
            value_counts = X_encoded[col].value_counts()
            # Mantener top 10 categorías más frecuentes
            top_categories = value_counts.head(10).index
            X_encoded.loc[~X_encoded[col].isin(top_categories), col] = 'Otros'
            category_mapping[col] = {
                'top_categories': top_categories.tolist(),
                'total_categories': len(value_counts),
                'kept_categories': 10
            }
        
        # Aplicar One-Hot Encoding
        X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols, drop_first=True)
        
        # Guardar nombres de características
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"📊 Características después de codificación: {X_encoded.shape[1]}")
        
        # Guardar información de codificación
        encoding_info = {
            'categorical_columns': categorical_cols.tolist(),
            'category_mapping': category_mapping,
            'final_features': self.feature_names,
            'encoding_method': 'One-Hot with top 10 categories + Others'
        }
        
        self._save_json(encoding_info, 'encoding_info.json')
        
        # Guardar dataset codificado
        X_encoded.to_csv(os.path.join(self.run_dir, 'X_encoded.csv'), index=False)
        
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
        
        # Guardar distribuciones originales
        original_train_dist = y_train.value_counts(normalize=True).to_dict()
        
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
        
        # Guardar información de división y balanceamiento
        split_info = {
            'test_size': test_size,
            'balance_method': balance_method,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'original_train_distribution': original_train_dist,
            'final_train_distribution': pd.Series(y_train).value_counts(normalize=True).to_dict()
        }
        
        self._save_json(split_info, 'data_split_info.json')
        
        # Guardar conjuntos de datos
        pd.DataFrame(X_train, columns=self.feature_names).to_csv(
            os.path.join(self.run_dir, 'X_train.csv'), index=False
        )
        pd.DataFrame(X_test, columns=self.feature_names).to_csv(
            os.path.join(self.run_dir, 'X_test.csv'), index=False
        )
        pd.Series(y_train).to_csv(
            os.path.join(self.run_dir, 'y_train.csv'), index=False
        )
        pd.Series(y_test).to_csv(
            os.path.join(self.run_dir, 'y_test.csv'), index=False
        )
        
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
            'Naive Bayes': MultinomialNB()
        }
        
        # Guardar configuración de modelos
        model_configs = {}
        for name, model in self.models.items():
            model_configs[name] = model.get_params()
        
        self._save_json(model_configs, 'model_configurations.json')
        self._save_json(class_weight_dict, 'class_weights.json')
        
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
            
            # Guardar modelo entrenado
            model_path = os.path.join(self.run_dir, f'model_{name.replace(" ", "_").lower()}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Guardar predicciones por separado (train y test tienen diferentes tamaños)
            # Predicciones de entrenamiento
            pred_train_df = pd.DataFrame({
                'y_true': self.y_train,
                'y_pred': y_pred_train
            })
            pred_train_df.to_csv(
                os.path.join(self.run_dir, f'predictions_train_{name.replace(" ", "_").lower()}.csv'),
                index=False
            )
            
            # Predicciones de prueba
            pred_test_df = pd.DataFrame({
                'y_true': self.y_test,
                'y_pred': y_pred_test
            })
            pred_test_df.to_csv(
                os.path.join(self.run_dir, f'predictions_test_{name.replace(" ", "_").lower()}.csv'),
                index=False
            )
        
        # Guardar resumen de resultados
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'train_accuracy': float(result['train_accuracy']),
                'test_accuracy': float(result['test_accuracy']),
                'train_f1': float(result['train_f1']),
                'test_f1': float(result['test_f1'])
            }
        
        self._save_json(results_summary, 'model_results_summary.json')
    
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
        
        # Guardar información de optimización
        optimization_info = {
            'model_name': model_name,
            'param_grid': param_grid,
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'optimized_test_accuracy': float(test_accuracy),
            'optimized_test_f1': float(test_f1),
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'params': grid_search.cv_results_['params']
            }
        }
        
        self._save_json(optimization_info, f'hyperparameter_optimization_{model_name.replace(" ", "_").lower()}.json')
        
        # Guardar modelo optimizado
        optimized_model_path = os.path.join(self.run_dir, f'optimized_model_{model_name.replace(" ", "_").lower()}.pkl')
        with open(optimized_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
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
            
            # Guardar importancias
            importance_df.to_csv(
                os.path.join(self.run_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.csv'),
                index=False
            )
            
            # Visualizar top 20 características
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title(f'Top 20 Características Más Importantes - {model_name}')
            plt.xlabel('Importancia')
            plt.tight_layout()
            
            # Guardar gráfico
            plot_path = os.path.join(self.run_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
        
        # Guardar gráfico
        plot_path = os.path.join(self.run_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Tabla resumen
        results_df = pd.DataFrame({
            'Modelo': models,
            'CV F1-Score': [f"{score:.3f}" for score in cv_means],
            'Test Accuracy': [f"{score:.3f}" for score in test_accuracies],
            'Test F1-Score': [f"{score:.3f}" for score in test_f1_scores]
        })
        
        # Guardar tabla resumen
        results_df.to_csv(os.path.join(self.run_dir, 'model_comparison_table.csv'), index=False)
        
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
        class_report = classification_report(self.y_test, y_pred_test)
        print(class_report)
        
        # Guardar classification report
        with open(os.path.join(self.run_dir, f'classification_report_{best_model_name.replace(" ", "_").lower()}.txt'), 'w') as f:
            f.write(class_report)
        
        # Classification report como diccionario
        report_dict = classification_report(self.y_test, y_pred_test, output_dict=True)
        self._save_json(report_dict, f'classification_report_{best_model_name.replace(" ", "_").lower()}.json')
        
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
        
        # Guardar matriz de confusión
        cm_path = os.path.join(self.run_dir, f'confusion_matrix_{best_model_name.replace(" ", "_").lower()}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Guardar matriz de confusión como CSV
        cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
        cm_df.to_csv(os.path.join(self.run_dir, f'confusion_matrix_{best_model_name.replace(" ", "_").lower()}.csv'))
        
        # Métricas por clase
        metrics_df = pd.DataFrame(report_dict).transpose()
        metrics_df = metrics_df[metrics_df.index != 'accuracy'].round(3)
        
        # Guardar métricas por clase
        metrics_df.to_csv(os.path.join(self.run_dir, f'metrics_by_class_{best_model_name.replace(" ", "_").lower()}.csv'))
        
        print("\n📊 MÉTRICAS POR CLASE:")
        print("=" * 50)
        print(metrics_df)
        
        return metrics_df
    
    def _save_json(self, data, filename):
        """Guarda datos en formato JSON"""
        filepath = os.path.join(self.run_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def save_pipeline_summary(self, results):
        """Guarda un resumen completo del pipeline"""
        summary = {
            'timestamp': self.timestamp,
            'run_directory': self.run_dir,
            'data_path': self.data_path,
            'best_model': results['best_model'],
            'dataset_shape': self.df.shape,
            'feature_count': len(self.feature_names),
            'models_trained': list(self.models.keys()),
            'results_summary': {
                model: {
                    'test_accuracy': float(self.results[model]['test_accuracy']),
                    'test_f1': float(self.results[model]['test_f1']),
                    'cv_mean': float(self.results[model]['cv_mean'])
                }
                for model in self.results.keys()
            },
            'files_generated': [
                'dataset_info.json',
                'feature_engineering_info.json',
                'encoding_info.json',
                'data_split_info.json',
                'model_configurations.json',
                'model_results_summary.json',
                'model_comparison_table.csv',
                'model_comparison.png',
                f'classification_report_{results["best_model"].replace(" ", "_").lower()}.txt',
                f'confusion_matrix_{results["best_model"].replace(" ", "_").lower()}.png'
            ]
        }
        
        self._save_json(summary, 'pipeline_summary.json')
        
        # También crear un README con instrucciones
        readme_content = f"""# Resultados del Pipeline de Clasificación de Homicidios

## Información General
- **Fecha de ejecución**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Directorio de resultados**: {self.run_dir}
- **Mejor modelo**: {results['best_model']}
- **Precisión del mejor modelo**: {self.results[results['best_model']]['test_accuracy']:.3f}
- **F1-Score del mejor modelo**: {self.results[results['best_model']]['test_f1']:.3f}

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
"""
        
        with open(os.path.join(self.run_dir, 'README.md'), 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
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
        optimized_model = None
        if optimize_best:
            optimized_model = self.hyperparameter_tuning(best_model_name)
        
        # 11. Reporte detallado
        metrics_df = self.generate_detailed_report(best_model_name)
        
        # 12. Guardar resumen del pipeline
        results = {
            'best_model': best_model_name,
            'results_summary': results_df,
            'feature_importance': importance_df,
            'detailed_metrics': metrics_df,
            'models': self.models,
            'results': self.results,
            'optimized_model': optimized_model
        }
        
        self.save_pipeline_summary(results)
        
        print(f"\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"📁 Resultados guardados en: {self.run_dir}")
        print("=" * 60)
        
        return results

# Función de uso principal
def main():
    """
    Función principal para ejecutar el análisis de clasificación
    """
    # Ruta a los datos preprocesados
    DATA_PATH = '../data/processed/homicidios_procesado.csv'
    
    # Directorio de salida personalizable
    OUTPUT_DIR = '../results'
    
    # Crear instancia del pipeline
    classifier = HomicideClassificationPipeline(DATA_PATH, OUTPUT_DIR)
    
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
    print(f"✅ Todos los resultados guardados en: {classifier.run_dir}")
    
    # Mostrar estructura de archivos generados
    print("\n📁 ARCHIVOS GENERADOS:")
    print("=" * 40)
    for root, dirs, files in os.walk(classifier.run_dir):
        level = root.replace(classifier.run_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{sub_indent}{file}')
    
    return results

# Función para cargar resultados de una ejecución anterior
def load_previous_results(run_directory):
    """
    Carga resultados de una ejecución anterior del pipeline
    """
    print(f"📂 Cargando resultados desde: {run_directory}")
    
    # Cargar resumen del pipeline
    summary_path = os.path.join(run_directory, 'pipeline_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"✅ Resumen cargado exitosamente")
        print(f"📊 Mejor modelo: {summary['best_model']}")
        print(f"📊 Modelos entrenados: {summary['models_trained']}")
        
        return summary
    else:
        print("❌ No se encontró el archivo de resumen")
        return None

# Función para comparar múltiples ejecuciones
def compare_runs(run_directories):
    """
    Compara resultados de múltiples ejecuciones del pipeline
    """
    print("🔄 Comparando múltiples ejecuciones...")
    
    comparison_data = []
    
    for run_dir in run_directories:
        summary = load_previous_results(run_dir)
        if summary:
            comparison_data.append({
                'run_id': os.path.basename(run_dir),
                'timestamp': summary['timestamp'],
                'best_model': summary['best_model'],
                'best_accuracy': max([result['test_accuracy'] for result in summary['results_summary'].values()]),
                'best_f1': max([result['test_f1'] for result in summary['results_summary'].values()])
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 COMPARACIÓN DE EJECUCIONES:")
        print("=" * 50)
        print(comparison_df.to_string(index=False))
        
        # Guardar comparación
        comparison_df.to_csv('../results/runs_comparison.csv', index=False)
        
        return comparison_df
    else:
        print("❌ No se pudieron cargar datos para comparación")
        return None

if __name__ == "__main__":
    results = main()