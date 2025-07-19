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
from sklearn.neural_network import MLPClassifier
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
    Con funcionalidades de guardado de resultados y modelos incluyendo redes neuronales de sklearn
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
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.n_classes = None
        
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
        
        # Determinar número de clases
        self.n_classes = len(y_raw.unique())
        print(f"📊 Número de clases: {self.n_classes}")
        
        # Guardar información de feature engineering
        feature_info = {
            'features_selected': features_selected,
            'available_features': available_features,
            'final_shape': X_raw.shape,
            'n_classes': self.n_classes,
            'class_distribution': class_distribution.to_dict(),
            'missing_handling': 'categorical: Sin información, numerical: median'
        }
        
        self._save_json(feature_info, 'feature_engineering_info.json')
        
        # Guardar datasets procesados
        X_raw.to_csv(os.path.join(self.run_dir, 'X_processed.csv'), index=False)
        y_raw.to_csv(os.path.join(self.run_dir, 'y_processed.csv'), index=False)
        
        return X_raw, y_raw
    
    def encode_features(self, X_raw, y_raw):
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
        
        # Codificar variable objetivo para consistencia
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_raw)
        
        print(f"📊 Características después de codificación: {X_encoded.shape[1]}")
        print(f"📊 Clases codificadas: {self.label_encoder.classes_}")
        
        # Guardar información de codificación
        encoding_info = {
            'categorical_columns': categorical_cols.tolist(),
            'category_mapping': category_mapping,
            'final_features': self.feature_names,
            'encoding_method': 'One-Hot with top 10 categories + Others',
            'target_classes': self.label_encoder.classes_.tolist(),
            'n_features': X_encoded.shape[1]
        }
        
        self._save_json(encoding_info, 'encoding_info.json')
        
        # Guardar dataset codificado
        X_encoded.to_csv(os.path.join(self.run_dir, 'X_encoded.csv'), index=False)
        
        # Guardar label encoder
        with open(os.path.join(self.run_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        return X_encoded, y_encoded
    
    def split_and_balance_data(self, X, y, test_size=0.2, balance_method='none'):
        """
        Divide los datos y aplica técnicas de balanceamiento si es necesario
        Incluye escalado para redes neuronales
        """
        print("🔄 Dividiendo y balanceando datos...")
        
        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Conjunto de entrenamiento: {X_train.shape}")
        print(f"📊 Conjunto de prueba: {X_test.shape}")
        
        # Guardar distribuciones originales
        original_train_dist = pd.Series(y_train).value_counts(normalize=True).to_dict()
        
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
        
        # Escalado para redes neuronales
        print("🔄 Escalando datos para redes neuronales...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Guardar información de división y balanceamiento
        split_info = {
            'test_size': test_size,
            'balance_method': balance_method,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'original_train_distribution': original_train_dist,
            'final_train_distribution': pd.Series(y_train).value_counts(normalize=True).to_dict(),
            'scaling_method': 'StandardScaler'
        }
        
        self._save_json(split_info, 'data_split_info.json')
        
        # Guardar conjuntos de datos
        pd.DataFrame(X_train, columns=self.feature_names).to_csv(
            os.path.join(self.run_dir, 'X_train.csv'), index=False
        )
        pd.DataFrame(X_test, columns=self.feature_names).to_csv(
            os.path.join(self.run_dir, 'X_test.csv'), index=False
        )
        pd.DataFrame(X_train_scaled, columns=self.feature_names).to_csv(
            os.path.join(self.run_dir, 'X_train_scaled.csv'), index=False
        )
        pd.DataFrame(X_test_scaled, columns=self.feature_names).to_csv(
            os.path.join(self.run_dir, 'X_test_scaled.csv'), index=False
        )
        pd.Series(y_train).to_csv(
            os.path.join(self.run_dir, 'y_train.csv'), index=False
        )
        pd.Series(y_test).to_csv(
            os.path.join(self.run_dir, 'y_test.csv'), index=False
        )
        
        # Guardar scaler
        with open(os.path.join(self.run_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def initialize_models(self):
        """
        Inicializa los modelos de clasificación incluyendo redes neuronales de sklearn
        """
        print("🤖 Inicializando modelos...")
        
        # Calcular pesos de clase para manejar desbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = {str(int(cls)): float(weight) for cls, weight in zip(np.unique(self.y_train), class_weights)}
        
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
            'Naive Bayes': MultinomialNB(),
            'Neural Network Simple': MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            'Neural Network Deep': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15
            ),
            'Neural Network Complex': MLPClassifier(
                hidden_layer_sizes=(150, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
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
            
            # Usar datos escalados para redes neuronales
            if 'Neural Network' in name:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=cv, scoring='f1_weighted')
            
            # Entrenar en todo el conjunto de entrenamiento
            model.fit(X_train_use, self.y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train_use)
            y_pred_test = model.predict(X_test_use)
            
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
                'y_pred_test': y_pred_test,
                'model_type': 'neural_network' if 'Neural Network' in name else 'traditional'
            }
            
            print(f"   ✅ CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            print(f"   ✅ Test Accuracy: {test_accuracy:.3f}")
            print(f"   ✅ Test F1-Score: {test_f1:.3f}")
            
            # Mostrar información adicional para redes neuronales
            if 'Neural Network' in name and hasattr(model, 'n_iter_'):
                print(f"   ✅ Iteraciones convergencia: {model.n_iter_}")
                print(f"   ✅ Loss final: {model.loss_:.4f}")
            
            # Guardar modelo entrenado
            model_path = os.path.join(self.run_dir, f'model_{name.replace(" ", "_").lower()}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Guardar predicciones
            pred_train_df = pd.DataFrame({
                'y_true': self.y_train,
                'y_pred': y_pred_train
            })
            pred_train_df.to_csv(
                os.path.join(self.run_dir, f'predictions_train_{name.replace(" ", "_").lower()}.csv'),
                index=False
            )
            
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
                'test_f1': float(result['test_f1']),
                'model_type': result['model_type']
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
            X_train_use = self.X_train
            X_test_use = self.X_test
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            base_model = GradientBoostingClassifier(random_state=42)
            X_train_use = self.X_train
            X_test_use = self.X_test
            
        elif 'Neural Network' in model_name:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
            base_model = MLPClassifier(
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
            X_train_use = self.X_train_scaled
            X_test_use = self.X_test_scaled
            
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
        
        grid_search.fit(X_train_use, self.y_train)
        
        print(f"🏆 Mejores parámetros: {grid_search.best_params_}")
        print(f"🏆 Mejor score CV: {grid_search.best_score_:.3f}")
        
        # Evaluar modelo optimizado
        best_model = grid_search.best_estimator_
        y_pred_test = best_model.predict(X_test_use)
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
            'cv_results_summary': {
                'mean_scores': [float(score) for score in grid_search.cv_results_['mean_test_score']],
                'std_scores': [float(score) for score in grid_search.cv_results_['std_test_score']]
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
        
        if model_name not in self.results:
            print(f"❌ Modelo {model_name} no encontrado")
            return None
        
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
    
    def plot_neural_network_analysis(self):
        """
        Análisis específico para redes neuronales de sklearn
        """
        print("🧠 Analizando redes neuronales...")
        
        neural_models = {k: v for k, v in self.results.items() if 'Neural Network' in k}
        
        if not neural_models:
            print("❌ No se encontraron redes neuronales entrenadas")
            return
        
        # Crear gráfico comparativo
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Preparar datos
        nn_names = list(neural_models.keys())
        test_accuracies = [neural_models[name]['test_accuracy'] for name in nn_names]
        test_f1s = [neural_models[name]['test_f1'] for name in nn_names]
        iterations = [neural_models[name]['model'].n_iter_ if hasattr(neural_models[name]['model'], 'n_iter_') else 0 for name in nn_names]
        losses = [neural_models[name]['model'].loss_ if hasattr(neural_models[name]['model'], 'loss_') else 0 for name in nn_names]
        
        # Accuracy por red neuronal
        axes[0, 0].bar(range(len(nn_names)), test_accuracies, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy por Red Neuronal')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_xticks(range(len(nn_names)))
        axes[0, 0].set_xticklabels([name.replace('Neural Network ', 'NN ') for name in nn_names], rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1-Score por red neuronal
        axes[0, 1].bar(range(len(nn_names)), test_f1s, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('F1-Score por Red Neuronal')
        axes[0, 1].set_ylabel('Test F1-Score')
        axes[0, 1].set_xticks(range(len(nn_names)))
        axes[0, 1].set_xticklabels([name.replace('Neural Network ', 'NN ') for name in nn_names], rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Iteraciones de convergencia
        axes[1, 0].bar(range(len(nn_names)), iterations, color='orange', alpha=0.7)
        axes[1, 0].set_title('Iteraciones para Convergencia')
        axes[1, 0].set_ylabel('Número de Iteraciones')
        axes[1, 0].set_xticks(range(len(nn_names)))
        axes[1, 0].set_xticklabels([name.replace('Neural Network ', 'NN ') for name in nn_names], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss final
        axes[1, 1].bar(range(len(nn_names)), losses, color='salmon', alpha=0.7)
        axes[1, 1].set_title('Loss Final')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xticks(range(len(nn_names)))
        axes[1, 1].set_xticklabels([name.replace('Neural Network ', 'NN ') for name in nn_names], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = os.path.join(self.run_dir, 'neural_networks_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Crear tabla resumen
        nn_summary = pd.DataFrame({
            'Red Neuronal': nn_names,
            'Test Accuracy': [f"{acc:.3f}" for acc in test_accuracies],
            'Test F1-Score': [f"{f1:.3f}" for f1 in test_f1s],
            'Iteraciones': iterations,
            'Loss Final': [f"{loss:.4f}" for loss in losses],
            'Arquitectura': [self.results[name]['model'].hidden_layer_sizes for name in nn_names]
        })
        
        print("\n🧠 RESUMEN DE REDES NEURONALES:")
        print("=" * 60)
        print(nn_summary.to_string(index=False))
        
        # Guardar tabla
        nn_summary.to_csv(os.path.join(self.run_dir, 'neural_networks_summary.csv'), index=False)
        
        return nn_summary
    
    def plot_results_comparison(self):
        """
        Visualiza la comparación de resultados entre todos los modelos
        """
        print("📊 Generando visualización comparativa...")
        
        # Preparar datos para visualización
        models = list(self.results.keys())
        cv_means = [self.results[model]['cv_mean'] for model in models]
        test_accuracies = [self.results[model]['test_accuracy'] for model in models]
        test_f1_scores = [self.results[model]['test_f1'] for model in models]
        
        # Separar modelos tradicionales y redes neuronales para coloreo
        colors = []
        for model in models:
            if self.results[model]['model_type'] == 'neural_network':
                colors.append('lightcoral')
            else:
                colors.append('lightblue')
        
        # Crear subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # CV F1-Score
        bars1 = axes[0].bar(range(len(models)), cv_means, color=colors, alpha=0.7)
        axes[0].set_title('F1-Score (Validación Cruzada)')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, score in zip(bars1, cv_means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Test Accuracy
        bars2 = axes[1].bar(range(len(models)), test_accuracies, color=colors, alpha=0.7)
        axes[1].set_title('Accuracy (Conjunto de Prueba)')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, acc in zip(bars2, test_accuracies):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Test F1-Score
        bars3 = axes[2].bar(range(len(models)), test_f1_scores, color=colors, alpha=0.7)
        axes[2].set_title('F1-Score (Conjunto de Prueba)')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, f1 in zip(bars3, test_f1_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Crear leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='Modelos Tradicionales'),
            Patch(facecolor='lightcoral', alpha=0.7, label='Redes Neuronales')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = os.path.join(self.run_dir, 'model_comparison_all.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Tabla resumen
        results_df = pd.DataFrame({
            'Modelo': models,
            'Tipo': [self.results[model]['model_type'] for model in models],
            'CV F1-Score': [f"{score:.3f}" for score in cv_means],
            'Test Accuracy': [f"{score:.3f}" for score in test_accuracies],
            'Test F1-Score': [f"{score:.3f}" for score in test_f1_scores]
        })
        
        # Guardar tabla resumen
        results_df.to_csv(os.path.join(self.run_dir, 'model_comparison_table_all.csv'), index=False)
        
        print("\n📋 Resumen de resultados (todos los modelos):")
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
        class_report = classification_report(self.y_test, y_pred_test, 
                                           target_names=self.label_encoder.classes_)
        print(class_report)
        
        # Guardar classification report
        with open(os.path.join(self.run_dir, f'classification_report_{best_model_name.replace(" ", "_").lower()}.txt'), 'w') as f:
            f.write(class_report)
        
        # Classification report como diccionario
        report_dict = classification_report(self.y_test, y_pred_test, 
                                          target_names=self.label_encoder.classes_,
                                          output_dict=True)
        self._save_json(report_dict, f'classification_report_{best_model_name.replace(" ", "_").lower()}.json')
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_, 
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Matriz de Confusión - {best_model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar matriz de confusión
        cm_path = os.path.join(self.run_dir, f'confusion_matrix_{best_model_name.replace(" ", "_").lower()}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Guardar matriz de confusión como CSV
        cm_df = pd.DataFrame(cm, index=self.label_encoder.classes_, 
                           columns=self.label_encoder.classes_)
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
        """Guarda datos en formato JSON con manejo de tipos numpy"""
        filepath = os.path.join(self.run_dir, filename)
        
        def convert_types(obj):
            """Convierte tipos numpy a tipos Python nativos"""
            if isinstance(obj, dict):
                return {str(k): convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convertir datos para evitar errores de serialización
        data_converted = convert_types(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_converted, f, indent=2, ensure_ascii=False, default=str)
    
    def save_pipeline_summary(self, results):
        """Guarda un resumen completo del pipeline"""
        summary = {
            'timestamp': self.timestamp,
            'run_directory': self.run_dir,
            'data_path': self.data_path,
            'best_model': results['best_model'],
            'dataset_shape': self.df.shape,
            'feature_count': len(self.feature_names),
            'n_classes': self.n_classes,
            'models_trained': list(self.models.keys()),
            'results_summary': {
                model: {
                    'test_accuracy': float(self.results[model]['test_accuracy']),
                    'test_f1': float(self.results[model]['test_f1']),
                    'model_type': self.results[model]['model_type']
                }
                for model in self.results.keys()
            }
        }
        
        self._save_json(summary, 'pipeline_summary.json')
        
        # README con instrucciones
        readme_content = f"""# Resultados del Pipeline de Clasificación de Homicidios

## Información General
- **Fecha de ejecución**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Directorio de resultados**: {self.run_dir}
- **Mejor modelo**: {results['best_model']}
- **Número de clases**: {self.n_classes}
- **Número de características**: {len(self.feature_names)}

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
- **Precisión del mejor modelo**: {self.results[results['best_model']]['test_accuracy']:.3f}
- **F1-Score del mejor modelo**: {self.results[results['best_model']]['test_f1']:.3f}

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
        X_encoded, y_encoded = self.encode_features(X_raw, y_raw)
        
        # 4. Dividir y balancear datos
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_scaled, self.X_test_scaled = self.split_and_balance_data(
            X_encoded, y_encoded, balance_method=balance_method
        )
        
        # 5. Inicializar modelos
        self.initialize_models()
        
        # 6. Entrenar y evaluar modelos
        self.train_and_evaluate_models()
        
        # 7. Análisis específico de redes neuronales
        neural_summary = self.plot_neural_network_analysis()
        
        # 8. Comparar resultados
        results_df = self.plot_results_comparison()
        
        # 9. Identificar mejor modelo
        f1_scores = [self.results[model]['test_f1'] for model in results_df['Modelo']]
        best_model_idx = np.argmax(f1_scores)
        best_model_name = results_df.loc[best_model_idx, 'Modelo']
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        print(f"🏆 Tipo: {self.results[best_model_name]['model_type']}")
        
        # 10. Análisis de importancia de características (si es aplicable)
        importance_df = None
        if hasattr(self.results[best_model_name]['model'], 'feature_importances_'):
            importance_df = self.analyze_feature_importance(best_model_name)
        
        # 11. Optimización de hiperparámetros (opcional)
        optimized_model = None
        if optimize_best:
            optimized_model = self.hyperparameter_tuning(best_model_name)
        
        # 12. Reporte detallado
        metrics_df = self.generate_detailed_report(best_model_name)
        
        # 13. Guardar resumen del pipeline
        results = {
            'best_model': best_model_name,
            'best_model_type': self.results[best_model_name]['model_type'],
            'results_summary': results_df,
            'neural_summary': neural_summary,
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
        
        # Mostrar resumen final
        print("\n📊 RESUMEN FINAL:")
        print("=" * 40)
        print(f"✅ Mejor modelo: {results['best_model']}")
        print(f"✅ Tipo de modelo: {results['best_model_type']}")
        print(f"✅ Accuracy: {self.results[best_model_name]['test_accuracy']:.3f}")
        print(f"✅ F1-Score: {self.results[best_model_name]['test_f1']:.3f}")
        
        # Comparar tipos de modelos
        traditional_models = {k: v for k, v in self.results.items() if v['model_type'] == 'traditional'}
        neural_models = {k: v for k, v in self.results.items() if v['model_type'] == 'neural_network'}
        
        if traditional_models:
            best_traditional = max(traditional_models.items(), key=lambda x: x[1]['test_f1'])
            print(f"🔹 Mejor modelo tradicional: {best_traditional[0]} (F1: {best_traditional[1]['test_f1']:.3f})")
        
        if neural_models:
            best_neural = max(neural_models.items(), key=lambda x: x[1]['test_f1'])
            print(f"🧠 Mejor red neuronal: {best_neural[0]} (F1: {best_neural[1]['test_f1']:.3f})")
        
        return results

# Funciones auxiliares

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
    print(f"✅ Tipo: {results['best_model_type']}")
    print("✅ Comparación completa entre modelos tradicionales y redes neuronales")
    print("✅ Análisis específico de redes neuronales de sklearn")
    print("✅ Métricas detalladas generadas")
    print("✅ Modelos listos para producción")
    print(f"✅ Todos los resultados guardados en: {classifier.run_dir}")
    
    return results

def train_only_neural_networks(data_path):
    """
    Función para entrenar solo redes neuronales
    """
    print("🧠 ENTRENAMIENTO ESPECÍFICO DE REDES NEURONALES")
    print("=" * 50)
    
    classifier = HomicideClassificationPipeline(data_path)
    
    # Preparar datos
    classifier.load_and_prepare_data()
    X_raw, y_raw = classifier.feature_engineering()
    X_encoded, y_encoded = classifier.encode_features(X_raw, y_raw)
    classifier.X_train, classifier.X_test, classifier.y_train, classifier.y_test, classifier.X_train_scaled, classifier.X_test_scaled = classifier.split_and_balance_data(
        X_encoded, y_encoded
    )
    
    # Solo redes neuronales
    classifier.models = {
        'Neural Network Simple': MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        'Neural Network Deep': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        ),
        'Neural Network Complex': MLPClassifier(
            hidden_layer_sizes=(150, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
    }
    
    # Entrenar y evaluar
    classifier.train_and_evaluate_models()
    neural_summary = classifier.plot_neural_network_analysis()
    
    # Encontrar mejor red neuronal
    best_neural = max(classifier.results.items(), key=lambda x: x[1]['test_f1'])
    
    print(f"\n🏆 Mejor red neuronal: {best_neural[0]}")
    print(f"🏆 F1-Score: {best_neural[1]['test_f1']:.3f}")
    
    return classifier, best_neural

if __name__ == "__main__":
    # Ejecutar pipeline completo
    results = main()