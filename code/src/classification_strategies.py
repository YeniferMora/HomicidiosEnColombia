import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class AdvancedClassificationAnalysis:
    """
    Análisis avanzado para clasificación de homicidios con técnicas especializadas
    para problemas multiclase desbalanceados
    """
    
    def __init__(self, classifier_pipeline):
        self.pipeline = classifier_pipeline
        self.advanced_results = {}
        
    def analyze_class_distribution(self):
        """
        Análisis detallado de la distribución de clases
        """
        print("📊 ANÁLISIS DE DISTRIBUCIÓN DE CLASES")
        print("=" * 50)
        
        # Distribución en conjunto completo
        y_complete = self.pipeline.df['Circunstancia del Hecho'].dropna()
        class_dist = y_complete.value_counts(normalize=True) * 100
        
        # Calcular métricas de desbalance
        majority_class_pct = class_dist.max()
        minority_class_pct = class_dist.min()
        imbalance_ratio = majority_class_pct / minority_class_pct
        
        print(f"📈 Distribución de clases (%):")
        for clase, porcentaje in class_dist.items():
            print(f"   {clase}: {porcentaje:.2f}%")
        
        print(f"\n📊 Métricas de desbalance:")
        print(f"   Clase mayoritaria: {majority_class_pct:.2f}%")
        print(f"   Clase minoritaria: {minority_class_pct:.2f}%")
        print(f"   Ratio de desbalance: {imbalance_ratio:.2f}:1")
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        class_dist.plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.7)
        axes[0].set_title('Distribución de Clases')
        axes[0].set_ylabel('Porcentaje (%)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico de pastel
        axes[1].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Distribución de Clases (Circular)')
        
        plt.tight_layout()
        plt.show()
        
        return class_dist, imbalance_ratio
    
    def apply_advanced_balancing_techniques(self):
        """
        Aplica técnicas avanzadas de balanceamiento de clases
        """
        print("⚖️ APLICANDO TÉCNICAS AVANZADAS DE BALANCEAMIENTO")
        print("=" * 60)
        
        X_train, y_train = self.pipeline.X_train, self.pipeline.y_train
        
        # Técnicas a probar
        balancing_techniques = {
            'SMOTE-ENN': SMOTEENN(random_state=42),
            'SMOTE-Tomek': SMOTETomek(random_state=42),
            'ADASYN': ADASYN(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
        }
        
        balanced_datasets = {}
        
        for name, technique in balancing_techniques.items():
            print(f"🔄 Aplicando {name}...")
            
            try:
                X_balanced, y_balanced = technique.fit_resample(X_train, y_train)
                
                # Verificar distribución después del balanceamiento
                balanced_dist = Counter(y_balanced)
                
                balanced_datasets[name] = {
                    'X': X_balanced,
                    'y': y_balanced,
                    'distribution': balanced_dist,
                    'technique': technique
                }
                
                print(f"   ✅ {name}: {X_balanced.shape[0]} muestras")
                print(f"   📊 Nueva distribución: {balanced_dist}")
                
            except Exception as e:
                print(f"   ❌ Error en {name}: {e}")
        
        return balanced_datasets
    
    def evaluate_balancing_impact(self, balanced_datasets):
        """
        Evalúa el impacto de las técnicas de balanceamiento
        """
        print("📈 EVALUANDO IMPACTO DEL BALANCEAMIENTO")
        print("=" * 50)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, accuracy_score
        
        # Modelo base para evaluación
        base_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        balancing_results = {}
        
        # Evaluar dataset original
        base_model.fit(self.pipeline.X_train, self.pipeline.y_train)
        y_pred_original = base_model.predict(self.pipeline.X_test)
        
        balancing_results['Original'] = {
            'accuracy': accuracy_score(self.pipeline.y_test, y_pred_original),
            'f1_weighted': f1_score(self.pipeline.y_test, y_pred_original, average='weighted'),
            'f1_macro': f1_score(self.pipeline.y_test, y_pred_original, average='macro')
        }
        
        # Evaluar cada técnica de balanceamiento
        for name, data in balanced_datasets.items():
            base_model.fit(data['X'], data['y'])
            y_pred = base_model.predict(self.pipeline.X_test)
            
            balancing_results[name] = {
                'accuracy': accuracy_score(self.pipeline.y_test, y_pred),
                'f1_weighted': f1_score(self.pipeline.y_test, y_pred, average='weighted'),
                'f1_macro': f1_score(self.pipeline.y_test, y_pred, average='macro')
            }
        
        # Crear DataFrame para visualización
        results_df = pd.DataFrame(balancing_results).T
        
        # Visualizar resultados
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['accuracy', 'f1_weighted', 'f1_macro']
        titles = ['Accuracy', 'F1-Score (Weighted)', 'F1-Score (Macro)']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            results_df[metric].plot(kind='bar', ax=axes[i], color='lightcoral', alpha=0.7)
            axes[i].set_title(title)
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Resultados de balanceamiento:")
        print(results_df.round(3))
        
        # Identificar mejor técnica
        best_technique = results_df['f1_macro'].idxmax()
        print(f"\n🏆 Mejor técnica de balanceamiento: {best_technique}")
        
        return results_df, best_technique
    
    def learning_curves_analysis(self, model_name='Random Forest'):
        """
        Analiza las curvas de aprendizaje para detectar overfitting/underfitting
        """
        print(f"📈 ANÁLISIS DE CURVAS DE APRENDIZAJE - {model_name}")
        print("=" * 60)
        
        model = self.pipeline.results[model_name]['model']
        
        # Calcular curvas de aprendizaje
        train_sizes, train_scores, val_scores = learning_curve(
            model, self.pipeline.X_train, self.pipeline.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        # Calcular estadísticas
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Visualizar
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1-Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Análisis de convergencia
        gap = train_mean - val_mean
        
        plt.subplot(2, 2, 2)
        plt.plot(train_sizes, gap, 'o-', color='green')
        plt.xlabel('Training Set Size')
        plt.ylabel('Training - Validation Gap')
        plt.title('Overfitting Analysis')
        plt.grid(True, alpha=0.3)
        
        # Análisis de varianza de validación
        plt.subplot(2, 2, 3)
        plt.plot(train_sizes, val_std, 'o-', color='orange')
        plt.xlabel('Training Set Size')
        plt.ylabel('Validation Score Std')
        plt.title('Model Stability')
        plt.grid(True, alpha=0.3)
        
        # Eficiencia del entrenamiento
        plt.subplot(2, 2, 4)
        efficiency = val_mean / train_sizes * 1000  # Score per 1000 samples
        plt.plot(train_sizes, efficiency, 'o-', color='purple')
        plt.xlabel('Training Set Size')
        plt.ylabel('Validation Score per 1K samples')
        plt.title('Training Efficiency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Diagnóstico
        final_gap = gap[-1]
        final_val_score = val_mean[-1]
        
        print(f"📊 Diagnóstico del modelo:")
        print(f"   F1-Score final de validación: {final_val_score:.3f}")
        print(f"   Gap entrenamiento-validación: {final_gap:.3f}")
        
        if final_gap > 0.1:
            print("   ⚠️ Posible overfitting detectado")
        elif final_val_score < 0.5:
            print("   ⚠️ Posible underfitting detectado")
        else:
            print("   ✅ Modelo bien balanceado")
        
        return train_sizes, train_scores, val_scores
    
    def multiclass_roc_analysis(self, model_name='Random Forest'):
        """
        Análisis ROC para clasificación multiclase
        """
        print(f"📊 ANÁLISIS ROC MULTICLASE - {model_name}")
        print("=" * 50)
        
        model = self.pipeline.results[model_name]['model']
        
        # Obtener probabilidades de predicción
        y_proba = model.predict_proba(self.pipeline.X_test)
        
        # Binarizar las clases
        classes = model.classes_
        y_test_bin = label_binarize(self.pipeline.y_test, classes=classes)
        
        # Calcular ROC para cada clase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        n_classes = len(classes)
        
        # ROC para cada clase vs todas las demás
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # ROC micro-promedio
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # ROC macro-promedio
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Visualización
        plt.figure(figsize=(15, 10))
        
        # Subplot para todas las clases individuales
        plt.subplot(2, 2, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC por Clase')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Subplot para promedios
        plt.subplot(2, 2, 2)
        plt.plot(fpr["micro"], tpr["micro"], 
                label=f'Micro-promedio (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-promedio (AUC = {roc_auc["macro"]:.2f})',
                color='navy', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Promedio')
        plt.legend()
        
        # AUC por clase
        plt.subplot(2, 2, 3)
        auc_scores = [roc_auc[i] for i in range(n_classes)]
        plt.bar(range(n_classes), auc_scores, color=colors, alpha=0.7)
        plt.xlabel('Clase')
        plt.ylabel('AUC Score')
        plt.title('AUC por Clase')
        plt.xticks(range(n_classes), [f'C{i}' for i in range(n_classes)], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Distribución de AUC
        plt.subplot(2, 2, 4)
        plt.hist(auc_scores, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(np.mean(auc_scores), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(auc_scores):.3f}')
        plt.xlabel('AUC Score')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"📊 Resumen ROC:")
        print(f"   AUC Micro-promedio: {roc_auc['micro']:.3f}")
        print(f"   AUC Macro-promedio: {roc_auc['macro']:.3f}")
        print(f"   AUC promedio por clase: {np.mean(auc_scores):.3f}")
        print(f"   Desviación estándar AUC: {np.std(auc_scores):.3f}")
        
        return roc_auc, fpr, tpr
    
    def error_analysis(self, model_name='Random Forest'):
        """
        Análisis detallado de errores de clasificación
        """
        print(f"🔍 ANÁLISIS DE ERRORES - {model_name}")
        print("=" * 50)
        
        model = self.pipeline.results[model_name]['model']
        y_pred = self.pipeline.results[model_name]['y_pred_test']
        y_true = self.pipeline.y_test
        
        # Identificar errores
        errors_mask = y_pred != y_true
        correct_mask = y_pred == y_true
        
        print(f"📊 Resumen de errores:")
        print(f"   Total de predicciones: {len(y_true)}")
        print(f"   Predicciones correctas: {np.sum(correct_mask)} ({np.mean(correct_mask)*100:.1f}%)")
        print(f"   Predicciones incorrectas: {np.sum(errors_mask)} ({np.mean(errors_mask)*100:.1f}%)")
        
        # Análisis de errores por clase
        classes = np.unique(y_true)
        error_analysis = {}
        
        for clase in classes:
            class_mask = y_true == clase
            class_errors = np.sum(errors_mask & class_mask)
            class_total = np.sum(class_mask)
            class_error_rate = class_errors / class_total if class_total > 0 else 0
            
            error_analysis[clase] = {
                'total_samples': class_total,
                'errors': class_errors,
                'error_rate': class_error_rate,
                'accuracy': 1 - class_error_rate
            }
        
        # Crear DataFrame para visualización
        error_df = pd.DataFrame(error_analysis).T
        
        # Visualizar errores por clase
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error rate por clase
        error_df['error_rate'].plot(kind='bar', ax=axes[0,0], color='salmon', alpha=0.7)
        axes[0,0].set_title('Tasa de Error por Clase')
        axes[0,0].set_ylabel('Tasa de Error')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Número de errores por clase
        error_df['errors'].plot(kind='bar', ax=axes[0,1], color='lightcoral', alpha=0.7)
        axes[0,1].set_title('Número de Errores por Clase')
        axes[0,1].set_ylabel('Número de Errores')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Matriz de confusión normalizada
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1,0],
                   xticklabels=classes, yticklabels=classes)
        axes[1,0].set_title('Matriz de Confusión (Normalizada)')
        axes[1,0].set_xlabel('Predicción')
        axes[1,0].set_ylabel('Real')
        
        # Análisis de confusión más común
        cm_abs = confusion_matrix(y_true, y_pred)
        np.fill_diagonal(cm_abs, 0)  # Ignorar diagonal (aciertos)
        
        # Encontrar las confusiones más frecuentes
        max_confusion_idx = np.unravel_index(np.argmax(cm_abs), cm_abs.shape)
        max_confusion_count = cm_abs[max_confusion_idx]
        
        confusion_pairs = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm_abs[i,j] > 0:
                    confusion_pairs.append((classes[i], classes[j], cm_abs[i,j]))
        
        # Ordenar por frecuencia de confusión
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Mostrar top confusiones
        if confusion_pairs:
            top_confusions = confusion_pairs[:5]
            true_classes, pred_classes, counts = zip(*top_confusions)
            
            axes[1,1].bar(range(len(top_confusions)), counts, color='orange', alpha=0.7)
            axes[1,1].set_title('Top 5 Confusiones Más Frecuentes')
            axes[1,1].set_ylabel('Número de Confusiones')
            axes[1,1].set_xticks(range(len(top_confusions)))
            axes[1,1].set_xticklabels([f'{tc[:10]}→{pc[:10]}' for tc, pc in zip(true_classes, pred_classes)], rotation=45)
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Análisis de errores por clase:")
        print(error_df.round(3))
        
        if confusion_pairs:
            print(f"\n🔍 Top 5 confusiones más frecuentes:")
            for i, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
                print(f"   {i}. {true_cls} → {pred_cls}: {count} casos")
        
        return error_df, confusion_pairs
    
    def generate_comprehensive_report(self):
        """
        Genera un reporte comprensivo del análisis avanzado
        """
        print("📄 GENERANDO REPORTE COMPRENSIVO")
        print("=" * 50)
        
        # 1. Análisis de distribución de clases
        class_dist, imbalance_ratio = self.analyze_class_distribution()
        
        # 2. Técnicas de balanceamiento
        balanced_datasets = self.apply_advanced_balancing_techniques()
        
        # 3. Evaluación del impacto del balanceamiento
        if balanced_datasets:
            balancing_results, best_technique = self.evaluate_balancing_impact(balanced_datasets)
        else:
            balancing_results, best_technique = None, "No disponible"
        
        # 4. Curvas de aprendizaje para el mejor modelo
        best_model = max(self.pipeline.results.keys(), 
                        key=lambda x: self.pipeline.results[x]['test_f1'])
        learning_results = self.learning_curves_analysis(best_model)
        
        # 5. Análisis ROC multiclase
        roc_results = self.multiclass_roc_analysis(best_model)
        
        # 6. Análisis de errores
        error_results = self.error_analysis(best_model)
        
        # Compilar reporte final
        comprehensive_report = {
            'dataset_info': {
                'imbalance_ratio': imbalance_ratio,
                'class_distribution': class_dist,
                'total_samples': len(self.pipeline.y_test)
            },
            'best_model': best_model,
            'best_balancing_technique': best_technique,
            'balancing_results': balancing_results,
            'roc_analysis': roc_results[0],  # Solo AUC scores
            'error_analysis': error_results[0]
        }
        
        print("\n✅ REPORTE COMPRENSIVO COMPLETADO")
        print("=" * 50)
        print(f"🏆 Mejor modelo: {best_model}")
        print(f"⚖️ Mejor técnica de balanceamiento: {best_technique}")
        print(f"📊 Ratio de desbalance: {imbalance_ratio:.2f}:1")
        
        return comprehensive_report

# Función de uso
def run_advanced_analysis(classifier_pipeline):
    """
    Ejecuta el análisis avanzado completo
    """
    print("🚀 INICIANDO ANÁLISIS AVANZADO DE CLASIFICACIÓN")
    print("=" * 60)
    
    # Crear instancia del análisis avanzado
    advanced_analyzer = AdvancedClassificationAnalysis(classifier_pipeline)
    
    # Ejecutar análisis comprensivo
    report = advanced_analyzer.generate_comprehensive_report()
    
    print("\n🎯 ANÁLISIS AVANZADO COMPLETADO")
    print("Recomendaciones generadas para mejorar el modelo")
    
    return advanced_analyzer, report

if __name__ == "__main__":
    from classification_pipeline import HomicideClassificationPipeline
    
    # Cargar el pipeline de clasificación
    classifier = HomicideClassificationPipeline('../data/processed/homicidios_procesado.csv')
    classifier.run_complete_pipeline()
    
    # Ejecutar análisis avanzado
    advanced_analyzer, report = run_advanced_analysis(classifier)
    
    # Imprimir reporte final
    print("\n📄 Reporte Final:")
    for key, value in report.items():
        print(f"{key}: {value}")