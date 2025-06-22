import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClusteringAnalysis:
    """
    Clase para realizar análisis de agrupación jerárquica en datos de homicidios.
    """
    
    def __init__(self, df):
        """
        Inicializa la clase con el DataFrame de datos.
        
        Parameters:
        df: DataFrame con los datos preprocesados
        """
        self.df = df.copy()
        self.df_encoded = None
        self.scaled_data = None
        self.feature_importance = None
        self.clustering_results = {}
        
    def prepare_data_for_clustering(self, top_n_features=15):
        """
        Prepara los datos para agrupación jerárquica.
        
        Parameters:
        top_n_features: Número de características más importantes a utilizar
        """
        print("🔄 Preparando datos para agrupación jerárquica...")
        
        # Seleccionar características más importantes basadas en entropía
        feature_scores = {
            'Municipio del hecho DANE': 5.198396,
            'Departamento del hecho DANE': 2.898963,
            'Mes del hecho': 2.483282,
            'Grupo de edad de la victima': 2.322627,
            'Dia del hecho': 1.924807,
            'Escenario del Hecho': 1.848538,
            'Actividad Durante el Hecho': 1.752705,
            'Escolaridad': 1.706606,
            'Rango de Hora del Hecho X 3 Horas': 1.539295,
            'Diagnostico Topográfico de la Lesión': 1.510262,
            'Presunto Agresor': 1.427516,
            'Circunstancia del Hecho': 1.107589,
            'Pertenencia Grupal': 0.990844,
            'Mecanismo Causal': 0.905714,
            'Ancestro Racial': 0.835886
        }
        
        # Seleccionar las top_n_features más importantes que existan en el DataFrame
        available_features = [col for col in feature_scores.keys() if col in self.df.columns]
        selected_features = available_features[:top_n_features]
        
        print(f"📊 Características seleccionadas: {len(selected_features)}")
        for i, feature in enumerate(selected_features, 1):
            print(f"   {i}. {feature} (score: {feature_scores[feature]:.3f})")
        
        # Crear DataFrame con características seleccionadas
        self.df_selected = self.df[selected_features].copy()
        
        # Codificar variables categóricas
        self.df_encoded = self.df_selected.copy()
        self.label_encoders = {}
        
        for col in self.df_encoded.columns:
            if self.df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                self.df_encoded[col] = le.fit_transform(self.df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Escalar los datos
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.df_encoded)
        
        print(f"✅ Datos preparados: {self.scaled_data.shape[0]} observaciones, {self.scaled_data.shape[1]} características")
        
        return self.scaled_data
    
    def perform_hierarchical_clustering(self, methods=['ward', 'complete', 'average'], 
                                      max_clusters=10, sample_size=5000):
        """
        Realiza agrupación jerárquica con diferentes métodos.
        
        Parameters:
        methods: Lista de métodos de enlace a probar
        max_clusters: Número máximo de clusters a evaluar
        sample_size: Tamaño de muestra para acelerar el proceso
        """
        print("🔄 Realizando agrupación jerárquica...")
        
        # Tomar muestra si el dataset es muy grande
        if len(self.scaled_data) > sample_size:
            print(f"📊 Tomando muestra de {sample_size} observaciones para acelerar el análisis...")
            sample_idx = np.random.choice(len(self.scaled_data), sample_size, replace=False)
            data_sample = self.scaled_data[sample_idx]
        else:
            data_sample = self.scaled_data
            sample_idx = np.arange(len(self.scaled_data))
        
        # Realizar clustering con diferentes métodos
        for method in methods:
            print(f"\n🔍 Analizando método: {method.upper()}")
            
            # Calcular matriz de enlace
            linkage_matrix = linkage(data_sample, method=method)
            
            # Evaluar diferentes números de clusters
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            for n_clusters in range(2, max_clusters + 1):
                # Obtener clusters
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                # Calcular métricas
                sil_score = silhouette_score(data_sample, clusters)
                cal_score = calinski_harabasz_score(data_sample, clusters)
                db_score = davies_bouldin_score(data_sample, clusters)
                
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
                davies_bouldin_scores.append(db_score)
            
            # Guardar resultados
            self.clustering_results[method] = {
                'linkage_matrix': linkage_matrix,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'sample_idx': sample_idx,
                'data_sample': data_sample
            }
            
            print(f"   ✅ Método {method} completado")
    
    def plot_dendrograms(self, figsize=(15, 10)):
        """
        Crea dendrogramas para los diferentes métodos de enlace.
        """
        print("📊 Generando dendrogramas...")
        
        n_methods = len(self.clustering_results)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, results) in enumerate(self.clustering_results.items()):
            ax = axes[i]
            
            # Crear dendrograma
            dendrogram(results['linkage_matrix'], 
                      ax=ax, 
                      truncate_mode='level',
                      p=10,  # Mostrar solo los últimos 10 niveles
                      show_leaf_counts=True)
            
            ax.set_title(f'Dendrograma - Método {method.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Índice de muestra o (tamaño del cluster)', fontsize=12)
            ax.set_ylabel('Distancia', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_clustering_metrics(self, figsize=(15, 5)):
        """
        Visualiza las métricas de evaluación de clustering.
        """
        print("📊 Visualizando métricas de clustering...")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Preparar datos para plotting
        n_clusters_range = range(2, len(self.clustering_results[list(self.clustering_results.keys())[0]]['silhouette_scores']) + 2)
        
        # Silhouette Score
        ax1 = axes[0]
        for method, results in self.clustering_results.items():
            ax1.plot(n_clusters_range, results['silhouette_scores'], 
                    marker='o', label=method.capitalize(), linewidth=2)
        ax1.set_xlabel('Número de Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score por Método')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calinski-Harabasz Index
        ax2 = axes[1]
        for method, results in self.clustering_results.items():
            ax2.plot(n_clusters_range, results['calinski_scores'], 
                    marker='s', label=method.capitalize(), linewidth=2)
        ax2.set_xlabel('Número de Clusters')
        ax2.set_ylabel('Calinski-Harabasz Index')
        ax2.set_title('Calinski-Harabasz Index por Método')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Davies-Bouldin Index
        ax3 = axes[2]
        for method, results in self.clustering_results.items():
            ax3.plot(n_clusters_range, results['davies_bouldin_scores'], 
                    marker='^', label=method.capitalize(), linewidth=2)
        ax3.set_xlabel('Número de Clusters')
        ax3.set_ylabel('Davies-Bouldin Index')
        ax3.set_title('Davies-Bouldin Index por Método')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def find_optimal_clusters(self):
        """
        Encuentra el número óptimo de clusters para cada método.
        """
        print("🔍 Buscando número óptimo de clusters...")
        
        optimal_results = {}
        
        for method, results in self.clustering_results.items():
            # Mejor silhouette score (más alto es mejor)
            best_sil_idx = np.argmax(results['silhouette_scores'])
            best_sil_clusters = best_sil_idx + 2
            
            # Mejor Calinski-Harabasz (más alto es mejor)
            best_cal_idx = np.argmax(results['calinski_scores'])
            best_cal_clusters = best_cal_idx + 2
            
            # Mejor Davies-Bouldin (más bajo es mejor)
            best_db_idx = np.argmin(results['davies_bouldin_scores'])
            best_db_clusters = best_db_idx + 2
            
            optimal_results[method] = {
                'silhouette_optimal': best_sil_clusters,
                'calinski_optimal': best_cal_clusters,
                'davies_bouldin_optimal': best_db_clusters,
                'silhouette_score': results['silhouette_scores'][best_sil_idx],
                'calinski_score': results['calinski_scores'][best_cal_idx],
                'davies_bouldin_score': results['davies_bouldin_scores'][best_db_idx]
            }
        
        # Mostrar resultados
        print("\n📋 RESULTADOS ÓPTIMOS POR MÉTODO:")
        print("=" * 80)
        
        for method, opt_results in optimal_results.items():
            print(f"\n🔹 MÉTODO: {method.upper()}")
            print(f"   Silhouette Score: {opt_results['silhouette_optimal']} clusters (score: {opt_results['silhouette_score']:.3f})")
            print(f"   Calinski-Harabasz: {opt_results['calinski_optimal']} clusters (score: {opt_results['calinski_score']:.3f})")
            print(f"   Davies-Bouldin: {opt_results['davies_bouldin_optimal']} clusters (score: {opt_results['davies_bouldin_score']:.3f})")
        
        return optimal_results
    
    def perform_final_clustering(self, method='ward', n_clusters=None):
        """
        Realiza el clustering final con el método y número de clusters especificados.
        
        Parameters:
        method: Método de enlace a usar
        n_clusters: Número de clusters (si es None, usa el óptimo según silhouette)
        """
        print(f"🎯 Realizando clustering final...")
        
        if n_clusters is None:
            # Usar el óptimo según silhouette score
            optimal_results = self.find_optimal_clusters()
            n_clusters = optimal_results[method]['silhouette_optimal']
        
        print(f"   Método: {method}")
        print(f"   Número de clusters: {n_clusters}")
        
        # Realizar clustering en todos los datos
        clustering_model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=method
        )
        
        cluster_labels = clustering_model.fit_predict(self.scaled_data)
        
        # Añadir labels al DataFrame original
        self.df['Cluster'] = cluster_labels
        
        # Calcular métricas finales
        final_silhouette = silhouette_score(self.scaled_data, cluster_labels)
        final_calinski = calinski_harabasz_score(self.scaled_data, cluster_labels)
        final_davies_bouldin = davies_bouldin_score(self.scaled_data, cluster_labels)
        
        print(f"\n📊 MÉTRICAS FINALES:")
        print(f"   Silhouette Score: {final_silhouette:.3f}")
        print(f"   Calinski-Harabasz Index: {final_calinski:.3f}")
        print(f"   Davies-Bouldin Index: {final_davies_bouldin:.3f}")
        
        return cluster_labels, {
            'silhouette': final_silhouette,
            'calinski': final_calinski,
            'davies_bouldin': final_davies_bouldin
        }
    
    def analyze_clusters(self, cluster_labels):
        """
        Analiza las características de cada cluster.
        """
        print("🔍 Analizando características de los clusters...")
        
        # Estadísticas básicas por cluster
        cluster_stats = self.df.groupby('Cluster').size().reset_index(name='Tamaño')
        cluster_stats['Porcentaje'] = (cluster_stats['Tamaño'] / len(self.df) * 100).round(2)
        
        print("\n📊 DISTRIBUCIÓN DE CLUSTERS:")
        print(cluster_stats.to_string(index=False))
        
        # Análisis de características categóricas principales
        categorical_features = ['Circunstancia del Hecho', 'Departamento del hecho DANE', 
                              'Escenario del Hecho', 'Grupo de edad de la victima']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                print(f"\n🔹 ANÁLISIS DE: {feature}")
                print("-" * 50)
                
                # Crear tabla de contingencia
                contingency = pd.crosstab(self.df['Cluster'], self.df[feature], normalize='index')
                contingency_pct = (contingency * 100).round(1)
                
                # Mostrar los valores más representativos por cluster
                for cluster in sorted(self.df['Cluster'].unique()):
                    top_values = contingency_pct.loc[cluster].nlargest(3)
                    print(f"   Cluster {cluster}: {', '.join([f'{idx} ({val}%)' for idx, val in top_values.items()])}")
    
    def visualize_clusters_pca(self, figsize=(12, 8)):
        """
        Visualiza los clusters usando PCA para reducción de dimensionalidad.
        """
        print("📊 Visualizando clusters con PCA...")
        
        # Aplicar PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.scaled_data)
        
        # Crear el plot
        plt.figure(figsize=figsize)
        
        # Obtener colores únicos para cada cluster
        unique_clusters = sorted(self.df['Cluster'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = self.df['Cluster'] == cluster
            plt.scatter(pca_data[mask, 0], pca_data[mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster}', 
                       alpha=0.6, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
        plt.title('Visualización de Clusters (PCA)', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"📈 Varianza explicada total: {pca.explained_variance_ratio_.sum():.1%}")
    
    def generate_cluster_report(self):
        """
        Genera un reporte completo del análisis de clustering.
        """
        print("\n" + "="*80)
        print("📋 REPORTE COMPLETO DE ANÁLISIS DE CLUSTERING")
        print("="*80)
        
        print(f"\n📊 INFORMACIÓN GENERAL:")
        print(f"   Total de observaciones: {len(self.df):,}")
        print(f"   Características utilizadas: {self.scaled_data.shape[1]}")
        print(f"   Número de clusters encontrados: {len(self.df['Cluster'].unique())}")
        
        # Distribución de clusters
        cluster_dist = self.df['Cluster'].value_counts().sort_index()
        print(f"\n📈 DISTRIBUCIÓN DE CLUSTERS:")
        for cluster, count in cluster_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"   Cluster {cluster}: {count:,} observaciones ({percentage:.1f}%)")
        
        # Características más discriminativas
        if hasattr(self, 'df_selected'):
            print(f"\n🎯 CARACTERÍSTICAS MÁS IMPORTANTES:")
            feature_importance = [
                'Municipio del hecho DANE', 'Departamento del hecho DANE', 
                'Mes del hecho', 'Grupo de edad de la victima', 'Dia del hecho'
            ]
            
            for i, feature in enumerate(feature_importance[:5], 1):
                if feature in self.df.columns:
                    print(f"   {i}. {feature}")
        
        print("\n" + "="*80)

# Función principal para ejecutar el análisis
def run_hierarchical_clustering_analysis(df):
    """
    Ejecuta el análisis completo de agrupación jerárquica.
    
    Parameters:
    df: DataFrame con los datos preprocesados
    """
    print("🚀 INICIANDO ANÁLISIS DE AGRUPACIÓN JERÁRQUICA")
    print("="*60)
    
    # Crear instancia del analizador
    analyzer = HierarchicalClusteringAnalysis(df)
    
    # 1. Preparar datos
    analyzer.prepare_data_for_clustering(top_n_features=10)
    
    # 2. Realizar clustering con diferentes métodos
    analyzer.perform_hierarchical_clustering(
        methods=['ward', 'complete', 'average'], 
        max_clusters=8,
        sample_size=3000
    )
    
    # 3. Visualizar dendrogramas
    analyzer.plot_dendrograms()
    
    # 4. Visualizar métricas
    analyzer.plot_clustering_metrics()
    
    # 5. Encontrar clustering óptimo
    optimal_results = analyzer.find_optimal_clusters()
    print("\n🔍 Resultados óptimos encontrados:")
    for method, results in optimal_results.items():
        print(f"   Método {method.upper()}:")
        print(f"   - Silhouette: {results['silhouette_optimal']} clusters (score: {results['silhouette_score']:.3f})")
        print(f"   - Calinski-Harabasz: {results['calinski_optimal']} clusters (score: {results['calinski_score']:.3f})")
        print(f"   - Davies-Bouldin: {results['davies_bouldin_optimal']} clusters (score: {results['davies_bouldin_score']:.3f})")
    
    # 6. Realizar clustering final
    cluster_labels, final_metrics = analyzer.perform_final_clustering(
        method='ward', 
        n_clusters=None
    )
    
    # 7. Analizar clusters
    analyzer.analyze_clusters(cluster_labels)
    
    # 8. Visualizar clusters
    analyzer.visualize_clusters_pca()
    
    # 9. Generar reporte
    analyzer.generate_cluster_report()
    
    return analyzer, cluster_labels, final_metrics

# Ejemplo de uso:

from config import PROCESSED_DATA_PATH
from data_loading import load_raw_data

df = load_raw_data(PROCESSED_DATA_PATH)
analyzer, clusters, metrics = run_hierarchical_clustering_analysis(df)