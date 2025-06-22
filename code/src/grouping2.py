import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalysis:
    def __init__(self, df):
        self.df = df.copy()
        self.df_encoded = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.high_entropy_features = []
        self.clustering_results = {}
        
    def prepare_data(self, entropy_threshold=1.0):
        """Prepara los datos para clustering basándose en las características de alta entropía"""
        
        # Seleccionar características de alta entropía basándose en tu análisis
        self.high_entropy_features = [
            'Municipio del hecho DANE', 'Departamento del hecho DANE', 
            'Mes del hecho', 'Grupo de edad de la victima',
            'Dia del hecho', 'Escenario del Hecho', 
            'Actividad Durante el Hecho', 'Escolaridad',
            'Rango de Hora del Hecho X 3 Horas', 'Diagnostico Topográfico de la Lesión',
            'Presunto Agresor', 'Circunstancia del Hecho', 'Pertenencia Grupal',
            'Mecanismo Causal', 'Ancestro Racial', 'Zona del Hecho'
        ]
        
        # Filtrar características que existen en el dataset
        available_features = [col for col in self.high_entropy_features if col in self.df.columns]
        
        print(f"Características disponibles para clustering: {len(available_features)}")
        print(f"Características: {available_features}")
        
        # Crear DataFrame con características seleccionadas
        df_selected = self.df[available_features].copy()
        
        # Codificar variables categóricas
        self.df_encoded = df_selected.copy()
        
        for col in df_selected.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            # Manejar valores faltantes
            mask = df_selected[col].notna()
            if mask.any():
                self.df_encoded.loc[mask, col] = le.fit_transform(df_selected.loc[mask, col])
                self.label_encoders[col] = le
            else:
                self.df_encoded[col] = 0
        
        # Rellenar valores faltantes con la mediana
        self.df_encoded = self.df_encoded.fillna(self.df_encoded.median())
        
        # Escalar los datos
        self.df_scaled = self.scaler.fit_transform(self.df_encoded)
        
        print(f"Forma de los datos preparados: {self.df_scaled.shape}")
        
        return self.df_scaled
    
    def find_optimal_clusters(self, max_clusters=10):
        """Encuentra el número óptimo de clusters usando diferentes métodos"""
        print("Buscando el número óptimo de clusters...")
        # Método del codo (Elbow Method)
        sse = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            print(f"Evaluando k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.df_scaled)
            sse.append(kmeans.inertia_)
            print(f"SSE para k={k}: {sse[-1]:.2f}")
            silhouette_scores.append(silhouette_score(self.df_scaled, kmeans.labels_))
            print(f"Evaluando k={k}: SSE={sse[-1]:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")
        
        # Visualizar resultados
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Método del codo
        ax1.plot(k_range, sse, 'bo-')
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('SSE (Suma de Errores Cuadráticos)')
        ax1.set_title('Método del Codo')
        ax1.grid(True)
        
        # Puntuación de Silhouette
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Número de Clusters (k)')
        ax2.set_ylabel('Puntuación de Silhouette')
        ax2.set_title('Análisis de Silhouette')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Encontrar k óptimo basado en silhouette
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Número óptimo de clusters basado en Silhouette: {optimal_k}")
        
        return optimal_k, silhouette_scores
    
    def apply_clustering_algorithms(self, n_clusters=None):
        """Aplica diferentes algoritmos de clustering"""
        
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
        
        algorithms = {
            'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Gaussian Mixture': GaussianMixture(n_components=n_clusters, random_state=42)
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            print(f"Aplicando {name}...")
            
            try:
                if name == 'Gaussian Mixture':
                    labels = algorithm.fit_predict(self.df_scaled)
                else:
                    labels = algorithm.fit_predict(self.df_scaled)
                
                # Calcular métricas
                n_clusters_found = len(np.unique(labels))
                if n_clusters_found > 1:
                    silhouette = silhouette_score(self.df_scaled, labels)
                    calinski = calinski_harabasz_score(self.df_scaled, labels)
                    davies_bouldin = davies_bouldin_score(self.df_scaled, labels)
                else:
                    silhouette = calinski = davies_bouldin = 0
                
                results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters_found,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies_bouldin,
                    'algorithm': algorithm
                }
                
                print(f"  - Clusters encontrados: {n_clusters_found}")
                print(f"  - Silhouette Score: {silhouette:.3f}")
                print(f"  - Calinski-Harabasz Score: {calinski:.3f}")
                print(f"  - Davies-Bouldin Score: {davies_bouldin:.3f}")
                print()
                
            except Exception as e:
                print(f"Error en {name}: {e}")
                continue
        
        self.clustering_results = results
        return results
    
    def visualize_clusters(self, method='PCA'):
        """Visualiza los clusters usando PCA o t-SNE"""
        
        if not self.clustering_results:
            print("Primero debe ejecutar apply_clustering_algorithms()")
            return
        
        # Reducción de dimensionalidad
        if method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(self.df_scaled)
            explained_var = reducer.explained_variance_ratio_
            title_suffix = f'(PCA - Varianza explicada: {explained_var.sum():.2%})'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_reduced = reducer.fit_transform(self.df_scaled[:5000])  # Limitar para t-SNE
            title_suffix = '(t-SNE)'
        
        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, (name, result) in enumerate(self.clustering_results.items()):
            if i >= 4:  # Solo mostrar primeros 4 algoritmos
                break
                
            labels = result['labels']
            if method == 'TSNE':
                labels = labels[:5000]  # Limitar para t-SNE
            
            # Crear scatter plot
            unique_labels = np.unique(labels)
            for j, label in enumerate(unique_labels):
                if label == -1:  # Ruido en DBSCAN
                    color = 'black'
                    alpha = 0.3
                    s = 20
                else:
                    color = colors[j % len(colors)]
                    alpha = 0.7
                    s = 30
                
                mask = labels == label
                axes[i].scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                              c=color, alpha=alpha, s=s, 
                              label=f'Cluster {label}' if label != -1 else 'Ruido')
            
            axes[i].set_title(f'{name} {title_suffix}\nSilhouette: {result["silhouette_score"]:.3f}')
            axes[i].set_xlabel(f'Componente 1' if method == 'PCA' else 'Dimensión 1')
            axes[i].set_ylabel(f'Componente 2' if method == 'PCA' else 'Dimensión 2')
            axes[i].grid(True, alpha=0.3)
            
            # Limitar leyenda a pocos elementos
            if len(unique_labels) <= 10:
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cluster_characteristics(self, algorithm_name='KMeans'):
        """Analiza las características de cada cluster"""
        
        if algorithm_name not in self.clustering_results:
            print(f"Algoritmo {algorithm_name} no encontrado")
            return
        
        labels = self.clustering_results[algorithm_name]['labels']
        
        # Agregar labels al DataFrame original
        df_analysis = self.df_encoded.copy()
        df_analysis['Cluster'] = labels
        
        # Análisis por cluster
        cluster_summary = []
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Ruido en DBSCAN
                continue
                
            cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df_analysis)) * 100
            
            print(f"\n=== CLUSTER {cluster_id} ===")
            print(f"Tamaño: {cluster_size} ({cluster_percentage:.1f}%)")
            
            # Características más comunes por cluster
            for col in self.high_entropy_features[:8]:  # Top 8 características
                if col in df_analysis.columns:
                    if col in self.label_encoders:
                        # Decodificar para mostrar valores originales
                        try:
                            mode_encoded = cluster_data[col].mode().iloc[0]
                            mode_decoded = self.label_encoders[col].inverse_transform([int(mode_encoded)])[0]
                            frequency = (cluster_data[col] == mode_encoded).sum()
                            percentage = (frequency / cluster_size) * 100
                            print(f"  {col}: {mode_decoded} ({percentage:.1f}%)")
                        except:
                            print(f"  {col}: No disponible")
                    else:
                        mode_value = cluster_data[col].mode()
                        if not mode_value.empty:
                            frequency = (cluster_data[col] == mode_value.iloc[0]).sum()
                            percentage = (frequency / cluster_size) * 100
                            print(f"  {col}: {mode_value.iloc[0]} ({percentage:.1f}%)")
            
            cluster_summary.append({
                'Cluster': cluster_id,
                'Tamaño': cluster_size,
                'Porcentaje': cluster_percentage
            })
        
        # Crear DataFrame resumen
        summary_df = pd.DataFrame(cluster_summary)
        
        # Visualizar distribución de clusters
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df['Cluster'], summary_df['Tamaño'], alpha=0.7)
        plt.xlabel('Cluster')
        plt.ylabel('Número de Casos')
        plt.title(f'Distribución de Casos por Cluster ({algorithm_name})')
        plt.grid(True, alpha=0.3)
        
        # Agregar porcentajes en las barras
        for i, (cluster, size, pct) in enumerate(zip(summary_df['Cluster'], 
                                                   summary_df['Tamaño'], 
                                                   summary_df['Porcentaje'])):
            plt.text(cluster, size + max(summary_df['Tamaño']) * 0.01, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return summary_df
    
    def compare_algorithms(self):
        """Compara el rendimiento de los diferentes algoritmos"""
        
        if not self.clustering_results:
            print("Primero debe ejecutar apply_clustering_algorithms()")
            return
        
        comparison_data = []
        
        for name, result in self.clustering_results.items():
            comparison_data.append({
                'Algoritmo': name,
                'N_Clusters': result['n_clusters'],
                'Silhouette_Score': result['silhouette_score'],
                'Calinski_Harabasz_Score': result['calinski_harabasz_score'],
                'Davies_Bouldin_Score': result['davies_bouldin_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Visualizar comparación
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Silhouette Score (más alto es mejor)
        axes[0].bar(comparison_df['Algoritmo'], comparison_df['Silhouette_Score'], alpha=0.7)
        axes[0].set_title('Silhouette Score\n(Más alto es mejor)')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Calinski-Harabasz Score (más alto es mejor)
        axes[1].bar(comparison_df['Algoritmo'], comparison_df['Calinski_Harabasz_Score'], 
                   alpha=0.7, color='orange')
        axes[1].set_title('Calinski-Harabasz Score\n(Más alto es mejor)')
        axes[1].set_ylabel('Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Davies-Bouldin Score (más bajo es mejor)
        axes[2].bar(comparison_df['Algoritmo'], comparison_df['Davies_Bouldin_Score'], 
                   alpha=0.7, color='red')
        axes[2].set_title('Davies-Bouldin Score\n(Más bajo es mejor)')
        axes[2].set_ylabel('Score')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df

from config import PROCESSED_DATA_PATH
from data_loading import load_raw_data

df = load_raw_data(PROCESSED_DATA_PATH)

# Ejemplo de uso
if __name__ == "__main__":
    
    # Crear instancia del analizador
    analyzer = ClusteringAnalysis(df)
    
    # Preparar datos
    analyzer.prepare_data()
    
    # Encontrar número óptimo de clusters
    optimal_k, silhouette_scores = analyzer.find_optimal_clusters()
    
    # Aplicar algoritmos de clustering
    results = analyzer.apply_clustering_algorithms(n_clusters=optimal_k)
    
    # Visualizar clusters
    analyzer.visualize_clusters(method='PCA')
    analyzer.visualize_clusters(method='TSNE')
    
    # Analizar características de clusters
    summary = analyzer.analyze_cluster_characteristics('KMeans')
    
    # Comparar algoritmos
    comparison = analyzer.compare_algorithms()
    
    print("Análisis de clustering completado!")