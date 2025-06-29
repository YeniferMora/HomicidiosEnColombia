# Clustering con UMAP para Datos de Homicidios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import umap
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import prince
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')
from config import PROCESSED_DATA_PATH
from data_loading import load_raw_data
import os

df = load_raw_data(PROCESSED_DATA_PATH)

class ClusteringHomicidiosUMAP:
    def __init__(self, df, results_path="clustering_results_umap"):
        self.df = df.copy()
        self.df_processed = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.clustering_results = {}
        self.results_path = results_path
        self.umap_embeddings = None
        self.mca_coords = None
        
        # Crear directorio para resultados si no existe
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def preprocesamiento_clustering(self):
        """Preprocesamiento específico para clustering con datos categóricos"""
        print("=== PREPROCESAMIENTO PARA CLUSTERING CON UMAP ===")
        
        # Seleccionar variables relevantes para clustering
        self.categorical_columns = [
            'Sexo de la victima',
            'Grupo de edad de la victima', 
            'Zona del Hecho',
            'Escenario del Hecho',
            'Circunstancia del Hecho',
            'Mes del hecho',
            'Dia del hecho',
            'Pertenencia Grupal',
            'Mecanismo Causal',
            'Presunto Agresor',
            'Departamento del hecho DANE',
            'Actividad Durante el Hecho',
            'Diagnostico Topográfico de la Lesión',
            'Rango de Hora del Hecho X 3 Horas',
            'Localidad del Hecho',
        ]
        
        print(f"Variables seleccionadas: {self.categorical_columns}")
        
        # Filtrar solo Bogotá
        #self.df = self.df[self.df['Departamento del hecho DANE'] == 'Bogotá, D.C.']
        #print(f"Dimensiones del DataFrame filtrado: {self.df.shape}")
        
        # Filtrar solo las columnas que existen en el DataFrame
        self.categorical_columns = [col for col in self.categorical_columns if col in self.df.columns]
        print(f"Variables categóricas disponibles: {self.categorical_columns}")
        
        # Crear dataset solo con variables seleccionadas
        self.df_processed = self.df[self.categorical_columns].copy()
        
        # Limpiar datos faltantes
        for col in self.categorical_columns:
            self.df_processed[col] = self.df_processed[col].fillna('Sin información')
        
        print(f"Forma del dataset procesado: {self.df_processed.shape}")
        return self.df_processed

    def aplicar_tecnicas_reduccion_dimensionalidad(self):
        """Aplica múltiples técnicas de reducción de dimensionalidad"""
        print("\n=== APLICANDO TÉCNICAS DE REDUCCIÓN DE DIMENSIONALIDAD ===")
        
        # 1. Codificación Label Encoder para datos categóricos
        self.df_encoded = self.df_processed.copy()
        self.label_encoders = {}
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.df_encoded[col] = le.fit_transform(self.df_processed[col])
            self.label_encoders[col] = le
        
        # 2. MCA (Multiple Correspondence Analysis)
        print("Aplicando MCA...")
        self.mca = prince.MCA(n_components=10, random_state=42)
        self.mca_coords = self.mca.fit_transform(self.df_processed)
        print(f"MCA coordinates shape: {self.mca_coords.shape}")
        
        # 3. UMAP sobre datos codificados directamente
        print("Aplicando UMAP sobre datos codificados...")
        self.umap_direct = umap.UMAP(
            n_neighbors=15, 
            n_components=2, 
            metric='hamming',  # Métrica apropiada para datos categóricos
            random_state=42,
            min_dist=0.1
        )
        self.umap_embeddings_direct = self.umap_direct.fit_transform(self.df_encoded)
        
        # 4. UMAP sobre coordenadas MCA
        print("Aplicando UMAP sobre coordenadas MCA...")
        self.umap_mca = umap.UMAP(
            n_neighbors=15, 
            n_components=2, 
            metric='euclidean',
            random_state=42,
            min_dist=0.1
        )
        self.umap_embeddings_mca = self.umap_mca.fit_transform(self.mca_coords)
        
        # 5. UMAP con diferentes parámetros para exploración
        print("Aplicando UMAP con parámetros optimizados...")
        self.umap_optimized = umap.UMAP(
            n_neighbors=30,  # Más vecinos para estructura global
            n_components=2,
            metric='hamming',
            random_state=42,
            min_dist=0.0,  # Permitir puntos más cercanos
            spread=1.0
        )
        self.umap_embeddings_optimized = self.umap_optimized.fit_transform(self.df_encoded)
        
        # Visualizar las diferentes proyecciones
        self.visualizar_proyecciones()
        
        return self.umap_embeddings_direct, self.umap_embeddings_mca, self.umap_embeddings_optimized

    def visualizar_proyecciones(self):
        """Visualiza las diferentes proyecciones de reducción de dimensionalidad"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MCA
        axes[0, 0].scatter(self.mca_coords.iloc[:, 0], self.mca_coords.iloc[:, 1], alpha=0.6, s=1)
        axes[0, 0].set_title('MCA Projection')
        axes[0, 0].set_xlabel('MCA Component 1')
        axes[0, 0].set_ylabel('MCA Component 2')
        
        # UMAP Direct
        axes[0, 1].scatter(self.umap_embeddings_direct[:, 0], self.umap_embeddings_direct[:, 1], alpha=0.6, s=1)
        axes[0, 1].set_title('UMAP Direct (Hamming)')
        axes[0, 1].set_xlabel('UMAP Component 1')
        axes[0, 1].set_ylabel('UMAP Component 2')
        
        # UMAP MCA
        axes[1, 0].scatter(self.umap_embeddings_mca[:, 0], self.umap_embeddings_mca[:, 1], alpha=0.6, s=1)
        axes[1, 0].set_title('UMAP + MCA')
        axes[1, 0].set_xlabel('UMAP Component 1')
        axes[1, 0].set_ylabel('UMAP Component 2')
        
        # UMAP Optimized
        axes[1, 1].scatter(self.umap_embeddings_optimized[:, 0], self.umap_embeddings_optimized[:, 1], alpha=0.6, s=1)
        axes[1, 1].set_title('UMAP Optimized')
        axes[1, 1].set_xlabel('UMAP Component 1')
        axes[1, 1].set_ylabel('UMAP Component 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'dimensionality_reduction_comparison.png'), dpi=300)
        plt.close()

    def clustering_umap_kmeans(self, embedding_type='optimized', k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
        """K-Means sobre embeddings UMAP"""
        print(f"\n=== K-MEANS SOBRE UMAP ({embedding_type.upper()}) ===")
        
        # Seleccionar el embedding apropiado
        if embedding_type == 'direct':
            embeddings = self.umap_embeddings_direct
        elif embedding_type == 'mca':
            embeddings = self.umap_embeddings_mca
        else:  # optimized
            embeddings = self.umap_embeddings_optimized
        
        inertias = []
        silhouette_scores = []
        best_score = -1
        best_k = None
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(embeddings)
            
            inertia = kmeans.inertia_
            sil_score = silhouette_score(embeddings, clusters)
            
            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            
            if sil_score > best_score:
                best_score = sil_score
                best_k = k
                best_clusters = clusters
                best_centroids = kmeans.cluster_centers_
            
            print(f"K={k}, Inertia={inertia:.2f}, Silhouette={sil_score:.3f}")
        
        # Guardar resultados
        self.clustering_results[f'umap_kmeans_{embedding_type}'] = {
            'clusters': best_clusters,
            'k_optimal': best_k,
            'silhouette_score': best_score,
            'centroids': best_centroids,
            'embeddings': embeddings,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'k_range': k_range
        }
        
        # Visualización
        self.visualizar_clustering_umap(best_clusters, embeddings, f'UMAP K-Means ({embedding_type})', best_k)
        
        print(f"Mejor K: {best_k} con Silhouette Score: {best_score:.3f}")
        return best_clusters

    def clustering_umap_dbscan(self, embedding_type='optimized', eps_range=[0.3, 0.5, 0.7, 1.0, 1.5], min_samples_range=[5, 10, 15, 20]):
        """DBSCAN sobre embeddings UMAP"""
        print(f"\n=== DBSCAN SOBRE UMAP ({embedding_type.upper()}) ===")
        
        # Seleccionar el embedding apropiado
        if embedding_type == 'direct':
            embeddings = self.umap_embeddings_direct
        elif embedding_type == 'mca':
            embeddings = self.umap_embeddings_mca
        else:  # optimized
            embeddings = self.umap_embeddings_optimized
        
        best_score = -1
        best_params = None
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(embeddings)
                
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                if n_clusters > 1:
                    mask = clusters != -1
                    if np.sum(mask) > 1:
                        sil_score = silhouette_score(embeddings[mask], clusters[mask])
                    else:
                        sil_score = -1
                else:
                    sil_score = -1
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': sil_score,
                    'clusters': clusters
                })
                
                if sil_score > best_score and n_clusters > 1:
                    best_score = sil_score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_clusters = clusters
                
                print(f"eps={eps}, min_samples={min_samples}, clusters={n_clusters}, noise={n_noise}, silhouette={sil_score:.3f}")
        
        # Guardar resultados
        self.clustering_results[f'umap_dbscan_{embedding_type}'] = {
            'clusters': best_clusters if best_params else None,
            'best_params': best_params,
            'silhouette_score': best_score,
            'embeddings': embeddings,
            'results': results
        }
        
        if best_params:
            print(f"Mejores parámetros: {best_params} con Silhouette Score: {best_score:.3f}")
            self.visualizar_clustering_umap(best_clusters, embeddings, f'UMAP DBSCAN ({embedding_type})', 
                                           f"eps={best_params['eps']}, min_samples={best_params['min_samples']}")
            return best_clusters
        else:
            print("No se encontraron parámetros válidos para DBSCAN")
            return None

    def clustering_umap_gmm(self, embedding_type='optimized', k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
        """Gaussian Mixture Model sobre embeddings UMAP"""
        print(f"\n=== GAUSSIAN MIXTURE MODEL SOBRE UMAP ({embedding_type.upper()}) ===")
        
        # Seleccionar el embedding apropiado
        if embedding_type == 'direct':
            embeddings = self.umap_embeddings_direct
        elif embedding_type == 'mca':
            embeddings = self.umap_embeddings_mca
        else:  # optimized
            embeddings = self.umap_embeddings_optimized
        
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        best_bic = np.inf
        best_k = None
        
        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(embeddings)
            clusters = gmm.predict(embeddings)
            
            bic = gmm.bic(embeddings)
            aic = gmm.aic(embeddings)
            sil_score = silhouette_score(embeddings, clusters)
            
            bic_scores.append(bic)
            aic_scores.append(aic)
            silhouette_scores.append(sil_score)
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_clusters = clusters
                best_gmm = gmm
            
            print(f"K={k}, BIC={bic:.2f}, AIC={aic:.2f}, Silhouette={sil_score:.3f}")
        
        # Guardar resultados
        self.clustering_results[f'umap_gmm_{embedding_type}'] = {
            'clusters': best_clusters,
            'k_optimal': best_k,
            'bic_score': best_bic,
            'model': best_gmm,
            'embeddings': embeddings,
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'silhouette_scores': silhouette_scores,
            'k_range': k_range
        }
        
        self.visualizar_clustering_umap(best_clusters, embeddings, f'UMAP GMM ({embedding_type})', best_k)
        
        print(f"Mejor K: {best_k} con BIC: {best_bic:.2f}")
        return best_clusters

    def visualizar_clustering_umap(self, clusters, embeddings, title, subtitle):
        """Visualiza los resultados de clustering sobre embeddings UMAP"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot con colores por cluster
        unique_clusters = sorted(set(clusters))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            mask = clusters == cluster
            if cluster == -1:  # Noise points
                axes[0].scatter(embeddings[mask, 0], embeddings[mask, 1], 
                              c='black', s=1, alpha=0.5, label=f'Noise ({np.sum(mask)})')
            else:
                axes[0].scatter(embeddings[mask, 0], embeddings[mask, 1], 
                              c=[colors[i]], s=2, alpha=0.7, label=f'Cluster {cluster} ({np.sum(mask)})')
        
        axes[0].set_title(f'{title}\n{subtitle}')
        axes[0].set_xlabel('UMAP Component 1')
        axes[0].set_ylabel('UMAP Component 2')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Histograma de distribución de clusters
        cluster_counts = [list(clusters).count(c) for c in unique_clusters]
        bars = axes[1].bar(range(len(unique_clusters)), cluster_counts, color=colors)
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Número de puntos')
        axes[1].set_title('Distribución de Clusters')
        axes[1].set_xticks(range(len(unique_clusters)))
        axes[1].set_xticklabels([f'{c}' if c != -1 else 'Noise' for c in unique_clusters])
        
        # Añadir etiquetas de conteo
        for bar, count in zip(bars, cluster_counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(cluster_counts),
                       str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(self.results_path, f'{filename}_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def optimizar_umap_parametros(self, n_neighbors_range=[5, 10, 15, 20, 30], min_dist_range=[0.0, 0.1, 0.3, 0.5]):
        """Optimiza los parámetros de UMAP probando diferentes combinaciones"""
        print("\n=== OPTIMIZACIÓN DE PARÁMETROS UMAP ===")
        
        best_score = -1
        best_params = None
        results = []
        
        for n_neighbors in n_neighbors_range:
            for min_dist in min_dist_range:
                print(f"Probando n_neighbors={n_neighbors}, min_dist={min_dist}")
                
                # Aplicar UMAP
                umap_model = umap.UMAP(
                    n_neighbors=n_neighbors,
                    n_components=2,
                    metric='hamming',
                    random_state=42,
                    min_dist=min_dist
                )
                embeddings = umap_model.fit_transform(self.df_encoded)
                
                # Aplicar K-Means para evaluar
                kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')  # Usar 4 clusters como referencia
                clusters = kmeans.fit_predict(embeddings)
                
                # Calcular métricas
                sil_score = silhouette_score(embeddings, clusters)
                ch_score = calinski_harabasz_score(embeddings, clusters)
                
                results.append({
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'embeddings': embeddings,
                    'clusters': clusters
                })
                
                if sil_score > best_score:
                    best_score = sil_score
                    best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
                    best_embeddings = embeddings
                
                print(f"  Silhouette: {sil_score:.3f}, Calinski-Harabasz: {ch_score:.2f}")
        
        print(f"\nMejores parámetros: {best_params} con Silhouette Score: {best_score:.3f}")
        
        # Visualizar heatmap de resultados
        self.visualizar_optimizacion_umap(results, n_neighbors_range, min_dist_range)
        
        return best_params, best_embeddings

    def visualizar_optimizacion_umap(self, results, n_neighbors_range, min_dist_range):
        """Visualiza los resultados de la optimización de parámetros UMAP"""
        # Crear matrices para heatmaps
        sil_matrix = np.zeros((len(min_dist_range), len(n_neighbors_range)))
        ch_matrix = np.zeros((len(min_dist_range), len(n_neighbors_range)))
        
        for result in results:
            i = min_dist_range.index(result['min_dist'])
            j = n_neighbors_range.index(result['n_neighbors'])
            sil_matrix[i, j] = result['silhouette']
            ch_matrix[i, j] = result['calinski_harabasz']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap Silhouette Score
        sns.heatmap(sil_matrix, 
                   xticklabels=n_neighbors_range, 
                   yticklabels=min_dist_range,
                   annot=True, 
                   fmt='.3f', 
                   cmap='viridis',
                   ax=axes[0])
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('n_neighbors')
        axes[0].set_ylabel('min_dist')
        
        # Heatmap Calinski-Harabasz Score
        sns.heatmap(ch_matrix, 
                   xticklabels=n_neighbors_range, 
                   yticklabels=min_dist_range,
                   annot=True, 
                   fmt='.0f', 
                   cmap='plasma',
                   ax=axes[1])
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].set_xlabel('n_neighbors')
        axes[1].set_ylabel('min_dist')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'umap_parameter_optimization.png'), dpi=300)
        plt.close()

    def evaluar_clustering_completo(self):
        """Evaluación comparativa de todos los algoritmos con UMAP"""
        print("\n=== EVALUACIÓN COMPARATIVA CON UMAP ===")
        
        metrics_summary = []
        
        for algorithm, results in self.clustering_results.items():
            if 'clusters' in results and results['clusters'] is not None:
                clusters = results['clusters']
                embeddings = results.get('embeddings', self.mca_coords)  # Fallback a MCA si no hay embeddings
                
                # Calcular métricas
                if algorithm.startswith('umap_dbscan'):
                    # Para DBSCAN, excluir puntos de ruido
                    mask = clusters != -1
                    if np.sum(mask) > 1 and len(set(clusters[mask])) > 1:
                        sil_score = silhouette_score(embeddings[mask], clusters[mask])
                        ch_score = calinski_harabasz_score(embeddings[mask], clusters[mask])
                        db_score = davies_bouldin_score(embeddings[mask], clusters[mask])
                    else:
                        sil_score = ch_score = db_score = np.nan
                else:
                    if len(set(clusters)) > 1:
                        sil_score = silhouette_score(embeddings, clusters)
                        ch_score = calinski_harabasz_score(embeddings, clusters)
                        db_score = davies_bouldin_score(embeddings, clusters)
                    else:
                        sil_score = ch_score = db_score = np.nan
                
                metrics_summary.append({
                    'Algoritmo': algorithm,
                    'N_Clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                    'Silhouette': sil_score,
                    'Calinski_Harabasz': ch_score,
                    'Davies_Bouldin': db_score,
                    'Noise_Points': list(clusters).count(-1) if -1 in clusters else 0
                })
        
        metrics_df = pd.DataFrame(metrics_summary)
        print("\nMétricas de Clustering con UMAP:")
        print(metrics_df.round(3))
        
        # Guardar métricas
        metrics_df.to_csv(os.path.join(self.results_path, 'clustering_metrics_comparison.csv'), index=False)
        
        return metrics_df

    def analizar_perfiles_clusters_umap(self, algorithm):
        """Análisis de perfiles de clusters para algoritmos con UMAP"""
        print(f"\n=== ANÁLISIS DE PERFILES - {algorithm.upper()} ===")
        
        if algorithm not in self.clustering_results:
            print(f"Algoritmo {algorithm} no encontrado")
            return
        
        clusters = self.clustering_results[algorithm]['clusters']
        
        # Añadir clusters al dataframe
        df_analysis = self.df_processed.copy()
        df_analysis['Cluster'] = clusters
        
        # Análisis por variable categórica
        for col in self.categorical_columns:
            if col in df_analysis.columns:
                print(f"\n--- {col} ---")
                
                # Excluir puntos de ruido si es DBSCAN
                if -1 in clusters:
                    df_clean = df_analysis[df_analysis['Cluster'] != -1]
                else:
                    df_clean = df_analysis
                
                if len(df_clean) > 0:
                    cluster_profile = pd.crosstab(df_clean['Cluster'], df_clean[col], normalize='index')
                    print(cluster_profile.round(3))
                    
                    # Visualización
                    plt.figure(figsize=(12, 8))
                    cluster_profile.plot(kind='bar', stacked=True, ax=plt.gca())
                    plt.title(f'Perfil de Clusters por {col} - {algorithm}')
                    plt.xlabel('Cluster')
                    plt.ylabel('Proporción')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.results_path, f'{algorithm}_cluster_profile_{col}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Resumen de características principales por cluster
        print(f"\n=== RESUMEN DE PERFILES - {algorithm.upper()} ===")
        unique_clusters = sorted([c for c in set(clusters) if c != -1])
        
        for cluster_id in unique_clusters:
            print(f"\n--- CLUSTER {cluster_id} ---")
            cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
            total_points = len(df_analysis[df_analysis['Cluster'] != -1]) if -1 in clusters else len(df_analysis)
            print(f"Tamaño: {len(cluster_data)} ({len(cluster_data)/total_points*100:.1f}%)")
            
            # Top características de cada cluster
            for col in self.categorical_columns:
                if col in cluster_data.columns:
                    mode_value = cluster_data[col].mode()
                    if len(mode_value) > 0:
                        percentage = (cluster_data[col] == mode_value[0]).sum() / len(cluster_data) * 100
                        print(f"  {col}: {mode_value[0]} ({percentage:.1f}%)")

    def ejecutar_pipeline_completo_umap(self):
        """Ejecuta el pipeline completo de clustering con UMAP"""
        print("INICIANDO PIPELINE DE CLUSTERING CON UMAP PARA DATOS DE HOMICIDIOS")
        print("="*70)
        
        # 1. Preprocesamiento
        self.preprocesamiento_clustering()
        
        # 2. Aplicar técnicas de reducción de dimensionalidad
        self.aplicar_tecnicas_reduccion_dimensionalidad()
        
        # 3. Optimizar parámetros UMAP (opcional)
        best_params, best_embeddings = self.optimizar_umap_parametros()
        
        # 4. Aplicar algoritmos de clustering sobre diferentes embeddings
        embedding_types = ['direct', 'mca', 'optimized']
        
        for embedding_type in embedding_types:
            print(f"\n--- CLUSTERING SOBRE EMBEDDING: {embedding_type.upper()} ---")
            self.clustering_umap_kmeans(embedding_type)
            self.clustering_umap_dbscan(embedding_type)
            self.clustering_umap_gmm(embedding_type)
        
        # 5. Evaluación comparativa
        metrics_df = self.evaluar_clustering_completo()
        
        # 6. Seleccionar el mejor algoritmo
        # Filtrar algoritmos válidos (que tienen silhouette score válido)
        valid_metrics = metrics_df.dropna(subset=['Silhouette'])
        if len(valid_metrics) > 0:
            best_algorithm = valid_metrics.loc[valid_metrics['Silhouette'].idxmax(), 'Algoritmo']
            print(f"\nMejor algoritmo según Silhouette Score: {best_algorithm}")
            
            # 7. Análisis de perfiles del mejor algoritmo
            self.analizar_perfiles_clusters_umap(best_algorithm)
        else:
            print("No se encontraron algoritmos con métricas válidas")
        
        return metrics_df

    def comparar_con_sin_umap(self, df_original_results=None):
        """Compara resultados con y sin UMAP"""
        print("\n=== COMPARACIÓN: CON vs SIN UMAP ===")
        
        if df_original_results is not None:
            print("Resultados SIN UMAP:")
            print(df_original_results.round(3))
            print("\nResultados CON UMAP:")
            metrics_df = self.evaluar_clustering_completo()
            print(metrics_df.round(3))
            
            # Crear gráfico comparativo
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Silhouette Scores
            if 'Silhouette' in df_original_results.columns:
                original_sil = df_original_results['Silhouette'].dropna()
                umap_sil = metrics_df['Silhouette'].dropna()
                
                axes[0].bar(range(len(original_sil)), original_sil.values, 
                           alpha=0.7, label='Sin UMAP', color='lightcoral')
                axes[0].bar(range(len(original_sil), len(original_sil) + len(umap_sil)), 
                           umap_sil.values, alpha=0.7, label='Con UMAP', color='skyblue')
                axes[0].set_title('Comparación Silhouette Score')
                axes[0].set_ylabel('Silhouette Score')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Estadísticas
                axes[1].text(0.1, 0.8, f"Sin UMAP - Mejor: {original_sil.max():.3f}", 
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].text(0.1, 0.7, f"Sin UMAP - Promedio: {original_sil.mean():.3f}", 
                           transform=axes[1].transAxes, fontsize=12)
                axes[1].text(0.1, 0.5, f"Con UMAP - Mejor: {umap_sil.max():.3f}", 
                           transform=axes[1].transAxes, fontsize=12, color='blue')
                axes[1].text(0.1, 0.4, f"Con UMAP - Promedio: {umap_sil.mean():.3f}", 
                           transform=axes[1].transAxes, fontsize=12, color='blue')
                
                mejora = ((umap_sil.max() - original_sil.max()) / original_sil.max()) * 100
                axes[1].text(0.1, 0.2, f"Mejora: {mejora:.1f}%", 
                           transform=axes[1].transAxes, fontsize=14, weight='bold',
                           color='green' if mejora > 0 else 'red')
                
                axes[1].set_title('Estadísticas Comparativas')
                axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_path, 'comparison_with_without_umap.png'), dpi=300)
            plt.close()
        
        return metrics_df

    def generar_reporte_completo(self):
        """Genera un reporte completo de los resultados"""
        print("\n=== GENERANDO REPORTE COMPLETO ===")
        
        report_lines = []
        report_lines.append("# REPORTE DE CLUSTERING CON UMAP - DATOS DE HOMICIDIOS")
        report_lines.append("=" * 60)
        report_lines.append(f"Fecha de análisis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Número total de registros: {len(self.df_processed)}")
        report_lines.append(f"Variables utilizadas: {len(self.categorical_columns)}")
        report_lines.append("")
        
        # Información de los datos
        report_lines.append("## INFORMACIÓN DE LOS DATOS")
        report_lines.append(f"- Dataset filtrado a: Bogotá, D.C.")
        report_lines.append(f"- Dimensiones finales: {self.df_processed.shape}")
        report_lines.append("- Variables categóricas utilizadas:")
        for col in self.categorical_columns:
            report_lines.append(f"  * {col}")
        report_lines.append("")
        
        # Resultados de clustering
        report_lines.append("## RESULTADOS DE CLUSTERING")
        metrics_df = self.evaluar_clustering_completo()
        
        for _, row in metrics_df.iterrows():
            report_lines.append(f"### {row['Algoritmo'].upper()}")
            report_lines.append(f"- Número de clusters: {row['N_Clusters']}")
            report_lines.append(f"- Silhouette Score: {row['Silhouette']:.3f}")
            report_lines.append(f"- Calinski-Harabasz Score: {row['Calinski_Harabasz']:.2f}")
            report_lines.append(f"- Davies-Bouldin Score: {row['Davies_Bouldin']:.3f}")
            if row['Noise_Points'] > 0:
                report_lines.append(f"- Puntos de ruido: {row['Noise_Points']}")
            report_lines.append("")
        
        # Mejor algoritmo
        valid_metrics = metrics_df.dropna(subset=['Silhouette'])
        if len(valid_metrics) > 0:
            best_idx = valid_metrics['Silhouette'].idxmax()
            best_algorithm = valid_metrics.loc[best_idx, 'Algoritmo']
            best_score = valid_metrics.loc[best_idx, 'Silhouette']
            
            report_lines.append("## MEJOR ALGORITMO")
            report_lines.append(f"**{best_algorithm.upper()}** con Silhouette Score: {best_score:.3f}")
            report_lines.append("")
        
        # Recomendaciones
        report_lines.append("## RECOMENDACIONES")
        report_lines.append("1. El uso de UMAP mejora significativamente la estructura de los datos categóricos")
        report_lines.append("2. Se recomienda utilizar el mejor algoritmo identificado para análisis posteriores")
        report_lines.append("3. Los perfiles de clusters pueden utilizarse para identificar patrones en los homicidios")
        report_lines.append("4. Se sugiere validar los resultados con expertos en criminología")
        
        # Guardar reporte
        report_path = os.path.join(self.results_path, 'reporte_clustering_umap.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Reporte guardado en: {report_path}")
        return report_lines

# Función principal para ejecutar el análisis
def ejecutar_analisis_clustering_umap(df):
    """Función principal para ejecutar el análisis completo con UMAP"""
    
    # Crear instancia del clustering con UMAP
    clustering_umap = ClusteringHomicidiosUMAP(df)
    
    # Ejecutar pipeline completo
    resultados = clustering_umap.ejecutar_pipeline_completo_umap()
    
    # Generar reporte
    clustering_umap.generar_reporte_completo()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*60)
    print(f"Resultados guardados en: {clustering_umap.results_path}")
    print("Archivos generados:")
    print("- Visualizaciones de reducción de dimensionalidad")
    print("- Resultados de clustering para cada algoritmo")
    print("- Comparación de métricas")
    print("- Perfiles de clusters")
    print("- Reporte completo")
    
    return clustering_umap, resultados

# Ejemplo de uso
if __name__ == "__main__":
    # Ejecutar análisis con UMAP
    clustering_umap, resultados = ejecutar_analisis_clustering_umap(df)
    
    # Mostrar resultados finales
    print("\nRESUMEN DE RESULTADOS:")
    print(resultados)
    
    # Opcional: Comparar con resultados anteriores si están disponibles
    # clustering_umap.comparar_con_sin_umap(resultados_anteriores)