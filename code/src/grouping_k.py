
# 1. Carga de datos
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

class ClusteringHomicidios:
    def __init__(self, df, results_path="clustering_results"):
        self.df = df.copy()
        self.df_processed = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.clustering_results = {}
        self.results_path = results_path
        
        # Crear directorio para resultados si no existe
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        
        
    def save_results(self, algorithm_name, data):
        """Guarda los resultados en un archivo CSV o JSON"""
        file_path = os.path.join(self.results_path, f"{algorithm_name}_results.json")
        pd.DataFrame(data).to_json(file_path, orient="records")
        print(f"Resultados de {algorithm_name} guardados en: {file_path}")

    def preprocesamiento_clustering(self):
        """Preprocesamiento específico para clustering con datos categóricos"""
        print("=== PREPROCESAMIENTO PARA CLUSTERING ===")
        
        # 1. Seleccionar variables relevantes para clustering
        # Basado en tu análisis de Cramer's V y entropía
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
            'Ancestro Racial',
            'Municipio del hecho DANE',
            'Departamento del hecho DANE',
            'Actividad Durante el Hecho',
            'Diagnostico Topográfico de la Lesión',
            'Rango de Hora del Hecho X 3 Horas',
        ]
        
        # Filtrar solo las columnas que existen en el DataFrame
        self.categorical_columns = [col for col in self.categorical_columns if col in self.df.columns]
        
        print(f"Variables categóricas seleccionadas: {self.categorical_columns}")
        
        # 2. Crear dataset solo con variables seleccionadas
        self.df_processed = self.df[self.categorical_columns].copy()
        
        # 3. Limpiar datos faltantes y valores raros
        for col in self.categorical_columns:
            # Reemplazar valores con baja frecuencia por "Otros"
            value_counts = self.df_processed[col].value_counts()
            threshold = len(self.df_processed) * 0.01  # 1% del total
            rare_values = value_counts[value_counts < threshold].index
            self.df_processed[col] = self.df_processed[col].replace(rare_values, 'Otros')
            
            # Manejar valores faltantes
            self.df_processed[col] = self.df_processed[col].fillna('Sin información')
        
        # 4. Aplicar MCA para reducción de dimensionalidad
        print("\nAplicando MCA para reducción de dimensionalidad...")
        self.mca = prince.MCA(n_components=10, random_state=42)
        self.mca_coords = self.mca.fit_transform(self.df_processed)
        
        # 5. Codificación para algoritmos que requieren datos numéricos
        self.df_encoded = self.df_processed.copy()
        self.label_encoders = {}
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.df_encoded[col] = le.fit_transform(self.df_processed[col])
            self.label_encoders[col] = le
        
        print(f"Forma del dataset procesado: {self.df_processed.shape}")
        print(f"Coordenadas MCA shape: {self.mca_coords.shape}")
        
        return self.df_processed
    
    def algoritmo_1_kmodes(self, k_range=[2, 3, 4, 5, 6, 7, 8, 9, 10]):
        """K-Modes: Algoritmo específico para datos categóricos"""
        print("\n=== ALGORITMO 1: K-MODES ===")
        
        best_cost = np.inf
        best_k = None
        costs = []
        results = []

        for k in k_range:
            try:
                kmodes = KModes(n_clusters=k, init='Huang', n_init=10, verbose=0, random_state=42)
                clusters = kmodes.fit_predict(self.df_processed)
                cost = kmodes.cost_
                costs.append(cost)
                results.append({
                    'k': k,
                    'cost': cost,
                    'clusters': list(clusters),
                    'centroids': kmodes.cluster_centroids_.tolist()
                })
                
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
                    best_clusters = clusters
                    best_centroids = kmodes.cluster_centroids_
                
                print(f"K={k}, Cost={cost:.2f}")
                
            except Exception as e:
                print(f"Error con K={k}: {e}")
                costs.append(np.inf)
        
        
        #self.save_results("kmodes", results)
        
        # Guardar resultados
        self.clustering_results['kmodes'] = {
            'clusters': best_clusters,
            'k_optimal': best_k,
            'cost': best_cost,
            'centroids': best_centroids,
            'costs': costs,
            'k_range': k_range
        }
        
        
        # Visualizar método del codo
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, costs, 'bo-')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Cost (Within-cluster sum of dissimilarities)')
        plt.title('Método del Codo - K-Modes')
        plt.grid(True)
        # Guardar figura
        plt.savefig(os.path.join(self.results_path, 'kmodes_elbow_method.png'))
        #Cerrar figura
        plt.close()
        
        print(f"Mejor K: {best_k} con cost: {best_cost:.2f}")
        return best_clusters
    
    def algoritmo_2_kmeans_mca(self, k_range=[2,3, 4, 5, 6, 7, 8,9,10]):
        """K-Means aplicado a coordenadas MCA"""
        print("\n=== ALGORITMO 2: K-MEANS CON MCA ===")
        
        inertias = []
        silhouette_scores = []
        best_score = -1
        best_k = None
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(self.mca_coords)
            
            inertia = kmeans.inertia_
            sil_score = silhouette_score(self.mca_coords, clusters)
            
            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            
            if sil_score > best_score:
                best_score = sil_score
                best_k = k
                best_clusters = clusters
                best_centroids = kmeans.cluster_centers_
            
            print(f"K={k}, Inertia={inertia:.2f}, Silhouette={sil_score:.3f}")
        
        # Guardar resultados
        self.clustering_results['kmeans_mca'] = {
            'clusters': best_clusters,
            'k_optimal': best_k,
            'silhouette_score': best_score,
            'centroids': best_centroids,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'k_range': k_range
        }
        
        # Visualizaciones
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Método del codo
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Método del Codo - K-Means MCA')
        ax1.grid(True)
        
        # Silhouette score
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Número de Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score - K-Means MCA')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'kmeans_mca_results.png'))
        # Cerrar figura
        plt.close()
        # self.save_results("kmeans_mca", {
        #     'clusters': best_clusters,
        #     'k_optimal': best_k,
        #     'silhouette_score': best_score,
        #     'centroids': best_centroids,
        #     'inertias': inertias,
        #     'silhouette_scores': silhouette_scores,
        #     'k_range': k_range
        # })
        
        print(f"Mejor K: {best_k} con Silhouette Score: {best_score:.3f}")
        return best_clusters
    
    def algoritmo_3_gaussian_mixture(self, k_range=[2,3, 4, 5, 6, 7, 8,9,10]):
        """Gaussian Mixture Model aplicado a coordenadas MCA"""
        print("\n=== ALGORITMO 3: GAUSSIAN MIXTURE MODEL ===")
        
        bic_scores = []
        aic_scores = []
        best_bic = np.inf
        best_k = None
        
        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.mca_coords)
            clusters = gmm.predict(self.mca_coords)
            
            bic = gmm.bic(self.mca_coords)
            aic = gmm.aic(self.mca_coords)
            
            bic_scores.append(bic)
            aic_scores.append(aic)
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_clusters = clusters
                best_gmm = gmm
            
            print(f"K={k}, BIC={bic:.2f}, AIC={aic:.2f}")
        
        # Guardar resultados
        self.clustering_results['gmm'] = {
            'clusters': best_clusters,
            'k_optimal': best_k,
            'bic_score': best_bic,
            'model': best_gmm,
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'k_range': k_range
        }
        
        # Visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(k_range, bic_scores, 'go-', label='BIC')
        ax1.plot(k_range, aic_scores, 'bo-', label='AIC')
        ax1.set_xlabel('Número de Componentes')
        ax1.set_ylabel('Information Criterion')
        ax1.set_title('BIC y AIC - Gaussian Mixture Model')
        ax1.legend()
        ax1.grid(True)
        
        # Distribución de clusters
        ax2.hist(best_clusters, bins=best_k, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title(f'Distribución de Clusters (K={best_k})')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'gmm_results.png'))
        # Cerrar figura
        plt.close()
                
        # self.save_results("gmm", {
        #     'clusters': best_clusters,
        #     'k_optimal': best_k,
        #     'bic_score': best_bic,
        #     'model': best_gmm,
        #     'bic_scores': bic_scores,
        #     'aic_scores': aic_scores,
        #     'k_range': k_range
        # })
        
        print(f"Mejor K: {best_k} con BIC: {best_bic:.2f}")
        return best_clusters
    
    def algoritmo_4_dbscan_mca(self, eps_range=[0.5, 1.0, 1.5, 2.0, 2.5], min_samples_range=[5, 10, 15, 20]):
        """DBSCAN aplicado a coordenadas MCA"""
        print("\n=== ALGORITMO 4: DBSCAN ===")
        
        best_score = -1
        best_params = None
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(self.mca_coords)
                
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                if n_clusters > 1:  # Necesitamos al menos 2 clusters para calcular silhouette
                    # Calcular silhouette sin incluir puntos de ruido
                    mask = clusters != -1
                    if np.sum(mask) > 1:
                        sil_score = silhouette_score(self.mca_coords[mask], clusters[mask])
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
        self.clustering_results['dbscan'] = {
            'clusters': best_clusters if best_params else None,
            'best_params': best_params,
            'silhouette_score': best_score,
            'results': results
        }
        
        #self.save_results("dbscan", results)
        
        if best_params:
            print(f"Mejores parámetros: {best_params} con Silhouette Score: {best_score:.3f}")
            
            # Visualización de resultados
            plt.figure(figsize=(12, 8))
            
            # Crear heatmap de parámetros
            eps_vals = sorted(set([r['eps'] for r in results]))
            min_samp_vals = sorted(set([r['min_samples'] for r in results]))
            
            heatmap_data = np.zeros((len(min_samp_vals), len(eps_vals)))
            
            for r in results:
                i = min_samp_vals.index(r['min_samples'])
                j = eps_vals.index(r['eps'])
                heatmap_data[i, j] = r['silhouette'] if r['silhouette'] > 0 else 0
            
            sns.heatmap(heatmap_data, 
                       xticklabels=eps_vals, 
                       yticklabels=min_samp_vals,
                       annot=True, 
                       fmt='.3f', 
                       cmap='viridis')
            plt.xlabel('eps')
            plt.ylabel('min_samples')
            plt.title('DBSCAN: Silhouette Score por Parámetros')
            # Guardar figura
            plt.savefig(os.path.join(self.results_path, 'dbscan_heatmap.png'))
            # Cerrar figura
            plt.close()
            
            return best_clusters
        else:
            print("No se encontraron parámetros válidos para DBSCAN")
            return None
    
    def evaluar_clustering(self):
        """Evaluación comparativa de todos los algoritmos"""
        print("\n=== EVALUACIÓN COMPARATIVA ===")
        
        # Métricas internas
        metrics_summary = []
        
        for algorithm, results in self.clustering_results.items():
            if 'clusters' in results and results['clusters'] is not None:
                clusters = results['clusters']
                
                # Calcular métricas internas usando coordenadas MCA
                if algorithm == 'dbscan':
                    # Para DBSCAN, excluir puntos de ruido
                    mask = clusters != -1
                    if np.sum(mask) > 1 and len(set(clusters[mask])) > 1:
                        sil_score = silhouette_score(self.mca_coords[mask], clusters[mask])
                        ch_score = calinski_harabasz_score(self.mca_coords[mask], clusters[mask])
                        db_score = davies_bouldin_score(self.mca_coords[mask], clusters[mask])
                    else:
                        sil_score = ch_score = db_score = np.nan
                else:
                    if len(set(clusters)) > 1:
                        sil_score = silhouette_score(self.mca_coords, clusters)
                        ch_score = calinski_harabasz_score(self.mca_coords, clusters)
                        db_score = davies_bouldin_score(self.mca_coords, clusters)
                    else:
                        sil_score = ch_score = db_score = np.nan
                
                metrics_summary.append({
                    'Algoritmo': algorithm,
                    'N_Clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                    'Silhouette': sil_score,
                    'Calinski_Harabasz': ch_score,
                    'Davies_Bouldin': db_score
                })
        
        metrics_df = pd.DataFrame(metrics_summary)
        print("\nMétricas Internas:")
        print(metrics_df.round(3))
        
        # Visualización comparativa
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribución de clusters para cada algoritmo
        for i, (algorithm, results) in enumerate(self.clustering_results.items()):
            if 'clusters' in results and results['clusters'] is not None:
                ax = axes[i//2, i%2]
                clusters = results['clusters']
                unique_clusters = sorted(set(clusters))
                
                counts = [list(clusters).count(c) for c in unique_clusters]
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
                
                bars = ax.bar(range(len(unique_clusters)), counts, color=colors)
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Frecuencia')
                ax.set_title(f'{algorithm.upper()}: Distribución de Clusters')
                ax.set_xticks(range(len(unique_clusters)))
                ax.set_xticklabels(unique_clusters)
                
                # Añadir etiquetas de frecuencia
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                           str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        # Guardar figura
        plt.savefig(os.path.join(self.results_path, 'clustering_comparison.png'))
        # Cerrar figura
        plt.close()
        
        return metrics_df
    
    def analizar_perfiles_clusters(self, algorithm='kmodes'):
        """Análisis de perfiles de clusters para el mejor algoritmo"""
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
            print(f"\n--- {col} ---")
            cluster_profile = pd.crosstab(df_analysis['Cluster'], df_analysis[col], normalize='index')
            print(cluster_profile.round(3))
            
            # Visualización
            plt.figure(figsize=(12, 8))
            cluster_profile.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(f'Perfil de Clusters por {col}')
            plt.xlabel('Cluster')
            plt.ylabel('Proporción')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            # Guardar figura
            plt.savefig(os.path.join(self.results_path, f'cluster_profile_{col}.png'))
            # Cerrar figura
            plt.close()
        
        # Resumen de características principales por cluster
        print("\n=== RESUMEN DE PERFILES ===")
        for cluster_id in sorted(set(clusters)):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            print(f"\n--- CLUSTER {cluster_id} ---")
            cluster_data = df_analysis[df_analysis['Cluster'] == cluster_id]
            print(f"Tamaño: {len(cluster_data)} ({len(cluster_data)/len(df_analysis)*100:.1f}%)")
            
            # Top características de cada cluster
            for col in self.categorical_columns:
                mode_value = cluster_data[col].mode()
                if len(mode_value) > 0:
                    percentage = (cluster_data[col] == mode_value[0]).sum() / len(cluster_data) * 100
                    print(f"{col}: {mode_value[0]} ({percentage:.1f}%)")
    
    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de clustering"""
        print("INICIANDO PIPELINE DE CLUSTERING PARA DATOS DE HOMICIDIOS")
        print("="*60)
        
        # 1. Preprocesamiento
        self.preprocesamiento_clustering()
        
        # 2. Aplicar algoritmos
        self.algoritmo_1_kmodes()
        self.algoritmo_2_kmeans_mca()
        self.algoritmo_3_gaussian_mixture()
        #self.algoritmo_4_dbscan_mca()
        
        # 3. Evaluación comparativa
        metrics_df = self.evaluar_clustering()
        
        # 4. Análisis de perfiles del mejor algoritmo
        # Seleccionar el mejor basado en Silhouette Score
        best_algorithm = metrics_df.loc[metrics_df['Silhouette'].idxmax(), 'Algoritmo']
        print(f"\nMejor algoritmo según Silhouette Score: {best_algorithm}")
        
        self.analizar_perfiles_clusters(best_algorithm)
        
        return metrics_df

# Ejemplo de uso
clustering = ClusteringHomicidios(df)
resultados = clustering.ejecutar_pipeline_completo()
# import os
# import multiprocessing

# print(f"Núcleos (lógicos): {os.cpu_count()}")
# print(f"Núcleos (físicos): {multiprocessing.cpu_count()}")
