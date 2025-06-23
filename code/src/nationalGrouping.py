import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DATA_PATH

class NationalGrouping:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_selected = None
        self.selected_features = [
            'Grupo de edad de la victima',
            'Sexo de la victima',
            'Escolaridad',
            'Mecanismo Causal',
            'Diagnostico Topográfico de la Lesión',
            'Presunto Agresor',
            'Ancestro Racial',
            'Zona del Hecho',
            'Escenario del Hecho',
            'Departamento del hecho DANE',
            'Mes del hecho',
            'Año del hecho'
        ]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()

    def load_and_balance_data(self):
        self.df = pd.read_csv(self.data_path)
        hombres = self.df[self.df['Sexo de la victima'] == 'Hombre']
        mujeres = self.df[self.df['Sexo de la victima'] == 'Mujer']
        n_mujeres = len(mujeres)
        n_hombres = n_mujeres + 3000
        hombres_sample = hombres.sample(n=n_hombres, random_state=42)
        print("Mujeres:", n_mujeres)
        print("Hombres seleccionados:", len(hombres_sample))
        grupo_balanceado = pd.concat([mujeres, hombres_sample]).reset_index(drop=True)
        self.df_selected = grupo_balanceado[self.selected_features].copy()

    def preprocess(self):
        X_encoded = self.encoder.fit_transform(self.df_selected)
        X_scaled = self.scaler.fit_transform(X_encoded)
        return X_scaled

    def cluster_dbscan(self, X_scaled, eps=90, min_samples=2):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        self.df_selected['cluster_dbscan'] = labels
        return labels

    def cluster_gmm(self, X_scaled, n_components=4):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(X_scaled)
        self.df_selected['cluster_gmm'] = labels
        return labels

    def visualize_clusters(self, X_scaled, labels, method_name="DBSCAN"):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2')
        plt.title(f"Visualización de Clusters con PCA ({method_name})")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def silhouette(self, X_scaled, labels):
        labels_unique = np.unique(labels)
        labels_without_noise = labels_unique[labels_unique != -1]
        if len(labels_without_noise) > 1:
            silhouette_avg = silhouette_score(X_scaled, labels)
            print(f"Coeficiente de silueta promedio: {silhouette_avg:.3f}")
        else:
            print("No se puede calcular el coeficiente de silueta: menos de 2 clusters detectados (sin contar el ruido).")

    def cluster_distribution(self, cluster_col):
        for col in self.selected_features:
            print(self.df_selected.groupby(cluster_col)[col].value_counts(normalize=True))

if __name__ == "__main__":
    ng = NationalGrouping(PROCESSED_DATA_PATH)
    ng.load_and_balance_data()
    X_scaled = ng.preprocess()

    # DBSCAN
    print("\n--- DBSCAN ---")
    dbscan_labels = ng.cluster_dbscan(X_scaled)
    ng.visualize_clusters(X_scaled, dbscan_labels, method_name="DBSCAN")
    ng.silhouette(X_scaled, dbscan_labels)
    ng.cluster_distribution('cluster_dbscan')

    # GMM
    print("\n--- Gaussian Mixture Model ---")
    gmm_labels = ng.cluster_gmm(X_scaled)
    ng.visualize_clusters(X_scaled, gmm_labels, method_name="GMM")
    ng.silhouette(X_scaled, gmm_labels)
    ng.cluster_distribution('cluster_gmm')
    