import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

def run_kmeans(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    return model.fit_predict(X)

def run_dbscan(X, eps=0.07, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def run_agglo(X, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X)

def run_gmm(X, n_components=5):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )
    return model.fit_predict(X)

def evaluate_clustering(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)
