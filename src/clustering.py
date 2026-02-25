from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


def run_kmeans(X, n_clusters=5):
    model = KMeans(
        n_clusters=n_clusters,
        n_init=50,
        random_state=42
    )
    return model.fit_predict(X)


def run_agglo(X, n_clusters=5):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )
    return model.fit_predict(X)


def evaluate_clustering(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)