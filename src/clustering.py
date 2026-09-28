from sklearn.cluster import DBSCAN, KMeans


def build_kmeans(
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )

def fit_kmeans(X, **kwargs):
    model = build_kmeans(**kwargs)
    labels = model.fit_predict(X)
    return model, labels


def build_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> DBSCAN:
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric)


def build_model(algorithm: str, **kwargs):
    if algorithm == "kmeans":
        return build_kmeans(**kwargs)
    if algorithm == "dbscan":
        return build_dbscan(**kwargs)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
