from sklearn.cluster import KMeans


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
