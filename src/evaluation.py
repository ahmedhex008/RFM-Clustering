import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

load_dotenv()

def clustering_metrics(X, labels) -> dict:
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("At least 2 clusters are required for evaluation.")
    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
        "n_clusters": len(unique),
    }

def cluster_summary(original_df, labels):
    result = original_df.copy()
    result["Cluster"] = labels
    return (
        result.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .agg(["count", "mean", "median"])
        .round(3)
    )

def inertia_for_k(X, k: int, random_state: int = 42, n_init: int = 10):
    model = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans(
        n_clusters=k, random_state=random_state, n_init=n_init
    )
    model.fit(X)
    return float(model.inertia_)
