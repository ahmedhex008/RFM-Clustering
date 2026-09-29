import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

load_dotenv()

def clustering_metrics(X, labels) -> dict:
    labels = np.asarray(labels)
    noise_points = int(np.sum(labels == -1))
    valid = labels != -1
    scored_X = X[valid] if hasattr(X, "iloc") else X[valid]
    scored_labels = labels[valid]
    unique = np.unique(scored_labels)
    scores = {
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }
    if 2 <= len(unique) < len(scored_labels):
        scores = {
            "silhouette_score": float(silhouette_score(scored_X, scored_labels)),
            "davies_bouldin_score": float(davies_bouldin_score(scored_X, scored_labels)),
            "calinski_harabasz_score": float(calinski_harabasz_score(scored_X, scored_labels)),
        }
    return {
        **scores,
        "n_clusters": len(unique),
        "noise_points": noise_points,
        "noise_ratio": float(noise_points / len(labels)),
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
