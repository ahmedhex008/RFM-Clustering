import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os

import hydra
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from omegaconf import DictConfig

from src.evaluation import cluster_summary, clustering_metrics, inertia_for_k


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    X = pd.read_csv(cfg.data.features_path)
    original = pd.read_csv(cfg.data.processed_path)
    model_cfg = cfg.model[cfg.model.algorithm]
    model = joblib.load(model_cfg.output_path)
    if hasattr(model, "predict"):
        labels = model.predict(X)
    elif hasattr(model, "labels_"):
        labels = model.labels_
    else:
        raise ValueError("Loaded clustering model cannot produce labels.")

    metrics = clustering_metrics(X, labels)
    summary = cluster_summary(original, labels)

    reports = Path(cfg.reports.dir)
    figures = Path(cfg.reports.figures_dir)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary.to_csv(reports / "cluster_summary.csv")

    if cfg.model.algorithm == "kmeans":
        ks = list(range(cfg.evaluation.min_k, cfg.evaluation.max_k + 1))
        inertias = [inertia_for_k(X, k, model_cfg.random_state, model_cfg.n_init) for k in ks]
        plt.figure(figsize=(8, 5))
        plt.plot(ks, inertias, marker="o")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Inertia")
        plt.title("Elbow Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "elbow_curve.png", dpi=150)
        plt.close()

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        silhouettes = []
        for k in ks:
            m = KMeans(n_clusters=k, random_state=model_cfg.random_state, n_init=model_cfg.n_init)
            l = m.fit_predict(X)
            silhouettes.append(silhouette_score(X, l))
        plt.figure(figsize=(8, 5))
        plt.plot(ks, silhouettes, marker="o")
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette Score by K")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures / "silhouette_scores.png", dpi=150)
        plt.close()

    # Cluster visualization using first two RFM dimensions
    plt.figure(figsize=(8, 5))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, alpha=0.55)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("Customer Clusters")
    plt.tight_layout()
    cluster_path = figures / "clusters.png"
    plt.savefig(cluster_path, dpi=150)
    plt.close()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"evaluation-{cfg.model.algorithm}") as run:
        mlflow.log_params({
            "algorithm": cfg.model.algorithm,
            "evaluation_min_k": cfg.evaluation.min_k,
            "evaluation_max_k": cfg.evaluation.max_k,
        })
        if cfg.model.algorithm == "kmeans":
            mlflow.log_param("n_clusters", model_cfg.n_clusters)
        mlflow.log_metrics({
            "silhouette_score": metrics["silhouette_score"],
            "davies_bouldin_score": metrics["davies_bouldin_score"],
            "calinski_harabasz_score": metrics["calinski_harabasz_score"],
            "discovered_clusters": float(metrics["n_clusters"]),
            "noise_points": float(metrics["noise_points"]),
            "noise_ratio": metrics["noise_ratio"],
        })
        mlflow.set_tag("stage", "evaluation")
        mlflow.log_artifacts(str(figures), artifact_path="figures")
        mlflow.log_artifact(str(reports / "metrics.json"), artifact_path="reports")
        mlflow.log_artifact(str(reports / "cluster_summary.csv"), artifact_path="reports")
        print(f"Evaluation MLflow run: {run.info.run_id}")

    print(json.dumps(metrics, indent=2))
    print(summary)

if __name__ == "__main__":
    main()
