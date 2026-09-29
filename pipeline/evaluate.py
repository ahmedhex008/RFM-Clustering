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
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.evaluation import cluster_summary, clustering_metrics, inertia_for_k

load_dotenv()

ALGORITHMS = ("kmeans", "dbscan")


def save_kmeans_diagnostics(X, model_cfg, evaluation_cfg, figures: Path):
    ks = list(range(evaluation_cfg.min_k, evaluation_cfg.max_k + 1))
    inertias = [inertia_for_k(X, k, model_cfg.random_state, model_cfg.n_init) for k in ks]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "elbow_curve.png", dpi=150)
    plt.close()

    silhouettes = []
    for k in ks:
        labels = KMeans(
            n_clusters=k,
            random_state=model_cfg.random_state,
            n_init=model_cfg.n_init,
        ).fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
    plt.figure(figsize=(8, 5))
    plt.plot(ks, silhouettes, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("KMeans Silhouette Score by K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures / "silhouette_scores.png", dpi=150)
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    X = pd.read_csv(cfg.data.features_path)
    original = pd.read_csv(cfg.data.processed_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    reports_root = Path(cfg.reports.dir)
    reports_root.mkdir(parents=True, exist_ok=True)

    for algorithm in ALGORITHMS:
        model_cfg = cfg.model[algorithm]
        model = joblib.load(model_cfg.output_path)
        if hasattr(model, "predict"):
            labels = model.predict(X)
        elif hasattr(model, "labels_"):
            labels = model.labels_
        else:
            raise ValueError(f"Loaded {algorithm} model cannot produce labels.")

        metrics = clustering_metrics(X, labels)
        summary = cluster_summary(original, labels)

        algorithm_reports = reports_root / algorithm
        figures = algorithm_reports / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        metrics_path = algorithm_reports / "metrics.json"
        summary_path = algorithm_reports / "cluster_summary.csv"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.to_csv(summary_path)

        if algorithm == "kmeans":
            save_kmeans_diagnostics(X, model_cfg, cfg.evaluation, figures)

        plt.figure(figsize=(8, 5))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, alpha=0.55)
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title(f"{algorithm.upper()} Customer Clusters")
        plt.tight_layout()
        plt.savefig(figures / "clusters.png", dpi=150)
        plt.close()

        with mlflow.start_run(run_name=f"evaluation-{algorithm}") as run:
            mlflow.log_params({
                "algorithm": algorithm,
                "evaluation_min_k": cfg.evaluation.min_k,
                "evaluation_max_k": cfg.evaluation.max_k,
            })
            if algorithm == "kmeans":
                mlflow.log_param("n_clusters", model_cfg.n_clusters)

            numeric_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if value is not None
            }
            mlflow.log_metrics(numeric_metrics)
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("dvc_stage", "evaluate")
            mlflow.log_artifacts(str(figures), artifact_path="figures")
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")
            mlflow.log_artifact(str(summary_path), artifact_path="reports")
            print(f"{algorithm} evaluation MLflow run: {run.info.run_id}")

        print(f"{algorithm.upper()} metrics:")
        print(json.dumps(metrics, indent=2))
        print(summary)


if __name__ == "__main__":
    main()
