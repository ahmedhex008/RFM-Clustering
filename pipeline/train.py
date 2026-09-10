import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

import hydra
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig

from src.clustering import build_kmeans
from src.utils import set_global_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_global_seed(cfg.seed)
    X = pd.read_csv(cfg.data.features_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model = build_kmeans(
        n_clusters=cfg.model.n_clusters,
        random_state=cfg.model.random_state,
        n_init=cfg.model.n_init,
        max_iter=cfg.model.max_iter,
    )

    with mlflow.start_run(run_name=f"kmeans-k{cfg.model.n_clusters}") as run:
        model.fit(X)

        mlflow.log_params({
            "algorithm": "kmeans",
            "n_clusters": cfg.model.n_clusters,
            "random_state": cfg.model.random_state,
            "n_init": cfg.model.n_init,
            "max_iter": cfg.model.max_iter,
            "features": ",".join(X.columns),
            "log_transform": cfg.preprocessing.log_transform,
            "scaler": cfg.preprocessing.scaler,
            "clip_quantiles": cfg.preprocessing.clip_quantiles,
        })
        mlflow.log_metric("inertia", float(model.inertia_))
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("dvc_stage", "train")

        model_path = Path(cfg.model.output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, name="kmeans_model")
        mlflow.log_artifact(str(model_path), artifact_path="local_model")
        run_id_path = Path(cfg.mlflow.run_id_path)
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run.info.run_id, encoding="utf-8")
        print(f"MLflow run: {run.info.run_id}")
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
