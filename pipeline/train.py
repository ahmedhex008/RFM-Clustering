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
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.clustering import build_model
from src.utils import set_global_seed

load_dotenv()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_global_seed(cfg.seed)
    X = pd.read_csv(cfg.data.features_path)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    algorithm = cfg.model.algorithm
    model_cfg = cfg.model[algorithm]
    model_kwargs = {
        key: model_cfg[key]
        for key in ("n_clusters", "random_state", "n_init", "max_iter", "eps", "min_samples", "metric")
        if key in model_cfg
    }
    model = build_model(algorithm, **model_kwargs)

    run_name = f"{algorithm}-k{model_cfg.n_clusters}" if algorithm == "kmeans" else (
        f"{algorithm}-eps{model_cfg.eps}-min{model_cfg.min_samples}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X)

        params = {
            "algorithm": algorithm,
            "features": ",".join(X.columns),
            "log_transform": cfg.preprocessing.log_transform,
            "scaler": cfg.preprocessing.scaler,
            "clip_quantiles": cfg.preprocessing.clip_quantiles,
        }
        params.update(model_kwargs)
        mlflow.log_params(params)
        if algorithm == "kmeans":
            mlflow.log_metric("inertia", float(model.inertia_))
            mlflow.log_metric("discovered_clusters", float(model.n_clusters))
        else:
            labels = model.labels_
            noise_points = int((labels == -1).sum())
            discovered_clusters = len(set(labels) - {-1})
            mlflow.log_metrics({
                "discovered_clusters": float(discovered_clusters),
                "noise_points": float(noise_points),
                "noise_ratio": float(noise_points / len(labels)),
            })
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("dvc_stage", "train")

        model_path = Path(model_cfg.output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, name=f"{algorithm}_model")
        mlflow.log_artifact(str(model_path), artifact_path="local_model")
        run_id_path = Path(cfg.mlflow.run_id_path)
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run.info.run_id, encoding="utf-8")
        print(f"MLflow run: {run.info.run_id}")
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
