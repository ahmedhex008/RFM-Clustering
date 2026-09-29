# RFM Customer Segmentation

An end-to-end, reproducible customer segmentation project using:

- Python + uv
- RFM features
- KMeans and DBSCAN clustering
- Hydra configuration
- DVC data and pipeline versioning
- MLflow experiment tracking
- DagsHub remote storage and MLflow
- Git + GitHub
- GitHub Actions CI

## RFM features

The clustering model uses:

- Recency
- Frequency
- Monetary

`Buyer` is treated as an identifier, not a model feature.

## Pipeline

```text
Final_File.xlsx
      |
      v
prepare
      |
      v
rfm_processed.csv
      |
      v
preprocess
      |
      v
X_processed.csv
      |
      v
train
      |
      v
    kmeans.pkl + dbscan.pkl
      |
      v
evaluate
      |
      +--> reports/kmeans/
      +--> reports/dbscan/
```

## Run

Use Python 3.11–3.13. Hydra 1.3 is not compatible with Python 3.14.

```bash
uv python install 3.13
uv sync --dev --python 3.13
uv run python pipeline/prepare.py
uv run python pipeline/preprocess.py
uv run python pipeline/train.py
uv run python pipeline/evaluate.py
```

The full pipeline trains and evaluates both KMeans and DBSCAN. Each algorithm
gets its own model file, reports directory, and MLflow runs. Adjust their
parameters in `configs/model.yaml` under `model.kmeans` and `model.dbscan`.

DBSCAN evaluation excludes noise points (`label=-1`) from clustering scores and
reports the number and ratio of noise points in `reports/dbscan/metrics.json`.
The MLflow training and evaluation runs also show `discovered_clusters`,
`noise_points`, and `noise_ratio`. These metrics appear only in new runs, so
rerun the commands after changing the pipeline.

Or run the complete DVC pipeline:

```bash
dvc repro
```

## Hydra experiments

```bash
uv run python pipeline/train.py model.kmeans.n_clusters=3
uv run python pipeline/train.py model.kmeans.n_clusters=4
uv run python pipeline/train.py preprocessing.scaler=robust
uv run python pipeline/train.py preprocessing.log_transform=false
```

## MLflow

Training and evaluation runs for both algorithms are logged to the DagsHub
MLflow server configured by `MLFLOW_TRACKING_URI` and credentials in `.env`.
Never commit `.env`.

## DVC

After `dvc init`:

```bash
dvc add data/raw/Final_File.xlsx
dvc repro
dvc status
dvc push
```

Commit the `.dvc` metadata, `dvc.yaml`, and `dvc.lock` to Git.

## Notes

This project uses log1p transformation, quantile clipping, and StandardScaler by default because RFM variables can be strongly skewed. These choices are configuration-driven and should be validated through experiments rather than treated as universally optimal.
