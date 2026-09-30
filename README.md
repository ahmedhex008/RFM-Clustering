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
  kmeans.pkl (default)
    |
    v
evaluate KMeans
    |
    +--> reports/kmeans/
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

KMeans is the default algorithm for training, evaluation, and `dvc repro`.
DBSCAN remains configured and available for experiments; it is not removed.
Select DBSCAN explicitly for either step:

```bash
uv run python pipeline/train.py model.algorithm=dbscan
uv run python pipeline/evaluate.py model.algorithm=dbscan
```

Each algorithm uses its own model file, reports directory, and MLflow runs.
Adjust their parameters in `configs/model.yaml` under `model.kmeans` and
`model.dbscan`.

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

## Prediction API and Streamlit UI

The API predicts the KMeans cluster for one customer's positive Recency,
Frequency, and Monetary values. It also returns each trained cluster's size,
median RFM profile, and a descriptive segment label.

Install the project dependencies, then start the API:

```bash
uv sync --dev
uv run uvicorn api.main:app --reload
```

In another terminal, start the Streamlit app:

```bash
uv run streamlit run streamlit_app.py
```

The form submits to the API's `/predict` endpoint. The API's interactive schema
is available at `http://127.0.0.1:8000/docs`. Ensure the prepare, preprocess,
and KMeans train stages have produced their model and data artifacts before
starting prediction.

### Run the API and UI with Docker Compose

The API and Streamlit use separate images. After training the KMeans model and
creating its preprocessing artifacts on the host, start both services with:

```bash
docker compose up --build api streamlit
```

Open the app at `http://localhost:8501`. The API is available at
`http://localhost:8000` and its documentation at `http://localhost:8000/docs`.
Compose connects Streamlit to the API using the internal `http://api:8000`
address and mounts the host's `models/` and `data/processed/` directories into
the API container read-only. These artifacts are not baked into either image.

Stop both containers with:

```bash
docker compose down
```

### Deploy the API and UI with Docker Swarm

Swarm stack deployment uses prebuilt images (it does not build Dockerfiles).
Train the model first, then build the same tagged images used by Compose:

```bash
docker compose build api streamlit
```

If Swarm is not already active on this machine, initialize it:

```bash
docker swarm init
```

The API needs the trained model and processed data on its node. Label the
manager node that has this project's `models/` and `data/processed/` folders,
then deploy the stack from the project root:

```powershell
$nodeId = docker info --format '{{.Swarm.NodeID}}'
docker node update --label-add rfm-artifacts=true $nodeId
docker stack deploy --compose-file swarm-stack.yaml rfm
```

Open the app at `http://localhost:8501`; the API is available at
`http://localhost:8000/docs`. Check service rollout with
`docker stack services rfm` and `docker stack ps rfm`. To remove the Swarm
services, run `docker stack rm rfm`.

The stack file uses bind mounts for trained artifacts, so API tasks are
constrained to nodes labeled `rfm-artifacts=true`. On a multi-node cluster,
copy the same `models/` and `data/processed/` artifacts to every node with
that label. Push the built images to a registry and update the image names in
`swarm-stack.yaml` before deploying to nodes that cannot access the local image
store. Stop the Compose services with `docker compose down` first if they are
already using ports 8000 or 8501.

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"recency\":30,\"frequency\":5,\"monetary\":100}"
```

## MLflow

Training and evaluation runs are logged to the DagsHub MLflow server configured
by `MLFLOW_TRACKING_URI` and credentials in `.env`. KMeans runs by default;
select `model.algorithm=dbscan` to log DBSCAN experiments. Never commit `.env`.

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
