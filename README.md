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

## CI with GitHub Actions

The workflow in `.github/workflows/ci.yaml` runs on pushes, pull requests, and
when started manually from the GitHub Actions tab. It checks Python 3.11, 3.12,
and 3.13 by installing the locked project dependencies, running the test suite,
reporting API and source coverage, and linting the application and test code
with Ruff. Coverage XML reports are attached to each Python-version run as
downloadable artifacts. A separate job builds both the API and Streamlit Docker
images to catch container build errors.

To use it, push this repository to GitHub and open the **Actions** tab. GitHub
runs the checks automatically for new pushes and pull requests; select **CI**
and **Run workflow** to start a manual run. A green result means the tests and
lint checks passed on all three Python versions and both images built
successfully. Download a `coverage-python-*` artifact from the run's summary to
inspect the XML coverage report. If a check fails, open its run and inspect
the failing job's logs.

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
the API container read-only, overriding the paths inside the API image.

Stop both containers with:

```bash
docker compose down
```

### Deploy the API and UI to local Kubernetes with Minikube

This manifest is for a local Minikube cluster. It uses the Docker images built
from this repository. The trained model and processed data are uploaded from
the project directory to Kubernetes Secrets and mounted into the API pod
read-only, so no long-running Minikube host mount is needed. Train the KMeans
model and generate the processed data before deploying.

Install Minikube and kubectl if needed, then open a new PowerShell window. Make
sure Docker Desktop is running and start the local cluster:

```powershell
minikube start --driver=docker
```

Build both images from the project root and load them into Minikube:

```powershell
docker build -f Dockerfile.api -t rfm-clustering-api:latest .
docker build -f Dockerfile.streamlit -t rfm-clustering-streamlit:latest .
minikube image load rfm-clustering-api:latest
minikube image load rfm-clustering-streamlit:latest
```

Apply the Kubernetes resources to create the namespace and workloads:

```powershell
kubectl apply -f k8s/minikube.yaml
```

Upload the prediction artifacts into namespace-scoped Secrets:

```powershell
kubectl create secret generic rfm-models --from-file=models/kmeans.pkl --from-file=models/scaler.pkl -n rfm --dry-run=client -o yaml | kubectl apply --server-side -f -
kubectl create secret generic rfm-processed-data --from-file=data/processed/rfm_processed.csv --from-file=data/processed/X_processed.csv -n rfm --dry-run=client -o yaml | kubectl apply --server-side -f -
```

Wait for the API and Streamlit deployments:

```powershell
kubectl rollout status deployment/api -n rfm
kubectl rollout status deployment/streamlit -n rfm
kubectl get pods,services -n rfm
```

Forward the Streamlit service to your computer and open `http://localhost:18501`:

```powershell
kubectl port-forward -n rfm service/streamlit 18501:8501
```

To check API health, port-forward its internal service in a separate terminal
and query its health endpoint:

```powershell
kubectl port-forward -n rfm service/api 18000:8000
Invoke-RestMethod http://localhost:18000/health
```

Inspect a failing pod with `kubectl logs -n rfm deployment/api` or
`kubectl describe pods -n rfm`. Remove the Kubernetes resources when finished:

```powershell
kubectl delete -f k8s/minikube.yaml
```

This setup is intended for local development. Kubernetes Secrets are not an
encrypted artifact store by default; for a remote cluster, use appropriately
secured secret management and persistent artifact storage.

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
