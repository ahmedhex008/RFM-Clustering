FROM python:3.13-slim

WORKDIR /app

# DVC requires a Git repository even when the container only runs the pipeline.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install uv from Python's package index, avoiding a dependency on GHCR.
RUN pip install --no-cache-dir uv

# Install the exact project dependencies recorded in uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Keep the DVC project metadata and pipeline code in the image. Data and
# generated artifacts are mounted from the host in compose.yaml.
COPY . .

# .git is excluded from the build context, so create a lightweight repository
# inside the image. Compose can still mount the host repository over this.
RUN git init --quiet /app

CMD ["uv", "run", "--frozen", "dvc", "repro"]
