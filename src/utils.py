from pathlib import Path
import json
import joblib
import numpy as np

def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: dict, path: str | Path) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def save_model(model, path: str | Path) -> None:
    path = ensure_parent(path)
    joblib.dump(model, path)

def load_model(path: str | Path):
    return joblib.load(path)

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
