import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import hydra
from omegaconf import DictConfig
from src.data_preprocessing import load_data, transform_rfm

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    df = load_data(cfg.data.processed_path)
    X, scaler = transform_rfm(
        df,
        log_transform=cfg.preprocessing.log_transform,
        clip_quantiles=cfg.preprocessing.clip_quantiles,
        lower_quantile=cfg.preprocessing.lower_quantile,
        upper_quantile=cfg.preprocessing.upper_quantile,
        scaler_name=cfg.preprocessing.scaler,
    )
    output = Path(cfg.data.features_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(output, index=False)

    scaler_path = Path(cfg.preprocessing.scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Saved features to {output}")
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    main()
