import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_preprocessing import clean_rfm, load_data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    df = load_data(cfg.data.raw_path)
    clean = clean_rfm(df)
    output = Path(cfg.data.processed_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    print(f"Saved {len(clean)} customers to {output}")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
