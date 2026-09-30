from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data_preprocessing import RFM_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CustomerSegmentPredictor:
    def __init__(self) -> None:
        model_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "model.yaml").model.kmeans
        preprocessing_cfg = OmegaConf.load(
            PROJECT_ROOT / "configs" / "preprocessing.yaml"
        ).preprocessing

        model_path = PROJECT_ROOT / model_cfg.output_path
        scaler_path = PROJECT_ROOT / preprocessing_cfg.scaler_path
        rfm_path = PROJECT_ROOT / "data" / "processed" / "rfm_processed.csv"
        features_path = PROJECT_ROOT / "data" / "processed" / "X_processed.csv"
        for path in (model_path, scaler_path, rfm_path, features_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required prediction artifact is missing: {path}. "
                    "Run the prepare, preprocess, and train pipeline first."
                )

        self.model: KMeans = joblib.load(model_path)
        self.scaler: StandardScaler | RobustScaler | None = joblib.load(scaler_path)
        self.log_transform = bool(preprocessing_cfg.log_transform)
        self.clip_quantiles = bool(preprocessing_cfg.clip_quantiles)
        self.lower_quantile = float(preprocessing_cfg.lower_quantile)
        self.upper_quantile = float(preprocessing_cfg.upper_quantile)
        self.training_data = pd.read_csv(rfm_path)[RFM_FEATURES]
        training_features = pd.read_csv(features_path)[RFM_FEATURES]

        if len(self.training_data) != len(training_features):
            raise ValueError(
                "Processed RFM data and model features have different row counts. "
                "Rerun the preprocessing and training pipeline."
            )

        training_labels = self.model.predict(training_features)
        self.segments = self._build_segments(training_labels)

    def _transform(self, values: pd.DataFrame) -> pd.DataFrame:
        transformed = values[RFM_FEATURES].copy()
        if self.clip_quantiles:
            for feature in RFM_FEATURES:
                lower = self.training_data[feature].quantile(self.lower_quantile)
                upper = self.training_data[feature].quantile(self.upper_quantile)
                transformed[feature] = transformed[feature].clip(lower, upper)

        if self.log_transform:
            transformed = pd.DataFrame(
                np.log1p(transformed.to_numpy()),
                columns=RFM_FEATURES,
                index=transformed.index,
            )

        if self.scaler is not None:
            transformed = pd.DataFrame(
                self.scaler.transform(transformed),
                columns=RFM_FEATURES,
                index=transformed.index,
            )
        return transformed

    def _build_segments(self, labels: np.ndarray) -> list[dict[str, Any]]:
        population_medians = self.training_data.median()
        segments = []
        labeled_data = self.training_data.copy()
        labeled_data["cluster_id"] = labels

        for cluster_id, group in labeled_data.groupby("cluster_id", sort=True):
            medians = group[RFM_FEATURES].median()
            profile = {
                feature.lower(): round(float(medians[feature]), 2)
                for feature in RFM_FEATURES
            }
            segment_name, description = self._describe_segment(
                medians, population_medians
            )
            segments.append({
                "cluster_id": int(cluster_id),
                "customer_count": len(group),
                "segment_name": segment_name,
                "description": description,
                "median_rfm": profile,
            })
        return segments

    @staticmethod
    def _describe_segment(
        medians: pd.Series, population_medians: pd.Series
    ) -> tuple[str, str]:
        recency_ratio = medians["Recency"] / population_medians["Recency"]
        frequency_ratio = medians["Frequency"] / population_medians["Frequency"]
        monetary_ratio = medians["Monetary"] / population_medians["Monetary"]

        recency, recency_description = (
            ("Recent", "more recently active")
            if recency_ratio < 0.85
            else (
                ("Less Recent", "less recently active")
                if recency_ratio > 1.15
                else ("Average Recency", "around average recency")
            )
        )
        frequency, frequency_description = (
            ("Frequent", "purchase more often")
            if frequency_ratio > 1.15
            else (
                ("Less Frequent", "purchase less often")
                if frequency_ratio < 0.85
                else ("Average Frequency", "purchase at a similar frequency")
            )
        )
        spend, spend_description = (
            ("Higher Spend", "spend more")
            if monetary_ratio > 1.15
            else (
                ("Lower Spend", "spend less")
                if monetary_ratio < 0.85
                else ("Average Spend", "spend a similar amount")
            )
        )
        name = f"{recency} / {frequency} / {spend}"
        description = (
            "Compared with the overall customer median, this group is "
            f"{recency_description}, {frequency_description}, and "
            f"{spend_description}."
        )
        return name, description

    def predict(self, values: Mapping[str, float]) -> dict[str, Any]:
        input_data = pd.DataFrame(
            [{feature: values[feature.lower()] for feature in RFM_FEATURES}]
        )
        label = int(self.model.predict(self._transform(input_data))[0])
        segment = next(
            (item for item in self.segments if item["cluster_id"] == label), None
        )
        if segment is None:
            raise ValueError(f"Model predicted cluster {label}, which has no profile.")
        return {
            "input": {key.lower(): float(values[key.lower()]) for key in RFM_FEATURES},
            "cluster_id": label,
            "segment_name": segment["segment_name"],
            "description": segment["description"],
            "segments": self.segments,
        }
