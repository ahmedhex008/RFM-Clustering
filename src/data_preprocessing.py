from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

RFM_FEATURES = ["Recency", "Frequency", "Monetary"]

def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return df

def validate_rfm_columns(df: pd.DataFrame) -> None:
    required = ["Buyer", *RFM_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def clean_rfm(df):
    """
    Remove rows with invalid RFM values (negative or zero values).
    
    Args:
        df: DataFrame with RFM columns
        
    Returns:
        Cleaned DataFrame with only valid rows
    """
    # Remove rows where any RFM metric is negative or zero
    df_clean = df[(df["Recency"] > 0) & (df["Frequency"] > 0) & (df["Monetary"] > 0)].copy()
    return df_clean

def transform_rfm(
    df: pd.DataFrame,
    log_transform: bool = True,
    clip_quantiles: bool = True,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    scaler_name: str = "standard",
):
    X = df[RFM_FEATURES].copy()

    if clip_quantiles:
        for col in RFM_FEATURES:
            lo = X[col].quantile(lower_quantile)
            hi = X[col].quantile(upper_quantile)
            X[col] = X[col].clip(lo, hi)

    if log_transform:
        X = np.log1p(X)

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "none":
        scaler = None
    else:
        raise ValueError("scaler_name must be standard, robust, or none")

    if scaler is not None:
        values = scaler.fit_transform(X)
    else:
        values = X.to_numpy()

    X_scaled = pd.DataFrame(values, columns=RFM_FEATURES, index=df.index)
    return X_scaled, scaler
