import pandas as pd
from src.data_preprocessing import clean_rfm, transform_rfm

def test_clean_rfm_removes_invalid_rows():
    df = pd.DataFrame({
        "Buyer": [1, 2, 2, 3],
        "Recency": [1, 2, -1, 3],
        "Frequency": [2, 3, 4, 5],
        "Monetary": [10, 20, 30, 40],
    })
    result = clean_rfm(df)
    assert len(result) == 2
    assert result["Buyer"].tolist() == [1, 2]

def test_transform_shape():
    df = pd.DataFrame({
        "Recency": [1, 2, 3],
        "Frequency": [2, 3, 4],
        "Monetary": [10, 20, 30],
    })
    X, scaler = transform_rfm(df)
    assert X.shape == (3, 3)
    assert list(X.columns) == ["Recency", "Frequency", "Monetary"]
