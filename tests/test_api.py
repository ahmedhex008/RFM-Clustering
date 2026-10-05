import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import main


class StubPredictor:
    def predict(self, values: dict[str, float]) -> dict:
        assert values == {"recency": 30.0, "frequency": 5.0, "monetary": 100.0}
        segment = {
            "cluster_id": 2,
            "customer_count": 12,
            "segment_name": "Recent / Frequent / Higher Spend",
            "description": "This is the predicted customer group.",
            "median_rfm": {"recency": 10.0, "frequency": 8.0, "monetary": 250.0},
        }
        return {
            "input": values,
            "cluster_id": segment["cluster_id"],
            "segment_name": segment["segment_name"],
            "description": segment["description"],
            "segments": [segment],
        }


def test_predict_returns_cluster_and_segment_profiles(monkeypatch):
    monkeypatch.setattr(main, "get_predictor", lambda: StubPredictor())

    response = TestClient(main.app).post(
        "/predict",
        json={"recency": 30, "frequency": 5, "monetary": 100},
    )

    assert response.status_code == 200
    assert response.json()["cluster_id"] == 2
    assert response.json()["segment_name"] == "Recent / Frequent / Higher Spend"
    assert response.json()["segments"][0]["median_rfm"]["monetary"] == 250.0


def test_health_endpoint_returns_ok():
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_rejects_nonpositive_rfm_values():
    response = TestClient(main.app).post(
        "/predict",
        json={"recency": 0, "frequency": 5, "monetary": 100},
    )

    assert response.status_code == 422


def test_predict_rejects_nonpositive_rfm_values():
    with pytest.raises(ValidationError):
        main.RFMInput(recency=0, frequency=5, monetary=100)
