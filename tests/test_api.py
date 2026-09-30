import pytest
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

    response = main.predict(
        main.RFMInput(recency=30, frequency=5, monetary=100)
    ).model_dump()

    assert response["cluster_id"] == 2
    assert response["segment_name"] == "Recent / Frequent / Higher Spend"
    assert response["segments"][0]["median_rfm"]["monetary"] == 250.0


def test_predict_rejects_nonpositive_rfm_values():
    with pytest.raises(ValidationError):
        main.RFMInput(recency=0, frequency=5, monetary=100)
