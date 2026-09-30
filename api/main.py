from functools import lru_cache
from math import isfinite

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.inference import CustomerSegmentPredictor

app = FastAPI(
    title="RFM Customer Segmentation API",
    description="Assign a customer to a trained KMeans RFM segment.",
    version="1.0.0",
)


class RFMInput(BaseModel):
    recency: float = Field(gt=0, description="Days since the customer's last purchase")
    frequency: float = Field(gt=0, description="Number of customer purchases")
    monetary: float = Field(gt=0, description="Customer monetary value")

    @field_validator("recency", "frequency", "monetary")
    @classmethod
    def must_be_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("RFM values must be finite numbers.")
        return value


class Segment(BaseModel):
    cluster_id: int
    customer_count: int
    segment_name: str
    description: str
    median_rfm: dict[str, float]


class Prediction(BaseModel):
    input: dict[str, float]
    cluster_id: int
    segment_name: str
    description: str
    segments: list[Segment]


@lru_cache(maxsize=1)
def get_predictor() -> CustomerSegmentPredictor:
    return CustomerSegmentPredictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: RFMInput) -> Prediction:
    return Prediction.model_validate(get_predictor().predict(customer.model_dump()))
