"""
backend/schemas.py — Pydantic request/response models for the FastAPI app.
"""

from typing import List
from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v


class PredictResponse(BaseModel):
    prediction: str        # "FAKE" or "REAL"
    score: float           # probability (0-1) or raw decision score
    score_type: str        # "confidence" or "decision_score"
    model_used: str        # name of the best model


class ModelMetrics(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]


class DatasetStats(BaseModel):
    total: int
    fake_count: int
    real_count: int


class StatsResponse(BaseModel):
    dataset: DatasetStats
    models: List[ModelMetrics]
    best_model: str
