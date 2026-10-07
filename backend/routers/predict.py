"""
backend/routers/predict.py — POST /predict endpoint.
"""

import sys
import os

from fastapi import APIRouter, HTTPException

# Ensure project root is on path so ml package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.preprocess import clean_text
from backend.schemas import PredictRequest, PredictResponse
from backend import model_loader as ml

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    cleaned = clean_text(request.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text contains no usable content after cleaning.")

    X = ml.vectorizer.transform([cleaned])
    label = ml.model.predict(X)[0]

    if hasattr(ml.model, "predict_proba"):
        proba = ml.model.predict_proba(X)[0]
        score = float(max(proba))
        score_type = "confidence"
    else:
        # LinearSVC: use absolute decision function value
        decision = ml.model.decision_function(X)[0]
        score = float(abs(decision))
        score_type = "decision_score"

    return PredictResponse(
        prediction=label,
        score=round(score, 4),
        score_type=score_type,
        model_used=ml.model_name,
    )
