"""
backend/routers/stats.py — GET /stats endpoint.

Reads from pre-loaded evaluation_report.json and dataset counts.
No recomputation happens at request time.
"""

from fastapi import APIRouter
from backend.schemas import StatsResponse, DatasetStats, ModelMetrics
from backend import model_loader as ml

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
def stats():
    dataset = DatasetStats(**ml.dataset_stats)

    models = [
        ModelMetrics(
            name=m["name"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            confusion_matrix=m["confusion_matrix"],
        )
        for m in ml.report["models"]
    ]

    return StatsResponse(
        dataset=dataset,
        models=models,
        best_model=ml.model_name,
    )
