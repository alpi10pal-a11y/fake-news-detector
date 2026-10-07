"""
backend/model_loader.py — loads all heavy assets once at application startup.

Exposes module-level singletons consumed by the routers:
    model        — fitted best model (sklearn estimator)
    vectorizer   — fitted TfidfVectorizer
    report       — parsed evaluation_report.json dict
    dataset_stats — {"total": int, "fake_count": int, "real_count": int}
    model_name   — str, name of the best model
"""

import os
import json
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MODEL_PATH      = os.path.join(ROOT, "ml", "models", "best_model.pkl")
_VECTORIZER_PATH = os.path.join(ROOT, "ml", "models", "tfidf_vectorizer.pkl")
_REPORT_PATH     = os.path.join(ROOT, "results", "evaluation_report.json")
_DATASET_PATH    = os.path.join(ROOT, "data", "dataset.csv")


def _load():
    missing = [p for p in (_MODEL_PATH, _VECTORIZER_PATH, _REPORT_PATH, _DATASET_PATH)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Required files not found (run ml/train.py and data/prepare_dataset.py first):\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    _model      = joblib.load(_MODEL_PATH)
    _vectorizer = joblib.load(_VECTORIZER_PATH)

    with open(_REPORT_PATH) as f:
        _report = json.load(f)

    df = pd.read_csv(_DATASET_PATH, usecols=["label"])
    counts = df["label"].value_counts()
    _stats = {
        "total":      int(len(df)),
        "fake_count": int(counts.get("FAKE", 0)),
        "real_count": int(counts.get("REAL", 0)),
    }

    return _model, _vectorizer, _report, _stats, _report["best_model"]


model, vectorizer, report, dataset_stats, model_name = _load()
