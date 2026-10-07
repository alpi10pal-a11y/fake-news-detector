"""
backend/model_loader.py — loads all heavy assets once at application startup.

Exposes module-level singletons consumed by the routers:
    model        — fitted best model (sklearn estimator)
    vectorizer   — fitted TfidfVectorizer
    report       — parsed evaluation_report.json dict
    dataset_stats — {"total": int, "fake_count": int, "real_count": int}
    model_name   — str, name of the best model

data/dataset.csv is optional: when absent (e.g. in deployment), dataset_stats
are derived from the confusion matrices stored in evaluation_report.json.
"""

import os
import json
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_MODEL_PATH      = os.path.join(ROOT, "ml", "models", "best_model.pkl")
_VECTORIZER_PATH = os.path.join(ROOT, "ml", "models", "tfidf_vectorizer.pkl")
_REPORT_PATH     = os.path.join(ROOT, "results", "evaluation_report.json")
_DATASET_PATH    = os.path.join(ROOT, "data", "dataset.csv")


def _stats_from_report(report: dict) -> dict:
    """Derive dataset counts from confusion matrices in the evaluation report.

    Each confusion matrix is [[TN, FP], [FN, TP]] where label order is
    FAKE=0, REAL=1.  We sum across all models' test sets then average, but
    since every model is evaluated on the same test split we just use the
    first model's matrix to avoid double-counting.
    """
    cm = report["models"][0]["confusion_matrix"]
    # cm[0] = [true_fake_correct, fake_predicted_as_real]
    # cm[1] = [real_predicted_as_fake, true_real_correct]
    fake_count = cm[0][0] + cm[0][1]
    real_count = cm[1][0] + cm[1][1]
    return {
        "total":      fake_count + real_count,
        "fake_count": fake_count,
        "real_count": real_count,
    }


def _load():
    missing = [p for p in (_MODEL_PATH, _VECTORIZER_PATH)
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Required model files not found (run ml/train.py first):\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    _model      = joblib.load(_MODEL_PATH)
    _vectorizer = joblib.load(_VECTORIZER_PATH)

    if os.path.exists(_REPORT_PATH):
        with open(_REPORT_PATH) as f:
            _report = json.load(f)
    else:
        _report = {"models": [], "best_model": "unknown"}

    if os.path.exists(_DATASET_PATH):
        import pandas as pd
        df = pd.read_csv(_DATASET_PATH, usecols=["label"])
        counts = df["label"].value_counts()
        _stats = {
            "total":      int(len(df)),
            "fake_count": int(counts.get("FAKE", 0)),
            "real_count": int(counts.get("REAL", 0)),
        }
    elif _report["models"]:
        _stats = _stats_from_report(_report)
    else:
        _stats = {"total": 0, "fake_count": 0, "real_count": 0}

    return _model, _vectorizer, _report, _stats, _report["best_model"]


model, vectorizer, report, dataset_stats, model_name = _load()
