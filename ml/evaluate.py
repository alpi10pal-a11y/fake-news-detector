"""
ml/evaluate.py — model evaluation utilities.

Used by ml/train.py to compute per-model metrics after fitting.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a fitted sklearn model against the test set.

    Returns a dict with:
      accuracy, precision, recall, f1  — all floats rounded to 4 dp
      confusion_matrix                 — [[TN, FP], [FN, TP]] as nested lists
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision":        round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":               round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": cm.tolist(),
    }
