"""
ml/train.py — full ML training pipeline.

Run from the fake-news-detector/ project root:
    python ml/train.py

Outputs:
    ml/models/best_model.pkl
    ml/models/tfidf_vectorizer.pkl
    results/evaluation_report.json
"""

import os
import sys
import json

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Ensure project root is on the path so `ml` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocess import clean_text
from ml.evaluate import evaluate_model

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data", "dataset.csv")
MODELS_DIR  = os.path.join(ROOT, "ml", "models")
RESULTS_DIR = os.path.join(ROOT, "results")

MODEL_OUT   = os.path.join(MODELS_DIR, "best_model.pkl")
TFIDF_OUT   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
REPORT_OUT  = os.path.join(RESULTS_DIR, "evaluation_report.json")

RANDOM_SEED = 42

# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
    ("LinearSVC",           LinearSVC(max_iter=2000, random_state=RANDOM_SEED)),
]


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        print("Run  python data/prepare_dataset.py  first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data()

    # ── 2. Preprocess text ────────────────────────────────────────────────────
    print("Cleaning text (this may take a minute)...")
    df["clean"] = df["text"].apply(clean_text)

    # ── 3. Train/test split (stratified) ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows\n")

    # ── 4. TF-IDF (fit on train only) ─────────────────────────────────────────
    print("Fitting TF-IDF vectorizer on training data...")
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n")

    # ── 5. Train & evaluate all models ────────────────────────────────────────
    results = []
    best_model_obj  = None
    best_model_name = None
    best_f1         = -1.0

    header = f"{'Model':<28}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}"
    print(header)
    print("-" * len(header))

    for name, model in MODELS:
        model.fit(X_train_tfidf, y_train)
        metrics = evaluate_model(model, X_test_tfidf, y_test)

        row = {
            "name":             name,
            "accuracy":         metrics["accuracy"],
            "precision":        metrics["precision"],
            "recall":           metrics["recall"],
            "f1":               metrics["f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        }
        results.append(row)

        print(
            f"{name:<28}  {metrics['accuracy']:>8.4f}  "
            f"{metrics['precision']:>9.4f}  {metrics['recall']:>6.4f}  {metrics['f1']:>6.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1         = metrics["f1"]
            best_model_name = name
            best_model_obj  = model

    print(f"\nBest model: {best_model_name}  (weighted F1 = {best_f1:.4f})")

    # ── 6. Save best model and vectorizer ─────────────────────────────────────
    joblib.dump(best_model_obj, MODEL_OUT)
    joblib.dump(vectorizer,     TFIDF_OUT)
    print(f"\nSaved model      : {MODEL_OUT}")
    print(f"Saved vectorizer : {TFIDF_OUT}")

    # ── 7. Write evaluation report ────────────────────────────────────────────
    report = {
        "models":     results,
        "best_model": best_model_name,
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report     : {REPORT_OUT}")
    print("\nSub-Task 3 complete.")


if __name__ == "__main__":
    main()
