# Fake News Detector

An end-to-end machine learning application that classifies news articles as **FAKE** or **REAL**.

| Layer | Technology |
|---|---|
| ML pipeline | scikit-learn · TF-IDF · Random Forest |
| Backend API | FastAPI · Uvicorn · Pydantic v2 |
| Frontend | React 19 · Vite · Tailwind CSS v4 · Recharts · Axios |

---

## Project Structure

```
fake-news-detector/
├── data/
│   ├── prepare_dataset.py   # Sub-task 1 — merges Fake.csv + True.csv → dataset.csv
│   ├── Fake.csv             # Raw source (Kaggle, not committed)
│   ├── True.csv             # Raw source (Kaggle, not committed)
│   └── dataset.csv          # Merged output (generated)
├── ml/
│   ├── preprocess.py        # Shared text-cleaning utility (train + serve)
│   ├── train.py             # Sub-task 3 — trains 4 models, saves best
│   ├── evaluate.py          # Metric helpers
│   └── models/
│       ├── best_model.pkl        # Serialised best model (generated)
│       └── tfidf_vectorizer.pkl  # Fitted vectoriser (generated)
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── model_loader.py      # Loads all artefacts once at startup
│   ├── schemas.py           # Pydantic request/response models
│   └── routers/
│       ├── predict.py       # POST /predict
│       └── stats.py         # GET  /stats
├── frontend/                # React/Vite SPA
│   ├── src/
│   │   ├── api/             # Axios client + API functions
│   │   ├── components/      # Navbar
│   │   ├── hooks/           # useHealth (live API status)
│   │   └── pages/           # Detector, Dashboard
│   └── vite.config.js
└── results/
    └── evaluation_report.json
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- The [Kaggle dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (`Fake.csv` and `True.csv`) placed inside `data/`

> **Skip steps 1–2** if `dataset.csv`, `best_model.pkl`, and `tfidf_vectorizer.pkl` already exist — they are included in this repo.

---

### Step 1 — Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

---

### Step 2 — Prepare dataset *(skip if `data/dataset.csv` exists)*

```bash
python data/prepare_dataset.py
```

Downloads `Fake.csv` + `True.csv`, merges them, shuffles, and writes `data/dataset.csv` (~44 900 rows).

---

### Step 3 — Train models *(skip if `ml/models/*.pkl` exist)*

```bash
python ml/train.py
```

Trains four classifiers, picks the best by weighted F1, and saves:
- `ml/models/best_model.pkl`
- `ml/models/tfidf_vectorizer.pkl`
- `results/evaluation_report.json`

Training takes ~2–5 minutes depending on hardware.

---

### Step 4 — Run the backend

```bash
uvicorn backend.main:app --reload
# → http://localhost:8001
# → http://localhost:8001/docs  (Swagger UI)
```

---

### Step 5 — Run the frontend

```bash
cd frontend
npm install        # first time only
npm run dev
# → http://localhost:5173
```

The Vite dev server proxies `/predict`, `/stats`, and `/health` to `http://localhost:8001`.

---

## API Endpoints

### `GET /health`

Returns `{"status": "ok"}` when the server and models are loaded.

---

### `POST /predict`

Classify a news article.

**Request**
```json
{ "text": "Your news article or headline here" }
```

**Response**
```json
{
  "prediction": "FAKE",
  "score": 0.82,
  "score_type": "confidence",
  "model_used": "Random Forest"
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | `"FAKE"` \| `"REAL"` | Classification result |
| `score` | float | Probability (0–1) for models with `predict_proba`; absolute decision value for LinearSVC |
| `score_type` | `"confidence"` \| `"decision_score"` | Indicates what `score` means |
| `model_used` | string | Name of the best model selected at training time |

Returns **422** if `text` is empty or contains only whitespace/stopwords.

---

### `GET /stats`

Returns dataset counts and evaluation metrics for all trained models.

**Response**
```json
{
  "dataset": { "total": 44898, "fake_count": 23481, "real_count": 21417 },
  "best_model": "Random Forest",
  "models": [
    {
      "name": "Logistic Regression",
      "accuracy": 0.9903, "precision": 0.9903, "recall": 0.9903, "f1": 0.9903,
      "confusion_matrix": [[4643, 53], [34, 4250]]
    },
    ...
  ]
}
```

---

## ML Approach

### Dataset

- Source: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- 23 481 fake articles + 21 417 real articles = **44 898 total** after cleaning
- `title` + `text` columns concatenated; rows with empty text after cleaning removed
- Shuffled with seed 42 for reproducibility

### Text Preprocessing ([`ml/preprocess.py`](ml/preprocess.py))

1. Lowercase
2. Strip all non-alphabetic characters (digits, punctuation, symbols)
3. Tokenise on whitespace
4. Remove English stopwords (NLTK)
5. Rejoin tokens

The **same** `clean_text()` function is used at both training time and inference time.

### Feature Extraction

TF-IDF vectoriser with:
- `max_features = 50 000`
- `ngram_range = (1, 2)` — unigrams and bigrams
- Fitted on the **training split only** to prevent data leakage

### Models Trained

| Model | Weighted F1 |
|---|---|
| **Random Forest** ✅ | **0.9968** |
| LinearSVC | 0.9967 |
| Logistic Regression | 0.9903 |
| Multinomial Naive Bayes | 0.9597 |

Train/test split: 80% / 20%, stratified, seed 42.

The **Random Forest** (100 estimators) achieved the highest weighted F1 and is served by the API.

---

## Evaluation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | 99.68% | 99.68% | 99.68% | 99.68% |
| LinearSVC | 99.67% | 99.67% | 99.67% | 99.67% |
| Logistic Regression | 99.03% | 99.03% | 99.03% | 99.03% |
| Multinomial Naive Bayes | 95.97% | 95.97% | 95.97% | 95.97% |

**Best model confusion matrix** (Random Forest, test set — 8 980 samples):

|  | Predicted REAL | Predicted FAKE |
|---|---|---|
| **Actual REAL** | 4 673 (TN) | 23 (FP) |
| **Actual FAKE** | 6 (FN) | 4 278 (TP) |

Full metrics are available at `GET /stats` and visualised on the Dashboard page.

---

## Limitations

- **Dataset bias**: Both source CSVs are from a single Kaggle dataset collected around 2016–2017 US politics. The model has strong positional and topical biases from this era and source.
- **Domain shift**: Articles from other topics, time periods, or writing styles may produce unreliable predictions.
- **Text-only features**: The model uses bag-of-words (TF-IDF) and cannot understand semantics, context, or named entities beyond surface token patterns.
- **Short inputs**: Very short headlines or single sentences contain few distinguishing tokens after stopword removal; confidence scores may be misleading.
- **No threshold calibration**: The 50% decision threshold is used as-is. In production, threshold tuning against a held-out validation set would be advisable.
- **Not production-ready**: No authentication, rate limiting, logging, or monitoring is implemented.

---

## Frontend Pages

### Detector (`/`)

- Paste any article or headline
- Live word count
- Example buttons (pre-filled fake / real samples)
- Colour-coded result badge with confidence bar
- Graceful error display when the API is unreachable

### Dashboard (`/dashboard`)

- Dataset statistics (total, fake count, real count)
- Bar chart comparing all four models across Accuracy, Precision, Recall, and F1
- Radar chart profile for the best model
- Confusion matrix with TN / FP / FN / TP breakdown
- Full sortable metrics table with best-model highlight

---

## Development

```bash
# Backend — auto-reload on file changes
uvicorn backend.main:app --reload

# Frontend — HMR dev server
cd frontend && npm run dev

# Frontend — production build
cd frontend && npm run build
```
