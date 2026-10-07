"""
backend/main.py — FastAPI application entry point.

Start the server from the fake-news-detector/ project root:
    uvicorn backend.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import predict, stats

app = FastAPI(
    title="Fake News Detector API",
    description="Classifies news articles as FAKE or REAL using a trained ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(stats.router)


@app.get("/health")
def health():
    return {"status": "ok"}
