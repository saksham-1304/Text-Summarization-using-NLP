"""FastAPI Application for Text Summarization.

Provides REST API endpoints for:
  - Summarization inference (single & batch)
  - Training pipeline trigger
  - Health check and model info
  - Web UI for interactive summarization

Endpoints:
  GET  /           - Web UI
  GET  /health     - Health check
  GET  /info       - Model info
  POST /predict    - Single text summarization
  POST /predict/batch - Batch summarization
  GET  /train      - Trigger training pipeline
"""

import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from textSummarizer.logging import logger


# ============================================================
# Request/Response Models (Pydantic)
# ============================================================

class SummarizeRequest(BaseModel):
    """Request body for single summarization."""
    text: str = Field(
        ...,
        min_length=10,
        description="Input text (dialogue or article) to summarize",
        json_schema_extra={
            "example": "Amanda: Hey, are we meeting today?\nJerry: Sure! What time?\nAmanda: How about 3pm at the coffee shop?\nJerry: Sounds good, see you there!"
        },
    )


class SummarizeResponse(BaseModel):
    """Response body for single summarization."""
    input_text: str
    summary: str
    input_length: int
    summary_length: int
    timestamp: str


class BatchSummarizeRequest(BaseModel):
    """Request body for batch summarization."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of texts to summarize (max 10)",
    )


class BatchSummarizeResponse(BaseModel):
    """Response body for batch summarization."""
    results: List[SummarizeResponse]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class TrainResponse(BaseModel):
    """Training trigger response."""
    status: str
    message: str


# ============================================================
# Application Lifespan (model loading)
# ============================================================

prediction_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global prediction_pipeline
    try:
        from textSummarizer.pipeline.prediction import PredictionPipeline
        prediction_pipeline = PredictionPipeline()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.warning(f"Model not loaded at startup: {e}")
        logger.warning("Train the model first using /train or `python main.py`")
    yield
    logger.info("Application shutting down")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Text Summarization API",
    description=(
        "Dialogue summarization API powered by BART-large-CNN fine-tuned on SAMSum dataset. "
        "Supports single and batch summarization with a modern web UI."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================
# Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_pipeline is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/info", tags=["System"])
async def model_info():
    """Get model and dataset information."""
    return {
        "model": "facebook/bart-large-cnn",
        "dataset": "SAMSum (16k dialogue-summary pairs)",
        "task": "Dialogue Summarization",
        "metrics": "ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum",
        "framework": "HuggingFace Transformers",
        "api_version": "2.0.0",
    }


@app.post("/predict", response_model=SummarizeResponse, tags=["Summarization"])
async def predict(request: SummarizeRequest):
    """Generate a summary for a single input text.

    Args:
        request: SummarizeRequest with the input text.

    Returns:
        SummarizeResponse with the generated summary.
    """
    if prediction_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first using /train or `python main.py`",
        )

    try:
        summary = prediction_pipeline.predict(request.text)
        return SummarizeResponse(
            input_text=request.text,
            summary=summary,
            input_length=len(request.text),
            summary_length=len(summary),
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchSummarizeResponse, tags=["Summarization"])
async def predict_batch(request: BatchSummarizeRequest):
    """Generate summaries for a batch of input texts.

    Args:
        request: BatchSummarizeRequest with list of texts.

    Returns:
        BatchSummarizeResponse with list of results.
    """
    if prediction_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )

    try:
        summaries = prediction_pipeline.predict_batch(request.texts)
        timestamp = datetime.utcnow().isoformat()

        results = [
            SummarizeResponse(
                input_text=text,
                summary=summary,
                input_length=len(text),
                summary_length=len(summary),
                timestamp=timestamp,
            )
            for text, summary in zip(request.texts, summaries)
        ]

        return BatchSummarizeResponse(results=results, count=len(results))
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/train", response_model=TrainResponse, tags=["Training"])
async def training():
    """Trigger the full training pipeline (runs in subprocess).

    Note: This is a long-running operation. For production, use
    a task queue (Celery/Redis) instead.
    """
    try:
        logger.info("Training pipeline triggered via API")
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return TrainResponse(
            status="started",
            message=(
                f"Training started (PID: {process.pid}). "
                "Monitor progress in logs/running_logs.log"
            ),
        )
    except Exception as e:
        logger.exception(f"Training trigger error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1,
    )
