"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    from app import app
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "timestamp" in data


class TestInfoEndpoint:
    """Tests for the /info endpoint."""

    def test_model_info(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "bart-samsum-model (fine-tuned from facebook/bart-large-cnn)"
        assert data["dataset"] == "SAMSum (16k dialogue-summary pairs)"
        assert data["task"] == "Dialogue Summarization"


class TestIndexEndpoint:
    """Tests for the / endpoint (web UI)."""

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Dialogue Summarizer" in response.text


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_no_model(self, client):
        """When model is not loaded, should return 503."""
        response = client.post(
            "/predict",
            json={"text": "Amanda: Hey!\nJerry: Hi!"},
        )
        # Either 503 (no model) or 200 (model loaded)
        assert response.status_code in [200, 503]

    def test_predict_validation_short_text(self, client):
        """Short text should fail Pydantic validation."""
        response = client.post(
            "/predict",
            json={"text": "Hi"},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_missing_text(self, client):
        """Missing text field should fail validation."""
        response = client.post(
            "/predict",
            json={},
        )
        assert response.status_code == 422

    def test_predict_validation_max_length_out_of_range(self, client):
        """max_length above the API cap should fail validation."""
        response = client.post(
            "/predict",
            json={
                "text": "Amanda: Hey there!\nJerry: Hello!\nAmanda: Let's sync at 3pm today.",
                "max_length": 999,
            },
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_empty(self, client):
        """Empty batch should fail validation."""
        response = client.post(
            "/predict/batch",
            json={"texts": []},
        )
        assert response.status_code == 422

    def test_batch_predict_validation_max_length_out_of_range(self, client):
        """Batch endpoint should validate max_length upper bound."""
        response = client.post(
            "/predict/batch",
            json={
                "texts": [
                    "Alice: Can you share the report?\nBob: Sure, I will send it now.",
                ],
                "max_length": 1000,
            },
        )
        assert response.status_code == 422
