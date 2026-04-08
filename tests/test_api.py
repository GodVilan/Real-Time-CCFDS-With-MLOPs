"""
Unit tests for the Credit Card Fraud Detection API.

These tests validate request/response schemas and input validation
without requiring a live MLflow connection or trained model.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# --------------------------------------------------
# Mock MLflow before importing the app
# --------------------------------------------------

mock_model = MagicMock()
mock_model.feature_names_in_ = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
mock_model.predict_proba = MagicMock(
    return_value=np.array([[0.95, 0.05]])
)


@pytest.fixture
def client():
    """Create a test client with mocked MLflow dependencies."""
    import os
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("API_KEY", "test-api-key")

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.sklearn.load_model", return_value=mock_model), \
         patch("api.main.load_production_threshold", return_value=0.5), \
         patch("api.main.THRESHOLD", 0.5), \
         patch("api.main.model", mock_model):

        from fastapi.testclient import TestClient
        from api.main import app
        yield TestClient(app)


def test_health_check(client):
    """Health endpoint should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_missing_api_key(client):
    """Prediction without API key should return 403."""
    response = client.post(
        "/predict",
        json={"features": [0.0] * 30}
    )
    assert response.status_code == 403


def test_predict_wrong_feature_count(client):
    """Prediction with wrong number of features should return 400."""
    response = client.post(
        "/predict",
        json={"features": [0.0] * 15},
        headers={"X-API-Key": "test-api-key"}
    )
    assert response.status_code == 400


def test_predict_valid_request(client):
    """Valid prediction request should return fraud assessment."""
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

    response = client.post(
        "/predict",
        json={"features": [0.0] * 30},
        headers={"X-API-Key": "test-api-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "fraud" in data
    assert "fraud_probability" in data
    assert "fraud_probability_percent" in data
    assert "decision_threshold" in data
    assert data["fraud"] in [0, 1]


def test_predict_response_schema(client):
    """Verify the prediction response has the correct schema and types."""
    mock_model.predict_proba.return_value = np.array([[0.20, 0.80]])

    response = client.post(
        "/predict",
        json={"features": [0.0] * 30},
        headers={"X-API-Key": "test-api-key"}
    )
    data = response.json()
    assert isinstance(data["fraud"], int)
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["fraud_probability_percent"], float)
    assert isinstance(data["decision_threshold"], float)
