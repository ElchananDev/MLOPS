"""בדיקות חוזה לשירות. הן מוודאות שהשירות מתנהג כמובטח, לא שהמודל טוב."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app

VALID_INPUT = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2,
}

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_prediction_contract(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}

    probabilities = list(body["probabilities"].values())
    assert len(probabilities) == 3
    assert all(0 <= value <= 1 for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
    assert body["model_version"]
    assert body["prediction_id"]


def test_invalid_measurement_is_rejected(client):
    invalid = {**VALID_INPUT, "petal_length_cm": -1}
    assert client.post("/predict", json=invalid).status_code == 422


def test_extra_field_is_rejected(client):
    invalid = {**VALID_INPUT, "unexpected_field": 123}
    assert client.post("/predict", json=invalid).status_code == 422


def test_missing_field_is_rejected(client):
    invalid = {k: v for k, v in VALID_INPUT.items() if k != "sepal_width_cm"}
    assert client.post("/predict", json=invalid).status_code == 422


def test_feature_order_is_preserved(client):
    """אותן מדידות בסדר שדות אחר חייבות להחזיר אותה תחזית."""
    reversed_input = dict(reversed(list(VALID_INPUT.items())))
    first = client.post("/predict", json=VALID_INPUT).json()
    second = client.post("/predict", json=reversed_input).json()
    assert first["prediction"] == second["prediction"]


def test_model_beats_baseline():
    """בדיקת מודל: התוצאה חייבת להיות טובה מהותית מ-Baseline."""
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_cv_macro_f1"] > metrics["baseline_cv_macro_f1"] + 0.20
    assert metrics["test_macro_f1"] >= 0.85
