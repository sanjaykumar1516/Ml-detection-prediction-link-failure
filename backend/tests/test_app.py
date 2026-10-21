
import sys
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import (
    app,
    generate_metrics,
    evaluate_thresholds,
    DEFAULT_THRESHOLDS
)

client = TestClient(app)


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_simulation_reproducibility():
    first = generate_metrics(
        "normal", 42, 1, DEFAULT_THRESHOLDS
    )

    second = generate_metrics(
        "normal", 42, 1, DEFAULT_THRESHOLDS
    )

    assert first["latency"] == second["latency"]
    assert first["packet_loss"] == second["packet_loss"]
    assert first["snr"] == second["snr"]


def test_outage():
    result = generate_metrics(
        "outage", 42, 1, DEFAULT_THRESHOLDS
    )

    assert result["packet_loss"] == 100
    assert result["quality"] == 0
    assert result["state"] == "OUTAGE"


def test_thresholds():
    metrics = {
        "latency": 400,
        "packet_loss": 20,
        "jitter": 60,
        "snr": 2
    }

    result = evaluate_thresholds(
        metrics, DEFAULT_THRESHOLDS
    )

    assert len(result) == 4


def test_invalid_coordinates():
    response = client.post(
        "/api/visibility",
        json={
            "satellite_id": "25544",
            "latitude": 120,
            "longitude": 77,
            "elevation_mask": 10
        }
    )

    assert response.status_code == 422


def test_invalid_scenario():
    client.post(
        "/api/simulation/start",
        json={"seed": 42}
    )

    response = client.post(
        "/api/simulation/scenario",
        json={"scenario": "unknown"}
    )

    assert response.status_code == 422


def test_failure_and_recovery():
    failure = generate_metrics(
        "outage", 42, 1, DEFAULT_THRESHOLDS
    )

    recovery = generate_metrics(
        "recovery", 42, 2, DEFAULT_THRESHOLDS
    )

    assert failure["state"] == "OUTAGE"
    assert recovery["quality"] > failure["quality"]


def test_ml_prediction():
    path = ROOT / "ml" / "models" / "jamming.joblib"

    if not path.exists():
        pytest.skip("Train models first")

    response = client.post(
        "/api/ml/jamming/predict",
        json={
            "latency": 180,
            "packet_loss": 30,
            "jitter": 45,
            "snr": 3
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]
