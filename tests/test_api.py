import pytest
from datetime import datetime
from src.api.schemas import PredictionInput, PredictionOutput, AlertPayload


def test_prediction_input_schema():
    data = {
        "features": [0.1] * 30,
        "feature_names": [f"f{i}" for i in range(30)],
        "window_timestamp_utc": "2026-01-01T12:00:00",
    }
    inp = PredictionInput(**data)
    assert len(inp.features) == 30
    assert len(inp.feature_names) == 30


def test_prediction_output_schema():
    data = {
        "flare_probability": 0.85,
        "severity_probs": {"B": 0.1, "C": 0.5, "M": 0.2, "X": 0.05},
        "expected_lead_time_min": 12.5,
        "lead_time_ci_90": [8.0, 17.0],
        "solar_state": "Precursor",
        "dominant_feature": "dsoft_dt_mean",
        "explanation_text": "Prediction driven by rapid soft X-ray rise",
        "model_uncertainty": 0.12,
        "inference_timestamp_utc": datetime.utcnow(),
    }
    out = PredictionOutput(**data)
    assert 0 <= out.flare_probability <= 1
    assert out.dominant_feature == "dsoft_dt_mean"
    assert out.lead_time_ci_90 == [8.0, 17.0]


def test_prediction_output_defaults():
    data = {
        "flare_probability": 0.1,
        "dominant_feature": "test",
        "explanation_text": "test",
        "model_uncertainty": 0.0,
    }
    out = PredictionOutput(**data)
    assert out.expected_lead_time_min is None
    assert out.severity_probs == {"B": 0, "C": 0, "M": 0, "X": 0}


def test_lead_time_null_when_low_prob():
    data = {
        "flare_probability": 0.2,
        "dominant_feature": "test",
        "explanation_text": "test",
        "model_uncertainty": 0.0,
    }
    out = PredictionOutput(**data)
    assert out.expected_lead_time_min is None


def test_alert_payload_schema():
    data = {
        "flare_probability": 0.8,
        "severity_probs": {"B": 0.1, "C": 0.6, "M": 0.1, "X": 0.0},
        "expected_lead_time_min": 10.0,
        "explanation_text": "Flare imminent",
        "deeplink": "/dashboard?time=2026-01-01T12:00:00",
        "timestamp_utc": datetime.utcnow(),
    }
    alert = AlertPayload(**data)
    assert alert.flare_probability == 0.8
    assert alert.deeplink.startswith("/dashboard")
