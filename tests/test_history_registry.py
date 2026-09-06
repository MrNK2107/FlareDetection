import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.api.history import InferenceHistory
from src.api.schemas import PredictionOutput, AlertPayload
from src.models.registry import ModelRegistry
from src.models.monitoring import tss_from_scores, rolling_tss, check_drift


# --------------------------- history ---------------------------

def _pred(prob=0.4, ts=None):
    return PredictionOutput(
        flare_probability=prob,
        severity_probs={"B": 0.2, "C": 0.1, "M": 0.05, "X": 0.0},
        expected_lead_time_min=10.0 if prob >= 0.3 else None,
        solar_state="Quiet",
        model_uncertainty=0.05,
        dominant_feature="f",
        explanation_text="test explanation",
        inference_timestamp_utc=ts or datetime(2026, 1, 1, 12, 0, 0),
    )


def test_history_round_trip(tmp_path):
    h = InferenceHistory(str(tmp_path / "hist.db"))
    h.insert_prediction(_pred(0.1))
    h.insert_prediction(_pred(0.9, ts=datetime(2026, 1, 2, 12, 0, 0)))
    df = h.query_predictions()
    assert len(df) == 2
    assert df.iloc[0]['flare_probability'] == 0.1
    window = h.query_predictions(start="2026-01-02", end="2026-01-03")
    assert len(window) == 1
    h.insert_alert(AlertPayload(
        flare_probability=0.9, severity_probs={}, expected_lead_time_min=5,
        explanation_text="x", deeplink="/d", timestamp_utc=datetime(2026, 1, 2, 12, 0, 0),
    ), channels=["webhook"])
    alerts = h.query_alerts()
    assert len(alerts) == 1
    assert json.loads(alerts.iloc[0]['channels']) == ["webhook"]


def test_history_csv_export(tmp_path):
    h = InferenceHistory(str(tmp_path / "hist.db"))
    h.insert_prediction(_pred(0.5))
    csv_text = h.predictions_csv()
    assert "flare_probability" in csv_text
    assert "0.5" in csv_text


# --------------------------- registry ---------------------------

def _metrics(tss=0.7, brier=0.05, far=0.1, det=0.9, mae=4.0):
    return {
        'tss': tss, 'brier_score': brier, 'false_alarm_rate': far,
        'detection_rate': det, 'lead_time_mae_min': mae,
    }


def test_registry_register_list_promote(tmp_path):
    reg = ModelRegistry(str(tmp_path / "registry"), str(tmp_path / "production.json"))
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"fake")
    v1 = reg.register("TestModel", _metrics(), {"model.pkl": str(artifact)})
    assert v1.endswith("-TestModel")
    assert len(reg.list_models()) == 1
    reg.promote(v1)
    prod = reg.get_production()
    assert prod['version'] == v1
    # immutability: registering same name twice creates distinct dirs
    v2 = reg.register("TestModel", _metrics(tss=0.8), {"model.pkl": str(artifact)})
    assert v2 != v1


def test_promotion_gate_all_metrics():
    better = _metrics(tss=0.8, brier=0.04, far=0.08, det=0.95, mae=3.0)
    worse = _metrics(tss=0.5, brier=0.06, far=0.2, det=0.7, mae=6.0)
    passed, _ = ModelRegistry.promotion_check(better, _metrics())
    assert passed
    passed, report = ModelRegistry.promotion_check(worse, _metrics())
    assert not passed
    # missing metric -> fail
    incomplete = {k: v for k, v in _metrics().items() if k != 'tss'}
    passed, report = ModelRegistry.promotion_check(incomplete, _metrics())
    assert not passed


# --------------------------- monitoring ---------------------------

def test_tss_from_scores():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert tss_from_scores(y_true, y_score) == 1.0
    y_score_bad = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    assert tss_from_scores(y_true, y_score_bad) == -1.0


def test_rolling_tss_periods():
    ts = pd.date_range("2026-01-01", periods=40, freq="6h")  # 10 days
    y_true = np.zeros(40)
    y_true[::5] = 1
    y_score = np.where(y_true == 1, 0.9, 0.1)
    rt = rolling_tss(ts, y_true, y_score, window_days=7.0)
    assert len(rt) == 2  # two 7-day buckets cover 10 days
    assert (rt['tss'] == 1.0).all()


def test_check_drift_triggers():
    ts = pd.date_range("2026-01-01", periods=400, freq="h")
    y_true = np.zeros(400, dtype=int)
    y_true[10:20] = 1      # flares in week 1, detected perfectly
    y_true[250:260] = 1    # flares later, missed entirely
    y_score = np.where(y_true == 1, 0.9, 0.1)
    y_score[240:] = 0.1    # model stops detecting after hour 240
    report = check_drift(
        ts, y_true, y_score, baseline_tss=1.0,
        new_labelled_samples=600,
    )
    assert report['alert_triggered'] is True
    assert report['volume_trigger_exceeded'] is True
    assert report['retrain_recommended'] is True


def test_check_drift_no_trigger_when_healthy():
    ts = pd.date_range("2026-02-01", periods=48, freq="h")
    y_true = np.zeros(48)
    y_true[5:10] = 1
    y_score = np.where(y_true == 1, 0.9, 0.1)
    report = check_drift(
        ts, y_true, y_score, baseline_tss=1.0,
        days_since_last_retrain=1.0,
        new_labelled_samples=10,
    )
    assert report['alert_triggered'] is False
    assert report['scheduled_check_due'] is False
    assert report['volume_trigger_exceeded'] is False
    assert report['retrain_recommended'] is False
