import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.api.schemas import PredictionInput, PredictionOutput, AlertPayload
from src.api.alerting import (
    AlertManager, EmailChannel, WebhookChannel, BrowserPushChannel,
)

MODELS_READY = Path('models/random_forest.pkl').exists() and Path('models/feature_names.json').exists()


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


def test_prediction_output_attention_and_version():
    out = PredictionOutput(
        flare_probability=0.7,
        dominant_feature="f",
        explanation_text="t",
        attention_weights=[0.1, 0.2, 0.7],
        model_version="transformer-v1",
    )
    assert out.attention_weights == [0.1, 0.2, 0.7]
    assert out.model_version == "transformer-v1"


def test_prediction_input_windows_optional():
    inp = PredictionInput(features=[0.1], feature_names=["f0"])
    assert inp.soft_window is None and inp.hard_window is None
    inp2 = PredictionInput(features=[0.1], feature_names=["f0"], soft_window=[1.0], hard_window=[2.0])
    assert inp2.soft_window == [1.0]


# --------------------------- alerting ---------------------------

def _prediction(prob: float) -> PredictionOutput:
    return PredictionOutput(
        flare_probability=prob,
        dominant_feature="f",
        explanation_text="t",
    )


def test_alert_rising_edge_and_suppression():
    mgr = AlertManager(config_path="nonexistent.yaml")
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    assert mgr.should_alert(0.2, t0) is False          # below threshold
    assert mgr.should_alert(0.8, t0) is True           # rising edge
    assert mgr.should_alert(0.9, t0 + timedelta(minutes=1)) is False  # no edge
    mgr.was_above_threshold = False
    assert mgr.should_alert(0.9, t0 + timedelta(minutes=5)) is False  # quiet window
    assert mgr.should_alert(0.9, t0 + timedelta(minutes=20)) is True  # after quiet window


def test_email_channel_noop_when_disabled():
    ch = EmailChannel({"enabled": False, "smtp_host": "smtp.test", "to_addrs": ["a@b.c"]})
    assert ch.send(AlertPayload(
        flare_probability=0.9, severity_probs={}, expected_lead_time_min=5,
        explanation_text="x", deeplink="/d", timestamp_utc=datetime.utcnow(),
    )) is False


def test_webhook_channel_posts_json(monkeypatch):
    captured = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=10):
        captured['data'] = json.loads(req.data.decode())
        captured['url'] = req.full_url
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ch = WebhookChannel({"enabled": True, "url": "https://hooks.example.com/x"})
    payload = AlertPayload(
        flare_probability=0.9, severity_probs={"B": 0.5}, expected_lead_time_min=5,
        explanation_text="x", deeplink="/d", timestamp_utc=datetime.utcnow(),
    )
    assert ch.send(payload) is True
    assert captured['data']['flare_probability'] == 0.9
    assert captured['url'] == "https://hooks.example.com/x"


def test_browser_channel_requires_broadcaster():
    ch = BrowserPushChannel({"enabled": True})
    payload = AlertPayload(
        flare_probability=0.9, severity_probs={}, expected_lead_time_min=5,
        explanation_text="x", deeplink="/d", timestamp_utc=datetime.utcnow(),
    )
    assert ch.send(payload) is False
    got = []
    ch.register_broadcaster(lambda p: got.append(p))
    assert ch.send(payload) is True
    assert got[0]['flare_probability'] == 0.9


# --------------------------- inference engine ---------------------------

@pytest.mark.skipif(not MODELS_READY, reason="trained RF artifacts not present")
def test_engine_payload_spec_compliance():
    from src.api.inference import InferenceEngine
    engine = InferenceEngine()
    with open('models/feature_names.json') as f:
        names = json.load(f)
    inp = PredictionInput(features=[0.0] * len(names), feature_names=names)
    out = engine.predict(inp)
    # docs/06 §4: all fields present with valid types
    assert 0.0 <= out.flare_probability <= 1.0
    assert set(out.severity_probs.keys()) == {"B", "C", "M", "X"}
    assert sum(out.severity_probs.values()) <= 1.0 + 1e-6
    if out.flare_probability < 0.3:
        assert out.expected_lead_time_min is None
    assert isinstance(out.solar_state, str)
    assert isinstance(out.explanation_text, str) and out.explanation_text
    assert out.model_uncertainty >= 0
    assert out.inference_timestamp_utc is not None
    # docs/07 §3: top SHAP features serialized for the dashboard panel
    assert out.top_features is not None and len(out.top_features) > 0
    assert all(set(f.keys()) >= {'name', 'value', 'shap', 'direction'} for f in out.top_features)
    # timestamps must be timezone-aware UTC
    assert out.inference_timestamp_utc.tzinfo is not None


@pytest.mark.skipif(not MODELS_READY, reason="trained RF artifacts not present")
def test_engine_rejects_wrong_feature_count():
    from src.api.inference import InferenceEngine
    engine = InferenceEngine()
    inp = PredictionInput(features=[0.0] * 3, feature_names=["a", "b", "c"])
    with pytest.raises(ValueError):
        engine.predict(inp)


STREAMING_READY = Path('data/processed/synchronized_clean.parquet').exists()


def test_broadcast_schedules_onto_event_loop():
    """Regression: _broadcast used asyncio.get_running_loop() from a worker
    thread (asyncio.to_thread), which raised RuntimeError and silently dropped
    every WS message. It must schedule onto the captured main loop instead."""
    import asyncio
    import threading
    import src.api.server as server_mod

    async def scenario():
        received = []

        class FakeWS:
            async def send_text(self, text):
                received.append(text)

        ws = FakeWS()
        server_mod.connected_ws.add(ws)
        server_mod.loop = asyncio.get_running_loop()
        try:
            # simulate the worker-thread tick calling _broadcast
            t = threading.Thread(
                target=server_mod._broadcast,
                args=(json.dumps({"type": "telemetry", "points": []}),),
            )
            t.start()
            t.join(timeout=5)
            await asyncio.sleep(0.2)  # let the scheduled coroutine run
        finally:
            server_mod.connected_ws.discard(ws)
            server_mod.loop = None
        assert received, "broadcast from worker thread delivered nothing"
        assert json.loads(received[0])["type"] == "telemetry"

    asyncio.run(scenario())


def test_history_query_normalizes_tz_bounds(tmp_path):
    """Naive start/end strings must be interpreted as UTC so SQLite string
    comparisons are consistent with the stored tz-aware isoformat rows."""
    from src.api.history import InferenceHistory
    db = InferenceHistory(db_path=str(tmp_path / "h.db"))
    out = PredictionOutput(flare_probability=0.9)
    db.insert_prediction(out)
    # naive bound far in the future must exclude the row (no crash / no match)
    df = db.query_predictions(start="2999-01-01T00:00:00")
    assert len(df) == 0
    # Z-suffixed and +00:00 bounds must both match
    assert len(db.query_predictions(start="2000-01-01T00:00:00Z")) == 1
    assert len(db.query_predictions(start="2000-01-01T00:00:00+00:00")) == 1


@pytest.mark.skipif(not STREAMING_READY, reason="synchronized telemetry not present")
def test_telemetry_streamer_tick_payload():
    from src.api.streaming import TelemetryStreamer
    s = TelemetryStreamer()
    tick = s.current_tick()
    # real telemetry + real features, no fake zero vectors
    assert len(tick['points']) > 0
    assert set(tick['points'][0].keys()) >= {'timestamp', 'soft_flux', 'hard_flux'}
    assert len(tick['features']) == len(tick['feature_names'])
    assert len(tick['soft_window']) > 0 and len(tick['soft_window']) == len(tick['hard_window'])
    assert not all(v == 0 for v in tick['features'][:10])
    # step() advances position and stays valid
    tick2 = s.step()
    assert tick2 is not None
