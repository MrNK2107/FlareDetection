# Task 8: Inference API

**Dependencies**: Task 6 (Baseline Models), Task 7 (Explainability)
**Estimated effort**: Medium
**PRD Reference**: Phase 5 — Forecasting Output Specification (§3.5); Dashboard & Alerting (§3.7)

## Objective

Build a FastAPI server that loads the trained Random Forest model, accepts feature vectors, and returns the structured prediction payload defined in PRD §3.5. Include WebSocket support for real-time streaming.

## Files to Create

- `src/api/__init__.py`
- `src/api/server.py`
- `src/api/schemas.py`
- `src/api/inference.py`
- `src/api/alerting.py`
- `tests/test_api.py`

## Implementation Steps

### 1. `schemas.py` — Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime

class PredictionInput(BaseModel):
    """Input to the prediction endpoint."""
    features: List[float] = Field(..., description="Feature vector")
    feature_names: List[str] = Field(..., description="Feature names")
    window_timestamp_utc: Optional[datetime] = None

class PredictionOutput(BaseModel):
    """Structured prediction payload per PRD §3.5."""

    flare_probability: float = Field(..., ge=0, le=1)
    severity_probs: Dict[str, float] = Field(...)  # {'B': ..., 'C': ..., 'M': ..., 'X': ...}
    expected_lead_time_min: Optional[float] = None
    lead_time_ci_90: Optional[List[float]] = None
    solar_state: str = "Unknown"
    state_transition_probs: Optional[Dict[str, float]] = None
    dominant_feature: str
    explanation_text: str
    model_uncertainty: float = Field(..., ge=0)
    inference_timestamp_utc: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "flare_probability": 0.87,
                "severity_probs": {"B": 0.1, "C": 0.45, "M": 0.30, "X": 0.02},
                "expected_lead_time_min": 12.5,
                "lead_time_ci_90": [8.2, 16.8],
                "solar_state": "Precursor",
                "dominant_feature": "dsoft_dt_mean",
                "explanation_text": "Prediction driven by rapid soft X-ray rise...",
                "model_uncertainty": 0.12,
                "inference_timestamp_utc": "2026-07-01T12:00:00Z"
            }
        }

class AlertPayload(BaseModel):
    """Alert payload for push notifications."""
    flare_probability: float
    severity_probs: Dict[str, float]
    expected_lead_time_min: Optional[float]
    explanation_text: str
    deeplink: str
    timestamp_utc: datetime
```

### 2. `inference.py` — Inference Engine

```python
class InferenceEngine:
    """
    Handles model loading, prediction, and explanation for the API.

    This is the core runtime component that the server calls.
    """

    def __init__(
        self,
        model_path: str = "models/random_forest.pkl",
        feature_names_path: str = "models/feature_names.json",
        threshold_path: str = "models/decision_threshold.json"
    ):
        ...
        self.model = joblib.load(model_path)
        with open(feature_names_path) as f:
            self.feature_names = json.load(f)
        self.explainer = RFExplainer(model_path=model_path)
        self.explainer.set_feature_names(self.feature_names)
        self.explanation_gen = ExplanationGenerator()

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Run full inference: predict → explain → format output.

        Steps:
        1. Validate feature vector (correct length, no NaN)
        2. Get probability prediction from model
        3. Get SHAP explanation
        4. Estimate uncertainty (MC Dropout or variance of trees)
        5. Generate explanation text
        6. Populate lead_time (if probability > 0.3)
        7. Return structured PredictionOutput
        """
        ...

    def estimate_uncertainty(self, X: np.ndarray) -> float:
        """
        Estimate epistemic uncertainty using tree variance.
        For Random Forest: std of predicted probabilities across trees.
        """
        ...

    def estimate_lead_time(self, features: np.ndarray, probability: float) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Estimate lead time until flare peak.

        For MVP: Use a heuristic based on soft flux derivative and
        cross-channel lag. If probability < 0.3, return None.

        Returns (expected_minutes, [ci_lower, ci_upper]) or (None, None).
        """
        ...
```

**Edge cases**:
- Feature vector with NaN → reject with 422 error
- Model not loaded → return 503 Service Unavailable
- probability < 0.3 → lead_time = None (per PRD spec)
- All trees predict 0 → uncertainty = 0 (high certainty about no flare)

### 3. `server.py` — FastAPI Application

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

inference_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global inference_engine
    inference_engine = InferenceEngine()
    yield

app = FastAPI(
    title="FlareClassifier — Solar Flare Forecasting API",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": inference_engine is not None}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict flare probability for a given feature vector.

    Returns structured payload per PRD Phase 5 specification.
    """
    ...

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming.

    Accepts feature vectors as JSON messages, returns predictions.
    Used by the dashboard for live updates.

    Message format (client → server):
        {"features": [...], "feature_names": [...], "window_timestamp_utc": "..."}

    Response format (server → client):
        Full PredictionOutput as JSON
    """
    ...

@app.get("/metrics")
async def get_metrics():
    """
    Return current model performance metrics.
    """
    ...
```

**Edge cases**:
- WebSocket client disconnects → handle WebSocketDisconnect gracefully
- Multiple WebSocket connections → maintain connection pool
- Model inference takes too long → set timeout, return partial result
- Concurrent requests → FastAPI handles with async; ensure model is thread-safe

### 4. `alerting.py` — Alert System (Basic)

```python
class AlertManager:
    """
    Manages alert thresholds, suppression windows, and dispatch.

    For MVP: WebSocket push + console logging.
    Full implementation: email (SMTP), webhook (JSON POST), Web Push API.
    """

    def __init__(self, threshold: float = 0.5, quiet_window_minutes: float = 15.0):
        self.threshold = threshold
        self.quiet_window = timedelta(minutes=quiet_window_minutes)
        self.last_alert_time = None

    def should_alert(self, probability: float, timestamp: datetime) -> bool:
        """
        Returns True if:
        1. probability crosses threshold AND
        2. sufficient time since last alert (rising edge only)
        """
        ...

    def dispatch_alert(self, prediction: PredictionOutput) -> AlertPayload:
        """Create and dispatch alert payload."""
        ...
```

**Edge cases**:
- probability stays above threshold for hours → only alert on rising edge, not sustained
- Multiple rapid threshold crossings → suppressed by quiet window
- First prediction ever → no last_alert_time, alert if threshold crossed

### 5. Write Tests

| Test | Description |
|------|-------------|
| `test_health_endpoint` | GET /health returns 200 with status |
| `test_predict_valid_input` | POST /predict with valid features returns 200 |
| `test_predict_invalid_input` | POST with wrong feature count returns 422 |
| `test_predict_nan_input` | POST with NaN features returns 422 |
| `test_prediction_output_schema` | Response matches PredictionOutput schema |
| `test_lead_time_null_when_low_prob` | prob < 0.3 → lead_time is None |
| `test_websocket_echo` | WebSocket echo test (send → receive) |
| `test_alert_threshold` | should_alert returns True on rising edge |
| `test_alert_suppression` | Repeated triggers within quiet window suppress |

## Acceptance Criteria

- [ ] `uvicorn src.api.server:app` starts without errors
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `POST /predict` returns valid PredictionOutput for valid input
- [ ] Invalid input returns appropriate error (422)
- [ ] WebSocket endpoint accepts connections and streams predictions
- [ ] lead_time is None when probability < 0.3
- [ ] Alert system suppresses duplicate alerts within 15-minute window
- [ ] All tests pass: `pytest tests/test_api.py -v`

## Verification

```bash
# Start server in background
Start-Job { python -m uvicorn src.api.server:app --port 8000 }
Start-Sleep 5

# Test health
python -c "
import requests, json
r = requests.get('http://localhost:8000/health')
print(f'Health: {r.json()}')

# Test prediction
features = [0.0] * 30  # Replace with actual feature count
r2 = requests.post('http://localhost:8000/predict', json={
    'features': features,
    'feature_names': [f'f{i}' for i in range(len(features))]
})
print(f'Predict status: {r2.status_code}')
if r2.status_code == 200:
    data = r2.json()
    print(f'Flare probability: {data[\"flare_probability\"]}')
    print(f'Dominant feature: {data[\"dominant_feature\"]}')
    print(f'Explanation: {data[\"explanation_text\"][:100]}...')
"
```
