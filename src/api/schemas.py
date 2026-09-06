from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="Engineered feature vector")
    feature_names: List[str] = Field(..., description="Feature names")
    window_timestamp_utc: Optional[datetime] = None
    # Raw windows at DL cadence enable the Transformer path (docs/05 §2).
    soft_window: Optional[List[float]] = Field(None, description="Soft X-ray window at DL cadence")
    hard_window: Optional[List[float]] = Field(None, description="Hard X-ray window at DL cadence")


class PredictionOutput(BaseModel):
    # docs/06 §4 inference payload specification
    flare_probability: float = Field(..., ge=0, le=1)
    severity_probs: Dict[str, float] = Field(default_factory=lambda: {"B": 0, "C": 0, "M": 0, "X": 0})
    expected_lead_time_min: Optional[float] = None
    lead_time_ci_90: Optional[List[float]] = None
    solar_state: str = "Unknown"
    state_transition_probs: Optional[Dict[str, float]] = None
    dominant_feature: str = ""
    explanation_text: str = ""
    model_uncertainty: float = Field(default=0.0, ge=0)
    inference_timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    # docs/07 §2: attention weights serialized with the inference payload
    attention_weights: Optional[List[float]] = None
    model_version: str = "baseline-rf"


class AlertPayload(BaseModel):
    flare_probability: float
    severity_probs: Dict[str, float]
    expected_lead_time_min: Optional[float]
    explanation_text: str
    deeplink: str
    timestamp_utc: datetime
