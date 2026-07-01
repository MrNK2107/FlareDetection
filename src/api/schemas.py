from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="Feature vector")
    feature_names: List[str] = Field(..., description="Feature names")
    window_timestamp_utc: Optional[datetime] = None


class PredictionOutput(BaseModel):
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


class AlertPayload(BaseModel):
    flare_probability: float
    severity_probs: Dict[str, float]
    expected_lead_time_min: Optional[float]
    explanation_text: str
    deeplink: str
    timestamp_utc: datetime
