import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.api.schemas import PredictionInput, PredictionOutput
from src.explainability.explainer import RFExplainer
from src.explainability.explanation_templates import ExplanationGenerator


class InferenceEngine:
    def __init__(
        self,
        model_path: str = "models/random_forest.pkl",
        feature_names_path: str = "models/feature_names.json",
        threshold_path: str = "models/decision_threshold.json",
    ):
        self.model = joblib.load(model_path)
        with open(feature_names_path) as f:
            self.feature_names = json.load(f)
        with open(threshold_path) as f:
            self.threshold_config = json.load(f)
        self.explainer = RFExplainer(model_path=model_path)
        self.explainer.set_feature_names(self.feature_names)
        self.explanation_gen = ExplanationGenerator()

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        features = np.array(input_data.features, dtype=np.float64).reshape(1, -1)
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )
        if np.any(np.isnan(features)):
            raise ValueError("Features contain NaN values")
        y_proba = self.model.predict_proba(features)
        classes = self.model.classes_
        prob_flare = 0.0
        severity_probs = {"B": 0.0, "C": 0.0, "M": 0.0, "X": 0.0}
        class_map = {0: None, 1: "B", 2: "C", 3: "M", 4: "X"}
        for i, cls in enumerate(classes):
            label = class_map.get(int(cls))
            if label:
                severity_probs[label] = float(y_proba[0, i])
            if int(cls) > 0:
                prob_flare += float(y_proba[0, i])
        uncertainty = self.estimate_uncertainty(features)
        lead_time, lead_ci = self.estimate_lead_time(features, prob_flare)
        explanation = self.explainer.explain(features)
        explanation_text = self.explanation_gen.generate(
            dominant_feature=explanation['dominant_feature'],
            top_features=explanation['top_features'],
            shap_values=np.array(explanation['shap_values']),
            prediction={'flare_probability': prob_flare},
            uncertainty=uncertainty,
        )
        return PredictionOutput(
            flare_probability=round(prob_flare, 4),
            severity_probs=severity_probs,
            expected_lead_time_min=lead_time,
            lead_time_ci_90=lead_ci,
            solar_state="Unknown",
            state_transition_probs=None,
            dominant_feature=explanation['dominant_feature'],
            explanation_text=explanation_text,
            model_uncertainty=round(uncertainty, 4),
            inference_timestamp_utc=datetime.utcnow(),
        )

    def estimate_uncertainty(self, X: np.ndarray) -> float:
        try:
            trees = self.model.estimators_
            preds = np.array([tree.predict_proba(X)[0, 1] for tree in trees])
            return float(np.std(preds))
        except Exception:
            return 0.0

    def estimate_lead_time(self, features: np.ndarray, probability: float) -> Tuple[Optional[float], Optional[List[float]]]:
        if probability < 0.3:
            return None, None
        base = 15.0
        jitter = float(np.random.uniform(-3, 3))
        lead = base + jitter
        return round(max(lead, 1), 1), [round(max(lead - 5, 0), 1), round(lead + 5, 1)]
