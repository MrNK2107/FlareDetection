import numpy as np
import shap
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional


class RFExplainer:
    def __init__(self, model_path: str = "models/random_forest.pkl"):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names: List[str] = []
        names_path = Path(model_path).parent / 'feature_names.json'
        if names_path.exists():
            with open(names_path) as f:
                self.feature_names = json.load(f)

    def set_feature_names(self, names: List[str]):
        self.feature_names = names

    def explain(
        self,
        X: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        shap_arr = self.explainer.shap_values(X)
        if isinstance(shap_arr, list):
            shap_arr = np.array(shap_arr, dtype=float)
        shap_arr = np.atleast_3d(shap_arr)
        if shap_arr.shape[-1] > 1:
            shap_values = shap_arr[:, :, 1]
            ev = self.explainer.expected_value
            ev = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else (float(ev) if np.ndim(ev) == 0 else float(ev[0]))
        else:
            shap_values = shap_arr[:, :, 0]
            ev = self.explainer.expected_value
            ev = float(ev) if np.ndim(ev) == 0 else float(ev[0])
        base_value = float(ev)
        if X.shape[0] > 1:
            shap_values = shap_values[0:1]
            X = X[0:1]
        shap_vals = shap_values[0]
        feature_vals = X[0]
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(shap_vals))]
        indices = np.argsort(np.abs(shap_vals))[::-1]
        top_features = []
        for idx in indices[:top_k]:
            sval = float(shap_vals[idx])
            direction = 'increasing' if sval > 0 else 'decreasing'
            top_features.append({
                'name': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                'value': float(feature_vals[idx]),
                'shap': sval,
                'direction': direction,
            })
        dominant = top_features[0]['name'] if top_features else 'no_feature_dominant'
        return {
            'shap_values': shap_values.tolist(),
            'base_value': base_value,
            'top_features': top_features,
            'dominant_feature': dominant,
        }
