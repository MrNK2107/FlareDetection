import json
import joblib
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.api.schemas import PredictionInput, PredictionOutput
from src.explainability.explainer import RFExplainer
from src.explainability.explanation_templates import ExplanationGenerator

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEVERITY_CLASSES = ['B', 'C', 'M', 'X']


class InferenceEngine:
    """Serves predictions from the best available model.

    Deep path (Transformer) is used when model + raw windows are provided;
    otherwise the classical RF path is used. Payload always matches docs/06 §4.
    """

    def __init__(self, config_path: str = "config/config.yaml", rf_model_path: str = "models/random_forest.pkl"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        # classical path
        self.model = joblib.load(rf_model_path)
        with open('models/feature_names.json') as f:
            self.feature_names = json.load(f)
        self.explainer = RFExplainer(model_path='models/random_forest.pkl')
        self.explainer.set_feature_names(self.feature_names)
        self.explanation_gen = ExplanationGenerator()
        # optional deep path
        self.transformer = None
        self.transformer_meta = None
        try:
            self._load_transformer()
        except Exception as e:
            print(f"Transformer not available for inference ({e}); using RF path")
        # optional lead-time fallback model
        self.lead_time = None
        try:
            from src.models.lead_time import LeadTimePredictor
            self.lead_time = LeadTimePredictor()
        except Exception:
            pass
        # optional HMM state machine
        self.state_machine = None
        try:
            from src.models.state_machine import StateMachineInference
            self.state_machine = StateMachineInference()
        except Exception:
            pass

    def _load_transformer(self) -> None:
        from src.models.transformer import DualStreamTransformer
        blob = torch.load('models/transformer.pt', map_location=DEVICE, weights_only=False)
        cfg = blob['config']
        model = DualStreamTransformer(
            n_engineered_features=cfg['n_engineered_features'],
            d_model=cfg['d_model'], nhead=cfg['nhead'], num_layers=cfg['num_layers'],
            fusion_dim=cfg['fusion_dim'], dropout=cfg['dropout'],
        )
        model.load_state_dict(blob['state_dict'])
        model.to(DEVICE).eval()
        self.transformer = model
        with open('models/transformer_meta.json') as f:
            self.transformer_meta = json.load(f)

    # ------------------------------------------------------------------ utils
    def _align_vector(self, features: np.ndarray, names) -> np.ndarray:
        lookup = {n: v for n, v in zip(names, np.asarray(features).ravel())}
        return np.array([lookup.get(c, 0.0) for c in self.feature_names], dtype=np.float64)

    def _infer_state(self, features: np.ndarray, names) -> Tuple[str, Optional[Dict[str, float]]]:
        if self.state_machine is None:
            return "Unknown", None
        try:
            return self.state_machine.infer(features, names)
        except Exception:
            return "Unknown", None

    def _explain(self, features_2d: np.ndarray, prob: float, uncertainty: float):
        explanation = self.explainer.explain(features_2d)
        text = self.explanation_gen.generate(
            dominant_feature=explanation['dominant_feature'],
            top_features=explanation['top_features'],
            shap_values=np.array(explanation['shap_values']),
            prediction={'flare_probability': prob},
            uncertainty=uncertainty,
        )
        return explanation, text

    @staticmethod
    def _top_features_payload(explanation: Dict) -> List[Dict]:
        """Serialize top SHAP features for the dashboard explainability panel
        (docs/07 §3); rounded for compact WS transport."""
        return [
            {
                'name': f['name'],
                'value': round(float(f['value']), 6),
                'shap': round(float(f['shap']), 6),
                'direction': f['direction'],
            }
            for f in explanation.get('top_features', [])
        ]

    @staticmethod
    def _severity_dict(sev_probs: np.ndarray) -> Dict[str, float]:
        return {cls: round(float(p), 6) for cls, p in zip(SEVERITY_CLASSES, sev_probs)}

    # ---------------------------------------------------------------- predict
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        features = np.array(input_data.features, dtype=np.float64).ravel()
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {len(features)}"
            )
        if np.any(np.isnan(features)):
            raise ValueError("Features contain NaN values")
        has_windows = (
            self.transformer is not None
            and input_data.soft_window is not None
            and input_data.hard_window is not None
        )
        if has_windows:
            out = self._predict_deep(input_data, features)
        else:
            out = self._predict_classical(input_data, features)
        # docs/06 §4: lead time is null when flare_probability < 0.3
        if out.flare_probability < 0.3:
            out.expected_lead_time_min = None
            out.lead_time_ci_90 = None
        out.inference_timestamp_utc = datetime.now(timezone.utc)
        return out

    # ------------------------------------------------------------- classical
    def _predict_classical(self, input_data: PredictionInput, features: np.ndarray) -> PredictionOutput:
        X = features.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        prob_flare = 0.0
        severity = {c: 0.0 for c in SEVERITY_CLASSES}
        class_map = {0: None, 1: 'B', 2: 'C', 3: 'M', 4: 'X'}
        for i, cls in enumerate(classes):
            label = class_map.get(int(cls))
            if label:
                severity[label] = float(proba[i])
            if int(cls) > 0:
                prob_flare += float(proba[i])
        prob_flare = float(min(prob_flare, 1.0))
        uncertainty = self._rf_uncertainty(X)
        explanation, text = self._explain(X, prob_flare, uncertainty)
        state, trans = self._infer_state(features, input_data.feature_names)
        lead, ci = self._lead_time_classical(features, input_data.feature_names, prob_flare)
        return PredictionOutput(
            flare_probability=round(prob_flare, 4),
            severity_probs=self._severity_dict([severity[c] for c in SEVERITY_CLASSES]),
            expected_lead_time_min=lead,
            lead_time_ci_90=ci,
            solar_state=state,
            state_transition_probs=trans,
            dominant_feature=explanation['dominant_feature'],
            explanation_text=text,
            model_uncertainty=round(uncertainty, 4),
            attention_weights=None,
            top_features=self._top_features_payload(explanation),
            model_version="baseline-rf",
        )

    def _rf_uncertainty(self, X: np.ndarray) -> float:
        try:
            per_tree = np.array([
                tree.predict_proba(X)[0, 1:].sum() for tree in self.model.estimators_
            ])
            return float(np.std(per_tree))
        except Exception:
            return 0.0

    def _lead_time_classical(
        self, features: np.ndarray, names, prob: float
    ) -> Tuple[Optional[float], Optional[List[float]]]:
        if prob < 0.3 or self.lead_time is None:
            return None, None
        try:
            return self.lead_time.predict(features, names)
        except Exception:
            return None, None

    # ------------------------------------------------------------------ deep
    def _predict_deep(self, input_data: PredictionInput, features: np.ndarray) -> PredictionOutput:
        from src.models.transformer import _mc_dropout_predict
        meta = self.transformer_meta
        stats = meta['norm_stats']
        soft = np.asarray(input_data.soft_window, dtype=np.float64)
        hard = np.asarray(input_data.hard_window, dtype=np.float64)
        t = min(len(soft), int(meta['window_steps']))
        soft, hard = soft[:t], hard[:t]
        soft_n = (soft - stats['soft_mean']) / stats['soft_std']
        hard_n = (hard - stats['hard_mean']) / stats['hard_std']
        eng = self._align_vector(features, input_data.feature_names)
        eng_n = (eng - np.array(stats['eng_mean'])) / np.array(stats['eng_std'])
        sb = torch.from_numpy(soft_n.astype(np.float32)).reshape(1, -1).to(DEVICE)
        hb = torch.from_numpy(hard_n.astype(np.float32)).reshape(1, -1).to(DEVICE)
        eb = torch.from_numpy(eng_n.astype(np.float32)).reshape(1, -1).to(DEVICE)

        k = int(meta.get('mc_dropout_samples', 20))
        mc = _mc_dropout_predict(self.transformer, sb, hb, eb, k)
        prob_flare = float(np.clip(mc['flare_prob_mean'][0], 0.0, 1.0))
        uncertainty = float(mc['flare_prob_std'][0])

        self.transformer.eval()
        with torch.no_grad():
            out = self.transformer(sb, hb, eb)
        sev = torch.softmax(out['severity_logits'], dim=1)[0].cpu().numpy()
        severity = sev * prob_flare  # joint probs; sum(severity) <= p_flare

        # lead time + CI from MC samples (log-space percentiles)
        lead, ci = None, None
        if prob_flare >= 0.3:
            log_samples = np.clip(mc['lead_log_mean'][:, 0] if mc['lead_log_mean'].ndim > 1 else mc['lead_log_mean'],
                                  np.log(1.0), np.log(1440.0))
            lead_samples = np.exp(log_samples)
            lead = float(np.clip(np.median(lead_samples), 1.0, 1440.0))
            ci = [round(float(np.percentile(lead_samples, 5)), 1),
                  round(float(np.percentile(lead_samples, 95)), 1)]

        # attention over the soft timeline (mean over batch & query positions)
        attn = self.transformer.last_attention  # (B, T_hard, T_soft)
        attn_vec = attn.mean(dim=(0, 1)).cpu().numpy() if attn is not None else None

        explanation, text = self._explain(eng.reshape(1, -1), prob_flare, uncertainty)
        state, trans = self._infer_state(eng, input_data.feature_names)
        # Downsample attention over the soft timeline to <=64 points for WS
        # transport (raw T=1200 per tick is wasteful; shape is preserved).
        attention_payload = None
        if attn_vec is not None:
            n_attn = len(attn_vec)
            step = max(1, n_attn // 64)
            attention_payload = [round(float(a), 6) for a in attn_vec[::step][:64]]
        return PredictionOutput(
            flare_probability=round(prob_flare, 4),
            severity_probs=self._severity_dict(severity),
            expected_lead_time_min=round(lead, 1) if lead is not None else None,
            lead_time_ci_90=ci,
            solar_state=state,
            state_transition_probs=trans,
            dominant_feature=explanation['dominant_feature'],
            explanation_text=text,
            model_uncertainty=round(uncertainty, 4),
            attention_weights=attention_payload,
            top_features=self._top_features_payload(explanation),
            model_version="transformer-v1",
        )
