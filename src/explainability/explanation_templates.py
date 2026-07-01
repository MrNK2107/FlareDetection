import numpy as np
from typing import Dict, List, Optional


class ExplanationGenerator:
    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        return {
            'soft_rise': (
                'Prediction driven by rapid soft X-ray rise '
                '(SHAP rank 1, {shap_rank_1_value:+.2f}). '
                'Soft flux increasing above quiet-Sun baseline.'
            ),
            'hard_rise': (
                'Emerging hard X-ray acceleration detected '
                '(SHAP rank 1, {shap_rank_1_value:+.2f}). '
                'Non-thermal electron emission is increasing.'
            ),
            'cross_channel': (
                'Cross-channel signature detected: '
                'lag correlation peaked at {peak_lag_s:.0f}s '
                '(SHAP rank 1, {shap_rank_1_value:+.2f}).'
            ),
            'precursor': (
                'Precursor signature: soft X-ray rising while '
                'hard X-ray remains at quiet levels. '
                'Thermal fraction elevated ({thermal_fraction:.2f}).'
            ),
            'thermal': (
                'Thermal emission features dominant '
                '(SHAP rank 1, {shap_rank_1_value:+.2f}). '
                'Quiet-Sun or early-stage activity.'
            ),
            'quiet': (
                'All channels at quiet-Sun levels. '
                'No significant deviation detected across features. '
                'Flare probability is low.'
            ),
            'uncertainty': (
                'Model uncertainty is high (std={uncertainty:.3f}). '
                'Features are near decision boundary. '
                'Dominant signal: {dominant_feature}.'
            ),
            'generic': (
                'Prediction primarily influenced by {dominant_feature} '
                '(SHAP rank 1, {shap_rank_1_value:+.2f}). '
                'Model confidence: {confidence:.0%}.'
            ),
        }

    def _select_template(self, dominant_feature: str, top_features: List[Dict]) -> str:
        df_lower = dominant_feature.lower()
        if 'soft_rising' in df_lower or 'dsoft' in df_lower:
            return 'soft_rise'
        elif 'hard_accelerating' in df_lower or 'dhard' in df_lower:
            return 'hard_rise'
        elif 'peak_lag' in df_lower or 'correlation' in df_lower:
            return 'cross_channel'
        elif 'precursor' in df_lower:
            return 'precursor'
        elif 'thermal' in df_lower:
            return 'thermal'
        elif 'quiet' in df_lower or np.abs(top_features[0]['shap']) < 0.01 if top_features else True:
            return 'quiet'
        else:
            return 'generic'

    def generate(
        self,
        dominant_feature: str,
        top_features: List[Dict],
        shap_values: np.ndarray,
        prediction: Dict,
        uncertainty: float = 0.0,
    ) -> str:
        if uncertainty > 0.3:
            template_key = 'uncertainty'
        else:
            template_key = self._select_template(dominant_feature, top_features)
        template = self.templates.get(template_key, self.templates['generic'])
        context = {
            'dominant_feature': dominant_feature,
            'shap_rank_1_value': top_features[0]['shap'] if top_features else 0.0,
            'shap_rank_1_name': top_features[0]['name'] if top_features else '',
            'uncertainty': uncertainty,
            'confidence': prediction.get('flare_probability', 0.0),
            'peak_lag_s': 0.0,
            'thermal_fraction': 0.0,
        }
        for feat in top_features[:5]:
            name = feat['name'].lower()
            if 'lag' in name or 'peak_lag' in name:
                context['peak_lag_s'] = abs(feat['value'])
            if 'thermal' in name:
                context['thermal_fraction'] = feat['value']
        try:
            text = template.format(**context)
        except KeyError:
            text = template
        return text
