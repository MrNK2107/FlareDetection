# Task 7: Explainability

**Dependencies**: Task 6 (Baseline Models)
**Estimated effort**: Medium
**PRD Reference**: Phase 6 — Explainability (§3.6)

## Objective

Implement SHAP-based feature attribution for the Random Forest model and an auto-generated explanation text system.

## Files to Create

- `src/explainability/__init__.py`
- `src/explainability/explainer.py`
- `src/explainability/explanation_templates.py`
- `tests/test_explainability.py`

## Implementation Steps

### 1. SHAP Explainer for Random Forest

```python
import shap
import joblib

class RFExplainer:
    """
    SHAP-based explainer for Random Forest model.

    Since Random Forest is a tree-based model, use shap.TreeExplainer
    which is exact and fast (no sampling approximation needed).
    """

    def __init__(self, model_path: str = "models/random_forest.pkl"):
        """Load model and initialize TreeExplainer."""
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = None

    def set_feature_names(self, names: List[str]):
        self.feature_names = names

    def explain(
        self,
        X: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Compute SHAP values for a single prediction.

        Args:
            X: Feature vector (1, n_features) or (n_windows, n_features)

        Returns:
            {
                'shap_values': np.ndarray,     # SHAP values for each feature
                'base_value': float,           # Expected model output
                'top_features': List[Dict],    # Top-k features with:
                    {'name': str, 'value': float, 'shap': float, 'direction': 'increasing'|'decreasing'}
                'dominant_feature': str,       # Name of top feature
            }
        """
        ...
```

**Edge cases**:
- All features have near-zero SHAP → model is uncertain; dominant_feature = "no feature dominant"
- Single sample with all zeros → SHAP values sum to -base_value (model predicts low probability)
- X has wrong number of features → raise descriptive error

### 2. Template-Based Explanation Generator

```python
class ExplanationGenerator:
    """
    Generate human-readable explanation text from SHAP + attention outputs.

    PRD: "Use a template engine (not another LLM call) to produce explanation_text
    from structured SHAP + attention outputs."
    """

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """
        Template categories based on dominant feature type:
        - 'soft_rise': 'Prediction driven by rapid soft X-ray rise...'
        - 'hard_rise': 'Emerging hard X-ray acceleration detected...'
        - 'cross_channel': 'Cross-channel lag correlation narrowed...'
        - 'thermal': 'Thermal emission fraction indicates...'
        - 'precursor': 'Precursor signature detected...'
        - 'quiet': 'All channels at quiet-Sun levels...'
        - 'uncertainty': 'Model uncertainty is high...'
        """
        ...

    def _select_template(self, dominant_feature: str, top_features: List[Dict]) -> str:
        """Select the best template based on dominant feature name."""
        ...

    def _format_template(
        self,
        template: str,
        top_features: List[Dict],
        shap_values: np.ndarray,
        prediction: Dict
    ) -> str:
        """
        Fill template variables from computed features.

        All template variables must be traced to specific computed features —
        no hallucinated reasons (PRD requirement).
        """
        ...

    def generate(
        self,
        dominant_feature: str,
        top_features: List[Dict],
        shap_values: np.ndarray,
        prediction: Dict,
        X_raw: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate explanation string.

        Example output:
        'Prediction driven by rapid soft X-ray rise over the past 6 minutes
        (SHAP rank 1, +0.32) combined with emerging hard X-ray acceleration.
        Cross-channel lag narrowed from 90s to 12s in the last window.'
        """
        ...
```

**Template variables** (traced to computed features):
- `{soft_rise_duration}` → derived from dsoft_dt threshold crossings
- `{hard_rise_flag}` → from hard_accelerating
- `{peak_lag_s}` → from lag_correlation feature
- `{lag_change}` → difference in peak_lag between consecutive windows
- `{thermal_fraction}` → from thermal_fraction feature
- `{shap_rank_N_name}` → feature name at rank N
- `{shap_rank_N_value}` → SHAP value at rank N, formatted

**Edge cases**:
- `dominant_feature` doesn't match any template → use generic template
- SHAP values are all near-zero → use 'uncertainty' template
- Template variable not available → output template as-is (visible gap for debugging)

### 3. Write Tests

| Test | Description |
|------|-------------|
| `test_explainer_initializes` | TreeExplainer loads with valid RF model |
| `test_explain_returns_top_features` | explain() returns top_k features |
| `test_top_features_are_sorted` | Features sorted by |SHAP| descending |
| `test_dominant_feature_in_top_k` | dominant_feature appears in top_features |
| `test_explanation_contains_feature_name` | Generated text mentions the dominant feature |
| `test_explanation_no_hallucination` | All template variables map to real feature names |
| `test_uncertainty_template` | Low-confidence prediction → uncertainty template |
| `test_shap_values_sum` | SHAP values + base_value ≈ model prediction |

## Acceptance Criteria

- [ ] SHAP TreeExplainer works with saved Random Forest model
- [ ] explain() returns correct structure for single and batch inputs
- [ ] Template generator produces readable, non-hallucinated text
- [ ] All template variables trace to specific features
- [ ] Uncertainty handling: low-confidence predictions produce appropriate text
- [ ] All tests pass: `pytest tests/test_explainability.py -v`

## Verification

```bash
python -c "
from src.explainability.explainer import RFExplainer
from src.explainability.explanation_templates import ExplanationGenerator
import numpy as np, joblib

# Load model and a sample
model = joblib.load('models/random_forest.pkl')
X_sample = np.random.randn(1, model.n_features_in_)

explainer = RFExplainer()
result = explainer.explain(X_sample)
print(f'Base value: {result[\"base_value\"]:.4f}')
print(f'Dominant feature: {result[\"dominant_feature\"]}')
for f in result['top_features'][:3]:
    print(f'  {f[\"name\"]}: SHAP={f[\"shap\"]:+.4f}, direction={f[\"direction\"]}')

gen = ExplanationGenerator()
text = gen.generate(
    dominant_feature=result['dominant_feature'],
    top_features=result['top_features'],
    shap_values=result['shap_values'],
    prediction={'flare_probability': 0.0}
)
print(f'Explanation: {text}')
"
```
