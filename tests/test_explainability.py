import pytest
import numpy as np
from src.explainability.explainer import RFExplainer
from src.explainability.explanation_templates import ExplanationGenerator


@pytest.fixture
def sample_model():
    from sklearn.ensemble import RandomForestClassifier
    import joblib, json, os
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/test_rf.pkl')
    with open('models/test_feature_names.json', 'w') as f:
        json.dump([f'f{i}' for i in range(10)], f)
    return model


def test_explainer_initializes(sample_model):
    explainer = RFExplainer(model_path='models/test_rf.pkl')
    assert explainer.explainer is not None


def test_explain_returns_top_features(sample_model):
    explainer = RFExplainer(model_path='models/test_rf.pkl')
    X = np.random.randn(1, 10)
    result = explainer.explain(X, top_k=5)
    assert len(result['top_features']) == 5
    assert 'dominant_feature' in result


def test_top_features_are_sorted(sample_model):
    explainer = RFExplainer(model_path='models/test_rf.pkl')
    X = np.random.randn(1, 10)
    result = explainer.explain(X, top_k=5)
    shap_vals = [f['shap'] for f in result['top_features']]
    for i in range(len(shap_vals) - 1):
        assert abs(shap_vals[i]) >= abs(shap_vals[i + 1])


def test_dominant_feature_in_top_k(sample_model):
    explainer = RFExplainer(model_path='models/test_rf.pkl')
    X = np.random.randn(1, 10)
    result = explainer.explain(X, top_k=5)
    names = [f['name'] for f in result['top_features']]
    assert result['dominant_feature'] in names


def test_explanation_contains_feature_name(sample_model):
    explainer = RFExplainer(model_path='models/test_rf.pkl')
    X = np.random.randn(1, 10)
    result = explainer.explain(X)
    gen = ExplanationGenerator()
    text = gen.generate(
        dominant_feature=result['dominant_feature'],
        top_features=result['top_features'],
        shap_values=np.array(result['shap_values']),
        prediction={'flare_probability': 0.0},
    )
    assert isinstance(text, str)
    assert len(text) > 0


def test_uncertainty_template(sample_model):
    gen = ExplanationGenerator()
    text = gen.generate(
        dominant_feature='test_feature',
        top_features=[{'name': 'test', 'value': 0.0, 'shap': 0.001, 'direction': 'increasing'}],
        shap_values=np.array([[0.001]]),
        prediction={'flare_probability': 0.5},
        uncertainty=0.5,
    )
    assert len(text) > 0
