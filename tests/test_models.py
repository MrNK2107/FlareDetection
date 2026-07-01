import pytest
import numpy as np
from src.models.baselines import train_logistic_regression, train_random_forest
from src.models.evaluate import compute_tss, evaluate_model


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 10))
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    return X, y


def test_logistic_regression_trains(sample_data):
    X, y = sample_data
    model, scaler = train_logistic_regression(X, y)
    assert hasattr(model, 'predict')
    preds = model.predict(scaler.transform(X))
    assert len(preds) == len(y)


def test_random_forest_trains(sample_data):
    X, y = sample_data
    model = train_random_forest(X, y)
    assert hasattr(model, 'predict')
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_tss_range(sample_data):
    X, y = sample_data
    model, scaler = train_logistic_regression(X, y)
    results = evaluate_model(model, X, y, "LR", scaler=scaler)
    assert -1.0 <= results['tss'] <= 1.0


def test_brier_score_range(sample_data):
    X, y = sample_data
    model, scaler = train_logistic_regression(X, y)
    results = evaluate_model(model, X, y, "LR", scaler=scaler)
    assert 0.0 <= results['brier_score'] <= 1.0


def test_majority_class_baseline():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    model, _ = train_logistic_regression(X, y)
    results = evaluate_model(model, X, y, "LR")
    assert results['tss'] is not None


def test_model_save_load(sample_data, tmp_path):
    import joblib
    X, y = sample_data
    model = train_random_forest(X, y)
    path = tmp_path / "test_model.pkl"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    preds_orig = model.predict(X)
    preds_loaded = loaded.predict(X)
    assert np.array_equal(preds_orig, preds_loaded)
