import pytest
import numpy as np
import pandas as pd
from src.models.baselines import train_logistic_regression, train_random_forest
from src.models.evaluate import compute_tss, evaluate_model
from src.models.train import prepare_training_data


def _write_split_fixtures(tmp_path, n=2000, test_days=30.0):
    """Two 12h clusters of windows separated by test_days of dead time."""
    ts = list(pd.date_range("2026-01-01", periods=n // 2, freq="min"))
    ts += list(pd.date_range("2026-01-01 12:00", periods=n // 2, freq="min"))
    ts = [t + pd.Timedelta(days=60) for t in ts]  # push past a 30d test window
    ts = ts[: n // 2] + [t + pd.Timedelta(days=60) for t in pd.date_range("2026-03-05", periods=n // 2, freq="min")]
    meta = pd.DataFrame({
        "window_start": pd.DatetimeIndex(ts),
        "label_code": np.repeat([0, 1], n // 2),
        "label": ["None"] * (n // 2) + ["C"] * (n // 2),
    })
    rng = np.random.default_rng(0)
    feats = pd.DataFrame(rng.standard_normal((n, 4)), columns=[f"f{i}" for i in range(4)])
    meta.to_parquet(tmp_path / "window_metadata.parquet", index=False)
    feats.to_parquet(tmp_path / "full_feature_matrix.parquet", index=False)
    return tmp_path


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


def test_temporal_split_orders_correctly(tmp_path):
    root = _write_split_fixtures(tmp_path)
    data = prepare_training_data(
        window_dir=str(root), processed_dir=str(root), test_days=30.0
    )
    ts = data["timestamps"]
    assert ts[data["train_indices"]].max() < ts[data["test_indices"]].min()
    assert data["n_test_positives"] > 0


def test_temporal_split_empty_side_raises(tmp_path):
    root = _write_split_fixtures(tmp_path)
    with pytest.raises(ValueError):
        prepare_training_data(
            window_dir=str(root), processed_dir=str(root), test_days=500.0
        )


# --------------------------- LSTM baseline ---------------------------

import torch
from src.models.lstm import FlareLSTM, TorchModelWrapper


def test_lstm_forward_shape():
    model = FlareLSTM(hidden_size=16, num_layers=2, dropout=0.1)
    x = torch.randn(4, 20, 2)
    out = model(x)
    assert out.shape == (4, 5)


def test_lstm_overfits_tiny_batch():
    torch.manual_seed(0)
    model = FlareLSTM(hidden_size=16, num_layers=1, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(8, 10, 2)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    first_loss = None
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if first_loss is None:
            first_loss = float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < first_loss * 0.2


def test_lstm_wrapper_predict_proba():
    torch.manual_seed(0)
    model = FlareLSTM(hidden_size=8, num_layers=1, dropout=0.0)
    wrapper = TorchModelWrapper(model)
    X_flat = np.random.randn(3, 20 * 2)
    proba = wrapper.predict_proba(X_flat)
    assert proba.shape == (3, 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = wrapper.predict(X_flat)
    assert preds.shape == (3,)


# --------------------------- Transformer ---------------------------

from src.models.transformer import (
    DualStreamTransformer, TransformerWrapper, FocalLoss, _lead_targets,
)


def test_transformer_forward_shapes_and_attention():
    torch.manual_seed(0)
    model = DualStreamTransformer(n_engineered_features=64, d_model=32, nhead=4, num_layers=1)
    model.eval()  # disable attn dropout so attention rows sum to 1
    B, T = 2, 40
    with torch.no_grad():
        out = model(torch.randn(B, T), torch.randn(B, T), torch.randn(B, 64), return_attention=True)
    assert out['flare_prob'].shape == (B,)
    assert out['lead_log_mean'].shape == (B,)
    assert out['severity_logits'].shape == (B, 4)
    assert out['attention'].shape == (B, T, T)
    # rows average to ~1 (avg over heads, eval mode)
    assert torch.allclose(out['attention'].sum(dim=-1), torch.ones(B, T), atol=1e-3)
    assert model.last_attention is not None


def test_focal_loss_penalizes_easy_examples_less():
    torch.manual_seed(0)
    focal = FocalLoss(gamma=2.0, alpha=0.75)
    logits = torch.tensor([3.0, 3.0, -3.0, -3.0])  # confident & correct
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    easy = focal(logits, targets)
    hard_logits = torch.tensor([0.1, 0.1, -0.1, -0.1])
    hard = focal(hard_logits, targets)
    assert easy < hard


def test_focal_loss_value_against_manual():
    p = 0.8
    logits = torch.log(torch.tensor([p / (1 - p)]))  # inverse-sigmoid(0.8)
    targets = torch.tensor([1.0])
    loss = FocalLoss(gamma=2.0, alpha=0.75)(logits, targets)
    expected = 0.75 * (1 - 0.8) ** 2 * (-np.log(0.8))
    assert abs(float(loss) - expected) < 1e-5


def test_transformer_wrapper_proba_valid():
    torch.manual_seed(0)
    model = DualStreamTransformer(n_engineered_features=4, d_model=16, nhead=4, num_layers=1)
    wrapper = TransformerWrapper(model)
    B, T = 3, 20
    X_flat = np.random.randn(B, 2 * T + 4).astype(np.float32)
    proba = wrapper.predict_proba(X_flat)
    assert proba.shape == (B, 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba >= 0).all()


def test_lead_targets_from_catalogue():
    meta = pd.DataFrame({
        'window_end': pd.date_range("2026-01-01", periods=5, freq="10min"),
    })
    cat = pd.DataFrame({
        'peak_utc': pd.to_datetime(["2026-01-01 00:25:00", "2026-01-01 01:10:00"]),
        'flare_class': ['C', 'M'],
    })
    targets, mask = _lead_targets(meta, cat)
    assert mask[0]
    # window_end[0] = 00:00 -> next peak 00:25 => 25 min
    assert targets[0] == pytest.approx(25.0, abs=0.5)
    # last window_end 00:40 -> peak 01:10 => 30 min
    assert mask[-1]
    assert targets[-1] == pytest.approx(30.0, abs=0.5)


def test_transformer_overfits_tiny_batch():
    torch.manual_seed(0)
    model = DualStreamTransformer(n_engineered_features=4, d_model=16, nhead=2, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sb, hb = torch.randn(4, 20), torch.randn(4, 20)
    eb = torch.randn(4, 4)
    yb = torch.tensor([0.0, 1.0, 1.0, 0.0])
    first = None
    for _ in range(80):
        opt.zero_grad()
        out = model(sb, hb, eb)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out['flare_logit'], yb)
        if first is None:
            first = float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.3
