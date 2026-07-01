import pytest
import numpy as np
from src.features.dynamics import compute_derivatives, compute_rolling_stats, extract_dynamics_features
from src.features.cross_channel import compute_lag_correlation, compute_phase_difference, extract_cross_channel_features
from src.features.pipeline import extract_features_for_window, extract_l1_features


def test_derivative_on_linear():
    x = np.linspace(0, 10, 100)
    d = compute_derivatives(x)
    assert np.allclose(d['d1'].mean(), 10 / 99, atol=0.01)
    assert abs(d['d2'].mean()) < 0.01


def test_derivative_on_constant():
    x = np.ones(100)
    d = compute_derivatives(x)
    assert np.allclose(d['d1'], 0)
    assert np.allclose(d['d2'], 0)


def test_lag_correlation_identical():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    result = compute_lag_correlation(x, x, max_lag_s=50)
    assert abs(result['peak_correlation']) > 0.99
    assert abs(result['peak_lag_s']) < 1.0


def test_lag_correlation_shifted():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    shift = 20
    y = np.roll(x, shift)
    result = compute_lag_correlation(x, y, max_lag_s=50)
    assert abs(result['peak_lag_s']) == pytest.approx(shift, abs=5) or abs(abs(result['peak_lag_s']) - shift) < 5


def test_l1_features():
    soft = np.random.randn(1200)
    hard = np.random.randn(1200)
    l1 = extract_l1_features(soft, hard)
    assert l1.shape[1] == 8
    assert 'soft_flux_mean' in l1.columns
    assert 'hard_flux_mean' in l1.columns


def test_dynamics_features():
    soft = np.random.randn(1200)
    hard = np.random.randn(1200)
    dyn = extract_dynamics_features(soft, hard)
    assert dyn.shape[1] >= 16


def test_cross_channel_features():
    soft = np.random.randn(1200)
    hard = np.roll(soft, 10)
    cc = extract_cross_channel_features(soft, hard)
    assert 'peak_correlation' in cc.columns
    assert 'peak_lag_s' in cc.columns


def test_feature_pipeline_shape():
    soft = np.random.randn(10, 1200)
    hard = np.random.randn(10, 1200)
    features = extract_features_for_window(soft[0], hard[0])
    assert len(features) >= 20


def test_phase_difference():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    y = np.sin(np.linspace(0, 4 * np.pi, 200) + np.pi / 4)
    phase = compute_phase_difference(x, y)
    assert isinstance(phase, float)
