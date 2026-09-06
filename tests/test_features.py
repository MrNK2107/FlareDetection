import pytest
import numpy as np
from src.features.dynamics import compute_derivatives, compute_rolling_stats, extract_dynamics_features
from src.features.cross_channel import compute_lag_correlation, compute_phase_difference, extract_cross_channel_features
from src.features.frequency import extract_frequency_features, compute_band_powers, compute_wavelet_energies
from src.features.changepoint import extract_changepoint_features, cusum_alarm, ruptures_breakpoints
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


# --------------------------- L4: frequency ---------------------------

def test_band_powers_sine_in_band():
    fs = 1.0
    t = np.arange(1200) / fs
    # 0.05 Hz sine -> inside band 2 (0.01-0.1 Hz)
    x = np.sin(2 * np.pi * 0.05 * t)
    bands = compute_band_powers(x, dt=fs)
    assert bands['fft_band2_power_frac'] > 0.5
    assert bands['fft_band1_power_frac'] < bands['fft_band2_power_frac']


def test_band_powers_constant_signal():
    bands = compute_band_powers(np.ones(300), dt=1.0)
    assert bands['fft_band1_power_frac'] == 0.0
    assert bands['spectral_entropy'] == 0.0


def test_spectral_entropy_white_noise_high():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1200)
    bands = compute_band_powers(x, dt=1.0)
    assert bands['spectral_entropy'] > 0.8


def test_wavelet_energy_dominant_scale():
    fs = 1.0
    t = np.arange(1200) / fs
    # 64s period sine -> dominant wavelet scale near 64s
    x = np.sin(2 * np.pi * t / 64.0)
    w = compute_wavelet_energies(x, dt=fs)
    assert w['wavelet_total_energy'] > 0
    assert 32.0 <= w['wavelet_dominant_scale_s'] <= 128.0


def test_extract_frequency_features_shape_and_nan_free():
    rng = np.random.default_rng(1)
    soft = rng.standard_normal(1200)
    hard = rng.standard_normal(1200)
    feats = extract_frequency_features(soft, hard, dt=1.0)
    assert feats.shape[1] == 16
    assert feats.notna().all().all()
    const = extract_frequency_features(np.ones(1200), np.ones(1200), dt=1.0)
    assert const.notna().all().all()


# --------------------------- L5: changepoint ---------------------------

def test_cusum_detects_step():
    x = np.concatenate([np.zeros(600), np.ones(600) * 3])
    res = cusum_alarm(x)
    assert res['cusum_alarmed'] == 1
    assert res['cusum_stat_max'] > 4.0


def test_cusum_quiet_series():
    rng = np.random.default_rng(2)
    res = cusum_alarm(rng.standard_normal(1200))
    assert res['cusum_alarmed'] == 0


def test_ruptures_detects_step():
    x = np.concatenate([np.zeros(600), np.ones(600) * 3])
    bkps = ruptures_breakpoints(x)
    assert len(bkps) >= 1
    assert abs(bkps[-1] - 600) <= 20


def test_ruptures_constant_series_no_breaks():
    assert ruptures_breakpoints(np.ones(1200)) == []


def test_extract_changepoint_features_nan_free_and_detects():
    x = np.concatenate([np.zeros(900), np.ones(300) * 3])
    feats = extract_changepoint_features(x, np.ones(1200), dt=1.0)
    assert feats.notna().all().all()
    assert feats['soft_n_breakpoints'].iloc[0] >= 1
    assert feats['soft_time_since_last_break_s'].iloc[0] <= 300


def test_full_layer_stack_shape():
    rng = np.random.default_rng(3)
    soft = rng.standard_normal(1200)
    hard = rng.standard_normal(1200)
    s = extract_features_for_window(soft, hard)
    # L1(8) + L2(22) + L3(6) + L4(16) + L5(12) = 64
    assert len(s) == 64
    assert s.notna().all()
