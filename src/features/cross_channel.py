import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict


def compute_lag_correlation(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    max_lag_s: int = 120,
    dt: float = 1.0
) -> Dict:
    max_lag = min(max_lag_s, len(soft_flux) // 2)
    corr = np.correlate(soft_flux - soft_flux.mean(), hard_flux - hard_flux.mean(), mode='full')
    norm = np.std(soft_flux) * np.std(hard_flux) * len(soft_flux)
    if norm == 0:
        return {'peak_correlation': 0.0, 'peak_lag_s': 0.0, 'correlation_at_zero': 0.0}
    corr = corr / norm
    center = len(corr) // 2
    lags = np.arange(-max_lag, max_lag + 1) * dt
    lag_indices = np.arange(center - max_lag, center + max_lag + 1)
    lag_indices = lag_indices[(lag_indices >= 0) & (lag_indices < len(corr))]
    if len(lag_indices) == 0:
        return {'peak_correlation': 0.0, 'peak_lag_s': 0.0, 'correlation_at_zero': 0.0}
    lag_corr = corr[lag_indices]
    peak_idx = np.argmax(np.abs(lag_corr))
    return {
        'peak_correlation': float(lag_corr[peak_idx]),
        'peak_lag_s': float(lags[peak_idx]) if peak_idx < len(lags) else 0.0,
        'correlation_at_zero': float(corr[center]) if 0 <= center < len(corr) else 0.0,
    }


def compute_phase_difference(soft_flux: np.ndarray, hard_flux: np.ndarray) -> float:
    if len(soft_flux) < 3:
        return 0.0
    try:
        soft_analytic = signal.hilbert(soft_flux)
        hard_analytic = signal.hilbert(hard_flux)
        soft_phase = np.unwrap(np.angle(soft_analytic))
        hard_phase = np.unwrap(np.angle(hard_analytic))
        phase_diff = np.mean(soft_phase - hard_phase)
        return float(phase_diff)
    except Exception:
        return 0.0


def extract_cross_channel_features(
    soft_flux_z: np.ndarray,
    hard_flux_z: np.ndarray,
    dt: float = 1.0,
    max_lag_s: int = 120
) -> pd.DataFrame:
    lag_info = compute_lag_correlation(soft_flux_z, hard_flux_z, max_lag_s, dt)
    soft_orig = soft_flux_z
    hard_orig = hard_flux_z
    soft_orig = soft_orig - soft_orig.min() + 1e-12
    hard_orig = hard_orig - hard_orig.min() + 1e-12
    flux_ratio = float(np.mean(soft_orig / hard_orig))
    flux_diff = float(np.mean(soft_flux_z - hard_flux_z))
    phase_diff = compute_phase_difference(soft_flux_z, hard_flux_z)
    features = {
        'flux_ratio': flux_ratio,
        'flux_difference': flux_diff,
        'peak_correlation': lag_info['peak_correlation'],
        'peak_lag_s': lag_info['peak_lag_s'],
        'correlation_at_zero': lag_info['correlation_at_zero'],
        'phase_difference': phase_diff,
    }
    return pd.DataFrame([features])
