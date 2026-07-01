import numpy as np
import pandas as pd
from typing import Dict, List


def compute_derivatives(series: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
    d1 = np.gradient(series, dt)
    d2 = np.gradient(d1, dt)
    return {'d1': d1, 'd2': d2}


def compute_rolling_stats(
    series: np.ndarray,
    window_sizes_s: List[int] = [60, 300],
    dt: float = 1.0
) -> Dict[str, float]:
    result = {}
    for w in window_sizes_s:
        n = int(w / dt)
        if n < 1:
            continue
        if n >= len(series):
            n = len(series)
        arr = pd.Series(series)
        mean_val = float(arr.rolling(n, min_periods=1).mean().iloc[-1])
        std_val = float(arr.rolling(n, min_periods=1).std().iloc[-1])
        var_val = float(arr.rolling(n, min_periods=1).var().iloc[-1])
        result[f'mean_{w}s'] = mean_val
        result[f'std_{w}s'] = std_val
        result[f'var_{w}s'] = var_val if not np.isnan(var_val) else 0.0
    return result


def extract_dynamics_features(
    soft_flux_z: np.ndarray,
    hard_flux_z: np.ndarray,
    dt: float = 1.0,
    rolling_windows_s: List[int] = [60, 300]
) -> pd.DataFrame:
    sd = compute_derivatives(soft_flux_z, dt)
    hd = compute_derivatives(hard_flux_z, dt)
    features = {
        'dsoft_dt_mean': float(np.mean(sd['d1'])),
        'dsoft_dt_max': float(np.max(sd['d1'])),
        'dsoft_dt_std': float(np.std(sd['d1'])),
        'd2soft_dt2_mean': float(np.mean(sd['d2'])),
        'd2soft_dt2_max': float(np.max(sd['d2'])),
        'dhard_dt_mean': float(np.mean(hd['d1'])),
        'dhard_dt_max': float(np.max(hd['d1'])),
        'dhard_dt_std': float(np.std(hd['d1'])),
        'd2hard_dt2_mean': float(np.mean(hd['d2'])),
        'd2hard_dt2_max': float(np.max(hd['d2'])),
    }
    for prefix, arr in [('soft', soft_flux_z), ('hard', hard_flux_z)]:
        stats = compute_rolling_stats(arr, rolling_windows_s, dt)
        for key, val in stats.items():
            features[f'{prefix}_{key}'] = val
    return pd.DataFrame([features])
