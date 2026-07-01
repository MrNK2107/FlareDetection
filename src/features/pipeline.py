import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional
from joblib import Parallel, delayed

from src.features.dynamics import extract_dynamics_features
from src.features.cross_channel import extract_cross_channel_features


def extract_l1_features(soft_flux_z: np.ndarray, hard_flux_z: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([{
        'soft_flux_mean': float(np.mean(soft_flux_z)),
        'soft_flux_std': float(np.std(soft_flux_z)),
        'soft_flux_min': float(np.min(soft_flux_z)),
        'soft_flux_max': float(np.max(soft_flux_z)),
        'hard_flux_mean': float(np.mean(hard_flux_z)),
        'hard_flux_std': float(np.std(hard_flux_z)),
        'hard_flux_min': float(np.min(hard_flux_z)),
        'hard_flux_max': float(np.max(hard_flux_z)),
    }])


def extract_features_for_window(
    soft_flux_z: np.ndarray,
    hard_flux_z: np.ndarray,
    dt: float = 1.0,
    rolling_windows_s: List[int] = None,
    max_lag_s: int = 120
) -> pd.Series:
    if rolling_windows_s is None:
        rolling_windows_s = [60, 300]
    l1 = extract_l1_features(soft_flux_z, hard_flux_z)
    l2 = extract_dynamics_features(soft_flux_z, hard_flux_z, dt, rolling_windows_s)
    l3 = extract_cross_channel_features(soft_flux_z, hard_flux_z, dt, max_lag_s)
    combined = pd.concat([l1, l2, l3], axis=1)
    return combined.iloc[0]


def extract_all_features(
    X_soft: np.ndarray,
    X_hard: np.ndarray,
    config_path: str = "config/config.yaml",
    n_jobs: int = -1
) -> pd.DataFrame:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    fc = config.get('features', {})
    rolling_windows = fc.get('rolling_windows_s', [60, 300])
    dt = 1.0
    max_lag_s = 120
    def extract_one(i):
        return extract_features_for_window(
            X_soft[i], X_hard[i], dt, rolling_windows, max_lag_s
        )
    n = len(X_soft)
    if n == 0:
        return pd.DataFrame()
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_one)(i) for i in range(n)
    )
    return pd.DataFrame(results)
