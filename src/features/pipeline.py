import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor

from src.features.dynamics import extract_dynamics_features
from src.features.cross_channel import extract_cross_channel_features

try:
    from src.features.frequency import extract_frequency_features
    _HAS_L4 = True
except ImportError:
    _HAS_L4 = False

try:
    from src.features.changepoint import extract_changepoint_features
    _HAS_L5 = True
except ImportError:
    _HAS_L5 = False


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
    max_lag_s: int = 120,
    layers: Optional[List[str]] = None,
) -> pd.Series:
    if rolling_windows_s is None:
        rolling_windows_s = [60, 300]
    if layers is None:
        layers = ['L1', 'L2', 'L3', 'L4', 'L5']
    l1 = extract_l1_features(soft_flux_z, hard_flux_z)
    frames = [l1]
    if 'L2' in layers:
        frames.append(extract_dynamics_features(soft_flux_z, hard_flux_z, dt, rolling_windows_s))
    if 'L3' in layers:
        frames.append(extract_cross_channel_features(soft_flux_z, hard_flux_z, dt, max_lag_s))
    if 'L4' in layers and _HAS_L4:
        frames.append(extract_frequency_features(soft_flux_z, hard_flux_z, dt))
    if 'L5' in layers and _HAS_L5:
        frames.append(extract_changepoint_features(soft_flux_z, hard_flux_z, dt))
    combined = pd.concat(frames, axis=1)
    return combined.iloc[0]


def _extract_one(args) -> "pd.Series":
    (soft, hard), opts = args
    return extract_features_for_window(soft, hard, **opts)


def extract_features_batch(
    view,
    indices: np.ndarray,
    config_path: str = "config/config.yaml",
    config: Optional[Dict] = None,
    n_jobs: int = -1,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Extract features for the given window indices from a WindowView.

    Windows are passed to workers as zero-copy buffer views (no memmap forks,
    no pickling of the source array)."""
    if config is None:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    fc = config.get('features', {})
    layers = fc.get('layers', ['L1', 'L2', 'L3', 'L4', 'L5'])
    rolling_windows = fc.get('rolling_windows_s', [60, 300])
    max_lag_s = int(fc.get('max_lag_s', 120))
    opts = {
        'dt': 1.0,
        'rolling_windows_s': rolling_windows,
        'max_lag_s': max_lag_s,
        'layers': layers,
    }
    n = len(indices)
    if n == 0:
        return pd.DataFrame()
    jobs = [((view.get_pair(i)), opts) for i in indices]
    if n_jobs == 0 or n < 8:
        results = [_extract_one(j) for j in jobs]
    else:
        workers = min(max_workers, n)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_extract_one, jobs, chunksize=max(1, n // (workers * 4))))
    return pd.DataFrame(results)
