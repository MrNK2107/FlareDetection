import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from joblib import Parallel, delayed

from src.physics.stage_indicators import (
    extract_stage_indicators,
    compute_thresholds_from_training,
)
from src.physics.derived_ratios import extract_derived_ratios
from src.physics.historical_context import extract_historical_features


def compute_physics_features(
    X_soft: np.ndarray,
    X_hard: np.ndarray,
    dsoft_dt_all: Optional[np.ndarray] = None,
    d2hard_dt2_all: Optional[np.ndarray] = None,
    window_timestamps: Optional[List[pd.Timestamp]] = None,
    flare_history: Optional[List[pd.Timestamp]] = None,
    config: Optional[Dict] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    if dsoft_dt_all is None:
        dsoft_dt_all = np.gradient(X_soft, axis=1)
    if d2hard_dt2_all is None:
        dhard_dt_all = np.gradient(X_hard, axis=1)
        d2hard_dt2_all = np.gradient(dhard_dt_all, axis=1)
    thresholds = compute_thresholds_from_training(
        dsoft_dt_all.flatten(), d2hard_dt2_all.flatten()
    )
    def extract_one(i):
        soft = X_soft[i]
        hard = X_hard[i]
        dt = 1.0
        dsoft = np.gradient(soft, dt)
        dhard = np.gradient(hard, dt)
        d2hard = np.gradient(dhard, dt)
        features = {}
        si = extract_stage_indicators(
            dsoft, d2hard, hard,
            thresholds['dsoft_dt_threshold'],
            thresholds['d2hard_dt2_threshold'],
        )
        features.update(si)
        dr = extract_derived_ratios(soft, hard, dsoft, dhard)
        features.update(dr)
        if window_timestamps is not None and i < len(window_timestamps):
            hf = extract_historical_features(
                window_timestamps[i],
                flare_history or []
            )
            features.update(hf)
        else:
            features['time_since_last_flare_s'] = 1e6
            features['cycle_proxy'] = 0.5
        return features
    n = len(X_soft)
    if n == 0:
        return pd.DataFrame()
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_one)(i) for i in range(n)
    )
    return pd.DataFrame(results)
