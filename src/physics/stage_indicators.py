import numpy as np
from typing import Dict


def compute_soft_rising(
    dsoft_dt: np.ndarray,
    threshold: float,
    consecutive_windows: int = 3
) -> float:
    if len(dsoft_dt) < consecutive_windows:
        return 0.0
    above = dsoft_dt > threshold
    count = 0
    for val in above:
        if val:
            count += 1
            if count >= consecutive_windows:
                return 1.0
        else:
            count = 0
    return 0.0


def compute_hard_accelerating(
    d2hard_dt2: np.ndarray,
    threshold: float,
    consecutive_windows: int = 2
) -> float:
    if len(d2hard_dt2) < consecutive_windows:
        return 0.0
    above = d2hard_dt2 > threshold
    count = 0
    for val in above:
        if val:
            count += 1
            if count >= consecutive_windows:
                return 1.0
        else:
            count = 0
    return 0.0


def compute_precursor_candidate(
    soft_rising: float,
    hard_flux: np.ndarray,
    hard_quiet_baseline: float = 1e-8,
    threshold_factor: float = 1.5
) -> float:
    if soft_rising == 0:
        return 0.0
    hard_mean = float(np.mean(hard_flux))
    if hard_mean < threshold_factor * hard_quiet_baseline:
        return 1.0
    return 0.0


def extract_stage_indicators(
    dsoft_dt: np.ndarray,
    d2hard_dt2: np.ndarray,
    hard_flux: np.ndarray,
    dsoft_dt_threshold: float,
    d2hard_dt2_threshold: float,
    hard_quiet_baseline: float = 1e-8
) -> Dict[str, float]:
    sr = compute_soft_rising(dsoft_dt, dsoft_dt_threshold)
    ha = compute_hard_accelerating(d2hard_dt2, d2hard_dt2_threshold)
    pc = compute_precursor_candidate(sr, hard_flux, hard_quiet_baseline)
    return {
        'soft_rising': sr,
        'hard_accelerating': ha,
        'precursor_candidate': pc,
    }


def compute_thresholds_from_training(
    dsoft_dt_all: np.ndarray,
    d2hard_dt2_all: np.ndarray,
    percentile: float = 95.0
) -> Dict[str, float]:
    return {
        'dsoft_dt_threshold': float(np.percentile(dsoft_dt_all, percentile)),
        'd2hard_dt2_threshold': float(np.percentile(d2hard_dt2_all, percentile)),
    }
