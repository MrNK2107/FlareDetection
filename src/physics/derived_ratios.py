import numpy as np
from typing import Dict


def compute_thermal_fraction(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    epsilon: float = 1e-12,
    smoothing_s: int = 30
) -> float:
    denominator = soft_flux + hard_flux + epsilon
    fraction = soft_flux / denominator
    if smoothing_s > 1 and len(fraction) > smoothing_s:
        fraction = np.convolve(fraction, np.ones(smoothing_s) / smoothing_s, mode='valid')
    return float(np.mean(fraction))


def compute_nonthermal_index(
    dhard_dt: np.ndarray,
    dsoft_dt: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    denominator = np.abs(dsoft_dt) + epsilon
    index = dhard_dt / denominator
    index = np.clip(index, 0, 100)
    return float(np.mean(index))


def extract_derived_ratios(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    dsoft_dt: np.ndarray,
    dhard_dt: np.ndarray
) -> Dict[str, float]:
    return {
        'thermal_fraction': compute_thermal_fraction(soft_flux, hard_flux),
        'nonthermal_index': compute_nonthermal_index(dhard_dt, dsoft_dt),
    }
