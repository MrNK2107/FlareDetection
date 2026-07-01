import pytest
import numpy as np
from src.physics.stage_indicators import (
    compute_soft_rising, compute_hard_accelerating,
    compute_precursor_candidate, extract_stage_indicators,
)
from src.physics.derived_ratios import (
    compute_thermal_fraction, compute_nonthermal_index,
    extract_derived_ratios,
)
from src.physics.historical_context import (
    compute_time_since_last_flare, compute_cycle_proxy,
)
import pandas as pd


def test_soft_rising_detection():
    dsoft = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    result = compute_soft_rising(dsoft, threshold=0.05, consecutive_windows=3)
    assert result == 1.0


def test_soft_rising_quiet():
    dsoft = np.zeros(100)
    result = compute_soft_rising(dsoft, threshold=0.5, consecutive_windows=3)
    assert result == 0.0


def test_precursor_detection():
    sr = 1.0
    hard = np.array([1e-9, 1e-9, 1e-9])
    result = compute_precursor_candidate(sr, hard, hard_quiet_baseline=1e-8)
    assert result == 1.0


def test_precursor_not_during_flare():
    sr = 1.0
    hard = np.array([1e-5, 1e-5, 1e-5])
    result = compute_precursor_candidate(sr, hard, hard_quiet_baseline=1e-8)
    assert result == 0.0


def test_thermal_fraction_quiet():
    soft = np.array([1e-7, 1e-7, 1e-7])
    hard = np.array([1e-8, 1e-8, 1e-8])
    tf = compute_thermal_fraction(soft, hard)
    assert 0.8 < tf < 1.0


def test_thermal_fraction_during_flare():
    soft = np.array([1e-5, 2e-5, 3e-5])
    hard = np.array([1e-5, 3e-5, 5e-5])
    tf = compute_thermal_fraction(soft, hard)
    assert 0.3 < tf < 0.8


def test_nonthermal_index():
    dhard = np.array([0, 1, 5, 10])
    dsoft = np.array([0, 0.1, 0.2, 0.3])
    ni = compute_nonthermal_index(dhard, dsoft)
    assert ni > 0


def test_time_since_last_flare():
    ts = pd.Timestamp("2026-01-02 12:00:00")
    flares = [pd.Timestamp("2026-01-01 00:00:00")]
    seconds = compute_time_since_last_flare(ts, flares)
    assert abs(seconds - 129600) < 10


def test_cycle_proxy_range():
    ts = pd.Timestamp("2026-01-01")
    proxy = compute_cycle_proxy(ts)
    assert 0.0 <= proxy <= 1.0
