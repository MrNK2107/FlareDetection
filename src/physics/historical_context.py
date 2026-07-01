import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def compute_time_since_last_flare(
    window_timestamp: pd.Timestamp,
    flare_timestamps: List[pd.Timestamp]
) -> float:
    if not flare_timestamps:
        return 1e6
    prior = [t for t in flare_timestamps if t <= window_timestamp]
    if not prior:
        return 1e6
    most_recent = max(prior)
    return (window_timestamp - most_recent).total_seconds()


def compute_cycle_proxy(
    window_timestamp: pd.Timestamp,
    solar_max_date: pd.Timestamp = pd.Timestamp("2025-07-01"),
    cycle_length_years: float = 11.0
) -> float:
    cycle_days = cycle_length_years * 365.25
    days_since_max = (window_timestamp - solar_max_date).total_seconds() / 86400
    proxy = days_since_max / cycle_days + 0.5
    return float(np.clip(proxy, 0.0, 1.0))


def extract_historical_features(
    window_timestamp: pd.Timestamp,
    flare_history: List[pd.Timestamp]
) -> Dict[str, float]:
    return {
        'time_since_last_flare_s': compute_time_since_last_flare(window_timestamp, flare_history),
        'cycle_proxy': compute_cycle_proxy(window_timestamp),
    }
