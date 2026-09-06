"""Temporal split helpers shared by all model trainers.

CRITICAL RULE (docs/06): temporal splits only — the test set is the LAST
`test_days` of window time. Never random split (leaks future into training).
"""
import numpy as np
import pandas as pd


def adaptive_split_time(
    ts: pd.DatetimeIndex,
    test_days: float,
    min_train_fraction: float = 0.5,
) -> tuple:
    """Return (split_time, actual_test_days) for a temporal split.

    If the requested `test_days` exceeds the data span enough to empty the
    train side, fall back to a split that keeps at least
    `min_train_fraction` of the span for training, with a printed warning.
    """
    ts = pd.DatetimeIndex(ts)
    t_min, t_max = ts.min(), ts.max()
    span_days = (t_max - t_min).total_seconds() / 86400.0
    max_test_days = span_days * (1.0 - min_train_fraction)
    if span_days > 0 and test_days > max_test_days:
        actual = max(max_test_days, 0.0)
        print(
            f"  WARNING: requested test_days={test_days:g} exceeds "
            f"{1 - min_train_fraction:.0%} of the {span_days:.1f}-day data span; "
            f"using {actual:.2f} days to keep a non-empty train split"
        )
        return t_max - pd.Timedelta(days=actual), actual
    return t_max - pd.Timedelta(days=test_days), test_days


def temporal_train_test_masks(
    ts: pd.DatetimeIndex,
    test_days: float,
    min_train_fraction: float = 0.5,
) -> tuple:
    """Boolean (train_mask, test_mask) over `ts`, adaptive when the span is short."""
    split_time, _ = adaptive_split_time(ts, test_days, min_train_fraction)
    train_mask = np.asarray(ts < split_time)
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Temporal split produced an empty side (train={int(train_mask.sum())}, "
            f"test={int(test_mask.sum())}); dataset too short"
        )
    return train_mask, test_mask
