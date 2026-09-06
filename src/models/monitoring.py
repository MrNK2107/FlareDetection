"""Continuous learning pipeline (docs/08 §5.1): retraining triggers.

- Scheduled: monthly rolling 30-day TSS compared against deployment baseline.
- Alert-triggered: TSS drop > 0.1 below baseline on any 7-day window.
- Data volume: > 500 new labelled flare samples since last registration.

The core functions operate on plain arrays (y_true / y_score / timestamps) so
they are unit-testable; scripts/evaluate_drift.py wires them to real data.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def tss_from_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr - fpr)


def rolling_tss(
    ts: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_score: np.ndarray,
    window_days: float = 7.0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """TSS over consecutive non-overlapping `window_days` periods."""
    df = pd.DataFrame({'ts': pd.DatetimeIndex(ts), 'y_true': y_true, 'y_score': y_score})
    df = df.sort_values('ts').reset_index(drop=True)
    rows = []
    if df.empty:
        return pd.DataFrame(columns=['period_start', 'period_end', 'tss', 'n'])
    period_start = df['ts'].iloc[0]
    while period_start < df['ts'].iloc[-1]:
        period_end = period_start + pd.Timedelta(days=window_days)
        chunk = df[(df['ts'] >= period_start) & (df['ts'] < period_end)]
        if len(chunk):
            rows.append({
                'period_start': period_start,
                'period_end': period_end,
                'tss': tss_from_scores(chunk['y_true'].values, chunk['y_score'].values, threshold),
                'n': len(chunk),
            })
        period_start = period_end
    return pd.DataFrame(rows)


def check_drift(
    ts,
    y_true: np.ndarray,
    y_score: np.ndarray,
    baseline_tss: float,
    drop_threshold: float = 0.1,
    window_days: float = 7.0,
    days_since_last_retrain: Optional[float] = None,
    scheduled_interval_days: float = 30.0,
    new_labelled_samples: int = 0,
    volume_trigger: int = 500,
    decision_threshold: float = 0.5,
) -> Dict:
    """Evaluate the three retraining triggers (docs/08 §5.1)."""
    rt = rolling_tss(pd.DatetimeIndex(ts), y_true, y_score,
                     window_days=window_days, threshold=decision_threshold)
    windows = rt.to_dict('records')
    alert_triggered = any(
        w['n'] > 0 and (baseline_tss - w['tss']) > drop_threshold for w in windows
    )
    scheduled_due = (
        days_since_last_retrain is None
        or days_since_last_retrain >= scheduled_interval_days
    )
    volume_exceeded = new_labelled_samples > volume_trigger
    report = {
        'baseline_tss': float(baseline_tss),
        'rolling_windows': [
            {**w,
             'period_start': str(w['period_start']),
             'period_end': str(w['period_end'])}
            for w in windows
        ],
        'worst_tss': float(rt['tss'].min()) if len(rt) else None,
        'alert_triggered': bool(alert_triggered),
        'scheduled_check_due': bool(scheduled_due),
        'volume_trigger_exceeded': bool(volume_exceeded),
        'new_labelled_samples': int(new_labelled_samples),
        'retrain_recommended': bool(alert_triggered or scheduled_due or volume_exceeded),
    }
    return report


def save_drift_report(report: Dict, path: str = "data/processed/drift_report.json") -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    return path
