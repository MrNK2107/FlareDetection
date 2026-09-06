"""Lead-time prediction (docs/06 §4): minutes until flare peak.

Primary path is the Transformer lead-time head (see transformer.py). This
module provides the classical fallback: GradientBoostingRegressor trained on
flare windows with residual-quantile 90% CI. Replaces the previous placeholder
(base + random jitter)."""
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.ensemble import GradientBoostingRegressor

from src.models.splits import temporal_train_test_masks


def build_lead_time_dataset(
    processed_dir: str = "data/processed",
    external_dir: str = "data/external",
    test_days: float = 30.0,
) -> Optional[Dict]:
    processed_dir = Path(processed_dir)
    meta = pd.read_parquet(processed_dir / 'window_metadata.parquet')
    features = pd.read_parquet(processed_dir / 'full_feature_matrix.parquet')
    n = min(len(meta), len(features))
    meta, features = meta.iloc[:n].reset_index(drop=True), features.iloc[:n].reset_index(drop=True)
    catalogue_path = Path(external_dir) / 'flare_catalogue.parquet'
    if not catalogue_path.exists() or (meta['label_code'] > 0).sum() < 10:
        return None
    catalogue = pd.read_parquet(catalogue_path)
    ts = pd.DatetimeIndex(pd.to_datetime(meta['window_end']))
    # keep the returned dataset aligned with the adaptive masks
    peaks = pd.DatetimeIndex(pd.to_datetime(catalogue['peak_utc'])).sort_values()
    peak_arr = peaks.values.astype('datetime64[s]').astype(np.int64)
    ts_arr = ts.values.astype('datetime64[s]').astype(np.int64)
    next_peak_idx = np.searchsorted(peak_arr, ts_arr, side='left')
    valid = next_peak_idx < len(peak_arr)
    idx = np.where(valid)[0]
    minutes = (peak_arr[next_peak_idx[idx]] - ts_arr[idx]) / 60.0
    minutes = np.clip(minutes, 1.0, 1440.0)
    cols = [c for c in features.columns if c != 'window_start']
    X = features.iloc[idx][cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = minutes
    window_ts = ts[idx]
    train_mask, test_mask = temporal_train_test_masks(window_ts, test_days)
    return {
        'X': X, 'y': y, 'cols': cols,
        'train_idx': np.where(train_mask)[0],
        'test_idx': np.where(test_mask)[0],
    }


def train_lead_time(config_path: str = "config/config.yaml") -> Optional[Dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data = build_lead_time_dataset(
        processed_dir=config['data']['processed_dir'],
        external_dir=config['data']['external_dir'],
        test_days=float(config.get('evaluation', {}).get('test_days', 30)),
    )
    if data is None:
        print("    Lead-time training skipped (insufficient labelled windows)")
        return None
    X, y = data['X'], data['y']
    train_idx, test_idx = data['train_idx'], data['test_idx']
    if len(train_idx) < 10:
        print("    Lead-time training skipped (too few training windows)")
        return None
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X[train_idx], y[train_idx])
    residuals = np.abs(model.predict(X[train_idx]) - y[train_idx])
    ci_half = float(np.quantile(residuals, 0.90))
    metrics = {}
    if len(test_idx) >= 3:
        pred = model.predict(X[test_idx])
        metrics['lead_time_mae_min'] = float(np.mean(np.abs(pred - y[test_idx])))
    joblib.dump({
        'model': model,
        'feature_cols': data['cols'],
        'ci_half_min': ci_half,
        'metrics': metrics,
    }, 'models/lead_time.pkl')
    print(f"    Lead-time model trained (MAE={metrics.get('lead_time_mae_min', float('nan')):.1f} min)")
    return metrics


class LeadTimePredictor:
    def __init__(self, path: str = 'models/lead_time.pkl'):
        blob = joblib.load(path)
        self.model = blob['model']
        self.feature_cols = blob['feature_cols']
        self.ci_half = float(blob['ci_half_min'])

    def predict(self, features: np.ndarray, feature_names) -> Tuple[float, Tuple[float, float]]:
        lookup = {n: v for n, v in zip(feature_names, np.asarray(features).ravel())}
        vec = np.array([lookup.get(c, 0.0) for c in self.feature_cols], dtype=np.float64).reshape(1, -1)
        lead = float(self.model.predict(vec)[0])
        lead = float(np.clip(lead, 1.0, 1440.0))
        ci = [max(lead - self.ci_half, 0.0), lead + self.ci_half]
        return round(lead, 1), [round(ci[0], 1), round(ci[1], 1)]


if __name__ == "__main__":
    import sys
    train_lead_time(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
