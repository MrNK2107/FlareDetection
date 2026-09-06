"""Evaluate model drift and retraining triggers (docs/08 §5.1).

Scores the most recent `--days` of feature windows with the production model
(models/ random forest by default) and compares rolling TSS against the
deployment baseline recorded in the model registry production metadata.

Usage: python scripts/evaluate_drift.py [--days 7] [--config config/config.yaml]
Exit code 1 if retraining is recommended (usable from cron/CI).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.monitoring import check_drift, save_drift_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=float, default=7.0)
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    processed_dir = Path(config['data']['processed_dir'])
    meta = pd.read_parquet(processed_dir / 'window_metadata.parquet')
    features = pd.read_parquet(processed_dir / 'full_feature_matrix.parquet')
    n = min(len(meta), len(features))
    meta, features = meta.iloc[:n].reset_index(drop=True), features.iloc[:n].reset_index(drop=True)

    ts = pd.DatetimeIndex(pd.to_datetime(meta['window_start']))
    cutoff = ts.max() - pd.Timedelta(days=args.days)
    recent = ts >= cutoff

    model_path = Path('models/random_forest.pkl')
    model = joblib.load(model_path)
    scaler_path = Path('models/logistic_regression_scaler.pkl')  # RF needs no scaler
    cols = [c for c in features.columns if c != 'window_start']
    X = np.nan_to_num(features[cols].to_numpy(dtype=np.float64), nan=0.0)

    proba = model.predict_proba(X[recent])
    y_score = proba[:, 1:].sum(axis=1)
    y_true = (meta['label_code'].to_numpy(dtype=int)[recent] > 0).astype(int)

    baseline = 0.0
    prod_path = Path('models/production.json')
    if prod_path.exists():
        with open(prod_path) as f:
            baseline = float(json.load(f).get('metrics', {}).get('tss', 0.0))

    new_labelled = int(y_true.sum())
    report = check_drift(
        ts[recent], y_true, y_score,
        baseline_tss=baseline,
        new_labelled_samples=new_labelled,
        days_since_last_retrain=None,  # registry timestamp fills this in production use
    )
    out = save_drift_report(report)
    print(f"Drift report written to {out}")
    print(f"  baseline_tss={report['baseline_tss']:.3f} worst_7d_tss={report['worst_tss']}")
    print(f"  alert_triggered={report['alert_triggered']} "
          f"scheduled_due={report['scheduled_check_due']} "
          f"volume_exceeded={report['volume_trigger_exceeded']}")
    if report['retrain_recommended']:
        print("RETRAINING RECOMMENDED")
        sys.exit(1)


if __name__ == "__main__":
    main()
