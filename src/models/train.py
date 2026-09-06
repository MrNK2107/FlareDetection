import numpy as np
import pandas as pd
import yaml
import json
import joblib
from pathlib import Path
from typing import Dict, Optional

from src.models.baselines import train_logistic_regression, train_random_forest
from src.models.evaluate import evaluate_model, save_evaluation_results
from src.models.splits import adaptive_split_time, temporal_train_test_masks


def prepare_training_data(
    window_dir: str = "data/windows",
    processed_dir: str = "data/processed",
    test_days: float = 30.0,
    features_filename: str = "full_feature_matrix.parquet",
    metadata_filename: str = "window_metadata.parquet",
) -> Dict:
    """Load features + labels and perform a timestamp-based temporal split.

    CRITICAL: temporal split only — the test set is the LAST `test_days` of
    window time. Never random split (leaks future into training).
    """
    processed_dir = Path(processed_dir)
    meta_path = processed_dir / metadata_filename
    feature_path = processed_dir / features_filename
    if not meta_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            f"Need {meta_path} and {feature_path}; run the ingestion pipeline first"
        )
    meta = pd.read_parquet(meta_path)
    features = pd.read_parquet(feature_path)
    n = min(len(meta), len(features))
    meta = meta.iloc[:n].reset_index(drop=True)
    features = features.iloc[:n].reset_index(drop=True)
    feature_names = [c for c in features.columns if c != 'window_start']
    X = features[feature_names].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = meta['label_code'].to_numpy(dtype=np.int32) if 'label_code' in meta.columns else np.zeros(n, dtype=np.int32)
    ts = pd.DatetimeIndex(pd.to_datetime(meta['window_start']))
    split_time, _ = adaptive_split_time(ts, test_days)
    train_mask, test_mask = temporal_train_test_masks(ts, test_days)
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    assert ts[train_indices].max() < ts[test_indices].min(), "Temporal split violated"
    n_test_pos = int((y[test_indices] > 0).sum())
    if n_test_pos == 0:
        print("  WARNING: test period has no positive windows; TSS will be 0")
    return {
        'X_train': X[train_indices],
        'X_test': X[test_indices],
        'y_train': y[train_indices],
        'y_test': y[test_indices],
        'feature_names': feature_names,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'timestamps': ts,
        'split_time': split_time,
        'n_test_positives': n_test_pos,
    }


def train_all_baselines(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config['models']
    eval_cfg = config.get('evaluation', {})
    data = prepare_training_data(
        window_dir=config['data']['windows_dir'],
        processed_dir=config['data']['processed_dir'],
        test_days=float(eval_cfg.get('test_days', 30)),
    )
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
    print(f"  Split at {data['split_time']}: "
          f"{len(X_train):,} train / {len(X_test):,} test "
          f"({data['n_test_positives']} test positives)")
    results = {}
    print("  Training Logistic Regression...")
    lr_model, lr_scaler = train_logistic_regression(
        X_train, y_train,
        class_weight=mc['logistic_regression']['class_weight'],
        max_iter=mc['logistic_regression']['max_iter'],
    )
    lr_results = evaluate_model(lr_model, X_test, y_test, "LogisticRegression", scaler=lr_scaler)
    results['LogisticRegression'] = lr_results
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(lr_scaler, 'models/logistic_regression_scaler.pkl')
    print(f"    TSS={lr_results['tss']:.4f}, Brier={lr_results['brier_score']:.4f}")
    print("  Training Random Forest...")
    rf_model = train_random_forest(
        X_train, y_train,
        n_estimators=mc['random_forest']['n_estimators'],
        max_depth=mc['random_forest']['max_depth'],
        class_weight=mc['random_forest']['class_weight'],
    )
    rf_results = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    results['RandomForest'] = rf_results
    joblib.dump(rf_model, 'models/random_forest.pkl')
    print(f"    TSS={rf_results['tss']:.4f}, Brier={rf_results['brier_score']:.4f}")
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    if rf_results.get('feature_importance'):
        fi_sorted = sorted(
            rf_results['feature_importance'].items(),
            key=lambda x: x[1], reverse=True
        )
        fi_df = pd.DataFrame([
            {'feature': feature_names[int(k.split('_')[1])] if k.startswith('feature_') else k,
             'importance': v}
            for k, v in fi_sorted[:10]
        ])
        fi_df.to_csv('models/feature_importance.csv', index=False)
    save_evaluation_results(results)
    with open('models/decision_threshold.json', 'w') as f:
        json.dump({'threshold': 0.5}, f)
    return results


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    train_all_baselines(config)
