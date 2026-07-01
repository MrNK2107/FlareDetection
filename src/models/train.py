import numpy as np
import pandas as pd
import yaml
import json
import joblib
from pathlib import Path
from typing import Dict, Optional

from src.models.baselines import train_logistic_regression, train_random_forest
from src.models.evaluate import evaluate_model, save_evaluation_results


def prepare_training_data(
    window_dir: str = "data/windows",
    processed_dir: str = "data/processed",
    test_months: float = 2.0,
) -> Dict:
    window_dir = Path(window_dir)
    processed_dir = Path(processed_dir)
    X_soft = np.load(window_dir / 'X_soft.npy')
    X_hard = np.load(window_dir / 'X_hard.npy')
    y = np.load(window_dir / 'y_labels.npy')
    meta_path = processed_dir / 'window_metadata.parquet'
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        n_total = len(meta)
        n_test = int(n_total * test_months / 3.0)
        n_test = min(max(n_test, 1), n_total - 1)
        train_indices = np.arange(0, n_total - n_test)
        test_indices = np.arange(n_total - n_test, n_total)
    else:
        n_total = len(y)
        n_test = max(1, int(n_total * 0.3))
        train_indices = np.arange(0, n_total - n_test)
        test_indices = np.arange(n_total - n_test, n_total)
    feature_path = processed_dir / 'feature_matrix.parquet'
    if feature_path.exists():
        features = pd.read_parquet(feature_path)
        X = features.values
        feature_names = list(features.columns)
    else:
        X = np.column_stack([X_soft, X_hard])
        feature_names = [f'soft_{i}' for i in range(X_soft.shape[1])] + [f'hard_{i}' for i in range(X_hard.shape[1])]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': feature_names,
        'train_indices': train_indices, 'test_indices': test_indices,
    }


def train_all_baselines(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mc = config['models']
    data = prepare_training_data(
        window_dir=config['data']['windows_dir'],
        processed_dir=config['data']['processed_dir'],
    )
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
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
