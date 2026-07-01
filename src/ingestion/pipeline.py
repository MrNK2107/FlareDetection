import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.ingestion.synchronizer import synchronize
from src.ingestion.cleaner import clean
from src.ingestion.normalizer import normalize
from src.ingestion.windowing import generate_windows


def run_ingestion_pipeline(
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    window_output_dir: Optional[str] = None,
    config_path: str = "config/config.yaml"
) -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if input_path is None:
        input_path = str(Path(config['data']['raw_dir']) / 'training_data.parquet')
    if output_dir is None:
        output_dir = config['data']['processed_dir']
    if window_output_dir is None:
        window_output_dir = config['data']['windows_dir']
    ing_cfg = config['ingestion']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(window_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Loading raw data from {input_path}")
    df = pd.read_parquet(input_path)
    n_original = len(df)
    print(f"  Synchronizing timestamps at {ing_cfg['target_frequency_hz']} Hz")
    df = synchronize(df, target_frequency_hz=ing_cfg['target_frequency_hz'])
    print(f"  Cleaning data")
    df = clean(df)
    n_after_clean = len(df)
    missing_rate = 1.0 - (n_after_clean / n_original) if n_original > 0 else 0.0
    print(f"  Normalizing with {ing_cfg['z_score_window_h']}h rolling window")
    df = normalize(df, window_hours=ing_cfg['z_score_window_h'])
    processed_path = Path(output_dir) / 'synchronized_clean.parquet'
    df.to_parquet(processed_path, index=False)
    print(f"  Generating windows ({ing_cfg['window_length_s']}s window, {ing_cfg['window_stride_s']}s stride)")
    X_soft, X_hard, y, meta = generate_windows(
        df,
        window_length_s=ing_cfg['window_length_s'],
        stride_s=ing_cfg['window_stride_s'],
        forecast_horizon_s=ing_cfg['forecast_horizon_s'],
    )
    if len(X_soft) == 0:
        print("  WARNING: No windows generated!")
        return {
            'n_rows_original': n_original,
            'n_rows_after_cleaning': n_after_clean,
            'n_windows': 0,
            'class_distribution': {},
            'missing_data_rate': missing_rate,
        }
    np.save(Path(window_output_dir) / 'X_soft.npy', X_soft)
    np.save(Path(window_output_dir) / 'X_hard.npy', X_hard)
    np.save(Path(window_output_dir) / 'y_labels.npy', y)
    if meta is not None:
        meta.to_parquet(Path(output_dir) / 'window_metadata.parquet', index=False)
    class_dist = pd.Series(y).value_counts().sort_index().to_dict()
    return {
        'n_rows_original': n_original,
        'n_rows_after_cleaning': n_after_clean,
        'n_windows': len(X_soft),
        'class_distribution': {str(k): int(v) for k, v in class_dist.items()},
        'missing_data_rate': missing_rate,
    }
