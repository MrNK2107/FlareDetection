import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.ingestion.synchronizer import synchronize
from src.ingestion.cleaner import clean
from src.ingestion.normalizer import normalize
from src.ingestion.windowing import (
    WindowView,
    build_window_labels,
    build_window_metadata,
    build_dl_windows,
    generate_windows,
)


def _load_raw(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    if 'flare_class' in df.columns:
        df['flare_class'] = df['flare_class'].astype('category')
    return df


def _select_training_windows(
    labels: np.ndarray,
    quiet_ratio: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Indices of all flare-bearing windows plus a random sample of quiet
    windows (ratio controls quiet:positive balance). Feature extraction and
    training then operate only on the selected subset."""
    rng = np.random.default_rng(seed)
    positive_idx = np.where(labels > 0)[0]
    quiet_idx = np.where(labels == 0)[0]
    n_quiet = min(len(quiet_idx), int(len(positive_idx) * quiet_ratio))
    quiet_sample = rng.choice(quiet_idx, size=n_quiet, replace=False) if n_quiet > 0 else np.array([], dtype=int)
    selected = np.sort(np.concatenate([positive_idx, quiet_sample]))
    return selected


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
    eval_cfg = config.get('evaluation', {})
    feature_stride_s = int(ing_cfg.get('feature_stride_s', 60))
    dl_stride_s = int(ing_cfg.get('dl_window_stride_s', 120))
    dl_cadence_s = int(ing_cfg.get('dl_internal_cadence_s', 10))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(window_output_dir).mkdir(parents=True, exist_ok=True)

    print(f"  Loading raw data from {input_path}")
    df = _load_raw(input_path)
    n_original = len(df)

    print(f"  Synchronizing timestamps at {ing_cfg['target_frequency_hz']} Hz")
    df = synchronize(df, target_frequency_hz=ing_cfg['target_frequency_hz'])
    print("  Cleaning data")
    df = clean(df)
    n_after_clean = len(df)
    missing_rate = 1.0 - (n_after_clean / n_original) if n_original > 0 else 0.0
    print(f"  Normalizing with {ing_cfg['z_score_window_h']}h rolling window")
    df = normalize(df, window_hours=ing_cfg['z_score_window_h'])
    processed_path = Path(output_dir) / 'synchronized_clean.parquet'
    df.to_parquet(processed_path, index=False)

    # --- Fine-cadence labels over the full dataset (spec stride, vectorized)
    stride_s = int(ing_cfg['window_stride_s'])
    labels_all = build_window_labels(
        df,
        window_length_s=ing_cfg['window_length_s'],
        stride_s=stride_s,
        forecast_horizon_s=ing_cfg['forecast_horizon_s'],
    )
    idx = pd.DatetimeIndex(pd.to_datetime(df['timestamp_utc']))
    n_rows = len(df)
    starts_all = np.arange(0, max(n_rows - ing_cfg['window_length_s'] - ing_cfg['forecast_horizon_s'] + 1, 0), stride_s)
    n_common = min(len(starts_all), len(labels_all))
    pd.DataFrame({
        'window_start': idx[starts_all[:n_common]],
        'label_code': labels_all[:n_common],
    }).to_parquet(Path(output_dir) / 'window_labels_all.parquet', index=False)
    print(f"  Computed {n_common:,} labels at {stride_s}s stride")

    # --- Candidate windows for feature extraction at the coarser stride
    labels_feat = build_window_labels(
        df,
        window_length_s=ing_cfg['window_length_s'],
        stride_s=feature_stride_s,
        forecast_horizon_s=ing_cfg['forecast_horizon_s'],
    )
    meta_feat = build_window_metadata(
        df,
        window_length_s=ing_cfg['window_length_s'],
        stride_s=feature_stride_s,
        forecast_horizon_s=ing_cfg['forecast_horizon_s'],
        labels=labels_feat,
    )
    quiet_ratio = int(eval_cfg.get('quiet_subsample_ratio', 8))
    selected = _select_training_windows(labels_feat, quiet_ratio=quiet_ratio)
    if len(selected) == 0:
        print("  WARNING: No windows generated!")
        return {
            'n_rows_original': n_original,
            'n_rows_after_cleaning': n_after_clean,
            'n_windows': 0,
            'n_selected_windows': 0,
            'class_distribution': {},
            'missing_data_rate': missing_rate,
        }

    # --- Selective feature extraction via zero-copy window views
    values = df[['soft_flux_z', 'hard_flux_z']].to_numpy(dtype=np.float64)
    view = WindowView(
        values,
        window_length_s=ing_cfg['window_length_s'],
        stride_s=feature_stride_s,
        forecast_horizon_s=ing_cfg['forecast_horizon_s'],
    )
    from src.features.pipeline import extract_features_batch
    print(f"  Extracting features for {len(selected):,} selected windows "
          f"(stride {feature_stride_s}s, quiet ratio {quiet_ratio})")
    features = extract_features_batch(view, selected, config=config)
    features.insert(0, 'window_start', meta_feat['window_start'].values[selected])
    features.to_parquet(Path(output_dir) / 'feature_matrix.parquet', index=False)

    selected_meta = meta_feat.iloc[selected].reset_index(drop=True)
    selected_meta.to_parquet(Path(output_dir) / 'window_metadata.parquet', index=False)

    # --- Physics features appended to the full matrix
    from src.physics.pipeline import compute_physics_features
    print("  Computing physics features")
    win_soft = np.stack([view.get_soft(i) for i in selected])
    win_hard = np.stack([view.get_hard(i) for i in selected])
    # Real flare history for time_since_last_flare_s (doc/04 §3): from the
    # generator's catalogue. Without it the feature degrades to a constant.
    flare_history = []
    catalogue_path = Path(config['data']['external_dir']) / 'flare_catalogue.parquet'
    if catalogue_path.exists():
        cat = pd.read_parquet(catalogue_path)
        if len(cat):
            flare_history = list(pd.DatetimeIndex(pd.to_datetime(cat['start_utc'])))
    physics_features = compute_physics_features(
        win_soft, win_hard,
        window_timestamps=list(selected_meta['window_start']),
        flare_history=flare_history,
        config=config,
    )
    full_features = pd.concat([features.reset_index(drop=True), physics_features.reset_index(drop=True)], axis=1)
    full_features.to_parquet(Path(output_dir) / 'full_feature_matrix.parquet', index=False)
    print(f"  -> {full_features.shape[1]} features for {full_features.shape[0]} windows")

    # --- DL windows (mean-pooled, float32 memmap)
    print(f"  Building DL windows (stride {dl_stride_s}s, cadence {dl_cadence_s}s)")
    dl_stats = build_dl_windows(
        df,
        out_dir=window_output_dir,
        window_length_s=ing_cfg['window_length_s'],
        stride_s=dl_stride_s,
        cadence_s=dl_cadence_s,
        forecast_horizon_s=ing_cfg['forecast_horizon_s'],
    )
    print(f"  -> {dl_stats.get('n_windows', 0):,} DL windows")

    class_dist = pd.Series(labels_feat[selected]).value_counts().sort_index().to_dict()
    return {
        'n_rows_original': n_original,
        'n_rows_after_cleaning': n_after_clean,
        'n_windows': int(len(labels_feat)),
        'n_selected_windows': int(len(selected)),
        'n_dl_windows': int(dl_stats.get('n_windows', 0)),
        'class_distribution': {str(k): int(v) for k, v in class_dist.items()},
        'missing_data_rate': missing_rate,
    }
