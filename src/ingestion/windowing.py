import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


def generate_windows(
    df: pd.DataFrame,
    window_length_s: int = 1200,
    stride_s: int = 10,
    forecast_horizon_s: int = 1800,
    label_column: str = 'flare_class'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    if df.empty:
        return np.array([]), np.array([]), np.array([]), None
    if 'timestamp_utc' in df.columns:
        ts = pd.to_datetime(df['timestamp_utc'])
        df_idx = df.set_index(ts)
    else:
        df_idx = df
    soft_col = 'soft_flux_z' if 'soft_flux_z' in df_idx.columns else ('soft_flux' if 'soft_flux' in df_idx.columns else None)
    hard_col = 'hard_flux_z' if 'hard_flux_z' in df_idx.columns else ('hard_flux' if 'hard_flux' in df_idx.columns else None)
    if soft_col is None or hard_col is None:
        raise ValueError("DataFrame must contain soft_flux_z (or soft_flux) and hard_flux_z (or hard_flux)")
    data = df_idx[[soft_col, hard_col]].values
    n_total = len(data)
    windows = []
    labels = []
    meta_rows = []
    class_map = {'None': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
    for start in range(0, n_total - window_length_s - forecast_horizon_s + 1, stride_s):
        end = start + window_length_s
        forecast_end = end + forecast_horizon_s
        future_labels = df_idx[label_column].iloc[end:forecast_end] if label_column in df_idx.columns else pd.Series(['None'])
        if future_labels.empty:
            continue
        non_none = future_labels[future_labels != 'None']
        if not non_none.empty:
            max_class = max(non_none, key=lambda x: class_map.get(x, 0))
        else:
            max_class = 'None'
        win_soft = data[start:end, 0].astype(np.float64)
        win_hard = data[start:end, 1].astype(np.float64)
        if np.isnan(win_soft).any() or np.isnan(win_hard).any():
            continue
        windows.append((win_soft, win_hard))
        labels.append(class_map.get(max_class, 0))
        if isinstance(df_idx.index, pd.DatetimeIndex):
            meta_rows.append({
                'window_start': df_idx.index[start],
                'window_end': df_idx.index[end - 1],
                'forecast_end': df_idx.index[min(forecast_end - 1, n_total - 1)],
                'label': max_class,
            })
    if not windows:
        return np.array([]), np.array([]), np.array([]), None
    X_soft = np.stack([w[0] for w in windows])
    X_hard = np.stack([w[1] for w in windows])
    y = np.array(labels, dtype=np.int32)
    meta = pd.DataFrame(meta_rows) if meta_rows else None
    return X_soft, X_hard, y, meta
