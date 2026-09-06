import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

CLASS_MAP = {'None': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


def _get_flux_columns(df: pd.DataFrame) -> Tuple[str, str]:
    soft_col = 'soft_flux_z' if 'soft_flux_z' in df.columns else 'soft_flux'
    hard_col = 'hard_flux_z' if 'hard_flux_z' in df.columns else 'hard_flux'
    if soft_col not in df.columns or hard_col not in df.columns:
        raise ValueError(
            "DataFrame must contain soft_flux_z (or soft_flux) and hard_flux_z (or hard_flux)"
        )
    return soft_col, hard_col


def _get_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if 'timestamp_utc' in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df['timestamp_utc']))
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    raise ValueError("DataFrame must have timestamp_utc column or DatetimeIndex")


def compute_window_starts(
    n_total: int,
    window_length_s: int,
    stride_s: int,
    forecast_horizon_s: int = 0,
) -> np.ndarray:
    last_start = n_total - window_length_s - forecast_horizon_s
    if last_start < 0:
        return np.array([], dtype=np.int64)
    return np.arange(0, last_start + 1, stride_s, dtype=np.int64)


def build_window_labels(
    df: pd.DataFrame,
    window_length_s: int = 1200,
    stride_s: int = 10,
    forecast_horizon_s: int = 1800,
    label_column: str = 'flare_class',
) -> np.ndarray:
    """Vectorized labeling: max flare class in the future window
    [window_end, window_end + horizon). Same semantics as the per-window loop,
    computed with a rolling max filter."""
    if label_column not in df.columns:
        return np.array([], dtype=np.int32)
    codes = df[label_column].map(CLASS_MAP).fillna(0).to_numpy(dtype=np.float64)
    n = len(codes)
    starts = compute_window_starts(n, window_length_s, stride_s, forecast_horizon_s)
    if len(starts) == 0:
        return np.array([], dtype=np.int32)
    rolling_max = pd.Series(codes).rolling(forecast_horizon_s, min_periods=1).max().to_numpy()
    j = starts + window_length_s + forecast_horizon_s - 1
    labels = np.nan_to_num(rolling_max[np.clip(j, 0, n - 1)], nan=0.0).astype(np.int32)
    return labels


def build_window_metadata(
    df: pd.DataFrame,
    window_length_s: int = 1200,
    stride_s: int = 10,
    forecast_horizon_s: int = 1800,
    labels: Optional[np.ndarray] = None,
    label_column: str = 'flare_class',
) -> Optional[pd.DataFrame]:
    idx = _get_index(df)
    n = len(df)
    starts = compute_window_starts(n, window_length_s, stride_s, forecast_horizon_s)
    if len(starts) == 0:
        return None
    meta = pd.DataFrame({
        'window_start': idx[starts],
        'window_end': idx[np.minimum(starts + window_length_s - 1, n - 1)],
        'forecast_end': idx[np.minimum(starts + window_length_s + forecast_horizon_s - 1, n - 1)],
    })
    if labels is not None:
        meta['label_code'] = labels.astype(int)
        meta['label'] = [CLASS_NAMES.get(int(l), 'None') for l in labels]
    elif label_column in df.columns:
        labels = build_window_labels(df, window_length_s, stride_s, forecast_horizon_s, label_column)
        meta['label_code'] = labels.astype(int)
        meta['label'] = [CLASS_NAMES.get(int(l), 'None') for l in labels]
    return meta


class WindowView:
    """Zero-copy strided view over a (n, 2) value array.

    Window k covers rows [k*stride, k*stride + window_length). Avoids
    materializing overlapping windows (which would need ~30 TB at 180d/1s/10s).
    """

    def __init__(
        self,
        values: np.ndarray,
        window_length_s: int,
        stride_s: int,
        forecast_horizon_s: int = 0,
    ):
        self.values = np.ascontiguousarray(values)
        self.window_length_s = window_length_s
        self.stride_s = stride_s
        self.starts = compute_window_starts(
            len(values), window_length_s, stride_s, forecast_horizon_s
        )

    def __len__(self) -> int:
        return len(self.starts)

    def get_soft(self, i: int) -> np.ndarray:
        s = self.starts[i]
        return self.values[s:s + self.window_length_s, 0]

    def get_hard(self, i: int) -> np.ndarray:
        s = self.starts[i]
        return self.values[s:s + self.window_length_s, 1]

    def get_pair(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        s = self.starts[i]
        block = self.values[s:s + self.window_length_s]
        return block[:, 0], block[:, 1]


def build_dl_windows(
    df: pd.DataFrame,
    out_dir: str = 'data/windows',
    window_length_s: int = 1200,
    stride_s: int = 120,
    cadence_s: int = 10,
    forecast_horizon_s: int = 1800,
    dtype=np.float32,
) -> dict:
    """Materialize raw DL windows: mean-pooled to `cadence_s` cadence, written
    as float32 memmaps (X_soft.npy / X_hard.npy / y_labels.npy) plus DL window
    metadata. These canonical names back the CLAUDE.md verification checks."""
    soft_col, hard_col = _get_flux_columns(df)
    data = df[[soft_col, hard_col]].to_numpy(dtype=np.float64)
    n = len(data)
    if n == 0:
        return {'n_windows': 0}
    starts = compute_window_starts(n, window_length_s, stride_s, forecast_horizon_s)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if len(starts) == 0:
        return {'n_windows': 0}
    n_steps = window_length_s // cadence_s
    X_soft = np.lib.format.open_memmap(
        out_path / 'X_soft.npy', mode='w+', dtype=dtype, shape=(len(starts), n_steps)
    )
    X_hard = np.lib.format.open_memmap(
        out_path / 'X_hard.npy', mode='w+', dtype=dtype, shape=(len(starts), n_steps)
    )
    labels = build_window_labels(df, window_length_s, stride_s, forecast_horizon_s)
    meta = build_window_metadata(df, window_length_s, stride_s, forecast_horizon_s, labels)
    pool = cadence_s
    for k in range(len(starts)):
        s = starts[k]
        block = data[s:s + window_length_s].reshape(n_steps, pool, 2)
        X_soft[k] = block[:, :, 0].mean(axis=1)
        X_hard[k] = block[:, :, 1].mean(axis=1)
    X_soft.flush()
    X_hard.flush()
    np.save(out_path / 'y_labels.npy', labels.astype(np.int32))
    if meta is not None:
        meta.to_parquet(out_path / 'dl_window_metadata.parquet', index=False)
    return {
        'n_windows': len(starts),
        'n_steps': n_steps,
        'cadence_s': cadence_s,
        'stride_s': stride_s,
    }


def generate_windows(
    df: pd.DataFrame,
    window_length_s: int = 1200,
    stride_s: int = 10,
    forecast_horizon_s: int = 1800,
    label_column: str = 'flare_class'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Legacy materializing windowing (used by unit tests / small data)."""
    if df.empty:
        return np.array([]), np.array([]), np.array([]), None
    soft_col, hard_col = _get_flux_columns(df)
    data = df[[soft_col, hard_col]].to_numpy(dtype=np.float64)
    n_total = len(data)
    starts = compute_window_starts(n_total, window_length_s, stride_s, forecast_horizon_s)
    if len(starts) == 0:
        return np.array([]), np.array([]), np.array([]), None
    labels = build_window_labels(df, window_length_s, stride_s, forecast_horizon_s, label_column)
    meta = build_window_metadata(df, window_length_s, stride_s, forecast_horizon_s, labels)
    has_nan = np.isnan(data).any()
    windows = []
    kept_labels = []
    kept_meta_rows = []
    for k, s in enumerate(starts):
        win = data[s:s + window_length_s]
        if has_nan and (np.isnan(win).any()):
            continue
        windows.append(win)
        kept_labels.append(labels[k])
        if meta is not None:
            kept_meta_rows.append(meta.iloc[k])
    if not windows:
        return np.array([]), np.array([]), np.array([]), None
    X = np.stack(windows)
    X_soft = np.ascontiguousarray(X[:, :, 0])
    X_hard = np.ascontiguousarray(X[:, :, 1])
    y = np.array(kept_labels, dtype=np.int32)
    kept_meta = pd.DataFrame(kept_meta_rows).reset_index(drop=True) if kept_meta_rows else None
    return X_soft, X_hard, y, kept_meta
