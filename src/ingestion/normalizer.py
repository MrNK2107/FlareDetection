import pandas as pd
import numpy as np


def rolling_zscore(
    series: pd.Series,
    window_hours: float = 6.0,
    min_periods: int = 60
) -> pd.Series:
    window = pd.Timedelta(hours=window_hours)
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    z = z.fillna(0.0)
    mean_z = z.mean()
    std_z = z.std()
    if abs(mean_z) > 0.01 or abs(std_z - 1) > 0.01:
        pass
    return z


def normalize(df: pd.DataFrame, window_hours: float = 6.0) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp_utc' in df.columns:
        ts = pd.to_datetime(df['timestamp_utc'])
        idx_df = df.set_index(ts)
    else:
        idx_df = df
    if 'soft_flux' in idx_df.columns:
        z_soft = rolling_zscore(idx_df['soft_flux'], window_hours)
        df['soft_flux_z'] = z_soft.values
    if 'hard_flux' in idx_df.columns:
        z_hard = rolling_zscore(idx_df['hard_flux'], window_hours)
        df['hard_flux_z'] = z_hard.values
    return df
