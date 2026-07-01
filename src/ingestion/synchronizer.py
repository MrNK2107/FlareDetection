import pandas as pd
import numpy as np
from typing import Tuple


def synchronize(
    df: pd.DataFrame,
    target_frequency_hz: float = 1.0,
    gap_threshold_s: float = 5.0
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            'timestamp_utc', 'soft_flux', 'hard_flux',
            'soft_quality_flag', 'hard_quality_flag'
        ])
    df = df.copy()
    if 'timestamp_utc' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df = df.set_index('timestamp_utc').sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        raise ValueError("Input must have 'timestamp_utc' column or DatetimeIndex")
    target_interval = pd.Timedelta(seconds=1.0 / target_frequency_hz)
    agg_dict = {
        'soft_flux': 'mean',
        'hard_flux': 'mean',
        'soft_quality_flag': 'max',
        'hard_quality_flag': 'max',
    }
    if 'flare_class' in df.columns:
        def keep_last_label(series):
            non_none = series[series != 'None']
            return non_none.iloc[-1] if len(non_none) > 0 else 'None'
        agg_dict['flare_class'] = keep_last_label
    resampled = df.resample(target_interval).agg(agg_dict)
    resampled.index.name = 'timestamp_utc'
    resampled = flag_gap_regions(resampled, gap_threshold_s)
    gap_mask = resampled['_gap_duration'] > gap_threshold_s
    resampled['soft_quality_flag'] = resampled['soft_quality_flag'].fillna(0).astype(int)
    resampled['hard_quality_flag'] = resampled['hard_quality_flag'].fillna(0).astype(int)
    resampled.loc[gap_mask, 'soft_quality_flag'] = -1
    resampled.loc[gap_mask, 'hard_quality_flag'] = -1
    resampled[['soft_flux', 'hard_flux']] = resampled[['soft_flux', 'hard_flux']].ffill()
    resampled = resampled.drop(columns=['_gap_duration'], errors='ignore')
    resampled = resampled.reset_index()
    return resampled


def flag_gap_regions(
    df: pd.DataFrame,
    gap_threshold_s: float = 5.0
) -> pd.DataFrame:
    df = df.copy()
    df['_gap_duration'] = 0.0
    if len(df) < 2:
        return df
    soft_nan = df['soft_flux'].isna()
    hard_nan = df['hard_flux'].isna()
    any_nan = soft_nan | hard_nan
    if not any_nan.any():
        return df
    gap_groups = (any_nan != any_nan.shift(1).fillna(False)).cumsum()
    gap_lengths = any_nan.groupby(gap_groups).transform('sum') * (
        (df.index[1] - df.index[0]).total_seconds() if len(df) > 1 else 1.0
    )
    df['_gap_duration'] = gap_lengths.fillna(0.0)
    return df
