import pandas as pd
import numpy as np
from typing import Tuple


def remove_glitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    glitch = (
        (df.get('soft_quality_flag', pd.Series(0)).isin([1, 2])) |
        (df.get('hard_quality_flag', pd.Series(0)).isin([1, 2]))
    )
    return df[~glitch].copy()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if 'timestamp_utc' in df.columns:
        return df.drop_duplicates(subset='timestamp_utc', keep='first')
    return df[~df.index.duplicated(keep='first')]


def clip_to_physical_range(
    df: pd.DataFrame,
    soft_range: Tuple[float, float] = (1e-9, 1e-2),
    hard_range: Tuple[float, float] = (1e-10, 1e-3)
) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    n_clipped = 0
    if 'soft_flux' in df.columns:
        soft_mask = df['soft_flux'].notna() & (
            (df['soft_flux'] < soft_range[0]) | (df['soft_flux'] > soft_range[1])
        )
        n_clipped += soft_mask.sum()
        df['soft_flux'] = df['soft_flux'].clip(soft_range[0], soft_range[1])
    if 'hard_flux' in df.columns:
        hard_mask = df['hard_flux'].notna() & (
            (df['hard_flux'] < hard_range[0]) | (df['hard_flux'] > hard_range[1])
        )
        n_clipped += hard_mask.sum()
        df['hard_flux'] = df['hard_flux'].clip(hard_range[0], hard_range[1])
    return df, int(n_clipped)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df, _ = clip_to_physical_range(df)
    df = deduplicate(df)
    df = remove_glitch_rows(df)
    return df
