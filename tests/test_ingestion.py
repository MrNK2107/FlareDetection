import pytest
import numpy as np
import pandas as pd
from src.ingestion.synchronizer import synchronize
from src.ingestion.cleaner import clean, remove_glitch_rows, deduplicate, clip_to_physical_range
from src.ingestion.normalizer import normalize, rolling_zscore
from src.ingestion.windowing import generate_windows


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 3600
    return pd.DataFrame({
        'timestamp_utc': pd.date_range("2026-01-01", periods=n, freq='s'),
        'soft_flux': 1e-7 + rng.normal(0, 1e-9, n),
        'hard_flux': 1e-8 + rng.normal(0, 1e-10, n),
        'soft_quality_flag': np.zeros(n, dtype=int),
        'hard_quality_flag': np.zeros(n, dtype=int),
        'flare_class': ['None'] * n,
    })


def test_synchronize_basic(sample_df):
    result = synchronize(sample_df)
    assert 'timestamp_utc' in result.columns
    assert len(result) >= len(sample_df) * 0.99
    assert not result['soft_flux'].isna().all()


def test_synchronize_empty():
    empty = pd.DataFrame(columns=['timestamp_utc', 'soft_flux', 'hard_flux', 'soft_quality_flag', 'hard_quality_flag'])
    result = synchronize(empty)
    assert len(result) == 0


def test_clean_deduplicate(sample_df):
    duped = pd.concat([sample_df, sample_df.iloc[:5]])
    result = deduplicate(duped)
    assert len(result) == len(sample_df)


def test_clean_remove_glitch(sample_df):
    df = sample_df.copy()
    df.loc[0, 'soft_quality_flag'] = 2
    result = remove_glitch_rows(df)
    assert len(result) == len(df) - 1


def test_clean_clip(sample_df):
    df = sample_df.copy()
    df.loc[0, 'soft_flux'] = 100.0
    result, n = clip_to_physical_range(df)
    assert n >= 1
    assert result['soft_flux'].max() <= 1e-2


def test_normalize_basic(sample_df):
    result = normalize(sample_df)
    assert 'soft_flux_z' in result.columns
    assert 'hard_flux_z' in result.columns


def test_rolling_zscore_properties():
    idx = pd.date_range("2026-01-01", periods=10000, freq='s')
    series = pd.Series(np.random.randn(10000), index=idx)
    z = rolling_zscore(series, window_hours=0.5)
    assert abs(z.mean()) < 0.1
    assert abs(z.std() - 1.0) < 0.1


def test_windowing_basic(sample_df):
    normed = normalize(sample_df)
    X_soft, X_hard, y, meta = generate_windows(normed)
    if len(X_soft) == 0:
        pytest.skip("Not enough data for window generation with these params")
    assert X_soft.shape[0] == X_hard.shape[0] == len(y)
    assert X_soft.shape[1] == 1200


def test_windowing_empty():
    empty = pd.DataFrame(columns=['timestamp_utc', 'soft_flux', 'hard_flux', 'soft_quality_flag', 'hard_quality_flag'])
    X_soft, X_hard, y, meta = generate_windows(empty)
    assert len(X_soft) == 0
