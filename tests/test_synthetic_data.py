import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from src.data_generation.synthetic_flare_generator import SyntheticFlareGenerator


@pytest.fixture(scope="session")
def gen():
    return SyntheticFlareGenerator(overrides={
        'synthetic_data': {'total_days': 0.5, 'min_gap_s': 120, 'events_per_day': {'B': 30, 'C': 30, 'M': 30, 'X': 30}},
    })


@pytest.fixture(scope="session")
def small_df(gen):
    return gen.generate_dataset()


def test_quiet_baseline(gen, small_df):
    quiet = small_df[small_df.flare_class == 'None'].head(100)
    assert quiet['soft_flux'].mean() == pytest.approx(1e-7, rel=0.5)
    assert quiet['hard_flux'].mean() == pytest.approx(1e-8, rel=0.5)


def test_flare_peak_values(gen, small_df):
    for fc in ['B', 'C', 'M', 'X']:
        flare_rows = small_df[small_df.flare_class == fc]
        if len(flare_rows) > 0:
            low, high = gen.FLARE_PEAK_RANGES[fc]
            peak = flare_rows['soft_flux'].max()
            assert low * 0.5 <= peak <= high * 2, f"{fc}: peak {peak} not in range [{low}, {high}]"


def test_label_accuracy(gen, small_df):
    non_none = small_df[small_df.flare_class != 'None']
    assert len(non_none) > 0, "No flare events generated"
    for fc in ['B', 'C', 'M', 'X']:
        assert fc in non_none['flare_class'].values, f"Class {fc} not found"


def test_output_schema(gen, small_df):
    expected = {'timestamp_utc', 'soft_flux', 'hard_flux', 'soft_quality_flag', 'hard_quality_flag', 'flare_class'}
    assert expected.issubset(set(small_df.columns)), f"Missing columns: {expected - set(small_df.columns)}"


def test_quality_flags(gen, small_df):
    flagged = small_df[small_df['soft_quality_flag'] > 0]
    assert len(flagged) > 0, "No quality flags generated"
    assert flagged['soft_flux'].isna().any(), "Flagged rows should have NaN flux"


def test_flare_catalogue_exported(gen, small_df):
    cat = gen.get_flare_catalogue()
    assert len(cat) > 0, "Catalogue should contain events"
    assert {'flare_class', 'start_utc', 'peak_utc', 'end_utc', 'peak_soft_flux', 'duration_min'}.issubset(cat.columns)
    assert len(cat) == len(cat.drop_duplicates(subset=['start_utc', 'flare_class']))


def test_chunked_generation_reproducible(tmp_path):
    cfg = {
        'data': {'raw_dir': str(tmp_path / 'raw'), 'external_dir': str(tmp_path / 'ext')},
        'synthetic_data': {
            'total_days': 2, 'chunk_days': 1, 'seed': 7,
            'quiet_baseline_soft': 1e-7, 'quiet_baseline_hard': 1e-8,
            'noise_std_soft': 1e-9, 'noise_std_hard': 1e-10,
            'dropout_prob': 0.001,
            'events_per_day': {'B': 1, 'C': 1, 'M': 1, 'X': 0.5},
        },
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    g = SyntheticFlareGenerator(str(cfg_path))
    g.generate_dataset()
    cat1 = g.get_flare_catalogue()
    g2 = SyntheticFlareGenerator(str(cfg_path))
    g2.generate_dataset()
    cat2 = g2.get_flare_catalogue()
    pd.testing.assert_frame_equal(cat1, cat2)
    assert len(cat1) > 0


def test_chunked_file_exists_and_sized(tmp_path):
    cfg = {
        'data': {'raw_dir': str(tmp_path / 'raw'), 'external_dir': str(tmp_path / 'ext')},
        'synthetic_data': {
            'total_days': 2, 'chunk_days': 1, 'seed': 7,
            'quiet_baseline_soft': 1e-7, 'quiet_baseline_hard': 1e-8,
            'noise_std_soft': 1e-9, 'noise_std_hard': 1e-10,
            'dropout_prob': 0.001,
            'events_per_day': {'B': 1, 'C': 1, 'M': 1, 'X': 0.5},
        },
    }
    cfg_path = tmp_path / 'cfg.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    g = SyntheticFlareGenerator(str(cfg_path))
    g.generate_dataset()
    raw = Path(cfg['data']['raw_dir']) / 'training_data.parquet'
    assert raw.exists()
    n_rows = pd.read_parquet(raw, columns=['soft_flux']).shape[0]
    assert n_rows == 2 * 86400
