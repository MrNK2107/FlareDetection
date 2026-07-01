import pytest
import numpy as np
import pandas as pd
from src.data_generation.synthetic_flare_generator import SyntheticFlareGenerator


@pytest.fixture
def gen():
    return SyntheticFlareGenerator()


def test_quiet_baseline(gen):
    df = gen.generate_dataset()
    quiet = df[df.flare_class == 'None'].head(100)
    assert quiet['soft_flux'].mean() == pytest.approx(1e-7, rel=0.5)
    assert quiet['hard_flux'].mean() == pytest.approx(1e-8, rel=0.5)


def test_flare_peak_values(gen):
    df = gen.generate_dataset()
    for fc in ['B', 'C', 'M', 'X']:
        flare_rows = df[df.flare_class == fc]
        if len(flare_rows) > 0:
            low, high = gen.FLARE_PEAK_RANGES[fc]
            peak = flare_rows['soft_flux'].max()
            assert low * 0.5 <= peak <= high * 2, f"{fc}: peak {peak} not in range [{low}, {high}]"


def test_label_accuracy(gen):
    df = gen.generate_dataset()
    non_none = df[df.flare_class != 'None']
    assert len(non_none) > 0, "No flare events generated"
    for fc in ['B', 'C', 'M', 'X']:
        assert fc in non_none['flare_class'].values, f"Class {fc} not found"


def test_output_schema(gen):
    df = gen.generate_dataset()
    expected = {'timestamp_utc', 'soft_flux', 'hard_flux', 'soft_quality_flag', 'hard_quality_flag', 'flare_class'}
    assert expected.issubset(set(df.columns)), f"Missing columns: {expected - set(df.columns)}"


def test_quality_flags(gen):
    df = gen.generate_dataset()
    flagged = df[df['soft_quality_flag'] > 0]
    assert len(flagged) > 0, "No quality flags generated"
    assert flagged['soft_flux'].isna().any(), "Flagged rows should have NaN flux"
