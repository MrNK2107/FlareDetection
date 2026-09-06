import numpy as np
import pandas as pd
import pytest
import joblib

from src.models.state_machine import (
    assign_state_names, STATE_NAMES, StateMachineInference,
)
from hmmlearn import hmm


def _state_means_df():
    """Six states with distinct, physically interpretable profiles."""
    return pd.DataFrame({
        'soft_flux_max':   [0.1, 0.4, 0.8, 1.6, 2.5, 1.0],   # idx: quiet..peak-ish
        'soft_flux_mean':  [0.0, 0.2, 0.5, 1.0, 1.8, 0.6],
        'dsoft_dt_mean':   [0.00, 0.01, 0.05, 0.30, -0.10, -0.25],
    }, index=[0, 1, 2, 3, 4, 5])


def test_assign_state_names_deterministic():
    sm = _state_means_df()
    m1 = assign_state_names(sm, list(sm.columns))
    m2 = assign_state_names(sm, list(sm.columns))
    assert m1 == m2
    # highest soft_flux_max (idx 4) -> Peak; lowest (idx 0) -> Quiet
    assert m1[4] == 'Peak'
    assert m1[0] == 'Quiet'
    # most positive slope (idx 3) -> Initiation; most negative (idx 5) -> Decay
    assert m1[3] == 'Initiation'
    assert m1[5] == 'Decay'
    assert set(m1.values()) <= set(STATE_NAMES)


def test_assign_state_names_missing_cols_falls_back():
    sm = pd.DataFrame(np.random.rand(6, 2))
    m = assign_state_names(sm, list(sm.columns))
    assert len(m) == 6


def test_hmm_trains_and_transitions_sum_to_one():
    rng = np.random.default_rng(0)
    # 3 well-separated blobs over 2 features
    X = np.vstack([
        rng.normal(0, 0.1, (200, 2)),
        rng.normal(5, 0.1, (200, 2)),
        rng.normal(10, 0.1, (200, 2)),
    ])
    m = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=0)
    m.fit(X)
    assert np.allclose(m.transmat_.sum(axis=1), 1.0, atol=1e-6)
    states = m.predict(X)
    assert len(states) == len(X)
    assert set(np.unique(states)) <= {0, 1, 2}


def test_state_machine_inference_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal(0, 0.1, (150, 2)),
        rng.normal(4, 0.1, (150, 2)),
    ])
    m = hmm.GaussianHMM(n_components=2, covariance_type='diag', n_iter=50, random_state=0)
    m.fit(X)
    feature_cols = ['a', 'b']
    blob = {
        'model': m,
        'feature_cols': feature_cols,
        'scaler_mean': X.mean(axis=0),
        'scaler_std': X.std(axis=0) + 1e-8,
        'state_name_map': {0: 'Quiet', 1: 'Peak'},
    }
    path = tmp_path / 'hmm.pkl'
    joblib.dump(blob, path)
    smi = StateMachineInference(str(path))
    state, trans = smi.infer(np.array([4.0, 4.0]), feature_cols)
    assert state in ('Quiet', 'Peak')
    assert set(trans.keys()) == {'Quiet', 'Peak'}
    assert abs(sum(trans.values()) - 1.0) < 1e-6
    # high-value sample should land in the 'Peak' blob
    assert state == 'Peak'
