"""Solar State Machine (docs/05 §4): 6 latent states inferred unsupervised
from the engineered feature matrix with a Gaussian HMM.

States: S0 Quiet, S1 Energy Accumulation, S2 Precursor, S3 Initiation,
S4 Peak, S5 Decay. Hidden-state indices are mapped to these semantic names
deterministically via feature-profile ranking (assign_state_names), and the
learned transition matrix exposes P(next_state | current_state) for the API.
"""
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

from hmmlearn import hmm

from src.models.splits import temporal_train_test_masks

STATE_NAMES = ['Quiet', 'Energy Accumulation', 'Precursor', 'Initiation', 'Peak', 'Decay']
FEATURE_COLS_FOR_RANKING = ['soft_flux_max', 'soft_flux_mean', 'dsoft_dt_mean']


def _feature_matrix(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(processed_dir / 'full_feature_matrix.parquet')
    meta = pd.read_parquet(processed_dir / 'window_metadata.parquet')
    n = min(len(features), len(meta))
    features = features.iloc[:n].reset_index(drop=True)
    meta = meta.iloc[:n].reset_index(drop=True)
    cols = [c for c in features.columns if c != 'window_start']
    X = features[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame(X, columns=cols)
    # Constant columns carry no information and make the Gaussian covariance
    # degenerate (e.g. unfilled historical-context features) -> drop them.
    var = df.var(axis=0)
    keep = var[var > 1e-12].index
    dropped = sorted(set(cols) - set(keep))
    if dropped:
        print(f"    Dropping constant feature columns: {dropped}")
    return df[list(keep)], meta


def assign_state_names(state_means: pd.DataFrame, feature_cols) -> Dict[int, str]:
    """Deterministic mapping from hidden-state index to semantic name via
    feature-profile ranking:
      Quiet      = lowest soft_flux_max
      Peak       = highest soft_flux_max
      Decay      = most negative dsoft_dt_mean of the rest
      Initiation = most positive dsoft_dt_mean of the rest
      Precursor  = higher soft_flux_max of the last two
      Energy Accumulation = the remaining one
    """
    cols = [c for c in FEATURE_COLS_FOR_RANKING if c in feature_cols]
    if not cols:
        return {i: STATE_NAMES[i] for i in range(len(state_means))}
    soft_max_col = 'soft_flux_max' if 'soft_flux_max' in cols else cols[0]
    slope_col = 'dsoft_dt_mean' if 'dsoft_dt_mean' in cols else cols[-1]
    sm = state_means[soft_max_col]
    slope = state_means[slope_col]
    remaining = list(state_means.index)
    mapping: Dict[int, str] = {}
    quiet_idx = sm.idxmin()
    mapping[quiet_idx] = 'Quiet'
    remaining.remove(quiet_idx)
    peak_idx = sm[remaining].idxmax()
    mapping[peak_idx] = 'Peak'
    remaining.remove(peak_idx)
    if len(remaining) >= 2:
        decay_idx = slope[remaining].idxmin()
        mapping[decay_idx] = 'Decay'
        remaining.remove(decay_idx)
        init_idx = slope[remaining].idxmax()
        mapping[init_idx] = 'Initiation'
        remaining.remove(init_idx)
    if len(remaining) >= 2:
        prec_idx = sm[remaining].idxmax()
        mapping[prec_idx] = 'Precursor'
        remaining.remove(prec_idx)
    for idx in remaining:
        mapping[idx] = 'Energy Accumulation'
    return mapping


def train_state_machine(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    hc = config['models']['hmm']
    eval_cfg = config.get('evaluation', {})
    seed = config['synthetic_data']['seed']
    processed_dir = Path(config['data']['processed_dir'])

    X_df, meta = _feature_matrix(processed_dir)
    ts = pd.DatetimeIndex(pd.to_datetime(meta['window_start']))
    train_mask, test_mask = temporal_train_test_masks(
        ts, float(eval_cfg.get('test_days', 30))
    )
    X_train = X_df[train_mask]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_z = (X_train - mean) / std

    best_model, best_score = None, -np.inf
    for i in range(int(hc.get('n_init', 3))):
        m = hmm.GaussianHMM(
            n_components=int(hc['n_states']), covariance_type='diag',
            n_iter=int(hc['n_iter']), random_state=seed + i, verbose=False,
        )
        try:
            m.fit(X_train_z)
            score = m.score(X_train_z)
        except Exception:
            continue
        if score > best_score:
            best_model, best_score = m, score
    if best_model is None:
        raise RuntimeError("HMM fitting failed on all restarts")

    state_means_z = pd.DataFrame(
        best_model.means_, columns=X_df.columns
    )[list(X_df.columns)]
    name_map = assign_state_names(state_means_z, list(X_df.columns))

    # ------- validation against flare labels (docs/05 §4)
    X_all_z = (X_df - mean) / std
    states = best_model.predict(X_all_z)
    labels = meta['label_code'].to_numpy(dtype=int) if 'label_code' in meta.columns else np.zeros(len(meta), dtype=int)
    init_peak = [i for i, n in name_map.items() if n in ('Initiation', 'Peak')]
    in_s34 = np.isin(states, init_peak)
    cooccurrence = float((labels[in_s34 & test_mask] > 0).mean()) if (in_s34 & test_mask).any() else 0.0
    positive_test = test_mask & (labels > 0)
    detection_in_s34 = float(in_s34[positive_test].mean()) if positive_test.any() else 0.0
    per_state_flare_rate = {
        name_map[int(k)]: float((labels[states == k] > 0).mean())
        for k in np.unique(states)
    }
    validation = {
        's3_s4_cooccurrence_with_flares': cooccurrence,
        'flare_windows_in_s3_s4': detection_in_s34,
        'per_state_flare_rate': per_state_flare_rate,
        'target_note': 'docs/05 target: S3/S4 co-occur with catalogue events at >70% (physics-level goal on synthetic data)',
        'log_likelihood': float(best_score),
    }
    with open('models/hmm_validation.json', 'w') as f:
        json.dump(validation, f, indent=2)

    joblib.dump({
        'model': best_model,
        'feature_cols': list(X_df.columns),
        'scaler_mean': mean,
        'scaler_std': std,
        'state_name_map': {int(k): v for k, v in name_map.items()},
    }, 'models/hmm.pkl')
    print(f"    HMM log-likelihood={best_score:.1f}; "
          f"S3/S4 co-occurrence={cooccurrence:.2%}, flares-in-S3/S4={detection_in_s34:.2%}")
    return validation


class StateMachineInference:
    """Loads the trained HMM for the inference API."""

    def __init__(self, path: str = 'models/hmm.pkl'):
        blob = joblib.load(path)
        self.model = blob['model']
        self.feature_cols = blob['feature_cols']
        self.mean = blob['scaler_mean']
        self.std = blob['scaler_std']
        self.state_name_map = {int(k): v for k, v in blob['state_name_map'].items()}

    def infer(self, features: np.ndarray, feature_names) -> Tuple[str, Dict[str, float]]:
        """Map an incoming feature vector onto HMM feature order, infer the
        current state, and return (state_name, P(next_state | current))."""
        lookup = {n: v for n, v in zip(feature_names, np.asarray(features).ravel())}
        vec = np.array([lookup.get(c, 0.0) for c in self.feature_cols], dtype=np.float64)
        X = ((vec - self.mean) / self.std).reshape(1, -1)
        state = int(self.model.predict(X)[0])
        trans_row = self.model.transmat_[state]
        transition_probs = {self.state_name_map[int(j)]: float(trans_row[j]) for j in range(len(trans_row))}
        return self.state_name_map[state], transition_probs


if __name__ == "__main__":
    import sys
    train_state_machine(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
