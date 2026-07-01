# Task 4: Feature Engineering

**Dependencies**: Task 3 (Data Ingestion)
**Estimated effort**: Medium
**PRD Reference**: Phase 2 — Feature Engineering (§3.2); Layers L1, L2, L3

## Objective

Extract features from the windowed normalized data. Implement L1 (raw), L2 (dynamics), and L3 (cross-channel) feature layers. For MVP, L4 (frequency) and L5 (changepoint) are optional.

## Files to Create

- `src/features/__init__.py`
- `src/features/dynamics.py`
- `src/features/cross_channel.py`
- `src/features/pipeline.py`
- `tests/test_features.py`

## Implementation Steps

### 1. `dynamics.py` — L2: Dynamics Features

```python
def compute_derivatives(series: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute first and second derivatives using np.gradient.

    Returns:
        {
            'dsoft_dt': np.ndarray,
            'd2soft_dt2': np.ndarray,
            'dhard_dt': np.ndarray,
            'd2hard_dt2': np.ndarray,
        }
    """
    ...

def compute_rolling_stats(
    series: np.ndarray,
    window_sizes_s: List[int] = [60, 300],
    dt: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute rolling mean, std, and variance for each window size.

    Returns:
        {
            'soft_mean_60s': np.ndarray,
            'soft_std_60s': np.ndarray,
            ...
        }
    """
    ...

def extract_dynamics_features(
    soft_flux_z: np.ndarray,
    hard_flux_z: np.ndarray,
    dt: float = 1.0,
    rolling_windows_s: List[int] = [60, 300]
) -> pd.DataFrame:
    """
    Extract all L2 features for a single window.

    Returns:
        DataFrame with one row, columns for each feature.
    """
    ...
```

**Edge cases**:
- Constant-value window → derivatives are all zero, rolling stats are zero
- Very short window (< 60s) → return NaN for rolling stats that exceed window length
- NaN in input → propagate; derivative of NaN is NaN

### 2. `cross_channel.py` — L3: Cross-Channel Features

```python
def compute_flux_ratio(soft_flux: np.ndarray, hard_flux: np.ndarray, epsilon: float = 1e-12) -> float:
    """soft/hard ratio. Return single float (mean over window)."""
    ...

def compute_flux_difference(soft_flux: np.ndarray, hard_flux: np.ndarray) -> float:
    """soft - hard difference. Return mean over window."""
    ...

def compute_lag_correlation(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    max_lag_s: int = 120,
    dt: float = 1.0
) -> Dict:
    """
    Compute Pearson correlation between soft and hard at lags from 0 to max_lag_s.

    Returns:
        {
            'peak_correlation': float,   # max correlation value
            'peak_lag_s': float,         # lag at which correlation peaks
            'correlation_at_zero': float # correlation at lag 0
        }
    This is one of the strongest precursor features per PRD.
    """
    ...

def compute_phase_difference(soft_flux: np.ndarray, hard_flux: np.ndarray, dt: float = 1.0) -> float:
    """
    Compute instantaneous phase difference via Hilbert transform.
    Returns mean phase difference over the window (in radians).
    """
    ...

def extract_cross_channel_features(
    soft_flux_z: np.ndarray,
    hard_flux_z: np.ndarray,
    dt: float = 1.0,
    max_lag_s: int = 120
) -> pd.DataFrame:
    """
    Extract all L3 features for a single window.

    Returns:
        DataFrame with one row.
    """
    ...
```

**Edge cases**:
- Both channels are identical → peak correlation = 1.0 at lag 0, phase difference = 0
- One channel is all zeros → ratio is 0 or inf (use epsilon); correlation is undefined → return NaN
- Lag > window length → clip max_lag to window_length / 2
- Hilbert transform on very short windows → phase may be unstable

### 3. `pipeline.py` — Feature Extraction Pipeline

```python
def extract_features_for_window(
    soft_flux_z: np.ndarray,
    hard_flux_z: np.ndarray,
    dt: float = 1.0,
    config: Optional[Dict] = None
) -> pd.Series:
    """
    Extract all configured features for a single window.

    Returns a flat Series with feature names as index.
    """
    ...

def extract_all_features(
    X_soft: np.ndarray,
    X_hard: np.ndarray,
    config_path: str = "config/config.yaml",
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Extract features for all windows.

    Args:
        X_soft: (n_windows, window_length)
        X_hard: (n_windows, window_length)

    Returns:
        DataFrame: (n_windows, n_features) with named columns
    """
    # Use joblib Parallel for speed
    ...

def get_feature_names() -> List[str]:
    """Return ordered list of all feature column names."""
    ...
```

**L1 features** (direct from windowed data):
- soft_flux_mean, soft_flux_std, soft_flux_min, soft_flux_max
- hard_flux_mean, hard_flux_std, hard_flux_min, hard_flux_max

**L2 features** (from dynamics):
- dsoft_dt_mean, dsoft_dt_max
- dhard_dt_mean, dhard_dt_max
- d2soft_dt2_mean, d2hard_dt2_mean
- soft_mean_60s, soft_std_60s, soft_mean_300s, soft_std_300s
- hard_mean_60s, hard_std_60s, hard_mean_300s, hard_std_300s

**L3 features** (from cross-channel):
- flux_ratio, flux_difference
- peak_correlation, peak_lag_s, correlation_at_zero
- phase_difference

### 4. Write Tests

| Test | Description |
|------|-------------|
| `test_derivative_on_linear` | Linear ramp: derivative is constant |
| `test_derivative_on_constant` | Constant signal: derivative is 0 |
| `test_rolling_stats_basic` | Known window produces correct mean/std |
| `test_lag_correlation_identical` | Identical signals: peak_correlation=1 at lag 0 |
| `test_lag_correlation_shifted` | Shifted signals: peak at correct lag |
| `test_flux_ratio` | ratio = soft/hard, verify against manual calc |
| `test_feature_pipeline_shape` | Output shape is (n_windows, n_features) |
| `test_feature_names_are_unique` | All feature column names are unique and non-empty |

## Acceptance Criteria

- [ ] L1 features extractable from any valid window
- [ ] L2 derivatives produce correct shapes
- [ ] L3 lag correlation finds correct peak for artificially shifted signals
- [ ] Feature matrix shape: (n_windows, n_features) with n_features >= 20
- [ ] All features are finite (no NaN/Inf) for valid windows
- [ ] Pipeline completes: feature extraction on full dataset finishes in reasonable time
- [ ] All tests pass: `pytest tests/test_features.py -v`

## Verification

```bash
python -c "
import numpy as np
from src.features.pipeline import extract_all_features
# Load windowed data
X_soft = np.load('data/windows/X_soft.npy', mmap_mode='r')[:100]
X_hard = np.load('data/windows/X_hard.npy', mmap_mode='r')[:100]
features = extract_all_features(X_soft, X_hard)
print(f'Extracted {features.shape[1]} features for {features.shape[0]} windows')
print(f'Feature names: {list(features.columns[:8])}...')
assert features.shape[1] >= 20, f'Expected >=20 features, got {features.shape[1]}'
assert not features.isna().any().any(), 'Found NaN values in features'
print('OK')
"
```
