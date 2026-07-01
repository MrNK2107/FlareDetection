# Task 5: Physics-Aware Feature Layer

**Dependencies**: Task 4 (Feature Engineering)
**Estimated effort**: Small-Medium
**PRD Reference**: Phase 3 — Physics-Aware Feature Layer (§3.3)

## Objective

Implement solar physics domain knowledge as additional features: stage indicators, derived ratios, and historical context.

## Files to Create

- `src/physics/__init__.py`
- `src/physics/stage_indicators.py`
- `src/physics/derived_ratios.py`
- `src/physics/historical_context.py`
- `src/physics/pipeline.py`
- `tests/test_physics.py`

## Implementation Steps

### 1. `stage_indicators.py`

```python
def compute_soft_rising(
    dsoft_dt: np.ndarray,
    threshold_percentile: float = 95.0,
    consecutive_windows: int = 3
) -> float:
    """
    Returns 1 if dSoft/dt > threshold for `consecutive_windows` consecutive values.
    Threshold = percentile of dSoft/dt in quiet-Sun baseline.

    For synthetic data, compute threshold from the training data distribution.
    Returns float (0 or 1) for a single window.
    """
    ...

def compute_hard_accelerating(
    d2hard_dt2: np.ndarray,
    threshold_percentile: float = 95.0,
    consecutive_windows: int = 2
) -> float:
    """
    Returns 1 if d²Hard/dt² > threshold for `consecutive_windows` consecutive values.
    """
    ...

def compute_precursor_candidate(
    soft_rising: float,
    hard_flux: np.ndarray,
    hard_quiet_baseline: float = 1e-8,
    threshold_factor: float = 1.5
) -> float:
    """
    Returns 1 if soft_rising AND hard_flux < threshold_factor * hard_quiet_baseline.

    This captures the precursor phase where soft X-ray is rising but hard X-ray
    hasn't yet responded (non-thermal emission hasn't started).
    """
    ...

def extract_stage_indicators(
    dsoft_dt: np.ndarray,
    d2hard_dt2: np.ndarray,
    hard_flux: np.ndarray,
    dsoft_dt_threshold: float,
    d2hard_dt2_threshold: float,
    hard_quiet_baseline: float = 1e-8
) -> Dict[str, float]:
    """
    Extract all stage indicator features for one window.

    Returns:
        {
            'soft_rising': 0 or 1,
            'hard_accelerating': 0 or 1,
            'precursor_candidate': 0 or 1
        }
    """
    ...
```

**Edge cases**:
- All dsoft_dt values are negative → soft_rising = 0
- hard_flux all zeros → hard_accelerating still valid (d²/dt² of zeros = 0)
- Borderline: exactly `consecutive_windows` values above threshold → soft_rising = 1

### 2. `derived_ratios.py`

```python
def compute_thermal_fraction(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    epsilon: float = 1e-12,
    smoothing_s: int = 30
) -> float:
    """
    thermal_fraction = soft_flux / (soft_flux + hard_flux)
    Smoothed over smoothing_s samples.

    High values = thermal emission dominates (quiet or early precursor)
    Low values = non-thermal emission significant (impulsive phase)
    """
    ...

def compute_nonthermal_index(
    dhard_dt: np.ndarray,
    dsoft_dt: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    nonthermal_index = (dHard/dt) / (dSoft/dt + epsilon)

    Rapidly rising hard flux relative to soft flux indicates
    non-thermal electron acceleration (impulsive phase onset).
    """
    ...

def extract_derived_ratios(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    dsoft_dt: np.ndarray,
    dhard_dt: np.ndarray
) -> Dict[str, float]:
    """
    Returns:
        {
            'thermal_fraction': float,
            'nonthermal_index': float
        }
    """
    ...
```

**Edge cases**:
- Both channels zero → thermal_fraction = 0.5 (by convention with epsilon)
- dSoft/dt = 0 → nonthermal_index blows up; clamp to [0, 100] range
- All hard_flux = 0 → thermal_fraction = 1.0 (purely thermal)

### 3. `historical_context.py`

```python
def compute_time_since_last_flare(
    window_timestamp: pd.Timestamp,
    flare_timestamps: List[pd.Timestamp]
) -> float:
    """
    Returns seconds since the most recent flare event.
    If no prior flare, return a large sentinel value (e.g., 1e6).
    """
    ...

def compute_cycle_proxy(
    window_timestamp: pd.Timestamp,
    solar_max_date: pd.Timestamp = pd.Timestamp("2025-07-01"),
    cycle_length_years: float = 11.0
) -> float:
    """
    Normalized day-of-cycle: 0.0 at solar minimum, 1.0 at next minimum.

    Useful proxy for background flare probability which varies with
    the 11-year solar cycle.
    """
    ...

def extract_historical_features(
    window_timestamp: pd.Timestamp,
    flare_history: List[pd.Timestamp]
) -> Dict[str, float]:
    """
    Extract historical context features.

    Returns:
        {
            'time_since_last_flare_s': float,
            'cycle_proxy': float        # [0, 1]
        }
    """
    ...
```

**Edge cases**:
- No prior flares → time_since_last_flare = 1e6 (sentinel)
- Timestamp before solar_max_date → cycle_proxy still works (negative values clamped to 0)
- Flare history empty → sentinel value, log warning on first call

### 4. `pipeline.py` — Physics Pipeline

```python
def compute_physics_features(
    X_soft: np.ndarray,
    X_hard: np.ndarray,
    dsoft_dt: np.ndarray,
    dhard_dt: np.ndarray,
    d2hard_dt2: np.ndarray,
    window_timestamps: Optional[List[pd.Timestamp]] = None,
    flare_history: Optional[List[pd.Timestamp]] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Compute all physics-aware features for all windows.

    Returns DataFrame with columns for each physics feature.
    """
    ...

def compute_thresholds_from_training(
    dsoft_dt_all: np.ndarray,
    d2hard_dt2_all: np.ndarray,
    percentile: float = 95.0
) -> Dict[str, float]:
    """
    Compute dsoft_dt and d2hard_dt2 thresholds from training data.
    These are used by stage_indicator functions.
    """
    ...
```

### 5. Write Tests

| Test | Description |
|------|-------------|
| `test_soft_rising_detection` | Window with rising soft flux returns soft_rising=1 |
| `test_soft_rising_quiet` | Window with flat/falling soft flux returns 0 |
| `test_precursor_detection` | Soft rising + hard quiet → precursor=1 |
| `test_precursor_not_during_flare` | Both channels elevated → precursor=0 |
| `test_thermal_fraction_quiet` | Quiet Sun: thermal_fraction ≈ 1 |
| `test_thermal_fraction_during_flare` | During flare: thermal_fraction < 1 |
| `test_nonthermal_index` | Rapid hard rise → large nonthermal_index |
| `test_time_since_last_flare` | Correctly counts seconds since prior flare |
| `test_cycle_proxy_range` | Value is in [0, 1] for any timestamp |

## Acceptance Criteria

- [ ] soft_rising correctly identifies rising soft flux windows
- [ ] precursor_candidate correctly identifies precursor state
- [ ] thermal_fraction in [0, 1] for all inputs
- [ ] All physics features are finite (no NaN/Inf)
- [ ] Physics features append cleanly to feature matrix from Task 4
- [ ] All tests pass: `pytest tests/test_physics.py -v`

## Verification

```bash
python -c "
import numpy as np
from src.physics.stage_indicators import compute_soft_rising, compute_precursor_candidate
# Test with rising soft flux
rising = np.linspace(0, 1, 100)
dsoft = np.gradient(rising)
result = compute_soft_rising(dsoft, threshold_percentile=50.0, consecutive_windows=3)
print(f'Rising signal → soft_rising={result}')
assert result == 1.0, 'Expected soft_rising=1 for rising signal'
# Test with quiet signal
quiet = np.zeros(100)
dsoft_q = np.gradient(quiet)
result_q = compute_soft_rising(dsoft_q, threshold_percentile=99.0, consecutive_windows=3)
print(f'Quiet signal → soft_rising={result_q}')
print('OK')
"
```
