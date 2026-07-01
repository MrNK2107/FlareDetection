# Task 3: Data Ingestion Pipeline

**Dependencies**: Task 2 (Synthetic Data Generator)
**Estimated effort**: Medium
**PRD Reference**: Phase 1 — Data Engineering (§3.1.1–3.1.4)

## Objective

Build the data ingestion pipeline: timestamp synchronization, cleaning, normalization, and windowing. This is the most critical phase — every downstream component depends on aligned, clean data.

## Files to Create

- `src/ingestion/__init__.py`
- `src/ingestion/synchronizer.py`
- `src/ingestion/cleaner.py`
- `src/ingestion/normalizer.py`
- `src/ingestion/windowing.py`
- `src/ingestion/pipeline.py` (orchestrator)
- `tests/test_ingestion.py`

## Implementation Steps

### 1. `synchronizer.py` — Timestamp Synchronization

```python
def synchronize(
    df: pd.DataFrame,
    target_frequency_hz: float = 1.0,
    gap_threshold_s: float = 5.0
) -> pd.DataFrame:
    """
    Resample both channels to a common UTC-aligned frequency.

    Steps:
    1. Ensure timestamp_utc is monotonic and datetime type
    2. Set timestamp_utc as index
    3. Resample to target frequency using mean aggregation
    4. Forward-fill gaps shorter than gap_threshold_s
    5. Flag gaps longer than gap_threshold_s (set quality_flag to -1 for gap)
    6. Forward-fill longer gaps (flagged for downstream uncertainty)

    Returns:
        DataFrame with columns:
            timestamp_utc, soft_flux, hard_flux,
            soft_quality_flag, hard_quality_flag
    """
    ...

def flag_gap_regions(
    df: pd.DataFrame,
    gap_threshold_s: float = 5.0
) -> pd.DataFrame:
    """
    Identify and flag gap regions in the synchronized data.

    Adds column 'gap_flag': 0 = normal, 1 = short gap (filled), 2 = long gap (filled+flagged)
    """
    ...
```

**Edge cases**:
- Empty DataFrame → return empty with correct schema
- Non-monotonic timestamps → sort before resampling
- All NaN in a channel → keep but flag all as quality=-1
- Irregular cadence → resample handles this; verify output is regular

### 2. `cleaner.py` — Data Cleaning

```python
def remove_glitch_rows(
    df: pd.DataFrame,
    quality_column: str = 'soft_quality_flag'
) -> pd.DataFrame:
    """
    Remove rows where either quality flag indicates sensor glitch or calibration event.
    Quality flags: 0=good, 1=glitch, 2=calibration, -1=gap_filled
    """
    ...

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps, keep first occurrence."""
    ...

def clip_to_physical_range(
    df: pd.DataFrame,
    soft_range: Tuple[float, float] = (1e-9, 1e-2),
    hard_range: Tuple[float, float] = (1e-10, 1e-3)
) -> Tuple[pd.DataFrame, int]:
    """
    Clip values to physically plausible ranges.
    Log clipped rows count.
    Returns (df, n_clipped).
    """
    ...

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps in order: dedup → clip → remove glitch rows."""
    ...
```

**Edge cases**:
- All rows are glitches → return empty DataFrame (handle downstream)
- Timestamps with sub-millisecond duplicates → keep first, log warning
- Values exactly at boundary → don't clip (use strict inequality)

### 3. `normalizer.py` — Rolling Z-Score Normalization

```python
def rolling_zscore(
    series: pd.Series,
    window_hours: float = 6.0,
    min_periods: int = 60
) -> pd.Series:
    """
    Apply rolling z-score normalization.

    z(t) = (x(t) - mean(t-window to t)) / std(t-window to t)

    Uses expanding window at the start (first `min_periods` samples).
    """
    ...

def normalize(df: pd.DataFrame, window_hours: float = 6.0) -> pd.DataFrame:
    """
    Apply rolling z-score independently to soft_flux and hard_flux.
    Preserves quality flags and metadata columns.

    Returns DataFrame with added columns:
        soft_flux_z, hard_flux_z
    """
    ...
```

**Edge cases**:
- Window contains all-constant values (std=0) → return 0 for those points
- Series shorter than min_periods → use available data with warning
- NaN values in window → exclude from mean/std calculation

### 4. `windowing.py` — Sliding Window Generation

```python
def generate_windows(
    df: pd.DataFrame,
    window_length_s: int = 1200,
    stride_s: int = 10,
    forecast_horizon_s: int = 1800,
    label_column: str = 'flare_class'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sliding windows for ML training.

    Args:
        df: Normalized DataFrame with timestamp index
        window_length_s: 1200s = 20 minutes
        stride_s: 10 seconds
        forecast_horizon_s: 1800s = 30 minutes forward
        label_column: Ground truth label column

    Returns:
        X_soft: (n_windows, window_length) soft flux z-scores
        X_hard: (n_windows, window_length) hard flux z-scores
        y: (n_windows,) labels encoded as:
            0=None, 1=B, 2=C, 3=M, 4=X
    """
    ...

def get_window_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return metadata for each window: start_time, end_time, label.
    Used for tracking and debugging.
    """
    ...
```

**Edge cases**:
- Windows at the end of the dataset (can't get 30-min future label) → drop
- Windows spanning gap regions (high uncertainty) → add a `has_gap` flag column
- No flares in dataset → all labels are 0 (None); still generates valid windows

### 5. `pipeline.py` — Orchestrator

```python
def run_ingestion_pipeline(
    input_path: str = "data/raw/training_data.parquet",
    output_dir: str = "data/processed",
    window_output_dir: str = "data/windows",
    config_path: str = "config/config.yaml"
) -> Dict:
    """
    Run the complete ingestion pipeline:
    1. Load raw data
    2. Synchronize
    3. Clean
    4. Normalize
    5. Generate windows
    6. Save all outputs

    Returns:
        Dict with:
            'n_rows_original': int
            'n_rows_after_cleaning': int
            'n_windows': int
            'class_distribution': Dict
            'missing_data_rate': float
    """
    ...
```

## Acceptance Criteria

- [ ] `synchronize()` produces regular 1-second cadence output
- [ ] Missing data rate after cleaning < 2%
- [ ] Rolling z-score mean ≈ 0 and std ≈ 1 for any 6-hour slice (within 0.01 tolerance)
- [ ] Windows shape: X_soft.shape == (n, 1200), X_hard.shape == (n, 1200)
- [ ] Label encoding: 0=None, 1=B, 2=C, 3=M, 4=X
- [ ] 100% of windows have aligned timestamps (no NaN in window)
- [ ] Pipeline runs end-to-end: `python -c "from src.ingestion.pipeline import run_ingestion_pipeline; stats = run_ingestion_pipeline(); print(stats)"`
- [ ] All tests pass: `pytest tests/test_ingestion.py -v`

## Verification

```bash
python -c "
from src.ingestion.pipeline import run_ingestion_pipeline
stats = run_ingestion_pipeline()
print(f'Original rows: {stats[\"n_rows_original\"]:,}')
print(f'After cleaning: {stats[\"n_rows_after_cleaning\"]:,}')
print(f'Missing data rate: {stats[\"missing_data_rate\"]:.4%}')
print(f'Windows generated: {stats[\"n_windows\"]:,}')
print(f'Class distribution: {stats[\"class_distribution\"]}')
"
```
