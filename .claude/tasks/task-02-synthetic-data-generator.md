# Task 2: Synthetic Data Generator

**Dependencies**: Task 1 (Scaffolding)
**Estimated effort**: Medium
**PRD Reference**: Phase 1 — Data Ingestion (Data source: synthetic)

## Objective

Create a parameterized synthetic data generator that produces realistic SoLEXS (soft X-ray) and HEL1OS (hard X-ray) time-series telemetry with embedded flare events of classes B, C, M, X.

## Files to Create

- `src/data_generation/__init__.py`
- `src/data_generation/synthetic_flare_generator.py`
- `tests/test_synthetic_data.py`

## Physics Model

The generator must model the following physical behavior:

1. **Quiet Sun**: Soft flux ~1e-7 W/m², hard flux ~1e-8 W/m² with pink noise (1/f)
2. **Precursor phase** (5-15 min before peak): Soft flux starts rising gradually; hard flux remains near baseline
3. **Impulsive phase**: Hard flux spikes sharply (non-thermal electron acceleration); soft flux continues rising
4. **Peak**: Both channels peak (soft lags hard by 2-5 minutes for large flares)
5. **Decay**: Soft flux decays exponentially (thermal); hard flux drops rapidly
6. **Flare classes**: B (~1e-7), C (~1e-6), M (~1e-5), X (~1e-4) peak soft flux

## Implementation Steps

### 1. Implement `SyntheticFlareGenerator` class

```python
class SyntheticFlareGenerator:
    """
    Generates synthetic SoLEXS + HEL1OS telemetry with labeled flare events.

    Parameters follow config.yaml synthetic_data section.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        ...

    def generate_flare_profile(
        self,
        flare_class: str,        # 'B', 'C', 'M', 'X'
        duration_minutes: float, # total flare duration
        rise_time_ratio: float,  # fraction of duration spent rising (0.2-0.4)
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single flare event (soft_flux, hard_flux arrays).

        Profile shape:
        - Precursor: gradual soft rise (5-15 min before impulsive phase)
        - Impulsive: hard flux spike (gaussian-like, 1-3 min FWHM)
        - Peak: soft reaches maximum (lags hard by delay parameter)
        - Decay: exponential decay (tau_soft >> tau_hard)

        Returns:
            soft_flux: array of flux values at 1s cadence
            hard_flux: array of flux values at 1s cadence
        """
        ...

    def generate_timeseries(
        self,
        duration_days: float,
        flare_events: List[Dict],
        seed: Optional[int] = None,
        dropout_prob: float = 0.001
    ) -> pd.DataFrame:
        """
        Generate a continuous time series with embedded flare events.

        Args:
            duration_days: Total length of the time series
            flare_events: List of dicts with keys:
                - 'flare_class': str
                - 'start_time': pd.Timestamp (or offset in seconds)
                - 'duration_minutes': float
                - 'rise_time_ratio': float (optional, default 0.3)
                - 'peak_ratio': float (optional, deviation from typical peak)
            dropout_prob: Probability of a telemetry dropout at each timestep

        Returns:
            DataFrame with columns:
                timestamp_utc, soft_flux, hard_flux,
                soft_quality_flag, hard_quality_flag,
                flare_class (ground truth label)
        """
        ...

    def generate_dataset(
        self,
        config_overrides: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Convenience method: generate full training dataset per config.
        Schedules flare events randomly based on config flare_rates.
        """
        ...
```

### 2. Implement Flare Profile Shapes

Use mathematical functions for realistic profiles:

- **Soft X-ray profile**: `soft(t) = baseline + A_soft * (1 - exp(-(t/tau_rise))) * exp(-(t/tau_decay))`  
  Use a smoothed piecewise function with smoothstep transitions between phases.

- **Hard X-ray profile**: `hard(t) = baseline + A_hard * exp(-((t - t_peak)/sigma)^2)`  
  Gaussian spike near the impulsive phase, with faster decay than soft.

- **Precursor**: Add a small linear ramp in soft flux starting 5-15 min before hard onset.

- **Noise model**: Pink noise (1/f) using the Voss-McCartney algorithm or numpy FFT method.

- **Quality flags**: Randomly set to 1 (glitch) for 0.1% of samples, or during dropout windows.

### 3. Implement `generate_dataset()`

- Read flare rates from config (e.g., 8 B-class, 4 C-class, 2 M-class, 1 X-class over 90 days)
- Schedule flares with random inter-flare intervals (minimum 2 hours between flares)
- Generate quiet periods between flares
- Embed quality flags for dropouts and glitches
- Save to `data/raw/training_data.parquet`
- Return the DataFrame

### 4. Write Tests

| Test | Description |
|------|-------------|
| `test_quiet_baseline` | No flare input: flux within 10% of configured baseline |
| `test_flare_peak_values` | Each class produces peak within correct order-of-magnitude range |
| `test_hard_lags_soft` | Hard flux peak occurs before or at soft flux peak (check delay) |
| `test_quality_flags` | Dropout probability matches config within 20% relative error |
| `test_label_accuracy` | Generated flare events have correct class labels |
| `test_noise_statistics` | Noise has approximately 1/f power spectrum |
| `test_output_schema` | DataFrame has all required columns |

## Acceptance Criteria

- [ ] `generate_dataset()` produces a valid Parquet file at `data/raw/training_data.parquet`
- [ ] Generated data has: timestamp_utc, soft_flux, hard_flux, soft_quality_flag, hard_quality_flag, flare_class
- [ ] Flare peak fluxes span B/C/M/X ranges correctly
- [ ] All tests pass: `pytest tests/test_synthetic_data.py -v`
- [ ] Visual inspection: run a quick plot to verify profiles look realistic

## Verification

```bash
python -c "
from src.data_generation.synthetic_flare_generator import SyntheticFlareGenerator
gen = SyntheticFlareGenerator()
df = gen.generate_dataset()
print(f'Generated {len(df):,} rows ({len(df)/86400:.1f} days)')
print(f'Flare events: {df[df.flare_class != \"None\"].flare_class.value_counts().to_dict()}')
print(f'Columns: {list(df.columns)}')
"
```
