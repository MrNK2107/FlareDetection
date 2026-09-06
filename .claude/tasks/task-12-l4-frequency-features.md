# Task 12: L4 — Frequency Features

**Dependencies**: Task 11
**PRD Reference**: docs/04 §L4; PRD Phase 2 feature table

## Objective
Add frequency-domain features per channel: FFT power in bands [0.001–0.01 Hz], [0.01–0.1 Hz], [0.1–1 Hz]; Morlet wavelet energy across scales 8–256s; spectrogram (spectral) entropy.

## Implementation Notes
- `scipy.signal.welch` with `fs=1/dt` for PSD; band powers via trapezoidal integration of PSD in band limits; normalize by total power.
- `pywt.cwt` (Morlet, scales for periods 8–256s) → wavelet energy per scale → aggregate: mean, max, scale-of-max.
- Spectral entropy: Shannon entropy of normalized PSD.
- Features per channel (soft, hard): `fft_band1/2/3_power_frac`, `wavelet_energy_mean/max`, `wavelet_dominant_scale_s`, `spectral_entropy` → 16 features total.
- 20-min window at 60s cadence (DL arrays) → 20 samples; for 10s-cadence feature extraction the arrays come from the windowing step — use the DL memmap windows for L4 (frequency features need long uniform cadence; document this).

## Files
- `src/features/frequency.py` (new) — `extract_frequency_features(soft, hard, dt) -> pd.Series`
- `src/features/pipeline.py` — integrate L4 behind `features.layers` config flag
- `tests/test_features.py` — sine-in-band has dominant band fraction; white noise entropy high; constant signal → low band-1 power; shape test
- `models/feature_names.json` regenerated

## Acceptance Criteria
- [ ] L4 features deterministic and NaN-free (constant signals handled)
- [ ] Unit tests pass; RF retrains with L4 included and TSS doesn't collapse
