# Task 16: Solar State Machine (HMM)

**Dependencies**: Task 15 (feature/flag infrastructure), Task 13
**PRD Reference**: docs/05 §4; docs/06 §4 (solar_state + state_transition_probs payload fields)

## Objective
Unsupervised 6-state HMM over engineered features → per-window state + transition probabilities; validated against flare catalogue (S3/S4 co-occurrence >70% with events).

## Implementation Notes
- States: S0 Quiet, S1 Energy Accumulation, S2 Precursor, S3 Initiation, S4 Peak, S5 Decay.
- New dep: `hmmlearn` (small, pure wheel). Fit GaussianHMM(n_components=6, covariance_type='diag') on standardized feature matrix (train period only), fixed seed, several restarts → best log-likelihood; **state ordering**: map states to semantic labels by their mean feature profile (e.g., S4 Peak = highest soft_flux_mean/flare co-occurrence; S0 Quiet = lowest activity) — deterministic assignment function, not eyeballing.
- Transition probabilities: from HMM's learned transition matrix; per-inference output = row for current state.
- Validation metric script: P(S3/S4 | flare ongoing) and co-occurrence rate vs flare catalogue from window metadata; write to `models/hmm_validation.json`.
- API integration: `InferenceEngine` loads HMM; per predict → current state label + transition probs row. Dashboard state colors per docs/08 (Quiet green, Energy Accum yellow, Precursor orange, Initiation red, Peak dark red, Decay purple).

## Files
- `src/models/state_machine.py` (new)
- `scripts/train_hmm.py`
- `src/api/inference.py` — wire solar_state + state_transition_probs into payload (replaces "Unknown")
- `tests/test_models.py` or new `tests/test_state_machine.py` — 6 states emitted; transition rows sum to ~1; state assignment deterministic; validation file written

## Acceptance Criteria
- [ ] HMM trains on feature matrix, deterministic with seed
- [ ] `solar_state` and `state_transition_probs` populated in API payload
- [ ] State/flare co-occurrence validation computed and saved
- [ ] Tests pass
