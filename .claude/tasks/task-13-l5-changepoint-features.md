# Task 13: L5 — Changepoint Features

**Dependencies**: Task 11
**PRD Reference**: docs/04 §L5

## Objective
Add change-point features per channel: CUSUM alarm flag, ruptures PELT breakpoint count + distance (seconds since last break), and a lightweight Bayesian changepoint posterior summary.

## Implementation Notes
- CUSUM: standard two-sided cumulative sum on mean-centered series with drift k=0.5σ, threshold h=4σ → binary alarm flag + amplitude at alarm.
- ruptures: `ruptures.Pelt(model="rbf").fit(signal).predict(pen=...)` → breakpoints; features: `n_breakpoints`, `time_since_last_break_s`, `breakpoint_recency_weighted_count`.
- Bayesian CP: offline Bayesian online changepoint detection (BOCPD) with Normal-Gamma model — features: `bocpd_max_prob`, `bocpd_runlength_mean`. Implement compactly with numpy (no heavy dependency); it's O(T·K) on 20–1200 samples, fine.
- Features per channel: 5 → 10 total. Guard: ruptures on tiny/constant arrays must not raise.

## Files
- `src/features/changepoint.py` (new)
- `src/features/pipeline.py` — integrate behind `features.layers` flag
- `tests/test_features.py` — step-change detection tests; constant-series robustness; determinism

## Acceptance Criteria
- [ ] A synthetic step change in the last third of a window is detected (n_breakpoints ≥ 1, time_since_last_break small)
- [ ] Constant input → 0 breaks, no exceptions, no NaN
- [ ] Full feature matrix rebuilds with L1–L5 + physics; RF retrains
