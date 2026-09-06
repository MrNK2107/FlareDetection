# Task 20: End-to-End Pipeline + Evaluation vs PRD Targets + Verification

**Dependencies**: Tasks 11–19
**PRD Reference**: docs/06 §2–§3 (metric targets); CLAUDE.md verification section

## Objective
Wire the full pipeline (extended to the new stages), run it end-to-end on the 180-day dataset, and verify PRD metric targets + all CLAUDE.md verification items.

## Implementation Notes
- Extend `scripts/run_pipeline.py`: generate → ingest → features (L1–L5) → physics → train baselines → LSTM → transformer → HMM → registry registration → evaluation report.
- `scripts/verify.py` (idempotent checks for CI and manual runs):
  - `models/random_forest.pkl` exists; `models/evaluation_results.json` has all 5 models' metrics
  - `data/windows/X_soft.npy` correct shape
  - Payload schema test (docs/06 §4 compliance)
  - Metric targets: report TSS / Brier / FAR / detection-rate per model vs PRD targets table; **note**: targets (TSS>0.6, Brier<0.08, FAR<25%, det>80%) are physics-level goals — on synthetic data report and assert *sanity* levels, not the PRD targets, and document that.
  - Model registry has ≥1 registered + promoted model
- README rewrite: setup (CPU torch note), pipeline, task map, deviations section.

## Files
- `scripts/run_pipeline.py` (extend), `scripts/verify.py` (new), `README.md` (new), `CLAUDE.md` (verification section update)

## Acceptance Criteria
- [ ] Full pipeline runs end-to-end with the new stages on 180-day data
- [ ] All 5 model families have metrics in evaluation_results.json
- [ ] verify.py passes all checks
- [ ] Full `pytest tests/` suite passes
- [ ] README documents deviations (hybrid stride, test-period length, file registry)
