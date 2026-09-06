# Task 18: Inference History + Hindcast + CSV Export + Model Registry + Continuous Learning

**Dependencies**: Task 17
**PRD Reference**: docs/08 §4 (historical view), §5 (continuous learning); docs/09 (model versioning)

## Objective
Persist inference history for hindcast/export, add a timestamped model registry with a promotion gate, and implement the retraining-trigger logic.

## Implementation Notes
- **Inference store**: SQLite via stdlib `sqlite3` (`data/processed/inference_history.db`): table `predictions(ts_utc, flare_probability, severity_json, expected_lead_time_min, solar_state, model_version, explanation_text)` + table `alerts`. Batch inserts; index on ts_utc. `/history` API endpoints: `GET /history/predictions?start&end&format=json|csv` (CSV = streaming `text/csv`, 10s-cadence payloads), `GET /history/alerts`.
- **Hindcast overlay data**: same endpoint serves probability curve vs flare events (events from window metadata / generator catalogue import).
- **Model registry** (`src/models/registry.py`): `models/registry/<utc-timestamp>-<modelname>/` containing model file + `metadata.json` (train window, data window, feature names, git SHA if available, metrics). `register(model_paths, metrics, config)` + `load_latest()` + `list_models()`.
- **Deployment gate** (`scripts/promote_model.py`): new candidate must beat current production registry entry on **all 5 metrics** (TSS, Brier, FAR, detection rate, lead-time MAE) to be promoted; else rejected with report. `models/production.json` points to promoted version. Never overwrite weights — immutable dirs.
- **Shadow mode**: API env flag `SHADOW_MODEL=path` — inference computes shadow predictions, logs both to history (tagged), serves production. (72-hour shadow run = operational procedure; the mechanism is what we ship.)
- **Continuous learning triggers** (`src/models/monitoring.py` + `scripts/evaluate_drift.py`):
  - Scheduled: monthly job comparing rolling 30-day TSS vs deployment baseline (from registry metadata).
  - Alert-triggered: 7-day rolling TSS drop > 0.1 → retrain flag + log.
  - Data volume: >500 new labeled events since last registration → queue retraining.
  - Output: `data/processed/drift_report.json` + exit codes for CI/cron.

## Files
- `src/api/history.py`, `src/api/server.py` (routes), `src/models/registry.py`, `src/models/monitoring.py`, `scripts/promote_model.py`, `scripts/evaluate_drift.py`, tests

## Acceptance Criteria
- [ ] Predictions + alerts persist; CSV export endpoint streams correct schema
- [ ] Registry registers/loads/lists; promotion gate enforces all-5-metrics rule
- [ ] Shadow-mode flag logs without serving
- [ ] Drift report generated from history with the three triggers
- [ ] Tests pass
