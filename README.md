# FlareClassifier — Aditya-L1 Solar Flare Forecasting System

End-to-end flare forecasting MVP: synthetic telemetry → ingestion → feature
engineering (L1–L5) → physics layer → ML (classical + deep) → inference API →
real-time dashboard. Implements the full spec in `docs/` at MVP fidelity;
deviations are listed below.

## Quickstart

```bash
# 1. Install (CPU torch is sufficient; ~2 GB with CUDA wheels if available)
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# 2. Full pipeline: generate 180 days of synthetic data, ingest, extract
#    features (L1-L5 + physics), train LogisticRegression/RandomForest/LSTM/
#    Transformer/HMM/lead-time, register + promote models (~30-60 min)
python scripts/run_pipeline.py

# 3. Verify artifacts, metrics, payload spec compliance, and registry
python scripts/verify.py

# 4. Run tests
python -m pytest tests/ -v

# 5. Start the API + dashboard
./scripts/start.ps1           # PowerShell
# or individually:
python -m uvicorn src.api.server:app --port 8000
cd src/dashboard && npm install && npm run dev
```

## Pipeline stages (scripts/run_pipeline.py)

| Stage | Output |
|---|---|
| 1. Synthetic data generation (180 d, Poisson flare events, catalogue) | `data/raw/`, `data/external/flare_catalogue.parquet` |
| 2. Ingestion: sync → clean → normalize → hybrid-stride windowing | `data/processed/`, `data/windows/` (float32 memmaps) |
| 3. Baselines (LogReg + RF) on engineered features | `models/*.pkl`, `models/evaluation_results.json` |
| 4. LSTM baseline on raw DL windows | `models/lstm.pt` |
| 5. Dual-Stream Transformer (cross-attention + MC-Dropout) | `models/transformer.pt` |
| 6. 6-state Gaussian HMM solar state machine | `models/hmm.pkl`, `models/hmm_validation.json` |
| 7. GradientBoosting lead-time model with 90% CI | `models/lead_time.pkl` |
| 8. Registry registration + first promotion | `models/registry/`, `models/production.json` |

## Task map

`.claude/tasks/task-01…20` document each increment, from scaffolding through
data scale-up (11), L4 frequency + L5 changepoint features (12–13), LSTM (14),
Transformer (15), HMM state machine (16), lead time + alerts (17), history +
registry + continuous learning (18), dashboard completion (19), and E2E
verification (20). `MVP_PLAN.md` is the master plan.

## Operations

- **Replay streaming**: the API replays `data/processed/synchronized_clean.parquet`
  over WebSocket at configurable speed (pause/resume/×1/×10/×60), computing real
  features per tick — no live instrument in the single-operator MVP.
- **Alerts**: rising-edge at threshold 0.5 (configurable) with 15-min quiet
  window; email (SMTP), webhook (JSON POST), browser (WS push + Notification API).
- **History**: SQLite (`data/processed/inference_history.db`); hindcast overlay,
  CSV export, `/history/predictions|alerts|events` endpoints.
- **Model registry**: immutable timestamped version dirs under `models/registry/`;
  `scripts/promote_model.py` promotes a candidate only if it beats production on
  all 5 gate metrics (TSS, Brier, FAR, detection rate, lead-time MAE).
- **Shadow mode**: `SHADOW_MODEL=path/to/model.pkl` env var logs shadow
  predictions (tagged `shadow:`) without serving them.
- **Drift / retraining triggers**: `scripts/evaluate_drift.py` computes rolling
  7-day TSS vs the deployment baseline plus data-volume trigger; exits 1 when
  retraining is recommended (cron/CI friendly).

## Deviations from docs/ (rationale in MVP_PLAN.md decision log)

- **Hybrid stride**: features every 60 s, DL raw windows every 120 s decimated
  to 10 s cadence (full 10 s everywhere ≈ 30 TB for 180 d @ 1 Hz).
- **30-day held-out test period** instead of 6 months (180-day dataset;
  adaptive fallback keeps ≥50% train when the span is short).
- **File-based registry** instead of MLflow (single-operator deployment).
- **Browser alerts via WebSocket + Notification API** instead of VAPID Web Push.
- **BOCPD** implemented compactly in numpy on a decimated series.
- **PRD metric targets** (TSS>0.6, Brier<0.08, FAR<25%, det>80%) are
  physics-level goals; on synthetic data `scripts/verify.py` reports them
  without asserting.
