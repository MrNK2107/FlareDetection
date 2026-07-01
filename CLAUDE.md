# FlareClassifier — Agent Instructions

## Project
Aditya-L1 Solar Flare Forecasting System MVP. End-to-end pipeline: synthetic data → ingestion → features → physics → ML → API → dashboard.

## Commands

| Command | Description |
|---------|-------------|
| `python scripts/run_pipeline.py` | Full pipeline: generate data → ingest → feature extract → train |
| `.\scripts\start.ps1` | Start API server + dashboard |
| `python -m pytest tests/ -v` | Run all tests |
| `python -m pytest tests/test_XXX.py -v` | Run specific test file |
| `python -m uvicorn src.api.server:app --port 8000` | Start API only |
| `cd src/dashboard && npm run dev` | Start dashboard only |

## Task Tracking

Tasks in `.claude/tasks/task-*.md`. Execute in numeric order (sequential dependencies).

## Architecture

```
data/raw/             → Synthetic telemetry (Parquet)
data/processed/       → Cleaned, normalized data + feature matrix
data/windows/         → Windowed numpy arrays for ML
src/ingestion/        → Sync, clean, normalize, window
src/features/         → L1 (raw), L2 (dynamics), L3 (cross-channel)
src/physics/          → Stage indicators, thermal fraction
src/models/           → Logistic Regression + Random Forest
src/explainability/   → SHAP + template explanations
src/api/              → FastAPI + WebSocket server
src/dashboard/        → React + Recharts frontend
```

## Verification

After pipeline run:
- `models/random_forest.pkl` exists
- `models/evaluation_results.json` has metrics
- `data/windows/X_soft.npy` has correct shape
- `python -m pytest tests/ -v` passes
