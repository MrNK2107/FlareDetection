# Task 10: Integration & Final Testing

**Dependencies**: All Tasks 1–9
**Estimated effort**: Medium
**PRD Reference**: Entire system

## Objective

Integrate all components into a working end-to-end system, write integration tests, and produce a startup script that launches the entire stack.

## Files to Create

- `scripts/run_pipeline.py` — Generate data → ingest → feature extract → train
- `scripts/start.sh` — Start API + Dashboard
- `scripts/start.ps1` — Start API + Dashboard (PowerShell)
- `tests/test_integration.py`
- `tests/test_end_to_end.py`

## Implementation Steps

### 1. Pipeline Script (`scripts/run_pipeline.py`)

```python
"""
Full pipeline runner: generates synthetic data, runs ingestion,
feature extraction, physics layer, and model training.

Usage: python scripts/run_pipeline.py [--config config/config.yaml]

This is the single entrypoint to reproduce the entire ML pipeline.
"""
import argparse
import yaml
from pathlib import Path

def run_pipeline(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("FlareClassifier Pipeline v0.1.0")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic data...")
    from src.data_generation.synthetic_flare_generator import SyntheticFlareGenerator
    gen = SyntheticFlareGenerator(config_path)
    df = gen.generate_dataset()
    print(f"  → {len(df):,} rows generated")
    print(f"  → Flare distribution: {df[df.flare_class != 'None'].flare_class.value_counts().to_dict()}")

    # Step 2: Ingestion pipeline
    print("\n[2/5] Running ingestion pipeline...")
    from src.ingestion.pipeline import run_ingestion_pipeline
    stats = run_ingestion_pipeline(config_path=config_path)
    print(f"  → {stats['n_windows']:,} windows created")
    print(f"  → Missing data rate: {stats['missing_data_rate']:.4%}")

    # Step 3: Feature extraction
    print("\n[3/5] Extracting features...")
    import numpy as np
    X_soft = np.load(f"{config['data']['windows_dir']}/X_soft.npy")
    X_hard = np.load(f"{config['data']['windows_dir']}/X_hard.npy")
    from src.features.pipeline import extract_all_features
    features = extract_all_features(X_soft, X_hard, config_path=config_path)
    print(f"  → {features.shape[1]} features for {features.shape[0]} windows")

    # Step 4: Physics layer
    print("\n[4/5] Computing physics features...")
    from src.physics.pipeline import compute_physics_features
    physics_features = compute_physics_features(
        X_soft, X_hard, config=config
    )
    print(f"  → {physics_features.shape[1]} physics features added")

    # Step 5: Train models
    print("\n[5/5] Training baseline models...")
    from src.models.train import train_all_baselines
    results = train_all_baselines(config_path=config_path)
    for model_name, metrics in results.items():
        print(f"  → {model_name}: TSS={metrics['tss']:.4f}, "
              f"Brier={metrics['brier_score']:.4f}, "
              f"FAR={metrics['false_alarm_rate']:.4f}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    return results

if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    run_pipeline(config)
```

### 2. Startup Script (`scripts/start.ps1`)

```powershell
# FlareClassifier startup script (PowerShell)
param(
    [string]$ConfigPath = "config/config.yaml"
)

Write-Host "Starting FlareClassifier..." -ForegroundColor Cyan

# Step 1: Verify pipeline has been run
if (-not (Test-Path "models/random_forest.pkl")) {
    Write-Host "Models not found. Running pipeline first..." -ForegroundColor Yellow
    python scripts/run_pipeline.py $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pipeline failed. Exiting." -ForegroundColor Red
        exit 1
    }
}

# Step 2: Start API server
Write-Host "Starting API server on port 8000..." -ForegroundColor Green
$apiJob = Start-Job { python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 }
Start-Sleep 3

# Step 3: Start dashboard
Write-Host "Starting dashboard dev server on port 5173..." -ForegroundColor Green
$dashboardJob = Start-Job {
    Set-Location src/dashboard
    npm run dev
}

Write-Host "FlareClassifier is running!" -ForegroundColor Cyan
Write-Host "  API:      http://localhost:8000" -ForegroundColor Yellow
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  Dashboard: http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
try {
    while ($true) { Start-Sleep 1 }
} finally {
    Stop-Job $apiJob
    Stop-Job $dashboardJob
    Remove-Job $apiJob
    Remove-Job $dashboardJob
}
```

### 3. Integration Tests (`tests/test_integration.py`)

| Test | Description |
|------|-------------|
| `test_full_pipeline_flow` | Generate data → ingest → feature extract → train. Verify all intermediate artifacts exist |
| `test_api_loads_model` | Start API server, verify it loads the trained model |
| `test_api_to_dashboard_flow` | Verify the full prediction payload can be JSON-serialized and matches dashboard expected format |
| `test_artifact_consistency` | Feature names in model match feature names in API |
| `test_data_idempotent` | Running pipeline twice with same seed produces identical results |

### 4. End-to-End Test (`tests/test_end_to_end.py`)

```python
"""
End-to-end test: simulates a streaming scenario.

1. Generate synthetic data with known flare events
2. Run the full pipeline
3. Start API server
4. Simulate streaming inputs via WebSocket
5. Verify predictions are returned for each input
6. Verify alert threshold logic
7. Verify dashboard can parse the output

This test takes ~2 minutes to run.
Mark with @pytest.mark.slow
"""
@pytest.mark.slow
def test_end_to_end_streaming():
    # 1. Setup: run pipeline with small dataset
    # 2. Start API server in subprocess
    # 3. Connect WebSocket client
    # 4. Send 100 feature vectors at 10s simulated intervals
    # 5. Collect predictions
    # 6. Assert: all predictions received, valid schema
    # 7. Assert: at least one alert triggered (if flare in data)
    # 8. Assert: server health endpoint returns ok
    ...
```

### 5. CLAUDE.md — Project Agent Config

```markdown
# FlareClassifier — Agent Instructions

## Commands

- Run full pipeline: `python scripts/run_pipeline.py`
- Start system: `powershell scripts/start.ps1`
- Run tests: `pytest tests/ -v`
- Run specific test: `pytest tests/test_ingestion.py -v`
- Run integration tests: `pytest tests/test_integration.py -v`
- Start API only: `uvicorn src.api.server:app --port 8000`
- Start dashboard only: `cd src/dashboard && npm run dev`

## Task Tracking

Tasks are in `.claude/tasks/task-*.md`. Execute in numeric order.
Each task depends on the previous one.

## Verification

After running pipeline, check:
- `models/random_forest.pkl` exists
- `models/evaluation_results.json` has TSS > 0.3
- `data/windows/` has .npy files with valid shapes

## Config

All parameters in `config/config.yaml`. Tune data generation, ingestion,
and model parameters there.
```

## Acceptance Criteria

- [ ] `python scripts/run_pipeline.py` completes end-to-end without errors
- [ ] All intermediate artifacts exist (parquet, npy, pkl, json)
- [ ] API server starts and responds to health check
- [ ] Integration tests pass: `pytest tests/test_integration.py -v`
- [ ] End-to-end test passes: `pytest tests/test_end_to_end.py -v --runslow`
- [ ] Dashboard builds: `cd src/dashboard && npm run build`
- [ ] All previous unit tests still pass: `pytest tests/ -v`

## Verification

```bash
# Full integration check
python scripts/run_pipeline.py

echo "---"
python -c "
import json
results = json.load(open('models/evaluation_results.json'))
for model, metrics in results.items():
    print(f'{model}: TSS={metrics[\"tss\"]:.4f} (target: >0.3), '
          f'Brier={metrics[\"brier_score\"]:.4f} (target: <0.15)')
"

echo "---"
pytest tests/ -v --ignore=tests/test_end_to_end.py
```
