# Task 1: Project Scaffolding

**Dependencies**: None
**Estimated effort**: Small
**PRD Reference**: Technology Stack (§7)

## Objective

Set up the Python project structure, dependencies, configuration, and tooling so all subsequent tasks have a consistent foundation.

## Files to Create

- `pyproject.toml`
- `.gitignore`
- `config/config.yaml`
- `src/__init__.py` (empty)
- `tests/__init__.py` (empty)
- `notebooks/.gitkeep`
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `data/windows/.gitkeep`
- `data/external/.gitkeep`

## Implementation Steps

### 1. Create `pyproject.toml`

Use `setuptools` with a `[project]` section. Dependencies:

```
python >=3.11
numpy
pandas
scipy
scikit-learn
pywavelets
ruptures
tsfresh
torch    # Only for potential future use; not required for baselines
pytorch-lightning
shap
captum
fastapi
uvicorn[standard]
websockets
mlflow
pyyaml
h5py
pytest
pytest-cov
jupyter
matplotlib
seaborn
```

Include a `[project.optional-dependencies]` group for `dev` with testing tools.

### 2. Create `config/config.yaml`

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  windows_dir: "data/windows"
  external_dir: "data/external"

ingestion:
  target_frequency_hz: 1.0
  gap_threshold_s: 5
  z_score_window_h: 6
  window_length_s: 1200      # 20 minutes
  window_stride_s: 10
  forecast_horizon_s: 1800   # 30 minutes

synthetic_data:
  total_days: 90
  seed: 42
  quiet_baseline_soft: 1e-7
  quiet_baseline_hard: 1e-8
  noise_std_soft: 1e-9
  noise_std_hard: 1e-10
  dropout_prob: 0.001
  flare_rates:
    B: 8   # per 90 days
    C: 4
    M: 2
    X: 1

features:
  layers: [L1, L2, L3]
  rolling_windows_s: [60, 300]

models:
  random_forest:
    n_estimators: 200
    max_depth: 15
    class_weight: "balanced"
  logistic_regression:
    class_weight: "balanced"
    max_iter: 1000

evaluation:
  test_months: 2
  tss_threshold: 0.6
  brier_threshold: 0.08
  false_alarm_threshold: 0.25
```

### 3. Create `.gitignore`

```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
data/raw/*
data/processed/*
data/windows/*
*.parquet
*.h5
mlruns/
node_modules/
.vite/
```

### 4. Create empty `__init__.py` files

- `src/__init__.py`
- `tests/__init__.py`

### 5. Create `.gitkeep` files in all data subdirectories

## Acceptance Criteria

- [ ] `pip install -e ".[dev]"` completes without errors
- [ ] `python -c "import numpy; import pandas; import scipy; import sklearn; import torch; import shap; import fastapi"` succeeds
- [ ] `config/config.yaml` is parseable: `python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"`
- [ ] Directory structure matches the file map in MVP_PLAN.md
- [ ] `.gitignore` ignores all listed patterns

## Verification

```bash
python -c "
import yaml
cfg = yaml.safe_load(open('config/config.yaml'))
print(f'Config loaded: {len(cfg)} sections')
assert 'ingestion' in cfg
assert 'models' in cfg
assert 'synthetic_data' in cfg
print('OK')
"
```
