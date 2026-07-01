# FlareClassifier MVP — Agentic Implementation Plan

## Understanding Summary

- **What**: End-to-end MVP of Aditya-L1 Solar Flare Forecasting System
- **Why**: Demonstrate AI-powered early warning for solar flares from X-ray telemetry
- **Who**: Single operator/scientist using a web dashboard
- **Data**: Synthetic SoLEXS (soft X-ray) + HEL1OS (hard X-ray) time-series telemetry
- **Scope**: All 7 PRD phases at minimal fidelity — data pipeline → baseline ML → API → dashboard
- **Key constraint**: Agentic execution — every task is self-contained with clear inputs/outputs/verification

## Assumptions

| # | Assumption | Source |
|---|-----------|--------|
| A1 | No real Aditya-L1 telemetry available | User confirmed |
| A2 | MVP uses classical baselines (LR + RF), not Transformer | PRD Phase 4: "Baseline models required first" |
| A3 | Synthetic data embeds ground-truth labels (known flare events) | Enables verification without external catalogue |
| A4 | Rolling z-score normalization (6h window) per PRD 3.1.3 | PRD requirement |
| A5 | 20-minute sliding windows, 10s stride, 30-min forecast horizon | PRD 3.1.4 |
| A6 | Temporal train/test split (no random split) | PRD Evaluation Framework |
| A7 | Dashboard served locally (single-operator, no auth) | PRD §8 — Out of Scope |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AGENT EXECUTION LAYER                          │
│  Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → ...       │
└─────────────────────────────────────────────────────────────────────┘

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Synthetic │→  │  Data    │→  │ Feature  │→  │ Physics  │→  │   ML     │
│  Data Gen │   │Ingestion │   │  Engine  │   │  Layer   │   │  Model   │
│  (Task 2) │   │ (Task 3) │   │ (Task 4) │   │ (Task 5) │   │ (Task 6) │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘
                                                                  │
                         ┌────────────────────────────────────────┘
                         ▼
                    ┌──────────┐   ┌──────────┐   ┌──────────┐
                    │   API    │←──│ Explain  │   │Dashboard │
                    │ (Task 8) │   │ (Task 7) │   │ (Task 9) │
                    └────┬─────┘   └──────────┘   └──────────┘
                         │
                    ┌────▼─────┐
                    │Integration│
                    │ (Task 10) │
                    └──────────┘
```

## Task Dependency Graph

```
Task 1 (Scaffolding) ───► all tasks
      │
      ▼
Task 2 (Data Gen) ──────► Task 3 (Ingestion)
                              │
                              ▼
                         Task 4 (Features)
                              │
                              ▼
                         Task 5 (Physics)
                              │
                              ▼
                         Task 6 (Models) ──► Task 7 (Explain)
                              │                    │
                              ▼                    │
                         Task 8 (API) ◄────────────┘
                              │
                              ▼
                         Task 9 (Dashboard)
                              │
                              ▼
                         Task 10 (Integration)
```

## Technology Choices

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Data format | Parquet | Columnar, efficient for time-series |
| Feature extraction | pandas + numpy + scipy | PRD stack |
| ML baselines | scikit-learn | LogisticRegression + RandomForest |
| Deep learning (future) | PyTorch | When we add Transformer |
| Experiment tracking | MLflow | Model versioning, metric history |
| API framework | FastAPI | Async, WebSocket support |
| Dashboard | React + Recharts + shadcn/ui | PRD recommendation |
| Synthetic data | Custom generator (numpy) | Need control over flare parameters |

## File Map

```
FlareClassifier/
├── CLAUDE.md                         # Project agent config
├── MVP_PLAN.md                       # This file
├── .claude/tasks/task-*.md           # Individual task instructions
├── pyproject.toml                    # Python deps + project config
├── requirements.txt                  # Pinned deps (generated from pyproject.toml)
├── config/
│   └── config.yaml                   # Pipeline parameters
├── data/
│   ├── raw/                          # Generated synthetic data
│   ├── processed/                    # Cleaned, normalized data
│   ├── windows/                      # Windowed training data
│   └── external/                     # GOES catalogue (if available)
├── src/
│   ├── data_generation/
│   │   └── synthetic_flare_generator.py
│   ├── ingestion/
│   │   ├── synchronizer.py
│   │   ├── cleaner.py
│   │   ├── normalizer.py
│   │   └── windowing.py
│   ├── features/
│   │   ├── dynamics.py
│   │   ├── cross_channel.py
│   │   └── pipeline.py
│   ├── physics/
│   │   ├── stage_indicators.py
│   │   ├── derived_ratios.py
│   │   └── historical_context.py
│   ├── models/
│   │   ├── baselines.py
│   │   └── evaluate.py
│   ├── explainability/
│   │   └── explainer.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   ├── schemas.py
│   │   └── inference.py
│   └── dashboard/
│       ├── package.json
│       ├── vite.config.ts
│       ├── tsconfig.json
│       ├── index.html
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── components/
│           │   ├── TimeSeriesPlot.tsx
│           │   ├── ProbabilityGauge.tsx
│           │   ├── StateIndicator.tsx
│           │   ├── ShapChart.tsx
│           │   └── ExplanationCard.tsx
│           └── api/
│               └── websocket.ts
├── tests/
│   ├── test_synthetic_data.py
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_physics.py
│   ├── test_models.py
│   ├── test_explainability.py
│   └── test_api.py
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_analysis.ipynb
    └── 03_model_evaluation.ipynb
```

## Decision Log

| Decision | Alternative | Rationale |
|----------|-------------|-----------|
| Synthetic data first | Wait for real data | Unblocks all development |
| Classical baselines (LR+RF) | Transformer/Deep Learning | PRD requirement; faster iteration |
| FastAPI + WebSocket | REST-only | Real-time dashboard requirement |
| Rolling z-score norm | Min-max or global norm | PRD 3.1.3 requirement |
| 20-min windows, 10s stride | Other sizes | PRD 3.1.4 requirement |
| Temporal split | Random split | PRD: no future leakage |
| Parquet storage | CSV/HDF5 | PRD recommendation + performance |

## Execution Guide

The agent should execute tasks in order. After each task:

1. Verify all acceptance criteria pass
2. Run the task's test suite
3. Mark the task as complete before proceeding
4. If a task fails, fix it before moving to the next
