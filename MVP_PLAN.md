# FlareClassifier — Full-Spec Implementation Plan (v1.0)

## Status
MVP (Tasks 1–10) is **complete**: all 7 pipeline stages exist, 49 tests pass, models trained, API + dashboard skeleton operational. This plan closes the gap between the MVP and the full spec in `docs/` (which supersedes MVP scope).

## Gap Analysis (MVP → full spec)
| Area | Missing | Spec ref |
|---|---|---|
| Features | L4 frequency, L5 changepoint layers | docs/04 |
| Models | LSTM baseline; Dual-Stream Temporal Transformer (PyTorch not yet a dep) | docs/05 |
| State machine | No HMM → `solar_state` always "Unknown", `state_transition_probs` null | docs/05 §4 |
| Lead time | Placeholder (`15 + jitter ±3`), no model, no real CI | docs/06 §4 |
| Evaluation | No M+X-specific detection rate, no lead-time MAE; 7-day data vs 6-month held-out requirement | docs/06 |
| API | Attention weights not serialized; no deep-model SHAP path | docs/07 |
| Dashboard | Sends fake zero feature vectors; plots probabilities as flux; no attention heatmap/countdown/history | docs/08 |
| Alerts | `print()` only — no email/webhook/browser channels | docs/08 §3 |
| History | No inference-payload storage → no hindcast, no CSV export | docs/08 §4 |
| Continuous learning | No retraining triggers, registry, promotion gate, shadow mode | docs/08 §5 |

## User Decisions (locked)
1. **Full PyTorch** — LSTM baseline + Dual-Stream Transformer with attention explainability + MC-Dropout uncertainty.
2. **180 days of synthetic data** with **hybrid strides** (features at 10s, raw DL windows at 60s, float32 memmap).
3. **File-based model registry** (timestamped dirs + metadata + promotion gate) instead of MLflow.
4. **Plan + implement immediately** after plan approval.

## Decision Log (deviations from docs/, with rationale)
| Decision | Alternative | Rationale |
|---|---|---|
| Hybrid stride: features 10s, DL raw windows 60s | 10s everywhere | 180d@1Hz with 10s-stride raw windows ≈ 30 TB — infeasible; 60s ≈ GBs. Feature/label semantics preserved (labels still derived from 10s future windows) |
| 30-day held-out test period | 6 months | 6-month test set on a 6-month+ dataset leaves no training data; 30 days on 180 days gives 5:1 train:test — accepted deviation, documented |
| File registry | MLflow/W&B | Single-operator deployment; zero new heavy deps; spec's *behavior* (versioning, never overwrite, promotion gate) preserved |
| Browser push via WebSocket events | VAPID Web Push | No push server/keys needed; single-operator; VAPID noted as stretch |
| L4/L5 computed on DL-window cadence arrays | per-10s-row streaming | Frequency/changepoint features need long uniform windows; cost otherwise prohibitive |
| BOCPD implemented in numpy | full BOCPD lib | Small surface; avoids unmaintained deps |

## Task Order (dependencies enforced)
```
Task 11 (data scale + hybrid windowing + temporal split)
  → Task 12 (L4 frequency features)
  → Task 13 (L5 changepoint features)
  → Task 14 (LSTM baseline)
  → Task 15 (Dual-Stream Transformer)
  → Task 16 (HMM state machine → API payload)
  → Task 17 (model-based lead time + CI; alert channels; payload compliance)
  → Task 18 (SQLite history + hindcast/CSV + registry + promotion gate + shadow + drift)
  → Task 19 (dashboard: real data, attention heatmap, countdown, history view)
  → Task 20 (E2E pipeline + verify script + README + final verification)
```
Tasks 12/13 can parallelize after 11; 14 after 11+13; 15 after 14; 16 after 15; 17 after 15+16; 18 after 17; 19 after 15–18; 20 last.

## Execution Guide
- Execute tasks in numeric order; after each: run task tests + `pytest tests/`, fix failures before proceeding.
- New deps: `torch` (CPU), `hmmlearn`. Everything else already present (`pywavelets`, `ruptures` are in requirements).
- Final verification per CLAUDE.md: models exist, `models/evaluation_results.json` complete, `data/windows/X_soft.npy` shape correct, full test suite green.
