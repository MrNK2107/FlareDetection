# Task 19: Dashboard Completion

**Dependencies**: Tasks 15–18
**PRD Reference**: docs/08 §1–§4 (real-time view, explainability panel, alerts, historical view)

## Objective
Make the dashboard spec-compliant: real data flow (no fake zero-vectors), attention heatmap, lead-time countdown, alert notifications, and historical view.

## Implementation Notes
- **Real data flow (root fix)**: backend streams the *actual* synchronized flux series (from `data/processed/synchronized_clean.parquet` tail) + computed feature vectors over WebSocket, replacing the frontend's fake `new Array(43).fill(0)` and the "flux = B+C probs" hack. Server pushes `{type: 'telemetry', points: [...]}` and `{type: 'prediction', ...}` messages; frontend updates by type. Replay engine drives the stream from historical data at configurable speed (demo mode) — single-operator MVP has no live instrument.
- **Attention heatmap**: serialize top-level attention weights (20-min window, 2 channels aggregated) in prediction payload; new `AttentionHeatmap.tsx` renders window × channel gradient (bright yellow/white = high attention); highlight driver windows (weight > 0.1).
- **Lead-time countdown**: `CountdownTimer.tsx` — visible only when `flare_probability > 0.3`; MM:SS format from `expected_lead_time_min`; CI band display.
- **State indicator colors** per docs/08: Quiet green, Energy Accum yellow, Precursor orange, Initiation red, Peak dark red, Decay purple; animated arrow to most-probable next state (already partially there — add colors).
- **Gauge bands**: green <30%, amber 30–70%, red >70% (verify existing component).
- **Alerts**: browser Notification API on `alert` WS message; in-app alert banner + history list.
- **Historical view**: date-range picker → `GET /history/predictions?format=json` → hindcast probability curve vs flare event markers; CSV export button hitting the CSV endpoint.
- SHAP chart: switch from hardcoded to real `top_features` from payload (exists in explainer output).

## Files
- `src/api/server.py` (telemetry streaming + replay task), `src/dashboard/src/App.tsx` (message routing by type), new components `AttentionHeatmap.tsx`, `CountdownTimer.tsx`, `HistoryView.tsx`, `alert handling` in `websocket.ts`
- `tests/test_api.py` — WS message types + telemetry payload shape test

## Acceptance Criteria
- [ ] No fake/zero features sent; dashboard renders real synchronized flux + real features
- [ ] Attention heatmap, countdown, gauge bands, state colors match spec
- [ ] Historical view + CSV export work against the history API
- [ ] `npm run build` passes; WS tests pass
