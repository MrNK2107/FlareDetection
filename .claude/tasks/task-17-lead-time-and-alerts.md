# Task 17: Real Lead Time + Full Prediction Payload + Alert Channels

**Dependencies**: Task 15 (lead-time head), Task 16
**PRD Reference**: docs/06 §4 payload spec; docs/08 §3 (alert channels, suppression)

## Objective
Replace the placeholder lead-time estimator (`15 + random ±3`) with the Transformer lead-time head (primary) and an RF-quantile fallback (for classical-only serving), and complete the payload + alerting to spec.

## Implementation Notes
- `src/models/lead_time.py`: wrapper exposing `predict(features_row, window) -> (lead_min, ci90)` using transformer head; fallback: `RandomForestRegressor`/`GradientBoostingRegressor` trained on flare windows (target = minutes-to-peak), with residual-based 90% CI from training quantiles.
- Payload completeness check vs docs/06 §4 table: all 10 fields present and correctly typed; `expected_lead_time_min=None` iff `flare_probability < 0.3`; severity probs sum ≤ 1; `inference_timestamp_utc` timezone-aware UTC.
- Alerts (`src/api/alerting.py`):
  - Channel abstraction: `AlertChannel` with `send(payload)`; implement `EmailChannel` (smtplib, config-driven, no-op when unconfigured), `WebhookChannel` (JSON POST via urllib/requests, Slack/Teams-compatible), `BrowserPushChannel` (Web Push — implement as server-sent WebSocket "alert" event to dashboard; VAPID push as stretch).
  - Rising-edge trigger at 0.5 with configurable quiet window (default 15 min) — already exists; keep + route to channels; persist alert history rows (feeds Task 18).
- Tests: channel no-op when unconfigured; webhook payload shape; suppression logic; payload spec compliance test (null rule, sums, types).

## Files
- `src/models/lead_time.py` (new), `src/api/alerting.py` (extend), `src/api/inference.py` (wire lead time), `tests/test_api.py`
- Config: `alerting:` section (smtp/webhook settings, quiet window, threshold)

## Acceptance Criteria
- [ ] Lead time comes from a trained model, CI is model-derived (no random jitter)
- [ ] Payload matches docs/06 §4 exactly (schema test enforces)
- [ ] Alert channels implemented + tested (email/webhook no-op-safe), alert history persisted
- [ ] Tests pass
