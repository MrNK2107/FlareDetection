# Task 14: LSTM Baseline

**Dependencies**: Tasks 11, 13
**PRD Reference**: docs/05 §1 (LSTM: 2-layer, hidden 128, raw windowed signals)

## Objective
Implement the neural baseline LSTM on raw windowed signals (no feature engineering), trained on the DL memmap windows, evaluated with the same metric suite (TSS, Brier, FAR, detection rate, accuracy as secondary only).

## Implementation Notes
- New dependency: torch (CPU). Add to pyproject/requirements with CPU wheel index note in README.
- Architecture: input (2 channels) → Linear embed → 2-layer LSTM(hidden=128, batch_first, dropout=0.2) → attention-free mean+max pool → FC head → 5-class logits (None/B/C/M/X) + binary flare logit.
- Loss: weighted CE (inverse-frequency class weights). Class imbalance handled by weights + threshold tuning after training.
- Train on DL memmap windows with temporal split (train idx / test idx by timestamp metadata); batch via `np.memmap` slices; standardize per-channel using train-set stats.
- Save: `models/lstm.pt` (state_dict + config via torch.save), `models/lstm_meta.json` (normalization stats, threshold, feature mode).
- Evaluate with `evaluate_model`-equivalent for torch (predict_proba wrapper) so results join the same `evaluation_results.json`.

## Files
- `src/models/lstm.py` (new: model + train/eval functions)
- `scripts/train_lstm.py` (entry)
- `tests/test_models.py` — model forward shape test; overfit-tiny-batch test (loss decreases); save/load round-trip; TSS range on synthetic eval

## Acceptance Criteria
- [ ] Training runs on CPU (<15 min on the 180-day DL windows)
- [ ] Tiny-batch overfit test passes (train loss → ~0 on 1 batch)
- [ ] Evaluation metrics written into the shared results file
- [ ] All tests pass
