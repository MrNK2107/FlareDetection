# Task 11: Data Scale-Up + Hybrid-Stride Windowing + Temporal Split

**Dependencies**: Tasks 1–10 (complete)
**PRD Reference**: Phase 1 (§3.1.4), Evaluation Framework §1

## Objective
Scale synthetic data to 180 days (6-month held-out requirement), keep the full pipeline tractable via hybrid strides, and make the train/test split spec-compliant (test set = last 6 months... of a 180-day dataset means train/test by time, test = final 30 days minimum; see decision log).

## Key Decisions
- **Hybrid stride**: feature/label windows at 10s stride (streaming, no materialization); raw window arrays for DL at 60s stride, float32, memmap-backed. Reason: 180d × 86400s × 10s-stride raw windows ≈ 30 TB — impossible; 60s stride ≈ manageable (~GBs).
- Window length stays 20 min; forecast horizon 30 min; labels B/C/M/X/None from future-window max class.

## Files
- `config/config.yaml` — `total_days: 180`, `raw_window_stride_s: 60`, `test_days: 30`, L4/L5 layer flags
- `src/data_generation/synthetic_flare_generator.py` — scale flare rates proportionally, keep seeded/reproducible; write in monthly chunks to bound memory
- `src/ingestion/windowing.py` — add `build_raw_window_memmap(stride_s=60, dtype=float32)` writing `data/windows/X_soft_dl.npy` / `X_hard_dl.npy` via `np.lib.format.open_memmap`
- `src/models/train.py::prepare_training_data` — temporal split by timestamp: `test_days` from config (default 30), assert max(train_ts) < min(test_ts); return timestamps
- `tests/test_ingestion.py`, `tests/test_models.py` — add: split temporality test, memmap shape/stride test, chunked generation reproducibility test

## Acceptance Criteria
- [ ] Generator produces 180 days in <10 min, <2 GB peak RAM, same seed → identical flare catalogue
- [ ] Raw DL windows exist as float32 memmaps with stride 60s and aligned metadata timestamps
- [ ] Temporal split: all test timestamps strictly after all train timestamps
- [ ] Label distribution reported; B+ events exist in both train and test
- [ ] `pytest tests/` passes
