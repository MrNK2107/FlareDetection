# Task 15: Dual-Stream Temporal Transformer (Primary Architecture)

**Dependencies**: Tasks 11, 14
**PRD Reference**: docs/05 §2, §3 (architecture + focal loss); docs/07 §2 (attention serialization)

## Objective
Implement the target architecture: independent soft/hard encoders (1D Temporal CNN kernels 3/7/15 → positional encoding → Transformer encoder 4 heads/2 layers/d_model=128, weights NOT shared), cross-attention layer (Q=hard, K/V=soft, weights stored), fusion with engineered feature vector → linear projection to 256-dim, and 3 heads: flare probability (sigmoid), lead time regression (log-scale), severity softmax [B/C/M/X].

## Implementation Notes
- `src/models/transformer.py`:
  - `StreamEncoder`: Conv1d stack (kernel 3→7→15, channels → d_model=128) + sinusoidal positional encoding + `nn.TransformerEncoder` (2 layers, 4 heads, d_model=128, batch_first, dropout=0.1 **attn-dropout enabled for MC-Dropout**).
  - `CrossAttention`: `nn.MultiheadAttention` (Q from hard stream, K/V from soft stream); return attention weights (avg over heads) — these are the explainability artifact.
  - `Fusion`: concat(cross-attended hard summary, pooled soft context, engineered feature vector) → Linear → 256 → LayerNorm → ReLU.
  - Heads: (1) flare sigmoid, (2) lead-time regression on log-minutes (+ variance head for CI), (3) severity softmax over {B,C,M,X} given flare.
  - `forward(..., return_attention=True)`.
- Losses: Focal loss (gamma=2, alpha=0.75) for flare head — **no BCE**; weighted CE for severity; Huber on log-lead-time. Total: `0.5*L_focal + 0.3*L_sev + 0.2*L_lead`.
- MC-Dropout uncertainty: K=20 stochastic forwards at inference → std of flare probability + percentile CI on lead time (`lead_time_ci_90`).
- Engineered feature input: use the RF feature matrix row aligned by window timestamp (join on metadata); z-score with train stats; feed same feature count as training.
- Training: same temporal split; label heads from window labels (lead-time target = time from window end to next flare peak, from generator's flare catalogue; censored/None windows excluded from lead-time loss via mask).
- Save `models/transformer.pt` + `models/transformer_meta.json` (feature names, norm stats, head config).

## Files
- `src/models/transformer.py`, `src/models/focal_loss.py`, `scripts/train_transformer.py`
- `tests/test_models.py` — forward shapes; attention weights returned and normalized; focal loss vs CE sanity; overfit-tiny-batch; save/load round-trip

## Acceptance Criteria
- [ ] CPU-trainable (<30 min for a few epochs on DL windows)
- [ ] Attention weights extractable per forward pass, shape (T,)
- [ ] Payload fields flare_probability / severity_probs / lead time + CI / model_uncertainty all producible from this model
- [ ] Tests pass
