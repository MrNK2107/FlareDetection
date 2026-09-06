"""Dual-Stream Temporal Transformer (docs/05 §2).

Independent soft/hard encoders (Temporal CNN kernels 3/7/15 -> positional
encoding -> Transformer encoder 4 heads / 2 layers / d_model=128, weights NOT
shared), cross-attention (Q=hard, K/V=soft, weights stored), fusion with the
engineered feature vector -> 256-dim, and three heads:
  (1) flare probability (sigmoid), (2) lead time regression on log-minutes,
  (3) severity softmax over {B, C, M, X}.
Loss: 0.5*focal + 0.3*weighted-CE-severity + 0.2*huber(log-lead-time).
Uncertainty: MC-Dropout (attn dropout enabled for MC sampling at inference).
"""
import json
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.models.focal_loss import FocalLoss
from src.models.evaluate import evaluate_model, save_evaluation_results

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEVERITY_CLASSES = ['B', 'C', 'M', 'X']


class StreamEncoder(nn.Module):
    """Temporal CNN (kernels 3, 7, 15) -> positional encoding -> Transformer."""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=15, padding=7),
            nn.ReLU(),
        )
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) -> (B, T, d_model)
        z = self.conv(x.unsqueeze(1)).transpose(1, 2)
        z = self.pos_encoding(z)
        return self.transformer(z)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class CrossAttention(nn.Module):
    """Q from hard stream, K/V from soft stream. Attention weights returned."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, hard: torch.Tensor, soft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attn(hard, soft, soft, average_attn_weights=True)
        return attended, weights  # weights: (B, T_hard, T_soft)


class DualStreamTransformer(nn.Module):
    def __init__(
        self,
        n_engineered_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        conv_kernels: Tuple[int, ...] = (3, 7, 15),
        fusion_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert tuple(conv_kernels) == (3, 7, 15), "conv_kernels fixed to (3, 7, 15) per spec"
        self.soft_encoder = StreamEncoder(d_model, nhead, num_layers, dropout)
        self.hard_encoder = StreamEncoder(d_model, nhead, num_layers, dropout)
        self.cross_attention = CrossAttention(d_model, nhead, dropout)
        self.feature_proj = nn.Linear(n_engineered_features, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.flare_head = nn.Linear(fusion_dim, 1)
        self.lead_time_head = nn.Linear(fusion_dim, 2)  # log-minutes mean, log-variance
        self.severity_head = nn.Linear(fusion_dim, len(SEVERITY_CLASSES))
        self.last_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        soft: torch.Tensor,
        hard: torch.Tensor,
        engineered: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        soft_enc = self.soft_encoder(soft)   # (B, T, d)
        hard_enc = self.hard_encoder(hard)   # (B, T, d)
        cross, attn_w = self.cross_attention(hard_enc, soft_enc)
        self.last_attention = attn_w.detach()
        pooled_soft = soft_enc.mean(dim=1)
        pooled_cross = cross.mean(dim=1)
        feat = self.feature_proj(engineered)
        fused = self.fusion(torch.cat([pooled_cross, pooled_soft, feat], dim=1))
        flare_logit = self.flare_head(fused).squeeze(-1)
        lead_mean, lead_logvar = self.lead_time_head(fused).split(1, dim=-1)
        severity_logits = self.severity_head(fused)
        out = {
            'flare_logit': flare_logit,
            'flare_prob': torch.sigmoid(flare_logit),
            'lead_log_mean': lead_mean.squeeze(-1),
            'lead_log_var': lead_logvar.squeeze(-1),
            'severity_logits': severity_logits,
        }
        if return_attention:
            out['attention'] = attn_w
        return out


class TransformerWrapper:
    """sklearn-style adapter for evaluate_model()."""

    def __init__(self, model: DualStreamTransformer, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.device = device
        self.classes_ = np.arange(5)

    def predict_proba(self, X_flat: np.ndarray) -> np.ndarray:
        self.model.eval()
        n, d = X_flat.shape
        t = d - self.model.feature_proj.in_features
        t = t // 2
        soft = X_flat[:, :t]
        hard = X_flat[:, t:2 * t]
        eng = X_flat[:, 2 * t:]
        with torch.no_grad():
            out = self.model(
                torch.from_numpy(soft.astype(np.float32)).to(self.device),
                torch.from_numpy(hard.astype(np.float32)).to(self.device),
                torch.from_numpy(eng.astype(np.float32)).to(self.device),
            )
        p_flare = out['flare_prob'].cpu().numpy()
        sev = torch.softmax(out['severity_logits'], dim=1).cpu().numpy()
        proba = np.column_stack([1 - p_flare, sev * p_flare[:, None]])
        return proba

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_flat).argmax(axis=1)


def _mc_dropout_predict(model, soft, hard, eng, n_samples: int) -> Dict[str, np.ndarray]:
    """MC-Dropout: enable dropout at inference, run n_samples stochastic forwards."""
    model.train()  # enables dropout layers
    # keep BN-like layers frozen by only enabling dropout modules
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        else:
            m.eval()
    probs, lead_means, lead_vars = [], [], []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(soft, hard, eng)
            probs.append(out['flare_prob'].cpu().numpy())
            lead_means.append(out['lead_log_mean'].cpu().numpy())
            lead_vars.append(out['lead_log_var'].cpu().numpy())
    probs = np.stack(probs)         # (K, B)
    lead_means = np.stack(lead_means)
    lead_vars = np.stack(lead_vars)
    model.eval()
    return {
        'flare_prob_samples': probs,
        'flare_prob_mean': probs.mean(axis=0),
        'flare_prob_std': probs.std(axis=0),
        'lead_log_mean': lead_means.mean(axis=0),
        'lead_total_var': lead_vars.mean(axis=0) + lead_means.var(axis=0),
    }


def _lead_targets(meta: pd.DataFrame, catalogue: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Minutes from window_end to next flare peak (from the generator's flare
    catalogue). Returns (target_minutes, mask) where mask=False for windows
    with no future flare (censored -> excluded from lead-time loss)."""
    ts = pd.DatetimeIndex(pd.to_datetime(meta['window_end']))
    targets = np.full(len(meta), np.nan)
    if len(catalogue) == 0:
        return targets, np.zeros(len(meta), dtype=bool)
    peaks = pd.DatetimeIndex(pd.to_datetime(catalogue['peak_utc'])).sort_values()
    peak_arr = peaks.values.astype('datetime64[s]').astype(np.int64)
    ts_arr = ts.values.astype('datetime64[s]').astype(np.int64)
    next_peak_idx = np.searchsorted(peak_arr, ts_arr, side='left')
    valid = next_peak_idx < len(peak_arr)
    idx = np.where(valid)[0]
    delta_s = peak_arr[next_peak_idx[idx]] - ts_arr[idx]
    targets[idx] = delta_s / 60.0
    # exclude targets beyond 24h as effectively censored
    mask = valid & (targets <= 1440)
    targets = np.clip(targets, 1.0, 1440.0)
    return targets, mask


def train_transformer(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tc = config['models']['transformer']
    eval_cfg = config.get('evaluation', {})
    seed = config['synthetic_data']['seed']

    # data: DL windows + engineered features joined on window_start
    window_dir = Path(config['data']['windows_dir'])
    processed_dir = Path(config['data']['processed_dir'])
    X_soft = np.load(window_dir / 'X_soft.npy', mmap_mode='r')
    X_hard = np.load(window_dir / 'X_hard.npy', mmap_mode='r')
    y = np.load(window_dir / 'y_labels.npy')
    dl_meta = pd.read_parquet(window_dir / 'dl_window_metadata.parquet')
    features = pd.read_parquet(processed_dir / 'full_feature_matrix.parquet')
    feature_cols = [c for c in features.columns if c != 'window_start']
    features = features.set_index('window_start')
    feats = features.reindex(pd.DatetimeIndex(pd.to_datetime(dl_meta['window_start'])))
    n = min(len(X_soft), len(X_hard), len(y), len(dl_meta), len(feats))
    X_soft, X_hard, y = X_soft[:n], X_hard[:n], y[:n]
    dl_meta = dl_meta.iloc[:n].reset_index(drop=True)
    eng = feats.iloc[:n][feature_cols].to_numpy(dtype=np.float64)
    eng = np.nan_to_num(eng, nan=0.0, posinf=0.0, neginf=0.0)

    ts = pd.DatetimeIndex(pd.to_datetime(dl_meta['window_start']))
    split_time = ts.max() - pd.Timedelta(days=float(eval_cfg.get('test_days', 30)))
    train_mask = ts < split_time
    test_mask = ~train_mask
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Temporal split produced an empty side; reduce test_days")

    catalogue_path = Path(config['data']['external_dir']) / 'flare_catalogue.parquet'
    catalogue = pd.read_parquet(catalogue_path) if catalogue_path.exists() else pd.DataFrame()
    lead_targets, lead_mask_all = _lead_targets(dl_meta, catalogue)

    # normalization stats from train split
    eng_mean = eng[train_idx].mean(axis=0)
    eng_std = eng[train_idx].std(axis=0) + 1e-8
    eng_norm = (eng - eng_mean) / eng_std
    soft_all = np.asarray(X_soft, dtype=np.float64)
    hard_all = np.asarray(X_hard, dtype=np.float64)
    stats = {
        'soft_mean': float(soft_all[train_idx].mean()), 'soft_std': float(soft_all[train_idx].std() + 1e-8),
        'hard_mean': float(hard_all[train_idx].mean()), 'hard_std': float(hard_all[train_idx].std() + 1e-8),
    }
    soft_norm = (soft_all - stats['soft_mean']) / stats['soft_std']
    hard_norm = (hard_all - stats['hard_mean']) / stats['hard_std']

    # subsample train set for tractability
    max_train = tc.get('max_train_windows')
    if max_train is not None and len(train_idx) > max_train:
        rng = np.random.default_rng(seed)
        pos = train_idx[y[train_idx] > 0]
        quiet = train_idx[y[train_idx] == 0]
        n_quiet = max(0, max_train - len(pos))
        train_idx = np.sort(np.concatenate([
            pos, rng.choice(quiet, size=min(n_quiet, len(quiet)), replace=False)
        ]))

    torch.manual_seed(seed)
    model = DualStreamTransformer(
        n_engineered_features=eng.shape[1],
        d_model=tc['d_model'], nhead=tc['nhead'], num_layers=tc['num_layers'],
        conv_kernels=tuple(tc.get('conv_kernels', [3, 7, 15])),
        fusion_dim=tc['fusion_dim'], dropout=tc['dropout'],
    ).to(DEVICE)
    focal = FocalLoss(gamma=tc['focal_gamma'], alpha=tc['focal_alpha'])
    sev_weights = torch.tensor(
        np.bincount(y[train_idx], minlength=5)[1:].sum() / np.maximum(np.bincount(y[train_idx], minlength=5)[1:], 1),
        dtype=torch.float32, device=DEVICE,
    )
    huber = nn.HuberLoss(delta=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    w_flare, w_sev, w_lead = tc.get('loss_weights', [0.5, 0.3, 0.2])

    batch_size = tc['batch_size']
    flare_y = (y > 0).astype(np.float32)
    lead_train_mask = lead_mask_all[train_idx]
    for epoch in range(tc['epochs']):
        perm = np.random.permutation(len(train_idx))
        totals = {'loss': 0.0, 'focal': 0.0, 'sev': 0.0, 'lead': 0.0}
        n_batches = 0
        for s in range(0, len(train_idx), batch_size):
            sel = train_idx[perm[s:s + batch_size]]
            sb = torch.from_numpy(soft_norm[sel].astype(np.float32)).to(DEVICE)
            hb = torch.from_numpy(hard_norm[sel].astype(np.float32)).to(DEVICE)
            eb = torch.from_numpy(eng_norm[sel].astype(np.float32)).to(DEVICE)
            fb = torch.from_numpy(flare_y[sel]).to(DEVICE)
            yb = torch.from_numpy(y[sel].astype(np.int64)).to(DEVICE)
            opt.zero_grad()
            out = model(sb, hb, eb)
            l_flare = focal(out['flare_logit'], fb)
            # severity: CE over 4 classes; for quiet windows use index 0 target
            sev_target = torch.clamp(yb - 1, min=0)
            sev_mask = (yb > 0).float()
            l_sev_raw = torch.nn.functional.cross_entropy(
                out['severity_logits'], sev_target, weight=sev_weights, reduction='none'
            )
            l_sev = (l_sev_raw * sev_mask).sum() / torch.clamp(sev_mask.sum(), min=1.0)
            lm = torch.from_numpy(lead_train_mask).to(DEVICE)
            sel_lead_mask = lm[torch.arange(len(sel), device=DEVICE)]
            if sel_lead_mask.any():
                log_minutes = torch.log(torch.from_numpy(
                    np.clip(lead_targets[train_idx][perm[s:s + batch_size]][sel_lead_mask.cpu().numpy()], 1.0, 1440.0)
                ).float()).to(DEVICE)
                l_lead = huber(out['lead_log_mean'][sel_lead_mask], log_minutes)
            else:
                l_lead = torch.tensor(0.0, device=DEVICE)
            loss = w_flare * l_flare + w_sev * l_sev + w_lead * l_lead
            loss.backward()
            opt.step()
            totals['loss'] += float(loss)
            totals['focal'] += float(l_flare)
            totals['sev'] += float(l_sev)
            totals['lead'] += float(l_lead)
            n_batches += 1
        print(f"    epoch {epoch + 1}/{tc['epochs']} "
              f"loss={totals['loss'] / n_batches:.4f} "
              f"(focal={totals['focal'] / n_batches:.4f} sev={totals['sev'] / n_batches:.4f} "
              f"lead={totals['lead'] / n_batches:.4f})")

    # ------- evaluation (shared metric suite)
    wrapper = TransformerWrapper(model)
    t = X_soft.shape[1]
    X_test_flat = np.concatenate([
        soft_norm[test_idx].astype(np.float32),
        hard_norm[test_idx].astype(np.float32),
        eng_norm[test_idx].astype(np.float32),
    ], axis=1)
    results = evaluate_model(wrapper, X_test_flat, y[test_idx], "Transformer")
    print(f"    TSS={results['tss']:.4f}, Brier={results['brier_score']:.4f}")

    # MC-Dropout uncertainty check on a test slice
    k = tc.get('mc_dropout_samples', 20)
    sample_slice = slice(0, min(64, len(test_idx)))
    mc = _mc_dropout_predict(
        model,
        torch.from_numpy(soft_norm[test_idx][sample_slice].astype(np.float32)).to(DEVICE),
        torch.from_numpy(hard_norm[test_idx][sample_slice].astype(np.float32)).to(DEVICE),
        torch.from_numpy(eng_norm[test_idx][sample_slice].astype(np.float32)).to(DEVICE),
        k,
    )
    results['mc_dropout_std_mean'] = float(mc['flare_prob_std'].mean())
    print(f"    MC-dropout std (mean)={results['mc_dropout_std_mean']:.4f}")

    save_path = Path('models/transformer.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'n_engineered_features': eng.shape[1],
            'd_model': tc['d_model'], 'nhead': tc['nhead'], 'num_layers': tc['num_layers'],
            'fusion_dim': tc['fusion_dim'], 'dropout': tc['dropout'],
        },
        'norm_stats': {
            'eng_mean': eng_mean.tolist(), 'eng_std': eng_std.tolist(),
            'feature_names': feature_cols,
            **stats,
        },
        'window_steps': t,
    }, save_path)
    with open('models/transformer_meta.json', 'w') as f:
        json.dump({
            'norm_stats': {'eng_mean': eng_mean.tolist(), 'eng_std': eng_std.tolist(),
                           'feature_names': feature_cols, **stats},
            'window_steps': t,
            'split_time': str(split_time),
            'mc_dropout_samples': k,
        }, f, indent=2)

    results_path = Path('models/evaluation_results.json')
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results['Transformer'] = results
    save_evaluation_results(all_results, str(results_path))
    return results


if __name__ == "__main__":
    import sys
    train_transformer(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
