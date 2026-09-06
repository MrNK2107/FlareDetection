"""LSTM baseline: 2-layer, hidden 128, on raw windowed signals (no feature
engineering), per docs/05 §1. Trains on the DL window memmaps produced by the
ingestion pipeline (data/windows/X_soft.npy / X_hard.npy @ dl cadence)."""
import json
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

from src.models.evaluate import evaluate_model, save_evaluation_results
from src.models.splits import temporal_train_test_masks

CLASS_MAP = {'None': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FlareLSTM(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2, n_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2)
        out, _ = self.lstm(x)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


class TorchModelWrapper:
    """sklearn-style adapter so torch models plug into evaluate_model()."""

    def __init__(self, model: nn.Module, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.device = device
        self.classes_ = np.arange(5)

    def _to_windows(self, X_flat: np.ndarray) -> torch.Tensor:
        n, d = X_flat.shape
        t = d // 2
        x = X_flat.reshape(n, t, 2).astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_windows(X_flat))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X_flat: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._to_windows(X_flat))
        return torch.softmax(logits, dim=1).cpu().numpy()


def load_dl_dataset(
    window_dir: str = "data/windows",
    metadata_filename: str = "dl_window_metadata.parquet",
    test_days: float = 30.0,
    max_train_windows: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    window_dir = Path(window_dir)
    X_soft = np.load(window_dir / 'X_soft.npy', mmap_mode='r')
    X_hard = np.load(window_dir / 'X_hard.npy', mmap_mode='r')
    y = np.load(window_dir / 'y_labels.npy')
    meta = pd.read_parquet(window_dir / metadata_filename)
    n = min(len(X_soft), len(X_hard), len(y), len(meta))
    meta = meta.iloc[:n].reset_index(drop=True)
    ts = pd.DatetimeIndex(pd.to_datetime(meta['window_start']))
    train_mask, test_mask = temporal_train_test_masks(ts, test_days)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if max_train_windows is not None and len(train_idx) > max_train_windows:
        rng = np.random.default_rng(seed)
        pos = train_idx[y[train_idx] > 0]
        quiet = train_idx[y[train_idx] == 0]
        n_quiet = max(0, max_train_windows - len(pos))
        keep = np.sort(np.concatenate([
            pos, rng.choice(quiet, size=min(n_quiet, len(quiet)), replace=False)
        ]))
        train_idx = keep
    return {
        'X_soft': X_soft, 'X_hard': X_hard, 'y': y,
        'train_idx': train_idx, 'test_idx': test_idx, 'timestamps': ts,
        'split_time': split_time,
    }


def _channel_norm_stats(X_soft, X_hard, idx) -> Dict[str, float]:
    soft_sample = np.asarray(X_soft[idx], dtype=np.float64)
    hard_sample = np.asarray(X_hard[idx], dtype=np.float64)
    return {
        'soft_mean': float(soft_sample.mean()), 'soft_std': float(soft_sample.std() + 1e-8),
        'hard_mean': float(hard_sample.mean()), 'hard_std': float(hard_sample.std() + 1e-8),
    }


def _make_batch(X_soft, X_hard, idx, stats) -> np.ndarray:
    soft = (np.asarray(X_soft[idx], dtype=np.float64) - stats['soft_mean']) / stats['soft_std']
    hard = (np.asarray(X_hard[idx], dtype=np.float64) - stats['hard_mean']) / stats['hard_std']
    return np.stack([soft, hard], axis=-1)  # (n, T, 2)


def train_lstm(config_path: str = "config/config.yaml") -> Dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    lc = config['models']['lstm']
    eval_cfg = config.get('evaluation', {})
    data = load_dl_dataset(
        window_dir=config['data']['windows_dir'],
        test_days=float(eval_cfg.get('test_days', 30)),
        max_train_windows=lc.get('max_train_windows'),
        seed=config['synthetic_data']['seed'],
    )
    y = data['y']
    stats = _channel_norm_stats(data['X_soft'], data['X_hard'], data['train_idx'])
    class_counts = np.bincount(y[data['train_idx']], minlength=5).astype(np.float64)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=DEVICE)

    torch.manual_seed(config['synthetic_data']['seed'])
    model = FlareLSTM(
        hidden_size=lc['hidden_size'], num_layers=lc['num_layers'], dropout=lc['dropout'],
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lc['lr'])
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_idx = data['train_idx']
    batch_size = lc['batch_size']
    model.train()
    for epoch in range(lc['epochs']):
        perm = np.random.permutation(len(train_idx))
        total_loss, n_batches = 0.0, 0
        for s in range(0, len(train_idx), batch_size):
            sel = train_idx[perm[s:s + batch_size]]
            xb = torch.from_numpy(_make_batch(data['X_soft'], data['X_hard'], sel, stats).astype(np.float32)).to(DEVICE)
            yb = torch.from_numpy(y[sel].astype(np.int64)).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
        print(f"    epoch {epoch + 1}/{lc['epochs']} loss={total_loss / max(n_batches, 1):.4f}")

    wrapper = TorchModelWrapper(model)
    t = data['X_soft'].shape[1]
    X_test_flat = _make_batch(data['X_soft'], data['X_hard'], data['test_idx'], stats)
    X_test_flat = X_test_flat.reshape(len(data['test_idx']), -1)
    results = evaluate_model(wrapper, X_test_flat, y[data['test_idx']], "LSTM")
    print(f"    TSS={results['tss']:.4f}, Brier={results['brier_score']:.4f}")

    save_path = Path('models/lstm.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'config': {k: lc[k] for k in ('hidden_size', 'num_layers', 'dropout')},
        'norm_stats': stats,
        'window_steps': t,
    }, save_path)
    with open('models/lstm_meta.json', 'w') as f:
        json.dump({'norm_stats': stats, 'window_steps': t, 'split_time': str(data['split_time'])}, f, indent=2)

    # merge into shared evaluation results
    results_path = Path('models/evaluation_results.json')
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results['LSTM'] = results
    save_evaluation_results(all_results, str(results_path))
    return results


if __name__ == "__main__":
    import sys
    train_lstm(sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml")
