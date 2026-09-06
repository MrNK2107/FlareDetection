"""Telemetry replay streamer (docs/08 §1.1).

The single-operator MVP has no live instrument feed, so the server replays the
synchronized telemetry and extracts the REAL feature pipeline per tick —
replacing the dashboard's previous fake zero-vector flow. Per tick it emits:
  - the past 60 minutes of dual-channel flux (10s cadence) for the plot
  - the past 20-minute window (DL cadence) for the deep path
  - the engineered feature vector (same pipeline as training)
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional

from src.features.pipeline import extract_features_for_window

PLOT_POINTS = 360          # 60 min at 10s cadence
PLOT_CADENCE_S = 10
DL_WINDOW_SAMPLES = 120    # 20 min at 10s cadence


class TelemetryStreamer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        data_path = Path(config['data']['processed_dir']) / 'synchronized_clean.parquet'
        df = pd.read_parquet(
            data_path,
            columns=['timestamp_utc', 'soft_flux_z', 'hard_flux_z', 'flare_class'],
        )
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(df['timestamp_utc']))
        self.soft = df['soft_flux_z'].to_numpy(dtype=np.float64)
        self.hard = df['hard_flux_z'].to_numpy(dtype=np.float64)
        self.flare_class = df['flare_class'].to_numpy(dtype=object)
        self.window_length_s = int(config['ingestion']['window_length_s'])
        self.layers = config.get('features', {}).get('layers', ['L1', 'L2', 'L3', 'L4', 'L5'])
        self.rolling_windows_s = config.get('features', {}).get('rolling_windows_s', [60, 300])
        self.position = self.window_length_s  # first valid window end
        self.speed = 1.0
        self.replay_step_s = 60.0
        self.paused = False
        self.flare_history = self._build_flare_history()

    def _build_flare_history(self) -> list:
        ts = self.timestamps
        mask = self.flare_class != 'None'
        changed = np.diff(np.concatenate([[False], mask.view(bool), [False]]))
        starts = np.where(changed)[0][::2]
        return [ts[i] for i in starts]

    def step(self) -> Optional[Dict]:
        """Advance one replay step and compute the tick payload."""
        if self.paused:
            return None
        self.position = min(self.position + int(self.replay_step_s * self.speed), len(self.soft) - 1)
        if self.position >= len(self.soft) - 1:
            self.position = self.window_length_s  # loop the replay
        return self.current_tick()

    def current_tick(self) -> Dict:
        pos = self.position
        w0 = pos - self.window_length_s
        win_soft = self.soft[w0:pos]
        win_hard = self.hard[w0:pos]
        # plot points: past 60 min at 10s cadence
        p0 = max(pos - PLOT_POINTS * PLOT_CADENCE_S, 0)
        sl = slice(p0, pos, PLOT_CADENCE_S)
        points = [
            {
                'timestamp': self.timestamps[i].isoformat(),
                'soft_flux': float(self.soft[i]),
                'hard_flux': float(self.hard[i]),
                'flare_class': str(self.flare_class[i]),
            }
            for i in range(p0, pos, PLOT_CADENCE_S)
        ]
        # real feature extraction (same pipeline as training)
        feats = extract_features_for_window(
            win_soft, win_hard, dt=1.0,
            rolling_windows_s=self.rolling_windows_s, layers=self.layers,
        )
        feature_names = list(feats.index)
        features = feats.to_numpy(dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        # raw windows at DL cadence for the deep path
        d = DL_WINDOW_SAMPLES
        soft_dl = win_soft[-d * PLOT_CADENCE_S::PLOT_CADENCE_S]
        hard_dl = win_hard[-d * PLOT_CADENCE_S::PLOT_CADENCE_S]
        return {
            'window_timestamp_utc': self.timestamps[pos - 1].isoformat(),
            'feature_names': feature_names,
            'features': features.tolist(),
            'soft_window': soft_dl.tolist(),
            'hard_window': hard_dl.tolist(),
            'points': points,
        }
