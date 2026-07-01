import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SyntheticFlareGenerator:
    FLARE_PEAK_RANGES = {
        'B': (1e-7, 1e-6),
        'C': (1e-6, 1e-5),
        'M': (1e-5, 1e-4),
        'X': (1e-4, 1e-3),
    }
    FLARE_ORDER = {'B': 1, 'C': 2, 'M': 3, 'X': 4, 'None': 0}

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        sd = self.config['synthetic_data']
        self.quiet_baseline_soft = sd['quiet_baseline_soft']
        self.quiet_baseline_hard = sd['quiet_baseline_hard']
        self.noise_std_soft = sd['noise_std_soft']
        self.noise_std_hard = sd['noise_std_hard']
        self.dropout_prob = sd['dropout_prob']
        self.total_days = sd['total_days']
        self.seed = sd['seed']
        self.flare_rates = sd['flare_rates']

    def _pink_noise(self, n_samples: int, std: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        white = rng.normal(0, std, n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-10
        fft /= np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=n_samples)
        return pink * (std / np.std(pink))

    def generate_flare_profile(
        self,
        flare_class: str,
        duration_minutes: float = 30.0,
        rise_time_ratio: float = 0.3,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_samples = int(duration_minutes * 60)
        t = np.linspace(0, duration_minutes * 60, n_samples)
        peak_idx = int(n_samples * rise_time_ratio)
        peak_time = t[peak_idx]
        low, high = self.FLARE_PEAK_RANGES[flare_class]
        peak_soft = 10 ** rng.uniform(np.log10(low), np.log10(high))
        hard_peak_ratio = rng.uniform(0.3, 0.8)
        peak_hard = peak_soft * hard_peak_ratio
        precursor_start = int(n_samples * max(0, rise_time_ratio - 0.15))
        soft = np.ones(n_samples) * self.quiet_baseline_soft
        hard = np.ones(n_samples) * self.quiet_baseline_hard
        soft_rise = peak_soft - self.quiet_baseline_soft
        for i in range(n_samples):
            if i < precursor_start:
                soft[i] = self.quiet_baseline_soft + soft_rise * 0.02 * (i / max(precursor_start, 1))
            elif i <= peak_idx:
                frac = (i - precursor_start) / max(peak_idx - precursor_start, 1)
                smooth = frac ** 2 * (3 - 2 * frac)
                soft[i] = self.quiet_baseline_soft + soft_rise * smooth
                hard_frac = max(0, (i - peak_idx + 60) / 60) if i >= peak_idx - 60 else 0
                hard_frac = min(1, hard_frac)
                hard[i] = self.quiet_baseline_hard + (peak_hard - self.quiet_baseline_hard) * hard_frac
            else:
                decay = (i - peak_idx) / max(n_samples - peak_idx, 1)
                soft[i] = self.quiet_baseline_soft + soft_rise * np.exp(-decay * 3)
                hard[i] = self.quiet_baseline_hard + (peak_hard - self.quiet_baseline_hard) * np.exp(-decay * 6)
        return soft, hard

    def generate_timeseries(
        self,
        duration_days: float,
        flare_events: List[Dict],
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_samples = int(duration_days * 86400)
        total_seconds = duration_days * 86400
        soft_flux = np.ones(n_samples) * self.quiet_baseline_soft
        hard_flux = np.ones(n_samples) * self.quiet_baseline_hard
        flare_class_arr = np.full(n_samples, 'None', dtype=object)
        soft_quality = np.zeros(n_samples, dtype=np.int8)
        hard_quality = np.zeros(n_samples, dtype=np.int8)
        pink_noise_seed = (seed or 0) + 1000
        soft_flux += self._pink_noise(n_samples, self.noise_std_soft, pink_noise_seed)
        hard_flux += self._pink_noise(n_samples, self.noise_std_hard, pink_noise_seed + 1)
        for event in flare_events:
            start_s = event['start_time']
            if isinstance(start_s, pd.Timestamp):
                start_s = int((start_s - pd.Timestamp("2026-01-01")).total_seconds())
            start_s = int(start_s)
            duration_min = event.get('duration_minutes', 30)
            fc = event['flare_class']
            n_flare = int(duration_min * 60)
            if start_s + n_flare >= n_samples:
                n_flare = n_samples - start_s
            if n_flare <= 0:
                continue
            fseed = (seed or 0) + start_s
            soft_prof, hard_prof = self.generate_flare_profile(fc, duration_min, seed=fseed)
            min_len = min(n_flare, len(soft_prof))
            soft_flux[start_s:start_s + min_len] = soft_prof[:min_len]
            hard_flux[start_s:start_s + min_len] = hard_prof[:min_len]
            flare_class_arr[start_s:start_s + min_len] = fc
        dropout_mask = rng.random(n_samples) < self.dropout_prob
        if dropout_mask.any():
            soft_flux[dropout_mask] = np.nan
            hard_flux[dropout_mask] = np.nan
            soft_quality[dropout_mask] = 1
            hard_quality[dropout_mask] = 1
        glitch_mask = rng.random(n_samples) < 0.001
        soft_quality[glitch_mask] = 2
        hard_quality[glitch_mask] = 2
        timestamps = pd.date_range(
            start="2026-01-01",
            periods=n_samples,
            freq='s'
        )[:n_samples]
        return pd.DataFrame({
            'timestamp_utc': timestamps,
            'soft_flux': soft_flux,
            'hard_flux': hard_flux,
            'soft_quality_flag': soft_quality,
            'hard_quality_flag': hard_quality,
            'flare_class': flare_class_arr,
        })

    def generate_dataset(self, config_overrides: Optional[Dict] = None) -> pd.DataFrame:
        cfg = self.config['synthetic_data']
        rng = np.random.default_rng(self.seed)
        total_seconds = self.total_days * 86400
        flare_events = []
        for fc, rate in cfg['flare_rates'].items():
            for _ in range(rate):
                start_s = int(rng.integers(3600, total_seconds - 7200))
                flare_events.append({
                    'flare_class': fc,
                    'start_time': start_s,
                    'duration_minutes': float(rng.uniform(15, 60)),
                })
        for event in flare_events:
            event['start_time'] = int(event['start_time'])
        flare_events.sort(key=lambda e: e['start_time'])
        min_gap = 3600
        filtered = []
        for e in flare_events:
            if not filtered or e['start_time'] - filtered[-1]['start_time'] >= min_gap:
                filtered.append(e)
        df = self.generate_timeseries(self.total_days, filtered, seed=self.seed)
        output_dir = Path(cfg['raw_dir'] if 'raw_dir' in cfg else self.config['data']['raw_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / 'training_data.parquet', index=False)
        return df
