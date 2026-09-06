import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLASS_MAP = {'None': 0, 'B': 1, 'C': 2, 'M': 3, 'X': 4}
START_EPOCH = pd.Timestamp("2026-01-01")


class SyntheticFlareGenerator:
    FLARE_PEAK_RANGES = {
        'B': (1e-7, 1e-6),
        'C': (1e-6, 1e-5),
        'M': (1e-5, 1e-4),
        'X': (1e-4, 1e-3),
    }
    FLARE_ORDER = {'B': 1, 'C': 2, 'M': 3, 'X': 4, 'None': 0}
    RISE_TIME_RATIO = 0.3

    def __init__(self, config_path: str = "config/config.yaml", overrides: Optional[Dict] = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        if overrides:
            for section, values in overrides.items():
                if isinstance(values, dict) and isinstance(self.config.get(section), dict):
                    self.config[section] = {**self.config[section], **values}
                else:
                    self.config[section] = values
        sd = self.config['synthetic_data']
        self.quiet_baseline_soft = sd['quiet_baseline_soft']
        self.quiet_baseline_hard = sd['quiet_baseline_hard']
        self.noise_std_soft = sd['noise_std_soft']
        self.noise_std_hard = sd['noise_std_hard']
        self.dropout_prob = sd['dropout_prob']
        self.total_days = float(sd['total_days'])
        self.seed = sd['seed']
        self.chunk_days = int(sd.get('chunk_days', 15))
        self.min_gap_s = int(sd.get('min_gap_s', 3600))
        # Per-day expected event counts (Poisson). Legacy integer flare_rates
        # (total counts) are converted to per-day rates for compatibility.
        if 'events_per_day' in sd:
            self.flare_rates_per_day = {k: float(v) for k, v in sd['events_per_day'].items()}
        else:
            self.flare_rates_per_day = {
                k: float(v) / self.total_days for k, v in sd['flare_rates'].items()
            }

    # ------------------------------------------------------------------ noise
    def _pink_noise(self, n_samples: int, std: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        white = rng.normal(0, std, n_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = 1e-10
        fft /= np.sqrt(freqs)
        fft[0] = 0  # pink noise has no DC component; keeps baseline unbiased
        pink = np.fft.irfft(fft, n=n_samples)
        return pink * (std / np.std(pink))

    # ------------------------------------------------------------------ flares
    def generate_flare_profile(
        self,
        flare_class: str,
        duration_minutes: float = 30.0,
        rise_time_ratio: float = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rise_time_ratio is None:
            rise_time_ratio = self.RISE_TIME_RATIO
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

    def _assign_flare_events(self) -> List[Dict]:
        """Draw per-day Poisson event counts per class, enforce min separation."""
        rng = np.random.default_rng(self.seed)
        total_seconds = self.total_days * 86400
        flare_events = []
        for fc, rate_per_day in self.flare_rates_per_day.items():
            n_events = int(rng.poisson(rate_per_day * self.total_days))
            for _ in range(n_events):
                start_s = int(rng.integers(3600, total_seconds - 7200))
                flare_events.append({
                    'flare_class': fc,
                    'start_time': start_s,
                    'duration_minutes': float(rng.uniform(15, 60)),
                })
        flare_events.sort(key=lambda e: e['start_time'])
        min_gap = self.min_gap_s
        filtered = []
        for e in flare_events:
            if not filtered or e['start_time'] - filtered[-1]['start_time'] >= min_gap:
                filtered.append(e)
        return filtered

    # --------------------------------------------------------------- timeseries
    def generate_timeseries(
        self,
        duration_days: float,
        flare_events: List[Dict],
        seed: Optional[int] = None,
        catalogue_out: Optional[List[Dict]] = None,
        block_start_utc: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Generate one contiguous block of telemetry.

        If catalogue_out is a list, per-event catalogue rows (with absolute UTC
        peak times) are appended to it. Events truncated at the block edge are
        not catalogued.
        """
        if block_start_utc is None:
            block_start_utc = START_EPOCH
        rng = np.random.default_rng(seed)
        n_samples = int(duration_days * 86400)
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
                start_s = int((start_s - START_EPOCH).total_seconds())
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
            if catalogue_out is not None and min_len == len(soft_prof):
                peak_rel = int(np.argmax(soft_prof))
                catalogue_out.append({
                    'flare_class': fc,
                    'start_utc': block_start_utc + pd.Timedelta(seconds=start_s),
                    'peak_utc': block_start_utc + pd.Timedelta(seconds=start_s + peak_rel),
                    'end_utc': block_start_utc + pd.Timedelta(seconds=start_s + min_len - 1),
                    'peak_soft_flux': float(soft_prof[peak_rel]),
                    'duration_min': float(duration_min),
                })
        dropout_mask = rng.random(n_samples) < self.dropout_prob
        if dropout_mask.any():
            soft_flux[dropout_mask] = np.nan
            hard_flux[dropout_mask] = np.nan
            soft_quality[dropout_mask] = 1
            hard_quality[dropout_mask] = 1
        glitch_mask = rng.random(n_samples) < 0.001
        soft_quality[glitch_mask] = 2
        hard_quality[glitch_mask] = 2
        timestamps = pd.date_range(start=block_start_utc, periods=n_samples, freq='s')
        return pd.DataFrame({
            'timestamp_utc': timestamps,
            'soft_flux': soft_flux,
            'hard_flux': hard_flux,
            'soft_quality_flag': soft_quality,
            'hard_quality_flag': hard_quality,
            'flare_class': flare_class_arr,
        })

    # ----------------------------------------------------------------- dataset
    def generate_dataset(self, config_overrides: Optional[Dict] = None) -> pd.DataFrame:
        """Generate the full dataset. Large datasets are written chunk-by-chunk
        into a single parquet file (one row group per chunk) to bound peak
        memory; small ones (<= chunk_days) are also returned in memory.
        Always writes the flare catalogue to data/external/flare_catalogue.parquet.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        raw_dir = Path(self.config['data']['raw_dir'])
        external_dir = Path(self.config['data']['external_dir'])
        raw_dir.mkdir(parents=True, exist_ok=True)
        external_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / 'training_data.parquet'
        catalogue_path = external_dir / 'flare_catalogue.parquet'

        flare_events = self._assign_flare_events()

        catalogue: List[Dict] = []
        if self.total_days <= self.chunk_days:
            df = self.generate_timeseries(
                self.total_days, flare_events, seed=self.seed,
                catalogue_out=catalogue, block_start_utc=START_EPOCH,
            )
            df.to_parquet(raw_path, index=False)
        else:
            writer = None
            for chunk_idx in range(0, int(self.total_days), self.chunk_days):
                block_start_s = chunk_idx * 86400
                block_end_s = min(block_start_s + self.chunk_days * 86400, self.total_days * 86400)
                block_days = (block_end_s - block_start_s) / 86400
                chunk_events = []
                for e in flare_events:
                    if block_start_s <= e['start_time'] < block_end_s:
                        ce = dict(e)
                        ce['start_time'] = e['start_time'] - block_start_s
                        chunk_events.append(ce)
                chunk_seed = self.seed + (chunk_idx // self.chunk_days) * 7919
                chunk_df = self.generate_timeseries(
                    block_days, chunk_events, seed=chunk_seed, catalogue_out=catalogue,
                    block_start_utc=START_EPOCH + pd.Timedelta(seconds=block_start_s),
                )
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(raw_path, table.schema)
                writer.write_table(table)
            if writer is not None:
                writer.close()
        catalogue_df = pd.DataFrame(catalogue).sort_values('peak_utc') if catalogue else pd.DataFrame(
            columns=['flare_class', 'start_utc', 'peak_utc', 'end_utc', 'peak_soft_flux', 'duration_min']
        )
        catalogue_df.to_parquet(catalogue_path, index=False)

        if self.total_days <= self.chunk_days:
            return df
        return pd.read_parquet(raw_path, columns=['flare_class'])

    def get_flare_catalogue(self) -> pd.DataFrame:
        path = Path(self.config['data']['external_dir']) / 'flare_catalogue.parquet'
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()
