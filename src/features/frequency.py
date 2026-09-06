import numpy as np
import pandas as pd
from scipy import signal

try:
    import pywt
    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _HAS_PYWT = False

WAVELET = 'morl'
TARGET_PERIODS_S = np.geomspace(8.0, 256.0, 8)
FFT_BANDS = [(0.001, 0.01), (0.01, 0.1), (0.1, 1.0)]


def _constant_defaults(prefix: str) -> dict:
    return {
        f'{prefix}_fft_band1_power_frac': 0.0,
        f'{prefix}_fft_band2_power_frac': 0.0,
        f'{prefix}_fft_band3_power_frac': 0.0,
        f'{prefix}_spectral_entropy': 0.0,
        f'{prefix}_wavelet_energy_mean': 0.0,
        f'{prefix}_wavelet_energy_max': 0.0,
        f'{prefix}_wavelet_dominant_scale_s': 0.0,
        f'{prefix}_wavelet_total_energy': 0.0,
    }


def compute_band_powers(x: np.ndarray, dt: float) -> dict:
    """FFT band power fractions + spectral entropy via Welch PSD."""
    fs = 1.0 / dt
    nperseg = min(len(x), 256)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    total = float(np.trapezoid(psd, freqs)) if hasattr(np, 'trapezoid') else float(np.trapz(psd, freqs))
    if total <= 0:
        return {f'fft_band{i}_power_frac': 0.0 for i in (1, 2, 3)} | {'spectral_entropy': 0.0}
    nyq = fs / 2.0
    out = {}
    for i, (lo, hi) in enumerate(FFT_BANDS, start=1):
        lo_c, hi_c = min(lo, nyq), min(hi, nyq)
        if hi_c <= lo_c:
            out[f'fft_band{i}_power_frac'] = 0.0
            continue
        mask = (freqs >= lo_c) & (freqs <= hi_c)
        band = float(np.trapezoid(psd[mask], freqs[mask])) if mask.sum() >= 2 else 0.0
        out[f'fft_band{i}_power_frac'] = band / total
    p = psd / psd.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum() / np.log(len(psd))) if len(p) > 0 else 0.0
    out['spectral_entropy'] = entropy
    return out


def compute_wavelet_energies(x: np.ndarray, dt: float) -> dict:
    """Morlet CWT energy across target periods (seconds)."""
    if not _HAS_PYWT:
        return {
            'wavelet_energy_mean': 0.0,
            'wavelet_energy_max': 0.0,
            'wavelet_dominant_scale_s': 0.0,
            'wavelet_total_energy': 0.0,
        }
    fs = 1.0 / dt
    scales = []
    for period_s in TARGET_PERIODS_S:
        freq = 1.0 / (period_s * fs)  # cycles per sample
        s = float(pywt.frequency2scale(WAVELET, freq))
        scales.append(max(s, 1.0))
    scales = np.array(sorted(set(scales)))
    try:
        coefs, _ = pywt.cwt(x, scales, WAVELET)
        energies = (np.abs(coefs) ** 2).sum(axis=1)
    except Exception:
        energies = np.zeros(len(scales))
    total = float(energies.sum())
    if total <= 0:
        return {
            'wavelet_energy_mean': 0.0,
            'wavelet_energy_max': 0.0,
            'wavelet_dominant_scale_s': 0.0,
            'wavelet_total_energy': 0.0,
        }
    dom_idx = int(np.argmax(energies))
    # scale -> pseudo-period in seconds
    dom_freq = pywt.scale2frequency(WAVELET, scales[dom_idx])
    dom_period_s = 1.0 / (dom_freq * fs) if dom_freq > 0 else 0.0
    return {
        'wavelet_energy_mean': float(energies.mean()),
        'wavelet_energy_max': float(energies.max()),
        'wavelet_dominant_scale_s': float(dom_period_s),
        'wavelet_total_energy': total,
    }


def extract_frequency_features(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    dt: float = 1.0,
) -> pd.DataFrame:
    features = {}
    for prefix, arr in [('soft', soft_flux), ('hard', hard_flux)]:
        x = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
        x = x - x.mean()
        if np.std(x) < 1e-30:
            features.update(_constant_defaults(prefix))
            continue
        features.update({f'{prefix}_{k}': v for k, v in compute_band_powers(x, dt).items()})
        features.update({f'{prefix}_{k}': v for k, v in compute_wavelet_energies(x, dt).items()})
    return pd.DataFrame([features])
