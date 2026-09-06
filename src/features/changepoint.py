import numpy as np
import pandas as pd


def cusum_alarm(x: np.ndarray, k_sigma: float = 0.5, h_sigma: float = 4.0) -> dict:
    """Two-sided CUSUM control chart on standardized series."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-30:
        return {'cusum_alarmed': 0, 'cusum_stat_max': 0.0}
    z = (x - mu) / sigma
    cp = np.maximum.accumulate(np.maximum(np.cumsum(z - k_sigma), 0))
    cn = np.maximum.accumulate(np.maximum(np.cumsum(-z - k_sigma), 0))
    stat = float(max(cp.max(), cn.max()))
    return {'cusum_alarmed': int(stat >= h_sigma), 'cusum_stat_max': stat}


def ruptures_breakpoints(x: np.ndarray, pen: float = None) -> list:
    """PELT breakpoint detection; returns breakpoint indices (end-exclusive)."""
    import ruptures as rpt
    x = np.nan_to_num(np.asarray(x, dtype=np.float64)).reshape(-1, 1)
    if len(x) < 10 or np.std(x) < 1e-30:
        return []
    if pen is None:
        pen = 3.0 * np.log(len(x))
    try:
        algo = rpt.Pelt(model='rbf').fit(x)
        bkps = algo.predict(pen=pen)
        return [int(b) for b in bkps[:-1]]
    except Exception:
        return []


def _bocpd_runlengths(x: np.ndarray, hazard: float = 0.005, max_run: int = 200) -> dict:
    """Compact Bayesian online changepoint detection (Normal-Gamma prior),
    vectorized over the run-length distribution. Returns summary of the
    changepoint posterior. Input should be standardized."""
    from scipy import stats
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    T = len(x)
    mu0 = 0.0
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = max(np.var(x) if T > 1 else 1.0, 1e-12)
    R = np.zeros(1)
    growth_probs_all = np.zeros(T)
    runlength_sum = 0.0
    for t in range(T):
        if len(R) == 0:
            R = np.array([0.0])
            continue
        r = np.arange(len(R))
        kappa = kappa0 * (1.0) ** r + r  # kappa grows with run length
        alpha = alpha0 + r
        beta = beta0 + 0.5 * r * np.var(x[: t + 1]) if t > 0 else np.full(1, beta0)
        pred = stats.t.pdf(
            x[t],
            df=np.maximum(2 * alpha, 1e-9),
            loc=mu0,
            scale=np.sqrt(beta * (kappa + 1) / (alpha * kappa)),
        )
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        joint = pred * R
        cp_prob = float(joint.sum())
        if cp_prob <= 0:
            cp_prob = 1e-12
        growth = joint / cp_prob
        R_new = np.concatenate(([cp_prob], hazard * growth))
        R_new[1:] += (1 - hazard) * growth
        if len(R_new) > max_run:
            R_new[max_run - 1:] = R_new[max_run - 1:].sum()
            R_new = R_new[:max_run]
        R_new /= R_new.sum()
        R = R_new
        growth_probs_all[t] = min(cp_prob, 1.0)
        runlength_sum += float((np.arange(len(R)) * R).sum())
    return {
        'bocpd_max_prob': float(growth_probs_all.max()) if T else 0.0,
        'bocpd_mean_runlength': runlength_sum / max(T, 1),
    }


def extract_changepoint_features(
    soft_flux: np.ndarray,
    hard_flux: np.ndarray,
    dt: float = 1.0,
    bocpd_decimation: int = 10,
) -> pd.DataFrame:
    """L5 features per channel: CUSUM alarm, ruptures PELT breakpoints,
    BOCPD posterior summary. BOCPD runs on a decimated series for tractability
    (see MVP_PLAN.md decision log)."""
    features = {}
    for prefix, arr in [('soft', soft_flux), ('hard', hard_flux)]:
        x = np.nan_to_num(np.asarray(arr, dtype=np.float64))
        features[f'{prefix}_cusum_alarmed'] = cusum_alarm(x)['cusum_alarmed']
        features[f'{prefix}_cusum_stat_max'] = cusum_alarm(x)['cusum_stat_max']
        bkps = ruptures_breakpoints(x)
        features[f'{prefix}_n_breakpoints'] = len(bkps)
        if len(bkps) > 0:
            features[f'{prefix}_time_since_last_break_s'] = (len(x) - bkps[-1]) * dt
        else:
            features[f'{prefix}_time_since_last_break_s'] = float(len(x)) * dt
        xd = x[::bocpd_decimation]
        xd = (xd - xd.mean()) / (xd.std() + 1e-30)
        try:
            boc = _bocpd_runlengths(xd)
            features[f'{prefix}_bocpd_max_prob'] = boc['bocpd_max_prob']
            features[f'{prefix}_bocpd_mean_runlength_s'] = boc['bocpd_mean_runlength'] * dt * bocpd_decimation
        except Exception:
            features[f'{prefix}_bocpd_max_prob'] = 0.0
            features[f'{prefix}_bocpd_mean_runlength_s'] = float(len(xd)) * dt * bocpd_decimation
    return pd.DataFrame([features])
