"""Illiquid real estate return model: unsmoothing, state-space filter, diagnostics, bootstrap.

Appraisal-based real estate returns are smoothed, which understates volatility.
This module recovers the latent "true" return and produces IID invariants that
can feed scenario generation.

Contents
- geltner_unsmooth / fgw_unsmooth: reverse-filter baselines (Geltner 1993;
  Fisher-Geltner-Webb 1994).
- LatentReturnModel / kalman_smooth / moment_fit: linear-Gaussian state-space
  model with AR(1) momentum, observed through smoothed noisy appraisal series.
- acf / ljung_box / arch_lm: residual diagnostics (invariants must be IID).
- block_bootstrap_with_parameter_uncertainty: block bootstrap of residual rows.
- simulate_smoothed_observations: synthetic known-truth world for validation.

Dependencies: numpy, scipy only. Returns are quarterly log returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def geltner_unsmooth(observed: np.ndarray, smoothing: float) -> np.ndarray:
    """Geltner reverse filter: r_t = (o_t - c o_{t-1}) / (1 - c), c in [0, 1).

    Drops the first observation (no prior value).
    """
    o = np.asarray(observed, dtype=float).ravel()
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    return (o[1:] - smoothing * o[:-1]) / (1.0 - smoothing)


def fgw_unsmooth(observed: np.ndarray) -> tuple[np.ndarray, float]:
    """Fisher-Geltner-Webb: choose c so the unsmoothed series has ~zero lag-1 ACF.

    Returns (unsmoothed, c). Can over-unsmooth if observed autocorrelation is
    partly genuine momentum, so report volatility across a range of c.
    """
    o = np.asarray(observed, dtype=float).ravel()
    best_c, best_acf = 0.0, np.inf
    for c in np.linspace(0.0, 0.99, 200):
        r = geltner_unsmooth(o, c)
        acf1 = np.corrcoef(r[1:], r[:-1])[0, 1]
        if abs(acf1) < abs(best_acf):
            best_c, best_acf = c, acf1
    return geltner_unsmooth(o, best_c), float(best_c)


@dataclass(frozen=True)
class LatentReturnModel:
    """Latent r_t = mu + rho (r_{t-1} - mu) + eta_t, eta ~ N(0, q).

    Observer i: o_i,t = (1 - c_i) r_t + c_i r_{t-1} + eps_i, eps ~ N(0, r_i).
    """

    mu: float
    rho: float
    q: float
    smoothing: tuple[float, ...]
    obs_var: tuple[float, ...]

    def __post_init__(self) -> None:
        if not -1 < self.rho < 1 or self.q <= 0:
            raise ValueError("need |rho| < 1 and q > 0")
        if len(self.smoothing) != len(self.obs_var):
            raise ValueError("one smoothing weight and obs variance per observer")


def kalman_smooth(observations: np.ndarray, model: LatentReturnModel) -> dict:
    """Kalman filter + RTS smoother.

    observations: (T, n_observers); NaN marks a missing observer-quarter.
    Returns dict with latent_smoothed (T,), latent_var (T,), invariant (T-1,).
    """
    obs = np.asarray(observations, dtype=float)
    if obs.ndim == 1:
        obs = obs[:, None]
    if obs.shape[1] != len(model.smoothing):
        raise ValueError("one observation column per observer")
    T = obs.shape[0]
    F = np.array([[model.rho, 0.0], [1.0, 0.0]])
    const = np.array([model.mu * (1 - model.rho), 0.0])
    Q = np.diag([model.q, 1e-12])
    c = np.asarray(model.smoothing)
    H_full = np.column_stack([1 - c, c])
    R = np.asarray(model.obs_var)

    x = np.array([model.mu, model.mu])
    P = np.eye(2) * model.q / (1 - model.rho**2)
    xs, Ps, x_pred, P_pred = [], [], [], []
    for t in range(T):
        if t > 0:
            x = const + F @ x
            P = F @ P @ F.T + Q
        x_pred.append(x.copy())
        P_pred.append(P.copy())
        ok = ~np.isnan(obs[t])
        if ok.any():
            H, y = H_full[ok], obs[t, ok]
            S = H @ P @ H.T + np.diag(R[ok])
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (y - H @ x)
            P = (np.eye(2) - K @ H) @ P
        xs.append(x.copy())
        Ps.append(P.copy())

    sm_x, sm_P = [None] * T, [None] * T
    sm_x[-1], sm_P[-1] = xs[-1], Ps[-1]
    for t in range(T - 2, -1, -1):
        J = Ps[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        sm_x[t] = xs[t] + J @ (sm_x[t + 1] - x_pred[t + 1])
        sm_P[t] = Ps[t] + J @ (sm_P[t + 1] - P_pred[t + 1]) @ J.T

    latent = np.array([s[0] for s in sm_x])
    invariant = latent[1:] - (model.mu + model.rho * (latent[:-1] - model.mu))
    return {
        "latent_smoothed": latent,
        "latent_var": np.array([p[0, 0] for p in sm_P]),
        "invariant": invariant,
    }


def moment_fit(observations: np.ndarray) -> LatentReturnModel:
    """Crude moment fit for a single observer (starting point, not EM)."""
    o = np.asarray(observations, dtype=float)
    if o.ndim == 2:
        if o.shape[1] != 1:
            raise ValueError("moment_fit handles one observer")
        o = o[:, 0]
    series = o[~np.isnan(o)]
    acf1 = float(np.corrcoef(series[1:], series[:-1])[0, 1])
    c0 = float(np.clip(acf1 / (1 + acf1), 0.0, 0.95))
    var_obs = float(series.var(ddof=1))
    q = max(var_obs * (1 - c0) ** 2, 1e-8)
    unsmoothed, c_fgw = fgw_unsmooth(series)
    rho = float(np.clip(np.corrcoef(unsmoothed[1:], unsmoothed[:-1])[0, 1], -0.9, 0.9))
    return LatentReturnModel(
        mu=float(series.mean()), rho=rho, q=q,
        smoothing=(c_fgw,), obs_var=(var_obs * 0.05,),
    )


def acf(series: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations for lags 0..nlags."""
    x = np.asarray(series, dtype=float).ravel()
    x = x - x.mean()
    denom = x @ x
    return np.array([1.0] + [(x[k:] @ x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box(series: np.ndarray, nlags: int) -> dict:
    """Ljung-Box Q test for autocorrelation up to nlags."""
    x = np.asarray(series, dtype=float).ravel()
    n = len(x)
    r = acf(x, nlags)[1:]
    q_stat = n * (n + 2) * np.sum(r**2 / (n - np.arange(1, nlags + 1)))
    return {"Q": float(q_stat), "df": nlags, "p_value": float(stats.chi2.sf(q_stat, nlags))}


def arch_lm(series: np.ndarray, nlags: int = 5) -> dict:
    """Engle ARCH-LM test: regress squared residuals on their lags."""
    x = np.asarray(series, dtype=float).ravel()
    e2 = (x - x.mean()) ** 2
    n = len(e2) - nlags
    y = e2[nlags:]
    X = np.column_stack([np.ones(n)] + [e2[nlags - k - 1: len(e2) - k - 1] for k in range(nlags)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    lm = n * r2
    return {"LM": float(lm), "df": nlags, "p_value": float(stats.chi2.sf(lm, nlags))}


def block_bootstrap_with_parameter_uncertainty(
    residuals: np.ndarray,
    horizon_periods: int,
    n_scenarios: int,
    block_length: int,
    seed: int,
    param_draws: tuple[np.ndarray, ...] = (),
    refit_fn=None,
) -> np.ndarray:
    """Stationary block bootstrap of residual rows; returns (n_scenarios, d).

    Whole rows are resampled so cross-dependence is preserved. Each scenario is
    the sum of horizon_periods resampled rows (log returns add). If param_draws
    and refit_fn are given, scenarios are split into one batch per draw and the
    residual pool is re-derived with refit_fn(*params) for each batch.
    """
    x = np.atleast_2d(np.asarray(residuals, dtype=float))
    if x.shape[0] == 1 and np.asarray(residuals).ndim == 1:
        x = x.T
    T, d = x.shape
    if not 1 <= block_length <= T:
        raise ValueError("bad block length")
    rng = np.random.default_rng(seed)
    n_batches = len(param_draws[0]) if param_draws else 1
    per_batch = n_scenarios // n_batches
    out = []
    for b in range(n_batches):
        pool = x
        if param_draws and refit_fn is not None:
            pool = np.asarray(refit_fn(*[pd[b] for pd in param_draws]), dtype=float)
            if pool.ndim == 1:
                pool = pool[:, None]
        starts = rng.integers(0, len(pool), size=(per_batch, horizon_periods))
        lengths = rng.geometric(1.0 / block_length, size=(per_batch, horizon_periods))
        idx = (starts + np.cumsum(lengths, axis=1) - lengths) % len(pool)
        out.append(pool[idx.reshape(-1)].reshape(per_batch, horizon_periods, d).sum(axis=1))
    scen = np.concatenate(out, axis=0)
    if scen.shape[0] < n_scenarios:
        scen = np.concatenate([scen, scen[: n_scenarios - scen.shape[0]]])
    return scen


def simulate_smoothed_observations(
    n_periods: int, mu: float, rho: float, q: float,
    smoothing: tuple[float, ...], obs_var: tuple[float, ...], seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic world with known truth. Returns (observations (T, k), truth (T,))."""
    rng = np.random.default_rng(seed)
    innov = rng.normal(0.0, np.sqrt(q), n_periods + 1)
    r = np.empty(n_periods + 1)
    r[0] = mu
    for t in range(1, n_periods + 1):
        r[t] = mu + rho * (r[t - 1] - mu) + innov[t]
    obs = np.column_stack([
        (1 - c) * r[1:] + c * r[:-1] + rng.normal(0.0, np.sqrt(rv), n_periods)
        for c, rv in zip(smoothing, obs_var)
    ])
    return obs, r[1:]
