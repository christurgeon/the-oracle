"""Pure functions for offline Gaussian EMOS fitting.

Used by `scripts/fit_emos.py`. Hermetic — does not import from production
strategy/execution code.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)

# Optimizer bounds — see spec §"Optimizer bounds".
# c lower bound is 1e-6 (not 0) to prevent sqrt(0) NaN in sigma_fit.
_BOUNDS = [
    (None, None),  # a: unbounded
    (0.3, 1.5),  # b: tightened from literature [0.7, 1.3] with slack
    (1e-6, None),  # c: strictly positive
    (0.0, None),  # d: nonneg
]
_INIT = np.array([0.0, 1.0, 1.0, 1.0])

# Fit-time σ/bracket-width gate.
#
# σ_emos = √(c + d · ens_std²). When σ_emos ≫ bracket_width the Gaussian
# predictive is essentially uniform across any one bracket and calibration is
# structurally meaningless at the trading geometry. This failure mode caused
# LAX/high's 2026-05-18 rollback (live ROI = −39.6% on n≈11 trades): the fit
# was clean by every prior gate (BSS positive, only `unstable_d` warning —
# which was overridden) but σ_emos(p99) / 1°F ≈ 5.76.
#
# `incompatible_sigma` is intentionally non-overridable in promotion logic.
# The runtime gate (`weather_evaluation.py` extrapolation guard) is a separate
# decision — see the audit's "follow-ups deferred" section.
DEFAULT_BRACKET_WIDTH = 2.0  # NWS-rounded settle window: [floor − 0.5, cap + 0.5)
DEFAULT_INCOMPATIBLE_SIGMA_THRESHOLD = 2.0

BOOTSTRAP_RESAMPLE_ROW = "row"
BOOTSTRAP_RESAMPLE_DATE_CLUSTER = "date_cluster"

# With 2 dates, half of all date-resamples contain a single date, so the CI is a
# coarse mixture. Cluster-mode fits and gates refuse cells below this count.
MIN_CLUSTER_DATES = 10


def sigma_emos_at(*, c: float, d: float, ens_std: float) -> float:
    """Predictive σ of the Gaussian-EMOS predictive distribution at one ens_std.

    σ_emos = √(c + d · ens_std²), matching `_emos_mu_sigma` with the same
    σ²-floor (1e-12) used inside the fitter.

    Negative `c` or `d` are user errors (the optimizer's bounds preclude
    them); raise rather than silently squaring through the floor.
    """
    if c < 0 or d < 0:
        raise ValueError("c and d must be non-negative")
    return float(math.sqrt(max(c + d * ens_std * ens_std, 1e-12)))


def sigma_to_bracket_ratio(
    *, c: float, d: float, ens_std: float, bracket_width: float
) -> float:
    """Dimensionless `σ_emos(ens_std) / bracket_width`.

    >2 is the structural-incompatibility band on 1°F brackets — see the
    `DEFAULT_INCOMPATIBLE_SIGMA_THRESHOLD` constant docstring above.
    """
    if bracket_width <= 0:
        raise ValueError("bracket_width must be positive")
    return sigma_emos_at(c=c, d=d, ens_std=ens_std) / bracket_width


def crps_gaussian(
    obs: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma), obs) = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    where z = (obs - mu) / sigma. Lower is better. Same units as obs.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive")

    obs = np.asarray(obs, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    z = (obs - mu) / sigma
    return np.asarray(
        sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - _INV_SQRT_PI),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class EmosFit:
    """Fitted Gaussian-EMOS coefficients for one (city, var) cell."""

    a: float
    b: float
    c: float
    d: float


def _emos_mu_sigma(
    params: np.ndarray, ens_mean: np.ndarray, ens_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = params
    mu = a + b * ens_mean
    sigma = np.sqrt(np.maximum(c + d * ens_std**2, 1e-12))
    return np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)


def _neg_mean_crps(
    params: np.ndarray, ens_mean: np.ndarray, ens_std: np.ndarray, obs: np.ndarray
) -> float:
    mu, sigma = _emos_mu_sigma(params, ens_mean, ens_std)
    return float(np.mean(crps_gaussian(obs, mu, sigma)))


def fit_cell(ens_mean: np.ndarray, ens_std: np.ndarray, obs: np.ndarray) -> EmosFit:
    """Fit Gaussian EMOS (a, b, c, d) by minimizing mean CRPS via L-BFGS-B."""
    ens_mean = np.asarray(ens_mean, dtype=np.float64)
    ens_std = np.asarray(ens_std, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if not (
        np.all(np.isfinite(ens_mean)) and np.all(np.isfinite(ens_std)) and np.all(np.isfinite(obs))
    ):
        raise ValueError("ens_mean, ens_std, obs must all be finite")
    if np.any(ens_std <= 0):
        raise ValueError("ens_std must be strictly positive")
    if not (len(ens_mean) == len(ens_std) == len(obs)):
        raise ValueError("ens_mean, ens_std, obs must have equal length")

    result = minimize(
        _neg_mean_crps,
        _INIT,
        args=(ens_mean, ens_std, obs),
        method="L-BFGS-B",
        bounds=_BOUNDS,
    )
    a, b, c, d = result.x
    return EmosFit(a=float(a), b=float(b), c=float(c), d=float(d))


# CV split fractions of unique forecast_dates (sorted ascending).
# Forward chaining: train [0, q_i), validate [q_i, q_{i+1}).
_CV_QUANTILES = (0.6, 0.68, 0.76, 0.84, 0.92, 1.0)


def cv_fold_iter(
    forecast_dates: np.ndarray, n_folds: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) for 5-fold expanding-window time-series CV.

    Splits are made on unique forecast_dates (preserving co-membership of rows
    that share a date — e.g., rows for different leads on the same day end up in
    the same fold). Default fractions: train [0, 0.6), validate [0.6, 0.68); then
    [0, 0.68) / [0.68, 0.76); etc., yielding 5 folds.
    """
    if n_folds != 5:
        raise NotImplementedError("only n_folds=5 is supported in v0")
    forecast_dates = np.asarray(forecast_dates)
    unique_dates = np.unique(forecast_dates)
    n_unique = len(unique_dates)
    if n_unique < n_folds + 1:
        raise ValueError(
            f"need at least {n_folds + 1} unique forecast_dates for {n_folds}-fold "
            f"CV, got {n_unique}"
        )
    cuts = [int(np.floor(q * n_unique)) for q in _CV_QUANTILES]
    for i in range(n_folds):
        train_dates = set(unique_dates[: cuts[i]].tolist())
        val_dates = set(unique_dates[cuts[i] : cuts[i + 1]].tolist())
        train_mask = np.array([d in train_dates for d in forecast_dates])
        val_mask = np.array([d in val_dates for d in forecast_dates])
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


def date_cluster_rows(dates: Sequence[Any] | np.ndarray) -> list[np.ndarray]:
    """Row-index arrays, one per unique date, for cluster resampling.

    Raises ValueError on null dates or fewer than 2 unique dates.
    """
    arr = np.asarray(dates)
    if pd.isna(arr).any():
        raise ValueError("dates must not contain nulls")
    _, inverse = np.unique(arr, return_inverse=True)
    n_dates = int(inverse.max()) + 1 if len(arr) else 0
    if n_dates < 2:
        raise ValueError(f"cluster bootstrap needs at least 2 unique dates, got {n_dates}")
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=n_dates))[:-1]
    return np.split(order, bounds)


def bootstrap_cell(
    ens_mean: np.ndarray,
    ens_std: np.ndarray,
    obs: np.ndarray,
    *,
    n_iter: int = 400,
    seed: int,
    bracket_width: float = DEFAULT_BRACKET_WIDTH,
    incompatible_sigma_threshold: float = DEFAULT_INCOMPATIBLE_SIGMA_THRESHOLD,
    dates: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, object]:
    """Bootstrap (a, b, c, d) and return 2.5/50/97.5 percentiles + CI warnings.

    Returns dict shaped:
        {
            "a":  {"p2.5", "p50", "p97.5"},
            "b":  {...},
            "c":  {...},
            "d":  {...},
            "ci_warnings": list[str],
            "sigma_emos_at_p50":  float,  # σ_emos at the median bootstrap (c,d) and p50 ens_std
            "sigma_emos_at_p99":  float,  # σ_emos at the median bootstrap (c,d) and p99 ens_std
            "sigma_to_bracket_ratio_p99": float,  # ratio at p99 ens_std vs bracket_width
            "effective_n_dates": int,  # only when `dates` is given
        }

    Parameters
    ----------
    bracket_width
        Trading bracket width in the same units as `obs` (°F for KXHIGH/KXLOWT).
        Default 2.0 is the NWS-rounded settle window for Kalshi's 1°F daily
        weather brackets: [floor − 0.5, cap + 0.5). Intraday strategies with
        wider settle windows (e.g., 5°F or 10°F) should pass their own value.
    incompatible_sigma_threshold
        Multiplier above which `sigma_to_bracket_ratio_p99` triggers the
        `incompatible_sigma` warning. Default 2.0. The warning is intentionally
        non-overridable in `scripts/compare_emos_coefficients.py:_stage_b_decision`.
    dates
        Per-row cluster label. When given, each iteration resamples whole dates
        with replacement; row-level resampling understates CI widths because
        rows share settled observations. See `date_cluster_rows`.
    """
    rng = np.random.default_rng(seed)
    n = len(ens_mean)
    date_rows: list[np.ndarray] | None = None
    if dates is not None:
        if len(dates) != n:
            raise ValueError("dates, ens_mean, ens_std, obs must have equal length")
        date_rows = date_cluster_rows(dates)
    samples = np.zeros((n_iter, 4), dtype=np.float64)
    for i in range(n_iter):
        if date_rows is None:
            idx = rng.integers(0, n, size=n)
        else:
            drawn = rng.integers(0, len(date_rows), size=len(date_rows))
            idx = np.concatenate([date_rows[j] for j in drawn])
        # Note: fit_cell is module-level so tests can monkey-patch it.
        fit = fit_cell(ens_mean[idx], ens_std[idx], obs[idx])
        samples[i] = (fit.a, fit.b, fit.c, fit.d)

    a_pct = {
        "p2.5": float(np.percentile(samples[:, 0], 2.5)),
        "p50": float(np.percentile(samples[:, 0], 50)),
        "p97.5": float(np.percentile(samples[:, 0], 97.5)),
    }
    b_pct = {
        "p2.5": float(np.percentile(samples[:, 1], 2.5)),
        "p50": float(np.percentile(samples[:, 1], 50)),
        "p97.5": float(np.percentile(samples[:, 1], 97.5)),
    }
    c_pct = {
        "p2.5": float(np.percentile(samples[:, 2], 2.5)),
        "p50": float(np.percentile(samples[:, 2], 50)),
        "p97.5": float(np.percentile(samples[:, 2], 97.5)),
    }
    d_pct = {
        "p2.5": float(np.percentile(samples[:, 3], 2.5)),
        "p50": float(np.percentile(samples[:, 3], 50)),
        "p97.5": float(np.percentile(samples[:, 3], 97.5)),
    }

    # CI-width / boundary warnings.
    warnings: list[str] = []
    b_width = b_pct["p97.5"] - b_pct["p2.5"]
    if b_width > 0.4:
        warnings.append("unstable_b")
    d_width = d_pct["p97.5"] - d_pct["p2.5"]
    if d_width > 4.0:
        warnings.append("unstable_d")
    if d_pct["p50"] <= 1e-6:
        warnings.append("d_at_zero")

    # σ/bracket-width incompatibility gate. Uses the median bootstrap
    # estimates of (c, d) — same scalars persisted to coefficients.json as
    # the EMOS variance shape — evaluated at the p50 and p99 of the actual
    # `ens_std` training distribution. This catches structural fits where
    # σ_emos(p99) ≫ bracket_width even when (a, b, c, d) are individually
    # well-fit and BSS is positive (the LAX/high 2026-05-18 failure mode).
    ens_std_p50 = float(np.percentile(ens_std, 50))
    ens_std_p99 = float(np.percentile(ens_std, 99))
    c_med = c_pct["p50"]
    d_med = d_pct["p50"]
    sigma_p50 = sigma_emos_at(c=c_med, d=d_med, ens_std=ens_std_p50)
    sigma_p99 = sigma_emos_at(c=c_med, d=d_med, ens_std=ens_std_p99)
    # Asymmetry vs `sigma_to_bracket_ratio` (~line 69, which raises): we
    # short-circuit to +inf here so a misconfigured `bracket_width <= 0`
    # surfaces as the `incompatible_sigma` warning (ratio > threshold) and
    # the cell is flagged at the refit gate, rather than crashing the whole
    # batch fit. The pure helper raises because it has no warning channel
    # and callers are expected to be tests or one-off scripts.
    ratio_p99 = sigma_p99 / bracket_width if bracket_width > 0 else float("inf")
    if ratio_p99 > incompatible_sigma_threshold:
        warnings.append("incompatible_sigma")

    out: dict[str, object] = {
        "a": a_pct,
        "b": b_pct,
        "c": c_pct,
        "d": d_pct,
        "ci_warnings": warnings,
        "sigma_emos_at_p50": sigma_p50,
        "sigma_emos_at_p99": sigma_p99,
        "sigma_to_bracket_ratio_p99": ratio_p99,
    }
    if date_rows is not None:
        out["effective_n_dates"] = len(date_rows)
    return out


_DEFAULT_BRACKET_OFFSETS = (-5.0, -2.5, 0.0, 2.5, 5.0)

# Default bracket half-width for the synthetic Brier diagnostic.
#
# 0.5°F → 1°F-wide bracket = KXHIGH / KXLOWT trading geometry. The earlier
# 2.5° (= 5°F-wide) default flattered cells with large `d`: σ_emos ≫ 1°F
# looks calibrated on 5°F brackets but is structurally meaningless at the
# bracket width Kalshi actually trades. The LAX/high rollback
# (live ROI = −39.6% with σ_emos(p99) ≈ 5.76°F) is the canonical example.
#
# Intraday or other strategies with wider bracket geometry MUST pass an
# explicit `half_width` (e.g., 1.0 for 2°F, 2.5 for 5°F) to
# `synthetic_bracket_brier`. The parameter is back-compatible.
_DEFAULT_HALF_WIDTH = 0.5


def synthetic_bracket_brier(
    ens_mean: np.ndarray,
    ens_std: np.ndarray,
    obs: np.ndarray,
    fit: EmosFit,
    *,
    half_width: float = _DEFAULT_HALF_WIDTH,
    offsets: tuple[float, ...] = _DEFAULT_BRACKET_OFFSETS,
) -> tuple[float, float]:
    """Mean Brier score over a synthetic ±half_width bracket grid.

    For each row, evaluates `len(offsets)` brackets centered at
    ens_mean + offset, each ±half_width wide. Outcome is 1 if obs is in the
    bracket else 0. Returns (brier_emos, brier_raw_gaussian) where:

      - brier_emos uses sigma_fit = sqrt(c + d * ens_std**2), mu_fit = a + b*ens_mean
      - brier_raw_gaussian uses (mu=ens_mean, sigma=ens_std) directly

    See spec §"Synthetic-bracket Brier" — this is a fitter sanity check, not a
    production replay.
    """
    ens_mean = np.asarray(ens_mean, dtype=np.float64)
    ens_std = np.asarray(ens_std, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)

    mu_fit = fit.a + fit.b * ens_mean
    sigma_fit = np.sqrt(np.maximum(fit.c + fit.d * ens_std**2, 1e-12))

    brier_emos_sum = 0.0
    brier_raw_sum = 0.0
    for offset in offsets:
        lo = ens_mean + offset - half_width
        hi = ens_mean + offset + half_width
        outcome = ((obs >= lo) & (obs <= hi)).astype(np.float64)

        p_emos = norm.cdf((hi - mu_fit) / sigma_fit) - norm.cdf((lo - mu_fit) / sigma_fit)
        p_raw = norm.cdf((hi - ens_mean) / ens_std) - norm.cdf((lo - ens_mean) / ens_std)

        brier_emos_sum += float(np.mean((p_emos - outcome) ** 2))
        brier_raw_sum += float(np.mean((p_raw - outcome) ** 2))

    n_offsets = len(offsets)
    return brier_emos_sum / n_offsets, brier_raw_sum / n_offsets


def skip_rule(
    n: int, *, min_n: int = 30, low_n_threshold: int = 60
) -> tuple[bool, bool, str | None]:
    """Decide skip / low_n flag for a cell. Returns (should_skip, low_n, reason).

    n < min_n            -> (True,  False, "n=<n> below floor")
    min_n <= n < low_n_t -> (False, True,  None)
    n >= low_n_threshold -> (False, False, None)
    """
    if n < min_n:
        return True, False, f"n={n} below floor"
    if n < low_n_threshold:
        return False, True, None
    return False, False, None


def _oos_crps_a1(
    df: pd.DataFrame,
) -> tuple[float, list[float]]:
    """5-fold CV mean CRPS using A1 (pooled across leads)."""
    forecast_dates = df["forecast_date"].to_numpy()
    fold_crps: list[float] = []
    for train_idx, val_idx in cv_fold_iter(forecast_dates):
        fit = fit_cell(
            df["ens_mean"].to_numpy()[train_idx],
            df["ens_std"].to_numpy()[train_idx],
            df["obs"].to_numpy()[train_idx],
        )
        mu, sigma = _emos_mu_sigma(
            np.array([fit.a, fit.b, fit.c, fit.d]),
            df["ens_mean"].to_numpy()[val_idx],
            df["ens_std"].to_numpy()[val_idx],
        )
        fold_crps.append(float(np.mean(crps_gaussian(df["obs"].to_numpy()[val_idx], mu, sigma))))
    return float(np.mean(fold_crps)), fold_crps


def _oos_crps_a2(df: pd.DataFrame) -> float:
    """5-fold CV mean CRPS using A2 (per-lead). Diagnostic only.

    Returns NaN if any lead has too few unique forecast_dates for CV.
    """
    leads = sorted(df["lead"].unique())
    per_lead_means: list[float] = []
    for lead in leads:
        sub = df[df["lead"] == lead].reset_index(drop=True)
        try:
            per_fold = []
            for train_idx, val_idx in cv_fold_iter(sub["forecast_date"].to_numpy()):
                fit = fit_cell(
                    sub["ens_mean"].to_numpy()[train_idx],
                    sub["ens_std"].to_numpy()[train_idx],
                    sub["obs"].to_numpy()[train_idx],
                )
                mu, sigma = _emos_mu_sigma(
                    np.array([fit.a, fit.b, fit.c, fit.d]),
                    sub["ens_mean"].to_numpy()[val_idx],
                    sub["ens_std"].to_numpy()[val_idx],
                )
                per_fold.append(
                    float(np.mean(crps_gaussian(sub["obs"].to_numpy()[val_idx], mu, sigma)))
                )
            if per_fold:
                per_lead_means.append(float(np.mean(per_fold)))
        except ValueError:
            return float("nan")
    if not per_lead_means:
        return float("nan")
    return float(np.mean(per_lead_means))


def fit_one_cell(
    df: pd.DataFrame,
    *,
    seed: int,
    bootstrap_iters: int = 400,
    bracket_width: float = DEFAULT_BRACKET_WIDTH,
    incompatible_sigma_threshold: float = DEFAULT_INCOMPATIBLE_SIGMA_THRESHOLD,
    cluster_by_date: bool = False,
    cell: str = "cell",
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit one (city, var) cell pooled across leads.

    Returns (coefficients_entry, diagnostics_entry) — the values that get
    stored under cells["<city>/<var>"] in coefficients.json and
    coefficients.diagnostics.json respectively. The diagnostics entry
    additionally carries `sigma_emos_at_p50`, `sigma_emos_at_p99`, and
    `sigma_to_bracket_ratio_p99` (computed inside `bootstrap_cell`); the
    `incompatible_sigma` warning, if any, is in `ci_warnings`.

    `df` must contain columns: forecast_date, lead, ens_mean, ens_std, obs.
    Caller is responsible for filtering to settled rows.

    `bracket_width` defaults to 2.0 (NWS-rounded settle window for KXHIGH/KXLOWT
    1°F brackets). Intraday strategies with wider settle windows must pass their
    own width to avoid spurious `incompatible_sigma` warnings on wide-bracket cells.

    `cluster_by_date=True` resamples whole `forecast_date`s in the bootstrap
    (see `bootstrap_cell`) and adds `effective_n_dates` to `diag["bootstrap"]`.
    Cells with fewer than `MIN_CLUSTER_DATES` unique dates are skipped; null
    dates raise ValueError naming `cell`.
    """
    n = len(df)
    n_per_lead = {int(lead): int(c) for lead, c in df["lead"].value_counts().items()}

    should_skip, low_n, reason = skip_rule(n)
    if cluster_by_date:
        forecast_dates = df["forecast_date"]
        if forecast_dates.isna().any():
            raise ValueError(f"{cell}: forecast_date must not contain nulls for cluster bootstrap")
        n_dates = int(forecast_dates.nunique())
        if not should_skip and n_dates < MIN_CLUSTER_DATES:
            should_skip = True
            reason = (
                f"{n_dates} unique forecast_dates below cluster-bootstrap "
                f"floor of {MIN_CLUSTER_DATES}"
            )
    if should_skip:
        return (
            {"skipped": True, "reason": reason},
            {"n_train": n, "n_per_lead": n_per_lead, "skipped": True},
        )

    ens_mean = df["ens_mean"].to_numpy()
    ens_std = df["ens_std"].to_numpy()
    obs = df["obs"].to_numpy()

    fit = fit_cell(ens_mean, ens_std, obs)
    in_sample_crps = float(
        np.mean(
            crps_gaussian(
                obs,
                *_emos_mu_sigma(np.array([fit.a, fit.b, fit.c, fit.d]), ens_mean, ens_std),
            )
        )
    )

    oos_crps_a1, fold_crps = _oos_crps_a1(df)
    oos_crps_a2 = _oos_crps_a2(df)
    brier_emos, brier_raw = synthetic_bracket_brier(ens_mean, ens_std, obs, fit)
    bootstrap = bootstrap_cell(
        ens_mean,
        ens_std,
        obs,
        n_iter=bootstrap_iters,
        seed=seed,
        bracket_width=bracket_width,
        incompatible_sigma_threshold=incompatible_sigma_threshold,
        dates=df["forecast_date"].to_numpy() if cluster_by_date else None,
    )
    ci_warnings = cast(list[str], bootstrap.pop("ci_warnings"))
    # Lift σ diagnostics from the bootstrap envelope to top-level diag for the
    # external coefficients.diagnostics.json schema.
    sigma_at_p50 = cast(float, bootstrap.pop("sigma_emos_at_p50"))
    sigma_at_p99 = cast(float, bootstrap.pop("sigma_emos_at_p99"))
    sigma_ratio_p99 = cast(float, bootstrap.pop("sigma_to_bracket_ratio_p99"))

    leads_pooled = sorted(int(lead) for lead in df["lead"].unique())

    coef: dict[str, object] = {
        "a": fit.a,
        "b": fit.b,
        "c": fit.c,
        "d": fit.d,
        "n_train": n,
        "leads_pooled": leads_pooled,
        "low_n": low_n,
        "ci_warnings": ci_warnings,
    }
    diag: dict[str, object] = {
        "n_train": n,
        "n_per_lead": n_per_lead,
        "in_sample_crps": in_sample_crps,
        "oos_crps_a1": oos_crps_a1,
        "oos_crps_a2": oos_crps_a2,
        "oos_brier_emos": brier_emos,
        "oos_brier_raw": brier_raw,
        # Geometry the Brier diagnostic was scored against, persisted so
        # future tooling can refuse a cross-vintage compare (pre-correction
        # diagnostics used 2.5°F half-width; current default is 0.5°F).
        # Distinct from `bracket_width` (drives the σ-gate), even though
        # they align by Kalshi convention today.
        "synthetic_brier_half_width": _DEFAULT_HALF_WIDTH,
        "fold_crps": fold_crps,
        "bootstrap": bootstrap,
        "ci_warnings": ci_warnings,
        "sigma_emos_at_p50": sigma_at_p50,
        "sigma_emos_at_p99": sigma_at_p99,
        "sigma_to_bracket_ratio_p99": sigma_ratio_p99,
        "bracket_width": bracket_width,
        "incompatible_sigma_threshold": incompatible_sigma_threshold,
    }
    return coef, diag
