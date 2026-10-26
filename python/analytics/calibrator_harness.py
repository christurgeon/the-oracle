"""Per-cell calibrator evaluation harness.

Phase 1A research tool. Scores three candidates (identity, iso_v2_global,
emos) per (city, var) cell using 5-fold expanding-window time-series CV
with per-fold refits, paired block-bootstrap CIs, and an AND-gate verdict
rule. Outputs data/calibration/per_cell_winners.json.

The harness composes existing scaffolding:
- scripts._calibration_common.build_production_mirrored_pairs_with_provenance
- python.analytics.emos_fit.cv_fold_iter / fit_one_cell
- python.analytics.isotonic_fit.fit_isotonic_pairs
- python.data.emos.emos_probability
- python.data.calibration.IsotonicCalibrator (for in_sample_reference)

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from python.analytics.emos_fit import cv_fold_iter, fit_one_cell
from python.analytics.fees import order_fee
from python.analytics.isotonic_fit import fit_isotonic_pairs
from python.analytics.sizing import fractional_kelly
from python.common import Side
from python.data.bias_correction import BiasTable
from python.data.calibration import IsotonicCalibrator
from python.data.emos import (
    LONG_TO_SHORT_VAR,
    EmosCoefficients,
    EmosTable,
    emos_probability,
)
from python.data.weather import EnsembleForecast
from python.utils.atomic_write import atomic_write_text
from scripts._calibration_common import (
    build_forecast_snapshot_dates,
    build_production_mirrored_pairs_with_provenance,
    load_snapshot_data,
)

# Project imports needed by the replay-parity assertions live below the
# assertion call site; they import cleanly when the assertions pass.

logger = logging.getLogger(__name__)


_PARITY_CHECKED: bool = False


def _assert_replay_parity() -> None:
    """FR-16 module-load assertions. Called once at import time.

    Extracted to a function so all imports stay at the top of the module
    (no scatter of noqa: E402 below the assertion block). Raises
    AssertionError on any FR-16 violation; the CLI translates that to
    exit code 2 (FM-3).

    The ``_PARITY_CHECKED`` guard makes the assertion fire exactly once per
    process — direct callers (test fixtures, future tasks) can't accidentally
    re-run the 6 FR-16 module imports the assertion did.
    """
    global _PARITY_CHECKED
    if _PARITY_CHECKED:
        return
    _PARITY_CHECKED = True

    from python.analytics.emos_fit import DEFAULT_BRACKET_WIDTH
    from python.analytics.outcomes import derive_outcome_yes
    from python.data.emos import emos_probability
    from python.data.weather import compute_blended_probability
    from python.strategies.weather_evaluation import DAILY_WEATHER_BRACKET_WIDTH_DEG_F
    from python.strategies.weather_parsing import _DIRECTION_TO_STRIKE_LABEL

    # FR-9 / FM-6: bracket-width parity with runtime + emos_fit.
    if (
        DAILY_WEATHER_BRACKET_WIDTH_DEG_F != DEFAULT_BRACKET_WIDTH
        or DAILY_WEATHER_BRACKET_WIDTH_DEG_F != 2.0
    ):
        raise AssertionError(
            f"FR-9/FM-6 bracket-width mismatch: "
            f"DAILY_WEATHER_BRACKET_WIDTH_DEG_F={DAILY_WEATHER_BRACKET_WIDTH_DEG_F}, "
            f"DEFAULT_BRACKET_WIDTH={DEFAULT_BRACKET_WIDTH}; expected both == 2.0. "
            f"Refusing to run — see harness FR-16 / FM-6."
        )

    # FR-16.2: NWS-window asymmetry preserved (callables present).
    assert callable(compute_blended_probability), (
        "FR-16.2: compute_blended_probability not callable"
    )
    assert callable(derive_outcome_yes), "FR-16.2: derive_outcome_yes not callable"

    # FR-16.3: EMOS swap path uses emos_probability.
    assert callable(emos_probability), "FR-16.3: emos_probability not callable"

    # FR-16.5: canonical strike labels.
    assert set(_DIRECTION_TO_STRIKE_LABEL.values()) == {"greater", "less", "between"}, (
        "FR-16.5: _DIRECTION_TO_STRIKE_LABEL values do not match canonical strike labels"
    )


_assert_replay_parity()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARNESS_VERSION: str = "0.1.0"
SCHEMA_VERSION: int = 2
LOGSKILL_EPS: float = 1e-3
BOOTSTRAP_ITER: int = 400
BOOTSTRAP_BLOCK_MEAN_LENGTH: int = 3  # Politis-Romano mean block length (forecast_dates)
WINNER_GATE_LOGSKILL_LB_MIN: float = 0.01  # raised from 0.0; F-1 multiple-comparison fix
WINNER_GATE_FOLD_CONSISTENCY_MIN: int = 4
WINNER_GATE_N_CLIPPED_PCT_MAX: float = 0.05
EMOS_GATE_LOGSKILL_LB_VS_ISO_MIN: float = -0.02
EMOS_GATE_BSS_LB_VS_ISO_MIN: float = -0.05
TIEBREAKER_LOGSKILL_LB_BAND: float = 0.005
FR11_2_N_VAL_TOTAL_MIN: int = 75  # raised from 30 per B.4 / M-1 (CV quantile span)
FLAGS_BELOW_POWER_FLOOR_N: int = 200  # warning threshold for under-power regime
# Width of the "borderline winner" band above WINNER_GATE_LOGSKILL_LB_MIN.
GATE_MARGIN_THIN_WIDTH: float = 0.01

# Stage-1 simulated-ROI trading filters. These mirror the production trading
# gate in python/config.yaml (weather.block_raw_prob_range,
# weather.min_calibrated_prob, weather.kelly_fraction); keep them in sync if
# config.yaml changes. Used by both simulate_stage1_roi defaults and the
# run_harness payload provenance block.
_SIM_BLOCK_RAW_PROB_RANGE: tuple[float, float] = (0.446, 0.883)
_SIM_MIN_CALIBRATED_PROB: float = 0.110
_SIM_KELLY_FRACTION: float = 0.075


def _clip(p: float, eps: float = LOGSKILL_EPS) -> float:
    """Two-sided clamp of a probability into ``[eps, 1 - eps]``.

    Shared by every candidate ``score()`` and the log-loss metrics so the
    clip boundary is defined in exactly one place.
    """
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureRow:
    """One scoring observation. Fields are the union of what any candidate needs.

    EMOS candidates consume gfs_ens_mean / gfs_ens_std (GFS-only post-bias,
    post-spread-inflation; matches python.strategies.weather_evaluation).
    Isotonic / identity candidates consume blended_raw_prob (output of
    compute_blended_probability — bias-corrected, spread-inflated, blended).
    """

    # Identity / locator
    city: str
    var: str  # "temperature_2m_max" | "temperature_2m_min"
    forecast_date: str  # ISO date "YYYY-MM-DD"; the date being forecast
    snapshot_date: str  # ISO date "YYYY-MM-DD"; collection date of the snapshot
    # snapshot_date <= forecast_date; lead = (forecast_date − snapshot_date).days
    lead: int  # 0 / 1 / 2; calendar days from snapshot to forecast date
    ticker: str

    # Market geometry
    strike_type: Literal["greater", "less", "between"]
    floor: float | None  # °F; None for one-sided brackets on the lower side
    cap: float | None  # °F; None for one-sided brackets on the upper side
    market_yes_price: float  # Kalshi YES price ∈ [0, 1]

    # Probabilities / ensemble stats
    blended_raw_prob: float  # compute_blended_probability output
    gfs_ens_mean: float  # GFS-only post-bias, post-inflation
    gfs_ens_std: float  # ddof=1 sample std
    n_members: int

    # Outcome (None on inference-only rows; populated on training/scoring rows)
    outcome: int | None

    # Open extension point for Phase 2 candidates (regime labels, options-IV, etc.)
    extras: dict[str, float] = field(default_factory=dict)


class FitState(Protocol):
    """Marker protocol. Each candidate defines its own FitState concrete type.

    The harness scoring loop does not introspect; dispatch is by candidate
    class + fit_semantics, not by FitState's structure.
    """


@dataclass(frozen=True)
class VetoResult:
    """Returned by CalibratorCandidate.validate."""

    ok: bool
    warnings: list[str]  # e.g., ["unstable_d"], ["incompatible_sigma", "unstable_b"]


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------


class CalibratorCandidate(ABC):
    """Phase 1A calibrator candidate interface.

    Subclasses MUST set `name` and `fit_semantics` as class attributes.
    `fit_semantics="per_fold_refit"` means fit() runs once per CV fold.
    `fit_semantics="load_artifact"` means load() runs once at run start;
    fit() is never called (LoadArtifactCandidate enforces this).
    `fit_semantics="compose"` is reserved for Phase 2 stackers/BMA.
    """

    name: str
    fit_semantics: Literal["per_fold_refit", "load_artifact", "compose"]

    @abstractmethod
    def fit(self, train_rows: list[FeatureRow]) -> FitState:
        """Fit the candidate on training rows. Returns an opaque fit-state object."""

    @abstractmethod
    def score(self, row: FeatureRow, fit_state: FitState) -> float:
        """Return calibrated P(YES) for the row, clipped to [LOGSKILL_EPS, 1 - LOGSKILL_EPS]."""

    @abstractmethod
    def validate(self, fit_state: FitState) -> VetoResult:
        """Return structural-veto status. Called after each fold's fit."""


class LoadArtifactCandidate(CalibratorCandidate):
    """Specialization for the in_sample_reference path.

    The harness scoring loop never calls fit() on these candidates — only
    load() then score(). fit() raises NotImplementedError defensively so a
    wiring bug surfaces loudly instead of silently re-fitting.
    """

    def fit(self, train_rows: list[FeatureRow]) -> FitState:
        raise NotImplementedError(
            "LoadArtifactCandidate uses load(), not fit(). "
            "Harness scoring loop is responsible for dispatching on fit_semantics."
        )

    @abstractmethod
    def load(self, artifact_path: Path) -> FitState:
        """Load the deployed artifact from disk. Called once at run start."""


# ---------------------------------------------------------------------------
# IdentityCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _IdentityFitState:
    """Empty fit state. Identity has no parameters."""


class IdentityCandidate(CalibratorCandidate):
    """No-op calibrator: score(row) = row.blended_raw_prob, clipped."""

    name = "identity"
    fit_semantics = "per_fold_refit"

    def fit(self, train_rows: list[FeatureRow]) -> FitState:
        return _IdentityFitState()

    def score(self, row: FeatureRow, fit_state: FitState) -> float:
        return _clip(row.blended_raw_prob)

    def validate(self, fit_state: FitState) -> VetoResult:
        return VetoResult(ok=True, warnings=[])


# ---------------------------------------------------------------------------
# IsoV2GlobalCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _IsoFitState:
    """Per-fold isotonic fit: breakpoints + values for linear interpolation."""

    breakpoints: list[float]
    values: list[float]


class IsoV2GlobalCandidate(CalibratorCandidate):
    """Isotonic candidate that REFITS PER FOLD on training rows' raw_prob.

    Despite the name 'iso_v2_global', this candidate is NOT the deployed
    isotonic_v2.json artifact. It is the same calibrator FAMILY refit on
    each fold's training partition. The deployed artifact is scored
    separately by IsoV2DeployedReference (in_sample_reference column).
    """

    name = "iso_v2_global"
    fit_semantics = "per_fold_refit"

    def fit(self, train_rows: list[FeatureRow]) -> FitState:
        if not train_rows:
            raise ValueError("IsoV2GlobalCandidate.fit: train_rows is empty")
        # Only include rows where outcome is observed.
        pairs = np.array(
            [
                (r.blended_raw_prob, float(r.outcome))
                for r in train_rows
                if r.outcome is not None
            ],
            dtype=np.float64,
        )
        if len(pairs) == 0:
            raise ValueError("IsoV2GlobalCandidate.fit: no rows with outcome present")
        breakpoints, values = fit_isotonic_pairs(pairs, anchor_endpoints=True)
        return _IsoFitState(breakpoints=breakpoints, values=values)

    def score(self, row: FeatureRow, fit_state: FitState) -> float:
        assert isinstance(fit_state, _IsoFitState)
        bp = fit_state.breakpoints
        vl = fit_state.values
        x = row.blended_raw_prob

        if x <= bp[0]:
            p = vl[0]
        elif x >= bp[-1]:
            p = vl[-1]
        else:
            lo, hi = 0, len(bp) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if bp[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            x0, x1 = bp[lo], bp[hi]
            y0, y1 = vl[lo], vl[hi]
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            p = y0 + t * (y1 - y0)

        return _clip(p)

    def validate(self, fit_state: FitState) -> VetoResult:
        # Isotonic never fires EMOS-style structural vetoes. n_clipped > 5%
        # is checked at the metric aggregation layer (Task 14), not here.
        return VetoResult(ok=True, warnings=[])


# ---------------------------------------------------------------------------
# EmosCandidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EmosFitState:
    """Per-fold EMOS fit: (a, b, c, d) coefficients + bootstrap diagnostics."""

    a: float
    b: float
    c: float
    d: float
    ci_warnings: list[str]
    fit_failure: bool = False


class EmosCandidate(CalibratorCandidate):
    """EMOS candidate that REFITS PER FOLD on training rows' (gfs_ens_mean, gfs_ens_std, obs).

    Scoring path reproduces the runtime swap in weather_evaluation.py:
    calls python.data.emos.emos_probability(coeffs, ens_mean, ens_std, ...) with a
    temporary EmosCoefficients built from refitted (a, b, c, d) — NOT through
    Calibrator.calibrate().

    Requires row.extras["obs_temp_f"] (actual observed temperature °F) on each
    training row; the corpus loader (Task 8) populates this. fit() raises
    ValueError if no rows carry obs_temp_f.
    """

    name = "emos"
    fit_semantics = "per_fold_refit"

    def fit(self, train_rows: list[FeatureRow]) -> FitState:
        if not train_rows:
            raise ValueError("EmosCandidate.fit: train_rows is empty")

        df = pd.DataFrame(
            [
                {
                    "forecast_date": r.forecast_date,
                    "lead": r.lead,
                    "ens_mean": r.gfs_ens_mean,
                    "ens_std": r.gfs_ens_std,
                    "obs": r.extras["obs_temp_f"],
                }
                for r in train_rows
                if r.outcome is not None and "obs_temp_f" in r.extras
            ]
        )
        if len(df) == 0:
            raise ValueError(
                "EmosCandidate.fit: no rows with outcome AND extras['obs_temp_f'] "
                "present. The corpus loader (load_feature_rows) is responsible "
                "for populating extras['obs_temp_f'] on every settled row."
            )

        try:
            coef, _diag = fit_one_cell(df, seed=42, bootstrap_iters=BOOTSTRAP_ITER)
        except Exception:
            # L-BFGS-B convergence failure or any other fit exception.
            return _EmosFitState(
                a=0.0, b=1.0, c=1.0, d=0.0,
                ci_warnings=["emos_fit_failure"], fit_failure=True,
            )

        if coef.get("skipped"):
            return _EmosFitState(
                a=0.0, b=1.0, c=1.0, d=0.0,
                ci_warnings=["emos_skip:" + str(coef.get("reason", "unknown"))],
                fit_failure=True,
            )

        return _EmosFitState(
            a=cast(float, coef["a"]),
            b=cast(float, coef["b"]),
            c=cast(float, coef["c"]),
            d=cast(float, coef["d"]),
            ci_warnings=cast(list[str], coef.get("ci_warnings", [])),
            fit_failure=False,
        )

    def score(self, row: FeatureRow, fit_state: FitState) -> float:
        assert isinstance(fit_state, _EmosFitState)
        if fit_state.fit_failure:
            # A failed-fit fold cannot score this row. The CV loop catches
            # this exception, marks the fold ineligible, and records null
            # in fold_logskill for this candidate (A2 fix per Task 16).
            raise RuntimeError("EmosCandidate.score called on failed-fit fold")

        # Build a temporary EmosCoefficients from refitted (a, b, c, d) so
        # we can delegate to emos_probability() — reproducing the runtime swap
        # in weather_evaluation.py bit-identically.
        coeffs = EmosCoefficients(
            a=fit_state.a,
            b=fit_state.b,
            c=fit_state.c,
            d=fit_state.d,
            n_train=0,       # sentinel — not used by emos_probability math
            low_n=False,     # sentinel
            ci_warnings=fit_state.ci_warnings,
            leads_pooled=[],  # sentinel
        )

        prob, _cdf_floor, _cdf_cap = emos_probability(
            coeffs,
            row.gfs_ens_mean,
            row.gfs_ens_std,
            strike_type=row.strike_type,
            floor=row.floor,
            cap=row.cap,
        )

        return _clip(prob)

    def validate(self, fit_state: FitState) -> VetoResult:
        assert isinstance(fit_state, _EmosFitState)
        if fit_state.fit_failure:
            return VetoResult(ok=False, warnings=["emos_fit_failure"])
        warnings = list(fit_state.ci_warnings)
        # ok=False iff σ-veto fired (non-overridable per emos_fit.py:40-44).
        # unstable_b / d_at_zero / unstable_d are aggregated at fold level in
        # compute_eligibility (Task 14): unstable_b / d_at_zero gated at ≥3/5;
        # unstable_d flag-only (matches data/emos/overrides.json convention).
        ok = "incompatible_sigma" not in warnings
        return VetoResult(ok=ok, warnings=warnings)


# ---------------------------------------------------------------------------
# IsoV2DeployedReference (in_sample_reference audit candidate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _IsoDeployedFitState:
    """Loaded IsotonicCalibrator state. fallback_reason is non-None on load failure."""

    calibrator: IsotonicCalibrator
    fallback_reason: str | None


class IsoV2DeployedReference(LoadArtifactCandidate):
    """Loads the deployed data/calibration/isotonic_v2.json artifact.

    Scores in-sample on the full corpus for the in_sample_reference column.
    Verdict path NEVER consults this candidate — see FR-13.
    """

    name = "iso_v2_deployed_reference"
    fit_semantics = "load_artifact"

    def load(self, artifact_path: Path) -> FitState:
        calibrator = IsotonicCalibrator(path=artifact_path)
        return _IsoDeployedFitState(
            calibrator=calibrator,
            fallback_reason=calibrator.fallback_reason,
        )

    def score(self, row: FeatureRow, fit_state: FitState) -> float:
        assert isinstance(fit_state, _IsoDeployedFitState)
        p = fit_state.calibrator.calibrate(
            row.blended_raw_prob, strike_type=row.strike_type
        )
        return _clip(p)

    def validate(self, fit_state: FitState) -> VetoResult:
        assert isinstance(fit_state, _IsoDeployedFitState)
        return VetoResult(
            ok=fit_state.fallback_reason is None,
            warnings=(
                [f"fallback:{fit_state.fallback_reason}"]
                if fit_state.fallback_reason
                else []
            ),
        )


# ---------------------------------------------------------------------------
# EmosDeployedReference (in_sample_reference audit candidate)
# ---------------------------------------------------------------------------

# Single source of truth for the long↔short var translation lives in
# python.data.emos.LONG_TO_SHORT_VAR (next to EmosTable._CANONICAL_VAR_LABELS).
# Aliased here under the prior local name for backward-readability of call sites.
_VAR_TO_SHORT = LONG_TO_SHORT_VAR


@dataclass(frozen=True)
class _EmosDeployedFitState:
    """Loaded EmosTable + fallback diagnostics."""

    table: EmosTable | None
    fallback_reason: str | None


class EmosDeployedReference(LoadArtifactCandidate):
    """Loads the deployed data/emos/coefficients.json via EmosTable.

    Scores in-sample on the full corpus for the in_sample_reference column.
    Per-cell coefficients are looked up by (city, var, lead).
    Cells with no entry produce nan (assembler renders as null in JSON output).
    Verdict path NEVER consults this candidate — see FR-13.
    """

    name = "emos_deployed_reference"
    fit_semantics = "load_artifact"

    def load(self, artifact_path: Path) -> FitState:
        try:
            table = EmosTable.load(artifact_path)
            if not table.is_loaded:
                return _EmosDeployedFitState(
                    table=None,
                    fallback_reason=f"EmosTable.load failed: {table.fallback_reason}",
                )
            return _EmosDeployedFitState(table=table, fallback_reason=None)
        except Exception as exc:
            return _EmosDeployedFitState(
                table=None,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

    def score(self, row: FeatureRow, fit_state: FitState) -> float:
        assert isinstance(fit_state, _EmosDeployedFitState)
        if fit_state.fallback_reason is not None or fit_state.table is None:
            return float("nan")

        # EmosTable requires canonical short var label ("high" / "low").
        var_short = _VAR_TO_SHORT.get(row.var, row.var)
        coeffs = fit_state.table.get(city=row.city, var=var_short, lead=row.lead)
        if coeffs is None:
            return float("nan")

        prob, _cdf_floor, _cdf_cap = emos_probability(
            coeffs,
            row.gfs_ens_mean,
            row.gfs_ens_std,
            strike_type=row.strike_type,
            floor=row.floor,
            cap=row.cap,
        )

        return _clip(prob)

    def validate(self, fit_state: FitState) -> VetoResult:
        assert isinstance(fit_state, _EmosDeployedFitState)
        if fit_state.fallback_reason:
            return VetoResult(ok=False, warnings=[f"fallback:{fit_state.fallback_reason}"])
        return VetoResult(ok=True, warnings=[])


# ---------------------------------------------------------------------------
# Corpus loader (Task 8)
# ---------------------------------------------------------------------------


def _lead_from_dates(snapshot_date: str, forecast_date: str) -> int:
    """Compute the lead (calendar days from snapshot to forecast date).

    Both arguments are ISO date strings "YYYY-MM-DD".

    The pre-B.5 implementation scanned the snapshot directory tree at score time
    and returned ``min(forecast_date − collection_date)`` across all snapshot dirs
    that contained weather forecast data.  Because daily 02:00 UTC snapshots
    always won the ``min()`` selection (same-day snapshot = lead 0), every row
    collapsed to lead=0.  Bootstrap blocks ``(forecast_date, lead)`` were thus
    single-lead-per-date; CIs were artificially tight.

    B.5 fix: ``snapshot_date`` is propagated through
    ``build_forecast_snapshot_dates`` → ``load_feature_rows`` → ``FeatureRow``;
    lead is the simple calendar difference.

    Returns:
        Non-negative integer days.

    Raises:
        AssertionError: if ``snapshot_date > forecast_date`` (structural
            invariant; should never happen in production data).
    """
    snap = date.fromisoformat(snapshot_date)
    fcst = date.fromisoformat(forecast_date)
    lead = (fcst - snap).days
    assert lead >= 0, (
        f"_lead_from_dates: negative lead "
        f"(snapshot={snapshot_date}, forecast={forecast_date}, lead={lead})"
    )
    return lead


def _compute_lead_for(row: FeatureRow) -> int:
    """Compute the lead for a ``FeatureRow`` from its snapshot/forecast dates.

    Thin delegate to :func:`_lead_from_dates`.
    """
    return _lead_from_dates(row.snapshot_date, row.forecast_date)


def _compute_gfs_stats(
    gfs_members: list[float],
    *,
    bias_table: BiasTable | None,
    daily_var: str,
    city: str,
    forecast_date: date,
    spread_inflation_overrides: dict[str, dict[str, float]] | None,
) -> tuple[float, float, int]:
    """Compute (post-bias, post-inflation) ens_mean, ens_std (ddof=1), n_members.

    Mirrors the GFS-only path in
    ``scripts._calibration_common.build_production_mirrored_pairs_with_provenance``
    so EMOS scoring sees the same distribution that production uses.
    """
    from python.data.bias_correction import apply_bias_correction
    from python.data.weather import get_spread_inflation_factor

    forecast = EnsembleForecast(
        forecast_date=forecast_date,
        city=city,
        temperatures=list(gfs_members),
        model="gfs_seamless",
    )

    if bias_table is not None and bias_table.loaded:
        forecast, _ = apply_bias_correction(
            forecast,
            bias_table,
            daily_var,
            shrinkage=0.3,
            max_correction_f=5.0,
        )

    factor = get_spread_inflation_factor(city, daily_var, spread_inflation_overrides)
    if factor != 1.0:
        forecast = forecast.inflated(factor)

    members = list(forecast.temperatures)
    n = len(members)
    mean = sum(members) / n
    # ddof=1 sample std
    variance = sum((m - mean) ** 2 for m in members) / (n - 1) if n > 1 else 0.0
    std = variance ** 0.5
    return mean, std, n


def load_feature_rows(
    snapshots_dir: Path,
    *,
    bias_table_path: Path | None,
    ensemble_weights: dict[str, float],
    spread_inflation_overrides: dict[str, dict[str, float]] | None = None,
) -> list[FeatureRow]:
    """Load a list of :class:`FeatureRow` from a snapshots directory.

    Composes :func:`~scripts._calibration_common.load_snapshot_data` and
    :func:`~scripts._calibration_common.build_production_mirrored_pairs_with_provenance`
    to inherit:

    - CHI pre-fix exclusion (FR-16.6)
    - NWS-window asymmetry and preliminary-settlement drop (FR-16.2)
    - Pipeline order (FR-16.4)

    Augments each pair with GFS-only ``ens_mean``/``ens_std`` (post-bias,
    post-spread-inflation, ddof=1) for EMOS candidates and populates
    ``extras["obs_temp_f"]`` (required by :class:`EmosCandidate`\\ .fit).
    """
    forecasts, settlements, markets = load_snapshot_data(snapshots_dir)

    # Build snapshot_date carrier for lead computation (B.5 fix).
    # Maps (city, forecast_date_str, daily_var, model) → snapshot_date_str.
    # Mirrors load_snapshot_data's sorted last-writer-wins traversal, so the
    # snapshot_date stored here matches the forecast entry actually used above.
    fc_snapshot_dates = build_forecast_snapshot_dates(snapshots_dir)

    # Resolve bias table once.
    bias_table: BiasTable | None = None
    if bias_table_path is not None and bias_table_path.exists():
        bias_table = BiasTable(bias_table_path)
        if not bias_table.loaded:
            bias_table = None

    # Build a ticker→market dict for O(1) lookup in the loop below.
    markets_by_ticker: dict[str, dict[str, Any]] = {
        m["ticker"]: m for m in markets if "ticker" in m
    }

    rows: list[FeatureRow] = []
    for (
        raw_prob,
        outcome,
        city,
        daily_var,
        strike_type,
        fc_date,
        _models_used,
        _bias_correction_applied,
        ticker,
    ) in build_production_mirrored_pairs_with_provenance(
        forecasts=forecasts,
        settlements=settlements,
        markets=markets,
        bias_table=bias_table,
        ensemble_weights=ensemble_weights,
        spread_inflation_overrides=spread_inflation_overrides,
    ):
        # Recover the market entry for floor/cap/yes_price.
        mkt = markets_by_ticker.get(ticker)
        if mkt is None:
            continue

        floor_s = mkt.get("floor_strike")
        cap_s = mkt.get("cap_strike")

        # best-effort yes price: prefer ask (taker), fall back to bid, then 0.5.
        yes_price_raw = mkt.get("yes_ask_dollars") or mkt.get("yes_bid_dollars")
        market_yes_price = float(yes_price_raw) if yes_price_raw is not None else 0.5

        # Recover settlement obs (settlements keyed by (city, observation_date)).
        st = settlements.get((city, str(fc_date)))
        if st is None:
            continue
        obs_field = "high_f" if daily_var == "temperature_2m_max" else "low_f"
        obs_temp = st.get(obs_field)
        if obs_temp is None:
            continue

        # GFS-only post-bias / post-inflation stats for EMOS.
        gfs_fc = forecasts.get((city, str(fc_date), daily_var, "gfs_seamless"))
        if gfs_fc is None:
            continue
        gfs_members = list(
            gfs_fc.get("temperatures") or gfs_fc.get("ensemble_members") or []
        )
        if len(gfs_members) < 25:
            continue

        gfs_mean, gfs_std, n_members = _compute_gfs_stats(
            gfs_members,
            bias_table=bias_table,
            daily_var=daily_var,
            city=city,
            forecast_date=fc_date,
            spread_inflation_overrides=spread_inflation_overrides,
        )

        # Resolve snapshot_date for this forecast (B.5: calendar-diff lead).
        # Fall back to forecast_date itself (lead=0) if the carrier is missing,
        # which can happen for synthetic test rows without a real snapshot tree.
        fc_date_str = str(fc_date)
        snapshot_date_str = fc_snapshot_dates.get(
            (city, fc_date_str, daily_var, "gfs_seamless"),
            fc_date_str,  # fallback: lead=0
        )

        lead = _lead_from_dates(snapshot_date_str, fc_date_str)

        rows.append(
            FeatureRow(
                city=city,
                var=daily_var,
                forecast_date=fc_date_str,
                snapshot_date=snapshot_date_str,
                lead=lead,
                ticker=ticker,
                strike_type=strike_type,  # type: ignore[arg-type]
                floor=float(floor_s) if floor_s is not None else None,
                cap=float(cap_s) if cap_s is not None else None,
                market_yes_price=market_yes_price,
                blended_raw_prob=raw_prob,
                gfs_ens_mean=gfs_mean,
                gfs_ens_std=gfs_std,
                n_members=n_members,
                outcome=int(outcome),
                extras={"obs_temp_f": float(obs_temp)},
            )
        )

    # Lead-blindness regression guard (B.5): for each cell, when that cell has
    # at least one settled FeatureRow where snapshot_date < forecast_date
    # (i.e., data was collected in advance AND the outcome is settled), any cell
    # with ≥30 rows should show more than one distinct lead value.  Fires if
    # build_forecast_snapshot_dates silently broke and all leads collapsed to 0
    # despite multi-day-ahead collection.
    # Does NOT fire for fixture corpora where every snapshot_date == forecast_date
    # (same-day collection is valid; leads are legitimately all 0 there).
    # Does NOT fire for off-list cities (e.g. PHIL) whose ahead-of-day
    # fc_snapshot_dates entries are only for unsettled future forecast_dates —
    # those cities produce no settled FeatureRows with non-zero leads because the
    # outcomes haven't been observed yet, which is correct behaviour.
    # Guard criterion: a cell "should have multi-lead rows" only when it already
    # has at least one row with snapshot_date < forecast_date (i.e., at least one
    # settled ahead-of-day row).  We infer this from the rows themselves rather
    # than from fc_snapshot_dates so that future-date carrier entries (not yet
    # settled) don't trigger a false positive.
    by_cell_leads: dict[tuple[str, str], list[int]] = defaultdict(list)
    cells_with_settled_ahead: set[tuple[str, str]] = set()
    for r in rows:
        by_cell_leads[(r.city, r.var)].append(r.lead)
        if r.snapshot_date < r.forecast_date:
            cells_with_settled_ahead.add((r.city, r.var))

    if cells_with_settled_ahead:
        for (lead_city, lead_var), cell_leads in by_cell_leads.items():
            if (lead_city, lead_var) not in cells_with_settled_ahead:
                # All settled rows for this cell were same-day; all-zero leads OK.
                continue
            if len(cell_leads) >= 30 and len(set(cell_leads)) == 1:
                raise AssertionError(
                    f"Lead-blindness regression: cell {lead_city}/{lead_var} has "
                    f"{len(cell_leads)} rows but only one distinct lead value "
                    f"({cell_leads[0]}) even though this cell has settled "
                    f"ahead-of-day FeatureRows. "
                    f"Check FeatureRow.snapshot_date propagation in "
                    f"build_forecast_snapshot_dates."
                )

    return rows


# ---------------------------------------------------------------------------
# CV scoring loop
# ---------------------------------------------------------------------------


def cv_score_cell(
    rows: list[FeatureRow],
    *,
    candidates: list[CalibratorCandidate],
) -> dict[str, Any]:
    """Per-cell 5-fold expanding-window CV scoring.

    For each fold:
      - Fit each per_fold_refit candidate on the train indices.
      - Run validate() on the fit state; mark per-fold ineligibility on σ-veto / failure.
      - Score val rows; record per-row predictions and per-fold mean (or null on failure).

    Returns:
        {
            "n_val_total": int,
            "n_unique_forecast_dates": int,
            "per_candidate": {
                "<name>": {
                    "val_rows": list[FeatureRow],              # flattened (legacy)
                    "val_predictions": list[float],            # parallel to val_rows
                    "fold_val_rows": list[list[FeatureRow] | None],   # 5 elts, per-fold
                    "fold_predictions": list[list[float] | None],     # 5 elts, // val_rows
                    "fold_warnings": list[list[str]],          # 5 elts, warnings/fold
                    "fold_eligible": list[bool],               # 5 elts, eligibility/fold
                }
            }
        }

    The ``fold_val_rows`` field (A2 fix) lets score_cell compute per-fold LogSkill
    by pairing this candidate's fold predictions against identity's fold
    predictions on the same fold's val rows. The legacy flattened ``val_rows`` /
    ``val_predictions`` are retained for backward-compat consumers.

    Raises:
        ValueError: when len(set(forecast_dates)) < 6 (FR-11.1 — caller treats as insufficient_n).
    """
    forecast_dates = np.array([r.forecast_date for r in rows])
    n_unique = len(set(forecast_dates.tolist()))
    if n_unique < 6:
        raise ValueError(
            f"cv_score_cell: only {n_unique} unique forecast_dates; "
            f"need ≥6 for 5-fold CV (caller should map to insufficient_n verdict)"
        )

    # Build per-candidate state.
    per_candidate: dict[str, dict[str, Any]] = {
        c.name: {
            "val_rows": [],
            "val_predictions": [],
            "fold_val_rows": [None, None, None, None, None],
            "fold_predictions": [None, None, None, None, None],
            "fold_warnings": [[], [], [], [], []],
            "fold_eligible": [False, False, False, False, False],
        }
        for c in candidates
    }

    fold_idx = 0
    for train_idx, val_idx in cv_fold_iter(forecast_dates):
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        for candidate in candidates:
            try:
                fit_state = candidate.fit(train_rows)
            except Exception as exc:
                per_candidate[candidate.name]["fold_warnings"][fold_idx].append(
                    f"fit_exception:{type(exc).__name__}"
                )
                per_candidate[candidate.name]["fold_eligible"][fold_idx] = False
                continue

            veto = candidate.validate(fit_state)
            per_candidate[candidate.name]["fold_warnings"][fold_idx].extend(veto.warnings)
            if not veto.ok:
                per_candidate[candidate.name]["fold_eligible"][fold_idx] = False
                continue

            # Score val rows.
            per_candidate[candidate.name]["fold_eligible"][fold_idx] = True
            fold_preds: list[float] = []
            fold_val_rows_for_this_fold: list[FeatureRow] = []
            for vr in val_rows:
                try:
                    p = candidate.score(vr, fit_state)
                except Exception as exc:
                    per_candidate[candidate.name]["fold_warnings"][fold_idx].append(
                        f"score_exception:{type(exc).__name__}"
                    )
                    p = float("nan")
                fold_preds.append(p)
                fold_val_rows_for_this_fold.append(vr)
                per_candidate[candidate.name]["val_rows"].append(vr)
                per_candidate[candidate.name]["val_predictions"].append(p)

            per_candidate[candidate.name]["fold_predictions"][fold_idx] = fold_preds
            per_candidate[candidate.name]["fold_val_rows"][fold_idx] = fold_val_rows_for_this_fold

        fold_idx += 1

    # Compute n_val_total from the candidate with the most val rows (all candidates
    # see the same val partitions; differences arise only when a candidate is
    # ineligible in a fold).
    n_val_total = (
        max(len(per_candidate[c.name]["val_rows"]) for c in candidates)
        if candidates
        else 0
    )

    return {
        "n_val_total": n_val_total,
        "n_unique_forecast_dates": n_unique,
        "per_candidate": per_candidate,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _log_loss_row(p: float, y: int) -> float:
    """Per-row log-loss with clipping."""
    p = _clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


def compute_logskill(
    candidate_predictions: list[float],
    identity_predictions: list[float],
    outcomes: list[int],
) -> float:
    """Per-cell LogSkill score relative to identity baseline.

    LogSkill = 1 - mean(LL_cand) / mean(LL_identity)
    Higher is better; positive = candidate beats identity on log-loss.
    Paired on identical val rows (caller guarantees parallel lists).
    """
    if not (len(candidate_predictions) == len(identity_predictions) == len(outcomes)):
        raise ValueError("compute_logskill: input lists must be equal length")
    if not candidate_predictions:
        raise ValueError("compute_logskill: empty inputs")

    n = len(outcomes)
    ll_cand_sum = sum(_log_loss_row(p, y) for p, y in zip(candidate_predictions, outcomes))
    ll_id_sum = sum(_log_loss_row(p, y) for p, y in zip(identity_predictions, outcomes))

    mean_cand = ll_cand_sum / n
    mean_id = ll_id_sum / n
    if mean_id == 0.0:
        return 0.0
    return 1.0 - mean_cand / mean_id


def count_clipped(predictions: list[float], eps: float = LOGSKILL_EPS) -> int:
    """Number of predictions that hit the clip boundary."""
    return sum(1 for p in predictions if p <= eps or p >= 1.0 - eps)


def compute_brier(predictions: list[float], outcomes: list[int]) -> float:
    if not predictions or len(predictions) != len(outcomes):
        raise ValueError("compute_brier: inputs must be equal length and non-empty")
    n = len(predictions)
    return sum((p - y) ** 2 for p, y in zip(predictions, outcomes)) / n


def compute_bss(
    candidate_predictions: list[float],
    identity_predictions: list[float],
    outcomes: list[int],
) -> float:
    brier_cand = compute_brier(candidate_predictions, outcomes)
    brier_id = compute_brier(identity_predictions, outcomes)
    if brier_id == 0.0:
        return 0.0
    return 1.0 - brier_cand / brier_id


# ---------------------------------------------------------------------------
# Simulated Stage-1 ROI (Task 12)
# ---------------------------------------------------------------------------


def simulate_stage1_roi(
    *,
    candidate_predictions: list[float],
    blended_raw_probs: list[float],
    market_yes_prices: list[float],
    outcomes: list[int],
    forecast_dates: list[str],
    leads: list[int],
    kelly_fraction: float = _SIM_KELLY_FRACTION,
    max_pct: float = 0.05,
    bankroll: float = 1000.0,
    block_raw_prob_range: tuple[float, float] = _SIM_BLOCK_RAW_PROB_RANGE,
    min_calibrated_prob: float = _SIM_MIN_CALIBRATED_PROB,
) -> dict[str, Any]:
    """Simulate Stage-1 P&L under production trading filters (FR-5.3, FR-17).

    Applies (in order) the production gating filters before each row's
    contribution is included in the ROI sum:

    1. Plateau-block on raw_prob: skip when
       ``block_raw_prob_range[0] <= raw_prob <= block_raw_prob_range[1]``.
    2. Calibrated-floor: skip when ``p_cand < min_calibrated_prob``.
    3. Kelly sizing: skip when :func:`fractional_kelly` returns
       ``Side.NONE`` or ``dollars <= 0``.

    Returns:
        {
            "roi_point": float | None,             # sum(pnl)/sum(wagered); None if insufficient
            "per_row_pnl": list[float],            # parallel to inputs (0.0 for filtered rows)
            "per_row_wagered": list[float],        # parallel to inputs (0.0 for filtered rows)
            "per_row_block_key": list[tuple],      # parallel to inputs (forecast_date, lead)
            "n_val_traded": int,
            "status": "ok" | "insufficient_data",
        }

    ``status == "insufficient_data"`` (and ``roi_point is None``) when
    ``n_val_traded < 30`` OR ``sum_wagered == 0.0`` — Task 13's
    block-bootstrap CI requires both a populated pnl/wagered stream and a
    minimum n to be statistically meaningful.

    The per_row_* lists are length-``n`` parallel to the inputs (filtered
    rows carry ``0.0``); ``per_row_block_key`` is always populated from
    ``(forecast_dates[i], leads[i])`` and is consumed by Task 13's paired
    block-bootstrap.

    Raises:
        ValueError: when input lists differ in length.
    """
    n = len(candidate_predictions)
    if not (
        len(blended_raw_probs) == n
        and len(market_yes_prices) == n
        and len(outcomes) == n
        and len(forecast_dates) == n
        and len(leads) == n
    ):
        raise ValueError(
            "simulate_stage1_roi: all input lists must have the same length; "
            f"got candidate_predictions={n}, blended_raw_probs={len(blended_raw_probs)}, "
            f"market_yes_prices={len(market_yes_prices)}, outcomes={len(outcomes)}, "
            f"forecast_dates={len(forecast_dates)}, leads={len(leads)}"
        )

    pnls: list[float] = [0.0] * n
    wagereds: list[float] = [0.0] * n
    block_keys: list[tuple[str, int]] = [
        (forecast_dates[i], leads[i]) for i in range(n)
    ]

    n_traded = 0
    for i in range(n):
        p_cand = candidate_predictions[i]
        raw_p = blended_raw_probs[i]
        price = market_yes_prices[i]
        outcome = outcomes[i]

        # Filter 1: plateau-block on raw_prob.
        if block_raw_prob_range[0] <= raw_p <= block_raw_prob_range[1]:
            continue
        # Filter 2: calibrated-floor.
        if p_cand < min_calibrated_prob:
            continue

        # Filter 3: Kelly sizing (returns SizingResult with side + dollars).
        sizing = fractional_kelly(
            p_cand,
            price,
            kelly_fraction=kelly_fraction,
            max_pct=max_pct,
            bankroll=bankroll,
        )
        if sizing.side == Side.NONE or sizing.dollars <= 0:
            continue

        # YES vs NO side handling. Kelly picks the side; we honor it.
        # YES: buy at `price`, win pays 1.0 when outcome==1.
        # NO:  buy at `1 - price`, win pays 1.0 when outcome==0.
        if sizing.side == Side.YES:
            price_paid = price
            won = outcome == 1
        else:  # Side.NO
            price_paid = 1.0 - price
            won = outcome == 0

        # Defensive: degenerate price (≤0 or ≥1) is already filtered by
        # fractional_kelly (returns Side.NONE), but keep a guard so the
        # divisor below can't blow up if upstream invariants slip.
        if price_paid <= 0.0 or price_paid >= 1.0:
            continue

        contracts = max(int(sizing.dollars / max(price_paid, 0.01)), 1)
        wagered = contracts * price_paid
        fee_buy = order_fee(price_paid, contracts)  # taker default
        pnl_per_contract = (1.0 - price_paid) if won else (-price_paid)
        pnl = contracts * pnl_per_contract - fee_buy

        pnls[i] = pnl
        wagereds[i] = wagered
        n_traded += 1

    sum_wagered = sum(wagereds)
    if n_traded < 30 or sum_wagered == 0.0:
        return {
            "roi_point": None,
            "per_row_pnl": pnls,
            "per_row_wagered": wagereds,
            "per_row_block_key": block_keys,
            "n_val_traded": n_traded,
            "status": "insufficient_data",
        }

    return {
        "roi_point": sum(pnls) / sum_wagered,
        "per_row_pnl": pnls,
        "per_row_wagered": wagereds,
        "per_row_block_key": block_keys,
        "n_val_traded": n_traded,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Task 13: Paired block-bootstrap CI
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    *,
    values: list[float],
    block_keys: list[tuple[str, int]],
    n_iter: int = BOOTSTRAP_ITER,
    seed: int,
    block_mean_length: int = BOOTSTRAP_BLOCK_MEAN_LENGTH,
) -> dict[str, float]:
    """Politis-Romano stationary block bootstrap (Politis & Romano 1994).

    Groups rows by forecast_date (first element of block_keys), then resamples
    geometrically-distributed blocks of dates with mean length `block_mean_length`.
    This handles multi-day synoptic autocorrelation that single-date blocks
    underestimate by √(L_eff).

    Returns:
        {mean, ci_lb, ci_ub, p2_5, p50, p97_5, p99, mean_rows_per_block}
    """
    if len(values) != len(block_keys):
        raise ValueError("block_bootstrap_ci: values and block_keys must be equal length")
    if not values:
        return {
            "mean": 0.0, "ci_lb": 0.0, "ci_ub": 0.0,
            "p2_5": 0.0, "p50": 0.0, "p97_5": 0.0, "p99": 0.0,
            "mean_rows_per_block": 0.0,
        }

    # Group rows by forecast_date.
    by_date: dict[str, list[float]] = {}
    for v, (fdate, _lead) in zip(values, block_keys):
        by_date.setdefault(fdate, []).append(v)
    dates = sorted(by_date.keys())
    n_dates = len(dates)
    n = len(values)
    # mean_rows_per_block is now informational: per-date row count (not per-resampled-block).
    mean_rows_per_block = n / max(n_dates, 1)

    rng = np.random.default_rng(seed=seed & 0xFFFFFFFF)
    p_continue = 1.0 - 1.0 / block_mean_length  # geometric block-length parameter

    iter_means: list[float] = []
    for _ in range(n_iter):
        rows_resampled: list[float] = []
        while len(rows_resampled) < n:
            # Start a new block at a random date.
            d_idx = int(rng.integers(0, n_dates))
            # Extend block geometrically.
            while True:
                rows_resampled.extend(by_date[dates[d_idx]])
                if len(rows_resampled) >= n:
                    break
                if rng.random() >= p_continue:
                    break
                d_idx = (d_idx + 1) % n_dates  # wrap around (stationary)
        rows_resampled = rows_resampled[:n]
        iter_means.append(float(np.mean(rows_resampled)))

    arr = np.asarray(iter_means)
    return {
        "mean": float(np.mean(arr)),
        "ci_lb": float(np.percentile(arr, 2.5)),
        "ci_ub": float(np.percentile(arr, 97.5)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p50": float(np.percentile(arr, 50)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "p99": float(np.percentile(arr, 99)),
        "mean_rows_per_block": float(mean_rows_per_block),
    }


def _cell_seed(city: str, var: str, candidate_name: str) -> int:
    """Per-cell deterministic seed.

    A1 fix: Python's hash() is randomized per interpreter invocation unless
    PYTHONHASHSEED is set. Using SHA-256 over a stable byte representation
    guarantees bit-identical reruns across processes without env-var coupling.
    """
    digest = hashlib.sha256(f"{city}|{var}|{candidate_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def block_bootstrap_ratio_ci(
    *,
    numerators: list[float],
    denominators: list[float],
    block_keys: list[tuple[str, int]],
    n_iter: int = BOOTSTRAP_ITER,
    seed: int,
) -> dict[str, float]:
    """Paired block-bootstrap on a RATIO statistic sum(num) / sum(denom).

    A3 fix: ROI per FR-6 is "the ratio sum(pnl) / sum(wagered) recomputed per
    resample." block_bootstrap_ci returns mean-of-pnl, which is only equal to
    the ratio when all denominators are equal (Kelly sizes vary, so they
    aren't). This helper recomputes the ratio per bootstrap iteration.

    Returns the same shape as block_bootstrap_ci: {mean, ci_lb, p2_5, p50,
    p97_5, ci_ub, mean_rows_per_block}. `mean` is the point-estimate ratio.
    Returns NaN-fielded result when >50% of resamples have sum_denom == 0
    (caller treats this as `simulated_roi.status = "insufficient_data"`).
    """
    # NB: deliberately NOT a chained `!=` comparison — Python's `A != B != C`
    # short-circuits as `(A != B) and (B != C)`, which silently passes when
    # `A == B` but `B != C`. Wrap chained `==` in `not (...)` instead.
    if not (len(numerators) == len(denominators) == len(block_keys)):
        raise ValueError(
            "block_bootstrap_ratio_ci: numerators, denominators, and block_keys "
            "must be equal length"
        )
    if not numerators:
        return {
            "mean": 0.0,
            "ci_lb": 0.0,
            "p2_5": 0.0,
            "p50": 0.0,
            "p97_5": 0.0,
            "ci_ub": 0.0,
            "mean_rows_per_block": 0.0,
        }

    blocks: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for num, denom, k in zip(numerators, denominators, block_keys, strict=True):
        blocks[k].append((num, denom))
    block_list = list(blocks.values())
    n_blocks = len(block_list)
    mean_rows_per_block = len(numerators) / n_blocks if n_blocks > 0 else 0.0

    point_sum_num = sum(numerators)
    point_sum_denom = sum(denominators)
    point_ratio = point_sum_num / point_sum_denom if point_sum_denom > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boot_ratios = np.empty(n_iter, dtype=np.float64)
    for it in range(n_iter):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        sum_num = 0.0
        sum_denom = 0.0
        for idx in sampled:
            for num, denom in block_list[idx]:
                sum_num += num
                sum_denom += denom
        boot_ratios[it] = (sum_num / sum_denom) if sum_denom > 0 else float("nan")

    finite = boot_ratios[np.isfinite(boot_ratios)]
    if len(finite) < int(0.5 * n_iter):
        # >50% of bootstrap iterations failed (degenerate denominator).
        return {
            "mean": float("nan"),
            "ci_lb": float("nan"),
            "p2_5": float("nan"),
            "p50": float("nan"),
            "p97_5": float("nan"),
            "ci_ub": float("nan"),
            "mean_rows_per_block": mean_rows_per_block,
        }

    return {
        "mean": point_ratio,
        "ci_lb": float(np.percentile(finite, 2.5)),
        "p2_5": float(np.percentile(finite, 2.5)),
        "p50": float(np.percentile(finite, 50)),
        "p97_5": float(np.percentile(finite, 97.5)),
        "ci_ub": float(np.percentile(finite, 97.5)),
        "mean_rows_per_block": mean_rows_per_block,
    }


# ---------------------------------------------------------------------------
# Task 14: Structural-veto eligibility + n_clipped veto (FR-9)
# ---------------------------------------------------------------------------


def compute_eligibility(
    *,
    fold_warnings: list[list[str]],
    fold_eligible: list[bool],
    n_clipped: int,
    n_val_total: int,
) -> dict[str, Any]:
    """Apply FR-9 structural-veto rules + A4 fit-failure aggregation.

    Args:
        fold_warnings: 5-element list of per-fold warning lists. Includes
            ``"fit_exception:<X>"`` strings for L-BFGS-B / fitter failures
            (set by :func:`cv_score_cell`).
        fold_eligible: 5-element list of per-fold eligibility (``False`` when
            ``fit()`` raised OR ``validate()`` returned ``ok=False``). A4 fix:
            FM-4 requires ≥3/5 ``False`` entries → candidate _ineligible.
        n_clipped: count of predictions hitting the ``[eps, 1-eps]`` clip.
        n_val_total: total val rows across folds (denominator for n_clipped pct).

    Returns:
        ``{
            "eligible": bool,
            "reasons": list[str],          # ineligibility reasons (empty when eligible)
            "per_fold_diagnostics": {
                "sigma_veto_folds": list[int],
                "unstable_b_folds": list[int],
                "d_at_zero_folds": list[int],
                "unstable_d_folds": list[int],
            },
        }``

        ``unstable_d`` is recorded in ``per_fold_diagnostics.unstable_d_folds``
        but never added to ``reasons`` — per FR-9 it is flag-only and does NOT
        block eligibility (matches ``data/emos/overrides.json`` convention).

    Raises:
        ValueError: when ``fold_warnings`` or ``fold_eligible`` is not length 5.
    """
    if len(fold_warnings) != 5:
        raise ValueError(
            f"compute_eligibility: expected 5 folds, got {len(fold_warnings)}"
        )
    if len(fold_eligible) != 5:
        raise ValueError(
            f"compute_eligibility: expected 5 fold_eligible entries, "
            f"got {len(fold_eligible)}"
        )

    sigma_folds = [i for i, w in enumerate(fold_warnings) if "incompatible_sigma" in w]
    unstable_b_folds = [i for i, w in enumerate(fold_warnings) if "unstable_b" in w]
    d_at_zero_folds = [i for i, w in enumerate(fold_warnings) if "d_at_zero" in w]
    unstable_d_folds = [i for i, w in enumerate(fold_warnings) if "unstable_d" in w]

    reasons: list[str] = []

    # σ-veto in ANY fold (non-overridable per emos_fit.py:40-44).
    # One reason per offending fold so the report identifies which fold tripped.
    for i in sigma_folds:
        reasons.append(f"incompatible_sigma:fold_{i}")

    # unstable_b ≥ 3/5
    if len(unstable_b_folds) >= 3:
        reasons.append("unstable_b:>=3/5")

    # d_at_zero ≥ 3/5
    if len(d_at_zero_folds) >= 3:
        reasons.append("d_at_zero:>=3/5")

    # A4 fix: fit-failure aggregation per FM-4. A fold where fit() raised OR
    # validate() returned ok=False is recorded in fold_eligible. ≥3 False
    # entries → _ineligible (the candidate cannot reliably fit).
    n_fold_failures = sum(1 for e in fold_eligible if not e)
    if n_fold_failures >= 3:
        reasons.append("fold_fit_failures:>=3/5")

    # n_clipped > 5% (strict inequality — exact 5% boundary is still eligible).
    if n_val_total > 0 and (n_clipped / n_val_total) > WINNER_GATE_N_CLIPPED_PCT_MAX:
        reasons.append("n_clipped:>5%")

    return {
        "eligible": len(reasons) == 0,
        "reasons": reasons,
        "per_fold_diagnostics": {
            "sigma_veto_folds": sigma_folds,
            "unstable_b_folds": unstable_b_folds,
            "d_at_zero_folds": d_at_zero_folds,
            "unstable_d_folds": unstable_d_folds,
        },
    }


# ---------------------------------------------------------------------------
# Verdict rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateResult:
    """Per-cell aggregated result for a candidate.

    Used by compute_verdict to apply FR-7 + FR-8 gates. Built by the scoring
    pipeline (Task 17) from cv_score_cell + metrics + bootstrap CIs.
    """

    name: str
    logskill_lb: float                              # FR-7.1: vs identity
    roi_point: float                                # diagnostic only (ROI demoted from FR-7)
    roi_ci_lb: float                                # diagnostic only
    fold_logskill_positive_count: int               # FR-7.2 numerator (denominator always 5)
    eligible: bool                                  # FR-9

    # EMOS-only (FR-8); identity / iso leave these at defaults.
    logskill_lb_vs_iso: float = 0.0
    bss_lb_vs_iso: float = 0.0
    emos_gate_pass: Any = True                      # bool | "not_evaluated"
    evaluable_folds_vs_iso: int = 5

    # Phase 1A.1 additions
    logskill_mean: float = 0.0          # bootstrap mean (tiebreaker + worse_than_identity)
    logskill_p99: float = 0.0           # bootstrap p99 (worse_than_identity rule)
    ineligibility_reasons: list[str] = field(default_factory=list)


def _passes_fr7(cand: CandidateResult) -> bool:
    """FR-7 winner gate (Phase 1A.1 amended; 4 clauses, ROI demoted).

    Clauses (evaluated in order; eligibility short-circuits):
      1. Candidate is not ineligible per FR-9.
      2. LogSkill_LB(X vs identity) > +0.01 (multiple-comparison effect-size threshold).
      3. LogSkill_per_fold(X) > 0 in ≥ 4 of 5 folds.
      4. LogSkill_mean(X vs identity) > 0 (point estimate; guards against
         pathological LB > 0 / mean ≤ 0 cases from skewed bootstrap distributions).

    ROI clauses removed per Phase 1A.1 amendment — ROI at n≈150 is
    sampling-noise dominated. simulated_roi fields remain in the schema as
    diagnostic only.
    """
    if not cand.eligible:
        return False
    if cand.logskill_lb <= WINNER_GATE_LOGSKILL_LB_MIN:
        return False
    if cand.fold_logskill_positive_count < WINNER_GATE_FOLD_CONSISTENCY_MIN:
        return False
    if cand.logskill_mean <= 0:
        return False
    return True


def _passes_fr8(cand: CandidateResult) -> bool:
    """FR-8 only applies to the EMOS candidate."""
    if cand.name != "emos":
        return True  # vacuously
    if cand.emos_gate_pass == "not_evaluated":
        return True  # FR-8 baseline ineligible → fall through to FR-7
    if cand.logskill_lb_vs_iso <= EMOS_GATE_LOGSKILL_LB_VS_ISO_MIN:
        return False
    if cand.bss_lb_vs_iso <= EMOS_GATE_BSS_LB_VS_ISO_MIN:
        return False
    return True


def _format_ineligibility_summary(non_identity_candidates: list[CandidateResult]) -> str:
    """Build a 'name:reason1,reason2' comma-joined summary of ineligible candidates.

    Returns empty string if no candidate is ineligible.
    """
    parts: list[str] = []
    for c in non_identity_candidates:
        if not c.eligible:
            reasons = ",".join(c.ineligibility_reasons) if c.ineligibility_reasons else "ineligible"
            parts.append(f"{c.name}:{reasons}")
    return ", ".join(parts)


def compute_verdict(
    candidates: list[CandidateResult],
    *,
    n_val_total: int,
    n_val_traded: int,
    n_unique_strikes: int,
) -> dict[str, Any]:
    """Apply FR-7 + FR-8 + FR-10 + FR-11 to produce the per-cell verdict (Phase 1A.1).

    6-state palette: winner, insufficient_evidence, worse_than_identity,
    insufficient_n, insufficient_diversity, no_separation.

    FR-7 winner gate (`_passes_fr7`) has 4 clauses: eligible + LogSkill_LB > +0.01
    + fold ≥ 4/5 + logskill_mean > 0. Clause 4 is a pathological-bootstrap-distribution
    guard (skewed CIs where LB > 0.01 but mean ≤ 0) and is a tightening, not loosening,
    of the design doc's stated 3-clause gate.

    Returns:
        {verdict, verdict_message, winner_candidate}
    """
    # FR-11 gate ladder.
    if n_val_total < FR11_2_N_VAL_TOTAL_MIN:
        return {
            "verdict": "insufficient_n",
            "verdict_message": (
                f"need more settled forecast_dates "
                f"(n_val_total={n_val_total}; need ≥ {FR11_2_N_VAL_TOTAL_MIN})"
            ),
            "winner_candidate": None,
        }
    if n_unique_strikes < 5:
        return {
            "verdict": "insufficient_diversity",
            "verdict_message": "bracket distribution too narrow to discriminate",
            "winner_candidate": None,
        }

    # Filter to non-identity candidates that pass FR-7 AND FR-8.
    non_identity = [c for c in candidates if c.name != "identity"]
    passing = [c for c in non_identity if _passes_fr7(c) and _passes_fr8(c)]

    if passing:
        # Sort by LogSkill_LB descending, then logskill_mean as secondary tiebreaker.
        passing.sort(key=lambda c: (-c.logskill_lb, -c.logskill_mean))
        if len(passing) >= 2:
            top, second = passing[0], passing[1]
            ls_diff = abs(top.logskill_lb - second.logskill_lb)
            mean_diff = abs(top.logskill_mean - second.logskill_mean)
            if ls_diff < TIEBREAKER_LOGSKILL_LB_BAND and mean_diff < TIEBREAKER_LOGSKILL_LB_BAND:
                return {
                    "verdict": "no_separation",
                    "verdict_message": "candidates tied within noise; defaulting to identity",
                    "winner_candidate": None,
                }
        winner = passing[0]
        return {
            "verdict": "winner",
            "verdict_message": f"winner: {winner.name}",
            "winner_candidate": winner.name,
        }

    # No winner. Check worse_than_identity: ≥2 eligible non-identity candidates,
    # ALL with logskill_mean < 0 AND logskill_p99 < 0.
    eligible_non_id = [c for c in non_identity if c.eligible]
    if len(eligible_non_id) >= 2 and all(
        (c.logskill_mean < 0 and c.logskill_p99 < 0) for c in eligible_non_id
    ):
        ineligible_summary = _format_ineligibility_summary(non_identity)
        return {
            "verdict": "worse_than_identity",
            "verdict_message": (
                f"all {len(eligible_non_id)} eligible candidates confidently worse than identity"
                + (f"; ineligible: [{ineligible_summary}]" if ineligible_summary else "")
            ),
            "winner_candidate": None,
        }

    # Default fallthrough: insufficient_evidence with enumerated eligibility.
    eligible_summary = ", ".join(c.name for c in non_identity if c.eligible) or "none"
    ineligible_summary = _format_ineligibility_summary(non_identity)
    return {
        "verdict": "insufficient_evidence",
        "verdict_message": (
            f"no candidate confidently beats identity at current n; "
            f"eligible: [{eligible_summary}]"
            + (f", ineligible: [{ineligible_summary}]" if ineligible_summary else "")
        ),
        "winner_candidate": None,
    }


# ---------------------------------------------------------------------------
# Task 16: Per-cell scoring orchestrator
# ---------------------------------------------------------------------------


def score_cell(
    *,
    city: str,
    var: str,
    rows: list[FeatureRow],
    candidates: list[CalibratorCandidate],
) -> dict[str, Any]:
    """End-to-end per-cell scoring: CV → metrics → bootstrap → eligibility → verdict.

    Returns the per-cell entry portion of per_cell_winners.json schema (excluding
    the surrounding production / disagreement_with_production / in_sample_reference
    fields that the assembler adds in Task 17).
    """
    n_unique = len({r.forecast_date for r in rows})

    _empty_flags = {
        "pnl_logskill_disagreement": False,
        "metric_disagreement": False,
        "gate_margin_thin": False,
        "below_power_floor": True,
    }

    # FR-11.1: catch unique-dates < 6.
    if n_unique < 6:
        return {
            "verdict": "insufficient_n",
            "verdict_message": "need more settled forecast_dates",
            "winner_candidate": None,
            "n": {"val_total": 0, "val_traded": 0, "unique_forecast_dates": n_unique},
            "candidates": {},
            "flags": _empty_flags,
        }

    # FR-11.2: catch n_val_total < FR11_2_N_VAL_TOTAL_MIN (we approximate via row
    # count; CV val partitions sum to the full row set since identity / iso never
    # fail to score an eligible fold).
    if len(rows) < FR11_2_N_VAL_TOTAL_MIN:
        return {
            "verdict": "insufficient_n",
            "verdict_message": (
                f"need more settled forecast_dates "
                f"(n={len(rows)}; need ≥ {FR11_2_N_VAL_TOTAL_MIN})"
            ),
            "winner_candidate": None,
            "n": {"val_total": len(rows), "val_traded": 0, "unique_forecast_dates": n_unique},
            "candidates": {},
            "flags": _empty_flags,
        }

    # FR-11.3: unique-strike-tuple diversity.
    unique_strikes = len({(r.strike_type, r.floor, r.cap) for r in rows})
    if unique_strikes < 5:
        return {
            "verdict": "insufficient_diversity",
            "verdict_message": "bracket distribution too narrow to discriminate",
            "winner_candidate": None,
            "n": {
                "val_total": len(rows),
                "val_traded": 0,
                "unique_forecast_dates": n_unique,
            },
            "candidates": {},
            "flags": _empty_flags,
        }

    cv_out = cv_score_cell(rows, candidates=candidates)

    # Identity must be present for vs-identity computations.
    identity_block = cv_out["per_candidate"].get("identity")
    if identity_block is None:
        raise ValueError("score_cell: IdentityCandidate must be in candidates list")

    # Build per-candidate aggregated blocks.
    per_cand_block: dict[str, Any] = {}
    candidate_results: list[CandidateResult] = []

    iso_block = cv_out["per_candidate"].get("iso_v2_global")  # may be None if not in candidates

    for cand in candidates:
        cand_block = cv_out["per_candidate"][cand.name]
        val_rows = cand_block["val_rows"]
        val_preds = cand_block["val_predictions"]

        if not val_rows:
            # Candidate failed every fold. Keep this block's keyset in sync with
            # block_bootstrap_ci's return shape — schema validator requires every
            # quantile (p2.5, p50, p97.5, p99) regardless of fold outcome.
            empty_block: dict[str, Any] = {
                "logskill": {"mean": 0.0, "p2.5": 0.0, "p50": 0.0, "p97.5": 0.0, "p99": 0.0},
                "brier": {"mean": 0.0, "p2.5": 0.0, "p50": 0.0, "p97.5": 0.0},
                "simulated_roi": {
                    "mean": None, "ci_lb": None, "ci_ub": None,
                    "n_val_traded": 0, "status": "insufficient_data",
                    "roi_ci_lb_informative": False,
                },
                "fold_logskill": [None] * 5,
                "n_clipped": 0,
                "mean_rows_per_block": 0.0,
                # Validator invariant: cell_bootstrap_degenerate ↔ mean_rows_per_block < 2.0.
                # With zero rows the cell trivially satisfies the degenerate condition.
                "cell_bootstrap_degenerate": True,
                "eligibility": "_ineligible",
                "ineligibility_reasons": ["all_folds_failed"],
                "per_fold_diagnostics": {
                    "sigma_veto_folds": [],
                    "unstable_b_folds": [],
                    "d_at_zero_folds": [],
                    "unstable_d_folds": [],
                },
            }
            # EMOS candidate must carry emos_promotion_gate_decision on every
            # path the validator inspects (assembler invariant: cname == "emos"
            # iff key present). When EMOS fails every fold there is no paired
            # vs-iso bootstrap data, so emit the underpowered defaults.
            if cand.name == "emos":
                empty_block["emos_promotion_gate_decision"] = {
                    "logskill_lb_vs_iso": None,
                    "bss_lb_vs_iso": None,
                    "structural_warnings": [],
                    "gate_pass": "not_evaluated",
                    "evaluable_folds": 0,
                    "fr8_underpowered": True,
                }
            per_cand_block[cand.name] = empty_block
            continue

        # Aggregate identity's predictions over the same val rows.
        # NOTE: Identity scores every fold, so its val_rows ⊇ this candidate's val_rows.
        # Build a dict keyed by ticker for pairing.
        id_pred_by_ticker = dict(
            zip([r.ticker for r in identity_block["val_rows"]], identity_block["val_predictions"])
        )
        # Defensive: expanding-window CV does not repeat rows, so identity's
        # ticker set should be unique. If not, dict(zip(...)) silently drops
        # earlier entries — fail loudly instead.
        assert len(id_pred_by_ticker) == len(identity_block["val_rows"]), (
            "duplicate tickers in identity val_rows"
        )
        identity_preds_aligned = [id_pred_by_ticker[r.ticker] for r in val_rows]
        outcomes = [r.outcome for r in val_rows]

        # LogSkill point estimate.
        logskill_point = compute_logskill(val_preds, identity_preds_aligned, outcomes)

        # Per-row log-loss differences for block bootstrap.
        ll_cand = [_log_loss_row(p, y) for p, y in zip(val_preds, outcomes)]
        ll_id = [_log_loss_row(p, y) for p, y in zip(identity_preds_aligned, outcomes)]
        ll_diff = [a - b for a, b in zip(ll_id, ll_cand)]  # higher = candidate better
        block_keys = [(r.forecast_date, r.lead) for r in val_rows]

        ls_boot = block_bootstrap_ci(
            values=ll_diff,
            block_keys=block_keys,
            n_iter=BOOTSTRAP_ITER,
            seed=_cell_seed(city, var, cand.name + ":logskill"),
        )

        # Brier metrics.
        brier_point = compute_bss(val_preds, identity_preds_aligned, outcomes)
        brier_row_diff = [
            (p_id - o) ** 2 - (p_c - o) ** 2
            for p_c, p_id, o in zip(val_preds, identity_preds_aligned, outcomes)
        ]
        brier_boot = block_bootstrap_ci(
            values=brier_row_diff,
            block_keys=block_keys,
            n_iter=BOOTSTRAP_ITER,
            seed=_cell_seed(city, var, cand.name + ":brier"),
        )

        # ROI — A3 fix: ratio bootstrap, not mean-of-pnl + divide-by-mean-wagered.
        roi_result = simulate_stage1_roi(
            candidate_predictions=val_preds,
            blended_raw_probs=[r.blended_raw_prob for r in val_rows],
            market_yes_prices=[r.market_yes_price for r in val_rows],
            outcomes=outcomes,
            forecast_dates=[r.forecast_date for r in val_rows],
            leads=[r.lead for r in val_rows],
        )
        if roi_result["status"] == "ok":
            roi_boot = block_bootstrap_ratio_ci(
                numerators=roi_result["per_row_pnl"],
                denominators=roi_result["per_row_wagered"],
                block_keys=roi_result["per_row_block_key"],
                n_iter=BOOTSTRAP_ITER,
                seed=_cell_seed(city, var, cand.name + ":roi"),
            )
            roi_block = {
                "mean": roi_result["roi_point"],
                "ci_lb": roi_boot["ci_lb"],
                "ci_ub": roi_boot["ci_ub"],
                "n_val_traded": roi_result["n_val_traded"],
                "status": "ok",
                "roi_ci_lb_informative": (
                    roi_boot["ci_ub"] - roi_boot["ci_lb"] < 0.20
                    if math.isfinite(roi_boot["ci_ub"]) and math.isfinite(roi_boot["ci_lb"])
                    else False
                ),
            }
        else:
            roi_block = {
                "mean": None,
                "ci_lb": None,
                "ci_ub": None,
                "n_val_traded": roi_result["n_val_traded"],
                "status": "insufficient_data",
                "roi_ci_lb_informative": False,
            }

        # n_clipped + eligibility — A4 fix: pass fold_eligible so fit failures
        # at ≥3/5 mark candidate _ineligible.
        n_clipped = count_clipped(val_preds)
        eligibility_result = compute_eligibility(
            fold_warnings=cand_block["fold_warnings"],
            fold_eligible=cand_block["fold_eligible"],
            n_clipped=n_clipped,
            n_val_total=len(val_preds),
        )

        # Per-fold LogSkill — A2 fix: compute from per-fold val rows + identity
        # per-fold predictions, NOT the aggregate logskill_point placeholder.
        identity_fold_val_rows = identity_block["fold_val_rows"]
        identity_fold_preds = identity_block["fold_predictions"]
        fold_logskill: list[float | None] = []
        for f_i in range(5):
            cand_fold_preds = cand_block["fold_predictions"][f_i]
            cand_fold_rows = cand_block["fold_val_rows"][f_i]
            if cand_fold_preds is None or not cand_block["fold_eligible"][f_i]:
                fold_logskill.append(None)
                continue
            id_fold_preds = identity_fold_preds[f_i]
            id_fold_rows = identity_fold_val_rows[f_i]
            if id_fold_preds is None or id_fold_rows is None:
                fold_logskill.append(None)
                continue
            # Pair cand vs identity on this fold's shared val rows by ticker.
            id_by_ticker = dict(zip(
                [r.ticker for r in id_fold_rows], id_fold_preds
            ))
            paired_cand_preds: list[float] = []
            paired_id_preds: list[float] = []
            paired_outcomes: list[int] = []
            for cr, cp in zip(cand_fold_rows, cand_fold_preds):
                ip = id_by_ticker.get(cr.ticker)
                if ip is None or cr.outcome is None:
                    continue
                paired_cand_preds.append(cp)
                paired_id_preds.append(ip)
                paired_outcomes.append(cr.outcome)
            if not paired_outcomes:
                fold_logskill.append(None)
                continue
            fold_ls = compute_logskill(paired_cand_preds, paired_id_preds, paired_outcomes)
            fold_logskill.append(fold_ls)

        # FR-8 default for EMOS when iso_v2_global is absent from the candidates
        # list — validator invariant: emos_promotion_gate_decision present iff
        # name=="emos". Without this fallback, an EMOS-only run would emit a
        # block missing the key and trip validate_schema. Mirrors the empty-fold
        # underpowered defaults.
        fr8_block: dict[str, Any] | None = None
        if cand.name == "emos" and iso_block is None:
            fr8_block = {
                "logskill_lb_vs_iso": None,
                "bss_lb_vs_iso": None,
                "structural_warnings": [],
                "gate_pass": "not_evaluated",
                "evaluable_folds": 0,
                "fr8_underpowered": True,
            }
        if cand.name == "emos" and iso_block is not None:
            # FR-8: bootstrap LogSkill / BSS deltas of EMOS vs iso (per-row).
            iso_pred_by_ticker = dict(
                zip([r.ticker for r in iso_block["val_rows"]], iso_block["val_predictions"])
            )
            # Defensive: same duplicate-ticker guard as identity above.
            assert len(iso_pred_by_ticker) == len(iso_block["val_rows"]), (
                "duplicate tickers in iso val_rows"
            )
            iso_preds_aligned = [
                iso_pred_by_ticker.get(r.ticker) for r in val_rows
            ]
            # Restrict to rows where iso also produced a prediction.
            paired = [
                (p_c, p_iso, p_id, o, (r.forecast_date, r.lead))
                for p_c, p_iso, p_id, o, r in zip(
                    val_preds, iso_preds_aligned, identity_preds_aligned, outcomes, val_rows
                )
                if p_iso is not None
            ]
            if len(paired) >= 5:
                ll_emos = [_log_loss_row(t[0], t[3]) for t in paired]
                ll_iso = [_log_loss_row(t[1], t[3]) for t in paired]
                ll_emos_vs_iso = [a - b for a, b in zip(ll_iso, ll_emos)]
                br_emos = [(t[0] - t[3]) ** 2 for t in paired]
                br_iso = [(t[1] - t[3]) ** 2 for t in paired]
                br_emos_vs_iso = [a - b for a, b in zip(br_iso, br_emos)]
                bk = [t[4] for t in paired]
                ls_vs_iso_boot = block_bootstrap_ci(
                    values=ll_emos_vs_iso, block_keys=bk,
                    n_iter=BOOTSTRAP_ITER,
                    seed=_cell_seed(city, var, "emos:ls_vs_iso"),
                )
                bss_vs_iso_boot = block_bootstrap_ci(
                    values=br_emos_vs_iso, block_keys=bk,
                    n_iter=BOOTSTRAP_ITER,
                    seed=_cell_seed(city, var, "emos:bss_vs_iso"),
                )
                evaluable_folds = sum(
                    1 for f in iso_block["fold_eligible"] if f
                )
                # Width scaling if underpowered.
                scaling = (5.0 / max(evaluable_folds, 1)) ** 0.5 if evaluable_folds < 5 else 1.0
                ls_lb_vs_iso = ls_vs_iso_boot["ci_lb"] * scaling
                bss_lb_vs_iso = bss_vs_iso_boot["ci_lb"] * scaling
                gate_pass = (
                    ls_lb_vs_iso > EMOS_GATE_LOGSKILL_LB_VS_ISO_MIN
                    and bss_lb_vs_iso > EMOS_GATE_BSS_LB_VS_ISO_MIN
                )
                fr8_block = {
                    "logskill_lb_vs_iso": ls_lb_vs_iso,
                    "bss_lb_vs_iso": bss_lb_vs_iso,
                    "structural_warnings": [],
                    "gate_pass": gate_pass,
                    "evaluable_folds": evaluable_folds,
                    "fr8_underpowered": evaluable_folds < 5,
                }
            else:
                fr8_block = {
                    "logskill_lb_vs_iso": None,
                    "bss_lb_vs_iso": None,
                    "structural_warnings": [],
                    "gate_pass": "not_evaluated",
                    "evaluable_folds": 0,
                    "fr8_underpowered": True,
                }

        block = {
            "logskill": {
                "mean": logskill_point,
                "p2.5": ls_boot["p2_5"], "p50": ls_boot["p50"], "p97.5": ls_boot["p97_5"],
                "p99": ls_boot["p99"],
            },
            "brier": {
                "mean": brier_point,
                "p2.5": brier_boot["p2_5"], "p50": brier_boot["p50"], "p97.5": brier_boot["p97_5"],
            },
            "simulated_roi": roi_block,
            "fold_logskill": fold_logskill,
            "n_clipped": n_clipped,
            "mean_rows_per_block": ls_boot["mean_rows_per_block"],
            "cell_bootstrap_degenerate": ls_boot["mean_rows_per_block"] < 2.0,
            "eligibility": "eligible" if eligibility_result["eligible"] else "_ineligible",
            "ineligibility_reasons": eligibility_result["reasons"],
            "per_fold_diagnostics": eligibility_result["per_fold_diagnostics"],
        }
        if cand.name == "emos":
            # Always emit on emos (the fallback above guarantees fr8_block is set
            # even when iso_block is missing). Validator invariant: key present
            # iff name=="emos".
            assert fr8_block is not None, (
                "fr8_block must be populated for EMOS candidate (iso-present or fallback)"
            )
            block["emos_promotion_gate_decision"] = fr8_block

        per_cand_block[cand.name] = block

        # Build CandidateResult for verdict.
        positive_fold_count = sum(
            1 for f in fold_logskill if f is not None and f > 0
        )
        candidate_results.append(CandidateResult(
            name=cand.name,
            logskill_lb=ls_boot["ci_lb"],
            roi_point=roi_block["mean"] if roi_block["mean"] is not None else -1.0,
            roi_ci_lb=roi_block["ci_lb"] if roi_block["ci_lb"] is not None else -1.0,
            fold_logskill_positive_count=positive_fold_count,
            eligible=eligibility_result["eligible"],
            logskill_lb_vs_iso=cast(
                float, (fr8_block or {}).get("logskill_lb_vs_iso", 0.0) or 0.0
            ),
            bss_lb_vs_iso=cast(
                float, (fr8_block or {}).get("bss_lb_vs_iso", 0.0) or 0.0
            ),
            emos_gate_pass=(fr8_block or {}).get("gate_pass", True),
            evaluable_folds_vs_iso=cast(
                int, (fr8_block or {}).get("evaluable_folds", 5)
            ),
            logskill_mean=ls_boot["mean"],
            logskill_p99=ls_boot["p99"],
            ineligibility_reasons=list(eligibility_result["reasons"]),
        ))

    verdict = compute_verdict(
        candidate_results,
        n_val_total=cv_out["n_val_total"],
        n_val_traded=max(
            (per_cand_block[c.name]["simulated_roi"]["n_val_traded"] for c in candidates),
            default=0,
        ),
        n_unique_strikes=unique_strikes,
    )

    # Cross-metric disagreement flags.
    # pnl_logskill_disagreement fires ONLY when ROI is informative AND negative.
    # At the current corpus where the simulator is structurally pessimistic, most
    # cells have ROI hitting the bet-fully-lost floor — that signal would be a
    # false alarm (E-17 amended).
    winner_block = (
        per_cand_block.get(verdict["winner_candidate"])
        if verdict.get("winner_candidate")
        else None
    )
    roi_info = (winner_block or {}).get("simulated_roi", {})
    roi_informative = bool(roi_info.get("roi_ci_lb_informative", False))
    roi_ci_lb = roi_info.get("ci_lb")
    roi_mean = roi_info.get("mean")
    pnl_disagreement = bool(
        winner_block is not None
        and roi_informative
        and roi_ci_lb is not None
        and roi_ci_lb > -0.95
        and roi_mean is not None
        and roi_mean <= 0
    )

    # gate_margin_thin: winner LB inside the GATE_MARGIN_THIN_WIDTH band above the gate.
    gate_margin_thin = False
    if verdict.get("verdict") == "winner" and winner_block is not None:
        winner_lb = winner_block["logskill"]["p2.5"]
        thin_band_upper = WINNER_GATE_LOGSKILL_LB_MIN + GATE_MARGIN_THIN_WIDTH
        if WINNER_GATE_LOGSKILL_LB_MIN < winner_lb < thin_band_upper:
            gate_margin_thin = True

    # below_power_floor: cell is below the statistical-power floor for confident detection.
    n_val_total_for_flags = cv_out["n_val_total"]

    flags = {
        "pnl_logskill_disagreement": pnl_disagreement,
        "metric_disagreement": False,
        "gate_margin_thin": gate_margin_thin,
        "below_power_floor": n_val_total_for_flags < FLAGS_BELOW_POWER_FLOOR_N,
    }

    return {
        **verdict,
        "n": {
            "val_total": cv_out["n_val_total"],
            "val_traded": max(
                (per_cand_block[c.name]["simulated_roi"]["n_val_traded"] for c in candidates),
                default=0,
            ),
            "unique_forecast_dates": cv_out["n_unique_forecast_dates"],
        },
        "candidates": per_cand_block,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Production lookup + in-sample reference (Task 17, FR-13)
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_key(city: str, var: str) -> str:
    """Slash-separated cell key for ``weather.emos_enabled_cells`` lookups."""
    return f"{city}/{_VAR_TO_SHORT.get(var, var)}"


def _cell_key_underscore(city: str, var: str) -> str:
    """Underscore-separated cell key for ``weather.excluded_city_vars`` lookups."""
    return f"{city}_{_VAR_TO_SHORT.get(var, var)}"


def resolve_production(
    *,
    city: str,
    var: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the current production calibrator for a (city, var) cell.

    Returns a struct: {kind, artifact_path, artifact_hash, source_config_key}.
    kind ∈ {"isotonic", "emos", "excluded"}.
    """
    weather = config.get("strategies", {}).get("weather", {})
    cell_key = _cell_key(city, var)
    cell_key_us = _cell_key_underscore(city, var)

    excluded = weather.get("excluded_city_vars", []) or []
    if cell_key_us in excluded:
        return {
            "kind": "excluded",
            "artifact_path": None,
            "artifact_hash": None,
            "source_config_key": "weather.excluded_city_vars",
        }

    emos_cells = weather.get("emos_enabled_cells", []) or []
    if cell_key in emos_cells:
        path = "data/emos/coefficients.json"
        return {
            "kind": "emos",
            "artifact_path": path,
            "artifact_hash": _file_sha256(Path(path)),
            "source_config_key": "weather.emos_enabled_cells",
        }

    # Unified isotonic_path default: matches build_in_sample_reference + run_harness
    # call sites that hardcode isotonic_v2.json. With a stripped config (no
    # `weather.isotonic_path` key), production.artifact_path would otherwise
    # silently report the legacy isotonic.json while scoring happens against _v2.
    iso_path = weather.get("isotonic_path", "data/calibration/isotonic_v2.json")
    return {
        "kind": "isotonic",
        "artifact_path": iso_path,
        "artifact_hash": _file_sha256(Path(iso_path)),
        "source_config_key": "weather.isotonic_path",
    }


def build_in_sample_reference(
    *,
    rows: list[FeatureRow],
    iso_v2_path: Path,
    emos_coefficients_path: Path,
) -> dict[str, Any]:
    """Score deployed artifacts in-sample on the full corpus for audit.

    Returns the in_sample_reference column entries with the "LEAKAGE — not a
    verdict" note. Entries whose artifact is in fallback report null values.
    """
    iso_ref = IsoV2DeployedReference()
    iso_state = iso_ref.load(iso_v2_path)
    iso_veto = iso_ref.validate(iso_state)

    emos_ref = EmosDeployedReference()
    emos_state = emos_ref.load(emos_coefficients_path)
    emos_veto = emos_ref.validate(emos_state)

    iso_entry: dict[str, Any] = {"note": "LEAKAGE — not a verdict"}
    emos_entry: dict[str, Any] = {"note": "LEAKAGE — not a verdict"}

    if not iso_veto.ok or not rows:
        iso_entry["logskill_in_sample"] = None
        iso_entry["brier_in_sample"] = None
        if iso_veto.warnings:
            iso_entry["fallback_reason"] = iso_veto.warnings[0]
    else:
        outcomes = [r.outcome for r in rows if r.outcome is not None]
        id_cand = IdentityCandidate()
        id_state = id_cand.fit([])
        id_preds = [id_cand.score(r, id_state) for r in rows if r.outcome is not None]
        iso_preds = [iso_ref.score(r, iso_state) for r in rows if r.outcome is not None]
        iso_entry["logskill_in_sample"] = compute_logskill(iso_preds, id_preds, outcomes)
        iso_entry["brier_in_sample"] = compute_brier(iso_preds, outcomes)

    if not emos_veto.ok or not rows:
        emos_entry["logskill_in_sample"] = None
        emos_entry["brier_in_sample"] = None
        if emos_veto.warnings:
            emos_entry["fallback_reason"] = emos_veto.warnings[0]
    else:
        # EMOS lookup is per-cell — if no cell matches, fallback.
        outcomes = [r.outcome for r in rows if r.outcome is not None]
        id_cand = IdentityCandidate()
        id_state = id_cand.fit([])
        id_preds = [id_cand.score(r, id_state) for r in rows if r.outcome is not None]
        emos_preds = [emos_ref.score(r, emos_state) for r in rows if r.outcome is not None]
        # Filter out NaN scores (cells with no EMOS entry).
        valid = [
            (p, ip, o)
            for p, ip, o in zip(emos_preds, id_preds, outcomes, strict=True)
            if not math.isnan(p)
        ]
        if valid:
            valid_emos = [t[0] for t in valid]
            valid_id = [t[1] for t in valid]
            valid_out = [t[2] for t in valid]
            emos_entry["logskill_in_sample"] = compute_logskill(
                valid_emos, valid_id, valid_out
            )
            emos_entry["brier_in_sample"] = compute_brier(valid_emos, valid_out)
        else:
            emos_entry["logskill_in_sample"] = None
            emos_entry["brier_in_sample"] = None
            emos_entry["fallback_reason"] = "no_emos_coverage"

    return {
        "iso_v2_deployed_reference": iso_entry,
        "emos_deployed_reference": emos_entry,
    }


# ---------------------------------------------------------------------------
# Task 18: Atomic write + schema-invariant validation
# ---------------------------------------------------------------------------


def atomic_write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Atomic write protocol: tempfile in output_path.parent + dual fsync + os.replace.

    Per spec FR-12 and FM-5.
    """
    atomic_write_text(output_path, json.dumps(payload, indent=2), fsync=True)


class SchemaValidationError(ValueError):
    """Schema invariant failure (FM-8). Distinct subclass so the CLI can
    route to exit code 4 by isinstance check instead of brittle substring
    matching on the message."""


def validate_schema(
    payload: dict[str, Any],
    *,
    expected_cell_keys: set[str] | None = None,
) -> None:
    """Enforce per-cell invariants from spec §"Per-cell schema invariants".

    B4 fix: expanded from 3 invariants to the full ~12 listed in spec.
    R2 fix: when ``expected_cell_keys`` is provided AND ``payload["provisional"]``
    is False (full run, not --cell-allowlist), enforce FR-14 14-cell coverage.
    Raises ValueError on any violation; harness CLI translates to exit 4 (FM-8).
    """
    # Top-level invariants.
    if "schema_version" not in payload:
        raise SchemaValidationError("payload: schema_version missing")
    schema_version = payload.get("schema_version")
    if schema_version != 2:
        raise SchemaValidationError(
            f"schema invariant: payload schema_version must be 2, got {schema_version!r} "
            f"(v1 is retired; re-run the harness)"
        )
    if "provisional" not in payload:
        raise SchemaValidationError("payload: provisional flag missing (set False for full runs)")

    cells = payload.get("cells", {})
    valid_verdicts = {
        "winner", "insufficient_evidence", "worse_than_identity",
        "insufficient_n", "insufficient_diversity", "no_separation",
    }
    valid_production_kinds = {"isotonic", "emos", "excluded"}
    valid_roi_statuses = {"ok", "insufficient_data"}

    for key, cell in cells.items():
        verdict = cell.get("verdict")
        if verdict not in valid_verdicts:
            raise SchemaValidationError(
                f"cell {key}: verdict={verdict!r} not in {sorted(valid_verdicts)}"
            )

        # winner_candidate ↔ verdict.
        if verdict == "winner":
            if cell.get("winner_candidate") is None:
                raise SchemaValidationError(
                    f"cell {key}: verdict=winner but winner_candidate is null"
                )
        else:
            if cell.get("winner_candidate") is not None:
                raise SchemaValidationError(
                    f"cell {key}: verdict={verdict!r} but winner_candidate is non-null"
                )

        # production struct shape.
        production = cell.get("production")
        if not isinstance(production, dict):
            raise SchemaValidationError(f"cell {key}: production missing or not a dict")
        prod_kind = production.get("kind")
        if prod_kind not in valid_production_kinds:
            raise SchemaValidationError(
                f"cell {key}: production.kind={prod_kind!r} not in {sorted(valid_production_kinds)}"
            )
        if prod_kind == "isotonic" and not production.get("artifact_path"):
            raise SchemaValidationError(
                f"cell {key}: production.kind=isotonic requires artifact_path"
            )
        if prod_kind == "excluded" and production.get("artifact_path") is not None:
            raise SchemaValidationError(
                f"cell {key}: production.kind=excluded must have artifact_path=null"
            )

        # excluded_in_production ↔ production.kind == excluded.
        excluded_flag = cell.get("excluded_in_production")
        if (prod_kind == "excluded") != bool(excluded_flag):
            raise SchemaValidationError(
                f"cell {key}: excluded_in_production={excluded_flag} but "
                f"production.kind={prod_kind!r}"
            )

        # provisional_in_allowlist only when payload.provisional is True.
        if cell.get("provisional_in_allowlist") and not payload.get("provisional"):
            raise SchemaValidationError(
                f"cell {key}: provisional_in_allowlist=true but payload.provisional=false"
            )

        # flags: gate_margin_thin and below_power_floor must be present and bool (v2).
        cell_flags = cell.get("flags", {})
        for flag_name in ("gate_margin_thin", "below_power_floor"):
            if flag_name not in cell_flags:
                raise SchemaValidationError(
                    f"cell {key}: flags.{flag_name} missing (required in schema_version=2)"
                )
            if not isinstance(cell_flags[flag_name], bool):
                raise SchemaValidationError(
                    f"cell {key}: flags.{flag_name} must be bool, "
                    f"got {type(cell_flags[flag_name])!r}"
                )

        for cname, cblock in cell.get("candidates", {}).items():
            # logskill.p99 required in schema v2.
            ls_block = cblock.get("logskill", {})
            if "p99" not in ls_block:
                raise SchemaValidationError(
                    f"cell {key}/{cname}: logskill.p99 missing (required in schema_version=2)"
                )

            # fold_logskill length 5 when present.
            fl = cblock.get("fold_logskill")
            if fl is not None and len(fl) != 5:
                raise SchemaValidationError(
                    f"cell {key}/{cname}: fold_logskill length {len(fl)} != 5"
                )

            # per_fold_diagnostics: each *_folds field is list[int] ∈ [0, 4].
            diag = cblock.get("per_fold_diagnostics", {})
            for fname, fvals in diag.items():
                if not isinstance(fvals, list):
                    raise SchemaValidationError(
                        f"cell {key}/{cname}: per_fold_diagnostics.{fname} not a list"
                    )
                for v in fvals:
                    if not isinstance(v, int) or v < 0 or v > 4:
                        raise SchemaValidationError(
                            f"cell {key}/{cname}: per_fold_diagnostics.{fname} "
                            f"contains non-int-in-[0,4]: {v!r}"
                        )

            # emos_promotion_gate_decision present iff candidate is emos.
            has_emos_gate = "emos_promotion_gate_decision" in cblock
            if (cname == "emos") != has_emos_gate:
                raise SchemaValidationError(
                    f"cell {key}/{cname}: emos_promotion_gate_decision present={has_emos_gate} "
                    f"but candidate name is {cname!r}"
                )

            # simulated_roi.status enum.
            roi = cblock.get("simulated_roi", {})
            roi_status = roi.get("status")
            if roi_status not in valid_roi_statuses:
                raise SchemaValidationError(
                    f"cell {key}/{cname}: simulated_roi.status={roi_status!r} not in "
                    f"{sorted(valid_roi_statuses)}"
                )

            # roi_ci_lb_informative ↔ CI width when status=ok.
            if roi_status == "ok":
                ci_lb = roi.get("ci_lb")
                ci_ub = roi.get("ci_ub")
                if ci_lb is None or ci_ub is None:
                    raise SchemaValidationError(
                        f"cell {key}/{cname}: simulated_roi.status=ok but ci_lb/ci_ub is null"
                    )
                expected_informative = (ci_ub - ci_lb) < 0.20
                if bool(roi.get("roi_ci_lb_informative")) != expected_informative:
                    raise SchemaValidationError(
                        f"cell {key}/{cname}: roi_ci_lb_informative inconsistent with CI width"
                    )

            # cell_bootstrap_degenerate ↔ mean_rows_per_block < 2.0.
            mrpb = cblock.get("mean_rows_per_block")
            if mrpb is not None:
                expected_degen = mrpb < 2.0
                if bool(cblock.get("cell_bootstrap_degenerate")) != expected_degen:
                    raise SchemaValidationError(
                        f"cell {key}/{cname}: cell_bootstrap_degenerate inconsistent "
                        f"with mean_rows_per_block={mrpb}"
                    )

            # fr8_underpowered ↔ evaluable_folds < 5 (EMOS only).
            if cname == "emos" and has_emos_gate:
                gate = cblock["emos_promotion_gate_decision"]
                eval_folds = gate.get("evaluable_folds")
                expected_underpowered = eval_folds is not None and eval_folds < 5
                if bool(gate.get("fr8_underpowered")) != expected_underpowered:
                    raise SchemaValidationError(
                        f"cell {key}/emos: fr8_underpowered inconsistent with "
                        f"evaluable_folds={eval_folds}"
                    )

        # in_sample_reference always present + has both entries.
        isr = cell.get("in_sample_reference")
        if not isinstance(isr, dict):
            raise SchemaValidationError(f"cell {key}: in_sample_reference missing or not a dict")
        if isr == {}:
            raise SchemaValidationError(f"cell {key}: in_sample_reference is empty {{}}")
        required_refs = {"iso_v2_deployed_reference", "emos_deployed_reference"}
        if set(isr.keys()) != required_refs:
            raise SchemaValidationError(
                f"cell {key}: in_sample_reference keys {set(isr.keys())} != {required_refs}"
            )

    # FR-14 14-cell coverage — R2 fix: enforce here on full runs (not allowlist)
    # so an operator running the CLI by hand fails fast on missing cells. When
    # provisional=True (--cell-allowlist mode), skip this check because the
    # operator deliberately re-ran a subset.
    if expected_cell_keys is not None and not payload.get("provisional"):
        actual = set(cells.keys())
        missing = expected_cell_keys - actual
        unexpected = actual - expected_cell_keys
        if missing or unexpected:
            raise SchemaValidationError(
                f"FR-14 14-cell coverage failure: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )


# ---------------------------------------------------------------------------
# Task 19: Top-level run_harness orchestrator
# ---------------------------------------------------------------------------


# A6 fix: candidate name → production kind translation. winner_candidate is
# named after the candidate class ("iso_v2_global", "emos", "identity") but
# production.kind uses production vocabulary ("isotonic", "emos", "excluded").
# Disagreement = winner_kind != production.kind, not string-compare across
# vocabularies. "identity" winning IS a recommendation different from any
# active-trading production calibrator (recommendation: stop calibrating, run
# raw); R3 fix below treats it as a disagreement on actively-traded cells.
_CANDIDATE_NAME_TO_PRODUCTION_KIND = {
    "identity": "identity",     # not a production kind, but distinct from isotonic/emos
    "iso_v2_global": "isotonic",
    "emos": "emos",
}


def _compute_disagreement(
    *,
    verdict: str,
    winner_candidate: str | None,
    production_kind: str,
) -> bool:
    """R3 fix: flag disagreement on identity-wins and worse_than_identity on trading cells.

    The harness's job is to surface operator-actionable signal. The original
    A6 fix (just winner_kind != production.kind) missed two important cases:

    1. Identity wins on a cell currently running isotonic or emos → operator
       should know "calibration is hurting you on this cell, run raw instead."
    2. worse_than_identity verdict on a cell currently trading → operator should
       know "harness says no calibrator beats raw on this cell, but you trade it."

    Rule: disagreement = True iff (a) harness recommends a calibrator different
    from production's kind, OR (b) harness recommends worse_than_identity on a
    cell that's actively trading (production.kind != "excluded"), OR (c) harness
    recommends winner: identity on an actively-trading cell.

    Symmetric agreement cases:
    - production excluded + verdict insufficient_evidence → agreement (consistent).
    - production excluded + verdict winner: <X> → disagreement (harness says
      this cell is now tradeable, but production excludes it).
    """
    if production_kind == "excluded":
        # Production not trading. Agreement iff verdict is insufficient_evidence /
        # worse_than_identity or any insufficient_* / no_separation state
        # (informational, no actionable change). A winner verdict (other than identity)
        # flags the cell as ready-to-trade — operator should reconsider the exclusion.
        return verdict == "winner" and winner_candidate != "identity"

    # Production actively trading: cases that disagree.
    if verdict == "worse_than_identity":
        return True  # harness says no calibrator beats raw on a cell production trades
    if verdict == "winner" and winner_candidate is not None:
        winner_kind = _CANDIDATE_NAME_TO_PRODUCTION_KIND.get(winner_candidate)
        return winner_kind != production_kind
    # insufficient_evidence / insufficient_n / insufficient_diversity / no_separation:
    # informational, not actionable as disagreement (harness can't make a recommendation).
    return False


def _compute_corpus_hash(snapshots_dir: Path) -> str:
    """A7 fix: hash input file SHAs + settlement_status, not derived FeatureRow fields.

    Per spec A-7: sha256 over sorted tuples (snapshot_date, run_id,
    settlement_status, sha256(forecasts.json), sha256(settlements.json)).
    Catches NWS settlement corrections (value changes that preserve ticker
    + forecast_date — masked by the prior derived-field hash).
    """
    if not snapshots_dir.is_dir():
        return "sha256:no-snapshots-dir"

    entries: list[str] = []
    for day_dir in sorted(snapshots_dir.iterdir()):
        if not day_dir.is_dir() or day_dir.name == "index.jsonl":
            continue
        # Handle both flat <date>/ and nested <date>/<run_id>/ layouts.
        if (day_dir / "weather_forecasts.json").exists():
            run_dirs = [(day_dir, "_flat")]
        else:
            run_dirs = [
                (sub, sub.name) for sub in sorted(day_dir.iterdir()) if sub.is_dir()
            ]
        for run_dir, run_id in run_dirs:
            fc_file = run_dir / "weather_forecasts.json"
            st_file = run_dir / "settlements.json"
            fc_hash = (
                hashlib.sha256(fc_file.read_bytes()).hexdigest() if fc_file.exists() else "missing"
            )
            st_hash = (
                hashlib.sha256(st_file.read_bytes()).hexdigest() if st_file.exists() else "missing"
            )
            # settlement_status derived from the settlements file (best-effort);
            # we hash the file bytes anyway, so the field is informational.
            settlement_status = "unknown"
            if st_file.exists():
                try:
                    st_data = json.loads(st_file.read_text())
                    if isinstance(st_data, list) and st_data:
                        prelim_count = sum(1 for s in st_data if s.get("is_preliminary"))
                        settlement_status = f"prelim_{prelim_count}_of_{len(st_data)}"
                except (json.JSONDecodeError, OSError):
                    pass
            entries.append(f"{day_dir.name}|{run_id}|{settlement_status}|{fc_hash}|{st_hash}")

    if not entries:
        return "sha256:empty-corpus"

    combined = "\n".join(sorted(entries))
    return "sha256:" + hashlib.sha256(combined.encode("utf-8")).hexdigest()


def run_harness(
    *,
    snapshots_dir: Path,
    config_path: Path,
    bias_table_path: Path | None = None,
    cell_allowlist: list[str] | None = None,
    write_output_to: Path | None = None,
    write_diagnostics_to: Path | None = None,
) -> dict[str, Any]:
    """Run the full harness end-to-end.

    Reads config, loads corpus via load_feature_rows, shards by (city, var),
    scores each cell via score_cell, assembles output, validates schema,
    atomically writes if write_output_to is provided. Returns the payload dict.

    B1 fix: when cell_allowlist is non-None AND write_output_to points at an
    existing JSON, read it first and carry forward non-allowlisted cells
    with provisional_in_allowlist=true (read-modify-write per E-16). Sets
    payload.provisional=true at top level.
    """
    config = yaml.safe_load(config_path.read_text())
    weather = config.get("strategies", {}).get("weather", {})
    ensemble_weights = weather.get("ensemble_weights") or {"gfs_seamless": 1.0}
    spread_inflation_overrides = weather.get("spread_inflation_overrides", {})

    logger.info("harness: loading feature rows from %s", snapshots_dir)
    all_rows = load_feature_rows(
        snapshots_dir=snapshots_dir,
        bias_table_path=bias_table_path,
        ensemble_weights=ensemble_weights,
        spread_inflation_overrides=spread_inflation_overrides,
    )
    logger.info("harness: loaded %d feature rows", len(all_rows))

    # Restrict to config-declared cities so historical snapshot expansions
    # (e.g. BOS / PHIL collected briefly in 2026-05-12+) don't violate the
    # FR-14 14-cell coverage invariant. The config city list is the canonical
    # production trading universe; off-list cells are corpus residue.
    _config_cities_for_filter = weather.get("cities")
    if _config_cities_for_filter:
        _allowed = set(_config_cities_for_filter)
        all_rows = [r for r in all_rows if r.city in _allowed]

    # Shard by (city, var).
    by_cell: dict[tuple[str, str], list[FeatureRow]] = {}
    for r in all_rows:
        by_cell.setdefault((r.city, r.var), []).append(r)

    # B1: in allowlist mode, load prior JSON for carry-forward.
    prior_payload: dict[str, Any] | None = None
    if cell_allowlist and write_output_to and write_output_to.exists():
        try:
            prior_payload = json.loads(write_output_to.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"--cell-allowlist mode requires the existing output JSON at "
                f"{write_output_to} to be parseable (read-modify-write). "
                f"Refusing to silently produce a partial file. Error: {exc}"
            ) from exc

    cell_entries: dict[str, Any] = {}

    # First, populate carry-forward cells from prior JSON if in allowlist mode.
    if prior_payload is not None:
        for key, prior_cell in (prior_payload.get("cells") or {}).items():
            if cell_allowlist and key in cell_allowlist:
                continue  # this cell will be re-scored below
            prior_cell["provisional_in_allowlist"] = True
            cell_entries[key] = prior_cell

    # Then score the (re-)requested cells.
    _cells_to_score = [
        (city, var, rows)
        for (city, var), rows in by_cell.items()
        if not cell_allowlist or _cell_key(city, var) in cell_allowlist
    ]
    logger.info("harness: scoring %d cell(s)", len(_cells_to_score))
    for _idx, (city, var, rows) in enumerate(_cells_to_score, start=1):
        key = _cell_key(city, var)
        logger.info(
            "harness: [%d/%d] scoring %s (n_rows=%d)",
            _idx, len(_cells_to_score), key, len(rows),
        )

        cell_result = score_cell(
            city=city, var=var, rows=rows,
            candidates=[IdentityCandidate(), IsoV2GlobalCandidate(), EmosCandidate()],
        )
        cell_result["production"] = resolve_production(city=city, var=var, config=config)

        # R3 fix: widened disagreement detection — fires on identity-wins,
        # worse_than_identity on actively-trading cells, and winners on excluded cells.
        # See _compute_disagreement docstring for the symmetric agreement rules.
        cell_result["disagreement_with_production"] = _compute_disagreement(
            verdict=cell_result["verdict"],
            winner_candidate=cell_result.get("winner_candidate"),
            production_kind=cell_result["production"]["kind"],
        )
        cell_result["excluded_in_production"] = (
            cell_result["production"]["kind"] == "excluded"
        )
        cell_result["in_sample_reference"] = build_in_sample_reference(
            rows=rows,
            iso_v2_path=Path(weather.get("isotonic_path", "data/calibration/isotonic_v2.json")),
            emos_coefficients_path=Path("data/emos/coefficients.json"),
        )
        cell_entries[key] = cell_result

    # FR-14 invariant: synthesise insufficient_n placeholder entries for any
    # config-declared cell that produced zero feature rows (e.g. SEA/low when
    # settlements never aligned with markets in the corpus window). Without
    # this, the by_cell loop above skips the cell entirely and the FR-14
    # validator fails on `missing=...`. Skip in --cell-allowlist mode since
    # the operator deliberately picked a subset.
    if not cell_allowlist:
        _config_cities_for_pad = weather.get("cities") or []
        _seen_keys = set(cell_entries.keys())
        for _city in _config_cities_for_pad:
            for _short, _var_long in (
                ("high", "temperature_2m_max"),
                ("low", "temperature_2m_min"),
            ):
                _key = f"{_city}/{_short}"
                if _key in _seen_keys:
                    continue
                _placeholder: dict[str, Any] = {
                    "verdict": "insufficient_n",
                    "verdict_message": "no feature rows in corpus for this cell",
                    "winner_candidate": None,
                    "n": {"val_total": 0, "val_traded": 0, "unique_forecast_dates": 0},
                    "candidates": {},
                    "flags": {
                        "pnl_logskill_disagreement": False,
                        "metric_disagreement": False,
                        "gate_margin_thin": False,
                        "below_power_floor": True,
                    },
                }
                _placeholder["production"] = resolve_production(
                    city=_city, var=_var_long, config=config,
                )
                _placeholder["disagreement_with_production"] = _compute_disagreement(
                    verdict=_placeholder["verdict"],
                    winner_candidate=_placeholder["winner_candidate"],
                    production_kind=_placeholder["production"]["kind"],
                )
                _placeholder["excluded_in_production"] = (
                    _placeholder["production"]["kind"] == "excluded"
                )
                _placeholder["in_sample_reference"] = build_in_sample_reference(
                    rows=[],
                    iso_v2_path=Path(weather.get(
                        "isotonic_path", "data/calibration/isotonic_v2.json",
                    )),
                    emos_coefficients_path=Path("data/emos/coefficients.json"),
                )
                cell_entries[_key] = _placeholder

    payload = {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run_timestamp": datetime.now(UTC).isoformat(),
        "provisional": cell_allowlist is not None,
        "corpus": {
            "snapshots_dir": str(snapshots_dir),
            "date_range": [
                min(r.forecast_date for r in all_rows) if all_rows else None,
                max(r.forecast_date for r in all_rows) if all_rows else None,
            ],
            "n_days": len({r.forecast_date for r in all_rows}),
            # A7 fix: hash input file SHAs, not derived ticker+forecast_date.
            "corpus_hash": _compute_corpus_hash(snapshots_dir),
        },
        "config": {
            "candidates": ["identity", "iso_v2_global", "emos"],
            "audit_candidates": ["iso_v2_deployed_reference", "emos_deployed_reference"],
            "partition_rule": "cv_fold_iter:5fold:expanding_window",
            "bracket_width": 2.0,
            "logskill_eps": LOGSKILL_EPS,
            "bootstrap_iter": BOOTSTRAP_ITER,
            "bootstrap_block": ["forecast_date", "lead"],
            "kelly_knobs": {
                "kelly_fraction": _SIM_KELLY_FRACTION, "max_pct": 0.05, "bankroll": 1000.0,
            },
            "trading_filters": {
                "block_raw_prob_range": list(_SIM_BLOCK_RAW_PROB_RANGE),
                "min_calibrated_prob": _SIM_MIN_CALIBRATED_PROB,
            },
            "winner_gate": {
                "logskill_lb_min": WINNER_GATE_LOGSKILL_LB_MIN,
                "fold_consistency_min": WINNER_GATE_FOLD_CONSISTENCY_MIN,
                "n_clipped_pct_max": WINNER_GATE_N_CLIPPED_PCT_MAX,
            },
            "emos_promotion_gate": {
                "logskill_lb_vs_iso_min": EMOS_GATE_LOGSKILL_LB_VS_ISO_MIN,
                "bss_lb_vs_iso_min": EMOS_GATE_BSS_LB_VS_ISO_MIN,
            },
            "tiebreaker": {
                "logskill_lb_band": TIEBREAKER_LOGSKILL_LB_BAND,
            },
        },
        "cells": cell_entries,
    }

    # R2 fix: enforce FR-14 14-cell coverage on full runs.
    # Cities sourced from config; if config doesn't expose an explicit list,
    # fall back to the cells observed in the corpus (keeps the test honest if
    # the config schema evolves).
    config_cities = weather.get("cities")
    if not config_cities:
        config_cities = sorted({r.city for r in all_rows})
    expected_cell_keys = {
        f"{city}/{short}"
        for city in config_cities
        for short in ("high", "low")
    }
    validate_schema(payload, expected_cell_keys=expected_cell_keys)

    if write_output_to is not None:
        atomic_write_json(payload, write_output_to)

    return payload


# ---------------------------------------------------------------------------
# Public surface (locked at Task 2 — see test_all_public_surface_locked)
# ---------------------------------------------------------------------------


__all__ = [
    # Constants
    "HARNESS_VERSION",
    "SCHEMA_VERSION",
    "LOGSKILL_EPS",
    "BOOTSTRAP_ITER",
    "WINNER_GATE_LOGSKILL_LB_MIN",
    "WINNER_GATE_FOLD_CONSISTENCY_MIN",
    "WINNER_GATE_N_CLIPPED_PCT_MAX",
    "EMOS_GATE_LOGSKILL_LB_VS_ISO_MIN",
    "EMOS_GATE_BSS_LB_VS_ISO_MIN",
    "TIEBREAKER_LOGSKILL_LB_BAND",
    "FR11_2_N_VAL_TOTAL_MIN",
    "FLAGS_BELOW_POWER_FLOOR_N",
    # Types
    "FeatureRow",
    "FitState",
    "VetoResult",
    "CandidateResult",
    "CalibratorCandidate",
    "LoadArtifactCandidate",
    # Candidates (Tasks 3-7)
    "IdentityCandidate",
    "IsoV2GlobalCandidate",
    "EmosCandidate",
    "IsoV2DeployedReference",
    "EmosDeployedReference",
    # Data loader (Task 8)
    "load_feature_rows",
    # CV + scoring (Tasks 9, 16)
    "cv_score_cell",
    "score_cell",
    # Metrics (Tasks 10-12)
    "compute_logskill",
    "compute_brier",
    "compute_bss",
    "count_clipped",
    "simulate_stage1_roi",
    # Bootstrap (Task 13)
    "block_bootstrap_ci",
    "block_bootstrap_ratio_ci",
    # Eligibility + verdict (Tasks 14, 15)
    "compute_eligibility",
    "compute_verdict",
    # Output assembly (Tasks 17-19) + holistic cleanup (Task 26)
    "resolve_production",
    "build_in_sample_reference",
    "validate_schema",
    "SchemaValidationError",
    "atomic_write_json",
    "run_harness",
]
