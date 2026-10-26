"""Shadow A/B replay: compare a candidate EMOS coefficients file against
production on `state/eval_audit.jsonl` × `state/forecasts_raw.jsonl` × snapshots.

For each cell with enough settled shadow rows, recomputes EMOS probability with
both prod and candidate coefficient sets (using the same pure function used at
runtime), computes per-cell Brier + BSS_candidate_vs_prod with a percentile
bootstrap CI, and prints a markdown decision table.

Usage::

    PYTHONPATH=. uv run python scripts/compare_emos_coefficients.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from python.analytics.emos_fit import (
    BOOTSTRAP_RESAMPLE_DATE_CLUSTER,
    BOOTSTRAP_RESAMPLE_ROW,
    MIN_CLUSTER_DATES,
    date_cluster_rows,
)
from python.analytics.outcomes import derive_outcome_yes
from python.data.emos import EmosCoefficients, EmosTable, emos_probability
from scripts._calibration_common import prefer_latest_settlement

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure functions — covered by tests/test_compare_emos_coefficients.py.
# --------------------------------------------------------------------------- #


def resolve_outcome(
    strike_type: str,
    floor: float | None,
    cap: float | None,
    observed: float,
) -> int:
    """Kalshi-convention outcome resolution.

    Delegates to ``python.analytics.outcomes.derive_outcome_yes`` (canonical
    single source of truth — see that module's docstring for the 42/42
    right-boundary audit). Returns 1 for YES, 0 for NO. Raises ValueError if
    the strike parameters are inconsistent with ``strike_type``.

    Prior to 2026-05-19 this function held its own inline rule with
    right-exclusive ``between`` (``floor <= obs < cap``), which mis-labelled
    every ``observed == cap`` case as NO and inflated EMOS-promotion BSS
    against the wrong ground truth.
    """
    yes = derive_outcome_yes(strike_type, floor, cap, observed)
    if yes is None:
        raise ValueError(
            f"cannot resolve outcome: strike_type={strike_type!r}, "
            f"floor={floor}, cap={cap}"
        )
    return 1 if yes else 0


def _paired_resample_means(
    prod: np.ndarray,
    cand: np.ndarray,
    n_iters: int,
    seed: int,
    dates: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-iteration bootstrap means of `prod` and `cand` on shared resamples.

    With `dates=None`, resamples rows. Otherwise resamples whole dates with
    replacement and takes every row of each drawn date: audit rows for one
    forecast_date share a settled observation, so row resampling understates
    the CI. Both score arrays use the same draw, so BSS and LogSkill callers
    sharing a seed also share resamples. Raises ValueError on null dates or
    fewer than 2 unique dates; callers guard with `MIN_CLUSTER_DATES`.
    """
    n = len(prod)
    rng = np.random.default_rng(seed)
    if dates is None:
        idx = rng.integers(0, n, size=(n_iters, n))
        return prod[idx].mean(axis=1), cand[idx].mean(axis=1)
    if len(dates) != n:
        raise ValueError("dates must match the score arrays in length")
    date_rows = date_cluster_rows(dates)
    draws = rng.integers(0, len(date_rows), size=(n_iters, len(date_rows)))
    prod_means = np.empty(n_iters)
    cand_means = np.empty(n_iters)
    for i in range(n_iters):
        idx = np.concatenate([date_rows[j] for j in draws[i]])
        prod_means[i] = prod[idx].mean()
        cand_means[i] = cand[idx].mean()
    return prod_means, cand_means


def bootstrap_bss_ci(
    sq_err_prod: list[float],
    sq_err_cand: list[float],
    n_iters: int,
    seed: int,
    ci: float = 0.95,
    dates: Sequence[Any] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap for BSS = 1 - mean(sq_err_cand)/mean(sq_err_prod).

    Resamples row indices with replacement so prod/cand pairs stay correlated;
    `dates` switches to whole-date resampling (see `_paired_resample_means`).
    Returns (lower, upper) at (1-ci)/2 and 1-(1-ci)/2 percentiles.
    """
    if len(sq_err_prod) != len(sq_err_cand):
        raise ValueError("sq_err arrays must have equal length")
    n = len(sq_err_prod)
    if n == 0:
        return 0.0, 0.0
    prod = np.asarray(sq_err_prod, dtype=float)
    cand = np.asarray(sq_err_cand, dtype=float)
    prod_means, cand_means = _paired_resample_means(prod, cand, n_iters, seed, dates)
    # Where prod_mean is zero, BSS is undefined; treat as 0.
    bss = np.where(prod_means > 0, 1.0 - cand_means / prod_means, 0.0)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(bss, alpha))
    hi = float(np.quantile(bss, 1.0 - alpha))
    return lo, hi


def bootstrap_log_skill_ci(
    nll_prod: list[float],
    nll_cand: list[float],
    n_iters: int,
    seed: int,
    ci: float = 0.95,
    dates: Sequence[Any] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap for LogSkill = 1 - mean(nll_cand)/mean(nll_prod).

    LogSkill is the logarithmic scoring rule analogue of BSS. Both are proper,
    but log-score is Kelly-aligned (equivalent up to constants to expected
    log-growth of a fractional Kelly bettor) and penalises tail mis-calibration
    linearly-but-unboundedly, where squared-error BSS penalises it quadratically.

    Resamples row indices with replacement so prod/cand pairs stay correlated;
    `dates` switches to whole-date resampling (see `_paired_resample_means`).
    Returns (lower, upper) at (1-ci)/2 and 1-(1-ci)/2 percentiles.
    """
    if len(nll_prod) != len(nll_cand):
        raise ValueError("nll arrays must have equal length")
    n = len(nll_prod)
    if n == 0:
        return 0.0, 0.0
    prod = np.asarray(nll_prod, dtype=float)
    cand = np.asarray(nll_cand, dtype=float)
    prod_means, cand_means = _paired_resample_means(prod, cand, n_iters, seed, dates)
    skill = np.where(prod_means > 0, 1.0 - cand_means / prod_means, 0.0)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(skill, alpha))
    hi = float(np.quantile(skill, 1.0 - alpha))
    return lo, hi


def compute_mu_shift_drift(
    ens_means: list[float],
    *,
    prod_a: float,
    prod_b: float,
    cand_a: float,
    cand_b: float,
) -> float:
    """Median absolute per-row drift between (a + (b-1)·mean) computed on prod vs candidate."""
    if not ens_means:
        return 0.0
    drifts: list[float] = []
    for m in ens_means:
        mu_shift_prod = prod_a + (prod_b - 1.0) * m
        mu_shift_cand = cand_a + (cand_b - 1.0) * m
        drifts.append(abs(mu_shift_cand - mu_shift_prod))
    return float(statistics.median(drifts))


# Non-overridable absolute-reject warnings. These reflect structural
# incompatibilities (predictive σ vs Kalshi bracket geometry) — not fit
# uncertainty (which `unstable_d` covers and operators can override via
# `data/emos/overrides.json`). If the candidate carries any of these, the cell
# is rejected even when prod also carries the same warning.
_NON_OVERRIDABLE_WARNINGS = frozenset({"incompatible_sigma"})


def decide_cell(
    *,
    n: int,
    n_min: int,
    bss_point: float,
    bss_ci_lower: float,
    mu_drift: float,
    mu_drift_max: float,
    has_new_warnings: bool,
    bss_ci_lower_floor: float = -0.05,
    cand_warnings: frozenset[str] | set[str] | None = None,
    log_skill_point: float | None = None,
    log_skill_ci_lower: float | None = None,
    log_skill_ci_lower_floor: float = -0.02,
    n_dates: int | None = None,
) -> str:
    """Promotion decision: PROMOTE | KEEP_PROD | INSUFFICIENT_N.

    Gate (all required for PROMOTE):
      - n >= n_min
      - n_dates is None or n_dates >= MIN_CLUSTER_DATES (pass the unique-date
        count in cluster mode only; the CI floors are negative, so a
        degenerate CI would otherwise pass)
      - bss_point >= 0  (candidate at least matches prod on settled shadow rows)
      - bss_ci_lower >= bss_ci_lower_floor  (within bootstrap noise band)
      - mu_drift <= mu_drift_max  (coefficient shift bounded)
      - has_new_warnings is False  (no new CI warnings introduced)
      - cand_warnings ∩ _NON_OVERRIDABLE_WARNINGS is empty (e.g.,
        `incompatible_sigma` — non-overridable even if prod has it too).
      - When provided: log_skill_point >= 0 AND log_skill_ci_lower >=
        log_skill_ci_lower_floor. LogSkill is the Kelly-aligned analogue of
        BSS (proper, knob-invariant) and catches tail mis-calibration that
        squared-error BSS under-weights — the LAX/high failure mode.
    """
    if n < n_min:
        return "INSUFFICIENT_N"
    if n_dates is not None and n_dates < MIN_CLUSTER_DATES:
        return "INSUFFICIENT_N"
    if cand_warnings and (set(cand_warnings) & _NON_OVERRIDABLE_WARNINGS):
        return "KEEP_PROD"
    if has_new_warnings:
        return "KEEP_PROD"
    if mu_drift > mu_drift_max:
        return "KEEP_PROD"
    if bss_point < 0.0:
        return "KEEP_PROD"
    if bss_ci_lower < bss_ci_lower_floor:
        return "KEEP_PROD"
    if log_skill_point is not None and log_skill_point < 0.0:
        return "KEEP_PROD"
    if log_skill_ci_lower is not None and log_skill_ci_lower < log_skill_ci_lower_floor:
        return "KEEP_PROD"
    return "PROMOTE"


def compute_squared_errors_per_cell(
    rows: Iterable[dict[str, Any]],
    coeffs_table: dict[str, EmosCoefficients],
) -> dict[str, list[float]]:
    """Per-cell list of `(emos_prob - outcome)^2` for each settled row.

    Rows whose `(city, var)` cell is not in `coeffs_table` are skipped silently
    (no fit for that cell). Rows are expected to carry: city, var, strike_type,
    floor, cap, ens_mean_raw, ens_std_raw, observed.
    """
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        cell = f"{r['city']}/{r['var']}"
        coeffs = coeffs_table.get(cell)
        if coeffs is None:
            continue
        prob, _, _ = emos_probability(
            coeffs,
            ens_mean=float(r["ens_mean_raw"]),
            ens_std=float(r["ens_std_raw"]),
            strike_type=r["strike_type"],
            floor=r["floor"],
            cap=r["cap"],
        )
        outcome = resolve_outcome(
            r["strike_type"], r.get("floor"), r.get("cap"), float(r["observed"])
        )
        out[cell].append((prob - outcome) ** 2)
    return dict(out)


_LOG_LOSS_EPS = 1e-9


def compute_log_losses_per_cell(
    rows: Iterable[dict[str, Any]],
    coeffs_table: dict[str, EmosCoefficients],
    eps: float = _LOG_LOSS_EPS,
) -> dict[str, list[float]]:
    """Per-cell list of binary cross-entropy values: -[y·log(p) + (1-y)·log(1-p)].

    Companion to compute_squared_errors_per_cell for the logarithmic scoring
    rule. Same row contract; same silent-skip behaviour for cells missing
    from coeffs_table. `eps` clamps `p` away from {0, 1} so log is finite —
    EMOS Gaussian-CDF probabilities saturate to ±0 in the tails which would
    otherwise crash on log(0). NaN is rejected explicitly so the clip never
    silently absorbs an upstream bug (the audit-2026-05-23 review flagged
    that clip(NaN, ...) is implementation-dependent).
    """
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        cell = f"{r['city']}/{r['var']}"
        coeffs = coeffs_table.get(cell)
        if coeffs is None:
            continue
        prob, _, _ = emos_probability(
            coeffs,
            ens_mean=float(r["ens_mean_raw"]),
            ens_std=float(r["ens_std_raw"]),
            strike_type=r["strike_type"],
            floor=r["floor"],
            cap=r["cap"],
        )
        if not math.isfinite(prob):
            raise ValueError(
                f"emos_probability returned non-finite value {prob!r} for cell={cell} "
                f"row={r!r}"
            )
        outcome = resolve_outcome(
            r["strike_type"], r.get("floor"), r.get("cap"), float(r["observed"])
        )
        p = min(max(prob, eps), 1.0 - eps)
        nll = -(outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p))
        out[cell].append(nll)
    return dict(out)


def load_forecasts_raw_index(
    path: Path,
) -> dict[tuple[str, str, str, str], tuple[float, float]]:
    """Index `state/forecasts_raw.jsonl` (+ rotated siblings) by
    (city, var, forecast_date, scan_id).

    Rotation: the writer emits `state/forecasts_raw.jsonl` (active) and
    rotated siblings `state/forecasts_raw-YYYY-MM-DD.jsonl`. Both are
    concatenated into one index.

    Returns {key: (ens_mean_raw_ddof1, ens_std_raw_ddof1)}. Rows missing
    raw stats are skipped silently.
    """
    p = Path(path)
    paths: list[Path] = [p] if p.exists() else []
    # Rotated siblings live alongside the active file, named with the date
    # in the stem (e.g., forecasts_raw-2026-05-07.jsonl).
    if p.parent.exists():
        stem = p.stem  # forecasts_raw
        paths.extend(sorted(p.parent.glob(f"{stem}-*.jsonl")))

    idx: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for fpath in paths:
        with fpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                city = r.get("city")
                var = r.get("var")
                fd = r.get("forecast_date")
                scan_id = r.get("scan_id")
                mean = r.get("ens_mean_raw_ddof1") or r.get("gfs_ens_mean_raw_ddof1")
                std = r.get("ens_std_raw_ddof1") or r.get("gfs_ens_std_raw_ddof1")
                if None in (city, var, fd, scan_id, mean, std):
                    continue
                idx[(city, var, str(fd), str(scan_id))] = (float(mean), float(std))
    return idx


# --------------------------------------------------------------------------- #
# CLI orchestration — composes the pure functions above into the full pipeline.
# --------------------------------------------------------------------------- #


def _iter_audit_rows(
    path: Path,
    start: dt.date | None,
    end: dt.date | None,
) -> Iterator[dict[str, Any]]:
    """Stream eval_audit.jsonl + rotated siblings, yielding rows with emos_prob_raw
    populated and within window.

    Rotation: active file is `eval_audit.jsonl`; rotated siblings are
    `eval_audit.jsonl.YYYY-MM-DD`. Both are concatenated.
    """
    p = Path(path)
    paths: list[Path] = [p] if p.exists() else []
    if p.parent.exists():
        paths.extend(sorted(p.parent.glob(f"{p.name}.*")))

    for fpath in paths:
        with fpath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("emos_prob_raw") is None:
                    continue
                fd_str = r.get("forecast_date")
                if not fd_str:
                    continue
                try:
                    fd = dt.date.fromisoformat(fd_str)
                except ValueError:
                    continue
                if start is not None and fd < start:
                    continue
                if end is not None and fd > end:
                    continue
                yield r


def _load_settlements_index(
    snapshots_dir: Path, dates: set[dt.date]
) -> dict[tuple[str, str, str], tuple[float, str]]:
    """Build a (city, var, observation_date) → (observed_temp, status) index.

    Settlement records use schema: {city, observation_date, high_f, low_f,
    is_preliminary, ...}. We expand each row into two index entries — one for
    var="high" (high_f) and one for var="low" (low_f).

    A single ``merged`` dict is built across ALL snapshot directories in the
    input set so ``prefer_latest_settlement`` arbitrates every (city,
    observation_date) tuple, including the cross-snapshot-day case where a
    final issued on day T+1 must beat a preliminary issued on day T. Building
    per-day merged dicts and then expanding into ``out`` via last-write-wins
    was the original bug (#2 in the 2026-05-11 audit): set iteration over
    ``dates`` is hash-randomized, so per-run results were non-deterministic.
    """
    snap_dates_sorted = sorted(dates)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for d in snap_dates_sorted:
        snap_day = snapshots_dir / d.isoformat()
        if not snap_day.exists():
            continue
        for run_dir in sorted(snap_day.iterdir()):
            if not run_dir.is_dir():
                continue
            settle_path = run_dir / "settlements.json"
            if not settle_path.exists():
                continue
            try:
                data = json.loads(settle_path.read_text())
            except json.JSONDecodeError:
                continue
            for s in data:
                city = s.get("city")
                obs_date = s.get("observation_date")
                if not (city and obs_date):
                    continue
                key = (city, obs_date)
                existing = merged.get(key)
                # prefer_latest_settlement(existing, new) -> True if new replaces existing
                if existing is None or prefer_latest_settlement(existing, s):
                    merged[key] = s

    out: dict[tuple[str, str, str], tuple[float, str]] = {}
    for (city, obs_date), s in merged.items():
        status = "preliminary" if s.get("is_preliminary") else "final"
        high_f = s.get("high_f")
        low_f = s.get("low_f")
        if high_f is not None:
            out[(city, "high", obs_date)] = (float(high_f), status)
        if low_f is not None:
            out[(city, "low", obs_date)] = (float(low_f), status)
    return out


_STRIKE_TYPE_MAP = {"above": "greater", "below": "less", "between": "between"}


# EMOS dispatch went live with the Stage A cutover (2026-05-11).
# Pre-cutover audit rows with emos_prob_raw populated had isotonic as their
# actual live calibrator, but live_calibrator_kind hadn't been added to the
# schema yet — so it shows up absent on those rows. The Stage B comparison
# treats absent-kind on pre-cutover rows as "isotonic fired" (which it was).
EMOS_CUTOVER_DATE = dt.date.fromisoformat("2026-05-11")


def _is_isotonic_baseline_row(r: dict[str, Any]) -> bool:
    """True if isotonic was the live calibrator for this audit row.

    Captures both schemas:
    - Post-cutover: ``live_calibrator_kind in {"isotonic", "isotonic_fallback"}``.
    - Pre-cutover: ``live_calibrator_kind`` absent AND
      ``forecast_date < EMOS_CUTOVER_DATE`` — EMOS dispatch did not exist yet,
      so isotonic was the only calibrator that could have fired on a row that
      reached the EMOS shadow block. (The row reaches Stage B only if
      ``emos_prob_raw is not None``, gated by ``_iter_audit_rows`` upstream;
      that implies the EMOS block ran, which implies the live calibrator was
      already chosen before the block — pre-cutover that calibrator was
      always isotonic.)

    Excludes:
    - ``"emos_raw"``: EMOS was live; including would double-count candidate
      EMOS against itself.
    - ``"identity"``: calibrator object was None at construction time
      (backtest / identity passthrough). In the normal live pipeline these
      rows already have ``emos_prob_raw is None`` so ``_iter_audit_rows``
      filters them upstream; this branch is defensive in case a future path
      lets them through.
    - Post-cutover rows with missing kind: unexpected schema gap; treat as
      unknown rather than guess.
    - 2026-05-11 rows with missing kind: conservatively dropped. Such rows
      *might* be from scans before the deploy that added the field that day (isotonic
      definitely fired) or from a schema bug post-deploy. We can't tell,
      and the at-most-handful-of-rows impact is negligible against an ~89k
      shadow corpus.

    Uses ``dt.date`` for the comparison so a stringified date with a time
    component or alternate format raises ``ValueError`` and returns False
    instead of silently lexicographing wrong.
    """
    kind = r.get("live_calibrator_kind")
    if kind in ("isotonic", "isotonic_fallback"):
        return True
    if kind is None:
        try:
            fd = dt.date.fromisoformat(str(r.get("forecast_date") or ""))
        except ValueError:
            return False
        return fd < EMOS_CUTOVER_DATE
    return False


def _join_settled_rows(
    audit_rows: Iterable[dict[str, Any]],
    forecasts_raw_idx: dict[tuple[str, str, str, str], tuple[float, float]],
    settlements: dict[tuple[str, str, str], tuple[float, str]],
    *,
    final_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join audit + raw + settlements; returns enriched rows and join-loss counters.

    Maps audit strike_type ("above"/"below"/"between") to EMOS canonical form
    ("greater"/"less"/"between") so downstream code can call emos_probability.
    """
    counters: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for r in audit_rows:
        counters["audit_total"] += 1
        city = r.get("city")
        var = r.get("var")  # "high" / "low"
        fd_str = r.get("forecast_date")
        scan_id = r.get("scan_id")
        if None in (city, var, fd_str, scan_id):
            counters["dropped_missing_key"] += 1
            continue
        raw = forecasts_raw_idx.get((city, var, str(fd_str), str(scan_id)))
        if raw is None:
            counters["dropped_no_raw_stats"] += 1
            continue
        settle = settlements.get((city, var, str(fd_str)))
        if settle is None:
            counters["dropped_no_settlement"] += 1
            continue
        observed, status = settle
        if final_only and status != "final":
            counters["dropped_preliminary"] += 1
            continue
        raw_strike = r.get("strike_type")
        strike_type = _STRIKE_TYPE_MAP.get(raw_strike, raw_strike)
        if strike_type not in ("greater", "less", "between"):
            counters["dropped_bad_strike_type"] += 1
            continue
        out.append({
            "city": city,
            "var": var,
            "forecast_date": str(fd_str),
            "scan_id": scan_id,
            "strike_type": strike_type,
            "floor": r.get("floor"),
            "cap": r.get("cap"),
            "ens_mean_raw": raw[0],
            "ens_std_raw": raw[1],
            "observed": observed,
            "settlement_status": status,
            "emos_prob_raw": r.get("emos_prob_raw"),
            "calibrated_prob": r.get("calibrated_prob"),
            "live_calibrator_kind": r.get("live_calibrator_kind"),
            "ticker": r.get("ticker"),
        })
        counters["joined"] += 1
    return out, dict(counters)


def _table_to_dict(table: EmosTable) -> dict[str, EmosCoefficients]:
    """Flatten an EmosTable to a {cell: EmosCoefficients} dict."""
    return dict(table.cells)


def _assert_bit_for_bit_replay(
    rows: list[dict[str, Any]],
    prod_table: dict[str, EmosCoefficients],
    tol: float = 1e-9,
) -> int:
    """For live-EMOS rows, replaying with prod coeffs must reproduce logged calibrated_prob."""
    checked = 0
    for r in rows:
        if r.get("live_calibrator_kind") != "emos_raw":
            continue
        cell = f"{r['city']}/{r['var']}"
        coeffs = prod_table.get(cell)
        if coeffs is None:
            continue
        recomputed, _, _ = emos_probability(
            coeffs,
            ens_mean=float(r["ens_mean_raw"]),
            ens_std=float(r["ens_std_raw"]),
            strike_type=r["strike_type"],
            floor=r["floor"],
            cap=r["cap"],
        )
        logged = float(r["calibrated_prob"])
        if abs(recomputed - logged) > tol:
            raise AssertionError(
                f"bit-for-bit replay mismatch for {cell} ticker={r.get('ticker')!r}: "
                f"recomputed={recomputed:.12f} logged={logged:.12f} Δ={abs(recomputed - logged):.2e}"
            )
        checked += 1
    return checked


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prod-coeffs", type=Path, default=Path("data/emos/coefficients.prod.bak.json"))
    p.add_argument(
        "--candidate-coeffs", type=Path, default=Path("data/emos/candidate/coefficients.json")
    )
    p.add_argument(
        "--candidate-diagnostics",
        type=Path,
        default=Path("data/emos/candidate/coefficients.diagnostics.json"),
    )
    p.add_argument(
        "--prod-diagnostics",
        type=Path,
        default=Path("data/emos/coefficients.diagnostics.prod.bak.json"),
    )
    p.add_argument("--eval-audit", type=Path, default=Path("state/eval_audit.jsonl"))
    p.add_argument("--forecasts-raw", type=Path, default=Path("state/forecasts_raw.jsonl"))
    p.add_argument("--snapshots-dir", type=Path, default=Path("data/snapshots"))
    p.add_argument("--start-date", type=dt.date.fromisoformat, default=None)
    p.add_argument("--end-date", type=dt.date.fromisoformat, default=None)
    p.add_argument("--lookback-days", type=int, default=30)
    p.add_argument("--n-min", type=int, default=15)
    p.add_argument("--mu-drift-max", type=float, default=1.0)
    p.add_argument("--bss-ci-lower-floor", type=float, default=-0.05)
    # Default tightened to -0.02 (vs BSS's -0.05) per the 2026-05-23
    # calibration audit: at EMOS's typical p-regime (median ~0.07, p25 ~0.04
    # on the live shadow corpus), a -0.05 LogSkill floor corresponds to a
    # roughly -0.01 BSS floor — i.e. functionally MORE permissive than BSS's
    # own -0.05. The 5/48 BSS/LogSkill bootstrap disagreements on the audit
    # corpus all went LogSkill-greenlight / BSS-reject, confirming that
    # -0.05 left LogSkill with zero independent veto power. -0.02 matches
    # BSS's effective bite at the EMOS p-regime; revisit when n_shadow ≥ 30
    # per cell at refit.
    p.add_argument("--log-skill-ci-lower-floor", type=float, default=-0.02)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument(
        "--cluster-by-date",
        action="store_true",
        help=(
            "Resample whole forecast_dates instead of rows (row CIs are too narrow). "
            "Required before any EMOS re-promotion; see docs/calibration-runbook.md. "
            f"Cells with fewer than {MIN_CLUSTER_DATES} dates are scored INSUFFICIENT_N."
        ),
    )
    p.add_argument(
        "--live-cells",
        nargs="+",
        default=["MIA/high", "NYC/low"],
        help="Cells whose decision drives the overall swap recommendation.",
    )
    p.add_argument("--final-only", action="store_true", default=True)
    p.add_argument("--include-preliminary", dest="final_only", action="store_false")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _stage_b_decision(
    *,
    n: int,
    n_min_iso: int,
    bss_vs_iso_point: float,
    bss_vs_iso_ci_lower: float,
    has_new_warnings: bool,
    cand_warnings: frozenset[str] | set[str] | None = None,
    log_skill_vs_iso_point: float | None = None,
    log_skill_vs_iso_ci_lower: float | None = None,
    n_dates: int | None = None,
) -> str:
    """Stage B per-cell gate against isotonic (existing project rule).

    `cand_warnings` participates only for `_NON_OVERRIDABLE_WARNINGS` (e.g.,
    `incompatible_sigma`): even if prod and candidate both carry the warning
    (so `has_new_warnings` would be False), the candidate is still blocked.

    When provided, log_skill_vs_iso_* enforces the Kelly-aligned scoring rule
    alongside BSS — both must clear 0 (and CI must be > 0) for eligibility.

    `n_dates` (cluster mode only) below `MIN_CLUSTER_DATES` yields INSUFFICIENT_N.
    """
    if n < n_min_iso:
        return "INSUFFICIENT_N"
    if n_dates is not None and n_dates < MIN_CLUSTER_DATES:
        return "INSUFFICIENT_N"
    if cand_warnings and (set(cand_warnings) & _NON_OVERRIDABLE_WARNINGS):
        return "BLOCKED_WARNINGS"
    if has_new_warnings:
        return "BLOCKED_WARNINGS"
    if bss_vs_iso_point <= 0.0:
        return "NOT_BETTER_THAN_ISO"
    if bss_vs_iso_ci_lower <= 0.0:
        return "CI_LOWER_NEGATIVE"
    if log_skill_vs_iso_point is not None and log_skill_vs_iso_point <= 0.0:
        return "LOG_SKILL_NOT_BETTER"
    if log_skill_vs_iso_ci_lower is not None and log_skill_vs_iso_ci_lower <= 0.0:
        return "LOG_SKILL_CI_LOWER_NEGATIVE"
    return "STAGE_B_ELIGIBLE"


def _cluster_gate(
    cell: str, dates: Sequence[Any] | None, scope: str
) -> tuple[int | None, bool]:
    """Unique-date count and whether the cell is too thin to bootstrap by date."""
    if dates is None:
        return None, False
    n_dates = len(set(dates))
    too_few = n_dates < MIN_CLUSTER_DATES
    if too_few:
        logger.warning(
            "%s (%s): %d unique forecast_dates < %d; cluster CI skipped, decision INSUFFICIENT_N",
            cell, scope, n_dates, MIN_CLUSTER_DATES,
        )
    return n_dates, too_few


def _warn_resample_mismatch(candidate_resample: str | None, resample: str) -> None:
    if candidate_resample is not None and candidate_resample != resample:
        logger.warning(
            "candidate diagnostics were fit with bootstrap_resample=%s but this comparison "
            "uses %s; the comparison is not like-for-like",
            candidate_resample, resample,
        )


def _build_stage_b_row(
    cell: str,
    settled_rows: list[dict[str, Any]],
    cand_table: dict[str, EmosCoefficients],
    cand_warnings: set[str],
    *,
    bootstrap_iters: int,
    seed: int,
    n_min_iso: int = 10,
    cluster_by_date: bool = False,
) -> dict[str, Any] | None:
    """Build the Stage B (candidate-EMOS vs logged-isotonic) row for one cell.

    Returns ``None`` when:
    - The candidate has no coefficients for the cell.
    - Fewer than one isotonic-baseline row is available (n=0). The gate at
      ``_stage_b_decision`` handles n_min_iso for the *decision string*, but
      we still emit the row with INSUFFICIENT_N so the operator sees it.

    Filters via ``_is_isotonic_baseline_row`` so pre-cutover rows where
    ``live_calibrator_kind`` is absent are included as isotonic baselines.
    """
    coeffs = cand_table.get(cell)
    if coeffs is None:
        return None

    iso_aligned_sq_cand: list[float] = []
    iso_aligned_sq_iso: list[float] = []
    iso_aligned_nll_cand: list[float] = []
    iso_aligned_nll_iso: list[float] = []
    iso_aligned_dates: list[str] = []
    for r in settled_rows:
        if f"{r['city']}/{r['var']}" != cell:
            continue
        if not _is_isotonic_baseline_row(r):
            continue
        cal = r.get("calibrated_prob")
        if cal is None:
            continue
        outcome = resolve_outcome(
            r["strike_type"], r.get("floor"), r.get("cap"), float(r["observed"])
        )
        p_cand, _, _ = emos_probability(
            coeffs,
            ens_mean=float(r["ens_mean_raw"]),
            ens_std=float(r["ens_std_raw"]),
            strike_type=r["strike_type"],
            floor=r["floor"],
            cap=r["cap"],
        )
        iso_aligned_dates.append(r["forecast_date"])
        iso_aligned_sq_cand.append((p_cand - outcome) ** 2)
        iso_aligned_sq_iso.append((float(cal) - outcome) ** 2)
        if not (math.isfinite(p_cand) and math.isfinite(float(cal))):
            raise ValueError(
                f"non-finite probability in Stage B row: p_cand={p_cand!r} "
                f"cal={cal!r} cell={cell} ticker={r.get('ticker')!r}"
            )
        p_cand_c = min(max(p_cand, _LOG_LOSS_EPS), 1.0 - _LOG_LOSS_EPS)
        p_iso_c = min(max(float(cal), _LOG_LOSS_EPS), 1.0 - _LOG_LOSS_EPS)
        iso_aligned_nll_cand.append(
            -(outcome * math.log(p_cand_c) + (1 - outcome) * math.log(1.0 - p_cand_c))
        )
        iso_aligned_nll_iso.append(
            -(outcome * math.log(p_iso_c) + (1 - outcome) * math.log(1.0 - p_iso_c))
        )

    if not iso_aligned_sq_iso:
        return None

    n = len(iso_aligned_sq_iso)
    boot_dates = iso_aligned_dates if cluster_by_date else None
    n_dates, too_few_dates = _cluster_gate(cell, boot_dates, "stage B")
    m_iso = sum(iso_aligned_sq_iso) / n
    m_cand_iso = sum(iso_aligned_sq_cand) / n
    bss_iso = 1.0 - m_cand_iso / m_iso if m_iso > 0 else 0.0
    ci_iso_lo, ci_iso_hi = (
        (math.nan, math.nan)
        if too_few_dates
        else bootstrap_bss_ci(
            iso_aligned_sq_iso, iso_aligned_sq_cand,
            n_iters=bootstrap_iters, seed=seed, dates=boot_dates,
        )
    )
    nll_iso = sum(iso_aligned_nll_iso) / n
    nll_cand_iso = sum(iso_aligned_nll_cand) / n
    log_skill_iso = 1.0 - nll_cand_iso / nll_iso if nll_iso > 0 else 0.0
    log_skill_iso_lo, log_skill_iso_hi = (
        (math.nan, math.nan)
        if too_few_dates
        else bootstrap_log_skill_ci(
            iso_aligned_nll_iso, iso_aligned_nll_cand,
            n_iters=bootstrap_iters, seed=seed, dates=boot_dates,
        )
    )
    decision = _stage_b_decision(
        n=n,
        n_min_iso=n_min_iso,
        bss_vs_iso_point=bss_iso,
        bss_vs_iso_ci_lower=ci_iso_lo,
        has_new_warnings=bool(cand_warnings),
        cand_warnings=cand_warnings,
        log_skill_vs_iso_point=log_skill_iso,
        log_skill_vs_iso_ci_lower=log_skill_iso_lo,
        n_dates=n_dates,
    )
    row: dict[str, Any] = {"cell": cell, "n": n}
    if n_dates is not None:
        row["n_dates"] = n_dates
    row.update({
        "Brier_iso": f"{m_iso:.4f}",
        "Brier_cand_emos": f"{m_cand_iso:.4f}",
        "BSS_emos_vs_iso": f"{bss_iso:+.4f} [{ci_iso_lo:+.3f}, {ci_iso_hi:+.3f}]",
        "LogSkill_emos_vs_iso": (
            f"{log_skill_iso:+.4f} [{log_skill_iso_lo:+.3f}, {log_skill_iso_hi:+.3f}]"
        ),
        "cand_warnings": ",".join(sorted(cand_warnings)) or "-",
        "stage_b": decision,
    })
    return row


def _format_markdown_table(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_(no rows)_\n"
    headers = list(rows[0].keys())
    out = [f"### {title}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Date window
    today = dt.date.today()
    end_date = args.end_date or today
    start_date = args.start_date or (end_date - dt.timedelta(days=args.lookback_days))
    logger.info("date window: %s → %s", start_date, end_date)

    # Load coefficients (use EmosTable so we apply the same bounds/contract as runtime)
    prod_table = _table_to_dict(EmosTable.load(args.prod_coeffs))
    cand_table = _table_to_dict(EmosTable.load(args.candidate_coeffs))
    logger.info(
        "loaded coefficients: prod=%d cells, candidate=%d cells",
        len(prod_table),
        len(cand_table),
    )

    # Load candidate diagnostics for ci_warnings comparison
    bootstrap_resample = (
        BOOTSTRAP_RESAMPLE_DATE_CLUSTER if args.cluster_by_date else BOOTSTRAP_RESAMPLE_ROW
    )
    cand_diag_doc = json.loads(args.candidate_diagnostics.read_text())
    _warn_resample_mismatch(cand_diag_doc.get("bootstrap_resample"), bootstrap_resample)
    cand_diag = cand_diag_doc["cells"]
    prod_diag = json.loads(args.prod_diagnostics.read_text())["cells"]

    # Load forecasts_raw index
    logger.info("loading forecasts_raw index from %s", args.forecasts_raw)
    raw_idx = load_forecasts_raw_index(args.forecasts_raw)
    logger.info("forecasts_raw index size: %d", len(raw_idx))

    # Stream audit rows
    logger.info("streaming eval_audit from %s", args.eval_audit)
    audit_rows = list(_iter_audit_rows(args.eval_audit, start_date, end_date))
    logger.info("eval_audit rows in window with emos_prob_raw: %d", len(audit_rows))

    # Settlements
    dates_needed = {dt.date.fromisoformat(r["forecast_date"]) for r in audit_rows}
    logger.info("settlement dates needed: %d", len(dates_needed))
    settlements = _load_settlements_index(args.snapshots_dir, dates_needed)
    logger.info("settlements index size: %d", len(settlements))

    # Join
    settled_rows, counters = _join_settled_rows(
        audit_rows, raw_idx, settlements, final_only=args.final_only
    )
    logger.info("join counters: %s", counters)
    logger.info("settled rows: %d", len(settled_rows))

    if not settled_rows:
        logger.error("no settled rows after join — aborting")
        return 2

    # Bit-for-bit replay invariant on live-EMOS rows
    checked = _assert_bit_for_bit_replay(settled_rows, prod_table)
    logger.info("bit-for-bit replay invariant verified on %d live-EMOS rows", checked)

    # Per-cell squared errors for prod and candidate
    sq_prod = compute_squared_errors_per_cell(settled_rows, prod_table)
    sq_cand = compute_squared_errors_per_cell(settled_rows, cand_table)

    # Per-cell log-losses (Kelly-aligned proper scoring rule). Same row order
    # as the squared-error accumulator, so pair-aligned bootstrap is valid.
    nll_prod = compute_log_losses_per_cell(settled_rows, prod_table)
    nll_cand = compute_log_losses_per_cell(settled_rows, cand_table)

    # Group ens_mean for μ_shift drift. (The Stage B vs-iso accumulator was
    # previously built here too, but it was only used as a presence guard for
    # the second-pass alignment loop; _build_stage_b_row now handles both
    # via _is_isotonic_baseline_row in one pass per cell.)
    per_cell_means: dict[str, list[float]] = defaultdict(list)
    per_cell_dates: dict[str, list[str]] = defaultdict(list)
    for r in settled_rows:
        cell = f"{r['city']}/{r['var']}"
        per_cell_means[cell].append(float(r["ens_mean_raw"]))
        per_cell_dates[cell].append(r["forecast_date"])

    # Build decision rows
    main_table: list[dict[str, Any]] = []
    stage_b_table: list[dict[str, Any]] = []
    all_cells = sorted(set(prod_table.keys()) | set(cand_table.keys()))
    out_records: list[dict[str, Any]] = []

    for cell in all_cells:
        if cell not in sq_prod or cell not in sq_cand:
            continue
        n = len(sq_prod[cell])
        if n == 0:
            continue
        m_prod = sum(sq_prod[cell]) / n
        m_cand = sum(sq_cand[cell]) / n
        bss = (1.0 - m_cand / m_prod) if m_prod > 0 else 0.0
        cell_dates = per_cell_dates[cell] if args.cluster_by_date else None
        n_dates, too_few_dates = _cluster_gate(cell, cell_dates, "main table")
        ci_lo, ci_hi = (
            (math.nan, math.nan)
            if too_few_dates
            else bootstrap_bss_ci(
                sq_prod[cell], sq_cand[cell],
                n_iters=args.bootstrap_iters, seed=args.seed, dates=cell_dates,
            )
        )

        nll_prod_cell = nll_prod.get(cell, [])
        nll_cand_cell = nll_cand.get(cell, [])
        # Pair-alignment invariant: NLL and squared-error accumulators iterate
        # the same `settled_rows` list under the same skip rule, so per-cell
        # lengths must match. A mismatch means a future skip-rule divergence
        # silently desynchronised the pairs — the bootstrap would still run
        # but on misaligned data, masquerading as a valid CI. (Audit
        # 2026-05-23, code-auditor HIGH #1.)
        if len(nll_prod_cell) != n or len(nll_cand_cell) != n:
            raise RuntimeError(
                f"NLL/squared-error length mismatch for {cell}: "
                f"sq={n} nll_prod={len(nll_prod_cell)} nll_cand={len(nll_cand_cell)}"
            )
        ll_prod = sum(nll_prod_cell) / n
        ll_cand = sum(nll_cand_cell) / n
        log_skill = (1.0 - ll_cand / ll_prod) if ll_prod > 0 else 0.0
        log_skill_lo, log_skill_hi = (
            (math.nan, math.nan)
            if too_few_dates
            else bootstrap_log_skill_ci(
                nll_prod_cell, nll_cand_cell,
                n_iters=args.bootstrap_iters, seed=args.seed, dates=cell_dates,
            )
        )

        prod_c = prod_table.get(cell)
        cand_c = cand_table.get(cell)
        mu_drift = (
            compute_mu_shift_drift(
                per_cell_means[cell],
                prod_a=prod_c.a, prod_b=prod_c.b,
                cand_a=cand_c.a, cand_b=cand_c.b,
            )
            if prod_c and cand_c
            else float("nan")
        )

        prod_warnings = set((prod_diag.get(cell) or {}).get("ci_warnings") or [])
        cand_warnings = set((cand_diag.get(cell) or {}).get("ci_warnings") or [])
        new_warnings = cand_warnings - prod_warnings

        decision = decide_cell(
            n=n,
            n_min=args.n_min,
            bss_point=bss,
            bss_ci_lower=ci_lo,
            mu_drift=mu_drift,
            mu_drift_max=args.mu_drift_max,
            has_new_warnings=bool(new_warnings),
            bss_ci_lower_floor=args.bss_ci_lower_floor,
            cand_warnings=cand_warnings,
            log_skill_point=log_skill,
            log_skill_ci_lower=log_skill_lo,
            log_skill_ci_lower_floor=args.log_skill_ci_lower_floor,
            n_dates=n_dates,
        )

        table_row: dict[str, Any] = {"cell": cell, "n": n}
        if n_dates is not None:
            table_row["n_dates"] = n_dates
        table_row.update({
            "Brier_prod": f"{m_prod:.4f}",
            "Brier_cand": f"{m_cand:.4f}",
            "ΔBrier": f"{(m_cand - m_prod):+.4f}",
            "BSS": f"{bss:+.4f} [{ci_lo:+.3f}, {ci_hi:+.3f}]",
            "LogSkill": f"{log_skill:+.4f} [{log_skill_lo:+.3f}, {log_skill_hi:+.3f}]",
            "μ_drift_°F": f"{mu_drift:.2f}",
            "new_warnings": ",".join(sorted(new_warnings)) or "-",
            "decision": decision,
        })
        main_table.append(table_row)

        # Stage B: candidate EMOS vs isotonic. Per-cell row build via
        # _build_stage_b_row, which uses _is_isotonic_baseline_row so
        # pre-cutover audit rows (missing live_calibrator_kind) participate.
        stage_b_row = _build_stage_b_row(
            cell,
            settled_rows,
            cand_table,
            cand_warnings,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed,
            cluster_by_date=args.cluster_by_date,
        )
        if stage_b_row is not None:
            stage_b_table.append(stage_b_row)

        record: dict[str, Any] = {"cell": cell, "n": n}
        if n_dates is not None:
            record["n_dates"] = n_dates
        record.update({
            "brier_prod": m_prod,
            "brier_cand": m_cand,
            "bss_cand_vs_prod": bss,
            "bss_ci": [ci_lo, ci_hi],
            "nll_prod": ll_prod,
            "nll_cand": ll_cand,
            "log_skill_cand_vs_prod": log_skill,
            "log_skill_ci": [log_skill_lo, log_skill_hi],
            "mu_drift_degf": mu_drift,
            "new_warnings": sorted(new_warnings),
            "decision": decision,
        })
        out_records.append(record)

    # Overall swap recommendation
    live_decisions = {
        c["cell"]: c["decision"]
        for c in main_table
        if c["cell"] in set(args.live_cells)
    }
    promote_all_live = bool(live_decisions) and all(
        d == "PROMOTE" for d in live_decisions.values()
    )

    # Print report
    print()
    print("=" * 80)
    print("EMOS Refit Shadow Validation Report")
    print("=" * 80)
    print(f"prod coeffs:      {args.prod_coeffs}  sha256={_sha256(args.prod_coeffs)[:12]}")
    print(f"candidate coeffs: {args.candidate_coeffs}  sha256={_sha256(args.candidate_coeffs)[:12]}")
    print(f"window: {start_date} → {end_date}  ({(end_date - start_date).days + 1} days)")
    print(f"bootstrap resample: {bootstrap_resample}")
    print(f"settled rows: {len(settled_rows)}  bit-for-bit replay verified on {checked} live-EMOS rows")
    print()
    print(_format_markdown_table(main_table, "Per-cell A/B (candidate vs prod EMOS)"))
    print(_format_markdown_table(stage_b_table, "Stage B candidates (candidate EMOS vs logged isotonic)"))
    print(f"Live cells: {args.live_cells}")
    print(f"Live-cell decisions: {live_decisions}")
    print(f"OVERALL: {'PROMOTE' if promote_all_live else 'KEEP_PROD'}")
    print()

    if args.json is None:
        ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        args.json = Path(f"data/emos/refit_validation_{ts}.json")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "prod_coeffs": str(args.prod_coeffs),
        "candidate_coeffs": str(args.candidate_coeffs),
        "prod_sha256": _sha256(args.prod_coeffs),
        "candidate_sha256": _sha256(args.candidate_coeffs),
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "settled_rows": len(settled_rows),
        "bit_for_bit_checked": checked,
        "n_min": args.n_min,
        "mu_drift_max": args.mu_drift_max,
        "bss_ci_lower_floor": args.bss_ci_lower_floor,
        "log_skill_ci_lower_floor": args.log_skill_ci_lower_floor,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_resample": bootstrap_resample,
        "seed": args.seed,
        "cells": out_records,
        "live_cells": list(args.live_cells),
        "live_cell_decisions": live_decisions,
        "overall": "PROMOTE" if promote_all_live else "KEEP_PROD",
        "stage_b": stage_b_table,
    }, indent=2))
    logger.info("wrote JSON report to %s", args.json)
    return 0 if promote_all_live else 1


if __name__ == "__main__":
    raise SystemExit(main())
