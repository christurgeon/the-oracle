"""Intraday ASOS HIGH shadow reliability report — live-flip decision instrument.

Reads ``state/predictions.jsonl``, filters ``strategy=="intraday_asos_high"``,
joins to NWS-settled temperatures from ``data/snapshots/*/settlements.json``,
and produces the §9 per-bin reliability table that governs the live-flip
decision for the shadow-only ``intraday_asos_high`` strategy.

Headline metric is CALIBRATION (reliability table of ``posterior_p_raw`` vs
bracket outcome).  If P&L is shown it is labelled NON-DECISION-GRADE.

This report makes the eventual live-capital decision for the shadow-only
``intraday_asos_high`` strategy, so it is built to defend against three
statistical defects an audit found in the first cut:

  * **Per-settlement dedup (CRITICAL-1).** In shadow mode the strategy re-logs a
    fresh prediction for the SAME open market on EVERY orchestrator tick (no
    fills → no dedup), so dozens of autocorrelated tick-rows exist per
    settlement event.  The §9 ``n`` gates must count INDEPENDENT settlements,
    not tick-rows.  ``dedup_to_settlement_events`` collapses each
    ``(city, forecast_date, market_ticker)`` event to the single
    decision-relevant tick — the SHARPEST posterior (smallest ``sigma_gain``),
    tie-broken by latest ``ts``.

  * **σ_gain stratification (CRITICAL-2).** ``guard_outcome=="pass"`` pools
    no-edge wide-σ morning rows with sharp post-peak rows, masking the binding
    low-p NO bin.  The §9 verdict is computed ONLY on the SHARP band
    (``sigma_gain <= SHARP_SIGMA_GAIN_MAX``); the wide band is a separate
    diagnostic.

  * **Wilson-CI kill criteria (CRITICAL-3).** A bare point comparison at n≈20
    has no power (the binomial CI half-width dwarfs the 0.05 tolerance).  The
    verdict uses a Wilson score 95% CI and fires PASS/FAIL only when the CI is
    on the correct side of the threshold; otherwise INCONCLUSIVE.

The strategy corpus is empty on first run (~10 weeks of forward shadow data
needed before the kill-criteria verdicts have power); the script exits 0 and
prints a clear message rather than crashing.

Run (cached):  PYTHONPATH=. uv run python scripts/intraday_asos_shadow_report.py
Run (fresh):   uv run python scripts/intraday_asos_shadow_report.py --pull-vps
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.analytics.prediction_log import (  # noqa: E402
    PredictionRecord,
    _deserialize,
)
from scripts._asos_reliability_bins import (  # noqa: E402
    ASOS_BINS,
    compute_reliability_bins,
)
from scripts._calibration_common import prefer_latest_settlement  # noqa: E402
from scripts._vps import VPS_STATE_DIR, pull_audit_files, require_vps_host  # noqa: E402

STRATEGY_NAME = "intraday_asos_high"

DEFAULT_PREDICTIONS = _PROJECT_ROOT / "state" / "predictions.jsonl"
DEFAULT_SNAPSHOTS = _PROJECT_ROOT / "data" / "snapshots"
DEFAULT_CONFIG = _PROJECT_ROOT / "python" / "config.yaml"

# §9 kill-criterion thresholds (pre-registered, spec §9).
LOW_P_THRESHOLD = 0.10  # posterior_p_raw < this → NO-side bin
HIGH_P_THRESHOLD = 0.95  # posterior_p_raw > this → high-confidence YES bin
LOW_P_FAIL_TOLERANCE = 0.05  # realized YES rate must be within pred + this
HIGH_P_MIN_REALIZED = 0.85  # realized YES rate floor for the high-p bin
KILL_N_MIN = 20  # min INDEPENDENT in-bin settlements for a PASS/FAIL verdict

# CRITICAL-2 — σ_gain stratification.
# The 1°F Kalshi bracket is the natural scale: a posterior is only "sharp enough
# to have edge" when its σ_gain sits at or below the bracket width.  Wide-σ
# morning rows (σ_gain well above 1°F) carry no real edge and, if pooled, dilute
# the binding low-p bin's realized rate.  The §9 verdict is computed ONLY on the
# sharp band; the wide band is reported as a separate diagnostic.
#
# NOTE: this cutoff is a FORWARD-DATA decision, not a value overfit to the
# 22-day prototype.  1.2°F gives a small margin above the 1°F bracket so a
# posterior that is "sharper than the bracket" (the regime the strategy's edge
# relies on) is retained; revisit once forward σ_gain distribution is observed.
SHARP_SIGMA_GAIN_MAX = 1.2

# CRITICAL-3 — Wilson score interval z for a two-sided 95% CI.  The CI
# half-width is reported per bin so the operator can see the actual power: a
# PASS/FAIL only fires when the CI lands cleanly on one side of the threshold,
# so n must be large enough for the interval to be tight.
WILSON_Z_95 = 1.959963984540054

# MAJOR-2 — informational-only cells (logged, never traded) excluded from the
# §9 headline cohort.  Spec §7: read from config; CLI override available.
DEFAULT_INFORMATIONAL_CELLS = ["LAX/high"]

_VARIABLE_TO_SETTLEMENT_FIELD = {
    "high": "high_f",
    "temperature_2m_max": "high_f",
    "low": "low_f",
    "temperature_2m_min": "low_f",
}

VPS_PREDICTIONS_PATH = f"{VPS_STATE_DIR}/predictions.jsonl"


# ---------------------------------------------------------------------------
# Wilson score confidence interval (CRITICAL-3) — pure function
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Return the Wilson score (lower, upper) CI for a binomial proportion.

    The Wilson interval is well-behaved at small n and near 0/1 (unlike the
    normal-approximation interval), which is exactly the regime the §9 gate
    operates in (n≈20, realized rate ≈ 0.1).

    For ``n == 0`` returns ``(0.0, 1.0)`` (maximally uninformative).
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes ({successes}) must be in [0, n] (n={n})")

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lower = center - margin
    upper = center + margin
    # Clamp to [0, 1] for numerical safety.
    return (max(0.0, lower), min(1.0, upper))


# ---------------------------------------------------------------------------
# Per-settlement dedup (CRITICAL-1) — pure function
# ---------------------------------------------------------------------------


def _settlement_event_key(rec: PredictionRecord) -> tuple[str, str, str]:
    """Dedup key for one settlement event: ``(city, forecast_date, ticker)``.

    ``forecast_date`` is the market's settlement date — ``expiration_date`` on
    the record (the spec calls this ``forecast_date``/expiration interchangeably).
    """
    city = rec.city or "unknown"
    fdate = rec.expiration_date.isoformat() if rec.expiration_date else "unknown"
    return (city, fdate, rec.market_ticker)


def _dedup_sort_key(rec: PredictionRecord) -> tuple[float, float]:
    """Ranking key within one settlement event: prefer SHARPEST then LATEST.

    The decision-relevant tick is the latest fresh-ASOS tick closest to peak —
    operationalised as the SHARPEST posterior (smallest ``sigma_gain``, that's
    the post-peak max-information tick the strategy's edge relies on), tie-broken
    by latest ``ts``.

    Returned key sorts ASCENDING so the WINNER is the MINIMUM:
      * primary:   ``sigma_gain`` (None → +inf so it always loses to a real σ)
      * secondary: ``-timestamp`` (later ts wins the σ tie)
    """
    sigma = rec.sigma_gain if rec.sigma_gain is not None else math.inf
    ts = rec.timestamp.timestamp()
    return (sigma, -ts)


def dedup_to_settlement_events(recs: list[PredictionRecord]) -> list[PredictionRecord]:
    """Collapse autocorrelated tick-rows to ONE row per settlement event.

    CRITICAL-1: in shadow mode the strategy re-logs the SAME open market on every
    orchestrator tick, so the raw corpus counts dozens of correlated rows as
    independent ``n``.  This collapses each ``(city, forecast_date, ticker)``
    event to its single decision-relevant tick (sharpest posterior, tie-broken by
    latest ``ts`` — see ``_dedup_sort_key``).

    Pure: no I/O, deterministic.  Output order is sorted by event key for
    reproducibility.
    """
    by_event: dict[tuple[str, str, str], PredictionRecord] = {}
    for rec in recs:
        key = _settlement_event_key(rec)
        cur = by_event.get(key)
        if cur is None or _dedup_sort_key(rec) < _dedup_sort_key(cur):
            by_event[key] = rec
    return [by_event[k] for k in sorted(by_event)]


def is_sharp(rec: PredictionRecord) -> bool:
    """True iff the posterior is sharp enough to carry edge (CRITICAL-2).

    Sharp ⇔ ``sigma_gain <= SHARP_SIGMA_GAIN_MAX``.  A row with no ``sigma_gain``
    is treated as NOT sharp (conservative — it cannot enter the headline cohort).
    """
    return rec.sigma_gain is not None and rec.sigma_gain <= SHARP_SIGMA_GAIN_MAX


# ---------------------------------------------------------------------------
# Settlement join (mirrors scripts/intraday_shadow_report.py pattern exactly)
# ---------------------------------------------------------------------------


def load_settlements(snapshots_dir: Path) -> dict[tuple[str, str], dict]:
    """Return final-only settlements keyed by ``(city, observation_date_iso)``.

    Scans all snapshot subdirectories (flat ``<date>/`` and nested
    ``<date>/<run_id>/``) for ``settlements.json`` files.  Merges by
    ``prefer_latest_settlement``; preliminary winners are dropped after merge.
    """
    settlements: dict[tuple[str, str], dict] = {}
    if not snapshots_dir.is_dir():
        return settlements

    def _merge(path: Path) -> None:
        try:
            entries = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        for s in entries:
            city = s.get("city")
            obs_date = s.get("observation_date")
            if not city or not obs_date:
                continue
            key = (city, obs_date)
            existing = settlements.get(key)
            if existing is None or prefer_latest_settlement(existing, s):
                settlements[key] = s

    for day_dir in sorted(snapshots_dir.iterdir()):
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        st_file = day_dir / "settlements.json"
        if st_file.exists():
            _merge(st_file)
        else:
            for run_dir in sorted(day_dir.iterdir()):
                if run_dir.is_dir():
                    run_st = run_dir / "settlements.json"
                    if run_st.exists():
                        _merge(run_st)

    return {k: v for k, v in settlements.items() if not v.get("is_preliminary")}


def lookup_actual_temp(
    settlements: dict[tuple[str, str], dict],
    city: str,
    daily_variable: str,
    observation_date_iso: str,
) -> float | None:
    """Return the NWS-settled temperature for (city, variable, date), or None."""
    st = settlements.get((city, observation_date_iso))
    if st is None:
        return None
    field = _VARIABLE_TO_SETTLEMENT_FIELD.get(daily_variable)
    if field is None:
        return None
    return st.get(field)


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------


def evaluate_outcome(rec: PredictionRecord, actual_temp: float) -> int | None:
    """Return 1 if the YES bracket was hit, 0 if not, None if indeterminate."""
    yes = derive_outcome_yes(rec.strike_type, rec.floor_strike, rec.cap_strike, actual_temp)
    if yes is None:
        return None
    return 1 if yes else 0


# ---------------------------------------------------------------------------
# Config: informational cells (MAJOR-2)
# ---------------------------------------------------------------------------


def load_informational_cells(config_path: Path) -> list[str]:
    """Read ``strategies.intraday_asos_high.informational_cells`` from config.

    Falls back to ``DEFAULT_INFORMATIONAL_CELLS`` if the file or key is absent.
    """
    try:
        import yaml  # local import; pyyaml is a core dep but keep the helper hermetic

        cfg = yaml.safe_load(config_path.read_text())
    except (OSError, ImportError, ValueError):
        return list(DEFAULT_INFORMATIONAL_CELLS)
    try:
        cells = cfg["strategies"]["intraday_asos_high"]["informational_cells"]
    except (KeyError, TypeError):
        return list(DEFAULT_INFORMATIONAL_CELLS)
    if not isinstance(cells, list):
        return list(DEFAULT_INFORMATIONAL_CELLS)
    return [str(c) for c in cells]


def _cell_of(rec: PredictionRecord) -> str:
    return f"{rec.city or 'unknown'}/{rec.daily_variable or 'unknown'}"


# ---------------------------------------------------------------------------
# §9 kill-criteria evaluation (CRITICAL-3: Wilson-CI gated)
# ---------------------------------------------------------------------------


def _bin_summary(outcomes: list[int], mean_pred: float | None) -> dict[str, Any]:
    """Wilson-CI summary block for a bin's realized outcomes."""
    n = len(outcomes)
    successes = sum(outcomes)
    realized = successes / n if n else None
    lo, hi = wilson_ci(successes, n)
    half_width = (hi - lo) / 2 if n else None
    return {
        "n": n,
        "mean_pred": mean_pred,
        "mean_real": realized,
        "wilson_lo": lo if n else None,
        "wilson_hi": hi if n else None,
        "wilson_half_width": half_width,
    }


def _eval_low_p_bin(probs: list[float], outcomes: list[int]) -> dict[str, Any]:
    """Low-p NO bin verdict via Wilson CI on the realized YES rate.

    FAIL  iff Wilson LOWER bound > pred + tolerance (confidently miscalibrated).
    PASS  iff Wilson UPPER bound ≤ pred + tolerance (confidently fine).
    else  INSUFFICIENT-N (n < KILL_N_MIN) or INCONCLUSIVE (CI straddles).
    """
    n = len(probs)
    mean_pred = sum(probs) / n if n else None
    stats = _bin_summary(outcomes, mean_pred)
    threshold = (mean_pred + LOW_P_FAIL_TOLERANCE) if mean_pred is not None else None

    if n < KILL_N_MIN:
        verdict = "INSUFFICIENT-N"
    elif threshold is not None and stats["wilson_lo"] > threshold:
        verdict = "FAIL"
    elif threshold is not None and stats["wilson_hi"] <= threshold:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "criterion": (
            f"posterior_p_raw < {LOW_P_THRESHOLD}: Wilson-95% CI of realized vs "
            f"pred+{LOW_P_FAIL_TOLERANCE} at n≥{KILL_N_MIN} (sharp band only)"
        ),
        "verdict": verdict,
        "fail_threshold": threshold,
        **stats,
    }


def _eval_high_p_bin(probs: list[float], outcomes: list[int]) -> dict[str, Any]:
    """High-p bin verdict via Wilson CI vs the 0.85 realized floor.

    FAIL  iff Wilson UPPER bound < HIGH_P_MIN_REALIZED (confidently below floor).
    PASS  iff Wilson LOWER bound ≥ HIGH_P_MIN_REALIZED (confidently at/above).
    else  INSUFFICIENT-N / INCONCLUSIVE.
    """
    n = len(probs)
    mean_pred = sum(probs) / n if n else None
    stats = _bin_summary(outcomes, mean_pred)

    if n < KILL_N_MIN:
        verdict = "INSUFFICIENT-N"
    elif stats["wilson_hi"] < HIGH_P_MIN_REALIZED:
        verdict = "FAIL"
    elif stats["wilson_lo"] >= HIGH_P_MIN_REALIZED:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "criterion": (
            f"posterior_p_raw > {HIGH_P_THRESHOLD}: Wilson-95% CI vs realized "
            f"≥ {HIGH_P_MIN_REALIZED} at n≥{KILL_N_MIN} (sharp band only)"
        ),
        "verdict": verdict,
        **stats,
    }


def _eval_kill_criteria_raw(
    low_p_probs: list[float],
    low_p_outcomes: list[int],
    high_p_probs: list[float],
    high_p_outcomes: list[int],
    all_probs: list[float],
    all_outcomes: list[int],
) -> dict[str, Any]:
    """Evaluate §9 kill criteria from raw prob/outcome lists (Wilson-CI gated).

    Inputs are expected to be the SHARP-band, DEDUPED, informational-excluded
    cohort (the caller is responsible for that filtering); this function is the
    pure statistical verdict engine.  Separated from ``build_report`` so it is
    testable independently.
    """
    low = _eval_low_p_bin(low_p_probs, low_p_outcomes)
    high = _eval_high_p_bin(high_p_probs, high_p_outcomes)

    # Calibration-in-large (secondary, report-only; spec §1.2 warns it is the
    # WRONG instrument here — kept only as a sanity diagnostic).
    n_all = len(all_probs)
    mean_pred = sum(all_probs) / n_all if n_all else None
    mean_real = sum(all_outcomes) / n_all if n_all else None
    cal_gap = (
        abs(mean_pred - mean_real) if (mean_pred is not None and mean_real is not None) else None
    )

    return {
        "low_p_no_bin": low,
        "high_p_bin": high,
        "calibration_in_large": {
            "note": "secondary — report-only; NOT the live-flip headline (spec §1.2)",
            "n": n_all,
            "mean_pred": mean_pred,
            "mean_real": mean_real,
            "abs_gap": cal_gap,
        },
    }


# ---------------------------------------------------------------------------
# Reliability tables + kill verdict for an arbitrary cohort of records
# ---------------------------------------------------------------------------


def _pairs(
    recs: list[PredictionRecord],
    outcome_by_key: dict[tuple[str, str, str], int],
) -> tuple[list[float], list[int]]:
    """Return (probs, outcomes) for records that have both a posterior and an outcome."""
    probs: list[float] = []
    outcomes: list[int] = []
    for r in recs:
        p = r.posterior_p_raw
        if p is None:
            continue
        oc = outcome_by_key.get(_settlement_event_key(r))
        if oc is None:
            continue
        probs.append(p)
        outcomes.append(oc)
    return probs, outcomes


def _kill_for_cohort(
    recs: list[PredictionRecord],
    outcome_by_key: dict[tuple[str, str, str], int],
) -> dict[str, Any]:
    """Compute the §9 Wilson-CI kill verdict for a (sharp, deduped) cohort."""
    probs, outcomes = _pairs(recs, outcome_by_key)
    low_p_probs = [p for p in probs if p < LOW_P_THRESHOLD]
    low_p_outcomes = [outcomes[i] for i, p in enumerate(probs) if p < LOW_P_THRESHOLD]
    high_p_probs = [p for p in probs if p > HIGH_P_THRESHOLD]
    high_p_outcomes = [outcomes[i] for i, p in enumerate(probs) if p > HIGH_P_THRESHOLD]
    return _eval_kill_criteria_raw(
        low_p_probs,
        low_p_outcomes,
        high_p_probs,
        high_p_outcomes,
        probs,
        outcomes,
    )


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------


def build_report(
    predictions_path: Path,
    snapshots_dir: Path,
    *,
    informational_cells: list[str] | None = None,
) -> dict[str, Any]:
    """Build the full calibration report dict.  Pure — no I/O side effects.

    Pipeline (defends the three audit defects + the Stage-0a frame cut):
      1. parse + filter to ``intraday_asos_high`` (with parse-loss reconciliation)
      2. PARTITION by ``frame_version``: frame-v2 (corrected LST frame) is the
         BINDING cohort; frame-v1 (legacy civil/DST frame) is a separate,
         clearly-labelled diagnostic — frames are NEVER pooled (Stage 0a)
      3. per frame: DEDUP tick-rows → one row per settlement event (CRITICAL-1)
      4. join settlements + derive outcomes
      5. split guard_outcome=="pass" → SHARP / WIDE σ bands (CRITICAL-2)
      6. exclude informational cells from the headline cohort (MAJOR-2)
      7. Wilson-CI §9 verdict on the sharp, enabled cohort + per-cell (CRITICAL-3)
    """
    if informational_cells is None:
        informational_cells = list(DEFAULT_INFORMATIONAL_CELLS)
    informational_set = set(informational_cells)

    # --- parse-loss reconciliation (MINOR) ---------------------------------
    total_raw_lines = 0
    parsed_records = 0
    dropped_rows = 0
    asos_recs: list[PredictionRecord] = []
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                total_raw_lines += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    dropped_rows += 1
                    continue
                rec = _deserialize(obj)
                if rec is None:
                    dropped_rows += 1
                    continue
                parsed_records += 1
                if rec.strategy == STRATEGY_NAME:
                    asos_recs.append(rec)

    settlements = load_settlements(snapshots_dir)

    # --- FRAME CUT (Stage 0a): never silently mix climate-day frames -------
    # frame_version 2 = corrected LST frame (binding cohort); 1 = legacy
    # civil/DST frame (contaminated boundary hour). Pre-v7 rows deserialize to 1.
    # The §9 verdicts are computed on frame-v2 ONLY; frame-v1 is a separate,
    # clearly-labelled diagnostic so a stale frame can never carry the decision.
    frame_v2_recs = [r for r in asos_recs if r.frame_version >= 2]
    frame_v1_recs = [r for r in asos_recs if r.frame_version < 2]

    parse_recon_common = {
        "total_raw_lines": total_raw_lines,
        "parsed_records": parsed_records,
        "rows_dropped": dropped_rows,
    }

    headline = _build_cohort_report(frame_v2_recs, settlements, informational_set)
    frame_v1 = _build_cohort_report(frame_v1_recs, settlements, informational_set)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": STRATEGY_NAME,
        "config": {
            "informational_cells": sorted(informational_set),
            "sharp_sigma_gain_max": SHARP_SIGMA_GAIN_MAX,
            "kill_n_min": KILL_N_MIN,
        },
        "frame_partition": {
            "binding_frame_version": 2,
            "frame_v2_rows": len(frame_v2_recs),
            "frame_v1_rows": len(frame_v1_recs),
            "frame_v2_settlement_events": headline["summary"]["n_settlement_events"],
            "frame_v1_settlement_events": frame_v1["summary"]["n_settlement_events"],
        },
        "parse_reconciliation": {
            **parse_recon_common,
            "intraday_asos_high_records": len(asos_recs),
            # Headline counts reflect the binding frame-v2 cohort.
            "frame_v2_records": len(frame_v2_recs),
            "frame_v1_records": len(frame_v1_recs),
            "deduped_settlement_events": headline["summary"]["n_settlement_events"],
        },
        # HEADLINE: every binding table/verdict is the frame-v2 cohort.
        "summary": headline["summary"],
        "reliability_sharp_enabled": headline["reliability_sharp_enabled"],
        "reliability_wide_diagnostic": headline["reliability_wide_diagnostic"],
        "reliability_informational": headline["reliability_informational"],
        "kill_criteria": headline["kill_criteria"],
        "by_cell": headline["by_cell"],
        # DIAGNOSTIC: legacy civil-frame (frame_version 1) cohort, kept fully
        # separate and clearly labelled. NOT decision-grade.
        "frame_v1_diagnostic": frame_v1,
    }
    return report


def _build_cohort_report(
    asos_recs: list[PredictionRecord],
    settlements: dict[tuple[str, str], dict],
    informational_set: set[str],
) -> dict[str, Any]:
    """Build the dedup→join→σ-band→Wilson-CI cohort report for one record set.

    Pure.  Called once per climate-day frame so frames are never pooled (Stage
    0a).  Returns the ``summary`` / reliability-table / ``kill_criteria`` /
    ``by_cell`` block for the supplied records.
    """
    # --- CRITICAL-1: dedup tick-rows to one row per settlement event -------
    n_asos_rows_pre_dedup = len(asos_recs)
    deduped = dedup_to_settlement_events(asos_recs)
    n_settlement_events = len(deduped)

    # --- settlement join + outcome derivation (on deduped events) ----------
    settled_pass: list[PredictionRecord] = []  # guard_outcome == "pass", settled
    settled_all: list[PredictionRecord] = []  # all guard outcomes, settled
    pending_count = 0
    outcome_by_key: dict[tuple[str, str, str], int] = {}

    for rec in deduped:
        exp_iso = rec.expiration_date.isoformat() if rec.expiration_date else None
        if exp_iso is None or rec.city is None or rec.daily_variable is None:
            pending_count += 1
            continue
        actual = lookup_actual_temp(settlements, rec.city, rec.daily_variable, exp_iso)
        if actual is None:
            pending_count += 1
            continue
        outcome = evaluate_outcome(rec, actual)
        if outcome is None:
            pending_count += 1
            continue
        outcome_by_key[_settlement_event_key(rec)] = outcome
        settled_all.append(rec)
        if rec.guard_outcome == "pass":
            settled_pass.append(rec)

    # --- CRITICAL-2: σ-band split on the trade-eligible (guard=="pass") rows
    sharp_pass = [r for r in settled_pass if is_sharp(r)]
    wide_pass = [r for r in settled_pass if not is_sharp(r)]

    # --- MAJOR-2: exclude informational cells from the headline cohort -----
    sharp_enabled = [r for r in sharp_pass if _cell_of(r) not in informational_set]
    sharp_informational = [r for r in sharp_pass if _cell_of(r) in informational_set]

    # --- reliability tables ------------------------------------------------
    def _bins(recs: list[PredictionRecord]) -> list[dict[str, Any]]:
        probs, outcomes = _pairs(recs, outcome_by_key)
        return compute_reliability_bins(probs, outcomes, ASOS_BINS)

    # --- §9 headline verdict: SHARP band, ENABLED cells, deduped -----------
    kill = _kill_for_cohort(sharp_enabled, outcome_by_key)

    # --- per-cell §9 verdicts (the live flip is per-cell, spec §8.3) -------
    cells: dict[str, list[PredictionRecord]] = defaultdict(list)
    for r in sharp_pass:
        cells[_cell_of(r)].append(r)
    by_cell: dict[str, Any] = {}
    for cell in sorted(cells):
        cell_recs = cells[cell]
        cell_kill = _kill_for_cohort(cell_recs, outcome_by_key)
        probs_c, _ = _pairs(cell_recs, outcome_by_key)
        by_cell[cell] = {
            "n": len(probs_c),
            "informational": cell in informational_set,
            "kill_criteria": cell_kill,
            "reliability": _bins(cell_recs),
        }

    return {
        "summary": {
            "n_total_strategy_rows": n_asos_rows_pre_dedup,
            "n_settlement_events": n_settlement_events,
            "n_settled_all_guard": len(settled_all),
            "n_settled_trade_eligible": len(settled_pass),
            "n_settled_sharp_pass": len(sharp_pass),
            "n_settled_wide_pass": len(wide_pass),
            "n_settled_sharp_enabled": len(sharp_enabled),
            "n_settled_sharp_informational": len(sharp_informational),
            "n_pending": pending_count,
        },
        # PRIMARY: sharp-band, enabled-cell reliability table (the §9 cohort).
        "reliability_sharp_enabled": _bins(sharp_enabled),
        # DIAGNOSTIC: wide-σ band (excluded from headline — CRITICAL-2).
        "reliability_wide_diagnostic": _bins(wide_pass),
        # DIAGNOSTIC: informational cells (LAX), kept separate (MAJOR-2).
        "reliability_informational": _bins(sharp_informational),
        # §9 kill criteria — HEADLINE = sharp band, enabled cells only.
        "kill_criteria": kill,
        # Per-cell §9 verdicts (live flip is per-cell).
        "by_cell": by_cell,
    }


# ---------------------------------------------------------------------------
# Console renderer
# ---------------------------------------------------------------------------


def _fmt_float(v: float | None, prec: int = 4) -> str:
    return f"{v:.{prec}f}" if v is not None else "n/a"


def _render_kill_block(lines: list[str], kc: dict[str, Any], indent: str = "  ") -> None:
    for key in ("low_p_no_bin", "high_p_bin"):
        v = kc[key]
        lines.append(f"{indent}[{v['verdict']:>14}]  {v['criterion']}")
        ci = (
            f"CI=[{_fmt_float(v['wilson_lo'], 3)},{_fmt_float(v['wilson_hi'], 3)}]"
            if v["wilson_lo"] is not None
            else "CI=n/a"
        )
        lines.append(
            f"{indent}                 n={v['n']}  "
            f"real={_fmt_float(v['mean_real'])}  "
            f"pred={_fmt_float(v['mean_pred'])}  {ci}  "
            f"half_width={_fmt_float(v['wilson_half_width'], 3)}"
        )
    cal = kc["calibration_in_large"]
    lines.append(
        f"{indent}[SECONDARY ONLY]  calibration-in-large: "
        f"n={cal['n']}  |pred−real|={_fmt_float(cal['abs_gap'])}"
    )


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    s = report["summary"]
    pr = report["parse_reconciliation"]
    lines.append("=== INTRADAY ASOS HIGH SHADOW RELIABILITY REPORT ===")
    lines.append(f"generated: {report['generated_at']}")
    lines.append("HEADLINE: CALIBRATION on SHARP band, ENABLED cells, DEDUPED settlements")
    lines.append(
        f"sharp σ_gain ≤ {report['config']['sharp_sigma_gain_max']} °F  |  "
        f"informational (excluded): {report['config']['informational_cells']}"
    )
    lines.append("")

    # --- FRAME CUT (Stage 0a) ---
    fp = report.get("frame_partition")
    if fp is not None:
        lines.append("--- FRAME PARTITION (Stage 0a — frames NOT pooled) ---")
        lines.append(
            f"BINDING = frame_version {fp['binding_frame_version']} (LST climate-day frame)"
        )
        lines.append(
            f"frame-v2 (LST, BINDING):   rows={fp['frame_v2_rows']}  "
            f"settlement_events={fp['frame_v2_settlement_events']}"
        )
        lines.append(
            f"frame-v1 (civil, DIAG):    rows={fp['frame_v1_rows']}  "
            f"settlement_events={fp['frame_v1_settlement_events']}"
        )
        lines.append("")

    # --- parse-loss reconciliation (MINOR) ---
    lines.append("--- PARSE RECONCILIATION ---")
    lines.append(f"total raw lines:            {pr['total_raw_lines']}")
    lines.append(f"parsed records:             {pr['parsed_records']}")
    lines.append(f"rows dropped (malformed):   {pr['rows_dropped']}")
    lines.append(f"intraday_asos_high records: {pr['intraday_asos_high_records']}")
    lines.append(f"deduped settlement events:  {pr['deduped_settlement_events']}")
    lines.append("")

    if s["n_settled_sharp_pass"] == 0 and s["n_settled_trade_eligible"] == 0:
        lines.append(
            f"0 settled frame-v2 (LST) predictions (trade-eligible). "
            f"settlement_events={s['n_settlement_events']}  pending={s['n_pending']}"
        )
        lines.append(
            "No BINDING (frame-v2) data to report.  Re-run after the LST frame "
            "accrues ~10 weeks of forward shadow."
        )
        lines.append("")
        # Still surface the legacy frame-v1 cohort as a clearly-labelled
        # diagnostic — right after the frame fix this is where ALL the data is.
        _render_frame_v1_diagnostic(lines, report)
        return "\n".join(lines)

    lines.append("--- SUMMARY (settlement events, post-dedup) ---")
    lines.append(f"settlement events:           {s['n_settlement_events']}")
    lines.append(f"settled (all guard):         {s['n_settled_all_guard']}")
    lines.append(f"settled trade-eligible:      {s['n_settled_trade_eligible']}")
    lines.append(f"  ├─ sharp band:             {s['n_settled_sharp_pass']}")
    lines.append(f"  │   ├─ enabled (headline): {s['n_settled_sharp_enabled']}")
    lines.append(f"  │   └─ informational:      {s['n_settled_sharp_informational']}")
    lines.append(f"  └─ wide band (diag):       {s['n_settled_wide_pass']}")
    lines.append(f"pending (no settlement):     {s['n_pending']}")
    lines.append("")

    def _render_bins(label: str, bins: list[dict[str, Any]]) -> None:
        lines.append(f"--- {label} ---")
        if not bins:
            lines.append("  (no data)")
        else:
            hdr = f"  {'bin':>14}  {'n':>5}  {'mean_pred':>10}  {'mean_real':>10}"
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            for row in bins:
                lines.append(
                    f"  {row['bin']:>14}  {row['n']:>5}  {row['pred']:>10.4f}  {row['real']:>10.4f}"
                )
        lines.append("")

    _render_bins(
        "RELIABILITY — SHARP band, ENABLED cells (HEADLINE COHORT)",
        report["reliability_sharp_enabled"],
    )
    _render_bins(
        "RELIABILITY — WIDE σ band (diagnostic, excluded from headline)",
        report["reliability_wide_diagnostic"],
    )
    _render_bins(
        "RELIABILITY — informational cells (LAX, never traded)",
        report["reliability_informational"],
    )

    lines.append("--- §9 KILL CRITERIA — HEADLINE (sharp band, enabled cells) ---")
    _render_kill_block(lines, report["kill_criteria"])
    lines.append("")

    if report["by_cell"]:
        lines.append("--- §9 KILL CRITERIA — PER CELL (live flip is per-cell) ---")
        for cell, v in sorted(report["by_cell"].items()):
            tag = " [informational]" if v["informational"] else ""
            lines.append(f"  ◆ {cell}{tag}  (n={v['n']})")
            _render_kill_block(lines, v["kill_criteria"], indent="      ")
            lines.append("")

    _render_frame_v1_diagnostic(lines, report)

    return "\n".join(lines)


def _render_frame_v1_diagnostic(lines: list[str], report: dict[str, Any]) -> None:
    """Render the legacy civil-frame (frame_version 1) cohort, clearly labelled.

    This is a DIAGNOSTIC only — the contaminated boundary hour means it is NOT
    decision-grade.  It is shown so the operator can see how much of the corpus
    is pre-fix and how the legacy frame's bins compare, never to gate the flip.
    """
    fv1 = report.get("frame_v1_diagnostic")
    if fv1 is None:
        return
    s1 = fv1["summary"]
    lines.append(
        "=== FRAME-v1 DIAGNOSTIC (legacy civil/DST frame — NOT DECISION-GRADE) ==="
    )
    if s1["n_settled_sharp_pass"] == 0 and s1["n_settled_trade_eligible"] == 0:
        lines.append(
            f"frame-v1 settlement_events={s1['n_settlement_events']}  "
            f"pending={s1['n_pending']}  (no trade-eligible settled rows)"
        )
        lines.append("")
        return
    lines.append(
        f"settlement events={s1['n_settlement_events']}  "
        f"settled trade-eligible={s1['n_settled_trade_eligible']}  "
        f"sharp-enabled={s1['n_settled_sharp_enabled']}  pending={s1['n_pending']}"
    )
    bins = fv1["reliability_sharp_enabled"]
    lines.append("--- frame-v1 RELIABILITY — SHARP band, ENABLED cells (DIAGNOSTIC) ---")
    if not bins:
        lines.append("  (no data)")
    else:
        hdr = f"  {'bin':>14}  {'n':>5}  {'mean_pred':>10}  {'mean_real':>10}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for row in bins:
            lines.append(
                f"  {row['bin']:>14}  {row['n']:>5}  {row['pred']:>10.4f}  {row['real']:>10.4f}"
            )
    lines.append("--- frame-v1 §9 KILL CRITERIA (DIAGNOSTIC, do not act on) ---")
    _render_kill_block(lines, fv1["kill_criteria"])
    lines.append("")


# ---------------------------------------------------------------------------
# VPS pull helpers
# ---------------------------------------------------------------------------


def pull_predictions_from_vps(local_path: Path) -> None:
    """SCP the predictions.jsonl from the VPS."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{require_vps_host()}:{VPS_PREDICTIONS_PATH}"
    print("Pulling predictions.jsonl from VPS...")
    result = subprocess.run(
        ["scp", remote, str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        sys.exit(1)
    n = sum(1 for line in local_path.read_text().splitlines() if line.strip())
    print(f"  -> {n} prediction rows saved to {local_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Intraday ASOS HIGH shadow reliability report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Headline is CALIBRATION on the SHARP σ band, ENABLED cells, DEDUPED\n"
            "to one row per settlement event.  §9 verdicts use a Wilson-95% CI:\n"
            "  low-p NO bin: FAIL if CI-lower > pred+0.05, PASS if CI-upper ≤ pred+0.05;\n"
            "  high-p bin:   FAIL if CI-upper < 0.85, PASS if CI-lower ≥ 0.85;\n"
            "  else INSUFFICIENT-N / INCONCLUSIVE."
        ),
    )
    p.add_argument(
        "--pull-vps",
        action="store_true",
        help="SCP latest predictions.jsonl from VPS before reporting",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Pull audit files from VPS (mirrors intraday_shadow_report.py convention)",
    )
    p.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
        help=f"Path to predictions.jsonl (default: {DEFAULT_PREDICTIONS})",
    )
    p.add_argument(
        "--snapshots",
        default=str(DEFAULT_SNAPSHOTS),
        help=f"Snapshots dir for NWS settlements (default: {DEFAULT_SNAPSHOTS})",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Config file for informational_cells (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--informational-cells",
        default=None,
        help=(
            "Comma-separated cells to exclude from the §9 headline cohort "
            "(overrides config; default reads strategies.intraday_asos_high.informational_cells)"
        ),
    )
    p.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write structured JSON report to this path",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    predictions_path = Path(args.predictions)

    if args.pull_vps:
        pull_predictions_from_vps(predictions_path)

    if args.refresh:
        audit_dir = _PROJECT_ROOT / "data" / "vps_audit"
        pull_audit_files(audit_dir)

    if args.informational_cells is not None:
        informational_cells = [c.strip() for c in args.informational_cells.split(",") if c.strip()]
    else:
        informational_cells = load_informational_cells(Path(args.config))

    report = build_report(
        predictions_path,
        Path(args.snapshots),
        informational_cells=informational_cells,
    )
    print(render_report(report))

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"JSON report written to {out_path}")


if __name__ == "__main__":
    main()
