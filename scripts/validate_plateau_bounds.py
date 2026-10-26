"""Validate the ``block_raw_prob_range`` config against a live isotonic artifact.

Reads ``data/calibration/isotonic.json`` (or whatever ``--isotonic`` points
at), walks maximal runs of consecutive breakpoints sharing the same value,
and identifies tripwire-dangerous plateaus (``width >= MIN_WIDTH`` AND
``|value - center| >= MIN_DEVIATION``). Mirrors the geometry definition
in ``python.analytics.audit_tripwires.check_calibrator_plateau_coverage``.

**Held-out reliability check (F1.4):** After identifying the dangerous
plateau, this script now also validates the proposed ``block_raw_prob_range``
against settled pairs strictly *after* the fit's ``max(forecast_date)`` — the
in-sample artifact alone cannot justify suppressing trades; the empirical
observed-YES rate in the band must agree with the artifact's plateau value.

Gate (per F1.4): with ``n >= MIN_HELD_OUT_N`` (30) pairs in the proposed band:

* If ``|empirical_rate - plateau_value| < MAX_PLATEAU_DEVIATION`` (0.10) the
  bound is corroborated.
* Otherwise the bound is contradicted by held-out data — exit non-zero so a
  refit surfaces a structural drift before shipping.

With ``n < MIN_HELD_OUT_N`` the gate WARNS (stderr) but does not fail — refits
during corpus accumulation must not be blocked. The held-out window is
selected by ``--held-out-after`` (default: the artifact's own ``fit_date``).

Exit codes:

* ``0`` — exactly one dangerous plateau AND held-out check passes (or warns
  on insufficient n); proposed ``block_raw_prob_range`` printed to stdout.
* ``1`` — multiple dangerous plateaus; current single-range config schema
  cannot cover them. Plateau-eval plan §B.3 schema extension is required
  before shipping.
* ``2`` — zero dangerous plateaus; the filter would be a no-op on this
  artifact. Defensive exit — the caller must decide whether to disable
  ``block_raw_prob_range`` rather than silently passing.
* ``4`` — held-out reliability check FAILS at ``n >= MIN_HELD_OUT_N``:
  empirical rate diverges from the artifact's plateau value by more than
  ``MAX_PLATEAU_DEVIATION``. Operator decision required.
* ``5`` — floor-coupling gate FAILS: artifact ``values[0]`` exceeds
  ``min_calibrated_prob - FLOOR_DEFEAT_EPSILON``, meaning the configured
  calibrated-probability floor would no longer bite at the bottom of the
  curve. Refit must either lower the curve floor or the operator must
  raise ``min_calibrated_prob`` (separate decision). Added 2026-05-20
  after a live floor-bypass incident.

Usage::

    uv run python scripts/validate_plateau_bounds.py
    uv run python scripts/validate_plateau_bounds.py --isotonic path/to/isotonic.json
    uv run python scripts/validate_plateau_bounds.py --skip-held-out-check
    uv run python scripts/validate_plateau_bounds.py --skip-floor-check
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Tripwire thresholds — matches
# python/analytics/audit_tripwires.py::check_calibrator_plateau_coverage
# (currently min_width=0.25, min_center_deviation=0.15).
MIN_WIDTH = 0.25
MIN_DEVIATION = 0.15

# Held-out reliability gate (F1.4).
MIN_HELD_OUT_N = 30
MAX_PLATEAU_DEVIATION = 0.10
BOOTSTRAP_N = 5000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

# Floor / curve-bottom coupling gate (added 2026-05-20).
# The invariant: ``artifact["values"][0] <= min_calibrated_prob - FLOOR_DEFEAT_EPSILON``.
# Why epsilon=0.01: we need a strictly-less-than semantic with a small numerical
# safety margin. Picking 0 would admit ``values[0] == floor`` curves where the
# floor binds on its edge (no meaningful suppression); picking too large a
# value would false-positive on benign refits whose curve floor happens to be
# close to but still below the floor (e.g. 0.092 vs 0.10 — fine in practice).
# 0.01 separates "0.0797 passes / 0.115 fails" cleanly on the artifacts on
# disk as of the incident.
FLOOR_DEFEAT_EPSILON = 0.01


def find_dangerous_plateaus(
    breakpoints: list[float],
    values: list[float],
) -> list[tuple[float, float, float, float, float]]:
    """Return ``[(low, high, value, width, deviation), ...]`` for dangerous plateaus.

    A plateau is a maximal run of consecutive entries where
    ``values[j+1] == values[i]``. A plateau is *dangerous* when
    ``width >= MIN_WIDTH`` AND ``|value - center| >= MIN_DEVIATION``.
    """
    if len(breakpoints) != len(values):
        raise ValueError(
            f"breakpoints ({len(breakpoints)}) and values ({len(values)}) length mismatch"
        )

    dangerous: list[tuple[float, float, float, float, float]] = []
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1] == values[i]:
            j += 1
        if j > i:
            low = float(breakpoints[i])
            high = float(breakpoints[j])
            value = float(values[i])
            width = high - low
            center = (high + low) / 2.0
            deviation = abs(value - center)
            if width >= MIN_WIDTH and deviation >= MIN_DEVIATION:
                dangerous.append((low, high, value, width, deviation))
        i = j + 1
    return dangerous


def isotonic_value_at(
    breakpoints: list[float],
    values: list[float],
    x: float,
) -> float:
    """Linear interpolation matching ``python.data.calibration.IsotonicCalibrator``.

    Used to look up the artifact's "expected" value at the proposed band's
    midpoint — that's the value the held-out empirical rate has to agree with
    for the suppression to be earned.
    """
    if not breakpoints or not values:
        raise ValueError("breakpoints / values must be non-empty")
    if x <= breakpoints[0]:
        return values[0]
    if x >= breakpoints[-1]:
        return values[-1]
    lo, hi = 0, len(breakpoints) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if breakpoints[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = breakpoints[lo], breakpoints[hi]
    y0, y1 = values[lo], values[hi]
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def cluster_bootstrap_ci(
    pairs: Sequence[tuple[float, int, date]],
    *,
    n_boot: int = BOOTSTRAP_N,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Cluster bootstrap 95% CI on the mean outcome.

    Resample unit is ``forecast_date`` — F1.2 in the open-items doc cautions
    against row-bootstrap because intra-day pairs (city × var × strike) share
    the same weather realization and are NOT independent. Cluster-on-date
    treats each forecast_date as the iid unit.

    Returns ``(lo, hi)`` of the resampled-mean distribution. When pairs is
    empty returns ``(nan, nan)``.
    """
    if not pairs:
        return (float("nan"), float("nan"))
    by_date: dict[date, list[int]] = {}
    for _p, o, d in pairs:
        by_date.setdefault(d, []).append(int(o))
    dates = list(by_date.keys())
    if not dates:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means: list[float] = []
    n_dates = len(dates)
    for _ in range(n_boot):
        sampled = rng.choices(dates, k=n_dates)
        total = 0
        n = 0
        for d in sampled:
            outs = by_date[d]
            total += sum(outs)
            n += len(outs)
        if n > 0:
            means.append(total / n)
    if not means:
        return (float("nan"), float("nan"))
    means.sort()
    lo_idx = int(alpha / 2 * len(means))
    hi_idx = min(int((1 - alpha / 2) * len(means)), len(means) - 1)
    return (means[lo_idx], means[hi_idx])


def held_out_reliability_check(
    *,
    pairs: Sequence[tuple[float, int, date]],
    low: float,
    high: float,
    plateau_value: float,
    held_out_after: date,
    min_n: int = MIN_HELD_OUT_N,
    max_deviation: float = MAX_PLATEAU_DEVIATION,
) -> dict[str, Any]:
    """Pure held-out gate.

    ``pairs`` is ``[(raw_prob, outcome, forecast_date), ...]`` already restricted
    to forecast_dates strictly after ``held_out_after``. This function does NOT
    re-filter on date — callers are responsible for that. (Keeping the filter
    outside makes the gate trivially testable with synthetic data.)

    Returns a result dict with keys:

    * ``status`` — ``"ok" | "fail" | "warn_insufficient_n"``
    * ``n`` — pairs in ``[low, high]``
    * ``empirical_rate`` — observed YES rate (nan if n==0)
    * ``plateau_value`` — artifact value at band midpoint (passed-in)
    * ``deviation`` — ``|empirical_rate - plateau_value|`` (nan if n==0)
    * ``ci_lo`` / ``ci_hi`` — cluster-bootstrap 95% CI (nan if n==0)
    * ``min_n`` / ``max_deviation`` — thresholds used
    * ``held_out_after`` — ISO date string
    * ``low`` / ``high`` — band
    * ``message`` — human-readable summary
    """
    in_band = [(p, int(o), d) for (p, o, d) in pairs if low <= p <= high]
    n = len(in_band)
    held_out_after_str = held_out_after.isoformat()

    if n == 0:
        return {
            "status": "warn_insufficient_n",
            "n": 0,
            "empirical_rate": float("nan"),
            "plateau_value": plateau_value,
            "deviation": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "min_n": min_n,
            "max_deviation": max_deviation,
            "held_out_after": held_out_after_str,
            "low": low,
            "high": high,
            "message": (
                f"WARN: 0 held-out pairs in [{low:.3f}, {high:.3f}] after "
                f"{held_out_after_str}. Gate cannot fire; continue refits "
                "but accumulate corpus."
            ),
        }

    empirical = sum(o for _p, o, _d in in_band) / n
    deviation = abs(empirical - plateau_value)
    ci_lo, ci_hi = cluster_bootstrap_ci(in_band)

    if n < min_n:
        return {
            "status": "warn_insufficient_n",
            "n": n,
            "empirical_rate": empirical,
            "plateau_value": plateau_value,
            "deviation": deviation,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "min_n": min_n,
            "max_deviation": max_deviation,
            "held_out_after": held_out_after_str,
            "low": low,
            "high": high,
            "message": (
                f"WARN: only n={n} held-out pairs in [{low:.3f}, {high:.3f}] "
                f"after {held_out_after_str} (need >={min_n} to fire gate). "
                f"Observed empirical_rate={empirical:.4f}, plateau_value="
                f"{plateau_value:.4f}, deviation={deviation:.4f}, "
                f"95% cluster-bootstrap CI=[{ci_lo:.4f}, {ci_hi:.4f}]. "
                "Held-out signal directional only."
            ),
        }

    if deviation < max_deviation:
        status = "ok"
        message = (
            f"OK: n={n} held-out pairs in [{low:.3f}, {high:.3f}] after "
            f"{held_out_after_str}; empirical_rate={empirical:.4f} agrees "
            f"with plateau_value={plateau_value:.4f} (deviation={deviation:.4f}"
            f" < {max_deviation:.2f}); 95% cluster-bootstrap CI="
            f"[{ci_lo:.4f}, {ci_hi:.4f}]."
        )
    else:
        status = "fail"
        message = (
            f"FAIL: n={n} held-out pairs in [{low:.3f}, {high:.3f}] after "
            f"{held_out_after_str}; empirical_rate={empirical:.4f} diverges "
            f"from plateau_value={plateau_value:.4f} by deviation="
            f"{deviation:.4f} >= {max_deviation:.2f}; 95% cluster-bootstrap CI="
            f"[{ci_lo:.4f}, {ci_hi:.4f}]. The proposed block range is no "
            "longer corroborated by out-of-sample data — refit + operator "
            "review required before shipping."
        )

    return {
        "status": status,
        "n": n,
        "empirical_rate": empirical,
        "plateau_value": plateau_value,
        "deviation": deviation,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "min_n": min_n,
        "max_deviation": max_deviation,
        "held_out_after": held_out_after_str,
        "low": low,
        "high": high,
        "message": message,
    }


def floor_coupling_check(
    *,
    values_zero: float,
    min_calibrated_prob: float | None,
    epsilon: float = FLOOR_DEFEAT_EPSILON,
) -> dict[str, Any]:
    """Pure floor/curve-bottom coupling gate.

    Fails when the candidate artifact's ``values[0]`` (the calibrated
    probability at raw_ensemble_prob=0) is high enough that the configured
    ``min_calibrated_prob`` floor would stop biting at the bottom of the
    curve. Symbolic invariant::

        values[0] <= min_calibrated_prob - epsilon

    A violation means trades the floor was previously suppressing will
    fire after the new artifact promotes. Refit must either rework the
    curve so its floor lies below ``min_calibrated_prob - epsilon`` or
    the operator must explicitly raise ``min_calibrated_prob`` (a
    separate decision — this function deliberately does NOT propose a
    new floor value).

    When ``min_calibrated_prob`` is ``None`` (config-level floor disabled
    entirely) the gate is a no-op and returns ``status="skipped"`` —
    callers must treat that as "not failed" but should not infer the gate
    passed (there is no configured floor to defeat).

    Returns a dict with keys ``status`` (``"ok" | "fail" | "skipped"``),
    ``values_zero``, ``min_calibrated_prob``, ``epsilon``, ``threshold``
    (``= min_calibrated_prob - epsilon`` or ``None`` when skipped), and
    ``message`` (human-readable).
    """
    if min_calibrated_prob is None:
        return {
            "status": "skipped",
            "values_zero": values_zero,
            "min_calibrated_prob": None,
            "epsilon": epsilon,
            "threshold": None,
            "message": (
                "SKIPPED: strategies.weather.min_calibrated_prob is null in "
                "config; no configured floor to defeat. Floor-coupling gate "
                "is a no-op."
            ),
        }

    threshold = float(min_calibrated_prob) - float(epsilon)
    if values_zero > threshold:
        message = (
            f"FAIL: artifact values[0]={values_zero:.6f} exceeds threshold "
            f"{threshold:.6f} (= min_calibrated_prob={float(min_calibrated_prob):.3f}"
            f" - epsilon={epsilon:.3f}). The curve floor would defeat the "
            f"configured min_calibrated_prob floor — trades intended to be "
            f"suppressed would fire. Fix: either rework the fit so the curve "
            f"floor lands below {threshold:.4f}, OR explicitly raise "
            f"min_calibrated_prob to >= {values_zero + epsilon:.4f} as a "
            f"separate operator decision."
        )
        return {
            "status": "fail",
            "values_zero": values_zero,
            "min_calibrated_prob": float(min_calibrated_prob),
            "epsilon": epsilon,
            "threshold": threshold,
            "message": message,
        }

    return {
        "status": "ok",
        "values_zero": values_zero,
        "min_calibrated_prob": float(min_calibrated_prob),
        "epsilon": epsilon,
        "threshold": threshold,
        "message": (
            f"OK: artifact values[0]={values_zero:.6f} <= threshold "
            f"{threshold:.6f} (= min_calibrated_prob="
            f"{float(min_calibrated_prob):.3f} - epsilon={epsilon:.3f}); "
            f"configured floor will continue to bite at the bottom of the "
            f"curve."
        ),
    }


def _load_min_calibrated_prob_from_config(config_path: Path) -> float | None:
    """Read ``strategies.weather.min_calibrated_prob`` from a YAML config.

    Returns the configured float (incl. 0.0) or ``None`` if the field is
    explicitly null. Raises ``ValueError`` if the file is missing/unreadable
    or the key path is absent — promotion gating must fail loud rather than
    silently skip the floor check.
    """
    # Defer yaml import so non-promotion paths don't pay the import cost.
    import yaml  # noqa: PLC0415

    if not config_path.exists():
        raise ValueError(
            f"Config file not found: {config_path}. Floor-coupling gate "
            "requires an accessible config to read min_calibrated_prob."
        )
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Config file {config_path} is not valid YAML: {exc}") from exc

    try:
        weather = cfg["strategies"]["weather"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Config file {config_path} is missing the "
            "'strategies.weather' section required for floor-coupling check."
        ) from exc

    if "min_calibrated_prob" not in weather:
        raise ValueError(
            f"Config file {config_path} is missing "
            "'strategies.weather.min_calibrated_prob'. Add the key explicitly "
            "(use null to disable) or pass --skip-floor-check."
        )

    value = weather["min_calibrated_prob"]
    if value is None:
        return None
    return float(value)


def _load_held_out_pairs_from_snapshots(
    *,
    snapshots_dir: Path,
    held_out_after: date,
    ensemble_weights: dict[str, float] | None,
    bias_table_path: Path | None,
) -> list[tuple[float, int, date]]:
    """Build production-mirrored pairs and filter to forecast_date > held_out_after.

    Returns ``[(raw_prob, outcome, forecast_date), ...]`` ready for the gate.
    Imports are deferred so the unit-tested gate path doesn't pull the full
    production stack.
    """
    # Defer heavy imports so unit tests of the gate logic don't need numpy/scipy.
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from python.data.bias_correction import BiasTable  # noqa: PLC0415
    from scripts._calibration_common import (  # noqa: PLC0415
        build_production_mirrored_pairs,
        load_snapshot_data,
    )

    forecasts, settlements, markets = load_snapshot_data(snapshots_dir)

    bias_table: BiasTable | None = None
    if bias_table_path is not None and bias_table_path.exists():
        candidate = BiasTable(bias_table_path)
        if candidate.loaded:
            bias_table = candidate

    if ensemble_weights is None:
        ensemble_weights = {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45}

    pairs: list[tuple[float, int, date]] = []
    for p, o, _c, _v, _st, d in build_production_mirrored_pairs(
        forecasts=forecasts,
        settlements=settlements,
        markets=markets,
        bias_table=bias_table,
        ensemble_weights=ensemble_weights,
        spread_inflation_overrides=None,
    ):
        if d > held_out_after:
            pairs.append((float(p), int(o), d))
    return pairs


def _resolve_held_out_after(raw: dict[str, Any], explicit: str | None) -> date | None:
    """Resolve the held-out cutoff date.

    Priority: ``--held-out-after`` CLI value > artifact ``fit_date`` field.
    Returns ``None`` if neither is parseable (caller treats as "no held-out
    check available" — skipped, not failed).

    **Composition with F1.5** (``scripts/fit_isotonic.py``, ``scripts/fit_emos.py``):
    F1.5 changed ``fit_date`` from wall-clock ``date.today()`` to
    ``max(corpus.forecast_date)``. With F1.5 active, ``fit_date`` is the
    intended cutoff (held-out window starts the day AFTER the last training
    day). Without F1.5, ``fit_date == today`` means the held-out window
    extends only into the future → ``n==0`` and the gate silently warns
    forever. If you see ``status=warn_insufficient_n`` on artifacts that
    should have plenty of held-out data, check F1.5 is in the fitter.
    """
    if explicit:
        try:
            return datetime.strptime(explicit, "%Y-%m-%d").date()
        except ValueError:
            return None
    fit_date = raw.get("fit_date")
    if isinstance(fit_date, str):
        try:
            return datetime.strptime(fit_date, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--isotonic",
        type=Path,
        default=Path("data/calibration/isotonic.json"),
        help="Path to the isotonic calibrator JSON artifact.",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=Path("data/snapshots"),
        help="Snapshot directory for held-out pair construction.",
    )
    parser.add_argument(
        "--bias-table",
        type=Path,
        default=Path("data/gfs_bias/bias_correction.json"),
        help="Bias-correction table for production-mirrored pair construction.",
    )
    parser.add_argument(
        "--held-out-after",
        type=str,
        default=None,
        help=(
            "Cutoff date (YYYY-MM-DD); only forecast_date strictly > this "
            "value is held-out. Defaults to the artifact's fit_date."
        ),
    )
    parser.add_argument(
        "--skip-held-out-check",
        action="store_true",
        help=(
            "Skip the held-out reliability check entirely (use only when "
            "snapshots aren't available, e.g., CI dry-runs). Plateau-only "
            "exit codes apply."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("python/config.yaml"),
        help=(
            "Path to the YAML config file the floor-coupling gate reads "
            "'strategies.weather.min_calibrated_prob' from."
        ),
    )
    parser.add_argument(
        "--skip-floor-check",
        action="store_true",
        help=(
            "Skip the floor / curve-bottom coupling gate. Use only when "
            "validating against a config-less artifact (rare; this gate is "
            "the operator's primary protection against promoting a curve "
            "that defeats min_calibrated_prob — see exit code 5)."
        ),
    )
    args = parser.parse_args()

    raw = json.loads(args.isotonic.read_text())
    breakpoints = raw.get("breakpoints")
    values = raw.get("values")
    if not isinstance(breakpoints, list) or not isinstance(values, list):
        print(
            f"ERROR: {args.isotonic} is missing 'breakpoints' or 'values' arrays.",
            file=sys.stderr,
        )
        return 3

    # === Floor / curve-bottom coupling gate (exit 5) ===
    # Runs BEFORE plateau analysis so a curve that silently defeats
    # min_calibrated_prob fails fast — the operator's primary protection
    # against the 2026-05-19 incident (values[0]=0.1153 vs configured
    # floor 0.10 → four long-shot YES trades fired).
    if not values:
        print(
            f"ERROR: {args.isotonic} has empty 'values' array — cannot "
            "evaluate floor-coupling gate.",
            file=sys.stderr,
        )
        return 3

    if args.skip_floor_check:
        print(
            "Floor-coupling gate: SKIPPED (--skip-floor-check).",
            file=sys.stderr,
        )
    else:
        try:
            min_cal_prob = _load_min_calibrated_prob_from_config(args.config)
        except ValueError as exc:
            print(f"ERROR: floor-coupling gate could not load config: {exc}", file=sys.stderr)
            return 3

        floor_result = floor_coupling_check(
            values_zero=float(values[0]),
            min_calibrated_prob=min_cal_prob,
            epsilon=FLOOR_DEFEAT_EPSILON,
        )
        print("=== Floor / curve-bottom coupling gate ===")
        print(floor_result["message"])
        print("")
        if floor_result["status"] == "fail":
            return 5

    dangerous = find_dangerous_plateaus(breakpoints, values)

    if len(dangerous) == 0:
        print(
            "WARNING: no tripwire-dangerous plateau found in "
            f"{args.isotonic}. Filter would be a no-op on this artifact; "
            "caller must decide whether to disable block_raw_prob_range "
            "rather than silently shipping a stale range.",
            file=sys.stderr,
        )
        return 2

    if len(dangerous) > 1:
        print(
            f"ERROR: {len(dangerous)} dangerous plateaus found in "
            f"{args.isotonic}. Current single-range schema for "
            "block_raw_prob_range cannot cover them — see plateau-eval "
            "plan §B.3 for the schema-extension requirement.",
            file=sys.stderr,
        )
        for low, high, value, width, deviation in dangerous:
            print(
                f"  [{low:.3f}, {high:.3f}] -> {value:.4f} "
                f"(width={width:.3f}, |val-center|={deviation:.3f})",
                file=sys.stderr,
            )
        return 1

    low, high, value, width, deviation = dangerous[0]
    print(
        f"Dangerous plateau: [{low:.3f}, {high:.3f}] -> {value:.4f} "
        f"(width={width:.3f}, |val-center|={deviation:.3f})"
    )
    print(f"block_raw_prob_range: [{low:.3f}, {high:.3f}]")

    if args.skip_held_out_check:
        print(
            "Held-out reliability check: SKIPPED (--skip-held-out-check).",
            file=sys.stderr,
        )
        return 0

    cutoff = _resolve_held_out_after(raw, args.held_out_after)
    if cutoff is None:
        print(
            "Held-out reliability check: SKIPPED (no parseable cutoff; "
            "artifact missing 'fit_date' and --held-out-after not given).",
            file=sys.stderr,
        )
        return 0

    if not args.snapshots.exists():
        print(
            f"Held-out reliability check: SKIPPED (snapshots dir {args.snapshots} not found).",
            file=sys.stderr,
        )
        return 0

    plateau_value = isotonic_value_at(breakpoints, values, (low + high) / 2.0)

    pairs = _load_held_out_pairs_from_snapshots(
        snapshots_dir=args.snapshots,
        held_out_after=cutoff,
        ensemble_weights=None,
        bias_table_path=args.bias_table,
    )
    result = held_out_reliability_check(
        pairs=pairs,
        low=low,
        high=high,
        plateau_value=plateau_value,
        held_out_after=cutoff,
    )
    print("")
    print("=== Held-out reliability check (F1.4) ===")
    print(result["message"])
    print(
        f"  n={result['n']}, empirical_rate={result['empirical_rate']:.4f}, "
        f"plateau_value={result['plateau_value']:.4f}, "
        f"deviation={result['deviation']:.4f}, "
        f"CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}], "
        f"held_out_after={result['held_out_after']}, "
        f"min_n={result['min_n']}, max_deviation={result['max_deviation']}"
    )

    if result["status"] == "fail":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
