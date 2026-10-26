"""Pre-deploy validation gate for isotonic calibrator artifacts.

Splits production-mirrored pairs by forecast date: last ``holdout_days`` form
the held-out test set; the remainder are training. Refits a candidate model on
the training portion and scores it on holdout. Compared to:
  - raw-input Brier on holdout (calibration-helps gate, load-bearing)
  - live baseline calibrator's Brier on the same holdout (advisory context)

Plateau width and output-range checks are applied to the input candidate
artifact (which was fit on full data, not the training portion).

Usage::

    uv run python scripts/validate_calibrator.py \\
        --candidate data/calibration/isotonic_v3.staging.json \\
        --baseline data/calibration/isotonic.json \\
        --holdout-days 14

Requires sklearn for the on-the-fly refit on the training portion. Install via::

    uv sync --extra ml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from python.data.bias_correction import BiasTable

logger = logging.getLogger(__name__)


def _detect_plateaus(
    breakpoints: list[float],
    values: list[float],
    *,
    width_threshold: float,
) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(values) - 1:
        j = i
        while j + 1 < len(values) and abs(values[j + 1] - values[i]) < 1e-6:
            j += 1
        if j > i:
            lo = breakpoints[i]
            hi = breakpoints[j + 1] if j + 1 < len(breakpoints) else breakpoints[j]
            width = hi - lo
            if width > width_threshold:
                out.append({"lo": lo, "hi": hi, "value": values[i], "width": width})
        i = j + 1 if j > i else i + 1
    return out


def _interpolate(raw: float, breakpoints: list[float], values: list[float]) -> float:
    """Pure-Python isotonic predict — same algorithm as IsotonicCalibrator."""
    if raw <= breakpoints[0]:
        return values[0]
    if raw >= breakpoints[-1]:
        return values[-1]
    for k in range(len(breakpoints) - 1):
        if breakpoints[k] <= raw <= breakpoints[k + 1]:
            x0, x1 = breakpoints[k], breakpoints[k + 1]
            y0, y1 = values[k], values[k + 1]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (raw - x0) / (x1 - x0)
    return values[-1]


def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _load_pairs_with_dates(
    *,
    snapshots_dir: Path,
    bias_table_path: Path | None,
    ensemble_weights: dict[str, float],
) -> list[tuple]:
    from python.data.bias_correction import BiasTable
    from scripts._calibration_common import (
        build_production_mirrored_pairs,
        load_snapshot_data,
    )

    forecasts, settlements, markets = load_snapshot_data(snapshots_dir)

    bias_table: BiasTable | None
    if bias_table_path is not None and bias_table_path.exists():
        bias_table = BiasTable(bias_table_path)
        if not bias_table.loaded:
            bias_table = None
    else:
        bias_table = None

    return list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=bias_table,
            ensemble_weights=ensemble_weights,
            spread_inflation_overrides=None,
        )
    )


def validate_calibrator_artifact(
    *,
    candidate_path: Path,
    baseline_path: Path,
    snapshots_dir: Path,
    bias_table_path: Path | None,
    ensemble_weights: dict[str, float],
    holdout_days: int = 14,
    max_plateau_width: float = 0.05,
    min_calibration_brier_lift: float = 0.01,
    min_holdout_pairs: int = 150,
    strict_structural_checks: bool = True,
) -> dict:
    """Run all pre-deploy checks. Returns a report dict with a ``pass`` boolean.

    When ``strict_structural_checks=False``, plateau-width and output-range
    failures move to ``advisory_warnings`` and do not gate the pass.
    The Brier-lift gate and holdout-size guard remain gating in both modes.
    """
    candidate = json.loads(Path(candidate_path).read_text())
    baseline = json.loads(Path(baseline_path).read_text())

    fail_reasons: list[str] = []
    advisory_warnings: list[str] = []

    all_pairs = _load_pairs_with_dates(
        snapshots_dir=snapshots_dir,
        bias_table_path=bias_table_path,
        ensemble_weights=ensemble_weights,
    )
    if not all_pairs:
        return {
            "pass": False,
            "fail_reasons": ["no_pairs_loaded"],
            "advisory_warnings": [],
            "n_train_pairs": 0,
            "n_holdout_pairs": 0,
        }

    from datetime import timedelta

    pair_dates = sorted({p[5] for p in all_pairs})
    if len(pair_dates) <= holdout_days:
        return {
            "pass": False,
            "fail_reasons": ["insufficient_temporal_coverage"],
            "advisory_warnings": [],
            "n_train_pairs": 0,
            "n_holdout_pairs": 0,
            "n_unique_dates": len(pair_dates),
        }
    max_date = pair_dates[-1]
    holdout_start = max_date - timedelta(days=holdout_days - 1)
    train_pairs = [p for p in all_pairs if p[5] < holdout_start]
    holdout_pairs = [p for p in all_pairs if p[5] >= holdout_start]

    if len(holdout_pairs) < min_holdout_pairs:
        return {
            "pass": False,
            "fail_reasons": ["insufficient_holdout_pairs"],
            "advisory_warnings": [],
            "n_train_pairs": len(train_pairs),
            "n_holdout_pairs": len(holdout_pairs),
            "n_unique_dates": len(pair_dates),
            "holdout_start": str(holdout_start),
        }

    train_raws = [p[0] for p in train_pairs]
    train_outs = [p[1] for p in train_pairs]
    hold_raws = [p[0] for p in holdout_pairs]
    hold_outs = [p[1] for p in holdout_pairs]

    import numpy as np

    from scripts.fit_isotonic import fit_isotonic

    iso_train = fit_isotonic(np.array(train_raws), np.array(train_outs))
    cand_holdout_probs = [float(iso_train.predict([r])[0]) for r in hold_raws]
    cand_holdout_brier = _brier(cand_holdout_probs, hold_outs)

    raw_holdout_brier = _brier(hold_raws, hold_outs)

    base_holdout_probs = [
        _interpolate(r, baseline["breakpoints"], baseline["values"]) for r in hold_raws
    ]
    base_holdout_brier = _brier(base_holdout_probs, hold_outs)

    calibration_lift = raw_holdout_brier - cand_holdout_brier
    if calibration_lift < min_calibration_brier_lift:
        fail_reasons.append("calibration_no_lift_on_holdout")

    plateau_violations = _detect_plateaus(
        candidate["breakpoints"],
        candidate["values"],
        width_threshold=max_plateau_width,
    )
    if plateau_violations:
        if strict_structural_checks:
            fail_reasons.append("plateau_too_wide")
        else:
            advisory_warnings.append("plateau_too_wide")

    max_cal_value = max(candidate["values"])
    if max_cal_value <= 0.50:
        if strict_structural_checks:
            fail_reasons.append("output_range_capped")
        else:
            advisory_warnings.append("output_range_capped")

    return {
        "pass": not fail_reasons,
        "fail_reasons": fail_reasons,
        "advisory_warnings": advisory_warnings,
        "candidate_path": str(candidate_path),
        "baseline_path": str(baseline_path),
        "n_train_pairs": len(train_pairs),
        "n_holdout_pairs": len(holdout_pairs),
        "n_unique_dates": len(pair_dates),
        "holdout_start": str(holdout_start),
        "raw_holdout_brier": round(raw_holdout_brier, 6),
        "candidate_holdout_brier": round(cand_holdout_brier, 6),
        "baseline_holdout_brier": round(base_holdout_brier, 6),
        "calibration_lift_vs_raw": round(calibration_lift, 6),
        "candidate_vs_baseline_delta": round(cand_holdout_brier - base_holdout_brier, 6),
        "plateau_violations": plateau_violations,
        "max_calibrated_value": max_cal_value,
        "strict_structural_checks": strict_structural_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-deploy validation for isotonic calibrator (held-out-by-date)"
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to the candidate calibrator artifact (the one being validated).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_PROJECT_ROOT / "data" / "calibration" / "isotonic.json",
        help="Path to the current live calibrator artifact (used for context "
        "comparison; never gated).",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
        help="Snapshot directory containing daily forecasts/markets/settlements.",
    )
    parser.add_argument(
        "--bias-table",
        type=Path,
        default=_PROJECT_ROOT / "data" / "gfs_bias" / "bias_correction.json",
        help="Path to the GFS bias correction JSON. Falls back to no correction "
        "if missing or unloadable.",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="JSON dict of {model_name: weight}. Falls back to config.yaml "
        "weather strategy weights.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=14,
        help="Number of calendar days from the most recent forecast date to use "
        "as the held-out test set. Default: 14.",
    )
    parser.add_argument(
        "--max-plateau-width",
        type=float,
        default=0.05,
        help="Plateau width threshold (raw probability span). Plateaus wider "
        "than this fail Gate 2. Default: 0.05.",
    )
    parser.add_argument(
        "--min-calibration-brier-lift",
        type=float,
        default=0.01,
        help="Minimum Brier improvement of candidate over raw inputs on holdout. "
        "Load-bearing gate. Default: 0.01.",
    )
    parser.add_argument(
        "--min-holdout-pairs",
        type=int,
        default=150,
        help="Minimum holdout pair count below which the gate short-circuits to "
        "fail (Brier 95%% CI half-width gets too wide). Default: 150.",
    )
    parser.add_argument(
        "--advisory-structural-checks",
        action="store_true",
        help="Treat plateau-width and max-output checks as advisory (logged "
        "but not gating). Use when Track 2 (EMOS) hasn't yet shipped "
        "and the current isotonic fit cannot pass these structural "
        "checks by construction. Default: strict.",
    )
    parser.add_argument(
        "--allow-partial-model-coverage",
        action="store_true",
        help="Allow the input-contract preflight to proceed when one or more "
        "post-cutover triples have only a subset of the expected "
        "ensemble models present. Required for pre-cutover "
        "reproducibility (where ECMWF is in the weights but not yet "
        "published).",
    )
    parser.add_argument(
        "--allow-empty-cells",
        action="store_true",
        help="Allow the input-contract preflight to proceed when one or more "
        "triples have NO model rows present despite having a market and "
        "settlement.",
    )
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="Operator override: skip the production bias-correction load. "
        "Without this flag, the validator's preflight aborts if the "
        "bias file is missing or fails to load.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full report as JSON.",
    )
    args = parser.parse_args()

    if args.candidate.resolve() == args.baseline.resolve():
        parser.error(
            "--candidate and --baseline point to the same file; comparison would be meaningless"
        )

    from scripts._calibration_common import (
        load_snapshot_data as _load_snapshot_data,
    )
    from scripts._calibration_common import (
        run_input_contract_preflight,
    )

    forecasts, settlements, markets = _load_snapshot_data(args.snapshots)
    bias_table, ensemble_weights, _preflight_report = run_input_contract_preflight(
        args, forecasts, settlements, markets
    )

    bias_path = args.bias_table if (bias_table is not None and args.bias_table.exists()) else None

    report = validate_calibrator_artifact(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        snapshots_dir=args.snapshots,
        bias_table_path=bias_path,
        ensemble_weights=ensemble_weights,
        holdout_days=args.holdout_days,
        max_plateau_width=args.max_plateau_width,
        min_calibration_brier_lift=args.min_calibration_brier_lift,
        min_holdout_pairs=args.min_holdout_pairs,
        strict_structural_checks=not args.advisory_structural_checks,
    )

    print("=" * 70)
    print("  CALIBRATOR VALIDATION GATE (held-out-by-date)")
    print("=" * 70)
    print(f"  Candidate:                {args.candidate}")
    print(f"  Baseline (live):          {args.baseline}")
    mode = "advisory" if args.advisory_structural_checks else "strict"
    print(f"  Structural checks mode:   {mode}")
    print(f"  Train pairs:              {report.get('n_train_pairs', 0)}")
    print(f"  Holdout pairs (last {args.holdout_days}d): {report.get('n_holdout_pairs', 0)}")
    print(f"  Holdout starts:           {report.get('holdout_start', 'n/a')}")
    raw_b = report.get("raw_holdout_brier", float("nan"))
    cand_b = report.get("candidate_holdout_brier", float("nan"))
    base_b = report.get("baseline_holdout_brier", float("nan"))
    lift = report.get("calibration_lift_vs_raw", float("nan"))
    delta = report.get("candidate_vs_baseline_delta", float("nan"))
    max_cv = report.get("max_calibrated_value", float("nan"))
    print(f"  Raw holdout Brier:        {raw_b:.6f}")
    print(f"  Candidate holdout Brier:  {cand_b:.6f}")
    print(f"  Baseline holdout Brier:   {base_b:.6f}")
    print(f"  Calibration lift vs raw:  {lift:+.6f}  (req >= {args.min_calibration_brier_lift})")
    print(f"  Cand vs baseline delta:   {delta:+.6f}  (context only)")
    print(f"  Max cal value:            {max_cv:.4f}  (req > 0.50)")
    print(f"  Plateau violations:       {len(report.get('plateau_violations', []))}")
    for p in report.get("plateau_violations", []):
        print(
            f"    raw [{p['lo']:.3f}, {p['hi']:.3f}] -> {p['value']:.4f} (width {p['width']:.3f})"
        )
    print()
    print(f"  RESULT: {'PASS' if report['pass'] else 'FAIL'}")
    if report.get("fail_reasons"):
        print(f"  Fail reasons:    {', '.join(report['fail_reasons'])}")
    if report.get("advisory_warnings"):
        print(f"  Advisory warns:  {', '.join(report['advisory_warnings'])}")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  Report written to {args.output}")

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
