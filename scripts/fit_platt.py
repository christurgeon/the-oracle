"""Fit Platt (logistic) calibration models from snapshot forecast-outcome pairs.

Produces two artifacts:

* ``data/calibration/platt.json`` — single global logistic fit
  ``sigmoid(a + b * raw)``.
* ``data/calibration/platt_per_strike_type.json`` — three independent
  fits, one per ``strike_type`` ∈ {greater, less, between}. Loaded by
  :class:`python.data.calibration.PerTypePlattCalibrator`.

The fitter shares ``scripts/_calibration_common.py`` with
``fit_isotonic.py`` so the two fitters can't disagree on which pairs
count as training data.

Usage::

    uv run --extra ml python scripts/fit_platt.py
    uv run --extra ml python scripts/fit_platt.py --cross-validate
    uv run --extra ml python scripts/fit_platt.py --no-spread-inflation

Notes:
  * Platt MLE is unregularized in principle; we approximate with
    ``sklearn.linear_model.LogisticRegression(C=1e10)``.
  * Per-type fits emit a warning (but still write the artifact) when a
    strike-type has fewer than ``--min-per-type`` pairs — the default
    300 is the sample floor where 2-param Platt MLE asymptotics hold.
    The runtime calibrator does not itself enforce this; ship decisions
    must weigh per-type ``n`` against risk tolerance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.calibration import STRIKE_TYPES as _CANONICAL_STRIKE_TYPES  # noqa: E402
from python.data.weather import get_spread_inflation_factor  # noqa: E402
from scripts._calibration_common import (  # noqa: E402
    brier_score,
    build_pairs,
    calibration_table,
    load_snapshot_data,
    run_input_contract_preflight,
)

if TYPE_CHECKING:
    from python.data.bias_correction import BiasTable
    from scripts._calibration_input_contract import InputContractReport

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

STRIKE_TYPES: tuple[str, ...] = ("greater", "less", "between")
assert set(STRIKE_TYPES) == _CANONICAL_STRIKE_TYPES

# sklearn LogisticRegression with very large C approximates unregularized
# MLE — Platt's original formulation. Solver "lbfgs" handles the small
# 1-feature problem trivially; max_iter is generous for convergence
# on degenerate inputs (all-one class, extreme raw probabilities, etc.).
# ``l1_ratio=0`` (pure L2) paired with a huge C is equivalent to the
# legacy ``penalty="l2"`` API that sklearn 1.8 deprecated.
_LR_KW: dict[str, Any] = {
    "C": 1.0e10,
    "solver": "lbfgs",
    "max_iter": 10_000,
    "fit_intercept": True,
}


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _fit_platt(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    """Return ``(a, b)`` for ``sigmoid(a + b * raw) ≈ P(y=1 | raw)``.

    Degenerate cases (all-one class, zero-length input) raise
    ``ValueError`` — callers decide whether to skip the stratum or
    abort.
    """
    if raw_probs.size == 0:
        raise ValueError("fit_platt: no samples")
    unique_y = set(outcomes.tolist())
    if len(unique_y) < 2:
        raise ValueError(
            f"fit_platt: outcomes are single-class ({unique_y}); "
            "logistic regression requires both classes present"
        )

    lr = LogisticRegression(**_LR_KW)
    lr.fit(raw_probs.reshape(-1, 1), outcomes)
    a = float(lr.intercept_[0])
    b = float(lr.coef_[0][0])
    return a, b


def _sigmoid_predict(raw: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply Platt map to a vector of raw probabilities."""
    z = a + b * raw
    # Numerically stable logistic — same form as the runtime calibrator.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _cross_validate(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    raw_briers: list[float] = []
    cal_briers: list[float] = []
    skipped = 0
    for train_idx, test_idx in kf.split(raw_probs):
        x_train = raw_probs[train_idx]
        y_train = outcomes[train_idx]
        x_test = raw_probs[test_idx]
        y_test = outcomes[test_idx]
        try:
            a, b = _fit_platt(x_train, y_train)
        except ValueError:
            skipped += 1
            continue
        cal_test = _sigmoid_predict(x_test, a, b)
        raw_briers.append(float(np.mean((x_test - y_test) ** 2)))
        cal_briers.append(float(np.mean((cal_test - y_test) ** 2)))
    if not cal_briers:
        return {"error": "cv_all_folds_degenerate", "skipped": skipped}
    return {
        "n_folds": n_folds,
        "n_evaluated_folds": len(cal_briers),
        "n_skipped_folds": skipped,
        "raw_brier_mean": float(np.mean(raw_briers)),
        "raw_brier_std": float(np.std(raw_briers)),
        "cal_brier_mean": float(np.mean(cal_briers)),
        "cal_brier_std": float(np.std(cal_briers)),
        "improvement_pct": float(
            (np.mean(raw_briers) - np.mean(cal_briers)) / np.mean(raw_briers) * 100
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _pairs_to_arrays(
    pairs: list[tuple[float, int, str, str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=int)
    strike_types = [p[4] for p in pairs]
    return raw, y, strike_types


def _fit_global(
    raw: np.ndarray,
    y: np.ndarray,
    *,
    cross_validate: bool,
) -> dict[str, Any]:
    a, b = _fit_platt(raw, y)
    cal = _sigmoid_predict(raw, a, b)
    summary: dict[str, Any] = {
        "a": round(a, 6),
        "b": round(b, 6),
        "n_pairs": int(raw.size),
        "brier_raw": round(brier_score(raw, y), 6),
        "brier_calibrated": round(brier_score(cal, y), 6),
    }
    summary["improvement_pct"] = round(
        (summary["brier_raw"] - summary["brier_calibrated"]) / summary["brier_raw"] * 100.0,
        2,
    )
    if cross_validate:
        summary["cv"] = _cross_validate(raw, y)
    return summary


def _fit_per_type(
    raw: np.ndarray,
    y: np.ndarray,
    strike_types: list[str],
    *,
    min_per_type: int,
    cross_validate: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Fit one Platt per strike_type.  Returns (per_type_dict, warnings)."""
    per_type: dict[str, Any] = {}
    warnings: list[str] = []
    strike_arr = np.asarray(strike_types)

    for stype in STRIKE_TYPES:
        mask = strike_arr == stype
        n = int(mask.sum())
        sub_raw = raw[mask]
        sub_y = y[mask]

        entry: dict[str, Any] = {"n": n}
        if n < min_per_type:
            warnings.append(
                f"strike_type={stype} has n={n} < min_per_type={min_per_type}; "
                "Platt MLE asymptotics may not hold"
            )
        try:
            a, b = _fit_platt(sub_raw, sub_y)
        except ValueError as exc:
            raise RuntimeError(
                f"fit_platt per-type: failed for strike_type={stype} ({exc}); "
                "cannot produce a complete per-type artifact. Collect more "
                "snapshots or drop per_type_platt as a candidate."
            ) from exc
        cal = _sigmoid_predict(sub_raw, a, b)
        entry["a"] = round(a, 6)
        entry["b"] = round(b, 6)
        entry["brier_raw"] = round(brier_score(sub_raw, sub_y), 6)
        entry["brier_calibrated"] = round(brier_score(cal, sub_y), 6)
        if cross_validate:
            entry["cv"] = _cross_validate(sub_raw, sub_y)
        per_type[stype] = entry
    return per_type, warnings


def _print_cal_table(label: str, probs: np.ndarray, outcomes: np.ndarray) -> None:
    print(f"  {label}:")
    print(f"    {'Bin':<12} {'N':>5}  {'Pred':>6}  {'Obs':>6}  {'Dev':>6}")
    for row in calibration_table(probs.tolist(), outcomes.tolist()):
        b_ = row["bin"]
        n_ = row["n"]
        p_ = row["predicted"]
        o_ = row["observed"]
        d_ = row["deviation"]
        print(f"    {b_:<12} {n_:>5}  {p_:>5.3f}  {o_:>5.3f}  {d_:>+5.3f}")


def _build_pairs_with_optional_provenance(
    args: argparse.Namespace,
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]],
    settlements: dict[tuple[str, str], dict[str, Any]],
    markets: list[dict[str, Any]],
    bias_table: BiasTable | None,
    ensemble_weights: dict[str, Any],
    *,
    _report: InputContractReport,
) -> list[tuple[float, int, str, str, str]]:
    """Build pairs, optionally writing provenance rows via ``--write-pair-provenance``."""
    from scripts._calibration_common import build_production_mirrored_pairs
    from scripts._calibration_input_contract import (
        CalibratorInputContractError,
    )

    del _report  # reserved for future provenance enrichment

    if args.write_pair_provenance:
        try:
            from scripts._calibration_common import (  # type: ignore[attr-defined]
                build_production_mirrored_pairs_with_provenance,
            )
        except ImportError as exc:
            raise CalibratorInputContractError(
                "--write-pair-provenance requested but "
                "build_production_mirrored_pairs_with_provenance is not available."
            ) from exc
        from scripts._calibration_input_contract import (
            record_pair_provenance,
        )

        # record_pair_provenance opens with O_APPEND for atomic concurrent
        # writes. Re-running the script with the same target would accumulate
        # on top of the previous run's output. Truncate at start of run.
        args.write_pair_provenance.unlink(missing_ok=True)

        pairs_iter = build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=bias_table,
            ensemble_weights=ensemble_weights,
            spread_inflation_overrides=None,
        )
        pairs: list[tuple[float, int, str, str, str]] = []
        for row in pairs_iter:
            p, o, c, v, strike_type, fc_d, models_tuple, bias_applied_raw, ticker = row
            models_used = list(models_tuple)
            bias_applied = bool(bias_applied_raw)
            record_pair_provenance(
                args.write_pair_provenance,
                ticker=ticker,
                city=c,
                forecast_date=fc_d,
                daily_variable=v,
                strike_type=strike_type,
                models_used=models_used,
                bias_correction_applied=bias_applied,
                spread_inflation_factor=float(get_spread_inflation_factor(c, v)),
                ensemble_weights_at_fit=ensemble_weights,
                raw_prob=float(p),
                outcome=int(o),
            )
            pairs.append((p, o, c, v, strike_type))
        return pairs

    pairs_iter = build_production_mirrored_pairs(
        forecasts=forecasts,
        settlements=settlements,
        markets=markets,
        bias_table=bias_table,
        ensemble_weights=ensemble_weights,
        spread_inflation_overrides=None,
    )
    return [(p, o, c, v, st) for p, o, c, v, st, _d in pairs_iter]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt calibration models")
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
        help="Snapshot data directory",
    )
    parser.add_argument(
        "--output-global",
        type=Path,
        default=_PROJECT_ROOT / "data" / "calibration" / "platt.json",
        help="Output path for the single-curve Platt artifact",
    )
    parser.add_argument(
        "--output-per-type",
        type=Path,
        default=_PROJECT_ROOT / "data" / "calibration" / "platt_per_strike_type.json",
        help="Output path for the per-strike-type Platt artifact",
    )
    parser.add_argument(
        "--min-per-type",
        type=int,
        default=300,
        help="Warn when a strike-type has fewer than this many training pairs",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run 5-fold CV and include results in the artifacts",
    )
    parser.add_argument(
        "--legacy-gfs-only",
        action="store_true",
        help="Use the legacy GFS-only pair builder. Archaeology only — "
        "does NOT apply bias correction or GFS+ECMWF blending.",
    )
    parser.add_argument(
        "--no-spread-inflation",
        action="store_true",
        help="Fit on raw ensemble fractions (no spread inflation). Legacy mode only.",
    )
    parser.add_argument(
        "--bias-table",
        type=Path,
        default=_PROJECT_ROOT / "data" / "gfs_bias" / "bias_correction.json",
        help="Path to bias correction JSON (used when --legacy-gfs-only is NOT set).",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="JSON dict of model_name -> weight. Defaults to "
        "config.yaml strategies.weather.ensemble_weights.",
    )
    parser.add_argument(
        "--allow-partial-model-coverage",
        action="store_true",
        help="Allow the input-contract preflight to proceed when one or more "
        "post-cutover triples have only a subset of the expected ensemble "
        "models present. Required for pre-cutover reproducibility (where "
        "ECMWF is in the weights but not yet published).",
    )
    parser.add_argument(
        "--allow-empty-cells",
        action="store_true",
        help="Allow the input-contract preflight to proceed when one or more "
        "triples have NO model rows present despite having a market and "
        "settlement. Without this flag, any empty cell aborts the fit.",
    )
    parser.add_argument(
        "--no-bias-correction",
        action="store_true",
        help="Operator override: skip the production bias-correction load. "
        "Without this flag, production-mirrored mode aborts if the bias "
        "file is missing or fails to load. The artifact's "
        "bias_correction_applied metadata field records the override.",
    )
    parser.add_argument(
        "--write-pair-provenance",
        type=Path,
        default=None,
        help="Path to a JSONL file. When set, a provenance row is written "
        "for every pair the production-mirrored generator yields.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Only fit the per-strike-type artifact",
    )
    parser.add_argument(
        "--skip-per-type",
        action="store_true",
        help="Only fit the global artifact",
    )
    args = parser.parse_args()
    if args.no_spread_inflation and not args.legacy_gfs_only:
        parser.error(
            "--no-spread-inflation is only valid with --legacy-gfs-only "
            "(production-mirrored pipeline always applies spread inflation per config)"
        )

    logger.info("Loading snapshot data from %s", args.snapshots)
    forecasts, settlements, markets = load_snapshot_data(args.snapshots)
    logger.info(
        "Loaded %d forecasts, %d settlements, %d market entries",
        len(forecasts),
        len(settlements),
        len(markets),
    )

    # bias_table hoisted so metadata records whether it actually loaded.
    bias_table = None
    if args.legacy_gfs_only:
        pairs = build_pairs(
            forecasts,
            settlements,
            markets,
            use_spread_inflation=not args.no_spread_inflation,
            spread_inflation_factor_fn=get_spread_inflation_factor,
        )
        ensemble_weights = {"gfs_seamless": 1.0}
        logger.info("Built %d (legacy GFS-only) forecast-outcome pairs", len(pairs))
    else:
        bias_table, ensemble_weights, report = run_input_contract_preflight(
            args, forecasts, settlements, markets
        )
        pairs = _build_pairs_with_optional_provenance(
            args,
            forecasts,
            settlements,
            markets,
            bias_table,
            ensemble_weights,
            _report=report,
        )
        logger.info(
            "Built %d production-mirrored forecast-outcome pairs (bias=%s, weights=%s)",
            len(pairs),
            "ON" if bias_table is not None else "OFF",
            ensemble_weights,
        )

    if len(pairs) < 100:
        logger.error("Insufficient data: need at least 100 pairs, got %d", len(pairs))
        sys.exit(1)

    raw, y, strike_types = _pairs_to_arrays(pairs)

    print("\n" + "=" * 70)
    print("  PLATT CALIBRATION FIT")
    print("=" * 70)
    print(f"  Training pairs:   {raw.size}")
    print(f"  Spread inflation: {'ON' if not args.no_spread_inflation else 'OFF'}")
    print()

    now = str(date.today())
    fit_metadata = {
        "schema_version": 3,
        "fit_date": now,
        "input_pipeline": ("legacy_gfs_only" if args.legacy_gfs_only else "production_mirrored"),
        "bias_correction_applied": bias_table is not None,
        "ensemble_weights": ({"gfs_seamless": 1.0} if args.legacy_gfs_only else ensemble_weights),
        "spread_inflation": not args.no_spread_inflation,
        "source_snapshots_dir": str(args.snapshots),
    }

    # -- Global fit -------------------------------------------------------
    if not args.skip_global:
        gsummary = _fit_global(raw, y, cross_validate=args.cross_validate)
        print("  GLOBAL PLATT")
        print(f"    a = {gsummary['a']:+.4f}")
        print(f"    b = {gsummary['b']:+.4f}")
        print(
            f"    Brier: raw={gsummary['brier_raw']:.4f} → "
            f"cal={gsummary['brier_calibrated']:.4f} "
            f"({gsummary['improvement_pct']:+.1f}%)"
        )
        if "cv" in gsummary and "cal_brier_mean" in gsummary["cv"]:
            cv = gsummary["cv"]
            print(
                f"    CV: raw={cv['raw_brier_mean']:.4f} → "
                f"cal={cv['cal_brier_mean']:.4f} "
                f"({cv['improvement_pct']:+.1f}%)"
            )
        payload_global: dict[str, Any] = {**fit_metadata, **gsummary}
        args.output_global.parent.mkdir(parents=True, exist_ok=True)
        args.output_global.write_text(json.dumps(payload_global, indent=2) + "\n")
        print(f"  -> wrote {args.output_global}")
        cal_global = _sigmoid_predict(raw, gsummary["a"], gsummary["b"])
        _print_cal_table("BEFORE (global)", raw, y)
        _print_cal_table("AFTER  (global)", cal_global, y)
        print()

    # -- Per-type fit -----------------------------------------------------
    if not args.skip_per_type:
        per_type, pt_warnings = _fit_per_type(
            raw,
            y,
            strike_types,
            min_per_type=args.min_per_type,
            cross_validate=args.cross_validate,
        )
        print("  PER-STRIKE-TYPE PLATT")
        for stype in STRIKE_TYPES:
            e = per_type[stype]
            print(
                f"    {stype:<8} n={e['n']:>5}  a={e['a']:+.4f} b={e['b']:+.4f}  "
                f"Brier {e['brier_raw']:.4f} → {e['brier_calibrated']:.4f}"
            )
        if pt_warnings:
            print("  Warnings:")
            for w in pt_warnings:
                print(f"    ! {w}")
        payload_per_type = {
            **fit_metadata,
            "schema_version": 2,  # per-type artifact has a distinct schema
            "min_per_type": args.min_per_type,
            "n_pairs_total": int(raw.size),
            "per_type_warnings": pt_warnings,
            "per_type": per_type,
        }
        args.output_per_type.parent.mkdir(parents=True, exist_ok=True)
        args.output_per_type.write_text(json.dumps(payload_per_type, indent=2) + "\n")
        print(f"  -> wrote {args.output_per_type}")


if __name__ == "__main__":
    main()
