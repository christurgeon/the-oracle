"""Fit isotonic calibration model from snapshot forecast-outcome pairs.

Outputs a JSON lookup table (breakpoints + values) for production use
without requiring sklearn at runtime.

Usage::

    uv run python scripts/fit_isotonic.py [--snapshots DIR] [--output PATH]
    uv run python scripts/fit_isotonic.py --cross-validate
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, KFold

# Add project root to path so the script works standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.isotonic_fit import (  # noqa: E402
    default_anchor_weight,
    fit_isotonic_model,
)
from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.data.weather import get_spread_inflation_factor  # noqa: E402
from scripts._calibration_common import (  # noqa: E402
    VALID_STRIKE_TYPES,
    load_snapshot_data,
    parse_market_date,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIABLE_TO_SETTLEMENT_FIELD = {
    "temperature_2m_max": "high_f",
    "temperature_2m_min": "low_f",
}


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _build_pairs(
    forecasts: dict[tuple[str, str, str, str], dict],
    settlements: dict,
    markets: list[dict],
    *,
    model: str = "gfs_seamless",
    use_spread_inflation: bool = True,
) -> list[tuple[float, int, str, str, date]]:
    """Build ``(raw_fraction, binary_outcome, city, variable, forecast_date)`` pairs."""
    pairs: list[tuple[float, int, str, str, date]] = []
    seen: set[str] = set()

    for mkt in markets:
        ticker = mkt.get("ticker", "")
        strike_type = mkt.get("strike_type")
        if strike_type not in VALID_STRIKE_TYPES:
            continue

        city = mkt.get("city")
        daily_var = mkt.get("daily_variable")
        if not city or not daily_var:
            continue

        fc_date = parse_market_date(ticker)
        if not fc_date:
            continue

        if ticker in seen:
            continue
        seen.add(ticker)

        if city == "CHI" and fc_date < date(2026, 3, 20):
            continue

        fc = forecasts.get((city, str(fc_date), daily_var, model))
        if not fc:
            continue

        st = settlements.get((city, str(fc_date)))
        if not st:
            continue

        settle_field = _VARIABLE_TO_SETTLEMENT_FIELD.get(daily_var)
        if not settle_field:
            continue
        actual_temp = st.get(settle_field)
        if actual_temp is None:
            continue

        members = fc.get("temperatures", fc.get("ensemble_members", []))
        if len(members) < 25:
            continue

        if use_spread_inflation:
            factor = get_spread_inflation_factor(city, daily_var)
            mean_val = sum(members) / len(members)
            members = [mean_val + (m - mean_val) * factor for m in members]

        floor_s = mkt.get("floor_strike")
        cap_s = mkt.get("cap_strike")

        # Outcome semantics MUST match Kalshi settlement (delegated to
        # python.analytics.outcomes.derive_outcome_yes — single source of
        # truth). The ``prob`` lines are model probabilities and follow
        # the production ``EnsembleForecast.fraction_*`` conventions
        # (between is CLOSED, less is strict, greater is strict).
        if strike_type == "greater":
            # Kalshi rule: t > floor (STRICT). Matches production
            # EnsembleForecast.fraction_above and derive_outcome_yes.
            prob = sum(1 for m in members if m > floor_s) / len(members)
        elif strike_type == "less":
            prob = sum(1 for m in members if m < cap_s) / len(members)
        else:
            # Kalshi rule: floor <= t <= cap (CLOSED, right-inclusive).
            prob = sum(1 for m in members if floor_s <= m <= cap_s) / len(members)
        yes = derive_outcome_yes(strike_type, floor_s, cap_s, float(actual_temp))
        if yes is None:
            continue
        outcome = 1 if yes else 0

        pairs.append((prob, outcome, city, daily_var, fc_date))

    return pairs


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_isotonic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    anchor_endpoints: bool = True,
    anchor_weight: int | None = None,
) -> IsotonicRegression:
    """Legacy entry-point retained for CLI back-compat.

    Delegates to python.analytics.isotonic_fit.fit_isotonic_model (one fit,
    canonical math). Returns the fitted IsotonicRegression so existing
    callers (replay_isotonic_holdout.py, validate_calibrator.py) that
    consume model.predict / X_thresholds_ / y_thresholds_ keep working.
    """
    pairs = np.column_stack([np.asarray(features), np.asarray(targets)])
    return fit_isotonic_model(
        pairs,
        anchor_endpoints=anchor_endpoints,
        anchor_weight=anchor_weight,
    )


def cross_validate(
    features: np.ndarray,
    targets: np.ndarray,
    n_folds: int = 5,
    *,
    anchor_endpoints: bool = True,
    groups: np.ndarray | None = None,
) -> dict:
    """Cross-validate the isotonic fit.

    Two split protocols:
    - ``groups=None`` → legacy ``KFold(shuffle=True, random_state=42)``.
      Same-day rows can land in different folds. Kept for backwards
      comparability of historical CV numbers (e.g., the 2026-05-12 refit's
      ``cv_improvement_pct: 18.6``).
    - ``groups`` array → ``GroupKFold(n_splits=n_folds)`` keyed on the
      array (typically ``forecast_date`` integer ids). All rows that share
      a group end up in the same fold. This is the leakage-free protocol
      and the number that should drive promotion decisions.

    F1.1 (2026-05-19 adversarial review): the random-KFold number
    overstates true OOS generalization because dozens of pairs per
    forecast_date (across cities × strikes × types) leak across folds.
    """
    splitter: GroupKFold | KFold
    if groups is not None:
        groups_arr = np.asarray(groups)
        unique = np.unique(groups_arr)
        effective_folds = min(n_folds, len(unique))
        if effective_folds < 2:
            raise ValueError(
                f"GroupKFold requires ≥2 unique groups, got {len(unique)}. "
                "Either add more forecast_dates or fall back to random KFold."
            )
        splitter = GroupKFold(n_splits=effective_folds)
        split_iter = splitter.split(features, targets, groups=groups_arr)
        protocol = "GroupKFold"
        n_folds = effective_folds
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(features)
        protocol = "KFold"

    raw_briers = []
    cal_briers = []

    for train_idx, test_idx in split_iter:
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        iso = fit_isotonic(x_train, y_train, anchor_endpoints=anchor_endpoints)
        y_pred_cal = iso.predict(x_test)

        raw_briers.append(float(np.mean((x_test - y_test) ** 2)))
        cal_briers.append(float(np.mean((y_pred_cal - y_test) ** 2)))

    return {
        "n_folds": n_folds,
        "split_protocol": protocol,
        "raw_brier_mean": float(np.mean(raw_briers)),
        "raw_brier_std": float(np.std(raw_briers)),
        "cal_brier_mean": float(np.mean(cal_briers)),
        "cal_brier_std": float(np.std(cal_briers)),
        "improvement_pct": float(
            (np.mean(raw_briers) - np.mean(cal_briers)) / np.mean(raw_briers) * 100
        ),
    }


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _calibration_table(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    table = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred = float(probs[mask].mean())
        obs = float(outcomes[mask].mean())
        table.append(
            {
                "bin": f"{int(lo * 100)}-{int(hi * 100)}%",
                "n": n,
                "predicted": round(pred, 4),
                "observed": round(obs, 4),
                "deviation": round(obs - pred, 4),
            }
        )
    return table


def _build_pairs_with_optional_provenance(
    args: argparse.Namespace,
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]],
    settlements: dict[tuple[str, str], dict[str, Any]],
    markets: list[dict[str, Any]],
    bias_table: BiasTable | None,
    ensemble_weights: dict[str, Any],
    *,
    _report: InputContractReport,
) -> list[tuple[float, int, str, str, date]]:
    """Build pairs, optionally writing provenance rows via ``--write-pair-provenance``.

    Returns 5-tuples ``(raw_prob, outcome, city, daily_variable, forecast_date)``.
    The ``forecast_date`` is required for leakage-free CV via GroupKFold.
    """
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
        pairs: list[tuple[float, int, str, str, date]] = []
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
            pairs.append((p, o, c, v, fc_d))
        return pairs

    pairs_iter = build_production_mirrored_pairs(
        forecasts=forecasts,
        settlements=settlements,
        markets=markets,
        bias_table=bias_table,
        ensemble_weights=ensemble_weights,
        spread_inflation_overrides=None,
    )
    return [(p, o, c, v, fc_d) for p, o, c, v, _st, fc_d in pairs_iter]


# ---------------------------------------------------------------------------
# fit_date resolution (F1.5)
# ---------------------------------------------------------------------------


def _resolve_fit_date(pairs: list[tuple[Any, ...]]) -> date:
    """Return ``max(forecast_date)`` across ``pairs`` or fall back to today.

    F1.5 (2026-05-19 adversarial review): the previous behavior stamped
    artifacts with ``date.today()``. A backfill rebuild against older
    snapshots therefore looked freshly fit, silently buying one extra day
    of grace from any tripwire keyed on artifact age. Stamping
    ``max(corpus.forecast_date)`` ties the artifact's age to the data it
    saw, not when the script happened to run.

    Each pair is expected to be a tuple with ``forecast_date`` in the last
    position (legacy ``_build_pairs`` and ``_build_pairs_with_optional_provenance``
    both emit 5-tuples ending in ``forecast_date``). The forecast_date may
    be either a ``datetime.date`` or an ISO date string.
    """
    if not pairs:
        logger.warning(
            "fit_date fallback: pair list is empty; stamping today's date. "
            "Re-fit on a non-empty corpus to restore data-anchored staleness."
        )
        return date.today()

    fc_dates: list[date] = []
    for row in pairs:
        if not row:
            continue
        fc = row[-1]
        if isinstance(fc, date):
            fc_dates.append(fc)
        elif isinstance(fc, str):
            try:
                fc_dates.append(date.fromisoformat(fc))
            except ValueError:
                continue
    if not fc_dates:
        logger.warning(
            "fit_date fallback: no parseable forecast_date in pair tuples; "
            "stamping today's date."
        )
        return date.today()
    return max(fc_dates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit isotonic calibration model")
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=_PROJECT_ROOT / "data" / "snapshots",
        help="Snapshot data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "calibration" / "isotonic.json",
        help="Output JSON path for fitted model",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run k-fold cross-validation and print results",
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
        "--no-anchor-endpoints",
        action="store_true",
        help="Disable (0,0)/(1,1) anchors.",
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
    args = parser.parse_args()
    if args.no_spread_inflation and not args.legacy_gfs_only:
        parser.error(
            "--no-spread-inflation is only valid with --legacy-gfs-only "
            "(production-mirrored pipeline always applies spread inflation per config)"
        )
    anchor_endpoints = not args.no_anchor_endpoints

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
        pairs = _build_pairs(
            forecasts,
            settlements,
            markets,
            use_spread_inflation=not args.no_spread_inflation,
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

    raw_probs = np.array([p[0] for p in pairs])
    outcomes = np.array([p[1] for p in pairs])

    # Build forecast-date group ids for GroupKFold (F1.1).
    # Each pair tuple ends with its ``forecast_date`` (date or str).
    def _pair_date_str(row: tuple[Any, ...]) -> str:
        fc = row[-1]
        if isinstance(fc, date):
            return fc.isoformat()
        return str(fc)

    date_strs = [_pair_date_str(p) for p in pairs]
    unique_dates = sorted(set(date_strs))
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    groups_arr = np.array([date_to_id[d] for d in date_strs], dtype=np.int64)

    if args.cross_validate:
        print("\n" + "=" * 70)
        print("  CROSS-VALIDATION RESULTS")
        print("=" * 70)

        def _print_cv(label: str, cv_out: dict[str, Any]) -> None:
            raw_s = f"{cv_out['raw_brier_mean']:.4f} ± {cv_out['raw_brier_std']:.4f}"
            cal_s = f"{cv_out['cal_brier_mean']:.4f} ± {cv_out['cal_brier_std']:.4f}"
            print(f"  [{label}] protocol={cv_out['split_protocol']} "
                  f"folds={cv_out['n_folds']}")
            print(f"    Raw Brier:        {raw_s}")
            print(f"    Calibrated Brier: {cal_s}")
            print(f"    Improvement:      {cv_out['improvement_pct']:+.1f}%")

        # Random-KFold (legacy — same number historical refit reports cited).
        cv_random = cross_validate(raw_probs, outcomes, anchor_endpoints=anchor_endpoints)
        _print_cv("Random-KFold (legacy, leakage-prone)", cv_random)

        # GroupKFold (leakage-free; groups on forecast_date).
        try:
            cv_group = cross_validate(
                raw_probs,
                outcomes,
                anchor_endpoints=anchor_endpoints,
                groups=groups_arr,
            )
            _print_cv("GroupKFold (forecast_date, leakage-free)", cv_group)
            delta = cv_random["improvement_pct"] - cv_group["improvement_pct"]
            print(f"  Δ improvement (random − group): {delta:+.2f} pp  "
                  "(positive ⇒ measured leakage)")
        except ValueError as exc:
            print(f"  GroupKFold unavailable: {exc}")
        print()

    iso = fit_isotonic(raw_probs, outcomes, anchor_endpoints=anchor_endpoints)
    cal_probs = iso.predict(raw_probs)

    brier_raw = _brier_score(raw_probs, outcomes)
    brier_cal = _brier_score(cal_probs, outcomes)

    print("\n" + "=" * 70)
    print("  ISOTONIC CALIBRATION FIT")
    print("=" * 70)
    print(f"  Training pairs:     {len(pairs)}")
    inflate = "ON" if not args.no_spread_inflation else "OFF"
    print(f"  Spread inflation:   {inflate}")
    print(f"  Brier (raw):        {brier_raw:.4f}")
    print(f"  Brier (calibrated): {brier_cal:.4f}")
    pct = (brier_raw - brier_cal) / brier_raw * 100
    print(f"  Improvement:        {pct:+.1f}%")
    print(f"  Breakpoints:        {len(iso.X_thresholds_)}")
    print()

    def _print_cal_table(label: str, probs: np.ndarray) -> None:
        print(f"  {label}:")
        hdr = f"    {'Bin':<12} {'N':>5}  {'Pred':>6}  {'Obs':>6}  {'Dev':>6}"
        print(hdr)
        for row in _calibration_table(probs, outcomes):
            b, n = row["bin"], row["n"]
            p, o, d = row["predicted"], row["observed"], row["deviation"]
            print(f"    {b:<12} {n:>5}  {p:>5.3f}  {o:>5.3f}  {d:>+5.3f}")

    _print_cal_table("BEFORE isotonic", raw_probs)
    print()
    _print_cal_table("AFTER isotonic", cal_probs)

    breakpoints = iso.X_thresholds_.tolist()
    values = iso.y_thresholds_.tolist()

    output = {
        "schema_version": 3,
        "fit_date": _resolve_fit_date(pairs).isoformat(),
        "input_pipeline": ("legacy_gfs_only" if args.legacy_gfs_only else "production_mirrored"),
        "bias_correction_applied": bias_table is not None,
        "ensemble_weights": ({"gfs_seamless": 1.0} if args.legacy_gfs_only else ensemble_weights),
        "n_pairs": len(pairs),
        "spread_inflation": not args.no_spread_inflation,
        "anchor_endpoints": anchor_endpoints,
        "anchor_weight": default_anchor_weight(len(pairs)) if anchor_endpoints else 0,
        "brier_raw": round(brier_raw, 6),
        "brier_calibrated": round(brier_cal, 6),
        "improvement_pct": round((brier_raw - brier_cal) / brier_raw * 100, 2),
        "breakpoints": [round(x, 6) for x in breakpoints],
        "values": [round(x, 6) for x in values],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    logger.info("Saved isotonic model to %s", args.output)


if __name__ == "__main__":
    main()
