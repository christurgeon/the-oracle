"""Shared helpers for calibration fitting scripts."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys as _sys
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Allow ``python.*`` imports from this script-side module without requiring
# callers to set PYTHONPATH manually.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

if TYPE_CHECKING:
    from python.data.bias_correction import BiasTable
    from scripts._calibration_input_contract import InputContractReport

logger = logging.getLogger(__name__)


def prefer_latest_settlement(
    existing: Mapping[str, Any],
    new: Mapping[str, Any],
) -> bool:
    """Return True if ``new`` should replace ``existing``.

    Selection key is ``(not is_preliminary, issued_at, product_id)`` — strictly
    greater wins. **Finals always beat preliminaries**, regardless of
    ``issued_at`` ordering; among two records of the same preliminary status,
    later ``issued_at`` wins; ``product_id`` breaks ties on identical
    ``issued_at``.

    Pre-2026-05-11 the key was just ``(issued_at, product_id)``; that allowed a
    retroactive preliminary issued later in the day to overwrite the morning
    final, and downstream ``final_only=True`` filters then dropped the row
    entirely.
    """
    existing_key = (
        not bool(existing.get("is_preliminary", False)),
        str(existing.get("issued_at", "") or ""),
        str(existing.get("product_id", "") or ""),
    )
    new_key = (
        not bool(new.get("is_preliminary", False)),
        str(new.get("issued_at", "") or ""),
        str(new.get("product_id", "") or ""),
    )
    return new_key > existing_key


_VARIABLE_TO_SETTLEMENT_FIELD = {
    "temperature_2m_max": "high_f",
    "temperature_2m_min": "low_f",
}

VALID_STRIKE_TYPES: tuple[str, ...] = ("greater", "less", "between")

_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})-", re.IGNORECASE)
_MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

# Chicago snapshots before this date used an incorrect city coord; drop.
_CHI_SNAPSHOT_FIX_DATE = date(2026, 3, 20)


def parse_market_date(ticker: str) -> date | None:
    """Extract the target date from a market ticker like KXHIGHNY-26MAR26-T72."""
    m = _DATE_RE.search(ticker)
    if not m:
        return None
    y, mo, d = int(m.group(1)), _MONTH_MAP.get(m.group(2).upper()), int(m.group(3))
    if mo is None:
        return None
    try:
        return date(2000 + y, mo, d)
    except (ValueError, OverflowError):
        return None


def load_snapshot_data(
    snapshots_dir: Path,
) -> tuple[dict[tuple[str, str, str, str], dict], dict[tuple[str, str], dict], list[dict]]:
    """Load forecasts, settlements, and markets from all snapshot directories.

    Handles both flat ``<date>/`` and nested ``<date>/<run_id>/`` layouts.
    Forecasts are keyed on ``(city, forecast_date, daily_variable, model)``;
    legacy records without a ``model`` field default to ``"gfs_seamless"``.
    Settlements are merged by ``prefer_latest_settlement``; preliminary winners
    are dropped after the merge.
    """
    forecasts: dict[tuple[str, str, str, str], dict] = {}
    settlements: dict[tuple[str, str], dict] = {}
    markets: list[dict] = []

    def _load_dir(d: Path) -> None:
        fc_file = d / "weather_forecasts.json"
        if fc_file.exists():
            for fc in json.loads(fc_file.read_text()):
                model = fc.get("model", "gfs_seamless")
                key = (fc["city"], fc["forecast_date"], fc["daily_variable"], model)
                forecasts[key] = fc  # last writer wins; sorted traversal keeps latest run

        mkt_file = d / "markets.json"
        if mkt_file.exists():
            markets.extend(json.loads(mkt_file.read_text()))

        st_file = d / "settlements.json"
        if st_file.exists():
            for s in json.loads(st_file.read_text()):
                sk = (s["city"], s["observation_date"])
                existing = settlements.get(sk)
                if existing is None or prefer_latest_settlement(existing, s):
                    settlements[sk] = s

    for day_dir in sorted(snapshots_dir.iterdir()):
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        if (day_dir / "markets.json").exists():
            _load_dir(day_dir)
        else:
            for run_dir in sorted(day_dir.iterdir()):
                if run_dir.is_dir():
                    _load_dir(run_dir)

    # Drop preliminary winners after the merge so the merge itself is unbiased.
    settlements = {k: v for k, v in settlements.items() if not v.get("is_preliminary")}

    return forecasts, settlements, markets


def build_forecast_snapshot_dates(
    snapshots_dir: Path,
) -> dict[tuple[str, str, str, str], str]:
    """Build a mapping from forecast key to the snapshot date that last wrote it.

    Mirrors the iteration order of ``load_snapshot_data`` (sorted, last-writer-wins)
    so the returned snapshot_date for each ``(city, forecast_date, daily_var, model)``
    key matches the actual entry used by ``load_snapshot_data``.

    Returns:
        Dict mapping ``(city, forecast_date_str, daily_variable, model)`` →
        ``snapshot_date_str`` (ISO "YYYY-MM-DD"). Only includes keys for which
        the snapshot directory name is a valid ISO date.

    Used by ``python.analytics.calibrator_harness.load_feature_rows`` to populate
    ``FeatureRow.snapshot_date`` for lead computation.
    """
    snapshot_dates: dict[tuple[str, str, str, str], str] = {}

    def _scan_dir(d: Path, snapshot_date_str: str) -> None:
        fc_file = d / "weather_forecasts.json"
        if fc_file.exists():
            try:
                records = json.loads(fc_file.read_text())
            except (OSError, json.JSONDecodeError):
                return
            for fc in records:
                city = fc.get("city")
                fc_date = fc.get("forecast_date")
                daily_var = fc.get("daily_variable")
                model = fc.get("model", "gfs_seamless")
                if city and fc_date and daily_var:
                    key: tuple[str, str, str, str] = (city, fc_date, daily_var, model)
                    snapshot_dates[key] = snapshot_date_str  # last writer wins

    if not snapshots_dir.is_dir():
        return snapshot_dates

    for day_dir in sorted(snapshots_dir.iterdir()):
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        try:
            date.fromisoformat(day_dir.name)
            snapshot_date_str = day_dir.name
        except ValueError:
            continue
        if (day_dir / "weather_forecasts.json").exists():
            _scan_dir(day_dir, snapshot_date_str)
        elif (day_dir / "markets.json").exists():
            # Flat layout with no weather_forecasts at top level — skip.
            pass
        else:
            for run_dir in sorted(day_dir.iterdir()):
                if run_dir.is_dir():
                    _scan_dir(run_dir, snapshot_date_str)

    return snapshot_dates


def build_pairs(
    forecasts: dict[tuple[str, str, str, str], dict],
    settlements: dict[tuple[str, str], dict],
    markets: list[dict],
    *,
    model: str = "gfs_seamless",
    use_spread_inflation: bool = True,
    spread_inflation_factor_fn: Any = None,
) -> list[tuple[float, int, str, str, str]]:
    """Build ``(raw_fraction, outcome, city, variable, strike_type)`` pairs.

    ``spread_inflation_factor_fn`` is injected rather than imported so this
    module stays dependency-free and importable without the full production stack.
    """
    pairs: list[tuple[float, int, str, str, str]] = []
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

        if city == "CHI" and fc_date < _CHI_SNAPSHOT_FIX_DATE:
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

        if use_spread_inflation and spread_inflation_factor_fn is not None:
            factor = spread_inflation_factor_fn(city, daily_var)
            mean_val = sum(members) / len(members)
            members = [mean_val + (m - mean_val) * factor for m in members]

        floor_s = mkt.get("floor_strike")
        cap_s = mkt.get("cap_strike")

        # Outcome delegated to python.analytics.outcomes.derive_outcome_yes
        # — single source of truth for Kalshi settlement semantics
        # (between is CLOSED, right-inclusive; see outcomes.py docstring).
        from python.analytics.outcomes import derive_outcome_yes

        if strike_type == "greater":
            if floor_s is None:
                continue
            # Kalshi rule: t > floor (STRICT). Matches production
            # EnsembleForecast.fraction_above and derive_outcome_yes.
            prob = sum(1 for m in members if m > floor_s) / len(members)
        elif strike_type == "less":
            if cap_s is None:
                continue
            prob = sum(1 for m in members if m < cap_s) / len(members)
        else:  # between
            if floor_s is None or cap_s is None:
                continue
            # Kalshi rule: floor <= t <= cap (CLOSED) — matches
            # EnsembleForecast.fraction_between (production probability).
            prob = sum(1 for m in members if floor_s <= m <= cap_s) / len(members)
        yes = derive_outcome_yes(strike_type, floor_s, cap_s, float(actual_temp))
        if yes is None:
            continue
        outcome = 1 if yes else 0

        pairs.append((prob, outcome, city, daily_var, strike_type))

    return pairs


def brier_score(probs: list[float] | Any, outcomes: list[int] | Any) -> float:
    """Mean squared error of probability forecasts. Accepts lists or numpy arrays."""
    n = 0
    total = 0.0
    for p, y in zip(probs, outcomes, strict=False):
        total += (float(p) - float(y)) ** 2
        n += 1
    return total / n if n else 0.0


# ---------------------------------------------------------------------------
# Production-mirrored pair builder
# ---------------------------------------------------------------------------


def build_production_mirrored_pairs(
    *,
    forecasts: dict[tuple[str, str, str, str], dict],
    settlements: dict[tuple[str, str], dict],
    markets: list[dict],
    bias_table: BiasTable | None,
    ensemble_weights: dict[str, float],
    spread_inflation_overrides: dict[str, dict[str, float]] | None,
    bias_shrinkage: float = 0.3,
    bias_max_correction_f: float = 5.0,
) -> Iterator[tuple[float, int, str, str, str, date]]:
    """Yield ``(raw_prob, outcome, city, variable, strike_type, forecast_date)`` pairs.

    Mirrors the production pipeline: GFS bias correction → spread inflation →
    blended bracket fraction via ``compute_blended_probability``.

    Bracket fractions AND outcome both use CLOSED semantics
    (``floor <= t <= cap``) — matches Kalshi settlement (right-inclusive,
    verified 2026-05-19 truth audit) and production
    ``EnsembleForecast.fraction_between``. Outcome derivation is delegated
    to ``python.analytics.outcomes.derive_outcome_yes`` (single source of
    truth). Prior to 2026-05-19 outcome used half-open ``[floor, cap)``,
    which mis-labelled every ``actual_temp == cap`` case as NO.
    """
    from python.analytics.outcomes import derive_outcome_yes
    from python.common import Direction
    from python.data.bias_correction import apply_bias_correction
    from python.data.weather import (
        EnsembleForecast,
        compute_blended_probability,
        get_spread_inflation_factor,
    )

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

        if city == "CHI" and fc_date < _CHI_SNAPSHOT_FIX_DATE:
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

        per_model: dict[str, EnsembleForecast] = {}
        for model_name in ensemble_weights:
            fc_raw = forecasts.get((city, str(fc_date), daily_var, model_name))
            if not fc_raw:
                continue
            members = list(fc_raw.get("temperatures") or fc_raw.get("ensemble_members") or [])
            if len(members) < 25:
                continue

            forecast = EnsembleForecast(
                forecast_date=fc_date,
                city=city,
                temperatures=members,
                model=model_name,
            )

            # Bias correction (production: GFS only).
            if model_name == "gfs_seamless" and bias_table is not None and bias_table.loaded:
                forecast, _correction_f = apply_bias_correction(
                    forecast,
                    bias_table,
                    daily_var,
                    shrinkage=bias_shrinkage,
                    max_correction_f=bias_max_correction_f,
                )

            # Spread inflation.
            factor = get_spread_inflation_factor(city, daily_var, spread_inflation_overrides)
            if factor != 1.0:
                forecast = forecast.inflated(factor)

            per_model[model_name] = forecast

        if not per_model:
            continue

        direction = {
            "greater": Direction.ABOVE,
            "less": Direction.BELOW,
            "between": Direction.BETWEEN,
        }[strike_type]

        floor_s = mkt.get("floor_strike")
        cap_s = mkt.get("cap_strike")
        if strike_type in ("greater", "between") and floor_s is None:
            continue
        if strike_type in ("less", "between") and cap_s is None:
            continue
        raw_prob, _per_model_probs = compute_blended_probability(
            per_model,
            ensemble_weights,
            direction=direction,
            floor_strike=float(floor_s) if floor_s is not None else None,
            cap_strike=float(cap_s) if cap_s is not None else None,
        )

        # Kalshi-truth outcome (between is CLOSED, right-inclusive).
        yes = derive_outcome_yes(strike_type, floor_s, cap_s, float(actual_temp))
        if yes is None:
            continue
        outcome = 1 if yes else 0

        yield (raw_prob, outcome, city, daily_var, strike_type, fc_date)


def build_production_mirrored_pairs_with_provenance(
    *,
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]],
    settlements: dict[tuple[str, str], dict[str, Any]],
    markets: list[dict[str, Any]],
    bias_table: BiasTable | None,
    ensemble_weights: dict[str, float],
    spread_inflation_overrides: dict[str, dict[str, float]] | None,
    bias_shrinkage: float = 0.3,
    bias_max_correction_f: float = 5.0,
) -> Iterator[tuple[float, int, str, str, str, date, tuple[str, ...], bool, str]]:
    """Yield 9-tuple pairs: the 6 fields from ``build_production_mirrored_pairs``
    plus ``(models_used, bias_correction_applied, ticker)``.

    ``models_used`` is a sorted tuple of model names that actually contributed
    to ``compute_blended_probability`` for the yielded pair — reflects partial
    coverage when only one model was published.

    ``bias_correction_applied`` is ``True`` iff the bias-correction path ran:
    ``bias_table`` loaded AND ``"gfs_seamless"`` present in ``per_model``.

    Body is duplicated from ``build_production_mirrored_pairs`` rather than
    sharing an inner helper — yielding out of a shared helper would require
    a more complex generator-wrapper pattern for ~80 lines of duplication.

    Outcome is delegated to ``python.analytics.outcomes.derive_outcome_yes``
    (Kalshi rule for ``between``: CLOSED, right-inclusive).
    """
    from python.analytics.outcomes import derive_outcome_yes
    from python.common import Direction
    from python.data.bias_correction import apply_bias_correction
    from python.data.weather import (
        EnsembleForecast,
        compute_blended_probability,
        get_spread_inflation_factor,
    )

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

        if city == "CHI" and fc_date < _CHI_SNAPSHOT_FIX_DATE:
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

        per_model: dict[str, EnsembleForecast] = {}
        for model_name in ensemble_weights:
            fc_raw = forecasts.get((city, str(fc_date), daily_var, model_name))
            if not fc_raw:
                continue
            members = list(fc_raw.get("temperatures") or fc_raw.get("ensemble_members") or [])
            if len(members) < 25:
                continue

            forecast = EnsembleForecast(
                forecast_date=fc_date,
                city=city,
                temperatures=members,
                model=model_name,
            )

            # Bias correction (production: GFS only).
            if model_name == "gfs_seamless" and bias_table is not None and bias_table.loaded:
                forecast, _correction_f = apply_bias_correction(
                    forecast,
                    bias_table,
                    daily_var,
                    shrinkage=bias_shrinkage,
                    max_correction_f=bias_max_correction_f,
                )

            # Spread inflation.
            factor = get_spread_inflation_factor(city, daily_var, spread_inflation_overrides)
            if factor != 1.0:
                forecast = forecast.inflated(factor)

            per_model[model_name] = forecast

        if not per_model:
            continue

        direction = {
            "greater": Direction.ABOVE,
            "less": Direction.BELOW,
            "between": Direction.BETWEEN,
        }[strike_type]

        floor_s = mkt.get("floor_strike")
        cap_s = mkt.get("cap_strike")
        if strike_type in ("greater", "between") and floor_s is None:
            continue
        if strike_type in ("less", "between") and cap_s is None:
            continue
        raw_prob, _per_model_probs = compute_blended_probability(
            per_model,
            ensemble_weights,
            direction=direction,
            floor_strike=float(floor_s) if floor_s is not None else None,
            cap_strike=float(cap_s) if cap_s is not None else None,
        )

        # Kalshi-truth outcome (between is CLOSED, right-inclusive — see
        # python.analytics.outcomes.derive_outcome_yes for the rule).
        yes = derive_outcome_yes(strike_type, floor_s, cap_s, float(actual_temp))
        if yes is None:
            continue
        outcome = 1 if yes else 0

        models_used: tuple[str, ...] = tuple(sorted(per_model.keys()))
        bias_correction_applied = (
            bias_table is not None and bias_table.loaded and "gfs_seamless" in per_model
        )

        yield (
            raw_prob,
            outcome,
            city,
            daily_var,
            strike_type,
            fc_date,
            models_used,
            bias_correction_applied,
            ticker,
        )


def calibration_table(
    probs: list[float] | Any,
    outcomes: list[int] | Any,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Return a per-bin calibration table. Empty bins are skipped."""
    probs_list = [float(p) for p in probs]
    outcomes_list = [int(y) for y in outcomes]
    table: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [
            j
            for j, p in enumerate(probs_list)
            if (p >= lo and p < hi) or (i == n_bins - 1 and p == 1.0)
        ]
        n = len(idx)
        if n == 0:
            continue
        pred = sum(probs_list[j] for j in idx) / n
        obs = sum(outcomes_list[j] for j in idx) / n
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


# ---------------------------------------------------------------------------
# Production-mirrored preflight (shared by fit_isotonic and fit_platt)
# ---------------------------------------------------------------------------


def run_input_contract_preflight(
    args: argparse.Namespace,
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]],
    settlements: dict[tuple[str, str], dict[str, Any]],
    markets: list[dict[str, Any]],
) -> tuple[BiasTable | None, dict[str, Any], InputContractReport]:
    """Run the shared preflight and return ``(bias_table, ensemble_weights, report)``.

    Loads the bias table (fail-fast unless ``--no-bias-correction``), resolves
    ensemble weights (CLI override > config.yaml; fail-fast if missing), loads
    the data-availability calendar, calls ``assert_snapshot_input_contract``,
    and applies the five abort conditions.
    """
    import yaml as _yaml

    from python.data.bias_correction import BiasTable
    from scripts._calibration_input_contract import (
        MIN_TRIPLES_FOR_FIT,
        CalibratorInputContractError,
        DataAvailabilityCalendar,
        assert_snapshot_input_contract,
    )

    bias_table: BiasTable | None
    if args.no_bias_correction:
        bias_table = None
        logger.info(
            "--no-bias-correction set: skipping bias-correction load "
            "(metadata bias_correction_applied=false)"
        )
    else:
        if not args.bias_table.exists():
            raise CalibratorInputContractError(
                f"Bias-correction file not found at {args.bias_table}. "
                "Production-mirrored mode requires a loaded bias table; "
                "pass --no-bias-correction to override (artifact metadata "
                "will record bias_correction_applied=false)."
            )
        candidate: BiasTable = BiasTable(args.bias_table)
        if not candidate.loaded:
            raise CalibratorInputContractError(
                f"Bias-correction file at {args.bias_table} failed to load "
                "(BiasTable.loaded is False). Production-mirrored mode "
                "requires a loaded bias table; pass --no-bias-correction "
                "to override (artifact metadata will record "
                "bias_correction_applied=false)."
            )
        bias_table = candidate

    if args.ensemble_weights:
        ensemble_weights = json.loads(args.ensemble_weights)
    else:
        cfg_path = _PROJECT_ROOT / "python" / "config.yaml"
        cfg = _yaml.safe_load(cfg_path.read_text())
        ensemble_weights = cfg.get("strategies", {}).get("weather", {}).get("ensemble_weights")
        if not ensemble_weights:
            raise CalibratorInputContractError(
                "ensemble_weights not configured: expected key path "
                "'strategies.weather.ensemble_weights' in "
                f"{cfg_path}. Pass --ensemble-weights '{{...}}' to "
                "override for ablation runs."
            )

    calendar_path = _PROJECT_ROOT / "data" / "calibration" / "data_availability_calendar.json"
    if not calendar_path.exists():
        raise CalibratorInputContractError(
            f"Data-availability calendar not found at {calendar_path}. "
            "Regenerate via `scripts/_calibration_dry_run.py "
            "--regenerate-calendar` before fitting."
        )
    calendar = DataAvailabilityCalendar.from_path(calendar_path)

    report = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights=ensemble_weights,
        calendar=calendar,
    )
    print(report.format_for_console())

    if report.n_triples_total == 0:
        raise CalibratorInputContractError(
            "Input-contract preflight: n_triples_total == 0. "
            "Snapshots dir is empty or markets/settlements did not join."
        )
    if report.n_triples_exempt == report.n_triples_total:
        raise CalibratorInputContractError(
            "Input-contract preflight: every triple is 'exempt' (calendar "
            "says no model is published on any of these dates). The "
            "corpus is unusable for production-mirrored fitting."
        )
    if report.n_triples_full + report.n_triples_single_model_era < MIN_TRIPLES_FOR_FIT:
        raise CalibratorInputContractError(
            "Input-contract preflight: only "
            f"{report.n_triples_full + report.n_triples_single_model_era} "
            "triples in 'full' + 'single_model_era' buckets (need ≥100). "
            "Sample-size sanity gate failed before the fitter's own "
            "len(pairs) >= 100 check."
        )
    if report.n_triples_partial > 0 and not args.allow_partial_model_coverage:
        raise CalibratorInputContractError(
            f"Input-contract preflight: {report.n_triples_partial} "
            "post-cutover triples have only a subset of the expected "
            "ensemble models present. Pass --allow-partial-model-coverage "
            "to opt in."
        )
    if report.n_triples_empty > 0 and not args.allow_empty_cells:
        raise CalibratorInputContractError(
            f"Input-contract preflight: {report.n_triples_empty} triples "
            "have NO model rows present despite having a market and "
            "settlement. Pass --allow-empty-cells to opt in."
        )

    return bias_table, ensemble_weights, report
