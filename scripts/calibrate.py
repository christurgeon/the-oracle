"""
Calibration analysis for GFS ensemble weather forecasts vs NWS settlement actuals.

Loads daily snapshot data (forecasts, markets, settlements), joins forecast
probabilities to actual outcomes, and computes calibration metrics: Brier score
with Murphy decomposition, calibration curves with Wilson CIs, per-city bias,
lead-time analysis, PIT histogram, and strike-type breakdowns.

No API calls, no async — pure offline analysis of stored snapshot data.

Usage::

    uv run python scripts/calibrate.py [--snapshots DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# Add project root to path so the script works standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from python.common import Direction  # noqa: E402
from python.data.weather import (  # noqa: E402
    EnsembleForecast,
    compute_blended_probability,
    get_spread_inflation_factor,
)
from python.utils.ticker_dates import parse_ticker_date  # noqa: E402
from scripts._calibration_common import prefer_latest_settlement  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strike type → Direction mapping
# ---------------------------------------------------------------------------

_STRIKE_TYPE_TO_DIRECTION: dict[str, Direction] = {
    "greater": Direction.ABOVE,
    "less": Direction.BELOW,
    "between": Direction.BETWEEN,
}

# daily_variable → settlement field mapping
_VARIABLE_TO_SETTLEMENT_FIELD: dict[str, str] = {
    "temperature_2m_max": "high_f",
    "temperature_2m_min": "low_f",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastOutcome:
    """One forecast–outcome pair: the fundamental unit of calibration analysis."""

    city: str
    forecast_date: date
    daily_variable: str
    lead_time_days: int
    ensemble_temperatures: list[float]
    ensemble_mean: float
    ensemble_std: float
    member_count: int
    model_prob: float
    actual_temp: float
    outcome: bool  # True = YES resolved
    strike_type: str
    floor_strike: float | None
    cap_strike: float | None
    market_midpoint: float | None
    market_spread: float | None
    snapshot_date: date


@dataclass
class JoinStats:
    """Diagnostic counters for the forecast–settlement join."""

    total_markets: int = 0
    total_forecasts: int = 0
    total_settlements: int = 0
    matched_pairs: int = 0
    orphaned_no_settlement: int = 0
    orphaned_no_forecast: int = 0
    skipped_no_strike_type: int = 0
    skipped_no_forecast_date: int = 0
    skipped_preliminary: int = 0
    skipped_low_members: int = 0
    skipped_null_temp: int = 0
    settlement_dates: set[date] = field(default_factory=set)
    forecast_dates: set[date] = field(default_factory=set)
    member_counts: list[int] = field(default_factory=list)
    corrupt_settlements: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _derive_forecast_date(market: dict[str, Any]) -> date | None:
    """Derive forecast_date from a market dict (expiration_ts or ticker fallback)."""
    exp_ts = market.get("expiration_ts")
    if exp_ts is not None:
        try:
            dt = datetime.fromtimestamp(int(exp_ts), tz=UTC)
            return dt.date()
        except (OSError, OverflowError, ValueError):
            pass

    ticker = market.get("ticker", "")
    if ticker:
        return parse_ticker_date(ticker)

    return None


def _compute_model_prob(
    ensemble: EnsembleForecast,
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
) -> float | None:
    """Compute model probability using production NWS-rounded bracket math.

    Delegates to ``compute_blended_probability`` (single-model weight=1.0) so
    the ±0.5°F NWS-rounding window shipped 2026-05-21
    applies. Pre-2026-05-22 this called ``ensemble.fraction_*`` directly with
    no shift, which silently diverged from production for ~24h after the
    bracket fix shipped.
    """
    direction = _STRIKE_TYPE_TO_DIRECTION.get(strike_type)
    if direction is None:
        return None

    if direction == Direction.ABOVE and floor_strike is None:
        return None
    if direction == Direction.BELOW and cap_strike is None:
        return None
    if direction == Direction.BETWEEN and (floor_strike is None or cap_strike is None):
        return None

    blended, _per_model = compute_blended_probability(
        forecasts={"gfs_seamless": ensemble},
        weights={"gfs_seamless": 1.0},
        direction=direction,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
    )
    return blended


def _determine_outcome(
    actual_temp: float,
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
) -> bool | None:
    """Determine binary outcome using Kalshi-truth semantics.

    Delegates to ``python.analytics.outcomes.derive_outcome_yes`` — single
    source of truth. Prior to 2026-05-19 this function had drifted from
    the canonical helper (used ``>`` for greater instead of ``>=``); the
    delegation enforces consistency. For ``between`` the rule is
    CLOSED, right-inclusive (``floor <= t <= cap``).
    """
    return derive_outcome_yes(strike_type, floor_strike, cap_strike, actual_temp)


def _load_snapshot_dir(
    data_dir: Path,
    snap_date: date,
    all_markets: list[dict[str, Any]],
    all_forecasts: list[dict[str, Any]],
    all_settlements: list[dict[str, Any]],
    run_id: str | None = None,
) -> None:
    """Load JSON files from a single snapshot directory into the accumulator lists.

    Args:
        data_dir: Directory containing markets.json, weather_forecasts.json,
            and optionally settlements.json.
        snap_date: The snapshot date (used for ``_snapshot_date`` annotation).
        all_markets: Accumulator for market records.
        all_forecasts: Accumulator for forecast records.
        all_settlements: Accumulator for settlement records.
        run_id: Optional run identifier (e.g. "0200") to annotate records.
    """
    markets_path = data_dir / "markets.json"
    forecasts_path = data_dir / "weather_forecasts.json"
    settlements_path = data_dir / "settlements.json"

    if markets_path.exists():
        with open(markets_path) as f:
            markets = json.load(f)
        for m in markets:
            m["_snapshot_date"] = snap_date.isoformat()
            if run_id:
                m["_run_id"] = run_id
        all_markets.extend(markets)

    if forecasts_path.exists():
        with open(forecasts_path) as f:
            forecasts = json.load(f)
        for fc in forecasts:
            fc["_snapshot_date"] = snap_date.isoformat()
            if run_id:
                fc["_run_id"] = run_id
        all_forecasts.extend(forecasts)

    if settlements_path.exists():
        with open(settlements_path) as f:
            settlements = json.load(f)
        all_settlements.extend(settlements)


def load_snapshots(
    snapshots_dir: Path,
) -> tuple[
    list[dict[str, Any]],  # all markets across all snapshots
    list[dict[str, Any]],  # all forecasts
    list[dict[str, Any]],  # all settlements
    list[date],  # snapshot dates
]:
    """Load all snapshot data from data/snapshots/*/.

    Supports two directory layouts:

    - **Flat** (legacy): ``data/snapshots/YYYY-MM-DD/{markets,weather_forecasts,settlements}.json``
    - **Nested** (multi-run): ``data/snapshots/YYYY-MM-DD/<run_id>/{markets,...}.json``

    Both layouts can coexist. Within a date directory, if JSON files exist at
    the top level they are loaded (flat layout). Any subdirectories with names
    that look like run IDs (e.g. "0200", "1400") are also scanned.
    """
    all_markets: list[dict[str, Any]] = []
    all_forecasts: list[dict[str, Any]] = []
    all_settlements: list[dict[str, Any]] = []
    snapshot_dates: list[date] = []

    if not snapshots_dir.exists():
        logger.warning("Snapshots directory does not exist: %s", snapshots_dir)
        return all_markets, all_forecasts, all_settlements, snapshot_dates

    for child in sorted(snapshots_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            snap_date = date.fromisoformat(child.name)
        except ValueError:
            continue

        snapshot_dates.append(snap_date)

        # Flat layout: JSON files directly inside the date directory
        if (child / "markets.json").exists() or (child / "weather_forecasts.json").exists():
            _load_snapshot_dir(child, snap_date, all_markets, all_forecasts, all_settlements)

        # Nested layout: subdirectories (run IDs) inside the date directory
        for run_dir in sorted(child.iterdir()):
            if not run_dir.is_dir():
                continue
            # Skip hidden dirs; run IDs are typically "0200", "1400", etc.
            if run_dir.name.startswith("."):
                continue
            if (run_dir / "markets.json").exists() or (run_dir / "weather_forecasts.json").exists():
                _load_snapshot_dir(
                    run_dir,
                    snap_date,
                    all_markets,
                    all_forecasts,
                    all_settlements,
                    run_id=run_dir.name,
                )

    logger.info(
        "Loaded %d snapshot days: %d markets, %d forecasts, %d settlements",
        len(snapshot_dates),
        len(all_markets),
        len(all_forecasts),
        len(all_settlements),
    )
    return all_markets, all_forecasts, all_settlements, snapshot_dates


def build_forecast_outcome_pairs(
    all_markets: list[dict[str, Any]],
    all_forecasts: list[dict[str, Any]],
    all_settlements: list[dict[str, Any]],
    *,
    apply_spread_inflation: bool = False,
) -> tuple[list[ForecastOutcome], JoinStats]:
    """Join forecasts to settlements and produce calibration pairs.

    Settlement lookup is built across ALL snapshots — settlements from later
    snapshots verify forecasts from earlier ones. Prefers final reports over
    preliminary for the same (city, date).
    """
    stats = JoinStats()
    stats.total_markets = len(all_markets)
    stats.total_forecasts = len(all_forecasts)
    stats.total_settlements = len(all_settlements)

    # --- Build settlement lookup ---
    # Key: (city, observation_date) → {high_f, low_f, is_preliminary, ...}
    # Merge rule: the record with the latest (issued_at, product_id) wins.
    # NWS issues multiple products per observation date (morning partial,
    # afternoon partial, next-morning final); later issuance always subsumes
    # earlier. See prefer_latest_settlement in scripts/_calibration_common.
    #
    # DATA QUALITY: CHI settlements before 2026-03-20 were fetched from
    # O'Hare (ORD) instead of Midway (MDW). Kalshi settles KXHIGHCHI on
    # Midway. O'Hare temps differ by 1-3°F (urban heat island). These
    # tainted pairs are excluded from calibration.
    _chi_station_fix_date = date(2026, 3, 20)
    _chi_excluded = 0
    settlement_lookup: dict[tuple[str, date], dict[str, Any]] = {}
    for s in all_settlements:
        # Exclude tainted CHI settlements (wrong station before fix)
        if s["city"] == "CHI":
            obs_date_check = date.fromisoformat(s["observation_date"])
            if obs_date_check < _chi_station_fix_date:
                _chi_excluded += 1
                continue
        city = s["city"]
        obs_date = date.fromisoformat(s["observation_date"])
        key = (city, obs_date)

        stats.settlement_dates.add(obs_date)

        # Validate: high_f < low_f indicates corruption
        high_f = s.get("high_f")
        low_f = s.get("low_f")
        if high_f is not None and low_f is not None and high_f < low_f:
            stats.corrupt_settlements += 1
            logger.warning(
                "Corrupt settlement: %s %s high_f=%s < low_f=%s",
                city,
                obs_date,
                high_f,
                low_f,
            )

        existing = settlement_lookup.get(key)
        if existing is None or prefer_latest_settlement(existing, s):
            settlement_lookup[key] = s

    if _chi_excluded > 0:
        logger.warning(
            "Excluded %d CHI settlement(s) before %s (wrong station: ORD instead of MDW)",
            _chi_excluded,
            _chi_station_fix_date,
        )

    # --- Build forecast lookup ---
    # Key: (snapshot_date, city, forecast_date, daily_variable, model) → forecast dict.
    # GFS-only at the lookup site — production runtime serves GFS (with bias correction),
    # so the calibrator must train on the same distribution. ECMWF records are loaded
    # but never selected here.
    forecast_lookup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for fc in all_forecasts:
        snap_date = fc.get("_snapshot_date", "")
        model = fc.get("model")
        if model is None:
            raise ValueError(
                f"Forecast record missing required 'model' field: "
                f"snap_date={snap_date} city={fc.get('city')} "
                f"forecast_date={fc.get('forecast_date')}"
            )
        fc_key = (snap_date, fc["city"], fc["forecast_date"], fc["daily_variable"], model)
        forecast_lookup[fc_key] = fc
        stats.forecast_dates.add(date.fromisoformat(fc["forecast_date"]))

    # Track member count distribution
    for fc in all_forecasts:
        temps = fc.get("temperatures", [])
        stats.member_counts.append(len(temps))

    # --- Join markets to forecasts and settlements ---
    pairs: list[ForecastOutcome] = []

    for market in all_markets:
        strike_type = market.get("strike_type")
        if not strike_type or strike_type not in _STRIKE_TYPE_TO_DIRECTION:
            stats.skipped_no_strike_type += 1
            continue

        forecast_date = _derive_forecast_date(market)
        if forecast_date is None:
            stats.skipped_no_forecast_date += 1
            continue

        snap_date_str = market.get("_snapshot_date", "")
        snapshot_date = date.fromisoformat(snap_date_str) if snap_date_str else None
        if snapshot_date is None:
            continue

        city = market["city"]
        daily_variable = market["daily_variable"]

        # Find matching forecast from the same snapshot (GFS-only — see lookup build above).
        fc_key = (snap_date_str, city, forecast_date.isoformat(), daily_variable, "gfs_seamless")
        forecast = forecast_lookup.get(fc_key)
        if forecast is None:
            stats.orphaned_no_forecast += 1
            continue

        temperatures = forecast.get("temperatures", [])
        if len(temperatures) < 25:
            stats.skipped_low_members += 1
            continue

        # Find matching settlement (from any snapshot)
        settlement_field = _VARIABLE_TO_SETTLEMENT_FIELD.get(daily_variable)
        if settlement_field is None:
            continue

        settlement = settlement_lookup.get((city, forecast_date))
        if settlement is None:
            stats.orphaned_no_settlement += 1
            continue

        if settlement.get("is_preliminary", False):
            stats.skipped_preliminary += 1
            continue

        actual_temp = settlement.get(settlement_field)
        if actual_temp is None:
            stats.skipped_null_temp += 1
            continue

        actual_temp = float(actual_temp)

        # Reconstruct EnsembleForecast for model probability
        ensemble = EnsembleForecast(
            city=city,
            forecast_date=forecast_date,
            temperatures=temperatures,
        )

        # Apply spread inflation before computing probabilities (matches production)
        if apply_spread_inflation:
            factor = get_spread_inflation_factor(city, daily_variable)
            ensemble = ensemble.inflated(factor)

        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")
        if floor_strike is not None:
            floor_strike = float(floor_strike)
        if cap_strike is not None:
            cap_strike = float(cap_strike)

        model_prob = _compute_model_prob(ensemble, strike_type, floor_strike, cap_strike)
        if model_prob is None:
            continue

        outcome = _determine_outcome(actual_temp, strike_type, floor_strike, cap_strike)
        if outcome is None:
            continue

        # Market midpoint and spread
        yes_bid = market.get("yes_bid_dollars")
        yes_ask = market.get("yes_ask_dollars")
        midpoint = None
        spread = None
        if yes_bid is not None and yes_ask is not None:
            midpoint = (float(yes_bid) + float(yes_ask)) / 2.0
            spread = float(yes_ask) - float(yes_bid)

        lead_time = (forecast_date - snapshot_date).days

        pair = ForecastOutcome(
            city=city,
            forecast_date=forecast_date,
            daily_variable=daily_variable,
            lead_time_days=lead_time,
            ensemble_temperatures=ensemble.temperatures,
            ensemble_mean=ensemble.mean_temperature,
            ensemble_std=ensemble.std_dev,
            member_count=ensemble.member_count,
            model_prob=model_prob,
            actual_temp=actual_temp,
            outcome=outcome,
            strike_type=strike_type,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            market_midpoint=midpoint,
            market_spread=spread,
            snapshot_date=snapshot_date,
        )
        pairs.append(pair)

    stats.matched_pairs = len(pairs)
    return pairs, stats


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(frozen=True)
class CalibrationBin:
    """One bin in the calibration curve."""

    range_low: float
    range_high: float
    count: int
    mean_predicted: float
    observed_freq: float
    deviation: float  # observed - predicted (signed)
    ci_low: float
    ci_high: float
    ci_width: float


def compute_calibration_curve(
    pairs: list[ForecastOutcome], n_bins: int = 5
) -> list[CalibrationBin]:
    """Bin model probabilities and compute calibration metrics with Wilson CIs."""
    if not pairs:
        return []

    bin_width = 1.0 / n_bins
    bins: list[CalibrationBin] = []

    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width

        in_bin = [
            p
            for p in pairs
            if low <= p.model_prob < high or (i == n_bins - 1 and p.model_prob == high)
        ]

        if not in_bin:
            bins.append(
                CalibrationBin(
                    range_low=low,
                    range_high=high,
                    count=0,
                    mean_predicted=0.0,
                    observed_freq=0.0,
                    deviation=0.0,
                    ci_low=0.0,
                    ci_high=0.0,
                    ci_width=0.0,
                )
            )
            continue

        n = len(in_bin)
        mean_pred = statistics.mean(p.model_prob for p in in_bin)
        obs_freq = sum(1 for p in in_bin if p.outcome) / n
        ci_lo, ci_hi = _wilson_ci(obs_freq, n)

        bins.append(
            CalibrationBin(
                range_low=low,
                range_high=high,
                count=n,
                mean_predicted=mean_pred,
                observed_freq=obs_freq,
                deviation=obs_freq - mean_pred,
                ci_low=ci_lo,
                ci_high=ci_hi,
                ci_width=ci_hi - ci_lo,
            )
        )

    return bins


@dataclass(frozen=True)
class BrierDecomposition:
    """Brier score with Murphy decomposition."""

    brier_score: float
    reliability: float  # REL — lower is better
    resolution: float  # RES — higher is better
    uncertainty: float  # UNC — base rate variance
    brier_skill_score: float  # BSS = 1 - BS/UNC
    bs_ci_low: float
    bs_ci_high: float
    n: int


def compute_brier(
    pairs: list[ForecastOutcome], n_bins: int = 5, n_bootstrap: int = 10000
) -> BrierDecomposition | None:
    """Brier score with Murphy decomposition and block-bootstrap CI."""
    if not pairs:
        return None

    n = len(pairs)
    outcomes = [1.0 if p.outcome else 0.0 for p in pairs]
    preds = [p.model_prob for p in pairs]

    bs = sum((p - y) ** 2 for p, y in zip(preds, outcomes)) / n
    o_bar = sum(outcomes) / n
    unc = o_bar * (1.0 - o_bar)

    # Murphy decomposition using calibration bins
    bin_width = 1.0 / n_bins
    rel = 0.0
    res = 0.0
    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        in_bin = [
            (p, y)
            for p, y in zip(preds, outcomes)
            if low <= p < high or (i == n_bins - 1 and p == high)
        ]
        if not in_bin:
            continue
        n_b = len(in_bin)
        p_bar = statistics.mean(p for p, _ in in_bin)
        o_bar_b = statistics.mean(y for _, y in in_bin)
        rel += n_b * (p_bar - o_bar_b) ** 2
        res += n_b * (o_bar_b - o_bar) ** 2

    rel /= n
    res /= n
    bss = 1.0 - bs / unc if unc > 0 else 0.0

    # Block bootstrap by date (all cities on same date move together)
    date_groups: dict[date, list[int]] = {}
    for idx, p in enumerate(pairs):
        date_groups.setdefault(p.forecast_date, []).append(idx)

    date_keys = list(date_groups.keys())
    rng = random.Random(42)
    boot_bs: list[float] = []

    if len(date_keys) >= 2:
        for _ in range(n_bootstrap):
            sampled_indices: list[int] = []
            for _ in range(len(date_keys)):
                dk = rng.choice(date_keys)
                sampled_indices.extend(date_groups[dk])
            if not sampled_indices:
                continue
            bs_sample = sum((preds[i] - outcomes[i]) ** 2 for i in sampled_indices) / len(
                sampled_indices
            )
            boot_bs.append(bs_sample)

    if boot_bs:
        boot_bs.sort()
        ci_low = boot_bs[int(0.025 * len(boot_bs))]
        ci_high = boot_bs[int(0.975 * len(boot_bs))]
    else:
        ci_low = bs
        ci_high = bs

    return BrierDecomposition(
        brier_score=bs,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        brier_skill_score=bss,
        bs_ci_low=ci_low,
        bs_ci_high=ci_high,
        n=n,
    )


@dataclass(frozen=True)
class CityBias:
    """Per-city temperature bias metrics."""

    city: str
    daily_variable: str
    count: int
    bias: float  # mean(ensemble_mean - actual)
    mae: float
    rmse: float
    brier: float


def compute_city_bias(pairs: list[ForecastOutcome]) -> list[CityBias]:
    """Compute per-city, per-variable temperature bias."""
    if not pairs:
        return []

    groups: dict[tuple[str, str], list[ForecastOutcome]] = {}
    for p in pairs:
        key = (p.city, p.daily_variable)
        groups.setdefault(key, []).append(p)

    results: list[CityBias] = []
    for (city, dv), group in sorted(groups.items()):
        # Deduplicate by (forecast_date) to get unique forecast events
        seen_events: dict[date, ForecastOutcome] = {}
        for p in group:
            if p.forecast_date not in seen_events:
                seen_events[p.forecast_date] = p

        unique = list(seen_events.values())
        errors = [p.ensemble_mean - p.actual_temp for p in unique]
        abs_errors = [abs(e) for e in errors]
        sq_errors = [e**2 for e in errors]

        bias = statistics.mean(errors) if errors else 0.0
        mae = statistics.mean(abs_errors) if abs_errors else 0.0
        rmse = math.sqrt(statistics.mean(sq_errors)) if sq_errors else 0.0

        # Brier score uses all pairs (one per market bracket)
        brier = sum((p.model_prob - (1.0 if p.outcome else 0.0)) ** 2 for p in group) / len(group)

        results.append(
            CityBias(
                city=city,
                daily_variable=dv,
                count=len(group),
                bias=bias,
                mae=mae,
                rmse=rmse,
                brier=brier,
            )
        )
    return results


@dataclass(frozen=True)
class LeadTimeStats:
    """Metrics grouped by forecast lead time."""

    lead_time_days: int
    count: int
    brier: float
    rmse: float
    mean_spread: float
    spread_skill_ratio: float  # mean(std) / RMSE — should be ~1.0


def compute_lead_time(pairs: list[ForecastOutcome]) -> list[LeadTimeStats]:
    """Compute Brier score and RMSE by lead time."""
    if not pairs:
        return []

    groups: dict[int, list[ForecastOutcome]] = {}
    for p in pairs:
        groups.setdefault(p.lead_time_days, []).append(p)

    results: list[LeadTimeStats] = []
    for lt, group in sorted(groups.items()):
        brier = sum((p.model_prob - (1.0 if p.outcome else 0.0)) ** 2 for p in group) / len(group)

        # RMSE from unique forecast events
        seen: dict[tuple[str, date], ForecastOutcome] = {}
        for p in group:
            key = (p.city, p.forecast_date)
            if key not in seen:
                seen[key] = p
        unique = list(seen.values())
        sq_errors = [(p.ensemble_mean - p.actual_temp) ** 2 for p in unique]
        rmse = math.sqrt(statistics.mean(sq_errors)) if sq_errors else 0.0

        mean_spread = statistics.mean(p.ensemble_std for p in unique)
        ssr = mean_spread / rmse if rmse > 0 else float("inf")

        results.append(
            LeadTimeStats(
                lead_time_days=lt,
                count=len(group),
                brier=brier,
                rmse=rmse,
                mean_spread=mean_spread,
                spread_skill_ratio=ssr,
            )
        )
    return results


def compute_ece(pairs: list[ForecastOutcome], n_bins: int = 5) -> float:
    """Expected Calibration Error: weighted average of bin calibration errors."""
    if not pairs:
        return 0.0
    n = len(pairs)
    bin_width = 1.0 / n_bins
    ece = 0.0
    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        in_bin = [
            p
            for p in pairs
            if low <= p.model_prob < high or (i == n_bins - 1 and p.model_prob == high)
        ]
        if not in_bin:
            continue
        n_b = len(in_bin)
        obs = sum(1 for p in in_bin if p.outcome) / n_b
        pred = statistics.mean(p.model_prob for p in in_bin)
        ece += n_b * abs(obs - pred)
    return ece / n


def compute_log_loss(pairs: list[ForecastOutcome]) -> float:
    """Log loss with probabilities clipped to [0.01, 0.99]."""
    if not pairs:
        return 0.0
    total = 0.0
    for p in pairs:
        prob = max(0.01, min(0.99, p.model_prob))
        y = 1.0 if p.outcome else 0.0
        total += y * math.log(prob) + (1.0 - y) * math.log(1.0 - prob)
    return -total / len(pairs)


@dataclass(frozen=True)
class StrikeTypeStats:
    """Metrics grouped by strike type."""

    strike_type: str
    count: int
    brier: float
    ece: float
    mean_predicted: float
    observed_freq: float


def compute_strike_type_breakdown(
    pairs: list[ForecastOutcome],
) -> list[StrikeTypeStats]:
    """Compute metrics per strike type."""
    if not pairs:
        return []

    groups: dict[str, list[ForecastOutcome]] = {}
    for p in pairs:
        groups.setdefault(p.strike_type, []).append(p)

    results: list[StrikeTypeStats] = []
    for st, group in sorted(groups.items()):
        n = len(group)
        brier = sum((p.model_prob - (1.0 if p.outcome else 0.0)) ** 2 for p in group) / n
        ece = compute_ece(group)
        mean_pred = statistics.mean(p.model_prob for p in group)
        obs_freq = sum(1 for p in group if p.outcome) / n
        results.append(
            StrikeTypeStats(
                strike_type=st,
                count=n,
                brier=brier,
                ece=ece,
                mean_predicted=mean_pred,
                observed_freq=obs_freq,
            )
        )
    return results


@dataclass(frozen=True)
class PITBin:
    """One bin of the PIT (Probability Integral Transform) histogram."""

    range_low: float
    range_high: float
    observed: int
    expected: float


def compute_pit_histogram(pairs: list[ForecastOutcome], n_bins: int = 10) -> list[PITBin]:
    """PIT histogram: fraction of ensemble members <= actual, binned.

    Computed per unique (city, forecast_date, daily_variable) event.
    Should be approximately uniform if the ensemble is well-calibrated.
    """
    if not pairs:
        return []

    # Deduplicate to unique forecast events
    seen: dict[tuple[str, date, str], ForecastOutcome] = {}
    for p in pairs:
        key = (p.city, p.forecast_date, p.daily_variable)
        if key not in seen:
            seen[key] = p

    # Compute PIT values
    pit_values: list[float] = []
    for p in seen.values():
        if not p.ensemble_temperatures:
            continue
        n_below_or_equal = sum(1 for t in p.ensemble_temperatures if t <= p.actual_temp)
        pit = n_below_or_equal / len(p.ensemble_temperatures)
        pit_values.append(pit)

    if not pit_values:
        return []

    bin_width = 1.0 / n_bins
    expected_per_bin = len(pit_values) / n_bins
    bins: list[PITBin] = []
    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        count = sum(1 for v in pit_values if low <= v < high or (i == n_bins - 1 and v == high))
        bins.append(
            PITBin(
                range_low=low,
                range_high=high,
                observed=count,
                expected=expected_per_bin,
            )
        )
    return bins


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_data_quality(stats: JoinStats, snapshot_dates: list[date]) -> None:
    """Print the data quality / join diagnostics report."""
    _print_header("DATA QUALITY REPORT")

    print(f"  Snapshot days loaded:    {len(snapshot_dates)}")
    if snapshot_dates:
        print(f"  Snapshot date range:    {min(snapshot_dates)} → {max(snapshot_dates)}")
    print(f"  Total markets:          {stats.total_markets}")
    print(f"  Total forecasts:        {stats.total_forecasts}")
    print(f"  Total settlements:      {stats.total_settlements}")
    print()

    if stats.forecast_dates:
        print(f"  Forecast dates:         {sorted(stats.forecast_dates)}")
    if stats.settlement_dates:
        print(f"  Settlement dates:       {sorted(stats.settlement_dates)}")
    print()

    print(f"  Matched pairs:          {stats.matched_pairs}")
    print(f"  No settlement found:    {stats.orphaned_no_settlement}")
    print(f"  No forecast found:      {stats.orphaned_no_forecast}")
    print(f"  No strike type:         {stats.skipped_no_strike_type}")
    print(f"  No forecast date:       {stats.skipped_no_forecast_date}")
    print(f"  Preliminary settlement: {stats.skipped_preliminary}")
    print(f"  Low member count (<25): {stats.skipped_low_members}")
    print(f"  Null temperature:       {stats.skipped_null_temp}")
    print(f"  Corrupt settlements:    {stats.corrupt_settlements}")

    if stats.member_counts:
        from collections import Counter

        mc = Counter(stats.member_counts)
        print("\n  Ensemble member count distribution:")
        for count, freq in sorted(mc.items()):
            print(f"    {count} members: {freq} forecasts")


def print_zero_pairs_diagnostic(stats: JoinStats) -> None:
    """Print a clear explanation of why 0 pairs were joined."""
    _print_header("ZERO PAIRS DIAGNOSTIC")

    fc_dates = sorted(stats.forecast_dates) if stats.forecast_dates else []
    st_dates = sorted(stats.settlement_dates) if stats.settlement_dates else []

    print("  No forecast-outcome pairs could be joined.\n")

    if fc_dates:
        print(f"  Forecasts cover dates:    {fc_dates[0]} → {fc_dates[-1]}")
    else:
        print("  Forecasts cover dates:    (none)")

    if st_dates:
        print(f"  Settlements cover dates:  {st_dates[0]} → {st_dates[-1]}")
    else:
        print("  Settlements cover dates:  (none)")

    overlap = set(fc_dates) & set(st_dates)
    if overlap:
        print(f"\n  Overlapping dates: {sorted(overlap)}")
        print("  Pairs should exist — check if preliminary-only or missing fields.")
    else:
        print("\n  No date overlap between forecasts and settlements.")
        if fc_dates and st_dates:
            gap_days = (min(fc_dates) - max(st_dates)).days
            print(f"  Gap: {gap_days} day(s) between latest settlement and earliest forecast.")
            print("\n  Forecasts predict future dates; settlements record past dates.")
            print("  Once a forecast date passes and NWS publishes the settlement,")
            print("  the next snapshot will contain both → pairs will form.")
            if gap_days > 0:
                print(f"\n  Estimated first joinable data: ~{gap_days} day(s) from now.")


def print_overall_summary(
    pairs: list[ForecastOutcome],
    brier: BrierDecomposition,
    ece: float,
    log_loss: float,
    snapshot_dates: list[date],
) -> None:
    """Print the overall summary table."""
    _print_header("OVERALL SUMMARY")

    unique_events = len({(p.city, p.forecast_date, p.daily_variable) for p in pairs})
    sharpness = statistics.mean(abs(p.model_prob - 0.5) for p in pairs)

    print(f"  Forecast-outcome pairs: {brier.n}")
    print(f"  Unique forecast events: {unique_events}")
    print(f"  Days of data:           {len(snapshot_dates)}")
    print(f"  Base rate (YES):        {sum(1 for p in pairs if p.outcome) / len(pairs):.3f}")
    print()
    print(f"  Brier Score:            {brier.brier_score:.4f}")
    print(f"    95% CI (bootstrap):   [{brier.bs_ci_low:.4f}, {brier.bs_ci_high:.4f}]")
    print(f"  BSS (Brier Skill):      {brier.brier_skill_score:.4f}")
    print(f"    REL (reliability):    {brier.reliability:.4f}")
    print(f"    RES (resolution):     {brier.resolution:.4f}")
    print(f"    UNC (uncertainty):    {brier.uncertainty:.4f}")
    print(f"  ECE:                    {ece:.4f}")
    print(f"  Log Loss:               {log_loss:.4f}")
    print(f"  Sharpness:              {sharpness:.4f}")


def print_calibration_curve(bins: list[CalibrationBin]) -> None:
    """Print the calibration curve table."""
    _print_header("CALIBRATION CURVE")

    header = (
        f"  {'Bin':>12s}  {'N':>5s}  {'Predicted':>9s}  {'Observed':>8s}  "
        f"{'Deviation':>9s}  {'95% CI':>15s}  {'CI Width':>8s}"
    )
    print(header)
    print(f"  {'-' * 12}  {'-' * 5}  {'-' * 9}  {'-' * 8}  {'-' * 9}  {'-' * 15}  {'-' * 8}")

    for b in bins:
        label = f"{b.range_low:.0%}-{b.range_high:.0%}"
        if b.count == 0:
            print(
                f"  {label:>12s}  {b.count:>5d}  {'--':>9s}  {'--':>8s}  "
                f"{'--':>9s}  {'--':>15s}  {'--':>8s}"
            )
        else:
            ci_str = f"[{b.ci_low:.3f}, {b.ci_high:.3f}]"
            dev_str = f"{b.deviation:+.3f}"
            print(
                f"  {label:>12s}  {b.count:>5d}  {b.mean_predicted:>9.3f}  "
                f"{b.observed_freq:>8.3f}  {dev_str:>9s}  {ci_str:>15s}  {b.ci_width:>8.3f}"
            )


def print_city_bias(biases: list[CityBias]) -> None:
    """Print per-city bias table."""
    _print_header("PER-CITY TEMPERATURE BIAS")

    header = (
        f"  {'City':>5s}  {'Variable':>18s}  {'N':>5s}  "
        f"{'Bias':>7s}  {'MAE':>6s}  {'RMSE':>6s}  {'Brier':>7s}"
    )
    print(header)
    print(f"  {'-' * 5}  {'-' * 18}  {'-' * 5}  {'-' * 7}  {'-' * 6}  {'-' * 6}  {'-' * 7}")

    for b in biases:
        var_label = "temp_max" if "max" in b.daily_variable else "temp_min"
        print(
            f"  {b.city:>5s}  {var_label:>18s}  {b.count:>5d}  "
            f"{b.bias:>+7.2f}  {b.mae:>6.2f}  {b.rmse:>6.2f}  {b.brier:>7.4f}"
        )


def print_lead_time(lead_times: list[LeadTimeStats]) -> None:
    """Print lead-time analysis table."""
    _print_header("LEAD-TIME ANALYSIS")

    header = (
        f"  {'Lead':>5s}  {'N':>5s}  {'Brier':>7s}  {'RMSE':>6s}  {'Spread':>7s}  {'S/S Ratio':>9s}"
    )
    print(header)
    print(f"  {'-' * 5}  {'-' * 5}  {'-' * 7}  {'-' * 6}  {'-' * 7}  {'-' * 9}")

    for lt in lead_times:
        ssr_str = f"{lt.spread_skill_ratio:.3f}" if lt.spread_skill_ratio < 100 else "inf"
        print(
            f"  {lt.lead_time_days:>4d}d  {lt.count:>5d}  {lt.brier:>7.4f}  "
            f"{lt.rmse:>6.2f}  {lt.mean_spread:>7.2f}  {ssr_str:>9s}"
        )


def print_strike_type(breakdown: list[StrikeTypeStats]) -> None:
    """Print strike-type breakdown table."""
    _print_header("STRIKE TYPE BREAKDOWN")

    header = (
        f"  {'Type':>8s}  {'N':>5s}  {'Brier':>7s}  "
        f"{'ECE':>7s}  {'Predicted':>9s}  {'Observed':>8s}"
    )
    print(header)
    print(f"  {'-' * 8}  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 9}  {'-' * 8}")

    for s in breakdown:
        print(
            f"  {s.strike_type:>8s}  {s.count:>5d}  {s.brier:>7.4f}  "
            f"{s.ece:>7.4f}  {s.mean_predicted:>9.3f}  {s.observed_freq:>8.3f}"
        )


def print_pit_histogram(pit_bins: list[PITBin]) -> None:
    """Print PIT histogram table."""
    _print_header("PIT HISTOGRAM")

    print("  (Uniform = well-calibrated | U-shaped = underdispersed | Dome = overdispersed)\n")
    header = f"  {'Bin':>12s}  {'Observed':>8s}  {'Expected':>8s}  {'Ratio':>6s}  {'Bar':>20s}"
    print(header)
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 8}  {'-' * 6}  {'-' * 20}")

    max_obs = max((b.observed for b in pit_bins), default=1)
    for b in pit_bins:
        label = f"{b.range_low:.1f}-{b.range_high:.1f}"
        ratio = b.observed / b.expected if b.expected > 0 else 0.0
        bar_len = int(20 * b.observed / max_obs) if max_obs > 0 else 0
        bar = "#" * bar_len
        print(f"  {label:>12s}  {b.observed:>8d}  {b.expected:>8.1f}  {ratio:>6.2f}  {bar:<20s}")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(
    pairs: list[ForecastOutcome],
    brier: BrierDecomposition | None,
    cal_bins: list[CalibrationBin],
    city_biases: list[CityBias],
    lead_times: list[LeadTimeStats],
    strike_breakdown: list[StrikeTypeStats],
    pit_bins: list[PITBin],
    ece: float,
    log_loss_val: float,
    stats: JoinStats,
    output_path: Path,
) -> None:
    """Save machine-readable calibration results to JSON."""
    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_pairs": len(pairs),
        "join_stats": {
            "total_markets": stats.total_markets,
            "total_forecasts": stats.total_forecasts,
            "total_settlements": stats.total_settlements,
            "matched_pairs": stats.matched_pairs,
            "orphaned_no_settlement": stats.orphaned_no_settlement,
            "orphaned_no_forecast": stats.orphaned_no_forecast,
        },
    }

    if brier:
        result["overall"] = {
            "brier_score": brier.brier_score,
            "bss": brier.brier_skill_score,
            "reliability": brier.reliability,
            "resolution": brier.resolution,
            "uncertainty": brier.uncertainty,
            "bs_ci_low": brier.bs_ci_low,
            "bs_ci_high": brier.bs_ci_high,
            "ece": ece,
            "log_loss": log_loss_val,
            "base_rate": sum(1 for p in pairs if p.outcome) / len(pairs) if pairs else 0,
        }

    result["calibration_bins"] = [
        {
            "range": f"{b.range_low:.0%}-{b.range_high:.0%}",
            "count": b.count,
            "mean_predicted": round(b.mean_predicted, 4),
            "observed_freq": round(b.observed_freq, 4),
            "deviation": round(b.deviation, 4),
            "ci_low": round(b.ci_low, 4),
            "ci_high": round(b.ci_high, 4),
        }
        for b in cal_bins
    ]

    result["city_bias"] = [
        {
            "city": cb.city,
            "variable": cb.daily_variable,
            "count": cb.count,
            "bias": round(cb.bias, 2),
            "mae": round(cb.mae, 2),
            "rmse": round(cb.rmse, 2),
            "brier": round(cb.brier, 4),
        }
        for cb in city_biases
    ]

    result["lead_time"] = [
        {
            "lead_days": lt.lead_time_days,
            "count": lt.count,
            "brier": round(lt.brier, 4),
            "rmse": round(lt.rmse, 2),
            "mean_spread": round(lt.mean_spread, 2),
            "spread_skill_ratio": round(lt.spread_skill_ratio, 3),
        }
        for lt in lead_times
    ]

    result["strike_type"] = [
        {
            "type": s.strike_type,
            "count": s.count,
            "brier": round(s.brier, 4),
            "ece": round(s.ece, 4),
            "mean_predicted": round(s.mean_predicted, 3),
            "observed_freq": round(s.observed_freq, 3),
        }
        for s in strike_breakdown
    ]

    result["pit_histogram"] = [
        {
            "range": f"{b.range_low:.1f}-{b.range_high:.1f}",
            "observed": b.observed,
            "expected": round(b.expected, 1),
        }
        for b in pit_bins
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    snapshots_dir: Path,
    output_path: Path,
    *,
    spread_inflation: bool = False,
) -> None:
    """Run calibration analysis on stored snapshot data."""
    # Load all snapshot data
    all_markets, all_forecasts, all_settlements, snapshot_dates = load_snapshots(snapshots_dir)

    if not snapshot_dates:
        print("No snapshot data found in", snapshots_dir)
        return

    # Build forecast-outcome pairs
    pairs, stats = build_forecast_outcome_pairs(
        all_markets,
        all_forecasts,
        all_settlements,
        apply_spread_inflation=spread_inflation,
    )

    if spread_inflation:
        logger.info("Spread inflation ENABLED — per-city factors applied to ensemble")

    # Always print data quality first
    print_data_quality(stats, snapshot_dates)

    if not pairs:
        print_zero_pairs_diagnostic(stats)
        # Still export the (empty) JSON
        export_json(
            pairs=[],
            brier=None,
            cal_bins=[],
            city_biases=[],
            lead_times=[],
            strike_breakdown=[],
            pit_bins=[],
            ece=0.0,
            log_loss_val=0.0,
            stats=stats,
            output_path=output_path,
        )
        return

    # Auto-select bin count: 10 bins if N > 500, else 5
    n_bins = 10 if len(pairs) > 500 else 5

    # Compute all metrics
    cal_bins = compute_calibration_curve(pairs, n_bins=n_bins)
    brier = compute_brier(pairs, n_bins=n_bins)
    ece = compute_ece(pairs, n_bins=n_bins)
    ll = compute_log_loss(pairs)
    city_biases = compute_city_bias(pairs)
    lead_times = compute_lead_time(pairs)
    strike_breakdown = compute_strike_type_breakdown(pairs)
    pit_bins = compute_pit_histogram(pairs)

    # Print all tables
    if brier:
        print_overall_summary(pairs, brier, ece, ll, snapshot_dates)
    print_calibration_curve(cal_bins)
    print_city_bias(city_biases)
    print_lead_time(lead_times)
    print_strike_type(strike_breakdown)
    print_pit_histogram(pit_bins)

    # Export JSON
    export_json(
        pairs=pairs,
        brier=brier,
        cal_bins=cal_bins,
        city_biases=city_biases,
        lead_times=lead_times,
        strike_breakdown=strike_breakdown,
        pit_bins=pit_bins,
        ece=ece,
        log_loss_val=ll,
        stats=stats,
        output_path=output_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration analysis: GFS ensemble forecasts vs NWS settlement actuals.",
    )
    parser.add_argument(
        "--snapshots",
        default="data/snapshots",
        help="Snapshots directory (default: data/snapshots)",
    )
    parser.add_argument(
        "--output",
        default="data/calibration_results.json",
        help="Output JSON path (default: data/calibration_results.json)",
    )
    parser.add_argument(
        "--spread-inflation",
        action="store_true",
        default=False,
        help="Apply per-city ensemble spread inflation before computing probabilities",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args()
    main(
        snapshots_dir=Path(args.snapshots),
        output_path=Path(args.output),
        spread_inflation=args.spread_inflation,
    )
