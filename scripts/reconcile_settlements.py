"""
Settlement reconciliation: compare NWS-predicted results to Kalshi actuals.

Guards against silent changes in Kalshi's settlement source (e.g., switching
from Midway to O'Hare for Chicago, changing data providers, etc.).

For each recently settled weather market, this script:
  1. Looks up what the NWS CLI temperature was for that city/date
  2. Determines the expected result (YES/NO) based on the market's strike
  3. Compares to Kalshi's actual settlement result
  4. Flags any mismatches as potential settlement source changes

Usage::

    # Check settlements from collected snapshots
    uv run python scripts/reconcile_settlements.py

    # Check a specific date
    uv run python scripts/reconcile_settlements.py --date 2026-03-19

    # Fetch fresh settled markets from Kalshi API and reconcile
    uv run python scripts/reconcile_settlements.py --live

A mismatch does NOT necessarily mean a bug — it could indicate:
  - Kalshi changed the settlement station for a city
  - NWS revised a preliminary report
  - A bracket boundary edge case (temp exactly on the strike)
  - Our NWS parser missed a value

Any mismatch should be investigated before trusting calibration data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.outcomes import derive_outcome_yes  # noqa: E402
from scripts._calibration_common import prefer_latest_settlement  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationResult:
    """Result of comparing NWS-expected vs Kalshi-actual settlement."""

    ticker: str
    city: str
    event_date: str
    strike_type: str
    floor_strike: float | None
    cap_strike: float | None
    nws_temp: int | None  # NWS observed temperature (high or low)
    temp_type: str  # "high" or "low"
    expected_result: str | None  # "yes" or "no" based on NWS temp
    kalshi_result: str | None  # "yes" or "no" from Kalshi API
    match: bool | None  # True if they agree, None if can't determine


# ---------------------------------------------------------------------------
# Settlement logic
# ---------------------------------------------------------------------------


def determine_expected_result(
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
    observed_temp: int,
) -> str:
    """Given a market's strike and the actual temperature, what should the result be?

    Delegates to ``python.analytics.outcomes.derive_outcome_yes`` — single
    source of Kalshi-truth semantics (between is CLOSED, right-inclusive;
    greater is left-inclusive ``>=``; less is strict ``<``). Prior to
    2026-05-19 this used strict ``>`` for ``greater`` — corrected for
    consistency with the helper.
    """
    yes = derive_outcome_yes(strike_type, floor_strike, cap_strike, float(observed_temp))
    if yes is None:
        return "unknown"
    return "yes" if yes else "no"


def reconcile_market(
    market: dict[str, Any],
    settlements: dict[tuple[str, str, str], int],
) -> ReconciliationResult | None:
    """Reconcile a single settled market against NWS data.

    Args:
        market: Kalshi market dict (must have result, strike_type, etc.)
        settlements: Map of (city, date_iso, "high"|"low") -> observed temp

    Returns:
        ReconciliationResult or None if insufficient data.
    """
    ticker = market.get("ticker", "")
    result = market.get("result", "")
    strike_type = market.get("strike_type", "")
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")

    if not result or result not in ("yes", "no"):
        return None

    # Determine city and temp type from series ticker or event_ticker
    series = market.get("series_ticker", "")
    if not series:
        # Derive series from event_ticker (e.g., "KXHIGHNY-26MAR18" → "KXHIGHNY")
        event_ticker = market.get("event_ticker", "")
        parts = event_ticker.split("-")
        if parts:
            series = parts[0]
    city, temp_type = _parse_series(series)
    if city is None:
        return None

    # Determine event date
    event_date = market.get("event_date")
    if event_date is None:
        return None

    # Look up NWS observed temp
    key = (city, event_date, temp_type)
    nws_temp = settlements.get(key)

    expected = None
    match = None

    if nws_temp is not None and strike_type:
        if floor_strike is not None:
            floor_strike = float(floor_strike)
        if cap_strike is not None:
            cap_strike = float(cap_strike)
        expected = determine_expected_result(strike_type, floor_strike, cap_strike, nws_temp)
        if expected != "unknown":
            match = expected == result

    return ReconciliationResult(
        ticker=ticker,
        city=city,
        event_date=event_date,
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        nws_temp=nws_temp,
        temp_type=temp_type,
        expected_result=expected,
        kalshi_result=result,
        match=match,
    )


def _parse_series(series_ticker: str) -> tuple[str | None, str]:
    """Extract city and temp type from a series ticker like KXHIGHNY or KXLOWTCHI."""
    from python.strategies.weather_edge import WEATHER_SERIES

    if series_ticker in WEATHER_SERIES:
        city, daily_var = WEATHER_SERIES[series_ticker]
        temp_type = "low" if "min" in str(daily_var).lower() else "high"
        return city, temp_type
    return None, "high"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _ingest_settlements_file(
    settlements_path: Path,
    records_by_key: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Parse a single settlements.json and merge records into the keyed lookup.

    Rule: for each ``(city, observation_date)``, the record with the latest
    ``(issued_at, product_id)`` wins. See ``prefer_latest_settlement`` in
    ``scripts/_calibration_common`` for the canonical semantics.
    """
    try:
        with open(settlements_path) as f:
            records = json.load(f)
        for rec in records:
            city = rec.get("city", "")
            obs_date = rec.get("observation_date", "")
            if not city or not obs_date:
                continue
            key = (city, obs_date)
            existing = records_by_key.get(key)
            if existing is None or prefer_latest_settlement(existing, rec):
                records_by_key[key] = rec
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load %s: %s", settlements_path, exc)


def load_nws_settlements(data_dir: Path) -> dict[tuple[str, str, str], int]:
    """Load all NWS settlement observations from snapshot directories.

    Supports both flat (``<date>/settlements.json``) and nested
    (``<date>/<run_id>/settlements.json``) directory layouts.

    Returns:
        Map of (city, date_iso, "high"|"low") -> observed temperature.
    """
    # Intermediate: keep the full winning record per (city, obs_date) so the
    # merge tiebreak on (issued_at, product_id) is deterministic.
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue

        # Flat layout: settlements.json directly inside date dir
        flat_path = subdir / "settlements.json"
        if flat_path.exists():
            _ingest_settlements_file(flat_path, records_by_key)

        # Nested layout: settlements.json inside run-id subdirectories
        for run_dir in sorted(subdir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            nested_path = run_dir / "settlements.json"
            if nested_path.exists():
                _ingest_settlements_file(nested_path, records_by_key)

    # Explode to (city, date, "high"|"low") -> int for the downstream API.
    settlements: dict[tuple[str, str, str], int] = {}
    for (city, obs_date), rec in records_by_key.items():
        high = rec.get("high_f")
        low = rec.get("low_f")
        if high is not None:
            settlements[(city, obs_date, "high")] = int(high)
        if low is not None:
            settlements[(city, obs_date, "low")] = int(low)

    logger.info("Loaded %d NWS settlement observations", len(settlements))
    return settlements


def _recent_event_tickers(series_list: list[str], lookback_days: int = 5) -> list[str]:
    """Generate event tickers for recent trading days to limit API calls.

    Instead of fetching ALL settled markets (thousands), only query
    event tickers for the last N trading days. E.g., KXHIGHNY-26MAR18.
    """
    month_names = {
        1: "JAN",
        2: "FEB",
        3: "MAR",
        4: "APR",
        5: "MAY",
        6: "JUN",
        7: "JUL",
        8: "AUG",
        9: "SEP",
        10: "OCT",
        11: "NOV",
        12: "DEC",
    }

    today = date.today()
    tickers = []
    trading_days_found = 0
    days_back = 0
    while trading_days_found < lookback_days:
        days_back += 1
        d = today - timedelta(days=days_back)
        if d.weekday() >= 5:
            continue  # Skip weekends
        trading_days_found += 1
        date_str = f"{d.year % 100:02d}{month_names[d.month]}{d.day:02d}"
        for series in series_list:
            tickers.append(f"{series}-{date_str}")
    return tickers


async def fetch_settled_markets(
    series_list: list[str],
    lookback_days: int = 5,
) -> list[dict[str, Any]]:
    """Fetch recently settled markets from Kalshi API.

    Queries by event_ticker (series + date) to avoid fetching the entire
    history of settled markets. Only checks the last N days.
    """
    from python.data.kalshi_client import KalshiClient

    api_key = os.environ.get("ORACLE_KALSHI_API_KEY", "")
    key_path = os.environ.get("ORACLE_KALSHI_PRIVATE_KEY_PATH", "")

    if not api_key or not key_path:
        try:
            from python.settings import Settings

            settings = Settings.load()
            api_key = api_key or settings.kalshi_api_key
            key_path = key_path or settings.kalshi_private_key_path
        except Exception:
            pass

    if not api_key or not key_path:
        raise RuntimeError("Kalshi credentials not available")

    event_tickers = _recent_event_tickers(series_list, lookback_days)
    logger.info("Querying %d recent event tickers for settlement data", len(event_tickers))

    all_markets: list[dict[str, Any]] = []

    async with KalshiClient(api_key=api_key, private_key_path=key_path) as client:
        for i, event_ticker in enumerate(event_tickers):
            if i > 0:
                await asyncio.sleep(0.3)
            try:
                resp = await client.get_markets(event_ticker=event_ticker)
                markets = resp.get("markets", [])
                # Only keep settled markets with a result
                for m in markets:
                    if m.get("result") in ("yes", "no") and m.get("status") in (
                        "settled",
                        "finalized",
                    ):
                        all_markets.append(m)
            except Exception:
                logger.warning("Failed to fetch %s", event_ticker, exc_info=True)

    logger.info("Fetched %d settled markets from Kalshi", len(all_markets))
    return all_markets


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_reconciliation(results: list[ReconciliationResult]) -> None:
    """Print reconciliation results with mismatches highlighted."""
    if not results:
        print("\n  No settlements to reconcile.")
        return

    mismatches = [r for r in results if r.match is False]
    matched = [r for r in results if r.match is True]
    unknown = [r for r in results if r.match is None]

    print(f"\n{'=' * 90}")
    print("  SETTLEMENT RECONCILIATION")
    print(f"{'=' * 90}")
    print(f"  Matched:   {len(matched)}")
    print(f"  Mismatch:  {len(mismatches)}")
    print(f"  Unknown:   {len(unknown)} (missing NWS data)")

    if mismatches:
        print(f"\n  {'!' * 60}")
        print("  MISMATCHES DETECTED — investigate settlement source changes")
        print(f"  {'!' * 60}")
        print(
            f"\n  {'Ticker':<32} {'City':<5} {'Date':<12} {'Type':<8} "
            f"{'NWS':>5} {'Expected':<8} {'Kalshi':<8}"
        )
        print(f"  {'-' * 86}")
        for r in mismatches:
            strike_desc = ""
            if r.strike_type == "greater" and r.floor_strike is not None:
                strike_desc = f">{r.floor_strike:.0f}"
            elif r.strike_type == "less" and r.cap_strike is not None:
                strike_desc = f"<{r.cap_strike:.0f}"
            elif r.strike_type == "between":
                strike_desc = f"{r.floor_strike:.0f}-{r.cap_strike:.0f}"

            print(
                f"  {r.ticker:<32} {r.city:<5} {r.event_date:<12} "
                f"{strike_desc:<8} {r.nws_temp or '?':>5} "
                f"{r.expected_result or '?':<8} {r.kalshi_result or '?':<8}"
            )

    if matched:
        print(f"\n  All {len(matched)} matched settlements:")
        # Group by city
        by_city: dict[str, int] = {}
        for r in matched:
            by_city[r.city] = by_city.get(r.city, 0) + 1
        for city, count in sorted(by_city.items()):
            print(f"    {city}: {count} markets verified")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    data_dir: Path,
    target_date: str | None,
    live: bool,
) -> None:
    """Run settlement reconciliation."""
    from python.strategies.weather_edge import WEATHER_SERIES

    # Load NWS settlements from snapshots
    nws_settlements = load_nws_settlements(data_dir)

    if not nws_settlements:
        print(f"No NWS settlement data found in {data_dir}")
        return

    # Get settled markets (from Kalshi API or from stored snapshot data)
    if live:
        series_list = list(WEATHER_SERIES.keys())
        settled_markets = asyncio.run(fetch_settled_markets(series_list))
    else:
        # Load from snapshot market files (markets collected as "open" that
        # may have since settled — we need the result field, so this only
        # works if we re-fetch or have backtest data with results)
        print("  Note: offline mode requires settled market data with 'result' field.")
        print("  Use --live to fetch current settlements from Kalshi API.")
        print("  Checking snapshot data for any markets with result field...")

        settled_markets = []
        for subdir in sorted(data_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            # Collect markets.json from flat layout and nested run-id subdirs
            markets_files: list[Path] = []
            flat_path = subdir / "markets.json"
            if flat_path.exists():
                markets_files.append(flat_path)
            for run_dir in sorted(subdir.iterdir()):
                if run_dir.is_dir() and not run_dir.name.startswith("."):
                    nested_path = run_dir / "markets.json"
                    if nested_path.exists():
                        markets_files.append(nested_path)
            for markets_path in markets_files:
                try:
                    with open(markets_path) as f:
                        markets = json.load(f)
                    for m in markets:
                        if m.get("result") in ("yes", "no"):
                            settled_markets.append(m)
                except (json.JSONDecodeError, OSError):
                    continue

    if not settled_markets:
        print("  No settled markets found to reconcile.")
        return

    # Add event_date to markets that don't have it
    import re

    date_re = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})", re.IGNORECASE)
    month_map = {
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
    for m in settled_markets:
        if "event_date" not in m or m["event_date"] is None:
            event_ticker = m.get("event_ticker", m.get("ticker", ""))
            match = date_re.search(event_ticker)
            if match:
                y, mon, d = int(match.group(1)), match.group(2).upper(), int(match.group(3))
                month_num = month_map.get(mon)
                if month_num:
                    try:
                        m["event_date"] = date(2000 + y, month_num, d).isoformat()
                    except ValueError:
                        pass

    # Filter by date if specified
    if target_date:
        settled_markets = [m for m in settled_markets if m.get("event_date") == target_date]

    # Reconcile
    results: list[ReconciliationResult] = []
    for market in settled_markets:
        r = reconcile_market(market, nws_settlements)
        if r is not None:
            results.append(r)

    print_reconciliation(results)

    # Exit with error code if mismatches found (useful in CI)
    mismatches = [r for r in results if r.match is False]
    if mismatches:
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile NWS settlement temps against Kalshi's actual settlement results.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/snapshots",
        help="Snapshot directory with settlements.json files (default: data/snapshots)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Only reconcile markets for a specific event date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch settled markets from Kalshi API (requires credentials)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args()
    main(
        data_dir=Path(args.data_dir),
        target_date=args.date,
        live=args.live,
    )
