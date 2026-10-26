"""One-shot LIVE capture of KXRAIN API payloads into test fixtures.

Read-only: issues GET requests and writes only under tests/fixtures/kxrain/,
paced to stay under Kalshi's rate limit."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kalshi_client import KalshiClient  # noqa: E402
from python.data.kxrain_markets import (  # noqa: E402
    KXRAIN_CITY_STATION,
    KXRAIN_SERIES_TICKER,
    parse_kxrain_ticker,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("capture_kxrain_fixture")

DEFAULT_OUT = _PROJECT_ROOT / "tests" / "fixtures" / "kxrain"


def _get_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ORACLE_KALSHI_API_KEY", "")
    key_path = os.environ.get("ORACLE_KALSHI_PRIVATE_KEY_PATH", "")
    if not api_key or not key_path:
        try:
            from python.settings import Settings

            settings = Settings.load()
            api_key = api_key or settings.kalshi_api_key
            key_path = key_path or settings.kalshi_private_key_path
        except Exception:
            logger.debug("Settings.load() failed — relying on env vars only")
    if not api_key or not key_path:
        raise RuntimeError(
            "No Kalshi credentials. Set ORACLE_KALSHI_API_KEY and "
            "ORACLE_KALSHI_PRIVATE_KEY_PATH, or populate .env."
        )
    return api_key, key_path


def _print_epoch(trade: dict[str, Any]) -> float:
    return datetime.fromisoformat(trade["created_time"].replace("Z", "+00:00")).timestamp()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s (%d bytes)", path, path.stat().st_size)


async def _get_trades(
    client: KalshiClient,
    *,
    ticker: str,
    limit: int,
    cursor: str | None = None,
    min_ts: int | None = None,
    max_ts: int | None = None,
) -> dict[str, Any]:
    # Routed through client._request (despite no-auth) to share the RateLimiter/429 gate.
    params: dict[str, Any] = {"ticker": ticker, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    if min_ts is not None:
        params["min_ts"] = min_ts
    if max_ts is not None:
        params["max_ts"] = max_ts
    return await client._request("GET", "/markets/trades", params=params)


async def capture(out_dir: Path, pacing: float) -> dict[str, Any]:
    api_key, key_path = _get_credentials()
    manifest: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "series_ticker": KXRAIN_SERIES_TICKER,
        "requests_issued": 0,
    }

    async with KalshiClient(api_key=api_key, private_key_path=key_path) as client:

        async def paced() -> None:
            manifest["requests_issued"] += 1
            await asyncio.sleep(pacing)

        # Open markets page
        markets_page = await client.get_markets(
            series_ticker=KXRAIN_SERIES_TICKER, status="open", limit=100
        )
        await paced()
        markets = markets_page.get("markets", [])
        cursor = markets_page.get("cursor") or ""
        _write(out_dir / "live_markets.json", markets_page)

        manifest["markets"] = {
            "n_markets": len(markets),
            "cursor_after_first_page": cursor,
            "single_page": not cursor,
            "event_tickers": sorted({m.get("event_ticker", "") for m in markets}),
            "city_suffixes": sorted({m["ticker"].rsplit("-", 1)[-1] for m in markets}),
            "has_city_key": any("city" in m for m in markets),
            "yes_bid_dollars_type": type(markets[0].get("yes_bid_dollars")).__name__
            if markets
            else None,
            "floor_strike_type": type(markets[0].get("floor_strike")).__name__ if markets else None,
        }

        # Orderbook for the most liquid in-band market
        def _bid(m: dict[str, Any]) -> float:
            try:
                return float(m.get("yes_bid_dollars") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        in_band = [m for m in markets if 0.10 <= _bid(m) < 0.90]
        pool = in_band or markets
        pool = sorted(pool, key=lambda m: float(m.get("volume_fp") or 0.0), reverse=True)
        book_market = pool[0]
        orderbook = await client.get_orderbook(book_market["ticker"], depth=10)
        await paced()
        _write(out_dir / "live_orderbook.json", orderbook)
        manifest["orderbook"] = {
            "ticker": book_market["ticker"],
            "yes_bid_dollars": book_market.get("yes_bid_dollars"),
            "top_level_keys": sorted(orderbook.keys()),
        }

        # Trades page: taker_side vocabulary and print order
        trades_payload: dict[str, Any] | None = None
        trades_ticker: str | None = None
        for candidate in pool[:6]:
            payload = await _get_trades(client, ticker=candidate["ticker"], limit=100)
            await paced()
            if payload.get("trades"):
                trades_payload = payload
                trades_ticker = candidate["ticker"]
                break
        if trades_payload is None or trades_ticker is None:
            raise RuntimeError("No KXRAIN market with recent prints; retry during market hours")
        _write(out_dir / "live_trades.json", trades_payload)

        prints = trades_payload.get("trades", [])
        times = [t.get("created_time", "") for t in prints]
        manifest["trades"] = {
            "ticker": trades_ticker,
            "n_trades": len(prints),
            "cursor": trades_payload.get("cursor") or "",
            "taker_side_counts": dict(Counter(t.get("taker_side") for t in prints).most_common()),
            "field_names": sorted(prints[0].keys()),
            "first_created_time": times[0] if times else None,
            "last_created_time": times[-1] if times else None,
            "newest_first": bool(times) and times == sorted(times, reverse=True),
            "oldest_first": bool(times) and times == sorted(times),
        }

        # A tiny page shows which end of the window survives truncation.
        small = await _get_trades(client, ticker=trades_ticker, limit=2)
        await paced()
        small_trades = small.get("trades", [])
        manifest["trades"]["truncation_probe"] = {
            "limit": 2,
            "n_returned": len(small_trades),
            "cursor_present": bool(small.get("cursor")),
            "created_times": [t.get("created_time") for t in small_trades],
            "matches_head_of_full_page": [t.get("trade_id") for t in small_trades]
            == [t.get("trade_id") for t in prints[:2]],
        }

        # Anchor the probe on a held print: on a liquid market a "last 30m" filter proves nothing.
        newest = _print_epoch(prints[0])
        oldest = _print_epoch(prints[-1])
        min_ts = int(newest) - 1
        max_ts = int(newest) + 1
        probe = await _get_trades(
            client, ticker=trades_ticker, limit=100, min_ts=min_ts, max_ts=max_ts
        )
        await paced()
        probe_trades = probe.get("trades", [])
        manifest["trades"]["ts_filter_probe"] = {
            "min_ts": min_ts,
            "max_ts": max_ts,
            "window_seconds": 2,
            "n_returned": len(probe_trades),
            "n_unfiltered": len(prints),
            "unfiltered_spans_seconds": round(newest - oldest, 1),
            "filter_applied": len(probe_trades) < len(prints),
            "all_within_window": all(
                min_ts <= _print_epoch(t) <= max_ts for t in probe_trades
            ),
        }

        # A settled market: join key and expiration_value shape
        settled_page = await client.get_markets(
            series_ticker=KXRAIN_SERIES_TICKER, status="settled", limit=100
        )
        await paced()
        settled = settled_page.get("markets", [])
        if not settled:
            raise RuntimeError("No settled KXRAIN markets returned")
        # Prefer YES: expiration_value is numeric there, not "" / "T".
        yes_rows = [m for m in settled if str(m.get("result", "")).lower() == "yes"]
        settled_market = (yes_rows or settled)[0]
        _write(out_dir / "live_settled_market.json", settled_market)

        exp_values = Counter(repr(m.get("expiration_value")) for m in settled)
        manifest["settled"] = {
            "n_settled_returned": len(settled),
            "cursor": settled_page.get("cursor") or "",
            "chosen_ticker": settled_market["ticker"],
            "result": settled_market.get("result"),
            "expiration_value": settled_market.get("expiration_value"),
            "expiration_value_type": type(settled_market.get("expiration_value")).__name__,
            "last_price_dollars": settled_market.get("last_price_dollars"),
            "close_time": settled_market.get("close_time"),
            "rules_primary": settled_market.get("rules_primary"),
            "expiration_value_vocabulary": dict(exp_values.most_common(12)),
            "result_vocabulary": dict(Counter(m.get("result") for m in settled).most_common()),
        }

    # Offline checks over what we captured
    observed_cities = set(manifest["markets"]["city_suffixes"])
    manifest["station_map_check"] = {
        "observed_city_count": len(observed_cities),
        "map_city_count": len(KXRAIN_CITY_STATION),
        "in_api_not_in_map": sorted(observed_cities - set(KXRAIN_CITY_STATION)),
        "in_map_not_in_api": sorted(set(KXRAIN_CITY_STATION) - observed_cities),
        "matches": observed_cities == set(KXRAIN_CITY_STATION),
    }

    # Unparseable tickers are reported separately: a retired suffix differs from a wrong date.
    mismatches: list[str] = []
    unparseable: list[str] = []
    for m in markets + settled:
        key = parse_kxrain_ticker(m["ticker"])
        if key is None:
            unparseable.append(m["ticker"])
            continue
        stated = key.settlement_date.strftime("%b %-d, %Y")
        if stated not in (m.get("rules_primary") or ""):
            mismatches.append(f"{m['ticker']}: rules do not state {stated}")
    manifest["ticker_date_check"] = {
        "n_checked": len(markets) + len(settled),
        "n_parsed": len(markets) + len(settled) - len(unparseable),
        "mismatches": mismatches[:10],
        "unparseable_tickers": unparseable[:10],
        "all_parsed_dates_match": not mismatches,
    }

    _write(out_dir / "capture_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="fixture output directory")
    parser.add_argument("--pacing", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    manifest = asyncio.run(capture(args.out, args.pacing))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
