"""Pure row builders for the KXRAIN shadow-observer JSONL streams.

Hermetic, no I/O, no clock. No builder raises — a malformed payload yields
nulls plus a parse-failure count. Every row is length-checked against `PIPE_BUF` (see `encode_row`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from python.data.kxrain_decision import SKIP_REASONS, DecisionOutcome
from python.data.kxrain_markets import (
    KxRainMarketKey,
    lst_local_time,
    parse_kxrain_ticker,
    station_from_rules,
    station_mismatch,
)
from python.data.kxrain_trades import TradeWindow, to_dollar_price, to_fp_float

logger = logging.getLogger(__name__)

__all__ = [
    "CAPTURE_OUTCOMES",
    "FRAME_VERSION",
    "MAX_UNPARSEABLE_TICKERS",
    "PIPE_BUF",
    "PRICE_DECIMALS",
    "SCHEMA_VERSION",
    "TICK_OUTCOMES",
    "build_book_row",
    "build_decision_row",
    "build_oversize_stub_row",
    "build_settlement_row",
    "build_tick_row",
    "build_trades_row",
    "encode_row",
    "quantise_price",
]

SCHEMA_VERSION = 1
FRAME_VERSION = 2

# O_APPEND is atomic w.r.t. file size only up to PIPE_BUF (4096 on Linux);
# past it a concurrent writer can interleave and tear the line.
PIPE_BUF = 4096

_NEWLINE_BYTES = 1  # counts against PIPE_BUF too — the writer appends it in the same syscall

PRICE_DECIMALS = 4

# skipped_market_absent covers the census row, which has no fetch outcome.
CAPTURE_OUTCOMES: frozenset[str] = frozenset(
    {
        "ok",
        "error",
        "skipped_band",
        "skipped_not_current_day",
        "skipped_market_absent",
        "budget_abort",
        "wall_clock_abort",
        "backoff_abort",
    }
)

# Wider than ABORT_REASONS (a mid-pass abort on one market): a tick skipped
# before any capture began is skipped_backoff, never backoff_abort.
TICK_OUTCOMES: frozenset[str] = frozenset(
    {
        "ok",
        "skipped_no_city_in_window",
        "skipped_backoff",
        "markets_fetch_error",
        "budget_abort",
        "wall_clock_abort",
        "backoff_abort",
        "disk_disabled",
        "error",
    }
)

# The ticks row's only unbounded field; n_unparseable still carries the true count.
MAX_UNPARSEABLE_TICKERS = 5

_STUB_KEYS = ("schema_version", "decision_id", "scan_id", "ts", "ticker")

_SETTLED_RESULTS = frozenset({"yes", "no"})

_TRACE_TOKENS = frozenset({"", "t"})  # "T" and "" both mean no measurable precipitation


# Serialisation guard


def quantise_price(value: float | None) -> float | None:
    """Round a derived price to `PRICE_DECIMALS`. `None` for non-finite."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    return round(as_float, PRICE_DECIMALS)


def encode_row(row: Mapping[str, Any]) -> bytes | None:
    """Serialise a row exactly as `ForecastsRawWriter` will, or `None` if it (plus the
    newline it appends) would exceed `PIPE_BUF`, or is not serialisable."""
    try:
        encoded = json.dumps(row, default=str).encode("utf-8")
    except (TypeError, ValueError):
        logger.error("kxrain: row is not serialisable; dropping", exc_info=True)
        return None
    if len(encoded) + _NEWLINE_BYTES > PIPE_BUF:
        return None
    return encoded


def build_oversize_stub_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Join-keys-only replacement for a row that exceeded `PIPE_BUF`."""
    stub: dict[str, Any] = {key: row[key] for key in _STUB_KEYS if key in row}
    stub["oversize"] = True
    try:
        stub["oversize_original_bytes"] = len(json.dumps(row, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        stub["oversize_original_bytes"] = -1
    return stub


# Decision row


def _round_sum(values: Iterable[float]) -> float:
    # sum(()) returns int 0, not 0.0, which would flip a typed loader's dtype
    # on empty-book rows.
    return round(float(sum(values)), 4)


def _sha16(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _safe_local_time_lst(city: str, ts: datetime) -> str | None:
    # lst_local_time raises on a naive datetime or unknown city; never raise here.
    try:
        return lst_local_time(city, ts).strftime("%H:%M")
    except (ValueError, KeyError):
        logger.error("kxrain: cannot derive local_time_lst for %s at %r", city, ts)
        return None


def build_decision_row(
    *,
    key: KxRainMarketKey,
    market: Mapping[str, Any] | None,
    outcome: DecisionOutcome | None,
    scan_id: str,
    ts: datetime,
    in_window: bool,
    already_entered: bool,
    virtual_entry: bool,
    pass_index: int,
    book_capture: str,
    trades_capture: str,
    follow_capture: bool,
    observer_version: str,
    max_entries_per_day: int | None,
    assumed_max_walk_cents: int,
    trade_window: TradeWindow | None = None,
    window_end_ts: int | None = None,
    last_seen_yes_bid: float | None = None,
    last_seen_ts: str | None = None,
) -> dict[str, Any]:
    """One row per (tick x KXRAIN market). Always written; never raises.

    `market=None` builds the market-absent census row (`skip_reason="market_absent"`,
    `outcome` ignored). `pass_index=-1` means never captured this pass.
    `follow_capture` rows deliberately ignore the band/window predicates.
    """
    _check_capture(book_capture)
    _check_capture(trades_capture)

    absent = market is None
    payload: Mapping[str, Any] = market or {}

    yes_bid = to_dollar_price(payload.get("yes_bid_dollars"))
    yes_ask = to_dollar_price(payload.get("yes_ask_dollars"))
    spread = (
        quantise_price(yes_ask - yes_bid) if yes_bid is not None and yes_ask is not None else None
    )

    rules_primary = str(payload.get("rules_primary") or "")
    # station_mismatch reports both a genuine disagreement and unreadable rules
    # text as rules_station_ok=false — a blind tripwire is as reportable as a tripped one.
    rules_station = station_from_rules(rules_primary) if rules_primary else None
    rules_station_ok = (not absent) and station_mismatch(key, rules_primary) is None

    if outcome is None and not absent:
        # A present market with no outcome is a caller bug, not a census row.
        logger.error(
            "kxrain: %s is present but has no decision outcome; emitting a row with "
            "no skip_reason rather than a false market_absent census label",
            key.ticker,
        )

    if absent or outcome is None:
        eligible: bool | None = False
        skip_reason: str | None = "market_absent" if absent else None
        in_price_band = False
        no_ask: float | None = None
        no_trade_recent: bool | None = None
        yes_bid_ge_040 = False
        assumed_entry_price: float | None = None
    else:
        eligible = outcome.eligible
        skip_reason = outcome.skip_reason
        in_price_band = outcome.in_price_band
        no_ask = quantise_price(outcome.no_ask)
        no_trade_recent = outcome.no_trade_recent
        yes_bid_ge_040 = outcome.yes_bid_ge_040
        assumed_entry_price = quantise_price(outcome.assumed_entry_price)

    if skip_reason is not None and skip_reason not in SKIP_REASONS:
        logger.warning("kxrain: skip_reason %r is outside the vocabulary", skip_reason)

    oldest_print_ts = trade_window.oldest_print_ts if trade_window is not None else None
    effective_lookback = _effective_lookback_seconds(trade_window, window_end_ts)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frame_version": FRAME_VERSION,
        "decision_id": f"{scan_id}:{key.ticker}",
        "scan_id": scan_id,
        "ts": ts.isoformat(),
        "ticker": key.ticker,
        "city": key.city,
        "station": key.station,
        "settlement_date": key.settlement_date.isoformat(),
        "lst_offset_hours": key.lst_offset_hours,
        "local_time_lst": _safe_local_time_lst(key.city, ts),
        "in_window": bool(in_window),
        "yes_bid": quantise_price(yes_bid),
        "yes_ask": quantise_price(yes_ask),
        "no_ask": no_ask,
        # Turns no_ask = 1 - yes_bid from an assumption into a monitored invariant.
        "no_ask_api": quantise_price(to_dollar_price(payload.get("no_ask_dollars"))),
        "spread": spread,
        # Top-of-book size on every tick at zero extra HTTP cost.
        "yes_bid_size_fp": to_fp_float(payload.get("yes_bid_size_fp")),
        "yes_ask_size_fp": to_fp_float(payload.get("yes_ask_size_fp")),
        "volume_fp": to_fp_float(payload.get("volume_fp")),
        "open_interest": to_fp_float(payload.get("open_interest_fp")),
        "last_price_dollars": to_dollar_price(payload.get("last_price_dollars")),
        "close_time": payload.get("close_time") if not absent else None,
        "status": str(payload["status"]) if payload.get("status") is not None else None,
        "can_close_early": (
            bool(payload["can_close_early"]) if payload.get("can_close_early") is not None else None
        ),
        # Non-null only on the market-absent census row; always present, never omitted.
        "last_seen_yes_bid": quantise_price(last_seen_yes_bid),
        "last_seen_ts": last_seen_ts,
        "in_price_band": in_price_band,
        "n_no_trades_30m": trade_window.n_trades_no_side if trade_window is not None else None,
        "no_volume_30m": trade_window.no_volume if trade_window is not None else None,
        "last_no_trade_ts": trade_window.last_no_trade_ts if trade_window is not None else None,
        "last_no_trade_price": (
            quantise_price(trade_window.last_no_trade_price) if trade_window is not None else None
        ),
        "no_trade_recent": no_trade_recent,
        "effective_lookback_seconds": effective_lookback,
        "window_oldest_print_ts": oldest_print_ts,
        "already_entered_market_day": bool(already_entered),
        "max_entries_per_day": max_entries_per_day,
        "yes_bid_ge_040": yes_bid_ge_040,
        "eligible": eligible,
        "skip_reason": skip_reason,
        "virtual_entry": bool(virtual_entry),
        "assumed_entry_price": assumed_entry_price,
        "assumed_max_walk_cents": int(assumed_max_walk_cents),
        "pass_index": int(pass_index),
        "book_capture": book_capture,
        "trades_capture": trades_capture,
        "follow_capture": bool(follow_capture),
        "rules_station_ok": rules_station_ok,
        "rules_station": rules_station,
        "observer_version": observer_version,
    }

    return row


# ---------------------------------------------------------------------------
# Book row
# ---------------------------------------------------------------------------


def build_book_row(
    *,
    decision_id: str,
    ts: datetime,
    ticker: str,
    orderbook: Mapping[str, Any] | None,
    markets_yes_bid: float | None,
    depth_levels: int = 10,
) -> dict[str, Any]:
    """Depth snapshot for one decision. Never raises; an empty/unreadable book still produces a row.

    Kalshi's ladders are ascending by price, so the touch is `levels[-1]` —
    this builder sorts explicitly and keeps the touch LAST in all three
    ladders. `no_ask_levels` is derived by mirroring the YES-bid ladder
    (`p -> 1-p`), since that's the side a buy-NO rule would lift.
    """
    envelope: Mapping[str, Any] = {}
    if orderbook is not None:
        raw_envelope = orderbook.get("orderbook_fp")
        if isinstance(raw_envelope, Mapping):
            envelope = raw_envelope

    yes_levels, yes_failures = _parse_ladder(envelope.get("yes_dollars"))
    no_levels, no_failures = _parse_ladder(envelope.get("no_dollars"))

    depth = max(1, int(depth_levels))
    yes_captured = yes_levels[-depth:]
    no_captured = no_levels[-depth:]

    # Mirror in place so position, not price order, keeps identifying the touch.
    no_ask_captured: list[list[float]] = [
        [round(1.0 - price, PRICE_DECIMALS), size] for price, size in yes_captured
    ]

    within_1c_contracts = 0.0
    within_1c_dollars = 0.0
    if no_ask_captured:
        # Compare in whole cents: the ladder is cent-tick (``linear_cent``) and a
        # float epsilon here would silently drop or admit a whole level.
        touch_cents = round(no_ask_captured[-1][0] * 100)
        for price, size in no_ask_captured:
            if round(price * 100) <= touch_cents + 1:
                within_1c_contracts += size
                within_1c_dollars += price * size

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_version": FRAME_VERSION,
        "decision_id": decision_id,
        "ts": ts.isoformat(),
        "ticker": ticker,
        "yes_bid_levels": [[p, s] for p, s in yes_captured],
        "no_bid_levels": [[p, s] for p, s in no_captured],
        "no_ask_levels": no_ask_captured,
        "no_ask_depth_within_1c_contracts": within_1c_contracts,
        "no_ask_depth_within_1c_dollars": within_1c_dollars,
        # Staleness canary vs. book_top_yes_bid: two separate calls, may legitimately diverge.
        "markets_yes_bid": quantise_price(markets_yes_bid),
        "book_top_yes_bid": yes_captured[-1][0] if yes_captured else None,
        "total_yes_depth_dollars": _round_sum(p * s for p, s in yes_captured),
        "total_no_depth_dollars": _round_sum(p * s for p, s in no_captured),
        # Levels the API returned, before the depth cap.
        "n_yes_levels": len(yes_levels),
        "n_no_levels": len(no_levels),
        "raw_level_parse_failures": yes_failures + no_failures,
    }


# Trades row


def build_trades_row(
    *,
    decision_id: str,
    ts: datetime,
    ticker: str,
    window: TradeWindow,
    window_start_ts: int,
    window_end_ts: int,
    n_pages_fetched: int,
) -> dict[str, Any]:
    """One row per decision that fetched trades — not one row per print.

    `window_end_ts` and `n_pages_fetched` are required, not defaulted: a
    default would silently null `effective_lookback_seconds` or mis-assert
    the page count rather than fail loudly.
    """
    prints = [
        {
            "ts": p.ts,
            "taker_side": p.taker_side,
            "yes_price": quantise_price(p.yes_price),
            "no_price": quantise_price(p.no_price),
            "count": p.count,
        }
        for p in window.prints
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_version": FRAME_VERSION,
        "decision_id": decision_id,
        "ts": ts.isoformat(),
        "ticker": ticker,
        "window_start_ts": int(window_start_ts),
        "window_end_ts": int(window_end_ts),
        "n_trades_total": window.n_trades_total,
        "n_trades_no_side": window.n_trades_no_side,
        "no_volume_30m": window.no_volume,
        "yes_volume_30m": window.yes_volume,
        "vwap_no_30m": quantise_price(window.vwap_no),
        "prints": prints,
        "prints_truncated": window.prints_truncated,
        "page_truncated": window.page_truncated,
        "window_oldest_print_ts": window.oldest_print_ts,
        "effective_lookback_seconds": _effective_lookback_seconds(window, window_end_ts),
        # 1, or 2 when the bounded extra page was taken (non-empty cursor AND
        # zero NO prints — the only case truncation can flip the eligibility
        # answer).
        "n_pages_fetched": int(n_pages_fetched),
        # Same drift canary as raw_level_parse_failures on the book row.
        "print_parse_failures": window.parse_failures,
    }


# Tick row


def build_tick_row(
    *,
    scan_id: str,
    ts: datetime,
    process_start_ts: datetime,
    tick_outcome: str,
    n_cities_in_window: int,
    n_markets_returned: int,
    n_parsed: int,
    n_unparseable: int,
    unparseable_tickers: Sequence[str],
    markets_truncated: bool,
    markets_cycle_aborted: bool,
    markets_pages_consumed: int,
    census_suppressed: bool,
    requests_used: int,
    max_requests_per_tick: int,
    tick_seed: int,
    elapsed_seconds: float,
    backoff_remaining_at_start: float,
    backoff_accessor_present: bool,
    n_decision_rows: int,
    rows_lost: int,
    oversize_rows_dropped: int,
    observer_version: str,
    tick_interval_seconds: float,
) -> dict[str, Any]:
    """One row per tick ATTEMPT, including a tick that wrote nothing. Never raises.

    `process_start_ts` is constant across a process lifetime and changes on
    restart, making a restart gap detectable. `unparseable_tickers` is capped
    at `MAX_UNPARSEABLE_TICKERS`; `n_unparseable` still carries the true count.
    """
    _check_tick_outcome(tick_outcome)

    tickers = [str(t) for t in unparseable_tickers]
    capped_tickers = tickers[:MAX_UNPARSEABLE_TICKERS]

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_version": FRAME_VERSION,
        "scan_id": scan_id,
        "ts": ts.isoformat(),
        "process_start_ts": process_start_ts.isoformat(),
        "tick_outcome": tick_outcome,
        "n_cities_in_window": int(n_cities_in_window),
        "n_markets_returned": int(n_markets_returned),
        "n_parsed": int(n_parsed),
        "n_unparseable": int(n_unparseable),
        "unparseable_tickers": capped_tickers,
        "unparseable_tickers_truncated": len(tickers) > MAX_UNPARSEABLE_TICKERS,
        "markets_truncated": bool(markets_truncated),
        # census_suppressed is the on-disk record that rows were withheld.
        "markets_cycle_aborted": bool(markets_cycle_aborted),
        "markets_pages_consumed": int(markets_pages_consumed),
        "census_suppressed": bool(census_suppressed),
        "requests_used": int(requests_used),
        "max_requests_per_tick": int(max_requests_per_tick),
        "tick_seed": int(tick_seed),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "backoff_remaining_at_start": round(float(backoff_remaining_at_start), 3),
        # 0.0 is otherwise ambiguous between a clear gate and a missing one.
        "backoff_accessor_present": bool(backoff_accessor_present),
        "n_decision_rows": int(n_decision_rows),
        "rows_lost": int(rows_lost),  # process-cumulative; otherwise only in an ERROR log
        "oversize_rows_dropped": int(oversize_rows_dropped),
        "observer_version": observer_version,
        # Cadence can change via config alone, without a version bump.
        "tick_interval_seconds": float(tick_interval_seconds),
    }


# Settlement row


def build_settlement_row(
    market: Mapping[str, Any],
    *,
    ts: datetime,
) -> dict[str, Any] | None:
    """One row per settled market-day, or `None` if it must not be written.

    Refuses unless `result` is exactly `"yes"` or `"no"`. An open market
    carries `result: ""` (an empty string, not null) — parsing it anyway would
    fabricate a passing NO settlement in a corpus that only buys NO. Also
    refuses a ticker the station map can't resolve; callers must count both
    refusal kinds rather than skip them silently.
    """
    raw_result = market.get("result")
    result = raw_result.strip().lower() if isinstance(raw_result, str) else ""
    if result not in _SETTLED_RESULTS:
        return None

    key = parse_kxrain_ticker(str(market.get("ticker") or ""))
    if key is None:
        return None

    raw_value = market.get("expiration_value")
    expiration_value_raw = str(raw_value) if raw_value is not None else None
    expiration_value_in, trace = _parse_expiration_value(raw_value)
    derived_yes = expiration_value_in > 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_version": FRAME_VERSION,
        "ts": ts.isoformat(),
        "ticker": key.ticker,
        "city": key.city,
        "station": key.station,
        "settlement_date": key.settlement_date.isoformat(),
        # Stored verbatim: the field can carry "T" or "" as well as a number.
        "expiration_value_raw": expiration_value_raw,
        "expiration_value_in": expiration_value_in,
        "expiration_value_trace": trace,
        "result": result,
        "derived_yes": derived_yes,
        # False is a CRITICAL gate line. Recorded, never suppressed.
        "rule_consistent": derived_yes == (result == "yes"),
        # Audit only, not the outcome.
        "last_price_dollars": to_dollar_price(market.get("last_price_dollars")),
        "close_time": market.get("close_time"),
        "settlement_ts": market.get("settlement_ts"),  # API spelling; not "settled_ts"
        "status": str(market["status"]) if market.get("status") is not None else None,
        # In-process hash resets on restart; a durable record of the settlement source's rendering.
        "rules_secondary_sha256": _sha16(market.get("rules_secondary")),
        "volume_fp": to_fp_float(market.get("volume_fp")),
        "open_interest": to_fp_float(market.get("open_interest_fp")),
    }


# Internals


def _check_capture(value: str) -> None:
    if value not in CAPTURE_OUTCOMES:
        logger.warning("kxrain: capture outcome %r is outside the vocabulary", value)


def _check_tick_outcome(value: str) -> None:
    if value not in TICK_OUTCOMES:
        logger.warning("kxrain: tick_outcome %r is outside the vocabulary", value)


def _effective_lookback_seconds(
    window: TradeWindow | None, window_end_ts: int | None
) -> float | None:
    # None when there's no page or print on it — 0.0 or 1800.0 would both be
    # claims about coverage the data doesn't support.
    if window is None or window.oldest_print_epoch is None or window_end_ts is None:
        return None
    return round(float(window_end_ts) - window.oldest_print_epoch, 3)


def _parse_ladder(raw: Any) -> tuple[list[tuple[float, float]], int]:
    # Parses to ascending (price, size). Sizes stay float — live levels run
    # fractional (e.g. "31.14"), and truncating would under-count depth.
    levels: list[tuple[float, float]] = []
    failures = 0
    if not isinstance(raw, list):
        return levels, failures
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            failures += 1
            continue
        price = to_dollar_price(item[0])
        size = to_fp_float(item[1])
        if price is None or size is None:
            failures += 1
            continue
        quantised = quantise_price(price)
        if quantised is None:
            failures += 1
            continue
        levels.append((quantised, size))
    levels.sort(key=lambda level: level[0])
    return levels, failures


def _parse_expiration_value(raw: Any) -> tuple[float, bool]:
    # Open markets carry "" here and must parse to 0.0/trace rather than throw.
    if raw is None:
        return 0.0, True
    text = str(raw).strip()
    if text.lower() in _TRACE_TOKENS:
        return 0.0, True
    try:
        return float(text), False
    except ValueError:
        return 0.0, True
