"""Background observer that logs the KXRAIN forward corpus. Places no orders.

Shares disk, the shared ``RateLimiter`` and the 429 backoff gate with the live
bot, so it self-limits and always yields to trading.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import time as time_module
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from python.data.kalshi_client import paginate_detailed
from python.data.kxrain_budget import DEFAULT_OVERDRAFT_REQUESTS, RequestBudget
from python.data.kxrain_decision import DecisionOutcome, evaluate_decision
from python.data.kxrain_ledger import LEDGER_FILENAME, CorpusKey, KxRainLedger
from python.data.kxrain_markets import (
    KXRAIN_CITY_STD_OFFSET_HOURS,
    KXRAIN_SERIES_TICKER,
    KxRainMarketKey,
    in_entry_window,
    lst_day,
    parse_kxrain_ticker,
)
from python.data.kxrain_rows import (
    build_book_row,
    build_decision_row,
    build_oversize_stub_row,
    build_tick_row,
    build_trades_row,
    encode_row,
)
from python.data.kxrain_trades import TradeWindow, summarise_trade_window, to_dollar_price
from python.notifications.base import AlertLevel, Notifier
from python.utils.jsonl import ForecastsRawWriter

logger = logging.getLogger(__name__)

__all__ = [
    "BACKOFF_SKIP_THRESHOLD_SECONDS",
    "CONSECUTIVE_FAILURE_ALERT_THRESHOLD",
    "DEFAULT_MIN_FREE_DISK_BYTES",
    "KXRAIN_BOOK_FILENAME",
    "KXRAIN_DECISIONS_FILENAME",
    "KXRAIN_TICKS_FILENAME",
    "KXRAIN_TRADES_FILENAME",
    "CENSUS_ABSENT_TICKS_REQUIRED",
    "LIVENESS_ALERT_AFTER_SECONDS",
    "MARKETS_FETCH_MAX_PAGES",
    "OVERDRAFT_REQUESTS",
    "WRITE_FLUSH_EVERY_ROWS",
    "KxRainObserver",
    "KxRainObserverConfig",
    "MarketDataClient",
]

KXRAIN_DECISIONS_FILENAME = "kxrain_decisions.jsonl"
KXRAIN_BOOK_FILENAME = "kxrain_book.jsonl"
KXRAIN_TRADES_FILENAME = "kxrain_trades.jsonl"
KXRAIN_TICKS_FILENAME = "kxrain_ticks.jsonl"

BUILD_SHA_FILENAME = ".build_sha"

# Bounds a non-terminating cursor: the live universe fits on one page anyway.
MARKETS_FETCH_MAX_PAGES = 3

# Pinned rather than inherited from ``get_markets``'s default, which lives
# in another module.
MARKETS_PAGE_LIMIT = 100

# A short page with an empty cursor looks like a complete one, so absence
# must survive a second complete page before it becomes a census row.
CENSUS_ABSENT_TICKS_REQUIRED = 2

# More market-days than this vanishing in one tick is a short page with a
# terminal cursor, not a real early close.
CENSUS_MAX_ABSENT_PER_TICK = 3

OVERDRAFT_REQUESTS = DEFAULT_OVERDRAFT_REQUESTS

# The shared gate models a short retry series, not the multi-minute sliding-
# window cooldown, so above this the observer yields the whole tick.
BACKOFF_SKIP_THRESHOLD_SECONDS = 30.0

# Below this floor the observer permanently self-disables rather than
# retrying, to leave headroom for risk-state persistence and trade logging.
DEFAULT_MIN_FREE_DISK_BYTES = 2 * 1024**3

# Legitimately silent for hours a day, so deliberately not in
# ``audit_tripwires.STREAM_NAMES``; this in-window-only detector replaces it.
LIVENESS_ALERT_AFTER_SECONDS = 90 * 60.0

CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 5
FAILURE_ALERT_COOLDOWN_SECONDS = 3600.0

# Tick outcomes meaning the corpus did not accrue; `run()` counts these
# rather than exceptions, since `_tick` no longer lets any escape.
FAILING_TICK_OUTCOMES: frozenset[str] = frozenset({"error", "markets_fetch_error"})

# Flushed incrementally rather than in one end-of-tick batch, so a
# ``CancelledError`` during shutdown can't discard a tick's worth of rows.
WRITE_FLUSH_EVERY_ROWS = 10

# Bounded so a stalled disk cannot eat the systemd stop timeout that live
# order cancellation needs.
ACLOSE_TIMEOUT_SECONDS = 10.0


class MarketDataClient(Protocol):
    """Three read-only GETs. No order-placement method, so a stray edit
    reaching for one fails type-checking as well as the AST test."""

    async def get_markets(
        self,
        *,
        status: str | None = ...,
        series_ticker: str | None = ...,
        event_ticker: str | None = ...,
        cursor: str | None = ...,
        limit: int = ...,
    ) -> dict[str, Any]: ...

    async def get_orderbook(self, ticker: str, depth: int = ...) -> dict[str, Any]: ...

    async def get_market_trades(
        self,
        *,
        ticker: str,
        min_ts: int | None = ...,
        max_ts: int | None = ...,
        limit: int = ...,
        cursor: str | None = ...,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class KxRainObserverConfig:
    """A frozen dataclass, not Pydantic, so it's constructible in a test
    without ``Settings.load()``."""

    log_dir: Path
    series_ticker: str = KXRAIN_SERIES_TICKER
    tick_interval_seconds: float = 600.0
    entry_window_start_lst: time = time(10, 0)
    entry_window_end_lst: time = time(18, 0)
    yes_bid_min: float = 0.10
    yes_bid_max: float = 0.90
    no_trade_lookback_minutes: int = 30
    max_entries_per_day: int | None = None
    assumed_max_walk_cents: int = 1
    depth_levels: int = 10
    # Every market still gets a decision row; ``band`` only skips the
    # book/trade reads for out-of-band markets. ``all`` is a diagnostic mode.
    depth_capture_mode: str = "band"
    max_requests_per_tick: int = 60
    request_pacing_seconds: float = 1.0
    tick_wall_clock_budget_seconds: float = 180.0
    # Book+trade capture stays pinned ON this long after a virtual entry,
    # regardless of band or the window clock.
    post_entry_follow_minutes: int = 45
    min_free_disk_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES
    log_retention_days: int = 240
    trades_page_limit: int = 100


def _as_utc(value: datetime) -> datetime:
    """Naive datetimes are treated as UTC; a mixed-awareness subtraction raises."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _free_disk_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path``. Patched in tests."""
    return shutil.disk_usage(path).free


@dataclass
class _Candidate:
    """Per-market working state for one tick."""

    key: KxRainMarketKey
    market: dict[str, Any]
    yes_bid: float | None
    is_current_day: bool
    clock_in_window: bool
    in_window: bool
    in_price_band: bool
    already_entered: bool
    follow_active: bool
    pass_index: int = -1
    book_capture: str = "skipped_not_current_day"
    trades_capture: str = "skipped_not_current_day"
    trade_window: TradeWindow | None = None
    window_start_ts: int | None = None
    window_end_ts: int | None = None
    n_pages_fetched: int = 1
    orderbook: dict[str, Any] | None = None
    outcome: DecisionOutcome | None = None
    virtual_entry: bool = False


@dataclass
class _TickCounters:
    n_markets_returned: int = 0
    n_parsed: int = 0
    n_unparseable: int = 0
    unparseable_tickers: list[str] = field(default_factory=list)
    markets_truncated: bool = False
    markets_cycle_aborted: bool = False
    markets_pages_consumed: int = 0
    census_suppressed: bool = False
    n_decision_rows: int = 0

    @property
    def markets_page_complete(self) -> bool:
        """Whether absence from this tick's ``/markets`` listing means anything."""
        return not self.markets_truncated and not self.markets_cycle_aborted


class KxRainObserver:
    """Writes the KXRAIN forward corpus. Returns nothing, signals nothing,
    orders nothing. ``run()`` exits only on ``CancelledError``."""

    def __init__(
        self,
        kalshi_client: MarketDataClient,
        config: KxRainObserverConfig,
        *,
        notifier_getter: Callable[[], Notifier | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._client = kalshi_client
        self._config = config
        # Resolved lazily, never captured: the orchestrator's notifier does
        # not exist yet at construction time.
        self._notifier_getter = notifier_getter
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._monotonic: Callable[[], float] = monotonic or time_module.monotonic
        self._rng = random.Random(seed)
        # Its own single worker, never the shared default executor —
        # saturating that pool previously stalled the process.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kxrain-observer"
        )

        log_dir = Path(config.log_dir)
        self._writers: dict[str, ForecastsRawWriter] = {
            name: ForecastsRawWriter(
                log_dir / name,
                retention_days=config.log_retention_days,
                clock=self._clock,
            )
            for name in (
                KXRAIN_DECISIONS_FILENAME,
                KXRAIN_BOOK_FILENAME,
                KXRAIN_TRADES_FILENAME,
                KXRAIN_TICKS_FILENAME,
            )
        }
        self._pending: dict[str, list[dict[str, Any]]] = {name: [] for name in self._writers}

        self._ledger = KxRainLedger.load(log_dir / LEDGER_FILENAME)
        self._observer_version = _read_observer_version(log_dir)

        # Constant across one process lifetime, so a restart gap is
        # mechanically detectable in the ticks stream.
        self._process_start_ts = _as_utc(self._clock())
        self._last_decision_write_utc: datetime | None = None
        self._liveness_alert_monotonic: float | None = None
        self._last_tick_outcome: str = "ok"

        # Corpus key -> the instant post-entry follow capture expires.
        self._follow_until: dict[CorpusKey, datetime] = {}
        # Ticker -> (yes_bid, ts) for the market-absent census.
        self._last_seen: dict[str, tuple[float | None, str, KxRainMarketKey]] = {}
        # Ticker -> consecutive complete pages it has been missing from.
        self._absent_streak: dict[str, int] = {}

        self._disabled = False
        self._disk_alert_sent = False
        self._consecutive_failed_ticks = 0
        self._last_failure_alert_monotonic: float | None = None
        self._listing_alert_monotonic: float | None = None
        self._oversize_rows_dropped = 0
        self._rows_lost = 0
        self._backoff_accessor_present = True
        self._backoff_accessor_warned = False
        self._pacing_sleeps = 0


    @property
    def disabled(self) -> bool:
        """True once the disk self-limit has permanently stopped this process."""
        return self._disabled

    @property
    def writers(self) -> dict[str, ForecastsRawWriter]:
        return self._writers

    @property
    def oversize_rows_dropped(self) -> int:
        return self._oversize_rows_dropped

    @property
    def rows_lost(self) -> int:
        """Rows that were built and handed to a writer but never reached disk."""
        return self._rows_lost

    @property
    def pacing_sleeps(self) -> int:
        return self._pacing_sleeps

    @property
    def last_decision_write_utc(self) -> datetime | None:
        return self._last_decision_write_utc

    @property
    def last_tick_outcome(self) -> str:
        return self._last_tick_outcome

    @property
    def observer_version(self) -> str:
        return self._observer_version


    async def run(self) -> None:
        """Sleep-then-tick until cancelled. Never exits on anything else."""
        logger.info(
            "KxRainObserver starting (interval=%.0fs, log_dir=%s, version=%s)",
            self._config.tick_interval_seconds,
            self._config.log_dir,
            self._observer_version,
        )
        try:
            while True:
                try:
                    await self.run_once()
                    # Keyed on the outcome, not an escaping exception, so a
                    # crash loop doesn't reset this to 0 every tick.
                    if self._last_tick_outcome in FAILING_TICK_OUTCOMES:
                        self._consecutive_failed_ticks += 1
                        await self._maybe_alert_consecutive_failures()
                    else:
                        self._consecutive_failed_ticks = 0
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("KxRainObserver tick failed; continuing")
                    self._consecutive_failed_ticks += 1
                    await self._maybe_alert_consecutive_failures()
                await asyncio.sleep(self._config.tick_interval_seconds)
        except asyncio.CancelledError:
            logger.info("KxRainObserver cancelled")
            raise

    async def run_once(self) -> int:
        """One tick. Never raises; ``CancelledError`` flushes synchronously
        (a cancelled await says nothing about a thread) and re-raises."""
        if self._disabled:
            return 0
        try:
            return await self._tick()
        except asyncio.CancelledError:
            self._flush_sync()
            raise


    async def _tick(self) -> int:
        # ONE clock read drives every window/day/timestamp decision this
        # tick, so two markets can never land in different LST days.
        now = _as_utc(self._clock())

        scan_id = uuid.uuid4().hex
        tick_seed = self._rng.getrandbits(32)
        counters = _TickCounters()
        budget = RequestBudget(
            max_requests=self._config.max_requests_per_tick,
            wall_clock_budget_seconds=self._config.tick_wall_clock_budget_seconds,
            clock=self._monotonic,
            overdraft_requests=OVERDRAFT_REQUESTS,
        )

        # Before any HTTP: must not fill the disk risk-state persistence and
        # the trade log depend on.
        if not await self._check_disk(now):
            # One row at the transition; later ticks short-circuit on
            # ``_disabled``.
            await self._write_tick_row(
                scan_id=scan_id,
                now=now,
                tick_outcome="disk_disabled",
                n_cities_in_window=self._count_cities_in_window(now),
                counters=counters,
                budget=budget,
                tick_seed=tick_seed,
                backoff_remaining=self._backoff_remaining(),
            )
            return 0

        try:
            return await self._tick_body(now, scan_id, tick_seed, counters, budget)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Silence in the stream whose purpose is explaining gaps would
            # be the worst possible gap.
            logger.exception("KxRainObserver: the tick raised; recording an error tick row")
            await self._write_tick_row(
                scan_id=scan_id,
                now=now,
                tick_outcome="error",
                n_cities_in_window=self._count_cities_in_window(now),
                counters=counters,
                budget=budget,
                tick_seed=tick_seed,
                backoff_remaining=0.0,
            )
            return 0

    async def _tick_body(
        self,
        now: datetime,
        scan_id: str,
        tick_seed: int,
        counters: _TickCounters,
        budget: RequestBudget,
    ) -> int:
        self._prune_follow_windows(now)
        n_cities_in_window = self._count_cities_in_window(now)
        await self._check_liveness(now, n_cities_in_window)

        backoff_remaining = self._backoff_remaining()

        if n_cities_in_window == 0 and not self._follow_until:
            await self._write_tick_row(
                scan_id=scan_id,
                now=now,
                tick_outcome="skipped_no_city_in_window",
                n_cities_in_window=n_cities_in_window,
                counters=counters,
                budget=budget,
                tick_seed=tick_seed,
                backoff_remaining=backoff_remaining,
            )
            return 0

        if backoff_remaining > BACKOFF_SKIP_THRESHOLD_SECONDS:
            logger.info(
                "KxRainObserver: skipping tick — shared Kalshi backoff active for %.1fs",
                backoff_remaining,
            )
            await self._write_tick_row(
                scan_id=scan_id,
                now=now,
                tick_outcome="skipped_backoff",
                n_cities_in_window=n_cities_in_window,
                counters=counters,
                budget=budget,
                tick_seed=tick_seed,
                backoff_remaining=backoff_remaining,
            )
            return 0

        try:
            raw_markets = await self._fetch_markets(budget, counters)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("KxRainObserver: the markets fetch failed", exc_info=True)
            await self._write_tick_row(
                scan_id=scan_id,
                now=now,
                tick_outcome="markets_fetch_error",
                n_cities_in_window=n_cities_in_window,
                counters=counters,
                budget=budget,
                tick_seed=tick_seed,
                backoff_remaining=backoff_remaining,
            )
            return 0

        if not counters.markets_page_complete:
            await self._alert_incomplete_listing(counters)

        candidates = self._build_candidates(raw_markets, now, counters)

        capture_pass = self._order_capture_pass(candidates, tick_seed)
        # Rows are emitted as each market's fetches complete, since this pass
        # is the slow half of the tick and thus the half a shutdown lands in.
        written = await self._run_capture_pass(capture_pass, budget, now, scan_id, counters)
        written += await self._emit_rows(
            [c for c in candidates if c.pass_index < 0], scan_id, now, counters
        )
        written += await self._emit_census_rows(candidates, scan_id, now, counters)

        # A mid-pass ``backoff_abort`` is its own tick outcome: labelling it
        # "ok" would hide truncation in exactly the window that matters most.
        tick_outcome = budget.abort_reason or "ok"

        await self._write_tick_row(
            scan_id=scan_id,
            now=now,
            tick_outcome=tick_outcome,
            n_cities_in_window=n_cities_in_window,
            counters=counters,
            budget=budget,
            tick_seed=tick_seed,
            backoff_remaining=backoff_remaining,
        )
        return written


    async def _fetch_markets(
        self, budget: RequestBudget, counters: _TickCounters
    ) -> list[dict[str, Any]]:
        """The one paginated leg, bounded by ``MARKETS_FETCH_MAX_PAGES``."""
        def fetch(cursor: str | None) -> Any:
            # Charged before the await: a fetch that raises on a later page
            # must still record the requests it actually made.
            budget.spend(1)
            counters.markets_pages_consumed += 1
            return self._client.get_markets(
                status="open",
                series_ticker=self._config.series_ticker,
                cursor=cursor,
                limit=MARKETS_PAGE_LIMIT,
            )

        result = await paginate_detailed(
            method_name="kxrain_observer.markets",
            list_field="markets",
            fetch=fetch,
            page_delay=self._config.request_pacing_seconds,
            filter_repr=f"series_ticker={self._config.series_ticker!r}",
            max_pages=MARKETS_FETCH_MAX_PAGES,
        )
        counters.markets_truncated = bool(result.truncated)
        counters.markets_cycle_aborted = bool(result.cycle_aborted)
        return result.rows

    async def _alert_incomplete_listing(self, counters: _TickCounters) -> None:
        """Truncation blinds the decision stream and the census at once —
        an alert, not a log line."""
        last = self._listing_alert_monotonic
        if last is not None and self._monotonic() - last < FAILURE_ALERT_COOLDOWN_SECONDS:
            return
        self._listing_alert_monotonic = self._monotonic()
        await self._alert(
            f"KXRAIN observer: the /markets listing came back incomplete "
            f"(truncated={counters.markets_truncated}, "
            f"cycle_aborted={counters.markets_cycle_aborted}, "
            f"pages={counters.markets_pages_consumed}, parsed={counters.n_parsed}). "
            "The market-day census is suppressed while this lasts, and the census "
            "is a pre-registered gate input.",
            AlertLevel.WARNING,
        )

    def _build_candidates(
        self,
        raw_markets: list[dict[str, Any]],
        now: datetime,
        counters: _TickCounters,
    ) -> list[_Candidate]:
        counters.n_markets_returned = len(raw_markets)
        candidates: list[_Candidate] = []
        for market in raw_markets:
            if not isinstance(market, dict):
                # A 2xx body can json.loads to a non-dict entry.
                counters.n_unparseable += 1
                counters.unparseable_tickers.append(repr(market)[:64])
                logger.warning("KxRainObserver: a /markets entry is not an object")
                continue
            ticker = str(market.get("ticker") or "")
            key = parse_kxrain_ticker(ticker)
            if key is None:
                # Counted and reported, never silently dropped — a silent
                # drop is how a corpus quietly loses instruments.
                if ticker.upper().startswith(f"{KXRAIN_SERIES_TICKER}-"):
                    counters.n_unparseable += 1
                    counters.unparseable_tickers.append(ticker)
                continue
            counters.n_parsed += 1
            candidates.append(self._make_candidate(key, market, now))
        return candidates

    def _make_candidate(
        self, key: KxRainMarketKey, market: dict[str, Any], now: datetime
    ) -> _Candidate:
        try:
            is_current_day = key.settlement_date == lst_day(key.city, now)
            clock_in_window = in_entry_window(
                key.city,
                now,
                self._config.entry_window_start_lst,
                self._config.entry_window_end_lst,
            )
        except (KeyError, ValueError):
            logger.error("KxRainObserver: cannot evaluate the window for %s", key.ticker)
            is_current_day = False
            clock_in_window = False

        yes_bid = to_dollar_price(market.get("yes_bid_dollars"))
        in_price_band = (
            yes_bid is not None
            and self._config.yes_bid_min <= yes_bid < self._config.yes_bid_max
        )
        follow_until = self._follow_until.get(key.corpus_key)
        return _Candidate(
            key=key,
            market=market,
            yes_bid=yes_bid,
            is_current_day=is_current_day,
            clock_in_window=clock_in_window,
            # Both legs required: a next-day market is never in its own
            # entry window even while the clock is.
            in_window=is_current_day and clock_in_window,
            in_price_band=in_price_band,
            already_entered=self._ledger.already_entered(key.corpus_key),
            follow_active=follow_until is not None and now < follow_until,
        )

    def _order_capture_pass(
        self, candidates: list[_Candidate], tick_seed: int
    ) -> list[_Candidate]:
        """Rank by ``yes_bid`` descending, ties shuffled by a logged per-tick
        seed, so an abort doesn't silently exclude the same cities every time."""
        capture_all = self._config.depth_capture_mode == "all"
        selected = [
            c
            for c in candidates
            # current-LST-day AND in_entry_window AND in_price_band, and
            # nothing else — skipping ``already_entered`` markets would
            # delete the fidelity gate's only input.
            if (c.in_window and (c.in_price_band or capture_all)) or c.follow_active
        ]
        shuffled = list(selected)
        random.Random(tick_seed).shuffle(shuffled)
        shuffled.sort(key=lambda c: c.yes_bid if c.yes_bid is not None else -1.0, reverse=True)
        for index, candidate in enumerate(shuffled):
            candidate.pass_index = index
        for candidate in candidates:
            if candidate.pass_index >= 0:
                continue
            candidate.book_capture = candidate.trades_capture = (
                "skipped_band" if candidate.in_window else "skipped_not_current_day"
            )
        return shuffled


    async def _run_capture_pass(
        self,
        capture_pass: list[_Candidate],
        budget: RequestBudget,
        now: datetime,
        scan_id: str,
        counters: _TickCounters,
    ) -> int:
        written = 0
        for index, candidate in enumerate(capture_pass):
            await self._capture_one(candidate, index, budget, now)
            written += await self._emit_candidate(candidate, scan_id, now)
            await self._maybe_flush(now)
        counters.n_decision_rows += written
        return written

    async def _capture_one(
        self, candidate: _Candidate, index: int, budget: RequestBudget, now: datetime
    ) -> None:
        if budget.aborted and not candidate.follow_active:
            candidate.book_capture = candidate.trades_capture = budget.abort_reason or "error"
            return

        # The shared 429 gate is bidirectional, so re-check before every
        # per-market read and abandon the pass rather than queue behind it.
        if self._backoff_remaining() > 0.0:
            budget.mark_backoff_abort()
            candidate.book_capture = candidate.trades_capture = "backoff_abort"
            return

        if index > 0 and self._config.request_pacing_seconds > 0:
            self._pacing_sleeps += 1
            await asyncio.sleep(self._config.request_pacing_seconds)

        if budget.reserve_market_pair():
            await self._fetch_trades(candidate, budget)
            await self._fetch_book(candidate)
            return

        # The book is the one artifact here that cannot be reconstructed, so
        # a follow-capture market may still take it from the overdraft.
        candidate.trades_capture = budget.abort_reason or "budget_abort"
        if candidate.follow_active and budget.reserve_book_overdraft():
            await self._fetch_book(candidate)
        else:
            candidate.book_capture = budget.abort_reason or "budget_abort"

    async def _fetch_trades(self, candidate: _Candidate, budget: RequestBudget) -> None:
        """Fetch the lookback window's prints, plus at most one extra page.
        ``now`` stamps the row; the query bounds use the actual call time."""
        call_ts = int(self._clock().timestamp())
        window_start = call_ts - self._config.no_trade_lookback_minutes * 60
        candidate.window_start_ts = window_start
        candidate.window_end_ts = call_ts
        try:
            payload = await self._client.get_market_trades(
                ticker=candidate.key.ticker,
                min_ts=window_start,
                max_ts=call_ts,
                limit=self._config.trades_page_limit,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "KxRainObserver: get_market_trades(%s) failed", candidate.key.ticker, exc_info=True
            )
            candidate.trades_capture = "error"
            return

        if not isinstance(payload, dict):
            logger.warning(
                "KxRainObserver: get_market_trades(%s) returned %s, not an object",
                candidate.key.ticker,
                type(payload).__name__,
            )
            candidate.trades_capture = "error"
            return

        window = summarise_trade_window(payload)
        rows = list(payload.get("trades") or [])
        cursor = str(payload.get("cursor") or "")

        # One page can span less than the lookback window, so fetch one extra
        # page in the one case where a missed print could flip eligibility.
        if cursor and window.n_trades_no_side == 0 and budget.can_spend(1):
            budget.spend(1)
            try:
                extra = await self._client.get_market_trades(
                    ticker=candidate.key.ticker,
                    min_ts=window_start,
                    max_ts=call_ts,
                    limit=self._config.trades_page_limit,
                    cursor=cursor,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "KxRainObserver: second trades page for %s failed",
                    candidate.key.ticker,
                    exc_info=True,
                )
                extra = None
            if isinstance(extra, dict):
                rows.extend(list(extra.get("trades") or []))
                window = summarise_trade_window(
                    {"trades": rows, "cursor": str(extra.get("cursor") or "")}
                )
                candidate.n_pages_fetched = 2

        candidate.trade_window = window
        candidate.trades_capture = "ok"

    async def _fetch_book(self, candidate: _Candidate) -> None:
        try:
            orderbook = await self._client.get_orderbook(
                candidate.key.ticker, depth=self._config.depth_levels
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "KxRainObserver: get_orderbook(%s) failed", candidate.key.ticker, exc_info=True
            )
            candidate.book_capture = "error"
            return
        if not isinstance(orderbook, dict):
            logger.warning(
                "KxRainObserver: get_orderbook(%s) returned %s, not an object",
                candidate.key.ticker,
                type(orderbook).__name__,
            )
            candidate.book_capture = "error"
            return
        candidate.orderbook = orderbook
        candidate.book_capture = "ok"


    async def _emit_rows(
        self,
        candidates: list[_Candidate],
        scan_id: str,
        now: datetime,
        counters: _TickCounters,
    ) -> int:
        """Rows for markets that never entered the capture pass — no HTTP cost,
        and keeps row existence unconditioned on anything the rule looks at."""
        written = 0
        for candidate in candidates:
            written += await self._emit_candidate(candidate, scan_id, now)
            await self._maybe_flush(now)
        counters.n_decision_rows += written
        return written

    async def _emit_candidate(self, candidate: _Candidate, scan_id: str, now: datetime) -> int:
        await self._consume_virtual_entry(candidate, scan_id, now)
        decision_id = f"{scan_id}:{candidate.key.ticker}"

        self._queue(
            KXRAIN_DECISIONS_FILENAME,
            build_decision_row(
                key=candidate.key,
                market=candidate.market,
                outcome=candidate.outcome,
                scan_id=scan_id,
                ts=now,
                in_window=candidate.in_window,
                already_entered=candidate.already_entered,
                virtual_entry=candidate.virtual_entry,
                pass_index=candidate.pass_index,
                book_capture=candidate.book_capture,
                trades_capture=candidate.trades_capture,
                follow_capture=candidate.follow_active,
                observer_version=self._observer_version,
                max_entries_per_day=self._config.max_entries_per_day,
                assumed_max_walk_cents=self._config.assumed_max_walk_cents,
                trade_window=candidate.trade_window,
                window_end_ts=candidate.window_end_ts,
            ),
        )

        if candidate.trade_window is not None and candidate.window_start_ts is not None:
            self._queue(
                KXRAIN_TRADES_FILENAME,
                build_trades_row(
                    decision_id=decision_id,
                    ts=now,
                    ticker=candidate.key.ticker,
                    window=candidate.trade_window,
                    window_start_ts=candidate.window_start_ts,
                    window_end_ts=candidate.window_end_ts or candidate.window_start_ts,
                    n_pages_fetched=candidate.n_pages_fetched,
                ),
            )
        if candidate.book_capture == "ok":
            self._queue(
                KXRAIN_BOOK_FILENAME,
                build_book_row(
                    decision_id=decision_id,
                    ts=now,
                    ticker=candidate.key.ticker,
                    orderbook=candidate.orderbook,
                    markets_yes_bid=candidate.yes_bid,
                    depth_levels=self._config.depth_levels,
                ),
            )

        self._last_seen[candidate.key.ticker] = (
            candidate.yes_bid,
            now.isoformat(),
            candidate.key,
        )
        return 1

    async def _consume_virtual_entry(
        self, candidate: _Candidate, scan_id: str, now: datetime
    ) -> None:
        candidate.outcome = evaluate_decision(
            yes_bid=candidate.yes_bid,
            in_window=candidate.in_window,
            no_trades_30m=(
                candidate.trade_window.n_trades_no_side
                if candidate.trade_window is not None
                else None
            ),
            already_entered=candidate.already_entered,
            yes_bid_min=self._config.yes_bid_min,
            yes_bid_max=self._config.yes_bid_max,
        )
        if candidate.outcome.eligible is not True:
            return

        key = candidate.key.corpus_key
        try:
            # ``record_entry`` is a blocking write fired immediately per
            # entry; run inline it would stall the whole event loop.
            candidate.virtual_entry = await asyncio.wrap_future(
                self._executor.submit(
                    partial(
                        self._ledger.record_entry,
                        key,
                        decision_id=f"{scan_id}:{candidate.key.ticker}",
                        ts=now.isoformat(),
                        assumed_entry_price=candidate.outcome.assumed_entry_price,
                        scan_id=scan_id,
                    )
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("KxRainObserver: recording the virtual entry for %s failed", key)
            return

        if self._ledger.already_entered(key):
            self._follow_until[key] = now + timedelta(
                minutes=self._config.post_entry_follow_minutes
            )

    async def _emit_census_rows(
        self,
        candidates: list[_Candidate],
        scan_id: str,
        now: datetime,
        counters: _TickCounters,
    ) -> int:
        """A vanished market-day must be a row, not a gap. Absence counts only
        on a complete page, and only after :data:`CENSUS_ABSENT_TICKS_REQUIRED`
        of them, since a short page can look identical to a complete one."""
        present = {c.key.ticker for c in candidates}
        # The ticker suffix is a vendor symbol Kalshi can rename; both
        # spellings resolve to one ``corpus_key``, so a rename isn't an early close.
        present_keys = {c.key.corpus_key for c in candidates}
        for ticker in present:
            self._absent_streak.pop(ticker, None)

        if not counters.markets_page_complete:
            counters.census_suppressed = True
            logger.warning(
                "KxRainObserver: /markets came back incomplete (truncated=%s cycle=%s) "
                "— suppressing the market-absent census for this tick",
                counters.markets_truncated,
                counters.markets_cycle_aborted,
            )
            return 0

        stale: list[str] = []
        absent: list[tuple[str, KxRainMarketKey, date, bool, float | None, str]] = []
        for ticker, (last_bid, last_ts, key) in self._last_seen.items():
            if ticker in present:
                continue
            if key.corpus_key in present_keys:
                stale.append(ticker)
                continue
            try:
                today = lst_day(key.city, now)
                clock_in_window = in_entry_window(
                    key.city,
                    now,
                    self._config.entry_window_start_lst,
                    self._config.entry_window_end_lst,
                )
            except (KeyError, ValueError):
                stale.append(ticker)
                continue
            if key.settlement_date < today:
                # It simply settled; absence is only newsworthy while the
                # market-day should still be open.
                stale.append(ticker)
                continue
            absent.append((ticker, key, today, clock_in_window, last_bid, last_ts))

        if len(absent) > CENSUS_MAX_ABSENT_PER_TICK:
            # A short page with a terminal cursor carries no flag of its
            # own; leave streaks untouched, as on an incomplete page.
            counters.census_suppressed = True
            logger.warning(
                "KxRainObserver: %d market-days vanished in one tick (n_parsed=%d) "
                "— treating the page as short and suppressing the census",
                len(absent),
                counters.n_parsed,
            )
            for ticker in stale:
                self._last_seen.pop(ticker, None)
                self._absent_streak.pop(ticker, None)
            return 0

        written = 0
        for ticker, key, today, clock_in_window, last_bid, last_ts in absent:
            streak = self._absent_streak.get(ticker, 0) + 1
            self._absent_streak[ticker] = streak
            if streak < CENSUS_ABSENT_TICKS_REQUIRED:
                continue
            self._queue(
                KXRAIN_DECISIONS_FILENAME,
                build_decision_row(
                    key=key,
                    market=None,
                    outcome=None,
                    scan_id=scan_id,
                    ts=now,
                    in_window=(key.settlement_date == today) and clock_in_window,
                    already_entered=self._ledger.already_entered(key.corpus_key),
                    virtual_entry=False,
                    pass_index=-1,
                    book_capture="skipped_market_absent",
                    trades_capture="skipped_market_absent",
                    follow_capture=False,
                    observer_version=self._observer_version,
                    max_entries_per_day=self._config.max_entries_per_day,
                    assumed_max_walk_cents=self._config.assumed_max_walk_cents,
                    last_seen_yes_bid=last_bid,
                    last_seen_ts=last_ts,
                ),
            )
            written += 1
            await self._maybe_flush(now)
        for ticker in stale:
            self._last_seen.pop(ticker, None)
            self._absent_streak.pop(ticker, None)
        counters.n_decision_rows += written
        return written

    async def _write_tick_row(
        self,
        *,
        scan_id: str,
        now: datetime,
        tick_outcome: str,
        n_cities_in_window: int,
        counters: _TickCounters,
        budget: RequestBudget,
        tick_seed: int,
        backoff_remaining: float,
    ) -> None:
        """Written last, with the real counters, on every tick attempt."""
        self._last_tick_outcome = tick_outcome
        self._queue(
            KXRAIN_TICKS_FILENAME,
            build_tick_row(
                scan_id=scan_id,
                ts=now,
                process_start_ts=self._process_start_ts,
                tick_outcome=tick_outcome,
                n_cities_in_window=n_cities_in_window,
                n_markets_returned=counters.n_markets_returned,
                n_parsed=counters.n_parsed,
                n_unparseable=counters.n_unparseable,
                unparseable_tickers=counters.unparseable_tickers,
                markets_truncated=counters.markets_truncated,
                markets_cycle_aborted=counters.markets_cycle_aborted,
                markets_pages_consumed=counters.markets_pages_consumed,
                census_suppressed=counters.census_suppressed,
                requests_used=budget.requests_used,
                max_requests_per_tick=budget.max_requests,
                tick_seed=tick_seed,
                elapsed_seconds=budget.elapsed_seconds,
                backoff_remaining_at_start=backoff_remaining,
                backoff_accessor_present=self._backoff_accessor_present,
                n_decision_rows=counters.n_decision_rows,
                rows_lost=self._rows_lost,
                oversize_rows_dropped=self._oversize_rows_dropped,
                observer_version=self._observer_version,
                tick_interval_seconds=self._config.tick_interval_seconds,
            ),
        )
        await self._flush(now)


    def _queue(self, filename: str, row: dict[str, Any]) -> None:
        """Length-check a row as the writer will serialise it, then park it.
        ``ForecastsRawWriter`` has no PIPE_BUF check, so an oversized row would
        otherwise be silently truncated mid-JSON."""
        if encode_row(row) is None:
            self._oversize_rows_dropped += 1
            logger.error(
                "KxRainObserver: row for %s exceeds PIPE_BUF; writing a join-keys stub",
                row.get("decision_id") or row.get("scan_id"),
            )
            row = build_oversize_stub_row(row)
        self._pending[filename].append(row)

    async def _maybe_flush(self, now: datetime) -> None:
        if len(self._pending[KXRAIN_DECISIONS_FILENAME]) >= WRITE_FLUSH_EVERY_ROWS:
            await self._flush(now)

    async def _flush(self, now: datetime) -> None:
        """Write every parked batch off the event loop, exactly once. A thread
        can't be cancelled once running, so ``Future.cancel()`` is the only
        way to know whether to requeue a batch or leave it written."""
        for filename, rows in self._pending.items():
            if not rows:
                continue
            batch = list(rows)
            rows.clear()
            writer = self._writers[filename]
            before = writer.success_count
            try:
                future = self._executor.submit(writer.write_batch, batch)
            except RuntimeError:
                # The executor is already shut down; the loop is going away.
                rows[:0] = batch
                continue
            try:
                await asyncio.wrap_future(future)
            except asyncio.CancelledError:
                if future.cancel():
                    rows[:0] = batch
                raise
            except Exception:
                logger.exception("KxRainObserver: write_batch(%s) failed", filename)
            self._observe_persistence(filename, writer.success_count - before, len(batch), now)

    def _flush_sync(self) -> None:
        now = self._clock()
        for filename, rows in self._pending.items():
            if not rows:
                continue
            batch = list(rows)
            rows.clear()
            writer = self._writers[filename]
            before = writer.success_count
            try:
                writer.write_batch(batch)
            except Exception:
                logger.exception("KxRainObserver: synchronous write_batch(%s) failed", filename)
            self._observe_persistence(filename, writer.success_count - before, len(batch), now)

    def _observe_persistence(
        self, filename: str, persisted: int, attempted: int, now: datetime
    ) -> None:
        """Advance liveness on rows that reached disk, never on rows built —
        ``write_batch`` swallows every ``OSError``, so ``success_count`` is the
        only signal a row actually landed."""
        if persisted > 0:
            if filename == KXRAIN_DECISIONS_FILENAME:
                self._last_decision_write_utc = now
                self._liveness_alert_monotonic = None
            if persisted < attempted:
                self._rows_lost += attempted - persisted
                logger.error(
                    "KxRainObserver: %s persisted %d of %d rows", filename, persisted, attempted
                )
            return
        self._rows_lost += attempted
        logger.error(
            "KxRainObserver: %s persisted NONE of %d rows — the corpus is not accruing",
            filename,
            attempted,
        )


    async def _check_disk(self, now: datetime) -> bool:
        """Returns False once permanently self-disabled. A failed
        ``disk_usage`` call is not a reason to disable — fail open."""
        try:
            free = _free_disk_bytes(Path(self._config.log_dir))
        except Exception:
            logger.warning("KxRainObserver: cannot read free disk space", exc_info=True)
            return True
        if free >= self._config.min_free_disk_bytes:
            return True

        self._disabled = True
        logger.critical(
            "KxRainObserver: only %d bytes free under %s (floor %d) — permanently "
            "disabling the observer for this process. A logging-only feature must "
            "never starve risk-state persistence or state/trades.jsonl.",
            free,
            self._config.log_dir,
            self._config.min_free_disk_bytes,
        )
        if not self._disk_alert_sent:
            self._disk_alert_sent = True
            await self._alert(
                f"KXRAIN observer disabled at {now.isoformat()}: only "
                f"{free / 1024**3:.2f} GiB free under {self._config.log_dir}. "
                "Corpus logging has stopped; clear disk space and restart the bot.",
                AlertLevel.CRITICAL,
            )
        return False

    def _count_cities_in_window(self, now: datetime) -> int:
        count = 0
        for city in KXRAIN_CITY_STD_OFFSET_HOURS:
            try:
                if in_entry_window(
                    city,
                    now,
                    self._config.entry_window_start_lst,
                    self._config.entry_window_end_lst,
                ):
                    count += 1
            except (KeyError, ValueError):
                continue
        return count

    def _prune_follow_windows(self, now: datetime) -> None:
        expired = [key for key, until in self._follow_until.items() if now >= until]
        for key in expired:
            self._follow_until.pop(key, None)

    def _backoff_remaining(self) -> float:
        """Read the shared 429 gate through the public accessor only — never
        ``client._backoff_until``, which breaks a fake client."""
        accessor = getattr(self._client, "backoff_remaining_seconds", None)
        if not callable(accessor):
            self._note_backoff_accessor_missing(
                "KxRainObserver: the Kalshi client has no backoff_remaining_seconds(); "
                "the yield-to-trading rule is INERT"
            )
            return 0.0
        try:
            value = max(0.0, float(accessor()))
        except Exception:
            self._note_backoff_accessor_missing(
                "KxRainObserver: backoff_remaining_seconds() raised; the "
                "yield-to-trading rule is INERT for this tick"
            )
            return 0.0
        self._backoff_accessor_present = True
        return value

    def _note_backoff_accessor_missing(self, message: str) -> None:
        self._backoff_accessor_present = False
        if self._backoff_accessor_warned:
            return
        self._backoff_accessor_warned = True
        logger.warning(message)

    async def _check_liveness(self, now: datetime, n_cities_in_window: int) -> None:
        """One WARNING when the corpus goes silent while a city is in window.
        Tracks decision rows, not tick rows, which are written even on a
        failing fetch and would never trip the alert."""
        if n_cities_in_window == 0:
            return
        reference = self._last_decision_write_utc or self._process_start_ts
        if (now - reference).total_seconds() < LIVENESS_ALERT_AFTER_SECONDS:
            return
        # Repeats on a cooldown rather than once per process.
        last = self._liveness_alert_monotonic
        if last is not None and self._monotonic() - last < FAILURE_ALERT_COOLDOWN_SECONDS:
            return
        self._liveness_alert_monotonic = self._monotonic()
        logger.warning(
            "KxRainObserver liveness: no decision row written since %s while %d "
            "cities are in window",
            reference.isoformat(),
            n_cities_in_window,
        )
        await self._alert(
            f"KXRAIN observer liveness: no decision row since {reference.isoformat()} "
            f"while {n_cities_in_window} cities are inside the entry window. The "
            "forward corpus may be dead; it cannot be backfilled.",
            AlertLevel.WARNING,
        )

    async def _maybe_alert_consecutive_failures(self) -> None:
        if self._consecutive_failed_ticks < CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
            return
        now = self._monotonic()
        last = self._last_failure_alert_monotonic
        if last is not None and now - last < FAILURE_ALERT_COOLDOWN_SECONDS:
            return
        self._last_failure_alert_monotonic = now
        await self._alert(
            f"KXRAIN observer: {self._consecutive_failed_ticks} consecutive failed ticks. "
            "The forward corpus is not accruing.",
            AlertLevel.WARNING,
        )

    async def aclose(self, timeout: float = ACLOSE_TIMEOUT_SECONDS) -> None:
        """Flush what is parked and stop the worker, off the loop and bounded
        — run inline, both halves block and would starve the loop upstream of
        live order cancellation."""
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(loop.run_in_executor(None, self._close_sync), timeout)
        except TimeoutError:
            logger.error(
                "KxRainObserver.aclose timed out after %.0fs; abandoning the parked "
                "batch so shutdown can reach order cancellation",
                timeout,
            )

    def _close_sync(self) -> None:
        self._flush_sync()
        self._executor.shutdown(wait=True)

    async def _alert(self, message: str, level: AlertLevel) -> None:
        """Send one alert. Resolves the notifier lazily; never raises."""
        try:
            notifier = self._notifier_getter() if self._notifier_getter is not None else None
        except Exception:
            logger.debug("KxRainObserver: notifier getter failed", exc_info=True)
            return
        if notifier is None:
            return
        try:
            await notifier.send_alert(message, level=level)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("KxRainObserver: alert delivery failed", exc_info=True)


def _read_observer_version(log_dir: Path) -> str:
    """``state/.build_sha``, then ``ORACLE_BUILD_SHA``, then ``"unknown"``.
    Read once at construction; never subprocess on the tick path."""
    try:
        sha_file = log_dir / BUILD_SHA_FILENAME
        if sha_file.is_file():
            text = sha_file.read_text(encoding="utf-8").strip()
            if text:
                return text[:64]
    except OSError:
        logger.debug("KxRainObserver: cannot read %s", BUILD_SHA_FILENAME, exc_info=True)
    return (os.environ.get("ORACLE_BUILD_SHA") or "unknown").strip()[:64] or "unknown"
