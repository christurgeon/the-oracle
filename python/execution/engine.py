"""
Execution engine for The Oracle.

Takes TradeSignal objects, validates them through risk, and places orders
in one of three modes: shadow (log only), paper (Kalshi sandbox), live.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from python.analytics.fees import order_fee as kalshi_order_fee
from python.analytics.sizing import binary_contract_pnl, dollars_to_contracts
from python.common import ET, ExecutionMode, OrderStatus, Side
from python.data.contract_watcher import ContractWatcher
from python.data.kalshi_client import KalshiAPIError, KalshiClient, OrderValidationError
from python.data.kalshi_schemas import (
    SCHEMA_FILL,
    SCHEMA_MARKET_SETTLEMENT,
    SCHEMA_ORDER_RESPONSE,
    SCHEMA_ORDER_STATUS_CHECK,
    KalshiSchema,
)
from python.execution.trade_log import TradeLogEvent, is_pnl_relevant_row
from python.risk.manager import RiskDecision, RiskManager
from python.settings import Settings
from python.strategies.base import TradeSignal
from python.utils.jsonl import iter_jsonl

if TYPE_CHECKING:
    from python.execution.shadow_broker import ShadowPosition

logger = logging.getLogger(__name__)

# Log-prefix tags for grep/aggregation rules in operator dashboards.
KILL_SWITCH_TRUNCATED_PREFIX = "KILL_SWITCH_TRUNCATED"
KILL_SWITCH_TIMEOUT_PREFIX = "KILL_SWITCH_TIMEOUT"

# Statuses meaning "order is on the book and may still fill" — public so other
# modules (e.g., python.notifications.formatters) can reason about it without
# reaching into class internals.
PENDING_STATUSES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.PENDING, OrderStatus.PARTIAL, OrderStatus.RESTING}
)

# Statuses that unconditionally represent a fill worth announcing.
# EXPIRED/CANCELLED count only when fill_count > 0 (a partial position
# acquired before the remainder cancelled).
_ALERTABLE_FILL_STATUSES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.FILLED, OrderStatus.PARTIAL, OrderStatus.SHADOW}
)


def is_unalerted_new_fill(record: OrderRecord) -> bool:
    """True if ``record`` holds contracts not yet announced to the operator.

    Dedup is keyed on ``record.last_alerted_fill_count`` (default sentinel -1),
    so a resting partial re-polled every tick announces once per new fill
    quantity, not once per tick. Cases:

    - FILLED / PARTIAL / SHADOW: ``fill_count > last_alerted`` — the degraded
      FILLED/fill_count==0 record announces once (``0 > -1``).
    - EXPIRED / CANCELLED with ``fill_count > 0``: a partial position was
      acquired before the remainder was cancelled or TTL-expired (e.g. the
      cancel-race window in ``_expire_stale_orders``) — still a real fill.
    - Everything else (REJECTED/SKIPPED/ERROR/RESTING/PENDING/HYDRATED/…): never.
    """
    if record.status in _ALERTABLE_FILL_STATUSES:
        return record.fill_count > record.last_alerted_fill_count
    if record.status in (OrderStatus.EXPIRED, OrderStatus.CANCELLED):
        return record.fill_count > 0 and record.fill_count > record.last_alerted_fill_count
    return False


SETTLEMENT_RESULT_EXPIRED = "expired"


def _synthesize_close_time(ticker_date: Any) -> str:
    """Derive a close_time for legacy fills whose metadata lacks one.

    Weather markets settle the morning after the ticker date; picking
    00:00 ET next day guarantees the expiry sweep releases the position
    within a day of the ticker's ET date.
    """
    if ticker_date is None:
        return ""
    synth = datetime.combine(ticker_date + timedelta(days=1), datetime.min.time())
    return synth.replace(tzinfo=ET).isoformat()


# Kalshi API returns status strings that don't match our enum values 1:1:
# "canceled" (American spelling vs our British "cancelled") and "executed"
# (vs our "filled"). Without these aliases, fully-filled orders stay stuck
# at PENDING and canceled orders stay at RESTING — leaking exposure and
# spamming 404s on TTL-cancel retries.
_KALSHI_STATUS_ALIASES: dict[str, OrderStatus] = {
    "canceled": OrderStatus.CANCELLED,
    "executed": OrderStatus.FILLED,
}


def _parse_order_status(
    raw: str,
    fallback: OrderStatus = OrderStatus.PENDING,
    contract_watcher: ContractWatcher | None = None,
) -> OrderStatus:
    """Safely convert a Kalshi API status string to OrderStatus.

    *fallback* is returned when *raw* doesn't match any known status.
    Pass the record's current status in check_fills to avoid
    misclassifying an unknown terminal status as PENDING.
    """
    aliased = _KALSHI_STATUS_ALIASES.get(raw)
    if aliased is not None:
        return aliased
    try:
        return OrderStatus(raw)
    except ValueError:
        logger.warning("Unknown order status from Kalshi: %s — keeping %s", raw, fallback)
        if contract_watcher is not None:
            contract_watcher.warn(
                f"unknown_order_status:{raw}",
                f"Unknown Kalshi order status '{raw}' — keeping {fallback}. "
                f"API may have added a new terminal state.",
            )
        return fallback


# Forces compute_limit_cents to cross to the ask for taker signals; intentionally
# exceeds the config's cross_to_ask_max_spread_cents (le=2) bound — it's a
# caller-side override, not a config value.
_TAKER_ALWAYS_CROSS_CENTS = 100


# Silent order-placement-failure detectors: a failed placement produces neither
# a dead process nor a realized loss, so it slips past liveness monitors and
# P&L circuit breakers (the 2026-06-19 Kalshi V1-deprecation outage ran
# silently for 2 days / 5,296 failures). These watch the intent->execution gap
# directly.

# Statuses meaning the order was successfully submitted to Kalshi.
_PLACED_OK_STATUSES = (OrderStatus.PENDING, OrderStatus.FILLED, OrderStatus.PARTIAL)
# Consecutive live ticks with placement intent but zero successful placements
# before the divergence watchdog escalates.
_PLACEMENT_DIVERGENCE_TICKS = 3
# Min seconds between placement-failure alerts (dedup; shared by both detectors).
_PLACEMENT_ALERT_COOLDOWN_S = 900


def _is_non_transient_placement_error(exc: BaseException) -> bool:
    """True for placement failures that retrying won't fix — a deprecated
    endpoint (410), auth (401/403), or a bad request (400/422).

    429/5xx are already retried by ``KalshiClient._request``, so a KalshiAPIError
    surfacing here with a 4xx status (other than 429 rate-limiting) is a
    persistent break worth alerting on.
    """
    return isinstance(exc, KalshiAPIError) and 400 <= exc.status < 500 and exc.status != 429


def _batch_attempt_counts(records: list[OrderRecord]) -> tuple[int, int]:
    """``(attempted, placed_ok)`` for an executed batch.

    ``attempted`` counts only risk-approved placements — a genuine intent to
    trade — so it excludes REJECTED / SKIPPED / ZERO_SIZE records. ``placed_ok``
    counts those that actually reached Kalshi (PENDING/FILLED/PARTIAL). An
    ERROR record is an attempt that failed.
    """
    attempted = 0
    placed_ok = 0
    for r in records:
        if r.status in _PLACED_OK_STATUSES:
            attempted += 1
            placed_ok += 1
        elif r.status == OrderStatus.ERROR:
            attempted += 1
    return attempted, placed_ok


def compute_limit_cents(
    *,
    order_price_dollars: float,
    side_bid_cents: int,
    side_ask_cents: int,
    cross_to_ask_max_spread_cents: int = 1,
) -> int:
    """Compute the integer-cent limit price to submit to Kalshi.

    Replaces ``int(round(price * 100))`` whose banker's rounding turned the
    strategy's midpoint into a parity-driven coin flip between bid (would not
    fill) and ask (taker) on 1c spreads.  Rule:
      - spread <= ``cross_to_ask_max_spread_cents``
                         -> cross to the ask (deterministic taker, fills).
                            Default threshold = 1 (1c-spread only).  Set to 2
                            to also cross 2c spreads ("modified Lever A").
      - wider spread     -> post strictly inside, biased toward the bid (maker).
      - locked/crossed   -> submit at the ask.
    """
    if side_ask_cents <= side_bid_cents:
        return side_ask_cents
    spread = side_ask_cents - side_bid_cents
    if spread <= cross_to_ask_max_spread_cents:
        return side_ask_cents
    raw = order_price_dollars * 100.0
    inside_low = side_bid_cents + 1
    inside_high = side_ask_cents - 1
    capped = max(min(raw, float(inside_high)), float(inside_low))
    return int(capped)  # truncates toward zero -> toward the bid for positive prices


@dataclass(frozen=True)
class SettlementResult:
    """Result of settling a single position."""

    position_key: str
    market_ticker: str
    result: str  # "yes", "no", or "expired"
    pnl: float


@dataclass
class OrderRecord:
    """A record of an order placed (or shadow-logged) by the engine."""

    order_id: str  # Kalshi order ID, or "shadow-{n}" for shadow mode
    signal: TradeSignal
    risk_decision: RiskDecision
    mode: ExecutionMode
    signal_price: float  # market price when signal was generated
    order_price: float  # limit price submitted
    fill_price: float | None = None  # actual fill price (None if unfilled)
    fill_count: int = 0
    requested_count: int = 0
    status: OrderStatus = OrderStatus.PENDING
    slippage: float = 0.0  # fill_price - order_price
    last_recorded_dollars: float = 0.0  # dollars already reported to risk manager
    last_logged_fill_count: int = 0  # fill_count already written to the trade log
    # Sentinel -1 (not 0) so a degraded FILLED/fill_count==0 record still
    # announces once, while REJECTED/SKIPPED (fill_count 0) stay silent.
    last_alerted_fill_count: int = -1
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    kalshi_response: dict[str, Any] = field(default_factory=dict)
    # Populated by check_fills() from /portfolio/fills; 0 = no fill data yet.
    is_taker_fills: int = 0  # contracts filled as taker (Kalshi: is_taker=true)
    is_maker_fills: int = 0  # contracts filled as maker (Kalshi: is_taker=false)
    fees_from_api: float = 0.0  # cumulative fee in dollars reported by Kalshi
    settle_defer_count: int = 0
    settle_deferred: bool = False  # in-memory only; reset on _orders rebuild at restart
    # True iff the most recent get_fills call succeeded and the per-fill loop
    # completed without exception — gates the authoritative vs. fee-fallback
    # settlement path in _settle_orders.
    fills_payload_received: bool = False
    # Display-only fill summary restored from trades.jsonl at startup.
    # NEVER used by settlement or P&L code — only read by formatters.
    restored_fill_info: tuple[float, int] | None = None  # (vwap_fill_price, total_contracts)
    # Guards against double-writing a cancel row if check_fills observes the
    # terminal status twice in one process lifetime. Resets on restart;
    # hydration uses set semantics so cross-restart duplicates are benign.
    cancel_row_written: bool = False
    # Per-order TTL override (seconds); None -> execution.order_ttl_seconds.
    # Set from favorite_bias signal metadata so a taker order can rest at the
    # frozen signal-time ask longer than the global TTL.
    ttl_seconds_override: int | None = None


@dataclass(frozen=True)
class LivePosition:
    """API-derived position for engine-side hydration on restart / mid-run drift."""

    position_key: str
    market_ticker: str
    side: Side
    dollars: float


class ExecutionEngine:
    """
    Async execution engine.

    Modes:
      - shadow: log everything but place no orders
      - paper:  place orders on Kalshi demo/sandbox
      - live:   place real orders on Kalshi production
    """

    _PENDING_STATUSES = PENDING_STATUSES
    _TERMINAL_NO_FILL_STATUSES = {OrderStatus.CANCELLED, OrderStatus.EXPIRED}
    # HYDRATED is intentionally excluded so _settle_live can release exposure
    # even if pruning runs first.
    _PRUNABLE_STATUSES = {
        OrderStatus.SETTLED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
        OrderStatus.SKIPPED,
        OrderStatus.ERROR,
        OrderStatus.ZERO_SIZE,
        OrderStatus.EXTERNALLY_CLOSED,  # manual-close stamp
    }
    _ORDER_DEDUP_WINDOW = 30.0
    _MAX_RETAINED_ORDERS = 500

    def __init__(
        self,
        settings: Settings,
        kalshi_client: KalshiClient | None,
        risk_manager: RiskManager,
        contract_watcher: ContractWatcher | None = None,
        trade_log_lock: asyncio.Lock | None = None,
        notify_warning: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        notify_critical: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._settings = settings
        self._client = kalshi_client
        self._risk = risk_manager
        self._contract_watcher = contract_watcher
        self._mode = ExecutionMode(settings.execution.mode)

        self._orders: dict[str, OrderRecord] = {}
        self._id_counters: dict[str, int] = {}

        self._recent_orders: dict[tuple[str, Side], float] = {}

        self._trade_log_path = Path(settings.state_dir) / "trades.jsonl"
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Shared lock serializing all writes to trades.jsonl.  Injected by the
        # orchestrator so FillsReconciler can share the same lock.  Falls back
        # to a per-engine lock so existing tests that construct ExecutionEngine
        # without the argument still work.
        self._trade_log_lock: asyncio.Lock = (
            trade_log_lock if trade_log_lock is not None else asyncio.Lock()
        )

        # Emits a placement row immediately on successful place_order, before
        # any immediate-fill check_fills. Default off until operator verifies
        # downstream consumers handle event="placement".
        self._emit_placement_rows: bool = settings.execution.emit_placement_rows

        # Per-tick scratchpad for SETTLE FEE FALLBACK ids; consumed by _settle_live
        # to fire Telegram CRITICAL after the position settles. Cleared at start
        # of each _settle_live invocation.
        self._fee_fallback_records_this_tick: list[str] = []

        # Aggregate-deferral alert state.
        self._deferral_alert_last_fired: float | None = None
        self._deferral_alert_active_last_tick: bool = False
        # Order-placement-failure alert state (#1 fast 4xx + #2 divergence watchdog).
        self._placement_alert_last_fired: float | None = None
        self._consecutive_failed_place_ticks: int = 0
        # Optional warning/critical callbacks wired by orchestrator. None → log only.
        self._notify_warning: Callable[[str], Coroutine[Any, Any, None]] | None = notify_warning
        self._notify_critical: Callable[[str], Coroutine[Any, Any, None]] | None = notify_critical

        # Optional per-day-cap release callback, wired by orchestrator after
        # both engine and strategy are constructed. Fired by _handle_cancellation
        # on every 0-fill terminal status. Default None → no-op (backtest,
        # shadow paths, or strategies without a cap).
        self._on_cancel_callback: Callable[[str], Any] | None = None

        # state/settled_order_ids.json — separate dedup-token file for
        # crash-after-write idempotency.
        self._settled_ids_path = Path(settings.state_dir) / "settled_order_ids.json"
        self._settled_ids_lock: asyncio.Lock = asyncio.Lock()

        logger.info("Execution engine initialized in %s mode", self._mode.value)

    def _check_contract(self, data: dict[str, Any], schema: KalshiSchema, *, context: str) -> None:
        if self._contract_watcher is not None:
            self._contract_watcher.check(data, schema, context=context)

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def _next_id(self, prefix: str) -> str:
        """Generate a unique order ID like 'rejected-3' for non-API records."""
        self._id_counters[prefix] = self._id_counters.get(prefix, 0) + 1
        return f"{prefix}-{self._id_counters[prefix]}"

    @property
    def order_history(self) -> list[OrderRecord]:
        return list(self._orders.values())

    async def execute(self, signal: TradeSignal) -> OrderRecord:
        """
        Execute a single trade signal.

        1. Run risk check
        2. If approved, place order (or shadow-log)
        3. Return OrderRecord
        """
        decision = self._risk.check_signal(signal)

        if not decision.approved:
            logger.info(
                "Signal REJECTED for %s (%s): %s",
                signal.market_ticker,
                signal.strategy_name,
                "; ".join(decision.rejection_reasons),
            )
            record = OrderRecord(
                order_id=self._next_id("rejected"),
                signal=signal,
                risk_decision=decision,
                mode=self._mode,
                signal_price=signal.market_price,
                order_price=0.0,
                status=OrderStatus.REJECTED,
            )
            self._orders[record.order_id] = record
            return record

        # Convert to side-denominated price: signal.market_price is always
        # the YES midpoint.  NO orders need (1 - market_price).
        if signal.side == Side.YES:
            order_price = signal.market_price
        else:
            order_price = 1.0 - signal.market_price
        dollars = decision.adjusted_dollars

        # Use depth-adjusted VWAP fill price when the strategy performed an
        # orderbook depth check.  None/0.0 means no depth data — fall back to
        # the raw midpoint.
        meta_fill_price: float | None = signal.metadata.get("fill_price")
        if meta_fill_price and meta_fill_price > 0.0:
            # fill_price is side-denominated; convert NO back to YES-space
            # because dollars_to_contracts expects the YES midpoint.
            has_depth_price = True
            if signal.side == Side.YES:
                effective_price = meta_fill_price
            else:
                effective_price = 1.0 - meta_fill_price
        else:
            has_depth_price = False
            effective_price = signal.market_price

        contract_count = dollars_to_contracts(dollars, effective_price, signal.side)

        # Cap by strategy's depth-checked suggested_contracts when available.
        if has_depth_price and signal.suggested_contracts > 0:
            contract_count = min(contract_count, signal.suggested_contracts)

        if contract_count <= 0:
            logger.info(
                "Signal for %s resulted in 0 contracts (dollars=%.2f)",
                signal.market_ticker,
                dollars,
            )
            record = OrderRecord(
                order_id=self._next_id("zero-size"),
                signal=signal,
                risk_decision=decision,
                mode=self._mode,
                signal_price=signal.market_price,
                order_price=order_price,
                status=OrderStatus.ZERO_SIZE,
            )
            self._orders[record.order_id] = record
            return record

        if self._mode == ExecutionMode.SHADOW:
            return await self._shadow_execute(signal, decision, order_price, contract_count)
        else:
            return await self._live_execute(signal, decision, order_price, contract_count)

    async def execute_batch(self, signals: list[TradeSignal]) -> list[OrderRecord]:
        """Execute a batch of signals sequentially with per-tick dedup.

        Only the first signal per ``(market_ticker, side)`` key is executed;
        duplicates receive ``OrderStatus.SKIPPED``.  The dedup set resets
        each call (per-tick scope).

        Callers should sort *signals* by edge descending before passing them
        so the highest-edge signal wins.
        """
        seen: set[tuple[str, str]] = set()
        records: list[OrderRecord] = []
        for signal in signals:
            key = (signal.market_ticker, signal.side)
            if key in seen:
                decision = RiskDecision(
                    approved=False,
                    signal=signal,
                    adjusted_dollars=0.0,
                    rejection_reasons=[
                        f"DEDUP: duplicate signal for {signal.market_ticker} {signal.side}"
                    ],
                )
                record = OrderRecord(
                    order_id=self._next_id("dedup"),
                    signal=signal,
                    risk_decision=decision,
                    mode=self._mode,
                    signal_price=signal.market_price,
                    order_price=0.0,
                    status=OrderStatus.SKIPPED,
                )
                logger.info(
                    "Signal DEDUP-SKIPPED for %s %s (%s): already seen this tick",
                    signal.market_ticker,
                    signal.side,
                    signal.strategy_name,
                )
                records.append(record)
            else:
                seen.add(key)
                record = await self.execute(signal)
                records.append(record)
        await self._check_placement_divergence(records)
        return records

    async def _fire_placement_alert(self, msg: str, now_ts: float) -> None:
        """Fire a CRITICAL placement-failure alert, deduped by a shared cooldown.

        Both the fast 4xx detector and the divergence watchdog route through here
        so one incident produces one alert per cooldown window, not a storm.
        Best-effort: a notifier failure must never break execution.
        """
        if self._notify_critical is None:
            return
        last = self._placement_alert_last_fired
        if last is not None and (now_ts - last) < _PLACEMENT_ALERT_COOLDOWN_S:
            return
        self._placement_alert_last_fired = now_ts
        try:
            await self._notify_critical(msg)
        except Exception:
            logger.exception("placement-failure alert notify failed")

    async def _check_placement_divergence(self, records: list[OrderRecord]) -> None:
        """Watchdog #2: alert when orders are approved but none place for
        ``_PLACEMENT_DIVERGENCE_TICKS`` consecutive live ticks.

        Cause-agnostic — catches a deprecated endpoint, expired auth, an
        exchange-side reject, or an execution bug, including modes #1 can't
        classify. Only meaningful in live mode (shadow/paper never place).
        """
        if self._mode != ExecutionMode.LIVE:
            return
        attempted, placed_ok = _batch_attempt_counts(records)
        if attempted == 0:
            return  # no placement intent this tick — uninformative, leave counter
        if placed_ok > 0:
            self._consecutive_failed_place_ticks = 0
            return
        self._consecutive_failed_place_ticks += 1
        if self._consecutive_failed_place_ticks >= _PLACEMENT_DIVERGENCE_TICKS:
            await self._fire_placement_alert(
                f"🚨 EXECUTION BROKEN: {self._consecutive_failed_place_ticks} consecutive "
                f"ticks with order(s) approved but 0 placed ({attempted} attempted this "
                f"tick, all failed). The bot has trade intent but cannot place orders.",
                time.time(),
            )

    async def cancel_all_open(self) -> tuple[list[str], bool]:
        """
        Cancel all open orders. Used by kill switch.

        Returns ``(cancelled_order_ids, truncated)``. ``truncated`` is True
        iff the resting-orders fetch hit the MAX_PAGES guard, which means
        live orders beyond the cap may NOT have been cancelled. The
        kill-switch caller in ``orchestrator.py`` is expected to surface this
        via a second ``AlertLevel.CRITICAL`` Telegram alert so the operator
        is not relying on log-grep alone during a flatten event.

        Also logs CRITICAL with a ``KILL_SWITCH_TRUNCATED:`` prefix after
        the cancellation loop finishes, with the actual cancelled count.
        """
        if self._mode == ExecutionMode.SHADOW or self._client is None:
            logger.info("Shadow mode: would cancel all open orders")
            return [], False

        cancelled: list[str] = []
        truncated = False
        try:
            orders, truncated = await asyncio.wait_for(
                self._client.iter_all_orders(status="resting"),
                timeout=30.0,
            )
            for order in orders:
                order_id = order.get("order_id", "")
                try:
                    await self._client.cancel_order(order_id)
                    cancelled.append(order_id)
                    logger.info("Cancelled order %s", order_id)
                except Exception:
                    logger.exception("Failed to cancel order %s", order_id)
            if truncated:
                logger.critical(
                    "%s: cancelled %d resting orders before MAX_PAGES cap; more may be live",
                    KILL_SWITCH_TRUNCATED_PREFIX,
                    len(cancelled),
                )
        except TimeoutError:
            logger.critical(
                "%s: iter_all_orders did not return within 30s; resting orders "
                "likely NOT cancelled — manual intervention required",
                KILL_SWITCH_TIMEOUT_PREFIX,
            )
            truncated = True
        except Exception:
            logger.exception("Failed to fetch open orders for cancellation")
        return cancelled, truncated

    @staticmethod
    def _parse_kalshi_fp(value: Any) -> float:
        # Kalshi v2 returns portfolio numeric fields as JSON strings ("5.00", "0.7400").
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _effective_ttl_seconds(ttl_override: int | None, global_ttl: int) -> int:
        """Resolve one order's TTL: per-order override wins, else the global TTL."""
        return ttl_override if ttl_override is not None else global_ttl

    @staticmethod
    def _immediate_fill_count(order_data: dict[str, Any]) -> float:
        """Contracts filled synchronously on placement, across response shapes.

        The V2 create-order response (POST /portfolio/events/orders) reports
        synchronous fills as ``fill_count``; the legacy create response used
        ``fill_count_fp``. Prefer the V2 field when present.
        """
        for key in ("fill_count", "fill_count_fp"):
            if key in order_data:
                return ExecutionEngine._parse_kalshi_fp(order_data.get(key))
        return 0.0

    @staticmethod
    def _compute_fill_price(fills: list[dict[str, Any]]) -> float | None:
        """VWAP of the side-denominated fill price in dollars; None when no valid fills."""
        total_cost = 0.0
        total_count = 0.0
        for f in fills:
            count = ExecutionEngine._parse_kalshi_fp(f.get("count_fp"))
            if count <= 0:
                continue
            side = f.get("side")
            if side == "no":
                price = ExecutionEngine._parse_kalshi_fp(f.get("no_price_dollars"))
            elif side == "yes":
                price = ExecutionEngine._parse_kalshi_fp(f.get("yes_price_dollars"))
            else:
                continue  # malformed fill — skip rather than corrupt the average
            total_cost += price * count
            total_count += count
        if total_count <= 0:
            return None
        return total_cost / total_count

    async def _apply_fills_response(self, record: OrderRecord, fills: list[dict[str, Any]]) -> None:
        """Apply a ``/portfolio/fills`` payload to ``record``.

        Sets ``is_taker_fills``, ``is_maker_fills``, ``fees_from_api``,
        ``fill_price``, ``slippage``; runs ``_reconcile_exposure`` and
        appends a fill row for any new fills.  Sets
        ``fills_payload_received = True`` only after the per-fill loop
        completes cleanly.

        Shared by ``check_fills`` (normal polling) and
        ``_recover_after_immediate_404`` (404-fallback recovery), so the
        two paths can never drift in how they consume the same payload.

        The caller is responsible for setting ``record.fill_count`` to
        the authoritative total (typically from ``get_order``'s
        ``fill_count_fp`` or summing ``count_fp`` from the payload) and
        for resetting ``record.fills_payload_received = False`` before
        the call.
        """
        if not fills:
            return

        taker_count = 0
        maker_count = 0
        fees_total_dollars = 0.0
        for f in fills:
            self._check_contract(f, SCHEMA_FILL, context="fills")
            count = round(self._parse_kalshi_fp(f.get("count_fp")))
            if f.get("is_taker") is True:
                taker_count += count
            elif f.get("is_taker") is False:
                maker_count += count
            fees_total_dollars += self._parse_kalshi_fp(f.get("fee_cost"))
        record.is_taker_fills = taker_count
        record.is_maker_fills = maker_count
        record.fees_from_api = fees_total_dollars
        record.fills_payload_received = True  # loop completed cleanly

        fill_price = self._compute_fill_price(fills)
        if fill_price is None:
            return

        record.fill_price = fill_price
        record.slippage = record.fill_price - record.order_price
        total_dollars = record.fill_count * record.fill_price
        self._reconcile_exposure(record, total_dollars)

        new_fills = record.fill_count - record.last_logged_fill_count
        if new_fills > 0:
            log_entry = self._build_log_entry(
                record,
                timestamp=datetime.now(UTC).isoformat(),
                contracts=new_fills,
                dollars=new_fills * fill_price,
                fill_price=fill_price,
            )
            await self._append_trade_log(log_entry)
            record.last_logged_fill_count = record.fill_count

    async def _recover_after_immediate_404(self, record: OrderRecord) -> None:
        """Fallback fill-recovery when ``check_fills`` 404s right after an
        immediate-fill placement.

        Kalshi's ``/portfolio/orders/<oid>`` occasionally 404s right after a
        synchronous-fill placement (stale read index, propagation lag), leaving
        the ``OrderRecord`` stuck at ``status=FILLED, fill_count=0,
        fill_price=None`` — FILLED is outside ``_PENDING_STATUSES`` so the
        polling loop never re-checks it, so ``_settle_orders`` later writes $0
        P&L, skips the settlement row, and ``/positions`` reports untracked
        exposure. ``/portfolio/fills?order_id=<oid>`` uses a different,
        non-lagging index, so a direct lookup recovers the fill. Failures here
        are swallowed; ``FillsReconciler``'s periodic sweep is the durable
        backstop.

        Race defense vs ``FillsReconciler``: the reconciler may grab
        ``_trade_log_lock`` during our ``get_fills`` await (its write phase is
        locked, its fetch phase is not) and write a fill row for this
        ``order_id`` first. We must not append a duplicate — that would
        violate the "at most one event=='fill' row per order_id" invariant and
        inflate downstream P&L aggregators. We detect this inside the lock and
        pre-set ``record.last_logged_fill_count = total_count`` so
        ``_apply_fills_response``'s append is suppressed (new_fills == 0); the
        in-memory state update still runs so the engine agrees with disk +
        Kalshi.

        Trigger condition (checked by the caller): an
        ``already_filled=True`` placement whose subsequent ``check_fills``
        left ``record.fill_count == 0``.
        """
        if self._client is None:
            return
        try:
            fills_resp = await self._client.get_fills(order_id=record.order_id)
            fills = fills_resp.get("fills", []) if isinstance(fills_resp, dict) else []
            if not fills:
                # Race window: order shows executed but fills aren't
                # visible yet. The reconciler's next paginated sweep
                # will catch them.
                return

            # Sum count_fp across the payload — authoritative because
            # Kalshi's invariant is order.fill_count_fp ==
            # sum(fill.count_fp).  Avoids relying on the get_order
            # response we never received.
            total_count = sum(round(self._parse_kalshi_fp(f.get("count_fp"))) for f in fills)
            if total_count <= 0:
                return

            record.fill_count = total_count
            record.fills_payload_received = False  # reset before apply

            # Race-defense check (see docstring): if the reconciler
            # already wrote a fill row for this order_id during our
            # get_fills await, suppress our append by pre-setting
            # last_logged_fill_count.  Held under _trade_log_lock so the
            # check is atomic with the reconciler's read+write of its
            # own synthetic-row path.
            async with self._trade_log_lock:
                if self._fill_row_exists_for_order_id(record.order_id):
                    record.last_logged_fill_count = total_count
                    logger.warning(
                        "Immediate-fill recovery: reconciler wrote fill row "
                        "first for %s — applying state-only update to avoid "
                        "duplicate trade-log row",
                        record.order_id,
                    )

            await self._apply_fills_response(record, fills)

            # Mirrors check_fills' post-fill status transition, which the 404
            # skipped: the record still carries its placement status
            # (RESTING/PENDING) for a partial immediate fill (Kalshi reports
            # "executed" only on a full fill). Without this, the Telegram
            # batch-alert filter silently drops the fill notification (status
            # ∉ {FILLED, SHADOW, PARTIAL}) — happened in production 2026-06-30
            # (KXLOWT next-day books, 1/38 fills).
            #
            # fill_count is summed from the /portfolio/fills payload above,
            # which would undercount if Kalshi paginated it; an undercount
            # degrades to PARTIAL (self-heals next tick), never a premature
            # FILLED, so the failure direction is safe.
            if record.status not in self._TERMINAL_NO_FILL_STATUSES:
                if 0 < record.fill_count < record.requested_count:
                    record.status = OrderStatus.PARTIAL
                elif record.fill_count >= record.requested_count:
                    record.status = OrderStatus.FILLED

            logger.info(
                "Immediate-fill recovery: %s populated %d fill(s) via "
                "/portfolio/fills (get_order 404 fallback)",
                record.order_id,
                record.fill_count,
            )
        except Exception:
            logger.exception(
                "Immediate-fill recovery failed for %s — FillsReconciler "
                "will backfill the log row on its next paginated sweep",
                record.order_id,
            )

    def _fill_row_exists_for_order_id(self, order_id: str) -> bool:
        """True iff ``state/trades.jsonl`` already contains an
        ``event=='fill'`` row in ``mode=='live'`` for this ``order_id``.

        Mirrors the reconciler's ``_load_tracked_live_order_ids`` semantics
        but targeted at a single oid — recovery doesn't need the full set,
        only a one-shot dedup check.  Skips torn lines (``JSONDecodeError``)
        per the same convention.

        Caller MUST hold ``self._trade_log_lock`` so the check is atomic
        with the reconciler's locked write phase.
        """
        if not self._trade_log_path.exists():
            return False
        try:
            with self._trade_log_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("mode") != "live":
                        continue
                    if not is_pnl_relevant_row(row):
                        continue
                    if row.get("order_id") == order_id:
                        return True
        except OSError:
            logger.exception(
                "Trade-log read failed during recovery dedup check for %s "
                "— treating as not-present and proceeding to append",
                order_id,
            )
        return False

    def _reconcile_exposure(self, record: OrderRecord, total_dollars: float) -> None:
        """Reconcile confirmed fill cost against tentative exposure.

        Positive delta (actual > tentative) adds the gap via ``record_fill``.
        Negative delta (actual < tentative) releases the overage via
        ``release_exposure``.  Updates ``record.last_recorded_dollars`` to
        the confirmed total so subsequent calls are idempotent.
        """
        position_key = record.signal.position_key
        delta_dollars = total_dollars - record.last_recorded_dollars
        if delta_dollars > 0:
            self._risk.record_fill(
                market_id=position_key,
                strategy_name=record.signal.strategy_name,
                dollars=delta_dollars,
            )
        elif delta_dollars < 0:
            self._risk.release_exposure(
                market_id=position_key,
                dollars=-delta_dollars,
            )
        record.last_recorded_dollars = total_dollars

    def set_on_cancel_callback(self, callback: Callable[[str], Any] | None) -> None:
        """Register a callback invoked on every 0-fill terminal cancellation.

        Callback receives the cancelled ``order_id``. Return value is ignored.
        Used by the orchestrator to wire engine cancellations into
        ``PerDayEventCap.release`` so the cap slot is released within the
        same UTC day as the cancel (the durable cancel row handles the
        cross-restart case). Default ``None`` — backtest/shadow paths can
        leave this unset.
        """
        self._on_cancel_callback = callback

    async def _handle_cancellation(self, record: OrderRecord, order_id: str) -> None:
        """Release tentative exposure for a cancelled order with no fills.

        A cancelled order that had tentative exposure but zero fills should
        have its entire reservation released.  Partial-fill cancellations
        are already reconciled by ``_reconcile_exposure`` so only zero-fill
        cancellations are handled here.

        After the exposure release, writes a durable ``cancel`` row to
        ``trades.jsonl`` and fires ``_on_cancel_callback`` so the per-day
        cap releases the slot for same-day re-attempt. The row write is
        gated on ``execution.emit_placement_rows`` and live mode (same gate
        as placement rows).

        OSError on the row write does NOT block the in-process callback —
        same-day retry continues to work; the durable record is missing
        until UTC-midnight rollover heals.
        """
        # Clear deferral flag on cancellation so prune doesn't retain an
        # indefinitely-deferred CANCELLED record (memory leak).
        record.settle_deferred = False

        # Partial-fill cancellations are reconciled by _reconcile_exposure and
        # their cap slot is legitimately held by the fill — return early.
        if record.fill_count != 0:
            return

        # Per-day cap release path FIRST so it fires regardless of exposure
        # state. The orchestrator's commit() call bound this order_id to its
        # event key when the placement succeeded; the cap slot must release
        # even if last_recorded_dollars was already cleared by a reconciler
        # race or a prior _handle_cancellation invocation. asyncio.CancelledError
        # is re-raised so SIGTERM-during-write unwinds cleanly; the next process
        # rehydrates and any unwritten cancel self-heals at UTC midnight.
        if not record.cancel_row_written:
            try:
                await self._emit_cancel_row(
                    record, timestamp=datetime.now(UTC).isoformat()
                )
                record.cancel_row_written = True
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to write cancel row for %s", order_id)
            if self._on_cancel_callback is not None:
                try:
                    self._on_cancel_callback(order_id)
                except Exception:
                    logger.exception("on_cancel_callback failed for %s", order_id)

        if record.last_recorded_dollars <= 0:
            return

        position_key = record.signal.position_key
        self._risk.release_exposure(
            market_id=position_key,
            dollars=record.last_recorded_dollars,
        )
        record.last_recorded_dollars = 0.0
        logger.info(
            "Cancelled order %s: released tentative exposure for %s",
            order_id,
            position_key,
        )

    async def check_fills(self, order_id: str) -> OrderRecord | None:
        """
        Check fill status for an order and update the record.

        Returns the updated OrderRecord, or None if not found.
        """
        if self._mode == ExecutionMode.SHADOW or self._client is None:
            return None

        record = self._orders.get(order_id)
        if record is None:
            return None

        try:
            record.fills_payload_received = False  # reset; only set True on full-loop success
            try:
                # A 404 here is the expected query-exchange read-after-write race
                # immediately after a synchronous-fill placement: the order isn't
                # on the read index yet. fill_count stays 0; the caller recovers
                # via /portfolio/fills (_recover_after_immediate_404), with the
                # FillsReconciler as the durable backstop. Logged quietly here AND
                # client-side (expected_error_statuses) so this per-order event
                # never masks a real failure. Scoped to get_order only — a non-404,
                # or a 404 from any later call (e.g. get_fills), stays loud.
                order_resp = await self._client.get_order(
                    order_id, expected_error_statuses={404}
                )
            except KalshiAPIError as exc:
                if exc.status != 404:
                    raise
                logger.debug(
                    "check_fills: order %s not yet on query index (404, "
                    "propagation lag); leaving fill_count=0 for fills-based recovery",
                    order_id,
                )
                return record
            order_data = order_resp.get("order", order_resp)
            self._check_contract(
                order_data, SCHEMA_ORDER_STATUS_CHECK, context="order_status_check"
            )

            record.fill_count = round(self._parse_kalshi_fp(order_data.get("fill_count_fp")))
            record.status = _parse_order_status(
                order_data.get("status", OrderStatus.PENDING),
                fallback=record.status,
                contract_watcher=self._contract_watcher,
            )
            record.kalshi_response = order_data

            # Clear settle_deferred when check_fills observes a terminal-no-fill
            # status transition. Defense-in-depth alongside _handle_cancellation's
            # reset (which fires later when status is in _TERMINAL_NO_FILL_STATUSES).
            if record.status in self._TERMINAL_NO_FILL_STATUSES:
                record.settle_deferred = False

            if record.fill_count > 0:
                fills_resp = await self._client.get_fills(order_id=order_id)
                fills = fills_resp.get("fills", [])
                await self._apply_fills_response(record, fills)

            # Handle partial fills — don't overwrite a terminal no-fill status
            # (CANCELLED/EXPIRED) that was just set above; FILLED from the API is
            # fine to override with PARTIAL because "executed" with fill_count <
            # requested_count means liquidity was exhausted mid-fill.
            if (
                0 < record.fill_count < record.requested_count
                and record.status not in self._TERMINAL_NO_FILL_STATUSES
            ):
                record.status = OrderStatus.PARTIAL
                logger.info(
                    "Partial fill on %s: %d/%d contracts",
                    record.signal.market_ticker,
                    record.fill_count,
                    record.requested_count,
                )

            if record.status in self._TERMINAL_NO_FILL_STATUSES:
                await self._handle_cancellation(record, order_id)

        except Exception:
            logger.exception("Failed to check fills for order %s", order_id)

        return record

    async def check_all_pending_fills(
        self,
    ) -> tuple[list[OrderRecord], list[OrderRecord]]:
        """Check fill status for all pending/resting/partial orders.

        Called once per tick to update the risk manager with new fills.
        No-op in shadow mode (shadow records fills inline).

        Returns ``(updated_with_fills, zero_fill_cancellations)``. The
        cancellations list is the set of orders that left
        ``_PENDING_STATUSES`` this tick with ``fill_count == 0`` — both
        server-cancelled (Kalshi reports "canceled") and TTL-expired by
        the safety net below. The orchestrator iterates this list to
        release per-strategy ticker dedup state so a follow-up scan can
        re-consider the same market (Bug B).

        Partial-fill cancellations (``fill_count > 0`` then cancelled or
        TTL-expired) are NOT in the cancellations list because the
        strategy successfully traded the ticker — its dedup state must
        persist.
        """
        if self._mode == ExecutionMode.SHADOW or self._client is None:
            return [], []

        updated: list[OrderRecord] = []
        server_cancelled: list[OrderRecord] = []
        for order_id, record in list(self._orders.items()):
            if record.status not in self._PENDING_STATUSES:
                continue
            result = await self.check_fills(order_id)
            if result is None:
                continue
            if result.fill_count > 0:
                updated.append(result)
            elif result.status in self._TERMINAL_NO_FILL_STATUSES:
                server_cancelled.append(result)

        if updated:
            logger.info(
                "Fill check: %d order(s) updated, total exposure $%.2f",
                len(updated),
                self._risk.total_exposure,
            )

        expired = await self._expire_stale_orders()

        zero_fill = server_cancelled + [r for r in expired if r.fill_count == 0]
        return updated, zero_fill

    async def _expire_stale_orders(self) -> list[OrderRecord]:
        """Cancel orders that have exceeded the configured TTL.

        Client-side safety net: even when Kalshi's server-side ``expiration_ts``
        handles cancellation, this catches delayed status updates or orders placed
        before the feature was deployed (no ``expiration_ts`` set).

        TTL is resolved per order via
        ``_effective_ttl_seconds(record.ttl_seconds_override, global_ttl)``: an
        order carrying a positive ``ttl_seconds_override`` (e.g. set from a
        favorite_bias ``resting_ttl_seconds`` config) rests for that per-order
        TTL and will still expire even when the global
        ``execution.order_ttl_seconds`` is ``<= 0``.
        """
        global_ttl = self._settings.execution.order_ttl_seconds
        if self._client is None:
            return []

        now = datetime.now(UTC)
        expired: list[OrderRecord] = []

        for order_id, record in list(self._orders.items()):
            if record.status not in self._PENDING_STATUSES:
                continue
            rec_ttl = self._effective_ttl_seconds(record.ttl_seconds_override, global_ttl)
            if rec_ttl <= 0:
                continue  # TTL disabled for this order
            if now - record.timestamp < timedelta(seconds=rec_ttl):
                continue

            logger.info(
                "TTL expired for order %s (%s %s, age=%s, ttl=%ds)",
                order_id,
                record.signal.side,
                record.signal.market_ticker,
                now - record.timestamp,
                rec_ttl,
            )

            try:
                await self._client.cancel_order(order_id)
            except Exception:
                logger.exception("Failed to cancel TTL-expired order %s", order_id)
                # Keep the order in _PENDING_STATUSES so the next tick retries.
                # We cannot confirm the cancel succeeded, so marking EXPIRED and
                # releasing exposure would be unsafe — the order may still be live.
                continue

            # Re-poll Kalshi AFTER the cancel to absorb any fill that landed in
            # the cancel race window (TOCTOU: a fill between the pre-cancel status
            # read and the cancel taking effect). check_fills reconciles exposure
            # against the true post-cancel fill_count for every outcome — full fill
            # holds the cost, partial releases the unfilled remainder, zero is
            # handled below — replacing the old blind release that trusted a
            # possibly-stale fill_count and could mark a held position EXPIRED while
            # contracts were live at Kalshi.
            await self.check_fills(order_id)

            if record.requested_count > 0 and record.fill_count >= record.requested_count:
                # The order fully filled in the race window — it is a real
                # position, not an expiry. Leave the Kalshi-reported status and
                # the already-reconciled exposure intact; do not expire it.
                logger.warning(
                    "TTL-expired order %s filled during the cancel race window "
                    "(%d/%d contracts) — keeping position, not marking EXPIRED",
                    order_id,
                    record.fill_count,
                    record.requested_count,
                )
                continue

            if record.fill_count == 0:
                # Zero fills — release all tentative exposure, write the durable
                # cancel row, fire the per-day-cap callback. Idempotent: a no-op
                # if check_fills already ran _handle_cancellation when Kalshi
                # reported a terminal status, and still fires when Kalshi reports
                # the order non-terminal post-cancel (eventual consistency).
                await self._handle_cancellation(record, order_id)
            else:
                # Partial fill — check_fills already reconciled exposure down to
                # the filled amount (the unfilled remainder is released). This
                # branch bypasses _handle_cancellation, so clear settle_deferred.
                record.settle_deferred = False

            record.status = OrderStatus.EXPIRED
            expired.append(record)

        if expired:
            logger.info(
                "TTL expiry: %d order(s) expired, total exposure $%.2f",
                len(expired),
                self._risk.total_exposure,
            )
        return expired

    def collect_unalerted_fills(self) -> list[OrderRecord]:
        """Return tracked orders holding contracts not yet announced.

        Scans ``_orders`` for records satisfying ``is_unalerted_new_fill`` — the
        sole feed for fill notifications, covering synchronous, later-tick,
        degraded-recovery, and cancel-race fills. The caller
        (``Orchestrator._send_fill_alerts``) stamps ``last_alerted_fill_count``
        after a successful send, so a resting partial is announced once per new
        fill quantity, not once per tick.
        """
        return [r for r in self._orders.values() if is_unalerted_new_fill(r)]

    def prune_completed_orders(self) -> int:
        """Remove settled/cancelled/rejected orders to bound memory.

        Keeps the most recent ``_MAX_RETAINED_ORDERS`` non-prunable orders plus
        all orders in pending/active states.  Returns the number of records pruned.

        Call once per tick (after settlement) to prevent unbounded growth of
        ``_orders`` during multi-day continuous operation.
        """
        if len(self._orders) <= self._MAX_RETAINED_ORDERS:
            return 0

        excess = len(self._orders) - self._MAX_RETAINED_ORDERS
        prunable = [
            oid
            for oid, rec in self._orders.items()
            if rec.status in self._PRUNABLE_STATUSES and not rec.settle_deferred
        ]
        if not prunable:
            return 0

        prunable.sort(key=lambda oid: self._orders[oid].timestamp)
        to_remove = prunable[:excess]
        for oid in to_remove:
            del self._orders[oid]

        if to_remove:
            logger.info(
                "Pruned %d completed order records (%d remaining)",
                len(to_remove),
                len(self._orders),
            )
        return len(to_remove)

    async def settle_expired_positions(self) -> list[SettlementResult]:
        """
        Check held positions for expired/settled markets and release exposure.

        Without this, the position dict grows monotonically and eventually
        hits the 40% total exposure cap, halting all trading.

        - **Shadow mode**: checks ``close_time`` from signal metadata.
        - **Live/Paper mode**: polls the Kalshi API for each held market's
          status; settles positions whose market result is finalized.

        Returns list of SettlementResult objects for settled positions.
        """
        if self._mode == ExecutionMode.SHADOW:
            return await self._settle_shadow()
        return await self._settle_live()

    async def _settle_shadow(self) -> list[SettlementResult]:
        """Settle shadow positions whose close_time has passed.

        Settlement rows are written to the trade log *before* any risk
        manager mutation, so a crash mid-sweep leaves the log in a
        "closed" state: on next replay the broker drops the positions
        (safe), rather than resurrecting closed ones (unsafe).  All rows
        for one sweep are written in a single ``open()`` so the partial
        state is either "all settled" or "some prefix settled."
        """
        now = datetime.now(UTC)
        to_settle: list[OrderRecord] = []

        for record in list(self._orders.values()):
            if record.status != OrderStatus.SHADOW:
                continue

            close_time_str = record.signal.metadata.get("close_time", "")
            if not close_time_str:
                continue

            try:
                close_dt = datetime.fromisoformat(close_time_str)
            except (ValueError, TypeError):
                continue

            if now < close_dt:
                continue

            if record.signal.position_key not in self._risk.positions:
                record.status = OrderStatus.SETTLED
                continue

            to_settle.append(record)

        if not to_settle:
            return []

        entries = [self._build_settlement_entry(r, now=now) for r in to_settle]
        try:
            await self._append_trade_log_strict(entries)
        except OSError:
            logger.exception(
                "SHADOW SETTLE: failed to append %d settlement row(s) — "
                "skipping close_position so risk state stays consistent with the log",
                len(entries),
            )
            return []

        settled: list[SettlementResult] = []
        for record in to_settle:
            position_key = record.signal.position_key
            self._risk.close_position(position_key)
            settled.append(
                SettlementResult(
                    position_key=position_key,
                    market_ticker=record.signal.market_ticker,
                    result=SETTLEMENT_RESULT_EXPIRED,
                    pnl=0.0,
                )
            )
            record.status = OrderStatus.SETTLED
            logger.info(
                "SHADOW SETTLE: %s expired (close_time=%s) — exposure released",
                position_key,
                record.signal.metadata.get("close_time", ""),
            )

        logger.info(
            "Shadow settlement: %d position(s) released, total exposure now $%.2f",
            len(settled),
            self._risk.total_exposure,
        )
        return settled

    def _build_settlement_entry(
        self,
        record: OrderRecord,
        *,
        now: datetime,
        result: str = SETTLEMENT_RESULT_EXPIRED,
        pnl: float = 0.0,
    ) -> dict[str, Any]:
        """Build a settlement-row entry for the trade log."""
        entry = self._base_log_fields(
            record, event=TradeLogEvent.SETTLEMENT, timestamp=now.isoformat()
        )
        entry.update({"result": result, "pnl": pnl})
        return entry

    def hydrate_shadow_positions(self, positions: list[ShadowPosition]) -> None:
        """Register shadow positions replayed from the trade log at startup.

        Constructs minimal ``OrderRecord`` stubs so the normal
        :meth:`_settle_shadow` sweep can release positions opened by an
        earlier process.  The stubs are read only for their
        ``position_key``, ``market_ticker``, ``side``, and ``close_time``;
        pricing fields are placeholders and must not be trusted.
        """
        if self._mode is not ExecutionMode.SHADOW:
            return
        for pos in positions:
            close_time = pos.close_time or _synthesize_close_time(pos.ticker_date)
            signal = TradeSignal(
                market_id=pos.market_id,
                market_ticker=pos.market_ticker,
                model_probability=0.0,
                market_price=0.0,
                confidence=0.0,
                strategy_name=pos.strategy_name or "shadow_hydrated",
                side=pos.side,
                suggested_dollars=pos.dollars,
                suggested_contracts=1,
                metadata={"close_time": close_time} if close_time else {},
            )
            decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=pos.dollars)
            record = OrderRecord(
                order_id=self._next_id("shadow-hydrated"),
                signal=signal,
                risk_decision=decision,
                mode=self._mode,
                signal_price=signal.market_price,
                order_price=signal.market_price,
                fill_price=signal.market_price,
                fill_count=1,
                requested_count=1,
                status=OrderStatus.SHADOW,
                last_recorded_dollars=pos.dollars,
                # Already-known position replayed on restart — start it
                # already-alerted so the fill-announcement scan does not
                # re-announce a days-old shadow position as a fresh fill.
                last_alerted_fill_count=1,
            )
            self._orders[record.order_id] = record

    def hydrate_live_positions(self, positions: list[LivePosition]) -> int:
        """Register OrderRecord stubs for live/paper positions returned by Kalshi.

        Without this, _settle_live (which builds its ticker_map from self._orders)
        cannot close API-restored positions and exposure leaks until restart.
        Stubs use status=HYDRATED, fill_price=None, fill_count=0 so _settle_orders
        skips its P&L branch and only releases exposure.

        Idempotent: re-entry with the same input registers no new stubs.

        Returns the number of stubs registered.
        """
        if self._mode is ExecutionMode.SHADOW:
            return 0

        covered = {
            r.signal.position_key
            for r in self._orders.values()
            if r.status not in self._PRUNABLE_STATUSES
        }

        count = 0
        for pos in positions:
            if pos.position_key in covered:
                continue
            signal = TradeSignal(
                market_id=pos.position_key,
                market_ticker=pos.market_ticker,
                model_probability=0.0,
                market_price=0.0,
                confidence=0.0,
                strategy_name="live_hydrated",
                side=pos.side,
                suggested_dollars=pos.dollars,
                suggested_contracts=1,
                metadata={},
            )
            decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=pos.dollars)
            record = OrderRecord(
                order_id=self._next_id("live-hydrated"),
                signal=signal,
                risk_decision=decision,
                mode=self._mode,
                signal_price=0.0,
                order_price=0.0,
                fill_price=None,
                fill_count=0,
                requested_count=0,
                status=OrderStatus.HYDRATED,
                last_recorded_dollars=pos.dollars,
            )
            self._orders[record.order_id] = record
            covered.add(pos.position_key)
            count += 1

        if count > 0:
            logger.info("LIVE HYDRATE: registered %d orphan stub(s)", count)
        return count

    def restore_fills_from_log(self) -> int:
        """Populate display-only fill metadata on HYDRATED stubs from state/trades.jsonl.

        Called once at startup after hydrate_live_positions(). Scans the trade log
        for fill rows and aggregates per ticker (VWAP fill_price, total contracts).
        For each HYDRATED stub with a matching ticker, sets ``restored_fill_info``
        so the Telegram /positions formatter can show real fill data instead of
        "original fill data not available".

        Settlement and P&L code are NOT affected — only ``restored_fill_info`` is
        written, leaving fill_price=None, fill_count=0, status=HYDRATED intact so
        the hydrated-stub settlement branch in ``_settle_orders`` fires correctly.

        Safe to call multiple times — only touches HYDRATED records.
        Returns the number of stubs annotated.
        """
        if self._mode is ExecutionMode.SHADOW:
            return 0

        # Aggregate fill rows per ticker: (total_dollars, total_contracts)
        agg: dict[str, tuple[float, int]] = {}
        for row in iter_jsonl(self._trade_log_path):
            if not is_pnl_relevant_row(row):
                continue
            ticker = str(row.get("position_key") or row.get("market_ticker") or "")
            if not ticker:
                continue
            fill_price = row.get("fill_price")
            contracts = row.get("contracts", 0)
            if fill_price is None or not contracts:
                continue
            try:
                fill_price_f = float(fill_price)
                contracts_i = int(contracts)
                if contracts_i <= 0 or not (0.0 < fill_price_f <= 1.0):
                    continue
            except (ValueError, TypeError):
                continue
            prev_dollars, prev_count = agg.get(ticker, (0.0, 0))
            agg[ticker] = (prev_dollars + fill_price_f * contracts_i, prev_count + contracts_i)

        annotated = 0
        for record in self._orders.values():
            if record.status != OrderStatus.HYDRATED:
                continue
            ticker = record.signal.market_ticker
            if ticker not in agg:
                continue
            total_dollars, total_contracts = agg[ticker]
            if total_contracts == 0:
                continue
            record.restored_fill_info = (total_dollars / total_contracts, total_contracts)
            annotated += 1

        if annotated > 0:
            logger.info(
                "LIVE HYDRATE: restored fill display data for %d stub(s) from trade log",
                annotated,
            )
        return annotated

    async def _settle_live(self) -> list[SettlementResult]:
        """Poll Kalshi for settled markets and release exposure."""
        if self._client is None:
            return []

        position_keys = set(self._risk.positions.keys())
        if not position_keys:
            return []

        ticker_map: dict[str, str] = {}
        for record in self._orders.values():
            # EXTERNALLY_CLOSED excluded so settle doesn't chase a position
            # already closed via manual UI.
            if record.status in (
                OrderStatus.REJECTED,
                OrderStatus.SKIPPED,
                OrderStatus.ERROR,
                OrderStatus.EXTERNALLY_CLOSED,
            ):
                continue
            pk = record.signal.position_key
            if pk in position_keys:
                ticker_map[pk] = record.signal.market_ticker

        settled: list[SettlementResult] = []
        for position_key, ticker in ticker_map.items():
            # Backward race — ticker_map was snapshotted before the await above;
            # an interleaved reconciler tick may have popped this position via
            # _apply_cash_out. Skip to avoid double-booking settlement P&L on
            # top of the cash-out's realized P&L.
            if position_key not in self._risk.positions:
                logger.info(
                    "_settle_live: position_key=%s already closed by concurrent "
                    "cash-out apply; skipping settlement quadruple to avoid "
                    "double-book",
                    position_key,
                )
                continue
            try:
                resp = await self._client.get_market(ticker)
                market = resp.get("market", resp)
                self._check_contract(market, SCHEMA_MARKET_SETTLEMENT, context="settlement")
                status = market.get("status", "")
                result = market.get("result", "")

                # Kalshi market statuses: "open", "closed", "settled"
                if status != "settled" and not result:
                    continue

                # Single pass: compute P&L (total + per-strategy) + all_settled.
                # Any deferred record blocks the whole position from closing this tick.
                self._fee_fallback_records_this_tick.clear()
                total_pnl, strategy_pnls, all_settled, per_record_pnl = self._settle_orders(
                    position_key, result
                )
                if not all_settled:
                    # At least one record deferred — retry next tick. No log row,
                    # no risk-state mutation, exposure held.
                    continue

                # Write settlement rows to trades.jsonl before mutating risk
                # state. Skip hydrated stubs (fill_price is None or fill_count <= 0)
                # — no useful info to record. Skip order_ids already in the
                # dedup-token set (crash-after-write recovery: row is already on
                # disk and the in-memory record was already marked SETTLED by
                # _settle_orders).
                already_settled = await self._load_settled_ids()
                now = datetime.now(UTC)
                entries: list[dict[str, Any]] = []
                new_ids: set[str] = set()
                for record, rec_pnl in per_record_pnl:
                    if record.fill_price is None or record.fill_count <= 0:
                        continue
                    if record.order_id in already_settled:
                        continue
                    entries.append(
                        self._build_settlement_entry(record, now=now, result=result, pnl=rec_pnl)
                    )
                    new_ids.add(record.order_id)

                if entries:
                    try:
                        await self._append_trade_log_strict(entries)
                    except OSError:
                        logger.exception(
                            "LIVE SETTLE: failed to append %d settlement row(s) "
                            "for %s — skipping close_position so log+state stay "
                            "consistent",
                            len(entries),
                            position_key,
                        )
                        continue
                    try:
                        await self._persist_settled_ids(new_ids)
                    except OSError:
                        # Trade-log row is on disk; dedup file failed.
                        # Recovery rebuilds dedup file from trade log on next
                        # restart. Log and proceed with close_position.
                        logger.exception(
                            "LIVE SETTLE: trade-log row written but settled_ids "
                            "persist failed for %s — recovery on next start will "
                            "rebuild from trade log",
                            position_key,
                        )

                if self._fee_fallback_records_this_tick and self._notify_critical:
                    n = len(self._fee_fallback_records_this_tick)
                    msg = (
                        f"SETTLE FEE FALLBACK on {position_key}: {n} record(s) "
                        f"settled with recomputed fee. Reconcile manually."
                    )
                    asyncio.create_task(self._notify_critical(msg))

                self._risk.close_position(position_key, pnl=total_pnl)
                settled.append(
                    SettlementResult(
                        position_key=position_key,
                        market_ticker=ticker,
                        result=result,
                        pnl=total_pnl,
                    )
                )

                for sname, spnl in strategy_pnls.items():
                    self._risk.record_strategy_trade(sname, spnl, won=(spnl > 0))

                logger.info(
                    "LIVE SETTLE: %s result=%s pnl=%.2f — exposure released",
                    position_key,
                    result,
                    total_pnl,
                )
            except Exception:
                logger.exception("Failed to check settlement for %s", ticker)

        # Aggregate-deferral alert.
        cfg = self._settings.execution
        deferred = [
            r for r in self._orders.values() if r.settle_deferred and r.settle_defer_count >= 3
        ]
        deferred_dollars = sum(r.last_recorded_dollars for r in deferred)
        active = (
            len(deferred) >= cfg.deferral_aggregate_min_count
            and deferred_dollars >= cfg.deferral_aggregate_min_dollars
        )
        now_ts = time.time()
        should_fire = False
        if active and not self._deferral_alert_active_last_tick:
            should_fire = True  # rising edge
        elif active and self._deferral_alert_last_fired is not None:
            elapsed = now_ts - self._deferral_alert_last_fired
            if elapsed >= cfg.deferral_aggregate_cooldown_seconds:
                should_fire = True  # cooldown elapsed, state still active
        if should_fire:
            msg = (
                f"3+ positions deferred ≥3 ticks; total deferred exposure "
                f"${deferred_dollars:.2f}. Possible Kalshi 429 storm."
            )
            logger.warning(msg)
            if self._notify_warning is not None:
                asyncio.create_task(self._notify_warning(msg))
            self._deferral_alert_last_fired = now_ts
        self._deferral_alert_active_last_tick = active

        if settled:
            logger.info(
                "Settlement: %d position(s) released, total exposure now $%.2f",
                len(settled),
                self._risk.total_exposure,
            )
        return settled

    def _settle_orders(
        self, position_key: str, result: str
    ) -> tuple[float, dict[str, float], bool, list[tuple[OrderRecord, float]]]:
        """Compute P&L and settle records that have consistent fee data.

        Returns ``(total_pnl, strategy_pnls, all_settled, per_record_pnl)``
        where ``all_settled`` is True iff every record for the position either
        settled (authoritative or fallback) or was skipped as hydrated/unfilled
        this tick — i.e. no record was deferred.  ``per_record_pnl`` is a list
        of ``(record, rec_pnl)`` for every record that actually settled this
        tick (authoritative, fallback, or hydrated stub); deferred records are
        NOT included.
        """
        total_pnl = 0.0
        strategy_pnls: dict[str, float] = {}
        all_settled = True
        per_record_pnl: list[tuple[OrderRecord, float]] = []
        max_defer = self._settings.execution.max_settle_defer_ticks
        is_index = position_key.startswith(("KXINX", "KXNDX"))
        for record in self._orders.values():
            if record.signal.position_key != position_key:
                continue
            if record.status == OrderStatus.SETTLED:
                continue  # idempotent re-entry within the same tick

            # Hydrated stub or unfilled record — skip P&L. Status mutation
            # is safe here because there's no fee data to be raced on.
            if record.fill_price is None or record.fill_count <= 0:
                record.status = OrderStatus.SETTLED
                record.settle_deferred = False
                # Include in per_record_pnl with pnl=0.0 (consistent with
                # _settle_shadow which writes pnl=0 for shadow positions).
                per_record_pnl.append((record, 0.0))
                continue

            if record.fills_payload_received:
                # Authoritative path — fills payload received from Kalshi.
                # fees_from_api may be 0.0 for legitimate maker-rebate cases.
                fee = record.fees_from_api
                rec_pnl = binary_contract_pnl(
                    record.signal.side,
                    record.fill_price,
                    record.fill_count,
                    result,
                    total_fee=fee,
                )
                total_pnl += rec_pnl
                strategy_pnls[record.signal.strategy_name] = (
                    strategy_pnls.get(record.signal.strategy_name, 0.0) + rec_pnl
                )
                record.status = OrderStatus.SETTLED
                record.settle_deferred = False
                per_record_pnl.append((record, rec_pnl))
                continue

            # Race — fee data not yet consistent.
            record.settle_defer_count += 1
            record.settle_deferred = True
            if record.settle_defer_count <= max_defer:
                all_settled = False
                logger.warning(
                    "SETTLE DEFER: order_id=%s position=%s defer_count=%d "
                    "(fills_payload_received=%s, fill_count=%d) — retrying next tick",
                    record.order_id,
                    position_key,
                    record.settle_defer_count,
                    record.fills_payload_received,
                    record.fill_count,
                )
                # Keep prior status (FILLED/PARTIAL) — do NOT mark SETTLED.
                # Deferred records are NOT added to per_record_pnl.
                continue

            # Exhausted — fallback to conservative recompute.
            # all_settled remains True for this record (it did settle via fallback).
            fee = kalshi_order_fee(
                record.fill_price,
                record.fill_count,
                is_maker=False,
                is_index_market=is_index,
            )
            rec_pnl = binary_contract_pnl(
                record.signal.side,
                record.fill_price,
                record.fill_count,
                result,
                total_fee=fee,
            )
            total_pnl += rec_pnl
            strategy_pnls[record.signal.strategy_name] = (
                strategy_pnls.get(record.signal.strategy_name, 0.0) + rec_pnl
            )
            record.status = OrderStatus.SETTLED
            record.settle_deferred = False
            per_record_pnl.append((record, rec_pnl))
            logger.error(
                "SETTLE FEE FALLBACK: order_id=%s position=%s ticker_index=%s "
                "fill_count=%d is_taker_fills=%d is_maker_fills=%d "
                "fees_from_api=%.4f recomputed_fee=%.4f defer_count=%d",
                record.order_id,
                position_key,
                is_index,
                record.fill_count,
                record.is_taker_fills,
                record.is_maker_fills,
                record.fees_from_api,
                fee,
                record.settle_defer_count,
            )
            self._fee_fallback_records_this_tick.append(record.order_id)
        return total_pnl, strategy_pnls, all_settled, per_record_pnl

    async def _shadow_execute(
        self,
        signal: TradeSignal,
        decision: RiskDecision,
        order_price: float,
        contract_count: int,
    ) -> OrderRecord:
        """Log a trade without placing any order."""
        order_id = self._next_id("shadow")

        record = OrderRecord(
            order_id=order_id,
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.SHADOW,
            signal_price=signal.market_price,
            order_price=order_price,
            fill_price=order_price,  # side-denominated price
            fill_count=contract_count,
            requested_count=contract_count,
            status=OrderStatus.SHADOW,
        )
        self._orders[record.order_id] = record

        log_entry = self._build_log_entry(
            record,
            timestamp=record.timestamp.isoformat(),
            contracts=contract_count,
            dollars=decision.adjusted_dollars,
        )
        await self._append_trade_log(log_entry)

        # Update risk manager so position/exposure tracking works across cycles.
        # Without this, shadow mode never records fills and the risk manager
        # approves the same markets every tick.
        position_key = signal.position_key
        self._risk.record_fill(
            market_id=position_key,
            strategy_name=signal.strategy_name,
            dollars=decision.adjusted_dollars,
        )
        record.last_recorded_dollars = decision.adjusted_dollars

        logger.info(
            "SHADOW: %s %s %d contracts @ %.2f on %s (edge=%.3f, strat=%s)",
            signal.side,
            signal.market_ticker,
            contract_count,
            order_price,
            signal.market_id,
            signal.edge,
            signal.strategy_name,
        )
        return record

    def _make_error_record(
        self,
        signal: TradeSignal,
        decision: RiskDecision,
        order_price: float,
        contract_count: int,
    ) -> OrderRecord:
        """Build an OrderRecord for a failed order and store it in history."""
        record = OrderRecord(
            order_id=self._next_id("error"),
            signal=signal,
            risk_decision=decision,
            mode=self._mode,
            signal_price=signal.market_price,
            order_price=order_price,
            requested_count=contract_count,
            status=OrderStatus.ERROR,
        )
        self._orders[record.order_id] = record
        return record

    async def _live_execute(
        self,
        signal: TradeSignal,
        decision: RiskDecision,
        order_price: float,
        contract_count: int,
    ) -> OrderRecord:
        """Place a real order on Kalshi (production or sandbox)."""
        if self._client is None:
            raise RuntimeError("Cannot execute live/paper trade without a Kalshi client")

        # Idempotency guard — reject if we just placed an order for the same
        # market+side within the dedup window (protects against timeout retries).
        dedup_key = (signal.market_ticker, signal.side)
        now = time.monotonic()

        # Secondary guard: scan order history for any still-pending/partial/resting
        # order for the same (market_ticker, side) pair.  _recent_orders only covers
        # the last ORDER_DEDUP_WINDOW seconds; an order that rested longer than that
        # window but hasn't been filled or cancelled yet would slip through otherwise.
        for existing in self._orders.values():
            if (
                existing.status in self._PENDING_STATUSES
                and existing.signal.market_ticker == signal.market_ticker
                and existing.signal.side == signal.side
            ):
                logger.warning(
                    "Order dedup (pending scan): rejecting %s %s — order %s is still %s",
                    signal.side,
                    signal.market_ticker,
                    existing.order_id,
                    existing.status.value,
                )
                return self._make_error_record(signal, decision, order_price, contract_count)

        # Evict stale entries to bound memory
        self._recent_orders = {
            k: ts for k, ts in self._recent_orders.items() if now - ts < self._ORDER_DEDUP_WINDOW
        }
        last_submit = self._recent_orders.get(dedup_key, 0.0)
        if now - last_submit < self._ORDER_DEDUP_WINDOW:
            logger.warning(
                "Order dedup: rejecting duplicate %s %s (last submitted %.1fs ago)",
                signal.side,
                signal.market_ticker,
                now - last_submit,
            )
            return self._make_error_record(signal, decision, order_price, contract_count)

        # Convert price to cents using deterministic side-aware pricing.
        # Avoids the parity-driven banker's-rounding bug where 1c-spread orders
        # randomly landed at the bid (no fill) or ask (taker) based on cents
        # parity.  See compute_limit_cents() for the rule.
        yes_bid = signal.metadata.get("yes_bid")
        yes_ask = signal.metadata.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            # No book metadata -- deterministic truncate-toward-bid fallback.
            price_cents = int(order_price * 100)
        else:
            if signal.side == Side.YES:
                side_bid_cents = int(round(yes_bid * 100))
                side_ask_cents = int(round(yes_ask * 100))
            else:
                # NO bid = 1 - yes_ask; NO ask = 1 - yes_bid.
                side_bid_cents = int(round((1.0 - yes_ask) * 100))
                side_ask_cents = int(round((1.0 - yes_bid) * 100))
            taker = bool(signal.metadata.get("execution_taker"))
            # Side-denominated limit from a strategy that walked the book
            # (favorite_bias max_book_walk_cents); replaces the frozen top ask.
            limit_price = signal.metadata.get("limit_price")
            if taker and limit_price:
                side_ask_cents = int(round(float(limit_price) * 100))
            price_cents = compute_limit_cents(
                order_price_dollars=order_price,
                side_bid_cents=side_bid_cents,
                side_ask_cents=side_ask_cents,
                cross_to_ask_max_spread_cents=(
                    _TAKER_ALWAYS_CROSS_CENTS
                    if taker
                    else self._settings.execution.cross_to_ask_max_spread_cents
                ),
            )

        # Server-side order expiration (Kalshi auto-cancels after TTL).
        # favorite_bias may request a longer resting TTL via signal metadata so a
        # taker order rests at the frozen signal-time ask.
        raw_ttl_override = signal.metadata.get("resting_ttl_seconds")
        ttl_override = int(raw_ttl_override) if raw_ttl_override is not None else None
        ttl = self._effective_ttl_seconds(ttl_override, self._settings.execution.order_ttl_seconds)
        expiration_ts = int(time.time()) + ttl if ttl > 0 else None

        try:
            resp = await self._client.place_order(
                ticker=signal.market_ticker,
                side=signal.side.lower(),
                action="buy",
                order_type="limit",
                count=contract_count,
                yes_price=price_cents if signal.side == Side.YES else None,
                no_price=price_cents if signal.side == Side.NO else None,
                expiration_ts=expiration_ts,
            )
            order_data = resp.get("order", resp)
            self._check_contract(order_data, SCHEMA_ORDER_RESPONSE, context="order_placement")
            order_id = order_data.get("order_id", "unknown")

            record = OrderRecord(
                order_id=order_id,
                signal=signal,
                risk_decision=decision,
                mode=self._mode,
                signal_price=signal.market_price,
                order_price=order_price,
                requested_count=contract_count,
                status=_parse_order_status(
                    order_data.get("status", OrderStatus.PENDING),
                    contract_watcher=self._contract_watcher,
                ),
                kalshi_response=order_data,
                ttl_seconds_override=ttl_override,
            )
            self._orders[record.order_id] = record

            # Cancellation-race reconciliation: a prior process may have placed this
            # order on Kalshi but crashed before recording it; on restart hydration
            # created a stub for the same position_key. Releasing the stub's exposure
            # first preserves the invariant that record_fill(...) below produces the
            # same _positions[key] value as a clean placement would.
            position_key = signal.position_key
            for stub in list(self._orders.values()):
                if stub.status is OrderStatus.HYDRATED and stub.signal.position_key == position_key:
                    self._risk.release_exposure(position_key, stub.last_recorded_dollars)
                    stub.status = OrderStatus.CANCELLED
                    logger.warning(
                        "Reconciled HYDRATED stub %s ($%.2f) on placement of %s — "
                        "likely orphan from prior cancellation",
                        stub.order_id,
                        stub.last_recorded_dollars,
                        record.order_id,
                    )

            # Tentative exposure — prevents duplicate signals before check_fills.
            tentative_dollars = contract_count * (price_cents / 100.0)
            self._risk.record_fill(
                market_id=position_key,
                strategy_name=signal.strategy_name,
                dollars=tentative_dollars,
            )
            record.last_recorded_dollars = tentative_dollars
            self._recent_orders[dedup_key] = now

            logger.info(
                "%s: %s %s %d contracts @ %d¢ on %s (order_id=%s, tentative=$%.2f)",
                self._mode.value.upper(),
                signal.side,
                signal.market_ticker,
                contract_count,
                price_cents,
                signal.market_id,
                order_id,
                tentative_dollars,
            )

            # Emit placement row before any immediate-fill GET so the order_id
            # is visible to the trade log even if check_fills 404s.
            await self._emit_placement_row(record, timestamp=datetime.now(UTC).isoformat())

            # Immediate-fill bookkeeping. Kalshi can fill an order synchronously
            # when the placement is a taker; the response carries
            # ``status="executed"`` (mapped to FILLED by ``_parse_order_status``)
            # and/or a non-zero ``filled_count``. The polling loop
            # (``check_all_pending_fills``) skips records once they leave
            # ``_PENDING_STATUSES``, so unless we drive the post-fill path here
            # the record never gets a fill_price, exposure stays tentative, and
            # no trade-log row is written. Re-uses ``check_fills`` so the
            # polling and immediate-fill paths share one implementation;
            # idempotency is provided by ``record.last_recorded_dollars`` and
            # ``record.last_logged_fill_count``.
            already_filled = (
                record.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)
                or self._immediate_fill_count(order_data) > 0
            )
            if already_filled:
                try:
                    await self.check_fills(record.order_id)
                except Exception:
                    # check_fills swallows its own exceptions; kept
                    # defensively in case that contract changes.
                    logger.exception("Immediate-fill check_fills failed for %s", record.order_id)
                # Kalshi's /portfolio/orders/<oid> can 404 immediately
                # after a synchronous-fill placement (query-exchange
                # propagation lag).  check_fills swallows the error,
                # leaving fill_count == 0.  Fall back to /portfolio/fills
                # to recover; see _recover_after_immediate_404 for the
                # full rationale.
                if record.fill_count == 0:
                    await self._recover_after_immediate_404(record)
            return record

        except OrderValidationError:
            logger.critical(
                "INVALID ORDER PARAMS for %s — this is a bug in the execution pipeline",
                signal.market_ticker,
                exc_info=True,
            )
            return self._make_error_record(signal, decision, order_price, contract_count)
        except Exception as exc:
            logger.exception("Order placement failed for %s", signal.market_ticker)
            # Detector #1: a non-transient (4xx) rejection means placements are
            # broken now — escalate immediately (deduped) rather than bleed
            # silent ERROR records like the 2026-06-19 V1-deprecation outage.
            if _is_non_transient_placement_error(exc):
                status = getattr(exc, "status", "?")
                body = getattr(exc, "body", {})
                code = ""
                if isinstance(body, dict):
                    err = body.get("error")
                    code = (err.get("code") if isinstance(err, dict) else None) or ""
                await self._fire_placement_alert(
                    f"🚨 ORDER REJECTED (HTTP {status}) for {signal.market_ticker}: "
                    f"{code or exc}. Non-transient — order placement is FAILING.",
                    time.time(),
                )
            return self._make_error_record(signal, decision, order_price, contract_count)

    def _base_log_fields(
        self, record: OrderRecord, *, event: TradeLogEvent, timestamp: str
    ) -> dict[str, Any]:
        """Fields shared by every row type (fill, settlement, placement)."""
        return {
            "event": event.value,
            "order_id": record.order_id,
            "timestamp": timestamp,
            "market_id": record.signal.market_id,
            "market_ticker": record.signal.market_ticker,
            "position_key": record.signal.position_key,
            "strategy": record.signal.strategy_name,
            "side": record.signal.side,
            "mode": self._mode.value,
            # intended = Kelly target the strategy asked for; placed = what we
            # actually ordered (after risk/dollar/max_contracts caps). The fill
            # row's `contracts` carries the filled count, so filled/placed exposes
            # book/TTL shortfall and placed/intended the pre-place shrink.
            "intended_contracts": record.signal.suggested_contracts,
            "placed_contracts": record.requested_count,
        }

    async def _emit_placement_row(
        self,
        record: OrderRecord,
        *,
        timestamp: str,
    ) -> None:
        """Write a placement-event row immediately after a successful place_order.

        Two load-bearing purposes:
        1. Existence proof for the order_id even if check_fills 404s.
        2. Strategy-attribution side-channel — durable record of
           (order_id -> strategy) for synthetic-row attribution after restart.

        Gated on execution.emit_placement_rows. No-op in shadow/paper modes.
        """
        if not self._emit_placement_rows:
            return
        if self._mode != ExecutionMode.LIVE:
            return

        entry = self._base_log_fields(record, event=TradeLogEvent.PLACEMENT, timestamp=timestamp)
        # 'action' lets classify_fill distinguish bot-placed sells (future)
        # from manual sells (today). The bot only buys today, so this is
        # unconditionally "buy" for now.
        entry["action"] = "buy"
        # Placement rows omit fill-time fields — those are written by the
        # fill-event row when check_fills observes the fill.
        entry["fill_price"] = None
        entry["contracts"] = None
        entry["dollars"] = None
        entry["is_taker_fills"] = None
        entry["is_maker_fills"] = None
        entry["fees_from_api"] = None
        await self._append_trade_log(entry)

    async def _emit_cancel_row(
        self,
        record: OrderRecord,
        *,
        timestamp: str,
    ) -> None:
        """Write a cancel-event row after a 0-fill terminal cancellation.

        Mirrors ``_emit_placement_row`` in scope:
          - Gated on ``execution.emit_placement_rows``.
          - Live mode only.
          - Reuses ``_base_log_fields`` for schema parity with placement /
            fill rows.

        Carries ``record.signal.metadata`` (city, market_type, forecast_date)
        so ``per_day_event_cap.hydrate_per_day_state_from_trades_jsonl`` can
        derive the event key directly without falling back to ticker parsing.
        """
        if not self._emit_placement_rows:
            return
        if self._mode != ExecutionMode.LIVE:
            return

        entry = self._base_log_fields(record, event=TradeLogEvent.CANCEL, timestamp=timestamp)
        entry["metadata"] = dict(record.signal.metadata or {})
        await self._append_trade_log(entry)

    @staticmethod
    def _finite_or_none(v: float) -> float | None:
        """Return v if finite, else None — null in the log means
        'not set at fill time' (hydrated stubs)."""
        return v if math.isfinite(v) else None

    def _build_log_entry(
        self,
        record: OrderRecord,
        *,
        timestamp: str,
        contracts: int,
        dollars: float,
        fill_price: float | None = None,
    ) -> dict[str, Any]:
        """Build a fill-row entry for the trade log."""
        entry = self._base_log_fields(record, event=TradeLogEvent.FILL, timestamp=timestamp)
        entry.update(
            {
                "model_probability": record.signal.model_probability,
                "market_price": record.signal.market_price,
                "mid_at_signal": self._finite_or_none(record.signal.mid_at_signal),
                "exec_price": self._finite_or_none(record.signal.exec_price),
                "edge": record.signal.edge,
                "confidence": record.signal.confidence,
                "dollars": dollars,
                "contracts": contracts,
                "order_price": record.order_price,
                "is_taker_fills": record.is_taker_fills,
                "is_maker_fills": record.is_maker_fills,
                "fees_from_api": round(record.fees_from_api, 4),
                "metadata": dict(record.signal.metadata),
            }
        )

        # seconds_since_placement separates first-touch fills from patient fills;
        # fill_price_minus_signal_ask measures effective vs modeled (frozen-ask) price.
        # Additive fields, .get()-safe on legacy rows.
        secs: float | None
        try:
            secs = max(0.0, (datetime.fromisoformat(timestamp) - record.timestamp).total_seconds())
        except (ValueError, TypeError):
            secs = None
        entry["seconds_since_placement"] = round(secs, 1) if secs is not None else None

        signal_ask = self._finite_or_none(record.signal.exec_price)
        if fill_price is not None and signal_ask is not None:
            entry["fill_price_minus_signal_ask"] = round(fill_price - signal_ask, 4)
        else:
            entry["fill_price_minus_signal_ask"] = None

        if fill_price is not None:
            entry["fill_price"] = fill_price
        return entry

    async def _append_trade_log(self, entry: dict[str, Any]) -> None:
        """Append a single JSON line; swallow OSError (fire-and-forget)."""
        async with self._trade_log_lock:
            try:
                await asyncio.to_thread(self._write_log_lines, [entry], False)
            except OSError:
                logger.exception("Failed to write trade log")

    async def _append_trade_log_strict(self, entries: list[dict[str, Any]]) -> None:
        """Append one or more JSON lines atomically; raise on OSError.

        Used by settlement sweeps where the caller has a consistency
        invariant (downstream risk-manager mutation) that depends on the
        write landing.  All entries are written under a single ``open()``
        so a mid-batch crash leaves a smaller, still-coherent log. Pairs
        ``flush`` + ``fsync`` so the bytes survive a power loss / OOM kill
        — the strict path is the source of truth for shadow-position
        replay on restart, and must not silently diverge from the
        atomically-persisted risk_state.json.
        """
        if not entries:
            return
        async with self._trade_log_lock:
            await asyncio.to_thread(self._write_log_lines, entries, True)

    def _write_log_lines(self, entries: list[dict[str, Any]], sync: bool) -> None:
        with open(self._trade_log_path, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")
            if sync:
                f.flush()
                os.fsync(f.fileno())

    async def _load_settled_ids(self) -> set[str]:
        """Load the dedup-token set from disk.

        Returns empty set if file doesn't exist (first run).
        On JSON corruption, logs WARNING and rebuilds via full-file scan
        of trades.jsonl for event=settlement rows.
        """
        async with self._settled_ids_lock:
            if not self._settled_ids_path.exists():
                return set()
            try:
                content = await asyncio.to_thread(self._settled_ids_path.read_text)
                data = json.loads(content)
                if not isinstance(data, list):
                    raise json.JSONDecodeError("expected list", content, 0)
                return {str(x) for x in data}
            except json.JSONDecodeError:
                logger.warning("settled_order_ids.json corrupt — rebuilding from trades.jsonl")
                rebuilt = await self._rebuild_settled_ids_from_trade_log()
                await self._persist_settled_ids_locked(rebuilt)
                return rebuilt

    async def _persist_settled_ids(self, new_ids: set[str]) -> None:
        """Atomically merge new_ids into the dedup-token file."""
        async with self._settled_ids_lock:
            existing: set[str] = set()
            if self._settled_ids_path.exists():
                try:
                    content = await asyncio.to_thread(self._settled_ids_path.read_text)
                    data = json.loads(content)
                    if isinstance(data, list):
                        existing = {str(x) for x in data}
                except json.JSONDecodeError:
                    logger.warning("settled_order_ids.json corrupt during persist; overwriting")
            merged = existing | new_ids
            await self._persist_settled_ids_locked(merged)

    async def _persist_settled_ids_locked(self, ids: set[str]) -> None:
        """Caller MUST hold _settled_ids_lock. Atomic write via tempfile + os.replace."""

        def _write() -> None:
            self._settled_ids_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                dir=self._settled_ids_path.parent,
                prefix=".settled_ids_",
                suffix=".json",
                encoding="utf-8",
            ) as tmp:
                json.dump(sorted(ids), tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, self._settled_ids_path)

        await asyncio.to_thread(_write)

    async def _rebuild_settled_ids_from_trade_log(self) -> set[str]:
        """Full-file scan of trades.jsonl for event=settlement rows. Recovery only.

        Counts malformed lines and logs WARNING with the count if non-zero.
        """
        if not self._trade_log_path.exists():
            return set()

        def _scan() -> tuple[set[str], int]:
            local: set[str] = set()
            local_malformed = 0
            with open(self._trade_log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        local_malformed += 1
                        continue
                    if row.get("event") == TradeLogEvent.SETTLEMENT:
                        oid = row.get("order_id")
                        if oid:
                            local.add(str(oid))
            return local, local_malformed

        ids, malformed = await asyncio.to_thread(_scan)
        if malformed > 0:
            logger.warning(
                "settled_order_ids rebuild: skipped %d malformed line(s) in trades.jsonl",
                malformed,
            )
        logger.info("settled_order_ids rebuilt from trade log: %d ids", len(ids))
        return ids
