"""Fills reconciler: write-behind cache for state/trades.jsonl.

The system of record is /portfolio/fills (Kalshi). The local trade log
is treated as a cache; this reconciler runs every 5 ticks and writes
synthetic fill rows for any Kalshi-confirmed fill not already in the log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from python.common import OrderStatus
from python.data.kalshi_client import KalshiAPIError, paginate_detailed
from python.execution.cost_basis import (
    CostBasis,
    PartialBasis,
    cost_basis_for_position,
    partial_basis,
)
from python.execution.trade_log import (
    TRADE_LOG_EVENT_CASH_OUT,
    TRADE_LOG_EVENT_FILL,
    TRADE_LOG_EVENT_MANUAL_CLOSE,
    TRADE_LOG_EVENT_PLACEMENT,
)
from python.notifications.base import AlertLevel, Notifier
from python.utils.atomic_write import atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

    from python.risk.manager import RiskManager

logger = logging.getLogger(__name__)

_TERMINAL_ORDER_STATUSES: frozenset[OrderStatus] = frozenset({
    OrderStatus.SETTLED,
    OrderStatus.EXTERNALLY_CLOSED,
    OrderStatus.REJECTED,
    OrderStatus.SKIPPED,
    OrderStatus.ERROR,
})


def _contracts_from_count_fp(value: Any) -> int:
    """Convert Kalshi /portfolio/fills `count_fp` to a contract count.

    Despite the name, `count_fp` is a plain integer contract count, NOT a
    fixed-point scaled value. Verified by the existing reconciler and the
    regression test in tests/test_pull_post_mortem_data.py.

    Handles None, str, and numeric inputs uniformly so callers don't need
    defensive cleanup.
    """
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class FillAggregate:
    """Per-order reduction of Kalshi /portfolio/fills rows.

    Dollar and fee totals are unrounded — each caller applies its own
    ``round(..., 4)`` when writing its trade-log row.

    ``taker_contracts``/``maker_contracts`` are SUMS OF CONTRACTS, not fill
    counts, matching the engine's ``is_taker_fills`` semantic; a fill whose
    ``is_taker`` is absent or None counts toward neither.

    NOT total: a present-but-uncoercible ``count_fp`` or ``fee_cost``, a
    non-numeric price string, or an explicit ``"side": None`` all raise. Only
    a missing/``None`` price column is safe. Deliberately does NOT reuse the
    lenient ``_contracts_from_count_fp``; that leniency is needed only in
    ``_apply_manual_close``, above its ``PARTIAL_HELD`` guard.

    Kalshi returns None for the off-side price column, so the price read is
    ``float(x or 0)``, not ``.get(k, 0)``.
    """

    contracts: int
    dollars: float
    fees: float
    taker_contracts: int
    maker_contracts: int
    earliest_ts: str
    side: str

    @property
    def vwap(self) -> float:
        """Volume-weighted average fill price, 0.0 for a zero-contract order."""
        return self.dollars / self.contracts if self.contracts else 0.0


def _aggregate_fills(kalshi_fills: list[dict[str, Any]]) -> FillAggregate:
    """Reduce one order's Kalshi fills. See FillAggregate for the raise contract."""
    contracts = 0
    dollars = 0.0
    fees = 0.0
    taker_contracts = 0
    maker_contracts = 0
    earliest_ts = ""

    for fill in kalshi_fills:
        count = int(float(fill.get("count_fp", 0)))
        fill_side = fill.get("side", "").upper()
        price = float(
            (fill.get("yes_price_dollars") or 0)
            if fill_side == "YES"
            else (fill.get("no_price_dollars") or 0)
        )
        contracts += count
        dollars += count * price
        fees += float(fill.get("fee_cost", 0))
        if fill.get("is_taker") is True:
            taker_contracts += count
        elif fill.get("is_taker") is False:
            maker_contracts += count
        ts = fill.get("created_time", "")
        if ts and (not earliest_ts or ts < earliest_ts):
            earliest_ts = ts

    side = kalshi_fills[0].get("side", "").upper() if kalshi_fills else ""
    return FillAggregate(
        contracts=contracts,
        dollars=dollars,
        fees=fees,
        taker_contracts=taker_contracts,
        maker_contracts=maker_contracts,
        earliest_ts=earliest_ts,
        side=side,
    )


# Max fill lines in one backfill alert before truncating with "…and K more".
# candidate_oids is already capped at max_fills_per_tick (default 50), so this
# is a defense-in-depth guard against Telegram's ~4096-char message ceiling if
# that cap is ever raised.
_BACKFILL_ALERT_MAX_LINES = 40


def _format_backfill_fill_alert(rows: list[dict[str, Any]]) -> str:
    """Batched operator alert for BUY_ENTRY fills the reconciler back-filled
    this tick.

    Pure and defensive: every field is coerced so a malformed or present-but-
    wrong-typed synthetic row cannot raise (``.get(default)`` only substitutes
    for *absent* keys — a present ``None`` or non-numeric value would otherwise
    reach ``f"${x:.2f}"`` and raise). Mirrors
    ``TelegramNotifier.send_trade_batch_alert`` line format, tagged
    ``[reconciled]`` to distinguish from live placements. Caller wraps this in
    try/except regardless (belt-and-braces).
    """
    n = len(rows)
    lines = [f"BACKFILLED FILLS [reconciled] ({n})"]
    for row in rows[:_BACKFILL_ALERT_MAX_LINES]:
        side = str(row.get("side") or "?")
        ticker = str(row.get("market_ticker") or "?")
        try:
            contracts = int(row.get("contracts") or 0)
        except (TypeError, ValueError, OverflowError):
            contracts = 0
        try:
            price = float(row.get("fill_price") or 0.0)
        except (TypeError, ValueError, OverflowError):
            price = 0.0
        meta = row.get("metadata")
        strategy = (
            str(meta.get("strategy") or "unknown")
            if isinstance(meta, dict)
            else "unknown"
        )
        lines.append(f"  {side} {ticker} {contracts}x @ ${price:.2f} | {strategy}")
    hidden = n - min(n, _BACKFILL_ALERT_MAX_LINES)
    if hidden > 0:
        lines.append(f"  …and {hidden} more")
    return "\n".join(lines)


@dataclass(frozen=True)
class ClassifierContext:
    """Input to classify_fill — extensible without growing kwargs."""

    tracked_buy_oids: set[str]
    placement_index: dict[str, dict[str, str]]
    open_positions_by_ticker_side: dict[tuple[str, str], float]


@dataclass(frozen=True)
class TrackedOids:
    """Return type for _load_tracked_oids. Named fields so future close-event
    types can extend it without breaking call sites."""

    buys: set[str]
    manual_closes: set[str]
    cash_outs: set[str]


class FillKind(StrEnum):
    """Classification of one Kalshi /portfolio/fills record.

    Pure function of the fill + tracked-oid sets (no application state).
    Apply-side concerns (dedup, orphan, partial-sell hold) live in
    _apply_manual_close.

    Dispatch order (fail-safe priority): UNKNOWN → BUY_ENTRY →
    SELL_BOT_ANOMALY (matches BEFORE placement-action check; anomaly
    fails-safe, placement check fails-open) → SELL_BOT →
    SELL_MANUAL_CANDIDATE fallthrough.
    """

    BUY_ENTRY = "buy_entry"
    BUY_CASH_OUT = "buy_cash_out"
    SELL_BOT = "sell_bot"
    SELL_BOT_ANOMALY = "sell_bot_anomaly"
    SELL_MANUAL_CANDIDATE = "sell_manual_candidate"
    UNKNOWN = "unknown"


class ApplyOutcome(StrEnum):
    """Outcome of one apply-path dispatch (manual-close / cash-out / shakedown).

    Mirrors the FillKind / TradeLogEvent StrEnum pattern. Members compare equal
    to their string value and serialize (json.dumps / structlog) to the bare
    string, so on-disk `apply_path_outcome_counts` keys and the values asserted
    by existing tests stay byte-identical. Values are intentionally UPPERCASE
    (distinct from FillKind's lowercase values — no collision).
    """

    APPLIED = "APPLIED"
    ALREADY_CLOSED_DEDUP = "ALREADY_CLOSED_DEDUP"
    ORPHAN = "ORPHAN"
    ZERO_COST_BASIS = "ZERO_COST_BASIS"
    PARTIAL_HELD = "PARTIAL_HELD"
    EMPTY_FILLS_CASH_OUT = "EMPTY_FILLS_CASH_OUT"
    ALREADY_CLOSED_DEDUP_CASH_OUT = "ALREADY_CLOSED_DEDUP_CASH_OUT"
    RACE_LOST_TO_SETTLEMENT = "RACE_LOST_TO_SETTLEMENT"
    ZERO_COST_BASIS_CASH_OUT = "ZERO_COST_BASIS_CASH_OUT"
    INVALID_PARTIAL_CASH_OUT = "INVALID_PARTIAL_CASH_OUT"
    PNL_CAP_EXCEEDED_CASH_OUT = "PNL_CAP_EXCEEDED_CASH_OUT"
    APPLIED_FULL_CASH_OUT = "APPLIED_FULL_CASH_OUT"
    APPLIED_PARTIAL_CASH_OUT = "APPLIED_PARTIAL_CASH_OUT"
    SELL_BOT_ANOMALY = "SELL_BOT_ANOMALY"
    UNKNOWN = "UNKNOWN"
    CANARY_LIMIT_REACHED_CASH_OUT = "CANARY_LIMIT_REACHED_CASH_OUT"
    CANARY_LIMIT_REACHED = "CANARY_LIMIT_REACHED"
    BUY_ENTRY_HANDLED_ELSEWHERE = "BUY_ENTRY_HANDLED_ELSEWHERE"
    SHAKEDOWN_SUPPRESSED = "SHAKEDOWN_SUPPRESSED"
    SHAKEDOWN_ALERTED = "SHAKEDOWN_ALERTED"


def classify_fill(fill: dict[str, Any], ctx: ClassifierContext) -> FillKind:
    """Classify one fill record. Pure function.

    Dispatch order on action="buy":
      1. placement_index match → BUY_ENTRY (bot-placed always wins).
      2. (ticker, opp_side) in open_positions_by_ticker_side → BUY_CASH_OUT.
      3. Fallthrough → BUY_ENTRY (unattributed-but-legitimate; covers Day-1
         historical fills that pre-date placement-row writing).
    """
    action = fill.get("action")
    if action not in ("buy", "sell"):
        return FillKind.UNKNOWN

    if action == "buy":
        oid = fill.get("order_id", "")
        if oid in ctx.placement_index:
            return FillKind.BUY_ENTRY
        ticker = fill.get("market_ticker", "")
        side = (fill.get("side", "") or "").upper()
        opp = "NO" if side == "YES" else "YES" if side == "NO" else ""
        if opp and (ticker, opp) in ctx.open_positions_by_ticker_side:
            return FillKind.BUY_CASH_OUT
        return FillKind.BUY_ENTRY

    oid = fill.get("order_id", "")
    if oid in ctx.tracked_buy_oids:
        return FillKind.SELL_BOT_ANOMALY
    placement = ctx.placement_index.get(oid)
    if placement is not None and placement.get("action") == "sell":
        return FillKind.SELL_BOT
    return FillKind.SELL_MANUAL_CANDIDATE


STATE_FILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReconcileTick:
    """Result of one reconciler tick.

    status:
      - "clean" — tick completed, delta_count == 0
      - "warning" — tick completed with non-fatal issues (e.g., delta > 0,
        unparseable rows in log, etc.)
      - "critical" — tick completed but emitted a CRITICAL alert
        (e.g., truncation, cursor cycle abort, threshold breach)
      - "skipped" — tick did not run (reentrant, operator-disabled, etc.)
    """

    status: Literal["clean", "warning", "critical", "skipped"]
    delta_count: int
    duration_ms: int
    pages_consumed: int = 0
    # True when pagination hit max_pages_per_tick and a backlog may remain.
    truncated_at_max_pages: bool = False
    # True when seen_cursors detected a repeated cursor; the cycling page was
    # discarded to prevent row inflation (cursor-cycle defense, 2026-05-12).
    cycle_aborted_at_cursor: bool = False
    reason: str | None = None
    exc_class: str | None = None
    exc_repr: str | None = None
    order_ids_synthesised: list[str] = field(default_factory=list)
    strategy_resolution_breakdown: dict[str, int] = field(default_factory=dict)
    last_safe_ts_unix_before: int = 0
    last_safe_ts_unix_after: int = 0

    @classmethod
    def skipped(cls, reason: str) -> ReconcileTick:
        """Construct a skipped-tick result (reentrancy, operator disable, etc.)."""
        return cls(status="skipped", delta_count=0, duration_ms=0, reason=reason)


@dataclass(frozen=True)
class BackfillResult:
    """Result of an operator-driven backfill (CLI or programmatic)."""

    planned_writes: list[dict[str, Any]]
    written_count: int
    skipped_count: int
    unresolved_count: int
    aborted: bool
    abort_reason: str | None


class StrategyResolver(Protocol):
    """Resolves the originating strategy for a synthetic fill row.

    Called once per tick (with caching) for each candidate order_id.
    Uses a 3-step lookup chain: placement index, recent fills, trade log.
    """

    def __call__(
        self, order_id: str, kalshi_fill: dict[str, Any]
    ) -> tuple[str, str]:
        """Return (strategy, resolution_method).

        resolution_method is one of:
          "in_memory" — found in engine._orders
          "placement_scan" — found in a placement row
          "unresolved" — neither path found it; strategy will be "unknown"
        """
        ...


@dataclass
class _DispatchTally:
    """Mutable per-tick accumulator filled by FillsReconciler._classify_and_dispatch.

    The count dicts are logged as-is; their pre-seeded keys and insertion
    order keep funnel diffs stable.
    """

    synthetic_rows: list[dict[str, Any]] = field(default_factory=list)
    order_ids_synthesised: list[str] = field(default_factory=list)
    resolution_breakdown: dict[str, int] = field(
        default_factory=lambda: {
            "in_memory": 0,
            "placement_scan": 0,
            "unresolved": 0,
        }
    )
    classifier_decision_counts: dict[str, int] = field(
        default_factory=lambda: {
            FillKind.BUY_ENTRY.value: 0,
            FillKind.BUY_CASH_OUT.value: 0,  # pre-seed for stable funnel diffs
            FillKind.SELL_BOT.value: 0,
            FillKind.SELL_BOT_ANOMALY.value: 0,
            FillKind.SELL_MANUAL_CANDIDATE.value: 0,
            FillKind.UNKNOWN.value: 0,
        }
    )
    apply_path_outcome_counts: dict[str, int] = field(default_factory=dict)
    any_unknown: bool = False


@dataclass(frozen=True)
class _CashOutPlan:
    """Validated cash-out, ready to commit. Output of _plan_cash_out's guard phase."""

    risk: RiskManager
    hedge_side: str
    closed_side: str
    cb: CostBasis
    agg: FillAggregate
    pb: PartialBasis
    close_type: Literal["full", "partial"]
    realized_pnl: float
    contracts_residual: int


def _group_fills_by_order_id(
    fills: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_order_id: dict[str, list[dict[str, Any]]] = {}
    for fill in fills:
        oid = fill.get("order_id")
        if not oid:
            continue
        by_order_id.setdefault(oid, []).append(fill)
    return by_order_id


def _apply_latency_seconds(earliest_ts: str) -> float | None:
    """Seconds from the earliest fill timestamp to now; None if unparseable.

    Only ValueError is swallowed: a naive timestamp raises TypeError on the
    aware-minus-naive subtraction and must still propagate.
    """
    try:
        fill_dt = datetime.fromisoformat(earliest_ts.replace("Z", "+00:00"))
        return (datetime.now(UTC) - fill_dt).total_seconds()
    except ValueError:
        return None


def _warn_on_pod_attribution_split(
    risk: RiskManager, position_key: str, strategy: str
) -> None:
    authoritative = risk.strategy_for_position(position_key)
    if authoritative != "unknown" and authoritative != strategy:
        logger.warning(
            "Pod attribution split for %s: reconciler resolved "
            "strategy=%r (for circuit-breaker P&L) but risk manager "
            "attributes exposure to %r. Exposure decrements the risk "
            "manager's pod; circuit-breaker P&L credits the "
            "reconciler's. Investigate if recurring.",
            position_key,
            strategy,
            authoritative,
        )


def _remember_oid(oids: set[str], order_id: str) -> bool:
    """Add order_id to a sticky set; return True if it was already present."""
    already = order_id in oids
    oids.add(order_id)
    return already


def _build_cash_out_row(
    *,
    order_id: str,
    market_ticker: str,
    position_key: str,
    plan: _CashOutPlan,
    strategy: str,
    method: str,
    applied_at: str,
) -> dict[str, Any]:
    agg = plan.agg
    pb = plan.pb
    return {
        "event": TRADE_LOG_EVENT_CASH_OUT,
        "order_id": order_id,
        "market_ticker": market_ticker,
        "position_key": position_key,
        "side": plan.closed_side,  # alias for closed_side (CRIT-6 compat)
        "mode": "live",
        "timestamp": agg.earliest_ts,
        "applied_at": applied_at,
        "closed_side": plan.closed_side,
        "hedge_side": plan.hedge_side,
        "close_type": plan.close_type,
        "contracts_offset": agg.contracts,
        "contracts_residual": plan.contracts_residual,
        "hedge_price": round(agg.vwap, 4),
        "hedge_proceeds": round(agg.dollars, 4),
        "hedge_fees": round(agg.fees, 4),
        "cost_basis_offset_dollars": pb.offset_dollars,
        "cost_basis_offset_fees": pb.offset_fees,
        "cost_basis_residual_dollars": pb.residual_dollars,
        "cost_basis_residual_fees": pb.residual_fees,
        # Kalshi binary contracts pay $1.00 per offsetting contract.
        "collateral_return_estimate": round(agg.contracts * 1.00, 4),
        "realized_pnl": round(plan.realized_pnl, 4),
        "is_taker_fills": agg.taker_contracts,
        "is_maker_fills": agg.maker_contracts,
        "metadata": {
            "strategy": strategy,
            "reconciled": True,
            "reconcile_source": "fills_reconciler",
            "classification": "BUY_CASH_OUT",
            "strategy_resolution": method,
        },
    }


def _cash_out_alert_message(
    plan: _CashOutPlan, market_ticker: str, position_key: str
) -> str:
    contracts_offset = plan.agg.contracts
    hedge_vwap = plan.agg.vwap
    realized_pnl = plan.realized_pnl
    if plan.close_type == "full":
        return (
            f"🔄 Cash out reconciled: {market_ticker} pos={position_key} "
            f"[full] {contracts_offset}c offset @ {hedge_vwap:.4f} "
            f"realized_pnl=${realized_pnl:+.2f}. Exposure freed."
        )
    return (
        f"🔄 Cash out reconciled: {market_ticker} pos={position_key} "
        f"[partial: {contracts_offset} offset, {plan.contracts_residual} "
        f"contracts still held] @ {hedge_vwap:.4f} "
        f"realized_pnl=${realized_pnl:+.2f}. "
        f"Residual exposure ${plan.pb.residual_dollars:.2f}."
    )


class FillsReconciler:
    """Periodic reconciler that back-fills Kalshi-confirmed fills into state/trades.jsonl.

    Single-writer, single-reader; the orchestrator owns the only instance.
    """

    def __init__(
        self,
        *,
        state_path: Path,
        trade_log_path: Path,
        trade_log_lock: asyncio.Lock,
        kalshi_client: Any,  # KalshiClient — Any to avoid circular import
        engine: Any | None,  # ExecutionEngine — for in-memory order lookup
        notifier: Notifier | None,
        startup_safety_window_seconds: int = 24 * 3600,
        startup_max_window_seconds: int = 7 * 24 * 3600,
        eventual_consistency_window_seconds: int = 120,
        max_pages_per_tick: int = 20,
        max_fills_per_tick: int = 50,
        page_delay_ms: int = 250,
        placement_row_lookback: int = 10000,
        alert_threshold_consecutive: int = 3,
        alert_threshold_critical_delta: int = 3,
        critical_alert_cooldown_seconds: int = 3600,
        strategy_resolver: StrategyResolver | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._state_path = state_path
        self._trade_log_path = trade_log_path
        self._trade_log_lock = trade_log_lock
        self._kalshi_client = kalshi_client
        self._engine = engine
        self._notifier = notifier

        self._startup_safety_window_seconds = startup_safety_window_seconds
        self._startup_max_window_seconds = startup_max_window_seconds
        self._eventual_consistency_window_seconds = eventual_consistency_window_seconds
        self._max_pages_per_tick = max_pages_per_tick
        self._max_fills_per_tick = max_fills_per_tick
        self._page_delay_ms = page_delay_ms
        self._placement_row_lookback = placement_row_lookback
        self._alert_threshold_consecutive = alert_threshold_consecutive
        self._alert_threshold_critical_delta = alert_threshold_critical_delta
        self._critical_alert_cooldown_seconds = critical_alert_cooldown_seconds

        self._strategy_resolver: StrategyResolver = (
            strategy_resolver or self._default_strategy_resolver
        )
        self._clock = clock

        self._tick_lock = asyncio.Lock()
        self._disabled: bool = False

        # Per-tick placement_index built lazily by S10's resolver.
        # Shape: order_id -> {"strategy": str, "action": "buy" | "sell"}.
        # 'action' defaults to "buy" for legacy rows that predate the field
        # (the bot has only ever submitted buys until the manual-close feature).
        self._placement_index: dict[str, dict[str, str]] = {}

        # __init__ is sync; corrupt-state alerts are queued here and flushed
        # on the first tick() call. asyncio.create_task in __init__ would crash
        # in any sync test context (no running loop).
        self._pending_startup_alerts: list[tuple[AlertLevel, str]] = []

        # State loaded from disk (or fallback)
        self._last_safe_ts_unix: int = 0
        self._last_clean_completion_ts_unix: int = 0
        self._consecutive_failures: int = 0
        self._last_critical_alert_ts_unix: int = 0
        # Delta-path CRITICAL has its own cooldown timestamp so failure-path
        # and delta-path ladders don't suppress each other. The three critical
        # sub-modes (delta-exceeded / truncation / cursor-cycle) each get their
        # own timestamp so a truncation alert can't silence a structurally
        # different cursor-cycle alert (or vice versa) inside the cooldown window.
        self._last_delta_critical_alert_ts_unix: int = 0
        self._last_truncation_critical_alert_ts_unix: int = 0
        self._last_cycle_critical_alert_ts_unix: int = 0

        # Manual-close state. All default to "feature off / unlimited apply /
        # empty sets"; the orchestrator wires real values from settings on startup.
        self._manual_close_enabled: bool = False
        self._manual_close_apply_limit: int = 0  # 0 = unlimited
        self._manual_close_applies_this_restart: int = 0
        self._seen_anomalous_sells: set[str] = set()
        self._pending_partial_sells: dict[str, float] = {}
        self._shakedown_seen_oids: set[str] = set()
        self._shakedown_seen_oids_path: Path | None = None  # set by orchestrator
        # Persistent set of sell oids that had no entry-FILL cost basis (orphan
        # closes); prevents re-CRITICAL-alerting every tick.
        self._orphan_close_oids: set[str] = set()

        # Per-tick observability accumulators (reset at top of _tick_impl,
        # drained into the structured log at end of tick).
        self._manual_close_realized_pnl_cents_this_tick: int = 0
        self._manual_close_apply_latency_max_this_tick: float = 0.0

        # Cash-out reconciliation state. All default to "feature off / unlimited
        # apply / empty sets"; orchestrator wires real values from settings.
        self._co_apply_enabled: bool = False
        self._co_apply_limit: int = 0  # 0 = unlimited
        self._co_applies_this_restart: int = 0
        self._co_max_pnl_apply_dollars: float = 500.0
        self._shakedown_seen_oids_cash_out: set[str] = set()
        self._shakedown_seen_oids_cash_out_path: Path | None = None  # set by orchestrator

        # Five separate dedup sets — distinct failure-mode recoverability:
        self._co_race_lost_oids: set[str] = set()
        self._co_zero_basis_oids: set[str] = set()
        self._co_invalid_partial_oids: set[str] = set()
        self._co_pnl_cap_oids: set[str] = set()
        self._co_reverted_oids: set[str] = set()

        # Per-tick observability accumulators for cash-out (reset at top of _tick_impl).
        self._co_realized_pnl_cents_this_tick: int = 0
        self._co_apply_latency_max_this_tick: float = 0.0
        self._co_applies_this_tick: int = 0

        self._load_state()

    def _load_state(self) -> None:
        """Load state file or fall back to safety window."""
        now = int(self._clock())
        floor_min_ts = now - self._startup_safety_window_seconds

        if not self._state_path.exists():
            self._last_safe_ts_unix = floor_min_ts
            logger.info(
                "fills_reconciler state file missing; using safety window fallback"
            )
            return

        try:
            data = json.loads(self._state_path.read_text())
            if data.get("schema_version") != STATE_FILE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unknown schema_version: {data.get('schema_version')}"
                )
            persisted_cursor = int(data.get("last_safe_ts_unix", floor_min_ts))
            # Go back AT LEAST safety_window, more if persisted is older.
            cursor = min(persisted_cursor, floor_min_ts)
            # Cap the lookback at startup_max_window_seconds so a multi-month-old
            # persisted cursor doesn't trigger an unbounded first-tick scan.
            # Operator can use scripts/reconcile_fills.py for older windows.
            ceiling_min_ts = now - self._startup_max_window_seconds
            self._last_safe_ts_unix = max(cursor, ceiling_min_ts)
            self._last_clean_completion_ts_unix = int(
                data.get("last_clean_completion_ts_unix", 0)
            )
            self._consecutive_failures = int(data.get("consecutive_failures", 0))
            self._last_critical_alert_ts_unix = int(
                data.get("last_critical_alert_ts_unix", 0)
            )
            self._last_delta_critical_alert_ts_unix = int(
                data.get("last_delta_critical_alert_ts_unix", 0)
            )
            # Additive + defaulted: legacy state files (without these keys) load 0,
            # so no STATE_FILE_SCHEMA_VERSION bump is required.
            self._last_truncation_critical_alert_ts_unix = int(
                data.get("last_truncation_critical_alert_ts_unix", 0)
            )
            self._last_cycle_critical_alert_ts_unix = int(
                data.get("last_cycle_critical_alert_ts_unix", 0)
            )
            # Missing on legacy state files → empty set.
            self._orphan_close_oids = set(data.get("orphan_close_oids", []))
            self._co_race_lost_oids = set(data.get("co_race_lost_oids", []))
            self._co_zero_basis_oids = set(data.get("co_zero_basis_oids", []))
            self._co_invalid_partial_oids = set(data.get("co_invalid_partial_oids", []))
            self._co_pnl_cap_oids = set(data.get("co_pnl_cap_oids", []))
            self._co_reverted_oids = set(data.get("co_reverted_oids", []))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "fills_reconciler state corrupt: %s; resetting cursor", exc
            )
            self._last_safe_ts_unix = floor_min_ts
            self._consecutive_failures = 0
            self._last_critical_alert_ts_unix = 0
            self._last_delta_critical_alert_ts_unix = 0
            self._last_truncation_critical_alert_ts_unix = 0
            self._last_cycle_critical_alert_ts_unix = 0
            if self._notifier is not None:
                self._pending_startup_alerts.append(
                    (
                        AlertLevel.WARNING,
                        f"Fills reconciler state corrupt; resetting cursor "
                        f"to now − {self._startup_safety_window_seconds // 3600}h.",
                    )
                )

    async def _persist_state(self) -> None:
        """Atomic write of the reconciler state, then chmod 0600."""
        payload = {
            "schema_version": STATE_FILE_SCHEMA_VERSION,
            "last_safe_ts_unix": self._last_safe_ts_unix,
            "last_clean_completion_ts_unix": self._last_clean_completion_ts_unix,
            "consecutive_failures": self._consecutive_failures,
            "last_critical_alert_ts_unix": self._last_critical_alert_ts_unix,
            "last_delta_critical_alert_ts_unix": (
                self._last_delta_critical_alert_ts_unix
            ),
            "last_truncation_critical_alert_ts_unix": (
                self._last_truncation_critical_alert_ts_unix
            ),
            "last_cycle_critical_alert_ts_unix": (
                self._last_cycle_critical_alert_ts_unix
            ),
            # Sorted for deterministic file diffs; loaded back into a set.
            "orphan_close_oids": sorted(self._orphan_close_oids),
            # Five separate sets keep distinct cash-out failure modes
            # individually recoverable via revert_cash_out.py.
            "co_race_lost_oids": sorted(self._co_race_lost_oids),
            "co_zero_basis_oids": sorted(self._co_zero_basis_oids),
            "co_invalid_partial_oids": sorted(self._co_invalid_partial_oids),
            "co_pnl_cap_oids": sorted(self._co_pnl_cap_oids),
            "co_reverted_oids": sorted(self._co_reverted_oids),
        }
        await asyncio.to_thread(
            atomic_write_text, self._state_path, json.dumps(payload, indent=2)
        )
        await asyncio.to_thread(os.chmod, self._state_path, 0o600)

    def disable(self) -> None:
        """Disable the reconciler — next tick() returns skipped.

        Process-local; survives until orchestrator restart. Used by Telegram
        /disable_reconciler.
        """
        self._disabled = True

    def enable(self) -> None:
        """Re-arm the reconciler. Public counterpart to disable()."""
        self._disabled = False

    @property
    def is_disabled(self) -> bool:
        """Read-side of the operator-disable flag."""
        return self._disabled

    async def tick(self) -> ReconcileTick:
        """One reconciliation pass. Never raises.

        Two early-exit guards:
          1. Reentrancy: if _tick_lock is held, return skipped.
          2. Disabled flag: if _disabled, return skipped.
        Then enter the main flow via _tick_impl.
        """
        if self._tick_lock.locked():
            return ReconcileTick.skipped(reason="reentrant")

        if self._disabled:
            return ReconcileTick.skipped(reason="operator_disabled")

        async with self._tick_lock:
            return await self._tick_impl()

    async def _append_synthetic_rows(
        self, rows: list[dict[str, Any]]
    ) -> None:
        """Append synthetic fill rows to trades.jsonl.

        Caller MUST hold self._trade_log_lock — this method does not acquire
        it. Mirrors the engine's writer pattern (open append, json.dumps,
        no fsync — synthetic rows are recoverable on the next tick if lost).
        """

        def _write() -> None:
            with self._trade_log_path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")

        await asyncio.to_thread(_write)

    async def _tick_impl(self) -> ReconcileTick:
        """Main tick implementation.

        Sequence:
          1. Outside lock: get_fills paginated.
          2. Inside trade_log_lock: build placement_index, load tracked_oids,
             resolve strategies for missing oids, write batched synthetic rows.
          3. Outside lock: compute next cursor, persist state.

        The dedup invariant (at most one event=='fill' per order_id) is
        enforced by step 2 reading tracked_oids INSIDE the lock — between
        the read and the write, no engine writer can append a fill row,
        because the engine acquires the same lock for its writes.
        """
        started_at_ts = self._clock()
        # Hoisted out of the try so the failure path can populate the
        # cursor observability fields on ReconcileTick.
        cursor_before = self._last_safe_ts_unix
        try:
            # Per-tick observability accumulators (drained into the log at end of tick).
            self._manual_close_realized_pnl_cents_this_tick = 0
            self._manual_close_apply_latency_max_this_tick = 0.0
            self._co_realized_pnl_cents_this_tick = 0
            self._co_apply_latency_max_this_tick = 0.0
            self._co_applies_this_tick = 0

            # Phase 1 (outside lock): fetch from Kalshi
            fetch_result = await self._fetch_fills_paginated(
                min_ts=self._last_safe_ts_unix
            )
            pages_consumed = fetch_result["pages_consumed"]
            truncated = fetch_result["truncated"]
            cycle_aborted = fetch_result["cycle_aborted"]
            all_fills = fetch_result["fills"]
            by_order_id = _group_fills_by_order_id(all_fills)

            # Phase 2 (inside lock): tracked-oid read + synthetic-row writes
            async with self._trade_log_lock:
                await self._flush_pending_startup_alerts()
                self._build_placement_index()
                tracked = self._load_tracked_oids()
                candidate_oids = self._select_candidate_oids(by_order_id, tracked)
                tally = await self._classify_and_dispatch(
                    candidate_oids, by_order_id, tracked
                )
                if tally.synthetic_rows:
                    await self._append_synthetic_rows(tally.synthetic_rows)

            # Phase 3 (outside lock): cursor advance + state persist.
            now = int(self._clock())
            cursor_after = self._advance_cursor(
                truncated=truncated, any_unknown=tally.any_unknown, now=now
            )

            delta = len(tally.synthetic_rows)
            status = self._tick_status(
                delta=delta,
                truncated=truncated,
                cycle_aborted=cycle_aborted,
                now=now,
            )

            # Delta-driven Telegram alerts (distinct ladder from _handle_tick_failure,
            # separate cooldown timestamp). Must run before _persist_state so the
            # cooldown survives restart.
            await self._alert_on_status(
                status=status,
                delta_count=delta,
                truncated=truncated,
                cycle_aborted=cycle_aborted,
            )
            await self._alert_backfilled_rows(tally.synthetic_rows)

            await self._persist_state()
            # Reset failure counter only after persisted state is durable, so a
            # crash between reset and persist can't drop the on-disk counter
            # out of sync with the in-memory one.
            self._consecutive_failures = 0

            ended_at_ts = self._clock()
            duration_ms = int((ended_at_ts - started_at_ts) * 1000)

            logger.info(
                "reconciler_tick",
                extra=self._build_tick_log_extra(
                    started_at_ts=started_at_ts,
                    ended_at_ts=ended_at_ts,
                    duration_ms=duration_ms,
                    pages_consumed=pages_consumed,
                    truncated=truncated,
                    cycle_aborted=cycle_aborted,
                    fills_fetched=len(all_fills),
                    delta_count=delta,
                    order_ids_synthesised=tally.order_ids_synthesised,
                    resolution_breakdown=tally.resolution_breakdown,
                    alert_state=status,
                    cursor_before=cursor_before,
                    cursor_after=cursor_after,
                    classifier_decision_counts=tally.classifier_decision_counts,
                    apply_path_outcome_counts=tally.apply_path_outcome_counts,
                    pending_partial_sells_count=len(self._pending_partial_sells),
                ),
            )

            return ReconcileTick(
                status=status,
                delta_count=delta,
                duration_ms=duration_ms,
                pages_consumed=pages_consumed,
                truncated_at_max_pages=truncated,
                cycle_aborted_at_cursor=cycle_aborted,
                order_ids_synthesised=tally.order_ids_synthesised,
                strategy_resolution_breakdown=tally.resolution_breakdown,
                last_safe_ts_unix_before=cursor_before,
                last_safe_ts_unix_after=cursor_after,
            )
        except Exception as exc:
            return await self._tick_failure_result(exc, started_at_ts, cursor_before)

    def _select_candidate_oids(
        self, by_order_id: dict[str, list[dict[str, Any]]], tracked: TrackedOids
    ) -> set[str]:
        """Candidate oids for this tick. Caller MUST hold trade_log_lock."""
        # Dedup-before-classify: skip oids that are already in ANY
        # tracked set (buy fills + manual_close rows + cash_out rows).
        # Prevents classifier_decision_counts from inflating on every
        # tick that re-scans the 24h safety window for already-handled
        # closes / cash-outs.
        candidate_oids = (
            set(by_order_id.keys())
            - tracked.buys
            - tracked.manual_closes
            - tracked.cash_outs
        )

        # Group-cap: apply max_fills_per_tick to the COMBINED candidate
        # set (buys + sells).
        if len(candidate_oids) > self._max_fills_per_tick:
            logger.warning(
                "Reconciler over max_fills_per_tick (%d > %d); "
                "deferring %d candidates to next tick",
                len(candidate_oids),
                self._max_fills_per_tick,
                len(candidate_oids) - self._max_fills_per_tick,
            )
            candidate_oids = set(sorted(candidate_oids)[: self._max_fills_per_tick])
        return candidate_oids

    async def _classify_and_dispatch(
        self,
        candidate_oids: set[str],
        by_order_id: dict[str, list[dict[str, Any]]],
        tracked: TrackedOids,
    ) -> _DispatchTally:
        """Classify each candidate (sorted) and apply it. Caller MUST hold trade_log_lock."""
        tally = _DispatchTally()
        # Open-positions snapshot is the BUY_CASH_OUT classifier
        # discriminator: a BUY against an existing open (ticker, side)
        # is a cash-out, not a new entry.
        open_positions_by_ticker_side = self._build_open_positions_by_ticker_side()
        ctx = ClassifierContext(
            tracked_buy_oids=tracked.buys,
            placement_index=self._placement_index,
            open_positions_by_ticker_side=open_positions_by_ticker_side,
        )
        for oid in sorted(candidate_oids):
            fills_for_order = by_order_id[oid]
            kind = classify_fill(fills_for_order[0], ctx)
            tally.classifier_decision_counts[kind.value] = (
                tally.classifier_decision_counts.get(kind.value, 0) + 1
            )
            if kind == FillKind.UNKNOWN:
                tally.any_unknown = True
            if kind == FillKind.BUY_ENTRY:
                self._synthesise_buy_entry(oid, fills_for_order, tally)
            else:
                await self._dispatch_close_fill(
                    kind, oid, fills_for_order, tracked, tally
                )
        return tally

    def _synthesise_buy_entry(
        self,
        oid: str,
        fills_for_order: list[dict[str, Any]],
        tally: _DispatchTally,
    ) -> None:
        strategy, method = self._safe_resolve_strategy(oid, fills_for_order[0])
        tally.resolution_breakdown[method] = (
            tally.resolution_breakdown.get(method, 0) + 1
        )
        row = self._build_synthetic_row(
            order_id=oid,
            kalshi_fills=fills_for_order,
            strategy=strategy,
            resolution_method=method,
        )
        tally.synthetic_rows.append(row)
        tally.order_ids_synthesised.append(oid)

    async def _dispatch_close_fill(
        self,
        kind: FillKind,
        oid: str,
        fills_for_order: list[dict[str, Any]],
        tracked: TrackedOids,
        tally: _DispatchTally,
    ) -> None:
        # Sell or anomaly path — derive position_key by scanning
        # engine._orders for a record whose market_ticker matches.
        market_ticker = fills_for_order[0].get("market_ticker", "")
        position_key = self._derive_position_key_for_ticker(market_ticker)
        outcome = await self._apply_fill_kind(
            kind=kind,
            order_id=oid,
            kalshi_fills=fills_for_order,
            position_key=position_key,
            market_ticker=market_ticker,
            tracked_manual_close_oids=tracked.manual_closes,
            tracked_cash_out_oids=tracked.cash_outs,
        )
        tally.apply_path_outcome_counts[outcome] = (
            tally.apply_path_outcome_counts.get(outcome, 0) + 1
        )
        if outcome == ApplyOutcome.APPLIED:
            tally.order_ids_synthesised.append(oid)

    def _advance_cursor(self, *, truncated: bool, any_unknown: bool, now: int) -> int:
        """Return the cursor after this tick, advancing it unless the window is incomplete.

        When pagination truncated at max_pages we did NOT consume the full
        window — advancing the cursor would silently drop the abandoned
        pages forever. Hold the cursor steady so the next tick re-fetches.

        If any fill this tick was classified UNKNOWN (schema drift, missing
        action), do NOT advance the cursor. The fill re-appears next tick
        for human review.
        """
        if truncated or any_unknown:
            return self._last_safe_ts_unix
        cursor_after = self._compute_next_cursor(now=now)
        self._last_safe_ts_unix = cursor_after
        return cursor_after

    def _tick_status(
        self, *, delta: int, truncated: bool, cycle_aborted: bool, now: int
    ) -> Literal["clean", "warning", "critical"]:
        if delta == 0 and not truncated and not cycle_aborted:
            self._last_clean_completion_ts_unix = now
            return "clean"
        if (
            delta > self._alert_threshold_critical_delta
            or truncated
            or cycle_aborted
        ):
            return "critical"
        return "warning"

    async def _alert_backfilled_rows(self, synthetic_rows: list[dict[str, Any]]) -> None:
        """Per-fill detail alert for backfilled BUY_ENTRY rows.

        Additive, status-independent: any non-empty synthetic_rows means
        real positions were durably written this tick (post-append — an
        append failure diverts to the outer handler before here), so the
        operator should see them even alongside a truncation/cycle CRITICAL.
        """
        if synthetic_rows and self._notifier is not None:
            # The formatter is wrapped LOCALLY: it is evaluated before
            # _safe_alert is entered, so a formatter bug must be caught here or
            # it would escape into the outer tick handler and skip _persist_state
            # / bump _consecutive_failures / fire a spurious failure alert.
            try:
                backfill_msg = _format_backfill_fill_alert(synthetic_rows)
            except Exception:  # noqa: BLE001 — observational alert, never fatal
                logger.exception(
                    "backfill alert formatter raised; skipping detail alert"
                )
                backfill_msg = None
            if backfill_msg:
                await self._safe_alert(AlertLevel.INFO, backfill_msg)

    async def _tick_failure_result(
        self, exc: Exception, started_at_ts: float, cursor_before: int
    ) -> ReconcileTick:
        ended_at_ts = self._clock()
        duration_ms = int((ended_at_ts - started_at_ts) * 1000)
        self._consecutive_failures += 1
        await self._handle_tick_failure(exc)

        # Sane defaults for fields not yet computed when the exception
        # fired; alert_state is "warning" because tick() never raises and
        # the failure-path return value is always status="warning".
        logger.info(
            "reconciler_tick",
            extra=self._build_tick_log_extra(
                started_at_ts=started_at_ts,
                ended_at_ts=ended_at_ts,
                duration_ms=duration_ms,
                pages_consumed=0,
                truncated=False,
                cycle_aborted=False,
                fills_fetched=0,
                delta_count=0,
                order_ids_synthesised=[],
                resolution_breakdown={},
                alert_state="warning",
                cursor_before=cursor_before,
                cursor_after=self._last_safe_ts_unix,
                classifier_decision_counts={},
                apply_path_outcome_counts={},
                pending_partial_sells_count=len(self._pending_partial_sells),
                exc=exc,
            ),
        )

        return ReconcileTick(
            status="warning",
            delta_count=0,
            duration_ms=duration_ms,
            exc_class=type(exc).__name__,
            exc_repr=repr(exc)[:256],
            last_safe_ts_unix_before=cursor_before,
            last_safe_ts_unix_after=self._last_safe_ts_unix,
        )

    def _build_tick_log_extra(
        self,
        *,
        started_at_ts: float,
        ended_at_ts: float,
        duration_ms: int,
        pages_consumed: int,
        truncated: bool,
        cycle_aborted: bool,
        fills_fetched: int,
        delta_count: int,
        order_ids_synthesised: list[str],
        resolution_breakdown: dict[str, int],
        alert_state: str,
        cursor_before: int,
        cursor_after: int,
        classifier_decision_counts: dict[str, int] | None = None,
        apply_path_outcome_counts: dict[str, int] | None = None,
        pending_partial_sells_count: int = 0,
        exc: Exception | None = None,
    ) -> dict[str, Any]:
        """Build the structured-log `extra` dict for a reconciler tick.

        Single source of truth for the success and failure paths so the
        field contract can't drift between them. `exc` is only set on
        the failure path and adds `exc_class`/`exc_repr` to the dict.
        """
        extra: dict[str, Any] = {
            "event": "reconciler_tick",
            "schema_version": 1,
            "tick_id": str(uuid.uuid4()),
            "ts_start": datetime.fromtimestamp(started_at_ts, tz=UTC).isoformat(),
            "ts_end": datetime.fromtimestamp(ended_at_ts, tz=UTC).isoformat(),
            "duration_ms": duration_ms,
            "api_calls_made": pages_consumed,
            "pages_consumed": pages_consumed,
            "truncated_at_max_pages": truncated,
            "cycle_aborted_at_cursor": cycle_aborted,
            "fills_fetched": fills_fetched,
            "delta_count": delta_count,
            "order_ids_synthesised": order_ids_synthesised[:20],
            "strategy_resolution_breakdown": resolution_breakdown,
            "consecutive_failures": self._consecutive_failures,
            "alert_state": alert_state,
            "last_safe_ts_unix_before": cursor_before,
            "last_safe_ts_unix_after": cursor_after,
            "classifier_decision_counts": classifier_decision_counts or {},
            "apply_path_outcome_counts": apply_path_outcome_counts or {},
            "manual_close_applied": self._manual_close_enabled,
            "manual_close_realized_pnl_cents_total": (
                self._manual_close_realized_pnl_cents_this_tick
            ),
            "manual_close_apply_latency_seconds": round(
                self._manual_close_apply_latency_max_this_tick, 3
            ),
            "co_realized_pnl_cents_this_tick": self._co_realized_pnl_cents_this_tick,
            "co_apply_latency_seconds": (
                round(self._co_apply_latency_max_this_tick, 3)
                if self._co_applies_this_tick > 0
                else None
            ),
            "co_applies_this_tick": self._co_applies_this_tick,
            "pending_partial_sells_count": pending_partial_sells_count,
            "shakedown_suppressed_count": len(self._shakedown_seen_oids),
        }
        if exc is not None:
            extra["exc_class"] = type(exc).__name__
            extra["exc_repr"] = repr(exc)[:256]
        return extra

    async def _handle_tick_failure(self, exc: Exception) -> None:
        """API-degraded vs code-bug distinction with cooldown.

        Side effects: sends one alert via self._notifier and may update
        self._last_critical_alert_ts_unix. Caller MUST have already
        incremented self._consecutive_failures — this method reads the
        counter for thresholding but does not mutate it. Notifier errors
        are swallowed (logged) so tick() honours its never-raises contract.
        """
        if self._notifier is None:
            return

        is_api_degraded = isinstance(exc, KalshiAPIError)
        now = int(self._clock())
        message = (
            f"Fills reconciler: {self._consecutive_failures} consecutive "
            f"failures — last error: {type(exc).__name__}: {str(exc)[:200]}"
        )

        if is_api_degraded:
            level = AlertLevel.WARNING
        elif self._consecutive_failures < self._alert_threshold_consecutive:
            level = AlertLevel.WARNING
        else:
            in_cooldown = (
                now - self._last_critical_alert_ts_unix
                < self._critical_alert_cooldown_seconds
            )
            if in_cooldown:
                level = AlertLevel.WARNING
            else:
                level = AlertLevel.CRITICAL
                self._last_critical_alert_ts_unix = now

        try:
            await self._notifier.send_alert(level=level, message=message)
        except Exception:
            logger.exception(
                "fills_reconciler notifier failed during failure-handling"
            )

    async def _alert_on_status(
        self,
        *,
        status: Literal["clean", "warning", "critical", "skipped"],
        delta_count: int,
        truncated: bool,
        cycle_aborted: bool = False,
    ) -> None:
        """Delta-driven success-path alerts (separate cooldown from _handle_tick_failure).

        Mapping:
          - "clean":   no alert; clears ALL three critical-mode cooldowns (a clean
            tick means delta==0 AND not truncated AND not cycle-aborted, so the next
            critical of any mode should fire immediately).
          - "warning": WARNING on every tick (no cooldown).
          - "critical": CRITICAL on first occurrence; downgraded to WARNING
            during the cooldown window. Triggered by truncation, cursor cycle
            abort, OR delta > alert_threshold_critical_delta. Each of the three
            sub-modes keeps its OWN cooldown timestamp so structurally-different
            failure modes never silence each other.
          - "skipped": no alert.

        Notifier failures are swallowed (logged) so tick() never raises.
        """
        if status == "clean":
            self._last_delta_critical_alert_ts_unix = 0
            self._last_truncation_critical_alert_ts_unix = 0
            self._last_cycle_critical_alert_ts_unix = 0
            return

        if status == "skipped" or self._notifier is None:
            return

        if status == "warning":
            level = AlertLevel.WARNING
            message = (
                f"Fills reconciler: backfilled {delta_count} missing "
                f"fill row(s); local trade log was missing fills present "
                f"in /portfolio/fills."
            )
        else:  # status == "critical"
            if cycle_aborted:
                message = (
                    f"Fills reconciler: cursor cycle detected at "
                    f"/portfolio/fills (delta={delta_count}); pagination "
                    f"aborted to prevent row inflation. Pre-cycle pages "
                    f"yielded {delta_count} synthetic row(s); the cycling "
                    f"page was discarded."
                )
            elif truncated:
                message = (
                    f"Fills reconciler: pagination truncated at "
                    f"max_pages_per_tick (delta={delta_count}); backlog "
                    f"may exist."
                )
            else:
                message = (
                    f"Fills reconciler: backfilled {delta_count} rows — "
                    f"exceeds critical_delta="
                    f"{self._alert_threshold_critical_delta}."
                )
            now = int(self._clock())
            # Per-mode cooldown: read (and on fire, write) the timestamp for the
            # sub-mode that triggered this critical. Explicit if/elif/else (not
            # getattr/setattr) so the attribute names are typo-checkable. Branch
            # order mirrors the message selection above.
            if cycle_aborted:
                last_ts = self._last_cycle_critical_alert_ts_unix
            elif truncated:
                last_ts = self._last_truncation_critical_alert_ts_unix
            else:
                last_ts = self._last_delta_critical_alert_ts_unix
            in_cooldown = now - last_ts < self._critical_alert_cooldown_seconds
            if in_cooldown:
                level = AlertLevel.WARNING
            else:
                level = AlertLevel.CRITICAL
                if cycle_aborted:
                    self._last_cycle_critical_alert_ts_unix = now
                elif truncated:
                    self._last_truncation_critical_alert_ts_unix = now
                else:
                    self._last_delta_critical_alert_ts_unix = now

        try:
            await self._notifier.send_alert(level=level, message=message)
        except Exception:
            logger.exception(
                "fills_reconciler notifier failed during _alert_on_status"
            )

    async def _flush_pending_startup_alerts(self) -> None:
        """Emit any alerts queued during sync __init__.

        __init__ cannot fire async alerts; the corrupt-state recovery path
        queues them on self._pending_startup_alerts and the first tick()
        flushes them through the notifier.
        """
        if not self._pending_startup_alerts or self._notifier is None:
            return
        pending = list(self._pending_startup_alerts)
        self._pending_startup_alerts.clear()
        for level, message in pending:
            await self._notifier.send_alert(level=level, message=message)

    def _compute_next_cursor(self, *, now: int) -> int:
        """Compute the next cursor value.

        Invariant: new_cursor = max(
            last_safe_ts_unix,
            now - eventual_consistency_window_seconds,
        )
        Cursor advances to now - window if greater than persisted; otherwise
        stays put. Never regresses. The window prevents skipping fills that
        arrive late from Kalshi's eventually-consistent /portfolio/fills index.
        """
        floor = now - self._eventual_consistency_window_seconds
        return max(self._last_safe_ts_unix, floor)

    def _load_tracked_live_order_ids(self) -> set[str]:
        """Backward-compatible wrapper returning only the buy-oid set.

        New callers should prefer _load_tracked_oids() to get manual-close
        and cash-out oids in the same pass. Kept for compatibility with the
        backfill() path and tests.
        """
        return self._load_tracked_oids().buys

    def _load_tracked_oids(self) -> TrackedOids:
        """Single-pass scan returning TrackedOids(buys, manual_closes, cash_outs).

        MUST be called inside trade_log_lock — preserves the "engine cannot
        append between read and synthetic-write" invariant.

        Scans state/trades.jsonl once and partitions live-mode rows by event:
          - event=="fill"          → buys (entry dedup)
          - event=="manual_close"  → manual_closes (idempotency)
          - event=="cash_out"      → cash_outs (cash-out idempotency)

        Skips torn (JSONDecodeError) lines.
        """
        buys: set[str] = set()
        manual_closes: set[str] = set()
        cash_outs: set[str] = set()
        if not self._trade_log_path.exists():
            return TrackedOids(buys, manual_closes, cash_outs)

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
                oid = row.get("order_id")
                if not oid:
                    continue
                event = row.get("event", TRADE_LOG_EVENT_FILL)
                if event == TRADE_LOG_EVENT_FILL:
                    buys.add(oid)
                elif event == TRADE_LOG_EVENT_MANUAL_CLOSE:
                    manual_closes.add(oid)
                elif event == TRADE_LOG_EVENT_CASH_OUT:
                    cash_outs.add(oid)
        return TrackedOids(buys, manual_closes, cash_outs)

    def _default_strategy_resolver(
        self, order_id: str, kalshi_fill: dict[str, Any]
    ) -> tuple[str, str]:
        """Default 3-step strategy resolution chain.

        (a) in-memory: engine._orders[order_id].signal.strategy_name
        (b) placement-row scan: backward-scan trades.jsonl for matching oid
        (c) unresolved: return ("unknown", "unresolved")

        _build_placement_index() is called once per tick, then this resolver
        runs O(1) per order_id via the cached index. Without the cache, K
        missing fills in one tick would trigger K full log scans.
        """
        if self._engine is not None:
            record = getattr(self._engine, "_orders", {}).get(order_id)
            if record is not None:
                signal = getattr(record, "signal", None)
                strategy_name = getattr(signal, "strategy_name", None)
                if strategy_name:
                    return (strategy_name, "in_memory")

        placement_strategy = self._lookup_placement_strategy(order_id)
        if placement_strategy is not None:
            return (placement_strategy, "placement_scan")

        return ("unknown", "unresolved")

    def _lookup_placement_strategy(self, order_id: str) -> str | None:
        """Look up strategy from the cached placement-row index.

        `_build_placement_index()` is called at the top of each tick to
        rebuild this dict; here we just hash-look-up and extract strategy.
        """
        entry = self._placement_index.get(order_id)
        if entry is None:
            return None
        return entry.get("strategy")

    def _build_placement_index(self) -> None:
        """Build the (order_id -> {strategy, action}) index by scanning the
        most recent placement_row_lookback rows of the trade log.

        'action' is recorded so classify_fill can disambiguate bot-placed
        sells (future feature) from manual sells (today). Legacy rows lacking
        'action' default to "buy".

        Called once per tick — keeps strategy resolution O(N + K) instead of
        O(N * K). If multiple placement rows exist for the same order_id (rare
        — would mean the same order was re-placed), the LATEST row's entry wins
        (forward-iteration overwrite).
        """
        self._placement_index = {}
        if not self._trade_log_path.exists():
            return

        # Single-pass deque keeps memory bounded even for large logs.
        with self._trade_log_path.open() as f:
            lines: deque[str] = deque(f, maxlen=self._placement_row_lookback)

        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != TRADE_LOG_EVENT_PLACEMENT:
                continue
            oid = row.get("order_id")
            strategy = row.get("metadata", {}).get("strategy")
            if oid and strategy:
                action = row.get("action", "buy")  # legacy default
                self._placement_index[oid] = {
                    "strategy": str(strategy),
                    "action": str(action),
                }

    async def _fetch_fills_paginated(
        self, *, min_ts: int
    ) -> dict[str, Any]:
        """Fetch fills from Kalshi with pagination and rate-limit respect.

        Delegates to the shared ``paginate_detailed`` free function (also used by
        ``KalshiClient._iter_paginated`` for markets/positions/orders) so there is
        ONE cycle-defense / terminal-cursor implementation. The ``fetch`` lambda
        wraps ``self._kalshi_client.get_fills`` — test fakes only need ``get_fills``.
        Capped at ``max_pages_per_tick``; ``page_delay_ms`` is converted to seconds.
        The cycling page is discarded (cycle_aborted=True); a cap-hit with a pending
        cursor sets truncated=True; the two are mutually exclusive. The dict shape is
        unchanged for ``_tick_impl`` / the alert path.

        Returns:
            {
                "fills": list[dict],
                "fills_count": int,
                "pages_consumed": int,
                "truncated": bool,
                "cycle_aborted": bool,
            }
        """
        result = await paginate_detailed(
            method_name="fills_reconciler",
            list_field="fills",
            fetch=lambda cursor: self._kalshi_client.get_fills(
                min_ts=min_ts, cursor=cursor
            ),
            page_delay=self._page_delay_ms / 1000.0,
            filter_repr=f"min_ts={min_ts}",
            max_pages=self._max_pages_per_tick,
        )
        return {
            "fills": result.rows,
            "fills_count": len(result.rows),
            "pages_consumed": result.pages_consumed,
            "truncated": result.truncated,
            "cycle_aborted": result.cycle_aborted,
        }

    def _build_synthetic_row(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        strategy: str,
        resolution_method: str,
    ) -> dict[str, Any]:
        """Build a synthetic event='fill' row from Kalshi /portfolio/fills records.

        Aggregates via _aggregate_fills: VWAP price, summed contracts and
        fees, taker/maker contract counts, earliest fill timestamp.

        Metadata fields written:
          - reconciled: True
          - reconcile_source: "fills_reconciler"
          - strategy_resolution: <resolver_method>

        Raises:
            ValueError: if kalshi_fills is empty.
        """
        if not kalshi_fills:
            raise ValueError(f"No Kalshi fills for order_id={order_id}")

        sample = kalshi_fills[0]
        agg = _aggregate_fills(kalshi_fills)
        market_ticker = sample.get("market_ticker") or sample.get("ticker", "")

        return {
            "event": TRADE_LOG_EVENT_FILL,
            "order_id": order_id,
            "market_ticker": market_ticker,
            "side": agg.side,
            "mode": "live",
            "timestamp": agg.earliest_ts or "",
            "fill_price": round(agg.vwap, 4),
            "contracts": agg.contracts,
            "dollars": round(agg.dollars, 4),
            "is_taker_fills": agg.taker_contracts,
            "is_maker_fills": agg.maker_contracts,
            "fees_from_api": round(agg.fees, 4),
            "metadata": {
                "strategy": strategy,
                "reconciled": True,
                "reconcile_source": "fills_reconciler",
                "strategy_resolution": resolution_method,
            },
        }

    async def _apply_manual_close(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        tracked_manual_close_oids: set[str],
    ) -> ApplyOutcome:
        """Manual-close apply path. Caller MUST hold self._trade_log_lock.

        Returns one of:
          "APPLIED"              — full apply (row written, close_position called)
          "ALREADY_CLOSED_DEDUP" — idempotency no-op (oid already in log)
          "ORPHAN"               — no position in risk; WARNING alert
          "ZERO_COST_BASIS"      — sell exists but no entry FILL; CRITICAL alert (deduped)
          "PARTIAL_HELD"         — sell incomplete; held for next tick
        """
        if order_id in tracked_manual_close_oids:
            logger.info(
                "manual_close: dedup-skip oid=%s (already in trade log)", order_id
            )
            return ApplyOutcome.ALREADY_CLOSED_DEDUP

        # Position must exist in risk manager (use the public `positions`
        # property; respects RiskManager encapsulation).
        risk = getattr(self._engine, "_risk", None)
        risk_positions = risk.positions if risk is not None else {}
        if position_key not in risk_positions:
            logger.warning(
                "manual_close: ORPHAN sell oid=%s position_key=%s — no position to close",
                order_id, position_key,
            )
            await self._safe_alert(
                AlertLevel.WARNING,
                f"Sell fill {order_id} for {market_ticker} but no local position "
                f"({position_key}) to close — likely already-closed or external.",
            )
            return ApplyOutcome.ORPHAN
        assert risk is not None

        cb = cost_basis_for_position(
            self._trade_log_path,
            position_key=position_key,
            market_ticker=market_ticker,
        )
        if cb.total_contracts == 0:
            # Orphan close — sell exists but no entry FILL. Wrong P&L is worse
            # than missing P&L. Dedup via _orphan_close_oids so the same orphan
            # doesn't fire CRITICAL every tick.
            if order_id not in self._orphan_close_oids:
                self._orphan_close_oids.add(order_id)
                logger.error(
                    "manual_close: ZERO_COST_BASIS oid=%s position_key=%s — sell with no entry",
                    order_id, position_key,
                )
                await self._safe_alert(
                    AlertLevel.CRITICAL,
                    f"Orphan manual close for {market_ticker}: sell oid={order_id} "
                    f"but local cost basis is empty. Operator must reconcile.",
                )
            return ApplyOutcome.ZERO_COST_BASIS

        total_sold = sum(_contracts_from_count_fp(f.get("count_fp")) for f in kalshi_fills)
        if total_sold < cb.total_contracts:
            logger.info(
                "manual_close: PARTIAL_HELD oid=%s sold=%d basis=%d",
                order_id, total_sold, cb.total_contracts,
            )
            self._pending_partial_sells.setdefault(order_id, self._clock())
            return ApplyOutcome.PARTIAL_HELD

        # Drop any pending-partial entry now that we're applying.
        # When a partial sell completes on a later tick, we land here; the
        # earlier setdefault leaves a stale key without this cleanup.
        self._pending_partial_sells.pop(order_id, None)

        agg = _aggregate_fills(kalshi_fills)
        close_proceeds = agg.dollars
        exit_fees = agg.fees
        earliest_ts = agg.earliest_ts
        side_label = agg.side
        taker_count = agg.taker_contracts
        maker_count = agg.maker_contracts
        realized_pnl = close_proceeds - cb.total_dollars - cb.total_fees - exit_fees
        vwap = close_proceeds / total_sold if total_sold else 0.0

        # 3-step resolver chain + cost_basis fallback.
        strategy, method, fell_back = self._resolve_strategy_with_basis_fallback(
            order_id, kalshi_fills[0], cb
        )
        if fell_back:
            logger.warning(
                "manual_close: strategy attributed via cost_basis_fallback "
                "oid=%s position_key=%s strategy=%s — verify if multi-leg",
                order_id, position_key, strategy,
            )

        applied_at = datetime.now(UTC).isoformat()
        row = {
            "event": TRADE_LOG_EVENT_MANUAL_CLOSE,
            "order_id": order_id,
            "market_ticker": market_ticker,
            "position_key": position_key,
            "side": side_label,
            "mode": "live",
            "timestamp": earliest_ts,
            "applied_at": applied_at,
            "close_price": round(vwap, 4),
            "contracts_closed": total_sold,
            "close_proceeds": round(close_proceeds, 4),
            "cost_basis_dollars": round(cb.total_dollars, 4),
            "cost_basis_contracts": cb.total_contracts,
            "entry_fees": round(cb.total_fees, 4),
            "exit_fees": round(exit_fees, 4),
            "realized_pnl": round(realized_pnl, 4),
            "is_taker_fills": taker_count,
            "is_maker_fills": maker_count,
            "fees_from_api": round(exit_fees, 4),
            "metadata": {
                "strategy": strategy,
                "reconciled": True,
                "reconcile_source": "fills_reconciler",
                "classification": "SELL_MANUAL",
                "close_type": "full",
                "strategy_resolution": method,
            },
        }
        await self._append_synthetic_rows([row])
        _warn_on_pod_attribution_split(risk, position_key, strategy)
        risk.close_position(position_key, pnl=realized_pnl)
        # Circuit-breaker accounting must include manual-close P&L.
        # Mirrors _settle_live's record_strategy_trade call so per-strategy
        # rolling P&L tracking sees these closes.
        risk.record_strategy_trade(strategy, realized_pnl, won=(realized_pnl > 0))
        assert self._engine is not None  # risk came from self._engine._risk
        for rec in self._engine._orders.values():
            if getattr(rec, "signal", None) is None:
                continue
            if rec.signal.position_key == position_key:
                rec.status = OrderStatus.EXTERNALLY_CLOSED

        await self._safe_alert(
            AlertLevel.WARNING,
            f"🔄 Manual close reconciled: {market_ticker} pos={position_key} "
            f"{total_sold}c @ {vwap*100:.0f}¢ realized_pnl=${realized_pnl:+.2f}. "
            f"Exposure freed.",
        )

        # Per-tick observability accumulators.
        self._manual_close_realized_pnl_cents_this_tick += int(round(realized_pnl * 100))
        # Apply-latency: max over tick (applied_at − earliest sell fill created_time)
        if earliest_ts:
            lat = _apply_latency_seconds(earliest_ts)
            if lat is not None:
                self._manual_close_apply_latency_max_this_tick = max(
                    self._manual_close_apply_latency_max_this_tick, lat
                )
        return ApplyOutcome.APPLIED

    async def _apply_cash_out(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        tracked_cash_out_oids: set[str],
    ) -> ApplyOutcome:
        """Cash-out apply path. Caller MUST hold self._trade_log_lock.

        Returns one of:
          APPLIED_FULL_CASH_OUT | APPLIED_PARTIAL_CASH_OUT |
          ALREADY_CLOSED_DEDUP_CASH_OUT | ZERO_COST_BASIS_CASH_OUT |
          RACE_LOST_TO_SETTLEMENT | INVALID_PARTIAL_CASH_OUT |
          PNL_CAP_EXCEEDED_CASH_OUT
        """
        plan = await self._plan_cash_out(
            order_id=order_id,
            kalshi_fills=kalshi_fills,
            position_key=position_key,
            market_ticker=market_ticker,
            tracked_cash_out_oids=tracked_cash_out_oids,
        )
        if isinstance(plan, ApplyOutcome):
            return plan
        return await self._commit_cash_out(
            order_id=order_id,
            kalshi_fills=kalshi_fills,
            position_key=position_key,
            market_ticker=market_ticker,
            plan=plan,
        )

    async def _plan_cash_out(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        tracked_cash_out_oids: set[str],
    ) -> _CashOutPlan | ApplyOutcome:
        """Guard phase: reject with an outcome, or return the plan to commit.

        Each reject branch keeps its own sticky-set / alert-dedup behavior;
        the asymmetries between them are intentional and pinned by
        tests/test_cash_out_reject_semantics.py.
        """
        # Empty-fills defense. Caller invariant should prevent this; the
        # alternative is an uncaught IndexError on kalshi_fills[0].
        if not kalshi_fills:
            logger.error("_apply_cash_out: empty kalshi_fills for oid=%s", order_id)
            return ApplyOutcome.EMPTY_FILLS_CASH_OUT

        if order_id in tracked_cash_out_oids:
            return ApplyOutcome.ALREADY_CLOSED_DEDUP_CASH_OUT

        hedge_side = (kalshi_fills[0].get("side", "") or "").upper()
        closed_side = "NO" if hedge_side == "YES" else "YES"

        # Settlement-race defense: position must still exist in risk manager.
        # Use the public `positions` property (defensive copy) to mirror
        # _apply_manual_close encapsulation.
        risk = getattr(self._engine, "_risk", None)
        risk_positions = getattr(risk, "positions", {}) if risk is not None else {}
        if position_key not in risk_positions:
            return await self._reject_race_lost(
                order_id, position_key, market_ticker, hedge_side
            )
        # Narrow for mypy: if we got past the race-lost guard, the position
        # exists in risk.positions, so risk cannot be None.
        assert risk is not None

        cb = cost_basis_for_position(
            self._trade_log_path,
            position_key=position_key,
            market_ticker=market_ticker,
        )
        if cb.total_contracts == 0:
            return await self._reject_zero_basis(order_id, position_key, market_ticker)

        agg = _aggregate_fills(kalshi_fills)
        contracts_offset = agg.contracts

        if contracts_offset > cb.total_contracts:
            return await self._reject_overflow(
                order_id, position_key, market_ticker, contracts_offset, cb
            )

        try:
            pb = partial_basis(cb, contracts_offset)
        except ValueError:
            return await self._reject_basis_allocation(
                order_id, position_key, market_ticker, contracts_offset, cb
            )
        close_type: Literal["full", "partial"] = (
            "full" if contracts_offset == cb.total_contracts else "partial"
        )

        realized_pnl = (
            contracts_offset * 1.00
            - pb.offset_dollars
            - pb.offset_fees
            - agg.dollars
            - agg.fees
        )

        # P&L safety knob: bail to shakedown if magnitude exceeds the cap.
        if abs(realized_pnl) > self._co_max_pnl_apply_dollars:
            return await self._reject_pnl_cap(
                order_id, position_key, market_ticker, realized_pnl
            )

        return _CashOutPlan(
            risk=risk,
            hedge_side=hedge_side,
            closed_side=closed_side,
            cb=cb,
            agg=agg,
            pb=pb,
            close_type=close_type,
            realized_pnl=realized_pnl,
            contracts_residual=cb.total_contracts - contracts_offset,
        )

    async def _reject_race_lost(
        self, order_id: str, position_key: str, market_ticker: str, hedge_side: str
    ) -> ApplyOutcome:
        self._co_race_lost_oids.add(order_id)
        # Logger emit before Telegram so structlog forensics work even
        # if the notifier is down. Avoid dangling space on missing hedge_side.
        logger.warning(
            "_apply_cash_out: RACE_LOST oid=%s position_key=%s "
            "market_ticker=%s hedge_side=%s — position closed between "
            "classify and apply",
            order_id,
            position_key,
            market_ticker,
            hedge_side or "UNKNOWN",
        )
        await self._safe_alert(
            AlertLevel.WARNING,
            f"Cash out fill {market_ticker} oid={order_id} for hedge side "
            f"{hedge_side or 'UNKNOWN'}: position closed between classify "
            f"and apply — likely settlement race. No state mutation.",
        )
        return ApplyOutcome.RACE_LOST_TO_SETTLEMENT

    async def _reject_zero_basis(
        self, order_id: str, position_key: str, market_ticker: str
    ) -> ApplyOutcome:
        if order_id not in self._co_zero_basis_oids:
            self._co_zero_basis_oids.add(order_id)
            logger.error(
                "_apply_cash_out: ZERO_COST_BASIS oid=%s position_key=%s "
                "market_ticker=%s — position exists but no entry FILL rows",
                order_id,
                position_key,
                market_ticker,
            )
            await self._safe_alert(
                AlertLevel.CRITICAL,
                f"Cash out for {market_ticker} oid={order_id}: position exists "
                f"but no entry FILL rows; operator must reconcile.",
            )
        return ApplyOutcome.ZERO_COST_BASIS_CASH_OUT

    async def _reject_overflow(
        self,
        order_id: str,
        position_key: str,
        market_ticker: str,
        contracts_offset: int,
        cb: CostBasis,
    ) -> ApplyOutcome:
        self._co_invalid_partial_oids.add(order_id)
        logger.error(
            "_apply_cash_out: INVALID_PARTIAL (overflow) oid=%s "
            "position_key=%s market_ticker=%s contracts_offset=%d "
            "cb_total_contracts=%d",
            order_id,
            position_key,
            market_ticker,
            contracts_offset,
            cb.total_contracts,
        )
        await self._safe_alert(
            AlertLevel.CRITICAL,
            f"Cash out contracts ({contracts_offset}) exceeded local cost-basis "
            f"contracts ({cb.total_contracts}) for {market_ticker} "
            f"oid={order_id}; not applied.",
        )
        return ApplyOutcome.INVALID_PARTIAL_CASH_OUT

    async def _reject_basis_allocation(
        self,
        order_id: str,
        position_key: str,
        market_ticker: str,
        contracts_offset: int,
        cb: CostBasis,
    ) -> ApplyOutcome:
        # Dedup-gate the alert so retries don't re-page; the set add()
        # still happens unconditionally for sticky observability.
        already_alerted = _remember_oid(self._co_invalid_partial_oids, order_id)
        # Logger emit unconditional — forensics matter even on retry.
        logger.error(
            "_apply_cash_out: INVALID_PARTIAL (basis allocation) oid=%s "
            "position_key=%s market_ticker=%s contracts_offset=%d "
            "cb_total_contracts=%d",
            order_id,
            position_key,
            market_ticker,
            contracts_offset,
            cb.total_contracts,
        )
        if not already_alerted:
            await self._safe_alert(
                AlertLevel.CRITICAL,
                f"Cash out contracts ({contracts_offset}) failed basis allocation "
                f"for {market_ticker} oid={order_id}; not applied.",
            )
        return ApplyOutcome.INVALID_PARTIAL_CASH_OUT

    async def _reject_pnl_cap(
        self,
        order_id: str,
        position_key: str,
        market_ticker: str,
        realized_pnl: float,
    ) -> ApplyOutcome:
        already_alerted = _remember_oid(self._co_pnl_cap_oids, order_id)
        logger.error(
            "_apply_cash_out: PNL_CAP_EXCEEDED oid=%s position_key=%s "
            "market_ticker=%s realized_pnl=%.4f cap=%.4f",
            order_id,
            position_key,
            market_ticker,
            realized_pnl,
            self._co_max_pnl_apply_dollars,
        )
        if not already_alerted:
            await self._safe_alert(
                AlertLevel.CRITICAL,
                f"Cash out P&L exceeded cap: ${realized_pnl:+.2f} > "
                f"${self._co_max_pnl_apply_dollars:.2f} for {market_ticker} "
                f"oid={order_id}; NOT applied. Investigate.",
            )
        return ApplyOutcome.PNL_CAP_EXCEEDED_CASH_OUT

    async def _commit_cash_out(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        plan: _CashOutPlan,
    ) -> ApplyOutcome:
        """Commit phase: trade-log row first, then risk, then OrderRecord stamping."""
        risk = plan.risk
        pb = plan.pb
        close_type = plan.close_type
        realized_pnl = plan.realized_pnl
        contracts_offset = plan.agg.contracts

        # Strategy attribution: cost_basis_fallback is PRIMARY for cash-outs.
        strategy, method, _ = self._resolve_strategy_with_basis_fallback(
            order_id, kalshi_fills[0], plan.cb
        )

        # Atomic state mutation — ROW FIRST, then risk, then OrderRecord
        # stamping. The durable dedup token is the trade-log row, so if we
        # crash after the row write, the next-tick dedup pass skips this oid.
        applied_at = datetime.now(UTC).isoformat()
        row = _build_cash_out_row(
            order_id=order_id,
            market_ticker=market_ticker,
            position_key=position_key,
            plan=plan,
            strategy=strategy,
            method=method,
            applied_at=applied_at,
        )
        await self._append_synthetic_rows([row])

        _warn_on_pod_attribution_split(risk, position_key, strategy)
        if close_type == "full":
            risk.close_position(position_key, pnl=realized_pnl)
        else:
            risk.realize_partial(
                position_key,
                dollars_released=pb.offset_dollars,
                pnl=realized_pnl,
            )
            self._decrement_fill_counts_for_position(
                position_key=position_key,
                contracts_offset=contracts_offset,
                cb_total_contracts=plan.cb.total_contracts,
            )

        risk.record_strategy_trade(strategy, realized_pnl, won=(realized_pnl > 0))

        # OrderRecord stamping — FULL only; partial leaves records intact so
        # the residual leg can still settle via _settle_orders.
        if close_type == "full":
            assert self._engine is not None  # risk came from self._engine._risk
            for rec in self._engine._orders.values():
                sig = getattr(rec, "signal", None)
                if sig is None or sig.position_key != position_key:
                    continue
                if rec.status in _TERMINAL_ORDER_STATUSES:
                    continue
                rec.status = OrderStatus.EXTERNALLY_CLOSED

        logger.info(
            "_apply_cash_out: applied %s for oid=%s pos=%s contracts=%d "
            "pnl=%.4f",
            close_type,
            order_id,
            position_key,
            contracts_offset,
            realized_pnl,
        )
        await self._safe_alert(
            AlertLevel.WARNING,
            _cash_out_alert_message(plan, market_ticker, position_key),
        )

        self._co_realized_pnl_cents_this_tick += int(round(realized_pnl * 100))
        self._co_applies_this_tick += 1
        if plan.agg.earliest_ts:
            lat = _apply_latency_seconds(plan.agg.earliest_ts)
            if lat is not None:
                self._co_apply_latency_max_this_tick = max(
                    self._co_apply_latency_max_this_tick,
                    lat,
                )

        return (
            ApplyOutcome.APPLIED_FULL_CASH_OUT
            if close_type == "full"
            else ApplyOutcome.APPLIED_PARTIAL_CASH_OUT
        )

    async def _safe_alert(self, level: AlertLevel, message: str) -> None:
        """Wrap notifier.send_alert in try/except so a Telegram outage doesn't
        propagate out of the apply path and prevent cursor advance.

        Notifier protocol is send_alert(message: str, *, level=...) — level
        is keyword-only.
        """
        if self._notifier is None:
            return
        try:
            await self._notifier.send_alert(message, level=level)
        except Exception:  # noqa: BLE001 — observational alert, never fatal
            logger.exception("Notifier send_alert failed; continuing")

    async def _apply_fill_kind(
        self,
        *,
        kind: FillKind,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        tracked_manual_close_oids: set[str],
        tracked_cash_out_oids: set[str],
    ) -> ApplyOutcome:
        """Dispatch on classifier outcome. Caller holds trade_log_lock.

        BUY_ENTRY is handled by the existing _tick_impl loop; this dispatcher
        is for sell/anomaly/cash-out outcomes. Returns the outcome name for
        log/metrics. All alert calls go through self._safe_alert so notifier
        outages don't propagate out of the lock.
        """
        if kind == FillKind.SELL_BOT:
            raise NotImplementedError(
                "SELL_BOT classifier branch fired; bot has no sell path today. "
                "Investigate the placement-row that triggered it."
            )
        if kind == FillKind.SELL_BOT_ANOMALY:
            if order_id not in self._seen_anomalous_sells:
                self._seen_anomalous_sells.add(order_id)
                await self._safe_alert(
                    AlertLevel.CRITICAL,
                    f"Sell oid={order_id} matches existing buy oid for "
                    f"{market_ticker} — investigate.",
                )
            return ApplyOutcome.SELL_BOT_ANOMALY
        if kind == FillKind.UNKNOWN:
            await self._safe_alert(
                AlertLevel.CRITICAL,
                f"Reconciler classifier UNKNOWN for {market_ticker} "
                f"close_oid={order_id}; manual review needed.",
            )
            return ApplyOutcome.UNKNOWN
        if kind == FillKind.BUY_CASH_OUT:
            if not self._co_apply_enabled:
                return await self._handle_shakedown_cash_out(
                    order_id=order_id,
                    kalshi_fills=kalshi_fills,
                    position_key=position_key,
                    market_ticker=market_ticker,
                    tracked_cash_out_oids=tracked_cash_out_oids,
                )
            if (
                self._co_apply_limit > 0
                and self._co_applies_this_restart >= self._co_apply_limit
            ):
                logger.warning(
                    "cash_out: canary apply limit reached (%d); not applying oid=%s",
                    self._co_apply_limit, order_id,
                )
                return ApplyOutcome.CANARY_LIMIT_REACHED_CASH_OUT
            outcome = await self._apply_cash_out(
                order_id=order_id,
                kalshi_fills=kalshi_fills,
                position_key=position_key,
                market_ticker=market_ticker,
                tracked_cash_out_oids=tracked_cash_out_oids,
            )
            if outcome in (
                ApplyOutcome.APPLIED_FULL_CASH_OUT,
                ApplyOutcome.APPLIED_PARTIAL_CASH_OUT,
            ):
                self._co_applies_this_restart += 1
            return outcome
        if kind == FillKind.SELL_MANUAL_CANDIDATE:
            if not self._manual_close_enabled:
                # Shakedown mode (flag off): log and alert but don't apply
                return await self._handle_shakedown(
                    order_id=order_id,
                    kalshi_fills=kalshi_fills,
                    position_key=position_key,
                    market_ticker=market_ticker,
                    tracked_manual_close_oids=tracked_manual_close_oids,
                )
            # Canary apply limit (0 = unlimited).
            if (
                self._manual_close_apply_limit > 0
                and self._manual_close_applies_this_restart >= self._manual_close_apply_limit
            ):
                logger.warning(
                    "manual_close: canary apply limit reached (%d); not applying oid=%s",
                    self._manual_close_apply_limit, order_id,
                )
                return ApplyOutcome.CANARY_LIMIT_REACHED
            outcome = await self._apply_manual_close(
                order_id=order_id,
                kalshi_fills=kalshi_fills,
                position_key=position_key,
                market_ticker=market_ticker,
                tracked_manual_close_oids=tracked_manual_close_oids,
            )
            if outcome == ApplyOutcome.APPLIED:
                self._manual_close_applies_this_restart += 1
            return outcome
        # BUY_ENTRY not handled here — _tick_impl's existing loop owns it.
        # Return a sentinel rather than raising so a future caller that
        # passes BUY_ENTRY accidentally fails-soft.
        return ApplyOutcome.BUY_ENTRY_HANDLED_ELSEWHERE

    async def _handle_shakedown(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        tracked_manual_close_oids: set[str],
    ) -> ApplyOutcome:
        """Shakedown mode (flag off): classify, log+alert, persist seen.

        Apply-path is not exercised; only the classifier and cost-basis
        computation run. Persistent dedup prevents re-alerting the same
        historical close on every restart inside the 24h safety window.
        """
        if order_id in self._shakedown_seen_oids:
            return ApplyOutcome.SHAKEDOWN_SUPPRESSED
        if order_id in tracked_manual_close_oids:
            # Already applied in a prior session before flag was flipped off
            return ApplyOutcome.SHAKEDOWN_SUPPRESSED

        # Compute would-be values for the alert.
        cb = cost_basis_for_position(
            self._trade_log_path,
            position_key=position_key,
            market_ticker=market_ticker,
        )
        agg = _aggregate_fills(kalshi_fills)
        proceeds = agg.dollars
        exit_fees = agg.fees
        pnl = proceeds - cb.total_dollars - cb.total_fees - exit_fees

        # Local INFO log mirrors the alert so a silent Telegram drop (rate-limit
        # or notifier unavailable) is still observable via journald.
        logger.info(
            "shadow_manual_close_detected ticker=%s pos=%s oid=%s "
            "pnl_dollars=%.2f cost_basis_dollars=%.2f proceeds_dollars=%.2f fees_dollars=%.2f",
            market_ticker, position_key, order_id,
            pnl, cb.total_dollars, proceeds, cb.total_fees + exit_fees,
        )
        await self._safe_alert(
            AlertLevel.WARNING,
            f"[SHADOW] Manual close detected (NOT applied): "
            f"{market_ticker} pos={position_key} close_oid={order_id} "
            f"computed_pnl=${pnl:+.2f} (cost_basis=${cb.total_dollars:.2f}, "
            f"proceeds=${proceeds:.2f}, fees=${cb.total_fees + exit_fees:.2f})",
        )
        self._shakedown_seen_oids.add(order_id)
        self._save_shakedown_seen_oids()
        return ApplyOutcome.SHAKEDOWN_ALERTED

    def _load_shakedown_seen_oids(self) -> None:
        """Load persisted shakedown dedup set from disk."""
        path = getattr(self, "_shakedown_seen_oids_path", None)
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "seen_oids" in data:
                self._shakedown_seen_oids = set(data["seen_oids"])
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load shakedown_seen_oids; starting empty")
            # Defensive: corrupt file → queue a startup alert so operator notices.
            self._pending_startup_alerts.append((
                AlertLevel.WARNING,
                "shakedown_seen_oids.json failed to parse; starting with empty "
                "dedup set. Historical shadow alerts may re-fire.",
            ))

    def _save_shakedown_seen_oids(self) -> None:
        """Atomic write of shakedown dedup set.

        Uses tempfile + os.replace. The replace is the atomic step; the temp
        write is fsync-less because shakedown dedup is recoverable (worst case:
        one extra re-alert on next tick).
        """
        path = getattr(self, "_shakedown_seen_oids_path", None)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            payload = {
                "schema_version": 1,
                "seen_oids": sorted(self._shakedown_seen_oids),
            }
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            logger.exception("Failed to persist shakedown_seen_oids")

    async def _handle_shakedown_cash_out(
        self,
        *,
        order_id: str,
        kalshi_fills: list[dict[str, Any]],
        position_key: str,
        market_ticker: str,
        tracked_cash_out_oids: set[str],
    ) -> ApplyOutcome:
        """Shakedown mode for cash-out (flag off).

        Mirrors _handle_shakedown: classify + compute would-be P&L + WARNING
        alert + persistent seen-oid dedup. No state mutation.
        """
        if order_id in self._shakedown_seen_oids_cash_out:
            return ApplyOutcome.SHAKEDOWN_SUPPRESSED
        if order_id in tracked_cash_out_oids:
            return ApplyOutcome.SHAKEDOWN_SUPPRESSED

        cb = cost_basis_for_position(
            self._trade_log_path,
            position_key=position_key,
            market_ticker=market_ticker,
        )

        # Compute would-be P&L (cash-out formula: collateral_return - basis - fees - hedge).
        agg = _aggregate_fills(kalshi_fills)
        contracts_offset = agg.contracts
        hedge_proceeds = agg.dollars
        hedge_fees = agg.fees

        # For shadow: just compute the would-be P&L treating contracts_offset's
        # share of basis (note: doesn't allocate via partial_basis to keep the
        # shadow path side-effect free — math is correct as long as
        # contracts_offset <= cb.total_contracts).
        if cb.total_contracts > 0:
            basis_share = cb.total_dollars * contracts_offset / cb.total_contracts
            fees_share = cb.total_fees * contracts_offset / cb.total_contracts
        else:
            basis_share = 0.0
            fees_share = 0.0
        pnl = contracts_offset * 1.00 - basis_share - fees_share - hedge_proceeds - hedge_fees

        # Local INFO log mirrors the alert so a silent Telegram drop (rate-limit
        # or notifier unavailable) is still observable via journald.
        logger.info(
            "shadow_cash_out_detected ticker=%s pos=%s oid=%s "
            "pnl_dollars=%.2f basis_share_dollars=%.2f "
            "hedge_proceeds_dollars=%.2f fees_dollars=%.2f",
            market_ticker, position_key, order_id,
            pnl, basis_share, hedge_proceeds, fees_share + hedge_fees,
        )
        await self._safe_alert(
            AlertLevel.WARNING,
            f"[SHADOW] Cash out detected (NOT applied): {market_ticker} "
            f"pos={position_key} close_oid={order_id} computed_pnl=${pnl:+.2f} "
            f"(basis_share=${basis_share:.2f}, hedge_proceeds=${hedge_proceeds:.2f}, "
            f"fees=${(fees_share + hedge_fees):.2f})",
        )
        self._shakedown_seen_oids_cash_out.add(order_id)
        self._save_shakedown_seen_oids_cash_out()
        return ApplyOutcome.SHAKEDOWN_ALERTED

    def _load_shakedown_seen_oids_cash_out(self) -> None:
        """Load persisted shakedown dedup set for cash-out."""
        path = self._shakedown_seen_oids_cash_out_path
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "seen_oids" in data:
                self._shakedown_seen_oids_cash_out = set(data["seen_oids"])
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load shakedown_seen_oids_cash_out; starting empty")
            self._pending_startup_alerts.append((
                AlertLevel.WARNING,
                "shakedown_seen_oids_cash_out.json failed to parse; starting "
                "with empty dedup set. Historical shadow alerts may re-fire.",
            ))

    def _save_shakedown_seen_oids_cash_out(self) -> None:
        """Atomic write of cash-out shakedown dedup set."""
        path = self._shakedown_seen_oids_cash_out_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            payload = {
                "schema_version": 1,
                "seen_oids": sorted(self._shakedown_seen_oids_cash_out),
            }
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
        except OSError:
            logger.exception("Failed to persist shakedown_seen_oids_cash_out")

    def _derive_position_key_for_ticker(self, market_ticker: str) -> str:
        """Find the position_key for a market_ticker by scanning engine._orders.

        The naive `position_key = market_ticker` heuristic fails for any
        strategy that constructs position_key as `f"{ticker}-{side}"` or
        similar.

        Returns the first non-prunable record's position_key; falls back to
        the bare ticker when no record exists (cold restart with positions
        not yet hydrated).
        """
        engine = self._engine
        if engine is None:
            return market_ticker
        for rec in engine._orders.values():
            sig = getattr(rec, "signal", None)
            if sig is None:
                continue
            if sig.market_ticker == market_ticker:
                return str(sig.position_key)
        return market_ticker

    def _build_open_positions_by_ticker_side(self) -> dict[tuple[str, str], float]:
        """Build {(ticker, side): exposure_dollars} snapshot once per tick.

        For each position_key in risk._positions, walk engine._orders for an
        OrderRecord whose signal.position_key matches AND status is not terminal.
        Keys by (market_ticker, side.upper()); values are the exposure dollars.

        If no live record is found for a position (cold-start where the position
        came from risk.hydrate_from_state before engine.hydrate_live_positions), the
        entry is OMITTED — fail-safe: uncertain-side positions don't trigger
        cash-out detection.
        """
        result: dict[tuple[str, str], float] = {}
        engine = self._engine
        if engine is None:
            return result
        risk = getattr(engine, "_risk", None)
        if risk is None:
            return result
        # Use the public `positions` property (returns dict[str, float]) so this
        # survives a future copy-on-read refactor of RiskManager.
        positions = (
            risk.positions if hasattr(risk, "positions") else getattr(risk, "_positions", {})
        )

        for position_key, exposure_dollars in positions.items():
            for rec in engine._orders.values():
                sig = getattr(rec, "signal", None)
                if sig is None:
                    continue
                if sig.position_key != position_key:
                    continue
                if rec.status in _TERMINAL_ORDER_STATUSES:
                    continue
                side_str = (
                    sig.side.value.upper()
                    if hasattr(sig.side, "value")
                    else str(sig.side).upper()
                )
                result[(sig.market_ticker, side_str)] = float(exposure_dollars)
                break  # first non-terminal record wins
        return result

    def _decrement_fill_counts_for_position(
        self,
        *,
        position_key: str,
        contracts_offset: int,
        cb_total_contracts: int,
    ) -> None:
        """Proportionally reduce fill_count on matching OrderRecords (partial cash-out).

        Precondition: contracts_offset < cb_total_contracts (full closes use
        close_position + EXTERNALLY_CLOSED stamping; this path is partial-only).
        """
        assert contracts_offset < cb_total_contracts, (
            f"_decrement_fill_counts_for_position called on full close: "
            f"contracts_offset={contracts_offset}, cb_total_contracts={cb_total_contracts}"
        )

        engine = self._engine
        if engine is None:
            logger.critical(
                "_decrement_fill_counts_for_position: engine is None for "
                "position_key=%s contracts_offset=%d — residual settlement "
                "attribution at risk; investigate.",
                position_key,
                contracts_offset,
            )
            return
        matches = [
            r
            for r in engine._orders.values()
            if r.signal.position_key == position_key
            and r.status not in _TERMINAL_ORDER_STATUSES
            and isinstance(r.fill_count, int)
            and r.fill_count > 0
        ]
        if not matches:
            logger.critical(
                "_decrement_fill_counts_for_position: no matching live "
                "OrderRecords for position_key=%s contracts_offset=%d — "
                "residual settlement attribution at risk; investigate.",
                position_key,
                contracts_offset,
            )
            return

        sum_existing = sum(r.fill_count for r in matches)
        if sum_existing < contracts_offset:
            logger.critical(
                "_decrement_fill_counts_for_position: sum(fill_count)=%d < "
                "contracts_offset=%d for position_key=%s — clamping each record "
                "at 0; residual settlement attribution at risk.",
                sum_existing,
                contracts_offset,
                position_key,
            )
            for r in matches:
                old = r.fill_count
                r.fill_count = 0
                if r.fees_from_api is not None and old > 0:
                    r.fees_from_api = 0.0
            return

        if len(matches) == 1:
            rec = matches[0]
            old_count = rec.fill_count
            rec.fill_count = max(0, old_count - contracts_offset)
            if rec.fees_from_api is not None and old_count > 0:
                rec.fees_from_api = round(
                    rec.fees_from_api * (rec.fill_count / old_count), 4
                )
            return

        # Multi-record: proportional allocation.
        decrements: list[int] = [
            round(contracts_offset * r.fill_count / sum_existing) for r in matches
        ]
        drift = contracts_offset - sum(decrements)
        if drift != 0:
            largest_idx = max(range(len(matches)), key=lambda i: matches[i].fill_count)
            decrements[largest_idx] += drift
        for r, dec in zip(matches, decrements, strict=False):
            old_count = r.fill_count
            r.fill_count = max(0, old_count - dec)
            if r.fees_from_api is not None and old_count > 0:
                r.fees_from_api = round(
                    r.fees_from_api * (r.fill_count / old_count), 4
                )

        # Post-mutation invariant: clamp can theoretically leave sum != expected
        # under pathological inputs. Log CRITICAL; don't crash (row is already
        # on disk; divergence is forensic).
        expected_residual = cb_total_contracts - contracts_offset
        actual_residual = sum(r.fill_count for r in matches)
        if actual_residual != expected_residual:
            logger.critical(
                "_decrement_fill_counts_for_position: invariant violated for "
                "position_key=%s — expected sum(fill_count)=%d, got %d. "
                "Residual settlement attribution may diverge by %d contracts.",
                position_key,
                expected_residual,
                actual_residual,
                actual_residual - expected_residual,
            )

    def _safe_resolve_strategy(
        self, order_id: str, kalshi_fill: dict[str, Any]
    ) -> tuple[str, str]:
        """Wrapper around the configured resolver that catches exceptions.

        A resolver that raises must NOT bubble out of tick(). The synthetic row
        is still written with strategy='unknown' so the operator can investigate
        via the unresolved-row Telegram alert.
        """
        try:
            return self._strategy_resolver(order_id, kalshi_fill)
        except Exception as e:
            logger.warning(
                "Strategy resolver raised for order_id=%s (%s); treating as unresolved",
                order_id,
                type(e).__name__,
                exc_info=True,
            )
            return ("unknown", "unresolved")

    def _resolve_strategy_with_basis_fallback(
        self, order_id: str, kalshi_fill: dict[str, Any], cb: CostBasis
    ) -> tuple[str, str, bool]:
        """Resolve the strategy, falling back to the cost basis's last strategy.

        Returns (strategy, method, fell_back).
        """
        strategy, method = self._safe_resolve_strategy(order_id, kalshi_fill)
        if method == "unresolved" and cb.last_strategy:
            return cb.last_strategy, "cost_basis_fallback", True
        return strategy, method, False

    async def backfill(
        self,
        *,
        min_ts: int,
        max_ts: int,
        allowlist: set[str] | None = None,
        expect_n: int | None = None,
        allow_extras: bool = False,
        force_strategy: str | None = None,
        dry_run: bool = True,
    ) -> BackfillResult:
        """Operator-driven backfill.

        Reads /portfolio/fills for [min_ts, max_ts], filters out
        already-tracked order_ids, applies allowlist / expect-n / force-strategy
        guards, and writes synthetic rows (only if dry_run=False).
        """
        fetch = await self._fetch_fills_paginated(min_ts=min_ts)
        fills_in_window = [
            f for f in fetch["fills"]
            if min_ts <= self._parse_iso_to_unix(f.get("created_time", "")) <= max_ts
        ]
        by_oid: dict[str, list[dict[str, Any]]] = {}
        for f in fills_in_window:
            by_oid.setdefault(f.get("order_id", ""), []).append(f)
        by_oid.pop("", None)

        # Single locked critical section: tracked-OID read, guard filters,
        # synthetic-row build, and (apply path) write all happen under one
        # acquisition of trade_log_lock. The dedup invariant ("at most one
        # event=='fill' per order_id") cannot hold if the engine is allowed
        # to append a fill row between our `tracked` snapshot and our write.
        # Mirrors the _tick_impl lock window.
        result: BackfillResult
        async with self._trade_log_lock:
            self._build_placement_index()
            tracked = self._load_tracked_live_order_ids()
            missing = set(by_oid.keys()) - tracked

            if allowlist is not None:
                unexpected = missing - allowlist
                if unexpected and not allow_extras:
                    return BackfillResult(
                        planned_writes=[], written_count=0, skipped_count=0,
                        unresolved_count=0, aborted=True,
                        abort_reason=f"unexpected order_ids: {sorted(unexpected)}",
                    )
                missing = missing & allowlist

            if expect_n is not None and len(missing) != expect_n:
                return BackfillResult(
                    planned_writes=[], written_count=0, skipped_count=0,
                    unresolved_count=0, aborted=True,
                    abort_reason=(
                        f"expected {expect_n} planned writes, got {len(missing)}"
                    ),
                )

            # Build synthetic rows. sorted() — deterministic for operator
            # review and reproducible across runs (sets are hash-randomised).
            planned: list[dict[str, Any]] = []
            unresolved_count = 0
            for oid in sorted(missing):
                kalshi_fills = by_oid[oid]
                strategy, method = self._safe_resolve_strategy(oid, kalshi_fills[0])
                if method == "unresolved":
                    if force_strategy:
                        strategy = force_strategy
                        method = "operator_forced"
                    else:
                        unresolved_count += 1
                row = self._build_synthetic_row(
                    order_id=oid,
                    kalshi_fills=kalshi_fills,
                    strategy=strategy,
                    resolution_method=method,
                )
                if force_strategy and method == "operator_forced":
                    row["metadata"]["strategy_forced_to"] = force_strategy
                planned.append(row)

            if dry_run:
                result = BackfillResult(
                    planned_writes=planned, written_count=0, skipped_count=0,
                    unresolved_count=unresolved_count,
                    aborted=False, abort_reason=None,
                )
            else:
                await self._append_synthetic_rows(planned)
                result = BackfillResult(
                    planned_writes=planned, written_count=len(planned),
                    skipped_count=0, unresolved_count=unresolved_count,
                    aborted=False, abort_reason=None,
                )

        # Outside the lock: Telegram CRITICAL alert when force_strategy was
        # actually applied. Notifier failures must not break the call path;
        # a bare-await would surface the synthetic-row write as an exception
        # even though the rows are already on disk.
        if (
            not dry_run
            and force_strategy
            and self._notifier is not None
        ):
            n_forced = sum(
                1 for r in planned
                if r["metadata"].get("strategy_resolution") == "operator_forced"
            )
            if n_forced > 0:
                try:
                    await self._notifier.send_alert(
                        level=AlertLevel.CRITICAL,
                        message=(
                            f"Backfill applied --force-strategy={force_strategy} "
                            f"to {n_forced} rows; manual override audit trail "
                            f"in {self._state_path}"
                        ),
                    )
                except Exception:
                    logger.exception(
                        "fills_reconciler notifier failed during force_strategy alert"
                    )

        return result

    @staticmethod
    def _parse_iso_to_unix(iso: str) -> int:
        """ISO8601 -> Unix timestamp (seconds). Returns 0 on parse failure."""
        if not iso:
            return 0
        try:
            return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError):
            return 0
