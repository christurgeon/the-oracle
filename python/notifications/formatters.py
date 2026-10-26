"""Plain-text message formatters for Telegram notifications.

All functions return strings ready for Telegram ``parse_mode=None`` (plain text).
Unicode symbols provide visual structure without MarkdownV2 escaping headaches.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from python.common import OrderStatus

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable

    from python.analytics.attribution import AttributionSummary
    from python.analytics.bracket_metrics import BracketSumMetrics
    from python.analytics.calibration_monitor import (
        CalibrationSnapshot,
        DriftAlert,
        MonitorHealth,
    )
    from python.analytics.mtm import PortfolioMtm
    from python.common import Side
    from python.execution.engine import ExecutionEngine, OrderRecord
    from python.risk.manager import RiskManager
    from python.settings import Settings
    from python.strategies.base import Strategy


def format_status(
    risk: RiskManager,
    engine: ExecutionEngine,
    settings: Settings,
    strategies: list[Strategy],
    *,
    mtm: PortfolioMtm | None = None,
    mtm_attempted: bool = False,
) -> str:
    """Overview: mode, bankroll, P&L, exposure, positions, kill switch.

    ``risk.bankroll`` is cost-basis equity (cash + Σ market_exposure_dollars);
    it only moves on settlement. Pass ``mtm`` to additionally render the
    current mark-to-market portfolio value (cash + Σ contracts × bid) and the
    unrealized P&L gap. ``mtm_attempted=True`` with ``mtm is None`` renders
    an "MTM: unavailable" line so the operator can distinguish a failed fetch
    from a deliberately skipped one (shadow/paper).
    """
    exposure_pct = (risk.total_exposure / risk.bankroll * 100) if risk.bankroll > 0 else 0.0
    n_positions = len(risk.positions)

    kill = "ACTIVE" if settings.is_kill_switch_active() else "inactive"
    halted = "YES" if risk.is_halted else "no"

    strat_lines = []
    for s in strategies:
        tracker = risk.get_tracker(s.name)
        status = "ON" if s.enabled and tracker.enabled else "OFF"
        if not tracker.enabled:
            status += f" ({tracker.disabled_reason})"
        strat_lines.append(f"  {s.name}: {status}")
        # EMOS sticky status — only meaningful for the weather strategy.
        # Use getattr so this stays safe before C5/C7 attach `emos_table`.
        if s.name == "weather_edge":
            strat_lines.append(f"    emos_status: {_format_emos_status(s)}")

    effective_cap_pct = risk.effective_max_total_exposure_pct * 100
    baseline_cap_pct = settings.risk.max_total_exposure_pct * 100
    if risk.exposure_override is not None:
        cap_line = (
            f"  Exposure cap: {effective_cap_pct:.1f}% (override; "
            f"baseline {baseline_cap_pct:.1f}%)\n"
        )
    else:
        cap_line = f"  Exposure cap: {effective_cap_pct:.1f}%\n"

    bankroll_block = f"  Bankroll: ${risk.bankroll:,.2f}\n"
    if mtm is not None:
        unrealized = mtm.equity_mtm - risk.bankroll
        if mtm.degraded:
            tag = (
                f"   [partial: {len(mtm.breakdown)}/"
                f"{mtm.n_positions_fetched} priced]"
            )
        else:
            tag = ""
        bankroll_block += (
            f"  MTM:      ${mtm.equity_mtm:,.2f}{tag}\n"
            f"  Unrealized: {_signed_dollars(unrealized)}\n"
        )
    elif mtm_attempted:
        bankroll_block += "  MTM: unavailable (Kalshi fetch failed or timed out)\n"

    return (
        f"THE ORACLE STATUS\n"
        f"  Mode: {engine.mode.value}\n"
        f"{bankroll_block}"
        f"  Daily P&L: {_signed_dollars(risk.daily_pnl)}\n"
        f"  Drawdown: {risk.drawdown:.1%}\n"
        f"  Exposure: ${risk.total_exposure:,.2f} ({exposure_pct:.1f}%)\n"
        f"{cap_line}"
        f"  Open positions: {n_positions}\n"
        f"  Kill switch: {kill}\n"
        f"  Trading halted: {halted}\n"
        f"\nStrategies:\n" + "\n".join(strat_lines)
    )


def format_positions(risk: RiskManager, engine: ExecutionEngine) -> str:
    """List open positions across three buckets.

    ``risk.positions`` is the source of truth for total exposure.
    ``engine.order_history`` is partitioned per ``position_key`` into:

      * filled records (any non-HYDRATED, non-SETTLED record carrying
        actual fill data: ``fill_count > 0`` and ``fill_price`` set).
        Includes terminal-status records such as CANCELLED or EXPIRED
        that filled partially before the resting portion was cancelled
        — those filled contracts are still real exposure on Kalshi until
        the market settles.
      * hydrated stubs (HYDRATED — re-attached on restart by
        ``ExecutionEngine.hydrate_live_positions``; carry no fill metadata)
      * pending records (``PENDING_STATUSES`` — RESTING / PENDING / PARTIAL
        still working on the book)

    A position may surface in multiple buckets (e.g. a PARTIAL fill plus
    its resting remainder, or a HYDRATED stub plus subsequent post-restart
    fills). Tracked exposure that maps to none of the three buckets is
    surfaced under ``WARNING: untracked exposure`` — this indicates the
    risk manager holds dollars for a position the engine has no fill data
    for, suggesting either a pruned record without prior exposure release
    or a state-divergence bug worth investigating.
    """
    from collections import defaultdict

    from python.execution.engine import PENDING_STATUSES

    positions = risk.positions
    if not positions:
        return "No open positions."

    # Records whose fill data must NOT contribute to the filled bucket:
    #   HYDRATED — handled separately by the hydrated bucket (it tracks
    #     exposure via ``last_recorded_dollars`` rather than fill_count ×
    #     fill_price). HYDRATED stubs nominally have fill_count == 0 so
    #     they'd fail the fill-data gate anyway; the exclusion is
    #     defense-in-depth + documentation.
    #   SETTLED — the record's fill_count × fill_price has already been
    #     credited to bankroll via ``_risk.close_position(..., pnl=...)``;
    #     if the same position_key reappears in ``risk.positions`` (a new
    #     trade after settlement on a re-trading market), only the newer
    #     non-SETTLED records should contribute to displayed exposure.
    _excluded_from_filled = {OrderStatus.HYDRATED, OrderStatus.SETTLED}

    by_key: dict[str, list[OrderRecord]] = defaultdict(list)
    for record in engine.order_history:
        by_key[record.signal.position_key].append(record)

    filled_lines: list[str] = []
    working_lines: list[str] = []
    hydrated_lines: list[str] = []
    untracked_lines: list[str] = []
    total_filled = 0.0
    total_working = 0.0
    total_hydrated = 0.0

    for position_key in sorted(positions):
        total_dollars = positions[position_key]
        records = by_key.get(position_key, [])

        filled_records = [
            r for r in records
            if r.status not in _excluded_from_filled
            and r.fill_price is not None
            and r.fill_count > 0
        ]
        hydrated_records = [r for r in records if r.status == OrderStatus.HYDRATED]
        pending_records = [r for r in records if r.status in PENDING_STATUSES]

        filled_dollars: float = sum(
            r.fill_count * r.fill_price
            for r in filled_records
            if r.fill_price is not None
        )
        hydrated_dollars: float = sum(
            r.last_recorded_dollars for r in hydrated_records
        )
        # Defensive clamp: rounding from many small fills could push the
        # synthesized cost basis above the tracked exposure.
        filled_dollars = min(filled_dollars, total_dollars)
        hydrated_dollars = min(hydrated_dollars, total_dollars - filled_dollars)
        working_dollars = max(0.0, total_dollars - filled_dollars - hydrated_dollars)
        if working_dollars < 0.005:  # under display rounding ($0.01)
            working_dollars = 0.0

        if filled_dollars > 0 and filled_records:
            rep = max(filled_records, key=lambda r: r.timestamp)
            detail = (
                f" ({rep.signal.side.value}, {rep.fill_count} contracts "
                f"@ ${rep.fill_price:.2f})"
            )
            filled_lines.append(f"  {position_key}: ${filled_dollars:,.2f}{detail}")
            total_filled += filled_dollars

        if hydrated_dollars > 0 and hydrated_records:
            rep_h = max(hydrated_records, key=lambda r: r.timestamp)
            rfi = rep_h.restored_fill_info
            if rfi is not None:
                detail = f" ({rep_h.signal.side.value}, {rfi[1]} contracts @ ${rfi[0]:.2f})"
            else:
                detail = f" ({rep_h.signal.side.value})"
            hydrated_lines.append(f"  {position_key}: ${hydrated_dollars:,.2f}{detail}")
            total_hydrated += hydrated_dollars

        if working_dollars > 0:
            if pending_records:
                rep_w = max(pending_records, key=lambda r: r.timestamp)
                remaining = max(rep_w.requested_count - rep_w.fill_count, 0)
                price = rep_w.order_price if rep_w.order_price > 0 else 0.0
                detail = (
                    f" reserved ({rep_w.signal.side.value}, "
                    f"{remaining} contracts @ ${price:.2f})"
                )
                working_lines.append(
                    f"  {position_key}: ${working_dollars:,.2f}{detail}"
                )
                total_working += working_dollars
            else:
                untracked_lines.append(
                    f"  {position_key}: ${working_dollars:,.2f} "
                    f"(untracked — investigate)"
                )

    lines: list[str] = []
    sections = (
        ("OPEN POSITIONS (filled)", filled_lines),
        ("WORKING ORDERS (resting / pending)", working_lines),
        ("OPEN POSITIONS (restored on restart)", hydrated_lines),
        ("WARNING: untracked exposure", untracked_lines),
    )
    for header, body in sections:
        if not body:
            continue
        if lines:
            lines.append("")
        lines.append(header)
        lines.extend(body)

    lines.append("")
    lines.append(f"Total filled:   ${total_filled:,.2f}")
    lines.append(f"Total working:  ${total_working:,.2f}")
    if total_hydrated > 0:
        lines.append(f"Total recovered: ${total_hydrated:,.2f}")
    lines.append(f"Total exposure: ${risk.total_exposure:,.2f}")
    return "\n".join(lines)


def format_trades(engine: ExecutionEngine, limit: int = 10) -> str:
    """Last N executed trades (filled or shadow)."""
    records = [
        r for r in engine.order_history
        if r.status in (OrderStatus.FILLED, OrderStatus.SHADOW, OrderStatus.PARTIAL)
    ]
    if not records:
        return "No trades yet."

    records.sort(key=lambda r: r.timestamp, reverse=True)
    records = records[:limit]

    lines = [f"RECENT TRADES (last {len(records)})"]
    for r in records:
        side = r.signal.side.value
        ticker = r.signal.market_ticker
        price = r.fill_price if r.fill_price is not None else r.order_price
        edge = r.signal.edge
        lines.append(
            f"  {side} {ticker}\n"
            f"    {r.fill_count} contracts @ ${price:.2f} | "
            f"edge {edge:.1%} | {r.signal.strategy_name} | {r.mode.value}"
        )
    return "\n".join(lines)


def format_signals(engine: ExecutionEngine, limit: int = 10) -> str:
    """Last N signals (all statuses except HYDRATED restart-recovery stubs)."""
    records = [r for r in engine.order_history if r.status != OrderStatus.HYDRATED]
    if not records:
        return "No signals yet."

    records.sort(key=lambda r: r.timestamp, reverse=True)
    records = records[:limit]

    lines = [f"RECENT SIGNALS (last {len(records)})"]
    for r in records:
        side = r.signal.side.value
        ticker = r.signal.market_ticker
        edge = r.signal.edge
        status = r.status.value.upper()
        lines.append(
            f"  {status} | {side} {ticker} | "
            f"edge {edge:.1%} | {r.signal.strategy_name}"
        )
    return "\n".join(lines)


def format_risk(risk: RiskManager, settings: Settings) -> str:
    """Risk gauges: exposure, loss, drawdown, circuit breakers."""
    r = settings.risk
    exposure_pct = (risk.total_exposure / risk.bankroll) if risk.bankroll > 0 else 0.0
    daily_loss_pct = (-risk.daily_pnl / risk.bankroll) if risk.bankroll > 0 else 0.0

    kill = "ACTIVE" if settings.is_kill_switch_active() else "inactive"
    halted = "YES" if risk.is_halted else "no"

    exposure_cap = risk.effective_max_total_exposure_pct
    override_tag = " (override)" if risk.exposure_override is not None else ""
    lines = [
        "RISK STATUS",
        f"  Exposure: {exposure_pct:.1%} / {exposure_cap:.0%}{override_tag}",
        f"  Daily loss: {daily_loss_pct:.1%} / {r.max_daily_loss_pct:.0%}",
        f"  Drawdown: {risk.drawdown:.1%} / {r.max_drawdown_pct:.0%}",
        f"  Kill switch: {kill}",
        f"  Trading halted: {halted}",
    ]
    return "\n".join(lines)


def format_trade_alert(record: OrderRecord) -> str:
    """Push notification for a single trade execution."""
    sig = record.signal
    price = record.fill_price if record.fill_price is not None else record.order_price
    notional = record.fill_count * price
    return (
        f"TRADE EXECUTED\n"
        f"  Market: {sig.market_ticker}\n"
        f"  Side: {sig.side.value}\n"
        f"  Contracts: {record.fill_count} @ ${price:.2f}\n"
        f"  Total: ${notional:.2f}\n"
        f"  Edge: {sig.edge:.1%}\n"
        f"  Strategy: {sig.strategy_name}\n"
        f"  Mode: {record.mode.value}"
    )


def format_bracket_alert(metrics: BracketSumMetrics) -> str:
    """Push notification for a bracket sum anomaly."""
    net10 = metrics.net_profit_by_count.get(10, 0.0)
    lines = [
        f"BRACKET SUM ALERT: {metrics.event_ticker}",
        f"  {metrics.city} {metrics.forecast_date} | {metrics.n_brackets} brackets",
        f"  bid_sum: ${metrics.bid_sum:.3f}  ask_sum: ${metrics.ask_sum:.3f}",
        f"  bid_excess: {_signed_dollars(metrics.bid_excess)}",
        f"  spread_sum: ${metrics.spread_sum:.3f}",
        f"  min_bid_depth: {metrics.min_bid_depth}  min_ask_depth: {metrics.min_ask_depth}",
        f"  net@10 (maker): {_signed_dollars(net10)}",
    ]
    if metrics.n_missing_bids > 0 or metrics.n_missing_asks > 0:
        lines.append(
            f"  missing: {metrics.n_missing_bids} bids, {metrics.n_missing_asks} asks"
        )
    return "\n".join(lines)


def format_settlement_alert(ticker: str, result: str, pnl: float, bankroll: float) -> str:
    """Push notification for a single position settlement."""
    outcome = "WON" if pnl > 0 else ("LOST" if pnl < 0 else "FLAT")
    return (
        f"SETTLED: {ticker}\n"
        f"  Result: {result.upper()} | {outcome}\n"
        f"  P&L: ${pnl:+.2f}\n"
        f"  Bankroll: ${bankroll:,.2f}"
    )


def format_risk_alert(event: str, details: str) -> str:
    """Push notification for a risk event."""
    return f"RISK ALERT: {event}\n  {details}"


def format_help() -> str:
    """Help text listing all available commands."""
    return (
        "THE ORACLE COMMANDS\n"
        "  /status         - Bankroll, P&L, mode, strategies\n"
        "  /positions      - Open positions\n"
        "  /trades         - Recent executed trades\n"
        "  /signals        - Recent signals (all statuses)\n"
        "  /risk           - Risk gauges and limits\n"
        "  /attribution    - P&L decomposition (opt: 7d|30d|all)\n"
        "  /calibhealth    - Calibration monitor health\n"
        "  /ack <code>     - Record 'reviewed, no action' for an alert code\n"
        "  /acks           - List acknowledgements + alert-suppression counts\n"
        "  /kill           - Activate kill switch (immediate)\n"
        "  /unkill         - Deactivate kill switch (requires restart)\n"
        "  /disable_reconciler - Pause the fills reconciler (process-local)\n"
        "  /enable_reconciler  - Re-arm the fills reconciler\n"
        "  /enable <s>     - Re-enable a circuit-broken strategy\n"
        "  /setexposure N  - Set total-exposure cap (0<N<=100 pct, or 0<N<=1 frac)\n"
        "  /resetexposure  - Clear override; revert to config baseline\n"
        "  /loglevel       - Show/set temp log level (debug <1-30>m | info)\n"
        "  /agent          - AI diagnosis (optional: freeform query)\n"
        "  /help           - This message"
    )


# ---------------------------------------------------------------------------
# P&L attribution
# ---------------------------------------------------------------------------


def _fmt_money(v: float) -> str:
    if math.isnan(v):
        return "—"
    return f"${v:+,.2f}"


def _fmt_plain_money(v: float) -> str:
    if math.isnan(v):
        return "—"
    return f"${v:,.2f}"


def format_attribution(
    window_label: str,
    summary: AttributionSummary,
    by_side: dict[Side, AttributionSummary],
    by_city: dict[str, AttributionSummary],
) -> str:
    """P&L decomposition: predicted_alpha + model_error - slippage - fees."""
    if summary.n == 0:
        return f"ATTRIBUTION ({window_label})\n  No settled trades in window."

    lines = [
        f"ATTRIBUTION ({window_label})",
        f"  Trades: {summary.n} | Wins: {summary.wins} "
        f"({summary.win_rate_pct:.1f}%) | Contracts: {summary.total_contracts}",
        f"  Wagered: {_fmt_plain_money(summary.total_wagered)}",
        # Headline = recorded (realized) P&L; the decomposition below sums to the
        # decomposed total, which the reconciliation-drift line reconciles.
        f"  Net P&L: {_fmt_money(summary.total_recorded_pnl)} "
        f"(ROI {summary.roi_recorded_pct:+.2f}%)",
        "",
        "  Decomposition:",
    ]
    if summary.n_full_mode > 0:
        lines.append(
            f"    predicted_alpha: {_fmt_money(summary.total_predicted_alpha)} "
            f"(edge at mid, n_full={summary.n_full_mode})"
        )
        lines.append(
            f"    slippage_cost:   {_fmt_money(-summary.total_slippage_cost)} "
            f"(paid over mid)"
        )
    # True degraded rows = non-full minus recorded-only; only these back the
    # decomposable alpha_at_exec subtotal (recorded-only rows are NaN-filtered out).
    n_degraded_decomposable = summary.n_degraded_mode - summary.n_recorded_only
    if n_degraded_decomposable > 0:
        lines.append(
            f"    alpha_at_exec:   {_fmt_money(summary.total_predicted_alpha_at_exec)} "
            f"(n_degraded={n_degraded_decomposable}; no mid recorded)"
        )
    lines.append(f"    model_error:     {_fmt_money(summary.total_model_error)}")
    lines.append(f"    fees:            {_fmt_money(-summary.total_fees)}")
    if summary.n_recorded_only > 0:
        # Reconciler-synthetic rows with no model_probability/price — recorded
        # P&L only, not decomposable. Shown so the block closes to the headline.
        lines.append(
            f"    recorded_only:   {_fmt_money(summary.total_recorded_only_pnl)} "
            f"(n={summary.n_recorded_only}; not decomposable)"
        )

    if abs(summary.total_reconciliation_diff) > 0.01:
        lines.append(
            f"  !! Reconciliation drift: {_fmt_money(summary.total_reconciliation_diff)}"
        )

    if by_side:
        lines.append("")
        lines.append("  By side:")
        for side, s in by_side.items():
            lines.append(
                f"    {side.value:3s}: n={s.n:3d} pnl={_fmt_money(s.total_recorded_pnl)} "
                f"(ROI {s.roi_recorded_pct:+.1f}%)"
            )

    if by_city:
        lines.append("")
        lines.append("  By city:")
        for city, s in sorted(by_city.items(), key=lambda kv: -kv[1].total_recorded_pnl):
            lines.append(
                f"    {city:8s}: n={s.n:3d} pnl={_fmt_money(s.total_recorded_pnl)} "
                f"(ROI {s.roi_recorded_pct:+.1f}%)"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration monitor
# ---------------------------------------------------------------------------


def format_calibration_health(
    snapshot: CalibrationSnapshot | None, health: MonitorHealth
) -> str:
    """Self-observability of the calibration monitor."""
    lines = ["CALIBRATION MONITOR"]
    if snapshot is None:
        lines.append("  (no snapshot yet — waiting for first refresh)")
    else:
        lines.append(
            f"  Last refresh: {snapshot.generated_at.strftime('%Y-%m-%d %H:%MZ')}"
        )
        lines.append(f"  {snapshot.summary_line()}")
    lines.append(
        f"  Ticks: {health.total_ticks} ({health.total_errors} errors, "
        f"{health.consecutive_errors} consecutive)"
    )
    if health.last_error:
        lines.append(f"  Last error: {health.last_error}")
    if health.last_successful_tick_ts:
        lines.append(
            f"  Last success: {health.last_successful_tick_ts.strftime('%Y-%m-%d %H:%MZ')}"
        )
    return "\n".join(lines)


def format_calibration_drift_alert(alert: DriftAlert) -> str:
    prefix = "CALIBRATION DRIFT"
    return f"{prefix} [{alert.severity.upper()}] {alert.code}\n  {alert.message}"


# ---------------------------------------------------------------------------
# Alert acknowledgements
# ---------------------------------------------------------------------------


def format_acks(
    acked: dict[str, dict[str, Any]],
    owner_is_live: Callable[[str | None], bool],
    suppressions: Counter[tuple[str, str]],
    ackable: dict[str, tuple[str | None, str]] | None = None,
) -> str:
    """List human-recorded /ack verdicts and alert-suppression counters.

    An ack taken while its owner was NOT live is dropped the moment that
    owner goes live (the operator's judgment was made in a no-money
    context) — flagged INACTIVE here rather than listed as if still in
    force. See ``Orchestrator._ack_suppresses`` for the transition rule
    this mirrors.

    ``ackable`` (``Orchestrator._ackable``, spec §3.2) lists every code seen
    this process that ``/ack`` would actually accept — the codes
    ``telegram_bot.py``'s ``/ack`` rejection message points the operator at
    ("See /acks for codes seen so far"). Without this section that pointer
    led nowhere: ``/acks`` printed only ``acked``/``suppressions``, neither
    of which shows what is *available* to ack. Optional (default ``None``)
    for callers that predate this parameter; omits the section instead of
    listing nothing when a caller passes an empty dict deliberately.
    """
    lines = ["ACKNOWLEDGEMENTS"]
    if not acked:
        lines.append("  (none recorded)")
    else:
        for code, entry in sorted(acked.items()):
            owner = entry.get("owner")
            severity = entry.get("severity", "?")
            ts = entry.get("ts", "?")
            owner_live_at_ack = bool(entry.get("owner_live_at_ack", False))
            inactive = not owner_live_at_ack and owner_is_live(owner)
            status = " [INACTIVE - owner now live]" if inactive else ""
            lines.append(f"  {code}: {severity} owner={owner or '-'} acked={ts}{status}")

    if ackable is not None:
        lines.append("")
        lines.append("ACKNOWLEDGEABLE (seen this process)")
        if not ackable:
            lines.append("  (none)")
        else:
            for code, (owner, severity) in sorted(ackable.items()):
                lines.append(f"  {code}: {severity} owner={owner or '-'}")

    lines.append("")
    lines.append("SUPPRESSED (owner not live, or acknowledged)")
    if not suppressions:
        lines.append("  (none)")
    else:
        for (code, owner), count in suppressions.most_common():
            lines.append(f"  {code} owner={owner or '-'}: {count}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signed_dollars(value: float) -> str:
    """Format a dollar amount with explicit sign."""
    if value >= 0:
        return f"+${value:,.2f}"
    return f"-${abs(value):,.2f}"


def _format_emos_status(strat: Strategy) -> str:
    """Render the sticky EMOS status line for the weather strategy.

    Uses ``getattr`` because PR-2 tasks C5/C7 (which attach ``emos_table``
    to the strategy) may not have landed when this code runs. A missing
    attribute is treated identically to ``emos_table is None`` — i.e.
    "off (file missing)".
    """
    table = getattr(strat, "emos_table", None)
    if table is None:
        return "off (file missing)"
    if getattr(table, "is_loaded", False):
        artifact_hash = getattr(table, "artifact_hash", "") or ""
        cells = getattr(table, "cells", None) or ()
        return f"on ({artifact_hash[:8]}, n_cells={len(cells)})"
    fallback = getattr(table, "fallback_reason", "unknown") or "unknown"
    return f"off ({fallback})"


def format_emos_writer_counters(strat: Strategy) -> str | None:
    """Return the diagnose-only writer counters line, or ``None`` if absent.

    The forecasts-raw writer is attached by PR-2 task C7. Until that lands,
    or if the writer was disabled, this returns ``None`` so callers can
    skip the line cleanly.
    """
    writer = getattr(strat, "_forecasts_raw_writer", None)
    if writer is None:
        return None
    success_count = getattr(writer, "success_count", 0)
    last_error_at = getattr(writer, "last_error_at", None)
    return (
        f"forecasts_raw write_count={success_count} "
        f"last_error_at={last_error_at}"
    )
