"""
Central risk controller for The Oracle.

Validates every trade before execution against hard limits, tracks exposure
and drawdown, and enforces per-strategy circuit breakers and the kill switch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from python.risk.persistence import (
    RiskStateSnapshot,
    load_exposure_override,
    save_exposure_override,
)
from python.settings import Settings
from python.strategies.base import TradeSignal
from python.utils.ticker_dates import parse_ticker_date

logger = logging.getLogger(__name__)


SAVE_FAILURE_ALERT_THRESHOLD = 5
"""Number of consecutive save_state failures before a CRITICAL alert fires.

Tick interval is 60s (config.poll_interval_seconds), so 5 failures = ~5
minutes of broken persistence — short enough to act before the next
day-boundary reset, long enough to absorb a transient FS hiccup."""


STRATEGY_ERROR_ALERT_THRESHOLD = 10
"""Number of consecutive signal-generation errors before a CRITICAL alert fires.

Below this threshold the orchestrator emits WARNING per failure. The
counter is persisted so a long-burning failure across restarts
still trips the threshold."""


PERSISTENCE_BROKEN_ALERT_PREFIX = "PERSISTENCE BROKEN"
"""Log/alert tag for save-failure CRITICAL alerts.

Matches the ``KILL_SWITCH_*_PREFIX`` convention in execution.engine — used
by operator log-aggregation rules and by tests asserting the alert path."""


def _correlated_event_count(position_key: str, open_keys: Iterable[str]) -> int:
    """Count distinct OTHER forecast dates held in the same weather series.

    Weather position keys are ``<series>-<date>-<bracket>`` (e.g.
    ``KXHIGHAUS-26APR22-B79.5``: series ``KXHIGHAUS`` = variable+city, date
    ``26APR22``, bracket ``B79.5``).

    Brackets within one ``(series, date)`` event are *mutually exclusive*
    outcomes (the daily extreme lands in exactly one), so they collapse to a
    single unit and the candidate's own event is excluded. The returned count is
    the number of distinct OTHER dates currently held in the same series — i.e.
    same-city/same-variable exposure spread across forecast days, the dominant
    weather correlation (a heat/cold spell persists across days).

    Cross-city (air-mass) and HIGH-vs-LOW correlation are intentionally NOT
    modeled here: both need an empirical adjacency model and are unchanged from
    prior behavior. Non-weather keys (not ``KX…`` or fewer than 3 ``-``-parts)
    return 0 — the cap does not apply to them.
    """
    parts = position_key.split("-")
    if not position_key.upper().startswith("KX") or len(parts) < 3:
        return 0
    series, date = parts[0], parts[1]
    other_dates: set[str] = set()
    for key in open_keys:
        kp = key.split("-")
        if len(kp) >= 3 and kp[0] == series and kp[1] != date:
            other_dates.add(kp[1])
    return len(other_dates)


@dataclass(frozen=True)
class RiskDecision:
    """Result of a risk check on a single signal."""

    approved: bool
    signal: TradeSignal
    adjusted_dollars: float  # may be reduced from signal.suggested_dollars
    rejection_reasons: list[str] = field(default_factory=list)
    size_adjustments: list[str] = field(default_factory=list)


def _reject(
    signal: TradeSignal, reasons: list[str], adjustments: list[str] | None = None
) -> RiskDecision:
    return RiskDecision(
        approved=False,
        signal=signal,
        adjusted_dollars=0.0,
        rejection_reasons=reasons,
        size_adjustments=adjustments if adjustments is not None else [],
    )


@dataclass
class StrategyTracker:
    """Rolling stats for a single strategy's circuit breaker."""

    name: str
    trades: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    disabled_reason: str = ""
    last_data_ts: float = 0.0  # monotonic time of last data update
    consecutive_errors: int = 0  # signal-generation failures (not counted as trades)

    def record_trade(self, pnl: float, won: bool, timestamp: datetime | None = None) -> None:
        ts = timestamp or datetime.now(UTC)
        self.trades.append({"pnl": pnl, "won": won, "ts": ts})

    def rolling_trades(self, window_days: int) -> list[dict[str, Any]]:
        """Return trades within the last *window_days* days."""
        cutoff = datetime.now(UTC).timestamp() - window_days * 86400
        return [t for t in self.trades if t["ts"].timestamp() > cutoff]

    def rolling_pnl(self, window_days: int) -> float:
        """Sum P&L over the last N days."""
        return float(sum(t["pnl"] for t in self.rolling_trades(window_days)))

    def rolling_win_rate(self, window_days: int) -> tuple[float, int]:
        """Return (win_rate, num_trades) over the rolling window."""
        recent = self.rolling_trades(window_days)
        if not recent:
            return 0.0, 0
        wins = sum(1 for t in recent if t["won"])
        return wins / len(recent), len(recent)

    def prune(self, max_age_days: int) -> None:
        """Remove trades older than *max_age_days* to bound memory usage."""
        cutoff = datetime.now(UTC).timestamp() - max_age_days * 86400
        self.trades = [t for t in self.trades if t["ts"].timestamp() > cutoff]

    def update_data_freshness(self) -> None:
        self.last_data_ts = time.monotonic()

    def data_age(self) -> float:
        """Seconds since last data update. Infinity if never updated."""
        if self.last_data_ts == 0.0:
            return float("inf")
        return time.monotonic() - self.last_data_ts


class RiskManager:
    """
    Central risk controller.

    Enforces:
      - Per-market position cap:
        max_single_market_pct × effective_max_total_exposure_pct × bankroll
      - Total exposure cap (config max_total_exposure_pct, overridable via /setexposure)
      - Daily loss halt (max_daily_loss_pct)
      - Drawdown-based size reduction (max_drawdown_pct -> drawdown_size_reduction)
      - Stale data cutoff
      - Kill switch (state/STOP file)
      - Per-strategy circuit breakers

    Bankroll semantics
    ------------------
    ``_bankroll`` represents **total equity** (cash + open position
    notional + realized P&L since deposit) — *not* bare Kalshi cash.
    ``hydrate_positions`` is fed equity at startup; tentative fills move
    cash into open positions (no equity change, so ``_bankroll`` does
    not move); settlements credit realized P&L (``_bankroll += pnl``).
    Sizing/drawdown calculations therefore operate on total wealth.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._risk = settings.risk
        self._cb = settings.circuit_breakers

        self._bankroll = settings.bankroll
        self._peak_bankroll = settings.bankroll
        self._daily_start_bankroll = settings.bankroll

        # Position tracking: market_id -> dollars exposed
        self._positions: dict[str, float] = {}

        # Pod attribution (per-strategy virtual bankrolls). `_position_strategy`
        # maps market_id -> owning strategy (authoritative pod attribution);
        # `_strategy_exposure` is the derived open-cost-basis accumulator. Both
        # are reconstructed at hydrate from the durable trade log (Kalshi
        # positions are strategy-anonymous after restart). The "unknown" bucket
        # holds positions with no resolvable strategy — counts against the firm
        # cap (lives in _positions) but no pod book.
        self._position_strategy: dict[str, str] = {}
        self._strategy_exposure: dict[str, float] = {}

        # Observe-only pod-exposure logging — last logged snapshot for the
        # on-change guard (avoids per-tick log spam at 60s ticks).
        self._last_logged_pod_exposure: dict[str, float] = {}

        self._strategy_trackers: dict[str, StrategyTracker] = {}

        self._daily_pnl: float = 0.0
        self._trading_halted: bool = False

        # WebSocket data freshness (set externally) — initialized to 0 so
        # first tick correctly reports stale until real data arrives.
        self._last_ws_data_time: float = 0.0
        self._startup_monotonic: float = time.monotonic()

        # Monotonic timestamp of the most recent STALE_DATA rejection, or None.
        # Powers the WS-health watchdog's "actual harm" alert trigger. Wall-clock
        # recency (NOT a counter reset on fresh frames) so it decays on its own
        # on a quiet market.
        self._last_stale_reject_monotonic: float | None = None

        # Dirty flag for debounced persistence — set True by every
        # financially significant state mutation, cleared after save.
        self._state_dirty: bool = False

        # Consecutive save_state failures — in-memory only because
        # persisting a counter for "the persistence layer is broken"
        # into the file the failure is on is incoherent.
        self._consecutive_save_failures: int = 0

        # Kill-switch transition flag — log CRITICAL only on activation
        # (False → True), not every tick while STOP exists.  Resets when
        # the file disappears so re-activation is logged again.
        self._kill_switch_seen: bool = False

        # Runtime exposure override — tightened/loosened via Telegram
        # /setexposure without restart. ``None`` means "use config value".
        # Always represents a fraction in (0.0, 1.0].
        self._exposure_override: float | None = None

    # -- Public API ----------------------------------------------------------

    def check_signal(self, signal: TradeSignal) -> RiskDecision:
        """
        Run all risk checks on a trade signal.

        Returns a RiskDecision indicating approval/rejection and any
        size adjustments.

        Three explicit phases, evaluated in order:

        1. **Hard stops** -- kill switch, daily halt.  Return immediately
           if triggered; no further checks run.
        2. **Size adjustments** -- per-market cap, total exposure cap,
           drawdown.  These reduce ``dollars`` and record what happened
           in ``size_adjustments``.  A cap with zero remaining room is
           promoted to a soft-block rejection instead.
        3. **Soft blocks** -- stale data, circuit breakers, correlated
           positions.  These accumulate rejection reasons without
           modifying size.

        If any rejection reasons exist after phases 2--3, the signal is
        rejected with ``adjusted_dollars=0.0``.
        """
        hard_stop = self._hard_stop_decision(signal)
        if hard_stop is not None:
            return hard_stop

        reasons: list[str] = []
        adjustments: list[str] = []
        dollars = self._apply_size_caps(signal, signal.suggested_dollars, reasons, adjustments)
        reasons.extend(self._soft_block_reasons(signal))

        if reasons:
            return _reject(signal, reasons, adjustments)

        return RiskDecision(
            approved=True,
            signal=signal,
            adjusted_dollars=round(dollars, 2),
            rejection_reasons=[],
            size_adjustments=adjustments,
        )

    def _hard_stop_decision(self, signal: TradeSignal) -> RiskDecision | None:
        """Phase 1: kill switch, then daily halt.  ``None`` if neither fired."""
        if self._check_kill_switch():
            return _reject(signal, ["KILL_SWITCH: state/STOP file detected"])

        if self._trading_halted:
            return _reject(
                signal,
                [
                    f"DAILY_LOSS_HALT: daily P&L {self._daily_pnl:.2f} "
                    f"exceeded {self._risk.max_daily_loss_pct * 100:.0f}% "
                    f"of bankroll"
                ],
            )
        return None

    def _apply_size_caps(
        self,
        signal: TradeSignal,
        dollars: float,
        reasons: list[str],
        adjustments: list[str],
    ) -> float:
        """Phase 2: market cap, total exposure cap, drawdown -- in that order.

        The order is load-bearing: the drawdown step only logs and records an
        adjustment when it binds after the earlier reductions.
        """
        max_dollars = (
            self._risk.max_single_market_pct
            * self.effective_max_total_exposure_pct
            * self._bankroll
        )
        dollars = self._cap_market_exposure(
            signal.position_key, dollars, max_dollars, reasons, adjustments
        )
        dollars = self._cap_total_exposure(dollars, reasons, adjustments)
        return self._cap_drawdown(dollars, max_dollars, adjustments)

    def _cap_market_exposure(
        self,
        position_key: str,
        dollars: float,
        max_dollars: float,
        reasons: list[str],
        adjustments: list[str],
    ) -> float:
        current_exposure = self._positions.get(position_key, 0.0)
        remaining_room = max(max_dollars - current_exposure, 0.0)
        if dollars > remaining_room:
            if remaining_room <= 0:
                reasons.append(
                    f"MARKET_CAP: already at max exposure for {position_key} "
                    f"({current_exposure:.2f} / {max_dollars:.2f})"
                )
            else:
                adjustments.append(
                    f"MARKET_CAP: reduced ${dollars:.2f} → ${remaining_room:.2f} "
                    f"(existing exposure ${current_exposure:.2f})"
                )
                dollars = remaining_room
        return dollars

    def _cap_total_exposure(
        self, dollars: float, reasons: list[str], adjustments: list[str]
    ) -> float:
        """Total exposure cap (runtime override wins over config baseline)."""
        total_exposure = self.total_exposure
        max_total = self.effective_max_total_exposure_pct * self._bankroll
        exposure_room = max(max_total - total_exposure, 0.0)
        if dollars > exposure_room:
            if exposure_room <= 0:
                reasons.append(
                    f"TOTAL_EXPOSURE_CAP: total exposure {total_exposure:.2f} "
                    f"at max {max_total:.2f}"
                )
            else:
                adjustments.append(
                    f"TOTAL_EXPOSURE: reduced ${dollars:.2f} → "
                    f"${exposure_room:.2f} "
                    f"(total exposure ${total_exposure:.2f})"
                )
                dollars = exposure_room
        return dollars

    def _cap_drawdown(self, dollars: float, max_dollars: float, adjustments: list[str]) -> float:
        """Drawdown size reduction -- apply as a cap rather than a multiplier
        to avoid double-reduction with the bankroll-based position cap."""
        drawdown = self.drawdown
        if drawdown >= self._risk.max_drawdown_pct:
            reduced_cap = max_dollars * (1.0 - self._risk.drawdown_size_reduction)
            # Only act + log when the cap actually binds.  Earlier reductions
            # (market cap, total exposure cap) may have already taken dollars
            # below ``reduced_cap``; firing a warning then misleads operators
            # into thinking the drawdown rule reduced the size when it didn't.
            if dollars > reduced_cap:
                adjustments.append(
                    f"DRAWDOWN: reduced ${dollars:.2f} → ${reduced_cap:.2f} "
                    f"(drawdown {drawdown:.1%} >= "
                    f"{self._risk.max_drawdown_pct:.0%})"
                )
                dollars = reduced_cap
                logger.warning(
                    "Drawdown %.1f%% >= %.1f%% — capping position at $%.2f (%.0f%% reduction)",
                    drawdown * 100,
                    self._risk.max_drawdown_pct * 100,
                    reduced_cap,
                    self._risk.drawdown_size_reduction * 100,
                )
        return dollars

    def _soft_block_reasons(self, signal: TradeSignal) -> list[str]:
        """Phase 3: stale data, circuit breaker, correlated positions, in order."""
        reasons: list[str] = []

        # Stale data -- applies after the first data update OR if the bot has
        # been running long enough that the WebSocket should have connected.
        # _last_ws_data_time == 0 in shadow mode is fine (no WS needed), but
        # in live/paper mode it means the WS never connected — fail closed.
        if self._last_ws_data_time > 0.0:
            data_age = time.monotonic() - self._last_ws_data_time
            if data_age > self._risk.stale_data_cutoff_seconds:
                reasons.append(
                    f"STALE_DATA: data is {data_age:.1f}s old "
                    f"(cutoff {self._risk.stale_data_cutoff_seconds}s)"
                )
        elif not self._settings.is_shadow:
            # Live/paper mode but WebSocket never connected — block trading
            # after a grace period to allow initial connection.
            uptime = time.monotonic() - self._startup_monotonic
            if uptime > self._risk.stale_data_cutoff_seconds:
                reasons.append(
                    "STALE_DATA: WebSocket never connected "
                    f"(uptime {uptime:.0f}s, no data received)"
                )

        # Record STALE_DATA rejection recency for the WS-health watchdog's
        # "actual harm" trigger. Instrumentation only — does not alter the
        # gate above.
        if any(r.startswith("STALE_DATA") for r in reasons):
            self._last_stale_reject_monotonic = time.monotonic()

        # Per-strategy circuit breaker
        tracker = self._get_tracker(signal.strategy_name)
        if not tracker.enabled:
            reasons.append(f"STRATEGY_DISABLED: {signal.strategy_name} — {tracker.disabled_reason}")

        # Event-aware: see _correlated_event_count for what counts as correlated.
        position_key = signal.position_key
        corr_count = _correlated_event_count(position_key, self._positions)
        if corr_count >= self._risk.max_correlated_positions:
            # Safe re-split: corr_count > 0 implies the helper already validated
            # a KX-prefixed, ≥3-part key, so parts[0] is the series.
            series = position_key.split("-")[0]
            reasons.append(
                f"CORRELATED_POSITIONS: {corr_count} other forecast date(s) held "
                f"in series '{series}' >= max {self._risk.max_correlated_positions}"
            )

        return reasons

    def record_fill(
        self,
        market_id: str,
        strategy_name: str,
        dollars: float,
        pnl: float = 0.0,
    ) -> None:
        """
        Record a trade fill for position/exposure tracking.

        Call this after a trade is executed (or resolved) to update
        internal state.
        """
        prior = self._positions.get(market_id, 0.0)
        new_exposure = prior + dollars
        if new_exposure <= 0.0:
            self._positions.pop(market_id, None)
            # Drop attribution for a fully-closed market; reconcile accumulator.
            self._adjust_strategy_exposure(market_id, -prior)
            self._position_strategy.pop(market_id, None)
        else:
            self._positions[market_id] = new_exposure
            prior_owner = self._position_strategy.get(market_id)
            if prior_owner is not None and prior_owner != strategy_name:
                # Ownership reassignment on the same market (not reachable in
                # production today — deference + dedup prevent it — but keep the
                # per-pod invariant unconditional). Move the prior chunk off the
                # old owner and attribute the full new exposure to the new owner.
                self._adjust_strategy_exposure(market_id, -prior)  # old owner -= prior
                self._position_strategy[market_id] = strategy_name
                self._adjust_strategy_exposure(market_id, new_exposure)  # new owner += new
            else:
                self._position_strategy[market_id] = strategy_name
                self._adjust_strategy_exposure(market_id, dollars)

        self._book_pnl(pnl)

    def release_exposure(self, market_id: str, dollars: float) -> None:
        """Reduce (but not increase) a position by *dollars*.

        Used to adjust tentative exposure recorded at order submission when the
        confirmed fill is smaller than expected (lower fill price, partial or
        cancelled order).  The position is clamped to zero — it is never allowed
        to go negative.
        """
        current = self._positions.get(market_id, 0.0)
        applied = min(dollars, current)  # mirror the clamp used for _positions
        reduced = max(current - dollars, 0.0)
        self._adjust_strategy_exposure(market_id, -applied)
        if reduced == 0.0:
            self._positions.pop(market_id, None)
            self._position_strategy.pop(market_id, None)
        else:
            self._positions[market_id] = reduced
        self._state_dirty = True

    def close_position(self, market_id: str, pnl: float = 0.0) -> None:
        """Remove a position after it settles or is fully closed."""
        popped = self._positions.pop(market_id, 0.0)
        self._adjust_strategy_exposure(market_id, -popped)
        self._position_strategy.pop(market_id, None)
        self._book_pnl(pnl)

    def realize_partial(
        self,
        position_key: str,
        *,
        dollars_released: float,
        pnl: float,
    ) -> None:
        """Release partial exposure AND book realized P&L.

        Books P&L like close_position (via _book_pnl) but reduces (not pops)
        _positions[position_key] by dollars_released. The remaining exposure
        stays in the positions dict for the residual leg.

        For full closes, callers MUST use close_position instead.

        State-divergence safety: NEVER pops the position. Always clamps at
        0.0 via max(0.0, current - dollars_released). If dollars_released
        exceeds current, log CRITICAL — the position entry remains in
        _positions so _settle_orders can find matching OrderRecords and
        book the residual leg's P&L correctly at settlement. Popping would
        silently orphan the residual exposure (_settle_live skips positions
        not in risk.positions).
        """
        current = self._positions.get(position_key, 0.0)
        if dollars_released > current:
            logger.critical(
                "realize_partial: state divergence detected — "
                "dollars_released=%.4f > current=%.4f for %s; "
                "clamping exposure at 0.0 (NOT popping; residual "
                "OrderRecords must still settle)",
                dollars_released,
                current,
                position_key,
            )
        applied = min(dollars_released, current)  # mirror the _positions clamp
        self._adjust_strategy_exposure(position_key, -applied)
        self._positions[position_key] = max(0.0, current - dollars_released)
        self._book_pnl(pnl)

    def update_data_freshness(self, source: str = "websocket") -> None:
        """Call this whenever fresh market data is received."""
        self._last_ws_data_time = time.monotonic()

    def stale_data_rejected_within(self, seconds: float) -> bool:
        """True iff a STALE_DATA rejection occurred within the last ``seconds``.

        Wall-clock recency; decays on its own with no dependence on a fresh WS
        frame (a quiet market never delivers one). Used by the orchestrator's
        WS-health watchdog as its ground-truth "actual harm" alert trigger.
        """
        ts = self._last_stale_reject_monotonic
        if ts is None:
            return False
        return (time.monotonic() - ts) <= seconds

    def reset_daily(self) -> None:
        """Reset daily P&L tracking. Call at start of each trading day."""
        self._daily_pnl = 0.0
        self._daily_start_bankroll = self._bankroll
        self._trading_halted = False
        self._state_dirty = True
        logger.info("Daily risk counters reset. Bankroll: %.2f", self._bankroll)

    def check_circuit_breakers(self) -> list[str]:
        """
        Evaluate per-strategy circuit breakers and disable any that trip.

        Returns list of strategies that were disabled this check.
        """
        disabled: list[str] = []
        window = self._cb.rolling_window_days

        for name, tracker in self._strategy_trackers.items():
            tracker.prune(max_age_days=window * 2)

            if not tracker.enabled:
                continue

            recent = tracker.rolling_trades(window)

            rolling = sum(t["pnl"] for t in recent)
            loss_threshold = self._cb.max_loss_pct * self._bankroll
            if self._bankroll > 0 and rolling < -loss_threshold:
                self._trip_breaker(
                    name,
                    tracker,
                    f"Rolling {window}d P&L "
                    f"{rolling:.2f} < -{self._cb.max_loss_pct * 100:.0f}% "
                    f"of bankroll (${loss_threshold:.2f})",
                )
                disabled.append(name)
                continue

            n_trades = len(recent)
            if n_trades >= self._cb.min_trades_for_win_rate:
                wins = sum(1 for t in recent if t["won"])
                win_rate = wins / n_trades
                if win_rate < self._cb.min_win_rate:
                    self._trip_breaker(
                        name,
                        tracker,
                        f"Win rate {win_rate:.1%} < {self._cb.min_win_rate:.0%} "
                        f"over {n_trades} trades",
                    )
                    disabled.append(name)

            # Data freshness check (soft): currently inert / reserved. The
            # `data_staleness_multiplier` config field is not yet wired up here
            # because the expected per-strategy data interval is unknown at this
            # point. The hard freshness enforcement (stale_data_cutoff) lives in
            # check_signal().

        return disabled

    def record_strategy_trade(self, strategy_name: str, pnl: float, won: bool) -> None:
        """Record a settled trade for per-strategy circuit breaker tracking."""
        tracker = self._get_tracker(strategy_name)
        tracker.record_trade(pnl=pnl, won=won)
        self._state_dirty = True

    def enable_strategy(self, name: str) -> None:
        """Manually re-enable a strategy after investigation."""
        tracker = self._get_tracker(name)
        tracker.enabled = True
        tracker.disabled_reason = ""
        self._state_dirty = True
        logger.info("Strategy %s manually re-enabled", name)

    def record_strategy_error(self, name: str) -> int:
        """Increment a strategy's consecutive-error counter.

        Persisted across restarts so a long-burning failure that takes
        many process restarts to accumulate still trips the CRITICAL
        alert at STRATEGY_ERROR_ALERT_THRESHOLD.

        Returns the new counter value so callers can branch on the
        threshold without re-reading.
        """
        tracker = self._get_tracker(name)
        tracker.consecutive_errors += 1
        self._state_dirty = True
        return tracker.consecutive_errors

    def clear_strategy_errors(self, name: str) -> None:
        """Reset a strategy's consecutive-error counter to 0.

        No-op (and no dirty flag) if already 0 — the success path runs
        every tick and we don't want to force a save every minute when
        nothing changed.
        """
        tracker = self._get_tracker(name)
        if tracker.consecutive_errors != 0:
            tracker.consecutive_errors = 0
            self._state_dirty = True

    def record_save_failure(self) -> int:
        """Increment the consecutive-save-failure counter.

        Not persisted (the persistence layer is the thing that's
        failing, so persisting the counter into the same file is
        incoherent). Returns the new counter value.
        """
        self._consecutive_save_failures += 1
        return self._consecutive_save_failures

    def clear_save_failures(self) -> None:
        """Reset the consecutive-save-failure counter to 0.

        Called by the orchestrator after every successful save.
        """
        self._consecutive_save_failures = 0

    @property
    def consecutive_save_failures(self) -> int:
        """In-memory counter of consecutive save_state failures."""
        return self._consecutive_save_failures

    # -- Persistence ---------------------------------------------------------

    @property
    def state_dirty(self) -> bool:
        """True if state has changed since last :meth:`mark_state_clean`."""
        return self._state_dirty

    def mark_state_clean(self) -> None:
        """Clear the dirty flag after state has been persisted."""
        self._state_dirty = False

    def mark_dirty(self) -> None:
        """Set the dirty flag.

        Inverse of :meth:`mark_state_clean`. Used by the orchestrator's
        save-failure recovery path: if ``save_state`` returns False or the
        post-lock-release block raises, ``_save_under_lock`` re-arms the
        dirty flag here so the next tick retries persistence.
        """
        self._state_dirty = True

    def export_state(self) -> dict[str, Any]:
        """Return a freshly constructed dict of all persistable state.

        Returns deep copies — safe to hand off to a background thread for
        serialization without risk of concurrent mutation.
        """
        trackers: dict[str, dict[str, Any]] = {}
        for name, tracker in self._strategy_trackers.items():
            # Snapshot the list before iterating — prune() could mutate it
            # concurrently if called from check_circuit_breakers on the next tick.
            trades_snapshot = list(tracker.trades)
            trackers[name] = {
                "enabled": tracker.enabled,
                "disabled_reason": tracker.disabled_reason,
                "consecutive_errors": tracker.consecutive_errors,
                "trades": [
                    {"pnl": t["pnl"], "won": t["won"], "ts": t["ts"]} for t in trades_snapshot
                ],
            }
        return {
            "peak_bankroll": self._peak_bankroll,
            "daily_pnl": self._daily_pnl,
            "daily_start_bankroll": self._daily_start_bankroll,
            "trading_halted": self._trading_halted,
            "strategy_trackers": trackers,
        }

    def hydrate_positions(
        self,
        positions: dict[str, float],
        bankroll: float,
        *,
        position_strategy: dict[str, str] | None = None,
    ) -> None:
        """Set positions and total equity from Kalshi API / replay data.

        ``bankroll`` is interpreted as **total equity** (Kalshi cash + open
        position notional). ``position_strategy`` is the reconstructed
        ``market_id -> strategy`` attribution (built by the orchestrator from
        the durable trade log); markets absent from it land in the "unknown"
        bucket. Derives the per-strategy exposure accumulator by grouping
        ``positions`` under their attributed strategy.

        Directly assigns state — does **not** call :meth:`record_fill`.
        Call this **before** :meth:`hydrate_from_state`.
        """
        self._positions = dict(positions)
        self._bankroll = bankroll
        # Retain attribution ONLY for currently-open markets. The reconstructed
        # index spans every market that ever traded (trades.jsonl is append-only
        # and never pruned); keeping all of it would bloat this map with dead
        # closed-market entries on every hydrate. Bounding it to open positions
        # matches the map's purpose (open-position attribution).
        src_attribution = position_strategy or {}
        self._position_strategy = {
            market_id: src_attribution[market_id]
            for market_id in self._positions
            if market_id in src_attribution
        }
        self._strategy_exposure = {}
        for market_id, dollars in self._positions.items():
            strategy = self._position_strategy.get(market_id, "unknown")
            self._strategy_exposure[strategy] = (
                self._strategy_exposure.get(strategy, 0.0) + dollars
            )
        logger.info(
            "Hydrated from Kalshi API: %d positions, bankroll=$%.2f, "
            "exposure=$%.2f, pods=%s",
            len(positions),
            bankroll,
            sum(positions.values()),
            {k: round(v, 2) for k, v in self._strategy_exposure.items()},
        )

    def hydrate_exposure_override(self, state_dir: Path) -> None:
        """Restore persisted exposure override from disk at startup.

        Silently no-ops when the file is missing, corrupt, or out of bounds —
        the persistence layer logs the reason.
        """
        fraction = load_exposure_override(state_dir)
        if fraction is None:
            return
        # Direct assignment — skip set_exposure_override so the startup
        # message is explicit rather than a generic "override changed" line.
        self._exposure_override = fraction
        logger.info(
            "Restored persisted exposure override: max_total_exposure_pct=%.4f "
            "(config baseline=%.4f)",
            fraction,
            self._risk.max_total_exposure_pct,
        )

    def persist_exposure_override(self, state_dir: Path) -> None:
        """Write the current override (or clear the file) to ``state_dir``.

        Synchronous — called from Telegram handlers which already run off
        the tick hot path.  Atomic via the persistence helper.
        """
        save_exposure_override(state_dir, self._exposure_override)

    def hydrate_from_state(self, state: RiskStateSnapshot) -> None:
        """Restore persisted risk state.  Call **after** :meth:`hydrate_positions`.

        Restores peak bankroll (uses max of stored vs current), daily PnL,
        halt flag, and strategy tracker state.  ``last_data_ts`` stays 0.0
        (monotonic clock is not persistable).
        """
        self._peak_bankroll = max(state.peak_bankroll, self._bankroll)
        self._daily_pnl = state.daily_pnl
        self._daily_start_bankroll = state.daily_start_bankroll
        self._trading_halted = state.trading_halted

        for name, tracker_data in state.strategy_trackers.items():
            tracker = self._get_tracker(name)
            tracker.enabled = tracker_data.get("enabled", True)
            tracker.disabled_reason = tracker_data.get("disabled_reason", "")
            tracker.consecutive_errors = int(tracker_data.get("consecutive_errors", 0))
            # Corrupted state files could crash check_circuit_breakers if
            # "ts" is missing or not a datetime.
            valid_trades: list[dict[str, Any]] = []
            for t in tracker_data.get("trades", []):
                if isinstance(t, dict) and isinstance(t.get("ts"), datetime):
                    valid_trades.append(t)
                else:
                    logger.warning("Dropping malformed trade during hydration: %s", t)
            tracker.trades = valid_trades

        logger.info(
            "Hydrated risk state: peak=$%.2f, daily_pnl=$%.2f, halted=%s, trackers=%d",
            self._peak_bankroll,
            self._daily_pnl,
            self._trading_halted,
            len(state.strategy_trackers),
        )

    # -- Introspection -------------------------------------------------------

    @property
    def bankroll(self) -> float:
        """Total equity (cash + open position notional + realized P&L).

        See class docstring for the full semantics — sizing, drawdown,
        and reconciliation drift checks all operate on this equity
        figure rather than bare Kalshi cash.
        """
        return self._bankroll

    @property
    def peak_bankroll(self) -> float:
        return self._peak_bankroll

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def total_exposure(self) -> float:
        return sum(self._positions.values())

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def strategy_exposure(self) -> dict[str, float]:
        """Per-strategy open cost-basis (read-only snapshot).

        Keys are strategy names plus the "unknown" bucket. The sum equals
        ``total_exposure`` by construction.
        """
        return dict(self._strategy_exposure)

    def strategy_for_position(self, market_id: str) -> str:
        """Owning strategy for an open market, or "unknown" if unattributed."""
        return self._position_strategy.get(market_id, "unknown")

    def open_position_keys(self) -> list[str]:
        """Market IDs (tickers) of currently-open positions. For cross-strategy
        event-deference (favorite_bias). Read-only snapshot."""
        return list(self._positions.keys())

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    @property
    def drawdown(self) -> float:
        if self._peak_bankroll <= 0:
            return 0.0
        return 1.0 - (self._bankroll / self._peak_bankroll)

    @property
    def effective_max_total_exposure_pct(self) -> float:
        """Runtime override (if set) else the config baseline.

        Read on every ``check_signal``; kept as a property over a plain
        attribute so the override stays authoritative even if a caller
        reaches in and mutates ``_risk.max_total_exposure_pct`` directly.
        """
        if self._exposure_override is not None:
            return self._exposure_override
        return self._risk.max_total_exposure_pct

    def book_for(self, strategy_name: str) -> float:
        """Per-strategy virtual bankroll = allocation × total equity.

        Strategies absent from ``risk.strategy_allocations`` (and the empty-map
        default) fall back to full equity, preserving legacy single-bankroll
        behavior. A ``0.0`` allocation returns ``0.0`` (parks the strategy).
        """
        pct = self._risk.strategy_allocations.get(strategy_name)
        if pct is None:
            return self._bankroll
        return pct * self._bankroll

    def maybe_log_pod_exposure(self) -> None:
        """Log each pod's open exposure vs its book — on change only.

        Observe-only: logs whether a pod *would* breach its book but does NOT
        block. No-op when strategy_allocations is empty. Routes to logger.info
        (never Telegram). Guarded against per-tick spam by comparing a rounded
        snapshot to the last logged one.
        """
        if not self._risk.strategy_allocations:
            return
        snapshot = {
            name: round(self._strategy_exposure.get(name, 0.0), 2)
            for name in self._risk.strategy_allocations
        }
        # Include the "unknown" bucket (unattributed exposure) in the snapshot:
        # it must reach $0 / be fully traced before enforcement can turn on.
        # Tracked under a reserved key that cannot collide with a strategy name.
        snapshot["unknown"] = round(self._strategy_exposure.get("unknown", 0.0), 2)
        if snapshot == self._last_logged_pod_exposure:
            return
        self._last_logged_pod_exposure = snapshot
        for name, exposure in snapshot.items():
            if name == "unknown":
                logger.info(
                    "pod_exposure strategy=unknown exposure=$%.2f "
                    "(no book; counts against firm cap only)",
                    exposure,
                )
                continue
            book = self.book_for(name)
            logger.info(
                "pod_exposure strategy=%s exposure=$%.2f book=$%.2f "
                "would_breach=%s",
                name,
                exposure,
                book,
                exposure > book,
            )

    @property
    def exposure_override(self) -> float | None:
        """Raw override value or ``None`` when config baseline is in effect."""
        return self._exposure_override

    def set_exposure_override(self, fraction: float | None) -> float:
        """Set (or clear with ``None``) the runtime total-exposure cap.

        ``fraction`` must be in ``(0.0, 1.0]`` when non-``None``.  Out-of-range
        values raise ``ValueError`` — callers handle UX messaging.  Returns the
        resulting effective cap so callers can log/reply without re-reading.

        This mutates an in-memory attribute only; persistence is the caller's
        responsibility (the Telegram handler writes ``exposure_override.json``
        after the call succeeds).
        """
        if fraction is not None and not (0.0 < fraction <= 1.0):
            raise ValueError(f"exposure override must be in (0.0, 1.0], got {fraction!r}")
        prior = self.effective_max_total_exposure_pct
        self._exposure_override = fraction
        new_effective = self.effective_max_total_exposure_pct
        logger.info(
            "max_total_exposure_pct changed: %.4f -> %.4f (override=%s)",
            prior,
            new_effective,
            "cleared" if fraction is None else f"{fraction:.4f}",
        )
        return new_effective

    def get_tracker(self, strategy_name: str) -> StrategyTracker:
        return self._get_tracker(strategy_name)

    def count_same_day_bets(self, forecast_date: date) -> int:
        """Return the number of open positions whose ticker resolves to
        ``forecast_date``.

        Used by :func:`python.analytics.sizing.fractional_kelly` to apply the
        same-day correlation haircut. Reads the existing ``_positions`` dict
        (no new data structure introduced) and relies on
        :func:`python.utils.ticker_dates.parse_ticker_date` to derive the
        forecast date from each position key (market_ticker or market_id).

        Positions whose ticker is unparseable (non-weather series like
        ``KXINX``, or any future ticker shape this helper hasn't been taught)
        do NOT contribute to the count — fail-open so an unrelated series
        can't accidentally throttle weather sizing. Closed positions are
        already absent from ``_positions`` by construction
        (``close_position`` pops them).
        """
        return sum(1 for key in self._positions if parse_ticker_date(key) == forecast_date)

    # -- Internal ------------------------------------------------------------

    def _get_tracker(self, name: str) -> StrategyTracker:
        if name not in self._strategy_trackers:
            self._strategy_trackers[name] = StrategyTracker(name=name)
        return self._strategy_trackers[name]

    def _adjust_strategy_exposure(self, market_id: str, delta: float) -> None:
        """Apply ``delta`` (signed) to the owning pod's exposure accumulator.

        Resolves the strategy via the authoritative _position_strategy map,
        defaulting to "unknown" for positions with no attribution (never
        raises). Clamps the bucket at 0.0 so the accumulator never goes
        negative on an over-release.
        """
        strategy = self._position_strategy.get(market_id, "unknown")
        new_val = self._strategy_exposure.get(strategy, 0.0) + delta
        self._strategy_exposure[strategy] = max(0.0, new_val)

    def _book_pnl(self, pnl: float) -> None:
        """Credit realized P&L to equity and the daily counter, then re-check the halt."""
        self._bankroll += pnl
        self._daily_pnl += pnl
        self._peak_bankroll = max(self._peak_bankroll, self._bankroll)
        self._state_dirty = True
        self._check_daily_loss_halt()

    def _trip_breaker(self, name: str, tracker: StrategyTracker, reason: str) -> None:
        tracker.enabled = False
        tracker.disabled_reason = reason
        self._state_dirty = True
        logger.warning("Circuit breaker tripped for %s: %s", name, reason)

    def _check_daily_loss_halt(self) -> None:
        """Halt trading if daily P&L breaches the loss limit."""
        daily_loss_limit = self._risk.max_daily_loss_pct * self._daily_start_bankroll
        if self._daily_pnl <= -daily_loss_limit:
            self._trading_halted = True
            logger.critical(
                "DAILY LOSS HALT: P&L %.2f breached limit -%.2f",
                self._daily_pnl,
                daily_loss_limit,
            )

    def _check_kill_switch(self) -> bool:
        """Check if the STOP file exists. Fails closed on filesystem errors.

        CRITICAL log fires once on activation (transition False → True).
        Resets to False when the file disappears so a re-activation is
        logged again — the goal is to flag transitions, not to suppress
        the alert forever after the first hit.
        """
        if self._settings.is_kill_switch_active():
            if not self._kill_switch_seen:
                logger.critical("KILL SWITCH ACTIVATED: %s exists", self._settings.stop_file)
                self._kill_switch_seen = True
            return True
        if self._kill_switch_seen:
            logger.info("KILL SWITCH cleared: %s no longer exists", self._settings.stop_file)
            self._kill_switch_seen = False
        return False
