"""Tests for Telegram message formatters."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, PropertyMock

from python.common import ExecutionMode, OrderStatus, Side
from python.execution.engine import OrderRecord
from python.notifications.formatters import (
    format_acks,
    format_help,
    format_positions,
    format_risk,
    format_signals,
    format_status,
    format_trade_alert,
    format_trades,
)
from python.risk.manager import RiskDecision, StrategyTracker
from python.strategies.base import TradeSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_risk(
    bankroll: float = 10_000.0,
    peak_bankroll: float = 10_000.0,
    daily_pnl: float = 0.0,
    total_exposure: float = 0.0,
    positions: dict | None = None,
    is_halted: bool = False,
    drawdown: float = 0.0,
    effective_max_total_exposure_pct: float = 0.40,
    exposure_override: float | None = None,
) -> MagicMock:
    risk = MagicMock()
    risk.bankroll = bankroll
    risk.peak_bankroll = peak_bankroll
    risk.daily_pnl = daily_pnl
    risk.total_exposure = total_exposure
    risk.positions = positions or {}
    risk.is_halted = is_halted
    risk.drawdown = drawdown
    risk.effective_max_total_exposure_pct = effective_max_total_exposure_pct
    risk.exposure_override = exposure_override
    risk.get_tracker.return_value = StrategyTracker(name="weather_edge")
    return risk


def _mock_engine(records: list[OrderRecord] | None = None) -> MagicMock:
    engine = MagicMock()
    type(engine).mode = PropertyMock(return_value=ExecutionMode.SHADOW)
    engine.order_history = records or []
    return engine


def _mock_settings(kill_switch: bool = False) -> MagicMock:
    settings = MagicMock()
    settings.is_kill_switch_active.return_value = kill_switch
    settings.risk.max_total_exposure_pct = 0.40
    settings.risk.max_daily_loss_pct = 0.05
    settings.risk.max_drawdown_pct = 0.15
    return settings


def _make_signal(**overrides) -> TradeSignal:
    defaults = {
        "market_id": "MKT-001",
        "market_ticker": "KXHIGHNYC-26MAR22-T58",
        "model_probability": 0.60,
        "market_price": 0.50,
        "confidence": 0.8,
        "strategy_name": "weather_edge",
        "side": Side.YES,
        "suggested_dollars": 100.0,
        "suggested_contracts": 2,
    }
    defaults.update(overrides)
    return TradeSignal(**defaults)


def _make_record(status: OrderStatus = OrderStatus.SHADOW, **sig_overrides) -> OrderRecord:
    signal = _make_signal(**sig_overrides)
    decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=100.0)
    return OrderRecord(
        order_id="shadow-1",
        signal=signal,
        risk_decision=decision,
        mode=ExecutionMode.SHADOW,
        signal_price=0.50,
        order_price=0.50,
        fill_price=0.50,
        fill_count=2,
        requested_count=2,
        status=status,
        timestamp=datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFormatStatus:
    def test_basic_output(self):
        risk = _mock_risk(bankroll=10_245.50, daily_pnl=45.50, total_exposure=1_915.20)
        engine = _mock_engine()
        settings = _mock_settings()
        strategy = MagicMock()
        strategy.name = "weather_edge"
        strategy.enabled = True

        text = format_status(risk, engine, settings, [strategy])

        assert "THE ORACLE STATUS" in text
        assert "shadow" in text
        assert "$10,245.50" in text
        assert "+$45.50" in text
        assert "weather_edge: ON" in text

    def test_kill_switch_active(self):
        risk = _mock_risk()
        engine = _mock_engine()
        settings = _mock_settings(kill_switch=True)

        text = format_status(risk, engine, settings, [])

        assert "ACTIVE" in text

    def test_strategy_disabled(self):
        risk = _mock_risk()
        tracker = StrategyTracker(name="weather_edge", enabled=False, disabled_reason="loss limit")
        risk.get_tracker.return_value = tracker
        engine = _mock_engine()
        settings = _mock_settings()
        strategy = MagicMock()
        strategy.name = "weather_edge"
        strategy.enabled = True

        text = format_status(risk, engine, settings, [strategy])

        assert "OFF" in text
        assert "loss limit" in text

    def test_without_mtm_omits_mtm_lines(self):
        # Regression guard: omitting the mtm kwarg renders no MTM lines —
        # preserves shadow/paper-mode output exactly.
        risk = _mock_risk(bankroll=10_000.00)
        engine = _mock_engine()
        settings = _mock_settings()

        text = format_status(risk, engine, settings, [])

        assert "MTM" not in text
        assert "Unrealized" not in text

    def test_with_healthy_mtm_renders_three_lines(self):
        from python.analytics.mtm import PortfolioMtm

        risk = _mock_risk(bankroll=10_000.00)
        engine = _mock_engine()
        settings = _mock_settings()
        mtm = PortfolioMtm(
            cash=8_000.00,
            position_mtm=2_037.67,
            equity_mtm=10_037.67,
            n_positions_fetched=3,
            as_of_utc=datetime(2026, 5, 16, 14, 32, 8, tzinfo=UTC),
            degraded=False,
        )

        text = format_status(risk, engine, settings, [], mtm=mtm)

        assert "Bankroll: $10,000.00" in text
        assert "MTM:      $10,037.67" in text
        assert "Unrealized: +$37.67" in text
        assert "[partial" not in text
        assert "(cost basis)" not in text
        assert "as of" not in text

    def test_with_degraded_mtm_shows_partial_tag(self):
        from python.analytics.mtm import PortfolioMtm

        risk = _mock_risk(bankroll=10_000.00)
        engine = _mock_engine()
        settings = _mock_settings()
        mtm = PortfolioMtm(
            cash=8_000.00,
            position_mtm=1_500.00,
            equity_mtm=9_500.00,
            n_positions_fetched=5,
            as_of_utc=datetime(2026, 5, 16, 14, 32, 8, tzinfo=UTC),
            degraded=True,
            breakdown=tuple(
                ("T", Side.YES, 1.0, 0.5, 0.5) for _ in range(3)
            ),
        )

        text = format_status(risk, engine, settings, [], mtm=mtm)

        assert "[partial: 3/5 priced]" in text
        assert "Unrealized: -$500.00" in text

    def test_with_mtm_attempted_but_failed_shows_unavailable(self):
        risk = _mock_risk(bankroll=10_000.00)
        engine = _mock_engine()
        settings = _mock_settings()

        text = format_status(
            risk, engine, settings, [], mtm=None, mtm_attempted=True
        )

        assert "MTM: unavailable" in text
        assert "Unrealized" not in text


class TestFormatPositions:
    """``format_positions`` splits ``risk.positions`` into filled vs working
    sections by cross-referencing ``engine.order_history`` — fills come from
    OrderRecords in FILLED/PARTIAL/SHADOW; the residual exposure (RESTING /
    PENDING limit orders) is the working portion.
    """

    @staticmethod
    def _filled_record(
        position_key: str,
        *,
        ticker: str,
        side: Side = Side.YES,
        fill_count: int = 12,
        fill_price: float = 0.35,
        status: OrderStatus = OrderStatus.FILLED,
        requested_count: int | None = None,
    ) -> OrderRecord:
        signal = _make_signal(
            market_id=position_key,
            market_ticker=ticker,
            side=side,
        )
        decision = RiskDecision(approved=True, signal=signal, adjusted_dollars=100.0)
        return OrderRecord(
            order_id=f"ord-{position_key}",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=fill_price,
            order_price=fill_price,
            fill_price=fill_price,
            fill_count=fill_count,
            requested_count=requested_count if requested_count is not None else fill_count,
            status=status,
            timestamp=datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC),
        )

    @staticmethod
    def _hydrated_record(
        position_key: str,
        *,
        ticker: str,
        side: Side = Side.YES,
        last_recorded_dollars: float,
    ) -> OrderRecord:
        # Mirrors hydrate_live_positions stub seeding: zeroed metadata,
        # fill_price=None, last_recorded_dollars carries the hydrated exposure.
        signal = _make_signal(
            market_id=position_key,
            market_ticker=ticker,
            side=side,
            strategy_name="live_hydrated",
            model_probability=0.0,
            market_price=0.0,
            confidence=0.0,
        )
        decision = RiskDecision(
            approved=True, signal=signal, adjusted_dollars=last_recorded_dollars
        )
        return OrderRecord(
            order_id=f"live-hydrated-{position_key}",
            signal=signal,
            risk_decision=decision,
            mode=ExecutionMode.LIVE,
            signal_price=0.0,
            order_price=0.0,
            fill_price=None,
            fill_count=0,
            requested_count=0,
            status=OrderStatus.HYDRATED,
            last_recorded_dollars=last_recorded_dollars,
            timestamp=datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC),
        )

    def test_no_positions(self):
        risk = _mock_risk()
        engine = _mock_engine()
        assert format_positions(risk, engine) == "No open positions."

    def test_all_filled_no_working(self):
        """Every dollar of exposure is backed by a filled OrderRecord."""
        risk = _mock_risk(
            positions={"MKT-A": 4.20, "MKT-B": 7.50},
            total_exposure=11.70,
        )
        engine = _mock_engine([
            self._filled_record("MKT-A", ticker="KXHIGHNYC-T62", fill_count=12, fill_price=0.35),
            self._filled_record("MKT-B", ticker="KXLOWAUS-B74.5", fill_count=15, fill_price=0.50),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        # No working orders → that header must NOT appear.
        assert "WORKING ORDERS" not in text
        assert "$4.20" in text
        assert "$7.50" in text
        assert "Total filled:" in text
        assert "Total working:" in text  # footer always shown
        assert "Total exposure:" in text
        assert "$11.70" in text

    def test_all_working_no_fills(self):
        """RESTING limit orders show only in the working section."""
        risk = _mock_risk(
            positions={"MKT-WORK": 3.51},
            total_exposure=3.51,
        )
        # No FILLED/PARTIAL/SHADOW records — engine has e.g. only RESTING orders.
        engine = _mock_engine([
            self._filled_record(
                "MKT-WORK",
                ticker="KXLOWTAUS-B74.5",
                status=OrderStatus.RESTING,
                fill_count=0,
                fill_price=0.0,
                requested_count=6,
            ),
        ])

        text = format_positions(risk, engine)

        assert "WORKING ORDERS (resting / pending)" in text
        assert "OPEN POSITIONS (filled)" not in text
        assert "$3.51" in text
        assert "Total filled:" in text
        assert "Total working:" in text

    def test_mixed_filled_and_working(self):
        """A risk-tracker key with both filled and unfilled exposure shows
        in both sections."""
        risk = _mock_risk(
            positions={"MKT-MIX": 10.00, "MKT-FILLED": 4.20},
            total_exposure=14.20,
        )
        # MKT-MIX has $6 of fills + $4 RESTING; MKT-FILLED is fully filled.
        engine = _mock_engine([
            self._filled_record(
                "MKT-MIX", ticker="KXHIGHNYC-MIX",
                fill_count=12, fill_price=0.50, status=OrderStatus.FILLED,
            ),
            self._filled_record(
                "MKT-MIX", ticker="KXHIGHNYC-MIX",
                fill_count=0, fill_price=0.50, status=OrderStatus.RESTING,
                requested_count=20,
            ),
            self._filled_record(
                "MKT-FILLED", ticker="KXHIGHNYC-T62",
                fill_count=12, fill_price=0.35, status=OrderStatus.FILLED,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        assert "WORKING ORDERS (resting / pending)" in text
        # MKT-MIX shows in both sections.
        mix_count = text.count("MKT-MIX")
        assert mix_count >= 2

    def test_partial_fill_in_both_sections(self):
        """A PARTIAL fill counts as filled (its filled portion) plus working
        (the remainder)."""
        risk = _mock_risk(
            positions={"MKT-PART": 10.00},
            total_exposure=10.00,
        )
        engine = _mock_engine([
            self._filled_record(
                "MKT-PART",
                ticker="KXHIGHNYC-T62",
                fill_count=6,
                fill_price=0.50,  # filled portion = $3.00
                status=OrderStatus.PARTIAL,
                requested_count=20,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        assert "WORKING ORDERS (resting / pending)" in text

    def test_shadow_treated_as_filled(self):
        """SHADOW status is in FILLED_STATUSES (mirrors ``format_trades``)."""
        risk = _mock_risk(
            positions={"MKT-SHADOW": 1.00},
            total_exposure=1.00,
        )
        engine = _mock_engine([
            self._filled_record(
                "MKT-SHADOW",
                ticker="KXHIGHNYC-T62",
                fill_count=2,
                fill_price=0.50,
                status=OrderStatus.SHADOW,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        assert "WORKING ORDERS" not in text

    def test_total_exposure_invariant(self):
        """``Total exposure`` always equals ``risk.total_exposure``,
        regardless of how the records partition it."""
        risk = _mock_risk(
            positions={"MKT-A": 4.20, "MKT-B": 3.51, "MKT-C": 7.30},
            total_exposure=15.01,
        )
        # One filled, one all-working, one missing entirely — total must
        # still match risk.total_exposure.
        engine = _mock_engine([
            self._filled_record("MKT-A", ticker="A", fill_count=12, fill_price=0.35),
        ])

        text = format_positions(risk, engine)

        assert "$15.01" in text

    def test_pruned_records_defensive(self):
        """A tracked position with no matching records is a bug indicator
        (terminal-status records are pruned only after exposure is released),
        so it surfaces under WARNING: untracked exposure rather than silently
        appearing as a working order."""
        risk = _mock_risk(
            positions={"MKT-ORPHAN": 5.00},
            total_exposure=5.00,
        )
        engine = _mock_engine([])  # no order history

        text = format_positions(risk, engine)

        assert "WARNING: untracked exposure" in text
        assert "MKT-ORPHAN" in text
        assert "$5.00" in text
        assert "untracked — investigate" in text
        assert "WORKING ORDERS (resting / pending)" not in text

    def test_hydrated_in_recovered_section(self):
        """A HYDRATED stub (created by engine.hydrate_live_positions on
        restart) appears under OPEN POSITIONS (restored on restart) without a
        fake '0 contracts @ $0.00' detail. Without restored_fill_info it shows
        the 'restored from API' fallback message."""
        risk = _mock_risk(
            positions={"MKT-HYDRA": 5.00},
            total_exposure=5.00,
        )
        engine = _mock_engine([
            self._hydrated_record(
                "MKT-HYDRA",
                ticker="KXHIGHNYC-T62",
                side=Side.NO,
                last_recorded_dollars=5.00,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (restored on restart)" in text
        assert "MKT-HYDRA" in text
        assert "$5.00" in text
        assert "(NO)" in text
        assert "(NO)" in text
        assert "restored from API" not in text
        # The pre-fix bug: HYDRATED stubs leaked under WORKING ORDERS with
        # a misleading "0 contracts @ $0.00" detail.
        assert "0 contracts @ $0.00" not in text
        assert "WORKING ORDERS" not in text
        assert "Total recovered:" in text

    def test_hydrated_with_restored_fill_info(self):
        """A HYDRATED stub with restored_fill_info shows real fill detail
        instead of the 'restored from API' fallback."""
        risk = _mock_risk(
            positions={"MKT-HYDRA": 5.00},
            total_exposure=5.00,
        )
        rec = self._hydrated_record(
            "MKT-HYDRA",
            ticker="KXHIGHNYC-T62",
            side=Side.NO,
            last_recorded_dollars=5.00,
        )
        rec.restored_fill_info = (0.50, 10)  # vwap=$0.50, 10 contracts
        engine = _mock_engine([rec])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (restored on restart)" in text
        assert "10 contracts @ $0.50" in text

    def test_hydrated_with_subsequent_fill(self):
        """A HYDRATED stub plus a real FILLED record on the same position_key
        (post-restart trade activity) shows under both OPEN POSITIONS (filled)
        and OPEN POSITIONS (restored on restart), with no spurious WORKING
        ORDERS line."""
        risk = _mock_risk(
            # 5.00 hydrated + 4.20 newly filled
            positions={"MKT-MIX": 9.20},
            total_exposure=9.20,
        )
        engine = _mock_engine([
            self._hydrated_record(
                "MKT-MIX",
                ticker="KXHIGHNYC-T62",
                side=Side.NO,
                last_recorded_dollars=5.00,
            ),
            self._filled_record(
                "MKT-MIX",
                ticker="KXHIGHNYC-T62",
                side=Side.YES,
                fill_count=12,
                fill_price=0.35,
                status=OrderStatus.FILLED,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        assert "OPEN POSITIONS (restored on restart)" in text
        assert "WORKING ORDERS" not in text
        assert "$4.20" in text  # filled portion
        assert "Total recovered:" in text

    def test_terminal_records_excluded_from_working(self):
        """A position whose only record is in a terminal/non-pending status
        (e.g., CANCELLED) with residual tracked exposure surfaces under
        WARNING: untracked exposure — not WORKING ORDERS."""
        risk = _mock_risk(
            positions={"MKT-CXL": 3.00},
            total_exposure=3.00,
        )
        engine = _mock_engine([
            self._filled_record(
                "MKT-CXL",
                ticker="KXHIGHNYC-T62",
                fill_count=0,
                fill_price=0.0,
                status=OrderStatus.CANCELLED,
                requested_count=10,
            ),
        ])

        text = format_positions(risk, engine)

        assert "WARNING: untracked exposure" in text
        assert "MKT-CXL" in text
        assert "$3.00" in text
        assert "WORKING ORDERS" not in text

    def test_cancelled_with_partial_fill_in_filled_bucket(self):
        """A CANCELLED record with ``fill_count > 0`` represents a partial
        fill whose remainder was cancelled (server-side TTL expiry on the
        resting tail, or a user/strategy cancel after some fills). The
        filled contracts are real exposure that must appear in OPEN
        POSITIONS (filled), not under WARNING: untracked exposure.

        Production trigger (2026-05-13): 8 of 11 ``untracked`` positions
        were partial fills whose resting portion was cancelled by Kalshi's
        ``expiration_ts`` after the order_ttl_seconds=120 window.
        """
        risk = _mock_risk(
            positions={"MKT-PART-CXL": 2.72},
            total_exposure=2.72,
        )
        engine = _mock_engine([
            self._filled_record(
                "MKT-PART-CXL",
                ticker="KXLOWTNYC-26MAY14-B52.5",
                side=Side.NO,
                fill_count=4,
                fill_price=0.68,
                status=OrderStatus.CANCELLED,
                requested_count=36,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        assert "MKT-PART-CXL" in text
        assert "$2.72" in text
        assert "(NO, 4 contracts @ $0.68)" in text
        # Must NOT appear in any other bucket
        assert "untracked — investigate" not in text
        assert "WORKING ORDERS" not in text
        # Totals must add up
        assert "Total filled:   $2.72" in text

    def test_expired_with_partial_fill_in_filled_bucket(self):
        """Same invariant as CANCELLED-with-partial-fill, but for EXPIRED
        (the engine's own TTL safety net path at engine.py:753 sets
        EXPIRED rather than CANCELLED when ``cancel_order`` is invoked
        client-side rather than via server-side ``expiration_ts``)."""
        risk = _mock_risk(
            positions={"MKT-PART-EXP": 5.00},
            total_exposure=5.00,
        )
        engine = _mock_engine([
            self._filled_record(
                "MKT-PART-EXP",
                ticker="KXLOWTNYC-26MAY14-B52.5",
                side=Side.YES,
                fill_count=10,
                fill_price=0.50,
                status=OrderStatus.EXPIRED,
                requested_count=36,
            ),
        ])

        text = format_positions(risk, engine)

        assert "OPEN POSITIONS (filled)" in text
        assert "MKT-PART-EXP" in text
        assert "$5.00" in text
        assert "(YES, 10 contracts @ $0.50)" in text
        assert "untracked — investigate" not in text

    def test_filled_status_zero_fill_count_still_untracked(self):
        """Defensive: a record with ``status=FILLED`` but ``fill_count == 0,
        fill_price=None`` (the Path A pre-fix state) has no fill data to
        display, so it must remain in the untracked bucket — there's
        nothing meaningful to show in OPEN POSITIONS (filled).

        With Fix 1 (engine 404 recovery), this state should no longer be
        reachable in practice — but the formatter must not paper over it
        if it ever does occur."""
        risk = _mock_risk(
            positions={"MKT-EMPTY-FILLED": 14.80},
            total_exposure=14.80,
        )
        engine = _mock_engine([
            self._filled_record(
                "MKT-EMPTY-FILLED",
                ticker="KXHIGHAUS-26MAY13-B88.5",
                fill_count=0,
                fill_price=0.0,
                status=OrderStatus.FILLED,
                requested_count=20,
            ),
        ])
        # _filled_record sets fill_price=0.0 by default; force None to mirror
        # the actual pre-fix shape (fill_price stays None when check_fills 404s).
        engine.order_history[0].fill_price = None

        text = format_positions(risk, engine)

        assert "WARNING: untracked exposure" in text
        assert "MKT-EMPTY-FILLED" in text
        assert "OPEN POSITIONS (filled)" not in text


class TestFormatTrades:
    def test_no_trades(self):
        engine = _mock_engine()
        assert format_trades(engine) == "No trades yet."

    def test_with_trades(self):
        record = _make_record()
        engine = _mock_engine([record])
        text = format_trades(engine)
        assert "RECENT TRADES" in text
        assert "KXHIGHNYC-26MAR22-T58" in text
        assert "weather_edge" in text

    def test_filters_rejected(self):
        record = _make_record(status=OrderStatus.REJECTED)
        engine = _mock_engine([record])
        text = format_trades(engine)
        assert text == "No trades yet."


class TestFormatSignals:
    def test_with_signals(self):
        record = _make_record(status=OrderStatus.REJECTED)
        engine = _mock_engine([record])
        text = format_signals(engine)
        assert "REJECTED" in text
        assert "KXHIGHNYC-26MAR22-T58" in text

    def test_filters_hydrated(self):
        """HYDRATED stubs are restart-recovery artifacts, not signals — they
        should not appear in /signals output."""
        hydrated = _make_record(
            status=OrderStatus.HYDRATED,
            market_ticker="KXHIGHNYC-HYDRA",
            strategy_name="live_hydrated",
        )
        rejected = _make_record(
            status=OrderStatus.REJECTED,
            market_ticker="KXHIGHNYC-REJ",
        )
        engine = _mock_engine([hydrated, rejected])
        text = format_signals(engine)
        assert "HYDRATED" not in text
        assert "KXHIGHNYC-HYDRA" not in text
        assert "REJECTED" in text
        assert "KXHIGHNYC-REJ" in text

    def test_only_hydrated_returns_empty(self):
        """If history contains only HYDRATED records, /signals reports
        no signals."""
        engine = _mock_engine([_make_record(status=OrderStatus.HYDRATED)])
        text = format_signals(engine)
        assert text == "No signals yet."


class TestFormatRisk:
    def test_basic(self):
        risk = _mock_risk(total_exposure=4_000.0, daily_pnl=-100.0, drawdown=0.03)
        settings = _mock_settings()
        text = format_risk(risk, settings)
        assert "RISK STATUS" in text
        assert "40.0%" in text  # exposure


class TestFormatTradeAlert:
    def test_basic(self):
        record = _make_record()
        text = format_trade_alert(record)
        assert "TRADE EXECUTED" in text
        assert "YES" in text
        assert "KXHIGHNYC-26MAR22-T58" in text
        assert "weather_edge" in text


class TestFormatHelp:
    def test_lists_all_commands(self):
        text = format_help()
        for cmd in ["/status", "/positions", "/trades", "/signals",
                    "/risk", "/kill", "/unkill", "/enable", "/help"]:
            assert cmd in text


class TestFormatAcks:
    def test_omits_ackable_section_when_not_passed(self) -> None:
        """Backward compat: callers that predate the `ackable` param get the
        original two-section output, not an empty third section."""
        from collections import Counter

        text = format_acks(acked={}, owner_is_live=lambda _o: False, suppressions=Counter())
        assert "ACKNOWLEDGEABLE" not in text

    def test_lists_ackable_codes(self) -> None:
        """The /ack rejection message points to /acks for 'codes seen so
        far' — this is the section that must satisfy that pointer."""
        from collections import Counter

        code = "calibrator_artifact_stale:favorite_bias:2026-07-04"
        text = format_acks(
            acked={},
            owner_is_live=lambda _o: False,
            suppressions=Counter(),
            ackable={code: ("favorite_bias", "WARNING")},
        )
        assert "ACKNOWLEDGEABLE" in text
        assert code in text
        assert "favorite_bias" in text

    def test_empty_ackable_dict_shows_none(self) -> None:
        from collections import Counter

        text = format_acks(
            acked={}, owner_is_live=lambda _o: False, suppressions=Counter(), ackable={}
        )
        assert "ACKNOWLEDGEABLE" in text
        assert "(none)" in text
