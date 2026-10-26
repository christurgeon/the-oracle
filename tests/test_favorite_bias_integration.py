"""End-to-end integration smoke test for FavoriteBiasStrategy.

Drives a realistic, live-API-shaped market batch through the real pipeline and
asserts a valid favorite_bias order is produced and executes cleanly in SHADOW
mode.  No network: Kalshi I/O is mocked; all other classes are real.

What this covers (the "traced on paper but never executed" gap):
  1. generate_signals(): live-API-shaped dict batch → ≥1 qualifying TradeSignal
  2. Gate filtering: non-qualifying markets (longshot mid, between strike-type,
     lead-2 date) are correctly excluded
  3. Signal fields: side=YES, strategy_name, effective_edge, taker metadata
  4. execute_batch() in SHADOW mode: signal → shadow OrderRecord written to
     state/trades.jsonl; RiskManager exposure updated
  5. Taker pricing in LIVE-style path: compute_limit_cents with
     _TAKER_ALWAYS_CROSS_CENTS fires when execution_taker=True (verified via
     a mocked Kalshi client that captures the limit-price cents submitted)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.common import OrderStatus, Side
from python.execution.engine import (
    _TAKER_ALWAYS_CROSS_CENTS,
    ExecutionEngine,
    compute_limit_cents,
)
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.favorite_bias import FavoriteBiasStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tomorrow_ticker(city_prefix: str, strike: int) -> str:
    """Build a market ticker whose forecast_date is UTC today + 1 (lead=1)."""
    tomorrow = datetime.now(UTC).date() + timedelta(days=1)
    month = tomorrow.strftime("%b").upper()
    yr = tomorrow.year % 100
    return f"{city_prefix}-{yr:02d}{month}{tomorrow.day:02d}-T{strike}"


def _day2_ticker(city_prefix: str, strike: int) -> str:
    """Build a ticker whose forecast_date is UTC today + 2 (lead=2)."""
    day2 = datetime.now(UTC).date() + timedelta(days=2)
    month = day2.strftime("%b").upper()
    yr = day2.year % 100
    return f"{city_prefix}-{yr:02d}{month}{day2.day:02d}-T{strike}"


def _make_settings(tmp_path: Path, mode: str = "shadow") -> Settings:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return Settings(
        state_dir=str(state),
        execution={"mode": mode},  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )


def _strategy_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "enabled": True,
        "band_low": 0.40,
        "band_high": 0.75,
        "min_volume_fp": 200,
        "edge_floor": 0.02,
        "lead": 1,
        "strike_types": ["greater", "less"],
        "isotonic_path": "data/calibration/favorite_bias_isotonic.json",
        "kelly_fraction": 0.05,
        "max_pct": 0.05,
        "max_contracts_per_trade": 25,
        "max_bets_per_day_per_event": 1,
        "audit_path": "/tmp/fb_integration_audit.jsonl",
        "execution_taker": True,
        "cities": [],  # accept all cities — no city filter
    }
    base.update(overrides)
    return base


def _make_fake_client(markets_by_series: dict[str, list[dict[str, Any]]]) -> Any:
    """Return a mock Kalshi client whose iter_all_markets yields test markets.

    FavoriteBiasStrategy.scan() calls:
        markets, _trunc = await self._kalshi.iter_all_markets(
            status="open", series_ticker=series
        )
    So the mock must return a (list, bool) tuple, NOT just the list.
    """
    client = MagicMock()

    async def _iter(status: str = "open", series_ticker: str = "") -> tuple[list[Any], bool]:
        return markets_by_series.get(series_ticker, []), False

    client.iter_all_markets = AsyncMock(side_effect=_iter)
    return client


# ---------------------------------------------------------------------------
# Market dict factory (live-API shape — no city/daily_variable keys)
# ---------------------------------------------------------------------------


def _market(
    ticker: str,
    *,
    yes_bid: float,
    yes_ask: float,
    volume_fp: int = 500,
    strike_type: str = "greater",
    floor_strike: float | None = 60.0,
    cap_strike: float | None = None,
    close_time: str = "2099-01-01T00:00:00+00:00",
) -> dict[str, Any]:
    """Live-API-shaped market dict (no city/daily_variable keys)."""
    return {
        "ticker": ticker,
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "volume_fp": volume_fp,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "close_time": close_time,
        "expiration_ts": "",
    }


# ---------------------------------------------------------------------------
# 1.  generate_signals() integration
# ---------------------------------------------------------------------------


class TestGenerateSignalsE2E:
    """generate_signals() drives live-API-shaped dicts through the real pipeline."""

    @pytest.mark.asyncio
    async def test_qualifying_market_produces_signal(self, tmp_path: Path) -> None:
        """A lead-1 greater market in the favorite band should yield ≥1 signal."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
            floor_strike=60.0,
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()

        assert len(signals) >= 1, (
            f"Expected ≥1 signal from a qualifying favorite-band market; got 0. "
            f"Ticker={qualifying_ticker}, mid=0.54"
        )
        sig = signals[0]
        assert sig.strategy_name == "favorite_bias"
        assert sig.side == Side.YES
        assert sig.effective_edge > 0.0, "effective_edge must be > 0"

    @pytest.mark.asyncio
    async def test_signal_metadata_populated(self, tmp_path: Path) -> None:
        """The signal metadata must carry fill_price, execution_taker, city,
        market_type (var), and forecast_date — fields downstream engine and
        risk code read."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
            floor_strike=60.0,
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert len(signals) >= 1
        sig = signals[0]
        md = sig.metadata

        # fill_price must equal yes_ask (taker, spec §3.3)
        assert md.get("fill_price") == pytest.approx(0.55), (
            f"fill_price should be yes_ask=0.55; got {md.get('fill_price')}"
        )
        assert md.get("execution_taker") is True
        # city must be derived from the ticker (live-API has no city key)
        assert md.get("city") == "NYC", (
            f"city should be 'NYC' derived from KXHIGHNY; got {md.get('city')!r}"
        )
        assert md.get("market_type") in ("high", "low"), (
            f"market_type got {md.get('market_type')!r}"
        )
        assert md.get("forecast_date") is not None, "forecast_date must be populated"
        assert md.get("yes_ask") == pytest.approx(0.55)
        assert md.get("yes_bid") == pytest.approx(0.53)

    @pytest.mark.asyncio
    async def test_non_qualifying_markets_excluded(self, tmp_path: Path) -> None:
        """Non-qualifying markets must be filtered out; only the qualifying one survives."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        longshot_ticker = _tomorrow_ticker("KXHIGHCHI", 90)  # mid=0.25 → below band
        between_ticker = _tomorrow_ticker("KXHIGHMIA", 75)  # strike_type=between → rejected
        lead2_ticker = _day2_ticker("KXHIGHLAX", 70)  # lead=2 → rejected

        qualifying = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )
        longshot = _market(
            longshot_ticker,
            yes_bid=0.23,
            yes_ask=0.27,  # mid=0.25 → below band_low=0.40
            volume_fp=500,
            strike_type="less",
        )
        between_mkt = _market(
            between_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="between",  # rejected: not in strike_types=[greater,less]
            floor_strike=74.0,
            cap_strike=75.0,
        )
        lead2 = _market(
            lead2_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client(
            {
                "KXHIGHNY": [qualifying],
                "KXHIGHCHI": [longshot],
                "KXHIGHMIA": [between_mkt],
                "KXHIGHLAX": [lead2],
            }
        )
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        signal_tickers = {s.market_ticker for s in signals}

        assert qualifying_ticker in signal_tickers, (
            f"Qualifying lead-1 greater market not in signals: {signal_tickers}"
        )
        assert longshot_ticker not in signal_tickers, (
            "Longshot (mid=0.25, below band) must be filtered out"
        )
        assert between_ticker not in signal_tickers, (
            "Between strike_type must be filtered out (not in strike_types)"
        )
        assert lead2_ticker not in signal_tickers, (
            "Lead-2 market must be filtered out (lead=1 config)"
        )

    @pytest.mark.asyncio
    async def test_signals_all_yes_side(self, tmp_path: Path) -> None:
        """FavoriteBias is YES-only (spec §3.1(7)); no NO signals allowed."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        for sig in signals:
            assert sig.side == Side.YES, (
                f"FavoriteBias must be YES-only; got side={sig.side} for {sig.market_ticker}"
            )


# ---------------------------------------------------------------------------
# 2.  execute_batch() shadow execution integration
# ---------------------------------------------------------------------------


class TestShadowExecutionE2E:
    """Push real FavoriteBias signals through the real ExecutionEngine in SHADOW
    mode and verify the full seam produces correct state."""

    @pytest.mark.asyncio
    async def test_shadow_execute_produces_filled_record(self, tmp_path: Path) -> None:
        """A valid FB signal must produce a SHADOW OrderRecord (not REJECTED)."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert signals, "No signals generated — cannot test execution seam"

        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        records = await engine.execute_batch(signals)
        assert records, "execute_batch returned empty list"

        shadow_records = [r for r in records if r.status == OrderStatus.SHADOW]
        assert shadow_records, (
            f"Expected at least one SHADOW record; got statuses: "
            f"{[r.status for r in records]}"
        )

    @pytest.mark.asyncio
    async def test_shadow_execute_updates_risk_exposure(self, tmp_path: Path) -> None:
        """After SHADOW execution, RiskManager.total_exposure must be > 0."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert signals

        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        await engine.execute_batch(signals)

        assert risk.total_exposure > 0, (
            f"RiskManager.total_exposure should be > 0 after shadow fill; "
            f"got {risk.total_exposure}"
        )
        # Verify a position entry exists for the qualifying ticker
        assert any(
            qualifying_ticker in k for k in risk.positions.keys()
        ), f"Position for {qualifying_ticker} not found in risk.positions: {dict(risk.positions)}"

    @pytest.mark.asyncio
    async def test_shadow_execute_writes_trade_log(self, tmp_path: Path) -> None:
        """execute_batch in SHADOW mode must write at least one row to trades.jsonl."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert signals

        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        records = await engine.execute_batch(signals)
        shadow_recs = [r for r in records if r.status == OrderStatus.SHADOW]
        assert shadow_recs, "No shadow records produced"

        # Verify the trade log file was written and contains a shadow fill row
        trade_log = Path(settings.state_dir) / "trades.jsonl"
        assert trade_log.exists(), f"Trade log not found at {trade_log}"

        rows: list[dict[str, Any]] = []
        with trade_log.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        shadow_rows = [r for r in rows if r.get("mode") == "shadow"]
        assert shadow_rows, (
            f"No shadow rows in trade log; rows present: {[r.get('mode') for r in rows]}"
        )

        # The row must reference the qualifying market
        row_tickers = {r.get("market_ticker") for r in shadow_rows}
        assert qualifying_ticker in row_tickers, (
            f"Qualifying ticker {qualifying_ticker!r} not in trade log rows: {row_tickers}"
        )

    @pytest.mark.asyncio
    async def test_shadow_record_strategy_name_is_favorite_bias(self, tmp_path: Path) -> None:
        """The shadow fill row in trades.jsonl must have strategy=favorite_bias."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert signals

        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)
        await engine.execute_batch(signals)

        trade_log = Path(settings.state_dir) / "trades.jsonl"
        rows = []
        with trade_log.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        shadow_rows = [r for r in rows if r.get("mode") == "shadow"]
        assert shadow_rows
        strats = {r.get("strategy") for r in shadow_rows}
        assert "favorite_bias" in strats, (
            f"strategy field in trade log should be 'favorite_bias'; got {strats}"
        )


# ---------------------------------------------------------------------------
# 3.  Taker-pricing end-to-end via mocked Kalshi client (LIVE mode)
# ---------------------------------------------------------------------------


class TestTakerPricingE2E:
    """Verify that the taker wiring fires end-to-end when a FavoriteBias signal
    reaches _live_execute with execution_taker=True in the metadata.

    SHADOW mode skips compute_limit_cents (it uses the midpoint).  To exercise
    the taker wiring we run the engine in LIVE mode with a mocked Kalshi client
    and assert that place_order is called with price_cents equal to the yes_ask
    in cents.
    """

    @pytest.mark.asyncio
    async def test_taker_price_submitted_to_place_order(self, tmp_path: Path) -> None:
        """execution_taker=True must result in compute_limit_cents using
        _TAKER_ALWAYS_CROSS_CENTS (100) → limit price == yes_ask in cents,
        regardless of spread width.

        yes_bid=0.53, yes_ask=0.55 → spread=2c.  Default threshold=1 would post
        INSIDE the spread; taker threshold=100 must cross to ask (55¢).
        """
        yes_bid = 0.53
        yes_ask = 0.55
        expected_limit_cents = 55  # yes_ask in cents

        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume_fp=500,
            strike_type="greater",
        )

        # Generate a signal through the strategy (shadow client, real pipeline)
        fb_client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=fb_client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )
        signals = await strategy.generate_signals()
        assert signals, "No signals produced — cannot test taker path"
        sig = signals[0]
        assert sig.metadata.get("execution_taker") is True, (
            "Signal must carry execution_taker=True for this test to be valid"
        )

        # Mock Kalshi client for the live execution path
        kalshi_mock = AsyncMock()

        # Capture the cents submitted to place_order
        submitted_price_cents: list[int] = []

        async def _fake_place_order(
            *,
            ticker: str,
            side: str,
            action: str,
            order_type: str,
            count: int,
            yes_price: int | None = None,
            no_price: int | None = None,
            expiration_ts: int | None = None,
        ) -> dict[str, Any]:
            if yes_price is not None:
                submitted_price_cents.append(yes_price)
            return {
                "order": {
                    "order_id": "test-order-001",
                    "status": "resting",
                    "fill_count_fp": "0",
                }
            }

        kalshi_mock.place_order = AsyncMock(side_effect=_fake_place_order)

        settings = _make_settings(tmp_path, mode="live")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=kalshi_mock, risk_manager=risk)

        # Execute in LIVE mode — this calls _live_execute → compute_limit_cents
        record = await engine.execute(sig)

        # Verify place_order was actually called (not rejected/zero-size)
        assert kalshi_mock.place_order.called, (
            f"place_order was not called; record.status={record.status}. "
            f"This means the signal was rejected or zero-sized before reaching "
            f"_live_execute. Check risk limits or contract count."
        )
        assert submitted_price_cents, "No price cents captured from place_order call"
        actual_cents = submitted_price_cents[0]
        assert actual_cents == expected_limit_cents, (
            f"Taker pricing failed: expected limit_cents={expected_limit_cents} "
            f"(yes_ask in cents with _TAKER_ALWAYS_CROSS_CENTS={_TAKER_ALWAYS_CROSS_CENTS}), "
            f"got {actual_cents}. Check that execution_taker=True in metadata routes "
            f"to _TAKER_ALWAYS_CROSS_CENTS in _live_execute."
        )

    def test_compute_limit_cents_taker_override_wide_spread(self) -> None:
        """Unit-verify the compute_limit_cents taker mechanic directly.

        With yes_bid=53¢, yes_ask=55¢ (2c spread) and execution_taker=True:
          - Default cross_to_ask_max_spread_cents=1 would post at 54¢ (maker, inside)
          - _TAKER_ALWAYS_CROSS_CENTS=100 must produce 55¢ (taker, cross to ask)
        This is the same computation that _live_execute performs when it reads
        metadata["execution_taker"]=True.
        """
        yes_bid_cents = 53
        yes_ask_cents = 55
        order_price = 0.54  # midpoint

        # Default (maker path) would post inside
        maker_cents = compute_limit_cents(
            order_price_dollars=order_price,
            side_bid_cents=yes_bid_cents,
            side_ask_cents=yes_ask_cents,
            cross_to_ask_max_spread_cents=1,
        )
        assert maker_cents != yes_ask_cents, (
            f"Maker path should NOT cross to ask on 2c spread; got {maker_cents}"
        )

        # Taker path (execution_taker=True → _TAKER_ALWAYS_CROSS_CENTS=100)
        taker_cents = compute_limit_cents(
            order_price_dollars=order_price,
            side_bid_cents=yes_bid_cents,
            side_ask_cents=yes_ask_cents,
            cross_to_ask_max_spread_cents=_TAKER_ALWAYS_CROSS_CENTS,
        )
        assert taker_cents == yes_ask_cents, (
            f"Taker path must cross to ask ({yes_ask_cents}¢); got {taker_cents}¢"
        )

    @pytest.mark.asyncio
    async def test_taker_metadata_flag_survives_full_pipeline(self, tmp_path: Path) -> None:
        """Verify that execution_taker=True set in the strategy config is carried
        all the way through generate_signals → signal.metadata → execute()
        without being dropped or overwritten anywhere in the pipeline."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )

        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(execution_taker=True),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert signals, "No signals — cannot verify metadata survival"
        sig = signals[0]

        # Check the flag is set on the signal that reaches the engine
        assert sig.metadata.get("execution_taker") is True, (
            f"execution_taker=True was lost in the pipeline; "
            f"metadata keys present: {list(sig.metadata.keys())}"
        )
        # And fill_price (taker price = yes_ask) is load-bearing
        assert sig.metadata.get("fill_price") == pytest.approx(0.55), (
            f"fill_price should equal yes_ask=0.55; got {sig.metadata.get('fill_price')}"
        )
        # Verify exec_price on the signal itself also equals yes_ask
        assert sig.exec_price == pytest.approx(0.55), (
            f"sig.exec_price should be yes_ask=0.55; got {sig.exec_price}"
        )


# ---------------------------------------------------------------------------
# 4.  Full seam: strategy → engine → log in one call
# ---------------------------------------------------------------------------


class TestFullSeamE2E:
    """Exercise the complete strategy→risk→engine→log pipeline in one shot."""

    @pytest.mark.asyncio
    async def test_full_seam_shadow_mode(self, tmp_path: Path) -> None:
        """Prove that the full seam from generate_signals to trade log write
        works end-to-end without exceptions, rejection, or silent no-ops."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )
        # Also add a less-type qualifying market to confirm multi-market batch works
        qualifying_ticker_less = _tomorrow_ticker("KXLOWTNYC", 58)
        qualifying_market_less = _market(
            qualifying_ticker_less,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=300,
            strike_type="less",
            floor_strike=None,
            cap_strike=58.0,
        )

        client = _make_fake_client(
            {
                "KXHIGHNY": [qualifying_market],
                "KXLOWTNYC": [qualifying_market_less],
            }
        )
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(max_bets_per_day_per_event=2),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        # Step 1: generate
        signals = await strategy.generate_signals()
        assert signals, "No signals generated — seam is broken at generate_signals"
        assert all(s.strategy_name == "favorite_bias" for s in signals), (
            "strategy_name must be 'favorite_bias' on all signals"
        )

        # Step 2: execute
        records = await engine.execute_batch(signals)
        assert records, "execute_batch returned nothing"
        shadow_recs = [r for r in records if r.status == OrderStatus.SHADOW]
        assert shadow_recs, (
            f"No SHADOW records; statuses were: {[r.status for r in records]}"
        )

        # Step 3: risk state updated
        assert risk.total_exposure > 0, "RiskManager exposure not updated after execution"

        # Step 4: trade log written
        trade_log = Path(settings.state_dir) / "trades.jsonl"
        assert trade_log.exists()
        rows = [json.loads(line) for line in trade_log.read_text().splitlines() if line.strip()]
        assert rows, "Trade log is empty"
        assert any(r.get("strategy") == "favorite_bias" for r in rows), (
            f"No favorite_bias rows in trade log; strategies found: "
            f"{[r.get('strategy') for r in rows]}"
        )

    @pytest.mark.asyncio
    async def test_disabled_strategy_produces_no_signals_no_execution(
        self, tmp_path: Path
    ) -> None:
        """enabled=False must short-circuit generate_signals → execute_batch touches nothing."""
        qualifying_ticker = _tomorrow_ticker("KXHIGHNY", 60)
        qualifying_market = _market(
            qualifying_ticker,
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )
        client = _make_fake_client({"KXHIGHNY": [qualifying_market]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(enabled=False),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        signals = await strategy.generate_signals()
        assert signals == [], (
            f"enabled=False must return empty signals; got {signals}"
        )

        settings = _make_settings(tmp_path, mode="shadow")
        risk = RiskManager(settings)
        engine = ExecutionEngine(settings, kalshi_client=None, risk_manager=risk)

        records = await engine.execute_batch(signals)
        assert records == [], "No records expected when signals=[]"
        assert risk.total_exposure == 0.0, "Exposure should be 0 when no trades executed"


# ---------------------------------------------------------------------------
# 5.  Per-scan summary log (observability)
# ---------------------------------------------------------------------------


class TestPerScanSummaryLog:
    """generate_signals() must emit a single INFO summary line every scan so a
    healthy-but-idle scan (0 in-band candidates) is distinguishable from a
    non-running / silently-failing strategy — mirrors weather_edge's
    'generated N signal(s) from M market(s)' line.
    """

    _LOGGER = "python.strategies.favorite_bias"

    @pytest.mark.asyncio
    async def test_active_scan_logs_summary_with_counts(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One qualifying favorite + one below-band longshot → the summary must
        report 1 signal / 2 markets / 1 in-band / 0 suppressed."""
        qualifying = _market(
            _tomorrow_ticker("KXHIGHNY", 60),
            yes_bid=0.53,
            yes_ask=0.55,
            volume_fp=500,
            strike_type="greater",
        )
        longshot = _market(
            _tomorrow_ticker("KXHIGHCHI", 90),
            yes_bid=0.23,
            yes_ask=0.27,  # mid=0.25 → below band_low=0.40, not in-band
            volume_fp=500,
            strike_type="less",
        )
        client = _make_fake_client({"KXHIGHNY": [qualifying], "KXHIGHCHI": [longshot]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            await strategy.generate_signals()

        summaries = [
            r.getMessage() for r in caplog.records if "favorite_bias: generated" in r.getMessage()
        ]
        assert len(summaries) == 1, f"Expected exactly one summary line; got {summaries}"
        msg = summaries[0]
        assert "generated 1 signal(s) from 2 market(s)" in msg, msg
        assert "1 in-band" in msg, msg
        assert "0 suppressed by per-day cap" in msg, msg

    @pytest.mark.asyncio
    async def test_idle_scan_still_logs_summary(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A scan with only a below-band longshot (0 in-band, 0 signals) must
        STILL emit the summary — the whole point is to prove liveness when idle."""
        longshot = _market(
            _tomorrow_ticker("KXHIGHCHI", 90),
            yes_bid=0.23,
            yes_ask=0.27,
            volume_fp=500,
            strike_type="less",
        )
        client = _make_fake_client({"KXHIGHCHI": [longshot]})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            await strategy.generate_signals()

        summaries = [
            r.getMessage() for r in caplog.records if "favorite_bias: generated" in r.getMessage()
        ]
        assert len(summaries) == 1, f"Idle scan must still log a summary; got {summaries}"
        msg = summaries[0]
        assert "generated 0 signal(s) from 1 market(s)" in msg, msg
        assert "0 in-band" in msg, msg

    @pytest.mark.asyncio
    async def test_disabled_strategy_does_not_log_summary(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """enabled=False short-circuits before any scan — no summary line."""
        client = _make_fake_client({"KXHIGHNY": []})
        strategy = FavoriteBiasStrategy(
            config=_strategy_config(enabled=False),
            kalshi_client=client,
            bankroll=1000.0,
            trades_log_path=str(tmp_path / "trades.jsonl"),
        )

        with caplog.at_level(logging.INFO, logger=self._LOGGER):
            await strategy.generate_signals()

        summaries = [
            r.getMessage() for r in caplog.records if "favorite_bias: generated" in r.getMessage()
        ]
        assert summaries == [], f"Disabled strategy must not log a summary; got {summaries}"
