"""favorite_bias bounded book-walk: price the taker limit through the ask ladder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.execution.engine import ExecutionEngine
from python.risk.manager import RiskManager
from python.settings import Settings
from python.strategies.favorite_bias import (
    FavoriteBiasStrategy,
    plan_book_walk,
    yes_ask_levels,
)


def _book(yes_asks: list[tuple[float, float]]) -> dict[str, Any]:
    """Orderbook payload whose NO bids produce the given YES asks."""
    return {
        "orderbook_fp": {
            "no_dollars": [[f"{1 - p:.4f}", f"{q:.2f}"] for p, q in yes_asks],
            "yes_dollars": [["0.5300", "40.00"]],
        }
    }


def test_yes_ask_levels_inverts_no_bids_and_sorts() -> None:
    levels = yes_ask_levels(_book([(0.57, 30), (0.55, 10), (0.56, 20)]))
    assert levels == [(0.55, 10), (0.56, 20), (0.57, 30)]


def test_yes_ask_levels_empty_on_missing_book() -> None:
    assert yes_ask_levels({}) == []


def _plan(levels: list[tuple[float, int]], **kw: Any):
    args: dict[str, Any] = dict(
        model_prob=0.642,
        signal_ask=0.55,
        target_contracts=50,
        max_walk_cents=2,
        edge_floor=0.02,
    )
    args.update(kw)
    return plan_book_walk(levels, **args)


def test_plan_stops_at_cheapest_level_covering_target() -> None:
    plan = _plan([(0.55, 10), (0.56, 20), (0.57, 30)], target_contracts=25)
    assert plan is not None
    assert plan.limit_price == 0.56
    assert plan.contracts == 25
    assert plan.vwap == pytest.approx((10 * 0.55 + 15 * 0.56) / 25)


def test_plan_never_walks_past_max_walk_cents() -> None:
    plan = _plan([(0.55, 10), (0.56, 10), (0.57, 10), (0.58, 100)])
    assert plan is not None
    assert plan.limit_price == 0.57
    assert plan.contracts == 30


def test_plan_stops_where_marginal_edge_fails_floor() -> None:
    # g=0.605: edge at 0.56 = 0.605-0.56-0.02 = 0.025 (ok); at 0.57 = 0.015
    # < floor -> stop even though 0.57 is within the walk budget.
    plan = _plan([(0.55, 10), (0.56, 10), (0.57, 10)], model_prob=0.605)
    assert plan is not None
    assert plan.limit_price == 0.56


def test_plan_uses_cheaper_levels_when_book_improved() -> None:
    plan = _plan([(0.53, 40), (0.55, 40)], target_contracts=30)
    assert plan is not None
    assert plan.limit_price == 0.53


def test_plan_none_when_best_ask_moved_beyond_budget() -> None:
    assert _plan([(0.60, 100)]) is None


def test_plan_zero_walk_is_top_of_book_only() -> None:
    plan = _plan([(0.55, 10), (0.56, 10)], max_walk_cents=0)
    assert plan is not None
    assert plan.limit_price == 0.55
    assert plan.contracts == 10


def _ticker() -> str:
    d = datetime.now(UTC).date() + timedelta(days=1)
    return f"KXHIGHNY-{d.year % 100:02d}{d.strftime('%b').upper()}{d.day:02d}-T60"


def _client(book: dict[str, Any] | Exception) -> Any:
    ticker = _ticker()
    market = {
        "ticker": ticker,
        "yes_bid_dollars": 0.53,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "close_time": "2099-01-01T00:00:00+00:00",
    }
    client = MagicMock()

    async def _iter(status: str = "open", series_ticker: str = "") -> tuple[list[Any], bool]:
        return ([market] if series_ticker == "KXHIGHNY" else []), False

    client.iter_all_markets = AsyncMock(side_effect=_iter)
    if isinstance(book, Exception):
        client.get_orderbook = AsyncMock(side_effect=book)
    else:
        client.get_orderbook = AsyncMock(return_value=book)
    return client


def _strategy(client: Any, tmp_path: Path, **kw: Any) -> FavoriteBiasStrategy:
    cfg: dict[str, Any] = dict(
        enabled=True,
        band_low=0.50,
        band_high=0.75,
        min_volume_fp=50,
        edge_floor=0.01,
        lead=1,
        strike_types=["greater", "less"],
        isotonic_path="data/calibration/favorite_bias_isotonic.json",
        kelly_fraction=0.60,
        max_pct=0.20,
        max_contracts_per_trade=100,
        max_bets_per_day_per_event=1,
        audit_path=str(tmp_path / "audit.jsonl"),
        execution_taker=True,
        cities=[],
    )
    cfg.update(kw)
    return FavoriteBiasStrategy(
        config=cfg,
        kalshi_client=client,
        bankroll=400.0,
        trades_log_path=str(tmp_path / "trades.jsonl"),
    )


@pytest.mark.asyncio
async def test_walk_disabled_never_fetches_orderbook(tmp_path: Path) -> None:
    client = _client(_book([(0.55, 5)]))
    sigs = await _strategy(client, tmp_path).generate_signals()
    assert sigs
    client.get_orderbook.assert_not_called()
    assert "limit_price" not in sigs[0].metadata


@pytest.mark.asyncio
async def test_walk_sets_limit_and_vwap_and_keeps_signal_ask(tmp_path: Path) -> None:
    client = _client(_book([(0.55, 5), (0.56, 10), (0.57, 200)]))
    baseline = (await _strategy(_client({}), tmp_path).generate_signals())[0]
    sig = (await _strategy(client, tmp_path, max_book_walk_cents=2).generate_signals())[0]
    md = sig.metadata
    assert md["yes_ask"] == 0.55  # signal-time ask stays the reporting reference
    assert md["limit_price"] == 0.57
    assert md["book_walk_cents"] == 2
    assert 0.55 < md["fill_price"] < 0.57
    # Kelly re-run at the worse VWAP can only shrink the order.
    assert 1 <= sig.suggested_contracts <= baseline.suggested_contracts
    assert md["book_walk_visible_contracts"] == sig.suggested_contracts


@pytest.mark.asyncio
async def test_walk_fails_open_on_orderbook_error(tmp_path: Path) -> None:
    client = _client(RuntimeError("429"))
    sig = (await _strategy(client, tmp_path, max_book_walk_cents=2).generate_signals())[0]
    assert "limit_price" not in sig.metadata
    assert sig.metadata["fill_price"] == 0.55


@pytest.mark.asyncio
async def test_walk_keeps_signal_when_no_eligible_depth(tmp_path: Path) -> None:
    client = _client(_book([(0.62, 100)]))
    sig = (await _strategy(client, tmp_path, max_book_walk_cents=2).generate_signals())[0]
    assert "limit_price" not in sig.metadata


@pytest.mark.asyncio
async def test_live_engine_submits_walked_limit(tmp_path: Path) -> None:
    client = _client(_book([(0.55, 5), (0.56, 200)]))
    sig = (await _strategy(client, tmp_path, max_book_walk_cents=2).generate_signals())[0]
    assert sig.metadata["limit_price"] == 0.56

    submitted: list[int] = []

    async def _place(**kw: Any) -> dict[str, Any]:
        submitted.append(kw["yes_price"])
        return {"order": {"order_id": "o-1", "status": "resting", "fill_count_fp": "0"}}

    kalshi = AsyncMock()
    kalshi.place_order = AsyncMock(side_effect=_place)
    state = tmp_path / "state"
    state.mkdir()
    settings = Settings(
        state_dir=str(state),
        execution={"mode": "live"},  # type: ignore[arg-type]
        _env_file=None,  # type: ignore[call-arg]
    )
    engine = ExecutionEngine(settings, kalshi_client=kalshi, risk_manager=RiskManager(settings))
    await engine.execute(sig)
    assert submitted == [56]
