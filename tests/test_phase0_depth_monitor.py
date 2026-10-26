"""Unit tests for the Phase 0 passive depth-probing monitor.

Covers the M0 surface area:
  - sample-loop behaviour with no eligible same-day tickers
  - sample-loop behaviour on an iter_all_markets error
  - JSONL row schema correctness on a populated market dict
  - sample-loop cadence (the run-once entry point batches one tick)
  - read-budget invariant: NO ``get_orderbook`` call per tick
  - ``iter_all_markets`` truncation surfaces as a WARNING + counter
  - cancellation mid-tick exits cleanly
  - writer errors are swallowed (no propagation)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from python.data.phase0_depth_monitor import (
    PHASE0_DEPTH_SINK_FILENAME,
    PHASE0_ELIGIBLE_SERIES,
    PHASE0_SAMPLE_KIND,
    Phase0DepthMonitor,
    build_depth_sample,
)

# ---------------------------------------------------------------------------
# build_depth_sample (pure JSONL row constructor)
# ---------------------------------------------------------------------------


_NOW = datetime(2026, 5, 15, 18, 30, 0, tzinfo=UTC)


def _make_market(
    ticker: str = "KXHIGHMIA-26MAY15H1700-B85",
    *,
    yes_bid_dollars: str | None = "0.42",
    yes_ask_dollars: str | None = "0.45",
    yes_bid_size_fp: str | None = "30.00",
    yes_ask_size_fp: str | None = "25.00",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "yes_bid_dollars": yes_bid_dollars,
        "yes_ask_dollars": yes_ask_dollars,
        "yes_bid_size_fp": yes_bid_size_fp,
        "yes_ask_size_fp": yes_ask_size_fp,
    }


class TestBuildDepthSample:
    def test_populated_market_produces_full_schema_with_no_l2_data(self):
        # Without an ``orderbook`` payload, the row has TOB fields populated
        # and L2 fields = None (the prober's pre-L2-pass row template).
        market = _make_market(
            ticker="KXHIGHMIA-26MAY15H1700-B85",
            yes_bid_dollars="0.42",
            yes_ask_dollars="0.45",
            yes_bid_size_fp="30.00",
            yes_ask_size_fp="25.00",
        )
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW)
        assert row is not None
        assert row["sample_kind"] == PHASE0_SAMPLE_KIND
        assert row["ts"] == _NOW.isoformat()
        assert row["ts_unix"] == pytest.approx(_NOW.timestamp())
        assert row["ticker"] == "KXHIGHMIA-26MAY15H1700-B85"
        assert row["series_ticker"] == "KXHIGHMIA"
        assert row["cell"] == "MIA/high"
        assert row["city"] == "MIA"
        assert row["yes_bid"] == pytest.approx(0.42)
        assert row["yes_bid_size"] == 30
        # L2 fields are None when no orderbook payload is supplied.
        assert row["yes_bid_l2_size"] is None
        assert row["yes_bid_l2_price"] is None
        assert row["yes_ask_l2_size"] is None
        assert row["yes_ask_l2_price"] is None
        assert row["l2_present"] is False
        assert row["yes_levels_count"] is None
        assert row["no_levels_count"] is None
        assert row["yes_ask"] == pytest.approx(0.45)
        assert row["yes_ask_size"] == 25
        # NO-side fields not available from /markets — None for forward compat.
        assert row["no_bid"] is None
        assert row["no_bid_size"] is None
        assert row["no_bid_l2_size"] is None
        assert row["mid"] == pytest.approx((0.42 + 0.45) / 2.0)
        assert row["spread"] == pytest.approx(0.45 - 0.42)
        assert row["empty_book"] is False

    def test_populated_market_with_l2_orderbook_populates_l2_fields(self):
        # Sibling to the schema test above — when an ``orderbook_fp``
        # payload is supplied, the L2 fields read second-best level
        # depth from both sides.
        market = _make_market(
            ticker="KXHIGHMIA-26MAY15H1700-B85",
            yes_bid_dollars="0.42",
            yes_ask_dollars="0.45",
            yes_bid_size_fp="30.00",
            yes_ask_size_fp="25.00",
        )
        orderbook = {
            "orderbook_fp": {
                "yes_dollars": [["0.42", "30"], ["0.40", "50"]],
                "no_dollars": [["0.55", "25"], ["0.53", "75"]],
            }
        }
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=orderbook)
        assert row is not None
        assert row["yes_bid_l2_size"] == 50
        assert row["yes_bid_l2_price"] == pytest.approx(0.40)
        # NO L2 bid = 0.53 → derived YES L2 ask = 1 - 0.53 = 0.47.
        assert row["yes_ask_l2_size"] == 75
        assert row["yes_ask_l2_price"] == pytest.approx(0.47)
        assert row["yes_levels_count"] == 2
        assert row["no_levels_count"] == 2
        assert row["l2_present"] is True

    def test_empty_market_flags_empty_book(self):
        market = _make_market(
            yes_bid_dollars=None,
            yes_ask_dollars=None,
            yes_bid_size_fp="0",
            yes_ask_size_fp="0",
        )
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW)
        assert row is not None
        assert row["empty_book"] is True
        assert row["yes_bid"] is None
        assert row["yes_ask"] is None
        assert row["mid"] is None
        assert row["spread"] is None

    def test_lax_series_maps_to_lax_high_cell(self):
        market = _make_market(ticker="KXHIGHLAX-26MAY15H1700-B85")
        row = build_depth_sample(market, "KXHIGHLAX", now_utc=_NOW)
        assert row is not None
        assert row["cell"] == "LAX/high"
        assert row["city"] == "LAX"

    def test_unknown_series_returns_none(self):
        # Defensive: an out-of-scope series should be rejected, not coerced.
        market = _make_market(ticker="KXHIGHNY-26MAY15")
        row = build_depth_sample(market, "KXHIGHNY", now_utc=_NOW)
        assert row is None

    def test_malformed_size_is_silently_coerced(self):
        # _parse_fp_int returns None on garbage; row constructor must still
        # produce a row with size=0 and empty_book=True if both sides empty.
        market = _make_market(
            yes_bid_dollars=None,
            yes_ask_dollars=None,
            yes_bid_size_fp="nope",
            yes_ask_size_fp="bogus",
        )
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW)
        assert row is not None
        assert row["yes_bid_size"] == 0
        assert row["yes_ask_size"] == 0
        assert row["empty_book"] is True


# ---------------------------------------------------------------------------
# Phase0DepthMonitor.run_once
# ---------------------------------------------------------------------------


@pytest.fixture
def kalshi_client() -> MagicMock:
    """A MagicMock Kalshi client with the async methods the monitor uses.

    Note: as of the 2026-05-17 L2 reinstatement, the monitor DOES call
    ``get_orderbook`` for L2 capture. The default ``{"orderbook_fp": {}}``
    response yields L2 fields = None on every row (empty envelope ⇒
    no L2 data), preserving the pre-L2 assertion semantics in existing
    tests that don't care about L2.
    """
    client = MagicMock()
    client.iter_all_markets = AsyncMock(return_value=([], False))
    client.get_orderbook = AsyncMock(return_value={"orderbook_fp": {}})
    return client


def _today_market_ticker(series: str = "KXHIGHMIA") -> str:
    """Build a ticker whose embedded date == today (the monitor's eligibility key)."""
    from python.common import et_today

    d = et_today()
    return f"{series}-{d.strftime('%y%b%d').upper()}H1700-B85"


def _other_day_market_ticker(series: str = "KXHIGHMIA") -> str:
    return f"{series}-20JAN01H1700-B85"


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_no_eligible_tickers_writes_nothing(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        # iter_all_markets returns no rows: monitor must not call
        # get_orderbook (no eligible markets ⇒ no L2 pass) and must
        # not create the sink file.
        kalshi_client.iter_all_markets = AsyncMock(return_value=([], False))

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()

        assert rows_written == 0
        kalshi_client.get_orderbook.assert_not_called()
        assert not (tmp_path / PHASE0_DEPTH_SINK_FILENAME).exists()

    @pytest.mark.asyncio
    async def test_one_eligible_ticker_writes_one_row(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        ticker = _today_market_ticker("KXHIGHMIA")
        market = _make_market(
            ticker=ticker,
            yes_bid_dollars="0.42",
            yes_ask_dollars="0.45",
            yes_bid_size_fp="30",
            yes_ask_size_fp="25",
        )

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return ([market], False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()

        assert rows_written == 1
        # L2 capture pass: one get_orderbook call for the single eligible ticker.
        # (Empty envelope default ⇒ L2 fields remain None on the row.)
        assert kalshi_client.get_orderbook.await_count == 1

        sink = tmp_path / PHASE0_DEPTH_SINK_FILENAME
        assert sink.exists()
        line = sink.read_text().strip().splitlines()
        assert len(line) == 1
        row = json.loads(line[0])
        assert row["ticker"] == ticker
        assert row["cell"] == "MIA/high"
        assert row["yes_bid_size"] == 30
        assert row["yes_ask_size"] == 25
        assert row["sample_kind"] == PHASE0_SAMPLE_KIND

    @pytest.mark.asyncio
    async def test_iter_all_markets_is_one_call_per_series_not_per_market(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        """Read-budget regression guard for TOB pass.

        With 12 eligible same-day markets across both series, the
        monitor must hit ``iter_all_markets`` exactly ``len(series)``
        times (NOT 1 per market) for the TOB pass. The L2 pass DOES
        fire one ``get_orderbook`` per market (12 total) — that is the
        intended behavior post-2026-05-17.
        """
        mia = [
            _make_market(ticker=_today_market_ticker("KXHIGHMIA").replace("B85", f"B{i}"))
            for i in range(6)
        ]
        lax = [
            _make_market(ticker=_today_market_ticker("KXHIGHLAX").replace("B85", f"B{i}"))
            for i in range(6)
        ]

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return (mia, False)
            if series_ticker == "KXHIGHLAX":
                return (lax, False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()

        assert rows_written == 12
        # 1 iter_all_markets call per series, NOT 1 per market.
        assert kalshi_client.iter_all_markets.await_count == len(PHASE0_ELIGIBLE_SERIES)
        # L2 pass: one get_orderbook per eligible same-day market.
        assert kalshi_client.get_orderbook.await_count == 12

    @pytest.mark.asyncio
    async def test_iter_all_markets_truncation_logs_warning_and_counts(
        self, tmp_path: Path, kalshi_client: MagicMock, caplog: pytest.LogCaptureFixture
    ):
        """Fix #2: truncated pagination → WARNING + counter, tick still writes rows."""
        ticker = _today_market_ticker("KXHIGHMIA")
        market = _make_market(ticker=ticker)

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return ([market], True)  # truncated=True
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        with caplog.at_level(logging.WARNING, logger="python.data.phase0_depth_monitor"):
            rows_written = await monitor.run_once()

        # Tick still writes the rows it does have.
        assert rows_written == 1
        # Operator-visible counter incremented for the truncated series.
        assert monitor.markets_truncated_ticks == 1
        # WARNING fired with the expected operator phrase.
        assert any(
            "truncated market list for series KXHIGHMIA" in r.getMessage() for r in caplog.records
        ), "expected truncation WARNING was not logged"

    @pytest.mark.asyncio
    async def test_other_day_tickers_are_filtered_out(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        # Same-day eligibility check must drop tickers whose embedded date
        # doesn't match ``et_today``. With no eligible markets, the L2
        # pass never fires either.
        market = _make_market(ticker=_other_day_market_ticker("KXHIGHMIA"))
        kalshi_client.iter_all_markets = AsyncMock(return_value=([market], False))

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()
        assert rows_written == 0
        kalshi_client.get_orderbook.assert_not_called()

    @pytest.mark.asyncio
    async def test_iter_all_markets_error_does_not_block_other_series(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        # iter_all_markets failure for one series must not block the other.
        lax_ticker = _today_market_ticker("KXHIGHLAX")
        lax_market = _make_market(ticker=lax_ticker, yes_bid_size_fp="12", yes_ask_size_fp="7")

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                raise RuntimeError("simulated kalshi 500")
            if series_ticker == "KXHIGHLAX":
                return ([lax_market], False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()
        assert rows_written == 1
        sink_lines = (tmp_path / PHASE0_DEPTH_SINK_FILENAME).read_text().strip().splitlines()
        assert json.loads(sink_lines[0])["cell"] == "LAX/high"

    @pytest.mark.asyncio
    async def test_writer_error_is_swallowed(
        self, tmp_path: Path, kalshi_client: MagicMock, caplog: pytest.LogCaptureFixture
    ):
        """Fix #5: disk-full / permission-denied on write_batch must not propagate."""
        ticker = _today_market_ticker("KXHIGHMIA")
        market = _make_market(ticker=ticker)
        kalshi_client.iter_all_markets = AsyncMock(return_value=([market], False))

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)

        def _boom(_rows: Any) -> None:  # noqa: ANN401
            raise OSError("disk full")

        monitor._writer.write_batch = _boom  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger="python.data.phase0_depth_monitor"):
            rows_written = await monitor.run_once()

        assert rows_written == 0
        assert any("write_batch failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Phase0DepthMonitor.run — cancellation
# ---------------------------------------------------------------------------


class TestRunCancellation:
    @pytest.mark.asyncio
    async def test_cancel_mid_tick_exits_cleanly(self, tmp_path: Path, kalshi_client: MagicMock):
        """Fix #4: cancelling run() mid-tick must surface CancelledError.

        Guards against a future refactor widening the inner ``except`` to
        ``BaseException``, which would swallow ``CancelledError`` and
        deadlock orchestrator shutdown.
        """

        # Replace iter_all_markets with a never-finishing future so the
        # tick is wedged inside the outermost try-block.
        async def _hang(**_: Any) -> tuple[list[dict[str, Any]], bool]:
            await asyncio.Future()  # never resolves
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_hang)

        monitor = Phase0DepthMonitor(
            kalshi_client=kalshi_client,
            state_dir=tmp_path,
            interval_seconds=0.01,
        )

        task = asyncio.create_task(monitor.run())
        # Yield so the task enters its loop body.
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert task.cancelled()


# ---------------------------------------------------------------------------
# L2 depth capture — schema, level selection, guards, failure modes
# (2026-05-17 reinstatement)
# ---------------------------------------------------------------------------


def _l2_orderbook(
    *,
    yes_levels: list[list[str]] | None = None,
    no_levels: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Helper: build an ``orderbook_fp``-enveloped payload."""
    return {
        "orderbook_fp": {
            "yes_dollars": yes_levels if yes_levels is not None else [],
            "no_dollars": no_levels if no_levels is not None else [],
        }
    }


class TestL2SchemaShape:
    """Test #1 (THE critical C1 regression catcher) + envelope guards."""

    def test_l2_capture_uses_orderbook_fp_envelope(self):
        # Correct envelope → L2 fields populated.
        orderbook_with_envelope = {
            "orderbook_fp": {
                "yes_dollars": [["0.42", "100"], ["0.40", "50"]],
                "no_dollars": [["0.55", "200"], ["0.50", "75"]],
            }
        }
        market = _make_market(
            yes_bid_dollars="0.42",
            yes_ask_dollars="0.45",
        )
        row = build_depth_sample(
            market, "KXHIGHMIA", now_utc=_NOW, orderbook=orderbook_with_envelope
        )
        assert row is not None
        assert row["yes_bid_l2_size"] == 50
        assert row["yes_ask_l2_size"] == 75
        assert row["yes_bid_l2_price"] == pytest.approx(0.40)
        # NO L2 = 0.50 → derived YES L2 ask = 1 - 0.50 = 0.50.
        assert row["yes_ask_l2_price"] == pytest.approx(0.50)
        assert row["yes_levels_count"] == 2
        assert row["no_levels_count"] == 2
        assert row["l2_present"] is True

        # Regression: fixture missing the ``orderbook_fp`` envelope
        # (top-level ``yes_dollars``) must yield L2 = None — guards C1.
        orderbook_without_envelope = {
            "yes_dollars": [["0.42", "100"], ["0.40", "50"]],
            "no_dollars": [["0.55", "200"], ["0.50", "75"]],
        }
        row_no_envelope = build_depth_sample(
            market,
            "KXHIGHMIA",
            now_utc=_NOW,
            orderbook=orderbook_without_envelope,
        )
        assert row_no_envelope is not None
        assert row_no_envelope["yes_bid_l2_size"] is None
        assert row_no_envelope["yes_ask_l2_size"] is None
        assert row_no_envelope["yes_bid_l2_price"] is None
        assert row_no_envelope["yes_ask_l2_price"] is None
        assert row_no_envelope["l2_present"] is False
        assert row_no_envelope["yes_levels_count"] == 0
        assert row_no_envelope["no_levels_count"] == 0


class TestL2LevelSelection:
    """Test #2-#5: level selection invariants."""

    def test_l2_capture_sort_independent_of_input_order(self):
        # Same levels supplied in ascending OR descending price order
        # must yield identical L2 size/price.
        market = _make_market(yes_bid_dollars="0.42", yes_ask_dollars="0.45")
        ob_descending = _l2_orderbook(
            yes_levels=[["0.42", "100"], ["0.40", "50"]],
            no_levels=[["0.55", "200"], ["0.50", "75"]],
        )
        ob_ascending = _l2_orderbook(
            yes_levels=[["0.40", "50"], ["0.42", "100"]],
            no_levels=[["0.50", "75"], ["0.55", "200"]],
        )
        row_d = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=ob_descending)
        row_a = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=ob_ascending)
        assert row_d is not None and row_a is not None
        assert row_d["yes_bid_l2_size"] == row_a["yes_bid_l2_size"] == 50
        assert row_d["yes_ask_l2_size"] == row_a["yes_ask_l2_size"] == 75
        assert row_d["yes_bid_l2_price"] == row_a["yes_bid_l2_price"]
        assert row_d["yes_ask_l2_price"] == row_a["yes_ask_l2_price"]

    def test_l2_capture_handles_single_level_book(self):
        market = _make_market(yes_bid_dollars="0.42", yes_ask_dollars="0.45")
        ob = _l2_orderbook(
            yes_levels=[["0.42", "100"]],
            no_levels=[["0.55", "200"]],
        )
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=ob)
        assert row is not None
        assert row["yes_bid_l2_size"] is None
        assert row["yes_ask_l2_size"] is None
        assert row["yes_bid_l2_price"] is None
        assert row["yes_ask_l2_price"] is None
        assert row["yes_levels_count"] == 1
        assert row["no_levels_count"] == 1
        assert row["l2_present"] is False

    def test_l2_capture_handles_empty_orderbook(self):
        market = _make_market(yes_bid_dollars="0.42", yes_ask_dollars="0.45")
        ob = _l2_orderbook(yes_levels=[], no_levels=[])
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=ob)
        assert row is not None
        assert row["yes_bid_l2_size"] is None
        assert row["yes_ask_l2_size"] is None
        assert row["yes_bid_l2_price"] is None
        assert row["yes_ask_l2_price"] is None
        assert row["yes_levels_count"] == 0
        assert row["no_levels_count"] == 0
        assert row["l2_present"] is False

    def test_l2_internal_monotonicity_assertion_fires_on_unsorted_data(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test #5a: if the post-sort step is bypassed (e.g., a future
        refactor accidentally removes the sort), the internal monotonicity
        assertion must catch it.

        ``list.sort`` is an immutable C method on a builtin type and
        can't be monkey-patched directly. Instead we patch the
        module-level helper ``_parse_orderbook_level`` to return
        ``(price, qty)`` tuples whose price object is a wrapper that
        LIES during the sort key comparison (returns ``False`` from
        ``__lt__``, making Timsort's stable sort a no-op) but answers
        the assertion's ``<=`` check truthfully. The wire payload is
        otherwise unused. This deterministically constructs a post-sort
        state where ``parsed_yes[1][0] > parsed_yes[0][0]`` — the exact
        condition the new monotonicity guard catches.
        """
        from python.data import phase0_depth_monitor as mod

        class _UncomparablePrice:
            """Float wrapper: sort sees it as never-less-than (so
            in-place sort is a stable no-op), assertion sees the real
            value via ``__le__``."""

            def __init__(self, value: float) -> None:
                self.value = value

            def __lt__(self, other: object) -> bool:  # used by sort key
                return False

            def __le__(self, other: object) -> bool:  # used by assertion
                if isinstance(other, _UncomparablePrice):
                    return self.value <= other.value
                return self.value <= float(other)  # type: ignore[arg-type]

            def __eq__(self, other: object) -> bool:
                if isinstance(other, _UncomparablePrice):
                    return self.value == other.value
                return False

            def __hash__(self) -> int:
                return hash(self.value)

            def __repr__(self) -> str:
                return f"_UncomparablePrice({self.value})"

        # Only the YES side gets wrapped (the YES assertion fires
        # first; the NO side is left as normal floats in the correct
        # monotone order so that the NO ``_orderbook_price_to_probability``
        # inversion at line 294 doesn't blow up on the wrapper before the
        # YES assertion is reached).
        levels = iter(
            [
                (_UncomparablePrice(0.40), 100),  # parsed_yes[0]
                (_UncomparablePrice(0.42), 50),  # parsed_yes[1] (HIGHER → trips L2≤L1)
                (0.55, 20),  # parsed_no[0]  (well-ordered)
                (0.50, 10),  # parsed_no[1]
            ]
        )

        def _fake_parse(_level: object) -> tuple[Any, int] | None:
            return next(levels)

        monkeypatch.setattr(mod, "_parse_orderbook_level", _fake_parse)

        market = _make_market(yes_bid_dollars="0.45", yes_ask_dollars="0.50")
        bad_ob = _l2_orderbook(
            yes_levels=[["x", "1"], ["y", "2"]],
            no_levels=[["x", "1"], ["y", "2"]],
        )
        with pytest.raises(AssertionError, match="YES L2 monotonicity violation"):
            build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=bad_ob)

    def test_otm_trash_bin_pattern_does_not_raise(self):
        """Test #5b (LOAD-BEARING regression test): the exact production
        failure mode pinned.

        Pre-fix this tripped the cross-call assertion in 18-33% of probes:
        NBBO ``yes_bid=0.0`` from ``iter_all_markets`` vs. orderbook with
        YES bids at 0.01+. Post-fix this is normal data — a valid row is
        written with the L2 fields populated from the orderbook.

        If this test ever regresses (AssertionError raised, or row missing
        L2 fields), the cross-call assertion class has come back. DO NOT
        weaken this test without understanding why it was fixed.
        """
        market = _make_market(
            yes_bid_dollars="0.00",
            yes_ask_dollars="0.01",
            yes_bid_size_fp="0",
            yes_ask_size_fp="200000",
        )
        # Production OTM trash-bin pattern: orderbook has two YES levels
        # at 0.01-0.02 while iter_all_markets reports yes_bid=0.0. Two
        # NO levels mirror the deep-OTM other side.
        ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.0100", "20.00"], ["0.0200", "5.00"]],
                "no_dollars": [["0.9700", "1.00"], ["0.9500", "3.00"]],
            }
        }
        # Must NOT raise AssertionError.
        row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=ob)
        assert row is not None
        # After descending sort: YES = [(0.02, 5), (0.01, 20)] → L2 = (0.01, 20).
        assert row["yes_bid_l2_price"] == pytest.approx(0.01)
        assert row["yes_bid_l2_size"] == 20
        # NO descending: [(0.97, 1), (0.95, 3)] → L2 NO bid = 0.95 →
        # derived YES L2 ask = 1 - 0.95 = 0.05, size = 3.
        assert row["yes_ask_l2_price"] == pytest.approx(0.05)
        assert row["yes_ask_l2_size"] == 3
        assert row["yes_levels_count"] == 2
        assert row["no_levels_count"] == 2
        assert row["l2_present"] is True

    def test_out_of_range_price_logs_warning(self, caplog: pytest.LogCaptureFixture):
        """Test #5c: a Kalshi cents↔dollars encoding flip on the
        orderbook side would yield prices > 1.0. ``_parse_dollar_price``
        rejects them to None (so the level is dropped) AND logs WARNING
        so the schema flip is observable. This re-establishes the
        observability the removed cross-call assertion was claiming to
        provide (the only schema-drift class that survives the local
        sort + envelope checks).
        """
        market = _make_market(yes_bid_dollars="0.42", yes_ask_dollars="0.45")
        # Cents-encoded levels (the failure class the previous assertion
        # was trying to catch). ``_parse_dollar_price`` rejects each level
        # with a WARNING; the resulting parsed list is empty.
        ob = _l2_orderbook(
            yes_levels=[["88.00", "50"], ["89.00", "100"]],
            no_levels=[["55.00", "10"], ["50.00", "20"]],
        )
        with caplog.at_level(logging.WARNING, logger="python.data.phase0_depth_monitor"):
            row = build_depth_sample(market, "KXHIGHMIA", now_utc=_NOW, orderbook=ob)
        assert row is not None
        # All levels rejected as out-of-range → no L2 data populated.
        assert row["yes_bid_l2_size"] is None
        assert row["yes_ask_l2_size"] is None
        assert row["yes_levels_count"] == 0
        assert row["no_levels_count"] == 0
        # WARNING fired for at least one of the rejected prices.
        assert any(
            "Dollar price" in r.getMessage() and "outside [0, 1]" in r.getMessage()
            for r in caplog.records
        ), "expected out-of-range Dollar price WARNING was not logged"


class TestL2RateLimitAndWallClock:
    """Test #6-#8: pacing, wall-clock guard, market-count guard."""

    @pytest.mark.asyncio
    async def test_l2_capture_paces_calls_with_sleep(
        self, tmp_path: Path, kalshi_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """At least 11 inter-call sleeps for 12 markets (sleep is between
        consecutive get_orderbook calls only)."""
        from python.data import phase0_depth_monitor as mod

        mia = [
            _make_market(ticker=_today_market_ticker("KXHIGHMIA").replace("B85", f"B{i}"))
            for i in range(6)
        ]
        lax = [
            _make_market(ticker=_today_market_ticker("KXHIGHLAX").replace("B85", f"B{i}"))
            for i in range(6)
        ]

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return (mia, False)
            if series_ticker == "KXHIGHLAX":
                return (lax, False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)
        kalshi_client.get_orderbook = AsyncMock(return_value={"orderbook_fp": {}})

        # Record asyncio.sleep calls with the L2 pacing argument.
        sleep_call_args: list[float] = []
        real_sleep = asyncio.sleep

        async def _recording_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
            sleep_call_args.append(delay)
            # Yield so the loop can progress.
            return await real_sleep(0)

        monkeypatch.setattr(mod.asyncio, "sleep", _recording_sleep)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        await monitor.run_once()

        # 12 markets → 11 inter-call sleeps at PHASE0_L2_PACING_SLEEP_SECONDS.
        pacing_calls = [d for d in sleep_call_args if d == mod.PHASE0_L2_PACING_SLEEP_SECONDS]
        assert len(pacing_calls) >= 11

    @pytest.mark.asyncio
    async def test_l2_capture_wall_clock_aborts_after_budget(
        self, tmp_path: Path, kalshi_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """Patch ``time.monotonic`` to advance past the 60s budget mid-pass.

        Deterministic (no asyncio.sleep(10) hang) — matches plan
        constraint #7.
        """
        from python.data import phase0_depth_monitor as mod

        mia = [
            _make_market(ticker=_today_market_ticker("KXHIGHMIA").replace("B85", f"B{i}"))
            for i in range(6)
        ]
        lax = [
            _make_market(ticker=_today_market_ticker("KXHIGHLAX").replace("B85", f"B{i}"))
            for i in range(6)
        ]

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return (mia, False)
            if series_ticker == "KXHIGHLAX":
                return (lax, False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)
        kalshi_client.get_orderbook = AsyncMock(return_value={"orderbook_fp": {}})

        # ``monotonic`` advances by 30s per call. First call returns
        # pass_start = 0; subsequent calls inside the L2 loop guard fire
        # at 30, 60, 90, … so the guard trips on iteration 2 (i=1) when
        # ``elapsed = 60`` satisfies ``>= 60``.
        ticker = iter(range(0, 10_000, 30))

        def _fake_monotonic() -> float:
            return float(next(ticker))

        monkeypatch.setattr(mod.time, "monotonic", _fake_monotonic)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()

        assert rows_written == 12
        assert monitor.l2_wall_clock_aborts_ticks == 1
        # Some get_orderbook calls happened before the abort; the rest
        # didn't. Each row that didn't get an orderbook payload has L2 = None.
        sink_lines = (tmp_path / PHASE0_DEPTH_SINK_FILENAME).read_text().strip().splitlines()
        rows = [json.loads(line) for line in sink_lines]
        # At least one row has L2 = None (the wall-clock-aborted tail).
        assert any(r["yes_bid_l2_size"] is None for r in rows)

    @pytest.mark.asyncio
    async def test_l2_capture_market_count_guard_skips_when_over_threshold(
        self, tmp_path: Path, kalshi_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        from python.data import phase0_depth_monitor as mod

        monkeypatch.setattr(mod, "PHASE0_L2_MAX_MARKETS_PER_TICK", 5)

        mia = [
            _make_market(ticker=_today_market_ticker("KXHIGHMIA").replace("B85", f"B{i}"))
            for i in range(5)
        ]
        lax = [
            _make_market(ticker=_today_market_ticker("KXHIGHLAX").replace("B85", f"B{i}"))
            for i in range(5)
        ]

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return (mia, False)
            if series_ticker == "KXHIGHLAX":
                return (lax, False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()

        assert rows_written == 10
        # All rows L2 = None.
        sink_lines = (tmp_path / PHASE0_DEPTH_SINK_FILENAME).read_text().strip().splitlines()
        rows = [json.loads(line) for line in sink_lines]
        assert all(r["yes_bid_l2_size"] is None for r in rows)
        assert all(r["yes_ask_l2_size"] is None for r in rows)
        assert all(r["l2_present"] is False for r in rows)
        # No get_orderbook calls made (guard fires before fetch).
        kalshi_client.get_orderbook.assert_not_called()
        assert monitor.l2_skipped_ticks == 1


class TestL2FailureModes:
    """Test #9-#10: fail-soft + cancellation."""

    @pytest.mark.asyncio
    async def test_l2_capture_fail_soft_on_orderbook_error(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        ticker = _today_market_ticker("KXHIGHMIA")
        market = _make_market(ticker=ticker)

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return ([market], False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)
        kalshi_client.get_orderbook = AsyncMock(
            side_effect=RuntimeError("simulated /orderbook 500")
        )

        monitor = Phase0DepthMonitor(kalshi_client=kalshi_client, state_dir=tmp_path)
        rows_written = await monitor.run_once()

        assert rows_written == 1
        assert monitor.l2_fetch_errors_ticks == 1
        sink_lines = (tmp_path / PHASE0_DEPTH_SINK_FILENAME).read_text().strip().splitlines()
        row = json.loads(sink_lines[0])
        assert row["yes_bid_l2_size"] is None
        assert row["yes_ask_l2_size"] is None
        assert row["l2_present"] is False

    @pytest.mark.asyncio
    async def test_cancel_during_l2_pass_exits_cleanly(
        self, tmp_path: Path, kalshi_client: MagicMock
    ):
        """CancelledError raised inside the L2 pass must propagate."""
        ticker = _today_market_ticker("KXHIGHMIA")
        market = _make_market(ticker=ticker)

        async def _iter(*, status: str, series_ticker: str, **_):
            if series_ticker == "KXHIGHMIA":
                return ([market], False)
            return ([], False)

        kalshi_client.iter_all_markets = AsyncMock(side_effect=_iter)

        async def _hang(*_: Any, **__: Any) -> dict[str, Any]:
            await asyncio.Future()  # never resolves
            return {}

        kalshi_client.get_orderbook = AsyncMock(side_effect=_hang)

        monitor = Phase0DepthMonitor(
            kalshi_client=kalshi_client,
            state_dir=tmp_path,
            interval_seconds=0.01,
        )

        task = asyncio.create_task(monitor.run())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        assert task.cancelled()
        # No rows written (write happens after L2 pass completes).
        assert not (tmp_path / PHASE0_DEPTH_SINK_FILENAME).exists()


# ---------------------------------------------------------------------------
# Sink-path + eligible-series surface
# ---------------------------------------------------------------------------


def test_sink_path_property_under_state_dir(tmp_path: Path):
    client = MagicMock()
    monitor = Phase0DepthMonitor(kalshi_client=client, state_dir=tmp_path)
    assert monitor.sink_path == tmp_path / PHASE0_DEPTH_SINK_FILENAME


def test_eligible_series_locks_to_mia_lax_only():
    # M0's "auto-enabled when MIA/high or LAX/high in open-markets" rule
    # cannot drift without a fresh §8.2 audit (locked by spec §6.5).
    assert set(PHASE0_ELIGIBLE_SERIES.keys()) == {"KXHIGHMIA", "KXHIGHLAX"}
    assert PHASE0_ELIGIBLE_SERIES["KXHIGHMIA"] == ("MIA/high", "MIA")
    assert PHASE0_ELIGIBLE_SERIES["KXHIGHLAX"] == ("LAX/high", "LAX")
