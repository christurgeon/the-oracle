"""Safety tests for the KXRAIN shadow observer: zero orders, never raises,
and resource non-degradation (disk limit, request budget, backoff gate,
offloaded ledger writes) — not correctness tests."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data import kxrain_observer as observer_module  # noqa: E402
from python.data.kxrain_ledger import LEDGER_FILENAME, KxRainLedger  # noqa: E402
from python.data.kxrain_markets import KXRAIN_CITY_STATION  # noqa: E402
from python.data.kxrain_observer import (  # noqa: E402
    CENSUS_ABSENT_TICKS_REQUIRED,
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    KXRAIN_BOOK_FILENAME,
    KXRAIN_DECISIONS_FILENAME,
    KXRAIN_TICKS_FILENAME,
    KXRAIN_TRADES_FILENAME,
    WRITE_FLUSH_EVERY_ROWS,
    KxRainObserver,
    KxRainObserverConfig,
)
from python.data.kxrain_rows import TICK_OUTCOMES  # noqa: E402
from python.notifications.base import AlertLevel  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "kxrain"

# Every mapped city is inside [10:00, 18:00) LST at this instant, so one tick
# exercises the in-window path for the whole universe.
CAPTURE_TS = datetime(2026, 9, 18, 18, 44, 11, tzinfo=UTC)

# The 6 of 27 same-day markets whose yes_bid lands in [0.10, 0.90) on the
# captured page, in yes_bid-descending capture order.
FIXTURE_BAND_TICKERS = [
    "KXRAIN-26SEP18-MIA",
    "KXRAIN-26SEP18-MKE",
    "KXRAIN-26SEP18-CHI",
    "KXRAIN-26SEP18-DEN",
    "KXRAIN-26SEP18-ATL",
    "KXRAIN-26SEP18-LEX",
]




def _load(name: str) -> Any:
    path = FIXTURES / name
    if not path.exists():  # pragma: no cover - fixtures are committed
        pytest.skip(f"{name} not captured; run scripts/capture_kxrain_fixture.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def live_markets() -> list[dict[str, Any]]:
    return _load("live_markets.json")["markets"]


@pytest.fixture(scope="module")
def live_trades() -> dict[str, Any]:
    return _load("live_trades.json")


@pytest.fixture(scope="module")
def live_orderbook() -> dict[str, Any]:
    return _load("live_orderbook.json")


def _template_market(markets: list[dict[str, Any]]) -> dict[str, Any]:
    return copy.deepcopy(markets[0])


def make_market(
    template: dict[str, Any],
    ticker: str,
    yes_bid: float,
    *,
    yes_ask: float | None = None,
) -> dict[str, Any]:
    row = copy.deepcopy(template)
    row["ticker"] = ticker
    row["event_ticker"] = "-".join(ticker.split("-")[:2])
    row["yes_bid_dollars"] = f"{yes_bid:.4f}"
    ask = yes_ask if yes_ask is not None else min(1.0, yes_bid + 0.02)
    row["yes_ask_dollars"] = f"{ask:.4f}"
    row["no_ask_dollars"] = f"{1.0 - yes_bid:.4f}"
    row["rules_primary"] = ""
    return row


def trades_payload(source: dict[str, Any], *, side: str | None, cursor: str = "") -> dict[str, Any]:
    rows = source["trades"]
    if side is not None:
        rows = [r for r in rows if r.get("taker_side") == side]
    return {"trades": copy.deepcopy(rows), "cursor": cursor}




class FakeKalshiClient:
    """Read-only Kalshi stand-in. Records every call it receives."""

    def __init__(
        self,
        markets: list[dict[str, Any]],
        *,
        trades: dict[str, Any] | None = None,
        orderbook: dict[str, Any] | None = None,
        backoff_schedule: list[float] | None = None,
        markets_error: bool = False,
        trades_error: bool = False,
        book_error: bool = False,
        truncated: bool = False,
        extra_page: dict[str, Any] | None = None,
    ) -> None:
        self._markets = markets
        self._trades = trades or {"trades": [], "cursor": ""}
        self._orderbook = orderbook or {"orderbook_fp": {"yes_dollars": [], "no_dollars": []}}
        self._backoff_schedule = list(backoff_schedule or [])
        self._backoff_default = self._backoff_schedule[-1] if self._backoff_schedule else 0.0
        self.markets_error = markets_error
        self.trades_error = trades_error
        self.book_error = book_error
        # A cursor the server never clears — the non-terminating-pagination
        # shape ``max_pages`` exists to bound.
        self.endless_cursor = truncated
        # A -> B -> A cursor: ``paginate_detailed`` discards the repeating page
        # and reports ``cycle_aborted``, which the observer must not lose.
        self.cycling_cursor = False
        self._extra_page = extra_page
        self.calls: list[tuple[str, Any]] = []
        self.trade_call_kwargs: list[dict[str, Any]] = []

    async def get_markets(
        self,
        *,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.calls.append(("get_markets", series_ticker))
        if self.markets_error:
            raise RuntimeError("get_markets exploded")
        if self.cycling_cursor:
            return {"markets": [copy.deepcopy(m) for m in self._markets], "cursor": "LOOP"}
        offset = int(cursor) if cursor else 0
        page = self._markets[offset : offset + limit]
        next_offset = offset + len(page)
        if self.endless_cursor:
            next_cursor = str(next_offset + 1)
        else:
            next_cursor = str(next_offset) if next_offset < len(self._markets) else ""
        return {"markets": [copy.deepcopy(m) for m in page], "cursor": next_cursor}

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        self.calls.append(("get_orderbook", ticker))
        if self.book_error:
            raise RuntimeError("get_orderbook exploded")
        return copy.deepcopy(self._orderbook)

    async def get_market_trades(
        self,
        *,
        ticker: str,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("get_market_trades", ticker))
        self.trade_call_kwargs.append(
            {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts, "limit": limit, "cursor": cursor}
        )
        if self.trades_error:
            raise RuntimeError("get_market_trades exploded")
        if cursor and self._extra_page is not None:
            return copy.deepcopy(self._extra_page)
        return copy.deepcopy(self._trades)

    def backoff_remaining_seconds(self) -> float:
        if self._backoff_schedule:
            return self._backoff_schedule.pop(0)
        return self._backoff_default

    # -- accounting helpers used by the tests -------------------------------

    def count(self, method: str) -> int:
        return sum(1 for name, _ in self.calls if name == method)

    def tickers(self, method: str) -> list[str]:
        return [arg for name, arg in self.calls if name == method]


class HostileKalshiClient(FakeKalshiClient):
    """Raises on every method besides the three read-only GETs the observer may use."""

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("KxRainObserver called place_order")

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("KxRainObserver called cancel_order")

    def create_order(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("KxRainObserver called create_order")

    def batch_create_orders(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("KxRainObserver called batch_create_orders")

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"KxRainObserver touched a non-GET client attribute: {name!r}")


class ExplodingClient:
    """Raises on absolutely everything, including the backoff accessor."""

    async def get_markets(self, **kwargs: Any) -> Any:
        raise RuntimeError("no")

    async def get_orderbook(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no")

    async def get_market_trades(self, **kwargs: Any) -> Any:
        raise RuntimeError("no")

    def backoff_remaining_seconds(self) -> float:
        raise RuntimeError("no")


class NoBackoffAccessorClient(FakeKalshiClient):
    """A client predating ``KalshiClient.backoff_remaining_seconds``."""

    backoff_remaining_seconds = None  # type: ignore[assignment]




class RecordingNotifier:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, AlertLevel]] = []

    async def send_alert(
        self,
        message: str,
        *,
        level: AlertLevel = AlertLevel.INFO,
        failure_bucket: str | None = None,
    ) -> None:
        self.alerts.append((message, level))


class RaisingNotifier(RecordingNotifier):
    async def send_alert(
        self,
        message: str,
        *,
        level: AlertLevel = AlertLevel.INFO,
        failure_bucket: str | None = None,
    ) -> None:
        raise RuntimeError("telegram is down")




class MutableClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: Any) -> None:
        self.now = self.now + timedelta(**kwargs)


class MutableMonotonic:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make_observer(
    client: Any,
    tmp_path: Path,
    *,
    clock: MutableClock,
    notifier: Any = None,
    monotonic: Any = None,
    **config_overrides: Any,
) -> tuple[KxRainObserver, KxRainObserverConfig]:
    overrides: dict[str, Any] = {"request_pacing_seconds": 0.0}
    overrides.update(config_overrides)
    config = KxRainObserverConfig(log_dir=tmp_path, **overrides)
    obs = KxRainObserver(
        client,
        config,
        notifier_getter=(lambda: notifier),
        clock=clock,
        monotonic=monotonic,
        seed=1234,
    )
    return obs, config


def read_stream(tmp_path: Path, filename: str) -> list[dict[str, Any]]:
    """Read a stream including rotated siblings, in file order."""
    stem = Path(filename).stem
    rows: list[dict[str, Any]] = []
    for path in sorted(tmp_path.glob(f"{stem}*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def decisions(tmp_path: Path) -> list[dict[str, Any]]:
    return read_stream(tmp_path, KXRAIN_DECISIONS_FILENAME)


def books(tmp_path: Path) -> list[dict[str, Any]]:
    return read_stream(tmp_path, KXRAIN_BOOK_FILENAME)


def trades(tmp_path: Path) -> list[dict[str, Any]]:
    return read_stream(tmp_path, KXRAIN_TRADES_FILENAME)


def ticks(tmp_path: Path) -> list[dict[str, Any]]:
    return read_stream(tmp_path, KXRAIN_TICKS_FILENAME)




KXRAIN_MODULES = sorted(Path(observer_module.__file__).parent.glob("kxrain_*.py"))


def test_the_module_glob_covers_the_orchestrator_launched_tasks() -> None:
    names = {p.name for p in KXRAIN_MODULES}
    assert {"kxrain_observer.py", "kxrain_settlements.py"} <= names


FORBIDDEN_IMPORT_PREFIXES = ("python.execution", "python.risk", "python.strategies")


@pytest.mark.parametrize("path", KXRAIN_MODULES, ids=lambda p: p.name)
def test_no_forbidden_package_imports(path: Path) -> None:
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES)
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                offenders.append(module)
            if any(alias.name == "TradeSignal" for alias in node.names):
                offenders.append(f"{module}.TradeSignal")
    assert offenders == []


@pytest.mark.parametrize("path", KXRAIN_MODULES, ids=lambda p: p.name)
def test_no_place_order_identifier_anywhere(path: Path) -> None:
    assert "place_order" not in path.read_text()


def _is_close_time_passthrough(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    # True only for ``{"close_time": <x>.get("close_time")...}`` in a dict literal.
    parent = parents.get(node)
    if isinstance(parent, ast.Dict):
        return node in parent.keys
    call = parent
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "get"
        and call.args
        and call.args[0] is node
    ):
        return False
    value: ast.AST = call
    while isinstance(parents.get(value), (ast.IfExp, ast.BoolOp)):
        value = parents[value]
    owner = parents.get(value)
    if not isinstance(owner, ast.Dict) or value not in owner.values:
        return False
    key = owner.keys[owner.values.index(value)]
    return isinstance(key, ast.Constant) and key.value == "close_time"


@pytest.mark.parametrize("path", KXRAIN_MODULES, ids=lambda p: p.name)
def test_close_time_is_only_copied_into_rows_never_read(path: Path) -> None:
    # close_time used to track daylight midnight; deriving a day from it
    # would silently shift every market-day by an hour.
    tree = ast.parse(path.read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Constant) and node.value == "close_time"
            and not _is_close_time_passthrough(node, parents))
        or (isinstance(node, ast.Attribute) and node.attr == "close_time")
        or (isinstance(node, ast.Name) and node.id == "close_time")
    ]
    assert offenders == []


def test_module_imports_no_trade_signal() -> None:
    assert not hasattr(observer_module, "TradeSignal")




async def test_hostile_client_full_simulated_day(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = HostileKalshiClient(
        live_markets, trades=live_trades, orderbook=live_orderbook
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)

    for _ in range(12):
        await obs.run_once()
        clock.advance(minutes=10)

    assert client.count("get_orderbook") > 0
    assert len(decisions(tmp_path)) == 54 * 12




@pytest.fixture
async def fixture_tick(
    tmp_path: Path,
    live_markets: list[dict[str, Any]],
    live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> Any:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, config = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    return client, obs, config


async def test_fixture_tick_row_counts(tmp_path: Path, fixture_tick: Any) -> None:
    client, _, _ = fixture_tick
    assert len(decisions(tmp_path)) == 54
    assert len(books(tmp_path)) == len(FIXTURE_BAND_TICKERS)
    assert len(trades(tmp_path)) == len(FIXTURE_BAND_TICKERS)
    assert len(ticks(tmp_path)) == 1
    assert client.count("get_markets") == 1


async def test_fixture_tick_join_keys_line_up(tmp_path: Path, fixture_tick: Any) -> None:
    decision_ids = {row["decision_id"] for row in decisions(tmp_path)}
    scan_ids = {row["scan_id"] for row in decisions(tmp_path)}
    assert len(scan_ids) == 1
    tick_scan_id = ticks(tmp_path)[0]["scan_id"]
    assert scan_ids == {tick_scan_id}
    for row in books(tmp_path) + trades(tmp_path):
        assert row["decision_id"] in decision_ids
        assert row["decision_id"] == f"{tick_scan_id}:{row['ticker']}"


async def test_fixture_tick_field_types(tmp_path: Path, fixture_tick: Any) -> None:
    for row in decisions(tmp_path):
        assert row["schema_version"] == 1
        assert row["frame_version"] == 2
        assert isinstance(row["in_window"], bool)
        assert isinstance(row["pass_index"], int)
        assert isinstance(row["follow_capture"], bool)
        # Emitted as null everywhere, never omitted.
        assert "last_seen_yes_bid" in row
        assert "last_seen_ts" in row
        assert row["book_capture"] in {
            "ok",
            "error",
            "skipped_band",
            "skipped_not_current_day",
            "skipped_market_absent",
            "budget_abort",
            "wall_clock_abort",
            "backoff_abort",
        }
    for row in trades(tmp_path):
        # Sizes are FRACTIONAL floats end-to-end.
        assert isinstance(row["no_volume_30m"], float)
        assert isinstance(row["n_trades_no_side"], int)
        assert row["n_pages_fetched"] == 1
        assert isinstance(row["window_end_ts"], int)
    for row in books(tmp_path):
        assert isinstance(row["no_ask_depth_within_1c_dollars"], float)
        assert isinstance(row["n_yes_levels"], int)


async def test_fixture_tick_eligibility_truth_table(tmp_path: Path, fixture_tick: Any) -> None:
    # Hand-computed: all 27 cities are in window; the 27 SEP19 markets are
    # next-day, and of the 27 SEP18 markets exactly 6 sit in the price band.
    rows = {row["ticker"]: row for row in decisions(tmp_path)}
    assert len(rows) == 54

    next_day = [r for r in rows.values() if r["settlement_date"] == "2026-09-19"]
    assert len(next_day) == 27
    for row in next_day:
        assert row["in_window"] is False
        assert row["skip_reason"] == "out_of_window"
        assert row["eligible"] is False
        assert row["book_capture"] == "skipped_not_current_day"
        assert row["pass_index"] == -1

    for ticker in FIXTURE_BAND_TICKERS:
        row = rows[ticker]
        assert row["in_window"] is True
        assert row["in_price_band"] is True
        assert row["eligible"] is True
        assert row["skip_reason"] is None
        assert row["virtual_entry"] is True
        assert row["assumed_entry_price"] == pytest.approx(1.0 - row["yes_bid"], abs=1e-9)
        assert row["book_capture"] == "ok"
        assert row["trades_capture"] == "ok"

    same_day_out_of_band = [
        r
        for r in rows.values()
        if r["settlement_date"] == "2026-09-18" and r["ticker"] not in FIXTURE_BAND_TICKERS
    ]
    assert len(same_day_out_of_band) == 21
    for row in same_day_out_of_band:
        assert row["skip_reason"] == "price_band"
        assert row["eligible"] is False
        assert row["book_capture"] == "skipped_band"
        assert row["virtual_entry"] is False


async def test_depth_capture_mode_all_captures_every_in_window_market(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(
        client, tmp_path, clock=clock, depth_capture_mode="all", max_requests_per_tick=200
    )
    await obs.run_once()

    same_day = [r for r in decisions(tmp_path) if r["settlement_date"] == "2026-09-18"]
    assert len(same_day) == 27
    assert all(r["book_capture"] == "ok" for r in same_day)
    assert len(books(tmp_path)) == 27
    # Next-day markets still cost no HTTP.
    assert len(decisions(tmp_path)) == 54


async def test_capture_order_is_yes_bid_descending(tmp_path: Path, fixture_tick: Any) -> None:
    client, _, _ = fixture_tick
    assert client.tickers("get_market_trades") == FIXTURE_BAND_TICKERS
    rows = {row["ticker"]: row for row in decisions(tmp_path)}
    for index, ticker in enumerate(FIXTURE_BAND_TICKERS):
        assert rows[ticker]["pass_index"] == index


async def test_trades_first_then_book_per_market(tmp_path: Path, fixture_tick: Any) -> None:
    client, _, _ = fixture_tick
    pairs = [c for c in client.calls if c[0] in {"get_market_trades", "get_orderbook"}]
    assert len(pairs) == 2 * len(FIXTURE_BAND_TICKERS)
    for i in range(0, len(pairs), 2):
        assert pairs[i][0] == "get_market_trades"
        assert pairs[i + 1][0] == "get_orderbook"
        assert pairs[i][1] == pairs[i + 1][1]


async def test_tick_row_carries_real_counters(tmp_path: Path, fixture_tick: Any) -> None:
    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "ok"
    assert row["n_markets_returned"] == 54
    assert row["n_parsed"] == 54
    assert row["n_unparseable"] == 0
    assert row["n_cities_in_window"] == len(KXRAIN_CITY_STATION)
    assert row["n_decision_rows"] == 54
    # 1 page of /markets (charged as used, not as a flat guess) + 2 per market.
    assert row["requests_used"] == 1 + 2 * len(FIXTURE_BAND_TICKERS)
    assert row["markets_pages_consumed"] == 1
    assert row["markets_cycle_aborted"] is False
    assert row["census_suppressed"] is False
    assert isinstance(row["tick_seed"], int)
    assert row["observer_version"]
    assert row["process_start_ts"]


async def test_tick_row_records_the_configured_cadence(
    tmp_path: Path, fixture_tick: Any
) -> None:
    _, _, config = fixture_tick
    assert ticks(tmp_path)[0]["tick_interval_seconds"] == config.tick_interval_seconds


async def test_trades_window_uses_actual_call_time(tmp_path: Path, fixture_tick: Any) -> None:
    # ``ts`` is the frozen tick instant; min_ts/max_ts are the call time.
    client, _, config = fixture_tick
    for kwargs in client.trade_call_kwargs:
        assert kwargs["max_ts"] - kwargs["min_ts"] == config.no_trade_lookback_minutes * 60
    for row in trades(tmp_path):
        assert row["window_end_ts"] - row["window_start_ts"] == (
            config.no_trade_lookback_minutes * 60
        )
        assert row["ts"] == CAPTURE_TS.isoformat()




def simulated_universe(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Two cities x two event days, all four priced inside the band."""
    return [
        make_market(template, "KXRAIN-26SEP18-NYC", 0.40),
        make_market(template, "KXRAIN-26SEP18-LAX", 0.30),
        make_market(template, "KXRAIN-26SEP19-NYC", 0.50),
        make_market(template, "KXRAIN-26SEP19-LAX", 0.60),
    ]


async def test_simulated_day_one_virtual_entry_per_market_day(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    client = FakeKalshiClient(
        simulated_universe(_template_market(live_markets)),
        trades=live_trades,
        orderbook=live_orderbook,
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)

    n_ticks = 24
    for _ in range(n_ticks):
        await obs.run_once()
        clock.advance(minutes=10)

    rows = decisions(tmp_path)
    # A decision row for EVERY market on EVERY tick — the single thing
    # intraday_emos_high got wrong.
    assert len(rows) == 4 * n_ticks

    entries = [r for r in rows if r["virtual_entry"]]
    keys = [(r["station"], r["settlement_date"]) for r in entries]
    assert len(keys) == len(set(keys))
    assert set(keys) == {("NYC", "2026-09-18"), ("LAX", "2026-09-18")}

    ledger = KxRainLedger.load(tmp_path / LEDGER_FILENAME)
    assert len(ledger) == 2


async def test_already_entered_markets_still_produce_trades_rows(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # Skipping entered markets would delete the primary gate's only input.
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    client = FakeKalshiClient(
        [make_market(_template_market(live_markets), "KXRAIN-26SEP18-NYC", 0.40)],
        trades=live_trades,
        orderbook=live_orderbook,
    )
    obs, _ = make_observer(client, tmp_path, clock=clock, post_entry_follow_minutes=0)

    for _ in range(6):
        await obs.run_once()
        clock.advance(minutes=10)

    rows = decisions(tmp_path)
    assert rows[0]["virtual_entry"] is True
    after = rows[1:]
    assert all(r["skip_reason"] == "already_entered" for r in after)
    assert all(r["trades_capture"] == "ok" for r in after)
    # One trades row and one book row for every check, entered or not.
    assert len(trades(tmp_path)) == 6
    assert len(books(tmp_path)) == 6


async def test_follow_capture_continues_past_band_and_window_exit(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    in_band = make_market(template, "KXRAIN-26SEP18-NYC", 0.40)
    out_of_band = make_market(template, "KXRAIN-26SEP18-NYC", 0.95)

    clock = MutableClock(datetime(2026, 9, 18, 22, 40, tzinfo=UTC))  # 17:40 LST
    client = FakeKalshiClient([in_band], trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock, post_entry_follow_minutes=45)

    await obs.run_once()
    assert decisions(tmp_path)[-1]["virtual_entry"] is True

    # Price leaves the band, then the clock leaves the window; follow capture
    # must stay pinned on regardless.
    client._markets = [out_of_band]
    for _ in range(4):
        clock.advance(minutes=10)
        await obs.run_once()

    rows = decisions(tmp_path)[1:]
    assert len(rows) == 4
    assert [r["in_price_band"] for r in rows] == [False] * 4
    assert [r["in_window"] for r in rows] == [True, False, False, False]
    assert all(r["follow_capture"] is True for r in rows)
    assert all(r["book_capture"] == "ok" for r in rows)
    assert len(books(tmp_path)) == 5


async def test_follow_capture_expires(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)],
        trades=live_trades,
        orderbook=live_orderbook,
    )
    obs, _ = make_observer(client, tmp_path, clock=clock, post_entry_follow_minutes=20)
    await obs.run_once()

    client._markets = [make_market(template, "KXRAIN-26SEP18-NYC", 0.95)]
    clock.advance(minutes=30)
    await obs.run_once()

    last = decisions(tmp_path)[-1]
    assert last["follow_capture"] is False
    assert last["book_capture"] == "skipped_band"




async def test_restart_rehydrates_ledger_no_duplicate_entry(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A LAX market-day entered at 23:00 UTC is still live at 01:00 UTC the next day.
    template = _template_market(live_markets)
    market = make_market(template, "KXRAIN-26SEP18-LAX", 0.40)

    clock = MutableClock(datetime(2026, 9, 18, 23, 0, tzinfo=UTC))  # 16:00 LST
    client = FakeKalshiClient([market], trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert decisions(tmp_path)[-1]["virtual_entry"] is True

    # New process, new observer, past UTC midnight — same LST market-day.
    clock2 = MutableClock(datetime(2026, 9, 19, 1, 0, tzinfo=UTC))  # 17:00 LST, SEP18
    client2 = FakeKalshiClient([market], trades=live_trades, orderbook=live_orderbook)
    obs2, _ = make_observer(client2, tmp_path, clock=clock2)
    await obs2.run_once()

    rows = decisions(tmp_path)
    assert [r["virtual_entry"] for r in rows] == [True, False]
    assert rows[-1]["already_entered_market_day"] is True
    assert rows[-1]["skip_reason"] == "already_entered"


async def test_corrupt_ledger_starts_empty_and_never_raises(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    (tmp_path / LEDGER_FILENAME).write_text("{not json at all")
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    client = FakeKalshiClient(
        [make_market(_template_market(live_markets), "KXRAIN-26SEP18-NYC", 0.40)],
        trades=live_trades,
        orderbook=live_orderbook,
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert decisions(tmp_path)[-1]["virtual_entry"] is True


async def test_record_entry_runs_off_the_event_loop(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = KxRainLedger.record_entry

    def spy(self: KxRainLedger, *args: Any, **kwargs: Any) -> bool:
        observed.append(threading.current_thread().name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(KxRainLedger, "record_entry", spy)

    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    client = FakeKalshiClient(
        [make_market(_template_market(live_markets), "KXRAIN-26SEP18-NYC", 0.40)],
        trades=live_trades,
        orderbook=live_orderbook,
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    main_thread = threading.main_thread().name
    assert observed and all(name != main_thread for name in observed)




async def test_budget_abort_never_splits_a_pair(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock, max_requests_per_tick=6)
    await obs.run_once()

    rows = {r["ticker"]: r for r in decisions(tmp_path)}
    captured = [t for t in FIXTURE_BAND_TICKERS if rows[t]["book_capture"] == "ok"]
    aborted = [t for t in FIXTURE_BAND_TICKERS if rows[t]["book_capture"] == "budget_abort"]
    assert len(captured) == 2
    assert len(aborted) == 4
    for ticker in FIXTURE_BAND_TICKERS:
        row = rows[ticker]
        assert row["book_capture"] == row["trades_capture"]
    # The abort takes the LOWEST yes_bid markets, by construction of the pass order.
    assert captured == FIXTURE_BAND_TICKERS[:2]
    # An aborted market has unknown trades, so eligibility is UNKNOWN, not false.
    for ticker in aborted:
        assert rows[ticker]["eligible"] is None
        assert rows[ticker]["skip_reason"] == "trades_unavailable"
        assert rows[ticker]["virtual_entry"] is False
    assert ticks(tmp_path)[0]["tick_outcome"] == "budget_abort"
    assert len(books(tmp_path)) == 2
    assert len(trades(tmp_path)) == 2


async def test_wall_clock_abort(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    mono = MutableMonotonic()
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)

    original_get_orderbook = client.get_orderbook

    async def slow_book(ticker: str, depth: int = 10) -> dict[str, Any]:
        mono.t += 100.0
        return await original_get_orderbook(ticker, depth)

    client.get_orderbook = slow_book  # type: ignore[method-assign]

    obs, _ = make_observer(
        client, tmp_path, clock=clock, monotonic=mono, tick_wall_clock_budget_seconds=180.0
    )
    await obs.run_once()

    rows = {r["ticker"]: r for r in decisions(tmp_path)}
    aborted = [t for t in FIXTURE_BAND_TICKERS if rows[t]["book_capture"] == "wall_clock_abort"]
    assert aborted
    assert ticks(tmp_path)[0]["tick_outcome"] == "wall_clock_abort"
    assert len(decisions(tmp_path)) == 54


async def test_backoff_mid_pass_aborts_the_remainder(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    # Pre-pass read, then two clean markets, then the shared 429 gate goes active.
    client = FakeKalshiClient(
        live_markets,
        trades=live_trades,
        orderbook=live_orderbook,
        backoff_schedule=[0.0, 0.0, 0.0, 12.0],
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    rows = {r["ticker"]: r for r in decisions(tmp_path)}
    captured = [t for t in FIXTURE_BAND_TICKERS if rows[t]["book_capture"] == "ok"]
    aborted = [t for t in FIXTURE_BAND_TICKERS if rows[t]["book_capture"] == "backoff_abort"]
    assert len(captured) == 2
    assert len(aborted) == 4
    for ticker in aborted:
        assert rows[ticker]["trades_capture"] == "backoff_abort"
    assert len(decisions(tmp_path)) == 54
    # A mid-pass backoff truncation must be its own outcome, not "ok" —
    # backoff peaks in the same window decisions cluster in.
    assert ticks(tmp_path)[0]["tick_outcome"] == "backoff_abort"


async def test_short_backoff_does_not_skip_the_tick(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(
        live_markets, trades=live_trades, orderbook=live_orderbook, backoff_schedule=[5.0, 0.0]
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert ticks(tmp_path)[0]["tick_outcome"] == "ok"
    assert ticks(tmp_path)[0]["backoff_remaining_at_start"] == pytest.approx(5.0)
    assert len(decisions(tmp_path)) == 54


def test_kalshi_client_exposes_the_public_backoff_accessor() -> None:
    # A missing accessor would read defensively as "no backoff" forever
    # rather than fail, so its presence is pinned here.
    from python.data.kalshi_client import KalshiClient

    assert callable(getattr(KalshiClient, "backoff_remaining_seconds", None))


async def test_missing_backoff_accessor_is_tolerated(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = NoBackoffAccessorClient(
        live_markets, trades=live_trades, orderbook=live_orderbook
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert ticks(tmp_path)[0]["backoff_remaining_at_start"] == 0.0
    assert len(decisions(tmp_path)) == 54


async def test_a_missing_backoff_accessor_is_loud_and_recorded(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any], caplog: pytest.LogCaptureFixture,
) -> None:
    # ``backoff_remaining_at_start: 0.0`` looks identical whether the gate is
    # genuinely clear or the accessor is gone.
    clock = MutableClock(CAPTURE_TS)
    client = NoBackoffAccessorClient(
        live_markets, trades=live_trades, orderbook=live_orderbook
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    with caplog.at_level("WARNING"):
        await obs.run_once()
        clock.advance(minutes=10)
        await obs.run_once()

    assert all(row["backoff_accessor_present"] is False for row in ticks(tmp_path))
    warnings = [r for r in caplog.records if "backoff" in r.message.lower()]
    assert len(warnings) == 1  # once per process, not 66 times a day


async def test_a_present_backoff_accessor_is_recorded_as_present(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert ticks(tmp_path)[0]["backoff_accessor_present"] is True


async def test_a_raising_backoff_accessor_is_recorded_as_absent(tmp_path: Path) -> None:
    # Present but unusable is the same blindness as missing.
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(ExplodingClient(), tmp_path, clock=clock)
    await obs.run_once()
    assert ticks(tmp_path)[0]["backoff_accessor_present"] is False




async def test_tied_yes_bids_are_shuffled_by_the_tick_seed(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # All six band tickers have distinct yes_bid, so a no-op shuffle would
    # still pass every other assertion here.
    template = _template_market(live_markets)
    cities = ["NYC", "LAX", "MIA", "CHI", "ATL", "DEN"]
    universe = [make_market(template, f"KXRAIN-26SEP18-{c}", 0.40) for c in cities]
    clock_at = CAPTURE_TS

    orders: list[tuple[str, ...]] = []
    for seed in (1, 2, 3, 4):
        target = tmp_path / f"seed{seed}"
        client = FakeKalshiClient(universe, trades=live_trades, orderbook=live_orderbook)
        config = KxRainObserverConfig(log_dir=target, request_pacing_seconds=0.0)
        obs = KxRainObserver(client, config, clock=MutableClock(clock_at), seed=seed)
        await obs.run_once()
        rows = sorted(
            (r for r in decisions(target) if r["pass_index"] >= 0),
            key=lambda r: r["pass_index"],
        )
        assert len(rows) == len(cities)
        orders.append(tuple(r["ticker"] for r in rows))

    assert len(set(orders)) > 1


async def test_the_capture_order_is_reproducible_from_the_logged_seed(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    universe = [
        make_market(template, f"KXRAIN-26SEP18-{c}", 0.40)
        for c in ("NYC", "LAX", "MIA", "CHI", "ATL", "DEN")
    ]
    clock_at = CAPTURE_TS

    seen: list[tuple[Any, ...]] = []
    for run in ("a", "b"):
        target = tmp_path / run
        client = FakeKalshiClient(universe, trades=live_trades, orderbook=live_orderbook)
        config = KxRainObserverConfig(log_dir=target, request_pacing_seconds=0.0)
        obs = KxRainObserver(client, config, clock=MutableClock(clock_at), seed=99)
        await obs.run_once()
        order = tuple(
            r["ticker"]
            for r in sorted(
                (r for r in decisions(target) if r["pass_index"] >= 0),
                key=lambda r: r["pass_index"],
            )
        )
        seen.append((ticks(target)[0]["tick_seed"], order))

    assert seen[0] == seen[1]




async def test_skipped_no_city_in_window(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
) -> None:
    # 08:00 UTC: 03:00 LST in UTC-5, 00:00 LST in UTC-8. Nobody is in window.
    clock = MutableClock(datetime(2026, 9, 18, 8, 0, tzinfo=UTC))
    client = FakeKalshiClient(live_markets, trades=live_trades)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    written = await obs.run_once()

    assert written == 0
    assert client.calls == []
    assert decisions(tmp_path) == []
    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "skipped_no_city_in_window"
    assert row["n_cities_in_window"] == 0
    assert row["n_decision_rows"] == 0


async def test_skipped_backoff(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, backoff_schedule=[95.0])
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    assert client.calls == []
    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "skipped_backoff"
    assert row["backoff_remaining_at_start"] == pytest.approx(95.0)
    assert decisions(tmp_path) == []


async def test_markets_fetch_error_writes_a_tick_row_and_no_decisions(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, markets_error=True)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    written = await obs.run_once()

    assert written == 0
    assert decisions(tmp_path) == []
    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "markets_fetch_error"
    assert row["n_markets_returned"] == 0


async def test_every_tick_outcome_is_reachable_and_in_vocabulary(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    seen: set[str] = set()

    async def one(dirpath: Path, **kwargs: Any) -> None:
        clock = kwargs.pop("clock")
        client = kwargs.pop("client")
        obs, _ = make_observer(client, dirpath, clock=clock, **kwargs)
        await obs.run_once()
        seen.update(r["tick_outcome"] for r in ticks(dirpath))

    await one(
        tmp_path / "ok",
        clock=MutableClock(CAPTURE_TS),
        client=FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook),
    )
    await one(
        tmp_path / "nocity",
        clock=MutableClock(datetime(2026, 9, 18, 8, 0, tzinfo=UTC)),
        client=FakeKalshiClient(live_markets, trades=live_trades),
    )
    await one(
        tmp_path / "backoff",
        clock=MutableClock(CAPTURE_TS),
        client=FakeKalshiClient(live_markets, backoff_schedule=[95.0]),
    )
    await one(
        tmp_path / "fetcherr",
        clock=MutableClock(CAPTURE_TS),
        client=FakeKalshiClient(live_markets, markets_error=True),
    )
    await one(
        tmp_path / "budget",
        clock=MutableClock(CAPTURE_TS),
        client=FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook),
        max_requests_per_tick=6,
    )
    mono = MutableMonotonic()
    slow = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    original = slow.get_orderbook

    async def slow_book(ticker: str, depth: int = 10) -> dict[str, Any]:
        mono.t += 100.0
        return await original(ticker, depth)

    slow.get_orderbook = slow_book  # type: ignore[method-assign]
    obs, _ = make_observer(
        slow, tmp_path / "wall", clock=MutableClock(CAPTURE_TS), monotonic=mono
    )
    await obs.run_once()
    seen.update(r["tick_outcome"] for r in ticks(tmp_path / "wall"))

    await one(
        tmp_path / "midbackoff",
        clock=MutableClock(CAPTURE_TS),
        client=FakeKalshiClient(
            live_markets,
            trades=live_trades,
            orderbook=live_orderbook,
            backoff_schedule=[0.0, 0.0, 0.0, 12.0],
        ),
    )

    disk = tmp_path / "disk"
    obs, _ = make_observer(
        FakeKalshiClient(live_markets), disk, clock=MutableClock(CAPTURE_TS)
    )
    obs._config = KxRainObserverConfig(
        log_dir=disk, request_pacing_seconds=0.0, min_free_disk_bytes=2**62
    )
    await obs.run_once()
    seen.update(r["tick_outcome"] for r in ticks(disk))

    err = tmp_path / "err"
    obs, _ = make_observer(
        FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook),
        err,
        clock=MutableClock(CAPTURE_TS),
    )

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("inside the tick")

    obs._build_candidates = boom  # type: ignore[method-assign]
    await obs.run_once()
    seen.update(r["tick_outcome"] for r in ticks(err))

    assert seen == set(TICK_OUTCOMES)




async def test_disk_below_floor_self_disables_with_zero_http(
    tmp_path: Path, live_markets: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer_module, "_free_disk_bytes", lambda path: 1_000_000)
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier)

    for _ in range(3):
        await obs.run_once()

    assert obs.disabled is True
    assert client.calls == []
    assert decisions(tmp_path) == []
    assert len(notifier.alerts) == 1
    assert notifier.alerts[0][1] == AlertLevel.CRITICAL

    # Exactly one row, at the disabling transition, not every tick after.
    rows = ticks(tmp_path)
    assert [r["tick_outcome"] for r in rows] == ["disk_disabled"]


async def test_disk_above_floor_runs_normally(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observer_module, "_free_disk_bytes", lambda path: 50 * 1024**3)
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert obs.disabled is False
    assert len(decisions(tmp_path)) == 54


async def test_disk_usage_failure_does_not_disable(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(path: Path) -> int:
        raise OSError("statvfs failed")

    monkeypatch.setattr(observer_module, "_free_disk_bytes", boom)
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert obs.disabled is False
    assert len(decisions(tmp_path)) == 54




async def test_client_raising_on_everything_never_raises(tmp_path: Path) -> None:
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(ExplodingClient(), tmp_path, clock=clock)
    assert await obs.run_once() == 0
    assert ticks(tmp_path)[0]["tick_outcome"] == "markets_fetch_error"


async def test_trades_error_yields_unknown_not_ineligible(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_orderbook: dict[str, Any]
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades_error=True, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    rows = {r["ticker"]: r for r in decisions(tmp_path)}
    for ticker in FIXTURE_BAND_TICKERS:
        assert rows[ticker]["eligible"] is None
        assert rows[ticker]["skip_reason"] == "trades_unavailable"
        assert rows[ticker]["trades_capture"] == "error"
        assert rows[ticker]["virtual_entry"] is False
        # The book still came back, so it is still recorded.
        assert rows[ticker]["book_capture"] == "ok"
    assert trades(tmp_path) == []
    assert len(books(tmp_path)) == len(FIXTURE_BAND_TICKERS)
    assert KxRainLedger.load(tmp_path / LEDGER_FILENAME).keys() == []


async def test_book_error_still_writes_the_decision_row(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any]
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, book_error=True)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    rows = {r["ticker"]: r for r in decisions(tmp_path)}
    for ticker in FIXTURE_BAND_TICKERS:
        assert rows[ticker]["book_capture"] == "error"
        assert rows[ticker]["trades_capture"] == "ok"
        assert rows[ticker]["eligible"] is True
    assert books(tmp_path) == []
    assert len(trades(tmp_path)) == len(FIXTURE_BAND_TICKERS)


async def test_writer_oserror_is_swallowed(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)

    def boom(rows: list[dict[str, Any]]) -> None:
        raise OSError("disk on fire")

    for writer in obs.writers.values():
        writer.write_batch = boom  # type: ignore[method-assign]

    assert await obs.run_once() >= 0
    # Swallowed, but NOT invisible: the rows are counted lost and liveness is
    # never advanced, or the only dead-corpus detector stays quiet for 18 weeks.
    assert obs.rows_lost > 0
    assert obs.last_decision_write_utc is None


async def test_a_raising_tick_still_writes_a_tick_row(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A corpus gap must never lack an on-disk explanation.
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("something inside the tick exploded")

    obs._build_candidates = boom  # type: ignore[method-assign]

    assert await obs.run_once() == 0
    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "error"
    assert row["scan_id"]


async def test_a_non_dict_trades_payload_is_an_error_not_a_raise(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_orderbook: dict[str, Any]
) -> None:
    template = _template_market(live_markets)
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)], orderbook=live_orderbook
    )

    async def list_body(**kwargs: Any) -> Any:
        client.calls.append(("get_market_trades", kwargs.get("ticker")))
        return [{"not": "a dict"}]

    client.get_market_trades = list_body  # type: ignore[method-assign]
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    row = decisions(tmp_path)[0]
    assert row["trades_capture"] == "error"
    assert row["eligible"] is None
    assert ticks(tmp_path)[0]["tick_outcome"] == "ok"


async def test_a_non_dict_orderbook_is_an_error_not_a_raise(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any]
) -> None:
    template = _template_market(live_markets)
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)], trades=live_trades
    )

    async def list_body(ticker: str, depth: int = 10) -> Any:
        client.calls.append(("get_orderbook", ticker))
        return ["not", "a", "dict"]

    client.get_orderbook = list_body  # type: ignore[method-assign]
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    assert decisions(tmp_path)[0]["book_capture"] == "error"
    assert books(tmp_path) == []
    assert ticks(tmp_path)[0]["tick_outcome"] == "ok"


async def test_a_non_dict_market_entry_is_counted_not_fatal(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40), "KXRAIN-26SEP18-LAX"],  # type: ignore[list-item]
        trades=live_trades,
        orderbook=live_orderbook,
    )
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "ok"
    assert row["n_markets_returned"] == 2
    assert row["n_parsed"] == 1
    assert row["n_unparseable"] == 1
    assert len(decisions(tmp_path)) == 1


async def test_notifier_raising_does_not_break_the_tick(
    tmp_path: Path, live_markets: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer_module, "_free_disk_bytes", lambda path: 1)
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(
        FakeKalshiClient(live_markets), tmp_path, clock=clock, notifier=RaisingNotifier()
    )
    await obs.run_once()
    assert obs.disabled is True


async def test_none_notifier_getter_mid_shutdown(
    tmp_path: Path, live_markets: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(observer_module, "_free_disk_bytes", lambda path: 1)
    clock = MutableClock(CAPTURE_TS)
    config = KxRainObserverConfig(log_dir=tmp_path, request_pacing_seconds=0.0)

    def getter() -> Any:
        raise RuntimeError("notifier is gone")

    obs = KxRainObserver(
        FakeKalshiClient(live_markets),
        config,
        notifier_getter=getter,
        clock=clock,
        seed=1,
    )
    await obs.run_once()
    assert obs.disabled is True




async def test_extra_page_fetched_only_when_no_side_is_empty(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    yes_only = trades_payload(live_trades, side="yes", cursor="PAGE2")
    page_two = trades_payload(live_trades, side="no", cursor="PAGE3")
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)],
        trades=yes_only,
        orderbook=live_orderbook,
        extra_page=page_two,
    )
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    assert client.count("get_market_trades") == 2
    assert client.trade_call_kwargs[1]["cursor"] == "PAGE2"
    row = trades(tmp_path)[0]
    assert row["n_pages_fetched"] == 2
    assert row["n_trades_no_side"] > 0
    # The one case where truncation can flip the eligibility answer.
    assert decisions(tmp_path)[0]["eligible"] is True
    assert ticks(tmp_path)[0]["requests_used"] == 1 + 2 + 1


async def test_no_extra_page_when_a_no_print_is_already_present(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)],
        trades=trades_payload(live_trades, side=None, cursor="PAGE2"),
        orderbook=live_orderbook,
        extra_page=trades_payload(live_trades, side="no"),
    )
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert client.count("get_market_trades") == 1
    assert trades(tmp_path)[0]["n_pages_fetched"] == 1


async def test_no_extra_page_when_the_cursor_is_empty(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)],
        trades=trades_payload(live_trades, side="yes", cursor=""),
        orderbook=live_orderbook,
        extra_page=trades_payload(live_trades, side="no"),
    )
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()
    assert client.count("get_market_trades") == 1
    assert decisions(tmp_path)[0]["skip_reason"] == "no_recent_no_trade"
    assert decisions(tmp_path)[0]["eligible"] is False




async def test_unparseable_kxrain_tickers_are_counted_and_capped(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A suffix absent from the frozen 27-city map is counted, never dropped.
    template = _template_market(live_markets)
    universe = [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)]
    universe += [
        make_market(template, f"KXRAIN-26SEP18-ZZ{i}", 0.40) for i in range(8)
    ]
    client = FakeKalshiClient(universe, trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    row = ticks(tmp_path)[0]
    assert row["n_markets_returned"] == 9
    assert row["n_parsed"] == 1
    assert row["n_unparseable"] == 8
    assert len(row["unparseable_tickers"]) == 5
    assert row["unparseable_tickers_truncated"] is True
    assert len(decisions(tmp_path)) == 1


async def test_market_absent_census_row(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    present = make_market(template, "KXRAIN-26SEP18-NYC", 0.40)
    other = make_market(template, "KXRAIN-26SEP18-LAX", 0.95)
    client = FakeKalshiClient([present, other], trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = [other]
    for _ in range(CENSUS_ABSENT_TICKS_REQUIRED):
        clock.advance(minutes=10)
        await obs.run_once()

    rows = decisions(tmp_path)
    absent = [r for r in rows if r["skip_reason"] == "market_absent"]
    assert len(absent) == 1
    assert absent[0]["ticker"] == "KXRAIN-26SEP18-NYC"
    assert absent[0]["last_seen_yes_bid"] == pytest.approx(0.40)
    assert absent[0]["last_seen_ts"]
    assert absent[0]["book_capture"] == "skipped_market_absent"
    assert absent[0]["pass_index"] == -1


async def test_census_row_needs_two_consecutive_absent_ticks(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A short page with an empty cursor looks like a complete one, so one
    # bad tick must not fabricate a row.
    template = _template_market(live_markets)
    present = make_market(template, "KXRAIN-26SEP18-NYC", 0.40)
    other = make_market(template, "KXRAIN-26SEP18-LAX", 0.95)
    client = FakeKalshiClient([present, other], trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = [other]
    clock.advance(minutes=10)
    await obs.run_once()
    assert [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"] == []

    clock.advance(minutes=10)
    await obs.run_once()
    absent = [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"]
    assert len(absent) == 1
    assert absent[0]["ticker"] == "KXRAIN-26SEP18-NYC"


async def test_a_transient_absence_never_becomes_a_census_row(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    present = make_market(template, "KXRAIN-26SEP18-NYC", 0.40)
    other = make_market(template, "KXRAIN-26SEP18-LAX", 0.95)
    client = FakeKalshiClient([present, other], trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = [other]
    clock.advance(minutes=10)
    await obs.run_once()

    client._markets = [present, other]
    clock.advance(minutes=10)
    await obs.run_once()
    clock.advance(minutes=10)
    client._markets = [other]
    await obs.run_once()

    assert [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"] == []


async def test_a_mid_day_rename_is_not_a_vanished_market_day(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # The ticker suffix (e.g. PROV -> PVD) can be renamed mid-corpus; both
    # spellings must resolve to one corpus_key.
    template = _template_market(live_markets)
    old = make_market(template, "KXRAIN-26SEP15-PROV", 0.40)
    new = make_market(template, "KXRAIN-26SEP15-PVD", 0.40)
    client = FakeKalshiClient([old], trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 15, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = [new]
    for _ in range(3):
        clock.advance(minutes=10)
        await obs.run_once()

    assert [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"] == []


async def test_truncated_markets_page_emits_no_census_rows(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    present = make_market(template, "KXRAIN-26SEP18-NYC", 0.40)
    other = make_market(template, "KXRAIN-26SEP18-LAX", 0.95)
    client = FakeKalshiClient([present, other], trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = [other]
    client.endless_cursor = True
    for _ in range(3):
        clock.advance(minutes=10)
        await obs.run_once()

    assert [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"] == []
    row = ticks(tmp_path)[-1]
    assert row["markets_truncated"] is True
    assert row["census_suppressed"] is True


async def test_cursor_cycle_emits_no_census_rows(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    present = make_market(template, "KXRAIN-26SEP18-NYC", 0.40)
    other = make_market(template, "KXRAIN-26SEP18-LAX", 0.95)
    client = FakeKalshiClient([present, other], trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = [other]
    client.cycling_cursor = True
    for _ in range(3):
        clock.advance(minutes=10)
        await obs.run_once()

    assert [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"] == []
    row = ticks(tmp_path)[-1]
    assert row["markets_cycle_aborted"] is True
    assert row["census_suppressed"] is True




async def test_markets_fetch_is_bounded_by_max_pages(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    # An under-charged request count would hide a sustained overshoot
    # against the shared RateLimiter.
    client = FakeKalshiClient(live_markets)
    client.endless_cursor = True
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    assert client.count("get_markets") == observer_module.MARKETS_FETCH_MAX_PAGES
    row = ticks(tmp_path)[0]
    assert row["markets_pages_consumed"] == observer_module.MARKETS_FETCH_MAX_PAGES
    assert row["markets_truncated"] is True
    assert row["requests_used"] >= observer_module.MARKETS_FETCH_MAX_PAGES


async def test_a_markets_fetch_that_raises_mid_pagination_still_charges_its_pages(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    # Charging only on the success path would skip it entirely on exactly
    # the ticks where the API is misbehaving and it matters most.
    client = FakeKalshiClient(live_markets)
    calls = {"n": 0}

    async def fail_on_page_three(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        client.calls.append(("get_markets", None))
        if calls["n"] >= 3:
            raise RuntimeError("500 from Kalshi")
        return {"markets": [], "cursor": f"p{calls['n']}"}

    client.get_markets = fail_on_page_three  # type: ignore[method-assign]
    obs, _ = make_observer(client, tmp_path, clock=MutableClock(CAPTURE_TS))
    await obs.run_once()

    row = ticks(tmp_path)[0]
    assert row["tick_outcome"] == "markets_fetch_error"
    assert calls["n"] == 3
    assert row["markets_pages_consumed"] == 3
    assert row["requests_used"] == 3


async def test_the_markets_page_limit_is_pinned_not_inherited(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # The "3 pages covers the universe" bound depends on this value.
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    seen: list[int] = []
    original = client.get_markets

    async def record(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs.get("limit"))  # type: ignore[arg-type]
        return await original(**kwargs)

    client.get_markets = record  # type: ignore[method-assign]
    obs, _ = make_observer(client, tmp_path, clock=MutableClock(CAPTURE_TS))
    await obs.run_once()
    assert seen == [observer_module.MARKETS_PAGE_LIMIT]


async def test_a_mass_disappearance_is_a_short_page_not_an_early_close(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A short page with a terminal cursor carries no flag of its own, so
    # the census must tell it apart from a real mass early close by size.
    template = _template_market(live_markets)
    universe = [
        make_market(template, f"KXRAIN-26SEP18-{c}", 0.40)
        for c in ("NYC", "LAX", "MIA", "CHI", "ATL", "DEN")
    ]
    client = FakeKalshiClient(universe, trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = universe[:1]  # the page goes short, cursor still terminal
    for _ in range(4):
        clock.advance(minutes=10)
        await obs.run_once()

    assert [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"] == []
    assert ticks(tmp_path)[-1]["census_suppressed"] is True


async def test_a_single_early_close_still_produces_a_census_row(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # Positive control: the mass-absence guard must not swallow a real close.
    template = _template_market(live_markets)
    universe = [
        make_market(template, f"KXRAIN-26SEP18-{c}", 0.40)
        for c in ("NYC", "LAX", "MIA", "CHI", "ATL", "DEN")
    ]
    client = FakeKalshiClient(universe, trades=live_trades, orderbook=live_orderbook)
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    client._markets = universe[1:]  # exactly one market-day closes early
    for _ in range(CENSUS_ABSENT_TICKS_REQUIRED):
        clock.advance(minutes=10)
        await obs.run_once()

    absent = [r for r in decisions(tmp_path) if r["skip_reason"] == "market_absent"]
    assert len(absent) == 1
    assert absent[0]["ticker"] == "KXRAIN-26SEP18-NYC"
    assert ticks(tmp_path)[-1]["census_suppressed"] is False


async def test_an_incomplete_listing_raises_an_alert_not_only_a_log_line(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    notifier = RecordingNotifier()
    mono = MutableMonotonic()
    client = FakeKalshiClient(live_markets)
    client.endless_cursor = True
    obs, _ = make_observer(
        client, tmp_path, clock=MutableClock(CAPTURE_TS), notifier=notifier, monotonic=mono
    )
    await obs.run_once()
    assert [a for a in notifier.alerts if "incomplete" in a[0].lower()]

    await obs.run_once()  # inside the cooldown — no second alert
    assert len([a for a in notifier.alerts if "incomplete" in a[0].lower()]) == 1


async def test_markets_fetch_charges_the_pages_it_actually_used(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    template = _template_market(live_markets)
    client = FakeKalshiClient(
        [make_market(template, "KXRAIN-26SEP18-NYC", 0.40)],
        trades=trades_payload(live_trades, side=None, cursor=""),
        orderbook=live_orderbook,
    )
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    row = ticks(tmp_path)[0]
    assert row["markets_pages_consumed"] == 1
    assert row["markets_cycle_aborted"] is False
    assert row["census_suppressed"] is False
    assert row["requests_used"] == 1 + 2


async def test_oversize_row_is_replaced_by_a_join_keys_stub(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observer_module, "encode_row", lambda row: None)
    clock = MutableClock(datetime(2026, 9, 18, 15, 0, tzinfo=UTC))
    client = FakeKalshiClient(
        [make_market(_template_market(live_markets), "KXRAIN-26SEP18-NYC", 0.40)],
        trades=live_trades,
        orderbook=live_orderbook,
    )
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    row = decisions(tmp_path)[0]
    assert row["oversize"] is True
    assert "ticker" in row and "decision_id" in row
    assert obs.oversize_rows_dropped >= 1




async def test_liveness_alert_fires_once_when_the_corpus_goes_silent(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, markets_error=True)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier)

    for _ in range(12):  # 2 hours of failing ticks, cities in window throughout
        await obs.run_once()
        clock.advance(minutes=10)

    liveness = [a for a in notifier.alerts if "liveness" in a[0].lower()]
    assert len(liveness) == 1
    assert liveness[0][1] == AlertLevel.WARNING


async def test_liveness_alert_does_not_fire_out_of_window(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    notifier = RecordingNotifier()
    clock = MutableClock(datetime(2026, 9, 18, 6, 0, tzinfo=UTC))
    client = FakeKalshiClient(live_markets, markets_error=True)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier)

    for _ in range(12):
        await obs.run_once()
        clock.advance(minutes=10)

    assert [a for a in notifier.alerts if "liveness" in a[0].lower()] == []


async def test_liveness_fires_when_rows_are_built_but_never_reach_disk(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # The detector must count persisted rows, not constructed ones, or a
    # read-only ``state/`` would never trip it.
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier)

    # Mirrors what ForecastsRawWriter does on an OSError: swallow it and
    # leave success_count alone.
    for writer in obs.writers.values():
        writer.write_batch = lambda rows: None  # type: ignore[method-assign]

    for _ in range(12):  # 2 hours, cities in window, every tick "succeeding"
        await obs.run_once()
        clock.advance(minutes=10)

    assert decisions(tmp_path) == []
    liveness = [a for a in notifier.alerts if "liveness" in a[0].lower()]
    assert len(liveness) == 1
    assert liveness[0][1] == AlertLevel.WARNING


async def test_liveness_stays_quiet_while_rows_actually_reach_disk(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # Positive control for the test above: a healthy corpus must not alert.
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier)

    for _ in range(12):
        await obs.run_once()
        clock.advance(minutes=10)

    assert decisions(tmp_path)
    assert [a for a in notifier.alerts if "liveness" in a[0].lower()] == []


async def test_partial_batch_write_counts_as_liveness(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A batch that lands 1 of 10 rows is progress, not silence.
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier)

    writer = obs.writers[KXRAIN_DECISIONS_FILENAME]
    original = writer.write_batch

    def partial(rows: list[dict[str, Any]]) -> None:
        original(rows[:1])

    writer.write_batch = partial  # type: ignore[method-assign]

    for _ in range(12):
        await obs.run_once()
        clock.advance(minutes=10)

    assert [a for a in notifier.alerts if "liveness" in a[0].lower()] == []


async def test_consecutive_failed_ticks_alert(tmp_path: Path) -> None:
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    config = KxRainObserverConfig(
        log_dir=tmp_path, request_pacing_seconds=0.0, tick_interval_seconds=0.0
    )
    obs = KxRainObserver(
        FakeKalshiClient([]),
        config,
        notifier_getter=lambda: notifier,
        clock=clock,
        seed=1,
    )

    calls = {"n": 0}

    async def boom() -> int:
        calls["n"] += 1
        raise RuntimeError("tick exploded")

    obs.run_once = boom  # type: ignore[method-assign]

    task = asyncio.create_task(obs.run())
    for _ in range(200):
        await asyncio.sleep(0)
        if calls["n"] >= 6:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    failures = [a for a in notifier.alerts if "consecutive" in a[0].lower()]
    assert len(failures) == 1
    assert failures[0][1] == AlertLevel.WARNING


async def test_a_crash_looping_tick_still_raises_the_consecutive_failure_alert(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # If `_tick` swallowed the raise and returned 0, `run()` would reset the
    # counter every tick and this alert would become dead code.
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    config = KxRainObserverConfig(
        log_dir=tmp_path, request_pacing_seconds=0.0, tick_interval_seconds=0.0
    )
    obs = KxRainObserver(
        FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook),
        config,
        notifier_getter=lambda: notifier,
        clock=clock,
        seed=1,
    )

    async def boom(*args: Any, **kwargs: Any) -> int:
        raise TypeError("a dependency is broken")

    obs._tick_body = boom  # type: ignore[method-assign]

    task = asyncio.create_task(obs.run())
    for _ in range(400):
        await asyncio.sleep(0)
        if len(ticks(tmp_path)) >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD + 1:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert obs.last_tick_outcome == "error"
    assert obs._consecutive_failed_ticks >= CONSECUTIVE_FAILURE_ALERT_THRESHOLD
    assert [a for a in notifier.alerts if "consecutive" in a[0].lower()]


async def test_a_healthy_tick_clears_the_failure_counter(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    config = KxRainObserverConfig(
        log_dir=tmp_path, request_pacing_seconds=0.0, tick_interval_seconds=0.0
    )
    obs = KxRainObserver(
        FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook),
        config,
        notifier_getter=lambda: notifier,
        clock=clock,
        seed=1,
    )
    task = asyncio.create_task(obs.run())
    for _ in range(400):
        await asyncio.sleep(0)
        if len(ticks(tmp_path)) >= 3:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert obs.last_tick_outcome == "ok"
    assert obs._consecutive_failed_ticks == 0
    assert [a for a in notifier.alerts if "consecutive" in a[0].lower()] == []


async def test_the_liveness_alert_repeats_rather_than_firing_once_per_process(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    notifier = RecordingNotifier()
    clock = MutableClock(CAPTURE_TS)
    mono = MutableMonotonic()
    client = FakeKalshiClient(live_markets, markets_error=True)
    obs, _ = make_observer(client, tmp_path, clock=clock, notifier=notifier, monotonic=mono)

    for _ in range(40):  # ~7 h of failing ticks, cities in window throughout
        await obs.run_once()
        clock.advance(minutes=10)
        mono.t += 600.0

    liveness = [a for a in notifier.alerts if "liveness" in a[0].lower()]
    assert len(liveness) > 1
    assert all(level == AlertLevel.WARNING for _, level in liveness)


async def test_run_exits_only_on_cancelled_error(tmp_path: Path) -> None:
    clock = MutableClock(CAPTURE_TS)
    config = KxRainObserverConfig(
        log_dir=tmp_path, request_pacing_seconds=0.0, tick_interval_seconds=0.0
    )
    obs = KxRainObserver(FakeKalshiClient([]), config, clock=clock, seed=1)

    calls = {"n": 0}

    async def boom() -> int:
        calls["n"] += 1
        raise ValueError("tick exploded")

    obs.run_once = boom  # type: ignore[method-assign]

    task = asyncio.create_task(obs.run())
    for _ in range(200):
        await asyncio.sleep(0)
        if calls["n"] >= 3:
            break
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task




async def test_rows_are_written_incrementally_not_in_one_end_of_tick_batch(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)

    batches: list[int] = []
    writer = obs.writers[KXRAIN_DECISIONS_FILENAME]
    original = writer.write_batch

    def spy(rows: list[dict[str, Any]]) -> None:
        batches.append(len(rows))
        original(rows)

    writer.write_batch = spy  # type: ignore[method-assign]
    await obs.run_once()

    assert len(batches) > 1
    assert sum(batches) == 54


async def test_cancellation_mid_tick_flushes_synchronously(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)

    state = {"n": 0}
    original_book = client.get_orderbook

    async def cancel_on_third(ticker: str, depth: int = 10) -> dict[str, Any]:
        state["n"] += 1
        if state["n"] == 3:
            raise asyncio.CancelledError()
        return await original_book(ticker, depth)

    client.get_orderbook = cancel_on_third  # type: ignore[method-assign]

    obs, _ = make_observer(client, tmp_path, clock=clock)

    threads: list[str] = []
    writer = obs.writers[KXRAIN_DECISIONS_FILENAME]
    original = writer.write_batch

    def spy(rows: list[dict[str, Any]]) -> None:
        threads.append(threading.current_thread().name)
        original(rows)

    writer.write_batch = spy  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await obs.run_once()

    # Rows accumulated before the cancellation are on disk, not discarded.
    assert decisions(tmp_path)
    assert threads[-1] == threading.main_thread().name


async def test_cancellation_during_a_flush_does_not_double_write(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # A thread can't be cancelled once running: the await raises but the
    # write finishes anyway, so requeuing the batch would write it twice.
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)

    writer = obs.writers[KXRAIN_DECISIONS_FILENAME]
    original = writer.write_batch
    state = {"n": 0}

    def cancel_after_writing(rows: list[dict[str, Any]]) -> None:
        original(rows)
        state["n"] += 1
        if state["n"] == 1:
            raise asyncio.CancelledError()

    writer.write_batch = cancel_after_writing  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await obs.run_once()

    ids = [row["decision_id"] for row in decisions(tmp_path)]
    assert ids
    assert len(ids) == len(set(ids))


async def test_a_queued_write_cancelled_before_it_starts_is_handed_back(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # The other failure mode: `shield` alone would silently drop a batch
    # still queued when loop shutdown cancels it.
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)

    blocked = threading.Event()
    obs._executor.submit(blocked.wait)  # occupy the observer's single worker
    for i in range(10):
        obs._queue(KXRAIN_DECISIONS_FILENAME, {"decision_id": f"x{i}", "scan_id": "s"})

    task = asyncio.create_task(obs._flush(CAPTURE_TS))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(obs._pending[KXRAIN_DECISIONS_FILENAME]) == 10
    blocked.set()
    obs._flush_sync()

    ids = [row["decision_id"] for row in decisions(tmp_path)]
    assert len(ids) == 10
    assert len(set(ids)) == 10


async def test_cancelling_the_task_mid_write_loses_nothing_and_duplicates_nothing(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # The real shutdown shape: cancel the task while the thread is writing.
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)

    writer = obs.writers[KXRAIN_DECISIONS_FILENAME]
    original = writer.write_batch
    started = threading.Event()

    def slow(rows: list[dict[str, Any]]) -> None:
        started.set()
        time.sleep(0.15)
        original(rows)

    writer.write_batch = slow  # type: ignore[method-assign]

    task = asyncio.create_task(obs.run_once())
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)  # let the executor thread finish

    ids = [row["decision_id"] for row in decisions(tmp_path)]
    assert len(ids) >= WRITE_FLUSH_EVERY_ROWS
    assert len(ids) == len(set(ids))




async def test_now_utc_is_computed_once_per_tick(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    class DriftingClock:
        def __init__(self, start: datetime) -> None:
            self.now = start
            self.reads = 0

        def __call__(self) -> datetime:
            self.reads += 1
            # A full day per read: if `ts` were re-read per market the rows
            # would carry different days.
            self.now = self.now + timedelta(days=1)
            return self.now

    clock = DriftingClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    config = KxRainObserverConfig(log_dir=tmp_path, request_pacing_seconds=0.0)
    obs = KxRainObserver(client, config, clock=clock, seed=7)
    await obs.run_once()

    stamps = {row["ts"] for row in decisions(tmp_path)}
    assert len(stamps) == 1




async def test_pacing_sleeps_between_per_market_reads(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    # Burst-free is the point: the undocumented sliding window trips on bursts.
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock, request_pacing_seconds=0.001)
    await obs.run_once()
    assert obs.pacing_sleeps == len(FIXTURE_BAND_TICKERS) - 1


async def test_request_budget_is_never_exceeded(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    for cap in (3, 4, 8, 12, 60):
        target = tmp_path / f"cap{cap}"
        client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
        obs, _ = make_observer(client, target, clock=clock, max_requests_per_tick=cap)
        await obs.run_once()
        used = ticks(target)[0]["requests_used"]
        assert used <= cap + observer_module.OVERDRAFT_REQUESTS
        assert len(client.calls) <= used


def _iter_rows(tmp_path: Path) -> Iterator[dict[str, Any]]:
    for name in (
        KXRAIN_DECISIONS_FILENAME,
        KXRAIN_BOOK_FILENAME,
        KXRAIN_TRADES_FILENAME,
        KXRAIN_TICKS_FILENAME,
    ):
        yield from read_stream(tmp_path, name)


async def test_every_written_row_is_under_pipe_buf(
    tmp_path: Path, live_markets: list[dict[str, Any]], live_trades: dict[str, Any],
    live_orderbook: dict[str, Any],
) -> None:
    clock = MutableClock(CAPTURE_TS)
    client = FakeKalshiClient(live_markets, trades=live_trades, orderbook=live_orderbook)
    obs, _ = make_observer(client, tmp_path, clock=clock)
    await obs.run_once()

    rows = list(_iter_rows(tmp_path))
    assert rows
    for row in rows:
        assert len(json.dumps(row, default=str).encode()) + 1 <= 4096


async def test_aclose_does_not_starve_the_event_loop(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    # `aclose` runs upstream of live order cancellation; run inline, a
    # stalled disk would starve the whole loop and delay it.
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(FakeKalshiClient(live_markets), tmp_path, clock=clock)
    obs._flush_sync = lambda: time.sleep(0.4)  # type: ignore[method-assign]

    beats = {"n": 0}

    async def heartbeat() -> None:
        while True:
            beats["n"] += 1
            await asyncio.sleep(0.01)

    hb = asyncio.create_task(heartbeat())
    await obs.aclose()
    hb.cancel()

    assert beats["n"] > 5


async def test_aclose_gives_up_rather_than_hold_shutdown_open(
    tmp_path: Path, live_markets: list[dict[str, Any]]
) -> None:
    clock = MutableClock(CAPTURE_TS)
    obs, _ = make_observer(FakeKalshiClient(live_markets), tmp_path, clock=clock)
    obs._flush_sync = lambda: time.sleep(5.0)  # type: ignore[method-assign]

    start = time.monotonic()
    await obs.aclose(timeout=0.2)
    assert time.monotonic() - start < 2.0
