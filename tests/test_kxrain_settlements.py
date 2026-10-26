"""Tests for the KXRAIN settlement sweep.

The most dangerous failure mode under test: an open market carries `result: ""`,
which would parse to a fabricated NO settlement in a corpus that only buys NO.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_settlements import (  # noqa: E402
    KXRAIN_SETTLEMENTS_FILENAME,
    SETTLED_INDEX_FILENAME,
    SETTLEMENT_SWEEP_MAX_PAGES,
    KxRainSettlementSweep,
    SettledIndex,
)

FIXTURES = Path(__file__).parent / "fixtures" / "kxrain"
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def settled_market() -> dict[str, Any]:
    return json.loads((FIXTURES / "live_settled_market.json").read_text())


def make_settled(
    template: dict[str, Any],
    ticker: str,
    *,
    result: str = "no",
    expiration_value: str = "0.00",
) -> dict[str, Any]:
    """Clone the real settled payload with a new ticker/outcome, keeping the live shape."""
    row = copy.deepcopy(template)
    row["ticker"] = ticker
    row["event_ticker"] = "-".join(ticker.split("-")[:2])
    row["result"] = result
    row["expiration_value"] = expiration_value
    return row


class FakeClient:
    """Paginates a settled listing, newest-first, exactly as Kalshi does."""

    def __init__(
        self,
        markets: list[dict[str, Any]],
        *,
        page_size: int = 100,
        error_on_page: int | None = None,
        endless_cursor: bool = False,
    ) -> None:
        self._markets = markets
        self._page_size = page_size
        self._error_on_page = error_on_page
        self._endless_cursor = endless_cursor
        self.calls: list[dict[str, Any]] = []

    async def get_markets(
        self,
        *,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.calls.append({"status": status, "cursor": cursor, "limit": limit})
        if self._error_on_page is not None and len(self.calls) >= self._error_on_page:
            raise RuntimeError("500 from Kalshi")
        offset = int(cursor) if cursor else 0
        page = self._markets[offset : offset + self._page_size]
        nxt = offset + len(page)
        if self._endless_cursor:
            next_cursor = str(nxt + 1)
        else:
            next_cursor = str(nxt) if nxt < len(self._markets) else ""
        return {"markets": [copy.deepcopy(m) for m in page], "cursor": next_cursor}


def make_sweep(client: Any, tmp_path: Path, **kwargs: Any) -> KxRainSettlementSweep:
    return KxRainSettlementSweep(
        client, log_dir=tmp_path, clock=lambda: NOW, page_delay=0.0, **kwargs
    )


def rows(tmp_path: Path) -> list[dict[str, Any]]:
    stem = Path(KXRAIN_SETTLEMENTS_FILENAME).stem
    out: list[dict[str, Any]] = []
    for path in sorted(tmp_path.glob(f"{stem}*.jsonl")):
        out.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return out


# The phantom NO settlement


async def test_an_unsettled_market_is_never_written(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    open_market = make_settled(
        settled_market, "KXRAIN-26SEP19-NYC", result="", expiration_value=""
    )
    sweep = make_sweep(FakeClient([open_market]), tmp_path)
    result = await sweep.run_once()

    assert rows(tmp_path) == []
    assert result.n_written == 0
    assert result.n_refused_unsettled == 1


async def test_a_settled_market_is_written_with_the_outcome_derived(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    sweep = make_sweep(FakeClient([settled_market]), tmp_path)
    result = await sweep.run_once()

    assert result.n_written == 1
    row = rows(tmp_path)[0]
    assert row["ticker"] == "KXRAIN-26SEP17-PHIL"
    assert row["station"] == "PHL"
    assert row["result"] == "yes"
    assert row["expiration_value_in"] == pytest.approx(0.05)
    assert row["derived_yes"] is True
    assert row["rule_consistent"] is True
    # API spellings, verified against the captured market.
    assert row["status"] == "finalized"
    assert row["settlement_ts"]
    assert isinstance(row["open_interest"], float)


async def test_an_inconsistent_settlement_is_recorded_not_suppressed(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    liar = make_settled(
        settled_market, "KXRAIN-26SEP19-NYC", result="no", expiration_value="0.42"
    )
    sweep = make_sweep(FakeClient([liar]), tmp_path)
    result = await sweep.run_once()

    assert result.n_written == 1
    assert result.n_rule_inconsistent == 1
    assert rows(tmp_path)[0]["rule_consistent"] is False


# Idempotence on corpus_key, not on settlement_date


async def test_a_second_sweep_writes_nothing_new(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    universe = [
        make_settled(settled_market, f"KXRAIN-26SEP17-{c}") for c in ("NYC", "LAX", "MIA")
    ]
    client = FakeClient(universe)
    assert (await make_sweep(client, tmp_path).run_once()).n_written == 3
    second = await make_sweep(client, tmp_path).run_once()

    assert second.n_written == 0
    assert second.n_already_present == 3
    assert len(rows(tmp_path)) == 3


async def test_the_first_sweep_after_utc_midnight_marks_what_it_wrote(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # The daily sweep's write is always the first after rotation; a size delta taken
    # across the rotated file read as 0 bytes, so nothing was marked and every row
    # was rewritten the next day.
    now = [NOW]
    day1 = [make_settled(settled_market, f"KXRAIN-26SEP17-{c}") for c in ("NYC", "LAX", "MIA")]
    day2 = day1 + [make_settled(settled_market, f"KXRAIN-26SEP18-{c}") for c in ("NYC", "LAX")]
    sweep = KxRainSettlementSweep(
        FakeClient(day1), log_dir=tmp_path, clock=lambda: now[0], page_delay=0.0
    )
    assert (await sweep.run_once()).n_persisted == 3

    now[0] = NOW + timedelta(days=1)
    sweep._client = FakeClient(day2)
    second = await sweep.run_once()
    assert second.n_written == 2
    assert second.n_persisted == 2
    assert not second.anomaly

    third = await sweep.run_once()
    assert third.n_written == 0
    assert len(rows(tmp_path)) == 5


async def test_a_partial_sweep_retries_only_the_market_days_it_missed(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # Keying on the date alone would mark the whole settlement_date present,
    # losing the market-days a mid-sweep error never reached.
    universe = [
        make_settled(settled_market, f"KXRAIN-26SEP17-{c}")
        for c in ("NYC", "LAX", "MIA", "CHI", "ATL", "DEN")
    ]
    # Page 1 lands 3 market-days, page 2 raises: same date, partially written.
    client = FakeClient(universe, page_size=3, error_on_page=2)
    first = await make_sweep(client, tmp_path).run_once()
    assert first.n_written == 3
    assert first.error

    second = await make_sweep(FakeClient(universe, page_size=3), tmp_path).run_once()
    assert second.n_written == 3
    assert second.n_already_present == 3

    written = {r["ticker"] for r in rows(tmp_path)}
    assert written == {m["ticker"] for m in universe}


async def test_idempotence_survives_a_mid_day_rename(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    """PROV and PVD are one market-day; the index keys on station."""
    old = make_settled(settled_market, "KXRAIN-26SEP15-PROV")
    new = make_settled(settled_market, "KXRAIN-26SEP15-PVD")
    assert (await make_sweep(FakeClient([old]), tmp_path).run_once()).n_written == 1
    second = await make_sweep(FakeClient([new]), tmp_path).run_once()

    assert second.n_written == 0
    assert second.n_already_present == 1


# The page cap


async def test_the_sweep_is_bounded_to_its_page_cap(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # Following the growing backlog daily in full trips Kalshi's sliding window.
    universe = [
        make_settled(settled_market, f"KXRAIN-26SEP{d:02d}-NYC") for d in range(1, 20)
    ]
    client = FakeClient(universe, page_size=2)
    result = await make_sweep(client, tmp_path).run_once()

    assert len(client.calls) == SETTLEMENT_SWEEP_MAX_PAGES
    assert result.pages_consumed == SETTLEMENT_SWEEP_MAX_PAGES
    assert result.truncated is True
    assert result.n_written == 2 * SETTLEMENT_SWEEP_MAX_PAGES


async def test_a_deeper_backfill_is_opt_in(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    universe = [
        make_settled(settled_market, f"KXRAIN-26SEP{d:02d}-NYC") for d in range(1, 20)
    ]
    client = FakeClient(universe, page_size=2)
    result = await make_sweep(client, tmp_path, max_pages=10).run_once()

    assert len(client.calls) == 10
    assert result.n_written == 19


async def test_the_query_asks_for_settled_and_pins_the_page_size(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    """`status="settled"` is the QUERY; the market's own status is "finalized"."""
    client = FakeClient([settled_market])
    await make_sweep(client, tmp_path).run_once()
    assert client.calls[0]["status"] == "settled"
    assert client.calls[0]["limit"] == 100


# Universe shape and failure handling


async def test_an_unparseable_ticker_is_counted_separately_from_an_unsettled_one(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    universe = [
        settled_market,
        make_settled(settled_market, "KXRAIN-26SEP17-ZZZ"),
        make_settled(settled_market, "KXRAIN-26SEP17-NYC", result=""),
    ]
    result = await make_sweep(FakeClient(universe), tmp_path).run_once()

    assert result.n_written == 1
    assert result.n_unparseable_ticker == 1
    assert result.n_refused_unsettled == 1
    assert result.unparseable_tickers == ["KXRAIN-26SEP17-ZZZ"]


async def test_the_sweep_never_raises_and_reports_the_error(tmp_path: Path) -> None:
    class Exploding:
        async def get_markets(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("no")

    result = await make_sweep(Exploding(), tmp_path).run_once()
    assert result.error
    assert result.n_written == 0
    assert rows(tmp_path) == []


async def test_a_write_failure_does_not_mark_the_market_day_present(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    sweep = make_sweep(FakeClient([settled_market]), tmp_path)

    def boom(batch: list[dict[str, Any]]) -> None:
        raise OSError("disk on fire")

    sweep.writer.write_batch = boom  # type: ignore[method-assign]
    result = await sweep.run_once()
    assert result.n_persisted == 0

    second = await make_sweep(FakeClient([settled_market]), tmp_path).run_once()
    assert second.n_written == 1
    assert second.n_already_present == 0


# The durable index


def test_the_index_round_trips_through_disk(tmp_path: Path) -> None:
    index = SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)
    index.mark(("PHL", date(2026, 9, 17)), ts=NOW.isoformat())
    assert index.save_atomic() is True

    reloaded = SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)
    assert reloaded.contains(("PHL", date(2026, 9, 17))) is True
    assert reloaded.contains(("NYC", date(2026, 9, 17))) is False


def test_a_corrupt_index_starts_empty_rather_than_raising(tmp_path: Path) -> None:
    """A duplicate settlement row is a recorded blip; a crash on a live bot is not."""
    path = tmp_path / SETTLED_INDEX_FILENAME
    path.write_text("{not json at all")
    index = SettledIndex.load(path)
    assert len(index) == 0


def test_the_index_prunes_beyond_the_sweep_horizon(tmp_path: Path) -> None:
    index = SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)
    for day in range(1, 41):
        index.mark(("NYC", date(2026, 8, 1) + timedelta(days=day)), ts=NOW.isoformat())
    dropped = index.prune(keep_dates=30)
    assert dropped == 10
    assert len({key[1] for key in index.keys()}) == 30


# The sweep must never believe a row landed when it did not


async def test_an_oversize_row_is_refused_not_stubbed_and_marked(
    tmp_path: Path, settled_market: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The oversize stub drops station/settlement_date/result — marking the
    # market-day present would lose the outcome permanently.
    import python.data.kxrain_settlements as mod

    monkeypatch.setattr(mod, "encode_row", lambda row: None)
    sweep = make_sweep(FakeClient([settled_market]), tmp_path)
    result = await sweep.run_once()

    assert result.n_written == 0
    assert result.n_refused_oversize == 1
    assert rows(tmp_path) == []

    monkeypatch.undo()
    second = await make_sweep(FakeClient([settled_market]), tmp_path).run_once()
    assert second.n_written == 1
    assert second.n_already_present == 0


async def test_a_torn_write_does_not_mark_the_market_day_present(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # success_count alone is not proof a row reached disk (e.g. a torn ENOSPC write).
    universe = [
        make_settled(settled_market, f"KXRAIN-26SEP17-{c}") for c in ("NYC", "LAX")
    ]
    sweep = make_sweep(FakeClient(universe), tmp_path)
    writer = sweep.writer
    path = tmp_path / KXRAIN_SETTLEMENTS_FILENAME

    def short_write(batch: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(batch[0])[:40])  # torn, no newline
        writer.success_count += len(batch)

    writer.write_batch = short_write  # type: ignore[method-assign]
    result = await sweep.run_once()

    assert result.n_persisted == 0
    second = await make_sweep(FakeClient(universe), tmp_path).run_once()
    assert second.n_already_present == 0
    assert second.n_written == 2


async def test_a_daily_sweep_cannot_prune_away_a_backfills_marks(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # Count-based pruning let two callers destroy each other's marks.
    from datetime import date as _date

    universe = []
    for i in range(37):
        day = _date(2026, 9, 17) - timedelta(days=i)
        universe.append(
            make_settled(settled_market, f"KXRAIN-{day.strftime('%y%b%d').upper()}-NYC")
        )
    deep = {"max_pages": 40, "keep_dates": 40}
    assert (await make_sweep(FakeClient(universe, page_size=2), tmp_path, **deep)
            .run_once()).n_written == 37

    tomorrow = [make_settled(settled_market, "KXRAIN-26SEP18-NYC"), *universe]
    await make_sweep(FakeClient(tomorrow, page_size=2), tmp_path).run_once()

    again = await make_sweep(FakeClient(tomorrow, page_size=2), tmp_path, **deep).run_once()
    assert again.n_written == 0

    keys = [(r["station"], r["settlement_date"]) for r in rows(tmp_path)]
    assert len(keys) == len(set(keys))


async def test_a_sweep_that_returns_nothing_is_flagged_as_an_anomaly(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # The daily universe is >=27 market-days, so an empty listing is never legitimately healthy.
    empty = await make_sweep(FakeClient([]), tmp_path).run_once()
    assert empty.anomaly is True
    assert empty.n_returned == 0

    healthy = await make_sweep(FakeClient([settled_market]), tmp_path).run_once()
    assert healthy.anomaly is False


async def test_an_unfinalized_market_with_a_result_is_refused(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # Gating on result alone would let a pre-finalization result fabricate a NO win.
    early = make_settled(settled_market, "KXRAIN-26SEP19-NYC", result="no")
    early["status"] = "active"
    early["settlement_ts"] = None
    result = await make_sweep(FakeClient([early]), tmp_path).run_once()

    assert result.n_written == 0
    assert result.n_refused_not_finalized == 1
    assert rows(tmp_path) == []


async def test_a_rule_inconsistent_row_stays_retryable(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    liar = make_settled(
        settled_market, "KXRAIN-26SEP19-NYC", result="no", expiration_value="0.42"
    )
    assert (await make_sweep(FakeClient([liar]), tmp_path).run_once()).n_written == 1

    fixed = make_settled(
        settled_market, "KXRAIN-26SEP19-NYC", result="yes", expiration_value="0.42"
    )
    second = await make_sweep(FakeClient([fixed]), tmp_path).run_once()
    assert second.n_written == 1
    assert second.n_already_present == 0
    assert rows(tmp_path)[-1]["rule_consistent"] is True


async def test_the_index_is_merged_not_clobbered_on_save(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # A stale-loaded sweep must not clobber another writer's marks on save.
    from python.data.kxrain_settlements import SettledIndex

    slow = make_sweep(FakeClient([settled_market]), tmp_path)
    stale_index = SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)

    other = make_settled(settled_market, "KXRAIN-26SEP17-LAX")
    await make_sweep(FakeClient([other]), tmp_path).run_once()

    stale_index.mark(("PHL", date(2026, 9, 17)), ts=NOW.isoformat())
    assert stale_index.save_atomic() is True

    reloaded = SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)
    assert reloaded.contains(("PHL", date(2026, 9, 17))) is True
    assert reloaded.contains(("LAX", date(2026, 9, 17))) is True
    assert slow is not None


async def test_the_walk_stops_when_the_shared_backoff_goes_active(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    # A recovery walk must not deepen a live 429 cooldown that live orders queue behind.
    universe = [
        make_settled(settled_market, f"KXRAIN-26SEP17-{c}")
        for c in ("NYC", "LAX", "MIA", "CHI")
    ]

    class Throttled(FakeClient):
        def __init__(self, markets: list[dict[str, Any]]) -> None:
            super().__init__(markets, page_size=1)
            self.gate = [0.0, 0.0, 45.0]

        def backoff_remaining_seconds(self) -> float:
            return self.gate.pop(0) if self.gate else 45.0

    client = Throttled(universe)
    result = await make_sweep(client, tmp_path, max_pages=10).run_once()

    assert result.backoff_aborted is True
    assert result.anomaly is True
    assert len(client.calls) == 2
    assert result.n_written == 2  # what it did fetch is still recorded
    assert len(rows(tmp_path)) == 2

    second = await make_sweep(FakeClient(universe), tmp_path).run_once()
    assert second.n_written == 2
    assert second.n_already_present == 2


async def test_a_client_without_the_backoff_accessor_still_sweeps(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    result = await make_sweep(FakeClient([settled_market]), tmp_path).run_once()
    assert result.backoff_aborted is False
    assert result.n_written == 1


# The backfill script


def test_the_backfill_script_reports_every_field_it_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A bad field reference here passes ruff/mypy but dies at runtime.
    import importlib

    script = importlib.import_module("scripts.kxrain_backfill_settlements")
    from python.data.kxrain_settlements import SettlementSweepResult

    result = SettlementSweepResult(
        n_returned=5,
        n_written=4,
        n_persisted=3,
        n_already_present=1,
        n_refused_unsettled=1,
        n_refused_not_finalized=1,
        n_refused_oversize=1,
        n_unparseable_ticker=1,
        n_rule_inconsistent=1,
        unparseable_tickers=["KXRAIN-26SEP17-ZZZ"],
        pages_consumed=2,
        truncated=True,
        cycle_aborted=True,
        backoff_aborted=True,
        index_saved=False,
        error="RuntimeError: boom",
    )
    script.print_result(result)
    out = capsys.readouterr().out
    for fragment in (
        "CURSOR CYCLE",
        "BACKOFF",
        "TRUNCATED",
        "UNPARSEABLE",
        "RULE-INCONSISTENT",
        "ONLY 3 OF 4",
        "INDEX COULD NOT BE SAVED",
        "RuntimeError: boom",
    ):
        assert fragment in out


def test_the_backfill_script_exit_code_flags_lost_data() -> None:
    import importlib

    script = importlib.import_module("scripts.kxrain_backfill_settlements")
    from python.data.kxrain_settlements import SettlementSweepResult

    healthy = SettlementSweepResult(n_returned=27, n_written=27, n_persisted=27)
    assert script.exit_code(healthy) == 0
    for bad in (
        SettlementSweepResult(n_returned=0),
        SettlementSweepResult(n_returned=1, n_written=1, n_persisted=0),
        SettlementSweepResult(n_returned=1, cycle_aborted=True),
        SettlementSweepResult(n_returned=1, n_unparseable_ticker=1),
        SettlementSweepResult(n_returned=1, n_rule_inconsistent=1),
        SettlementSweepResult(n_returned=1, index_saved=False),
    ):
        assert script.exit_code(bad) == 1


def test_the_coverage_report_distinguishes_a_lost_index_from_a_lost_corpus(
    tmp_path: Path, settled_market: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib

    script = importlib.import_module("scripts.kxrain_backfill_settlements")
    (tmp_path / KXRAIN_SETTLEMENTS_FILENAME).write_text(
        json.dumps({"station": "PHL", "settlement_date": "2026-09-17"}) + "\n"
    )
    script.report_coverage(tmp_path)
    out = capsys.readouterr().out
    assert "market-days on disk:  1" in out
    assert "NOT in the index" in out


async def test_reconcile_indexes_rows_on_disk_so_the_next_sweep_writes_nothing(
    tmp_path: Path, settled_market: dict[str, Any]
) -> None:
    import importlib

    script = importlib.import_module("scripts.kxrain_backfill_settlements")
    universe = [make_settled(settled_market, f"KXRAIN-26SEP17-{c}") for c in ("NYC", "LAX")]
    await make_sweep(FakeClient(universe), tmp_path).run_once()
    (tmp_path / SETTLED_INDEX_FILENAME).unlink()  # rows on disk, index lost

    assert script.reconcile_index(tmp_path) == 2
    assert script.reconcile_index(tmp_path) == 0
    again = await make_sweep(FakeClient(universe), tmp_path).run_once()
    assert again.n_written == 0
    assert len(rows(tmp_path)) == 2


def test_pruning_survives_the_merge_that_precedes_it(tmp_path: Path) -> None:
    # merge_from_disk re-adds every key on disk, which can silently undo a
    # prune() done before it in the same save.
    from datetime import date as _date

    index = SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)
    for day in range(50):
        index.mark(("NYC", _date(2026, 1, 1) + timedelta(days=day)), ts=NOW.isoformat())
    assert index.save_atomic() is True
    assert len(SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)) == 50

    assert index.save_atomic(keep_dates=10) is True
    assert len(SettledIndex.load(tmp_path / SETTLED_INDEX_FILENAME)) == 10


def test_pruning_still_merges_a_concurrent_writers_marks(tmp_path: Path) -> None:
    from datetime import date as _date

    path = tmp_path / SETTLED_INDEX_FILENAME
    first = SettledIndex.load(path)
    first.mark(("NYC", _date(2026, 6, 1)), ts=NOW.isoformat())
    first.save_atomic()

    stale = SettledIndex.load(path)
    other = SettledIndex.load(path)
    other.mark(("LAX", _date(2026, 6, 1)), ts=NOW.isoformat())
    other.save_atomic()

    stale.mark(("MIA", _date(2026, 6, 1)), ts=NOW.isoformat())
    stale.save_atomic(keep_dates=240)

    reloaded = SettledIndex.load(path)
    assert reloaded.contains(("NYC", _date(2026, 6, 1)))
    assert reloaded.contains(("LAX", _date(2026, 6, 1)))
    assert reloaded.contains(("MIA", _date(2026, 6, 1)))
