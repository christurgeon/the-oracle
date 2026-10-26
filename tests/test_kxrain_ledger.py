"""Unit tests for ``python/data/kxrain_ledger.py``.

Covers fail-open loading (corrupt shapes start empty, never raise) and the
UTC-midnight straddle (pruning by the ledger's own dates, never "today")."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from python.data.kxrain_ledger import (  # noqa: E402
    DEFAULT_KEEP_DATES,
    LEDGER_FILENAME,
    LEDGER_SCHEMA_VERSION,
    KxRainLedger,
    decode_corpus_key,
    encode_corpus_key,
)
from python.data.kxrain_markets import parse_kxrain_ticker  # noqa: E402

MSY = ("MSY", date(2026, 9, 19))
SEA = ("SEA", date(2026, 9, 19))


def _path(tmp_path: Path) -> Path:
    return tmp_path / LEDGER_FILENAME


def _record(ledger: KxRainLedger, key, *, scan_id="scan1", price=0.94) -> bool:
    return ledger.record_entry(
        key,
        decision_id=f"{scan_id}:KXRAIN-26SEP19-{key[0]}",
        ts="2026-09-19T15:00:00+00:00",
        assumed_entry_price=price,
        scan_id=scan_id,
    )


# Key encoding


def test_corpus_key_round_trips():
    assert encode_corpus_key(MSY) == "MSY|2026-09-19"
    assert decode_corpus_key("MSY|2026-09-19") == MSY


def test_decode_rejects_junk():
    assert decode_corpus_key("MSY") is None
    assert decode_corpus_key("MSY|not-a-date") is None
    assert decode_corpus_key("") is None
    assert decode_corpus_key("A|B|C") is None


def test_key_matches_the_market_parser():
    key = parse_kxrain_ticker("KXRAIN-26SEP19-NOLA")
    assert key is not None
    assert encode_corpus_key(key.corpus_key) == "MSY|2026-09-19"


# Round trip


def test_record_then_reload(tmp_path):
    path = _path(tmp_path)
    ledger = KxRainLedger.load(path)
    assert ledger.already_entered(MSY) is False
    assert _record(ledger, MSY) is True
    assert ledger.already_entered(MSY) is True

    reloaded = KxRainLedger.load(path)
    assert reloaded.already_entered(MSY) is True
    entry = reloaded.entry(MSY)
    assert entry is not None
    assert entry.scan_id == "scan1"
    assert entry.assumed_entry_price == pytest.approx(0.94)
    assert entry.decision_id == "scan1:KXRAIN-26SEP19-MSY"


def test_record_entry_is_durable_immediately(tmp_path):
    path = _path(tmp_path)
    ledger = KxRainLedger.load(path)
    assert not path.exists()
    _record(ledger, MSY)
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["_schema_version"] == LEDGER_SCHEMA_VERSION
    assert "MSY|2026-09-19" in on_disk["entries"]


def test_on_disk_shape_matches_the_spec(tmp_path):
    path = _path(tmp_path)
    ledger = KxRainLedger.load(path)
    _record(ledger, MSY)
    on_disk = json.loads(path.read_text())
    assert set(on_disk) == {"_schema_version", "entries"}
    assert set(on_disk["entries"]["MSY|2026-09-19"]) == {
        "decision_id",
        "ts",
        "assumed_entry_price",
        "scan_id",
    }


def test_duplicate_record_keeps_the_first_entry(tmp_path, caplog):
    path = _path(tmp_path)
    ledger = KxRainLedger.load(path)
    _record(ledger, MSY, scan_id="first", price=0.94)
    with caplog.at_level(logging.WARNING):
        assert _record(ledger, MSY, scan_id="second", price=0.50) is False
    entry = ledger.entry(MSY)
    assert entry is not None
    assert entry.scan_id == "first"
    assert any("already" in r.message.lower() for r in caplog.records)


def test_different_stations_same_date_are_independent(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    _record(ledger, MSY)
    assert ledger.already_entered(SEA) is False
    _record(ledger, SEA)
    assert len(ledger) == 2


def test_no_state_file_is_written_until_an_entry_lands(tmp_path):
    path = _path(tmp_path)
    KxRainLedger.load(path)
    assert not path.exists()


# Fail-open loading


def test_missing_file_starts_empty(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    assert len(ledger) == 0
    assert ledger.already_entered(MSY) is False


def test_empty_file_starts_empty_and_warns(tmp_path, caplog):
    path = _path(tmp_path)
    path.write_text("")
    with caplog.at_level(logging.WARNING):
        ledger = KxRainLedger.load(path)
    assert len(ledger) == 0
    assert caplog.records


def test_truncated_json_starts_empty_and_warns(tmp_path, caplog):
    path = _path(tmp_path)
    path.write_text('{"_schema_version": 1, "entries": {"MSY|2026-09-1')
    with caplog.at_level(logging.WARNING):
        ledger = KxRainLedger.load(path)
    assert len(ledger) == 0
    assert caplog.records


def test_wrong_schema_version_starts_empty(tmp_path, caplog):
    path = _path(tmp_path)
    path.write_text(json.dumps({"_schema_version": 99, "entries": {"MSY|2026-09-19": {}}}))
    with caplog.at_level(logging.WARNING):
        ledger = KxRainLedger.load(path)
    assert len(ledger) == 0
    assert caplog.records


def test_top_level_list_starts_empty(tmp_path):
    path = _path(tmp_path)
    path.write_text("[1, 2, 3]")
    assert len(KxRainLedger.load(path)) == 0


def test_entries_not_a_mapping_starts_empty(tmp_path):
    path = _path(tmp_path)
    path.write_text(json.dumps({"_schema_version": 1, "entries": ["MSY|2026-09-19"]}))
    assert len(KxRainLedger.load(path)) == 0


def test_a_single_bad_entry_does_not_discard_the_good_ones(tmp_path, caplog):
    path = _path(tmp_path)
    path.write_text(
        json.dumps(
            {
                "_schema_version": LEDGER_SCHEMA_VERSION,
                "entries": {
                    "MSY|2026-09-19": {
                        "decision_id": "a:b",
                        "ts": "2026-09-19T15:00:00+00:00",
                        "assumed_entry_price": 0.94,
                        "scan_id": "a",
                    },
                    "not-a-key": {"decision_id": "x"},
                    "SEA|2026-09-19": "not-a-dict",
                },
            }
        )
    )
    with caplog.at_level(logging.WARNING):
        ledger = KxRainLedger.load(path)
    assert ledger.already_entered(MSY) is True
    assert ledger.already_entered(SEA) is False
    assert len(ledger) == 1
    assert caplog.records


def test_a_directory_where_the_file_should_be_starts_empty(tmp_path):
    path = _path(tmp_path)
    path.mkdir()
    assert len(KxRainLedger.load(path)) == 0


def test_save_to_an_unwritable_location_returns_false_and_does_not_raise(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    assert ledger.save_atomic(blocked / "sub" / LEDGER_FILENAME) is False


def test_record_entry_keeps_the_entry_in_memory_when_the_save_fails(tmp_path):
    blocked = tmp_path / "wall"
    blocked.write_text("not a directory")
    ledger = KxRainLedger(blocked / "sub" / LEDGER_FILENAME)
    assert _record(ledger, MSY) is False
    assert ledger.already_entered(MSY) is True


# Pruning


def test_prune_keeps_the_three_most_recent_settlement_dates(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    for day in range(14, 20):
        _record(ledger, ("MSY", date(2026, 9, day)))
    assert DEFAULT_KEEP_DATES == 3
    kept = {key[1] for key in ledger.keys()}
    assert kept == {date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 19)}


def test_prune_counts_dates_not_entries(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    for station in ("MSY", "SEA", "MIA", "ORD"):
        for day in (17, 18, 19):
            _record(ledger, (station, date(2026, 9, day)))
    assert len(ledger) == 12
    assert ledger.prune() == 0


def test_prune_returns_the_number_dropped(tmp_path):
    ledger = KxRainLedger(_path(tmp_path))
    for day in range(10, 20):
        ledger.record_entry(
            ("MSY", date(2026, 9, day)),
            decision_id="d",
            ts="t",
            assumed_entry_price=None,
            scan_id="s",
            prune_after=False,
        )
    assert len(ledger) == 10
    assert ledger.prune(keep_dates=2) == 8
    assert len(ledger) == 2


def test_prune_is_a_no_op_on_an_empty_ledger(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    assert ledger.prune() == 0


# UTC-midnight straddle


def test_utc_minus_8_city_is_not_double_entered_across_utc_midnight(tmp_path):
    # SEA's settlement day spans 08:00Z->08:00Z; a 01:00Z restart is a different UTC date.
    path = _path(tmp_path)
    ledger = KxRainLedger.load(path)
    ledger.record_entry(
        SEA,
        decision_id="tick-a:KXRAIN-26SEP19-SEA",
        ts="2026-09-19T23:50:00+00:00",
        assumed_entry_price=0.91,
        scan_id="tick-a",
    )

    after_restart = KxRainLedger.load(path)
    assert after_restart.already_entered(SEA) is True

    after_restart.record_entry(
        ("MSY", date(2026, 9, 20)),
        decision_id="tick-b:KXRAIN-26SEP20-NOLA",
        ts="2026-09-20T01:00:00+00:00",
        assumed_entry_price=0.88,
        scan_id="tick-b",
    )
    assert after_restart.already_entered(SEA) is True
    assert KxRainLedger.load(path).already_entered(SEA) is True


def test_pruning_never_consults_a_clock(tmp_path):
    # Pruning by "today" would drop a UTC-8 city's still-live market-day.
    ledger = KxRainLedger.load(_path(tmp_path))
    for day in (1, 2, 3):
        _record(ledger, ("MSY", date(2020, 1, day)))
    assert len(ledger) == 3


def test_recording_an_out_of_order_date_is_not_evicted_by_its_own_prune(tmp_path):
    path = _path(tmp_path)
    ledger = KxRainLedger.load(path)
    for day in (17, 18, 19):
        ledger.record_entry(
            ("MSY", date(2026, 9, day)),
            decision_id=f"t{day}", ts="x", assumed_entry_price=0.9, scan_id="s",
        )

    old = ("SEA", date(2026, 9, 10))
    assert ledger.record_entry(
        old, decision_id="old", ts="x", assumed_entry_price=0.9, scan_id="s"
    ) is True
    assert ledger.already_entered(old) is True
    assert KxRainLedger.load(path).already_entered(old) is True


def test_prune_protect_keeps_an_older_date(tmp_path):
    ledger = KxRainLedger.load(_path(tmp_path))
    for day in (17, 18, 19, 20):
        ledger.record_entry(
            ("MSY", date(2026, 9, day)),
            decision_id=f"t{day}", ts="x", assumed_entry_price=0.9, scan_id="s",
        )
    protected = ("SEA", date(2026, 9, 1))
    ledger._entries[protected] = ledger._entries[("MSY", date(2026, 9, 20))]
    ledger.prune(keep_dates=2, protect=protected)
    assert protected in ledger._entries
