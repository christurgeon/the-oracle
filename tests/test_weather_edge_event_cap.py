"""Integration tests for per-day event cap inside WeatherEdgeStrategy."""

from __future__ import annotations

import json
from pathlib import Path

from python.strategies.per_day_event_cap import PerDayEventCap
from python.utils.jsonl import PerDayCapBlockWriter


def test_audit_stream_writes_when_cap_fires(tmp_path: Path) -> None:
    """Smoke test: cap + writer wired together produce a JSONL row.

    Full strategy-level integration test lives in the
    test_weather_edge_event_cap_integration.py file. This narrow test
    verifies the cap's interaction with the production writer (not the
    fake one used in the unit tests for PerDayEventCap).
    """
    blocks_path = tmp_path / "blocks.jsonl"
    writer = PerDayCapBlockWriter(blocks_path)
    cap = PerDayEventCap(max_per_day=1, audit_writer=writer)
    cap.maybe_reset(utc_today="2026-05-20", hydrate_fn=lambda d: ({}, {}))

    from dataclasses import dataclass, field

    @dataclass
    class _Sig:
        market_ticker: str
        effective_edge: float
        confidence: float
        metadata: dict = field(default_factory=dict)

    def _sig(t: str, **md: object) -> _Sig:
        meta = {
            "city": "NYC",
            "market_type": "high",
            "forecast_date": "2026-05-21",
            "side": "NO",
            "strike_type": "between",
            "floor_strike": 72,
            "cap_strike": 73,
        }
        meta.update(md)
        return _Sig(market_ticker=t, effective_edge=0.10, confidence=0.5, metadata=meta)

    cap.filter_and_select([_sig("KX-A")], scan_id="s1", scan_ts="2026-05-20T14:00Z")
    cap.commit(("NYC", "high", "2026-05-21"), "oid-KX-A")
    cap.filter_and_select([_sig("KX-B")], scan_id="s2", scan_ts="2026-05-20T14:15Z")

    lines = blocks_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["reason"] == "PER_DAY_EVENT_CAP_HIT"
    assert row["blocked_signal"]["market_ticker"] == "KX-B"
