from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from python.strategies import favorite_bias as fb_mod

SCAN_TODAY = date(2026, 7, 2)


def _fb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **cfg_over: Any
) -> tuple[fb_mod.FavoriteBiasStrategy, list[dict[str, Any]]]:
    cfg = {
        "enabled": True,
        "isotonic_path": None,
        "climatology_path": None,
        "audit_path": str(tmp_path / "fb_audit.jsonl"),
        **cfg_over,
    }
    strat = fb_mod.FavoriteBiasStrategy(config=cfg, kalshi_client=None, bankroll=10_000.0)
    monkeypatch.setattr(strat, "_g", lambda mid: 0.62)
    rows: list[dict[str, Any]] = []
    monkeypatch.setattr(strat, "_audit_write", lambda row: rows.append(row))
    return strat, rows


def _mkt(yes_bid: float, yes_ask: float) -> fb_mod.FBMarket:
    return fb_mod.FBMarket(
        ticker="KXHIGHTBOS-26JUL03-T99",
        city="BOS",
        var="high",
        strike_type="less",
        floor_strike=None,
        cap_strike=99.0,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        mid=(yes_bid + yes_ask) / 2.0,
        volume_fp=300,
        forecast_date=date(2026, 7, 3),
        close_time="2026-07-04T05:00:00Z",
    )


def test_gate_off_by_default_trades_wide_spread(monkeypatch, tmp_path) -> None:
    strat, rows = _fb(monkeypatch, tmp_path)
    sig = strat._build_signal(_mkt(0.49, 0.56), scan_today=SCAN_TODAY)
    assert sig is not None
    assert rows[-1]["decision"] == "buy"


def test_gate_off_signal_identical_to_explicit_null(monkeypatch, tmp_path) -> None:
    default, _ = _fb(monkeypatch, tmp_path)
    explicit, _ = _fb(monkeypatch, tmp_path, max_spread=None)
    a = default._build_signal(_mkt(0.49, 0.56), scan_today=SCAN_TODAY)
    b = explicit._build_signal(_mkt(0.49, 0.56), scan_today=SCAN_TODAY)
    assert a is not None and b is not None
    assert replace(a, timestamp=b.timestamp) == b


def test_spread_above_cap_rejected(monkeypatch, tmp_path) -> None:
    strat, rows = _fb(monkeypatch, tmp_path, max_spread=0.05)
    sig = strat._build_signal(_mkt(0.49, 0.56), scan_today=SCAN_TODAY)
    assert sig is None
    assert rows[-1]["decision"] == "skip_spread"


def test_spread_equal_to_cap_passes(monkeypatch, tmp_path) -> None:
    assert 0.55 - 0.50 > 0.05  # float subtraction overshoots; the epsilon must absorb it
    strat, rows = _fb(monkeypatch, tmp_path, max_spread=0.05)
    sig = strat._build_signal(_mkt(0.50, 0.55), scan_today=SCAN_TODAY)
    assert sig is not None
    assert rows[-1]["decision"] == "buy"


def test_edge_skip_keeps_skip_label_when_spread_wide(monkeypatch, tmp_path) -> None:
    strat, rows = _fb(monkeypatch, tmp_path, max_spread=0.05)
    monkeypatch.setattr(strat, "_g", lambda mid: 0.50)
    assert strat._build_signal(_mkt(0.49, 0.56), scan_today=SCAN_TODAY) is None
    assert rows[-1]["decision"] == "skip"
