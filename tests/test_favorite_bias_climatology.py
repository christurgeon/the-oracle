from __future__ import annotations

import json
from datetime import date

from python.strategies import favorite_bias as fb_mod


def _fb(monkeypatch, tmp_path, clim_cells, **cfg_over):
    p = tmp_path / "clim.json"
    p.write_text(json.dumps({"cells": clim_cells}))
    cfg = {
        "enabled": True,
        "climatology_path": str(p),
        "isotonic_path": None,
        "audit_path": str(tmp_path / "fb_audit.jsonl"),  # never touch real state
        **cfg_over,
    }
    strat = fb_mod.FavoriteBiasStrategy(config=cfg, kalshi_client=None, bankroll=10_000.0)
    monkeypatch.setattr(strat, "_g", lambda mid: 0.62)  # deterministic edge
    return strat


def _mkt(strike_type="less", cap=99.0, floor=None, mid=0.48):
    return fb_mod.FBMarket(
        ticker="KXHIGHTBOS-26JUL03-T99",
        city="BOS",
        var="high",
        strike_type=strike_type,
        floor_strike=floor,
        cap_strike=cap,
        yes_bid=mid - 0.02,
        yes_ask=mid + 0.02,
        mid=mid,
        volume_fp=300,
        forecast_date=date(2026, 7, 3),
        close_time="2026-07-04T05:00:00Z",
    )


# ---- Task 6: instrumentation (suppress nothing) ----


def test_less_strike_uses_cap_strike(monkeypatch, tmp_path) -> None:
    strat = _fb(monkeypatch, tmp_path, {"BOS_high_7": {"mu": 85.0, "sigma": 4.0, "n": 29}})
    rows: list[dict] = []
    monkeypatch.setattr(strat, "_audit_write", lambda row: rows.append(row))
    strat._build_signal(_mkt(strike_type="less", cap=99.0), scan_today=date(2026, 7, 2))
    assert rows, "audit must fire for an in-band market"
    r = rows[-1]
    assert abs(r["z_strike"] - 3.5) < 1e-6  # (99 - 85) / 4, from cap_strike not floor
    assert r["clim_source"] == "month"
    assert r["clim_n"] == 29


def test_greater_strike_uses_floor_strike(monkeypatch, tmp_path) -> None:
    strat = _fb(monkeypatch, tmp_path, {"BOS_high_7": {"mu": 85.0, "sigma": 4.0, "n": 29}})
    rows: list[dict] = []
    monkeypatch.setattr(strat, "_audit_write", lambda row: rows.append(row))
    strat._build_signal(
        _mkt(strike_type="greater", cap=None, floor=93.0), scan_today=date(2026, 7, 2)
    )
    assert abs(rows[-1]["z_strike"] - 2.0) < 1e-6  # (93 - 85) / 4, from floor_strike


def test_missing_climatology_logs_null_never_crashes(monkeypatch, tmp_path) -> None:
    strat = _fb(monkeypatch, tmp_path, {})  # empty cells -> disabled-ish lookups
    rows: list[dict] = []
    monkeypatch.setattr(strat, "_audit_write", lambda row: rows.append(row))
    sig = strat._build_signal(_mkt(), scan_today=date(2026, 7, 2))
    r = rows[-1]
    assert r["z_strike"] is None and r["clim_source"] is None
    assert sig is not None  # instrumentation-only: buy still happens (guard unset)
    assert sig.metadata["z_strike"] is None


# ---- Task 7: tail-strike suppression guard (default-off) ----


def test_guard_suppresses_month_tail_when_armed(monkeypatch, tmp_path) -> None:
    strat = _fb(
        monkeypatch,
        tmp_path,
        {"BOS_high_7": {"mu": 85.0, "sigma": 4.0, "n": 29}},
        max_strike_climatology_z=2.0,
        clim_min_n=15,
    )
    rows: list[dict] = []
    monkeypatch.setattr(strat, "_audit_write", lambda row: rows.append(row))
    sig = strat._build_signal(_mkt(cap=99.0), scan_today=date(2026, 7, 2))  # z=3.5 >= 2.0
    assert sig is None
    assert rows[-1]["decision"] == "skip_clim_tail"


def test_guard_off_by_default(monkeypatch, tmp_path) -> None:
    strat = _fb(monkeypatch, tmp_path, {"BOS_high_7": {"mu": 85.0, "sigma": 4.0, "n": 29}})
    sig = strat._build_signal(_mkt(cap=99.0), scan_today=date(2026, 7, 2))
    assert sig is not None  # z=3.5 but guard unset -> trades


def test_guard_fails_open_on_pool_source(monkeypatch, tmp_path) -> None:
    strat = _fb(
        monkeypatch,
        tmp_path,
        {"BOS_high_pool": {"mu": 70.0, "sigma": 3.0, "n": 200}},  # only a pool cell
        max_strike_climatology_z=2.0,
        clim_min_n=15,
    )
    sig = strat._build_signal(_mkt(cap=99.0), scan_today=date(2026, 7, 2))
    assert sig is not None  # source != "month" -> never suppress


def test_guard_fails_open_on_thin_month_cell(monkeypatch, tmp_path) -> None:
    strat = _fb(
        monkeypatch,
        tmp_path,
        {"BOS_high_7": {"mu": 85.0, "sigma": 4.0, "n": 8}},  # n < clim_min_n
        max_strike_climatology_z=2.0,
        clim_min_n=15,
    )
    sig = strat._build_signal(_mkt(cap=99.0), scan_today=date(2026, 7, 2))
    assert sig is not None
