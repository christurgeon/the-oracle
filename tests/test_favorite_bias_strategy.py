"""Tests for FavoriteBiasStrategy (Task 3).

Tests call _g() which requires the Task 2 artifact at
data/calibration/favorite_bias_isotonic.json.
"""

from __future__ import annotations

import asyncio
from datetime import date

from python.common import Side
from python.strategies.favorite_bias import FavoriteBiasStrategy, _lead_days, _parse_market


def _cfg(**kw):
    base = dict(
        enabled=True,
        band_low=0.40,
        band_high=0.75,
        min_volume_fp=200,
        edge_floor=0.02,
        lead=1,
        strike_types=["greater", "less"],
        isotonic_path="data/calibration/favorite_bias_isotonic.json",
        kelly_fraction=0.05,
        max_pct=0.05,
        max_contracts_per_trade=25,
        max_bets_per_day_per_event=1,
        audit_path="/tmp/fb_audit.jsonl",
        execution_taker=True,
        cities=["NYC"],
    )
    base.update(kw)
    return base


def test_lead_days_from_ticker():
    # KXHIGHNY-26MAY20-T96 forecast 2026-05-20; scanning on 2026-05-19 => lead 1
    assert _lead_days("KXHIGHNY-26MAY20-T96", date(2026, 5, 19)) == 1
    assert _lead_days("KXHIGHNY-26MAY20-T96", date(2026, 5, 20)) == 0


def test_band_gate_rejects_outside_band():
    s = FavoriteBiasStrategy(config=_cfg(), kalshi_client=None)
    # mid 0.20 (longshot) -> not a candidate
    assert s._passes_band(0.20) is False
    assert s._passes_band(0.55) is True
    assert s._passes_band(0.80) is False


def test_evaluate_returns_g_of_mid_monotone():
    s = FavoriteBiasStrategy(config=_cfg(), kalshi_client=None)
    g_low = s._g(0.45)
    g_hi = s._g(0.65)
    assert 0.0 <= g_low <= 1.0 and 0.0 <= g_hi <= 1.0
    assert g_hi >= g_low  # monotone


def test_signal_only_yes_side_and_taker_metadata():
    s = FavoriteBiasStrategy(config=_cfg(), kalshi_client=None)
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "city": "NYC",
        "daily_variable": "temperature_2m_max",
        "yes_bid_dollars": 0.53,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
    }
    sig = s._build_signal(_parse_market(m), scan_today=date(2026, 5, 19))
    assert sig is not None
    assert sig.side == Side.YES
    assert sig.exec_price == 0.55
    assert sig.metadata["fill_price"] == 0.55  # taker fill (spec §3.3)
    assert sig.metadata["execution_taker"] is True
    assert sig.effective_edge > 0  # must be set (sort key)
    assert sig.strategy_name == "favorite_bias"


def test_parse_market_rejects_zero_bid():
    """Two-sided quote required: yes_bid_dollars=0.0 must be rejected."""
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "city": "NYC",
        "daily_variable": "temperature_2m_max",
        "yes_bid_dollars": 0.0,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
    }
    assert _parse_market(m) is None


def test_parse_market_rejects_crossed_quotes():
    """Crossed quote (bid >= ask) must be rejected."""
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "city": "NYC",
        "daily_variable": "temperature_2m_max",
        "yes_bid_dollars": 0.60,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
    }
    assert _parse_market(m) is None


def test_parse_market_accepts_valid_two_sided_quote():
    """A valid two-sided quote should parse successfully."""
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "city": "NYC",
        "daily_variable": "temperature_2m_max",
        "yes_bid_dollars": 0.53,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
    }
    fm = _parse_market(m)
    assert fm is not None
    assert abs(fm.mid - 0.54) < 1e-9


# ---- _build_signal gate tests ----


def _valid_market_dict(
    *,
    strike_type: str = "greater",
    volume_fp: int = 500,
    ticker: str = "KXHIGHNY-26MAY20-T60",
) -> dict:
    """Return a live-API-shaped market dict (no `city` / `daily_variable` fields)."""
    return {
        "ticker": ticker,
        "strike_type": strike_type,
        "floor_strike": 60.0,
        "cap_strike": None,
        "yes_bid_dollars": 0.53,
        "yes_ask_dollars": 0.55,
        "volume_fp": volume_fp,
        "expiration_ts": "",
    }


def test_city_filter_rejects_out_of_list():
    """cities=["NYC"] must reject a MIA-ticker market (city derived from ticker)."""
    s = FavoriteBiasStrategy(config=_cfg(cities=["NYC"]), kalshi_client=None)
    # Use a MIA ticker so _parse_market derives city="MIA" from the ticker prefix.
    m = _valid_market_dict(ticker="KXHIGHMIA-26MAY20-T80")
    sig = s._build_signal(_parse_market(m), scan_today=date(2026, 5, 19))
    assert sig is None


def test_city_filter_passes_matching_city():
    """cities=["NYC"] must accept a NYC-ticker market (city derived from ticker)."""
    s = FavoriteBiasStrategy(config=_cfg(cities=["NYC"]), kalshi_client=None)
    m = _valid_market_dict(ticker="KXHIGHNY-26MAY20-T60")
    # city is now derived from the ticker by _parse_market, not from the dict
    fm = _parse_market(m)
    assert fm is not None and fm.city == "NYC"
    # city gate passes, so any None is due to other gates (e.g. edge_floor), not city
    assert not s._cities or fm.city in s._cities


def test_city_filter_empty_allows_all():
    """Omitting 'cities' (empty set) should allow any city."""
    s = FavoriteBiasStrategy(config=_cfg(cities=[]), kalshi_client=None)
    for ticker, expected_city in [
        ("KXHIGHNY-26MAY20-T60", "NYC"),
        ("KXHIGHMIA-26MAY20-T80", "MIA"),
        ("KXHIGHLAX-26MAY20-T70", "LAX"),
        ("KXHIGHCHI-26MAY20-T55", "CHI"),
    ]:
        m = _valid_market_dict(ticker=ticker)
        fm = _parse_market(m)
        assert fm is not None and fm.city == expected_city
        # city gate should not block (fm.city filtering is bypassed)
        assert not s._cities or fm.city in s._cities


def test_rejects_below_min_volume():
    """volume_fp < min_volume_fp must cause _build_signal to return None."""
    s = FavoriteBiasStrategy(config=_cfg(cities=["NYC"]), kalshi_client=None)
    m = _valid_market_dict(volume_fp=10)  # below default min_volume_fp=200
    sig = s._build_signal(_parse_market(m), scan_today=date(2026, 5, 19))
    assert sig is None


def test_rejects_wrong_lead():
    """A market whose forecast_date is 2 days out (lead=2) must be rejected when lead=1."""
    s = FavoriteBiasStrategy(config=_cfg(cities=["NYC"], lead=1), kalshi_client=None)
    # ticker date 2026-05-21, scan_today 2026-05-19 => lead=2, not 1
    m = _valid_market_dict(ticker="KXHIGHNY-26MAY21-T60")
    sig = s._build_signal(_parse_market(m), scan_today=date(2026, 5, 19))
    assert sig is None


def test_rejects_excluded_strike_type():
    """strike_type='between' is not in strike_types=[greater,less] and must be rejected."""
    s = FavoriteBiasStrategy(
        config=_cfg(cities=["NYC"], strike_types=["greater", "less"]),
        kalshi_client=None,
    )
    m = _valid_market_dict(strike_type="between")
    sig = s._build_signal(_parse_market(m), scan_today=date(2026, 5, 19))
    assert sig is None


def test_excluded_city_var_gate_is_the_cause_of_the_drop():
    """The (city,var) gate — not some other filter — is what drops the cell: a
    BOS/high market that trades with no exclusion (default [] no-op) must return
    None once "BOS_high" is excluded. BOS high prefix is KXHIGHTBOS (extra-T quirk)."""
    fm = _parse_market(_valid_market_dict(ticker="KXHIGHTBOS-26MAY20-T80"))
    assert fm is not None and fm.city == "BOS" and fm.var == "high"
    # control: no exclusion configured -> the cell trades
    s_open = FavoriteBiasStrategy(config=_cfg(cities=[]), kalshi_client=None)
    assert s_open._excluded_city_vars == set()
    assert s_open._build_signal(fm, scan_today=date(2026, 5, 19)) is not None
    # excluded -> short-circuited to None
    s_excl = FavoriteBiasStrategy(
        config=_cfg(cities=[], excluded_city_vars=["BOS_high"]), kalshi_client=None
    )
    assert s_excl._build_signal(fm, scan_today=date(2026, 5, 19)) is None


def test_excluded_city_var_is_cell_granular_not_city_or_var_wide():
    """excluded_city_vars=["BOS_high"] blocks ONLY BOS/high — not BOS/low (same
    city, other var) nor NYC/high (other city, same var). The gate keys on the
    exact runtime cell f"{city}_{var}", so every other cell still produces a signal."""
    s = FavoriteBiasStrategy(
        config=_cfg(cities=[], excluded_city_vars=["BOS_high"]),
        kalshi_client=None,
    )
    for ticker, city, var in (
        ("KXLOWTBOS-26MAY20-T50", "BOS", "low"),  # same city, other var
        ("KXHIGHNY-26MAY20-T60", "NYC", "high"),  # other city, same var
    ):
        fm = _parse_market(_valid_market_dict(ticker=ticker))
        assert fm is not None and fm.city == city and fm.var == var
        # not the excluded cell -> gate must not fire -> a signal is produced
        assert s._build_signal(fm, scan_today=date(2026, 5, 19)) is not None


def test_disabled_or_fallback_generate_signals_empty():
    """generate_signals() with enabled=False must return [] (async)."""
    s = FavoriteBiasStrategy(config=_cfg(enabled=False, cities=["NYC"]), kalshi_client=None)

    async def _run() -> list:
        return await s.generate_signals()

    result = asyncio.run(_run())
    assert result == []


def test_mark_unmark_traded():
    """mark_traded adds to _traded_tickers; unmark_traded removes idempotently."""
    s = FavoriteBiasStrategy(config=_cfg(), kalshi_client=None)
    ticker = "KXHIGHNY-26MAY20-T60"
    assert ticker not in s._traded_tickers
    s.mark_traded(ticker)
    assert ticker in s._traded_tickers
    s.unmark_traded(ticker)
    assert ticker not in s._traded_tickers
    # idempotent — should not raise
    s.unmark_traded(ticker)


# ---------------------------------------------------------------------------
# New tests required by the merge-blocking bug fix (live-API shape + sizing)
# ---------------------------------------------------------------------------


def test_parse_market_live_api_shape():
    """_parse_market must derive city and var from the ticker when the live-API
    fields city/daily_variable are absent.  KXHIGHNY-... -> city="NYC", var="high"."""
    # Only the fields present in the live Kalshi /markets payload:
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "yes_bid_dollars": 0.53,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "expiration_ts": 1748822400,
    }
    fm = _parse_market(m)
    assert fm is not None, "_parse_market returned None for a valid live-API-shaped dict"
    assert fm.city == "NYC", f"expected city='NYC', got {fm.city!r}"
    assert fm.var == "high", f"expected var='high', got {fm.var!r}"
    assert abs(fm.mid - 0.54) < 1e-9


def test_parse_market_rejects_non_weather_ticker():
    """_parse_market must return None for a non-weather ticker (e.g. KXINX-...)
    because parse_ticker_city_var returns None for unknown series prefixes."""
    m = {
        "ticker": "KXINX-26MAR20H1600-B5250",
        "yes_bid_dollars": 0.40,
        "yes_ask_dollars": 0.42,
        "volume_fp": 300,
        "strike_type": "greater",
        "floor_strike": 5250.0,
        "cap_strike": None,
        "expiration_ts": 1742493600,
    }
    assert _parse_market(m) is None, "non-weather KXINX ticker must be rejected"


def test_parse_market_handles_live_string_typed_numerics():
    """REGRESSION (2026-05-25 live crash): the real Kalshi /markets API returns
    prices as STRINGS ("0.5600") and strikes as INTs (84); only the snapshot
    collector normalizes them to floats. _parse_market must coerce them, not
    raise TypeError on the `0.0 < yb < ya < 1.0` comparison."""
    # Valid two-sided favorite, live shape: string prices + int strike.
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "yes_bid_dollars": "0.5400",
        "yes_ask_dollars": "0.5600",
        "volume_fp": "500",  # live API may also string-type volume
        "strike_type": "greater",
        "floor_strike": 60,
        "cap_strike": None,
        "expiration_ts": 1748822400,
    }
    fm = _parse_market(m)
    assert fm is not None
    assert isinstance(fm.yes_ask, float) and isinstance(fm.yes_bid, float)
    assert abs(fm.mid - 0.55) < 1e-9
    assert fm.volume_fp == 500
    assert fm.floor_strike == 60.0 and isinstance(fm.floor_strike, float)

    # The exact market that crashed prod: string prices, zero bid, int strike.
    # Must be REJECTED (zero-bid longshot) WITHOUT raising.
    crasher = {
        "ticker": "KXHIGHNY-26MAY26-T84",
        "yes_bid_dollars": "0.0000",
        "yes_ask_dollars": "0.0100",
        "volume_fp": "12",
        "strike_type": "greater",
        "floor_strike": 84,
        "cap_strike": None,
        "expiration_ts": 1748822400,
    }
    assert _parse_market(crasher) is None  # no TypeError, cleanly skipped


def test_sizing_nonzero_at_pilot_bankroll():
    """Spec §3.2 H3(b): at a pilot-representative bankroll (1000.0) a favorite-band
    live-shaped market (mid≈0.55, yes_ask≈0.56, volume≥200, lead-1, greater) must
    produce suggested_contracts >= 1 (guards against a silently-dead pilot)."""
    s = FavoriteBiasStrategy(
        config=_cfg(
            cities=[],  # no city filter — accept any
            band_low=0.40,
            band_high=0.75,
            min_volume_fp=200,
            lead=1,
            strike_types=["greater"],
            kelly_fraction=0.05,
            max_pct=0.05,
            max_contracts_per_trade=25,
            edge_floor=0.01,  # lowered to ensure we don't miss due to rounding
        ),
        kalshi_client=None,
        bankroll=1000.0,
    )
    m = {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "yes_bid_dollars": 0.54,
        "yes_ask_dollars": 0.56,
        "volume_fp": 300,
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "expiration_ts": 1748822400,
    }
    sig = s._build_signal(_parse_market(m), scan_today=date(2026, 5, 19))
    assert sig is not None, (
        "No signal produced at bankroll=1000 for mid≈0.55 favorite market — "
        "check g(mid) artifact, kelly_fraction, or edge_floor"
    )
    assert sig.suggested_contracts >= 1, (
        f"suggested_contracts={sig.suggested_contracts} < 1 at bankroll=1000 "
        "— pilot would be silently dead (spec §3.2 H3(b))"
    )


def _fb_market_dict() -> dict:
    return {
        "ticker": "KXHIGHNY-26MAY20-T60",
        "strike_type": "greater",
        "floor_strike": 60.0,
        "cap_strike": None,
        "city": "NYC",
        "daily_variable": "temperature_2m_max",
        "yes_bid_dollars": 0.53,
        "yes_ask_dollars": 0.55,
        "volume_fp": 500,
    }


def test_signal_metadata_carries_resting_ttl_seconds() -> None:
    s = FavoriteBiasStrategy(config=_cfg(resting_ttl_seconds=600), kalshi_client=None)
    sig = s._build_signal(_parse_market(_fb_market_dict()), scan_today=date(2026, 5, 19))
    assert sig is not None
    assert sig.metadata.get("resting_ttl_seconds") == 600


def test_signal_metadata_resting_ttl_none_by_default() -> None:
    s = FavoriteBiasStrategy(config=_cfg(), kalshi_client=None)  # no override in _cfg
    sig = s._build_signal(_parse_market(_fb_market_dict()), scan_today=date(2026, 5, 19))
    assert sig is not None
    assert sig.metadata.get("resting_ttl_seconds") is None


def test_default_state_paths_follow_settings_not_cwd(tmp_path, monkeypatch):
    """In production cwd is /opt/oracle, so a cwd-relative default hits live state."""
    state_dir = tmp_path / "configured_state"
    monkeypatch.setenv("ORACLE_STATE_DIR", str(state_dir))
    cfg = _cfg()
    del cfg["audit_path"]
    s = FavoriteBiasStrategy(config=cfg, kalshi_client=None)
    assert s._audit_path == state_dir / "favorite_bias_audit.jsonl"
    assert s._trades_log_path == state_dir / "trades.jsonl"
