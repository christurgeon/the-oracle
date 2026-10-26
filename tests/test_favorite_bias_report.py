"""Tests for scripts/favorite_bias_report.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.favorite_bias_report import (
    CalRow,
    favorite_bias_pnl,
    filter_live_config,
    filter_live_config_settlement_rows,
    kill_criteria_verdict,
    load_buy_rows,
    load_settled_results,
    material_gap_threshold,
    outcome_source_summary,
    realized_vs_g,
    tripwire_verdict,
    unresolved_buy_counts,
)


def _fb_settled_row(
    *, won: bool, order_price: float, contracts: int = 10,
    mid: float | None = None, map_version: str | None = None,
    settlement_source: str | None = None,
) -> dict:
    """Minimal favorite_bias settled row in shadow_settlements.full.json shape."""
    pnl = (1.0 - order_price) * contracts if won else -(order_price * contracts)
    row = {
        "strategy": "favorite_bias",
        "settled": True,
        "side": "YES",
        "contracts": contracts,
        "model_probability": 0.55,
        "order_price": order_price,
        "exec_price": order_price,
        "pnl": round(pnl, 2),
        "fee": 0.0,
        "won": won,
        "market_ticker": "KXHIGHNY-26MAY27-T84",
    }
    if mid is not None:
        row["metadata"] = {"mid": mid, "map_version": map_version}
    if settlement_source is not None:
        row["settlement_source"] = settlement_source
    return row


def test_kill_criteria_verdict_matrix() -> None:
    # win-rate criterion armed only at n>=40
    assert kill_criteria_verdict(win_rate_pct=50.0, roi_pct=5.0, n=40) == "DISABLE"
    assert kill_criteria_verdict(win_rate_pct=60.0, roi_pct=5.0, n=40) == "OK"
    assert kill_criteria_verdict(win_rate_pct=50.0, roi_pct=5.0, n=39) == "INSUFFICIENT_N"
    # ROI floor fires at ANY n>0, BEFORE the n>=40 gate
    assert kill_criteria_verdict(win_rate_pct=80.0, roi_pct=-15.0, n=5) == "DISABLE"
    assert kill_criteria_verdict(win_rate_pct=80.0, roi_pct=-12.0, n=100) == "DISABLE"
    # healthy
    assert kill_criteria_verdict(win_rate_pct=80.0, roi_pct=5.0, n=50) == "OK"
    # zero data
    assert kill_criteria_verdict(win_rate_pct=None, roi_pct=None, n=0) == "INSUFFICIENT_N"


def test_kill_criteria_verdict_none_guards_no_typeerror() -> None:
    """None rates must not crash the comparison (settled-but-zero-wagered edge)."""
    assert kill_criteria_verdict(win_rate_pct=None, roi_pct=None, n=0) == "INSUFFICIENT_N"
    assert kill_criteria_verdict(win_rate_pct=80.0, roi_pct=None, n=50) == "OK"


def test_favorite_bias_pnl_filters_and_summarizes(tmp_path) -> None:
    rows = [
        _fb_settled_row(won=True, order_price=0.50),
        _fb_settled_row(won=True, order_price=0.50),
        _fb_settled_row(won=False, order_price=0.50),
        # a weather_edge row that must be excluded
        {**_fb_settled_row(won=True, order_price=0.50), "strategy": "weather_edge"},
    ]
    p = tmp_path / "settlements.full.json"
    p.write_text(json.dumps(rows))
    res = favorite_bias_pnl(p)
    assert res["n"] == 3
    assert res["wins"] == 2
    assert res["losses"] == 1
    assert res["win_rate_pct"] == pytest.approx(66.7, abs=0.1)


def test_favorite_bias_pnl_empty_file_returns_none_rates(tmp_path) -> None:
    res = favorite_bias_pnl(tmp_path / "does_not_exist.json")
    assert res["n"] == 0
    assert res["win_rate_pct"] is None
    assert res["roi_pct"] is None


def test_favorite_bias_pnl_without_live_config_is_unfiltered(tmp_path) -> None:
    """Backward compatibility: omitting live_config reproduces today's pooled behavior."""
    rows = [
        _fb_settled_row(won=True, order_price=0.50, mid=0.45, map_version="live"),  # retired band
        _fb_settled_row(won=True, order_price=0.50, mid=0.55, map_version="live"),  # live band
    ]
    p = tmp_path / "settlements.full.json"
    p.write_text(json.dumps(rows))
    res = favorite_bias_pnl(p)
    assert res["n"] == 2
    assert res["n_excluded_retired_config"] == 0
    assert res["n_excluded_no_mid"] == 0


def test_favorite_bias_pnl_applies_band_filter_when_live_config_given(tmp_path) -> None:
    rows = [
        # retired band -> excluded
        _fb_settled_row(won=False, order_price=0.50, mid=0.45, map_version="live"),
        # live band -> kept
        _fb_settled_row(won=True, order_price=0.50, mid=0.55, map_version="live"),
        # no metadata.mid at all -> excluded, counted separately
        {**_fb_settled_row(won=True, order_price=0.50), "market_ticker": "KXHIGHNY-26MAY28-T84"},
    ]
    p = tmp_path / "settlements.full.json"
    p.write_text(json.dumps(rows))
    res = favorite_bias_pnl(p, live_config=(0.50, 0.75))
    assert res["n"] == 1
    assert res["wins"] == 1
    assert res["n_excluded_retired_config"] == 1
    assert res["n_excluded_no_mid"] == 1


def test_favorite_bias_pnl_exclusion_counts_ignore_unsettled_rows(tmp_path) -> None:
    """Regression: exclusion counts must be measured against the SETTLED subset only.

    A prior version of this filter counted n_excluded_no_mid/n_excluded_retired_config
    over ALL favorite_bias rows (settled + open/pending), which double-counted open
    positions that were never part of the unfiltered n in the first place — a settled
    n=192 pooled a retired sub-0.50-band cohort with the live one, and 126 rows (not
    the correct 77) were reported as "out-of-band" because 49 of those 126 were actually
    unsettled rows, not settled ones. Open positions must be silently ignored by both
    counters, exactly as they are ignored by the unfiltered ``n``.
    """
    rows = [
        # settled, retired band -> excluded
        _fb_settled_row(won=False, order_price=0.50, mid=0.45, map_version="live"),
        # settled, live band -> kept
        _fb_settled_row(won=True, order_price=0.50, mid=0.55, map_version="live"),
        # settled, no metadata.mid at all -> excluded, counted separately
        {**_fb_settled_row(won=True, order_price=0.50), "market_ticker": "KXHIGHNY-26MAY28-T84"},
        # unsettled (open position), retired band -> must NOT be counted anywhere
        {
            "strategy": "favorite_bias",
            "settled": False,
            "metadata": {"mid": 0.40, "map_version": "live"},
        },
        # unsettled (open position), no mid at all -> must NOT be counted anywhere
        {"strategy": "favorite_bias", "settled": False, "metadata": {}},
    ]
    p = tmp_path / "settlements.full.json"
    p.write_text(json.dumps(rows))
    res = favorite_bias_pnl(p, live_config=(0.50, 0.75))
    assert res["n"] == 1
    assert res["wins"] == 1
    # Unchanged from the settled-only fixture: the two unsettled rows above are NOT counted.
    assert res["n_excluded_retired_config"] == 1
    assert res["n_excluded_no_mid"] == 1


def test_favorite_bias_pnl_map_version_without_live_config_raises(tmp_path) -> None:
    p = tmp_path / "settlements.full.json"
    p.write_text(json.dumps([]))
    with pytest.raises(ValueError):
        favorite_bias_pnl(p, map_version="live")


def test_favorite_bias_pnl_counts_backfilled_rows(tmp_path) -> None:
    """A row with no settlement_source key must count as 'kalshi' (matching
    the writer's own default in fetch_shadow_settlements.py), not as
    backfilled and not as an error."""
    rows = [
        _fb_settled_row(won=True, order_price=0.50, settlement_source="kalshi"),
        _fb_settled_row(won=True, order_price=0.50, settlement_source="nws_cli_backfill"),
        _fb_settled_row(won=False, order_price=0.50, settlement_source="nws_cli_backfill"),
        _fb_settled_row(won=True, order_price=0.50),  # no key at all -> counts as kalshi
    ]
    p = tmp_path / "settlements.full.json"
    p.write_text(json.dumps(rows))
    res = favorite_bias_pnl(p)
    assert res["n"] == 4
    assert res["n_backfilled"] == 2


def _cal_row(g, outcome, *, mid=0.55, ask=0.57, day="2026-08-01", map_version="live"):
    return CalRow(
        g=g, outcome=outcome, mid=mid, ask=ask, forecast_date=day, map_version=map_version
    )


def test_realized_vs_g_gap() -> None:
    rows = [_cal_row(g, y) for g, y in [(0.60, 1), (0.60, 1), (0.60, 0), (0.70, 1), (0.70, 1)]]
    res = realized_vs_g(rows)
    assert abs(res["mean_g"] - 0.64) < 1e-6
    assert abs(res["realized"] - 0.8) < 1e-6
    assert abs(res["gap"] - (0.8 - 0.64)) < 1e-6
    assert res["n"] == 5


def test_tripwire_requires_a_material_and_confident_overstatement() -> None:
    """Replaces the old bare sign test.

    `WARN if gap < 0` fired ~50% of the time on a perfectly-calibrated map (the
    gap's day-clustered SD is ~0.028) and produced a false demote recommendation
    at gap=-0.0016. WARN now needs the CI upper bound below the material
    threshold, so noise around zero cannot trigger it.
    """
    t = 0.05
    # Confidently worse than the threshold -> WARN.
    assert tripwire_verdict(-0.20, -0.09, 50, 30, t) == "WARN"
    # Confidently better -> OK.
    assert tripwire_verdict(-0.02, +0.10, 50, 30, t) == "OK"
    # Straddles the threshold -> reported, not alerted.
    assert tripwire_verdict(-0.20, +0.05, 50, 30, t) == "INCONCLUSIVE"


def test_tripwire_does_not_warn_on_a_noise_sized_negative_gap() -> None:
    """Regression for the 2026-09-01 false alarm: gap=-0.0016 recommended demote."""
    assert tripwire_verdict(-0.057, +0.053, 247, 89, 0.05) != "WARN"


def test_tripwire_dual_sample_gate() -> None:
    """n alone is not enough: measured flow is ~2.8 markets/day, so n=30 buys
    only ~11 independent day-clusters, not the 30 the old docstring assumed."""
    assert tripwire_verdict(-0.20, -0.09, 29, 30, 0.05) == "INSUFFICIENT_N"
    assert tripwire_verdict(-0.20, -0.09, 50, 19, 0.05) == "INSUFFICIENT_N"
    assert tripwire_verdict(-0.20, -0.09, 30, 20, 0.05) == "WARN"


def test_tripwire_inconclusive_without_a_ci() -> None:
    assert tripwire_verdict(None, None, 50, 30, 0.05) == "INCONCLUSIVE"


def test_material_threshold_tracks_the_books_mean_claimed_edge() -> None:
    """The demote condition is 'overstatement large enough to zero out the book'."""
    rows = [_cal_row(0.70, 1, ask=0.60) for _ in range(10)]  # claimed edge ~0.08 net of fee
    t = material_gap_threshold(rows)
    assert 0.05 < t < 0.10
    # Floor applies when the book's claimed edge is thin.
    thin = [_cal_row(0.60, 1, ask=0.59) for _ in range(10)]
    assert material_gap_threshold(thin) == pytest.approx(0.03)


def test_filter_live_config_drops_retired_band_and_map() -> None:
    """The pooled gap cancelled two opposite cohorts, one no longer traded."""
    rows = [
        _cal_row(0.62, 1, mid=0.45, map_version="live"),  # below band -> drop
        _cal_row(0.62, 1, mid=0.80, map_version="live"),  # at/above band_high -> drop
        _cal_row(0.62, 1, mid=0.55, map_version="old"),  # retired map -> drop
        _cal_row(0.62, 1, mid=0.55, map_version=None),  # unattributable -> drop
        _cal_row(0.62, 1, mid=0.55, map_version="live"),  # keep
    ]
    kept = filter_live_config(rows, map_version="live", band_low=0.50, band_high=0.75)
    assert len(kept) == 1
    assert kept[0].mid == 0.55 and kept[0].map_version == "live"


def test_filter_live_config_without_a_map_hash_still_filters_band() -> None:
    rows = [_cal_row(0.62, 1, mid=0.45), _cal_row(0.62, 1, mid=0.55)]
    kept = filter_live_config(rows, map_version=None, band_low=0.50, band_high=0.75)
    assert [r.mid for r in kept] == [0.55]


def _settlement_dict_row(
    *, mid: float, map_version: str | None, strategy: str = "favorite_bias"
) -> dict:
    return {"strategy": strategy, "metadata": {"mid": mid, "map_version": map_version}}


def test_filter_live_config_settlement_rows_drops_retired_band_and_map() -> None:
    rows = [
        _settlement_dict_row(mid=0.45, map_version="live"),  # below band -> drop
        _settlement_dict_row(mid=0.80, map_version="live"),  # at/above band_high -> drop
        _settlement_dict_row(mid=0.55, map_version="old"),  # retired map -> drop
        _settlement_dict_row(mid=0.55, map_version=None),  # unattributable -> drop
        _settlement_dict_row(mid=0.55, map_version="live"),  # keep
    ]
    kept = filter_live_config_settlement_rows(
        rows, map_version="live", band_low=0.50, band_high=0.75
    )
    assert len(kept) == 1
    assert kept[0]["metadata"]["mid"] == 0.55


def test_filter_live_config_settlement_rows_drops_rows_without_mid() -> None:
    rows = [
        {"strategy": "favorite_bias", "metadata": {}},
        _settlement_dict_row(mid=0.55, map_version="live"),
    ]
    kept = filter_live_config_settlement_rows(
        rows, map_version="live", band_low=0.50, band_high=0.75
    )
    assert len(kept) == 1


def test_filter_live_config_settlement_rows_without_a_map_hash_still_filters_band() -> None:
    rows = [
        _settlement_dict_row(mid=0.45, map_version="live"),
        _settlement_dict_row(mid=0.55, map_version="live"),
    ]
    kept = filter_live_config_settlement_rows(rows, map_version=None, band_low=0.50, band_high=0.75)
    assert [r["metadata"]["mid"] for r in kept] == [0.55]


def test_load_buy_rows_filters_decision_and_missing_settlement() -> None:
    """load_buy_rows skips non-buy rows and rows lacking a settlement."""
    # Build a tiny in-memory settlements dict:
    #   NYC observation_date 2026-05-20 high=85.0
    settlements: dict = {
        ("NYC", "2026-05-20"): {"high_f": 85.0, "low_f": 70.0, "is_preliminary": False},
    }

    # Audit rows:
    #   1. buy row, NYC/high, greater at 83 -> YES (85 > 83)
    #   2. skip row -> must be dropped
    #   3. buy row, CHI/high -> no settlement -> must be dropped
    audit_rows = [
        {
            "ts": "2026-05-19T20:00:00+00:00",
            "market_ticker": "KXHIGHNY-26MAY20-B83",
            "city": "NYC",
            "var": "high",
            "strike_type": "greater",
            "floor_strike": 83.0,
            "cap_strike": None,
            "mid": 0.55,
            "g_of_mid": 0.62,
            "decision": "buy",
            "volume_fp": 500,
        },
        {
            "ts": "2026-05-19T20:00:00+00:00",
            "market_ticker": "KXHIGHNY-26MAY20-B80",
            "city": "NYC",
            "var": "high",
            "strike_type": "greater",
            "floor_strike": 80.0,
            "cap_strike": None,
            "mid": 0.62,
            "g_of_mid": 0.70,
            "decision": "skip",  # must be dropped
            "volume_fp": 500,
        },
        {
            "ts": "2026-05-19T20:00:00+00:00",
            "market_ticker": "KXHIGHCHI-26MAY20-B75",
            "city": "CHI",
            "var": "high",
            "strike_type": "greater",
            "floor_strike": 75.0,
            "cap_strike": None,
            "mid": 0.57,
            "g_of_mid": 0.65,
            "decision": "buy",  # dropped: no CHI settlement
            "volume_fp": 500,
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in audit_rows:
            f.write(json.dumps(row) + "\n")
        audit_path = Path(f.name)

    try:
        result = load_buy_rows(audit_path, settlements)
    finally:
        audit_path.unlink()

    # Only the first row should survive: buy + NYC settlement exists
    assert len(result) == 1
    g_val, yes_outcome = result[0].g, result[0].outcome
    assert abs(g_val - 0.62) < 1e-9
    assert yes_outcome == 1  # 85 > 83 -> YES


def test_load_buy_rows_dedups_rescanned_ticker_keep_first() -> None:
    """The same open market is re-audited every scan tick; load_buy_rows must
    count it ONCE (distinct ticker), keeping the EARLIEST-ts buy decision — the
    scan that actually fired the order (per-day cap + mark_traded). Otherwise a
    persistently-quoted loser dominates the scan-weighted gap (the bug this fixes).
    """
    settlements: dict = {
        ("NYC", "2026-05-20"): {"high_f": 85.0, "low_f": 70.0, "is_preliminary": False},
    }

    def _row(ts: str, mid: float, g: float) -> dict:
        return {
            "ts": ts,
            "market_ticker": "KXHIGHNY-26MAY20-B83",
            "city": "NYC",
            "var": "high",
            "strike_type": "greater",
            "floor_strike": 83.0,
            "cap_strike": None,
            "mid": mid,
            "g_of_mid": g,
            "decision": "buy",
            "volume_fp": 500,
        }

    # Same ticker, three scans with UPWARD-drifting g (convergence). Rows are
    # written out of order to prove load_buy_rows sorts by ts, not file order.
    audit_rows = [
        _row("2026-05-19T22:00:00+00:00", 0.62, 0.70),  # last
        _row("2026-05-19T20:00:00+00:00", 0.52, 0.58),  # FIRST (fired the order)
        _row("2026-05-19T21:00:00+00:00", 0.57, 0.64),  # middle
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in audit_rows:
            f.write(json.dumps(row) + "\n")
        audit_path = Path(f.name)

    try:
        result = load_buy_rows(audit_path, settlements)
    finally:
        audit_path.unlink()

    # Exactly ONE tuple for the re-scanned ticker, carrying the earliest-ts g.
    assert len(result) == 1
    g_val, yes_outcome = result[0].g, result[0].outcome
    assert abs(g_val - 0.58) < 1e-9  # keep-first, NOT 0.70 (keep-last)
    assert yes_outcome == 1  # 85 > 83 -> YES


def test_load_buy_rows_null_ts_does_not_crash_or_win_keep_first() -> None:
    """A row with null/absent ts must not crash the `<` comparison nor spuriously
    win keep-first over a real-ts row (defensive against future/foreign writers)."""
    settlements: dict = {
        ("NYC", "2026-05-20"): {"high_f": 85.0, "low_f": 70.0, "is_preliminary": False},
    }

    def _row(ts, g: float) -> dict:
        row = {
            "market_ticker": "KXHIGHNY-26MAY20-B83",
            "city": "NYC",
            "var": "high",
            "strike_type": "greater",
            "floor_strike": 83.0,
            "cap_strike": None,
            "mid": 0.55,
            "g_of_mid": g,
            "decision": "buy",
            "volume_fp": 500,
        }
        if ts is not None:
            row["ts"] = ts
        return row

    audit_rows = [
        _row(None, 0.99),  # null ts — must NOT win keep-first
        _row("2026-05-19T20:00:00+00:00", 0.58),  # real earliest ts — wins
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in audit_rows:
            f.write(json.dumps(row) + "\n")
        audit_path = Path(f.name)

    try:
        result = load_buy_rows(audit_path, settlements)
    finally:
        audit_path.unlink()

    assert len(result) == 1
    g_val = result[0].g
    assert abs(g_val - 0.58) < 1e-9  # real-ts row wins, not the null-ts g=0.99


def test_load_buy_rows_keeps_distinct_tickers_separate() -> None:
    """Two different brackets in the same event are independent calibration
    points (different mids/g/outcomes) — dedup must NOT collapse them.

    NOTE: this is a post-dedup *correctness* check, not a regression test — it
    passes against the pre-dedup code too (one row per ticker). The regression is
    pinned by ``test_load_buy_rows_dedups_rescanned_ticker_keep_first``.
    """
    settlements: dict = {
        ("NYC", "2026-05-20"): {"high_f": 82.0, "low_f": 70.0, "is_preliminary": False},
    }

    def _row(ticker: str, floor: float, g: float) -> dict:
        return {
            "ts": "2026-05-19T20:00:00+00:00",
            "market_ticker": ticker,
            "city": "NYC",
            "var": "high",
            "strike_type": "greater",
            "floor_strike": floor,
            "cap_strike": None,
            "mid": 0.55,
            "g_of_mid": g,
            "decision": "buy",
            "volume_fp": 500,
        }

    audit_rows = [
        _row("KXHIGHNY-26MAY20-B80", 80.0, 0.60),  # 82 > 80 -> YES
        _row("KXHIGHNY-26MAY20-B84", 84.0, 0.55),  # 82 < 84 -> NO
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in audit_rows:
            f.write(json.dumps(row) + "\n")
        audit_path = Path(f.name)

    try:
        result = load_buy_rows(audit_path, settlements)
    finally:
        audit_path.unlink()

    assert len(result) == 2
    outcomes = sorted(r.outcome for r in result)
    assert outcomes == [0, 1]


def _settle_row(ticker: str, *, secs: float, won: bool, contracts: int, g: float,
                mid: float = 0.55, fill_minus_ask: float = 0.0,
                map_version: str | None = "live") -> dict:
    # Frozen-ask economics: win pays ~ (1 - price) per contract, loss ~ -price.
    price = 0.55
    pnl = contracts * ((1.0 - price) if won else -price)
    return {
        "strategy": "favorite_bias",
        "settled": True,
        "won": won,
        "contracts": contracts,
        "pnl": round(pnl, 2),
        "dollars": round(contracts * price, 2),
        "market_ticker": ticker,
        "seconds_since_placement": secs,
        "fill_price_minus_signal_ask": fill_minus_ask,
        "metadata": {"g_of_mid": g, "mid": mid, "map_version": map_version},
    }


def test_fill_timing_market_dedup_and_patient_classification() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    # ONE market, filled across immediate + patient timings -> exactly ONE PATIENT market
    # (contracts summed), NOT three rows and NOT split across cohorts.
    rows = [
        _settle_row("KXLOWTLAX-26JUL01-T60", secs=5, won=True, contracts=1, g=0.6),
        _settle_row("KXLOWTLAX-26JUL01-T60", secs=90, won=True, contracts=1, g=0.6),
        _settle_row("KXLOWTLAX-26JUL01-T60", secs=200, won=True, contracts=3, g=0.6),
    ]
    out = fill_timing_report(rows)
    assert out["patient"]["n_markets"] == 1
    assert out["patient"]["n_contracts"] == 5
    assert out["immediate"]["n_markets"] == 0
    assert out["verdict"] == "INSUFFICIENT_N"


def test_fill_timing_report_applies_band_filter_when_live_config_given() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    rows: list[dict] = []
    for d in range(1, 25):
        # Live-band immediate markets (kept).
        rows.append(_settle_row(
            f"KXLOWTLAX-26JUL{d:02d}-T60", secs=5, won=True, contracts=3, g=0.55, mid=0.55,
        ))
        # Live-band patient markets (kept).
        rows.append(_settle_row(
            f"KXHIGHNY-26JUL{d:02d}-T80", secs=300, won=True, contracts=3, g=0.55, mid=0.55,
        ))
        # Retired sub-0.50-band markets that must be excluded once live_config is passed.
        rows.append(_settle_row(
            f"KXLOWTPHIL-26JUL{d:02d}-T60", secs=5, won=False, contracts=3, g=0.30, mid=0.30,
        ))

    unfiltered = fill_timing_report(rows)
    filtered = fill_timing_report(rows, live_config=(0.50, 0.75))

    # Unfiltered pools all 3 tickers/day (72 markets); filtered drops the
    # sub-0.50 cohort (24 markets).
    assert unfiltered["immediate"]["n_markets"] + unfiltered["patient"]["n_markets"] == 72
    assert filtered["immediate"]["n_markets"] + filtered["patient"]["n_markets"] == 48


def test_fill_timing_report_map_version_without_live_config_raises() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    with pytest.raises(ValueError):
        fill_timing_report([], map_version="live")


def test_fill_timing_insufficient_n() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    rows = [
        _settle_row("KXLOWTLAX-26JUL01-T60", secs=200, won=True, contracts=3, g=0.6),
        _settle_row("KXHIGHNY-26JUL02-T80", secs=5, won=True, contracts=3, g=0.6),
    ]
    out = fill_timing_report(rows)  # 1 patient market < 20
    assert out["verdict"] == "INSUFFICIENT_N"


def test_fill_timing_no_red_flag_when_patient_matches_immediate() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    # 24 distinct days; each day has ONE immediate-only market (KXLOWT, all fills <=120s)
    # and ONE patient market (KXHIGH, a >120s fill). Identical outcomes -> edge_gap == 0
    # in every bootstrap draw -> CI excludes -3pp -> NO_RED_FLAG.
    rows: list[dict] = []
    for d in range(1, 25):
        rows.append(_settle_row(
            f"KXLOWTLAX-26JUL{d:02d}-T60", secs=5, won=True, contracts=3, g=0.55
        ))
        rows.append(_settle_row(
            f"KXHIGHNY-26JUL{d:02d}-T80", secs=300, won=True, contracts=3, g=0.55
        ))
    out = fill_timing_report(rows)
    assert out["patient"]["n_markets"] >= 20
    assert out["patient"]["n_days"] >= 10
    assert out["immediate"]["n_markets"] >= 20
    assert out["verdict"] == "NO_RED_FLAG"


def test_fill_timing_no_go_when_patient_loses() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    # Immediate markets win, patient markets lose -> edge_gap ~ -1.0 -> CI hi <= -3pp -> NO_GO.
    rows: list[dict] = []
    for d in range(1, 25):
        rows.append(_settle_row(
            f"KXLOWTLAX-26JUL{d:02d}-T60", secs=5, won=True, contracts=3, g=0.55
        ))
        rows.append(_settle_row(
            f"KXHIGHNY-26JUL{d:02d}-T80", secs=300, won=False, contracts=3, g=0.55
        ))
    out = fill_timing_report(rows)
    assert out["verdict"] == "NO_GO"


def test_fill_timing_inconclusive_when_immediate_cohort_empty() -> None:
    from scripts.favorite_bias_report import fill_timing_report

    # 24 distinct-day PATIENT-ONLY markets (every fill >120s) clears the patient
    # sample gate (>=20 markets, >=10 days) but leaves the immediate cohort empty.
    # _edge_gap_day_ci then degenerates on every bootstrap draw (immediate edge is
    # always None -> draw skipped), so it returns (None, None) -> gap_hi is None
    # -> INCONCLUSIVE (the "cannot compute the gap CI yet" branch, distinct from
    # NO_GO/NO_RED_FLAG which both require a computed gap_hi).
    rows: list[dict] = []
    for d in range(1, 25):
        rows.append(_settle_row(
            f"KXHIGHNY-26JUL{d:02d}-T80", secs=300, won=True, contracts=3, g=0.55
        ))
    out = fill_timing_report(rows)
    assert out["immediate"]["n_markets"] == 0
    assert out["patient"]["n_markets"] >= 20
    assert out["patient"]["n_days"] >= 10
    assert out["verdict"] == "INCONCLUSIVE"


def _buy_audit_row(ticker: str, *, ts: str = "2026-09-20T20:00:00+00:00", **over: object) -> dict:
    row = {
        "ts": ts,
        "market_ticker": ticker,
        "city": "NYC",
        "var": "high",
        "strike_type": "greater",
        "floor_strike": 83.0,
        "cap_strike": None,
        "mid": 0.55,
        "yes_ask": 0.57,
        "g_of_mid": 0.62,
        "map_version": "live",
        "decision": "buy",
    }
    row.update(over)
    return row


def _write_audit(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "favorite_bias_audit.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_buy_rows_uses_settled_result_without_a_snapshot(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, [_buy_audit_row("KXHIGHNY-26SEP21-T83")])
    rows = load_buy_rows(audit, {}, {"KXHIGHNY-26SEP21-T83": (0, "kalshi")})
    assert [(r.ticker, r.outcome, r.outcome_source) for r in rows] == [
        ("KXHIGHNY-26SEP21-T83", 0, "kalshi")
    ]


def test_load_buy_rows_settled_result_wins_over_snapshot(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, [_buy_audit_row("KXHIGHNY-26SEP21-T83")])
    snapshots = {("NYC", "2026-09-21"): {"high_f": 85.0, "low_f": 70.0}}
    rows = load_buy_rows(audit, snapshots, {"KXHIGHNY-26SEP21-T83": (0, "kalshi")})
    assert rows[0].outcome == 0
    assert rows[0].outcome_source == "kalshi"


def test_load_buy_rows_falls_back_to_snapshot(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path, [_buy_audit_row("KXHIGHNY-26SEP21-T83")])
    snapshots = {("NYC", "2026-09-21"): {"high_f": 85.0, "low_f": 70.0}}
    rows = load_buy_rows(audit, snapshots, {"KXHIGHCHI-26SEP21-T75": (0, "kalshi")})
    assert rows[0].outcome == 1
    assert rows[0].outcome_source == "nws_snapshot"


def test_load_buy_rows_settled_path_keeps_earliest_ts_and_sorts(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        [
            _buy_audit_row("KXHIGHNY-26SEP22-T83", ts="2026-09-21T21:00:00+00:00"),
            _buy_audit_row("KXHIGHNY-26SEP21-T83", ts="2026-09-20T22:00:00+00:00", g_of_mid=0.70),
            _buy_audit_row("KXHIGHNY-26SEP21-T83", ts="2026-09-20T20:00:00+00:00", g_of_mid=0.62),
        ],
    )
    results = {"KXHIGHNY-26SEP21-T83": (1, "kalshi"), "KXHIGHNY-26SEP22-T83": (1, "kalshi")}
    rows = load_buy_rows(audit, {}, results)
    assert [r.forecast_date for r in rows] == ["2026-09-21", "2026-09-22"]
    assert rows[0].g == pytest.approx(0.62)


def test_load_settled_results_reads_yes_side_result_and_source(tmp_path: Path) -> None:
    path = tmp_path / "full.json"
    path.write_text(
        json.dumps(
            [
                {"market_ticker": "A", "market_result": "yes", "side": "NO"},
                {"market_ticker": "B", "market_result": "no",
                 "settlement_source": "nws_cli_backfill"},
                {"market_ticker": "C", "market_result": None},
                {"market_ticker": "D", "market_result": "yes"},
                {"market_ticker": "D", "market_result": "no"},
            ]
        )
    )
    results, tickers = load_settled_results(path)
    assert results == {"A": (1, "kalshi"), "B": (0, "nws_cli_backfill")}
    assert tickers == {"A", "B", "C", "D"}


def test_load_settled_results_missing_or_malformed_file(tmp_path: Path) -> None:
    assert load_settled_results(tmp_path / "absent.json") == ({}, set())
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_settled_results(bad) == ({}, set())
    bad.write_text(json.dumps({"market_ticker": "A"}))
    assert load_settled_results(bad) == ({}, set())


def test_unresolved_buy_counts_splits_pending_from_absent(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path,
        [
            _buy_audit_row("KXHIGHNY-26SEP21-T83"),
            _buy_audit_row("KXHIGHNY-26SEP22-T83"),
            _buy_audit_row("KXHIGHNY-26SEP23-T83"),
            _buy_audit_row("KXHIGHNY-26SEP24-T83", mid=0.45),
            _buy_audit_row("KXHIGHNY-26SEP25-T83", map_version="old"),
        ],
    )
    counts = unresolved_buy_counts(
        audit,
        {"KXHIGHNY-26SEP21-T83"},
        {"KXHIGHNY-26SEP21-T83", "KXHIGHNY-26SEP22-T83"},
        map_version="live",
        band_low=0.50,
        band_high=0.75,
    )
    assert counts == {"pending": 1, "not_in_settlements": 1}


def test_outcome_source_summary_reports_newest_date_per_source() -> None:
    def row(day: str, source: str) -> CalRow:
        return CalRow(g=0.6, outcome=1, mid=0.55, ask=0.57, forecast_date=day,
                      map_version="live", outcome_source=source)

    summary = outcome_source_summary(
        [
            row("2026-09-01", "nws_snapshot"),
            row("2026-10-05", "kalshi"),
            row("2026-09-30", "kalshi"),
        ]
    )
    assert summary == {
        "nws_snapshot": {"n": 1, "newest_forecast_date": "2026-09-01"},
        "kalshi": {"n": 2, "newest_forecast_date": "2026-10-05"},
    }
