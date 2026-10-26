"""Daily-report boundary-filter kill-watch section (Path 7a, spec §6.5).

Verifies the section renders nothing when disabled, tolerates rows from
before deploy (missing audit fields), shows 'pending' below n=10, and
computes a day-clustered ROI CI with a keep/WATCH verdict above it.
"""

from __future__ import annotations

from scripts.daily_trade_report import render_boundary_filter_section


def _settled_row(*, city, var, pnl, dollars, day, strategy="weather_edge"):
    return {
        "strategy": strategy,
        "settled": True,
        "pnl": pnl,
        "dollars": dollars,
        "timestamp": f"{day}T14:00:00+00:00",
        "metadata": {"city": city, "market_type": var},
    }


def test_render_with_no_cells_configured_returns_empty():
    """Empty config → empty rendering, no crash."""
    assert render_boundary_filter_section([], {}) == ""


def test_render_with_old_rows_lacking_metadata_does_not_crash():
    """Rows missing metadata fall back to ticker parsing; never crash."""
    old_rows = [
        {
            "strategy": "weather_edge",
            "settled": True,
            "pnl": 5.0,
            "dollars": 50.0,
            "timestamp": "2026-05-01T14:00:00+00:00",
            "market_ticker": "KXLOWTMIA-26MAY01-B70-71",
        }
    ]
    out = render_boundary_filter_section(old_rows, {"MIA/low": 0.5})
    # n=1 < 10 → pending; no crash
    assert "pending" in out
    assert "MIA/low" in out


def test_render_with_n_below_10_shows_pending():
    rows = [
        _settled_row(city="MIA", var="low", pnl=1.0, dollars=10.0, day=f"2026-05-0{i}")
        for i in range(1, 6)  # 5 rows
    ]
    out = render_boundary_filter_section(rows, {"MIA/low": 0.5})
    assert "pending (n<10)" in out


def test_render_with_enough_rows_computes_ci_and_verdict():
    # 12 all-winning MIA/low rows across distinct days → ROI strongly positive,
    # CI lower bound > 0 → 'keep'.
    rows = [
        _settled_row(city="MIA", var="low", pnl=20.0, dollars=10.0, day=f"2026-05-{i:02d}")
        for i in range(1, 13)
    ]
    out = render_boundary_filter_section(rows, {"MIA/low": 0.5})
    assert "MIA/low" in out
    assert "pending" not in out
    assert "keep" in out


def test_render_flags_watch_when_ci_lower_negative():
    # Mixed wins/losses with a net-negative skew across days → CI lo < 0 → WATCH.
    rows = []
    for i in range(1, 13):
        pnl = -30.0 if i % 2 == 0 else 5.0
        rows.append(
            _settled_row(city="MIA", var="low", pnl=pnl, dollars=40.0, day=f"2026-05-{i:02d}")
        )
    out = render_boundary_filter_section(rows, {"MIA/low": 0.5})
    assert "WATCH" in out


def test_no_since_flags_contaminated_cohort():
    rows = [
        _settled_row(city="MIA", var="low", pnl=1.0, dollars=10.0, day=f"2026-05-{i:02d}")
        for i in range(1, 13)
    ]
    out = render_boundary_filter_section(rows, {"MIA/low": 0.5})
    assert "NOT post-enablement" in out
    assert "--boundary-since" in out


def test_since_scopes_to_post_enablement_cohort():
    # 6 pre-enablement losers + 12 post-enablement winners. Without --since the
    # cohort mixes both; with --since only the 12 post rows count.
    pre = [
        _settled_row(city="MIA", var="low", pnl=-50.0, dollars=10.0, day=f"2026-04-{i:02d}")
        for i in range(1, 7)
    ]
    post = [
        _settled_row(city="MIA", var="low", pnl=20.0, dollars=10.0, day=f"2026-05-{i:02d}")
        for i in range(1, 13)
    ]
    out = render_boundary_filter_section(pre + post, {"MIA/low": 0.5}, since="2026-05-01")
    assert "trades on/after 2026-05-01" in out
    # n should be 12 (post only), not 18.
    assert " 12 " in out
    assert "keep" in out  # post cohort all winners → CI lo > 0


def test_render_ignores_other_cells_and_strategies():
    rows = [
        _settled_row(city="NYC", var="high", pnl=20.0, dollars=10.0, day=f"2026-05-{i:02d}")
        for i in range(1, 13)
    ] + [
        _settled_row(
            city="MIA", var="low", pnl=20.0, dollars=10.0, day=f"2026-05-{i:02d}",
            strategy="favorite_bias",
        )
        for i in range(1, 13)
    ]
    out = render_boundary_filter_section(rows, {"MIA/low": 0.5})
    # MIA/low has 0 weather_edge settled rows here → pending
    assert "pending (n<10)" in out
