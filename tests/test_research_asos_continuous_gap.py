"""Unit tests for the ASOS continuous-gap probe pure helpers."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from python.common import Direction  # noqa: E402
from scripts.research_2026_06_10_asos_continuous_gap import (  # noqa: E402
    bracket_prob_shift,
    daily_max_by_lst_day,
    day_clustered_bootstrap_ci,
    decompose_gap,
    loo_month_stable,
)


def test_daily_max_groups_by_lst_day_not_utc():
    # AUS is UTC-6 (CST, no DST). 2026-06-02 04:00 UTC = 2026-06-01 22:00 LST,
    # so it belongs to the 2026-06-01 climate day, not 2026-06-02.
    rows = [
        (datetime(2026, 6, 1, 18, 0, tzinfo=UTC), 95.0),  # 12:00 LST 06-01
        (datetime(2026, 6, 2, 4, 0, tzinfo=UTC), 88.0),   # 22:00 LST 06-01
        (datetime(2026, 6, 2, 18, 0, tzinfo=UTC), 90.0),  # 12:00 LST 06-02
    ]
    out = daily_max_by_lst_day(rows, offset_hours=-6)
    from datetime import date
    assert out[date(2026, 6, 1)] == 95.0  # max(95, 88)
    assert out[date(2026, 6, 2)] == 90.0


def test_decompose_gap_components_sum_to_total():
    # cli=92, h1min=91 (continuous proxy), m_hourly=89 (sampled).
    # between-sample miss b = h1min - m_hourly = 2
    # quantization/continuous residual q = cli - h1min = 1
    # total G = cli - m_hourly = 3 = b + q
    d = decompose_gap(cli_high=92.0, h1min=91.0, m_hourly=89.0)
    assert d["between_sample"] == 2.0
    assert d["quant_residual"] == 1.0
    assert d["total"] == 3.0
    assert d["between_sample"] + d["quant_residual"] == d["total"]


def test_bracket_prob_shift_raises_above_prob_when_m_increases():
    # An ABOVE bracket (floor=90) gets MORE likely when the corrected M is +1.5F
    # higher (the daily max is closer to / past the strike). NOTE: bracket_probability
    # takes a Direction ENUM (not "greater"/"less") and keyword-only floor/cap/M.
    base, corrected, delta = bracket_prob_shift(
        strike_type=Direction.ABOVE, floor=90.0, cap=None,
        M=88.0, mu_gain=1.0, sigma_gain=1.0, nu=5.0, m_correction=1.5,
    )
    assert 0.0 <= base <= 1.0 and 0.0 <= corrected <= 1.0
    assert corrected > base          # more YES mass for an ABOVE strike
    assert delta == corrected - base


def test_bootstrap_ci_collapses_to_point_when_all_equal():
    # Every city-day has G=1.0 -> every resample mean is 1.0 -> CI is degenerate.
    lo, hi = day_clustered_bootstrap_ci([1.0] * 40, n_resamples=200, seed=7)
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_brackets_the_mean_with_spread():
    vals = [0.0, 0.5, 1.0, 1.5, 2.0] * 8  # mean 1.0, real spread
    lo, hi = day_clustered_bootstrap_ci(vals, n_resamples=500, seed=7)
    assert lo < 1.0 < hi
    assert lo >= 0.0  # sample is non-negative; CI of the mean stays in-range-ish


def test_loo_month_stable_true_when_all_months_clear_threshold():
    # Both months have G well above 0.5 -> dropping either keeps mean-0.5 sign.
    assert loo_month_stable([(6, 2.0), (6, 2.0), (7, 2.0), (7, 2.0)], threshold=0.5)


def test_loo_month_unstable_when_one_month_dominates():
    # Full mean 2.0 > 0.5, but dropping June leaves July's -1.0 < 0.5 -> sign flips.
    assert not loo_month_stable([(6, 5.0), (6, 5.0), (7, -1.0), (7, -1.0)], threshold=0.5)


def test_loo_month_unstable_with_single_month():
    # Can't assess stability across months with only one month -> not stable.
    assert not loo_month_stable([(6, 2.0), (6, 2.0)], threshold=0.5)


def test_probe_offsets_match_runtime_asos():
    from python.data.asos import CITY_STD_OFFSET_HOURS as RUNTIME
    from scripts.research_2026_06_10_asos_continuous_gap import (
        CITY_STD_OFFSET_HOURS as PROBE,
    )
    # The probe must frame the LST climate day identically to production, or
    # every H1min / M_hourly is computed against the wrong day boundary.
    for city, off in PROBE.items():
        assert RUNTIME[city] == off, (
            f"{city}: probe offset {off} != runtime {RUNTIME[city]}"
        )
