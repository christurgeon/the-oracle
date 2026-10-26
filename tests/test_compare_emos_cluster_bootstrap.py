"""Opt-in date-cluster resampling for the compare-script bootstrap CIs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from python.analytics.emos_fit import MIN_CLUSTER_DATES
from python.data.emos import EmosCoefficients
from scripts import compare_emos_coefficients as cmp
from scripts.compare_emos_coefficients import (
    _build_stage_b_row,
    _format_markdown_table,
    _parse_args,
    _stage_b_decision,
    bootstrap_bss_ci,
    bootstrap_log_skill_ci,
    decide_cell,
)


def _pair(n: int = 60) -> tuple[list[float], list[float]]:
    g = np.random.default_rng(1)
    return list(g.uniform(0.05, 0.4, n)), list(g.uniform(0.05, 0.4, n))


def _row_level_reference(
    prod: list[float], cand: list[float], n_iters: int, seed: int, ci: float = 0.95
) -> tuple[float, float]:
    p = np.asarray(prod, dtype=float)
    c = np.asarray(cand, dtype=float)
    n = len(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iters, n))
    prod_means = p[idx].mean(axis=1)
    cand_means = c[idx].mean(axis=1)
    skill = np.where(prod_means > 0, 1.0 - cand_means / prod_means, 0.0)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(skill, alpha)), float(np.quantile(skill, 1.0 - alpha))


def test_dates_none_is_bit_identical_to_row_path():
    p, c = _pair()
    expected = _row_level_reference(p, c, n_iters=200, seed=5)
    assert bootstrap_bss_ci(p, c, n_iters=200, seed=5) == expected
    assert bootstrap_log_skill_ci(p, c, n_iters=200, seed=5) == expected
    assert bootstrap_bss_ci(p, c, n_iters=200, seed=5, dates=None) == expected
    assert bootstrap_log_skill_ci(p, c, n_iters=200, seed=5, dates=None) == expected


def test_cluster_ci_is_wider_when_rows_duplicated_per_date():
    p, c = _pair(30)
    p3 = np.repeat(p, 3).tolist()
    c3 = np.repeat(c, 3).tolist()
    dates = np.repeat(np.arange(30), 3).tolist()
    lo_r, hi_r = bootstrap_bss_ci(p3, c3, n_iters=500, seed=2)
    lo_c, hi_c = bootstrap_bss_ci(p3, c3, n_iters=500, seed=2, dates=dates)
    assert (hi_c - lo_c) > (hi_r - lo_r)
    lo_r, hi_r = bootstrap_log_skill_ci(p3, c3, n_iters=500, seed=2)
    lo_c, hi_c = bootstrap_log_skill_ci(p3, c3, n_iters=500, seed=2, dates=dates)
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_pairing_preserved_under_dates():
    p, _ = _pair(30)
    dates = [f"2026-05-{1 + i // 3:02d}" for i in range(30)]
    assert bootstrap_bss_ci(p, list(p), n_iters=100, seed=4, dates=dates) == (0.0, 0.0)
    assert bootstrap_log_skill_ci(p, list(p), n_iters=100, seed=4, dates=dates) == (0.0, 0.0)


def test_bss_and_log_skill_share_resamples_under_dates():
    p, c = _pair(30)
    dates = np.repeat(np.arange(10), 3)
    assert bootstrap_bss_ci(p, c, n_iters=100, seed=9, dates=dates) == bootstrap_log_skill_ci(
        p, c, n_iters=100, seed=9, dates=dates
    )


@pytest.mark.parametrize("fn", [bootstrap_bss_ci, bootstrap_log_skill_ci])
def test_fewer_than_two_unique_dates_raises(fn):
    p, c = _pair(10)
    with pytest.raises(ValueError, match="at least 2 unique"):
        fn(p, c, n_iters=10, seed=0, dates=["d"] * 10)


@pytest.mark.parametrize("fn", [bootstrap_bss_ci, bootstrap_log_skill_ci])
@pytest.mark.parametrize("dates", [[float("nan"), 1.0] * 5, [None, "a"] * 5], ids=["nan", "none"])
def test_null_dates_rejected(fn, dates):
    p, c = _pair(10)
    with pytest.raises(ValueError, match="dates must not contain nulls"):
        fn(p, c, n_iters=10, seed=0, dates=dates)


def test_dates_length_must_match():
    p, c = _pair(10)
    with pytest.raises(ValueError, match="dates"):
        bootstrap_bss_ci(p, c, n_iters=10, seed=0, dates=["a", "b"])


def test_cluster_by_date_flag_defaults_off():
    assert _parse_args([]).cluster_by_date is False
    assert _parse_args(["--cluster-by-date"]).cluster_by_date is True


def test_help_text_mentions_min_cluster_dates(capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert f"fewer than {MIN_CLUSTER_DATES} dates are scored INSUFFICIENT_N" in out


_PASSING = dict(
    n=100,
    n_min=30,
    bss_point=0.2,
    bss_ci_lower=0.1,
    mu_drift=0.1,
    mu_drift_max=1.0,
    has_new_warnings=False,
)
_PASSING_B = dict(
    n=100,
    n_min_iso=10,
    bss_vs_iso_point=0.2,
    bss_vs_iso_ci_lower=0.1,
    has_new_warnings=False,
)


@pytest.mark.parametrize("n_dates", [1, 2, 5, MIN_CLUSTER_DATES - 1])
def test_decide_cell_too_few_dates_is_insufficient_n(n_dates):
    assert decide_cell(**_PASSING, n_dates=n_dates) == "INSUFFICIENT_N"


@pytest.mark.parametrize("n_dates", [None, MIN_CLUSTER_DATES])
def test_decide_cell_enough_or_unspecified_dates_can_promote(n_dates):
    assert decide_cell(**_PASSING, n_dates=n_dates) == "PROMOTE"


@pytest.mark.parametrize("n_dates", [1, 5])
def test_stage_b_decision_too_few_dates_is_insufficient_n(n_dates):
    assert _stage_b_decision(**_PASSING_B, n_dates=n_dates) == "INSUFFICIENT_N"


@pytest.mark.parametrize("n_dates", [None, MIN_CLUSTER_DATES])
def test_stage_b_decision_enough_or_unspecified_dates_can_pass(n_dates):
    assert _stage_b_decision(**_PASSING_B, n_dates=n_dates) == "STAGE_B_ELIGIBLE"


def _stage_b_inputs(n_unique_dates: int, n_rows: int = 12):
    cand_table = {
        "MIA/high": EmosCoefficients(
            a=0.0,
            b=1.0,
            c=0.5,
            d=0.0,
            n_train=145,
            low_n=False,
            ci_warnings=[],
            leads_pooled=[0],
            by_lead=None,
            ens_std_p99=2.0,
        ),
    }
    rows = [
        {
            "city": "MIA",
            "var": "high",
            "strike_type": "between",
            "floor": 88.0,
            "cap": 92.0,
            "ens_mean_raw": 90.0,
            "ens_std_raw": 1.0,
            "observed": 90.0,
            "calibrated_prob": 0.5,
            "forecast_date": f"2026-05-{1 + i % n_unique_dates:02d}",
        }
        for i in range(n_rows)
    ]
    return rows, cand_table


@pytest.mark.parametrize("n_unique_dates", [1, 5])
def test_stage_b_row_cluster_few_dates_is_insufficient_n(monkeypatch, caplog, n_unique_dates):
    def boom(*a, **k):
        raise AssertionError("bootstrap must not run below MIN_CLUSTER_DATES")

    monkeypatch.setattr(cmp, "bootstrap_bss_ci", boom)
    monkeypatch.setattr(cmp, "bootstrap_log_skill_ci", boom)
    rows, cand_table = _stage_b_inputs(n_unique_dates)
    with caplog.at_level("WARNING"):
        row = _build_stage_b_row(
            "MIA/high",
            rows,
            cand_table,
            set(),
            bootstrap_iters=50,
            seed=1,
            cluster_by_date=True,
        )
    assert row is not None
    assert row["stage_b"] == "INSUFFICIENT_N"
    assert row["n_dates"] == n_unique_dates
    assert row["BSS_emos_vs_iso"].startswith("+")
    assert "nan" in row["BSS_emos_vs_iso"]
    assert "nan" in row["LogSkill_emos_vs_iso"]
    assert any("MIA/high" in r.getMessage() for r in caplog.records)
    assert "MIA/high" in _format_markdown_table([row], "t")


def test_stage_b_row_cluster_enough_dates_runs_bootstrap():
    rows, cand_table = _stage_b_inputs(MIN_CLUSTER_DATES, n_rows=MIN_CLUSTER_DATES * 2)
    row = _build_stage_b_row(
        "MIA/high",
        rows,
        cand_table,
        set(),
        bootstrap_iters=50,
        seed=1,
        cluster_by_date=True,
    )
    assert row is not None
    assert row["stage_b"] != "INSUFFICIENT_N"
    assert row["n_dates"] == MIN_CLUSTER_DATES
    assert "nan" not in row["BSS_emos_vs_iso"]


def test_stage_b_row_row_mode_has_no_n_dates_key():
    rows, cand_table = _stage_b_inputs(1)
    row = _build_stage_b_row(
        "MIA/high",
        rows,
        cand_table,
        set(),
        bootstrap_iters=50,
        seed=1,
    )
    assert row is not None
    assert "n_dates" not in row
    assert not math.isnan(float(row["BSS_emos_vs_iso"].split()[0]))


def test_resample_mismatch_warns(caplog):
    with caplog.at_level("WARNING"):
        cmp._warn_resample_mismatch("row", "date_cluster")
    assert any("not like-for-like" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("candidate", [None, "date_cluster"])
def test_resample_match_or_absent_is_silent(caplog, candidate):
    with caplog.at_level("WARNING"):
        cmp._warn_resample_mismatch(candidate, "date_cluster")
    assert not caplog.records
