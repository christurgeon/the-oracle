"""Opt-in date-cluster bootstrap for ``bootstrap_cell`` / ``fit_one_cell``."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from python.analytics.emos_fit import (
    BOOTSTRAP_RESAMPLE_DATE_CLUSTER,
    BOOTSTRAP_RESAMPLE_ROW,
    MIN_CLUSTER_DATES,
    bootstrap_cell,
    date_cluster_rows,
    fit_one_cell,
)


def _cell_data(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    m = rng.normal(70, 5, n)
    s = rng.uniform(0.8, 2.0, n)
    o = m * 0.9 + 7 + rng.normal(0, 2, n)
    return m, s, o


def _width(ci: dict[str, object], key: str) -> float:
    pct = ci[key]
    assert isinstance(pct, dict)
    return float(pct["p97.5"] - pct["p2.5"])


def _record_fit_cell_calls(monkeypatch) -> list[np.ndarray]:
    from python.analytics import emos_fit

    seen: list[np.ndarray] = []

    def stub(ens_mean, ens_std, obs):
        seen.append(np.asarray(ens_mean).astype(int))
        return emos_fit.EmosFit(a=0.0, b=1.0, c=1.0, d=1.0)

    monkeypatch.setattr(emos_fit, "fit_cell", stub)
    return seen


@pytest.mark.parametrize("dates_kwarg", [False, True])
def test_bootstrap_cell_default_draws_row_indices_from_seeded_rng(monkeypatch, dates_kwarg):
    n, n_iter, seed = 25, 6, 7
    seen = _record_fit_cell_calls(monkeypatch)
    kwargs = {"dates": None} if dates_kwarg else {}
    got = bootstrap_cell(
        np.arange(n, dtype=float), np.ones(n), np.zeros(n), n_iter=n_iter, seed=seed, **kwargs
    )
    rng = np.random.default_rng(seed)
    expected = [rng.integers(0, n, size=n) for _ in range(n_iter)]
    assert len(seen) == n_iter
    for got_idx, want_idx in zip(seen, expected, strict=True):
        np.testing.assert_array_equal(got_idx, want_idx)
    assert "effective_n_dates" not in got


def test_min_cluster_dates_and_labels():
    assert MIN_CLUSTER_DATES == 10
    assert (BOOTSTRAP_RESAMPLE_ROW, BOOTSTRAP_RESAMPLE_DATE_CLUSTER) == ("row", "date_cluster")


def test_date_cluster_rows_partitions_rows_by_date():
    groups = date_cluster_rows(["b", "a", "b", "c", "a"])
    assert [g.tolist() for g in groups] == [[1, 4], [0, 2], [3]]


@pytest.mark.parametrize(
    "dates",
    [[0.0, float("nan")], [None, "a"], pd.Series([pd.NaT, pd.Timestamp("2026-05-01")]).to_numpy()],
    ids=["nan", "none", "nat"],
)
def test_date_cluster_rows_rejects_nulls(dates):
    with pytest.raises(ValueError, match="dates must not contain nulls"):
        date_cluster_rows(dates)


@pytest.mark.parametrize("dates", [[], ["d", "d"]])
def test_date_cluster_rows_requires_two_dates(dates):
    with pytest.raises(ValueError, match="at least 2 unique"):
        date_cluster_rows(dates)


def test_bootstrap_cell_cluster_is_wider_than_row_when_rows_duplicated_per_date():
    m, s, o = _cell_data(20)
    m, s, o = np.repeat(m, 3), np.repeat(s, 3), np.repeat(o, 3)
    dates = np.repeat(np.arange(20), 3)
    row = bootstrap_cell(m, s, o, n_iter=150, seed=11)
    cluster = bootstrap_cell(m, s, o, n_iter=150, seed=11, dates=dates)
    assert _width(cluster, "a") > _width(row, "a")
    assert _width(cluster, "b") > _width(row, "b")


def test_bootstrap_cell_reports_effective_n_dates():
    m, s, o = _cell_data(30)
    dates = np.repeat(np.arange(10), 3)
    ci = bootstrap_cell(m, s, o, n_iter=5, seed=1, dates=dates)
    assert ci["effective_n_dates"] == 10
    assert isinstance(ci["effective_n_dates"], int)


def test_bootstrap_cell_cluster_accepts_string_dates():
    m, s, o = _cell_data(6)
    dates = np.array(
        ["2026-05-01", "2026-05-01", "2026-05-02", "2026-05-02", "2026-05-03", "2026-05-03"]
    )
    ci = bootstrap_cell(m, s, o, n_iter=5, seed=1, dates=dates)
    assert ci["effective_n_dates"] == 3


def test_bootstrap_cell_cluster_requires_two_unique_dates():
    m, s, o = _cell_data(10)
    with pytest.raises(ValueError, match="at least 2 unique"):
        bootstrap_cell(m, s, o, n_iter=5, seed=1, dates=np.zeros(10, dtype=int))


@pytest.mark.parametrize(
    "dates",
    [
        np.array([0.0, np.nan] * 5),
        np.array([None, "a"] * 5, dtype=object),
    ],
    ids=["nan", "none"],
)
def test_bootstrap_cell_cluster_rejects_null_dates(dates):
    m, s, o = _cell_data(10)
    with pytest.raises(ValueError, match="dates must not contain nulls"):
        bootstrap_cell(m, s, o, n_iter=5, seed=1, dates=dates)


def test_bootstrap_cell_cluster_requires_matching_length():
    m, s, o = _cell_data(10)
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_cell(m, s, o, n_iter=5, seed=1, dates=np.arange(9))


def _cell_df(n_dates: int, rows_per_date: int = 3) -> pd.DataFrame:
    n = n_dates * rows_per_date
    m, s_, o = _cell_data(n)
    return pd.DataFrame(
        {
            "forecast_date": np.repeat(np.arange(n_dates), rows_per_date),
            "lead": np.tile(np.arange(rows_per_date), n_dates),
            "ens_mean": m,
            "ens_std": s_,
            "obs": o,
        }
    )


def test_fit_one_cell_cluster_by_date_lands_in_diag_bootstrap():
    df = _cell_df(30)
    _, diag_row = fit_one_cell(df, seed=3, bootstrap_iters=5)
    _, diag_cluster = fit_one_cell(df, seed=3, bootstrap_iters=5, cluster_by_date=True)
    assert "effective_n_dates" not in diag_row["bootstrap"]
    assert diag_cluster["bootstrap"]["effective_n_dates"] == 30


@pytest.mark.parametrize("n_dates", [1, 5, MIN_CLUSTER_DATES - 1])
def test_fit_one_cell_cluster_skips_cell_with_too_few_dates(n_dates):
    df = _cell_df(n_dates, rows_per_date=40 // n_dates + 1)
    assert len(df) >= 30
    coef, diag = fit_one_cell(df, seed=3, bootstrap_iters=5, cluster_by_date=True)
    assert coef["skipped"] is True
    assert f"{n_dates} unique forecast_dates" in coef["reason"]
    assert diag["skipped"] is True
    assert diag["n_train"] == len(df)


def test_fit_one_cell_row_mode_ignores_date_count():
    coef, _ = fit_one_cell(_cell_df(7, rows_per_date=6), seed=3, bootstrap_iters=5)
    assert "skipped" not in coef


def test_fit_one_cell_cluster_null_date_raises_naming_cell():
    df = _cell_df(12)
    df["forecast_date"] = df["forecast_date"].astype(float)
    df.loc[0, "forecast_date"] = np.nan
    with pytest.raises(ValueError, match="NYC/high"):
        fit_one_cell(df, seed=3, bootstrap_iters=5, cluster_by_date=True, cell="NYC/high")


def test_fit_emos_cluster_by_date_flag_defaults_off(monkeypatch):
    from scripts import fit_emos

    monkeypatch.setattr(sys, "argv", ["fit_emos"])
    assert fit_emos._parse_args().cluster_by_date is False
    monkeypatch.setattr(sys, "argv", ["fit_emos", "--cluster-by-date"])
    assert fit_emos._parse_args().cluster_by_date is True


def _write_corpus(path, one_date_cell_rows: int, many_date_cell_rows: int) -> None:
    rows = []
    for i in range(one_date_cell_rows):
        rows.append(("NYC", "high", pd.Timestamp("2024-01-01").date(), i))
    for i in range(many_date_cell_rows):
        rows.append(("MIA", "high", (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).date(), i))
    pd.DataFrame(
        [
            {
                "forecast_date": d,
                "lead": 0,
                "city": city,
                "var": var,
                "ens_mean": 70.0 + (i % 5),
                "ens_std": 2.0,
                "obs": 70.0 + (i % 7),
                "settlement_status": "final",
            }
            for city, var, d, i in rows
        ]
    ).to_parquet(path)


def _run_fit_emos(monkeypatch, tmp_path, *extra: str, one_date_rows: int = 40) -> tuple[dict, dict]:
    import json

    from scripts import fit_emos

    corpus = tmp_path / "corpus.parquet"
    _write_corpus(corpus, one_date_cell_rows=one_date_rows, many_date_cell_rows=40)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_emos",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(out_dir),
            "--bootstrap-iters",
            "5",
            "--seed",
            "0",
            *extra,
        ],
    )
    fit_emos.main()
    coef = json.loads((out_dir / "coefficients.json").read_text())["cells"]
    diag = json.loads((out_dir / "coefficients.diagnostics.json").read_text())["cells"]
    return coef, diag


def test_fit_emos_cluster_by_date_skips_one_date_cell(monkeypatch, tmp_path, caplog):
    with caplog.at_level("WARNING"):
        coef, diag = _run_fit_emos(monkeypatch, tmp_path, "--cluster-by-date")
    assert coef["NYC/high"]["skipped"] is True
    assert "unique forecast_dates" in coef["NYC/high"]["reason"]
    assert diag["NYC/high"]["skipped"] is True
    assert diag["NYC/high"]["n_train"] == 40
    assert "skipped" not in coef["MIA/high"]
    assert any("skipping NYC/high" in r.getMessage() for r in caplog.records)


def test_fit_emos_cluster_output_loads_in_emos_table(monkeypatch, tmp_path):
    from python.data.emos import EmosTable

    _run_fit_emos(monkeypatch, tmp_path, "--cluster-by-date")
    table = EmosTable.load(tmp_path / "out" / "coefficients.json")
    assert "NYC/high" in table.skipped_cells
    assert "MIA/high" in table.cells


def test_fit_emos_cluster_warns_once_about_row_tuned_thresholds(monkeypatch, tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _run_fit_emos(monkeypatch, tmp_path, "--cluster-by-date")
    msgs = [r.getMessage() for r in caplog.records if "tuned on row-level" in r.getMessage()]
    assert len(msgs) == 1


def test_fit_emos_row_mode_does_not_warn_about_thresholds(monkeypatch, tmp_path, caplog):
    with caplog.at_level("WARNING"):
        _run_fit_emos(monkeypatch, tmp_path, one_date_rows=0)
    assert not any("tuned on row-level" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("flags", "label"),
    [((), BOOTSTRAP_RESAMPLE_ROW), (("--cluster-by-date",), BOOTSTRAP_RESAMPLE_DATE_CLUSTER)],
)
def test_fit_emos_report_header_and_diagnostics_carry_resample_label(
    monkeypatch, tmp_path, capsys, flags, label
):
    import json

    _run_fit_emos(monkeypatch, tmp_path, *flags, one_date_rows=0)
    assert f"bootstrap resample: {label}" in capsys.readouterr().out
    diag_doc = json.loads((tmp_path / "out" / "coefficients.diagnostics.json").read_text())
    assert diag_doc["bootstrap_resample"] == label


def test_fit_emos_help_mentions_min_cluster_dates(monkeypatch, capsys):
    from scripts import fit_emos

    monkeypatch.setattr(sys, "argv", ["fit_emos", "--help"])
    with pytest.raises(SystemExit):
        fit_emos._parse_args()
    out = " ".join(capsys.readouterr().out.split())
    assert f"fewer than {MIN_CLUSTER_DATES} dates are skipped" in out
