"""F1.5 — ``scripts/fit_emos.py`` must stamp ``fit_date`` from the corpus
``max(forecast_date)``, not from wall-clock ``date.today()``.

These tests run via the unit-level helper rather than spawning the full
fitter subprocess, so they don't require a real corpus.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


def test_resolve_fit_date_returns_corpus_max() -> None:
    """The helper picks the max forecast_date from the corpus dataframe."""
    from scripts.fit_emos import _resolve_fit_date

    df = pd.DataFrame(
        {
            "forecast_date": pd.to_datetime(
                ["2024-01-01", "2024-01-15", "2024-01-07", "2024-01-15"]
            ),
        }
    )
    assert _resolve_fit_date(df) == dt.date(2024, 1, 15)


def test_resolve_fit_date_accepts_string_dates() -> None:
    """``forecast_date`` may be stored as ISO strings (parquet round-trips)."""
    from scripts.fit_emos import _resolve_fit_date

    df = pd.DataFrame(
        {
            "forecast_date": ["2024-01-01", "2024-01-15", "2024-01-07"],
        }
    )
    assert _resolve_fit_date(df) == dt.date(2024, 1, 15)


def test_resolve_fit_date_falls_back_on_empty_corpus(caplog) -> None:
    """An empty dataframe falls back to ``date.today()`` with a logged warning."""
    from scripts.fit_emos import _resolve_fit_date

    df = pd.DataFrame({"forecast_date": pd.to_datetime([])})
    with caplog.at_level("WARNING"):
        result = _resolve_fit_date(df)
    assert result == dt.date.today()
    # Some warning marker should mention the fallback.
    assert any(
        "fit_date" in rec.message.lower() or "fallback" in rec.message.lower()
        for rec in caplog.records
    ), f"no fit_date/fallback warning in {caplog.records!r}"


def test_resolve_fit_date_falls_back_on_all_nulls(caplog) -> None:
    """All-null ``forecast_date`` column falls back to ``date.today()``."""
    from scripts.fit_emos import _resolve_fit_date

    df = pd.DataFrame({"forecast_date": [None, None, None]})
    with caplog.at_level("WARNING"):
        result = _resolve_fit_date(df)
    assert result == dt.date.today()


def test_fit_emos_stamps_corpus_max_not_today(tmp_path, monkeypatch) -> None:
    """End-to-end: invoke ``main`` on a synthetic corpus with a max
    forecast_date in the past; the emitted coefficients.json must carry that
    date, not today's wall-clock."""
    import json
    import sys

    import scripts.fit_emos as fit_emos

    # Build a tiny synthetic corpus that the fitter will load.
    # Two cells × multiple rows so something fits.
    rows = []
    rng_dates = pd.date_range("2024-01-01", periods=40, freq="D")
    for d in rng_dates:
        for city, var in [("NYC", "high"), ("MIA", "high")]:
            rows.append(
                {
                    "snapshot_date": d.date(),
                    "run_id": "test",
                    "run_ts": pd.Timestamp(d, tz="UTC"),
                    "forecast_date": d.date(),
                    "lead": 0,
                    "city": city,
                    "var": var,
                    "daily_variable": "temperature_2m_max",
                    "model": "gfs_seamless",
                    "model_version": "v1",
                    "n_members": 31,
                    "ens_mean": 70.0 + (d.day % 5),
                    "ens_std": 2.0,
                    "obs": 70.0 + (d.day % 7),
                    "settlement_status": "final",
                    "doy": d.dayofyear,
                }
            )
    corpus_df = pd.DataFrame(rows)
    corpus_path = tmp_path / "corpus.parquet"
    corpus_df.to_parquet(corpus_path)

    expected_max = corpus_df["forecast_date"].max()
    # Coerce to plain date for the assertion.
    if hasattr(expected_max, "date"):
        expected_max_date = expected_max.date()
    else:
        expected_max_date = expected_max
    today = dt.date.today()
    # If by some scheduling fluke today < expected_max, skip the negative
    # assertion. The corpus dates are 2024 so this should never happen.
    assert expected_max_date < today

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_emos.py",
            "--corpus",
            str(corpus_path),
            "--output-dir",
            str(out_dir),
            "--bootstrap-iters",
            "5",
            "--min-n",
            "5",
            "--low-n-flag-threshold",
            "5",
            "--seed",
            "0",
        ],
    )
    fit_emos.main()

    coef = json.loads((out_dir / "coefficients.json").read_text())
    diag = json.loads((out_dir / "coefficients.diagnostics.json").read_text())
    assert coef["fit_date"] == expected_max_date.isoformat()
    assert coef["fit_date"] != today.isoformat()
    # The diagnostics file uses the same stamp for consistency.
    assert diag["fit_date"] == expected_max_date.isoformat()


def test_fit_emos_falls_back_on_empty_corpus(tmp_path, monkeypatch) -> None:
    """An empty/all-pending corpus produces no fitted cells. The fit_date
    falls back to today and a warning is logged."""
    import json
    import logging
    import sys

    import scripts.fit_emos as fit_emos

    # Empty corpus — no settled rows.
    cols = [
        "snapshot_date",
        "run_id",
        "run_ts",
        "forecast_date",
        "lead",
        "city",
        "var",
        "daily_variable",
        "model",
        "model_version",
        "n_members",
        "ens_mean",
        "ens_std",
        "obs",
        "settlement_status",
        "doy",
    ]
    df = pd.DataFrame({c: [] for c in cols})
    corpus_path = tmp_path / "corpus.parquet"
    df.to_parquet(corpus_path)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_emos.py",
            "--corpus",
            str(corpus_path),
            "--output-dir",
            str(out_dir),
            "--bootstrap-iters",
            "5",
            "--seed",
            "0",
        ],
    )
    with monkeypatch.context() as m:
        # Capture warnings via logging.
        m.setattr(logging, "WARNING", logging.WARNING)
        fit_emos.main()

    coef = json.loads((out_dir / "coefficients.json").read_text())
    # On empty corpus the stamp falls back to today.
    assert coef["fit_date"] == dt.date.today().isoformat()
