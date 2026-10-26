"""Tests for scripts/build_emos_corpus.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from scripts.build_emos_corpus import (
    _OUTPUT_COLUMNS,
    _build_rows,
    _iter_snapshot_runs,
    _load_all_settlements,
    _load_forecasts_per_run,
    _select_latest_run_per_day,
)


def _write_run(
    run_dir: Path,
    forecasts: list[dict] | None = None,
    settlements: list[dict] | None = None,
) -> None:
    """Write weather_forecasts.json and settlements.json into a run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weather_forecasts.json").write_text(json.dumps(forecasts or []))
    (run_dir / "settlements.json").write_text(json.dumps(settlements or []))


def _build_snapshot_tree(
    snapshots_dir: Path,
    days: dict[str, dict[str, dict] | None],
) -> None:
    """Build a synthetic snapshots tree under ``snapshots_dir``.

    ``days`` maps a snapshot_date string ("2026-04-25") to a dict of
    ``run_id`` -> ``{"forecasts": [...], "settlements": [...]}``.

    For pre-cutover flat layouts (where ``weather_forecasts.json``
    lives directly under the date dir, not in a run subdir), use the
    sentinel run_id ``"_flat_"``. The day_dir itself becomes the
    target.

    Passing ``None`` for a day is a no-op (the date directory is not
    even created); use it only if a test wants the snapshots dir to
    exist without specific day content.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    for date_str, runs in days.items():
        day_dir = snapshots_dir / date_str
        if runs is None:
            continue
        for run_id, payload in runs.items():
            payload = payload or {}
            target = day_dir if run_id == "_flat_" else day_dir / run_id
            _write_run(
                target,
                forecasts=payload.get("forecasts"),
                settlements=payload.get("settlements"),
            )


def test_iter_snapshot_runs_skips_non_dir_siblings(tmp_path: Path) -> None:
    """Spec test #12 — index.jsonl and heal_report_*.csv must be ignored."""

    snaps = tmp_path / "snapshots"
    _build_snapshot_tree(snaps, {"2026-04-25": {"2000": {}}})
    (snaps / "index.jsonl").write_text('{"date":"2026-04-25"}\n')
    (snaps / "heal_report_20260423T165611Z.csv").write_text("col\n")

    runs = list(_iter_snapshot_runs(snaps))
    assert len(runs) == 1
    snapshot_date, run_id, run_dir = runs[0]
    assert str(snapshot_date) == "2026-04-25"
    assert run_id == "2000"
    assert run_dir == snaps / "2026-04-25" / "2000"


def test_iter_snapshot_runs_handles_flat_layout(tmp_path: Path) -> None:
    """Spec test #11 — pre-cutover flat <date>/ becomes run_id='flat'."""

    snaps = tmp_path / "snapshots"
    _build_snapshot_tree(snaps, {"2026-03-20": {"_flat_": {}}})

    runs = list(_iter_snapshot_runs(snaps))
    assert len(runs) == 1
    snapshot_date, run_id, run_dir = runs[0]
    assert str(snapshot_date) == "2026-03-20"
    assert run_id == "flat"
    assert run_dir == snaps / "2026-03-20"


def test_iter_snapshot_runs_nested_wins_over_flat(tmp_path: Path) -> None:
    """Cutover day (e.g. 2026-03-26) has both a flat
    weather_forecasts.json AND a numbered run subdir. The nested
    run is later/newer; it must win. The flat row is suppressed."""

    snaps = tmp_path / "snapshots"
    _build_snapshot_tree(snaps, {"2026-03-26": {"_flat_": {}}})
    _build_snapshot_tree(snaps, {"2026-03-26": {"2000": {}}})

    runs = list(_iter_snapshot_runs(snaps))
    assert len(runs) == 1
    _snap, run_id, run_dir = runs[0]
    assert run_id == "2000"
    assert run_dir == snaps / "2026-03-26" / "2000"


def _make_forecast(
    *,
    city: str = "NYC",
    forecast_date: str = "2026-04-25",
    daily_variable: str = "temperature_2m_max",
    model: str = "gfs_seamless",
    model_version: str = "GFSv16",
    n_members: int = 30,
    base_temp: float = 60.0,
    fetched_at: float = 1777409706.88,
) -> dict:
    """Build a synthetic weather_forecasts.json row."""
    members = [base_temp + 0.1 * i for i in range(n_members)]
    return {
        "city": city,
        "forecast_date": forecast_date,
        "daily_variable": daily_variable,
        "temperatures": members,
        "mean": sum(members) / len(members),
        "std_dev": 1.0,
        "fetched_at": fetched_at,
        "model": model,
        "model_version": model_version,
    }


def test_load_forecasts_drops_below_min_members(tmp_path: Path) -> None:
    """Spec test #3 — n_members=24 drops; n_members=25 keeps."""

    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        forecasts=[
            _make_forecast(city="NYC", n_members=24),  # dropped
            _make_forecast(city="CHI", n_members=25),  # kept
        ],
    )

    rows = _load_forecasts_per_run(run_dir, min_members=25, allowed_models={"gfs_seamless"})
    cities = sorted(r["city"] for r in rows)
    assert cities == ["CHI"]


def test_load_forecasts_filters_models(tmp_path: Path) -> None:
    """Spec test #8 — ECMWF rows dropped when not in allowed_models."""

    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        forecasts=[
            _make_forecast(model="gfs_seamless"),
            _make_forecast(model="ecmwf_ifs025"),
        ],
    )

    gfs_only = _load_forecasts_per_run(run_dir, min_members=25, allowed_models={"gfs_seamless"})
    assert [r["model"] for r in gfs_only] == ["gfs_seamless"]

    both = _load_forecasts_per_run(
        run_dir, min_members=25, allowed_models={"gfs_seamless", "ecmwf_ifs025"}
    )
    assert sorted(r["model"] for r in both) == ["ecmwf_ifs025", "gfs_seamless"]


def test_load_forecasts_skips_unknown_daily_variable(tmp_path: Path) -> None:
    """Forecasts for variables we don't model are silently dropped."""

    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        forecasts=[
            _make_forecast(daily_variable="temperature_2m_max"),
            _make_forecast(daily_variable="precipitation_sum"),  # unsupported
        ],
    )

    rows = _load_forecasts_per_run(run_dir, min_members=25, allowed_models={"gfs_seamless"})
    assert len(rows) == 1
    assert rows[0]["daily_variable"] == "temperature_2m_max"


def test_load_forecasts_handles_malformed_json(tmp_path: Path, caplog) -> None:
    """Spec error-handling: malformed JSON logs a warning and returns
    [] — the corpus build must not abort on a single corrupt file."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "weather_forecasts.json").write_text("{not valid json")

    with caplog.at_level(logging.WARNING, logger="scripts.build_emos_corpus"):
        rows = _load_forecasts_per_run(run_dir, min_members=25, allowed_models={"gfs_seamless"})

    assert rows == []
    assert any("malformed forecast JSON" in rec.message for rec in caplog.records)


def test_latest_run_wins_within_day() -> None:
    """Spec test #1 — same (city, forecast_date, var, model) across runs:
    keep the row from the highest-run_id (lex)."""

    rows = [
        # Same target, two runs, different ens_mean — 2000 should win.
        ("2026-04-25", "1000", _make_forecast(city="NYC", base_temp=60.0)),
        ("2026-04-25", "2000", _make_forecast(city="NYC", base_temp=70.0)),
        # Different city, single run — should pass through unchanged.
        ("2026-04-25", "1500", _make_forecast(city="CHI", base_temp=50.0)),
    ]

    selected = list(_select_latest_run_per_day(rows))
    assert len(selected) == 2

    by_city = {(snap, run, fc["city"]): fc for snap, run, fc in selected}
    nyc_key = next(k for k in by_city if k[2] == "NYC")
    assert nyc_key[1] == "2000"
    assert by_city[nyc_key]["temperatures"][0] == 70.0


def test_latest_run_wins_under_reversed_input_order() -> None:
    """Same as test_latest_run_wins_within_day but with 2000 arriving
    BEFORE 1000. Without this, a naive last-seen algorithm would
    accidentally pass test_latest_run_wins_within_day. Locks the
    actual `>=` semantics on numeric run_ids."""

    rows = [
        ("2026-04-25", "2000", _make_forecast(city="NYC", base_temp=70.0)),
        ("2026-04-25", "1000", _make_forecast(city="NYC", base_temp=60.0)),
    ]

    selected = list(_select_latest_run_per_day(rows))
    assert len(selected) == 1
    _snap, run_id, fc = selected[0]
    assert run_id == "2000"
    assert fc["temperatures"][0] == 70.0


def test_flat_sentinel_loses_to_numeric_run() -> None:
    """Spec test #13 — flat layout never beats a numeric run, even though
    'flat' > '2000' under naive lex ordering."""

    rows = [
        ("2026-04-25", "flat", _make_forecast(city="NYC", base_temp=60.0)),
        ("2026-04-25", "2000", _make_forecast(city="NYC", base_temp=70.0)),
    ]

    selected = list(_select_latest_run_per_day(rows))
    assert len(selected) == 1
    snap, run_id, fc = selected[0]
    assert run_id == "2000"
    assert fc["temperatures"][0] == 70.0


def test_numeric_run_is_not_replaced_by_flat() -> None:
    """Spec test #13 (converse ordering) — flat arriving after a numeric
    run must not overwrite it. Locks the `>=` semantics so a future
    change to `>` would regress here."""

    rows = [
        ("2026-04-25", "2000", _make_forecast(city="NYC", base_temp=70.0)),
        ("2026-04-25", "flat", _make_forecast(city="NYC", base_temp=60.0)),
    ]

    selected = list(_select_latest_run_per_day(rows))
    assert len(selected) == 1
    _, run_id, fc = selected[0]
    assert run_id == "2000"
    assert fc["temperatures"][0] == 70.0


def test_dedup_is_per_snapshot_date() -> None:
    """Different snapshot_dates with the same forecast_target both survive
    — that's the whole point of EMOS keying on (snapshot_date, lead)."""

    rows = [
        ("2026-04-24", "2000", _make_forecast(city="NYC", forecast_date="2026-04-25")),
        ("2026-04-25", "2000", _make_forecast(city="NYC", forecast_date="2026-04-25")),
    ]

    selected = list(_select_latest_run_per_day(rows))
    assert len(selected) == 2
    snap_dates = sorted(s for s, _, _ in selected)
    assert snap_dates == ["2026-04-24", "2026-04-25"]


def _make_settlement(
    *,
    city: str = "NYC",
    cli_location: str | None = None,
    observation_date: str = "2026-04-25",
    high_f: float = 65.0,
    low_f: float = 50.0,
    is_preliminary: bool = False,
    issued_at: str = "2026-04-26T06:20:00+00:00",
    product_id: str = "p-final",
) -> dict:
    return {
        "city": city,
        "cli_location": cli_location or city,
        "observation_date": observation_date,
        "high_f": high_f,
        "low_f": low_f,
        "is_preliminary": is_preliminary,
        "issued_at": issued_at,
        "product_id": product_id,
        "snapshot_ts": 0.0,
    }


def test_settlements_final_supersedes_preliminary(tmp_path: Path) -> None:
    """Spec test #5 — afternoon prelim followed by next-morning final
    → final wins."""

    snaps = tmp_path / "snapshots"
    _build_snapshot_tree(
        snaps,
        {
            "2026-04-25": {
                "2000": {
                    "settlements": [
                        _make_settlement(
                            high_f=64.0,
                            is_preliminary=True,
                            issued_at="2026-04-25T20:34:00+00:00",
                            product_id="p-prelim",
                        ),
                    ],
                },
            },
            "2026-04-26": {
                "0500": {
                    "settlements": [
                        _make_settlement(
                            high_f=68.0,
                            is_preliminary=False,
                            issued_at="2026-04-26T06:20:00+00:00",
                            product_id="p-final",
                        ),
                    ],
                },
            },
        },
    )

    merged = _load_all_settlements(snaps)
    rec = merged[("NYC", "2026-04-25")]
    assert rec["high_f"] == 68.0
    assert rec["is_preliminary"] is False


def test_settlements_only_preliminary_is_kept(tmp_path: Path) -> None:
    """Spec test #6 — if no final ever lands, the preliminary stays
    available (the row builder tags it so the fitter can decide)."""

    snaps = tmp_path / "snapshots"
    _build_snapshot_tree(
        snaps,
        {
            "2026-04-29": {
                "2000": {
                    "settlements": [
                        _make_settlement(
                            observation_date="2026-04-29",
                            high_f=72.0,
                            is_preliminary=True,
                            issued_at="2026-04-29T20:34:00+00:00",
                            product_id="p-prelim-only",
                        ),
                    ],
                },
            },
        },
    )

    merged = _load_all_settlements(snaps)
    assert ("NYC", "2026-04-29") in merged
    rec = merged[("NYC", "2026-04-29")]
    assert rec["is_preliminary"] is True
    assert rec["high_f"] == 72.0


def test_settlements_skips_malformed_json(tmp_path: Path, caplog) -> None:
    """Spec error-handling: malformed settlements.json logs a warning
    and is skipped — the corpus build must not abort on one bad file.
    Mirrors test_load_forecasts_handles_malformed_json for symmetry."""

    snaps = tmp_path / "snapshots"
    day_dir = snaps / "2026-04-25" / "2000"
    day_dir.mkdir(parents=True)
    (day_dir / "weather_forecasts.json").write_text("[]")
    (day_dir / "settlements.json").write_text("{not valid json")

    with caplog.at_level(logging.WARNING, logger="scripts.build_emos_corpus"):
        result = _load_all_settlements(snaps)

    assert result == {}
    assert any("malformed settlement JSON" in rec.message for rec in caplog.records)


def test_row_builder_lead_and_doy() -> None:
    """Spec test #2 — snapshot 2026-04-25 → forecasts for +0/+1/+2
    yield lead 0/1/2, doy 115/116/117 (April 25 = day 115)."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-25")),
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-26")),
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-27")),
    ]
    settlements = {
        ("NYC", "2026-04-25"): _make_settlement(observation_date="2026-04-25"),
        ("NYC", "2026-04-26"): _make_settlement(observation_date="2026-04-26"),
        ("NYC", "2026-04-27"): _make_settlement(observation_date="2026-04-27"),
    }

    rows = list(_build_rows(forecasts, settlements))
    by_fc_date = {r["forecast_date"]: r for r in rows}
    # Pin the per-row pairing so a future refactor that derives doy
    # from snapshot_date (or lead from the wrong delta direction)
    # would fail here, not just on aggregate sort.
    assert by_fc_date["2026-04-25"]["lead"] == 0
    assert by_fc_date["2026-04-25"]["doy"] == 115
    assert by_fc_date["2026-04-26"]["lead"] == 1
    assert by_fc_date["2026-04-26"]["doy"] == 116
    assert by_fc_date["2026-04-27"]["lead"] == 2
    assert by_fc_date["2026-04-27"]["doy"] == 117


def test_row_builder_chi_pre_fix_excluded() -> None:
    """Spec test #4 — CHI forecast for 2026-03-19 dropped; 2026-03-20 kept."""

    forecasts = [
        (
            "2026-03-18",
            "2000",
            _make_forecast(city="CHI", forecast_date="2026-03-19"),
        ),
        (
            "2026-03-19",
            "2000",
            _make_forecast(city="CHI", forecast_date="2026-03-20"),
        ),
    ]
    settlements = {
        ("CHI", "2026-03-19"): _make_settlement(city="CHI", observation_date="2026-03-19"),
        ("CHI", "2026-03-20"): _make_settlement(city="CHI", observation_date="2026-03-20"),
    }

    rows = list(_build_rows(forecasts, settlements))
    assert [r["forecast_date"] for r in rows] == ["2026-03-20"]


def test_row_builder_drops_when_settlement_missing() -> None:
    """Spec test #7 — forecast with no matching settlement → no row."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-29")),  # unsettled
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-25")),  # settled
    ]
    settlements = {
        ("NYC", "2026-04-25"): _make_settlement(observation_date="2026-04-25"),
    }

    rows = list(_build_rows(forecasts, settlements))
    assert [r["forecast_date"] for r in rows] == ["2026-04-25"]


def test_row_builder_recomputes_stats_ignoring_json_mean() -> None:
    """Spec test #9 — ens_mean / ens_std recomputed from members,
    NOT trusted from the JSON 'mean' field."""
    import statistics as stats

    fc = _make_forecast(forecast_date="2026-04-25")
    fc["mean"] = 999.0  # poison the JSON value
    fc["std_dev"] = 999.0
    expected_mean = sum(fc["temperatures"]) / len(fc["temperatures"])
    expected_std = stats.stdev(fc["temperatures"])  # ddof=1

    forecasts = [("2026-04-25", "2000", fc)]
    settlements = {("NYC", "2026-04-25"): _make_settlement(observation_date="2026-04-25")}

    [row] = list(_build_rows(forecasts, settlements))
    assert row["ens_mean"] == pytest.approx(expected_mean)
    assert row["ens_std"] == pytest.approx(expected_std)


def test_row_builder_tags_settlement_status() -> None:
    """Final settlement → 'final'; preliminary settlement → 'preliminary'."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(city="NYC", forecast_date="2026-04-25")),
        ("2026-04-29", "2000", _make_forecast(city="CHI", forecast_date="2026-04-29")),
    ]
    settlements = {
        ("NYC", "2026-04-25"): _make_settlement(
            observation_date="2026-04-25", is_preliminary=False
        ),
        ("CHI", "2026-04-29"): _make_settlement(
            city="CHI", observation_date="2026-04-29", is_preliminary=True
        ),
    }

    rows = sorted(_build_rows(forecasts, settlements), key=lambda r: r["city"])
    assert [r["settlement_status"] for r in rows] == ["preliminary", "final"]


def test_row_builder_tags_null_is_preliminary_as_preliminary() -> None:
    """Defensive default: a settlement record with is_preliminary=None
    (legacy snapshots predating the field) must be tagged 'preliminary',
    not 'final'. `bool(None) is False` would otherwise wrongly classify."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-25")),
    ]
    settle = _make_settlement(observation_date="2026-04-25")
    settle["is_preliminary"] = None
    settlements = {("NYC", "2026-04-25"): settle}

    [row] = list(_build_rows(forecasts, settlements))
    assert row["settlement_status"] == "preliminary"


def test_row_builder_tags_missing_is_preliminary_as_preliminary() -> None:
    """Same defensive default for the field-absent case."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-25")),
    ]
    settle = _make_settlement(observation_date="2026-04-25")
    del settle["is_preliminary"]
    settlements = {("NYC", "2026-04-25"): settle}

    [row] = list(_build_rows(forecasts, settlements))
    assert row["settlement_status"] == "preliminary"


def test_row_builder_obs_field_matches_variable() -> None:
    """high → high_f; low → low_f. var derives from daily_variable."""

    forecasts = [
        (
            "2026-04-25",
            "2000",
            _make_forecast(daily_variable="temperature_2m_max"),
        ),
        (
            "2026-04-25",
            "2000",
            _make_forecast(daily_variable="temperature_2m_min"),
        ),
    ]
    settlements = {
        ("NYC", "2026-04-25"): _make_settlement(
            observation_date="2026-04-25", high_f=68.0, low_f=45.0
        ),
    }

    rows = sorted(_build_rows(forecasts, settlements), key=lambda r: r["var"])
    assert rows[0]["var"] == "high"
    assert rows[0]["obs"] == 68.0
    assert rows[1]["var"] == "low"
    assert rows[1]["obs"] == 45.0


def test_row_builder_skips_negative_lead() -> None:
    """forecast_date < snapshot_date should never happen; if it does,
    skip the row defensively."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-24")),
    ]
    settlements = {
        ("NYC", "2026-04-24"): _make_settlement(observation_date="2026-04-24"),
    }

    rows = list(_build_rows(forecasts, settlements))
    assert rows == []


def test_row_builder_negative_lead_logs_warning(caplog) -> None:
    """Spec — negative lead must log a warning, not silently skip."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-24")),
    ]
    settlements = {
        ("NYC", "2026-04-24"): _make_settlement(observation_date="2026-04-24"),
    }

    with caplog.at_level(logging.WARNING, logger="scripts.build_emos_corpus"):
        list(_build_rows(forecasts, settlements))

    assert any("Negative lead" in rec.message for rec in caplog.records)


def test_row_builder_handles_null_model_version() -> None:
    """JSON `null` for model_version must coerce to empty string,
    not flow through as Python None (which would poison parquet dtype)."""

    fc = _make_forecast(forecast_date="2026-04-25")
    fc["model_version"] = None  # simulate JSON null

    forecasts = [("2026-04-25", "2000", fc)]
    settlements = {("NYC", "2026-04-25"): _make_settlement(observation_date="2026-04-25")}

    [row] = list(_build_rows(forecasts, settlements))
    assert row["model_version"] == ""
    assert row["model_version"] is not None


def test_end_to_end_with_date_window(tmp_path: Path) -> None:
    """Spec test #10 — --start-date / --end-date bracket the output.
    Doubles as the end-to-end smoke test."""
    import subprocess
    import sys

    import pandas as pd

    snaps = tmp_path / "snapshots"
    _build_snapshot_tree(
        snaps,
        {
            "2026-04-23": {
                "2000": {
                    "forecasts": [
                        _make_forecast(forecast_date="2026-04-23"),
                    ],
                    "settlements": [_make_settlement(observation_date="2026-04-23")],
                },
            },
            "2026-04-25": {
                "2000": {
                    "forecasts": [
                        _make_forecast(forecast_date="2026-04-25"),
                    ],
                    "settlements": [_make_settlement(observation_date="2026-04-25")],
                },
            },
            "2026-04-27": {
                "2000": {
                    "forecasts": [
                        _make_forecast(forecast_date="2026-04-27"),
                    ],
                    "settlements": [_make_settlement(observation_date="2026-04-27")],
                },
            },
        },
    )

    out_path = tmp_path / "corpus.parquet"
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_emos_corpus.py"),
            "--snapshots",
            str(snaps),
            "--output",
            str(out_path),
            "--start-date",
            "2026-04-24",
            "--end-date",
            "2026-04-26",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    df = pd.read_parquet(out_path)
    assert sorted(df["snapshot_date"].unique()) == ["2026-04-25"]


def test_output_columns_match_build_rows_keys() -> None:
    """Lock the _OUTPUT_COLUMNS list against the actual _build_rows
    output keys so a future schema change can't silently drift."""

    forecasts = [
        ("2026-04-25", "2000", _make_forecast(forecast_date="2026-04-25")),
    ]
    settlements = {
        ("NYC", "2026-04-25"): _make_settlement(observation_date="2026-04-25"),
    }
    [row] = list(_build_rows(forecasts, settlements))
    assert set(row.keys()) == set(_OUTPUT_COLUMNS), (
        f"keys={sorted(row.keys())}, columns={sorted(_OUTPUT_COLUMNS)}"
    )
