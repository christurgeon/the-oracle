"""Subprocess tests for ``scripts/_calibration_dry_run.py``.

Exercises the real argparse / exit-code path by invoking the wrapper as a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "_calibration_dry_run.py"

# Calendar-anchor dates match the committed calendar so the synthesised tree
# behaves the same way the real corpus does after the 2026-04-24 cutover.
_GFS_FIRST_SEEN = date(2026, 3, 19)
_ECMWF_FIRST_SEEN = date(2026, 4, 24)
# Both models post-cutover (so triples bucket as "full" rather than
# "single_model_era"). We need n_full >= 100 to pass the sample-size gate.
_POST_CUTOVER_START = date(2026, 4, 25)
_N_DAYS = 50
_CITIES = ("NYC", "CHI")
_DAILY_VAR = "temperature_2m_max"
_ENSEMBLE_WEIGHTS_JSON = '{"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45}'


def _forecast_row(*, city: str, fc_date: str, model: str, n_members: int = 30) -> dict[str, Any]:
    return {
        "city": city,
        "forecast_date": fc_date,
        "daily_variable": _DAILY_VAR,
        "model": model,
        "model_version": "test",
        "temperatures": [60.0 + i * 0.1 for i in range(n_members)],
        "mean": 60.0,
        "std_dev": 1.0,
    }


def _settlement_row(*, city: str, obs_date: str) -> dict[str, Any]:
    return {
        "city": city,
        "observation_date": obs_date,
        "high_f": 65,
        "low_f": 50,
        "product_id": f"pid-{city}-{obs_date}",
        "issued_at": f"{obs_date}T18:00:00+00:00",
        "is_preliminary": False,
    }


def _market_row(*, ticker: str, city: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "city": city,
        "daily_variable": _DAILY_VAR,
        "strike_type": "greater",
        "floor_strike": 65.0,
        "cap_strike": None,
    }


def _build_clean_tree(snaps_root: Path) -> None:
    """Synthesise n_full >= 100 post-cutover triples in a flat ``<date>/`` layout."""
    snaps_root.mkdir(parents=True, exist_ok=True)
    # Earlier snapshots so build_data_availability_calendar picks the
    # committed first-seen dates rather than _POST_CUTOVER_START.
    _seed_first_seen_rows(snaps_root)

    for offset in range(_N_DAYS):
        fc_d = _POST_CUTOVER_START + timedelta(days=offset)
        fc_iso = fc_d.isoformat()
        day_dir = snaps_root / fc_iso
        day_dir.mkdir(exist_ok=True)
        forecasts: list[dict[str, Any]] = []
        settlements: list[dict[str, Any]] = []
        markets: list[dict[str, Any]] = []
        for city in _CITIES:
            for model in ("gfs_seamless", "ecmwf_ifs025"):
                forecasts.append(_forecast_row(city=city, fc_date=fc_iso, model=model))
            settlements.append(_settlement_row(city=city, obs_date=fc_iso))
            markets.append(
                _market_row(
                    ticker=f"KX-{city}-{fc_iso.replace('-', '')}",
                    city=city,
                )
            )
        (day_dir / "weather_forecasts.json").write_text(json.dumps(forecasts))
        (day_dir / "settlements.json").write_text(json.dumps(settlements))
        (day_dir / "markets.json").write_text(json.dumps(markets))


def _seed_first_seen_rows(snaps_root: Path) -> None:
    """Plant minimal forecasts so calendar auto-detect picks the committed dates."""
    # GFS first-seen anchor.
    gfs_dir = snaps_root / _GFS_FIRST_SEEN.isoformat()
    gfs_dir.mkdir(exist_ok=True)
    (gfs_dir / "weather_forecasts.json").write_text(
        json.dumps(
            [
                _forecast_row(
                    city="NYC",
                    fc_date=_GFS_FIRST_SEEN.isoformat(),
                    model="gfs_seamless",
                )
            ]
        )
    )
    # ECMWF first-seen anchor (one day before _POST_CUTOVER_START).
    ec_dir = snaps_root / _ECMWF_FIRST_SEEN.isoformat()
    ec_dir.mkdir(exist_ok=True)
    (ec_dir / "weather_forecasts.json").write_text(
        json.dumps(
            [
                _forecast_row(
                    city="NYC",
                    fc_date=_ECMWF_FIRST_SEEN.isoformat(),
                    model="ecmwf_ifs025",
                )
            ]
        )
    )


def _write_calendar(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-05-09T00:00:00+00:00",
                "method": "auto",
                "models": {
                    "gfs_seamless": {"first_seen_date": _GFS_FIRST_SEEN.isoformat()},
                    "ecmwf_ifs025": {"first_seen_date": _ECMWF_FIRST_SEEN.isoformat()},
                },
            }
        )
    )


def _run_wrapper(*extra_args: str, snapshots: Path, calendar: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(_SCRIPT),
        "--snapshots",
        str(snapshots),
        "--calendar",
        str(calendar),
        "--ensemble-weights",
        _ENSEMBLE_WEIGHTS_JSON,
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# T6a — clean tree exits 0
# ---------------------------------------------------------------------------


def test_t6a_clean_tree_exits_zero(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    cal = tmp_path / "cal.json"
    _build_clean_tree(snaps)
    _write_calendar(cal)

    proc = _run_wrapper(snapshots=snaps, calendar=cal)
    assert proc.returncode == 0, (
        f"expected returncode 0, got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # Sanity: report header should be in stdout.
    assert "Calibrator input-contract report" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# T6b — contaminated tree exits 1
# ---------------------------------------------------------------------------


def test_t6b_contaminated_tree_exits_one(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    cal = tmp_path / "cal.json"
    _build_clean_tree(snaps)
    _write_calendar(cal)

    # Drop ECMWF from a single post-cutover triple. _build_clean_tree wrote
    # full forecasts on every fc_date in [_POST_CUTOVER_START, +_N_DAYS); we
    # rewrite one of them with only the GFS row to create a partial triple.
    target_date = (_POST_CUTOVER_START + timedelta(days=10)).isoformat()
    target = snaps / target_date / "weather_forecasts.json"
    rows = json.loads(target.read_text())
    target_city = "NYC"
    rows = [r for r in rows if not (r["city"] == target_city and r["model"] == "ecmwf_ifs025")]
    target.write_text(json.dumps(rows))

    proc = _run_wrapper(snapshots=snaps, calendar=cal)
    assert proc.returncode == 1, (
        f"expected returncode 1, got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "ABORT" in proc.stdout, proc.stdout
    assert "partial" in proc.stdout.lower(), proc.stdout


# ---------------------------------------------------------------------------
# T6c — --regenerate-calendar writes the file
# ---------------------------------------------------------------------------


def test_t6c_regenerate_calendar_writes_file(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    cal = tmp_path / "cal.json"
    _build_clean_tree(snaps)
    # Note: cal.json deliberately does NOT exist at this point; the wrapper
    # must build it from the snapshot tree.

    proc = _run_wrapper("--regenerate-calendar", snapshots=snaps, calendar=cal)

    # Calendar must be written regardless of the gate result.
    assert cal.exists(), f"calendar was not written\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    payload = json.loads(cal.read_text())
    assert payload["schema_version"] == 1
    assert payload["method"] == "auto"
    assert payload["models"]["gfs_seamless"]["first_seen_date"] == _GFS_FIRST_SEEN.isoformat()
    assert payload["models"]["ecmwf_ifs025"]["first_seen_date"] == _ECMWF_FIRST_SEEN.isoformat()
    # Clean tree should still exit 0.
    assert proc.returncode == 0, (
        f"expected returncode 0, got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# T6d — regression aborts by default
# ---------------------------------------------------------------------------


def test_t6d_calendar_regression_aborts(tmp_path: Path) -> None:
    """Regression aborts by default."""
    snaps = tmp_path / "snapshots"
    cal = tmp_path / "cal.json"
    _build_clean_tree(snaps)

    # Pre-write a calendar whose gfs_seamless first-seen is 2026-01-01 — earlier
    # than any snapshot in the tree, so regeneration will move it LATER.
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "method": "manual",
                "models": {
                    "gfs_seamless": {"first_seen_date": "2026-01-01"},
                    "ecmwf_ifs025": {"first_seen_date": _ECMWF_FIRST_SEEN.isoformat()},
                },
            }
        )
    )
    original_content = cal.read_text()

    proc = _run_wrapper("--regenerate-calendar", snapshots=snaps, calendar=cal)

    assert proc.returncode == 1, (
        f"expected returncode 1, got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "ABORT" in proc.stdout, proc.stdout
    assert "gfs_seamless" in proc.stdout, proc.stdout
    assert "--allow-calendar-regression" in proc.stdout, proc.stdout
    # File must be unchanged.
    assert cal.read_text() == original_content, "calendar was overwritten despite regression abort"


# ---------------------------------------------------------------------------
# T6e — --allow-calendar-regression bypasses the abort
# ---------------------------------------------------------------------------


def test_t6e_allow_calendar_regression_honored(tmp_path: Path) -> None:
    """--allow-calendar-regression bypasses the abort and rewrites the file."""
    snaps = tmp_path / "snapshots"
    cal = tmp_path / "cal.json"
    _build_clean_tree(snaps)

    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "method": "manual",
                "models": {
                    "gfs_seamless": {"first_seen_date": "2026-01-01"},
                    "ecmwf_ifs025": {"first_seen_date": _ECMWF_FIRST_SEEN.isoformat()},
                },
            }
        )
    )

    proc = _run_wrapper(
        "--regenerate-calendar",
        "--allow-calendar-regression",
        snapshots=snaps,
        calendar=cal,
    )

    # The regression abort must not have fired.
    assert "ABORT — calendar regeneration would regress" not in proc.stdout, proc.stdout
    # The file must have been rewritten with the auto-detected date.
    assert cal.exists()
    payload = json.loads(cal.read_text())
    assert payload["models"]["gfs_seamless"]["first_seen_date"] != "2026-01-01", (
        "calendar was not rewritten"
    )
