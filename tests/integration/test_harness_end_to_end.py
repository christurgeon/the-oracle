"""Integration tests for the harness CLI (`scripts.run_calibrator_harness`)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Alphabetic 3-letter month codes — `_calibration_common.parse_market_date`
# requires this format (e.g. 26APR15). The plan's numeric form (260415) would
# silently produce 0 FeatureRows.
_MONTH_CODES = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def _write_fixture_corpus(tmp_path: Path, n_days: int = 60) -> Path:
    snap = tmp_path / "snapshots"
    for d_i in range(n_days):
        month_idx = d_i // 28
        day_num = (d_i % 28) + 1
        date_str = f"2026-{(month_idx + 1):02d}-{day_num:02d}"
        ticker_date = f"26{_MONTH_CODES[month_idx]}{day_num:02d}"
        day = snap / date_str
        day.mkdir(parents=True)

        forecasts = [
            {
                "city": "MIA",
                "forecast_date": date_str,
                "daily_variable": "temperature_2m_max",
                "model": "gfs_seamless",
                "temperatures": [80.0 + i * 0.3 for i in range(31)],
            }
        ]
        settlements = [
            {
                "city": "MIA",
                "observation_date": date_str,
                "high_f": 85.0 + d_i % 5,
                "low_f": 70.0,
                "is_preliminary": False,
                "issued_at": f"{date_str}T20:00:00Z",
                "product_id": "MIA-CLI",
            }
        ]
        markets = [
            {
                "ticker": f"KXHIGHMIA-{ticker_date}-T80",
                "strike_type": "greater",
                "floor_strike": 80.0 + (d_i % 5),
                "cap_strike": None,
                "city": "MIA",
                "daily_variable": "temperature_2m_max",
                "yes_ask_dollars": 0.4,
                "yes_bid_dollars": 0.38,
            }
        ]
        (day / "weather_forecasts.json").write_text(json.dumps(forecasts))
        (day / "settlements.json").write_text(json.dumps(settlements))
        (day / "markets.json").write_text(json.dumps(markets))
    return snap


def _write_fixture_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
strategies:
  weather:
    ensemble_weights: {gfs_seamless: 1.0}
    emos_enabled_cells: []
    excluded_city_vars: []
    isotonic_path: data/calibration/isotonic_v2.json
"""
    )
    return config_path


def test_i1_cli_runs_end_to_end(tmp_path):
    """I-1: CLI runs end-to-end on a tiny fixture corpus."""
    snap = _write_fixture_corpus(tmp_path)
    config_path = _write_fixture_config(tmp_path)
    output = tmp_path / "out.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_calibrator_harness",
            "--snapshots-dir", str(snap),
            "--config", str(config_path),
            "--bias-table", "/nonexistent",
            "--output", str(output),
            "--cell-allowlist", "MIA/high",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert output.exists()


def test_i2_schema_validation(tmp_path):
    """I-2: output JSON validates against schema."""
    snap = _write_fixture_corpus(tmp_path)
    config_path = _write_fixture_config(tmp_path)
    output = tmp_path / "out.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_calibrator_harness",
            "--snapshots-dir", str(snap),
            "--config", str(config_path),
            "--bias-table", "/nonexistent",
            "--output", str(output),
            "--cell-allowlist", "MIA/high",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2
    assert "MIA/high" in payload["cells"]
    cell = payload["cells"]["MIA/high"]
    assert cell["verdict"] in (
        "winner",
        "insufficient_evidence",
        "worse_than_identity",
        "insufficient_n",
        "insufficient_diversity",
        "no_separation",
    )


def test_i2_bis_exit_code_on_unwritable_output(tmp_path):
    """I-2-bis: CLI exits nonzero when output dir is unwritable."""
    snap = _write_fixture_corpus(tmp_path)
    config_path = _write_fixture_config(tmp_path)
    output_dir = tmp_path / "readonly"
    output_dir.mkdir()
    output_dir.chmod(0o500)
    output = output_dir / "out.json"
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.run_calibrator_harness",
                "--snapshots-dir", str(snap),
                "--config", str(config_path),
                "--bias-table", "/nonexistent",
                "--output", str(output),
                "--cell-allowlist", "MIA/high",
            ],
            capture_output=True,
            text=True,
        )
        # Exit 3 (write failure) or 1 (corpus failure) acceptable depending
        # on environment; assert it's nonzero.
        assert result.returncode != 0
    finally:
        output_dir.chmod(0o700)


def test_i3_deterministic_reruns(tmp_path):
    """I-3: two consecutive runs produce bit-identical output (excluding run_timestamp)."""
    snap = _write_fixture_corpus(tmp_path)
    config_path = _write_fixture_config(tmp_path)
    output_1 = tmp_path / "out_1.json"
    output_2 = tmp_path / "out_2.json"

    for output in (output_1, output_2):
        subprocess.run(
            [
                sys.executable, "-m", "scripts.run_calibrator_harness",
                "--snapshots-dir", str(snap),
                "--config", str(config_path),
                "--bias-table", "/nonexistent",
                "--output", str(output),
                "--cell-allowlist", "MIA/high",
            ],
            check=True,
            capture_output=True,
        )

    p1 = json.loads(output_1.read_text())
    p2 = json.loads(output_2.read_text())
    p1.pop("run_timestamp", None)
    p2.pop("run_timestamp", None)
    assert p1 == p2


def test_i4_spot_check_obvious_verdicts(tmp_path):
    """I-4: a tiny (5-day) corpus produces an ``insufficient_n`` verdict.

    Per spec I-4: NOT specific verdict identities (those are L-BFGS-B-fragile).
    Instead: structural verdicts that depend on row counts, not coefficient fits.
    """
    snap = tmp_path / "snapshots_sparse"
    for d_i in range(5):
        date_str = f"2026-04-{d_i + 1:02d}"
        day = snap / date_str
        day.mkdir(parents=True)
        (day / "weather_forecasts.json").write_text(
            json.dumps(
                [
                    {
                        "city": "MIA",
                        "forecast_date": date_str,
                        "daily_variable": "temperature_2m_max",
                        "model": "gfs_seamless",
                        "temperatures": [85.0 + i * 0.2 for i in range(31)],
                    }
                ]
            )
        )
        (day / "settlements.json").write_text(
            json.dumps(
                [
                    {
                        "city": "MIA",
                        "observation_date": date_str,
                        "high_f": 87.0,
                        "low_f": 70.0,
                        "is_preliminary": False,
                        "issued_at": f"{date_str}T20:00:00Z",
                        "product_id": "MIA-CLI",
                    }
                ]
            )
        )
        (day / "markets.json").write_text(
            json.dumps(
                [
                    {
                        "ticker": f"KXHIGHMIA-26APR{d_i + 1:02d}-T80",
                        "strike_type": "greater",
                        "floor_strike": 80.0,
                        "cap_strike": None,
                        "city": "MIA",
                        "daily_variable": "temperature_2m_max",
                        "yes_ask_dollars": 0.5,
                        "yes_bid_dollars": 0.48,
                    }
                ]
            )
        )

    config_path = _write_fixture_config(tmp_path)
    output = tmp_path / "out_sparse.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_calibrator_harness",
            "--snapshots-dir", str(snap),
            "--config", str(config_path),
            "--no-bias-correction",
            "--output", str(output),
            "--cell-allowlist", "MIA/high",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(output.read_text())
    assert payload["cells"]["MIA/high"]["verdict"] == "insufficient_n"


def test_i5_smoke_test_mode(tmp_path):
    """I-5: --smoke-test exits 0 with per-cell counts; does NOT write output."""
    snap = _write_fixture_corpus(tmp_path)
    config_path = _write_fixture_config(tmp_path)
    output = tmp_path / "out.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_calibrator_harness",
            "--snapshots-dir", str(snap),
            "--config", str(config_path),
            "--bias-table", "/nonexistent",
            "--output", str(output),
            "--smoke-test",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "MIA/high" in result.stdout
    assert not output.exists()


def test_i6_fourteen_cell_coverage():
    """I-6: harness produces verdicts for all 14 (city, var) cells.

    Skips when neither LIVE_CORPUS_AVAILABLE env var nor a full fixture corpus
    is available, to avoid silently passing in CI.
    """
    import os

    if os.environ.get("LIVE_CORPUS_AVAILABLE") != "1":
        fixture_corpus = Path("tests/fixtures/full_14_cell_corpus")
        if not fixture_corpus.exists():
            pytest.skip(
                "I-6 requires LIVE_CORPUS_AVAILABLE=1 OR "
                "tests/fixtures/full_14_cell_corpus/; neither found"
            )
        corpus_dir = fixture_corpus
    else:
        corpus_dir = Path("data/snapshots")

    output = Path(tempfile.mkdtemp()) / "out.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_calibrator_harness",
            "--snapshots-dir", str(corpus_dir),
            "--output", str(output),
            "--config", "python/config.yaml",
            "--bias-table", "data/gfs_bias/bias_correction.json",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(output.read_text())

    # B7 fix: read expected cities from config instead of hardcoding the 7-city set.
    # If config adds a city, the test silently passes despite missing coverage.
    import yaml
    config = yaml.safe_load(Path("python/config.yaml").read_text())
    weather = config.get("strategies", {}).get("weather", {})
    cities = weather.get("cities") or [
        # Fallback: enumerate from observed snapshot dirs if config doesn't expose
        # an explicit cities list. This keeps the test honest even if the config
        # schema changes shape (e.g., per-city dicts vs a flat list).
        row["city"]
        for snap_day in Path("data/snapshots").iterdir()
        if snap_day.is_dir() and (snap_day / "weather_forecasts.json").exists()
        for row in json.loads((snap_day / "weather_forecasts.json").read_text())
    ]
    cities = sorted(set(cities))
    expected_cells = {f"{city}/{var}" for city in cities for var in ("high", "low")}
    assert set(payload["cells"].keys()) == expected_cells, (
        f"FR-14 14-cell coverage: expected {expected_cells}, got {set(payload['cells'].keys())}"
    )


@pytest.mark.slow
def test_a9_perf_under_30min():
    """A-9: full 14-cell run completes in <=30 minutes on reference hardware.

    Skipped unless LIVE_CORPUS_AVAILABLE=1 (CI fixture is too small to be
    representative). Reference hardware: Mac M1 / a 2-vCPU VPS. On slower
    machines, expect the assertion to fail — that's the falsifier triggering,
    not a test bug. First operational run (Task 24) clocked
    7m51s on 62 settled days, well within the 30-min ceiling.
    """
    import os
    import time

    if os.environ.get("LIVE_CORPUS_AVAILABLE") != "1":
        pytest.skip("A-9 benchmark requires LIVE_CORPUS_AVAILABLE=1")

    output = Path(tempfile.mkdtemp()) / "out.json"
    start = time.monotonic()
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.run_calibrator_harness",
            "--snapshots-dir", "data/snapshots",
            "--output", str(output),
            "--config", "python/config.yaml",
            "--bias-table", "data/gfs_bias/bias_correction.json",
        ],
        capture_output=True, text=True,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, f"harness failed: {result.stderr}"
    assert elapsed < 1800.0, (
        f"A-9 falsifier: full run took {elapsed:.1f}s > 1800s (30 min). "
        f"First knob to reduce: BOOTSTRAP_ITER 400 -> 200."
    )
