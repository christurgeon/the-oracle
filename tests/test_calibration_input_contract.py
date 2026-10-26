"""Tests for the calibrator input-contract preflight + per-pair provenance."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calendar(models: dict[str, str], *, method: str = "auto"):
    from scripts._calibration_input_contract import DataAvailabilityCalendar

    return DataAvailabilityCalendar(
        schema_version=1,
        generated_at="2026-05-09T00:00:00+00:00",
        method=method,
        models={name: date.fromisoformat(d) for name, d in models.items()},
    )


def _forecast(
    *,
    city: str,
    fc_date: str,
    daily_var: str,
    model: str,
    n_members: int = 30,
) -> dict[str, Any]:
    return {
        "city": city,
        "forecast_date": fc_date,
        "daily_variable": daily_var,
        "model": model,
        "model_version": "test",
        "temperatures": [60.0 + i * 0.1 for i in range(n_members)],
        "mean": 60.0,
        "std_dev": 1.0,
    }


def _settlement(city: str, obs_date: str, *, high_f: int = 65, is_preliminary: bool = False):
    return {
        "city": city,
        "observation_date": obs_date,
        "high_f": high_f,
        "low_f": high_f - 15,
        "product_id": f"pid-{city}-{obs_date}",
        "issued_at": f"{obs_date}T18:00:00+00:00",
        "is_preliminary": is_preliminary,
    }


def _market(*, ticker: str, city: str, daily_var: str, strike_type: str = "greater"):
    return {
        "ticker": ticker,
        "city": city,
        "daily_variable": daily_var,
        "strike_type": strike_type,
        "floor_strike": 65.0,
        "cap_strike": None,
    }


# ---------------------------------------------------------------------------
# T1 — happy path
# ---------------------------------------------------------------------------


def test_t1_happy_path_full_coverage():
    """Two models, both present on every triple, all post-cutover."""
    from scripts._calibration_input_contract import assert_snapshot_input_contract

    cal = _calendar({"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"})

    fc_date = "2026-04-25"
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for model in ("gfs_seamless", "ecmwf_ifs025"):
        for city in ("NYC", "CHI"):
            forecasts[(city, fc_date, "temperature_2m_max", model)] = _forecast(
                city=city, fc_date=fc_date, daily_var="temperature_2m_max", model=model
            )

    settlements = {
        ("NYC", fc_date): _settlement("NYC", fc_date),
        ("CHI", fc_date): _settlement("CHI", fc_date),
    }
    markets = [
        _market(ticker="KXHIGHNY-26APR25-T65", city="NYC", daily_var="temperature_2m_max"),
        _market(ticker="KXHIGHCH-26APR25-T65", city="CHI", daily_var="temperature_2m_max"),
    ]

    rep = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        calendar=cal,
    )
    assert rep.n_triples_total == 2
    assert rep.n_triples_full == 2
    assert rep.n_triples_partial == 0
    assert rep.n_triples_empty == 0
    assert rep.n_triples_single_model_era == 0
    assert rep.n_triples_exempt == 0
    assert rep.cells_with_partial_models == []


# ---------------------------------------------------------------------------
# T2 — single-model-era window (everything pre-cutover)
# ---------------------------------------------------------------------------


def test_t2_single_model_era_pre_cutover():
    """Pre-cutover: only GFS expected, triples bucket as single_model_era."""
    from scripts._calibration_input_contract import assert_snapshot_input_contract

    cal = _calendar({"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"})

    fc_date = "2026-04-01"  # pre-cutover
    forecasts = {
        ("NYC", fc_date, "temperature_2m_max", "gfs_seamless"): _forecast(
            city="NYC", fc_date=fc_date, daily_var="temperature_2m_max", model="gfs_seamless"
        )
    }
    settlements = {("NYC", fc_date): _settlement("NYC", fc_date)}
    markets = [_market(ticker="KXHIGHNY-26APR01-T65", city="NYC", daily_var="temperature_2m_max")]

    rep = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        calendar=cal,
    )
    assert rep.n_triples_total == 1
    assert rep.n_triples_single_model_era == 1
    assert rep.n_triples_partial == 0
    assert rep.n_triples_full == 0
    assert rep.n_triples_empty == 0


# ---------------------------------------------------------------------------
# T3 — pre-cutover (mixed) window
# ---------------------------------------------------------------------------


def test_t3_mixed_window_buckets_correctly():
    """Pre-cutover buckets as single_model_era; post-cutover as full-coverage."""
    from scripts._calibration_input_contract import assert_snapshot_input_contract

    cal = _calendar({"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"})

    pre_d = "2026-04-15"
    post_d = "2026-04-25"

    forecasts: dict[tuple[str, str, str, str], dict[str, Any]] = {
        # Pre-cutover: only GFS exists.
        ("NYC", pre_d, "temperature_2m_max", "gfs_seamless"): _forecast(
            city="NYC", fc_date=pre_d, daily_var="temperature_2m_max", model="gfs_seamless"
        ),
        # Post-cutover: both models full.
        ("NYC", post_d, "temperature_2m_max", "gfs_seamless"): _forecast(
            city="NYC", fc_date=post_d, daily_var="temperature_2m_max", model="gfs_seamless"
        ),
        ("NYC", post_d, "temperature_2m_max", "ecmwf_ifs025"): _forecast(
            city="NYC", fc_date=post_d, daily_var="temperature_2m_max", model="ecmwf_ifs025"
        ),
    }
    settlements = {
        ("NYC", pre_d): _settlement("NYC", pre_d),
        ("NYC", post_d): _settlement("NYC", post_d),
    }
    markets = [
        _market(ticker="KXHIGHNY-26APR15-T65", city="NYC", daily_var="temperature_2m_max"),
        _market(ticker="KXHIGHNY-26APR25-T65", city="NYC", daily_var="temperature_2m_max"),
    ]

    rep = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        calendar=cal,
    )
    assert rep.n_triples_total == 2
    assert rep.n_triples_single_model_era == 1
    assert rep.n_triples_full == 1
    assert rep.n_triples_partial == 0
    assert rep.n_triples_empty == 0


# ---------------------------------------------------------------------------
# T4 — post-cutover ECMWF gap synthesised
# ---------------------------------------------------------------------------


def test_t4_post_cutover_ecmwf_gap_buckets_partial():
    """Post-cutover triple missing ECMWF: bucket=partial, count=1.

    Caller (the fitter) decides to abort; the helper itself does not raise.
    """
    from scripts._calibration_input_contract import assert_snapshot_input_contract

    cal = _calendar({"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"})

    fc_date = "2026-04-25"
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]] = {
        # Full triple
        ("NYC", fc_date, "temperature_2m_max", "gfs_seamless"): _forecast(
            city="NYC", fc_date=fc_date, daily_var="temperature_2m_max", model="gfs_seamless"
        ),
        ("NYC", fc_date, "temperature_2m_max", "ecmwf_ifs025"): _forecast(
            city="NYC", fc_date=fc_date, daily_var="temperature_2m_max", model="ecmwf_ifs025"
        ),
        # Gap triple (CHI missing ECMWF row)
        ("CHI", fc_date, "temperature_2m_max", "gfs_seamless"): _forecast(
            city="CHI", fc_date=fc_date, daily_var="temperature_2m_max", model="gfs_seamless"
        ),
    }
    settlements = {
        ("NYC", fc_date): _settlement("NYC", fc_date),
        ("CHI", fc_date): _settlement("CHI", fc_date),
    }
    markets = [
        _market(ticker="KXHIGHNY-26APR25-T65", city="NYC", daily_var="temperature_2m_max"),
        _market(ticker="KXHIGHCH-26APR25-T65", city="CHI", daily_var="temperature_2m_max"),
    ]

    rep = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        calendar=cal,
    )
    assert rep.n_triples_total == 2
    assert rep.n_triples_full == 1
    assert rep.n_triples_partial == 1
    assert rep.n_triples_empty == 0
    assert len(rep.cells_with_partial_models) == 1
    cell = rep.cells_with_partial_models[0]
    assert cell[0] == "CHI"
    assert cell[1] == fc_date
    assert cell[2] == "temperature_2m_max"
    assert "gfs_seamless" in cell[3]
    assert "ecmwf_ifs025" in cell[4]


# ---------------------------------------------------------------------------
# T5 — covered by fitter test files (test_fit_isotonic, test_fit_platt)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Covered by test_fit_isotonic.py.")
def test_t5_bias_table_not_loaded_aborts() -> None:
    pass


@pytest.mark.skip(reason="Covered by test_fit_isotonic.py.")
def test_t6_ensemble_weights_config_load_failure() -> None:
    pass


# ---------------------------------------------------------------------------
# T7 — write-pair-provenance writer round-trip
# ---------------------------------------------------------------------------


def test_t7_record_pair_provenance_writes_valid_jsonl(tmp_path: Path):
    from scripts._calibration_input_contract import record_pair_provenance

    out = tmp_path / "provenance.jsonl"
    rows_in = [
        {
            "ticker": "KXHIGHNY-26MAY08-B72.5",
            "city": "NYC",
            "forecast_date": date(2026, 5, 8),
            "daily_variable": "temperature_2m_max",
            "strike_type": "between",
            "models_used": ["gfs_seamless", "ecmwf_ifs025"],
            "bias_correction_applied": True,
            "spread_inflation_factor": 1.04,
            "ensemble_weights_at_fit": {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            "raw_prob": 0.412,
            "outcome": 0,
        },
        {
            "ticker": "KXHIGHCH-26MAY08-T68",
            "city": "CHI",
            "forecast_date": "2026-05-08",
            "daily_variable": "temperature_2m_max",
            "strike_type": "greater",
            "models_used": ["gfs_seamless"],
            "bias_correction_applied": False,
            "spread_inflation_factor": 1.0,
            "ensemble_weights_at_fit": {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            "raw_prob": 0.611,
            "outcome": 1,
        },
        {
            "ticker": "KXLOWMA-26MAY08-L48",
            "city": "MIA",
            "forecast_date": date(2026, 5, 8),
            "daily_variable": "temperature_2m_min",
            "strike_type": "less",
            "models_used": ["gfs_seamless", "ecmwf_ifs025"],
            "bias_correction_applied": True,
            "spread_inflation_factor": 1.18,
            "ensemble_weights_at_fit": {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            "raw_prob": 0.098,
            "outcome": 0,
        },
    ]
    for row in rows_in:
        record_pair_provenance(out, **row)

    text = out.read_text().strip().splitlines()
    assert len(text) == 3
    parsed = [json.loads(line) for line in text]
    for original, decoded in zip(rows_in, parsed, strict=True):
        assert decoded["schema_version"] == 1
        assert decoded["ticker"] == original["ticker"]
        assert decoded["city"] == original["city"]
        assert decoded["forecast_date"] == "2026-05-08"
        assert decoded["daily_variable"] == original["daily_variable"]
        assert decoded["strike_type"] == original["strike_type"]
        assert decoded["models_used"] == original["models_used"]
        assert decoded["bias_correction_applied"] is original["bias_correction_applied"]
        assert decoded["spread_inflation_factor"] == original["spread_inflation_factor"]
        assert decoded["ensemble_weights_at_fit"] == original["ensemble_weights_at_fit"]
        assert decoded["raw_prob"] == original["raw_prob"]
        assert decoded["outcome"] == original["outcome"]


def test_t7a_record_pair_provenance_payload_under_pipe_buf(tmp_path: Path):
    """Each appended row must fit within PIPE_BUF (~4 KB) for atomic os.write."""
    from scripts._calibration_input_contract import record_pair_provenance

    out = tmp_path / "provenance.jsonl"
    # Realistic row: full 2-model ensemble, between-strike, bias correction.
    record_pair_provenance(
        out,
        ticker="KXHIGHNY-26MAY08-B72.5",
        city="NYC",
        forecast_date=date(2026, 5, 8),
        daily_variable="temperature_2m_max",
        strike_type="between",
        models_used=["gfs_seamless", "ecmwf_ifs025"],
        bias_correction_applied=True,
        spread_inflation_factor=1.04,
        ensemble_weights_at_fit={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        raw_prob=0.412,
        outcome=0,
    )

    raw = out.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert lines, "writer produced no rows"
    for line in lines:
        assert len(line) <= 4096, "row exceeds PIPE_BUF — atomic-append guarantee at risk"


def test_t7b_record_pair_provenance_no_fd_leak_on_serialization_failure(tmp_path: Path):
    """Non-serializable input must raise before the file is opened (serialize-then-open order).

    If a future refactor opens the fd first, a serialization failure creates a
    zero-byte file and leaks the fd — the ``not out.exists()`` assertion catches that.
    """
    from scripts._calibration_input_contract import record_pair_provenance

    out = tmp_path / "provenance.jsonl"

    class _NotSerializable:
        """No JSON encoder — json.dumps must raise TypeError."""

    with pytest.raises(TypeError):
        record_pair_provenance(
            out,
            ticker="KXHIGHNY-26MAY08-T65",
            city="NYC",
            forecast_date=date(2026, 5, 8),
            daily_variable="temperature_2m_max",
            strike_type="greater",
            # list() coerces the iterable but each element is preserved;
            # json.dumps then chokes on _NotSerializable.
            models_used=[_NotSerializable()],  # type: ignore[list-item]
            bias_correction_applied=False,
            spread_inflation_factor=1.0,
            ensemble_weights_at_fit={"gfs_seamless": 1.0},
            raw_prob=0.5,
            outcome=0,
        )

    # File must not exist — serialization failed before os.open ran.
    assert not out.exists(), (
        "record_pair_provenance opened/created the file before json.dumps "
        "succeeded — fd-leak / zero-byte-file regression"
    )


# ---------------------------------------------------------------------------
# T8 — covered by test_calibration_common.py (sibling-generator tests)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Covered by test_calibration_common.py sibling-generator tests.")
def test_t8_pair_list_reproducibility() -> None:
    pass


# ---------------------------------------------------------------------------
# T9 — calendar auto-detect
# ---------------------------------------------------------------------------


def test_t9_build_data_availability_calendar_auto(tmp_path: Path):
    """Auto-detect picks earliest fc_date per model; ignores rows with <25 members."""
    from scripts._calibration_input_contract import (
        DataAvailabilityCalendar,
        build_data_availability_calendar,
    )

    snaps = tmp_path / "snapshots"
    snaps.mkdir()

    # 2026-03-19 / flat layout: GFS-only.
    d1 = snaps / "2026-03-19"
    d1.mkdir()
    (d1 / "weather_forecasts.json").write_text(
        json.dumps(
            [
                _forecast(
                    city="NYC",
                    fc_date="2026-03-20",
                    daily_var="temperature_2m_max",
                    model="gfs_seamless",
                )
            ]
        )
    )

    # 2026-04-24 / nested layout: GFS + ECMWF + a "novel" gfs_globalv2 with
    # one row that has only 10 members (must be ignored).
    d2 = snaps / "2026-04-24"
    d2.mkdir()
    (d2 / "0400").mkdir()
    rows: list[dict[str, Any]] = [
        _forecast(
            city="NYC",
            fc_date="2026-04-25",
            daily_var="temperature_2m_max",
            model="gfs_seamless",
        ),
        _forecast(
            city="NYC",
            fc_date="2026-04-25",
            daily_var="temperature_2m_max",
            model="ecmwf_ifs025",
        ),
        _forecast(
            city="NYC",
            fc_date="2026-04-25",
            daily_var="temperature_2m_max",
            model="gfs_globalv2",
            n_members=10,  # below floor — must be ignored
        ),
    ]
    (d2 / "0400" / "weather_forecasts.json").write_text(json.dumps(rows))

    # 2026-04-26 / nested layout: gfs_globalv2 with 30 members — this should
    # become its first-seen date.
    d3 = snaps / "2026-04-26"
    d3.mkdir()
    (d3 / "0400").mkdir()
    (d3 / "0400" / "weather_forecasts.json").write_text(
        json.dumps(
            [
                _forecast(
                    city="CHI",
                    fc_date="2026-04-27",
                    daily_var="temperature_2m_max",
                    model="gfs_globalv2",
                )
            ]
        )
    )

    out = build_data_availability_calendar(snaps)
    assert out["schema_version"] == 1
    assert out["method"] == "auto"
    assert out["models"]["gfs_seamless"]["first_seen_date"] == "2026-03-20"
    assert out["models"]["ecmwf_ifs025"]["first_seen_date"] == "2026-04-25"
    assert out["models"]["gfs_globalv2"]["first_seen_date"] == "2026-04-27"

    # Round-trip via DataAvailabilityCalendar.from_path.
    cal_path = tmp_path / "cal.json"
    cal_path.write_text(json.dumps(out))
    cal = DataAvailabilityCalendar.from_path(cal_path)
    assert cal.first_seen("gfs_seamless") == date(2026, 3, 20)
    assert cal.first_seen("ecmwf_ifs025") == date(2026, 4, 25)
    assert cal.first_seen("gfs_globalv2") == date(2026, 4, 27)
    assert cal.first_seen("does_not_exist") is None


# ---------------------------------------------------------------------------
# T10 — calendar unknown-model abort
# ---------------------------------------------------------------------------


def test_t10_calendar_unknown_model_in_weights_aborts():
    from scripts._calibration_input_contract import (
        CalibratorInputContractError,
        assert_snapshot_input_contract,
    )

    cal = _calendar({"gfs_seamless": "2026-03-19"})  # NO ecmwf_ifs025
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    settlements: dict[tuple[str, str], dict[str, Any]] = {}
    markets: list[dict[str, Any]] = []

    with pytest.raises(CalibratorInputContractError) as exc:
        assert_snapshot_input_contract(
            forecasts,
            settlements,
            markets,
            ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            calendar=cal,
        )
    msg = str(exc.value)
    assert "ecmwf_ifs025" in msg
    assert "regenerate" in msg.lower() or "calendar" in msg.lower()


# ---------------------------------------------------------------------------
# T11 — covered by test_validate_calibrator.py
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Covered by test_validate_calibrator.py.")
def test_t11_validate_calibrator_rejects_partial_holdout() -> None:
    pass


# ---------------------------------------------------------------------------
# T12 — calendar-staleness WARN on novel model in snapshots
# ---------------------------------------------------------------------------


def test_t12_calendar_staleness_warn_on_novel_snapshot_model(caplog: pytest.LogCaptureFixture):
    """Snapshot has a model the calendar doesn't know about. Helper logs a
    WARN, does NOT raise, and computes bucket counts against the calendar's
    expected_models (which excludes the novel model).
    """
    from scripts._calibration_input_contract import assert_snapshot_input_contract

    cal = _calendar({"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"})

    fc_date = "2026-04-25"
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]] = {
        ("NYC", fc_date, "temperature_2m_max", "gfs_seamless"): _forecast(
            city="NYC", fc_date=fc_date, daily_var="temperature_2m_max", model="gfs_seamless"
        ),
        ("NYC", fc_date, "temperature_2m_max", "ecmwf_ifs025"): _forecast(
            city="NYC", fc_date=fc_date, daily_var="temperature_2m_max", model="ecmwf_ifs025"
        ),
        # Novel model — present in snapshots, absent from calendar.
        ("NYC", fc_date, "temperature_2m_max", "gfs_globalv2"): _forecast(
            city="NYC", fc_date=fc_date, daily_var="temperature_2m_max", model="gfs_globalv2"
        ),
    }
    settlements = {("NYC", fc_date): _settlement("NYC", fc_date)}
    markets = [_market(ticker="KXHIGHNY-26APR25-T65", city="NYC", daily_var="temperature_2m_max")]

    caplog.set_level(logging.WARNING, logger="scripts._calibration_input_contract")

    rep = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        calendar=cal,
    )
    # No abort, single warning mentioning the novel model.
    warn_msgs = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("gfs_globalv2" in m for m in warn_msgs), warn_msgs

    # Bucket counts use expected_models = {gfs_seamless, ecmwf_ifs025}; the
    # triple is full-coverage against that set.
    assert rep.n_triples_total == 1
    assert rep.n_triples_full == 1
    assert rep.n_triples_partial == 0
    assert rep.n_triples_empty == 0


# ---------------------------------------------------------------------------
# T13 — manual-calendar override fires a WARN at load time
# ---------------------------------------------------------------------------


def test_t13_manual_calendar_logs_warn_on_load(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """Loading a calendar with ``method == "manual"`` logs a WARN so the
    override stays operator-visible. ``method == "auto"`` does not.
    """
    from scripts._calibration_input_contract import DataAvailabilityCalendar

    cal_path = tmp_path / "calendar.json"
    cal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-05-09T00:00:00Z",
                "method": "manual",
                "models": {
                    "gfs_seamless": {"first_seen_date": "2026-03-19"},
                    "ecmwf_ifs025": {"first_seen_date": "2026-04-24"},
                },
            }
        )
    )

    caplog.set_level(logging.WARNING, logger="scripts._calibration_input_contract")
    DataAvailabilityCalendar.from_path(cal_path)

    manual_warns = [r for r in caplog.records if "manual" in r.getMessage()]
    assert len(manual_warns) == 1, (
        f"expected exactly one WARN mentioning 'manual', got {len(manual_warns)}"
    )

    # Sanity: an auto calendar produces no manual-mode WARN.
    caplog.clear()
    cal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-05-09T00:00:00Z",
                "method": "auto",
                "models": {"gfs_seamless": {"first_seen_date": "2026-03-19"}},
            }
        )
    )
    DataAvailabilityCalendar.from_path(cal_path)
    assert not any("manual" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# T14 — gate skips markets with unsupported strike_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_strike_type",
    ["below", None],
    ids=["string_not_in_set", "none"],
)
def test_t14_gate_skips_markets_with_unsupported_strike_type(bad_strike_type):
    """Markets with strike_type outside the fitter's accepted set are not graded."""
    from scripts._calibration_input_contract import assert_snapshot_input_contract

    cal = _calendar({"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"})
    fc_date = "2026-04-25"

    forecasts: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for model in ("gfs_seamless", "ecmwf_ifs025"):
        for city in ("NYC", "CHI"):
            forecasts[(city, fc_date, "temperature_2m_max", model)] = _forecast(
                city=city, fc_date=fc_date, daily_var="temperature_2m_max", model=model
            )

    settlements = {
        ("NYC", fc_date): _settlement("NYC", fc_date),
        ("CHI", fc_date): _settlement("CHI", fc_date),
    }

    bogus_market: dict[str, Any] = {
        "ticker": "KXHIGHCH-26APR25-T65",
        "city": "CHI",
        "daily_variable": "temperature_2m_max",
        "strike_type": bad_strike_type,
        "floor_strike": 65.0,
        "cap_strike": None,
    }
    markets = [
        _market(ticker="KXHIGHNY-26APR25-T65", city="NYC", daily_var="temperature_2m_max"),
        bogus_market,
    ]

    rep = assert_snapshot_input_contract(
        forecasts,
        settlements,
        markets,
        ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
        calendar=cal,
    )
    assert rep.n_triples_total == 1
    assert rep.n_triples_full == 1
    assert rep.n_triples_partial == 0
    assert rep.n_triples_empty == 0


# ---------------------------------------------------------------------------
# Smoke — pin the on-disk committed calendar artifact
# ---------------------------------------------------------------------------


def test_committed_calendar_values():
    """Pin ``data/calibration/data_availability_calendar.json`` to committed values.

    If the calendar is regenerated and first-seen dates change, this test must
    be updated in the same commit.
    """
    cal_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "calibration"
        / "data_availability_calendar.json"
    )
    payload = json.loads(cal_path.read_text())

    assert payload["schema_version"] == 1
    assert set(payload["models"].keys()) == {"gfs_seamless", "ecmwf_ifs025"}
    assert payload["models"]["gfs_seamless"]["first_seen_date"] == "2026-03-19"
    assert payload["models"]["ecmwf_ifs025"]["first_seen_date"] == "2026-04-24"
