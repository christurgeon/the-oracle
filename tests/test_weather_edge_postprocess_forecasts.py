from __future__ import annotations

from datetime import date

from python.data.weather import DailyVariable, EnsembleForecast


def _forecast(temps: list[float]) -> EnsembleForecast:
    return EnsembleForecast(
        city="TEST", forecast_date=date(2026, 5, 6), temperatures=list(temps)
    )


def test_build_forecasts_raw_rows_var_serialization_high():
    from python.strategies.weather_edge import build_forecasts_raw_rows

    sink = {
        ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX): {
            "gfs_seamless": _forecast([70.0, 71.0, 72.0])
        }
    }
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 5),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["var"] == "high"  # NOT "TEMP_MAX"
    assert r["city"] == "AUS"
    assert r["forecast_date"] == "2026-05-06"
    assert r["lead"] == 1  # 6 - 5
    assert r["gfs_members_raw"] == [70.0, 71.0, 72.0]
    assert r["gfs_bias_correction_f"] == 0.0


def test_build_forecasts_raw_rows_var_serialization_low():
    from python.strategies.weather_edge import build_forecasts_raw_rows

    sink = {
        ("DEN", date(2026, 5, 6), DailyVariable.TEMP_MIN): {
            "gfs_seamless": _forecast([35.0, 36.0, 37.0])
        }
    }
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 6),
    )
    assert rows[0]["var"] == "low"
    assert rows[0]["lead"] == 0


def test_build_forecasts_raw_rows_records_bias_correction():
    from python.strategies.weather_edge import build_forecasts_raw_rows

    fkey = ("NYC", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink = {fkey: {"gfs_seamless": _forecast([70.0, 71.0, 72.0])}}
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={fkey: -1.5},
        utc_today=date(2026, 5, 6),
    )
    assert rows[0]["gfs_bias_correction_f"] == -1.5


# ---------------------------------------------------------------------------
# C1 — postprocess_forecasts.raw_member_sink out-param
# ---------------------------------------------------------------------------


def test_postprocess_forecasts_populates_raw_member_sink_pre_bias():
    """Sink is populated with a deepcopy of the pre-bias-correction GFS forecast."""
    from python.strategies.weather_edge import postprocess_forecasts

    fc = _forecast([70.0, 71.0, 72.0])
    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    forecast_cache = {fkey: {"gfs_seamless": fc}}
    sink: dict = {}

    postprocess_forecasts(
        forecast_cache=forecast_cache,
        bias_table=None,
        raw_member_sink=sink,
    )

    assert fkey in sink
    # Snapshot is pre-bias correction (and bias_table=None means same values).
    assert sink[fkey]["gfs_seamless"].temperatures == [70.0, 71.0, 72.0]
    # Distinct list identity — deepcopy, not aliased to forecast_cache.
    assert sink[fkey]["gfs_seamless"].temperatures is not fc.temperatures


def test_postprocess_forecasts_no_sink_when_kwarg_none():
    """Backward-compat: omitting raw_member_sink is safe (no exception)."""
    from python.strategies.weather_edge import postprocess_forecasts

    fc = _forecast([70.0, 71.0, 72.0])
    forecast_cache = {("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX): {"gfs_seamless": fc}}
    # raw_member_sink omitted — default is None.
    postprocess_forecasts(forecast_cache=forecast_cache, bias_table=None)
    # No exception; no sink expected.


def test_postprocess_forecasts_does_no_disk_io():
    """Spec invariant: postprocess_forecasts performs zero filesystem writes."""
    from unittest.mock import patch

    from python.strategies.weather_edge import postprocess_forecasts

    fc = _forecast([70.0, 71.0, 72.0])
    forecast_cache = {("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX): {"gfs_seamless": fc}}
    sink: dict = {}

    write_calls: list[tuple] = []
    real_write = __import__("os").write

    def tracking_write(fd, data):  # type: ignore[no-untyped-def]
        write_calls.append((fd, data))
        return real_write(fd, data)

    with patch("os.write", side_effect=tracking_write):
        postprocess_forecasts(
            forecast_cache=forecast_cache, bias_table=None, raw_member_sink=sink
        )

    assert not write_calls, "postprocess_forecasts must not write to disk"


# ---------------------------------------------------------------------------
# _extract — member-extraction helper (spec §5.3)
# ---------------------------------------------------------------------------


def test_extract_returns_none_when_model_absent():
    from python.strategies.weather_evaluation import _extract

    sink_entry = {"gfs_seamless": _forecast([70.0, 71.0, 72.0])}
    assert _extract(sink_entry, "ecmwf_ifs025") is None


def test_extract_returns_list_when_model_present():
    from python.strategies.weather_evaluation import _extract

    sink_entry = {"gfs_seamless": _forecast([70.0, 71.0, 72.0])}
    result = _extract(sink_entry, "gfs_seamless")
    assert result == [70.0, 71.0, 72.0]


def test_extract_returns_none_when_temperatures_empty():
    """Empty-list temperatures coerce to None — eliminates [] downstream."""
    from python.data.weather import EnsembleForecast
    from python.strategies.weather_evaluation import _extract

    sink_entry = {
        "gfs_seamless": EnsembleForecast(
            city="TEST", forecast_date=date(2026, 5, 6), temperatures=[]
        )
    }
    assert _extract(sink_entry, "gfs_seamless") is None


def test_extract_filters_non_finite_floats():
    """Spec §9 R6: NaN/Inf are filtered at extraction."""
    import math

    from python.strategies.weather_evaluation import _extract

    sink_entry = {"gfs_seamless": _forecast([70.0, math.nan, 72.0, math.inf, -math.inf, 74.0])}
    result = _extract(sink_entry, "gfs_seamless")
    assert result == [70.0, 72.0, 74.0]


def test_extract_returns_none_when_all_members_non_finite():
    """All-non-finite collapses to absent-model state."""
    import math

    from python.strategies.weather_evaluation import _extract

    sink_entry = {"gfs_seamless": _forecast([math.nan, math.inf, -math.inf])}
    assert _extract(sink_entry, "gfs_seamless") is None


# ---------------------------------------------------------------------------
# _stats — per-model mean/std helper (spec §5.3)
# ---------------------------------------------------------------------------


def test_stats_returns_none_pair_when_members_none():
    from python.strategies.weather_evaluation import _stats

    assert _stats(None) == (None, None)


def test_stats_returns_mean_and_zero_std_for_single_member():
    """1-member degenerate case: mean is the member, std is 0.0 (not None)."""
    from python.strategies.weather_evaluation import _stats

    mean, std = _stats([72.5])
    assert mean == 72.5
    assert std == 0.0


def test_stats_returns_sample_stdev_for_multi_member():
    """ddof=1 sample std. Spec §5.1 confirms ddof=1 convention."""
    import statistics

    from python.strategies.weather_evaluation import _stats

    members = [70.0, 71.0, 72.0, 73.0, 74.0]
    mean, std = _stats(members)
    assert mean == statistics.mean(members)
    assert abs(std - statistics.stdev(members)) < 1e-12


# ---------------------------------------------------------------------------
# postprocess_forecasts — nested sink shape (spec §5.2)
# ---------------------------------------------------------------------------


def test_postprocess_forecasts_sink_carries_both_models_pre_bias():
    """Nested sink: sink[fkey] is dict[model_name, EnsembleForecast]."""
    from python.strategies.weather_evaluation import postprocess_forecasts

    gfs = _forecast([70.0, 71.0, 72.0])
    ecmwf = _forecast([80.0, 81.0, 82.0])
    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    forecast_cache = {fkey: {"gfs_seamless": gfs, "ecmwf_ifs025": ecmwf}}
    sink: dict = {}

    postprocess_forecasts(
        forecast_cache=forecast_cache,
        bias_table=None,
        raw_member_sink=sink,
    )

    assert fkey in sink
    assert set(sink[fkey].keys()) == {"gfs_seamless", "ecmwf_ifs025"}
    assert sink[fkey]["gfs_seamless"].temperatures == [70.0, 71.0, 72.0]
    assert sink[fkey]["ecmwf_ifs025"].temperatures == [80.0, 81.0, 82.0]
    # Deepcopy invariant: distinct list identity from inputs.
    assert sink[fkey]["gfs_seamless"].temperatures is not gfs.temperatures
    assert sink[fkey]["ecmwf_ifs025"].temperatures is not ecmwf.temperatures


def test_postprocess_forecasts_sink_carries_ecmwf_only_when_gfs_absent():
    """Case A: ECMWF present, GFS absent. Sink has only the ECMWF key."""
    from python.strategies.weather_evaluation import postprocess_forecasts

    ecmwf = _forecast([80.0, 81.0, 82.0])
    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    forecast_cache = {fkey: {"ecmwf_ifs025": ecmwf}}
    sink: dict = {}

    postprocess_forecasts(
        forecast_cache=forecast_cache,
        bias_table=None,
        raw_member_sink=sink,
    )

    assert fkey in sink
    assert set(sink[fkey].keys()) == {"ecmwf_ifs025"}
    assert sink[fkey]["ecmwf_ifs025"].temperatures == [80.0, 81.0, 82.0]


# ---------------------------------------------------------------------------
# build_forecasts_raw_rows — new schema with ECMWF + missing-data policy
# (spec §5.1, §5.3)
# ---------------------------------------------------------------------------


def test_build_forecasts_raw_rows_emits_both_models():
    """Both models present → six per-model fields populated; per-model
    ddof=1 stats match statistics.mean/stdev on the respective members."""
    import statistics

    from python.strategies.weather_evaluation import build_forecasts_raw_rows

    gfs_members = [70.0, 71.0, 72.0, 73.0, 74.0]
    ecmwf_members = [80.0, 81.0, 82.0]
    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink = {
        fkey: {
            "gfs_seamless": _forecast(gfs_members),
            "ecmwf_ifs025": _forecast(ecmwf_members),
        }
    }
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 5),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["gfs_members_raw"] == gfs_members
    assert r["gfs_ens_mean_raw_ddof1"] == statistics.mean(gfs_members)
    assert abs(r["gfs_ens_std_raw_ddof1"] - statistics.stdev(gfs_members)) < 1e-12
    assert r["ecmwf_members_raw"] == ecmwf_members
    assert r["ecmwf_ens_mean_raw_ddof1"] == statistics.mean(ecmwf_members)
    assert abs(r["ecmwf_ens_std_raw_ddof1"] - statistics.stdev(ecmwf_members)) < 1e-12


def test_build_forecasts_raw_rows_emits_gfs_only_with_ecmwf_nulls():
    """Case B: GFS present, ECMWF absent. ECMWF fields explicitly None."""
    from python.strategies.weather_evaluation import build_forecasts_raw_rows

    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink = {fkey: {"gfs_seamless": _forecast([70.0, 71.0, 72.0])}}
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 5),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["gfs_members_raw"] == [70.0, 71.0, 72.0]
    assert r["gfs_ens_mean_raw_ddof1"] == 71.0
    assert r["ecmwf_members_raw"] is None
    assert r["ecmwf_ens_mean_raw_ddof1"] is None
    assert r["ecmwf_ens_std_raw_ddof1"] is None


def test_build_forecasts_raw_rows_emits_ecmwf_only_with_gfs_nulls():
    """Case A: ECMWF present, GFS absent. GFS fields explicitly None.
    bias_correction key still 0.0 per spec §5.3 R4."""
    from python.strategies.weather_evaluation import build_forecasts_raw_rows

    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink = {fkey: {"ecmwf_ifs025": _forecast([80.0, 81.0, 82.0])}}
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 5),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["gfs_members_raw"] is None
    assert r["gfs_ens_mean_raw_ddof1"] is None
    assert r["gfs_ens_std_raw_ddof1"] is None
    assert r["ecmwf_members_raw"] == [80.0, 81.0, 82.0]
    assert r["ecmwf_ens_mean_raw_ddof1"] == 81.0
    # bias_correction is always present per spec §5.3
    assert r["gfs_bias_correction_f"] == 0.0


def test_build_forecasts_raw_rows_drops_when_both_models_empty():
    """Case C: empty sink entry → no row emitted."""
    from python.strategies.weather_evaluation import build_forecasts_raw_rows

    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink: dict = {fkey: {}}  # both models absent
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 5),
    )
    assert rows == []


def test_build_forecasts_raw_rows_one_member_std_is_zero_per_model():
    """1-member-degenerate: _std_raw_ddof1 is 0.0 (not None).
    Distinguishes 'degenerate-but-present' from 'absent'."""
    from python.strategies.weather_evaluation import build_forecasts_raw_rows

    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink = {
        fkey: {
            "gfs_seamless": _forecast([72.5]),
            "ecmwf_ifs025": _forecast([81.0]),
        }
    }
    rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={},
        utc_today=date(2026, 5, 5),
    )
    assert rows[0]["gfs_ens_std_raw_ddof1"] == 0.0
    assert rows[0]["ecmwf_ens_std_raw_ddof1"] == 0.0
    # Per spec §5.3, absent ≠ degenerate: nulls would be wrong here.
    assert rows[0]["gfs_ens_std_raw_ddof1"] is not None
    assert rows[0]["ecmwf_ens_std_raw_ddof1"] is not None


# ---------------------------------------------------------------------------
# Schema-rename invariant test (spec §7.4)
# ---------------------------------------------------------------------------


def test_no_stale_keys_in_emitted_rows():
    """Locks against accidental reintroduction of pre-rename keys
    (spec §7.4). Drives both row producers and scans for the three
    stale keys."""
    from python.analytics.audit_log import RejectionReason
    from python.common import DailyVariable, Direction
    from python.strategies.weather_audit import _eval_audit_row
    from python.strategies.weather_evaluation import build_forecasts_raw_rows
    from python.strategies.weather_parsing import (
        ParsedMarket,
        ScanCandidate,
    )

    STALE_KEYS = {
        "bias_correction_applied_f",
        "ens_mean_raw_ddof1",
        "ens_std_raw_ddof1",
    }

    # Drive build_forecasts_raw_rows
    fkey = ("AUS", date(2026, 5, 6), DailyVariable.TEMP_MAX)
    sink = {
        fkey: {
            "gfs_seamless": _forecast([70.0, 71.0, 72.0]),
            "ecmwf_ifs025": _forecast([80.0, 81.0, 82.0]),
        }
    }
    forecasts_raw_rows = build_forecasts_raw_rows(
        scan_id="abc",
        raw_member_sink=sink,
        bias_corrections={fkey: -1.5},
        utc_today=date(2026, 5, 5),
    )
    assert forecasts_raw_rows, "expected one row from build_forecasts_raw_rows"
    forecasts_raw_keys = set(forecasts_raw_rows[0].keys())
    assert not (STALE_KEYS & forecasts_raw_keys), (
        f"stale keys in forecasts_raw row: {STALE_KEYS & forecasts_raw_keys}"
    )

    # Drive _eval_audit_row
    parsed = ParsedMarket(
        ticker="KXHIGHAUS-26MAY06-T70",
        city="AUS",
        forecast_date=date(2026, 5, 6),
        daily_variable=DailyVariable.TEMP_MAX,
        strike_type=Direction.ABOVE,
        floor_strike=70.0,
        cap_strike=None,
    )
    sc = ScanCandidate(
        parsed=parsed,
        market={"event_ticker": "KXHIGHAUS-26MAY06"},
        price_info=None,
    )
    eval_row = _eval_audit_row(
        scan_id="abc",
        sc=sc,
        price_info=None,
        reason=RejectionReason.EXCLUDED_CITY_VAR,
        gfs_bias_correction_f=-1.5,
    )
    eval_keys = set(eval_row.keys())
    assert not (STALE_KEYS & eval_keys), (
        f"stale keys in eval_audit row: {STALE_KEYS & eval_keys}"
    )
