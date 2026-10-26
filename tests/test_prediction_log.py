"""Tests for python.analytics.prediction_log."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.prediction_log import PredictionLog, PredictionRecord
from python.common import Side


def _rec(**overrides) -> PredictionRecord:
    defaults = {
        "timestamp": datetime.now(UTC),
        "strategy": "weather_edge",
        "market_ticker": "KXHIGHNY-26APR17-T80",
        "side": Side.YES,
        "city": "NYC",
        "daily_variable": "high",
        "strike_type": "greater",
        "floor_strike": 80.0,
        "cap_strike": None,
        "expiration_date": date(2026, 4, 17),
        "raw_ensemble_prob": 0.68,
        "calibrated_prob": 0.72,
        "market_price": 0.55,
        "exec_price": 0.57,
        "confidence": 0.85,
    }
    defaults.update(overrides)
    return PredictionRecord(**defaults)


class TestPredictionLog:
    def test_append_and_read_back(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        log.append(_rec())
        log.append(_rec(market_ticker="KXHIGHNY-26APR17-T85", calibrated_prob=0.40))

        records = list(log.iter_all())
        assert len(records) == 2
        assert records[0].calibrated_prob == 0.72
        assert records[1].calibrated_prob == 0.40
        assert records[0].side == Side.YES

    def test_rejects_naive_timestamp(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        naive = _rec(timestamp=datetime(2026, 4, 17, 12, 0))  # no tz
        with pytest.raises(ValueError):
            log.append(naive)

    def test_iter_since_filters(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        now = datetime.now(UTC)
        log.append(_rec(timestamp=now - timedelta(days=10)))
        log.append(_rec(timestamp=now - timedelta(hours=1)))

        cutoff = now - timedelta(days=1)
        recent = list(log.iter_since(cutoff))
        assert len(recent) == 1

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "predictions.jsonl"
        log = PredictionLog(path)
        log.append(_rec())
        with path.open("a") as f:
            f.write("{not-json}\n")
            f.write('{"missing_required_fields": true}\n')
        log.append(_rec(market_ticker="KXHIGHNY-T2"))

        records = list(log.iter_all())
        assert len(records) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "missing.jsonl")
        assert list(log.iter_all()) == []


class TestSchemaV2Fields:
    """New fields added in schema version 2."""

    def test_new_fields_serialize_and_round_trip(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        rec = _rec(
            gfs_prob=0.130,
            ecmwf_prob=0.155,
            ensemble_std_pooled=2.4,
        )
        log.append(rec)
        records = list(log.iter_all())
        assert len(records) == 1
        r = records[0]
        assert r.gfs_prob == pytest.approx(0.130)
        assert r.ecmwf_prob == pytest.approx(0.155)
        assert r.ensemble_std_pooled == pytest.approx(2.4)
        assert r.schema_version == 7

    def test_new_fields_default_to_none(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        rec = _rec()  # uses _rec() helper which has no new fields
        log.append(rec)
        records = list(log.iter_all())
        r = records[0]
        assert r.gfs_prob is None
        assert r.ecmwf_prob is None
        assert r.ensemble_std_pooled is None

    def test_schema_v1_record_deserializes_new_fields_as_none(self, tmp_path: Path) -> None:
        # Simulate an old record written without the new fields
        path = tmp_path / "predictions.jsonl"
        old_record = {
            "timestamp": "2026-04-01T12:00:00+00:00",
            "strategy": "weather_edge",
            "market_ticker": "KXHIGHNY-26APR17-T80",
            "side": "YES",
            "city": "NYC",
            "daily_variable": "high",
            "strike_type": "greater",
            "floor_strike": 80.0,
            "cap_strike": None,
            "expiration_date": "2026-04-17",
            "raw_ensemble_prob": 0.68,
            "calibrated_prob": 0.72,
            "market_price": 0.55,
            "exec_price": 0.57,
            "confidence": 0.85,
            "schema_version": 1,
        }
        path.write_text(json.dumps(old_record) + "\n")
        log = PredictionLog(path)
        records = list(log.iter_all())
        assert len(records) == 1
        r = records[0]
        assert r.gfs_prob is None
        assert r.ecmwf_prob is None
        assert r.ensemble_std_pooled is None


def test_prediction_record_schema_v3_has_is_same_day_market(tmp_path: Path) -> None:
    """v3 schema added is_same_day_market; v4 (Tier 2 C, 2026-05-19) adds
    live_calibrator_kind; v5 adds ens_mean/ens_std/emos_prob_raw; v6 adds the
    intraday-ASOS audit block; v7 adds frame_version. The v3-added field still
    round-trips at v7."""
    from python.analytics.prediction_log import SCHEMA_VERSION

    assert SCHEMA_VERSION == 7

    rec = _rec(is_same_day_market=True)
    log = PredictionLog(tmp_path / "predictions.jsonl")
    log.append(rec)
    out = list(log.iter_all())
    assert len(out) == 1
    assert out[0].is_same_day_market is True
    assert out[0].schema_version == 7


def test_prediction_record_v2_back_compat_defaults_is_same_day_market_false(
    tmp_path: Path,
) -> None:
    """Pre-v3 records (no is_same_day_market field) deserialize to False."""
    path = tmp_path / "predictions.jsonl"
    legacy = {
        "timestamp": "2026-04-01T12:00:00+00:00",
        "strategy": "weather_edge",
        "market_ticker": "KXHIGHNY-26APR01-T80",
        "side": "YES",
        "city": "NYC",
        "daily_variable": "high",
        "strike_type": "greater",
        "floor_strike": 80.0,
        "cap_strike": None,
        "expiration_date": None,
        "raw_ensemble_prob": 0.4,
        "calibrated_prob": 0.55,
        "market_price": 0.5,
        "exec_price": 0.52,
        "confidence": 0.8,
        "schema_version": 2,
    }
    path.write_text(json.dumps(legacy) + "\n")
    log = PredictionLog(path)
    out = list(log.iter_all())
    assert len(out) == 1
    assert out[0].is_same_day_market is False


class TestSchemaV5Fields:
    """New fields added in schema version 5 (intraday calibration completeness)."""

    def test_prediction_record_new_fields_default_none(self) -> None:
        """Backward compat: PredictionRecord can be constructed without new fields."""
        rec = _rec()
        assert rec.ens_mean is None
        assert rec.ens_std is None
        assert rec.emos_prob_raw is None

    def test_prediction_record_new_fields_serialize(self, tmp_path: Path) -> None:
        """New fields serialize to JSON when populated."""
        log = PredictionLog(tmp_path / "predictions.jsonl")
        rec = _rec(ens_mean=72.3, ens_std=2.1, emos_prob_raw=0.58)
        log.append(rec)
        path = tmp_path / "predictions.jsonl"
        raw = json.loads(path.read_text().strip())
        assert raw["ens_mean"] == pytest.approx(72.3)
        assert raw["ens_std"] == pytest.approx(2.1)
        assert raw["emos_prob_raw"] == pytest.approx(0.58)

    def test_prediction_record_new_fields_deserialize(self, tmp_path: Path) -> None:
        """New fields round-trip through the JSONL log."""
        log = PredictionLog(tmp_path / "predictions.jsonl")
        rec = _rec(ens_mean=68.0, ens_std=3.5, emos_prob_raw=0.42)
        log.append(rec)
        out = list(log.iter_all())
        assert len(out) == 1
        r = out[0]
        assert r.ens_mean == pytest.approx(68.0)
        assert r.ens_std == pytest.approx(3.5)
        assert r.emos_prob_raw == pytest.approx(0.42)

    def test_prediction_record_schema_version_5(self) -> None:
        """SCHEMA_VERSION is 7 after the v7 bump (v5 added ens_mean/ens_std/emos_prob_raw)."""
        from python.analytics.prediction_log import SCHEMA_VERSION

        assert SCHEMA_VERSION == 7

    def test_pre_v5_record_deserializes_new_fields_as_none(self, tmp_path: Path) -> None:
        """Pre-v5 records missing ens_mean/ens_std/emos_prob_raw deserialize to None."""
        path = tmp_path / "predictions.jsonl"
        legacy = {
            "timestamp": "2026-04-01T12:00:00+00:00",
            "strategy": "weather_edge",
            "market_ticker": "KXHIGHNY-26APR01-T80",
            "side": "YES",
            "city": "NYC",
            "daily_variable": "high",
            "strike_type": "greater",
            "floor_strike": 80.0,
            "cap_strike": None,
            "expiration_date": "2026-04-01",
            "raw_ensemble_prob": 0.4,
            "calibrated_prob": 0.55,
            "market_price": 0.5,
            "exec_price": 0.52,
            "confidence": 0.8,
            "live_calibrator_kind": "isotonic",
            "schema_version": 4,
        }
        path.write_text(json.dumps(legacy) + "\n")
        log = PredictionLog(path)
        out = list(log.iter_all())
        assert len(out) == 1
        r = out[0]
        assert r.ens_mean is None
        assert r.ens_std is None
        assert r.emos_prob_raw is None


class TestSchemaV7FrameVersion:
    """Schema v7: frame_version (LST climate-day frame audit field, §0a)."""

    def test_frame_version_defaults_to_one(self) -> None:
        """A record constructed without frame_version defaults to the legacy
        civil frame (1) — non-ASOS strategies never set it."""
        rec = _rec()
        assert rec.frame_version == 1

    def test_frame_version_survives_write_read_round_trip(self, tmp_path: Path) -> None:
        """frame_version=2 must survive a write→read cycle (a forgotten serializer
        entry would silently defeat the field — this is the §0a discipline test)."""
        log = PredictionLog(tmp_path / "predictions.jsonl")
        log.append(_rec(frame_version=2))
        log.append(_rec(market_ticker="KXHIGHNY-T2", frame_version=1))

        # On disk the field must be present and correct.
        raw_lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
        assert json.loads(raw_lines[0])["frame_version"] == 2
        assert json.loads(raw_lines[1])["frame_version"] == 1

        # And it must deserialize back unchanged.
        out = list(log.iter_all())
        assert len(out) == 2
        assert out[0].frame_version == 2
        assert out[1].frame_version == 1

    def test_pre_v7_record_deserializes_frame_version_as_one(self, tmp_path: Path) -> None:
        """Pre-v7 rows lack frame_version → deserialize to 1 (civil frame)."""
        path = tmp_path / "predictions.jsonl"
        legacy = {
            "timestamp": "2026-06-01T18:00:00+00:00",
            "strategy": "intraday_asos_high",
            "market_ticker": "KXHIGHMIA-26JUN01-B89.5",
            "side": "NO",
            "city": "MIA",
            "daily_variable": "high",
            "strike_type": "between",
            "floor_strike": 89.0,
            "cap_strike": 90.0,
            "expiration_date": "2026-06-01",
            "raw_ensemble_prob": None,
            "calibrated_prob": 0.006,
            "market_price": 0.12,
            "exec_price": 0.89,
            "confidence": 1.0,
            "posterior_p_raw": 0.006,
            "schema_version": 6,
        }
        path.write_text(json.dumps(legacy) + "\n")
        log = PredictionLog(path)
        out = list(log.iter_all())
        assert len(out) == 1
        assert out[0].frame_version == 1
