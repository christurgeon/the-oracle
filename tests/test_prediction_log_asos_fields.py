"""Tests for ASOS prediction-log fields (schema v6, Task 7.1).

Three contracts:
  (a) round-trip: a record carrying all new ASOS fields survives
      serialize → JSONL → deserialize with values intact.
  (b) backward-compat: a serialized dict MISSING the new ASOS keys still
      deserializes cleanly (fields default to None).
  (c) SCHEMA_VERSION (currently 7 after the v7 frame_version bump).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from python.analytics.prediction_log import SCHEMA_VERSION, PredictionLog, PredictionRecord
from python.common import Side


def _base_rec(**overrides) -> PredictionRecord:
    defaults = {
        "timestamp": datetime(2026, 6, 5, 12, 0, 0, tzinfo=UTC),
        "strategy": "intraday_asos_high",
        "market_ticker": "KXHIGHMIA-26JUN05-G88",
        "side": Side.YES,
        "city": "MIA",
        "daily_variable": "high",
        "strike_type": "greater",
        "floor_strike": 88.0,
        "cap_strike": None,
        "expiration_date": date(2026, 6, 5),
        "raw_ensemble_prob": 0.55,
        "calibrated_prob": 0.60,
        "market_price": 0.52,
        "exec_price": 0.54,
        "confidence": 0.80,
    }
    defaults.update(overrides)
    return PredictionRecord(**defaults)


class TestSchemaVersion:
    def test_schema_version_is_6(self) -> None:
        """SCHEMA_VERSION must be 7 after the v7 bump (v6 added the ASOS audit block)."""
        assert SCHEMA_VERSION == 7

    def test_new_record_carries_version_6(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        log.append(_base_rec())
        recs = list(log.iter_all())
        assert len(recs) == 1
        assert recs[0].schema_version == 7


class TestAsosFieldsRoundTrip:
    """(a) Round-trip: records carrying the new ASOS fields survive JSONL."""

    def test_all_new_fields_round_trip(self, tmp_path: Path) -> None:
        log = PredictionLog(tmp_path / "predictions.jsonl")
        rec = _base_rec(
            asos_M=87.4,
            asos_obs_age_h=2.5,
            mu_gain=1.2,
            sigma_gain=0.3,
            climo_n=45,
            climo_source="asos_klax_2020_2025",
            posterior_p_raw=0.62,
            boundary_distance_f=0.8,
            post_mean_f=88.1,
            guard_outcome="PASS",
            prelim_drift_f=0.4,
        )
        log.append(rec)
        recs = list(log.iter_all())
        assert len(recs) == 1
        r = recs[0]
        assert r.asos_M == pytest.approx(87.4)
        assert r.asos_obs_age_h == pytest.approx(2.5)
        assert r.mu_gain == pytest.approx(1.2)
        assert r.sigma_gain == pytest.approx(0.3)
        assert r.climo_n == 45
        assert r.climo_source == "asos_klax_2020_2025"
        assert r.posterior_p_raw == pytest.approx(0.62)
        assert r.boundary_distance_f == pytest.approx(0.8)
        assert r.post_mean_f == pytest.approx(88.1)
        assert r.guard_outcome == "PASS"
        assert r.prelim_drift_f == pytest.approx(0.4)

    def test_new_fields_in_raw_json(self, tmp_path: Path) -> None:
        """New fields are serialized into the raw JSON dict on disk."""
        log = PredictionLog(tmp_path / "predictions.jsonl")
        rec = _base_rec(asos_M=86.0, climo_n=30, guard_outcome="ASOS_STALE_OBS")
        log.append(rec)
        raw = json.loads((tmp_path / "predictions.jsonl").read_text().strip())
        assert raw["asos_M"] == pytest.approx(86.0)
        assert raw["climo_n"] == 30
        assert raw["guard_outcome"] == "ASOS_STALE_OBS"

    def test_none_valued_new_fields_serialize(self, tmp_path: Path) -> None:
        """New fields default to None and serialize as null."""
        log = PredictionLog(tmp_path / "predictions.jsonl")
        log.append(_base_rec())
        raw = json.loads((tmp_path / "predictions.jsonl").read_text().strip())
        for key in (
            "asos_M",
            "asos_obs_age_h",
            "mu_gain",
            "sigma_gain",
            "climo_n",
            "climo_source",
            "posterior_p_raw",
            "boundary_distance_f",
            "post_mean_f",
            "guard_outcome",
            "prelim_drift_f",
        ):
            assert key in raw, f"field {key!r} missing from serialized dict"
            assert raw[key] is None, f"field {key!r} expected None, got {raw[key]!r}"


class TestBackwardCompat:
    """(b) Old records missing the new keys still deserialize cleanly."""

    def test_pre_v6_record_missing_asos_fields_deserializes(self, tmp_path: Path) -> None:
        """A v5 record without ASOS fields deserializes with all new fields = None."""
        path = tmp_path / "predictions.jsonl"
        legacy = {
            "timestamp": "2026-05-01T12:00:00+00:00",
            "strategy": "intraday_emos_high",
            "market_ticker": "KXHIGHMIA-26MAY01-G86",
            "side": "YES",
            "city": "MIA",
            "daily_variable": "high",
            "strike_type": "greater",
            "floor_strike": 86.0,
            "cap_strike": None,
            "expiration_date": "2026-05-01",
            "raw_ensemble_prob": 0.50,
            "calibrated_prob": 0.55,
            "market_price": 0.50,
            "exec_price": 0.52,
            "confidence": 0.80,
            "ens_mean": 87.0,
            "ens_std": 2.1,
            "emos_prob_raw": 0.55,
            "schema_version": 5,
            # NOTE: none of the ASOS fields are present
        }
        path.write_text(json.dumps(legacy) + "\n")
        log = PredictionLog(path)
        recs = list(log.iter_all())
        assert len(recs) == 1
        r = recs[0]
        assert r.asos_M is None
        assert r.asos_obs_age_h is None
        assert r.mu_gain is None
        assert r.sigma_gain is None
        assert r.climo_n is None
        assert r.climo_source is None
        assert r.posterior_p_raw is None
        assert r.boundary_distance_f is None
        assert r.post_mean_f is None
        assert r.guard_outcome is None
        assert r.prelim_drift_f is None
        # Existing v5 fields still read correctly
        assert r.ens_mean == pytest.approx(87.0)
        assert r.ens_std == pytest.approx(2.1)
        assert r.emos_prob_raw == pytest.approx(0.55)

    def test_completely_legacy_v1_record_still_deserializes(self, tmp_path: Path) -> None:
        """Oldest-format v1 record without any optional fields deserializes."""
        path = tmp_path / "predictions.jsonl"
        v1 = {
            "timestamp": "2026-03-01T12:00:00+00:00",
            "strategy": "weather_edge",
            "market_ticker": "KXHIGHNY-26MAR01-T80",
            "side": "YES",
            "city": "NYC",
            "daily_variable": "high",
            "strike_type": "greater",
            "floor_strike": 80.0,
            "cap_strike": None,
            "expiration_date": "2026-03-01",
            "raw_ensemble_prob": 0.60,
            "calibrated_prob": 0.65,
            "market_price": 0.55,
            "exec_price": 0.57,
            "confidence": 0.85,
            "schema_version": 1,
        }
        path.write_text(json.dumps(v1) + "\n")
        log = PredictionLog(path)
        recs = list(log.iter_all())
        assert len(recs) == 1
        r = recs[0]
        # All new ASOS fields default to None
        assert r.asos_M is None
        assert r.climo_n is None
        assert r.guard_outcome is None
        # The schema version stored in the record reflects the serialized value
        assert r.schema_version == 1
