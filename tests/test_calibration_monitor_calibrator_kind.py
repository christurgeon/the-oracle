"""Tests for Tier 2 C — partition rolling Brier by ``live_calibrator_kind``.

Goal: the 30-day rolling Brier in ``calibration_monitor`` must partition by
``(city, daily_variable, live_calibrator_kind)`` so that pre-cutover
(isotonic-era) rows and post-cutover (emos_raw-era) rows don't mix into a
single number that hides cutover-window regressions.

Invariants enforced by the tests below:

1. ``CityVarBrier`` carries a ``live_calibrator_kind`` field.
2. ``_per_city_var_brier`` partitions by ``(city, daily_variable, kind)``
   AND emits an aggregate ``kind="all"`` row per (city, daily_variable) to
   preserve backwards-compat for consumers that only key on (city, var).
3. The ``brier_7d_vs_30d`` tripwire is suppressed when the 30d window mixes
   kinds for any per-cell pair; suppression must be observable (log line or
   alert detail), never silent.
4. The tripwire fires normally when both windows are single-kind across
   cells with post-cutover history.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.calibration_monitor import (
    CalibrationMonitor,
    CalibrationMonitorConfig,
    CityVarBrier,
    DriftAlert,
    SettledPair,
    _per_city_var_brier,
)
from python.analytics.prediction_log import PredictionLog, PredictionRecord
from python.common import Side

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _write_settlements(
    dir_: Path,
    city: str,
    obs_date: date,
    high_f: float,
    *,
    low_f: float = 50.0,
) -> None:
    d = dir_ / obs_date.isoformat() / "2000"
    d.mkdir(parents=True, exist_ok=True)
    settlement = [
        {
            "city": city,
            "observation_date": obs_date.isoformat(),
            "high_f": high_f,
            "low_f": low_f,
            "is_preliminary": False,
        }
    ]
    (d / "settlements.json").write_text(json.dumps(settlement))


def _pred_record(
    *,
    predicted: float,
    floor: float,
    obs_date: date,
    city: str = "MIA",
    daily_variable: str = "high",
    days_ago: int = 1,
    ticker_suffix: str = "",
    live_calibrator_kind: str | None = "isotonic",
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=datetime.now(UTC) - timedelta(days=days_ago),
        strategy="weather_edge",
        market_ticker=f"KX{daily_variable.upper()}{city}-{obs_date.isoformat()}-T{int(floor)}{ticker_suffix}",
        side=Side.YES,
        city=city,
        daily_variable=daily_variable,
        strike_type="greater",
        floor_strike=floor,
        cap_strike=None,
        expiration_date=obs_date,
        raw_ensemble_prob=predicted,
        calibrated_prob=predicted,
        market_price=0.50,
        exec_price=0.50,
        confidence=0.85,
        live_calibrator_kind=live_calibrator_kind,
    )


# ---------------------------------------------------------------------------
# Unit: _per_city_var_brier partitioning
# ---------------------------------------------------------------------------


def _pair(
    *,
    city: str = "MIA",
    daily_variable: str = "high",
    predicted: float,
    outcome: int,
    kind: str = "isotonic",
    days_ago: int = 1,
) -> SettledPair:
    return SettledPair(
        timestamp=datetime.now(UTC) - timedelta(days=days_ago),
        market_ticker=f"KX-{city}-{daily_variable}-{kind}-{predicted}-{outcome}",
        city=city,
        daily_variable=daily_variable,
        strike_type="greater",
        predicted_yes=predicted,
        outcome_yes=outcome,
        raw_ensemble_prob=predicted,
        observation_date=date(2026, 4, 15),
        actual_temp=85.0 if outcome else 70.0,
        live_calibrator_kind=kind,
    )


class TestPerCityVarBrierPartitioning:
    def test_partitioned_by_kind(self) -> None:
        """Two kinds (isotonic, emos_raw) for the same (city, var) produce
        TWO per-kind rows plus an aggregate ``all`` row."""
        pairs = [
            _pair(predicted=0.80, outcome=1, kind="isotonic"),
            _pair(predicted=0.80, outcome=1, kind="isotonic"),
            _pair(predicted=0.40, outcome=0, kind="emos_raw"),
            _pair(predicted=0.40, outcome=0, kind="emos_raw"),
        ]
        rows = _per_city_var_brier(pairs)
        # Index by (city, var, kind) for clarity.
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        # Per-kind rows are present.
        assert ("MIA", "high", "isotonic") in by_kind
        assert ("MIA", "high", "emos_raw") in by_kind
        # Plus an aggregate "all" row.
        assert ("MIA", "high", "all") in by_kind
        # Per-kind n=2 each, aggregate n=4.
        assert by_kind[("MIA", "high", "isotonic")].n == 2
        assert by_kind[("MIA", "high", "emos_raw")].n == 2
        assert by_kind[("MIA", "high", "all")].n == 4

    def test_aggregate_row_preserved(self) -> None:
        """Backwards-compat: even when only one kind is present, the
        aggregate ``all`` row is still emitted (so legacy consumers don't
        break)."""
        pairs = [
            _pair(predicted=0.80, outcome=1, kind="isotonic"),
            _pair(predicted=0.30, outcome=0, kind="isotonic"),
        ]
        rows = _per_city_var_brier(pairs)
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        assert ("MIA", "high", "isotonic") in by_kind
        assert ("MIA", "high", "all") in by_kind
        # The two rows should have the same Brier when the cell is single-kind.
        assert by_kind[("MIA", "high", "isotonic")].brier == pytest.approx(
            by_kind[("MIA", "high", "all")].brier
        )

    def test_unknown_kind_default_for_legacy_pairs(self) -> None:
        """A SettledPair without an explicit kind (legacy JSONL row) should
        be bucketed as ``"unknown"`` rather than silently dropped."""
        pairs = [
            _pair(predicted=0.80, outcome=1, kind="unknown"),
            _pair(predicted=0.20, outcome=0, kind="unknown"),
        ]
        rows = _per_city_var_brier(pairs)
        kinds = {r.live_calibrator_kind for r in rows}
        assert "unknown" in kinds
        assert "all" in kinds


# ---------------------------------------------------------------------------
# CityVarBrier dataclass shape
# ---------------------------------------------------------------------------


class TestCityVarBrierShape:
    def test_carries_live_calibrator_kind(self) -> None:
        b = CityVarBrier(
            city="MIA",
            daily_variable="high",
            n=10,
            brier=0.18,
            mean_predicted=0.41,
            mean_observed=0.45,
            live_calibrator_kind="emos_raw",
        )
        assert b.live_calibrator_kind == "emos_raw"


# ---------------------------------------------------------------------------
# Integration: full monitor pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brier_partitioned_by_kind(tmp_path: Path) -> None:
    """End-to-end: a prediction stream with both isotonic and emos_raw rows
    for MIA/high produces two separate Brier scores in the snapshot's
    ``per_city_var_30d`` plus an aggregate ``all`` row."""
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date.today() - timedelta(days=2)
    # Settlement: high=85 → YES wins for floor=80.
    _write_settlements(snaps, "MIA", obs, high_f=85.0)

    log = PredictionLog(log_path)
    # 4 isotonic-era rows (predicted 0.80, all win) → brier = (0.20)^2 = 0.04
    for i in range(4):
        log.append(
            _pred_record(
                predicted=0.80,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-iso-{i}",
                live_calibrator_kind="isotonic",
            )
        )
    # 4 emos_raw-era rows (predicted 0.40, all win) → brier = (0.60)^2 = 0.36
    for i in range(4):
        log.append(
            _pred_record(
                predicted=0.40,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-emos-{i}",
                live_calibrator_kind="emos_raw",
            )
        )

    cfg = CalibrationMonitorConfig(
        enabled=True, warmup_pairs=1, bin_min_n=1, settlement_source="nws"
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
    )
    snapshot = await monitor.force_refresh()
    assert snapshot is not None

    rows = {
        (r.city, r.daily_variable, r.live_calibrator_kind): r
        for r in snapshot.per_city_var_30d
    }
    assert ("MIA", "high", "isotonic") in rows
    assert ("MIA", "high", "emos_raw") in rows
    assert ("MIA", "high", "all") in rows
    assert rows[("MIA", "high", "isotonic")].brier == pytest.approx(0.04, abs=1e-6)
    assert rows[("MIA", "high", "emos_raw")].brier == pytest.approx(0.36, abs=1e-6)


@pytest.mark.asyncio
async def test_per_kind_partitioning_at_default_bin_min_n(tmp_path: Path) -> None:
    """With the production default ``bin_min_n=20`` and ≥25 pairs per
    ``live_calibrator_kind`` for a single (city, var), the per-kind
    ``CityVarBrier`` rows MUST still appear in ``snapshot.per_city_var_30d``.

    A regression where the default is misapplied to the per-kind partition
    (e.g., gating on global n vs per-kind n) would silently suppress these
    rows on cells right at the threshold. Companion to
    ``test_brier_partitioned_by_kind`` which uses ``bin_min_n=1`` for speed.
    """
    from python.analytics.calibration_monitor import DEFAULT_BIN_MIN_N

    assert DEFAULT_BIN_MIN_N == 20, "test assumes the production default of 20"

    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date.today() - timedelta(days=2)
    _write_settlements(snaps, "MIA", obs, high_f=85.0)

    log = PredictionLog(log_path)
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.80,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-iso-{i}",
                live_calibrator_kind="isotonic",
            )
        )
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.40,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-emos-{i}",
                live_calibrator_kind="emos_raw",
            )
        )

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=1,
        bin_min_n=DEFAULT_BIN_MIN_N,
        # Mute unrelated tripwires so they can't mask a partition regression.
        city_var_brier_threshold=99.0,
        bin_deviation_threshold=99.0,
        settlement_source="nws",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
    )
    snapshot = await monitor.force_refresh()
    assert snapshot is not None

    rows = {
        (r.city, r.daily_variable, r.live_calibrator_kind): r
        for r in snapshot.per_city_var_30d
    }
    assert ("MIA", "high", "isotonic") in rows, (
        "isotonic per-kind row missing under DEFAULT_BIN_MIN_N=20; "
        f"got keys={sorted(rows)}"
    )
    assert ("MIA", "high", "emos_raw") in rows, (
        "emos_raw per-kind row missing under DEFAULT_BIN_MIN_N=20; "
        f"got keys={sorted(rows)}"
    )
    assert ("MIA", "high", "all") in rows
    assert rows[("MIA", "high", "isotonic")].n == 25
    assert rows[("MIA", "high", "emos_raw")].n == 25
    assert rows[("MIA", "high", "all")].n == 50


@pytest.mark.asyncio
async def test_tripwire_suppressed_during_mixed_kind_30d(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Mid-window cutover: the 30d window mixes kinds, so the global
    ``brier_7d_vs_30d`` alert MUST NOT fire. Suppression must be observable
    (log line)."""
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date.today() - timedelta(days=2)
    # Settlement loses uniformly → both eras get bad Brier.
    _write_settlements(snaps, "MIA", obs, high_f=70.0)  # YES at floor 80 loses

    log = PredictionLog(log_path)
    # 25 pre-cutover (isotonic) rows ~25 days ago with mid Brier
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.50,
                floor=80.0,
                obs_date=obs,
                days_ago=25,
                ticker_suffix=f"-iso-{i}",
                live_calibrator_kind="isotonic",
            )
        )
    # 25 post-cutover (emos_raw) rows ~5 days ago with WORSE Brier
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.95,
                floor=80.0,
                obs_date=obs,
                days_ago=5,
                ticker_suffix=f"-emos-{i}",
                live_calibrator_kind="emos_raw",
            )
        )

    alerts: list[DriftAlert] = []

    async def _capture(alert: DriftAlert) -> None:
        alerts.append(alert)

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=10,
        bin_min_n=5,
        brier_warn_mult=1.3,
        brier_crit_mult=1.6,
        # Disable the other tripwires so the test isolates the 7d_vs_30d one.
        city_var_brier_threshold=99.0,
        bin_deviation_threshold=99.0,
        settlement_source="nws",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        alert_callback=_capture,
    )

    with caplog.at_level(logging.INFO, logger="python.analytics.calibration_monitor"):
        await monitor.force_refresh()

    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    codes = {a.code for a in alerts}
    assert "brier_7d_vs_30d" not in codes, (
        "tripwire must be suppressed when 30d window is mixed-kind; "
        f"got alerts={[a.code for a in alerts]}"
    )
    # The suppression must be observable: either a log entry or a recent_alert
    # entry. We look for the suppression keyword in caplog text.
    suppressed_hit = any(
        "mixed_kind" in rec.getMessage() or "mixed-kind" in rec.getMessage()
        for rec in caplog.records
    )
    assert suppressed_hit, (
        "expected a log line mentioning mixed_kind suppression; "
        f"caplog messages={[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_tripwire_fires_when_both_windows_single_kind(
    tmp_path: Path,
) -> None:
    """Both 7d and 30d windows entirely emos_raw, but 7d Brier is much
    worse than 30d → tripwire MUST fire."""
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    # Need a separate settlement date for 7d-window vs older-than-7d rows so
    # both groups join AND produce different outcomes (drives the Brier
    # divergence the tripwire keys on).
    obs_old = date.today() - timedelta(days=15)
    obs_new = date.today() - timedelta(days=2)
    # Old: settlement says YES wins for floor 80 (high=90), so predicted=0.90
    # rows → Brier ~0.01 (very good).
    _write_settlements(snaps, "MIA", obs_old, high_f=90.0)
    # New: settlement says YES loses for floor 80 (high=70), so predicted=0.90
    # rows → Brier ~0.81 (terrible — 7d worse than 30d).
    _write_settlements(snaps, "MIA", obs_new, high_f=70.0)

    log = PredictionLog(log_path)
    # 25 emos_raw rows ~15 days ago (in 30d window, NOT in 7d window).
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.90,
                floor=80.0,
                obs_date=obs_old,
                days_ago=15,
                ticker_suffix=f"-old-{i}",
                live_calibrator_kind="emos_raw",
            )
        )
    # 25 emos_raw rows ~2 days ago (in 7d window AND 30d window).
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.90,
                floor=80.0,
                obs_date=obs_new,
                days_ago=2,
                ticker_suffix=f"-new-{i}",
                live_calibrator_kind="emos_raw",
            )
        )

    alerts: list[DriftAlert] = []

    async def _capture(alert: DriftAlert) -> None:
        alerts.append(alert)

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=10,
        bin_min_n=5,
        brier_warn_mult=1.3,
        brier_crit_mult=1.6,
        # Mute the unrelated tripwires.
        city_var_brier_threshold=99.0,
        bin_deviation_threshold=99.0,
        settlement_source="nws",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        alert_callback=_capture,
    )

    await monitor.force_refresh()
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    codes = {a.code for a in alerts}
    assert "brier_7d_vs_30d" in codes, (
        "tripwire must fire when both windows are single-kind; "
        f"got alerts={[a.code for a in alerts]}"
    )


# ---------------------------------------------------------------------------
# Schema backwards-compat: legacy JSONL rows
# ---------------------------------------------------------------------------


def test_legacy_jsonl_without_kind_deserializes(tmp_path: Path) -> None:
    """A legacy ``predictions.jsonl`` row written before the schema bump
    (no ``live_calibrator_kind`` field) must still deserialize cleanly via
    ``PredictionLog.iter_all``."""
    log_path = tmp_path / "predictions.jsonl"
    # Simulate a v3 record on disk (no live_calibrator_kind field).
    legacy = {
        "timestamp": "2026-04-15T12:00:00+00:00",
        "strategy": "weather_edge",
        "market_ticker": "KXHIGHMIA-26APR15-T80",
        "side": "YES",
        "city": "MIA",
        "daily_variable": "high",
        "strike_type": "greater",
        "floor_strike": 80.0,
        "cap_strike": None,
        "expiration_date": "2026-04-15",
        "raw_ensemble_prob": 0.7,
        "calibrated_prob": 0.7,
        "market_price": 0.5,
        "exec_price": 0.5,
        "confidence": 0.85,
        "order_id": None,
        "isotonic_fit_date": "2026-05-01",
        "gfs_prob": None,
        "ecmwf_prob": None,
        "ensemble_std_pooled": None,
        "is_same_day_market": False,
        "schema_version": 3,
    }
    log_path.write_text(json.dumps(legacy) + "\n")
    log = PredictionLog(log_path)
    recs = list(log.iter_all())
    assert len(recs) == 1
    # New field defaults to None for legacy rows.
    assert recs[0].live_calibrator_kind is None
