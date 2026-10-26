"""Tests for python.analytics.calibration_monitor."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.calibration_monitor import (
    CalibrationMonitor,
    CalibrationMonitorConfig,
    DriftAlert,
    _AlertGate,
    _bin_index,
    _brier_score,
    _reliability_bins,
    _wilson_interval,
)
from python.analytics.prediction_log import PredictionLog, PredictionRecord
from python.common import Side

# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


class TestWilson:
    def test_zero_observations(self) -> None:
        assert _wilson_interval(0, 0) == (0.0, 0.0)

    def test_all_wins(self) -> None:
        lo, hi = _wilson_interval(10, 10)
        assert 0.7 < lo <= hi == 1.0

    def test_all_losses(self) -> None:
        lo, hi = _wilson_interval(0, 10)
        assert lo == 0.0
        assert 0.0 < hi < 0.3

    def test_midpoint(self) -> None:
        lo, hi = _wilson_interval(50, 100)
        assert 0.4 < lo < 0.5 < hi < 0.6


class TestBrierScore:
    def test_empty(self) -> None:
        b, n = _brier_score([])
        assert b is None
        assert n == 0

    def test_perfect_predictions(self) -> None:
        from python.analytics.calibration_monitor import SettledPair

        pairs = [
            SettledPair(
                timestamp=datetime.now(UTC),
                market_ticker="M",
                city="NYC",
                daily_variable="high",
                strike_type="greater",
                predicted_yes=1.0,
                outcome_yes=1,
                raw_ensemble_prob=None,
                observation_date=date(2026, 4, 15),
                actual_temp=80.0,
            )
        ]
        b, n = _brier_score(pairs)
        assert b == 0.0
        assert n == 1


class TestBinIndex:
    def test_boundaries(self) -> None:
        assert _bin_index(0.0) == 0
        assert _bin_index(0.05) == 0
        assert _bin_index(0.10) == 1
        assert _bin_index(0.95) == 9
        assert _bin_index(1.0) == 9


class TestReliabilityBins:
    def test_always_ten_bins(self) -> None:
        bins = _reliability_bins([])
        assert len(bins) == 10


# ---------------------------------------------------------------------------
# Alert gate
# ---------------------------------------------------------------------------


class TestAlertGate:
    def test_first_fire_allowed(self) -> None:
        g = _AlertGate(cooldown_seconds=3600)
        assert g.should_fire("x", "warning", 0.0)

    def test_cooldown_suppresses_same_severity(self) -> None:
        g = _AlertGate(cooldown_seconds=3600)
        g.record("x", "warning", 0.0)
        assert not g.should_fire("x", "warning", 1000.0)
        assert g.should_fire("x", "warning", 4000.0)

    def test_escalation_bypasses_cooldown(self) -> None:
        g = _AlertGate(cooldown_seconds=3600)
        g.record("x", "warning", 0.0)
        assert g.should_fire("x", "critical", 100.0)

    def test_clear_resets(self) -> None:
        g = _AlertGate(cooldown_seconds=3600)
        g.record("x", "warning", 0.0)
        g.clear("x")
        assert g.should_fire("x", "warning", 100.0)

    def test_load_state_v3_round_trip(self) -> None:
        """A state dict shaped like the current on-disk JSON must load and
        replay correctly. Pins the contract so a future schema bump can't
        silently drop latched alerts on process restart."""
        g = _AlertGate(cooldown_seconds=3600)
        raw = {
            "brier_7d_vs_30d": {"severity": "warning", "last_fired_ts": 1_700_000_000.0},
            "bin_deviation": {"severity": "critical", "last_fired_ts": 1_700_000_100.0},
        }
        g.load_state(raw)

        # Latched warning is still inside cooldown.
        assert not g.should_fire("brier_7d_vs_30d", "warning", 1_700_000_500.0)
        # Critical is preserved at the right severity rank (escalation logic
        # would otherwise allow a warning to re-fire after a critical latched).
        assert not g.should_fire("bin_deviation", "warning", 1_700_000_500.0)
        # Round-trip dump echoes the input shape.
        dumped = g.dump_state()
        assert dumped["brier_7d_vs_30d"]["severity"] == "warning"
        assert dumped["bin_deviation"]["severity"] == "critical"
        assert dumped["brier_7d_vs_30d"]["last_fired_ts"] == 1_700_000_000.0

    def test_load_state_drops_malformed_entries(self) -> None:
        """Malformed legacy entries are skipped, not crashed on. Exercises
        the defensive try/except in load_state."""
        g = _AlertGate(cooldown_seconds=3600)
        raw = {
            "good": {"severity": "warning", "last_fired_ts": 1.0},
            "missing_severity": {"last_fired_ts": 2.0},
            "bad_severity_value": {"severity": "purple", "last_fired_ts": 3.0},
            "bad_ts_type": {"severity": "warning", "last_fired_ts": "not-a-number"},
        }
        g.load_state(raw)
        dumped = g.dump_state()
        assert set(dumped.keys()) == {"good"}


# ---------------------------------------------------------------------------
# Integration: monitor + prediction log + settlement snapshots
# ---------------------------------------------------------------------------


def _write_settlements(dir_: Path, city: str, obs_date: date, high_f: float) -> None:
    d = dir_ / obs_date.isoformat() / "2000"
    d.mkdir(parents=True, exist_ok=True)
    settlement = [
        {
            "city": city,
            "observation_date": obs_date.isoformat(),
            "high_f": high_f,
            "low_f": 50.0,
            "is_preliminary": False,
        }
    ]
    (d / "settlements.json").write_text(json.dumps(settlement))


def _pred_record(
    *,
    predicted: float,
    floor: float,
    obs_date: date,
    city: str = "NYC",
    days_ago: int = 1,
    ticker_suffix: str = "",
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=datetime.now(UTC) - timedelta(days=days_ago),
        strategy="weather_edge",
        market_ticker=f"KXHIGH{city}-{obs_date.isoformat()}-T{int(floor)}{ticker_suffix}",
        side=Side.YES,
        city=city,
        daily_variable="high",
        strike_type="greater",
        floor_strike=floor,
        cap_strike=None,
        expiration_date=obs_date,
        raw_ensemble_prob=predicted,
        calibrated_prob=predicted,
        market_price=0.50,
        exec_price=0.50,
        confidence=0.85,
    )


@pytest.mark.asyncio
async def test_force_refresh_builds_snapshot(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date(2026, 4, 15)
    _write_settlements(snaps, "NYC", obs, high_f=85.0)  # YES wins (>=80)

    log = PredictionLog(log_path)
    log.append(_pred_record(predicted=0.80, floor=80.0, obs_date=obs))

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
    assert snapshot.n_pairs_total == 1
    assert snapshot.brier_all == pytest.approx(0.04, abs=1e-6)  # (0.8 - 1)^2


@pytest.mark.asyncio
async def test_unsettled_predictions_excluded(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date(2026, 4, 15)
    _write_settlements(snaps, "NYC", obs, high_f=85.0)

    log = PredictionLog(log_path)
    log.append(_pred_record(predicted=0.80, floor=80.0, obs_date=obs))
    # Unsettled: no settlement for this date
    log.append(
        _pred_record(predicted=0.50, floor=80.0, obs_date=date(2026, 5, 1))
    )

    cfg = CalibrationMonitorConfig(enabled=True, warmup_pairs=1, settlement_source="nws")
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
    )

    snapshot = await monitor.force_refresh()
    assert snapshot is not None
    assert snapshot.n_pairs_total == 1


@pytest.mark.asyncio
async def test_duplicate_tickers_deduplicated(tmp_path: Path) -> None:
    # Each scan tick that produces a passed signal appends a record. Treating
    # each as an independent Bernoulli observation inflates n, narrows Wilson
    # CIs, and produces false-positive Brier alerts. The monitor must collapse
    # duplicates by market_ticker before scoring AND keep the latest-timestamp
    # record so the most up-to-date forecast is what's scored.
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date(2026, 4, 15)
    _write_settlements(snaps, "NYC", obs, high_f=85.0)  # YES wins (>=80)

    log = PredictionLog(log_path)
    # Vary days_ago descending and calibrated_prob ascending so the LATEST
    # record (days_ago=1) carries predicted=0.80. If dedup kept the first
    # record instead, Brier would be (0.10 - 1)^2 = 0.81 — clearly wrong.
    for i, days_ago in enumerate(range(10, 0, -1)):
        log.append(
            _pred_record(
                predicted=0.10 + i * 0.07,
                floor=80.0,
                obs_date=obs,
                days_ago=days_ago,
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
    # 10 raw records, 1 unique ticker → 1 pair after dedup
    assert snapshot.n_pairs_total == 1
    # Latest record had predicted=0.10 + 9*0.07 = 0.73; outcome=YES (1).
    # Brier = (0.73 - 1)^2 = 0.0729. If first-wins, would be (0.10 - 1)^2 = 0.81.
    assert snapshot.brier_all == pytest.approx(0.0729, abs=1e-6)


@pytest.mark.asyncio
async def test_alert_fires_on_bad_bin(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date(2026, 4, 15)
    # Settlement says high=70 (fails all >=80 predictions)
    _write_settlements(snaps, "NYC", obs, high_f=70.0)

    log = PredictionLog(log_path)
    for i in range(25):
        # Predict 90% YES but always lose: gross miscalibration
        log.append(
            _pred_record(
                predicted=0.95,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-{i}",
            )
        )

    alerts: list[DriftAlert] = []

    async def _capture(alert: DriftAlert) -> None:
        alerts.append(alert)

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=10,
        bin_min_n=5,
        bin_deviation_threshold=0.15,
        city_var_min_n=5,
        city_var_brier_threshold=0.25,
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
    # Allow scheduled callbacks on the running loop to execute.
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    codes = {a.code for a in alerts}
    assert "bin_deviation" in codes or "city_var_brier" in codes


@pytest.mark.asyncio
async def test_bin_deviation_disabled_suppresses_only_that_alert(tmp_path: Path) -> None:
    """bin_deviation_enabled=False mutes bin_deviation but not other alerts."""
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date(2026, 4, 15)
    # Settlement high=70 fails all >=80 predictions → gross per-bin miss AND
    # bad city_var Brier, so we can prove only bin_deviation is silenced.
    _write_settlements(snaps, "NYC", obs, high_f=70.0)

    log = PredictionLog(log_path)
    for i in range(25):
        log.append(
            _pred_record(
                predicted=0.95,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-{i}",
            )
        )

    alerts: list[DriftAlert] = []

    async def _capture(alert: DriftAlert) -> None:
        alerts.append(alert)

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=10,
        bin_min_n=5,
        bin_deviation_threshold=0.15,
        bin_deviation_enabled=False,
        city_var_min_n=5,
        city_var_brier_threshold=0.25,
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
    assert "bin_deviation" not in codes
    # The economic alert is unaffected by the bin_deviation switch.
    assert "city_var_brier" in codes


@pytest.mark.asyncio
async def test_warmup_gates_alerts(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    obs = date(2026, 4, 15)
    _write_settlements(snaps, "NYC", obs, high_f=70.0)
    log = PredictionLog(log_path)
    for i in range(5):
        log.append(
            _pred_record(
                predicted=0.95,
                floor=80.0,
                obs_date=obs,
                ticker_suffix=f"-{i}",
            )
        )

    alerts: list[DriftAlert] = []

    async def _capture(alert: DriftAlert) -> None:
        alerts.append(alert)

    cfg = CalibrationMonitorConfig(enabled=True, warmup_pairs=20, settlement_source="nws")
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        alert_callback=_capture,
    )
    snapshot = await monitor.force_refresh()
    assert snapshot is not None
    assert snapshot.n_pairs_total == 5
    # Below warmup — no alerts should fire
    assert alerts == []


def test_settlement_loader_prefers_final(tmp_path: Path) -> None:
    from python.analytics.calibration_monitor import _load_settlements

    # Same obs_date with both preliminary and final — loader picks whichever
    # has the later (issued_at, product_id). The preliminary is older (issued
    # same afternoon); the final is issued next morning.
    a = tmp_path / "2026-04-15" / "1500"
    b = tmp_path / "2026-04-16" / "0800"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "settlements.json").write_text(
        json.dumps(
            [
                {
                    "city": "NYC",
                    "observation_date": "2026-04-15",
                    "high_f": 85,
                    "low_f": 50,
                    "is_preliminary": True,
                    "issued_at": "2026-04-15T21:00:00+00:00",
                    "product_id": "prelim-1",
                }
            ]
        )
    )
    (b / "settlements.json").write_text(
        json.dumps(
            [
                {
                    "city": "NYC",
                    "observation_date": "2026-04-15",
                    "high_f": 82,  # corrected
                    "low_f": 48,
                    "is_preliminary": False,
                    "issued_at": "2026-04-16T12:00:00+00:00",
                    "product_id": "final-1",
                }
            ]
        )
    )
    table = _load_settlements(tmp_path)
    entry = table[("NYC", date(2026, 4, 15))]
    assert entry.is_preliminary is False
    assert entry.high_f == 82


def test_alert_state_persists_across_restart(tmp_path: Path) -> None:
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"

    log = PredictionLog(log_path)

    cfg = CalibrationMonitorConfig(enabled=True, warmup_pairs=1)
    m1 = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
    )
    m1._gate.record("x", "warning", 12345.0)
    m1._persist_state()

    m2 = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
    )
    # Cooldown should apply because state was restored.
    assert not m2._gate.should_fire("x", "warning", 12346.0)


def test_severity_rank_includes_all_four_alert_levels() -> None:
    from python.analytics.calibration_monitor import _SEVERITY_RANK
    from python.notifications.base import AlertLevel

    assert _SEVERITY_RANK[AlertLevel.INFO] == 0
    assert _SEVERITY_RANK[AlertLevel.WARNING] == 1
    assert _SEVERITY_RANK[AlertLevel.ERROR] == 2
    assert _SEVERITY_RANK[AlertLevel.CRITICAL] == 3


# ---------------------------------------------------------------------------
# C12b — guard removal: reliability bins must remain consumer-stable when
# `raw_ensemble_prob` is always populated (calibrator-identity rows now
# carry the blended fraction instead of None). Regression check that the
# canonical bin builder `_reliability_bins` does NOT depend on
# `raw_ensemble_prob` in its bucketing logic.
# ---------------------------------------------------------------------------


class TestC12bGuardRemovalReliabilityBins:
    """Pre-PR-2 spec invariant #4: dropping the
    `raw_ensemble_prob if calibrator else None` guard must not drift the
    reliability-bin output. _reliability_bins buckets on `predicted_yes`
    only — `raw_ensemble_prob` is just attached metadata."""

    @staticmethod
    def _build_synthetic_corpus(
        *, raw_factory  # callable: (predicted, idx) -> float | None
    ) -> list:
        """Build N=200 SettledPair rows with deterministic predicted/outcome
        values. `raw_factory` decides what raw_ensemble_prob to attach."""
        from python.analytics.calibration_monitor import SettledPair

        pairs = []
        for i in range(200):
            # Spread predictions across [0, 1] so every decile bin is filled.
            predicted = (i + 0.5) / 200.0
            # Outcome: 1 with probability ~ predicted (rough monotonic so
            # bins look realistic but values are deterministic).
            outcome = 1 if (i % 5) < int(predicted * 5) else 0
            pairs.append(
                SettledPair(
                    timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
                    + timedelta(minutes=i),
                    market_ticker=f"M{i:03d}",
                    city="AUS",
                    daily_variable="high",
                    strike_type="greater",
                    predicted_yes=predicted,
                    outcome_yes=outcome,
                    raw_ensemble_prob=raw_factory(predicted, i),
                    observation_date=date(2026, 5, 1) + timedelta(days=i % 30),
                    actual_temp=70.0 + (i % 20),
                )
            )
        return pairs

    def test_reliability_bins_identical_before_and_after_guard_drop(self) -> None:
        """Build two synthetic corpora that differ ONLY in whether
        `raw_ensemble_prob` is populated under identity-calibrator rows;
        assert `_reliability_bins` produces byte-identical output."""
        # Pre-guard-drop semantics: identity-calibrator rows carry
        # raw_ensemble_prob == None.
        pre_drop = self._build_synthetic_corpus(
            raw_factory=lambda predicted, _i: None
        )
        # Post-guard-drop semantics: identity-calibrator rows carry the
        # blended fraction (== predicted_yes for the identity case).
        post_drop = self._build_synthetic_corpus(
            raw_factory=lambda predicted, _i: predicted
        )

        bins_pre = _reliability_bins(pre_drop)
        bins_post = _reliability_bins(post_drop)

        # Same shape (always 10 bins by contract).
        assert len(bins_pre) == 10
        assert len(bins_post) == 10

        # Each bin's bucketed stats are identical: bucketing depends on
        # predicted_yes only; raw_ensemble_prob is not referenced.
        for b_pre, b_post in zip(bins_pre, bins_post, strict=True):
            assert b_pre.lower == b_post.lower
            assert b_pre.upper == b_post.upper
            assert b_pre.n == b_post.n
            assert b_pre.mean_predicted == b_post.mean_predicted
            assert b_pre.mean_observed == b_post.mean_observed
            assert b_pre.wilson_low == b_post.wilson_low
            assert b_pre.wilson_high == b_post.wilson_high

    def test_reliability_bins_stable_under_mixed_raw_population(self) -> None:
        """Realistic mid-window scenario: half rows have raw_ensemble_prob
        (post-guard), half are legacy nulls. Must produce the same bins as
        the all-populated corpus."""
        all_populated = self._build_synthetic_corpus(
            raw_factory=lambda predicted, _i: predicted
        )
        mixed = self._build_synthetic_corpus(
            raw_factory=lambda predicted, i: predicted if i % 2 == 0 else None
        )

        bins_a = _reliability_bins(all_populated)
        bins_b = _reliability_bins(mixed)

        for b_a, b_b in zip(bins_a, bins_b, strict=True):
            assert b_a.n == b_b.n
            assert b_a.mean_predicted == b_b.mean_predicted
            assert b_a.mean_observed == b_b.mean_observed
