"""Tests for F8.1 — unify ``calibration_monitor`` on Kalshi-authoritative outcomes.

The calibration monitor previously derived outcomes from NWS snapshots; the
trade dashboards (shadow_dashboard.py, daily_trade_report.py) read
Kalshi-authoritative settlements from ``data/shadow_settlements.full.json``.
Disagreements between the two sources (right-boundary ``between``-bracket
settlement, stale NWS preliminary readings) skew /brier numbers used for kill
decisions away from the trade-pnl reality on /profit. This module adds tests
for the Kalshi-source path while keeping the NWS path as a configurable
fallback.

Invariants enforced:

1. With ``settlement_source="kalshi"``, the monitor joins on
   ``market_ticker`` and uses Kalshi ``market_result`` as the outcome —
   ignoring any NWS-derived value, even if the NWS path would disagree.
2. With ``settlement_source="nws"``, the legacy NWS join is used.
3. Tier 2 C partitioning by ``live_calibrator_kind`` is preserved under the
   Kalshi-source path.
4. Predictions without a matching Kalshi settlement are excluded from Brier
   without crashing the monitor.
5. When a synthetic case disagrees on outcome, Kalshi vs NWS sources produce
   different Brier numbers — sanity check that the switch actually does work.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.calibration_monitor import (
    CalibrationMonitor,
    CalibrationMonitorConfig,
    _load_kalshi_outcomes,
)
from python.analytics.prediction_log import PredictionLog, PredictionRecord
from python.common import Side

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_nws_settlements(
    dir_: Path,
    *,
    city: str,
    obs_date: date,
    high_f: float,
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


def _write_kalshi_settlements(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def _kalshi_row(*, ticker: str, market_result: str, settled: bool = True) -> dict:
    return {
        "market_ticker": ticker,
        "settled": settled,
        "market_result": market_result,
        "strategy": "weather_edge",
    }


def _pred_record(
    *,
    market_ticker: str,
    predicted: float,
    floor: float | None,
    cap: float | None,
    strike_type: str,
    obs_date: date,
    city: str = "NYC",
    daily_variable: str = "high",
    days_ago: int = 1,
    live_calibrator_kind: str | None = "isotonic",
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=datetime.now(UTC) - timedelta(days=days_ago),
        strategy="weather_edge",
        market_ticker=market_ticker,
        side=Side.YES,
        city=city,
        daily_variable=daily_variable,
        strike_type=strike_type,
        floor_strike=floor,
        cap_strike=cap,
        expiration_date=obs_date,
        raw_ensemble_prob=predicted,
        calibrated_prob=predicted,
        market_price=0.50,
        exec_price=0.50,
        confidence=0.85,
        live_calibrator_kind=live_calibrator_kind,
    )


# ---------------------------------------------------------------------------
# Pure unit: _load_kalshi_outcomes
# ---------------------------------------------------------------------------


class TestLoadKalshiOutcomes:
    def test_returns_market_ticker_to_outcome_map(self, tmp_path: Path) -> None:
        f = tmp_path / "settlements.json"
        _write_kalshi_settlements(
            f,
            [
                _kalshi_row(ticker="KXHIGHNY-26MAY10-T80", market_result="yes"),
                _kalshi_row(ticker="KXHIGHCHI-26MAY10-T75", market_result="no"),
            ],
        )
        out = _load_kalshi_outcomes(f)
        assert out == {
            "KXHIGHNY-26MAY10-T80": 1,
            "KXHIGHCHI-26MAY10-T75": 0,
        }

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        # Fresh deploy: file may not exist yet. Must not crash.
        out = _load_kalshi_outcomes(tmp_path / "missing.json")
        assert out == {}

    def test_unsettled_rows_excluded(self, tmp_path: Path) -> None:
        f = tmp_path / "settlements.json"
        _write_kalshi_settlements(
            f,
            [
                _kalshi_row(ticker="A", market_result="yes"),
                _kalshi_row(ticker="B", market_result="no", settled=False),
                {"market_ticker": "C", "settled": True, "market_result": None},
            ],
        )
        out = _load_kalshi_outcomes(f)
        assert out == {"A": 1}

    def test_duplicate_ticker_with_conflict_logs_keeps_first(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        f = tmp_path / "settlements.json"
        _write_kalshi_settlements(
            f,
            [
                _kalshi_row(ticker="A", market_result="yes"),
                _kalshi_row(ticker="A", market_result="no"),
            ],
        )
        with caplog.at_level(logging.WARNING):
            out = _load_kalshi_outcomes(f)
        assert out == {"A": 1}
        # Surface the data-quality issue, don't swallow.
        assert any("conflicting" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration: monitor + Kalshi source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brier_uses_kalshi_settlement_when_configured(tmp_path: Path) -> None:
    """When ``settlement_source='kalshi'``, the monitor uses the Kalshi
    market_result and IGNORES any NWS-derived value the snapshots might
    suggest. This is the core invariant of F8.1.

    Synthetic disagreement: prediction floor_strike=80 with NWS high_f=70
    (NWS says YES=0, prediction 0.95 → Brier = 0.95² ≈ 0.9025) vs Kalshi
    market_result="yes" (Kalshi says YES=1, prediction 0.95 → Brier =
    (0.95-1)² = 0.0025). The configured Kalshi value MUST win.
    """
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"
    kalshi_path = tmp_path / "kalshi.json"

    obs = date(2026, 5, 10)
    # NWS says high=70 → YES outcome under floor=80 is FALSE.
    _write_nws_settlements(snaps, city="NYC", obs_date=obs, high_f=70.0)
    # Kalshi disagrees: market_result="yes" → YES outcome is TRUE.
    _write_kalshi_settlements(
        kalshi_path,
        [_kalshi_row(ticker="KXHIGHNY-26MAY10-T80", market_result="yes")],
    )

    log = PredictionLog(log_path)
    log.append(
        _pred_record(
            market_ticker="KXHIGHNY-26MAY10-T80",
            predicted=0.95,
            floor=80.0,
            cap=None,
            strike_type="greater",
            obs_date=obs,
        )
    )

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=1,
        bin_min_n=1,
        settlement_source="kalshi",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        kalshi_settlements_path=kalshi_path,
    )

    snapshot = await monitor.force_refresh()
    assert snapshot is not None
    assert snapshot.n_pairs_total == 1
    # Kalshi YES=1 ⇒ Brier = (0.95 - 1)² = 0.0025. If the NWS value had been
    # used, this would be (0.95 - 0)² = 0.9025 — failing this assertion.
    assert snapshot.brier_all == pytest.approx(0.0025, abs=1e-6)


@pytest.mark.asyncio
async def test_brier_falls_back_to_nws_when_configured(tmp_path: Path) -> None:
    """When ``settlement_source='nws'``, the legacy NWS path is used.

    Same disagreement fixture as the Kalshi test: NWS says YES=0, Kalshi
    says YES=1. With the NWS source explicitly configured, the monitor
    should produce the NWS Brier (~0.9025) and NOT touch the Kalshi file.
    """
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"
    kalshi_path = tmp_path / "kalshi.json"

    obs = date(2026, 5, 10)
    _write_nws_settlements(snaps, city="NYC", obs_date=obs, high_f=70.0)
    _write_kalshi_settlements(
        kalshi_path,
        [_kalshi_row(ticker="KXHIGHNY-26MAY10-T80", market_result="yes")],
    )

    log = PredictionLog(log_path)
    log.append(
        _pred_record(
            market_ticker="KXHIGHNY-26MAY10-T80",
            predicted=0.95,
            floor=80.0,
            cap=None,
            strike_type="greater",
            obs_date=obs,
        )
    )

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=1,
        bin_min_n=1,
        settlement_source="nws",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        kalshi_settlements_path=kalshi_path,
    )

    snapshot = await monitor.force_refresh()
    assert snapshot is not None
    assert snapshot.n_pairs_total == 1
    # NWS path: predicted 0.95 vs outcome 0 → Brier = 0.9025.
    assert snapshot.brier_all == pytest.approx(0.9025, abs=1e-6)


@pytest.mark.asyncio
async def test_brier_partitioning_by_calibrator_kind_preserved(tmp_path: Path) -> None:
    """Regression guard for Tier 2 C.

    Under the Kalshi-source path, ``_per_city_var_brier`` must still emit
    one row per ``(city, daily_variable, live_calibrator_kind)`` plus the
    aggregate ``kind="all"`` row. This is the invariant the Kalshi switch
    must NOT regress.
    """
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"
    kalshi_path = tmp_path / "kalshi.json"

    obs = date(2026, 5, 10)
    # Kalshi settles: iso ticker = YES (good), emos ticker = NO (bad).
    _write_kalshi_settlements(
        kalshi_path,
        [
            _kalshi_row(ticker="KXHIGHMIA-26MAY10-T80", market_result="yes"),
            _kalshi_row(ticker="KXHIGHMIA-26MAY10-T82", market_result="no"),
        ],
    )

    log = PredictionLog(log_path)
    # Same (city, var), different kinds; same prediction value so any merge
    # to "all" would still average to a unique value distinguishable from
    # the per-kind rows.
    log.append(
        _pred_record(
            market_ticker="KXHIGHMIA-26MAY10-T80",
            predicted=0.90,
            floor=80.0,
            cap=None,
            strike_type="greater",
            obs_date=obs,
            city="MIA",
            daily_variable="high",
            live_calibrator_kind="isotonic",
        )
    )
    log.append(
        _pred_record(
            market_ticker="KXHIGHMIA-26MAY10-T82",
            predicted=0.90,
            floor=82.0,
            cap=None,
            strike_type="greater",
            obs_date=obs,
            city="MIA",
            daily_variable="high",
            live_calibrator_kind="emos_raw",
        )
    )

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=1,
        bin_min_n=1,
        settlement_source="kalshi",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        kalshi_settlements_path=kalshi_path,
    )

    snapshot = await monitor.force_refresh()
    assert snapshot is not None
    rows = snapshot.per_city_var_30d
    kinds = {r.live_calibrator_kind for r in rows if r.city == "MIA" and r.daily_variable == "high"}
    # Must have isotonic, emos_raw, AND the aggregate "all" row.
    assert {"isotonic", "emos_raw", "all"}.issubset(kinds), (
        f"Tier 2 C partitioning regressed under Kalshi source. Got kinds={kinds}"
    )

    # The per-kind rows should carry distinct Brier numbers (one was YES,
    # the other was NO — both predicted 0.90).
    by_kind = {
        r.live_calibrator_kind: r
        for r in rows
        if r.city == "MIA" and r.daily_variable == "high"
    }
    assert by_kind["isotonic"].brier == pytest.approx((0.90 - 1) ** 2, abs=1e-6)
    assert by_kind["emos_raw"].brier == pytest.approx((0.90 - 0) ** 2, abs=1e-6)
    # Aggregate is the mean across both: ((0.90-1)² + (0.90-0)²) / 2 = 0.41.
    assert by_kind["all"].brier == pytest.approx(0.41, abs=1e-6)
    assert by_kind["all"].n == 2


@pytest.mark.asyncio
async def test_unsettled_markets_excluded(tmp_path: Path) -> None:
    """Predictions whose tickers aren't in the Kalshi settlement file (market
    still open, or settlement not yet fetched) must be silently excluded from
    the Brier denominator. The monitor must NOT crash on the unmatched row.
    """
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path = tmp_path / "monitor.json"
    kalshi_path = tmp_path / "kalshi.json"

    obs = date(2026, 5, 10)
    # Only one of the two tickers is settled on the Kalshi side.
    _write_kalshi_settlements(
        kalshi_path,
        [_kalshi_row(ticker="KXHIGHNY-26MAY10-T80", market_result="yes")],
    )

    log = PredictionLog(log_path)
    # Settled: matches Kalshi.
    log.append(
        _pred_record(
            market_ticker="KXHIGHNY-26MAY10-T80",
            predicted=0.80,
            floor=80.0,
            cap=None,
            strike_type="greater",
            obs_date=obs,
        )
    )
    # Unsettled: no Kalshi row. Must be excluded.
    log.append(
        _pred_record(
            market_ticker="KXHIGHNY-26MAY11-T80",
            predicted=0.50,
            floor=80.0,
            cap=None,
            strike_type="greater",
            obs_date=date(2026, 5, 11),
        )
    )

    cfg = CalibrationMonitorConfig(
        enabled=True,
        warmup_pairs=1,
        bin_min_n=1,
        settlement_source="kalshi",
    )
    monitor = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path,
        config=cfg,
        kalshi_settlements_path=kalshi_path,
    )

    snapshot = await monitor.force_refresh()
    assert snapshot is not None
    assert snapshot.n_pairs_total == 1
    assert snapshot.brier_all == pytest.approx(0.04, abs=1e-6)  # (0.8 - 1)²


@pytest.mark.asyncio
async def test_kalshi_and_nws_disagreement_produces_different_brier(
    tmp_path: Path,
) -> None:
    """Sanity check that the settlement_source switch actually changes
    behavior. Synthetic case where one prediction disagrees between sources —
    same fixture under each source must produce a different Brier number.

    This catches "config knob exists but is no-op" regressions.
    """
    snaps = tmp_path / "snapshots"
    log_path = tmp_path / "predictions.jsonl"
    state_path_a = tmp_path / "monitor_a.json"
    state_path_b = tmp_path / "monitor_b.json"
    kalshi_path = tmp_path / "kalshi.json"

    obs = date(2026, 5, 10)
    # Disagreement fixture:
    #   - NWS high_f = 70 → derive_outcome_yes(greater, floor=80) → False → YES=0
    #   - Kalshi market_result = "yes" → YES=1
    _write_nws_settlements(snaps, city="NYC", obs_date=obs, high_f=70.0)
    _write_kalshi_settlements(
        kalshi_path,
        [_kalshi_row(ticker="KXHIGHNY-26MAY10-T80", market_result="yes")],
    )

    log = PredictionLog(log_path)
    log.append(
        _pred_record(
            market_ticker="KXHIGHNY-26MAY10-T80",
            predicted=0.60,
            floor=80.0,
            cap=None,
            strike_type="greater",
            obs_date=obs,
        )
    )

    base_cfg = dict(enabled=True, warmup_pairs=1, bin_min_n=1)

    cfg_kalshi = CalibrationMonitorConfig(**base_cfg, settlement_source="kalshi")
    monitor_kalshi = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path_a,
        config=cfg_kalshi,
        kalshi_settlements_path=kalshi_path,
    )
    snap_kalshi = await monitor_kalshi.force_refresh()
    assert snap_kalshi is not None

    cfg_nws = CalibrationMonitorConfig(**base_cfg, settlement_source="nws")
    monitor_nws = CalibrationMonitor(
        prediction_log=log,
        snapshots_dir=snaps,
        state_path=state_path_b,
        config=cfg_nws,
        kalshi_settlements_path=kalshi_path,
    )
    snap_nws = await monitor_nws.force_refresh()
    assert snap_nws is not None

    # Kalshi outcome=1 → Brier = (0.60 - 1)² = 0.16.
    # NWS outcome=0    → Brier = (0.60 - 0)² = 0.36.
    assert snap_kalshi.brier_all == pytest.approx(0.16, abs=1e-6)
    assert snap_nws.brier_all == pytest.approx(0.36, abs=1e-6)
    assert snap_kalshi.brier_all != snap_nws.brier_all, (
        "Settlement source switch is a no-op — both paths produced identical "
        "Brier on a hand-constructed disagreement fixture."
    )
