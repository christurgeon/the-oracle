"""Tests for the eval-audit-backed calibration corpus (2026-07-04 fix).

Root cause being fixed: weather_edge's ``predictions.jsonl`` stream is written
only for candidates that survive depth-check + Kelly sizing into a signal.
Setting ``risk.strategy_allocations.weather_edge: 0.0``
made ``RiskManager.book_for`` return a $0 pod, every candidate died at
``KELLY_ZERO``, and the calibration monitor's corpus silently froze on
2026-06-16 — while ``eval_audit.jsonl`` (written at Stage-1, before
depth/Kelly/allocation) kept accruing.

The fix repoints the monitor at the eval-audit stream via
``EvalAuditRecordSource`` and adds a semantic corpus-liveness alert.

Invariants enforced here:

1. **Parity/adaptation** — eval-audit rows adapt into ``PredictionRecord``s
   and join Kalshi outcomes into the same ``SettledPair`` population the
   legacy predictions path produced.
2. **Incident regression (load-bearing)** — a corpus consisting solely of
   ``KELLY_ZERO``-rejected evaluations (the $0-pod world) still yields
   ``n_pairs_30d > 0``: a parked pod can no longer dark the corpus.
3. **Corpus-went-dark alert** — if the pair count transitions >0 → 0 while
   the eval stream file is fresh (a join/schema break, not a quiet market),
   a ``corpus_dark`` WARNING fires. A pure file-freshness tripwire cannot
   catch this class of failure.
4. **Strategy filter** — rows stamped by other strategies are excluded;
   legacy weather rows (written before the ``strategy`` field landed) are
   included; rows without ``calibrated_prob`` are dropped.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.calibration_monitor import (
    CalibrationMonitor,
    CalibrationMonitorConfig,
)
from python.analytics.prediction_log import EvalAuditRecordSource

NOW = datetime.now(tz=UTC)


def _eval_row(
    *,
    ticker: str,
    ts: datetime | None = None,
    strategy: str | None = "weather_edge",
    calibrated_prob: float | None = 0.7,
    reason: str = "PASSED",
    city: str = "MIA",
    var: str = "high",
    mid: float | None = 0.55,
) -> dict:
    row = {
        "ts": (ts or NOW).isoformat(),
        "scan_id": "scan-1",
        "ticker": ticker,
        "event": ticker.rsplit("-", 1)[0],
        "city": city,
        "var": var,
        "forecast_date": (ts or NOW).date().isoformat(),
        "strike_type": "between",
        "floor": 84.0,
        "cap": 85.0,
        "mid": mid,
        "yes_bid": (mid - 0.01) if mid is not None else None,
        "yes_ask": (mid + 0.01) if mid is not None else None,
        "raw_ens_prob": 0.6,
        "calibrated_prob": calibrated_prob,
        "rejection_reason": reason,
        "live_calibrator_kind": "isotonic",
    }
    if strategy is not None:
        row["strategy"] = strategy
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _write_kalshi(path: Path, results: dict[str, str]) -> None:
    rows = [
        {
            "market_ticker": t,
            "settled": True,
            "market_result": res,
            "won": res == "yes",
        }
        for t, res in results.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def _make_monitor(
    tmp_path: Path,
    source: EvalAuditRecordSource,
    *,
    kalshi: dict[str, str],
) -> CalibrationMonitor:
    kalshi_path = tmp_path / "shadow_settlements.full.json"
    _write_kalshi(kalshi_path, kalshi)
    (tmp_path / "snapshots").mkdir(exist_ok=True)
    return CalibrationMonitor(
        prediction_log=source,
        snapshots_dir=tmp_path / "snapshots",
        state_path=tmp_path / "monitor.json",
        config=CalibrationMonitorConfig(enabled=True, settlement_source="kalshi"),
        kalshi_settlements_path=kalshi_path,
        corpus_stream_path=source.active_path,
    )


# ---------------------------------------------------------------------------
# 1. Adaptation / parity
# ---------------------------------------------------------------------------


class TestEvalAuditAdaptation:
    def test_rows_adapt_to_prediction_records(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(path, [_eval_row(ticker="KXHIGHMIA-26JUL03-B85")])
        recs = list(EvalAuditRecordSource(path).iter_all())
        assert len(recs) == 1
        r = recs[0]
        assert r.strategy == "weather_edge"
        assert r.market_ticker == "KXHIGHMIA-26JUL03-B85"
        assert r.city == "MIA"
        assert r.daily_variable == "high"
        assert r.calibrated_prob == pytest.approx(0.7)
        assert r.raw_ensemble_prob == pytest.approx(0.6)
        assert r.market_price == pytest.approx(0.55)
        assert r.expiration_date == NOW.date()
        assert r.live_calibrator_kind == "isotonic"

    def test_reads_rotated_siblings(self, tmp_path: Path) -> None:
        # Derived from NOW: _files() drops siblings older than its 32d window.
        rotated_day = (NOW - timedelta(days=2)).date().isoformat()
        active = tmp_path / "eval_audit.jsonl"
        _write_jsonl(active, [_eval_row(ticker="KXHIGHMIA-26JUL03-B85")])
        _write_jsonl(
            tmp_path / f"eval_audit.jsonl.{rotated_day}",
            [_eval_row(ticker="KXHIGHMIA-26JUL01-B84", ts=NOW - timedelta(days=2))],
        )
        _write_jsonl(
            tmp_path / f"eval_audit.jsonl.{rotated_day}.1",
            [_eval_row(ticker="KXHIGHMIA-26JUL01-B86", ts=NOW - timedelta(days=2))],
        )
        tickers = {r.market_ticker for r in EvalAuditRecordSource(active).iter_all()}
        assert tickers == {
            "KXHIGHMIA-26JUL03-B85",
            "KXHIGHMIA-26JUL01-B84",
            "KXHIGHMIA-26JUL01-B86",
        }

    @pytest.mark.asyncio
    async def test_pairs_built_from_eval_stream(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(
            path,
            [
                _eval_row(ticker="KXHIGHMIA-26JUL03-B85", calibrated_prob=0.8),
                _eval_row(ticker="KXHIGHMIA-26JUL03-B87", calibrated_prob=0.2),
            ],
        )
        monitor = _make_monitor(
            tmp_path,
            EvalAuditRecordSource(path),
            kalshi={"KXHIGHMIA-26JUL03-B85": "yes", "KXHIGHMIA-26JUL03-B87": "no"},
        )
        snap = await monitor.force_refresh()
        assert snap is not None
        assert snap.n_pairs_30d == 2


# ---------------------------------------------------------------------------
# 2. Incident regression — the $0-pod world cannot dark the corpus
# ---------------------------------------------------------------------------


class TestZeroPodRegression:
    @pytest.mark.asyncio
    async def test_kelly_zero_rows_still_produce_pairs(self, tmp_path: Path) -> None:
        """Allocation 0.0 ⇒ zero signals ⇒ every eval row is KELLY_ZERO.

        The corpus must still accrue: this encodes the 2026-06-16 incident.
        """
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(
            path,
            [
                _eval_row(
                    ticker=f"KXHIGHMIA-26JUL03-B8{i}",
                    reason="KELLY_ZERO",
                    calibrated_prob=0.6,
                )
                for i in range(3)
            ],
        )
        monitor = _make_monitor(
            tmp_path,
            EvalAuditRecordSource(path),
            kalshi={f"KXHIGHMIA-26JUL03-B8{i}": "yes" for i in range(3)},
        )
        snap = await monitor.force_refresh()
        assert snap is not None
        assert snap.n_pairs_30d > 0


# ---------------------------------------------------------------------------
# 3. Corpus-went-dark semantic alert
# ---------------------------------------------------------------------------


class TestCorpusDarkAlert:
    @pytest.mark.asyncio
    async def test_alert_fires_on_nonzero_to_zero_with_fresh_stream(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(path, [_eval_row(ticker="KXHIGHMIA-26JUL03-B85")])
        monitor = _make_monitor(
            tmp_path,
            EvalAuditRecordSource(path),
            kalshi={"KXHIGHMIA-26JUL03-B85": "yes"},
        )
        snap = await monitor.force_refresh()
        assert snap is not None and snap.n_pairs_30d == 1

        # Simulate a join/schema break: the stream stays FRESH (rows keep
        # landing) but no row matches the monitor's filter any more.
        _write_jsonl(path, [_eval_row(ticker="KXHIGHMIA-26JUL04-B85", strategy="renamed_strategy")])
        await monitor.force_refresh()
        codes = [a.code for a in monitor.recent_alerts]
        assert "corpus_dark" in codes

    @pytest.mark.asyncio
    async def test_no_alert_when_corpus_was_never_populated(self, tmp_path: Path) -> None:
        """Zero pairs from the start (fresh deploy) is not a transition — no alert."""
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(path, [_eval_row(ticker="KXHIGHMIA-26JUL03-B85")])
        monitor = _make_monitor(
            tmp_path,
            EvalAuditRecordSource(path),
            kalshi={},  # no settlements yet -> zero pairs, legitimately
        )
        await monitor.force_refresh()
        codes = [a.code for a in monitor.recent_alerts]
        assert "corpus_dark" not in codes


# ---------------------------------------------------------------------------
# 4. Strategy / schema filters
# ---------------------------------------------------------------------------


class TestRowFilters:
    def test_other_strategies_excluded_legacy_weather_included(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(
            path,
            [
                _eval_row(ticker="KXHIGHMIA-26JUL03-B85"),  # explicit weather_edge
                _eval_row(ticker="KXHIGHMIA-26JUL03-B86", strategy=None),  # legacy row
                _eval_row(ticker="KXHIGHMIA-26JUL03H1700-B85", strategy="intraday_asos_high"),
                _eval_row(ticker="KXHIGHMIA-26JUL03-B88", strategy="intraday_emos_high"),
            ],
        )
        tickers = {r.market_ticker for r in EvalAuditRecordSource(path).iter_all()}
        assert tickers == {"KXHIGHMIA-26JUL03-B85", "KXHIGHMIA-26JUL03-B86"}

    def test_rows_without_calibrated_prob_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(
            path,
            [
                _eval_row(ticker="KXHIGHMIA-26JUL03-B85", calibrated_prob=None),
                _eval_row(ticker="KXHIGHMIA-26JUL03-B86", calibrated_prob=0.4),
            ],
        )
        tickers = {r.market_ticker for r in EvalAuditRecordSource(path).iter_all()}
        assert tickers == {"KXHIGHMIA-26JUL03-B86"}

    def test_rows_without_price_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_audit.jsonl"
        _write_jsonl(path, [_eval_row(ticker="KXHIGHMIA-26JUL03-B85", mid=None)])
        assert list(EvalAuditRecordSource(path).iter_all()) == []


# ---------------------------------------------------------------------------
# 5. Producer stamps strategy on eval rows
# ---------------------------------------------------------------------------


class TestEvalRowCarriesStrategy:
    def test_eval_audit_row_default_strategy(self) -> None:
        # Minimal ScanCandidate stand-in via the real helper is heavyweight;
        # assert on the row contract through the public keyword instead.
        import inspect

        from python.strategies.weather_audit import _eval_audit_row

        sig = inspect.signature(_eval_audit_row)
        assert "strategy" in sig.parameters
        assert sig.parameters["strategy"].default == "weather_edge"


class TestStrikeTypeCanonicalization:
    """Eval rows stamp raw Direction values (above/below); the calibration
    pipeline's canonical labels are greater/less — unmapped, directional rows
    would be silently dropped by derive_outcome_yes on the NWS path."""

    @pytest.mark.parametrize(
        ("raw", "canonical"),
        [("above", "greater"), ("below", "less"), ("between", "between")],
    )
    def test_direction_values_map_to_canonical(
        self, tmp_path: Path, raw: str, canonical: str
    ) -> None:
        path = tmp_path / "eval_audit.jsonl"
        row = _eval_row(ticker="KXHIGHMIA-26JUL03-T85")
        row["strike_type"] = raw
        _write_jsonl(path, [row])
        recs = list(EvalAuditRecordSource(path).iter_all())
        assert len(recs) == 1
        assert recs[0].strike_type == canonical
