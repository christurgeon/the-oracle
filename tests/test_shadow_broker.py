"""Tests for ShadowBroker — the shadow-mode position reconstructor."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from python.common import Side
from python.execution.shadow_broker import ShadowBroker


def _write_log(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _fill(
    *,
    ticker: str,
    dollars: float,
    side: str = "NO",
    position_key: str | None = None,
    close_time: str = "2099-01-01T00:00:00+00:00",
    event: str | None = "fill",
    mode: str = "shadow",
) -> dict:
    entry: dict = {
        "market_ticker": ticker,
        "market_id": ticker,
        "strategy": "weather_edge",
        "side": side,
        "dollars": dollars,
        "contracts": 100,
        "order_price": 0.5,
        "mode": mode,
        "metadata": {"close_time": close_time},
    }
    if event is not None:
        entry["event"] = event
    if position_key is not None:
        entry["position_key"] = position_key
    return entry


def _settlement(
    ticker: str,
    side: str = "NO",
    position_key: str | None = None,
    mode: str = "shadow",
) -> dict:
    return {
        "event": "settlement",
        "market_ticker": ticker,
        "market_id": ticker,
        "position_key": position_key or ticker,
        "strategy": "weather_edge",
        "side": side,
        "result": "expired",
        "pnl": 0.0,
        "mode": mode,
    }


def _build_broker(log_path: Path) -> ShadowBroker:
    return ShadowBroker(
        trade_log_path=log_path,
        configured_bankroll=5000.0,
        today=date(2026, 4, 23),
    )


def _dollars_by_key(broker: ShadowBroker) -> dict[str, float]:
    return {p.position_key: p.dollars for p in broker.replay()}


class TestReplay:
    def test_empty_log(self, tmp_path: Path) -> None:
        assert _build_broker(tmp_path / "log.jsonl").replay() == []

    def test_missing_log_file(self, tmp_path: Path) -> None:
        assert _build_broker(tmp_path / "nonexistent.jsonl").replay() == []

    def test_fills_only_future_dated(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=250.0),
                _fill(ticker="KXHIGHAUS-26APR24-B80.5", dollars=300.0),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {
            "KXHIGHNY-26APR24-B70.5": 250.0,
            "KXHIGHAUS-26APR24-B80.5": 300.0,
        }

    def test_settlement_drops_matching_position(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=250.0),
                _fill(ticker="KXHIGHAUS-26APR24-B80.5", dollars=300.0),
                _settlement("KXHIGHNY-26APR24-B70.5"),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHAUS-26APR24-B80.5": 300.0}

    def test_legacy_entry_no_event_key_treated_as_fill(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(log, [_fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=250.0, event=None)])
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 250.0}

    def test_date_safety_net_keeps_yesterday_drops_two_days_old(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR21-B70.5", dollars=100.0),  # 2 days old → drop
                _fill(ticker="KXHIGHNY-26APR22-B70.5", dollars=200.0),  # yesterday → keep
                _fill(ticker="KXHIGHNY-26APR23-B70.5", dollars=300.0),  # today → keep
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {
            "KXHIGHNY-26APR22-B70.5": 200.0,
            "KXHIGHNY-26APR23-B70.5": 300.0,
        }

    def test_malformed_json_line_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        with open(log, "w") as f:
            f.write(json.dumps(_fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0)) + "\n")
            f.write("{this is not json\n")
            f.write(json.dumps(_fill(ticker="KXHIGHAUS-26APR24-B80.5", dollars=200.0)) + "\n")
        assert _dollars_by_key(_build_broker(log)) == {
            "KXHIGHNY-26APR24-B70.5": 100.0,
            "KXHIGHAUS-26APR24-B80.5": 200.0,
        }

    def test_unparseable_ticker_kept(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(log, [_fill(ticker="KXINX-26MAR20H1600", dollars=500.0)])
        assert _dollars_by_key(_build_broker(log)) == {"KXINX-26MAR20H1600": 500.0}

    def test_partial_fills_aggregate(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0),
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=75.0),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 175.0}

    def test_dollars_from_entry_not_from_price_times_contracts(self, tmp_path: Path) -> None:
        # The log has dollar/contract inconsistencies from old formulas; the
        # broker must use entry["dollars"] directly, not recompute from price.
        log = tmp_path / "log.jsonl"
        entry = _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=500.0)
        entry["contracts"] = 724
        entry["order_price"] = 0.31  # 724 * 0.31 = 224.44, not 500
        _write_log(log, [entry])
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 500.0}

    def test_settlement_for_unmatched_key_is_noop(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0),
                _settlement("KXAUS-26APR24-B80.5"),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 100.0}

    def test_explicit_position_key_overrides_ticker(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    dollars=100.0,
                    position_key="custom-key",
                ),
                _settlement("KXHIGHNY-26APR24-B70.5", position_key="custom-key"),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {}


class TestModeFlipFilter:
    """Live/paper fill rows in the log must not resurrect as shadow positions."""

    _BROKER_LOGGER = "python.execution.shadow_broker"

    @staticmethod
    def _skip_warnings(records: list) -> list:
        return [
            r for r in records
            if r.name == TestModeFlipFilter._BROKER_LOGGER
            and r.levelno == logging.WARNING
            and "skipped" in r.msg
        ]

    def test_live_fill_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=250.0),
                _fill(ticker="KXHIGHAUS-26APR24-B80.5", dollars=300.0, mode="live"),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 250.0}

    def test_paper_fill_skipped(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(log, [_fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0, mode="paper")])
        assert _dollars_by_key(_build_broker(log)) == {}

    def test_missing_mode_treated_as_shadow(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        legacy = _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0)
        del legacy["mode"]
        _write_log(log, [legacy])
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 100.0}

    def test_live_settlement_does_not_close_shadow_position(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=250.0),
                _settlement("KXHIGHNY-26APR24-B70.5", mode="live"),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {"KXHIGHNY-26APR24-B70.5": 250.0}

    def test_warning_emitted_with_skip_count(self, tmp_path: Path, caplog) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0),
                _fill(ticker="KXHIGHAUS-26APR24-B80.5", dollars=200.0, mode="live"),
                _fill(ticker="KXHIGHCHI-26APR24-B65.5", dollars=300.0, mode="live"),
            ],
        )

        with caplog.at_level(logging.WARNING, logger=self._BROKER_LOGGER):
            _build_broker(log).replay()

        warnings = self._skip_warnings(caplog.records)
        assert len(warnings) == 1
        assert warnings[0].args == (2,)

    def test_no_warning_when_no_mode_flip(self, tmp_path: Path, caplog) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(log, [_fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0)])

        with caplog.at_level(logging.WARNING, logger=self._BROKER_LOGGER):
            _build_broker(log).replay()

        assert self._skip_warnings(caplog.records) == []


class TestPositionMetadata:
    def test_side_parsed_from_entry(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(log, [_fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=100.0, side="NO")])
        [pos] = _build_broker(log).replay()
        assert pos.side == Side.NO

    def test_close_time_carried_from_first_fill(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    dollars=100.0,
                    close_time="2026-04-24T04:59:00+00:00",
                ),
                # Second fill with empty metadata must not clobber the first close_time.
                {
                    **_fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=50.0),
                    "metadata": {},
                },
            ],
        )
        [pos] = _build_broker(log).replay()
        assert pos.close_time == "2026-04-24T04:59:00+00:00"
        assert pos.dollars == 150.0


class TestBankroll:
    def test_bankroll_reflects_config(self, tmp_path: Path) -> None:
        broker = ShadowBroker(
            trade_log_path=tmp_path / "log.jsonl",
            configured_bankroll=5000.0,
            today=date(2026, 4, 23),
        )
        assert broker.bankroll == 5000.0


class TestCashOutRows:
    """ShadowBroker handling of cash_out / cash_out_force_applied / cash_out_reverted rows.

    Cash-out v1.1 §12 adds three new event types to ``state/trades.jsonl``:

      * ``cash_out`` and ``cash_out_force_applied`` adjust positions on
        replay — full closes pop the position (like ``settlement``), partial
        closes reduce ``dollars`` by ``cost_basis_offset_dollars``.
      * ``cash_out_reverted`` reverses the most recent cash_out /
        force_applied for the same position — partial reverts add the
        offset back. The full-revert case requires reconstructing cost
        basis from the prior fill row, which is operator-runbook recovery
        territory (the row alone is insufficient) — the replay treats it
        as a no-op.
    """

    @staticmethod
    def _cash_out_row(
        *,
        ticker: str,
        close_type: str,
        cost_basis_offset_dollars: float,
        contracts_offset: int = 25,
        event: str = "cash_out",
        position_key: str | None = None,
        mode: str = "shadow",
    ) -> dict:
        row: dict = {
            "event": event,
            "market_ticker": ticker,
            "market_id": ticker,
            "mode": mode,
            "close_type": close_type,
            "contracts_offset": contracts_offset,
            "cost_basis_offset_dollars": cost_basis_offset_dollars,
        }
        if position_key is not None:
            row["position_key"] = position_key
        return row

    def test_full_cash_out_pops_position(self, tmp_path: Path) -> None:
        """A full cash_out row drops the position like settlement does."""
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=2.34),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    close_type="full",
                    contracts_offset=26,
                    cost_basis_offset_dollars=2.34,
                ),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {}

    def test_partial_cash_out_reduces_dollars(self, tmp_path: Path) -> None:
        """A partial cash_out reduces dollars by cost_basis_offset_dollars."""
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=2.34),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    close_type="partial",
                    cost_basis_offset_dollars=2.25,
                ),
            ],
        )
        result = _dollars_by_key(_build_broker(log))
        assert "KXHIGHNY-26APR24-B70.5" in result, "partial cash-out must NOT pop the position"
        assert result["KXHIGHNY-26APR24-B70.5"] == pytest.approx(0.09, abs=0.01)

    def test_force_applied_and_reverted_compensate(self, tmp_path: Path) -> None:
        """force_applied subtracts dollars; reverted adds them back (partial case).

        Force-applied subtracts $2.25 from $2.34 → $0.09; reverted re-adds
        $2.25 → back to original $2.34.
        """
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=2.34),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    event="cash_out_force_applied",
                    close_type="partial",
                    cost_basis_offset_dollars=2.25,
                ),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    event="cash_out_reverted",
                    close_type="partial",
                    cost_basis_offset_dollars=2.25,
                ),
            ],
        )
        result = _dollars_by_key(_build_broker(log))
        assert "KXHIGHNY-26APR24-B70.5" in result
        assert result["KXHIGHNY-26APR24-B70.5"] == pytest.approx(2.34, abs=0.01)

    def test_force_applied_full_pops_position(self, tmp_path: Path) -> None:
        """cash_out_force_applied with close_type='full' pops the position."""
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=2.34),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    event="cash_out_force_applied",
                    close_type="full",
                    contracts_offset=26,
                    cost_basis_offset_dollars=2.34,
                ),
            ],
        )
        assert _dollars_by_key(_build_broker(log)) == {}

    def test_partial_cash_out_floors_at_zero(self, tmp_path: Path) -> None:
        """A partial cash_out offset larger than remaining dollars floors to 0."""
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=1.00),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    close_type="partial",
                    cost_basis_offset_dollars=2.50,
                ),
            ],
        )
        # Position survives in dict at $0.0; the date safety-net keeps it
        # because the ticker date is in range.
        result = _dollars_by_key(_build_broker(log))
        assert result.get("KXHIGHNY-26APR24-B70.5") == pytest.approx(0.0, abs=0.001)

    def test_full_revert_is_noop_no_basis_to_reconstruct(self, tmp_path: Path) -> None:
        """A standalone cash_out_reverted with close_type='full' is a no-op.

        The row alone is insufficient to reconstruct the prior cost basis
        (which lived on the popped position). Operator runbook covers
        manual recovery for full reverts; replay logs + skips. Here the
        prior fill already established the position, so a full-revert on
        a still-open position must not alter dollars.
        """
        log = tmp_path / "log.jsonl"
        _write_log(
            log,
            [
                _fill(ticker="KXHIGHNY-26APR24-B70.5", dollars=2.34),
                self._cash_out_row(
                    ticker="KXHIGHNY-26APR24-B70.5",
                    event="cash_out_reverted",
                    close_type="full",
                    contracts_offset=26,
                    cost_basis_offset_dollars=2.34,
                ),
            ],
        )
        result = _dollars_by_key(_build_broker(log))
        assert result.get("KXHIGHNY-26APR24-B70.5") == pytest.approx(2.34, abs=0.01)


def test_replay_skips_placement_rows(tmp_path: Path) -> None:
    """replay() must skip event='placement' rows even though they share
    most identity fields with fill rows. Otherwise placement rows would
    be replayed as ghost fills, double-counting positions."""
    log_path = tmp_path / "trades.jsonl"
    # Placement row includes dollars so it would be treated as a fill
    # without an explicit is_pnl_relevant_row guard.
    placement_row = {
        "event": "placement",
        "market_ticker": "KXHIGHNY-26APR24-B70.5",
        "market_id": "KXHIGHNY-26APR24-B70.5",
        "side": "NO",
        "mode": "shadow",
        "dollars": 65.0,
        "strategy": "weather_edge",
        "metadata": {"close_time": "2099-01-01T00:00:00+00:00"},
    }
    fill_row = {
        "event": "fill",
        "market_ticker": "KXHIGHNY-26APR24-B70.5",
        "market_id": "KXHIGHNY-26APR24-B70.5",
        "side": "NO",
        "mode": "shadow",
        "dollars": 65.0,
        "contracts": 5,
        "strategy": "weather_edge",
        "metadata": {"close_time": "2099-01-01T00:00:00+00:00"},
    }
    _write_log(log_path, [placement_row, fill_row])

    broker = ShadowBroker(
        trade_log_path=log_path,
        configured_bankroll=1000.0,
        today=date(2026, 4, 24),
    )
    positions = broker.replay()

    # Only the fill row contributes to the position; the placement row is ignored.
    # If placement rows were treated as fills, dollars would be 130.0 (double-counted).
    matched = [p for p in positions if p.market_ticker == "KXHIGHNY-26APR24-B70.5"]
    assert len(matched) == 1
    assert matched[0].dollars == pytest.approx(65.0)
