"""Tests for scripts/revert_cash_out.py (v1.1 §15)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "revert_cash_out.py"


def test_script_aborts_when_orchestrator_running(tmp_path, monkeypatch):
    """The script's first action is a pgrep check; abort if bot is up.

    We can't actually have an orchestrator running during tests, so we test
    by import-and-call the helper directly with a mocked subprocess.
    """
    import scripts.revert_cash_out as rcm

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="12345\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with __import__('pytest').raises(SystemExit) as exc_info:
        rcm._abort_if_bot_running()
    assert exc_info.value.code == 2


def test_script_parses_required_args():
    """--order-id and a mode flag are required."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--reset-dedup"],
        capture_output=True,
    )
    assert result.returncode != 0
    # argparse error message goes to stderr.
    assert b"--order-id" in result.stderr or b"required" in result.stderr


def test_script_aborts_without_yes_i_am_sure(tmp_path):
    """Missing --yes-i-am-sure returns 3."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--reset-dedup",
         "--order-id", "OID-1", "--state-dir", str(tmp_path)],
        capture_output=True,
    )
    assert result.returncode == 3
    assert b"yes-i-am-sure" in result.stderr


def test_reset_dedup_removes_oid_from_all_five_sets(tmp_path):
    """--reset-dedup removes the oid from all five _co_*_oids arrays."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    fills_state = state_dir / "fills_reconciler.json"
    fills_state.write_text(json.dumps({
        "schema_version": 1,
        "last_safe_ts_unix": 0,
        "co_race_lost_oids": ["OID-1", "OID-2"],
        "co_zero_basis_oids": ["OID-1"],
        "co_invalid_partial_oids": ["OID-1"],
        "co_pnl_cap_oids": ["OID-1"],
        "co_reverted_oids": ["OID-1"],
    }))
    # Also need an empty trade log to satisfy any cross-checks.
    (state_dir / "trades.jsonl").write_text("")
    # And risk_state.json to satisfy _abort_if_bot_running's state-dir check.
    (state_dir / "risk_state.json").write_text("{}")

    import scripts.revert_cash_out as rcm
    rc = rcm._reset_dedup(state_dir, "OID-1")
    assert rc == 0

    new_state = json.loads(fills_state.read_text())
    # OID-1 removed from all five sets; OID-2 preserved in co_race_lost_oids.
    assert "OID-1" not in new_state["co_race_lost_oids"]
    assert "OID-2" in new_state["co_race_lost_oids"]
    assert "OID-1" not in new_state["co_zero_basis_oids"]
    assert "OID-1" not in new_state["co_invalid_partial_oids"]
    assert "OID-1" not in new_state["co_pnl_cap_oids"]
    assert "OID-1" not in new_state["co_reverted_oids"]


def test_reset_dedup_writes_atomically(tmp_path):
    """The state file is written via temp + replace (atomic write)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    fills_state = state_dir / "fills_reconciler.json"
    original = {
        "schema_version": 1,
        "last_safe_ts_unix": 1234,
        "co_pnl_cap_oids": ["OID-1"],
    }
    fills_state.write_text(json.dumps(original))

    import scripts.revert_cash_out as rcm
    rcm._reset_dedup(state_dir, "OID-1")

    # File still readable + correctly formatted.
    after = json.loads(fills_state.read_text())
    assert after["schema_version"] == 1
    assert after["last_safe_ts_unix"] == 1234  # other fields preserved
    assert after["co_pnl_cap_oids"] == []


def test_revert_mode_writes_cash_out_reverted_row_and_adjusts_state(tmp_path):
    """--revert path: finds cash_out row, writes cash_out_reverted, adjusts state."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Seed a state with an applied full cash-out.
    trade_log = state_dir / "trades.jsonl"
    trade_log.write_text(json.dumps({
        "event": "cash_out",
        "order_id": "CO-1",
        "market_ticker": "KXT",
        "position_key": "KXT",
        "side": "YES",
        "mode": "live",
        "timestamp": "2026-05-17T17:18:00Z",
        "applied_at": "2026-05-17T17:18:30Z",
        "closed_side": "YES",
        "hedge_side": "NO",
        "close_type": "full",
        "contracts_offset": 26,
        "contracts_residual": 0,
        "hedge_price": 0.86,
        "hedge_proceeds": 22.36,
        "hedge_fees": 0.10,
        "cost_basis_offset_dollars": 2.34,
        "cost_basis_offset_fees": 0.10,
        "cost_basis_residual_dollars": 0.0,
        "cost_basis_residual_fees": 0.0,
        "realized_pnl": 1.14,
        "metadata": {"strategy": "weather_edge", "classification": "BUY_CASH_OUT"},
    }) + "\n")

    risk_state_path = state_dir / "risk_state.json"
    risk_state_path.write_text(json.dumps({
        "bankroll": 101.14,
        "daily_pnl": 1.14,
        "peak_bankroll": 101.14,
        "positions": {},
        "strategy_trackers": {
            "weather_edge": {
                "trades": [{"pnl": 1.14, "won": True, "ts": 12345}],
            },
        },
    }))

    fills_state_path = state_dir / "fills_reconciler.json"
    fills_state_path.write_text(json.dumps({
        "schema_version": 1,
        "last_safe_ts_unix": 0,
        "co_reverted_oids": [],
    }))

    import scripts.revert_cash_out as rcm
    rc = rcm._revert(state_dir, "CO-1", "test revert")
    assert rc == 0

    # 1. Trade log now has BOTH cash_out and cash_out_reverted rows.
    rows = [json.loads(line) for line in trade_log.read_text().splitlines() if line.strip()]
    cash_out_rows = [r for r in rows if r.get("event") == "cash_out"]
    reverted_rows = [r for r in rows if r.get("event") == "cash_out_reverted"]
    assert len(cash_out_rows) == 1
    assert len(reverted_rows) == 1

    rev = reverted_rows[0]
    assert rev["reverts_order_id"] == "CO-1"
    assert rev["realized_pnl"] == pytest.approx(-1.14, abs=0.001)  # SIGN-FLIPPED
    assert rev["close_type"] == "full"
    assert rev["contracts_offset"] == 26
    assert rev["metadata"]["reverts"] == "CO-1"
    assert rev["metadata"]["reason"] == "test revert"

    # 2. risk_state.json adjusted.
    new_risk = json.loads(risk_state_path.read_text())
    assert new_risk["daily_pnl"] == pytest.approx(0.0, abs=0.001)  # 1.14 - 1.14
    # The matching tracker entry was removed.
    assert len(new_risk["strategy_trackers"]["weather_edge"]["trades"]) == 0

    # 3. _co_reverted_oids includes CO-1.
    new_fills = json.loads(fills_state_path.read_text())
    assert "CO-1" in new_fills["co_reverted_oids"]


def test_revert_mode_fails_when_cash_out_row_not_found(tmp_path):
    """If the order_id has no matching cash_out row, return error code."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "trades.jsonl").write_text("")  # empty
    (state_dir / "risk_state.json").write_text("{}")
    (state_dir / "fills_reconciler.json").write_text(json.dumps({
        "schema_version": 1, "last_safe_ts_unix": 0, "co_reverted_oids": [],
    }))

    import scripts.revert_cash_out as rcm
    rc = rcm._revert(state_dir, "MISSING", "test")
    assert rc == 5  # error code for "no cash_out row found"


def test_apply_mode_writes_cash_out_force_applied_row(tmp_path, monkeypatch):
    """--apply reads the fill from a mocked /portfolio/fills, computes
    realized_pnl using _apply_cash_out's formula, writes a
    cash_out_force_applied row, and mutates risk_state.json directly.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Seed the trade log with an entry FILL row that gives us a cost basis.
    trade_log = state_dir / "trades.jsonl"
    trade_log.write_text(json.dumps({
        "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
        "side": "YES", "mode": "live",
        "fill_price": 0.05, "contracts": 1000, "fees_from_api": 0.50,
        "metadata": {"strategy": "weather_edge"},
    }) + "\n")

    risk_state = state_dir / "risk_state.json"
    risk_state.write_text(json.dumps({
        "bankroll": 100.0, "daily_pnl": 0.0, "peak_bankroll": 100.0,
        "positions": {"KXT": 50.0},
        "strategy_trackers": {"weather_edge": {"trades": []}},
    }))

    fills_state = state_dir / "fills_reconciler.json"
    fills_state.write_text(json.dumps({
        "schema_version": 1, "last_safe_ts_unix": 0,
        "co_pnl_cap_oids": ["CO-1"],
        "cash_out_max_pnl_apply_dollars": 500.0,
    }))

    # Monkeypatch the fill-fetch seam so the test doesn't hit Kalshi.
    import scripts.revert_cash_out as rcm
    fake_fill = {
        "order_id": "CO-1", "market_ticker": "KXT", "side": "no",
        "count_fp": 1000, "no_price_dollars": 0.99, "fee_cost": 0.50,
        "action": "buy", "is_taker": True,
        "created_time": "2026-05-17T17:18:00Z",
    }
    monkeypatch.setattr(rcm, "_fetch_fill_for_oid",
                        lambda state_dir, oid: fake_fill)

    rc = rcm._apply(state_dir, "CO-1", "operator confirmed legit P&L")
    assert rc == 0

    # Verify cash_out_force_applied row appended.
    rows = [json.loads(line) for line in trade_log.read_text().splitlines() if line.strip()]
    force_rows = [r for r in rows if r.get("event") == "cash_out_force_applied"]
    assert len(force_rows) == 1
    row = force_rows[0]
    assert row["close_type"] == "full"  # 1000 == cb.total_contracts
    assert row["contracts_offset"] == 1000
    assert row["realized_pnl"] == pytest.approx(-41.0, abs=0.5)
    # Metadata captures operator + bypassed cap.
    assert row["metadata"]["bypassed_cap_dollars"] == 500.0
    assert row["metadata"]["force_applied_for_oid"] == "CO-1"

    # Verify risk state mutated.
    new_state = json.loads(risk_state.read_text())
    # realized_pnl: 1000 * 1.00 - 50.00 - 0.50 - (1000 * 0.99) - 0.50
    #            = 1000 - 50 - 0.50 - 990 - 0.50 = -41.00
    assert new_state["daily_pnl"] == pytest.approx(-41.0, abs=0.5)
    # CRITICAL #2 regression guard: --apply must NOT mutate positions, bankroll,
    # or peak_bankroll, even when the (test-only) state file happens to contain
    # them.  Those fields are not in the persisted production schema and
    # self-heal from Kalshi on next hydrate (spec §15 line 1056).  Mutating them
    # corrupts peak_bankroll on real state files.
    assert new_state["positions"] == {"KXT": 50.0}, (
        "--apply must leave seeded 'positions' untouched (spec §15 1056)"
    )
    assert new_state["bankroll"] == 100.0, (
        "--apply must leave seeded 'bankroll' untouched (spec §15 1056)"
    )
    assert new_state["peak_bankroll"] == 100.0, (
        "--apply must leave seeded 'peak_bankroll' untouched (spec §15 1056)"
    )
    # And the strategy tracker DID record the new trade entry.
    new_trades = new_state["strategy_trackers"]["weather_edge"]["trades"]
    assert len(new_trades) == 1
    assert new_trades[0]["pnl"] == pytest.approx(-41.0, abs=0.5)

    # Verify oid removed from co_pnl_cap_oids.
    new_fills = json.loads(fills_state.read_text())
    assert "CO-1" not in new_fills["co_pnl_cap_oids"]


def test_telegram_post_failure_does_not_abort_mutation(monkeypatch, capsys):
    """If Telegram POST fails (network down), script logs error and continues.
    Best-effort contract - mutation completes regardless.
    """
    import scripts.revert_cash_out as rcm

    class FakeHttpx:
        @staticmethod
        def post(*args, **kwargs):
            raise ConnectionError("network down")

    monkeypatch.setattr(rcm, "httpx", FakeHttpx, raising=False)
    monkeypatch.setenv("ORACLE_TELEGRAM_BOT_TOKEN", "fake-token")
    monkeypatch.setenv("ORACLE_TELEGRAM_OWNER_CHAT_ID", "12345")

    # Call the helper directly - it should NOT raise.
    rcm._telegram_post_best_effort("test message")

    # Verify stderr captured the failure log.
    captured = capsys.readouterr()
    assert "Telegram POST failed" in captured.err
    assert "mutation completed regardless" in captured.err


def test_telegram_post_skipped_when_credentials_missing(monkeypatch, capsys):
    """No token or no chat_id -> log skip + return without POSTing."""
    import scripts.revert_cash_out as rcm

    monkeypatch.delenv("ORACLE_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ORACLE_TELEGRAM_OWNER_CHAT_ID", raising=False)

    rcm._telegram_post_best_effort("test message")

    captured = capsys.readouterr()
    assert "Telegram credentials not configured" in captured.err


# ---------------------------------------------------------------------------
# Regression tests for the Tasks 29-31 code-auditor review (CRITICAL #1-3).
# ---------------------------------------------------------------------------


def test_revert_mode_refuses_double_revert(tmp_path):
    """CRITICAL #3: Running --revert twice on the same oid should refuse + exit 9."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    trade_log = state_dir / "trades.jsonl"
    trade_log.write_text("\n".join([
        json.dumps({
            "event": "cash_out", "order_id": "CO-1",
            "market_ticker": "KXT", "position_key": "KXT", "side": "YES",
            "mode": "live", "timestamp": "2026-05-17T17:18:00Z",
            "applied_at": "2026-05-17T17:18:30Z",
            "closed_side": "YES", "hedge_side": "NO", "close_type": "full",
            "contracts_offset": 26, "contracts_residual": 0,
            "cost_basis_offset_dollars": 2.34, "cost_basis_offset_fees": 0.10,
            "realized_pnl": 1.14,
            "metadata": {"strategy": "weather_edge"},
        }),
        json.dumps({
            "event": "cash_out_reverted", "order_id": "REV-1",
            "reverts_order_id": "CO-1",
            "market_ticker": "KXT", "position_key": "KXT", "mode": "live",
            "realized_pnl": -1.14,
            "metadata": {"reverts": "CO-1"},
        }),
    ]) + "\n")
    (state_dir / "risk_state.json").write_text("{}")
    (state_dir / "fills_reconciler.json").write_text(
        json.dumps({"co_reverted_oids": ["CO-1"]})
    )

    import scripts.revert_cash_out as rcm
    rc = rcm._revert(state_dir, "CO-1", "second revert attempt")
    assert rc == 9


def test_apply_mode_refuses_double_apply(tmp_path, monkeypatch):
    """CRITICAL #3: Running --apply twice on the same oid should refuse + exit 10."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    trade_log = state_dir / "trades.jsonl"
    trade_log.write_text("\n".join([
        json.dumps({
            "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
            "side": "YES", "mode": "live",
            "fill_price": 0.05, "contracts": 1000, "fees_from_api": 0.50,
            "metadata": {"strategy": "weather_edge"},
        }),
        json.dumps({
            "event": "cash_out_force_applied", "order_id": "FA-1",
            "market_ticker": "KXT", "position_key": "KXT", "side": "YES",
            "mode": "live", "close_type": "full",
            "contracts_offset": 1000, "realized_pnl": -41.0,
            "metadata": {"force_applied_for_oid": "CO-1", "operator": "test"},
        }),
    ]) + "\n")
    (state_dir / "risk_state.json").write_text(json.dumps({
        "daily_pnl": -41.0,
        "strategy_trackers": {"weather_edge": {"trades": []}},
    }))
    (state_dir / "fills_reconciler.json").write_text(
        json.dumps({"co_pnl_cap_oids": []})
    )

    import scripts.revert_cash_out as rcm
    fake_fill = {
        "order_id": "CO-1", "market_ticker": "KXT", "side": "no",
        "count_fp": 1000, "no_price_dollars": 0.99, "fee_cost": 0.50,
        "action": "buy", "is_taker": True,
    }
    monkeypatch.setattr(rcm, "_fetch_fill_for_oid",
                        lambda state_dir, oid: fake_fill)

    rc = rcm._apply(state_dir, "CO-1", "second apply attempt")
    assert rc == 10


def test_apply_mode_uses_isoformat_ts_for_strategy_trackers(tmp_path, monkeypatch):
    """CRITICAL #1: trades[].ts must be ISO-format string, not Unix float,
    so risk persistence.deserialize_risk_state can read it.

    The production deserializer calls ``datetime.fromisoformat(t["ts"])``
    (python/risk/persistence.py:134); a float raises TypeError which is
    swallowed by the blanket except (line 156), discarding ALL risk state.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "trades.jsonl").write_text(json.dumps({
        "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
        "side": "YES", "mode": "live",
        "fill_price": 0.05, "contracts": 1000, "fees_from_api": 0.50,
        "metadata": {"strategy": "weather_edge"},
    }) + "\n")
    (state_dir / "risk_state.json").write_text(json.dumps({
        "daily_pnl": 0.0,
        "strategy_trackers": {"weather_edge": {"trades": []}},
    }))
    (state_dir / "fills_reconciler.json").write_text(json.dumps({
        "co_pnl_cap_oids": ["CO-1"],
    }))

    import scripts.revert_cash_out as rcm
    fake_fill = {
        "order_id": "CO-1", "market_ticker": "KXT", "side": "no",
        "count_fp": 1000, "no_price_dollars": 0.99, "fee_cost": 0.50,
        "action": "buy", "is_taker": True,
    }
    monkeypatch.setattr(rcm, "_fetch_fill_for_oid",
                        lambda state_dir, oid: fake_fill)

    rc = rcm._apply(state_dir, "CO-1", "test")
    assert rc == 0

    new_state = json.loads((state_dir / "risk_state.json").read_text())
    trades = new_state["strategy_trackers"]["weather_edge"]["trades"]
    assert len(trades) == 1
    ts = trades[0]["ts"]
    assert isinstance(ts, str), (
        f"trades[].ts must be ISO str for production deser; got "
        f"{type(ts).__name__}={ts!r}"
    )
    # Verify it's parseable by datetime.fromisoformat (production code's path).
    from datetime import datetime
    datetime.fromisoformat(ts)


def test_apply_mode_does_not_mutate_positions_or_bankroll(tmp_path, monkeypatch):
    """CRITICAL #2: --apply must NOT touch positions/bankroll/peak_bankroll.

    Those keys are not in the persisted-state schema and self-heal from Kalshi
    on next hydrate (spec §15 line 1056).  Mutating them silently corrupts
    peak_bankroll because the script computes ``max(real_peak, 0 + realized_pnl)``
    using a 0-default bankroll on the real schema.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "trades.jsonl").write_text(json.dumps({
        "event": "fill", "position_key": "KXT", "market_ticker": "KXT",
        "side": "YES", "mode": "live",
        "fill_price": 0.05, "contracts": 1000, "fees_from_api": 0.50,
        "metadata": {"strategy": "weather_edge"},
    }) + "\n")
    # Production state file does NOT contain "positions" or "bankroll" keys.
    risk_state_initial = {
        "daily_pnl": 0.0,
        "peak_bankroll": 100.0,  # MUST stay 100.0; we don't touch it
        "strategy_trackers": {"weather_edge": {"trades": []}},
    }
    (state_dir / "risk_state.json").write_text(json.dumps(risk_state_initial))
    (state_dir / "fills_reconciler.json").write_text(json.dumps({
        "co_pnl_cap_oids": ["CO-1"],
    }))

    import scripts.revert_cash_out as rcm
    fake_fill = {
        "order_id": "CO-1", "market_ticker": "KXT", "side": "no",
        "count_fp": 1000, "no_price_dollars": 0.99, "fee_cost": 0.50,
        "action": "buy", "is_taker": True,
    }
    monkeypatch.setattr(rcm, "_fetch_fill_for_oid",
                        lambda state_dir, oid: fake_fill)

    rc = rcm._apply(state_dir, "CO-1", "test")
    assert rc == 0

    new_state = json.loads((state_dir / "risk_state.json").read_text())
    assert "positions" not in new_state, "--apply must not add 'positions' key"
    assert "bankroll" not in new_state, "--apply must not add 'bankroll' key"
    assert new_state["peak_bankroll"] == 100.0, (
        "--apply must not mutate peak_bankroll"
    )
    # daily_pnl IS mutated (this is correct).
    assert new_state["daily_pnl"] == pytest.approx(-41.0, abs=0.5)
    # strategy_trackers IS mutated.
    assert len(new_state["strategy_trackers"]["weather_edge"]["trades"]) == 1
