"""Smoke tests for the shared conftest fixtures (Task 0 round-2).

These exercise the simple-helpers path of ``E2EHarness`` end-to-end so we
catch any regression that turns the factory back into a NotImplementedError
trap. The integration-glue methods are also covered (they MUST raise
NotImplementedError with a plan pointer — silent NoOps are explicitly
disallowed).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.conftest import (
    E2EHarness,
    make_minimal_reconciler,
    make_test_risk_manager,
)


def test_e2e_harness_simple_helpers(
    make_e2e_harness: Callable[..., E2EHarness],
) -> None:
    """The factory must construct without raising; the simple helpers must
    write to the trade log + risk-manager state; the integration-glue
    methods must raise NotImplementedError, not silently NoOp.
    """
    h = make_e2e_harness(full_wiring=False)  # MUST NOT raise (MAJOR-1 regression guard)

    # seed_fill writes a fill row.
    h.seed_fill(market_ticker="TEST", side="yes", price=0.5, contracts=10)
    assert h.read_trade_log_event_count("fill") == 1

    # set_position mutates the real RiskManager (MAJOR-2 regression guard).
    h.set_position("TEST", dollars=5.0)
    assert h.risk._positions["TEST"] == 5.0

    # starting_daily_pnl reads the captured value (MAJOR-3 regression guard).
    assert h.starting_daily_pnl == 0.0

    # write_placement_row appends a placement event.
    h.write_placement_row(
        market_ticker="TEST",
        order_id="OID-1",
        strategy="weather_edge",
    )
    assert h.read_trade_log_event_count("placement") == 1

    # read_trade_log_rows returns both rows in order.
    rows = h.read_trade_log_rows()
    assert len(rows) == 2
    assert rows[0]["event"] == "fill"
    assert rows[1]["event"] == "placement"

    # Integration-glue still raises on the simple-helpers path (full_wiring=False).
    # Task 18 implemented these methods behind the full_wiring opt-in; callers
    # that need them must construct with E2EHarness(tmp_path, full_wiring=True).
    with pytest.raises(NotImplementedError, match="full-wiring"):
        h.queue_cash_out_fill(
            market_ticker="TEST",
            hedge_side="no",
            price=0.5,
            contracts=10,
            order_id="X",
        )

    # set_cash_out_apply_enabled raises until reconciler wiring lands.
    with pytest.raises(NotImplementedError, match="Tasks 18/25/26/32"):
        h.set_cash_out_apply_enabled(True)


def test_e2e_harness_default_constructor(tmp_path: Path) -> None:
    """Direct construction (not via fixture) also works."""
    h = E2EHarness(tmp_path)
    assert h.tmp_path == tmp_path
    assert h.starting_bankroll == 10_000.0
    assert h.trade_log.exists()
    assert h.reconciler is None
    assert h.engine is None


def test_make_test_risk_manager_requires_state_dir(tmp_path: Path) -> None:
    """MINOR-11 regression guard: state_dir is required keyword-only,
    no /tmp fallback.
    """
    rm = make_test_risk_manager(bankroll=15_000.0, state_dir=tmp_path)
    assert rm.bankroll == 15_000.0
    assert rm._daily_pnl == 0.0


def test_make_minimal_reconciler_requires_paths(tmp_path: Path) -> None:
    """MINOR-10 regression guard: state_path and trade_log_path are required
    keyword-only, no shared /tmp defaults.
    """
    rec = make_minimal_reconciler(
        state_path=tmp_path / "reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    assert rec is not None
    # The capturing notifier is wired and has the sent_alerts sink.
    # The structural-subtype cast is intentional — the notifier protocol
    # in production doesn't declare sent_alerts (it's a test-only sink).
    assert rec._notifier.sent_alerts == []  # type: ignore[union-attr]


def test_make_minimal_reconciler_capturing_notifier_accepts_positional_message(
    tmp_path: Path,
) -> None:
    """MINOR-7 regression guard: the capturing notifier accepts ``message``
    positionally (matching the production Notifier protocol shape) AND
    keyword-only (matching the v1 _MockNotifier shape).
    """
    import asyncio

    rec = make_minimal_reconciler(
        state_path=tmp_path / "reconciler.json",
        trade_log_path=tmp_path / "trades.jsonl",
    )
    notifier = rec._notifier
    assert notifier is not None

    asyncio.run(notifier.send_alert("positional-msg"))
    asyncio.run(notifier.send_alert(level="warning", message="kw-msg"))  # type: ignore[arg-type]

    assert notifier.sent_alerts == [  # type: ignore[attr-defined]
        (None, "positional-msg"),
        ("warning", "kw-msg"),
    ]
