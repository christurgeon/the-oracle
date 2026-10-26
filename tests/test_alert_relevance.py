"""Tests for the alert-relevance suppression rule in ``Orchestrator._notify_alert``.

Fixtures (`orch`, `notifier`) and the `_stub` strategy factory live in
tests/conftest.py — see the "Alert-relevance fixtures" section there. They
are shared with Tasks 5, 7, 8, 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.conftest import _stub, _StubNotifier

if TYPE_CHECKING:
    from python.orchestrator import Orchestrator


async def test_owner_none_always_dispatches(orch: Orchestrator, notifier: _StubNotifier) -> None:
    assert await orch._notify_alert("infra", owner=None) is True
    assert len(notifier.sent) == 1


async def test_shadow_only_owner_is_suppressed(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=True)]
    assert await orch._notify_alert("cal", owner="weather_edge") is False
    assert notifier.sent == []


async def test_live_owner_dispatches(orch: Orchestrator, notifier: _StubNotifier) -> None:
    orch._strategies = [_stub("favorite_bias", enabled=True, shadow_only=False)]
    assert await orch._notify_alert("cal", owner="favorite_bias") is True
    assert len(notifier.sent) == 1


async def test_disabled_owner_is_suppressed(orch: Orchestrator, notifier: _StubNotifier) -> None:
    """The enabled= arm of configured_to_trade, not just shadow_only=.

    Without this, only the shadow_only branch of _stub is ever exercised —
    a strategy disabled outright (enabled=False) must suppress identically.
    """
    orch._strategies = [_stub("weather_edge", enabled=False, shadow_only=False)]
    assert await orch._notify_alert("cal", owner="weather_edge") is False
    assert notifier.sent == []


async def test_unresolved_owner_dispatches_and_warns(
    orch: Orchestrator, notifier: _StubNotifier, caplog: pytest.LogCaptureFixture
) -> None:
    """Fail OPEN. A typo must be noisy, never silent."""
    orch._strategies = []
    with caplog.at_level("WARNING"):
        assert await orch._notify_alert("cal", owner="typo") is True
    assert "matches no registered strategy" in caplog.text
    assert len(notifier.sent) == 1


async def test_property_raising_still_dispatches(
    orch: Orchestrator, notifier: _StubNotifier
) -> None:
    """Fail open on exception too — _notify_alert must never raise."""

    class _Boom:
        name = "boom"

        @property
        def configured_to_trade(self) -> bool:
            raise RuntimeError("kaboom")

    orch._strategies = [_Boom()]  # type: ignore[list-item]  # duck-typed double
    assert await orch._notify_alert("cal", owner="boom") is True
    assert len(notifier.sent) == 1


async def test_missing_notifier_dispatches_true_without_sending(orch: Orchestrator) -> None:
    """A missing notifier (Telegram disabled, or _start_telegram failed) is
    NOT suppression — it must return True so callers still record the alert
    as handled. Returning False here would be indistinguishable from a real
    suppression to a caller checking the return value alone, and — per
    _start_telegram (orchestrator.py:~2388) setting self._notifier = None on
    ANY startup failure — would let a hostile-looking but purely
    infrastructural outage silently defeat Task 5's dedup/persist gating.
    """
    orch._notifier = None
    assert await orch._notify_alert("infra", owner=None) is True


async def test_suppression_is_counted(orch: Orchestrator) -> None:
    orch._strategies = [_stub("weather_edge", enabled=True, shadow_only=True)]
    await orch._notify_alert(
        "[bias_table_stale:x] m", owner="weather_edge", code="bias_table_stale:x"
    )
    assert orch._alert_suppressions[("bias_table_stale:x", "weather_edge")] == 1


async def test_unshadowing_restores_dispatch(orch: Orchestrator, notifier: _StubNotifier) -> None:
    s = _stub("weather_edge", enabled=True, shadow_only=True)
    orch._strategies = [s]
    assert await orch._notify_alert("cal", owner="weather_edge") is False
    s._config["shadow_only"] = False
    assert await orch._notify_alert("cal", owner="weather_edge") is True


def test_only_the_two_analytical_sites_pass_owner() -> None:
    """The kill switch must never become suppressible.

    ``owner=`` and ``code=`` are each INDEPENDENTLY sufficient to make a
    ``_notify_alert`` call suppressible: ``owner=`` feeds `_alert_is_relevant`
    (owner-liveness suppression), and ``code=`` alone gates the ack rule —
    `_notify_alert` checks ``if code is not None and self._ack_suppresses(...)``
    with no owner involved (orchestrator.py:2590-ish). A guard that only
    detects ``owner=`` would pass a kill-switch call that grew a bare
    ``code="kill_switch"`` — no owner needed to make that call ack-suppressible.
    So this guard flags a call site that passes EITHER keyword.

    Whole-file assertion: if any call site in python/orchestrator.py other
    than the two named below ever grows an owner= or code= argument, this
    fails — covering _notify_warning_telegram / _notify_critical_telegram and
    every other in-file caller. It does NOT cover python/data/contract_watcher.py
    (the AST tree parsed here is orchestrator.py only) — the two real
    ``alert_fn(message, AlertLevel.WARNING)`` calls there
    (contract_watcher.py:106,111) are positional, and their actual
    protection is the keyword-only ``*`` in _notify_alert's signature, not
    this guard.

    Asserts against ``_dispatch_tripwire_alerts`` rather than
    ``_audit_tripwire_loop`` because Task 5 extracted the tripwire dispatch
    body (with its ``owner=``-passing call) into that new method — the
    dispatch call site itself no longer passes ``owner=`` directly.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("python/orchestrator.py").read_text())

    class _Attribute(ast.NodeVisitor):
        """Attribute each call to its NEAREST enclosing function.

        A plain ast.walk from every FunctionDef double-counts calls inside
        any nested function def — a stack gives the innermost owner only.
        """

        def __init__(self) -> None:
            self.stack: list[str] = []
            self.offenders: set[str] = set()

        def _fn(self, node: ast.AST) -> None:
            self.stack.append(node.name)  # type: ignore[attr-defined]
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _fn  # noqa: N815 — ast.NodeVisitor naming
        visit_AsyncFunctionDef = _fn  # noqa: N815 — ast.NodeVisitor naming

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "_notify_alert"
                and any(kw.arg in ("owner", "code") for kw in node.keywords)
            ):
                self.offenders.add(self.stack[-1] if self.stack else "<module>")
            self.generic_visit(node)

    visitor = _Attribute()
    visitor.visit(tree)
    offenders = visitor.offenders

    assert offenders == {"_on_calibration_drift", "_dispatch_tripwire_alerts"}, (
        "only the calibration-drift and tripwire-sweep sites may pass owner= or code=; "
        f"found: {sorted(offenders)}"
    )
