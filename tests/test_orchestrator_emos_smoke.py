"""Orchestrator startup-load smoke_test + alert behavior (M1 audit fixes).

Pre-M1, ``weather_edge.setup()`` did ``EmosTable.load`` -> ``smoke_test`` and,
on failure, downgraded to a disabled sentinel AND dispatched an ERROR-level
Telegram alert (``failure_bucket="emos_load:_global"``) when
``emos_shadow_log: true``.

Post-M1, the load was hoisted into the orchestrator but the smoke_test +
alert behavior was dropped — that gap is the HIGH the code-quality auditor
flagged. These tests pin the restored behavior at the orchestrator load site
(``Orchestrator._build_emos_holder``).

Legacy alert path being mirrored: python/strategies/weather_edge.py:340-350.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from python.data.emos import EmosTable, EmosTableHolder
from python.notifications.base import AlertLevel
from python.orchestrator import Orchestrator


def _valid_payload() -> dict:
    return {
        "fit_date": "2026-05-04",
        "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": -4.0,
                "b": 1.05,
                "c": 15.0,
                "d": 0.4,
                "n_train": 127,
                "leads_pooled": [0, 1, 2],
                "ens_std_p99": 6.5,
            },
        },
    }


def _bare_orch_with_settings(
    *,
    coeffs_path: Path,
    emos_shadow_log: bool,
    notifier,
) -> Orchestrator:
    """Build a minimal Orchestrator skeleton that exposes the attributes
    ``_build_emos_holder`` reads: ``_settings.strategies.weather`` and
    ``_notifier``. All other state is irrelevant for this method."""
    orch = Orchestrator.__new__(Orchestrator)
    weather_cfg = SimpleNamespace(
        emos_coefficients_path=str(coeffs_path),
        emos_shadow_log=emos_shadow_log,
    )
    orch._settings = SimpleNamespace(  # type: ignore[assignment]
        strategies=SimpleNamespace(weather=weather_cfg),
    )
    orch._notifier = notifier  # type: ignore[assignment]
    return orch


@pytest.mark.asyncio
async def test_build_emos_holder_runs_smoke_and_falls_back_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load a valid artifact; force ``smoke_test`` to raise; assert
    ``_build_emos_holder`` downgrades to a disabled sentinel with a
    ``smoke-error:`` reason (matching the legacy weather_edge.setup()
    fallback path)."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_valid_payload()))

    def _raising_smoke(self) -> bool:
        raise RuntimeError("synthetic smoke failure")

    monkeypatch.setattr(EmosTable, "smoke_test", _raising_smoke)

    orch = _bare_orch_with_settings(
        coeffs_path=coeffs_path, emos_shadow_log=False, notifier=None
    )
    holder = await orch._build_emos_holder()

    assert isinstance(holder, EmosTableHolder)
    table = holder.table
    # Smoke failure must collapse to a disabled sentinel — not the loaded table.
    assert not table.is_loaded
    assert table.fallback_reason is not None
    assert table.fallback_reason.startswith("smoke-error:RuntimeError:")
    # The path is preserved so hot-reload can still re-attempt the artifact later.
    assert table._path == coeffs_path


@pytest.mark.asyncio
async def test_build_emos_holder_passes_through_loaded_table(
    tmp_path: Path,
) -> None:
    """Valid payload + passing smoke -> ``_build_emos_holder`` returns a
    loaded table wrapped in the holder, with no alert."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_valid_payload()))

    notifier = AsyncMock()
    orch = _bare_orch_with_settings(
        coeffs_path=coeffs_path, emos_shadow_log=True, notifier=notifier
    )
    holder = await orch._build_emos_holder()

    assert holder.table.is_loaded
    notifier.send_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_emos_holder_dispatches_telegram_alert_on_smoke_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``emos_shadow_log=True`` and smoke fails, an ERROR-level Telegram
    alert fires with ``failure_bucket="emos_load:_global"`` — mirroring
    legacy weather_edge.setup() (python/strategies/weather_edge.py:344-348)."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_valid_payload()))

    def _raising_smoke(self) -> bool:
        raise RuntimeError("synthetic smoke failure")

    monkeypatch.setattr(EmosTable, "smoke_test", _raising_smoke)

    notifier = AsyncMock()
    orch = _bare_orch_with_settings(
        coeffs_path=coeffs_path, emos_shadow_log=True, notifier=notifier
    )
    await orch._build_emos_holder()

    notifier.send_alert.assert_awaited_once()
    call_args = notifier.send_alert.await_args.args
    call_kwargs = notifier.send_alert.await_args.kwargs
    assert "EMOS load failed" in call_args[0]
    assert call_kwargs["level"] == AlertLevel.ERROR
    assert call_kwargs["failure_bucket"] == "emos_load:_global"


@pytest.mark.asyncio
async def test_build_emos_holder_skips_alert_when_shadow_log_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``emos_shadow_log=False``, smoke failure still downgrades to
    disabled, but NO Telegram alert fires (legacy WARNING-only branch at
    python/strategies/weather_edge.py:351-352)."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_valid_payload()))

    def _raising_smoke(self) -> bool:
        raise RuntimeError("synthetic smoke failure")

    monkeypatch.setattr(EmosTable, "smoke_test", _raising_smoke)

    notifier = AsyncMock()
    orch = _bare_orch_with_settings(
        coeffs_path=coeffs_path, emos_shadow_log=False, notifier=notifier
    )
    holder = await orch._build_emos_holder()

    assert not holder.table.is_loaded
    notifier.send_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_emos_holder_no_alert_when_notifier_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Telegram disabled paths leave ``self._notifier is None``; the
    orchestrator must skip the alert silently rather than AttributeError."""
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_valid_payload()))

    def _raising_smoke(self) -> bool:
        raise RuntimeError("synthetic smoke failure")

    monkeypatch.setattr(EmosTable, "smoke_test", _raising_smoke)

    orch = _bare_orch_with_settings(
        coeffs_path=coeffs_path, emos_shadow_log=True, notifier=None
    )
    holder = await orch._build_emos_holder()  # must not raise

    assert not holder.table.is_loaded


@pytest.mark.asyncio
async def test_build_emos_holder_alert_dispatch_failure_does_not_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notifier raising during alert dispatch must NOT crash startup —
    legacy code wraps the send_alert call in try/except (line 343-350).
    """
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(json.dumps(_valid_payload()))

    def _raising_smoke(self) -> bool:
        raise RuntimeError("synthetic smoke failure")

    monkeypatch.setattr(EmosTable, "smoke_test", _raising_smoke)

    notifier = AsyncMock()
    notifier.send_alert.side_effect = RuntimeError("notifier exploded")
    orch = _bare_orch_with_settings(
        coeffs_path=coeffs_path, emos_shadow_log=True, notifier=notifier
    )
    # Must not raise even though send_alert raises.
    holder = await orch._build_emos_holder()
    assert not holder.table.is_loaded
    notifier.send_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_emos_holder_alerts_on_file_missing(
    tmp_path: Path,
) -> None:
    """Missing artifact -> ``EmosTable.load`` returns ``_disabled`` with
    reason ``file-missing``; the alert path fires under ``emos_shadow_log``
    because the legacy ``except Exception`` branch caught the RuntimeError
    reraise for ``not is_loaded`` outcomes."""
    missing_path = tmp_path / "nope.json"
    notifier = AsyncMock()
    orch = _bare_orch_with_settings(
        coeffs_path=missing_path, emos_shadow_log=True, notifier=notifier
    )
    holder = await orch._build_emos_holder()

    assert not holder.table.is_loaded
    assert holder.table.fallback_reason == "file-missing"
    notifier.send_alert.assert_awaited_once()
    call_kwargs = notifier.send_alert.await_args.kwargs
    assert call_kwargs["failure_bucket"] == "emos_load:_global"
