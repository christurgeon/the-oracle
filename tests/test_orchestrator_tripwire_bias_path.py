"""Resolver for the bias-table freshness tripwire.

``Orchestrator._tripwire_bias_table_path`` must monitor the artifact the
strategy actually loads — the configured ``weather.bias_table_path`` (the live
NWS-anchored table) or the ERA5 canonical default when that is ``null`` — never
a hard-coded ERA5 path. The 2026-06-21 misdirected-tripwire bug: the loop hard-
coded ``data/gfs_bias/bias_correction.json``, so the alert false-alarmed on the
ERA5 rollback fallback while staying blind to the deployed NWS table aging out.

The resolver only reads ``self._settings.strategies.weather`` and constructs a
``Path``; it is exercised here with a duck-typed ``self`` so the test stays a
fast pure-logic check (no full ``Orchestrator`` / client construction).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from python.orchestrator import Orchestrator


def _fake_self(*, enabled: bool, table_path: str | None) -> SimpleNamespace:
    weather = SimpleNamespace(
        bias_correction_enabled=enabled,
        bias_table_path=table_path,
    )
    settings = SimpleNamespace(strategies=SimpleNamespace(weather=weather))
    return SimpleNamespace(_settings=settings)


def test_resolves_configured_live_path() -> None:
    """The deployed NWS table path flows straight through."""
    fake = _fake_self(enabled=True, table_path="data/nws_bias/bias_correction.json")
    assert Orchestrator._tripwire_bias_table_path(fake) == Path(
        "data/nws_bias/bias_correction.json"
    )


def test_null_config_falls_back_to_era5_default() -> None:
    """``bias_table_path: null`` rolls back to the ERA5 canonical table — the
    tripwire must follow it there (matches ``BiasTable`` default resolution).
    """
    fake = _fake_self(enabled=True, table_path=None)
    assert Orchestrator._tripwire_bias_table_path(fake) == Path(
        "data/gfs_bias/bias_correction.json"
    )


def test_disabled_correction_skips_check() -> None:
    """Bias correction off → nothing is loaded → no artifact to monitor.
    ``None`` makes ``run_all`` skip the freshness check (no spurious alert).
    """
    fake = _fake_self(enabled=False, table_path="data/nws_bias/bias_correction.json")
    assert Orchestrator._tripwire_bias_table_path(fake) is None
