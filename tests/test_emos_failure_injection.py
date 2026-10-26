"""Tests for WeatherEdgeStrategy.setup() EMOS load failure handling (PR-2 / C7).

The strategy MUST never refuse to start because of an EMOS coefficients
issue — every failure mode falls back to identity (no shadow, no harm).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from python.strategies.weather_edge import WeatherEdgeStrategy


def _make_strategy(tmp_path: Path, **cfg: object) -> WeatherEdgeStrategy:
    """Construct a minimal WeatherEdgeStrategy bound to ``tmp_path`` as
    the state directory. ``setup()`` writes ``forecasts_raw.jsonl`` here."""
    strat = WeatherEdgeStrategy(
        config=dict(cfg),
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
    )
    # Override the production ``Settings()`` instance so ``state_dir`` points
    # at the test tmpdir. Only ``state_dir`` is read by setup().
    strat._settings = SimpleNamespace(state_dir=str(tmp_path))  # type: ignore[assignment]
    return strat


@pytest.mark.asyncio
async def test_setup_with_missing_coefficients_falls_back_to_identity(tmp_path: Path):
    """Missing artifact => setup() completes silently, EMOS off."""
    strat = _make_strategy(
        tmp_path,
        emos_coefficients_path=str(tmp_path / "missing.json"),
    )
    await strat.setup()
    assert strat.emos_table is None or not strat.emos_table.is_loaded


@pytest.mark.asyncio
async def test_setup_with_corrupt_artifact_keeps_emos_disabled(tmp_path: Path):
    """Corrupt JSON => EmosTable.load returns disabled (parse-error). The
    strategy logs and falls back to identity."""
    coeffs = tmp_path / "coefficients.json"
    coeffs.write_text("{not valid json")
    strat = _make_strategy(
        tmp_path,
        emos_coefficients_path=str(coeffs),
        emos_shadow_log=False,
    )
    await strat.setup()
    assert strat.emos_table is None or not strat.emos_table.is_loaded


@pytest.mark.asyncio
async def test_setup_with_valid_artifact_loads_table(tmp_path: Path):
    """Happy path: a valid coefficients.json loads, smoke-tests cleanly,
    and the strategy ends up with a usable EmosTable."""
    coeffs = tmp_path / "coefficients.json"
    coeffs.write_text(
        json.dumps(
            {
                "fit_date": "2026-05-04",
                "schema_version": 1,
                "cells": {
                    "AUS/high": {
                        "a": 0.0,
                        "b": 1.0,
                        "c": 1.0,
                        "d": 0.0,
                        "n_train": 50,
                        "leads_pooled": [0, 1, 2],
                        "ens_std_p99": 5.0,
                    },
                },
            }
        )
    )
    strat = _make_strategy(
        tmp_path,
        emos_coefficients_path=str(coeffs),
        emos_shadow_log=True,
    )
    await strat.setup()
    assert strat.emos_table is not None and strat.emos_table.is_loaded
    # Writer is wired regardless of artifact load success — the side stream is
    # always cheap and the orchestrator side-stream write checks both writer
    # presence AND a non-empty raw_member_sink.
    assert strat._forecasts_raw_writer is not None


@pytest.mark.asyncio
async def test_setup_with_corrupt_artifact_emits_alert_when_shadow_log_on(tmp_path: Path):
    """When ``emos_shadow_log`` is on AND a notifier is wired, a load failure
    fires an ERROR-level alert with ``failure_bucket=emos_load:_global``."""

    class _RecordingNotifier:
        def __init__(self) -> None:
            self.alerts: list[dict[str, object]] = []

        async def send_alert(
            self,
            message: str,
            *,
            level=None,
            failure_bucket: str | None = None,
        ) -> None:
            self.alerts.append(
                {
                    "message": message,
                    "level": level,
                    "failure_bucket": failure_bucket,
                }
            )

    notifier = _RecordingNotifier()
    coeffs = tmp_path / "coefficients.json"
    coeffs.write_text("{not valid json")
    strat = WeatherEdgeStrategy(
        config={
            "emos_coefficients_path": str(coeffs),
            "emos_shadow_log": True,
        },
        kalshi_client=None,  # type: ignore[arg-type]
        weather_client=None,  # type: ignore[arg-type]
        bankroll=1000,
        notifier=notifier,  # type: ignore[arg-type]
    )
    strat._settings = SimpleNamespace(state_dir=str(tmp_path))  # type: ignore[assignment]
    await strat.setup()

    assert len(notifier.alerts) == 1
    assert notifier.alerts[0]["failure_bucket"] == "emos_load:_global"
    # ERROR-level by AlertLevel enum value
    from python.notifications.base import AlertLevel

    assert notifier.alerts[0]["level"] == AlertLevel.ERROR


# ---------------------------------------------------------------------------
# C15 — operator-drill failure injection (continued)
#
# Wave 3 covered: missing artifact, corrupt-at-startup, valid-load happy path,
# and ERROR-Telegram dispatch under flag=on. The two drills below close the
# remaining production-on-fire scenarios:
#   1. Disk-full on the side-stream writer mid-scan.
#   2. Corrupt-artifact dropped between scans (mid-run hot-reload failure).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disk_full_on_writer_logs_error_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Disk-full drill: ``ForecastsRawWriter.write_batch`` MUST handle
    ``OSError`` from ``os.write`` gracefully — set ``last_error_at``, log
    ERROR, return without raising, leave ``success_count`` unchanged at 0.
    This is the contract the strategy's ``asyncio.to_thread(write_batch, …)``
    relies on so a full disk never crashes the trading loop.
    """
    from python.utils.jsonl import ForecastsRawWriter

    writer = ForecastsRawWriter(tmp_path / "forecasts_raw.jsonl")
    assert writer.success_count == 0
    assert writer.last_error_at is None

    # Patch os.write at the module level used by ForecastsRawWriter so the
    # open() succeeds but the per-row write raises (ENOSPC).
    def _raise_no_space(*_args, **_kwargs):  # pragma: no cover - exception path
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("python.utils.jsonl.os.write", _raise_no_space)

    rows: list[dict[str, object]] = [
        {
            "scan_id": "scan-abc",
            "city": "AUS",
            "var": "high",
            "forecast_date": "2026-05-06",
            "lead": 0,
            "gfs_members_raw": [70.0] * 30,
            "gfs_ens_mean_raw_ddof1": 70.0,
            "gfs_ens_std_raw_ddof1": 1.0,
            "ecmwf_members_raw": [70.0] * 50,
            "ecmwf_ens_mean_raw_ddof1": 70.0,
            "ecmwf_ens_std_raw_ddof1": 1.0,
            "gfs_bias_correction_f": 0.0,
        }
    ]

    caplog.set_level("ERROR", logger="python.utils.jsonl")
    # MUST NOT raise — the strategy's side-stream contract requires graceful
    # absorption of disk failures.
    writer.write_batch(rows)

    # Contract:
    #   - success_count never advanced past 0 (write failed before increment).
    #   - last_error_at was set to a unix timestamp.
    #   - ERROR log fired with "write failed" in the body.
    #   - No exception escaped to the caller.
    assert writer.success_count == 0
    assert writer.last_error_at is not None
    assert writer.last_error_at > 0
    assert any(
        "forecasts_raw write failed" in record.message
        for record in caplog.records
        if record.levelname == "ERROR"
    )


@pytest.mark.asyncio
async def test_corrupt_artifact_mid_run_keeps_prior_table_then_recovers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Corrupt-mid-run drill: a strategy starts with a valid coefficients
    file (loaded at setup). Between scans an operator-induced bad artifact
    is dropped on disk (e.g. ``b=2.0`` violates the ``[0.3, 1.5]`` bound).
    ``check_and_reload`` MUST return ``None`` (load failure on bounds), the
    prior in-memory table stays intact, and an ERROR log fires. When the
    operator restores a valid artifact, the NEXT ``check_and_reload``
    returns a fresh instance with a different hash.
    """
    import time

    from python.data.emos import EmosTable

    coeffs = tmp_path / "coefficients.json"
    valid_payload: dict[str, object] = {
        "fit_date": "2026-05-06",
        "training_window_days": 46,
        "lead_handling": "pooled",
        "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": -4.0,
                "b": 1.05,
                "c": 15.0,
                "d": 0.4,
                "n_train": 127,
                "leads_pooled": [0, 1, 2],
                "ens_std_p99": 6.0,
            },
        },
    }
    coeffs.write_text(json.dumps(valid_payload))

    table = EmosTable.load(coeffs)
    assert table.is_loaded, f"setup must load cleanly; got {table.fallback_reason}"
    pre_hash = table.artifact_hash
    pre_coeffs = table.get(city="AUS", var="high", lead=0)
    assert pre_coeffs is not None
    pre_b = pre_coeffs.b

    # --- Drop a bad artifact (b out of bounds → _BoundsError on parse) ---
    bad_payload: dict[str, object] = json.loads(json.dumps(valid_payload))
    bad_cells = bad_payload["cells"]  # type: ignore[index]
    bad_cells["AUS/high"]["b"] = 2.0  # type: ignore[index]
    time.sleep(0.01)  # advance mtime
    coeffs.write_text(json.dumps(bad_payload))

    caplog.set_level("ERROR", logger="python.data.emos")
    result_bad = table.check_and_reload()
    # Contract: bad reload returns None, prior table preserved.
    assert result_bad is None
    assert table.artifact_hash == pre_hash
    prior_coeffs = table.get(city="AUS", var="high", lead=0)
    assert prior_coeffs is not None
    assert prior_coeffs.b == pre_b
    assert table.is_loaded  # still usable

    # An ERROR log fired (load failure mentions "load" in the body).
    assert any(
        ("load failed" in r.message.lower() or "smoke failed" in r.message.lower())
        for r in caplog.records
        if r.levelname == "ERROR"
    )

    # --- Restore a valid artifact (different content → different hash) ---
    caplog.clear()
    restored_payload: dict[str, object] = json.loads(json.dumps(valid_payload))
    restored_payload["fit_date"] = "2026-05-07"  # bump → bytes diff → hash diff
    time.sleep(0.01)
    coeffs.write_text(json.dumps(restored_payload))

    new_table = table.check_and_reload()
    assert new_table is not None
    assert new_table.is_loaded
    assert new_table.artifact_hash != pre_hash
    # Prior instance is still untouched (R2-H9 invariant) — the swap is
    # caller-driven via the returned instance, not in-place.
    assert table.artifact_hash == pre_hash
