"""Preflight-wiring tests for ``scripts/fit_isotonic.py``.

Exercises the input-contract abort policy via stubbed pair-builders; the heavy
``IsotonicRegression`` fit step is only invoked in tests that explicitly need it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn", reason="ML extras not installed (uv sync --extra ml)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _empty_snapshot_dir(tmp_path: Path) -> Path:
    """Build an empty (but well-formed) snapshots tree."""
    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)
    (snap_dir / "weather_forecasts.json").write_text(json.dumps([]))
    (snap_dir / "settlements.json").write_text(json.dumps([]))
    (snap_dir / "markets.json").write_text(json.dumps([]))
    return tmp_path / "snapshots"


def _write_calendar(path: Path, models: dict[str, str] | None = None) -> Path:
    """Write a calendar JSON to ``path`` and return it."""
    if models is None:
        models = {"gfs_seamless": "2026-03-19", "ecmwf_ifs025": "2026-04-24"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-05-09T00:00:00+00:00",
                "method": "auto",
                "models": {name: {"first_seen_date": d} for name, d in models.items()},
            }
        )
    )
    return path


def _write_bias(path: Path) -> Path:
    """Write a minimal but ``loaded == True`` bias table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # BiasTable.loaded == bool(self._data); a non-empty dict suffices.
    path.write_text(json.dumps({"NYC": {"high": {"by_month": {}}}}))
    return path


def _make_report(
    *,
    total: int = 200,
    full: int = 200,
    partial: int = 0,
    empty: int = 0,
    single_model_era: int = 0,
    exempt: int = 0,
):
    from scripts._calibration_input_contract import InputContractReport

    return InputContractReport(
        n_triples_total=total,
        n_triples_full=full,
        n_triples_partial=partial,
        n_triples_empty=empty,
        n_triples_single_model_era=single_model_era,
        n_triples_exempt=exempt,
        n_settlements_final=total,
        n_settlements_dropped_preliminary=0,
    )


def _stub_pair_builder(monkeypatch, pairs=None) -> dict:
    import scripts._calibration_common as cc

    called = {"helper": False, "kwargs": None}

    def fake_helper(**kwargs):
        called["helper"] = True
        called["kwargs"] = kwargs
        return iter(pairs or [])

    monkeypatch.setattr(cc, "build_production_mirrored_pairs", fake_helper)
    return called


def _stub_preflight(monkeypatch, report) -> dict:
    import scripts._calibration_input_contract as ic

    called = {"preflight": False, "kwargs": None}

    def fake(*args, **kwargs):
        called["preflight"] = True
        called["kwargs"] = kwargs
        return report

    monkeypatch.setattr(ic, "assert_snapshot_input_contract", fake)
    return called


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fit_isotonic.py", *args])


# ---------------------------------------------------------------------------
# T3a — calendar missing
# ---------------------------------------------------------------------------


def test_t3a_calendar_missing_raises(tmp_path: Path, monkeypatch) -> None:
    """If the data-availability calendar is missing, abort with a clear message."""
    import scripts._calibration_common as _cc
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    # Point both _PROJECT_ROOTs at tmp_path — run_input_contract_preflight now
    # lives in _calibration_common and uses that module's _PROJECT_ROOT.
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_cc, "_PROJECT_ROOT", tmp_path)
    # Calendar deliberately not written.

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="calendar"):
        fit_iso.main()


# ---------------------------------------------------------------------------
# T3b — partial-coverage abort + flag honor
# ---------------------------------------------------------------------------


def test_t3b_partial_coverage_aborts_without_flag(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, partial=10))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="--allow-partial-model-coverage"):
        fit_iso.main()


def test_t3b_partial_coverage_flag_proceeds(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, partial=10))
    called = _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--allow-partial-model-coverage",
    )

    with pytest.raises(SystemExit):
        fit_iso.main()
    assert called["helper"] is True


# ---------------------------------------------------------------------------
# T3c — empty-cells abort + flag honor
# ---------------------------------------------------------------------------


def test_t3c_empty_cells_aborts_without_flag(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, empty=10))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="--allow-empty-cells"):
        fit_iso.main()


def test_t3c_empty_cells_flag_proceeds(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=190, empty=10))
    called = _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--allow-empty-cells",
    )

    with pytest.raises(SystemExit):
        fit_iso.main()
    assert called["helper"] is True


# ---------------------------------------------------------------------------
# T3d — bias-table missing without override
# ---------------------------------------------------------------------------


def test_t3d_missing_bias_table_aborts_without_override(tmp_path: Path, monkeypatch) -> None:
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    bias_path = tmp_path / "missing_bias.json"  # NOT written

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias_path),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="--no-bias-correction"):
        fit_iso.main()


def test_t3d_missing_bias_table_no_bias_correction_proceeds_and_records(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    bias_path = tmp_path / "missing_bias.json"  # NOT written

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    pairs = [(0.4, 0, "NYC", "temperature_2m_max", "greater", "2026-04-15")] * 60 + [
        (0.6, 1, "NYC", "temperature_2m_max", "greater", "2026-04-15")
    ] * 60
    _stub_pair_builder(monkeypatch, pairs=pairs)

    out_path = tmp_path / "isotonic.json"
    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(out_path),
        "--bias-table",
        str(bias_path),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
    )

    fit_iso.main()
    artifact = json.loads(out_path.read_text())
    assert artifact["bias_correction_applied"] is False


# ---------------------------------------------------------------------------
# T3e — config-load failure
# ---------------------------------------------------------------------------


def test_t3e_missing_ensemble_weights_in_config_aborts(tmp_path: Path, monkeypatch) -> None:
    """If config has no strategies.weather.ensemble_weights and no CLI override → abort."""
    import scripts._calibration_common as _cc
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")

    # Synthesise a config.yaml at the test's _PROJECT_ROOT that lacks the key.
    cfg_dir = tmp_path / "python"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text("strategies:\n  weather:\n    other_key: 1\n")

    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(_cc, "_PROJECT_ROOT", tmp_path)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
    )

    with pytest.raises(
        CalibratorInputContractError,
        match="strategies.weather.ensemble_weights",
    ):
        fit_iso.main()


def test_t3e_explicit_ensemble_weights_bypass_config(tmp_path: Path, monkeypatch) -> None:
    """--ensemble-weights overrides the config-load and proceeds."""
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")

    # Config still missing the key — the CLI arg should win.
    cfg_dir = tmp_path / "python"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text("strategies:\n  weather:\n    other_key: 1\n")

    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))
    called = _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(SystemExit):
        fit_iso.main()
    assert called["helper"] is True


# ---------------------------------------------------------------------------
# T3h — n_triples_total == 0 abort
# ---------------------------------------------------------------------------


def test_t3h_total_zero_aborts(tmp_path: Path, monkeypatch) -> None:
    """If the preflight reports n_triples_total == 0 the fitter aborts."""
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=0, full=0))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="n_triples_total"):
        fit_iso.main()


# ---------------------------------------------------------------------------
# T3i — all-exempt abort
# ---------------------------------------------------------------------------


def test_t3i_all_exempt_aborts(tmp_path: Path, monkeypatch) -> None:
    """If every triple is calendar-exempt the fitter aborts."""
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    # n_triples_exempt == n_triples_total > 0 → abort with "exempt" message.
    _stub_preflight(monkeypatch, _make_report(total=200, full=0, exempt=200))
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="exempt"):
        fit_iso.main()


# ---------------------------------------------------------------------------
# T3f — --write-pair-provenance happy path (post-Task 5)
# ---------------------------------------------------------------------------


def test_t3f_write_pair_provenance_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    """--write-pair-provenance writes one JSONL row per yielded pair with schema_version=1."""
    import scripts._calibration_common as cc
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    assert hasattr(cc, "build_production_mirrored_pairs_with_provenance")

    # Stub the sibling generator to yield 9-tuples we control. Need ≥100
    # pairs across both classes so the IsotonicRegression fit step doesn't
    # short-circuit on len(pairs)<100.
    sibling_pairs = []
    for i in range(60):
        # 60 negatives at low raw, 60 positives at high raw.
        sibling_pairs.append(
            (
                0.3,
                0,
                "NYC",
                "temperature_2m_max",
                "greater",
                "2026-04-15",
                ("gfs_seamless",),
                False,
                f"KXHIGHNY-26APR15-T70-N{i}",
            )
        )
        sibling_pairs.append(
            (
                0.7,
                1,
                "NYC",
                "temperature_2m_max",
                "greater",
                "2026-04-15",
                ("gfs_seamless", "ecmwf_ifs025"),
                True,
                f"KXHIGHNY-26APR15-T70-P{i}",
            )
        )

    def fake_sibling(**kwargs):
        return iter(sibling_pairs)

    monkeypatch.setattr(cc, "build_production_mirrored_pairs_with_provenance", fake_sibling)

    out_provenance = tmp_path / "provenance.jsonl"
    out_artifact = tmp_path / "out.json"
    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(out_artifact),
        "--bias-table",
        str(tmp_path / "missing_bias.json"),  # use --no-bias-correction below
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
        "--write-pair-provenance",
        str(out_provenance),
    )

    fit_iso.main()

    assert out_provenance.exists()
    rows = [json.loads(line) for line in out_provenance.read_text().splitlines() if line.strip()]
    assert len(rows) == len(sibling_pairs)
    for row in rows:
        assert row["schema_version"] == 1
        assert row["city"] == "NYC"
        assert row["daily_variable"] == "temperature_2m_max"
        assert row["strike_type"] == "greater"
        assert "raw_prob" in row
        assert "outcome" in row
        assert "models_used" in row
        assert isinstance(row["models_used"], list)  # JSON has no tuple type
        assert "bias_correction_applied" in row
        assert isinstance(row["bias_correction_applied"], bool)
        assert "spread_inflation_factor" in row
        assert "ensemble_weights_at_fit" in row
        assert "ticker" in row
        assert row["ticker"].startswith("KXHIGHNY-26APR15-T70-")


# ---------------------------------------------------------------------------
# T3f1 — --write-pair-provenance truncates a stale target on re-run
# ---------------------------------------------------------------------------


def test_t3f1_write_pair_provenance_truncates_stale_target(tmp_path: Path, monkeypatch) -> None:
    """A pre-existing target file must be truncated before the new run starts.

    Regression: record_pair_provenance uses O_APPEND for atomic concurrent
    writes. Without truncation at the start of the loop, re-running the
    fitter against the same target accumulates on top of the prior run's
    output, producing duplicate rows and confusing the operator.
    """
    import scripts._calibration_common as cc
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    sibling_pairs = []
    for i in range(60):
        sibling_pairs.append(
            (0.3, 0, "NYC", "temperature_2m_max", "greater", "2026-04-15",
             ("gfs_seamless",), False, f"KXHIGHNY-26APR15-T70-N{i}")
        )
        sibling_pairs.append(
            (0.7, 1, "NYC", "temperature_2m_max", "greater", "2026-04-15",
             ("gfs_seamless", "ecmwf_ifs025"), True, f"KXHIGHNY-26APR15-T70-P{i}")
        )

    monkeypatch.setattr(
        cc,
        "build_production_mirrored_pairs_with_provenance",
        lambda **_kw: iter(sibling_pairs),
    )

    out_provenance = tmp_path / "provenance.jsonl"
    # Pre-write a stale row that must NOT survive the run.
    out_provenance.write_text('{"schema_version":1,"ticker":"STALE-FROM-PREVIOUS-RUN"}\n')

    _set_argv(
        monkeypatch,
        "--snapshots", str(snapshots),
        "--output", str(tmp_path / "out.json"),
        "--bias-table", str(tmp_path / "missing_bias.json"),
        "--ensemble-weights", json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
        "--write-pair-provenance", str(out_provenance),
    )

    fit_iso.main()

    rows = [json.loads(line) for line in out_provenance.read_text().splitlines() if line.strip()]
    assert len(rows) == len(sibling_pairs), (
        f"expected exactly {len(sibling_pairs)} rows; got {len(rows)} "
        "(stale rows from prior run leaked through)"
    )
    assert not any(r.get("ticker") == "STALE-FROM-PREVIOUS-RUN" for r in rows)


# ---------------------------------------------------------------------------
# T3g — sample-size guard
# ---------------------------------------------------------------------------


def test_t3g_sample_size_guard_aborts_below_100(tmp_path: Path, monkeypatch) -> None:
    """If full + single_model_era < 100 the preflight aborts before fitting."""
    import scripts.fit_isotonic as fit_iso
    from scripts._calibration_input_contract import CalibratorInputContractError

    snapshots = _empty_snapshot_dir(tmp_path)
    bias = _write_bias(tmp_path / "bias.json")
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    # 50 full + 40 single = 90 < 100 → abort.
    _stub_preflight(
        monkeypatch,
        _make_report(total=90, full=50, single_model_era=40),
    )
    _stub_pair_builder(monkeypatch)

    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(tmp_path / "out.json"),
        "--bias-table",
        str(bias),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
    )

    with pytest.raises(CalibratorInputContractError, match="≥100|>=100|100"):
        fit_iso.main()
