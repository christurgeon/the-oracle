"""EMOS shadow analyzer tests."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# D3 + D9: self-anchored reconciliation envelope (frozen-population seal)
# ---------------------------------------------------------------------------

_FITTABLE_CELLS: set[tuple[str, str]] = {("AUS", "high")}
_PRE_FLAG_FLIP_CUTOFF = date(2026, 5, 9)


def _write_eval_audit(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _write_settlements(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def _write_iso_artifact(path: Path, payload: dict | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload or {"fit_date": "2026-05-01"}, sort_keys=True).encode()
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


# ---------------------------------------------------------------------------
# Step 1: loader (_load_reconciliation_population)
# ---------------------------------------------------------------------------


def test_loader_injects_settlement_and_iso_hash_fields(tmp_path):
    """Loader walks eval_audit, joins settlements, injects fields on every row."""
    from scripts.analyze_emos_shadow import _load_reconciliation_population

    iso_path = tmp_path / "calibration" / "isotonic.json"
    expected_iso_hash = _write_iso_artifact(iso_path)

    state_dir = tmp_path / "state"
    eval_rows = [
        {
            "scan_id": "scan_001",
            "ticker": "KXHIGHNY-26MAY05-T70",
            "city": "NYC",
            "var": "high",
            "forecast_date": "2026-05-05",
            "strike_type": "above",
            "floor": 70.0,
            "cap": None,
            "raw_ens_prob": 0.45,
        },
        {
            "scan_id": "scan_002",
            "ticker": "KXLOWTCHI-26MAY05-T50",
            "city": "CHI",
            "var": "low",
            "forecast_date": "2026-05-05",
            "strike_type": "below",
            "floor": None,
            "cap": 50.0,
            "raw_ens_prob": 0.42,
        },
    ]
    _write_eval_audit(state_dir / "eval_audit.jsonl", eval_rows)

    snapshots_dir = tmp_path / "snapshots"
    # Settlement: NYC high=72 on 2026-05-05, CHI low=48 on 2026-05-05.
    _write_settlements(
        snapshots_dir / "2026-05-06" / "settlements.json",
        [
            {
                "city": "NYC",
                "observation_date": "2026-05-05",
                "high_f": 72,
                "low_f": 58,
                "is_preliminary": False,
                "issued_at": "2026-05-06T06:00:00+00:00",
                "product_id": "p-nyc",
            },
            {
                "city": "CHI",
                "observation_date": "2026-05-05",
                "high_f": 65,
                "low_f": 48,
                "is_preliminary": False,
                "issued_at": "2026-05-06T06:00:00+00:00",
                "product_id": "p-chi",
            },
        ],
    )

    out = _load_reconciliation_population(
        state_dir=state_dir,
        snapshots_dir=snapshots_dir,
        start_date=date(2026, 5, 5),
        end_date=date(2026, 5, 6),
        iso_artifact_path=iso_path,
    )

    assert len(out) == 2
    for row in out:
        assert row["isotonic_artifact_hash"] == expected_iso_hash
        assert row["settlement_status"] == "final"

    by_scan = {r["scan_id"]: r for r in out}
    # NYC: actual high 72 >= floor 70 → outcome 1
    assert by_scan["scan_001"]["settled_value"] == 1
    # CHI: actual low 48 < cap 50 → outcome 1
    assert by_scan["scan_002"]["settled_value"] == 1
    # Original fields preserved.
    assert by_scan["scan_001"]["raw_ens_prob"] == 0.45
    assert by_scan["scan_001"]["ticker"] == "KXHIGHNY-26MAY05-T70"


def test_loader_drops_rows_with_missing_settlement_files(tmp_path):
    """If no snapshot has a settlement for the row, settlement_status='missing'."""
    from scripts.analyze_emos_shadow import _load_reconciliation_population

    iso_path = tmp_path / "calibration" / "isotonic.json"
    _write_iso_artifact(iso_path)

    state_dir = tmp_path / "state"
    _write_eval_audit(
        state_dir / "eval_audit.jsonl",
        [
            {
                "scan_id": "scan_orphan",
                "city": "LAX",
                "var": "high",
                "forecast_date": "2026-05-05",
                "strike_type": "above",
                "floor": 80.0,
                "cap": None,
                "raw_ens_prob": 0.4,
            }
        ],
    )

    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()  # no settlement files at all

    out = _load_reconciliation_population(
        state_dir=state_dir,
        snapshots_dir=snapshots_dir,
        start_date=date(2026, 5, 5),
        end_date=date(2026, 5, 6),
        iso_artifact_path=iso_path,
    )

    assert len(out) == 1
    assert out[0]["settlement_status"] == "missing"
    assert out[0]["settled_value"] is None


def test_loader_uses_latest_settlement_via_prefer_latest_settlement(tmp_path):
    """Two snapshots, preliminary first then final: final wins."""
    from scripts.analyze_emos_shadow import _load_reconciliation_population

    iso_path = tmp_path / "calibration" / "isotonic.json"
    _write_iso_artifact(iso_path)

    state_dir = tmp_path / "state"
    _write_eval_audit(
        state_dir / "eval_audit.jsonl",
        [
            {
                "scan_id": "scan_late",
                "city": "AUS",
                "var": "high",
                "forecast_date": "2026-05-05",
                "strike_type": "above",
                "floor": 75.0,
                "cap": None,
                "raw_ens_prob": 0.45,
            }
        ],
    )

    snapshots_dir = tmp_path / "snapshots"
    # Earlier snapshot has preliminary high=70 (would yield outcome 0).
    _write_settlements(
        snapshots_dir / "2026-05-05" / "settlements.json",
        [
            {
                "city": "AUS",
                "observation_date": "2026-05-05",
                "high_f": 70,
                "low_f": 55,
                "is_preliminary": True,
                "issued_at": "2026-05-05T20:00:00+00:00",
                "product_id": "p-prelim",
            },
        ],
    )
    # Later snapshot has final high=80 (yields outcome 1) — must win.
    _write_settlements(
        snapshots_dir / "2026-05-06" / "settlements.json",
        [
            {
                "city": "AUS",
                "observation_date": "2026-05-05",
                "high_f": 80,
                "low_f": 60,
                "is_preliminary": False,
                "issued_at": "2026-05-06T06:00:00+00:00",
                "product_id": "p-final",
            },
        ],
    )

    out = _load_reconciliation_population(
        state_dir=state_dir,
        snapshots_dir=snapshots_dir,
        start_date=date(2026, 5, 5),
        end_date=date(2026, 5, 6),
        iso_artifact_path=iso_path,
    )

    assert len(out) == 1
    assert out[0]["settlement_status"] == "final"
    # AUS final high=80 >= floor 75 → outcome 1 (preliminary 70 would have been 0)
    assert out[0]["settled_value"] == 1


def _make_reference_rows(
    n_yes: int,
    n_total: int,
    *,
    raw_ens_prob: float = 0.478,
    city: str = "AUS",
    var: str = "high",
    forecast_date: str = "2026-05-06",
) -> list[dict]:
    """Build n_total synthetic post-loader eval_audit row dicts that pass every
    reconciliation filter. n_yes have settled_value=1, the rest settled_value=0.

    Note: rows already have settlement_status and settled_value populated; in
    production these come from _load_reconciliation_population. Tests bypass
    the loader to construct precise filter populations directly."""
    rows: list[dict] = []
    for i in range(n_total):
        rows.append(
            {
                "raw_ens_prob": raw_ens_prob,
                "settlement_status": "final",
                "city": city,
                "var": var,
                "forecast_date": forecast_date,
                "settled_value": 1 if i < n_yes else 0,
                # discriminator so each row hashes uniquely
                "_row_idx": i,
            }
        )
    return rows


def test_compute_broken_bin_population_returns_filter_results(tmp_path):
    """Renamed compute function applies the bin/cell/cutoff/final filters and
    returns (n, mean_predicted, observed_freq, row_hashes). No `matched` field,
    no comparison against a hardcoded constant."""
    from scripts.analyze_emos_shadow import compute_broken_bin_population

    rows = _make_reference_rows(n_yes=16, n_total=51)
    result = compute_broken_bin_population(
        rows,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
    )

    assert not hasattr(result, "matched")
    assert result.n == 51
    assert abs(result.mean_predicted - 0.478) < 1e-9
    assert abs(result.observed_freq - 16 / 51) < 1e-9
    assert len(result.eval_audit_row_hashes) == 51
    for h in result.eval_audit_row_hashes:
        assert isinstance(h, str)
        assert len(h) == 16
        # 16-char hex string
        int(h, 16)


def test_compute_broken_bin_population_filters_out_of_bin_rows(tmp_path):
    """Rows whose raw_ens_prob is outside [0.40, 0.50) are filtered out."""
    from scripts.analyze_emos_shadow import compute_broken_bin_population

    in_bin = _make_reference_rows(n_yes=5, n_total=10, raw_ens_prob=0.45)
    out_of_bin = _make_reference_rows(n_yes=2, n_total=4, raw_ens_prob=0.30)
    # Force unique idx so row hashes don't collide
    for i, r in enumerate(out_of_bin):
        r["_row_idx"] = 100 + i

    result = compute_broken_bin_population(
        in_bin + out_of_bin,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
    )

    assert result.n == 10
    assert abs(result.mean_predicted - 0.45) < 1e-9


# ---------------------------------------------------------------------------
# Step 3: pre-seal sanity validator
# ---------------------------------------------------------------------------


def test_seal_sanity_aborts_outside_bin():
    """If the recomputed mean_predicted is outside [_BIN_RANGE_LOW, _BIN_RANGE_HIGH),
    the seal is corrupt (filter wasn't applied or rows were synthesized wrong)."""
    from scripts.analyze_emos_shadow import (
        ReconciliationResult,
        _validate_seal_inputs,
    )

    bad_result = ReconciliationResult(
        n=20,
        mean_predicted=0.60,
        observed_freq=0.30,
        eval_audit_row_hashes=["aa" * 8] * 20,
    )

    with pytest.raises(RuntimeError, match="mean_predicted.*outside bin"):
        _validate_seal_inputs(bad_result, _FITTABLE_CELLS)


def test_seal_sanity_aborts_below_absolute_floor():
    """n < absolute minimum (10) aborts regardless of --first-run-n-floor."""
    from scripts.analyze_emos_shadow import (
        ReconciliationResult,
        _validate_seal_inputs,
    )

    bad_result = ReconciliationResult(
        n=8,
        mean_predicted=0.45,
        observed_freq=0.30,
        eval_audit_row_hashes=["aa" * 8] * 8,
    )

    with pytest.raises(RuntimeError, match="absolute floor|n=8"):
        _validate_seal_inputs(bad_result, _FITTABLE_CELLS)


def test_seal_sanity_aborts_with_empty_fittable_cells():
    """Empty fittable_cells means the loader returned nothing meaningful —
    refuse to seal."""
    from scripts.analyze_emos_shadow import (
        ReconciliationResult,
        _validate_seal_inputs,
    )

    ok_result = ReconciliationResult(
        n=20,
        mean_predicted=0.45,
        observed_freq=0.30,
        eval_audit_row_hashes=["aa" * 8] * 20,
    )

    with pytest.raises(RuntimeError, match="fittable_cells is empty"):
        _validate_seal_inputs(ok_result, set())


def test_seal_sanity_passes_on_valid_inputs():
    """Happy path: in-bin mean, n at absolute floor, valid obs, non-empty cells."""
    from scripts.analyze_emos_shadow import (
        ReconciliationResult,
        _validate_seal_inputs,
    )

    ok_result = ReconciliationResult(
        n=10,
        mean_predicted=0.45,
        observed_freq=0.30,
        eval_audit_row_hashes=["aa" * 8] * 10,
    )
    _validate_seal_inputs(ok_result, _FITTABLE_CELLS)


# ---------------------------------------------------------------------------
# Step 4: frozen population I/O + two-branch reconcile_or_use_envelope
# ---------------------------------------------------------------------------


def test_write_and_load_frozen_population_round_trips(tmp_path):
    """Frozen JSONL writer + loader preserve every row byte-stably."""
    from scripts.analyze_emos_shadow import (
        _load_frozen_population,
        _write_frozen_population,
    )

    rows = _make_reference_rows(n_yes=3, n_total=7)
    frozen_path = tmp_path / "frozen.jsonl"
    _write_frozen_population(rows, frozen_path)

    loaded = _load_frozen_population(frozen_path)
    assert loaded == rows


def test_write_reconciliation_envelope_includes_window_and_cells(tmp_path):
    """Envelope payload includes fittable_cells_sealed, start/end dates, cutoff."""
    from scripts.analyze_emos_shadow import (
        ReconciliationResult,
        write_reconciliation_envelope,
    )

    result = ReconciliationResult(
        n=20,
        mean_predicted=0.45,
        observed_freq=0.30,
        eval_audit_row_hashes=["aa" * 8] * 20,
    )
    envelope_path = tmp_path / "envelope.json"
    write_reconciliation_envelope(
        result,
        envelope_path=envelope_path,
        isotonic_artifact_hash="ISOHASH",
        git_rev="rev123",
        query_string="see scripts/analyze_emos_shadow.py",
        fittable_cells_sealed=["AUS/high", "NYC/high"],
        start_date_sealed="2026-05-01",
        end_date_sealed="2026-05-08",
        pre_flag_flip_cutoff_date_sealed="2026-05-09",
    )

    payload = json.loads(envelope_path.read_text())
    assert payload["fittable_cells_sealed"] == ["AUS/high", "NYC/high"]
    assert payload["start_date_sealed"] == "2026-05-01"
    assert payload["end_date_sealed"] == "2026-05-08"
    assert payload["pre_flag_flip_cutoff_date_sealed"] == "2026-05-09"
    assert payload["isotonic_artifact_hash"] == "ISOHASH"


def test_first_run_seals_envelope_and_writes_frozen(tmp_path):
    """Branch A: no envelope on disk + n >= floor → seal + freeze + return True/sealed=True."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    iso_hash = _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=5, n_total=15)
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    passed, env, was_sealed = reconcile_or_use_envelope(
        rows,
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
        git_rev="abc",
    )
    assert passed is True
    assert was_sealed is True
    assert envelope_path.exists()
    assert frozen_path.exists()
    assert env["recomputed_n"] == 15
    assert env["isotonic_artifact_hash"] == iso_hash
    assert env["fittable_cells_sealed"] == ["AUS/high"]


def test_first_run_aborts_when_below_floor(tmp_path):
    """Branch A: no envelope + n < first_run_n_floor → ABORT."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=2, n_total=12)

    with pytest.raises(RuntimeError, match="First-run seal aborted|below floor"):
        reconcile_or_use_envelope(
            rows,
            iso_artifact_path=iso_path,
            envelope_path=tmp_path / "envelope.json",
            frozen_path=tmp_path / "frozen.jsonl",
            fittable_cells=_FITTABLE_CELLS,
            pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 8),
            first_run_n_floor=30,
        )


def test_first_run_aborts_when_filter_population_is_empty(tmp_path):
    """Branch A: empty-pop guard fires before the floor check."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    # No fittable cells matches → population is empty.
    rows = _make_reference_rows(n_yes=5, n_total=15)

    with pytest.raises(RuntimeError, match="filtered population is empty"):
        reconcile_or_use_envelope(
            rows,
            iso_artifact_path=iso_path,
            envelope_path=tmp_path / "envelope.json",
            frozen_path=tmp_path / "frozen.jsonl",
            fittable_cells={("MIA", "low")},  # no row matches
            pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 8),
            first_run_n_floor=10,
        )


def test_subsequent_run_verifies_against_frozen_population(tmp_path):
    """Branch B: envelope + frozen both exist → read frozen, verify, no reseal."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=5, n_total=15)
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    # Seal first.
    reconcile_or_use_envelope(
        rows,
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
    )

    # Verify (ignore the `rows` argument — Branch B reads from frozen).
    passed, env, was_sealed = reconcile_or_use_envelope(
        [],  # rows are ignored in Branch B
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
    )
    assert passed is True
    assert was_sealed is False
    assert env["recomputed_n"] == 15


def test_subsequent_run_aborts_when_frozen_rows_tampered(tmp_path):
    """Branch B: frozen file mutated post-seal → row-hash mismatch → ABORT."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=5, n_total=15)
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    reconcile_or_use_envelope(
        rows,
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
    )

    # Tamper: append an extra row that changes the hash set.
    with frozen_path.open("a") as fh:
        fh.write(json.dumps({"tampered": True}) + "\n")

    with pytest.raises(RuntimeError, match="frozen.*sha256|row hashes do not match|tampering"):
        reconcile_or_use_envelope(
            [],
            iso_artifact_path=iso_path,
            envelope_path=envelope_path,
            frozen_path=frozen_path,
            fittable_cells=_FITTABLE_CELLS,
            pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 8),
            first_run_n_floor=10,
        )


def test_subsequent_run_aborts_when_iso_artifact_changed(tmp_path):
    """Branch B: iso artifact bytes changed post-seal → file-level hash guard → ABORT."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=5, n_total=15)
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    reconcile_or_use_envelope(
        rows,
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
    )

    # Modify the iso artifact bytes.
    _write_iso_artifact(iso_path, payload={"fit_date": "2026-06-01"})

    with pytest.raises(RuntimeError, match="isotonic artifact|iso_hash"):
        reconcile_or_use_envelope(
            [],
            iso_artifact_path=iso_path,
            envelope_path=envelope_path,
            frozen_path=frozen_path,
            fittable_cells=_FITTABLE_CELLS,
            pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 8),
            first_run_n_floor=10,
        )


def test_subsequent_run_aborts_when_fittable_cells_changed(tmp_path):
    """Branch B: caller's fittable_cells differs from sealed set → ABORT."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=5, n_total=15)
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    reconcile_or_use_envelope(
        rows,
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
    )

    with pytest.raises(RuntimeError, match="fittable_cells"):
        reconcile_or_use_envelope(
            [],
            iso_artifact_path=iso_path,
            envelope_path=envelope_path,
            frozen_path=frozen_path,
            fittable_cells={("AUS", "high"), ("NYC", "low")},  # differs from sealed
            pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 8),
            first_run_n_floor=10,
        )


def test_subsequent_run_aborts_when_window_changed(tmp_path):
    """Branch B: caller's start/end/cutoff differs from sealed → ABORT."""
    from scripts.analyze_emos_shadow import reconcile_or_use_envelope

    iso_path = tmp_path / "iso.json"
    _write_iso_artifact(iso_path)
    rows = _make_reference_rows(n_yes=5, n_total=15)
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    reconcile_or_use_envelope(
        rows,
        iso_artifact_path=iso_path,
        envelope_path=envelope_path,
        frozen_path=frozen_path,
        fittable_cells=_FITTABLE_CELLS,
        pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 8),
        first_run_n_floor=10,
    )

    with pytest.raises(RuntimeError, match="window|start_date|end_date"):
        reconcile_or_use_envelope(
            [],
            iso_artifact_path=iso_path,
            envelope_path=envelope_path,
            frozen_path=frozen_path,
            fittable_cells=_FITTABLE_CELLS,
            pre_flag_flip_cutoff_date=_PRE_FLAG_FLIP_CUTOFF,
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 12),  # widened
            first_run_n_floor=10,
        )


# ---------------------------------------------------------------------------
# Step 5: reseal helper (operator escape hatch)
# ---------------------------------------------------------------------------


def test_reseal_backs_up_old_files_before_overwriting(tmp_path):
    """`_backup_for_reseal` moves existing envelope + frozen to .bak.<ts>,
    preserving original bytes. After the call neither original file remains,
    and backup files contain the original content."""
    from scripts.analyze_emos_shadow import _backup_for_reseal

    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"
    envelope_path.write_text('{"recomputed_n": 15}')
    frozen_path.write_text('{"x": 1}\n')

    backups = _backup_for_reseal(envelope_path, frozen_path)
    assert "envelope" in backups
    assert "frozen" in backups
    assert backups["envelope"].exists()
    assert backups["frozen"].exists()
    assert backups["envelope"].read_text() == '{"recomputed_n": 15}'
    assert backups["frozen"].read_text() == '{"x": 1}\n'
    assert not envelope_path.exists()
    assert not frozen_path.exists()


def test_reseal_helper_is_noop_when_no_files_exist(tmp_path):
    """If neither envelope nor frozen exist, _backup_for_reseal returns empty dict."""
    from scripts.analyze_emos_shadow import _backup_for_reseal

    backups = _backup_for_reseal(
        tmp_path / "absent_envelope.json",
        tmp_path / "absent_frozen.jsonl",
    )
    assert backups == {}


# ---------------------------------------------------------------------------
# Step 6: CLI helpers (_resolve_fittable_cells, _resolve_git_rev)
# ---------------------------------------------------------------------------


def test_resolve_fittable_cells_from_coefficients_file(tmp_path):
    """Reads coefficients.json cells, subtracts excluded_city_vars from config."""
    from scripts.analyze_emos_shadow import _resolve_fittable_cells

    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text(
        json.dumps(
            {
                "cells": {
                    "AUS/high": {},
                    "AUS/low": {},
                    "DEN/high": {},
                    "DEN/low": {},
                    "NYC/high": {},
                    "SEA/high": {},
                }
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather:\n"
        "  excluded_city_vars:\n"
        "    - DEN_low\n"
        "    - DEN_high\n"
        "    - SEA_high\n"
    )

    cells = _resolve_fittable_cells(
        coeffs_path=coeffs_path, config_path=config_path, override=None
    )
    assert cells == {("AUS", "high"), ("AUS", "low"), ("NYC", "high")}


def test_resolve_fittable_cells_with_override(tmp_path):
    """CLI override string parses to a set, bypassing config-derived value."""
    from scripts.analyze_emos_shadow import _resolve_fittable_cells

    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text('{"cells": {"AUS/high": {}}}')
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weather:\n  excluded_city_vars: []\n")

    cells = _resolve_fittable_cells(
        coeffs_path=coeffs_path,
        config_path=config_path,
        override="NYC/high,CHI/low",
    )
    assert cells == {("NYC", "high"), ("CHI", "low")}


def test_main_dry_run_first_run_seals(tmp_path, monkeypatch, capsys):
    """End-to-end main() smoke: seals envelope + frozen on first --dry-run."""
    from scripts.analyze_emos_shadow import main

    # Build a tiny realistic fixture: state/eval_audit.jsonl + snapshots + iso + coeffs + config.
    state_dir = tmp_path / "state"
    _write_eval_audit(
        state_dir / "eval_audit.jsonl",
        [
            {
                "scan_id": f"s{i:03d}",
                "city": "AUS",
                "var": "high",
                "forecast_date": "2026-05-05",
                "strike_type": "above",
                "floor": 75.0,
                "cap": None,
                "raw_ens_prob": 0.45,
            }
            for i in range(15)
        ],
    )
    snapshots_dir = tmp_path / "snapshots"
    _write_settlements(
        snapshots_dir / "2026-05-06" / "settlements.json",
        [
            {
                "city": "AUS",
                "observation_date": "2026-05-05",
                "high_f": 80,
                "low_f": 65,
                "is_preliminary": False,
                "issued_at": "2026-05-06T06:00:00+00:00",
                "product_id": "p-aus",
            },
        ],
    )
    iso_path = tmp_path / "isotonic.json"
    _write_iso_artifact(iso_path)
    coeffs_path = tmp_path / "coefficients.json"
    coeffs_path.write_text('{"cells": {"AUS/high": {}}}')
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weather:\n  excluded_city_vars: []\n")
    envelope_path = tmp_path / "envelope.json"
    frozen_path = tmp_path / "frozen.jsonl"

    rc = main(
        [
            "--dry-run",
            "--start-date",
            "2026-05-01",
            "--end-date",
            "2026-05-08",
            "--decision-mode",
            "30-day-system",
            "--state-dir",
            str(state_dir),
            "--snapshots-dir",
            str(snapshots_dir),
            "--iso-artifact-path",
            str(iso_path),
            "--coeffs-path",
            str(coeffs_path),
            "--config-path",
            str(config_path),
            "--envelope-path",
            str(envelope_path),
            "--frozen-path",
            str(frozen_path),
            "--first-run-n-floor",
            "10",
        ]
    )
    assert rc == 0
    assert envelope_path.exists()
    assert frozen_path.exists()
    out = capsys.readouterr().out
    assert "sealed" in out
    assert "n=15" in out
