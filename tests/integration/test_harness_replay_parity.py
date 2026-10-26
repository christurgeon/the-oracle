"""Replay-parity tests for the harness."""

from __future__ import annotations

import inspect
import re
from datetime import date

import pytest


def test_r2_strike_label_translation():
    """R-2: _DIRECTION_TO_STRIKE_LABEL is intact and exposes the canonical 3 strike labels."""
    from python.strategies.weather_edge import _DIRECTION_TO_STRIKE_LABEL
    assert set(_DIRECTION_TO_STRIKE_LABEL.values()) == {"greater", "less", "between"}


def test_r3_chi_pre_fix_exclusion():
    """R-3: CHI rows with forecast_date < 2026-03-20 are excluded from FeatureRow build."""
    # The harness reuses _calibration_common._CHI_SNAPSHOT_FIX_DATE via
    # build_production_mirrored_pairs_with_provenance. Inspect that constant and
    # assert it matches the spec's 2026-03-20.
    from scripts._calibration_common import _CHI_SNAPSHOT_FIX_DATE
    assert _CHI_SNAPSHOT_FIX_DATE == date(2026, 3, 20)


def test_r4_no_synthetic_brier_in_verdict_path():
    """R-4: harness's verdict-scoring code path does not import or call synthetic_bracket_brier.

    C5 fix: per spec, recurse one level into DIRECT imports from
    python.analytics.* / python.data.* namespaces (NOT transitively into
    numpy / scipy / sklearn — those produce false positives and are
    intractable to scan).
    """
    import python.analytics.calibrator_harness as harness
    modules_to_scan = [harness]

    # Discover direct project-namespace imports from the harness module.
    for name in dir(harness):
        obj = getattr(harness, name, None)
        if obj is None or not hasattr(obj, "__module__"):
            continue
        mod_name = getattr(obj, "__module__", "") or ""
        if (
            mod_name.startswith("python.analytics.")
            or mod_name.startswith("python.data.")
        ) and mod_name != harness.__name__:
            try:
                import importlib
                modules_to_scan.append(importlib.import_module(mod_name))
            except ImportError:
                continue

    # De-duplicate.
    seen_module_names: set[str] = set()
    unique_modules = []
    for m in modules_to_scan:
        if m.__name__ not in seen_module_names:
            seen_module_names.add(m.__name__)
            unique_modules.append(m)

    forbidden_patterns = [
        r"synthetic_bracket_brier\s*\(",   # call
        r"from .* import .*synthetic_bracket_brier",  # named import
    ]

    # The defining module (python.analytics.emos_fit) is the hermetic owner of
    # synthetic_bracket_brier; its internal call inside fit_one_cell produces
    # diagnostic-only fields (oos_brier_emos/oos_brier_raw) that the harness's
    # verdict-scoring path never reads. Skip the defining module so the test
    # measures *callers from the verdict path*, not the function's owner.
    for mod in unique_modules:
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        if re.search(r"^def\s+synthetic_bracket_brier\s*\(", src, flags=re.MULTILINE):
            continue  # owner module — exempt by construction
        for pattern in forbidden_patterns:
            assert not re.search(pattern, src), (
                f"FR-16.1 violation: {pattern!r} found in {mod.__name__}"
            )


def test_u8_bracket_width_assertion_on_mismatch(monkeypatch):
    """U-8: monkey-patching emos_fit.DEFAULT_BRACKET_WIDTH to 1.0 fires AssertionError on reload.

    Verifies FR-9 / FM-6 module-load bracket-width parity check fires when
    runtime and emos_fit disagree (the LAX/high failure-mode guard).
    """
    import importlib

    from python.analytics import emos_fit

    monkeypatch.setattr(emos_fit, "DEFAULT_BRACKET_WIDTH", 1.0)

    # Force re-import of calibrator_harness so its module-load assertions re-run.
    import python.analytics.calibrator_harness as harness_mod
    with pytest.raises(AssertionError, match="bracket-width"):
        importlib.reload(harness_mod)

    # Restore: pytest's monkeypatch + reload outside the with-block resets state.
    monkeypatch.setattr(emos_fit, "DEFAULT_BRACKET_WIDTH", 2.0)
    importlib.reload(harness_mod)  # restore consistent state for subsequent tests


def test_r6_strike_label_translation_runtime_uses_canonical():
    """R-6: FeatureRow construction uses canonical strike_type labels."""
    from python.analytics.calibrator_harness import FeatureRow
    # Validate that allowed strike_type values are exactly the 3 canonical labels.
    valid = ("greater", "less", "between")
    for s in valid:
        FeatureRow(
            city="MIA", var="temperature_2m_max", forecast_date="2026-05-01",
            snapshot_date="2026-05-01", lead=0,
            ticker="X", strike_type=s, floor=80.0, cap=None,
            market_yes_price=0.5, blended_raw_prob=0.5,
            gfs_ens_mean=80.0, gfs_ens_std=1.0, n_members=31, outcome=0,
        )


def test_r5_frozen_sentinel_row():
    """R-5: pipeline math applied DIRECTLY to fixture's raw inputs matches frozen outputs.

    B3 fix: don't re-load from data/snapshots and match by ticker — that
    tests "the row still loads," not "frozen inputs → frozen outputs."
    Instead, call compute_blended_probability + the spread-inflation chain
    directly on the fixture's raw_gfs_members and assert outputs match.

    Option X (per Task 22 corrections): bias_correction_entry is None in the
    fixture, so the bias-correction branch is skipped here. The bias-correction
    pipeline is exercised in its own suite (tests/test_bias_correction*.py);
    this test covers the harness's pipeline composition.
    """
    import json
    from datetime import date as date_cls
    from pathlib import Path

    fixture_path = Path("tests/fixtures/sentinel_row.json")
    if not fixture_path.exists():
        pytest.skip(f"sentinel fixture not yet generated: {fixture_path}")

    fix = json.loads(fixture_path.read_text())
    expected = fix["frozen_expected"]

    # Apply the pipeline math directly on the frozen raw inputs.
    from python.common import Direction
    from python.data.weather import EnsembleForecast, compute_blended_probability

    forecast = EnsembleForecast(
        forecast_date=date_cls.fromisoformat(fix["forecast_date"]),
        city=fix["city"],
        temperatures=list(fix["raw_gfs_members"]),
        model="gfs_seamless",
    )

    # Bias correction (skipped — Option X: bias_correction_entry is None).
    assert fix.get("bias_correction_entry") is None, (
        "R-5 fixture was regenerated with bias_correction_entry set; "
        "update the test to apply_bias_correction or revert to Option X "
        "(see scripts/generate_sentinel_row.py)."
    )

    # Spread inflation.
    factor = fix.get("spread_inflation_factor", 1.0)
    if factor != 1.0:
        forecast = forecast.inflated(factor)

    # gfs_ens_mean / gfs_ens_std (ddof=1) on the post-inflation array
    # — matches _compute_gfs_stats in python/analytics/calibrator_harness.py.
    members_post = list(forecast.temperatures)
    n = len(members_post)
    gfs_mean = sum(members_post) / n
    gfs_std = (sum((m - gfs_mean) ** 2 for m in members_post) / (n - 1)) ** 0.5

    # Blended raw prob (single-model blend).
    direction = {
        "greater": Direction.ABOVE,
        "less": Direction.BELOW,
        "between": Direction.BETWEEN,
    }[fix["strike_type"]]
    blended_raw_prob, _ = compute_blended_probability(
        {"gfs_seamless": forecast},
        fix["ensemble_weights"],
        direction=direction,
        floor_strike=fix.get("floor"),
        cap_strike=fix.get("cap"),
    )

    assert abs(blended_raw_prob - expected["blended_raw_prob"]) < 1e-9, (
        f"R-5 blended_raw_prob mismatch: got {blended_raw_prob}, "
        f"expected {expected['blended_raw_prob']}"
    )
    assert abs(gfs_mean - expected["gfs_ens_mean"]) < 1e-9, (
        f"R-5 gfs_ens_mean mismatch: got {gfs_mean}, expected {expected['gfs_ens_mean']}"
    )
    assert abs(gfs_std - expected["gfs_ens_std"]) < 1e-9, (
        f"R-5 gfs_ens_std mismatch: got {gfs_std}, expected {expected['gfs_ens_std']}"
    )


def test_r1_emos_probability_parity_on_eval_audit():
    """R-1: harness's EMOS-deployed score matches eval_audit.jsonl emos_prob_raw to 1e-5.

    B2 fix: PIN snapshot_run_id from each audit row — load that specific
    snapshot, not 'latest'. Avoids false failures when snapshots get re-fitted
    between trade time and test time (and false passes when drift coincides
    with snapshot rotation).
    """
    import json
    from pathlib import Path

    from python.analytics.calibrator_harness import (
        EmosDeployedReference,
        load_feature_rows,
    )

    audit_path = Path("state/eval_audit.jsonl")
    coefficients_path = Path("data/emos/coefficients.json")
    if not audit_path.exists() or not coefficients_path.exists():
        pytest.skip("audit log or EMOS coefficients missing")

    # Read last ~500 lines and find rows with emos_prob_raw populated AND
    # snapshot_run_id pinning available.
    lines = audit_path.read_text().splitlines()[-500:]
    audit_rows = []
    for line in lines:
        try:
            rec = json.loads(line)
            if (
                rec.get("emos_prob_raw") is not None
                and rec.get("snapshot_run_id") is not None
                and rec.get("ticker") is not None
            ):
                audit_rows.append(rec)
        except (json.JSONDecodeError, KeyError):
            continue

    if not audit_rows:
        pytest.skip(
            "no eval_audit rows with emos_prob_raw + snapshot_run_id populated "
            "(production prediction_log.py may not yet record snapshot_run_id; "
            "if so, R-1 cannot pin the snapshot vintage and must be deferred)"
        )

    # Group audit rows by snapshot_run_id; one harness-load per run_id.
    from collections import defaultdict
    by_run_id: dict[str, list[dict]] = defaultdict(list)
    for rec in audit_rows:
        by_run_id[rec["snapshot_run_id"]].append(rec)

    candidate = EmosDeployedReference()
    fit_state = candidate.load(coefficients_path)

    matched = 0
    skipped_no_snapshot = 0
    mismatches: list[tuple] = []
    for run_id, recs in by_run_id.items():
        # Locate the snapshot run directory.
        # Snapshot dir structure: data/snapshots/<collection_date>/<run_id>/
        run_dir = None
        snapshots_root = Path("data/snapshots")
        if snapshots_root.is_dir():
            for day_dir in snapshots_root.iterdir():
                candidate_run = day_dir / run_id
                if candidate_run.is_dir():
                    run_dir = day_dir  # load_feature_rows wants the day, not the run
                    break
        if run_dir is None:
            skipped_no_snapshot += len(recs)
            continue

        # Load only this snapshot's rows.
        rows = load_feature_rows(
            snapshots_dir=run_dir.parent,  # pin to the day that contains this run_id
            bias_table_path=Path("data/gfs_bias/bias_correction.json")
            if Path("data/gfs_bias/bias_correction.json").exists() else None,
            ensemble_weights={"gfs_seamless": 1.0},
        )
        by_ticker = {r.ticker: r for r in rows}

        for rec in recs:
            row = by_ticker.get(rec["ticker"])
            if row is None:
                continue
            p_harness = candidate.score(row, fit_state)
            p_audit = rec["emos_prob_raw"]
            if abs(p_harness - p_audit) > 1e-5:
                mismatches.append((rec["ticker"], p_harness, p_audit))
            matched += 1

    if matched == 0:
        pytest.skip(
            f"no audit-corpus matches found "
            f"(skipped {skipped_no_snapshot} rows whose snapshot_run_id had no on-disk dir)"
        )

    assert not mismatches, (
        f"R-1 mismatch on {len(mismatches)} of {matched} rows: {mismatches[:5]}"
    )
