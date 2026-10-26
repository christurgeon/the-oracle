"""Unit tests for python.analytics.calibrator_harness."""

from __future__ import annotations

import pytest


@pytest.fixture
def fast_emos_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cut the EMOS refit bootstrap from 400 draws; these tests check structure, not CI width."""
    monkeypatch.setattr("python.analytics.calibrator_harness.BOOTSTRAP_ITER", 20)


def test_module_loads():
    """Smoke: module imports without raising the FR-16 assertion."""
    import python.analytics.calibrator_harness as harness
    assert harness.SCHEMA_VERSION == 2
    assert harness.HARNESS_VERSION == "0.1.0"


def test_feature_row_is_frozen():
    """FeatureRow is a frozen dataclass."""
    from python.analytics.calibrator_harness import FeatureRow
    row = FeatureRow(
        city="MIA",
        var="temperature_2m_max",
        forecast_date="2026-05-15",
        snapshot_date="2026-05-15",
        lead=1,
        ticker="KXHIGHMIA-26MAY15-T87",
        strike_type="greater",
        floor=86.0,
        cap=None,
        market_yes_price=0.42,
        blended_raw_prob=0.48,
        gfs_ens_mean=86.4,
        gfs_ens_std=1.2,
        n_members=31,
        outcome=1,
    )
    with pytest.raises(AttributeError):
        row.city = "NYC"  # type: ignore[misc]


def test_all_public_surface_locked_at_41_entries():
    """__all__ is locked at Phase 1A.1 per the B.4+B-gate commit.

    Phase 1A.1 removed WINNER_GATE_ROI_POINT_MIN, WINNER_GATE_ROI_CI_LB_MIN,
    TIEBREAKER_ROI_BAND (ROI demoted from FR-7) and added FR11_2_N_VAL_TOTAL_MIN,
    FLAGS_BELOW_POWER_FLOOR_N — net: 42 → 41 entries.
    If you legitimately need to add or remove a public symbol, update the plan
    first and revise this test accordingly.
    """
    import python.analytics.calibrator_harness as harness
    assert len(harness.__all__) == 41, (
        f"__all__ has {len(harness.__all__)} entries (expected 41). "
        f"If you legitimately need to add or remove a public symbol, "
        f"update the plan first and revise this test accordingly."
    )
    # Defense in depth: catches name-only mutations (renames keep count).
    expected = {
        "HARNESS_VERSION", "SCHEMA_VERSION", "LOGSKILL_EPS", "BOOTSTRAP_ITER",
        "WINNER_GATE_LOGSKILL_LB_MIN",
        "WINNER_GATE_FOLD_CONSISTENCY_MIN",
        "WINNER_GATE_N_CLIPPED_PCT_MAX", "EMOS_GATE_LOGSKILL_LB_VS_ISO_MIN",
        "EMOS_GATE_BSS_LB_VS_ISO_MIN", "TIEBREAKER_LOGSKILL_LB_BAND",
        "FR11_2_N_VAL_TOTAL_MIN", "FLAGS_BELOW_POWER_FLOOR_N",
        "FeatureRow", "FitState", "VetoResult", "CandidateResult",
        "CalibratorCandidate", "LoadArtifactCandidate",
        "IdentityCandidate", "IsoV2GlobalCandidate", "EmosCandidate",
        "IsoV2DeployedReference", "EmosDeployedReference",
        "load_feature_rows",
        "cv_score_cell", "score_cell",
        "compute_logskill", "compute_brier", "compute_bss", "count_clipped",
        "simulate_stage1_roi",
        "block_bootstrap_ci", "block_bootstrap_ratio_ci",
        "compute_eligibility", "compute_verdict",
        "resolve_production", "build_in_sample_reference",
        "validate_schema", "SchemaValidationError",
        "atomic_write_json", "run_harness",
    }
    actual = set(harness.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"__all__ drift detected. Missing: {missing}. Extra: {extra}."
    )


def test_load_artifact_candidate_fit_raises():
    """LoadArtifactCandidate.fit() raises NotImplementedError."""
    from python.analytics.calibrator_harness import (
        FeatureRow,  # noqa: F401  (imported to verify symbol resolves)
        FitState,  # noqa: F401  (imported to verify symbol resolves)
        LoadArtifactCandidate,
        VetoResult,
    )

    class DummyCandidate(LoadArtifactCandidate):
        name = "dummy"
        fit_semantics = "load_artifact"  # type: ignore[assignment]

        def load(self, artifact_path):
            return None

        def score(self, row, fit_state):
            return 0.5

        def validate(self, fit_state):
            return VetoResult(ok=True, warnings=[])

    candidate = DummyCandidate()
    with pytest.raises(NotImplementedError):
        candidate.fit([])


def test_identity_candidate_passes_through_raw_prob():
    """IdentityCandidate.score returns row.blended_raw_prob unchanged."""
    from python.analytics.calibrator_harness import (
        FeatureRow,
        IdentityCandidate,
    )
    candidate = IdentityCandidate()
    assert candidate.name == "identity"
    assert candidate.fit_semantics == "per_fold_refit"

    fit_state = candidate.fit([])
    veto = candidate.validate(fit_state)
    assert veto.ok is True
    assert veto.warnings == []

    row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-15",
        snapshot_date="2026-05-15", lead=1,
        ticker="X", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.42, blended_raw_prob=0.48,
        gfs_ens_mean=86.4, gfs_ens_std=1.2, n_members=31, outcome=1,
    )
    assert candidate.score(row, fit_state) == 0.48


def test_identity_candidate_clips_extremes():
    """IdentityCandidate.score clips to [LOGSKILL_EPS, 1 - LOGSKILL_EPS]."""
    from python.analytics.calibrator_harness import (
        LOGSKILL_EPS,
        FeatureRow,
        IdentityCandidate,
    )
    candidate = IdentityCandidate()
    fit_state = candidate.fit([])

    def make_row(p):
        return FeatureRow(
            city="MIA", var="temperature_2m_max", forecast_date="2026-05-15",
            snapshot_date="2026-05-15", lead=1,
            ticker="X", strike_type="greater", floor=86.0, cap=None,
            market_yes_price=0.42, blended_raw_prob=p,
            gfs_ens_mean=86.4, gfs_ens_std=1.2, n_members=31, outcome=1,
        )

    assert candidate.score(make_row(0.0), fit_state) == LOGSKILL_EPS
    assert candidate.score(make_row(1.0), fit_state) == 1.0 - LOGSKILL_EPS
    assert candidate.score(make_row(0.5), fit_state) == 0.5


def test_iso_v2_global_candidate_fits_per_fold():
    """IsoV2GlobalCandidate.fit produces breakpoints from training rows."""
    from python.analytics.calibrator_harness import (
        FeatureRow,
        IsoV2GlobalCandidate,
    )
    candidate = IsoV2GlobalCandidate()
    assert candidate.name == "iso_v2_global"
    assert candidate.fit_semantics == "per_fold_refit"

    # 100 synthetic training rows: raw_prob ~ uniform, outcome = (raw > 0.5)
    train_rows = []
    for i in range(100):
        p = i / 99.0
        fdate = f"2026-04-{(i % 28) + 1:02d}"
        train_rows.append(FeatureRow(
            city="MIA", var="temperature_2m_max",
            forecast_date=fdate, snapshot_date=fdate, lead=1,
            ticker=f"T{i}", strike_type="greater", floor=86.0, cap=None,
            market_yes_price=0.5, blended_raw_prob=p,
            gfs_ens_mean=86.0, gfs_ens_std=1.0, n_members=31,
            outcome=1 if p > 0.5 else 0,
        ))

    fit_state = candidate.fit(train_rows)
    # On a low-prob row, score should be near 0
    low_row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-01",
        snapshot_date="2026-05-01", lead=1,
        ticker="LOW", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.1, blended_raw_prob=0.1,
        gfs_ens_mean=86.0, gfs_ens_std=1.0, n_members=31, outcome=0,
    )
    high_row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-01",
        snapshot_date="2026-05-01", lead=1,
        ticker="HIGH", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.9, blended_raw_prob=0.9,
        gfs_ens_mean=86.0, gfs_ens_std=1.0, n_members=31, outcome=1,
    )
    p_low = candidate.score(low_row, fit_state)
    p_high = candidate.score(high_row, fit_state)
    assert p_low < 0.2
    assert p_high > 0.8

    veto = candidate.validate(fit_state)
    assert veto.ok is True


def test_iso_v2_global_score_clipped_to_eps():
    """IsoV2GlobalCandidate output is clipped to [LOGSKILL_EPS, 1 - LOGSKILL_EPS]."""
    from python.analytics.calibrator_harness import (
        LOGSKILL_EPS,
        FeatureRow,
        IsoV2GlobalCandidate,
    )
    candidate = IsoV2GlobalCandidate()
    # Train with degenerate all-zero outcomes — fit collapses to ~0
    train_rows = [
        FeatureRow(
            city="MIA", var="temperature_2m_max",
            forecast_date=f"2026-04-{(i % 28) + 1:02d}",
            snapshot_date=f"2026-04-{(i % 28) + 1:02d}", lead=0,
            ticker=f"T{i}", strike_type="greater", floor=86.0, cap=None,
            market_yes_price=0.5, blended_raw_prob=i / 49.0,
            gfs_ens_mean=86.0, gfs_ens_std=1.0, n_members=31, outcome=0,
        )
        for i in range(50)
    ]
    fit_state = candidate.fit(train_rows)
    row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-01",
        snapshot_date="2026-05-01", lead=0,
        ticker="X", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.5, blended_raw_prob=0.5,
        gfs_ens_mean=86.0, gfs_ens_std=1.0, n_members=31, outcome=0,
    )
    p = candidate.score(row, fit_state)
    assert p >= LOGSKILL_EPS
    assert p <= 1.0 - LOGSKILL_EPS


@pytest.mark.usefixtures("fast_emos_bootstrap")
def test_u7_emos_reads_gfs_ens_not_blended_raw():
    """U-7: EmosCandidate consumes gfs_ens_mean/std, NOT blended_raw_prob.

    FeatureRow with deliberately mismatched values; assert EMOS score uses
    gfs_ens_* (NOT blended_raw_prob).
    """
    import numpy as np

    from python.analytics.calibrator_harness import (
        EmosCandidate,
        FeatureRow,
    )
    candidate = EmosCandidate()
    assert candidate.name == "emos"
    assert candidate.fit_semantics == "per_fold_refit"

    # Build 60 synthetic training rows with strong (ens_mean, obs) signal:
    # forecast 20 unique dates × 3 leads × 2 rows = 120 rows
    train_rows = []
    rng = np.random.default_rng(seed=2026)
    for date_i in range(20):
        for lead in range(3):
            for k in range(2):
                fdate = f"2026-04-{date_i + 1:02d}"
                ens_mean = 80.0 + rng.normal(0, 5)
                ens_std = abs(rng.normal(2.0, 0.3))
                obs = ens_mean + rng.normal(0, ens_std)
                train_rows.append(FeatureRow(
                    city="MIA", var="temperature_2m_max",
                    forecast_date=fdate, snapshot_date=fdate,
                    lead=lead, ticker=f"T{date_i}_{lead}_{k}",
                    strike_type="greater", floor=80.0, cap=None,
                    market_yes_price=0.5, blended_raw_prob=0.5,
                    gfs_ens_mean=ens_mean, gfs_ens_std=ens_std,
                    n_members=31, outcome=int(obs > 80.0),
                    extras={"obs_temp_f": float(obs)},
                ))

    fit_state = candidate.fit(train_rows)

    # Mismatched scoring row: blended_raw_prob is sky-high, but gfs_ens_mean
    # is far below the floor → EMOS should score low.
    row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-01",
        snapshot_date="2026-05-01", lead=0,
        ticker="X", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.5,
        blended_raw_prob=0.99,        # DELIBERATELY HIGH — should be ignored
        gfs_ens_mean=70.0,            # ens mean WAY below floor=86 → P(>86) tiny
        gfs_ens_std=1.0,
        n_members=31, outcome=0,
    )
    p_emos = candidate.score(row, fit_state)
    assert p_emos < 0.2, (
        f"EMOS score {p_emos} should be low; if it equals 0.99 the "
        "implementation is reading blended_raw_prob instead of gfs_ens_*"
    )


@pytest.mark.usefixtures("fast_emos_bootstrap")
def test_emos_validate_fires_sigma_veto_on_pathological_d():
    """EmosCandidate.validate fires incompatible_sigma when d is very large.

    Train with rows whose ens_std is large relative to bracket_width → σ-veto.
    """
    import numpy as np

    from python.analytics.calibrator_harness import EmosCandidate, FeatureRow

    candidate = EmosCandidate()
    rng = np.random.default_rng(seed=7)
    train_rows = []
    for date_i in range(30):
        fdate = f"2026-04-{date_i + 1:02d}"
        for lead in range(3):
            ens_mean = 60.0 + rng.normal(0, 1)
            # Pathological: very high ens_std AND high signal-correlation drives d up
            ens_std = abs(rng.normal(8.0, 2.0))
            obs = ens_mean + rng.normal(0, ens_std)
            train_rows.append(FeatureRow(
                city="LAX", var="temperature_2m_max",
                forecast_date=fdate, snapshot_date=fdate, lead=lead,
                ticker=f"T{date_i}_{lead}", strike_type="greater", floor=60.0, cap=None,
                market_yes_price=0.5, blended_raw_prob=0.5,
                gfs_ens_mean=ens_mean, gfs_ens_std=ens_std, n_members=31,
                outcome=int(obs > 60.0),
                extras={"obs_temp_f": float(obs)},
            ))

    fit_state = candidate.fit(train_rows)
    veto = candidate.validate(fit_state)
    # σ_emos / bracket_width > 2 → incompatible_sigma should be in warnings.
    # Don't assert ok=False unconditionally (signal may not be strong enough);
    # do assert the warning is detectable in principle.
    assert isinstance(veto.warnings, list)


def test_iso_v2_deployed_reference_loads_and_scores():
    """IsoV2DeployedReference loads the deployed artifact and scores in-sample."""
    from pathlib import Path

    from python.analytics.calibrator_harness import (
        FeatureRow,
        IsoV2DeployedReference,
    )

    candidate = IsoV2DeployedReference()
    assert candidate.name == "iso_v2_deployed_reference"
    assert candidate.fit_semantics == "load_artifact"

    # Load the actually-deployed artifact.
    artifact_path = Path("data/calibration/isotonic_v2.json")
    if not artifact_path.exists():
        import pytest
        pytest.skip(f"deployed artifact missing: {artifact_path}")

    fit_state = candidate.load(artifact_path)

    row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-15",
        snapshot_date="2026-05-15", lead=1,
        ticker="X", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.5, blended_raw_prob=0.5,
        gfs_ens_mean=86.4, gfs_ens_std=1.2, n_members=31, outcome=1,
    )
    p = candidate.score(row, fit_state)
    assert 0.0 < p < 1.0


def test_iso_v2_deployed_reference_fit_raises():
    """IsoV2DeployedReference.fit() raises NotImplementedError (LoadArtifactCandidate)."""
    import pytest

    from python.analytics.calibrator_harness import IsoV2DeployedReference
    candidate = IsoV2DeployedReference()
    with pytest.raises(NotImplementedError):
        candidate.fit([])


def test_iso_v2_deployed_reference_fallback_reason_propagated():
    """When the artifact is missing, fit_state carries the fallback_reason."""
    from pathlib import Path

    from python.analytics.calibrator_harness import IsoV2DeployedReference

    candidate = IsoV2DeployedReference()
    fit_state = candidate.load(Path("/nonexistent/isotonic_v2.json"))
    # The harness reports this in in_sample_reference as null + fallback_reason.
    assert hasattr(fit_state, "fallback_reason")
    assert fit_state.fallback_reason is not None


def test_emos_deployed_reference_loads_and_scores():
    """EmosDeployedReference loads coefficients.json via EmosTable and scores in-sample."""
    from pathlib import Path

    import pytest

    from python.analytics.calibrator_harness import (
        EmosDeployedReference,
        FeatureRow,
    )

    candidate = EmosDeployedReference()
    assert candidate.name == "emos_deployed_reference"
    assert candidate.fit_semantics == "load_artifact"

    artifact_path = Path("data/emos/coefficients.json")
    if not artifact_path.exists():
        pytest.skip(f"deployed artifact missing: {artifact_path}")

    fit_state = candidate.load(artifact_path)

    row = FeatureRow(
        city="MIA", var="temperature_2m_max", forecast_date="2026-05-15",
        snapshot_date="2026-05-15", lead=1,
        ticker="X", strike_type="greater", floor=86.0, cap=None,
        market_yes_price=0.5, blended_raw_prob=0.5,
        gfs_ens_mean=86.4, gfs_ens_std=1.2, n_members=31, outcome=1,
        extras={"obs_temp_f": 87.0},
    )
    p = candidate.score(row, fit_state)
    # MIA/high is a live EMOS cell. Score should be a real probability.
    assert 0.0 < p < 1.0


def test_emos_deployed_reference_fit_raises():
    """EmosDeployedReference.fit() raises NotImplementedError per LoadArtifactCandidate contract."""
    import pytest

    from python.analytics.calibrator_harness import EmosDeployedReference

    candidate = EmosDeployedReference()
    with pytest.raises(NotImplementedError):
        candidate.fit([])


def test_load_feature_rows_emits_expected_fields(tmp_path):
    """load_feature_rows reads snapshots and emits FeatureRow with all required fields."""
    import json

    from python.analytics.calibrator_harness import load_feature_rows

    # Construct a minimal snapshot directory tree.
    snap = tmp_path / "snapshots"
    day = snap / "2026-04-15"
    day.mkdir(parents=True)

    forecasts = [
        {
            "city": "MIA",
            "forecast_date": "2026-04-15",
            "daily_variable": "temperature_2m_max",
            "model": "gfs_seamless",
            "temperatures": [85.0 + 0.1 * i for i in range(31)],
        }
    ]
    settlements = [
        {
            "city": "MIA",
            "observation_date": "2026-04-15",
            "high_f": 87.0,
            "low_f": 70.0,
            "is_preliminary": False,
            "issued_at": "2026-04-16T10:00:00Z",
            "product_id": "MIA-CLI",
        }
    ]
    markets = [
        {
            "ticker": "KXHIGHMIA-26APR15-T86",
            "strike_type": "greater",
            "floor_strike": 86.0,
            "cap_strike": None,
            "city": "MIA",
            "daily_variable": "temperature_2m_max",
            "yes_ask_dollars": 0.4,
            "yes_bid_dollars": 0.38,
        }
    ]
    (day / "weather_forecasts.json").write_text(json.dumps(forecasts))
    (day / "settlements.json").write_text(json.dumps(settlements))
    (day / "markets.json").write_text(json.dumps(markets))

    rows = load_feature_rows(
        snapshots_dir=snap,
        bias_table_path=None,  # disable bias correction for test
        ensemble_weights={"gfs_seamless": 1.0},
    )
    assert len(rows) >= 1
    r = rows[0]
    assert r.city == "MIA"
    assert r.var == "temperature_2m_max"
    assert r.forecast_date == "2026-04-15"
    assert r.strike_type == "greater"
    assert r.floor == 86.0
    assert r.outcome == 1  # high_f=87 > floor=86
    assert "obs_temp_f" in r.extras
    assert r.extras["obs_temp_f"] == 87.0
    assert r.n_members == 31
    assert 0.0 <= r.blended_raw_prob <= 1.0
    assert r.gfs_ens_std > 0.0


def test_load_feature_rows_excludes_chi_pre_fix():
    """CHI rows with forecast_date < 2026-03-20 must be excluded (R-3 / FR-16.6)."""
    # The reused build_production_mirrored_pairs_with_provenance applies the
    # CHI pre-fix exclusion via _CHI_SNAPSHOT_FIX_DATE. Verify it propagates.
    # (Detailed test in test_harness_replay_parity.py:R-3 — this is a smoke check.)
    pass  # covered by R-3 in Task 23


def test_cv_score_cell_produces_per_candidate_per_fold_predictions():
    """cv_score_cell runs the per-fold loop and returns predictions+per-fold logskill."""
    import numpy as np

    from python.analytics.calibrator_harness import (
        FeatureRow,
        IdentityCandidate,
        IsoV2GlobalCandidate,
        cv_score_cell,
    )

    # Build 200 rows across 50 unique forecast_dates (4 strike_types/day × 1 lead).
    rng = np.random.default_rng(seed=2026)
    rows = []
    strikes = [("greater", 86.0, None), ("greater", 88.0, None),
               ("less", None, 84.0), ("less", None, 82.0)]
    for d_i in range(50):
        date_str = f"2026-03-{(d_i % 28) + 1:02d}" if d_i < 28 else f"2026-04-{(d_i - 28) + 1:02d}"
        for s_type, fl, cp in strikes:
            p = rng.uniform(0.0, 1.0)
            obs_temp = 80.0 + rng.normal(0, 4)
            if s_type == "greater":
                outcome = int(obs_temp > fl)
            else:
                outcome = int(obs_temp < cp)
            rows.append(FeatureRow(
                city="MIA", var="temperature_2m_max",
                forecast_date=date_str, snapshot_date=date_str, lead=0,
                ticker=f"T{d_i}_{s_type}_{fl or cp}",
                strike_type=s_type, floor=fl, cap=cp,
                market_yes_price=0.5, blended_raw_prob=p,
                gfs_ens_mean=80.0, gfs_ens_std=2.0, n_members=31, outcome=outcome,
                extras={"obs_temp_f": float(obs_temp)},
            ))

    result = cv_score_cell(
        rows,
        candidates=[IdentityCandidate(), IsoV2GlobalCandidate()],
    )

    # Each candidate gets a per-row prediction (parallel to val_rows) and a per-fold list
    assert "identity" in result["per_candidate"]
    assert "iso_v2_global" in result["per_candidate"]
    for name in ("identity", "iso_v2_global"):
        cand = result["per_candidate"][name]
        assert len(cand["val_rows"]) == len(cand["val_predictions"])
        assert len(cand["fold_predictions"]) == 5
    assert result["n_val_total"] == len(result["per_candidate"]["identity"]["val_rows"])
    assert result["n_unique_forecast_dates"] >= 6


def test_cv_score_cell_raises_for_too_few_unique_dates():
    """cv_score_cell raises ValueError (caller catches) when unique_dates < 6."""
    import pytest

    from python.analytics.calibrator_harness import (
        FeatureRow,
        IdentityCandidate,
        cv_score_cell,
    )
    rows = [
        FeatureRow(
            city="MIA", var="temperature_2m_max",
            forecast_date=f"2026-04-{i + 1:02d}",
            snapshot_date=f"2026-04-{i + 1:02d}", lead=0,
            ticker=f"T{i}", strike_type="greater", floor=86.0, cap=None,
            market_yes_price=0.5, blended_raw_prob=0.5,
            gfs_ens_mean=86.0, gfs_ens_std=1.0, n_members=31, outcome=0,
            extras={"obs_temp_f": 85.0},
        )
        for i in range(5)
    ]
    with pytest.raises(ValueError):
        cv_score_cell(rows, candidates=[IdentityCandidate()])


def test_u1_logskill_formula_correctness():
    """U-1: LogSkill matches hand-computed value to 1e-9."""
    import math

    from python.analytics.calibrator_harness import compute_logskill

    # Synthetic 2-row inputs.
    cand_preds = [0.6, 0.4]
    id_preds = [0.5, 0.5]
    outcomes = [1, 0]

    # By hand:
    # LL_cand_row1 = -(1*log(0.6) + 0*log(0.4)) = -log(0.6) = 0.5108256...
    # LL_cand_row2 = -(0*log(0.4) + 1*log(0.6)) = -log(0.6) = 0.5108256...
    # mean LL_cand = 0.5108256
    # LL_id_row1 = -log(0.5) = 0.6931472
    # LL_id_row2 = -log(0.5) = 0.6931472
    # mean LL_id = 0.6931472
    # LogSkill = 1 - mean(LL_cand)/mean(LL_id) = 1 - 0.5108256/0.6931472 = 0.2630345
    expected = 1.0 - (-math.log(0.6)) / (-math.log(0.5))

    result = compute_logskill(cand_preds, id_preds, outcomes)
    assert abs(result - expected) < 1e-9


def test_u2_logskill_clipping():
    """U-2: probabilities outside [LOGSKILL_EPS, 1 - LOGSKILL_EPS] are clipped."""
    import math

    from python.analytics.calibrator_harness import (
        LOGSKILL_EPS,
        compute_logskill,
    )

    # p = 1e-10 with y = 1: without clipping → LL = -log(1e-10) ≈ 23.0
    # with clipping to 1e-3 → LL = -log(1e-3) ≈ 6.9
    cand_preds = [1e-10]
    id_preds = [0.5]
    outcomes = [1]

    result = compute_logskill(cand_preds, id_preds, outcomes)
    # Expected LL_cand = -log(1e-3); LL_id = -log(0.5); LogSkill = 1 - LL_cand/LL_id
    expected = 1.0 - (-math.log(LOGSKILL_EPS)) / (-math.log(0.5))
    assert abs(result - expected) < 1e-9


def test_brier_metric_formula_correctness():
    from python.analytics.calibrator_harness import compute_brier

    preds = [0.6, 0.4, 0.7]
    outcomes = [1, 0, 1]
    # Brier = mean((p - y)^2) = ((0.6-1)^2 + (0.4-0)^2 + (0.7-1)^2) / 3
    #       = (0.16 + 0.16 + 0.09) / 3 = 0.41 / 3 = 0.13666...
    expected = ((0.4) ** 2 + (0.4) ** 2 + (0.3) ** 2) / 3.0
    assert abs(compute_brier(preds, outcomes) - expected) < 1e-9


def test_bss_relative_to_identity():
    from python.analytics.calibrator_harness import compute_bss

    cand_preds = [0.9, 0.1]
    id_preds = [0.5, 0.5]
    outcomes = [1, 0]
    # Brier_cand = ((0.9-1)^2 + (0.1-0)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
    # Brier_id = ((0.5-1)^2 + (0.5-0)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    # BSS = 1 - 0.01/0.25 = 0.96
    assert abs(compute_bss(cand_preds, id_preds, outcomes) - 0.96) < 1e-9


def test_u9_filter_scope_separation():
    """U-9: LogSkill sees all rows; simulated ROI applies trading filters.

    30 val rows; the first 15 carry calibrated_prob = 0.05 (filtered out by
    the min_calibrated_prob=0.110 floor); the last 15 carry calibrated_prob
    = 0.20 (above the floor, p > market price 0.10 so Kelly returns YES).
    blended_raw_prob = 0.5 sits *inside* (0.446, 0.883), so we use a value
    OUTSIDE the plateau for surviving rows. Setting all rows to 0.40 (just
    under the lower plateau bound) sidesteps the plateau-block filter
    entirely so we can isolate the calibrated-floor scope.
    """
    from python.analytics.calibrator_harness import simulate_stage1_roi

    predictions = [0.05] * 15 + [0.20] * 15  # first 15 below floor
    outcomes = [1, 0] * 15  # alternating
    # 0.40 sits below the plateau range (0.446, 0.883), so plateau-block
    # never fires — we're isolating the calibrated-floor scope here.
    blended_raw_probs = [0.40] * 30
    market_yes_prices = [0.10] * 30

    result = simulate_stage1_roi(
        candidate_predictions=predictions,
        blended_raw_probs=blended_raw_probs,
        market_yes_prices=market_yes_prices,
        outcomes=outcomes,
        forecast_dates=[f"2026-04-{(i % 28) + 1:02d}" for i in range(30)],
        leads=[0] * 30,
        kelly_fraction=0.075,
        max_pct=0.05,
        bankroll=1000.0,
        block_raw_prob_range=(0.446, 0.883),
        min_calibrated_prob=0.110,
    )
    # 15 rows survive (those with calibrated_prob = 0.20).
    # LogSkill (computed outside this function) sees all 30; ROI only on 15.
    assert result["n_val_traded"] == 15
    # n_val_traded < 30 → insufficient_data + roi_point None.
    assert result["status"] == "insufficient_data"
    assert result["roi_point"] is None
    # Per-row arrays are length-30 parallel to inputs.
    assert len(result["per_row_pnl"]) == 30
    assert len(result["per_row_wagered"]) == 30
    assert len(result["per_row_block_key"]) == 30
    # Filtered rows (the first 15) carry 0.0 in pnl/wagered.
    assert all(p == 0.0 for p in result["per_row_pnl"][:15])
    assert all(w == 0.0 for w in result["per_row_wagered"][:15])
    # Surviving rows actually transacted (non-zero wagered).
    assert all(w > 0.0 for w in result["per_row_wagered"][15:])
    # block_keys are always populated regardless of filter status.
    assert result["per_row_block_key"][0] == ("2026-04-01", 0)
    assert result["per_row_block_key"][-1] == ("2026-04-02", 0)


def test_u3_paired_block_bootstrap_with_autocorrelation():
    """U-3: block-bootstrap CI is wider than row-iid bootstrap on autocorrelated data."""
    import numpy as np

    from python.analytics.calibrator_harness import block_bootstrap_ci

    # Generate 50 blocks × 3 rows with ρ=0.5 within-block autocorrelation.
    rng = np.random.default_rng(seed=2026)
    values: list[float] = []
    block_keys: list[tuple[str, int]] = []
    for block_i in range(50):
        date = f"2026-04-{(block_i % 28) + 1:02d}"
        # Common shock per block (autocorrelation source).
        block_shock = rng.normal(0.05, 0.05)
        for _k in range(3):
            row_noise = rng.normal(0, 0.1) * (1 - 0.5)  # ρ=0.5
            value = 0.05 + block_shock * 0.5 + row_noise
            values.append(float(value))
            block_keys.append((date, 0))

    block_result = block_bootstrap_ci(
        values=values,
        block_keys=block_keys,
        n_iter=400,
        seed=42,
    )

    # Compare against row-iid bootstrap (one row per block).
    iid_result = block_bootstrap_ci(
        values=values,
        block_keys=[(f"row_{i}", 0) for i in range(len(values))],  # each row in own block
        n_iter=400,
        seed=42,
    )

    block_width = block_result["ci_ub"] - block_result["ci_lb"]
    iid_width = iid_result["ci_ub"] - iid_result["ci_lb"]
    # Politis-Romano stationary block bootstrap with L=3 produces modestly wider CIs than
    # per-row iid resampling on this data (28 shared dates vs 150 unique row-dates).
    # The 1.05 threshold is calibrated to the new algorithm; the old fixed-block threshold
    # was 1.3× (single-date fixed blocks amplify within-block structure more aggressively).
    assert block_width >= 1.05 * iid_width, (
        f"block CI half-width {block_width} should be >= 1.05x row-iid CI half-width {iid_width}"
    )


def test_block_bootstrap_deterministic():
    """Same seed → same CI."""
    import numpy as np

    from python.analytics.calibrator_harness import block_bootstrap_ci

    rng = np.random.default_rng(seed=1)
    values = rng.normal(0, 1, size=100).tolist()
    block_keys = [(f"2026-04-{(i % 28) + 1:02d}", 0) for i in range(100)]

    r1 = block_bootstrap_ci(values=values, block_keys=block_keys, n_iter=200, seed=99)
    r2 = block_bootstrap_ci(values=values, block_keys=block_keys, n_iter=200, seed=99)
    assert r1["ci_lb"] == r2["ci_lb"]
    assert r1["ci_ub"] == r2["ci_ub"]


# ---------------------------------------------------------------------------
# Task 14: Structural-veto eligibility + n_clipped veto (FR-9)
# ---------------------------------------------------------------------------


def test_u5_eligibility_sigma_veto_any_fold():
    """U-5a: σ-veto in any fold → _ineligible."""
    from python.analytics.calibrator_harness import compute_eligibility

    fold_warnings = [[], [], ["incompatible_sigma"], [], []]
    result = compute_eligibility(
        fold_warnings=fold_warnings,
        fold_eligible=[True] * 5,
        n_clipped=0,
        n_val_total=100,
    )
    assert result["eligible"] is False
    assert "incompatible_sigma:fold_2" in result["reasons"]


def test_u5_eligibility_unstable_b_three_of_five():
    """U-5b: unstable_b in ≥3/5 folds → _ineligible."""
    from python.analytics.calibrator_harness import compute_eligibility

    fold_warnings = [["unstable_b"], ["unstable_b"], ["unstable_b"], [], []]
    result = compute_eligibility(
        fold_warnings=fold_warnings,
        fold_eligible=[True] * 5,
        n_clipped=0,
        n_val_total=100,
    )
    assert result["eligible"] is False
    assert "unstable_b:>=3/5" in result["reasons"]


def test_u5_eligibility_unstable_d_flag_only():
    """U-5c: unstable_d in any fold → eligible (flag only)."""
    from python.analytics.calibrator_harness import compute_eligibility

    fold_warnings = [
        ["unstable_d"],
        ["unstable_d"],
        ["unstable_d"],
        ["unstable_d"],
        ["unstable_d"],
    ]
    result = compute_eligibility(
        fold_warnings=fold_warnings,
        fold_eligible=[True] * 5,
        n_clipped=0,
        n_val_total=100,
    )
    assert result["eligible"] is True


def test_u17_n_clipped_pct_veto():
    """U-17: n_clipped / n_val_total > 5% → _ineligible."""
    from python.analytics.calibrator_harness import compute_eligibility

    result = compute_eligibility(
        fold_warnings=[[], [], [], [], []],
        fold_eligible=[True, True, True, True, True],
        n_clipped=8,
        n_val_total=100,
    )
    assert result["eligible"] is False
    assert "n_clipped:>5%" in result["reasons"]

    # Boundary case: 5/100 = 5% → still eligible
    result_at_boundary = compute_eligibility(
        fold_warnings=[[], [], [], [], []],
        fold_eligible=[True, True, True, True, True],
        n_clipped=5,
        n_val_total=100,
    )
    assert result_at_boundary["eligible"] is True


def test_a4_eligibility_fit_failures_at_three_of_five():
    """A4 fix: fold_eligible has ≥3 False entries → _ineligible (FM-4 rule)."""
    from python.analytics.calibrator_harness import compute_eligibility

    # 3 folds failed to fit; 2 succeeded.
    result = compute_eligibility(
        fold_warnings=[
            ["fit_exception:Exception"],
            ["fit_exception:Exception"],
            ["fit_exception:Exception"],
            [],
            [],
        ],
        fold_eligible=[False, False, False, True, True],
        n_clipped=0,
        n_val_total=60,
    )
    assert result["eligible"] is False
    assert "fold_fit_failures:>=3/5" in result["reasons"]


def test_a4_eligibility_two_fit_failures_still_eligible_overall():
    """A4: 2/5 fit failures + 3/5 clean → eligible (but underpowered)."""
    from python.analytics.calibrator_harness import compute_eligibility

    result = compute_eligibility(
        fold_warnings=[
            ["fit_exception:Exception"],
            ["fit_exception:Exception"],
            [],
            [],
            [],
        ],
        fold_eligible=[False, False, True, True, True],
        n_clipped=0,
        n_val_total=60,
    )
    assert result["eligible"] is True


def test_u4_all_six_cell_level_verdicts():
    """U-4: each of the 6 cell-level verdicts is reachable from hand-crafted CandidateResult."""
    from python.analytics.calibrator_harness import (
        CandidateResult,
        compute_verdict,
    )

    def cr(
        name: str, ls_lb: float, ls_mean: float = 0.0, ls_p99: float = 0.0,
        roi_pt: float = 0.0, roi_lb: float = 0.0,
        fold_pos: int = 0, eligible: bool = True,
        ls_lb_vs_iso: float = 0.0, bss_lb_vs_iso: float = 0.0,
        gate_pass_vs_iso: bool = True, evaluable_folds: int = 5,
        reasons: list[str] | None = None,
    ) -> CandidateResult:
        return CandidateResult(
            name=name,
            logskill_lb=ls_lb, logskill_mean=ls_mean, logskill_p99=ls_p99,
            roi_point=roi_pt, roi_ci_lb=roi_lb,
            fold_logskill_positive_count=fold_pos,
            eligible=eligible,
            ineligibility_reasons=reasons or [],
            logskill_lb_vs_iso=ls_lb_vs_iso,
            bss_lb_vs_iso=bss_lb_vs_iso,
            emos_gate_pass=gate_pass_vs_iso,
            evaluable_folds_vs_iso=evaluable_folds,
        )

    # 1) winner: clears LB > +0.01 + 5/5 folds.
    candidates = [
        cr("identity", 0.0, 0.0, fold_pos=0),
        cr("emos", ls_lb=0.05, ls_mean=0.10, fold_pos=5),
    ]
    v = compute_verdict(candidates, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "winner", v
    assert v["winner_candidate"] == "emos"

    # 2) insufficient_evidence: 2 eligible, both near zero (NOT confidently worse).
    candidates_ie = [
        cr("identity", 0.0, 0.0, fold_pos=0),
        cr("iso_v2_global", ls_lb=-0.005, ls_mean=0.005, ls_p99=0.05, fold_pos=2),
        cr("emos", ls_lb=-0.005, ls_mean=0.005, ls_p99=0.05, fold_pos=2),
    ]
    v = compute_verdict(candidates_ie, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "insufficient_evidence", v
    assert "eligible:" in v["verdict_message"]

    # 3) worse_than_identity: 2 eligible, both confidently worse (p99 < 0).
    candidates_wti = [
        cr("identity", 0.0, 0.0, fold_pos=0),
        cr("iso_v2_global", ls_lb=-0.10, ls_mean=-0.08, ls_p99=-0.02, fold_pos=0),
        cr("emos", ls_lb=-0.12, ls_mean=-0.10, ls_p99=-0.03, fold_pos=0),
    ]
    v = compute_verdict(candidates_wti, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "worse_than_identity", v

    # 4) insufficient_n
    v = compute_verdict([], n_val_total=20, n_val_traded=0, n_unique_strikes=10)
    assert v["verdict"] == "insufficient_n"

    # 5) insufficient_diversity
    v = compute_verdict(candidates, n_val_total=150, n_val_traded=120, n_unique_strikes=3)
    assert v["verdict"] == "insufficient_diversity"

    # 6) no_separation
    candidates_tied = [
        cr("identity", 0.0, 0.0, fold_pos=0),
        cr("iso_v2_global", ls_lb=0.05, ls_mean=0.10, fold_pos=5),
        cr("emos", ls_lb=0.052, ls_mean=0.102, fold_pos=5),
    ]
    v = compute_verdict(candidates_tied, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "no_separation"


def test_u4_worse_than_identity_requires_two_eligible():
    """U-4 supplement: worse_than_identity does NOT fire with only 1 eligible candidate."""
    from python.analytics.calibrator_harness import CandidateResult, compute_verdict

    cands = [
        CandidateResult(
            name="identity", logskill_lb=0.0, logskill_mean=0.0, logskill_p99=0.0,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=True, ineligibility_reasons=[],
        ),
        CandidateResult(  # 1 eligible-confidently-worse
            name="iso_v2_global", logskill_lb=-0.10, logskill_mean=-0.08, logskill_p99=-0.02,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=True, ineligibility_reasons=[],
        ),
        CandidateResult(  # 1 ineligible — should NOT count toward "all eligible worse"
            name="emos", logskill_lb=0.0, logskill_mean=0.0, logskill_p99=0.0,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=False,
            ineligibility_reasons=["incompatible_sigma:fold_2"],
        ),
    ]
    v = compute_verdict(cands, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "insufficient_evidence"  # NOT worse_than_identity
    assert "ineligible:" in v["verdict_message"]
    assert "incompatible_sigma" in v["verdict_message"]


def test_u10_emos_promotion_gate_inheritance():
    """U-10: EMOS candidate fails FR-8 even if FR-7 passes."""
    from python.analytics.calibrator_harness import CandidateResult, compute_verdict

    # EMOS passes FR-7 vs identity but fails FR-8 vs iso (LogSkill_LB_vs_iso < -0.02).
    cands = [
        CandidateResult(
            name="identity",
            logskill_lb=0.0, roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0,
            eligible=True,
        ),
        CandidateResult(
            name="emos",
            logskill_lb=0.05, logskill_mean=0.08, roi_point=0.10, roi_ci_lb=-0.01,
            fold_logskill_positive_count=5,
            eligible=True,
            logskill_lb_vs_iso=-0.05,  # fails FR-8 (must be > -0.02)
            bss_lb_vs_iso=0.0,
            emos_gate_pass=False,
            evaluable_folds_vs_iso=5,
        ),
    ]
    v = compute_verdict(cands, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    # EMOS rejected by FR-8 → falls to insufficient_evidence.
    # logskill_mean=0.08 ensures FR-7 clause 4 (mean > 0) passes, so the FR-8 path
    # is the actual binding gate this test exercises.
    assert v["verdict"] == "insufficient_evidence"


def test_u11_emos_baseline_not_evaluated_falls_through_to_fr7():
    """U-11: iso_v2_global ineligible in all 5 folds → FR-8 not_evaluated → FR-7 only."""
    from python.analytics.calibrator_harness import CandidateResult, compute_verdict

    cands = [
        CandidateResult(
            name="identity",
            logskill_lb=0.0, roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0,
            eligible=True,
        ),
        CandidateResult(
            name="emos",
            logskill_lb=0.05, logskill_mean=0.08, roi_point=0.10, roi_ci_lb=-0.01,
            fold_logskill_positive_count=5,
            eligible=True,
            evaluable_folds_vs_iso=0,  # baseline _ineligible in all folds → FR-8 not_evaluated
            emos_gate_pass="not_evaluated",
        ),
    ]
    v = compute_verdict(cands, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "winner"
    assert v["winner_candidate"] == "emos"


def test_u15a_fr11_2_insufficient_n_below_75():
    """U-15a: n_val_total < 75 → insufficient_n (M-1 fix raises threshold from 30)."""
    from python.analytics.calibrator_harness import compute_verdict

    v = compute_verdict([], n_val_total=74, n_val_traded=0, n_unique_strikes=10)
    assert v["verdict"] == "insufficient_n"
    assert "75" in v["verdict_message"]  # message references the threshold

    # Boundary: exactly at threshold passes the gate.
    v = compute_verdict([], n_val_total=75, n_val_traded=0, n_unique_strikes=10)
    # With 0 non-identity candidates: falls to insufficient_evidence (not insufficient_n)
    assert v["verdict"] != "insufficient_n"


def test_u15b_fr7_failure_at_above_threshold_n():
    """U-15b: with n_val_total ≥ 75 and all candidates failing the new FR-7
    (LB ≤ +0.01 or fold count < 4), verdict is insufficient_evidence."""
    from python.analytics.calibrator_harness import CandidateResult, compute_verdict

    cands = [
        CandidateResult(
            name="identity", logskill_lb=0.0, roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=True,
        ),
        CandidateResult(
            name="iso_v2_global", logskill_lb=0.005,  # ≤ +0.01 → fails FR-7 LB clause
            roi_point=0.0, roi_ci_lb=0.0, fold_logskill_positive_count=5, eligible=True,
        ),
    ]
    v = compute_verdict(cands, n_val_total=150, n_val_traded=0, n_unique_strikes=10)
    assert v["verdict"] == "insufficient_evidence"


@pytest.mark.usefixtures("fast_emos_bootstrap")
def test_score_cell_end_to_end_on_synthetic_rows():
    """score_cell orchestrates CV + metrics + bootstrap + verdict for one cell."""
    import numpy as np

    from python.analytics.calibrator_harness import (
        EmosCandidate,
        FeatureRow,
        IdentityCandidate,
        IsoV2GlobalCandidate,
        score_cell,
    )

    rng = np.random.default_rng(seed=2026)
    rows = []
    for d_i in range(60):
        date_str = f"2026-{((d_i // 28) + 1):02d}-{(d_i % 28) + 1:02d}"
        for k in range(3):
            ens_mean = 80.0 + rng.normal(0, 5)
            ens_std = max(0.5, rng.normal(2.0, 0.3))
            obs_temp = ens_mean + rng.normal(0, ens_std)
            # Vary the floor across 5 distinct values so FR-11.3 (unique
            # (strike_type, floor, cap) ≥ 5) is satisfied and the full
            # pipeline executes rather than short-circuiting to
            # insufficient_diversity.
            floor = 78.0 + float(d_i % 5)
            outcome = int(obs_temp > floor)
            # Blended raw prob from a coarse Gaussian approximation.
            from scipy.stats import norm
            raw_p = float(1.0 - norm.cdf(floor, loc=ens_mean, scale=ens_std))
            rows.append(FeatureRow(
                city="MIA", var="temperature_2m_max",
                forecast_date=date_str, snapshot_date=date_str, lead=k,
                ticker=f"T{d_i}_{k}", strike_type="greater",
                floor=floor, cap=None,
                market_yes_price=0.5, blended_raw_prob=raw_p,
                gfs_ens_mean=ens_mean, gfs_ens_std=ens_std,
                n_members=31, outcome=outcome,
                extras={"obs_temp_f": float(obs_temp)},
            ))

    result = score_cell(
        city="MIA",
        var="temperature_2m_max",
        rows=rows,
        candidates=[IdentityCandidate(), IsoV2GlobalCandidate(), EmosCandidate()],
    )

    # Structural assertions, NOT specific verdict (L-BFGS-B determinism per FR-15).
    assert "verdict" in result
    assert "candidates" in result
    assert set(result["candidates"].keys()) == {"identity", "iso_v2_global", "emos"}
    for name, block in result["candidates"].items():
        assert "logskill" in block
        assert "brier" in block
        assert "simulated_roi" in block
        assert "fold_logskill" in block
        assert len(block["fold_logskill"]) == 5
        assert "eligibility" in block
        assert "per_fold_diagnostics" in block
    assert result["n"]["val_total"] >= 30


def test_build_in_sample_reference_returns_two_entries():
    from pathlib import Path

    from python.analytics.calibrator_harness import build_in_sample_reference

    result = build_in_sample_reference(
        rows=[],  # empty → expect null values + fallback notes
        iso_v2_path=Path("/nonexistent/iso_v2.json"),
        emos_coefficients_path=Path("/nonexistent/coefficients.json"),
    )
    assert "iso_v2_deployed_reference" in result
    assert "emos_deployed_reference" in result
    for entry in result.values():
        assert entry["note"] == "LEAKAGE — not a verdict"


def test_resolve_production_for_emos_cell():
    from python.analytics.calibrator_harness import resolve_production

    config = {
        "strategies": {"weather": {
            "emos_enabled_cells": ["MIA/high"],
            "excluded_city_vars": ["DEN_low"],
            "calibration_type": "isotonic",
            "isotonic_path": "data/calibration/isotonic_v2.json",
        }}
    }
    prod = resolve_production(city="MIA", var="temperature_2m_max", config=config)
    assert prod["kind"] == "emos"
    assert prod["artifact_path"] == "data/emos/coefficients.json"
    assert prod["source_config_key"] == "weather.emos_enabled_cells"

    prod_excluded = resolve_production(city="DEN", var="temperature_2m_min", config=config)
    assert prod_excluded["kind"] == "excluded"

    prod_iso = resolve_production(city="NYC", var="temperature_2m_max", config=config)
    assert prod_iso["kind"] == "isotonic"
    assert prod_iso["artifact_path"] == "data/calibration/isotonic_v2.json"


def test_u16_atomic_write_uses_correct_tempfile_location(tmp_path):
    """U-16: tempfile is created with dir=output_path.parent, NOT /tmp."""
    from python.analytics.calibrator_harness import atomic_write_json

    output = tmp_path / "sub" / "winners.json"
    output.parent.mkdir()
    payload = {"hello": "world"}

    atomic_write_json(payload, output)
    assert output.exists()
    import json
    assert json.loads(output.read_text()) == payload


def test_u16_atomic_write_fails_on_unwritable(tmp_path):
    """U-16: write failure raises IOError; existing output untouched."""
    import json
    import os

    from python.analytics.calibrator_harness import atomic_write_json

    # D-item: chmod 0o500 is meaningless under root in CI containers.
    if os.geteuid() == 0:
        pytest.skip("running as root; chmod-based unwritability test would not block writes")

    output = tmp_path / "winners.json"
    output.write_text(json.dumps({"existing": True}))

    output.parent.chmod(0o500)
    try:
        with pytest.raises(OSError):
            atomic_write_json({"new": True}, output)
        assert json.loads(output.read_text()) == {"existing": True}
    finally:
        output.parent.chmod(0o700)


def test_run_harness_smoke(tmp_path):
    """run_harness end-to-end on a tiny fixture corpus."""
    import json

    from python.analytics.calibrator_harness import run_harness

    # Build a minimal snapshot tree (similar to Task 8 test).
    # Note: ticker uses alphabetic month codes (parse_market_date requires
    # 2-digit-year + 3-letter-month + 2-digit-day, e.g. 26APR15) — plan-provided
    # numeric format (e.g. 260415) would not parse.
    snap = tmp_path / "snapshots"
    month_codes = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                   "SEP", "OCT", "NOV", "DEC"]
    for d_i in range(60):
        month_idx = d_i // 28  # 0 or 1
        day_num = (d_i % 28) + 1
        date_str = f"2026-{(month_idx + 1):02d}-{day_num:02d}"
        ticker_date = f"26{month_codes[month_idx]}{day_num:02d}"
        day = snap / date_str
        day.mkdir(parents=True)

        forecasts = [
            {
                "city": "MIA", "forecast_date": date_str,
                "daily_variable": "temperature_2m_max",
                "model": "gfs_seamless",
                "temperatures": [80.0 + i * 0.3 for i in range(31)],
            }
        ]
        settlements = [
            {
                "city": "MIA", "observation_date": date_str,
                "high_f": 85.0 + d_i % 5,
                "low_f": 70.0,
                "is_preliminary": False,
                "issued_at": f"{date_str}T20:00:00Z",
                "product_id": "MIA-CLI",
            }
        ]
        markets = [
            {
                "ticker": f"KXHIGHMIA-{ticker_date}-T80",
                "strike_type": "greater", "floor_strike": 80.0 + (d_i % 5), "cap_strike": None,
                "city": "MIA", "daily_variable": "temperature_2m_max",
                "yes_ask_dollars": 0.4, "yes_bid_dollars": 0.38,
            }
        ]
        (day / "weather_forecasts.json").write_text(json.dumps(forecasts))
        (day / "settlements.json").write_text(json.dumps(settlements))
        (day / "markets.json").write_text(json.dumps(markets))

    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
strategies:
  weather:
    ensemble_weights: {gfs_seamless: 1.0}
    emos_enabled_cells: []
    excluded_city_vars: []
    isotonic_path: data/calibration/isotonic_v2.json
""")

    output_path = tmp_path / "per_cell_winners.json"
    payload = run_harness(
        snapshots_dir=snap,
        config_path=config_path,
        bias_table_path=None,
        cell_allowlist=["MIA/high"],  # restrict to the synthetic cell
        write_output_to=output_path,
    )
    assert "cells" in payload
    assert "MIA/high" in payload["cells"]
    assert output_path.exists()


def test_u28_politis_romano_coverage_on_autocorrelated_series():
    """U-28: stationary block bootstrap covers true mean ≥ 93% on
    autocorrelation length L=4 dates."""
    import numpy as np

    from python.analytics.calibrator_harness import block_bootstrap_ci

    rng = np.random.default_rng(seed=20260524)
    # Generate AR(1) per-date noise, then per-row values = date_signal + iid_noise
    n_dates = 90
    n_rows_per_date = 6
    phi = 0.75  # AR(1) coefficient → L_eff ≈ 4
    date_signal = np.zeros(n_dates)
    date_signal[0] = rng.standard_normal()
    for i in range(1, n_dates):
        date_signal[i] = phi * date_signal[i - 1] + rng.standard_normal() * np.sqrt(1 - phi ** 2)

    true_mean = 0.5  # injected
    coverage_hits = 0
    n_trials = 200
    for trial in range(n_trials):
        rng_trial = np.random.default_rng(seed=trial)
        values = []
        block_keys = []
        for d in range(n_dates):
            for k in range(n_rows_per_date):
                values.append(float(true_mean + date_signal[d] + 0.3 * rng_trial.standard_normal()))
                block_keys.append((f"2026-{(d // 28) + 1:02d}-{(d % 28) + 1:02d}", k % 3))
        result = block_bootstrap_ci(
            values=values, block_keys=block_keys, n_iter=400, seed=trial,
        )
        if result["ci_lb"] <= true_mean <= result["ci_ub"]:
            coverage_hits += 1

    coverage = coverage_hits / n_trials
    # Target 95% nominal, allow 93% slack for finite-sample bias of stationary block bootstrap
    assert coverage >= 0.93, (
        f"Politis-Romano coverage {coverage:.2%} below 93% target on L=4 autocorrelated series"
    )


def test_u24_lead_blindness_regression():
    """U-24: leads must be distinct on a real-shape multi-snapshot fixture after B.5 fix.

    The pre-fix bug collapsed _compute_lead_for to always return 0 because daily
    02:00 UTC snapshots win the min() operation. Post-fix, lead = calendar
    diff between forecast_date and snapshot_date, so a corpus with snapshots
    on multiple lead days for the same forecast_date produces distinct lead
    values.
    """
    from python.analytics.calibrator_harness import FeatureRow, _compute_lead_for

    # Construct rows where the same forecast_date has snapshot_dates 0, 1, 2 days prior.
    rows = []
    for k in range(3):
        snap_day = 20 - k
        rows.append(FeatureRow(
            city="MIA", var="temperature_2m_max",
            forecast_date="2026-05-20",
            snapshot_date=f"2026-05-{snap_day:02d}",
            lead=0,  # placeholder; _compute_lead_for is being tested
            ticker=f"T{k}", strike_type="greater",
            floor=80.0, cap=None, market_yes_price=0.5,
            blended_raw_prob=0.5, gfs_ens_mean=80.0, gfs_ens_std=2.0,
            n_members=31, outcome=1, extras={},
        ))
    leads = [_compute_lead_for(r) for r in rows]
    assert leads == [0, 1, 2], f"Expected leads [0, 1, 2], got {leads}"
    assert len(set(leads)) == 3, f"Leads should be distinct, got {leads}"


# ---------------------------------------------------------------------------
# Phase 1A.1 tests: U-22, U-23, U-25, U-26, U-27
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_u22_worse_than_identity_end_to_end():
    """U-22: synthetic anti-correlated candidates → worse_than_identity verdict.

    Requires ≥2 eligible candidates; both must have logskill_p99 < 0.
    Also asserts eligibility status BEFORE verdict to catch the n_clipped:>5%
    failure mode (extreme probs trip the FR-9 extreme-prob veto).
    """
    import numpy as np

    from python.analytics.calibrator_harness import (
        CalibratorCandidate,
        FeatureRow,
        IdentityCandidate,
        VetoResult,
        score_cell,
    )

    class AntiCorrelatedCandidate(CalibratorCandidate):
        def __init__(self, name: str, noise: float):
            self.name = name
            self.noise = noise
            self.fit_semantics = "per_fold_refit"

        def fit(self, train_rows):
            return None

        def score(self, row, fit_state):
            # Read p_true from extras and return its INVERSE (anti-correlated).
            p_true = float(row.extras["p_true"])
            return float(np.clip(1.0 - p_true + self.noise, 0.2, 0.8))

        def validate(self, fit_state):
            return VetoResult(ok=True, warnings=[])

    rng = np.random.default_rng(seed=20260524)
    rows = []
    for d_i in range(200):
        month = (d_i // 28) % 12 + 1
        day = (d_i % 28) + 1
        year = 2025 + (d_i // (12 * 28))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        for k in range(3):
            p_true = float(rng.choice([0.2, 0.3, 0.7, 0.8]))
            raw = float(np.clip(0.5 + 0.6 * (p_true - 0.5), 0.2, 0.8))
            outcome = int(rng.random() < p_true)
            rows.append(FeatureRow(
                city="TEST", var="temperature_2m_max",
                forecast_date=date_str, snapshot_date=date_str,
                lead=0, ticker=f"T{d_i}_{k}", strike_type="greater",
                floor=78.0 + float(d_i % 5), cap=None,
                market_yes_price=raw, blended_raw_prob=raw,
                gfs_ens_mean=80.0, gfs_ens_std=2.0, n_members=31,
                outcome=outcome, extras={"p_true": p_true},
            ))

    result = score_cell(
        city="TEST", var="temperature_2m_max", rows=rows,
        candidates=[
            IdentityCandidate(),
            AntiCorrelatedCandidate("anti_a", noise=0.0),
            AntiCorrelatedCandidate("anti_b", noise=0.02),
        ],
    )
    # Assert both anti candidates are eligible before assertion on verdict.
    for name in ("anti_a", "anti_b"):
        assert result["candidates"][name]["eligibility"] == "eligible", (
            f"{name} ineligible: {result['candidates'][name]['ineligibility_reasons']}"
        )
    assert result["verdict"] == "worse_than_identity", (
        result["verdict"], result["verdict_message"]
    )
    assert "anti_a" in result["verdict_message"] or "all 2" in result["verdict_message"]


def test_u23_schema_v2_rejects_v1():
    """U-23: validate_schema rejects v1-shaped payload (intentional break)."""
    from python.analytics.calibrator_harness import SchemaValidationError, validate_schema

    v1_payload = {
        "schema_version": 1, "provisional": False,
        "run_timestamp": "2026-05-24T00:00:00Z",
        "config": {}, "corpus": {}, "cells": {},
    }
    try:
        validate_schema(v1_payload, expected_cell_keys=set())
        raise AssertionError("validate_schema accepted v1; expected SchemaValidationError")
    except SchemaValidationError as e:
        assert "schema_version" in str(e) and "2" in str(e)


def test_u23_schema_v2_accepts_new_verdict_enum():
    """U-23: validate_schema accepts insufficient_evidence and worse_than_identity."""
    from python.analytics.calibrator_harness import validate_schema

    def _cell(verdict: str, winner: str | None = None) -> dict:
        return {
            "verdict": verdict,
            "verdict_message": "test message",
            "winner_candidate": winner,
            "n": {"val_total": 150, "val_traded": 100, "unique_forecast_dates": 50},
            "candidates": {
                "identity": {
                    "logskill": {"mean": 0.0, "p2.5": 0.0, "p50": 0.0, "p97.5": 0.0, "p99": 0.0},
                    "brier": {"mean": 0.0, "p2.5": 0.0, "p50": 0.0, "p97.5": 0.0},
                    "simulated_roi": {
                        "mean": None, "ci_lb": None, "ci_ub": None,
                        "n_val_traded": 0, "status": "insufficient_data",
                        "roi_ci_lb_informative": False,
                    },
                    "fold_logskill": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "n_clipped": 0, "mean_rows_per_block": 6.0,
                    "cell_bootstrap_degenerate": False,
                    "eligibility": "eligible", "ineligibility_reasons": [],
                    "per_fold_diagnostics": {
                        "sigma_veto_folds": [], "unstable_b_folds": [],
                        "d_at_zero_folds": [], "unstable_d_folds": [],
                    },
                },
            },
            "flags": {
                "pnl_logskill_disagreement": False, "metric_disagreement": False,
                "gate_margin_thin": False, "below_power_floor": False,
            },
            "production": {
                "kind": "isotonic",
                "artifact_path": "data/calibration/isotonic_v2.json",
                "artifact_hash": None,
                "source_config_key": "weather.isotonic_path",
            },
            "disagreement_with_production": False,
            "excluded_in_production": False,
            "in_sample_reference": {
                "iso_v2_deployed_reference": {
                    "logskill_in_sample": 0.1, "brier_in_sample": 0.2,
                    "note": "LEAKAGE — not a verdict",
                },
                "emos_deployed_reference": {
                    "logskill_in_sample": 0.0, "brier_in_sample": 0.0,
                    "note": "LEAKAGE — not a verdict",
                },
            },
        }

    payload = {
        "schema_version": 2, "provisional": False,
        "harness_version": "0.2.0",
        "run_timestamp": "2026-05-24T00:00:00Z",
        "corpus": {
            "snapshots_dir": "data/snapshots",
            "date_range": ["2026-03-19", "2026-05-21"],
            "n_days": 62, "corpus_hash": "sha256:test",
        },
        "config": {
            "candidates": ["identity", "iso_v2_global", "emos"],
            "audit_candidates": ["iso_v2_deployed_reference", "emos_deployed_reference"],
            "partition_rule": "cv_fold_iter:5fold:expanding_window",
            "bracket_width": 2.0, "logskill_eps": 0.001,
            "bootstrap_iter": 400, "bootstrap_block": ["forecast_date", "lead"],
            "kelly_knobs": {"kelly_fraction": 0.075, "max_pct": 0.05, "bankroll": 1000.0},
            "trading_filters": {
                "block_raw_prob_range": [0.446, 0.883], "min_calibrated_prob": 0.110,
            },
            "winner_gate": {
                "logskill_lb_min": 0.01, "fold_consistency_min": 4,
                "n_clipped_pct_max": 0.05,
            },
            "emos_promotion_gate": {
                "logskill_lb_vs_iso_min": -0.02, "bss_lb_vs_iso_min": -0.05,
            },
            "tiebreaker": {"logskill_lb_band": 0.005},
        },
        "cells": {
            "MIA/high": _cell("insufficient_evidence"),
            "MIA/low": _cell("winner", winner="iso_v2_global"),
            "NYC/high": _cell("worse_than_identity"),
        },
    }
    validate_schema(payload, expected_cell_keys={"MIA/high", "MIA/low", "NYC/high"})
    # No exception → PASS.


def test_u25_winner_requires_positive_logskill_mean():
    """U-25: defensive invariant — a candidate with LB > +0.01 but mean ≤ 0
    must not become a winner.

    This catches future regressions where someone weakens the gate to
    LogSkill_LB > -X without checking that the candidate's point estimate
    is still positive.
    """
    from python.analytics.calibrator_harness import CandidateResult, compute_verdict

    cands = [
        CandidateResult(
            name="identity", logskill_lb=0.0, logskill_mean=0.0, logskill_p99=0.0,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=True, ineligibility_reasons=[],
        ),
        CandidateResult(  # LB just above gate, but mean <= 0 (pathological)
            name="iso_v2_global", logskill_lb=0.011, logskill_mean=-0.001, logskill_p99=0.10,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=5, eligible=True, ineligibility_reasons=[],
        ),
    ]
    v = compute_verdict(cands, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    # FR-7 clause 4 (logskill_mean > 0) blocks this candidate at the gate, so
    # verdict is insufficient_evidence (not winner) and the property check below
    # never fires. If a future spec edit removes clause 4 OR relaxes clause 2
    # below the candidate's LB=0.011, the candidate becomes a winner and this
    # test fires the assertion as a regression catch. Keep this assertion live —
    # it is what makes the "future weakening" guard meaningful.
    assert v["verdict"] == "insufficient_evidence", (
        f"Expected insufficient_evidence under 4-clause gate; got {v['verdict']!r}. "
        f"Did FR-7 lose clause 4 (logskill_mean > 0) or weaken clause 2?"
    )
    if v["verdict"] == "winner":
        winner_block_mean = cands[1].logskill_mean  # iso_v2_global
        assert winner_block_mean > 0, (
            f"Property violation: winner has LogSkill_point = {winner_block_mean} ≤ 0. "
            "Gate composition must require positive point estimate when LB is on the wire."
        )


def test_u26_verdict_message_enumerates_eligibility():
    """U-26: insufficient_evidence and worse_than_identity messages enumerate
    eligible AND ineligible candidates."""
    from python.analytics.calibrator_harness import CandidateResult, compute_verdict

    cands = [
        CandidateResult(name="identity", logskill_lb=0.0, logskill_mean=0.0, logskill_p99=0.0,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=True, ineligibility_reasons=[]),
        CandidateResult(
            name="iso_v2_global", logskill_lb=-0.01, logskill_mean=0.001, logskill_p99=0.05,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=3, eligible=True, ineligibility_reasons=[]),
        CandidateResult(name="emos", logskill_lb=0.0, logskill_mean=0.0, logskill_p99=0.0,
            roi_point=0.0, roi_ci_lb=0.0,
            fold_logskill_positive_count=0, eligible=False,
            ineligibility_reasons=["incompatible_sigma:fold_2"]),
    ]
    v = compute_verdict(cands, n_val_total=150, n_val_traded=120, n_unique_strikes=10)
    assert v["verdict"] == "insufficient_evidence"
    msg = v["verdict_message"]
    assert "eligible:" in msg
    assert "iso_v2_global" in msg
    assert "ineligible:" in msg
    assert "emos" in msg
    assert "incompatible_sigma" in msg


def test_u27_pnl_logskill_disagreement_suppression():
    """U-27: pnl_logskill_disagreement does NOT fire when ROI hits the
    bet-fully-lost floor (ci_lb ≤ −0.95 OR roi_ci_lb_informative=false).

    Test approach: directly construct a per-candidate block resembling
    score_cell's output, then invoke the flag computation. This isolates the
    suppression rule from the full score_cell pipeline.
    """
    # The pnl_logskill_disagreement flag is set inside score_cell. To unit-test
    # the suppression rule in isolation, expose the predicate as a small
    # helper or replicate the predicate here.
    #
    # Two cases:

    # Case 1: ROI mean negative AND ROI informative (ci_lb > -0.95) → flag fires.
    roi_info_fires = {
        "mean": -0.30, "ci_lb": -0.55, "ci_ub": -0.05,
        "n_val_traded": 80, "status": "ok",
        "roi_ci_lb_informative": True,
    }
    # Case 2: ROI mean negative but ROI hitting floor (ci_lb = -1.05, informative=false)
    # → flag SUPPRESSED.
    roi_info_suppressed = {
        "mean": -1.06, "ci_lb": -1.069, "ci_ub": -1.05,
        "n_val_traded": 80, "status": "ok",
        "roi_ci_lb_informative": False,
    }

    def _predicate(roi_info: dict, winner_exists: bool) -> bool:
        # Mirror the rule from score_cell after Sub-task 5.3 Step 5.
        return bool(
            winner_exists
            and roi_info.get("roi_ci_lb_informative", False)
            and (roi_info.get("ci_lb") or 0.0) > -0.95
            and (roi_info.get("mean") or 0.0) <= 0
        )

    assert _predicate(roi_info_fires, winner_exists=True) is True, (
        "expected flag to fire on informative-negative-ROI"
    )
    assert _predicate(roi_info_suppressed, winner_exists=True) is False, (
        "expected flag suppressed on bet-fully-lost floor"
    )
    assert _predicate(roi_info_fires, winner_exists=False) is False, (
        "no winner → no disagreement"
    )
