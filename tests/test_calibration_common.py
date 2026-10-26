"""Tests for shared calibration helpers."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def test_build_production_mirrored_pairs_no_bias_table():
    """bias_table=None → no bias correction. Bracket AND outcome semantics
    are CLOSED [floor, cap] (Kalshi-truth, post 2026-05-19 fix).
    Members at 70.0 in [69, 70] → raw_prob=1.0; settlement 65 is outside
    [69, 70] → outcome 0.
    """
    from scripts._calibration_common import build_production_mirrored_pairs

    forecasts = {
        ("NYC", "2026-04-01", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [70.0] * 31,
        },
    }
    settlements = {("NYC", "2026-04-01"): {"high_f": 65.0, "low_f": 55.0}}
    markets = [
        {
            "ticker": "KXHIGHNY-26APR01-B69.5",
            "city": "NYC",
            "daily_variable": "temperature_2m_max",
            "strike_type": "between",
            "floor_strike": 69.0,
            "cap_strike": 70.0,
        },
    ]

    pairs = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 1.0, "ecmwf_ifs025": 1.0},
            spread_inflation_overrides=None,
        )
    )

    assert len(pairs) == 1
    raw_prob, outcome, city, var, strike_type, fc_date = pairs[0]
    assert city == "NYC"
    assert var == "temperature_2m_max"
    assert strike_type == "between"
    assert fc_date == date(2026, 4, 1)
    assert raw_prob == 1.0  # all 31 members at 70 are in CLOSED [69, 70]
    assert outcome == 0  # 65 is outside CLOSED [69, 70] (Kalshi rule)


def test_build_production_mirrored_pairs_blends_gfs_and_ecmwf():
    """Two-model blend uses ensemble_weights.

    GFS at 80 (zero in [83, 84]), ECMWF at 85 (zero in [83, 84]; closed bracket
    means 85 > 84 cap, so still 0). Blended = 0.
    """
    from scripts._calibration_common import build_production_mirrored_pairs

    forecasts = {
        ("AUS", "2026-04-02", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [80.0] * 31,
        },
        ("AUS", "2026-04-02", "temperature_2m_max", "ecmwf_ifs025"): {
            "temperatures": [85.0] * 51,
        },
    }
    settlements = {("AUS", "2026-04-02"): {"high_f": 84.0, "low_f": 70.0}}
    markets = [
        {
            "ticker": "KXHIGHAUS-26APR02-B83.5",
            "city": "AUS",
            "daily_variable": "temperature_2m_max",
            "strike_type": "between",
            "floor_strike": 83.0,
            "cap_strike": 84.0,
        },
    ]

    pairs = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            spread_inflation_overrides=None,
        )
    )

    assert len(pairs) == 1
    raw_prob, outcome, *_ = pairs[0]
    assert len(pairs[0]) == 6  # (raw_prob, outcome, city, var, strike_type, fc_date)
    assert raw_prob == 0.0  # neither model has members in [83, 84]
    # Kalshi rule: between is CLOSED, right-inclusive (floor <= t <= cap).
    # actual=84 IS in [83, 84] → outcome 1. Prior assertion (outcome == 0)
    # was guarding the right-exclusive bug; updated 2026-05-19. See
    # tests/test_outcomes_kalshi_semantics.py and the truth-audit citation
    # in python/analytics/outcomes.py docstring.
    assert outcome == 1


def test_build_production_mirrored_pairs_blend_partial_overlap():
    """Verify weighted blending when models disagree.

    Bracket [83, 84] settles YES iff NWS_int ∈ {83, 84} → true-temp window [82.5, 84.5].
    GFS members 16@83.0, 15@84.5 → all 31 in [82.5, 84.5] → fraction = 1.0
    ECMWF members all at 85.0 → fraction = 0
    Blend with weights 0.55/0.45 → 0.55 * 1.0 + 0.45 * 0 = 0.55
    """
    from scripts._calibration_common import build_production_mirrored_pairs

    forecasts = {
        ("AUS", "2026-04-03", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [83.0] * 16 + [84.5] * 15,  # 31 members, ~50/50
        },
        ("AUS", "2026-04-03", "temperature_2m_max", "ecmwf_ifs025"): {
            "temperatures": [85.0] * 51,
        },
    }
    settlements = {("AUS", "2026-04-03"): {"high_f": 84.0, "low_f": 70.0}}
    markets = [
        {
            "ticker": "KXHIGHAUS-26APR03-B83.5",
            "city": "AUS",
            "daily_variable": "temperature_2m_max",
            "strike_type": "between",
            "floor_strike": 83.0,
            "cap_strike": 84.0,
        },
    ]
    pairs = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45},
            spread_inflation_overrides=None,
        )
    )
    assert len(pairs) == 1
    assert len(pairs[0]) == 6  # (raw_prob, outcome, city, var, strike_type, fc_date)
    raw_prob = pairs[0][0]
    # AUS temperature_2m_max default spread inflation factor is 1.0
    # (per GFS_DERIVED_SPREAD_INFLATION in python/data/weather.py:207),
    # so members are NOT shifted: 16 stay at 83.0, 15 stay at 84.5.
    # NWS-rounded settle window for [83, 84] is [82.5, 84.5]: all 31 GFS
    # members count → GFS frac = 31/31 = 1.0. ECMWF at 85.0 outside → 0.
    # Blend = 0.55 * 1.0 + 0.45 * 0 = 0.55.
    expected = 0.55
    assert abs(raw_prob - expected) < 1e-6


def test_build_production_mirrored_pairs_skips_pre_chi_fix_date():
    """Chicago snapshots before 2026-03-20 are excluded (existing fitter behavior)."""
    from scripts._calibration_common import build_production_mirrored_pairs

    forecasts = {
        ("CHI", "2026-03-15", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [50.0] * 31,
        },
    }
    settlements = {("CHI", "2026-03-15"): {"high_f": 50.0, "low_f": 40.0}}
    markets = [
        {
            "ticker": "KXHIGHCHI-26MAR15-B49.5",
            "city": "CHI",
            "daily_variable": "temperature_2m_max",
            "strike_type": "between",
            "floor_strike": 49.0,
            "cap_strike": 50.0,
        },
    ]

    pairs = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides=None,
        )
    )
    assert pairs == []


def test_build_production_mirrored_pairs_uses_bias_table(tmp_path):
    """Helper invokes apply_bias_correction when bias_table is provided."""
    import json

    from python.data.bias_correction import BiasTable
    from scripts._calibration_common import build_production_mirrored_pairs

    # SNR = |mean_bias_f| / (std_bias_f / sqrt(count))
    #     = 5.0 / (1.0 / sqrt(50)) ≈ 35.4 — passes _MIN_SNR=2.0 gate.
    bias_path = tmp_path / "bias.json"
    bias_path.write_text(
        json.dumps(
            {
                "NYC": {
                    "high": {
                        "overall": {
                            "count": 100,
                            "mean_bias_f": 5.0,
                            "mae_f": 5.0,
                            "rmse_f": 5.1,
                            "std_bias_f": 1.0,
                        },
                        "by_month": {
                            "4": {
                                "count": 50,
                                "mean_bias_f": 5.0,
                                "mae_f": 5.0,
                                "rmse_f": 5.1,
                                "std_bias_f": 1.0,
                            }
                        },
                    }
                }
            }
        )
    )
    bias_table = BiasTable(bias_path)
    assert bias_table.loaded

    # All members at 75. Without correction → fraction_above(74) = 1.0.
    # With +5 mean_bias_f and 0.3 shrinkage → correction is -1.5°F → members at 73.5.
    # 73.5 < 74 (strict above) → fraction_above = 0.0.
    forecasts = {
        ("NYC", "2026-04-01", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [75.0] * 31,
        },
    }
    settlements = {("NYC", "2026-04-01"): {"high_f": 70.0, "low_f": 60.0}}
    markets = [
        {
            "ticker": "KXHIGHNY-26APR01-T74",
            "city": "NYC",
            "daily_variable": "temperature_2m_max",
            "strike_type": "greater",
            "floor_strike": 74.0,
            "cap_strike": None,
        },
    ]

    pairs_no_corr = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides=None,
        )
    )
    assert pairs_no_corr[0][0] == 1.0  # raw_prob, no correction

    pairs_with_corr = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=bias_table,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides=None,
        )
    )
    assert pairs_with_corr[0][0] == 0.0  # bias correction shifted members below strike


def test_build_production_mirrored_pairs_applies_spread_inflation():
    """Helper must invoke EnsembleForecast.inflated() when override factor != 1.0.

    AUS temperature_2m_max default is 1.0 (per GFS_DERIVED_SPREAD_INFLATION),
    so we use the override path explicitly to force a 2.0 factor.

    Members: 16 at 80.0, 15 at 84.0 (n=31).
    Ensemble mean = (16*80 + 15*84) / 31 = 2540/31 ≈ 81.9355.

    Bracket [80, 84] (production CLOSED):
      - Without inflation (factor=1.0): all 31 members are in [80, 84]
        → raw_prob = 1.0.
      - With inflation factor=2.0:
          inflated_t = mean + 2 * (t - mean)
          80.0 → 81.9355 + 2 * (80.0 - 81.9355) = 78.0645
          84.0 → 81.9355 + 2 * (84.0 - 81.9355) = 86.0645
        Neither 78.0645 nor 86.0645 is in [80, 84]
        → raw_prob = 0.0.

    The two raw_probs MUST differ — that's the proof the inflation step
    is actually executing in the helper's pipeline.
    """
    from scripts._calibration_common import build_production_mirrored_pairs

    forecasts = {
        ("AUS", "2026-04-04", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [80.0] * 16 + [84.0] * 15,
        },
    }
    settlements = {("AUS", "2026-04-04"): {"high_f": 82.0, "low_f": 70.0}}
    markets = [
        {
            "ticker": "KXHIGHAUS-26APR04-B82",
            "city": "AUS",
            "daily_variable": "temperature_2m_max",
            "strike_type": "between",
            "floor_strike": 80.0,
            "cap_strike": 84.0,
        },
    ]

    pairs_no_inflation = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides=None,  # AUS/temp_2m_max default = 1.0 (no-op)
        )
    )
    assert len(pairs_no_inflation) == 1
    assert pairs_no_inflation[0][0] == 1.0  # all 31 members in CLOSED [80, 84]

    pairs_inflated = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides={"AUS": {"temperature_2m_max": 2.0}},
        )
    )
    assert len(pairs_inflated) == 1
    # Inflated members at ~78.06 and ~86.06; neither in [80, 84] → 0.0.
    assert pairs_inflated[0][0] == 0.0
    # The two probabilities MUST differ — proves inflation step ran.
    assert pairs_no_inflation[0][0] != pairs_inflated[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Sibling generator: build_production_mirrored_pairs_with_provenance (Task 5)
# ─────────────────────────────────────────────────────────────────────────────


def test_t_cc_1_sibling_yields_9_tuple_with_matching_first_six_fields():
    """T-cc-1: sibling generator yields 9-tuple; first 6 fields bit-identical
    to the existing 6-tuple generator on the same inputs."""
    from scripts._calibration_common import (
        build_production_mirrored_pairs,
        build_production_mirrored_pairs_with_provenance,
    )

    forecasts = {
        ("NYC", "2026-04-25", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [70.0] * 31,
        },
        ("NYC", "2026-04-25", "temperature_2m_max", "ecmwf_ifs025"): {
            "temperatures": [72.0] * 51,
        },
    }
    settlements = {("NYC", "2026-04-25"): {"high_f": 71.0, "low_f": 60.0}}
    markets = [
        {
            "ticker": "KXHIGHNY-26APR25-T70",
            "city": "NYC",
            "daily_variable": "temperature_2m_max",
            "strike_type": "greater",
            "floor_strike": 70.0,
            "cap_strike": None,
        },
    ]

    weights = {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45}
    legacy_pairs = list(
        build_production_mirrored_pairs(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights=weights,
            spread_inflation_overrides=None,
        )
    )
    sibling_pairs = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights=weights,
            spread_inflation_overrides=None,
        )
    )

    assert len(legacy_pairs) == 1
    assert len(sibling_pairs) == 1
    assert len(sibling_pairs[0]) == 9
    # First 6 fields bit-identical.
    assert sibling_pairs[0][:6] == legacy_pairs[0]
    # Last three fields are the provenance fields.
    models_used = sibling_pairs[0][6]
    bias_applied = sibling_pairs[0][7]
    ticker = sibling_pairs[0][8]
    assert isinstance(models_used, tuple)
    assert isinstance(bias_applied, bool)
    assert ticker == "KXHIGHNY-26APR25-T70"


def test_t_cc_2_bias_correction_applied_truth_table(tmp_path):
    """T-cc-2: bias_correction_applied is True iff bias_table is loaded AND
    'gfs_seamless' is in per_model. Cover all four (loaded × gfs-present)
    combinations."""
    from python.data.bias_correction import BiasTable
    from scripts._calibration_common import build_production_mirrored_pairs_with_provenance

    # Build a minimal loaded BiasTable. SNR gate uses count/std_bias_f.
    bias_path = tmp_path / "bias.json"
    bias_path.write_text(
        json.dumps(
            {
                "NYC": {
                    "high": {
                        "overall": {
                            "count": 100,
                            "mean_bias_f": 1.0,
                            "mae_f": 1.0,
                            "rmse_f": 1.0,
                            "std_bias_f": 1.0,
                        },
                        "by_month": {},
                    }
                }
            }
        )
    )
    loaded_bias = BiasTable(bias_path)
    assert loaded_bias.loaded

    settlements = {("NYC", "2026-04-25"): {"high_f": 71.0, "low_f": 60.0}}
    markets = [
        {
            "ticker": "KXHIGHNY-26APR25-T70",
            "city": "NYC",
            "daily_variable": "temperature_2m_max",
            "strike_type": "greater",
            "floor_strike": 70.0,
            "cap_strike": None,
        },
    ]

    # Case A: bias_table loaded + gfs_seamless present → True.
    forecasts_with_gfs = {
        ("NYC", "2026-04-25", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [70.0] * 31,
        },
    }
    pair = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts_with_gfs,
            settlements=settlements,
            markets=markets,
            bias_table=loaded_bias,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides=None,
        )
    )[0]
    assert pair[7] is True, "loaded bias_table + GFS present → True"

    # Case B: bias_table=None + gfs_seamless present → False.
    pair = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts_with_gfs,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"gfs_seamless": 1.0},
            spread_inflation_overrides=None,
        )
    )[0]
    assert pair[7] is False, "bias_table=None → False even with GFS present"

    # Case C: bias_table loaded + only ECMWF present (no GFS) → False.
    forecasts_ecmwf_only = {
        ("NYC", "2026-04-25", "temperature_2m_max", "ecmwf_ifs025"): {
            "temperatures": [72.0] * 51,
        },
    }
    pair = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts_ecmwf_only,
            settlements=settlements,
            markets=markets,
            bias_table=loaded_bias,
            ensemble_weights={"ecmwf_ifs025": 1.0},
            spread_inflation_overrides=None,
        )
    )[0]
    assert pair[7] is False, "no GFS in per_model → False even with loaded bias_table"

    # Case D: bias_table=None + only ECMWF present → False.
    pair = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts_ecmwf_only,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"ecmwf_ifs025": 1.0},
            spread_inflation_overrides=None,
        )
    )[0]
    assert pair[7] is False, "neither bias_table nor GFS → False"


def test_t_cc_3_models_used_is_sorted_tuple():
    """T-cc-3: models_used must be a sorted tuple, not a set or list."""
    from scripts._calibration_common import build_production_mirrored_pairs_with_provenance

    # Provide both models. ensemble_weights iteration order would be
    # ECMWF-first; sorted output must place GFS first alphabetically.
    forecasts = {
        ("NYC", "2026-04-25", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [70.0] * 31,
        },
        ("NYC", "2026-04-25", "temperature_2m_max", "ecmwf_ifs025"): {
            "temperatures": [72.0] * 51,
        },
    }
    settlements = {("NYC", "2026-04-25"): {"high_f": 71.0, "low_f": 60.0}}
    markets = [
        {
            "ticker": "KXHIGHNY-26APR25-T70",
            "city": "NYC",
            "daily_variable": "temperature_2m_max",
            "strike_type": "greater",
            "floor_strike": 70.0,
            "cap_strike": None,
        },
    ]

    pair = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"ecmwf_ifs025": 0.45, "gfs_seamless": 0.55},
            spread_inflation_overrides=None,
        )
    )[0]
    models_used = pair[6]
    # Must be a tuple (NOT a set, NOT a list).
    assert isinstance(models_used, tuple)
    assert not isinstance(models_used, list)
    # Must be sorted.
    assert models_used == ("ecmwf_ifs025", "gfs_seamless")
    assert list(models_used) == sorted(models_used)

    # Single-model fallback: if only GFS is published, models_used reflects that.
    forecasts_gfs_only = {
        ("NYC", "2026-04-25", "temperature_2m_max", "gfs_seamless"): {
            "temperatures": [70.0] * 31,
        },
    }
    pair = list(
        build_production_mirrored_pairs_with_provenance(
            forecasts=forecasts_gfs_only,
            settlements=settlements,
            markets=markets,
            bias_table=None,
            ensemble_weights={"ecmwf_ifs025": 0.45, "gfs_seamless": 0.55},
            spread_inflation_overrides=None,
        )
    )[0]
    assert pair[6] == ("gfs_seamless",)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: prefer_latest_settlement — pure lex compare, returns bool
# ─────────────────────────────────────────────────────────────────────────────


def _record(*, issued_at: str, product_id: str, is_preliminary: bool = False) -> dict:
    """Minimal settlement record for layer-1 comparison tests."""
    return {
        "issued_at": issued_at,
        "product_id": product_id,
        "is_preliminary": is_preliminary,
    }


class TestPreferLatestSettlement:
    def test_later_issued_at_wins(self) -> None:
        existing = _record(issued_at="2026-05-04T18:00:00Z", product_id="CLI_NYC_001")
        new = _record(issued_at="2026-05-04T22:00:00Z", product_id="CLI_NYC_001")
        from scripts._calibration_common import prefer_latest_settlement

        assert prefer_latest_settlement(existing, new) is True
        assert prefer_latest_settlement(new, existing) is False

    def test_same_issued_at_later_product_id_wins(self) -> None:
        existing = _record(issued_at="2026-05-04T18:00:00Z", product_id="CLI_NYC_001")
        new = _record(issued_at="2026-05-04T18:00:00Z", product_id="CLI_NYC_002")
        from scripts._calibration_common import prefer_latest_settlement

        assert prefer_latest_settlement(existing, new) is True
        assert prefer_latest_settlement(new, existing) is False

    def test_final_beats_preliminary_regardless_of_issued_at(self) -> None:
        """A final settlement ALWAYS beats a preliminary one, even if the
        preliminary was issued later. Prior to 2026-05-11 the comparison
        ignored ``is_preliminary`` entirely; that allowed a retroactive
        preliminary at 17:45 to overwrite a final issued at 06:30 the same
        day. Downstream ``final_only`` filters then dropped the merged row,
        losing the final we already had. See
        ``memory/emos_corpus_pipeline_followups.md`` follow-up #4 for the
        forensic trace.
        """
        earlier_final = _record(
            issued_at="2026-05-04T06:00:00Z",
            product_id="CLI_NYC_001",
            is_preliminary=False,
        )
        later_preliminary = _record(
            issued_at="2026-05-04T18:00:00Z",
            product_id="CLI_NYC_002",
            is_preliminary=True,
        )
        from scripts._calibration_common import prefer_latest_settlement

        # Final stays even when paired against a later preliminary.
        assert prefer_latest_settlement(earlier_final, later_preliminary) is False
        # Existing preliminary is replaced by an earlier final.
        assert prefer_latest_settlement(later_preliminary, earlier_final) is True

    def test_two_finals_compare_by_issued_at(self) -> None:
        """Among two finals, the later ``issued_at`` wins (corrections after
        the initial post)."""
        earlier_final = _record(
            issued_at="2026-05-04T06:00:00Z",
            product_id="CLI_NYC_001",
            is_preliminary=False,
        )
        later_final = _record(
            issued_at="2026-05-04T22:00:00Z",
            product_id="CLI_NYC_001",
            is_preliminary=False,
        )
        from scripts._calibration_common import prefer_latest_settlement

        assert prefer_latest_settlement(earlier_final, later_final) is True
        assert prefer_latest_settlement(later_final, earlier_final) is False

    def test_two_preliminaries_compare_by_issued_at(self) -> None:
        """Among two preliminaries, the later one wins (typically the
        afternoon reissue with cleaner sensor data)."""
        earlier_prelim = _record(
            issued_at="2026-05-04T15:00:00Z",
            product_id="CLI_NYC_001",
            is_preliminary=True,
        )
        later_prelim = _record(
            issued_at="2026-05-04T18:00:00Z",
            product_id="CLI_NYC_001",
            is_preliminary=True,
        )
        from scripts._calibration_common import prefer_latest_settlement

        assert prefer_latest_settlement(earlier_prelim, later_prelim) is True
        assert prefer_latest_settlement(later_prelim, earlier_prelim) is False

    def test_identical_keys_returns_false(self) -> None:
        """``new > existing`` is strict; equal keys do NOT trigger replacement."""
        existing = _record(issued_at="2026-05-04T18:00:00Z", product_id="CLI_NYC_001")
        new = _record(issued_at="2026-05-04T18:00:00Z", product_id="CLI_NYC_001")
        from scripts._calibration_common import prefer_latest_settlement

        assert prefer_latest_settlement(existing, new) is False

    def test_missing_fields_default_to_empty_string(self) -> None:
        """Records missing ``issued_at`` or ``product_id`` compare against
        ``""``; this protects against silent KeyError regressions if the
        production schema ever shifts."""
        existing: dict = {}
        new = _record(issued_at="2026-05-04T18:00:00Z", product_id="CLI_NYC_001")
        from scripts._calibration_common import prefer_latest_settlement

        assert prefer_latest_settlement(existing, new) is True
        assert prefer_latest_settlement(new, existing) is False


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: load_snapshot_data — post-merge final-only filter
# ─────────────────────────────────────────────────────────────────────────────


def _write_snapshot(
    snap_dir: Path,
    *,
    forecasts: list[dict] | None = None,
    settlements: list[dict] | None = None,
    markets: list[dict] | None = None,
) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    if markets is not None:
        (snap_dir / "markets.json").write_text(json.dumps(markets))
    if forecasts is not None:
        (snap_dir / "weather_forecasts.json").write_text(json.dumps(forecasts))
    if settlements is not None:
        (snap_dir / "settlements.json").write_text(json.dumps(settlements))


class TestLoadSnapshotDataFinalOnlyFilter:
    """After the merge, records with ``is_preliminary=True`` are filtered out —
    the returned ``settlements`` dict contains only finalized records.
    """

    def test_final_survives_a_later_preliminary(self, tmp_path: Path) -> None:
        """Under the post-2026-05-11 contract, a later preliminary cannot
        overwrite an earlier final at layer 1, so layer 2's final-only
        filter has nothing to drop and the final survives the merge.

        Pre-2026-05-11 the late-preliminary won the issued_at race at layer 1
        and the final-only filter at layer 2 then dropped the entire row,
        losing the final we already had. See
        ``memory/emos_corpus_pipeline_followups.md`` follow-up #4."""
        from scripts._calibration_common import load_snapshot_data

        _write_snapshot(
            tmp_path / "2026-05-04",
            settlements=[
                {
                    "city": "NYC",
                    "observation_date": "2026-05-04",
                    "high_f": 70.0,
                    "low_f": 55.0,
                    "issued_at": "2026-05-04T06:00:00Z",
                    "product_id": "CLI_NYC_001",
                    "is_preliminary": False,
                }
            ],
            markets=[],
        )
        _write_snapshot(
            tmp_path / "2026-05-05",
            settlements=[
                {
                    "city": "NYC",
                    "observation_date": "2026-05-04",
                    "high_f": 71.0,
                    "low_f": 55.0,
                    "issued_at": "2026-05-05T18:00:00Z",  # later, but preliminary
                    "product_id": "CLI_NYC_002",
                    "is_preliminary": True,
                }
            ],
            markets=[],
        )

        _, settlements, _ = load_snapshot_data(tmp_path)

        assert ("NYC", "2026-05-04") in settlements
        kept = settlements[("NYC", "2026-05-04")]
        assert kept["is_preliminary"] is False
        assert kept["high_f"] == 70.0
        assert kept["product_id"] == "CLI_NYC_001"

    def test_final_winner_survives(self, tmp_path: Path) -> None:
        from scripts._calibration_common import load_snapshot_data

        _write_snapshot(
            tmp_path / "2026-05-04",
            settlements=[
                {
                    "city": "NYC",
                    "observation_date": "2026-05-04",
                    "high_f": 70.0,
                    "low_f": 55.0,
                    "issued_at": "2026-05-04T06:00:00Z",
                    "product_id": "CLI_NYC_001",
                    "is_preliminary": True,
                }
            ],
            markets=[],
        )
        _write_snapshot(
            tmp_path / "2026-05-05",
            settlements=[
                {
                    "city": "NYC",
                    "observation_date": "2026-05-04",
                    "high_f": 71.0,
                    "low_f": 55.0,
                    "issued_at": "2026-05-05T18:00:00Z",
                    "product_id": "CLI_NYC_002",
                    "is_preliminary": False,  # the final supersedes the prelim
                }
            ],
            markets=[],
        )

        _, settlements, _ = load_snapshot_data(tmp_path)

        # Final-version record kept under (city, observation_date).
        assert settlements[("NYC", "2026-05-04")]["high_f"] == 71.0
        assert settlements[("NYC", "2026-05-04")]["is_preliminary"] is False

    def test_finals_only_kept_when_no_preliminary_present(self, tmp_path: Path) -> None:
        """Sanity: the filter doesn't drop legitimate finals when nothing
        else is on disk."""
        from scripts._calibration_common import load_snapshot_data

        _write_snapshot(
            tmp_path / "2026-05-04",
            settlements=[
                {
                    "city": "NYC",
                    "observation_date": "2026-05-04",
                    "high_f": 70.0,
                    "low_f": 55.0,
                    "issued_at": "2026-05-04T06:00:00Z",
                    "product_id": "CLI_NYC_001",
                    "is_preliminary": False,
                }
            ],
            markets=[],
        )

        _, settlements, _ = load_snapshot_data(tmp_path)
        assert settlements[("NYC", "2026-05-04")]["high_f"] == 70.0
