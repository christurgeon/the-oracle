"""M8 — §8.2 integration test on the slim MIA fixture.

Shape-invariant: the scaffold's load → join → EMOS-recompute → outcome-derive
→ gate-stack pipeline produces the 6-bracket MIA/high evidence row set the
spike (`scripts/_step0_pattern1_spike_2026-05-13.py`) reported, with the
spike's EMOS posterior moments, outcomes against settled NWS=93 °F, gate
verdicts under `{cal_floor, plateau_filter}`, and a reproducible envelope SHA.

Spec §8.2; plan §D M8 + §F.1.

Fixture provenance: the slim trio under
`tests/research/fixtures/snapshots_step0_slim/2026-05-08/1500/` is built
deterministically by `scripts/_make_step0_slim_fixture.py` from the
2026-05-08/1500 snapshot + the 2026-05-09/2000 MIA settlement. Forecast
date == snapshot day (2026-05-08) so `allow_same_day_settlement=True` is
required to surface the rows the spike evaluated; this matches the spec
§8.2 narrative.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

from python.data.emos import EmosTable
from python.research.scaffold import (
    EvaluationRow,
    GateConfig,
    build_envelope,
    iter_evaluation_rows,
)

_PROD_COEFFS = Path("data/emos/coefficients.json")
_SLIM_FIXTURE = Path(__file__).parent / "fixtures" / "snapshots_step0_slim"

# Spike values (`_step0_pattern1_spike_2026-05-13.py`): MIA/high, 2026-05-08,
# lead=0 against current `data/emos/coefficients.json` (sha `3ca7b94d…`,
# pooled across leads).
_EXPECTED_MU = 92.736
_EXPECTED_SIGMA = 0.993
_POSTERIOR_TOL = 1e-3

# NWS final high for MIA on 2026-05-08 (from data/snapshots/2026-05-09/2000/
# settlements.json, `is_preliminary=false`).
_SETTLED_HIGH_F = 93.0

# Per the spike's settled-NWS=93 °F outcome map.
# Updated 2026-05-19 for Kalshi-truth semantics:
#  - ``between`` is CLOSED right-inclusive: B92.5 (floor=92, cap=93) with
#    actual=93 settles YES (42/42 audit; Kalshi rules + empirical).
#  - ``greater`` is STRICT: T93 (floor=93) with actual=93 settles NO.
#    Verified empirically — KXHIGHMIA-26MAY08-T93 settled NO on Kalshi
#    (rules_primary: "greater than 93°"). The prior True expectation
#    encoded a transient ``>=`` over-correction; reverted to strict here.
# See tests/test_outcomes_kalshi_semantics.py.
_EXPECTED_OUTCOMES = {
    "KXHIGHMIA-26MAY08-T93": False,
    "KXHIGHMIA-26MAY08-T86": False,
    "KXHIGHMIA-26MAY08-B86.5": False,
    "KXHIGHMIA-26MAY08-B88.5": False,
    "KXHIGHMIA-26MAY08-B90.5": False,
    "KXHIGHMIA-26MAY08-B92.5": True,
}


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def _expected_verdict(row: EvaluationRow, cfg: GateConfig) -> str | None:
    """Compute the expected gate-stack verdict for one row under
    `{cal_floor, plateau_filter}` using the same predicates as the
    scaffold's `_canonical_gate_order` (cal_floor → plateau_filter,
    first-failure-wins).
    """
    model_prob = row.emos_prob_bracket if row.emos_prob_bracket is not None else 0.0
    raw_prob = row.bracket_raw_prob_production_mirrored

    if cfg.min_calibrated_prob is not None and model_prob < cfg.min_calibrated_prob:
        return "CAL_FLOOR_BLOCK"
    rng = cfg.plateau_range
    if rng is not None and rng[0] <= raw_prob <= rng[1]:
        return "PLATEAU_BLOCK"
    return None


def test_step0_sanity() -> None:
    """Spec §8.2 — 5 invariants on the slim MIA fixture.

    1. exactly 6 rows for the MIA/high `KXHIGHMIA-26MAY08-*` brackets;
    2. each row's `(emos_mu, emos_sigma)` matches the spike's
       `(92.736, 0.993)` to 1e-3 (pooled-across-leads coefficients
       make this lead-source-agnostic — spec §8.2 (a));
    3. outcomes vs settled NWS=93 °F match the spike's map (T93 YES,
       all others NO);
    4. under `gates={cal_floor, plateau_filter}`, each row's
       `rejection_reason` matches the canonical-order verdict derived
       independently from `(raw_ensemble_prob, emos_prob_bracket)`;
    5. `build_envelope` is reproducible — two invocations return equal
       envelopes with a 64-char hex `snapshots_dir_sha`.
    """
    emos_table = EmosTable.load(_PROD_COEFFS)
    assert emos_table.is_loaded, f"prod EMOS table not loaded: {emos_table.fallback_reason}"

    rows = list(
        iter_evaluation_rows(
            _SLIM_FIXTURE,
            emos_table=emos_table,
            cell_filter=frozenset({"MIA/high"}),
            allow_same_day_settlement=True,
        )
    )

    # --- Invariant 1: row count -------------------------------------------
    assert len(rows) == 6, (
        f"expected exactly 6 rows for KXHIGHMIA-26MAY08-* / MIA/high; got "
        f"{len(rows)}: {[r.market_ticker for r in rows]}"
    )
    tickers = {r.market_ticker for r in rows}
    assert tickers == set(_EXPECTED_OUTCOMES), (
        f"emitted ticker set mismatch: got {sorted(tickers)} expected {sorted(_EXPECTED_OUTCOMES)}"
    )

    # --- Invariant 2: EMOS posterior moments ------------------------------
    for r in rows:
        assert abs(r.emos_mu - _EXPECTED_MU) < _POSTERIOR_TOL, (
            f"{r.market_ticker}: emos_mu={r.emos_mu:.6f}; expected "
            f"{_EXPECTED_MU} ± {_POSTERIOR_TOL}"
        )
        assert abs(r.emos_sigma - _EXPECTED_SIGMA) < _POSTERIOR_TOL, (
            f"{r.market_ticker}: emos_sigma={r.emos_sigma:.6f}; expected "
            f"{_EXPECTED_SIGMA} ± {_POSTERIOR_TOL}"
        )
        assert r.live_calibrator_kind == "emos_raw"

    # --- Invariant 3: outcomes vs settled NWS=93 °F ----------------------
    for r in rows:
        assert r.settled_temp_f == _SETTLED_HIGH_F, (
            f"{r.market_ticker}: settled_temp_f={r.settled_temp_f}; expected {_SETTLED_HIGH_F}"
        )
        assert r.settled_is_preliminary is False
        assert r.pending_settlement is False
        expected = _EXPECTED_OUTCOMES[r.market_ticker]
        assert r.outcome_yes is expected, (
            f"{r.market_ticker}: outcome_yes={r.outcome_yes}; expected {expected}"
        )

    # --- Invariant 4: gate verdict map -----------------------------------
    cfg = GateConfig()
    gated_rows = list(
        iter_evaluation_rows(
            _SLIM_FIXTURE,
            emos_table=emos_table,
            cell_filter=frozenset({"MIA/high"}),
            allow_same_day_settlement=True,
            gates=frozenset({"cal_floor", "plateau_filter"}),
            gate_config=cfg,
        )
    )
    assert len(gated_rows) == 6
    gated_by_ticker = {r.market_ticker: r for r in gated_rows}
    base_by_ticker = {r.market_ticker: r for r in rows}
    for ticker, gated in gated_by_ticker.items():
        base = base_by_ticker[ticker]
        assert (
            gated.bracket_raw_prob_production_mirrored == base.bracket_raw_prob_production_mirrored
        )
        assert gated.emos_prob_bracket == base.emos_prob_bracket
        expected_verdict = _expected_verdict(base, cfg)
        assert gated.rejection_reason == expected_verdict, (
            f"{ticker}: rejection_reason={gated.rejection_reason!r} "
            f"expected={expected_verdict!r} "
            f"(raw={base.bracket_raw_prob_production_mirrored:.4f}, "
            f"model={base.emos_prob_bracket})"
        )
        # Sanity: the cal_floor predicate (model_prob < 0.10) is the only one
        # that can fire on this fixture — all raw probs sit outside the
        # plateau range [0.649, 0.967].
        if expected_verdict is None:
            assert gated.rejection_payload is None
        else:
            assert gated.rejection_payload is not None

    # --- Invariant 5: envelope reproducibility ---------------------------
    env1 = build_envelope(_SLIM_FIXTURE, emos_table)
    env2 = build_envelope(_SLIM_FIXTURE, emos_table)
    assert env1 == env2
    assert env1.snapshots_dir_sha == env2.snapshots_dir_sha
    assert len(env1.snapshots_dir_sha) == 64
    assert all(c in "0123456789abcdef" for c in env1.snapshots_dir_sha)

    # Sanity cross-check that the spike's spot reading is reachable from
    # the slim corpus (raw_prob via member fraction matches the spike's
    # Gaussian-equivalent inputs to within rounding on T93 NO).
    t93 = base_by_ticker["KXHIGHMIA-26MAY08-T93"]
    # Member-fraction raw_prob can be 0/30 = 0 because every GFS member sits
    # < 93 °F; the EMOS posterior μ=92.736, σ=0.993 puts ~40% mass above 93 °F
    # via _normal_cdf, matching the spike's `EMOS only` arm value (~0.395).
    p_via_cdf = 1.0 - _normal_cdf(93.0, t93.emos_mu, t93.emos_sigma)
    assert t93.emos_prob_bracket is not None
    assert abs(t93.emos_prob_bracket - p_via_cdf) < 1e-9
