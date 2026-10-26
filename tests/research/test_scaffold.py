"""M2 + M3 + M4 gating tests.

See plan §D M2 / §D M3 / §D M4 (gating tests) and spec §4.2 / §4.6 / §5 /
§6.1 / §6.4 / §8.1 A13.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

import pytest

from python.data.emos import EmosTable, emos_probability
from python.research.scaffold import (
    GateConfig,
    ScaffoldEnvelope,
    SkipReason,
    apply_gate_stack,
    build_envelope,
    iter_evaluation_rows,
    recompute_emos_posterior,
)

_PROD_COEFFS_PATH = Path("data/emos/coefficients.json")

_PINNED_HASH = "3ca7b94d9d423fac63dbeb481f6684febd086131f7df54724f511ebc51770260"
_PINNED_COEFFS_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "coefficients_history_snapshots"
    / f"{_PINNED_HASH}.json"
)
_EVAL_AUDIT_EXCERPT_PATH = (
    Path(__file__).parent / "fixtures" / "eval_audit_excerpt.jsonl"
)


def test_emos_posterior_matches_emos_probability() -> None:
    """`recompute_emos_posterior` mirrors `emos.py:82-84` + `emos_probability`.

    For MIA/high coefficients at the canonical spike input
    (ens_mean=88.75, ens_std=0.677, between/[90.0, 91.0]), the helper's
    `(mu, sigma, prob)` must match the direct formulas to 1e-12.
    """
    table = EmosTable.load(_PROD_COEFFS_PATH)
    assert table.is_loaded, f"prod EMOS table not loaded: {table.fallback_reason}"
    coeffs = table.get(city="MIA", var="high", lead=0)
    assert coeffs is not None, "MIA/high coefficients missing from prod artifact"

    ens_mean = 88.75
    ens_std = 0.677
    floor = 90.0
    cap = 91.0

    mu, sigma, prob = recompute_emos_posterior(
        coeffs=coeffs,
        ens_mean=ens_mean,
        ens_std=ens_std,
        strike_type="between",
        floor=floor,
        cap=cap,
    )

    expected_mu = coeffs.a + coeffs.b * ens_mean
    expected_sigma = sqrt(max(coeffs.c + coeffs.d * ens_std**2, 1e-12))
    expected_prob, _, _ = emos_probability(
        coeffs,
        ens_mean,
        ens_std,
        strike_type="between",
        floor=floor,
        cap=cap,
    )

    assert abs(mu - expected_mu) < 1e-12
    assert abs(sigma - expected_sigma) < 1e-12
    assert abs(prob - expected_prob) < 1e-12


def test_emos_posterior_matches_eval_audit() -> None:
    """A13 — bit-identical match to `state/eval_audit.jsonl` for an EMOS-live row.

    Pinned to coefficient artifact sha
    ``3ca7b94d9d423fac63dbeb481f6684febd086131f7df54724f511ebc51770260``
    (the 2026-05-11 cutover artifact, per CLAUDE.md). The companion
    `eval_audit_excerpt.jsonl` fixture must contain at least one EMOS-live
    non-same-day row whose `emos_coeffs_artifact_hash` matches the pinned
    hash. If the excerpt is absent we skip per plan §F.2 — the operator
    must add it (the historical artifact never made it into the
    available `state/eval_audit.jsonl*` rotated files).

    See scaffold spec §8.1 A13 and plan §F.2.
    """
    if not _PINNED_COEFFS_PATH.exists():
        pytest.skip(
            "A13 fixture not available — pinned coefficients artifact "
            f"{_PINNED_HASH} missing from "
            "tests/research/fixtures/coefficients_history_snapshots/. "
            "See scaffold spec §8.1 A13, plan §F.2."
        )
    if not _EVAL_AUDIT_EXCERPT_PATH.exists():
        pytest.skip(
            "A13 fixture not available — eval_audit_excerpt.jsonl absent. "
            "No row in state/eval_audit.jsonl* carries "
            f"emos_coeffs_artifact_hash={_PINNED_HASH} as of 2026-05-14; "
            "see scaffold spec §8.1 A13, plan §F.2."
        )

    import json

    table = EmosTable.load(_PINNED_COEFFS_PATH)
    assert table.is_loaded, f"pinned EMOS table not loaded: {table.fallback_reason}"
    assert table.artifact_hash == _PINNED_HASH, (
        f"pinned artifact hash mismatch: got {table.artifact_hash!r}, "
        f"expected {_PINNED_HASH!r}"
    )

    with _EVAL_AUDIT_EXCERPT_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("emos_coeffs_artifact_hash") != _PINNED_HASH:
                continue
            if row.get("emos_prob_raw") is None:
                continue
            coeffs = table.get(city=row["city"], var=row["var"], lead=row["emos_lead"])
            assert coeffs is not None
            floor = row.get("floor")
            cap = row.get("cap")
            _, _, prob = recompute_emos_posterior(
                coeffs=coeffs,
                ens_mean=row["ens_mean"],
                ens_std=row["ens_std"],
                strike_type=row["strike_type"],
                floor=floor,
                cap=cap,
            )
            assert abs(prob - row["emos_prob_raw"]) < 0.005
            return

    pytest.skip(
        "A13 fixture excerpt present but contains no EMOS-live row pinned "
        f"to {_PINNED_HASH}; see scaffold spec §8.1 A13."
    )


# ---------------------------------------------------------------------------
# M3 — iter_evaluation_rows happy path
# ---------------------------------------------------------------------------

_M3_FIXTURE = Path(__file__).parent / "fixtures" / "snapshots_synthetic_m3"
_PROD_COEFFS = Path("data/emos/coefficients.json")


def _load_prod_table() -> EmosTable:
    table = EmosTable.load(_PROD_COEFFS)
    assert table.is_loaded, f"prod EMOS table not loaded: {table.fallback_reason}"
    return table


def test_iter_order_determinism() -> None:
    """Spec §6.1 + §6.4 — two invocations on identical inputs yield identical
    rows in identical `(snapshot_day asc, run_id asc, market_ticker asc)` order.
    """
    table = _load_prod_table()
    rows_first = list(iter_evaluation_rows(_M3_FIXTURE, emos_table=table))
    rows_second = list(iter_evaluation_rows(_M3_FIXTURE, emos_table=table))

    assert rows_first == rows_second
    assert len(rows_first) >= 1, "fixture should emit at least one row"

    # Independent verification that the canonical ordering holds.
    ordering_keys = [
        (r.snapshot_day, r.run_id, r.market_ticker) for r in rows_first
    ]
    assert ordering_keys == sorted(ordering_keys)


def test_load_snapshot_data_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec §8.1 — `load_snapshot_data` is invoked exactly once per
    `iter_evaluation_rows` call. Guards against per-run-dir reloads that
    would scale O(n_runs) of file I/O.
    """
    import python.research.scaffold as scaffold
    from scripts import _calibration_common

    original = _calibration_common.load_snapshot_data
    call_count = 0

    def counting(snapshots_dir):
        nonlocal call_count
        call_count += 1
        return original(snapshots_dir)

    monkeypatch.setattr(scaffold, "load_snapshot_data", counting)

    table = _load_prod_table()
    rows = list(iter_evaluation_rows(_M3_FIXTURE, emos_table=table))

    assert call_count == 1, f"expected exactly one load_snapshot_data call, got {call_count}"
    assert len(rows) >= 1


def test_iter_happy_path_minimal_fixture() -> None:
    """Spec §3.2 — the happy path populates all required `EvaluationRow`
    fields (identity, forecast inputs, EMOS posterior, market state,
    settlement, lead/temporal). No skip branches fire on this fixture.
    """
    table = _load_prod_table()
    rows = list(iter_evaluation_rows(_M3_FIXTURE, emos_table=table))

    assert len(rows) >= 1
    for r in rows:
        # Identity
        assert r.city == "MIA"
        assert r.var == "high"
        assert r.daily_variable == "temperature_2m_max"
        assert r.run_id == "1500"
        assert r.market_ticker.startswith("KXHIGHMIA-")
        assert r.event_ticker
        assert r.strike_type in ("greater", "less", "between")

        # Forecast inputs (ddof=1 recompute; spec §6.3)
        assert r.n_members >= 25
        assert r.ens_std > 0.0
        assert 0.0 <= r.bracket_raw_prob_single_model <= 1.0
        # With ensemble_weights=None and bias_table=None, the two raw-prob
        # columns are bit-identical (§3.2.1 invariant table row 1).
        assert r.bracket_raw_prob_production_mirrored == r.bracket_raw_prob_single_model

        # EMOS posterior — MIA/high is EMOS-live in prod, so live_calibrator_kind
        # should be "emos_raw" (not "identity").
        assert r.live_calibrator_kind == "emos_raw"
        assert r.emos_prob_bracket is not None
        assert 0.0 <= r.emos_prob_bracket <= 1.0
        assert r.emos_sigma > 0.0
        assert r.emos_coeffs_hash  # non-empty hash
        # lead_days is unclamped per v1.2 B10; for this fixture it should be 1
        # (snapshot 2026-05-07 / 2026-05-08 → fc 2026-05-08 / 2026-05-09).
        assert r.lead_days == 1
        assert r.emos_lead == 1

        # Settlement — fixture carries final settlements only.
        assert r.pending_settlement is False
        assert r.settled_temp_f is not None
        assert r.settled_is_preliminary is False
        assert r.outcome_yes is not None

        # Gate verdict — gates=None → no rejection annotation.
        assert r.rejection_reason is None
        assert r.rejection_payload is None


# ---------------------------------------------------------------------------
# M4 — skip semantics (spec §5 precedence + §4.1 filter validation)
# ---------------------------------------------------------------------------

# 25-member ensemble centred on 87 °F. The mean and spread are wide enough that
# every bracket lookup stays in a well-defined raw_prob region; we vary the
# market floor/cap per fixture rather than the forecast.
_BASE_TEMPS = [
    86.6, 85.3, 87.0, 87.3, 86.7, 87.8, 86.9, 86.8, 88.7, 88.3,
    87.9, 88.7, 89.2, 88.5, 86.9, 87.6, 87.7, 86.3, 86.6, 87.3,
    87.9, 89.2, 88.1, 86.8, 87.6,
]


def _write_snapshot(
    base: Path,
    *,
    snapshot_day: str,
    run_id: str,
    markets: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    settlements: list[dict[str, Any]],
) -> Path:
    """Write a single snapshot run dir under `base` and return its path."""
    run_dir = base / snapshot_day / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "markets.json").write_text(json.dumps(markets))
    (run_dir / "weather_forecasts.json").write_text(json.dumps(forecasts))
    (run_dir / "settlements.json").write_text(json.dumps(settlements))
    return run_dir


def _make_market(
    *,
    ticker: str,
    city: str,
    daily_variable: str,
    strike_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "series_ticker": ticker.rsplit("-", 1)[0],
        "city": city,
        "daily_variable": daily_variable,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "yes_bid_dollars": 0.23,
        "yes_ask_dollars": 0.24,
        "volume_fp": 100,
        "expiration_ts": 1778335200,
        "snapshot_ts": 1778170140.0,
    }


def _make_forecast(
    *,
    city: str,
    forecast_date: str,
    daily_variable: str,
    temperatures: list[float] | None = None,
) -> dict[str, Any]:
    temps = list(_BASE_TEMPS) if temperatures is None else temperatures
    return {
        "city": city,
        "forecast_date": forecast_date,
        "daily_variable": daily_variable,
        "temperatures": temps,
        "mean": sum(temps) / len(temps),
        "std_dev": 1.0,
        "fetched_at": 1778170148.0,
        "model": "gfs_seamless",
        "model_version": "GFSv16",
    }


def _make_settlement(
    *,
    city: str,
    observation_date: str,
    high_f: int | None = 91,
    low_f: int | None = 72,
    is_preliminary: bool = False,
    product_id: str = "fixture-final",
    issued_at: str = "2026-05-10T08:22:00+00:00",
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "city": city,
        "cli_location": city,
        "observation_date": observation_date,
        "product_id": product_id,
        "issued_at": issued_at,
        "is_preliminary": is_preliminary,
        "snapshot_ts": 1778359320.0,
    }
    if high_f is not None:
        rec["high_f"] = high_f
    if low_f is not None:
        rec["low_f"] = low_f
    return rec


def _prod_table() -> EmosTable:
    table = EmosTable.load(_PROD_COEFFS_PATH)
    assert table.is_loaded, f"prod EMOS table not loaded: {table.fallback_reason}"
    return table


class _SkipCapture:
    """Test helper: record every `on_skip` invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[SkipReason, dict[str, Any]]] = []

    def __call__(self, reason: SkipReason, context: Any) -> None:
        self.calls.append((reason, dict(context)))

    def reasons(self) -> list[SkipReason]:
        return [r for r, _ in self.calls]


# 1. CHI pre-fix-date skip — spec §5 precedence #3, mirrors
#    `_calibration_common.py:186-187` (date(2026, 3, 20)).
def test_iter_chi_pre_fix_skip(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    # Ticker carries a CHI forecast date BEFORE the fix date.
    bad_ticker = "KXHIGHCHI-26MAR15-T70"
    good_ticker = "KXHIGHMIA-26MAY09-T90"
    _write_snapshot(
        snapshots,
        snapshot_day="2026-03-14",
        run_id="1500",
        markets=[
            _make_market(
                ticker=bad_ticker,
                city="CHI",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=70.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="CHI",
                forecast_date="2026-03-15",
                daily_variable="temperature_2m_max",
            ),
        ],
        settlements=[
            _make_settlement(city="CHI", observation_date="2026-03-15"),
        ],
    )
    # Also include a non-CHI row so the iterator has a normal happy-path
    # comparison point. (We don't assert it emits because the forecast for
    # this row would still be CHI's; the assertion is that no CHI row fires.)

    capture = _SkipCapture()
    rows = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            on_skip=capture,
        )
    )

    # The CHI row must have been skipped.
    assert any(r is SkipReason.CHI_PRE_FIX_DATE for r in capture.reasons())
    chi_calls = [c for r, c in capture.calls if r is SkipReason.CHI_PRE_FIX_DATE]
    assert chi_calls
    assert chi_calls[0]["market_ticker"] == bad_ticker
    assert chi_calls[0]["city"] == "CHI"
    # No CHI rows in the output.
    assert not any(r.market_ticker.startswith("KXHIGHCHI-") for r in rows)
    # Sanity: the "good" ticker placeholder doesn't exist in fixture; we
    # still assert no emission for a CHI pre-fix forecast date.
    assert good_ticker not in {r.market_ticker for r in rows}


# 2. FORECAST_THIN_MEMBERS skip — spec §5 precedence #6, `_calibration_common.py:205`.
def test_iter_skips_on_thin_members(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    thin_temps = _BASE_TEMPS[:20]  # 20 < 25
    _write_snapshot(
        snapshots,
        snapshot_day="2026-05-07",
        run_id="1500",
        markets=[
            _make_market(
                ticker="KXHIGHMIA-26MAY08-T90",
                city="MIA",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=90.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="MIA",
                forecast_date="2026-05-08",
                daily_variable="temperature_2m_max",
                temperatures=thin_temps,
            ),
        ],
        settlements=[
            _make_settlement(city="MIA", observation_date="2026-05-08"),
        ],
    )

    capture = _SkipCapture()
    rows = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            on_skip=capture,
        )
    )
    assert SkipReason.FORECAST_THIN_MEMBERS in capture.reasons()
    thin_calls = [c for r, c in capture.calls if r is SkipReason.FORECAST_THIN_MEMBERS]
    assert thin_calls[0]["n_members"] == 20
    assert not rows


# 3. SAME_DAY_SUPPRESSED — spec §5 precedence #5; pinned to snapshot_day_utc
#    regardless of lead_source (v1.2 B8).
def test_iter_same_day_suppressed_by_default(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    same_day_ticker = "KXHIGHMIA-26MAY08-T90"
    _write_snapshot(
        snapshots,
        snapshot_day="2026-05-08",  # snapshot_day == forecast_date
        run_id="1500",
        markets=[
            _make_market(
                ticker=same_day_ticker,
                city="MIA",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=90.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="MIA",
                forecast_date="2026-05-08",
                daily_variable="temperature_2m_max",
            ),
        ],
        settlements=[
            _make_settlement(city="MIA", observation_date="2026-05-08"),
        ],
    )

    capture = _SkipCapture()
    rows_default = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            on_skip=capture,
        )
    )
    assert not rows_default
    assert SkipReason.SAME_DAY_SUPPRESSED in capture.reasons()
    same_day_calls = [
        c for r, c in capture.calls if r is SkipReason.SAME_DAY_SUPPRESSED
    ]
    assert same_day_calls[0]["snapshot_day_utc"] == "2026-05-08"
    assert same_day_calls[0]["market_ticker"] == same_day_ticker

    # Inverse — allow_same_day_settlement=True keeps the row.
    rows_allowed = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            allow_same_day_settlement=True,
        )
    )
    assert len(rows_allowed) == 1
    assert rows_allowed[0].market_ticker == same_day_ticker


# 4. require_emos fork — spec §5 precedence #7 / #7'. We need a cell absent
#    from `emos_table.cells`. Production has all 14 cells, so we build a
#    pruned EmosTable for this test.
def test_require_emos_drops_identity_rows(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    _write_snapshot(
        snapshots,
        snapshot_day="2026-05-07",
        run_id="1500",
        markets=[
            _make_market(
                ticker="KXHIGHDEN-26MAY08-T70",
                city="DEN",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=70.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="DEN",
                forecast_date="2026-05-08",
                daily_variable="temperature_2m_max",
            ),
        ],
        settlements=[
            _make_settlement(city="DEN", observation_date="2026-05-08"),
        ],
    )

    full = _prod_table()
    # Pruned table: drop DEN/high so the cell is absent.
    pruned_cells = {k: v for k, v in full.cells.items() if k != "DEN/high"}
    assert "DEN/high" not in pruned_cells
    pruned = EmosTable(
        cells=pruned_cells,
        artifact_hash=full.artifact_hash,
        artifact_signature=full.artifact_signature,
        artifact_loaded_at=full.artifact_loaded_at,
        fit_date=full.fit_date,
        training_window_days=full.training_window_days,
        lead_handling=full.lead_handling,
        schema_version=full.schema_version,
        n_skipped=full.n_skipped,
        skipped_cells=dict(full.skipped_cells),
        fallback_reason=full.fallback_reason,
        _path=full._path,
    )

    # Leg A — require_emos=False: row IS emitted with identity calibrator.
    rows_identity = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=pruned,
            require_emos=False,
        )
    )
    assert len(rows_identity) == 1
    assert rows_identity[0].live_calibrator_kind == "identity"
    assert (
        rows_identity[0].emos_prob_bracket
        == rows_identity[0].bracket_raw_prob_production_mirrored
    )

    # Leg B — require_emos=True: row skipped via NO_EMOS_COEFFS, with the
    # A11 context fields included.
    capture = _SkipCapture()
    rows_required = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=pruned,
            require_emos=True,
            on_skip=capture,
        )
    )
    assert not rows_required
    assert SkipReason.NO_EMOS_COEFFS in capture.reasons()
    ctx = next(c for r, c in capture.calls if r is SkipReason.NO_EMOS_COEFFS)
    assert ctx["emos_artifact_hash"] == pruned.artifact_hash
    assert ctx["emos_fit_date"] == pruned.fit_date
    assert "lead_days_requested" in ctx


# 5. cell_filter typo raises ValueError on construction — spec §4.1 A14.
def test_cell_filter_typo_raises(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    with pytest.raises(ValueError, match=r"MIA/high \(lowercase\)"):
        list(
            iter_evaluation_rows(
                snapshots,
                emos_table=_prod_table(),
                cell_filter=frozenset({"MIA/HIGH"}),
            )
        )


# 6. excluded_cell_vars separator pin (reviewer flag #3) — spec §4.1 B2.
def test_excluded_cell_vars_underscore_form(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    # Build a minimal snapshot so iteration is otherwise reachable.
    _write_snapshot(
        snapshots,
        snapshot_day="2026-05-07",
        run_id="1500",
        markets=[
            _make_market(
                ticker="KXHIGHMIA-26MAY08-T90",
                city="MIA",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=90.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="MIA",
                forecast_date="2026-05-08",
                daily_variable="temperature_2m_max",
            ),
        ],
        settlements=[
            _make_settlement(city="MIA", observation_date="2026-05-08"),
        ],
    )

    # Leg A — underscore form is accepted; no ValueError.
    list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            excluded_cell_vars=frozenset({"DEN_high"}),
        )
    )

    # Leg B — slash form must raise with the four required components.
    with pytest.raises(ValueError) as excinfo:
        list(
            iter_evaluation_rows(
                snapshots,
                emos_table=_prod_table(),
                excluded_cell_vars=frozenset({"DEN/high"}),
            )
        )
    msg = str(excinfo.value)
    assert "underscore form" in msg
    assert "weather_evaluation.py:591" in msg
    assert "slash form" in msg
    assert "DEN/high" in msg


# 7. Skip precedence — cell-absent (#7) fires BEFORE lead-out-of-horizon (#8).
#    Reviewer flag #1 second item. Forecast date 4 days out, cell pruned.
def test_skip_precedence_emos_cell_missing_with_lead_out_of_horizon(
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    # snapshot 2026-05-07 → forecast 2026-05-11 → lead_days = 4 (> 2 horizon).
    _write_snapshot(
        snapshots,
        snapshot_day="2026-05-07",
        run_id="1500",
        markets=[
            _make_market(
                ticker="KXHIGHDEN-26MAY11-T70",
                city="DEN",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=70.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="DEN",
                forecast_date="2026-05-11",
                daily_variable="temperature_2m_max",
            ),
        ],
        settlements=[
            _make_settlement(city="DEN", observation_date="2026-05-11"),
        ],
    )

    full = _prod_table()
    pruned_cells = {k: v for k, v in full.cells.items() if k != "DEN/high"}
    pruned = EmosTable(
        cells=pruned_cells,
        artifact_hash=full.artifact_hash,
        artifact_signature=full.artifact_signature,
        artifact_loaded_at=full.artifact_loaded_at,
        fit_date=full.fit_date,
        training_window_days=full.training_window_days,
        lead_handling=full.lead_handling,
        schema_version=full.schema_version,
        n_skipped=full.n_skipped,
        skipped_cells=dict(full.skipped_cells),
        fallback_reason=full.fallback_reason,
        _path=full._path,
    )

    capture = _SkipCapture()
    rows = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=pruned,
            require_emos=True,
            on_skip=capture,
        )
    )
    assert not rows
    # Cell-absent fires first; the lead-out-of-horizon reason must NOT appear.
    assert SkipReason.NO_EMOS_COEFFS in capture.reasons()
    assert SkipReason.EMOS_OUT_OF_HORIZON not in capture.reasons()
    ctx = next(c for r, c in capture.calls if r is SkipReason.NO_EMOS_COEFFS)
    assert ctx["lead_days_requested"] == 4


# 8. allow_preliminary toggle — spec §5 precedence #9 / #10. When the
#    settlement record is missing entirely, `pending_settlement=True` and
#    `outcome_yes=None`.
def test_allow_preliminary_true_yields_pending(tmp_path: Path) -> None:
    snapshots = tmp_path / "snaps"
    snapshots.mkdir()
    # No settlement file at all for this run dir → pending_settlement=True.
    _write_snapshot(
        snapshots,
        snapshot_day="2026-05-07",
        run_id="1500",
        markets=[
            _make_market(
                ticker="KXHIGHMIA-26MAY08-T90",
                city="MIA",
                daily_variable="temperature_2m_max",
                strike_type="greater",
                floor_strike=90.0,
                cap_strike=None,
            ),
        ],
        forecasts=[
            _make_forecast(
                city="MIA",
                forecast_date="2026-05-08",
                daily_variable="temperature_2m_max",
            ),
        ],
        settlements=[],
    )

    # Default (allow_preliminary=False) → row skipped with NO_SETTLEMENT.
    capture = _SkipCapture()
    rows_default = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            on_skip=capture,
        )
    )
    assert not rows_default
    assert SkipReason.NO_SETTLEMENT in capture.reasons()

    # allow_preliminary=True → row emitted with outcome_yes=None,
    # pending_settlement=True.
    rows_pending = list(
        iter_evaluation_rows(
            snapshots,
            emos_table=_prod_table(),
            allow_preliminary=True,
        )
    )
    assert len(rows_pending) == 1
    assert rows_pending[0].pending_settlement is True
    assert rows_pending[0].outcome_yes is None
    assert rows_pending[0].settled_temp_f is None


# ---------------------------------------------------------------------------
# M5 — apply_gate_stack + per-gate predicates (spec §4.5; plan §D M5)
# ---------------------------------------------------------------------------


def _gate_kwargs(
    *,
    raw_ensemble_prob: float = 0.20,
    model_prob: float = 0.20,
    market_mid: float | None = 0.50,
    market_spread: float | None = 0.05,
    strike_type: str = "between",
    is_excluded_cell: bool = False,
    live_calibrator_kind: str = "isotonic",
    strike_abs_z: float | None = None,
    config: GateConfig | None = None,
) -> dict[str, Any]:
    """Defaults that PASS every gate; tests override only what they trip."""
    return {
        "raw_ensemble_prob": raw_ensemble_prob,
        "model_prob": model_prob,
        "market_mid": market_mid,
        "market_spread": market_spread,
        "strike_type": strike_type,
        "is_excluded_cell": is_excluded_cell,
        "live_calibrator_kind": live_calibrator_kind,
        "strike_abs_z": strike_abs_z,
        "config": config if config is not None else GateConfig(),
    }


# 1. cal_floor fires at low model prob.
def test_gate_stack_cal_floor_fires_at_low_model() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"cal_floor"}),
        **_gate_kwargs(model_prob=0.05),
    )
    assert reason == "CAL_FLOOR_BLOCK"
    assert payload is not None
    assert payload["model_prob"] == 0.05


# 2. plateau fires when cal_floor passes (A1 reorder fix).
def test_gate_stack_plateau_fires_in_range_when_cal_passes() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"cal_floor", "plateau_filter"}),
        **_gate_kwargs(raw_ensemble_prob=0.7, model_prob=0.20),
    )
    assert reason == "PLATEAU_BLOCK"
    assert payload is not None
    assert payload["raw_ensemble_prob"] == 0.7


# 3. raw_prob_gate fires on between with raw below floor.
def test_gate_stack_raw_prob_gate_fires_between_low() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"cal_floor", "plateau_filter", "raw_prob_gate"}),
        **_gate_kwargs(
            strike_type="between",
            raw_ensemble_prob=0.01,
            model_prob=0.20,
        ),
    )
    assert reason == "RAW_PROB_GATE"
    assert payload is not None
    assert payload["raw_ensemble_prob"] == 0.01
    assert payload["min_raw_prob_for_between"] == 0.02


# 4. spread_filter fires when market_spread exceeds max_spread.
def test_gate_stack_spread_filter_fires() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"spread_filter"}),
        **_gate_kwargs(
            market_spread=0.20,
            config=GateConfig(max_spread=0.10),
        ),
    )
    assert reason == "SPREAD"
    assert payload is not None
    assert payload["market_spread"] == 0.20
    assert payload["max_spread"] == 0.10


# 5. threshold_raw_prob_clamp fires when raw > 0.95 on threshold market.
def test_gate_stack_threshold_raw_prob_clamp_fires() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"threshold_raw_prob_clamp"}),
        **_gate_kwargs(
            strike_type="greater",
            raw_ensemble_prob=0.97,
        ),
    )
    assert reason == "RAW_PROB_GATE"
    assert payload is not None
    assert payload["raw_ensemble_prob"] == 0.97
    assert payload["threshold_clamp_hi"] == 0.95


# 6. strike_distance — no-op on strike_abs_z=None; fires when z below floor (B5).
def test_gate_stack_strike_distance_noop_when_z_none() -> None:
    # Leg A — z=None is a strict pass.
    reason_none, payload_none = apply_gate_stack(
        gates=frozenset({"strike_distance"}),
        **_gate_kwargs(
            strike_type="greater",
            strike_abs_z=None,
            config=GateConfig(min_distance_from_strike_std=2.0),
        ),
    )
    assert reason_none is None
    assert payload_none is None

    # Leg B — z below floor trips.
    reason_low, payload_low = apply_gate_stack(
        gates=frozenset({"strike_distance"}),
        **_gate_kwargs(
            strike_type="greater",
            strike_abs_z=0.5,
            config=GateConfig(min_distance_from_strike_std=2.0),
        ),
    )
    assert reason_low == "STRIKE_DISTANCE"
    assert payload_low is not None
    assert payload_low["strike_abs_z"] == 0.5
    assert payload_low["min_distance_from_strike_std"] == 2.0


# 7. calibrator_disagreement BYPASS when caller claims live_calibrator_kind="emos_raw".
def test_gate_stack_disagreement_bypass_emos_raw() -> None:
    # Disagreement signature: (raw-mid)*(model-mid) = (-0.10)*(0.10) < 0.
    reason, payload = apply_gate_stack(
        gates=frozenset({"calibrator_disagreement"}),
        **_gate_kwargs(
            raw_ensemble_prob=0.40,
            model_prob=0.60,
            market_mid=0.50,
            live_calibrator_kind="emos_raw",
        ),
    )
    assert reason is None
    assert payload is None


# 8. disagreement FIRES when caller's claim is "isotonic" even on an emos cell (A8).
def test_gate_stack_disagreement_fires_isotonic_arm_on_emos_cell() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"calibrator_disagreement"}),
        **_gate_kwargs(
            raw_ensemble_prob=0.40,
            model_prob=0.60,
            market_mid=0.50,
            # Caller's CLAIM is isotonic — even if the row's emitted tag
            # is emos_raw, the gate fires because the bypass keys on the
            # caller's argument (v1.1 A8).
            live_calibrator_kind="isotonic",
        ),
    )
    assert reason == "CALIBRATOR_DISAGREEMENT"
    assert payload is not None
    assert payload["raw_ensemble_prob"] == 0.40
    assert payload["model_prob"] == 0.60
    assert payload["market_mid"] == 0.50


# 9. excluded_cell_gate tags the row (A2). Direct apply_gate_stack call.
def test_excluded_cell_gate_tags_only() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"excluded_cell_gate"}),
        **_gate_kwargs(is_excluded_cell=True),
    )
    assert reason == "EXCLUDED_CELL"
    assert payload is not None


# 10. excluded_cell_gate precedence over cal_floor (B1, reviewer flag #1).
def test_excluded_cell_gate_precedence_over_cal_floor() -> None:
    reason, payload = apply_gate_stack(
        gates=frozenset({"excluded_cell_gate", "cal_floor"}),
        **_gate_kwargs(
            is_excluded_cell=True,
            model_prob=0.05,  # would also trip cal_floor
        ),
    )
    assert reason == "EXCLUDED_CELL"
    assert payload is not None


# 11. gates=frozenset() (empty set) skips every gate → no rejection.
def test_gates_none_skips_all() -> None:
    # Direct call with empty frozenset — no gates evaluated.
    reason, payload = apply_gate_stack(
        gates=frozenset(),
        **_gate_kwargs(
            # Inputs that WOULD trip several gates if any were active.
            raw_ensemble_prob=0.7,
            model_prob=0.05,
            market_spread=0.50,
            strike_type="between",
            is_excluded_cell=True,
            config=GateConfig(max_spread=0.10),
        ),
    )
    assert reason is None
    assert payload is None

    # And the iterator's gates=None path: every emitted row has
    # rejection_reason=None (iter_evaluation_rows does not call
    # apply_gate_stack when gates is None).
    table = _load_prod_table()
    rows = list(iter_evaluation_rows(_M3_FIXTURE, emos_table=table, gates=None))
    assert len(rows) >= 1
    for r in rows:
        assert r.rejection_reason is None
        assert r.rejection_payload is None


# ---------------------------------------------------------------------------
# M6 — build_envelope (spec §6.2 / v1.1 A5 / v1.2 B6; plan §D M6)
# ---------------------------------------------------------------------------


def test_build_envelope_pure_function(tmp_path: Path) -> None:
    """v1.1 A5 — `build_envelope` is a pure function.

    Two calls with identical arguments produce equal envelopes, with no
    side effects (no files written, no global state mutated, no
    `load_snapshot_data` calls — the envelope only reads metadata).
    """
    table = _load_prod_table()
    # Copy snapshots dir contents under tmp_path so we can assert nothing
    # was written or mutated by build_envelope.
    work_dir = tmp_path / "snaps"
    work_dir.mkdir()
    for src in _M3_FIXTURE.rglob("*"):
        if src.is_file():
            rel = src.relative_to(_M3_FIXTURE)
            dest = work_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())

    before = sorted(p.relative_to(work_dir).as_posix() for p in work_dir.rglob("*"))

    envelope1 = build_envelope(work_dir, table, GateConfig())
    envelope2 = build_envelope(work_dir, table, GateConfig())

    assert envelope1 == envelope2
    assert isinstance(envelope1, ScaffoldEnvelope)

    after = sorted(p.relative_to(work_dir).as_posix() for p in work_dir.rglob("*"))
    assert before == after, "build_envelope must not write to snapshots_dir"


def test_build_envelope_none_gate_config_serializes_null() -> None:
    """v1.2 B6 — `gate_config=None` produces `gate_config_repr=None`, which
    serializes as JSON `null`. Non-None gate_config produces the dataclass
    repr.
    """
    table = _load_prod_table()

    envelope_none = build_envelope(_M3_FIXTURE, table, gate_config=None)
    assert envelope_none.gate_config_repr is None
    assert (
        json.dumps({"gate_config": envelope_none.gate_config_repr})
        == '{"gate_config": null}'
    )

    envelope_with = build_envelope(_M3_FIXTURE, table, gate_config=GateConfig())
    assert envelope_with.gate_config_repr is not None
    assert isinstance(envelope_with.gate_config_repr, str)
    assert envelope_with.gate_config_repr  # non-empty dataclass repr
    assert "GateConfig" in envelope_with.gate_config_repr


def test_build_envelope_sha_stable_across_runs() -> None:
    """Both SHAs (scaffold module + snapshots dir) reproduce on a second
    invocation against the same inputs; the full envelope is equal.
    """
    table = _load_prod_table()
    envelope1 = build_envelope(_M3_FIXTURE, table, GateConfig())
    envelope2 = build_envelope(_M3_FIXTURE, table, GateConfig())

    assert envelope1.snapshots_dir_sha == envelope2.snapshots_dir_sha
    assert envelope1.scaffold_module_sha == envelope2.scaffold_module_sha
    assert envelope1 == envelope2
    # Sanity — the two SHAs are non-empty hex strings of sha256 length.
    assert len(envelope1.snapshots_dir_sha) == 64
    assert len(envelope1.scaffold_module_sha) == 64
