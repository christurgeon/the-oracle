"""Tests for persistence-BSS diagnostic in calibration_monitor.

Adds a naive-persistence baseline to each per-cell row so the monitor can
distinguish "model is broken on this cell" from "this cell is intrinsically
hard". Persistence predicts today's bracket-membership from yesterday's
observed extremum at the same (city, var) station; the new fields on
``CityVarBrier`` carry persistence_brier, bss_vs_persistence, and
n_persistence per (city, var, kind).

Invariants:

1. ``_persistence_brier`` is a pure helper returning (mean_sq_err, n) over
   the subset of pairs with ``persistence_yes`` not None.
2. ``SettledPair`` carries an optional ``persistence_yes: int | None`` field.
3. ``CityVarBrier`` carries optional ``persistence_brier``,
   ``bss_vs_persistence``, ``n_persistence`` fields.
4. ``_per_city_var_brier`` computes BSS = 1 - live_brier / persistence_brier
   per (city, var, kind), with ``bss_vs_persistence = None`` when
   persistence_brier == 0 (degenerate denominator).
5. Both pair constructors (``_record_to_pair``, ``_record_to_pair_kalshi``)
   accept a settlements dict and set ``persistence_yes`` from
   ``derive_outcome_yes(strike_type, floor, cap, yesterday_temp)``. None
   when no D-1 entry exists or D-1 high/low is missing for the variable.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from python.analytics.calibration_monitor import (
    CityVarBrier,
    SettledPair,
    _load_settlements,
    _per_city_var_brier,
    _persistence_brier,
    _record_to_pair,
    _record_to_pair_kalshi,
    _SettlementEntry,
)
from python.analytics.prediction_log import PredictionRecord
from python.common import Side

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair(
    *,
    city: str = "MIA",
    daily_variable: str = "high",
    predicted: float,
    outcome: int,
    persistence: int | None = None,
    kind: str = "isotonic",
    days_ago: int = 1,
) -> SettledPair:
    return SettledPair(
        timestamp=datetime.now(UTC) - timedelta(days=days_ago),
        market_ticker=f"KX-{city}-{daily_variable}-{kind}-{predicted}-{outcome}-{persistence}",
        city=city,
        daily_variable=daily_variable,
        strike_type="greater",
        predicted_yes=predicted,
        outcome_yes=outcome,
        raw_ensemble_prob=predicted,
        observation_date=date(2026, 4, 15),
        actual_temp=85.0 if outcome else 70.0,
        live_calibrator_kind=kind,
        persistence_yes=persistence,
    )


def _pred_record(
    *,
    city: str = "MIA",
    daily_variable: str = "high",
    obs_date: date = date(2026, 4, 15),
    strike_type: str = "greater",
    floor: float | None = 80.0,
    cap: float | None = None,
    predicted: float = 0.6,
) -> PredictionRecord:
    return PredictionRecord(
        timestamp=datetime.now(UTC) - timedelta(days=1),
        strategy="weather_edge",
        market_ticker=(
            f"KX{daily_variable.upper()}{city}-{obs_date.isoformat()}"
            f"-T{int(floor or cap or 0)}"
        ),
        side=Side.YES,
        city=city,
        daily_variable=daily_variable,
        strike_type=strike_type,
        floor_strike=floor,
        cap_strike=cap,
        expiration_date=obs_date,
        raw_ensemble_prob=predicted,
        calibrated_prob=predicted,
        market_price=0.5,
        exec_price=0.5,
        confidence=0.85,
        live_calibrator_kind="isotonic",
    )


def _settlement(
    *, city: str, obs: date, high: float | None = 80.0, low: float | None = 60.0
) -> _SettlementEntry:
    return _SettlementEntry(
        city=city,
        observation_date=obs,
        high_f=high,
        low_f=low,
        is_preliminary=False,
        issued_at="2026-04-16T09:00:00Z",
        product_id="CLI",
    )


# ---------------------------------------------------------------------------
# _persistence_brier helper (pure math)
# ---------------------------------------------------------------------------


class TestPersistenceBrier:
    def test_empty(self) -> None:
        b, n = _persistence_brier([])
        assert b is None
        assert n == 0

    def test_all_none_skipped(self) -> None:
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=None),
            _pair(predicted=0.5, outcome=0, persistence=None),
        ]
        b, n = _persistence_brier(pairs)
        assert b is None
        assert n == 0

    def test_perfect_persistence(self) -> None:
        """Yesterday's bracket-membership exactly matches today's outcome."""
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=1),
            _pair(predicted=0.5, outcome=0, persistence=0),
            _pair(predicted=0.5, outcome=1, persistence=1),
        ]
        b, n = _persistence_brier(pairs)
        assert b == pytest.approx(0.0)
        assert n == 3

    def test_always_wrong_persistence(self) -> None:
        """Persistence predicts the opposite of today's outcome every time."""
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=0),
            _pair(predicted=0.5, outcome=0, persistence=1),
        ]
        b, n = _persistence_brier(pairs)
        assert b == pytest.approx(1.0)
        assert n == 2

    def test_mixed_skips_none(self) -> None:
        """Pairs with persistence_yes=None must be skipped, not counted as 0."""
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=1),  # err 0
            _pair(predicted=0.5, outcome=1, persistence=0),  # err 1
            _pair(predicted=0.5, outcome=0, persistence=None),  # skip
        ]
        b, n = _persistence_brier(pairs)
        assert b == pytest.approx(0.5)
        assert n == 2


# ---------------------------------------------------------------------------
# CityVarBrier persistence fields
# ---------------------------------------------------------------------------


class TestCityVarBrierPersistenceFields:
    def test_shape_has_new_fields(self) -> None:
        b = CityVarBrier(
            city="MIA",
            daily_variable="high",
            n=10,
            brier=0.18,
            mean_predicted=0.41,
            mean_observed=0.45,
            persistence_brier=0.22,
            bss_vs_persistence=0.18,
            n_persistence=10,
        )
        assert b.persistence_brier == 0.22
        assert b.bss_vs_persistence == pytest.approx(0.18)
        assert b.n_persistence == 10

    def test_defaults_for_legacy_construction(self) -> None:
        """Existing callsites constructing CityVarBrier without the new
        fields must keep working — defaults are None/0."""
        b = CityVarBrier(
            city="MIA",
            daily_variable="high",
            n=10,
            brier=0.18,
            mean_predicted=0.41,
            mean_observed=0.45,
        )
        assert b.persistence_brier is None
        assert b.bss_vs_persistence is None
        assert b.n_persistence == 0


class TestPerCityVarBrierWithPersistence:
    def test_bss_formula_basic(self) -> None:
        """Model picks each outcome correctly (Brier ≈ 0); persistence is
        wrong half the time (Brier = 0.5). BSS ≈ 1."""
        pairs = [
            _pair(predicted=1.0, outcome=1, persistence=1),  # live err 0, pers err 0
            _pair(predicted=1.0, outcome=1, persistence=0),  # live err 0, pers err 1
            _pair(predicted=0.0, outcome=0, persistence=1),  # live err 0, pers err 1
            _pair(predicted=0.0, outcome=0, persistence=0),  # live err 0, pers err 0
        ]
        rows = _per_city_var_brier(pairs)
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        row = by_kind[("MIA", "high", "isotonic")]
        # live Brier == 0, persistence Brier == 0.5 → BSS == 1.0
        assert row.brier == pytest.approx(0.0)
        assert row.persistence_brier == pytest.approx(0.5)
        assert row.bss_vs_persistence == pytest.approx(1.0)
        assert row.n_persistence == 4

    def test_bss_undefined_when_persistence_zero(self) -> None:
        """Persistence Brier = 0 (perfect baseline) → BSS undefined → None."""
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=1),
            _pair(predicted=0.5, outcome=0, persistence=0),
        ]
        rows = _per_city_var_brier(pairs)
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        row = by_kind[("MIA", "high", "isotonic")]
        assert row.persistence_brier == pytest.approx(0.0)
        assert row.bss_vs_persistence is None

    def test_n_persistence_can_be_less_than_n(self) -> None:
        """Some pairs may lack D-1 data; n_persistence counts only those
        with persistence_yes != None."""
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=1),
            _pair(predicted=0.5, outcome=0, persistence=None),
            _pair(predicted=0.5, outcome=1, persistence=0),
        ]
        rows = _per_city_var_brier(pairs)
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        row = by_kind[("MIA", "high", "isotonic")]
        assert row.n == 3
        assert row.n_persistence == 2

    def test_no_persistence_data_yields_none_fields(self) -> None:
        """Cell with no pairs carrying persistence_yes still emits the row;
        new fields are None/0."""
        pairs = [
            _pair(predicted=0.5, outcome=1, persistence=None),
            _pair(predicted=0.5, outcome=0, persistence=None),
        ]
        rows = _per_city_var_brier(pairs)
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        row = by_kind[("MIA", "high", "isotonic")]
        assert row.n_persistence == 0
        assert row.persistence_brier is None
        assert row.bss_vs_persistence is None

    def test_bss_uses_matched_subset_not_full_n(self) -> None:
        """BSS must use the SAME subset of pairs for numerator and
        denominator. Today's bug: brier averages over n=4, persistence_brier
        averages over n_persistence=2 — the ratio is not a proper BSS.

        Setup: 2 hard pairs without D-1 data where the model is wrong, and
        2 easy pairs with D-1 data where both model and persistence are right.
        On the matched subset (the 2 easy pairs), live Brier = 0 and
        persistence Brier = 0, so BSS is undefined (None). The buggy
        formulation would compute brier=0.5 (over all 4) divided by
        persistence_brier=0 (over the 2 easy) and return None for the wrong
        reason — same outcome by accident here, but the next test pins the
        case where the bias swings the BSS sign.
        """
        pairs = [
            _pair(predicted=1.0, outcome=0, persistence=None),  # hard, no D-1
            _pair(predicted=0.0, outcome=1, persistence=None),  # hard, no D-1
            _pair(predicted=1.0, outcome=1, persistence=1),  # easy
            _pair(predicted=0.0, outcome=0, persistence=0),  # easy
        ]
        rows = _per_city_var_brier(pairs)
        row = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}[
            ("MIA", "high", "isotonic")
        ]
        # Full-sample brier survives unchanged for backwards-compat.
        assert row.brier == pytest.approx(0.5)
        assert row.n == 4
        # Persistence subset has 2 pairs; both are easy → persistence_brier = 0.
        assert row.n_persistence == 2
        assert row.persistence_brier == pytest.approx(0.0)
        # BSS undefined because matched-subset persistence Brier is 0.
        assert row.bss_vs_persistence is None

    def test_bss_matched_subset_protects_against_selection_bias(self) -> None:
        """Concrete selection-bias scenario: model is GOOD on the matched
        subset (where D-1 is available) but bad overall. Buggy formula
        would mix full-n model error with subset persistence error and
        produce a misleading BSS. Matched-subset BSS correctly reads as
        strongly positive.
        """
        pairs = [
            # 4 hard pairs without D-1 — model wrong, doesn't enter BSS
            _pair(predicted=0.9, outcome=0, persistence=None),
            _pair(predicted=0.9, outcome=0, persistence=None),
            _pair(predicted=0.9, outcome=0, persistence=None),
            _pair(predicted=0.9, outcome=0, persistence=None),
            # 2 easy pairs WITH D-1 — model right; persistence wrong
            _pair(predicted=0.0, outcome=0, persistence=1),
            _pair(predicted=1.0, outcome=1, persistence=0),
        ]
        rows = _per_city_var_brier(pairs)
        row = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}[
            ("MIA", "high", "isotonic")
        ]
        # Buggy formula: brier on n=6 ≈ 0.54, persistence_brier on n=2 = 1.0,
        #   BSS = 1 - 0.54/1.0 ≈ 0.46 — wrong, contaminated by hard pairs.
        # Correct formula: live brier on matched subset = 0, persistence
        #   brier on matched subset = 1, BSS = 1.0.
        assert row.bss_vs_persistence == pytest.approx(1.0)

    def test_aggregate_all_row_carries_persistence_too(self) -> None:
        """Backwards-compat: the aggregate 'all' row also gets persistence
        fields populated."""
        pairs = [
            _pair(predicted=1.0, outcome=1, persistence=0, kind="isotonic"),
            _pair(predicted=1.0, outcome=1, persistence=0, kind="emos_raw"),
        ]
        rows = _per_city_var_brier(pairs)
        by_kind = {(r.city, r.daily_variable, r.live_calibrator_kind): r for r in rows}
        agg = by_kind[("MIA", "high", "all")]
        assert agg.n_persistence == 2
        # Both pairs: live brier 0, persistence brier 1 → BSS 1
        assert agg.persistence_brier == pytest.approx(1.0)
        assert agg.bss_vs_persistence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# D-1 lookup in pair constructors
# ---------------------------------------------------------------------------


class TestRecordToPairNwsPersistence:
    def test_d1_present_sets_persistence_yes(self) -> None:
        """D-1 settlement with high=82 and a 'greater than 80' bracket →
        persistence_yes = 1."""
        obs = date(2026, 5, 14)
        d1 = obs - timedelta(days=1)
        settlements = {
            ("NYC", obs): _settlement(city="NYC", obs=obs, high=83.0),
            ("NYC", d1): _settlement(city="NYC", obs=d1, high=82.0),
        }
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="greater", floor=80.0,
        )
        pair = _record_to_pair(rec, settlements)
        assert pair is not None
        assert pair.persistence_yes == 1

    def test_d1_below_floor_persistence_no(self) -> None:
        obs = date(2026, 5, 14)
        d1 = obs - timedelta(days=1)
        settlements = {
            ("NYC", obs): _settlement(city="NYC", obs=obs, high=83.0),
            ("NYC", d1): _settlement(city="NYC", obs=d1, high=75.0),
        }
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="greater", floor=80.0,
        )
        pair = _record_to_pair(rec, settlements)
        assert pair is not None
        assert pair.persistence_yes == 0

    def test_d1_missing_settlement_yields_none(self) -> None:
        obs = date(2026, 5, 14)
        settlements = {
            ("NYC", obs): _settlement(city="NYC", obs=obs, high=83.0),
            # No D-1 row
        }
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="greater", floor=80.0,
        )
        pair = _record_to_pair(rec, settlements)
        assert pair is not None
        assert pair.persistence_yes is None

    def test_d1_missing_variable_yields_none(self) -> None:
        """D-1 row exists but high_f is None for a 'high' market → None."""
        obs = date(2026, 5, 14)
        d1 = obs - timedelta(days=1)
        settlements = {
            ("NYC", obs): _settlement(city="NYC", obs=obs, high=83.0),
            ("NYC", d1): _settlement(city="NYC", obs=d1, high=None, low=60.0),
        }
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="greater", floor=80.0,
        )
        pair = _record_to_pair(rec, settlements)
        assert pair is not None
        assert pair.persistence_yes is None

    def test_d1_between_bracket(self) -> None:
        """Between bracket [70, 79] with D-1 high=75 → persistence_yes = 1."""
        obs = date(2026, 5, 14)
        d1 = obs - timedelta(days=1)
        settlements = {
            ("NYC", obs): _settlement(city="NYC", obs=obs, high=78.0),
            ("NYC", d1): _settlement(city="NYC", obs=d1, high=75.0),
        }
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="between", floor=70.0, cap=79.0,
        )
        pair = _record_to_pair(rec, settlements)
        assert pair is not None
        assert pair.persistence_yes == 1


class TestLoadSettlementsBoundedWindow:
    def _write(self, root: Path, obs: date, *, high: float = 80.0) -> None:
        d = root / obs.isoformat() / "2000"
        d.mkdir(parents=True, exist_ok=True)
        (d / "settlements.json").write_text(
            json.dumps([
                {
                    "city": "NYC",
                    "observation_date": obs.isoformat(),
                    "high_f": high,
                    "low_f": 60.0,
                    "is_preliminary": False,
                    "issued_at": "2026-05-01T09:00:00Z",
                    "product_id": "CLI",
                }
            ])
        )

    def test_default_no_window_loads_all(self, tmp_path: Path) -> None:
        old = date(2026, 1, 1)
        recent = date(2026, 5, 20)
        self._write(tmp_path, old)
        self._write(tmp_path, recent)
        table = _load_settlements(tmp_path)
        assert ("NYC", old) in table
        assert ("NYC", recent) in table

    def test_since_skips_older_directories(self, tmp_path: Path) -> None:
        old = date(2026, 1, 1)
        recent = date(2026, 5, 20)
        self._write(tmp_path, old)
        self._write(tmp_path, recent)
        table = _load_settlements(tmp_path, since=date(2026, 4, 1))
        assert ("NYC", old) not in table
        assert ("NYC", recent) in table


class TestRecordToPairKalshiPersistence:
    def test_d1_present_sets_persistence_yes(self) -> None:
        """Kalshi path: outcome comes from Kalshi map; persistence still
        derives from D-1 NWS settlement + today's bracket bounds."""
        obs = date(2026, 5, 14)
        d1 = obs - timedelta(days=1)
        settlements = {
            ("NYC", d1): _settlement(city="NYC", obs=d1, high=82.0),
        }
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="greater", floor=80.0,
        )
        outcomes = {rec.market_ticker: 1}
        pair = _record_to_pair_kalshi(rec, outcomes, settlements)
        assert pair is not None
        assert pair.outcome_yes == 1
        assert pair.persistence_yes == 1

    def test_missing_settlements_dict_still_yields_pair(self) -> None:
        """Empty settlements dict (fresh deploy, no NWS data yet) — pair
        still constructed via Kalshi outcome; persistence_yes is None."""
        obs = date(2026, 5, 14)
        rec = _pred_record(
            city="NYC", daily_variable="high", obs_date=obs,
            strike_type="greater", floor=80.0,
        )
        outcomes = {rec.market_ticker: 1}
        pair = _record_to_pair_kalshi(rec, outcomes, {})
        assert pair is not None
        assert pair.outcome_yes == 1
        assert pair.persistence_yes is None
