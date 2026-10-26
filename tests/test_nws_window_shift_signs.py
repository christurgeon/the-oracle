"""Regression test pinning NWS-rounding shift signs (bracket-window follow-up #4).

After the 2026-05-21 incident, the two shift sites must
preserve their signs: ``_apply_nws_window`` (`python/data/weather.py`) and
``_nws_shifted_emos_bounds`` (`python/strategies/weather_evaluation.py``)
both shift by ``NWS_ROUND_HALF_F = 0.5`` so BETWEEN expands to
``[floor − 0.5, cap + 0.5]`` and GREATER/LESS contract symmetrically.

The test has three layers and a parametric sanity sweep:

* **Layer A** — pure-function unit pinning of ``_nws_shifted_emos_bounds``.
* **Layer B** — unit pinning of ``_apply_nws_window`` via real
  ``EnsembleForecast`` instances at boundary positions.
* **Layer C** — empirical-decode probe over ``data/shadow_settlements.full.json``
  joined to ``data/snapshots/*/.../settlements.json``: build a single-member
  forecast at the observed NWS integer, call ``_apply_nws_window``, assert
  the returned 0/1 matches ``market_result``. A flipped sign in production
  produces systematic 0↔1 disagreement here.
* **Parametric sweep** — re-run a fixed subset under
  ``monkeypatch(NWS_ROUND_HALF_F, candidate ∈ {-0.5, 0.0, 0.5})``; only the
  production value passes every assertion.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from python.common import NWS_ROUND_HALF_F, Direction
from python.data.weather import EnsembleForecast, _apply_nws_window
from python.strategies.weather_evaluation import _nws_shifted_emos_bounds

REPO_ROOT = Path(__file__).resolve().parent.parent
SHADOW_SETTLEMENTS_PATH = REPO_ROOT / "data" / "shadow_settlements.full.json"
SNAPSHOTS_ROOT = REPO_ROOT / "data" / "snapshots"

# Historical 78/78 empirical-decode baseline that motivated the 2026-05-21
# rounding fix. The corpus has grown; we keep ≥78 as the floor.
EMPIRICAL_MIN_EXAMINED = 78


# ── Layer A — `_nws_shifted_emos_bounds` unit pinning ──────────────────────


def test_nws_shifted_bounds_between_expands() -> None:
    assert _nws_shifted_emos_bounds("between", 57.0, 58.0) == (56.5, 58.5)


def test_nws_shifted_bounds_greater_contracts_floor() -> None:
    assert _nws_shifted_emos_bounds("greater", 70.0, None) == (70.5, None)


def test_nws_shifted_bounds_less_contracts_cap() -> None:
    assert _nws_shifted_emos_bounds("less", None, 72.0) == (None, 71.5)


def test_nws_shifted_bounds_between_requires_both_bounds() -> None:
    with pytest.raises(ValueError, match="between"):
        _nws_shifted_emos_bounds("between", 57.0, None)
    with pytest.raises(ValueError, match="between"):
        _nws_shifted_emos_bounds("between", None, 58.0)


def test_nws_shifted_bounds_greater_requires_floor() -> None:
    with pytest.raises(ValueError, match="greater"):
        _nws_shifted_emos_bounds("greater", None, 70.0)


def test_nws_shifted_bounds_less_requires_cap() -> None:
    with pytest.raises(ValueError, match="less"):
        _nws_shifted_emos_bounds("less", 72.0, None)


def test_nws_shifted_bounds_rejects_unknown_strike_type() -> None:
    with pytest.raises(ValueError, match="unknown strike_type"):
        _nws_shifted_emos_bounds("equal", 70.0, 72.0)


def test_nws_round_half_f_constant_pinned() -> None:
    """One-line pin so a typo in ``python/common.py:20`` fails one assertion
    rather than rippling into ambiguous downstream failures."""
    assert NWS_ROUND_HALF_F == 0.5


# ── Layer B — `_apply_nws_window` via real EnsembleForecast ─────────────────


def _single_member(t: float) -> EnsembleForecast:
    """One-member forecast at temperature ``t``. ``city``/``forecast_date`` are
    immaterial to the window math; we set them to fixed values."""
    return EnsembleForecast(
        city="NYC",
        forecast_date=date(2026, 5, 1),
        temperatures=[t],
    )


@pytest.mark.parametrize(
    "t,expected",
    [
        # BETWEEN floor=70 cap=72 → expanded window [69.5, 72.5] (inclusive).
        (69.5, 1.0),  # left boundary (inclusive after expand)
        (72.5, 1.0),  # right boundary (inclusive after expand)
        (69.4, 0.0),  # just left of expanded floor
        (72.6, 0.0),  # just right of expanded cap
        (71.0, 1.0),  # interior
    ],
)
def test_apply_window_between_boundaries(t: float, expected: float) -> None:
    assert _apply_nws_window(_single_member(t), Direction.BETWEEN, 70.0, 72.0) == expected


@pytest.mark.parametrize(
    "t,expected",
    [
        # ABOVE floor=70 → contracted threshold floor + 0.5 = 70.5 (inclusive).
        (70.5, 1.0),  # boundary (inclusive after contract)
        (70.4, 0.0),  # just below contracted threshold
        (75.0, 1.0),  # well inside
        (70.0, 0.0),  # at unshifted floor — below contracted threshold
    ],
)
def test_apply_window_above_boundaries(t: float, expected: float) -> None:
    assert _apply_nws_window(_single_member(t), Direction.ABOVE, 70.0, None) == expected


@pytest.mark.parametrize(
    "t,expected",
    [
        # BELOW cap=72 → contracted threshold cap − 0.5 = 71.5 (strict <).
        (71.5, 0.0),  # boundary — strict less-than excludes
        (71.4, 1.0),  # just below threshold
        (60.0, 1.0),  # well inside
        (72.0, 0.0),  # at unshifted cap — above contracted threshold
    ],
)
def test_apply_window_below_boundaries(t: float, expected: float) -> None:
    assert _apply_nws_window(_single_member(t), Direction.BELOW, None, 72.0) == expected


def test_apply_window_above_requires_floor() -> None:
    with pytest.raises(ValueError, match="floor_strike required for ABOVE"):
        _apply_nws_window(_single_member(70.0), Direction.ABOVE, None, 72.0)


def test_apply_window_below_requires_cap() -> None:
    with pytest.raises(ValueError, match="cap_strike required for BELOW"):
        _apply_nws_window(_single_member(70.0), Direction.BELOW, 70.0, None)


def test_apply_window_between_requires_both_bounds() -> None:
    with pytest.raises(ValueError, match="BETWEEN"):
        _apply_nws_window(_single_member(70.0), Direction.BETWEEN, 70.0, None)
    with pytest.raises(ValueError, match="BETWEEN"):
        _apply_nws_window(_single_member(70.0), Direction.BETWEEN, None, 72.0)


# ── Layer C — empirical-decode corpus probe ─────────────────────────────────


def _build_settlement_index() -> dict[tuple[str, date], dict[str, Any]]:
    """Index ``(city, observation_date) → settlement record`` from every
    ``data/snapshots/*/.../settlements.json``. Among duplicates, prefer
    non-preliminary; among same-prelim-status, prefer latest ``snapshot_ts``.

    Handles both the older flat layout (``data/snapshots/<date>/settlements.json``)
    and the post-2026-05-20 HHMM layout
    (``data/snapshots/<date>/<HHMM>/settlements.json``).
    """
    index: dict[tuple[str, date], dict[str, Any]] = {}
    if not SNAPSHOTS_ROOT.exists():
        return index
    for path in SNAPSHOTS_ROOT.rglob("settlements.json"):
        try:
            records = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            city = rec.get("city")
            obs_str = rec.get("observation_date")
            if not isinstance(city, str) or not isinstance(obs_str, str):
                continue
            try:
                obs_date = date.fromisoformat(obs_str)
            except ValueError:
                continue
            key = (city, obs_date)
            existing = index.get(key)
            if existing is None:
                index[key] = rec
                continue
            existing_prelim = bool(existing.get("is_preliminary", True))
            new_prelim = bool(rec.get("is_preliminary", True))
            if existing_prelim and not new_prelim:
                index[key] = rec
            elif existing_prelim == new_prelim:
                if rec.get("snapshot_ts", 0) > existing.get("snapshot_ts", 0):
                    index[key] = rec
    return index


_DIRECTION_FROM_STRIKE_TYPE: dict[str, Direction] = {
    "between": Direction.BETWEEN,
    "greater": Direction.ABOVE,
    "less": Direction.BELOW,
}


def _market_result_to_expected_prob(market_result: str) -> float | None:
    """Map the Kalshi market resolution string to the expected 0/1 probability.

    Returns ``None`` for resolutions we don't probe (e.g., ``"void"``).
    """
    if market_result == "yes":
        return 1.0
    if market_result == "no":
        return 0.0
    return None


def test_empirical_decode_matches_market_resolution() -> None:
    """For every settled shadow trade with a final NWS observation, the
    production ``_apply_nws_window`` must return 1.0 iff the market resolved
    YES. A flipped shift sign produces systematic 0↔1 mismatches here.

    Skips records where:

    * The snapshot file for the forecast date is missing.
    * Only a preliminary NWS settlement is available (Hypothesis E
      prelim/final disagreements would create false sign-mismatch noise).
    * The applicable ``high_f`` / ``low_f`` is ``None`` (data quality).
    * The strike geometry is missing the required bound for its direction.

    Counts skipped records and asserts ≥78 non-skipped examined — the
    historical baseline that originally motivated the 2026-05-21 fix.
    """
    if not SHADOW_SETTLEMENTS_PATH.exists():
        pytest.skip(f"missing {SHADOW_SETTLEMENTS_PATH}")
    settlement_index = _build_settlement_index()
    records = json.loads(SHADOW_SETTLEMENTS_PATH.read_text())

    violations: list[str] = []
    examined = 0
    skipped_no_snapshot = 0
    skipped_preliminary_only = 0
    skipped_data_quality = 0

    for rec in records:
        if not rec.get("settled"):
            continue
        meta = rec.get("metadata", {})
        strike_type = rec.get("strike_type") or meta.get("strike_type")
        if strike_type not in _DIRECTION_FROM_STRIKE_TYPE:
            continue
        city = meta.get("city")
        market_type = meta.get("market_type")  # "high" or "low"
        forecast_date_str = meta.get("forecast_date")
        floor_strike = rec.get("floor_strike", meta.get("floor_strike"))
        cap_strike = rec.get("cap_strike", meta.get("cap_strike"))
        market_result = rec.get("market_result")
        expected = _market_result_to_expected_prob(str(market_result))
        if expected is None:
            continue
        if not all(
            [
                isinstance(city, str),
                market_type in {"high", "low"},
                isinstance(forecast_date_str, str),
            ]
        ):
            skipped_data_quality += 1
            continue
        try:
            forecast_date = date.fromisoformat(str(forecast_date_str))
        except ValueError:
            skipped_data_quality += 1
            continue
        settlement = settlement_index.get((str(city), forecast_date))
        if settlement is None:
            skipped_no_snapshot += 1
            continue
        if settlement.get("is_preliminary"):
            skipped_preliminary_only += 1
            continue
        nws_int = settlement.get("high_f") if market_type == "high" else settlement.get("low_f")
        if nws_int is None:
            skipped_data_quality += 1
            continue
        forecast = EnsembleForecast(
            city=str(city),
            forecast_date=forecast_date,
            temperatures=[float(nws_int)],
        )
        direction = _DIRECTION_FROM_STRIKE_TYPE[strike_type]
        try:
            prob = _apply_nws_window(
                forecast,
                direction,
                float(floor_strike) if floor_strike is not None else None,
                float(cap_strike) if cap_strike is not None else None,
            )
        except ValueError:
            # Missing floor/cap for the strike type — data quality, not a
            # sign-flip test target.
            skipped_data_quality += 1
            continue
        if prob != expected:
            violations.append(
                f"{rec.get('order_id')}: strike_type={strike_type} "
                f"floor={floor_strike} cap={cap_strike} NWS_int={nws_int} "
                f"market_result={market_result} prob={prob} (expected {expected})"
            )
        examined += 1

    assert not violations, (
        f"{len(violations)} sign-mismatch violations (showing first 10):\n"
        + "\n".join(violations[:10])
    )
    assert examined >= EMPIRICAL_MIN_EXAMINED, (
        f"examined only {examined} settled non-preliminary records; "
        f"need ≥{EMPIRICAL_MIN_EXAMINED}. "
        f"skipped_no_snapshot={skipped_no_snapshot} "
        f"skipped_preliminary_only={skipped_preliminary_only} "
        f"skipped_data_quality={skipped_data_quality}"
    )


# ── Parametric sanity sweep ────────────────────────────────────────────────


@pytest.mark.parametrize("candidate_shift", [-0.5, 0.0, 0.5])
def test_only_production_shift_passes_all_layers(
    candidate_shift: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-run a minimal subset of Layers A + B under three candidate shift
    values. Only the production value (0.5) makes every subset assertion
    hold. Locks in both sign and magnitude as a single-valued constraint.

    ``from X import Y`` creates a local binding, so monkeypatch must hit
    each importer module's reference (``python.data.weather`` and
    ``python.strategies.weather_evaluation``) in addition to the source.
    """
    monkeypatch.setattr("python.common.NWS_ROUND_HALF_F", candidate_shift)
    monkeypatch.setattr("python.data.weather.NWS_ROUND_HALF_F", candidate_shift)
    monkeypatch.setattr(
        "python.strategies.weather_evaluation.NWS_ROUND_HALF_F", candidate_shift
    )

    # Layer A subset: between-bounds shift.
    bounds = _nws_shifted_emos_bounds("between", 70.0, 72.0)
    # Layer B subset: BETWEEN at t=72.4 distinguishes positive vs zero/negative.
    prob_between = _apply_nws_window(_single_member(72.4), Direction.BETWEEN, 70.0, 72.0)
    # Layer B subset: ABOVE at t=70.4 distinguishes positive vs zero/negative.
    prob_above = _apply_nws_window(_single_member(70.4), Direction.ABOVE, 70.0, None)

    if candidate_shift == 0.5:
        # Production values — every assertion holds.
        assert bounds == (69.5, 72.5)
        assert prob_between == 1.0
        assert prob_above == 0.0
    else:
        # No other candidate may pass all three production assertions.
        production_holds = (
            bounds == (69.5, 72.5) and prob_between == 1.0 and prob_above == 0.0
        )
        assert not production_holds, (
            f"shift={candidate_shift} accidentally passes all production assertions"
        )
