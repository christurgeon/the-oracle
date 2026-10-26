"""Production-stack re-verification CLI for IntradayEmosHighStrategy.

Purpose
-------
Replays the **strategy's own evaluate path** (``IntradayEmosHighStrategy``)
on snapshot rows pulled from ``data/snapshots/<date>/<run_id>/`` and
diffs the resulting per-trade decision fields against the v6 audit's
fixture CSV (``workstream1_gate_audit_v6_trades.csv``).

Fields diffed (spec §4.5 lines 757-760):

  * ``model_probability``           (vs. CSV ``emos_prob_walkforward``)
  * ``market_price`` (yes_mid)      (vs. CSV ``yes_mid``)
  * ``side``                        (vs. CSV ``side``)
  * ``effective_edge``              (vs. CSV ``effective_edge``)
  * ``gate_pass / gate_reject``     (vs. v6 row existence — see below)
  * ``strike_abs_z``                (vs. CSV ``strike_abs_z``)

Plus defensive checks on the underlying EMOS posterior + strike-distance
inputs:

  * ``emos_prob_raw``               (vs. CSV ``emos_prob_walkforward``)
  * ``pooled_median``               (vs. CSV ``prod_median_f``)
  * ``ensemble_std_pooled``         (vs. CSV ``prod_pstdev_f``)

Gate-pass / gate-reject reason
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The v6 CSV contains only PASSED rows (rows that survived every gate).
For each v6 row we recompute the strategy's gate stack via
``IntradayEmosHighStrategy._build_eval_config()`` + ``evaluate_market_pure``;
a ``None`` ``_MarketCandidate`` means the strategy would have REJECTED
that row, which is a structural diff — surfaced as
``gate_pass: audit=PASSED, replay=<REJECTED|<gap reason>>``.

Production-faithful gate-input SHA256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Per spec §4.5 lines 764-767 the script asserts that the verifier's
gate-input layer hashes to the pinned v6 value::

    prod_strike_inputs_sha256 =
        a8f7ec115e8413bd191240ab67eebda574d13759c0c8303710be202a74c20683

The hash is computed by reading the v6 pinned strike-inputs fixture
(``data/research/workstream1_gate_audit_v6_prod_strike_inputs.json``)
as bytes — the fixture file IS the authoritative pin. A SHA mismatch
exits 1: the fixture file itself was mutated and the v6 audit's
bit-identical claim no longer applies. Snapshot-data drift inside
the v6 keyset is still caught — via the per-row replay diff on
``model_probability`` / ``emos_prob_raw`` / ``pooled_median``.

JSON sink
~~~~~~~~~
Every run emits ``data/research/intraday_replay_diff_<run_date>.json``
(or ``--output-json PATH`` override). Schema::

    {
      "run_date": "YYYY-MM-DD",
      "n_trades": int,                 # v6 audit fixture rows in the window
      "n_diffs": int,                  # rows with at least one field-diff
      "n_replay_errors": int,
      "gate_input_sha256": str,        # the verifier's computed SHA
      "expected_sha256": str,          # pinned v6 SHA
      "sha_match": bool,
      "exit_code": int,                # 0 / 1 / 2
      "diffs": [
        {
          "snapshot_day": "YYYY-MM-DD",
          "cell": "MIA/high" | "LAX/high",
          "ticker": str,
          "field": str,                # "model_probability" / "side" / ...
          "audit": str | float | None,
          "replay": str | float | None,
          "abs_delta": float | None,   # None for non-numeric fields
        },
        ...
      ],
      "replay_errors": [
        {"snapshot_day": str, "cell": str, "ticker": str, "error": str},
        ...
      ],
    }

CLI surface (plan M7 line 949)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--start-date YYYY-MM-DD --end-date YYYY-MM-DD --cell CELL``

  * ``--cell`` defaults to ``"all"`` (both v6-bound cells:
    ``MIA/high``, ``LAX/high``).
  * ``--tolerance`` (1e-9 default), ``--strict`` (documented no-op),
    ``--output-json PATH`` override.
  * v6 fixture paths are loaded from canonical locations under
    ``data/research/`` and are NOT exposed as CLI flags.

Spec §11.2 v1.3 IMPORTANT #1 honest disclosure: NO automatic Telegram
CRITICAL fires from this script. The alert path is the exit code, which
the operator monitors via shell ($? != 0).

Operator workflow (one-off, manual)::

    uv run python scripts/verify_intraday_replay.py \\
        --start-date 2026-04-05 --end-date 2026-05-09
    echo $?   # 0 = match; 1 = diff detected; 2 = fixture/env error

Contract sizing intentionally differs between strategy and audit (spec
§6.4 NBBO_SIZE_CAP=50 vs Kelly + depth-check): we do NOT diff
``suggested_contracts`` / ``suggested_dollars``. P&L is recomputed at
the audit's NBBO_SIZE_CAP convention so the math is reproducible from
the CSV alone.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Add project root to path so ``python.*`` imports resolve when invoked
# as a CLI from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from python.analytics.audit_log import RejectionReason, ScanFunnel  # noqa: E402
from python.common import DailyVariable, Direction, ExecutionMode, Side  # noqa: E402
from python.data.bias_correction import BiasTable  # noqa: E402
from python.data.emos import (  # noqa: E402
    EmosCoefficients,
    EmosTable,
    EmosTableHolder,
)
from python.data.weather import EnsembleForecast  # noqa: E402
from python.settings import Settings  # noqa: E402
from python.strategies.intraday_emos_high import IntradayEmosHighStrategy  # noqa: E402
from python.strategies.weather_evaluation import (  # noqa: E402
    evaluate_market_pure,
    postprocess_forecasts,
)
from python.strategies.weather_parsing import (  # noqa: E402
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
)
from scripts._calibration_common import load_snapshot_data  # noqa: E402

# v6 audit constants (mirrors audit_same_day_gate_2026-05-14_v6_prod_gate_math.py).
_NBBO_SIZE_CAP_AUDIT = 50
_BIAS_SHRINKAGE = 0.3
_BIAS_MAX_CORRECTION_F = 5.0
_ENSEMBLE_WEIGHTS: dict[str, float] = {"gfs_seamless": 0.55, "ecmwf_ifs025": 0.45}

# Default tolerance — IEEE-754 doubles deterministic at 1e-12; 1e-9 leaves
# headroom for scipy.norm.cdf least-significant-bit drift across versions.
_DEFAULT_TOLERANCE = 1e-9

# Pinned v6 gate-input SHA256.
EXPECTED_PROD_STRIKE_INPUTS_SHA256 = (
    "6e70b22fbee9570a47c9bf3859f16546130b53a2f2cd72098f2c8400466ccdfb"
)

# Canonical fixture paths (NOT CLI-exposed — these are the v6 anchors).
# The v6 fixture files are unpublished research data, not in this repository.
_DEFAULT_TRADES_CSV = Path("data/research/workstream1_gate_audit_v6_trades.csv")
_DEFAULT_COEFFS_JSON = Path("data/research/workstream1_gate_audit_v6_walkforward_coeffs.json")
_DEFAULT_STRIKE_INPUTS_JSON = Path(
    "data/research/workstream1_gate_audit_v6_prod_strike_inputs.json"
)
_DEFAULT_BIAS_PATH = Path("data/gfs_bias/bias_correction.json")
_DEFAULT_SNAPSHOTS_DIR = Path("data/snapshots")
_DEFAULT_OUTPUT_DIR = Path("data/research")

_V6_CELL_FILTER = frozenset({"MIA/high", "LAX/high"})

_STRIKE_LABEL_TO_DIRECTION = {
    "greater": Direction.ABOVE,
    "less": Direction.BELOW,
    "between": Direction.BETWEEN,
}


# ---------------------------------------------------------------------------
# Frozen-clock context manager
# ---------------------------------------------------------------------------
#
# evaluate_market_pure computes ``utc_today = datetime.now(UTC).date()`` to
# derive the EMOS dispatcher's lead_days check (lead < 0 → "lead_negative"
# warning + emos_prob_raw stays None → no swap). Replaying historical rows
# at HEAD-clock would always trip the negative-lead branch and the EMOS
# posterior would never compute — so the replay would diverge from v6 not
# because the strategy is broken but because clock-tied gates are tripping.
#
# Solution: monkey-patch ``datetime`` in the relevant production modules to
# return ``forecast_date + 1 day`` as ``now()`` so the lead computation
# resolves to ``lead = (forecast_date - snapshot_day_utc).days`` matching
# what the v6 audit assumed. Identical pattern to ``freezegun`` but with
# zero dependencies — touches only the in-module name bindings.


class _FrozenDateTime(datetime):
    """A ``datetime`` subclass whose ``now()`` returns a pinned value."""

    _frozen_now: datetime | None = None

    @classmethod
    def now(cls, tz: timezone | None = None) -> datetime:  # type: ignore[override]
        frozen = cls._frozen_now
        if frozen is None:
            return datetime.now(tz)
        if tz is not None:
            return frozen.astimezone(tz)
        return frozen.replace(tzinfo=None)


@contextlib.contextmanager
def _freeze_clock(snapshot_day: date) -> Iterator[None]:
    """Freeze ``datetime.now()`` to the snapshot_day at noon UTC for the
    duration of the ``with`` block.

    Noon UTC is a deliberate choice — it's late enough that the ET-derived
    et_today() returns the same date as snapshot_day (ET = UTC-5/-4),
    matching the v6 audit's row-emission assumption (rows are keyed on
    snapshot_day UTC).

    Patches three sites:
      * ``python.strategies.weather_evaluation.datetime``
      * ``python.strategies.intraday_emos_high.datetime``
      * ``python.common.datetime`` (for ``et_today()``)
    """
    # weather_evaluation imports ``datetime`` directly as a name binding.
    # We swap the module attribute, not the class — so other code that
    # imports ``datetime`` from ``datetime`` is unaffected.
    import python.common as _common_mod
    import python.strategies.intraday_emos_high as _intraday_mod
    import python.strategies.weather_evaluation as _weather_eval_mod

    pinned = datetime(
        snapshot_day.year,
        snapshot_day.month,
        snapshot_day.day,
        12,
        0,
        0,
        tzinfo=UTC,
    )
    _FrozenDateTime._frozen_now = pinned

    originals: list[tuple[Any, str, Any]] = []
    for module in (_weather_eval_mod, _intraday_mod, _common_mod):
        if hasattr(module, "datetime"):
            originals.append((module, "datetime", module.datetime))
            module.datetime = _FrozenDateTime  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module, name, prev in originals:
            setattr(module, name, prev)
        _FrozenDateTime._frozen_now = None


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------


def _load_v6_trades(path: Path) -> list[dict[str, Any]]:
    """Load the v6 trades CSV into a list of dicts (raw string values)."""
    if not path.exists():
        raise FileNotFoundError(f"v6 trades CSV not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_walkforward_coeffs_by_date(
    path: Path,
) -> dict[str, dict[str, EmosCoefficients]]:
    """Parse walk-forward coefficients into a ``snapshot_day -> cell -> coeffs``
    nested lookup.

    The fixture's structure is::

        {"by_date": {"2026-04-05": {"MIA/high": {"fit": {a,b,c,d}, ...}, ...}}}

    Cells whose ``fit`` is null (n_train < min_train_n) are dropped from the
    returned dict — the verifier marks affected v6 rows as missing-fit
    replay-errors if they reappear.
    """
    if not path.exists():
        raise FileNotFoundError(f"v6 walk-forward coeffs JSON not found: {path}")
    payload = json.loads(path.read_text())
    by_date = payload.get("by_date") or {}
    out: dict[str, dict[str, EmosCoefficients]] = {}
    for date_str, cells in by_date.items():
        out[date_str] = {}
        for cell, info in cells.items():
            fit = info.get("fit")
            if fit is None:
                continue
            out[date_str][cell] = EmosCoefficients(
                a=float(fit["a"]),
                b=float(fit["b"]),
                c=float(fit["c"]),
                d=float(fit["d"]),
                n_train=int(info.get("n_train", 0)),
                low_n=False,
                ci_warnings=[],
                leads_pooled=[],
                by_lead=None,
                ens_std_p99=None,
            )
    return out


# ---------------------------------------------------------------------------
# v6 production-faithful strike-distance input lookup
# ---------------------------------------------------------------------------


def compute_gate_input_sha256(
    *,
    strike_inputs_path: Path = _DEFAULT_STRIKE_INPUTS_JSON,
) -> tuple[str, str]:
    """Hash the v6 pinned strike-inputs fixture and return
    ``(sha256_hex, serialized_text)``.

    The fixture file is the authoritative pin — it carries the production-
    faithful ``(city, forecast_date, daily_variable) -> (pooled_median,
    pooled_pstdev)`` table as captured at the v6 freeze. Hashing the
    fixture (not a daily recompute from ``data/snapshots/``) is the only
    stable contract: the snapshot tree mutates continuously (GitHub
    Action re-runs overwrite trailing-edge forecast_dates with newer GFS
    inits; new forecast_dates accrue past the v6 window), so a recompute-
    over-snapshots SHA would false-alarm on every calendar tick.

    Snapshot-data drift inside the v6 keyset is still detected — but via
    the per-row replay diff (``model_probability`` / ``emos_prob_raw`` /
    ``pooled_median``), where a mutated ensemble member surfaces as a
    structural diff and flips the exit code. This separates the two
    concerns: SHA = "fixture file unchanged", per-row diff = "production
    strategy still reproduces v6 outputs on the underlying snapshots".
    """
    # read_bytes() (not read_text().encode()) — text-mode read would apply
    # platform-default line-ending translation; the pinned SHA was computed
    # against the LF bytes on disk.
    raw = strike_inputs_path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), raw.decode()


# ---------------------------------------------------------------------------
# Snapshot iteration + market replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MarketRow:
    """A v6 trade row joined to its source snapshot market dict.

    ``snapshot_dir`` is the directory containing the ``markets.json`` /
    ``weather_forecasts.json`` files for this trade's snapshot_day+run_id.
    """

    audit_row: dict[str, Any]
    snapshot_dir: Path
    market: dict[str, Any]


def _find_snapshot_market(
    snapshots_dir: Path,
    *,
    snapshot_day: str,
    run_id: str,
    ticker: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Locate the raw market dict for a v6 audit row.

    Looks under ``data/snapshots/<snapshot_day>/<run_id>/markets.json`` first
    (the v6 audit's source layout); falls back to the flat
    ``data/snapshots/<snapshot_day>/markets.json`` for legacy snapshots.
    """
    candidates = [
        snapshots_dir / snapshot_day / run_id / "markets.json",
        snapshots_dir / snapshot_day / "markets.json",
    ]
    for mkt_path in candidates:
        if not mkt_path.exists():
            continue
        try:
            rows = json.loads(mkt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            if row.get("ticker") == ticker:
                return mkt_path.parent, row
    return None


def _build_forecast_cache_from_global(
    forecasts_global: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    city: str,
    forecast_date: date,
    daily_variable: DailyVariable,
) -> dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]:
    """Project the global forecasts dict (keyed by string-date + str-var) into
    the per-fkey ``EnsembleForecast`` shape that ``evaluate_market_pure``
    expects, restricted to one ``(city, forecast_date, daily_variable)`` cell.

    The global cache is built once by ``load_snapshot_data`` (mirroring the
    v6 audit's ``_iter_snapshot_dirs`` semantic — last-writer-wins across
    runs) so a market on ``snapshot_day=A/run_id=B`` can use a forecast
    from ``snapshot_day=A/run_id=C`` if C is more recent. Per-row
    isolation is preserved because the returned dict is freshly
    constructed each call.
    """
    if daily_variable == DailyVariable.TEMP_MAX:
        fc_var_str = "temperature_2m_max"
    elif daily_variable == DailyVariable.TEMP_MIN:
        fc_var_str = "temperature_2m_min"
    else:
        return {}
    fc_date_str = forecast_date.isoformat()
    fkey = (city, forecast_date, daily_variable)
    out: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] = {fkey: {}}
    for model_name in _ENSEMBLE_WEIGHTS:
        fc = forecasts_global.get((city, fc_date_str, fc_var_str, model_name))
        if fc is None:
            continue
        members = list(fc.get("temperatures") or fc.get("ensemble_members") or [])
        if not members:
            continue
        out[fkey][model_name] = EnsembleForecast(
            forecast_date=forecast_date,
            city=city,
            temperatures=members,
            model=model_name,
        )
    return out


def _emos_table_for_snapshot(
    snapshot_day: str,
    walkforward: dict[str, dict[str, EmosCoefficients]],
) -> EmosTable:
    """Synthesize an in-memory ``EmosTable`` carrying the v6 walk-forward
    coefficients for ``snapshot_day``.

    Construction notes:
      * ``artifact_hash`` is a stable per-snapshot-day hash so a downstream
        consumer reading ``_MarketCandidate.emos_coeffs_artifact_hash`` can
        distinguish walk-forward fits across days.
      * ``fallback_reason=None`` + non-empty ``cells`` make ``is_loaded``
        return True (the strategy's setup precondition #1 gate).
      * Cells absent from the fixture for this date are omitted; the
        strategy's setup precondition #2 (enabled cells must exist) is
        intentionally bypassed by directly injecting the holder AFTER setup
        when the test fixture lacks one cell (see ``_replay_row`` for the
        missing-fit replay-error path).
    """
    cells = walkforward.get(snapshot_day, {})
    artifact_hash = hashlib.sha256(f"v6-walkforward:{snapshot_day}".encode()).hexdigest()
    return EmosTable(
        cells=dict(cells),
        artifact_hash=artifact_hash,
        artifact_signature=None,
        artifact_loaded_at=datetime.now().isoformat(),
        fit_date=snapshot_day,
        training_window_days=None,
        lead_handling="pooled_across_leads",
        schema_version=1,
        n_skipped=0,
        skipped_cells={},
        fallback_reason=None,
        _path=None,
    )


def _build_scan_candidate(market: dict[str, Any]) -> ScanCandidate | None:
    """Build a ``ScanCandidate`` for the v6-bound markets without going through
    ``parse_weather_market`` (which would re-parse the ticker via series-
    routing — unnecessary since the v6 CSV already pins the parsed fields).

    The intraday strategy is restricted to KXHIGHMIA + KXHIGHLAX; this
    helper reads ``ticker`` + ``strike_type`` + strikes + ``daily_variable``
    + ``city`` directly off the market dict and constructs the
    ``ParsedMarket`` + ``MarketPriceInfo`` pair used by
    ``evaluate_market_pure``.
    """
    ticker = market.get("ticker", "")
    if not ticker:
        return None
    strike_type_str = market.get("strike_type")
    if strike_type_str not in _STRIKE_LABEL_TO_DIRECTION:
        return None
    city = market.get("city")
    daily_var_str = market.get("daily_variable")
    if not city or not daily_var_str:
        return None

    # forecast_date is encoded in the ticker (e.g. KXHIGHMIA-26APR05-B95);
    # we lift it from the snapshot dict's ``observation_date`` first (more
    # robust than re-parsing the ticker), fall back to the embedded date.
    fc_date_str = market.get("observation_date")
    if not fc_date_str:
        from scripts._calibration_common import parse_market_date

        fc_date = parse_market_date(ticker)
        if fc_date is None:
            return None
    else:
        fc_date = date.fromisoformat(fc_date_str)

    daily_variable: DailyVariable
    if daily_var_str == "temperature_2m_max":
        daily_variable = DailyVariable.TEMP_MAX
    elif daily_var_str == "temperature_2m_min":
        daily_variable = DailyVariable.TEMP_MIN
    else:
        return None

    parsed = ParsedMarket(
        ticker=ticker,
        city=city,
        forecast_date=fc_date,
        strike_type=_STRIKE_LABEL_TO_DIRECTION[strike_type_str],
        floor_strike=(
            float(market["floor_strike"]) if market.get("floor_strike") is not None else None
        ),
        cap_strike=(float(market["cap_strike"]) if market.get("cap_strike") is not None else None),
        daily_variable=daily_variable,
    )

    yes_bid = market.get("yes_bid_dollars")
    yes_ask = market.get("yes_ask_dollars")
    yes_bid_f = float(yes_bid) if yes_bid is not None else None
    yes_ask_f = float(yes_ask) if yes_ask is not None else None
    mid = (
        (yes_bid_f + yes_ask_f) / 2.0
        if yes_bid_f is not None and yes_ask_f is not None
        else (yes_bid_f if yes_bid_f is not None else yes_ask_f or 0.0)
    )
    spread = (yes_ask_f - yes_bid_f) if yes_bid_f is not None and yes_ask_f is not None else None
    price_info = MarketPriceInfo(
        midpoint=mid,
        yes_bid=yes_bid_f,
        yes_ask=yes_ask_f,
        spread=spread,
        yes_bid_size=None,
        yes_ask_size=None,
        volume=market.get("volume"),
        open_interest=market.get("open_interest"),
    )

    return ScanCandidate(market=market, parsed=parsed, price_info=price_info)


# ---------------------------------------------------------------------------
# Per-row replay
# ---------------------------------------------------------------------------


# Production-only gate reasons — strategy applies these but v6 audit did
# NOT. A REJECTED-via-these-gates row is INFORMATIONAL, not a structural
# regression. (Spec §6.4 acknowledges sizing differs by design; these
# downstream gates that don't change the EMOS posterior or strike-distance
# decision are in the same equivalence class — they shape the trade-set,
# not the model.)
_PROD_ONLY_REJECTION_REASONS: frozenset[str] = frozenset(
    {
        RejectionReason.YES_FLOOR.value,
        RejectionReason.EDGE_ABOVE_MAX.value,
        RejectionReason.MODEL_MARKET_GAP_INTERIM.value,
        RejectionReason.NO_PRICE.value,
        RejectionReason.KELLY_ZERO.value,
        RejectionReason.KELLY_SIDE_FLIP.value,
        # Added post-v6 (2026-05-19): runtime σ_emos/bracket gate.
        # The v6 audit predates this gate — LAX/high rows that passed v6
        # can fail here (σ_emos > 4°F on high-d walk-forward fits). The
        # gate "shapes the trade-set, not the model" (doesn't alter EMOS
        # posterior or strike-distance inputs), so v6-fixture rows rejected
        # here are informational vs. the v6 baseline.
        RejectionReason.EMOS_SIGMA_INCOMPATIBLE.value,
    }
)


@dataclass(frozen=True)
class ReplayResult:
    """Per-(v6 row) replay outputs covering the spec §4.5 diff field set."""

    cell: str
    snapshot_day: str
    market_ticker: str

    # Outcome of the strategy's evaluate path on this row. ``None`` means
    # the strategy would have REJECTED the row at some gate.
    candidate_passed: bool

    # If candidate_passed is False, the rejection reason as a stable string
    # (matches ``RejectionReason`` value). ``None`` when the rejection
    # happened in a code path that doesn't update the funnel.
    rejection_reason: str | None

    # Spec §4.5 fields (lines 757-760). All ``None`` when
    # ``candidate_passed is False`` AND the field requires a passed
    # candidate to compute (e.g. ``side``, ``effective_edge``).
    model_probability: float | None
    market_price: float | None
    side: str | None
    effective_edge: float | None
    strike_abs_z: float | None

    # Defensive checks (kept from the previous pinned-list verifier so
    # any underlying EMOS posterior / strike-input drift surfaces as a
    # field-level diff before propagating to model_probability).
    emos_prob_raw: float | None
    pooled_median: float | None
    ensemble_std_pooled: float | None

    # P&L convention identical to v6 audit (NBBO_SIZE_CAP, side-denominated
    # fill_ask). Kept as a defensive check on the audit's fee math.
    pnl_ask: float | None

    # Replay-side Kelly contract count. Used by ``_classify_effective_edge``
    # to detect a Kelly-bucket flip (contracts==0 vs >0) caused by an
    # ``effective_edge`` divergence. ``None`` when the strategy rejected
    # before reaching Kelly.
    contracts: int | None


@dataclass(frozen=True)
class FieldDiff:
    field: str
    audit_value: float | str | None
    replay_value: float | str | None
    abs_diff: float | None


@dataclass
class TradeDiff:
    cell: str
    market_ticker: str
    snapshot_day: str
    fields: list[FieldDiff] = field(default_factory=list)
    error: str | None = None


def _parse_optional_float(s: Any) -> float | None:
    """CSV cells are strings; empty -> None. Non-string scalars pass through."""
    if s is None:
        return None
    if isinstance(s, str):
        if s == "":
            return None
        return float(s)
    return float(s)


def _parse_bool(s: str) -> bool:
    if s == "True":
        return True
    if s == "False":
        return False
    raise ValueError(f"expected 'True'/'False' boolean, got {s!r}")


def _realized_pnl_per_contract(outcome_yes: bool, side: str, fill: float) -> float:
    """Per-contract realized P&L; ``fill`` is side-denominated in [0, 1].

    Mirrors ``realized_pnl_per_contract`` in the v6 audit script (line 97).
    """
    win = (side == "YES" and outcome_yes) or (side == "NO" and not outcome_yes)
    return (1.0 - fill) if win else -fill


def _replay_pnl_ask(audit_row: dict[str, Any]) -> float:
    """P&L at the audit's NBBO_SIZE_CAP convention (defensive check)."""
    from python.analytics.fees import order_fee

    fill_ask = float(audit_row["fill_ask"])
    side = audit_row["side"]
    outcome_yes = _parse_bool(audit_row["outcome_yes"])
    order_level_fee = order_fee(fill_ask, contracts=_NBBO_SIZE_CAP_AUDIT, is_maker=False)
    return (
        _NBBO_SIZE_CAP_AUDIT * _realized_pnl_per_contract(outcome_yes, side, fill_ask)
        - order_level_fee
    )


async def replay_row(
    *,
    audit_row: dict[str, Any],
    snapshots_dir: Path,
    forecasts_global: dict[tuple[str, str, str, str], dict[str, Any]],
    walkforward: dict[str, dict[str, EmosCoefficients]],
    bias_table: BiasTable,
    settings: Settings,
) -> ReplayResult:
    """Replay one v6 audit row through ``IntradayEmosHighStrategy``'s evaluate
    path and capture the spec §4.5 diff field set.

    Constructs a real ``IntradayEmosHighStrategy`` with a snapshot-loaded
    ``EmosTableHolder`` (carrying the v6 walk-forward coefficients for this
    row's snapshot_day) and a stub ``WeatherClient`` whose
    ``get_multi_model_forecast`` returns snapshot-derived ``EnsembleForecast``
    objects. ``KalshiClient`` is never called — we operate on a single
    market row, not a full ``scan()`` sweep.

    The strategy's ``_build_eval_config()`` is the load-bearing call site:
    it carries the gate-stack params (``min_distance_from_strike_std=2.0``,
    ``edge_threshold=0.07``, plateau range, etc.) that we want to test for
    regressions.
    """
    cell = audit_row["cell"]
    snapshot_day = audit_row["snapshot_day"]
    run_id = audit_row.get("run_id", "")
    ticker = audit_row["market_ticker"]
    forecast_date_str = audit_row["forecast_date"]

    # Locate the snapshot market dict.
    located = _find_snapshot_market(
        snapshots_dir,
        snapshot_day=snapshot_day,
        run_id=str(run_id),
        ticker=ticker,
    )
    if located is None:
        return _replay_error_result(
            audit_row,
            f"market {ticker} not found under {snapshots_dir}/{snapshot_day}/{run_id}/",
        )
    _snapshot_dir, raw_market = located
    _ = forecast_date_str  # surfaces in audit_row; used only for SHA + diff
    _ = _snapshot_dir  # markets dict already pulled — kept for future debug

    sc = _build_scan_candidate(raw_market)
    if sc is None:
        return _replay_error_result(
            audit_row,
            f"could not parse ScanCandidate for market dict (ticker={ticker})",
        )

    # Build the per-snapshot-day EMOS holder (walk-forward fit). Drives the
    # strategy's per-cell EMOS dispatcher in evaluate_market_pure.
    table = _emos_table_for_snapshot(snapshot_day, walkforward)
    if cell not in table.cells:
        return _replay_error_result(
            audit_row,
            f"walk-forward fit missing for cell={cell} on snapshot_day={snapshot_day}",
        )
    holder = EmosTableHolder(table=table)

    # Build the forecast_cache shape evaluate_market_pure expects, projecting
    # the pre-loaded global forecasts dict (last-writer-wins across runs —
    # mirrors the v6 audit's load_snapshot_data semantic).
    fkey_target = (sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable)
    forecast_cache = _build_forecast_cache_from_global(
        forecasts_global,
        city=sc.parsed.city,
        forecast_date=sc.parsed.forecast_date,
        daily_variable=sc.parsed.daily_variable,
    )
    if not forecast_cache.get(fkey_target):
        return _replay_error_result(
            audit_row,
            f"forecast_cache missing key {fkey_target} (needed for ticker={ticker})",
        )

    # Stub WeatherClient: only ``get_multi_model_forecast`` and
    # ``compute_model_probability`` are reachable from evaluate_market_pure;
    # we inject a deepcopy-able real ``EnsembleForecast`` for the former and
    # the real method body for the latter.
    weather_client = MagicMock()

    async def _stub_get(
        *,
        city: str,
        forecast_date: date,
        daily_variable: DailyVariable,
        models: list[str],
    ) -> dict[str, EnsembleForecast]:
        out: dict[str, EnsembleForecast] = {}
        full = forecast_cache.get((city, forecast_date, daily_variable), {})
        for model_name in models:
            if model_name in full:
                out[model_name] = full[model_name]
        return out

    weather_client.get_multi_model_forecast = _stub_get
    weather_client.compute_model_probability = (
        # Real ensemble-fraction computation; evaluate_market_pure invokes
        # this only for the gfs_uncorrected_prob diagnostic (skipped unless
        # bias correction shift > 0.01°F).
        _ensemble_fraction
    )

    kalshi_client = MagicMock()

    # NOTE on gate alignment: the v6 audit's gate set is
    # ``{excluded_cell_gate, cal_floor, plateau_filter, raw_prob_gate,
    # strike_distance, spread_filter, threshold_raw_prob_clamp,
    # calibrator_disagreement}``. The strategy's evaluate_market_pure
    # additionally applies ``max_edge`` (phantom-quote rejection),
    # ``min_calibrated_prob``, plateau-block, model-market-gap, and a
    # YES-floor on threshold markets below $0.15 mid.
    #
    # For bit-identical replay against the v6 fixture we set these
    # production-only gates to neutral values (None / 1.0) so the only
    # divergence is in gates the audit also applied. Per spec §4.5 the
    # diff target is "the strategy's `evaluate` path on snapshot rows" —
    # specifically the (model_probability, market_price, side,
    # effective_edge, gate_pass/gate_reject, strike_abs_z) tuple — and
    # that tuple is what `evaluate_market_pure` produces. Production-only
    # gate trips would be a verifier false positive, not a strategy
    # regression vs. v6.
    config = {
        "enabled": True,
        "enabled_cells": [cell],
        "edge_threshold": 0.07,
        # max_edge=1.0 disables the phantom-quote rejection (v6 audit
        # does not have this gate). Strategy ships with 0.30 in prod.
        "max_edge": 1.0,
        "min_distance_from_strike_std": 2.0,
        "max_contracts_per_trade": 50,
        "max_trades_per_cell_per_day": 3,
        "emos_hot_reload_pause_seconds": 60,
        "ensemble_models": list(_ENSEMBLE_WEIGHTS.keys()),
        "ensemble_weights": dict(_ENSEMBLE_WEIGHTS),
        "bias_shrinkage": _BIAS_SHRINKAGE,
        "bias_max_correction_f": _BIAS_MAX_CORRECTION_F,
        "spread_inflation_factors": None,
        # cal_floor + plateau gates: audit applied min_calibrated_prob=0.10
        # and plateau_range=(0.649, 0.967) — production strategy keeps the
        # same values. Set them to match the audit so the diff fires on
        # genuine model differences, not gate-config differences.
        "min_calibrated_prob": 0.10,
        "block_raw_prob_range": [0.649, 0.967],
        "eval_audit_log": False,
        "candidate_audit_log": False,
        "scan_funnel_log": False,
        "orderbook_depth_check": False,
    }

    strategy = IntradayEmosHighStrategy(
        config=config,
        kalshi_client=kalshi_client,
        weather_client=weather_client,
        bankroll=10_000.0,
        market_filter={},
        settings=settings,
        emos_holder=holder,
        bias_table=bias_table,
        prediction_log=None,
        notifier=None,
    )

    # Apply postprocess_forecasts (bias correction + median/std snapshot).
    # We do this on a per-row copy of the cache so each replay is independent
    # — postprocess_forecasts mutates its input dict in place.
    cache_copy: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] = {
        fkey_target: dict(forecast_cache[fkey_target])
    }
    raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] = {}
    (
        cache_after,
        bias_corrections,
        raw_ensemble_stds,
        raw_ensemble_medians,
    ) = postprocess_forecasts(
        forecast_cache=cache_copy,
        bias_table=bias_table,
        bias_shrinkage=_BIAS_SHRINKAGE,
        bias_max_correction_f=_BIAS_MAX_CORRECTION_F,
        spread_inflation_overrides=None,
        raw_member_sink=raw_member_sink,
    )

    # Call the strategy's _build_eval_config() — this is the load-bearing
    # composition point: a future change to the strategy's gate-stack
    # parameters (edge_threshold, min_distance_from_strike_std, etc.) will
    # surface here.
    cfg = strategy._build_eval_config()  # type: ignore[reportPrivateUsage]

    # Freeze the clock at noon UTC of snapshot_day so the EMOS dispatcher's
    # ``lead = (forecast_date - utc_today).days`` computation in
    # evaluate_market_pure matches what the v6 audit assumed (rows are
    # keyed on snapshot_day UTC; the audit bypassed the lead check by
    # calling emos_probability directly).
    snapshot_day_d = date.fromisoformat(snapshot_day)
    funnel = ScanFunnel()
    with _freeze_clock(snapshot_day_d):
        candidate = evaluate_market_pure(
            sc,
            cache_after,
            bias_corrections,
            raw_ensemble_stds,
            raw_ensemble_medians,
            cfg,
            calibrator=None,
            shadow_calibrator=None,
            weights=dict(_ENSEMBLE_WEIGHTS),
            weather_client=weather_client,
            bankroll=10_000.0,
            excluded_city_vars=[],
            spread_inflation_overrides=None,
            emos_table=holder.table,
            raw_member_sink=raw_member_sink,
            emos_log_dedup={},
            emos_enabled_cells=frozenset([cell]),
            funnel=funnel,
        )

    pnl_ask = _replay_pnl_ask(audit_row)

    if candidate is None:
        # Identify which gate rejected via the funnel counters. Used to
        # classify the rejection as production-only (informational) vs
        # audit-shared (structural). See _PROD_ONLY_REJECTION_REASONS.
        rejection_reason = _infer_rejection_reason(funnel)
        return ReplayResult(
            cell=cell,
            snapshot_day=snapshot_day,
            market_ticker=ticker,
            candidate_passed=False,
            rejection_reason=rejection_reason,
            model_probability=None,
            market_price=None,
            side=None,
            effective_edge=None,
            strike_abs_z=None,
            emos_prob_raw=None,
            pooled_median=raw_ensemble_medians.get(fkey_target),
            ensemble_std_pooled=raw_ensemble_stds.get(fkey_target),
            pnl_ask=pnl_ask,
            contracts=None,
        )

    # Smoke-check: strategy.evaluate(market) must return the same model_prob.
    # This catches a future regression where the strategy's evaluate path
    # diverges from the underlying evaluate_market_pure composition (spec
    # §4.5 line 749). The clock stays frozen for this call so the EMOS
    # dispatcher's lead-days check sees snapshot_day UTC.
    with _freeze_clock(snapshot_day_d):
        try:
            evaluate_prob = await strategy.evaluate(sc)
        except Exception as exc:
            return _replay_error_result(
                audit_row,
                f"strategy.evaluate raised: {exc!r}",
            )
    if evaluate_prob is None or not math.isclose(
        evaluate_prob, candidate.model_prob, rel_tol=0.0, abs_tol=1e-9
    ):
        return _replay_error_result(
            audit_row,
            (
                f"strategy.evaluate divergence: "
                f"evaluate_prob={evaluate_prob!r} vs "
                f"candidate.model_prob={candidate.model_prob!r}"
            ),
        )

    return ReplayResult(
        cell=cell,
        snapshot_day=snapshot_day,
        market_ticker=ticker,
        candidate_passed=True,
        rejection_reason=None,
        model_probability=candidate.model_prob,
        market_price=candidate.price_info.midpoint,
        side=candidate.sizing.side.value if candidate.sizing.side != Side.NONE else None,
        effective_edge=candidate.effective_edge,
        strike_abs_z=candidate.strike_abs_z,
        emos_prob_raw=candidate.emos_prob_raw,
        pooled_median=candidate.ensemble_median_pooled,
        ensemble_std_pooled=candidate.ensemble_std_pooled,
        pnl_ask=pnl_ask,
        contracts=candidate.contracts,
    )


def _infer_rejection_reason(funnel: ScanFunnel) -> str | None:
    """Walk the funnel counters and return the first attribute that
    ticked. Order matters: scan-level filters first, then per-market
    eval gates, then Stage-2 (which evaluate_market_pure doesn't reach,
    but kept for completeness).
    """
    # Walk in funnel-order to find which counter incremented.
    counter_to_reason: list[tuple[str, str]] = [
        ("n_excluded_city_var", RejectionReason.EXCLUDED_CITY_VAR.value),
        ("n_forecast_gap", RejectionReason.FORECAST_MISSING.value),
        # Runtime σ_emos / bracket gate (Tier 2 A, 2026-05-19) fires after
        # the EMOS dispatcher and before the calibrated-floor block — keep
        # the ordering here in sync with weather_evaluation.py.
        ("n_emos_sigma_incompatible", RejectionReason.EMOS_SIGMA_INCOMPATIBLE.value),
        ("n_cal_floor_block", RejectionReason.CAL_FLOOR_BLOCK.value),
        ("n_plateau_block", RejectionReason.PLATEAU_BLOCK.value),
        ("n_raw_prob_reject", RejectionReason.RAW_PROB_GATE.value),
        ("n_strike_distance_reject", RejectionReason.STRIKE_DISTANCE.value),
        ("n_no_price", RejectionReason.NO_PRICE.value),
        ("n_spread_reject", RejectionReason.SPREAD.value),
        ("n_calibrator_disagreement", RejectionReason.CALIBRATOR_DISAGREEMENT.value),
        ("n_model_market_gap", RejectionReason.MODEL_MARKET_GAP_INTERIM.value),
        ("n_yes_floor", RejectionReason.YES_FLOOR.value),
        ("n_yes_inverted", RejectionReason.YES_INVERTED.value),
        ("n_no_inverted", RejectionReason.NO_INVERTED.value),
        ("n_edge_below", RejectionReason.EDGE_BELOW_THRESHOLD.value),
        ("n_edge_above_max", RejectionReason.EDGE_ABOVE_MAX.value),
        ("n_kelly_zero", RejectionReason.KELLY_ZERO.value),
        ("n_kelly_side_flip", RejectionReason.KELLY_SIDE_FLIP.value),
        (
            "n_after_trades_per_cell_per_day_cap",
            RejectionReason.TRADES_PER_CELL_CAP.value,
        ),
        ("n_nbm_bracket_position_veto", RejectionReason.NBM_BRACKET_POSITION_VETO.value),
        ("n_nbm_fetch_failed", RejectionReason.NBM_FETCH_FAILED.value),
        ("n_per_day_cap_blocked", RejectionReason.PER_DAY_EVENT_CAP_HIT.value),
        ("n_boundary_filter_block", RejectionReason.BOUNDARY_FILTER_BLOCK.value),
        # ASOS-nowcast physical plausibility guards (intraday_asos_high).
        ("n_asos_boundary_guard", RejectionReason.ASOS_BOUNDARY_GUARD.value),
        ("n_asos_stale_obs", RejectionReason.ASOS_STALE_OBS.value),
        ("n_asos_zero_headroom", RejectionReason.ASOS_ZERO_HEADROOM.value),
        ("n_asos_bad_prelim", RejectionReason.ASOS_BAD_PRELIM.value),
        ("n_asos_climo_unavailable", RejectionReason.ASOS_CLIMO_UNAVAILABLE.value),
    ]
    for attr, reason in counter_to_reason:
        if getattr(funnel, attr, 0) > 0:
            return reason
    return None


def _ensemble_fraction(
    *,
    forecast: EnsembleForecast,
    direction: Direction,
    floor_strike: float | None = None,
    cap_strike: float | None = None,
) -> float:
    """Standalone version of ``WeatherClient.compute_model_probability``
    that doesn't need a constructed client instance — used by the stub
    weather client in ``replay_row``.
    """
    temps = forecast.temperatures
    n = len(temps)
    if n == 0:
        return 0.0
    if direction == Direction.ABOVE:
        if floor_strike is None:
            raise ValueError("floor_strike required for ABOVE direction")
        # Kalshi rule: t > floor (STRICT). Matches production
        # EnsembleForecast.fraction_above and derive_outcome_yes.
        return sum(1 for t in temps if t > floor_strike) / n
    if direction == Direction.BELOW:
        if cap_strike is None:
            raise ValueError("cap_strike required for BELOW direction")
        return sum(1 for t in temps if t < cap_strike) / n
    if direction == Direction.BETWEEN:
        if floor_strike is None or cap_strike is None:
            raise ValueError("floor + cap required for BETWEEN direction")
        return sum(1 for t in temps if floor_strike <= t <= cap_strike) / n
    raise ValueError(f"unsupported direction: {direction!r}")


def _replay_error_result(audit_row: dict[str, Any], message: str) -> ReplayResult:
    """Pack a replay-error into the same ReplayResult shape — diff_row sees
    it and surfaces it as a ``TradeDiff`` with ``error`` set via the
    ``rejection_reason`` field.
    """
    return ReplayResult(
        cell=audit_row.get("cell", "?"),
        snapshot_day=audit_row.get("snapshot_day", "?"),
        market_ticker=audit_row.get("market_ticker", "?"),
        candidate_passed=False,
        rejection_reason=f"REPLAY_ERROR:{message}",
        model_probability=None,
        market_price=None,
        side=None,
        effective_edge=None,
        strike_abs_z=None,
        emos_prob_raw=None,
        pooled_median=None,
        ensemble_std_pooled=None,
        pnl_ask=None,
        contracts=None,
    )


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def _diff_numeric(
    name: str,
    audit_value: float | None,
    replay_value: float | None,
    tolerance: float,
) -> FieldDiff | None:
    if audit_value is None and replay_value is None:
        return None
    if audit_value is None or replay_value is None:
        return FieldDiff(name, audit_value, replay_value, None)
    abs_diff = abs(audit_value - replay_value)
    if abs_diff <= tolerance:
        return None
    return FieldDiff(name, audit_value, replay_value, abs_diff)


def _diff_categorical(
    name: str,
    audit_value: str | None,
    replay_value: str | None,
) -> FieldDiff | None:
    if audit_value == replay_value:
        return None
    return FieldDiff(name, audit_value, replay_value, None)


# ---------------------------------------------------------------------------
# Field-tolerance map + diff classification
# ---------------------------------------------------------------------------
#
# Most §4.5 fields hash to bit-identity at 1e-9 because they come from
# ``emos_probability`` (deterministic Gaussian-CDF math). Two exceptions
# have wider tolerances because production's compute path differs from
# the v6 audit's by design:
#
#   * ``effective_edge`` — production's ``ensemble_confidence`` multiplier
#     uses pooled blended ``ens_std`` (ddof=0 across gfs_seamless +
#     ecmwf_ifs025), while the v6 audit's uses single-model GFS ``ens_std``
#     (ddof=1). The EMOS posterior (``emos_probability``) is invariant
#     under this choice — the ddof+pooling asymmetry only reaches
#     ``effective_edge`` via the conf multiplier. Calibration-analyst
#     measurement (2026-05-15): max |Δ effective_edge| < 0.001 on the 7
#     measurable v6 threshold rows, with zero edge-floor flips and zero
#     Kelly-bucket flips. Option A tolerance pin: |Δ| ≤ 0.01 AND the
#     edge-floor side is preserved AND the Kelly sizing bucket is
#     preserved → INFORMATIONAL; anything outside that envelope →
#     STRUCTURAL (catches a real downstream regression).
#   * ``market_price`` (yes_mid) — Kalshi quote rounding can drift
#     0.005 across snapshot reloads on the same minute.
#
# Structural §4.5 fields (gate the exit code): ``model_probability``,
# ``side``, ``strike_abs_z``, ``gate_pass`` (audit-shared gates only),
# plus the defensive backbone ``emos_prob_raw`` / ``pooled_median`` /
# ``ensemble_std_pooled`` / ``pnl_ask``.
_FIELD_TOLERANCE_OVERRIDES: dict[str, float] = {
    "market_price": 0.005,
}

# Effective-edge classifier thresholds — see Option A note above. The
# floor + ceiling mirror production config (``edge_threshold=0.07``,
# ``max_edge=0.30``) so a divergence that would flip either gate at the
# prod settings is caught here even though the verifier neutralizes the
# ceiling for the audit-replay diff.
_EFFECTIVE_EDGE_TOLERANCE: float = 0.01
_PROD_EDGE_THRESHOLD: float = 0.07
_PROD_MAX_EDGE: float = 0.30

# Fields whose divergences are surfaced in the JSON sink + stderr but do
# NOT gate the exit code (see comment above). ``effective_edge`` is
# *conditionally* informational — see ``_classify_effective_edge``.
_INFORMATIONAL_FIELDS: frozenset[str] = frozenset()


def _classify_effective_edge(
    *,
    audit_edge: float | None,
    replay_edge: float | None,
    replay_contracts: int | None,
) -> tuple[str, FieldDiff | None]:
    """Return ``(classification, field_diff)`` for the effective_edge field.

    classification ∈ {"match", "informational", "structural"}:
      * "match"        — both None, or numerically identical.
      * "informational" — |Δ| ≤ 0.01 AND edge-floor side preserved AND
        Kelly bucket preserved (would-prod-Kelly-fire matches between
        audit and replay).
      * "structural"   — any guard trips.

    Kelly-bucket preservation here is the *production* outcome: would
    the trade have reached Kelly with prod's gate stack
    (``edge_threshold=0.07``, ``max_edge=0.30``)? A trade reaches Kelly
    iff its edge lies in ``[edge_threshold, max_edge]``. If audit_edge
    and replay_edge fall on opposite sides of either bound, the
    production outcome flips — that is the real sizing-impact regression
    we need to catch. ``replay_contracts`` is the verifier's neutralized-
    ceiling result (max_edge=1.0); we don't compare to it directly
    because the verifier intentionally bypasses the prod ceiling for the
    audit-shared diff.
    """
    if audit_edge is None and replay_edge is None:
        return "match", None
    if audit_edge is None or replay_edge is None:
        return "structural", FieldDiff("effective_edge", audit_edge, replay_edge, None)
    abs_diff = abs(audit_edge - replay_edge)
    if abs_diff == 0.0:
        return "match", None
    del replay_contracts  # see docstring — prod-outcome comparison instead.

    audit_above_floor = audit_edge >= _PROD_EDGE_THRESHOLD
    replay_above_floor = replay_edge >= _PROD_EDGE_THRESHOLD
    edge_floor_preserved = audit_above_floor == replay_above_floor
    audit_in_prod_band = audit_above_floor and audit_edge <= _PROD_MAX_EDGE
    replay_in_prod_band = replay_above_floor and replay_edge <= _PROD_MAX_EDGE
    kelly_bucket_preserved = audit_in_prod_band == replay_in_prod_band

    if abs_diff <= _EFFECTIVE_EDGE_TOLERANCE and edge_floor_preserved and kelly_bucket_preserved:
        return "informational", FieldDiff(
            "effective_edge_informational", audit_edge, replay_edge, abs_diff
        )
    return "structural", FieldDiff("effective_edge", audit_edge, replay_edge, abs_diff)


def diff_trade(
    audit_row: dict[str, Any],
    replay: ReplayResult,
    *,
    tolerance: float,
) -> TradeDiff:
    """Diff one v6 audit row vs. its replay; ``diff.fields`` is empty on
    bit-identical match (modulo per-field tolerance overrides).
    """
    out = TradeDiff(
        cell=replay.cell,
        market_ticker=replay.market_ticker,
        snapshot_day=replay.snapshot_day,
    )

    def _tol(name: str) -> float:
        return _FIELD_TOLERANCE_OVERRIDES.get(name, tolerance)

    strike_type = audit_row.get("strike_type", "")
    is_threshold = strike_type in ("greater", "less")

    # Spec §4.5 line 757-760: gate_pass / gate_reject is the most
    # important diff. Audit rows are PASSED-only (CSV is post-gate); a
    # replay-side REJECT is either:
    #   * STRUCTURAL — gate the audit ALSO applied (CAL_FLOOR_BLOCK,
    #     PLATEAU_BLOCK, RAW_PROB_GATE, STRIKE_DISTANCE, SPREAD,
    #     CALIBRATOR_DISAGREEMENT, EXCLUDED_CITY_VAR, EDGE_BELOW_THRESHOLD,
    #     YES_INVERTED, NO_INVERTED). Surfaces as a fail-class FieldDiff.
    #   * INFORMATIONAL — production-only gate (YES_FLOOR,
    #     EDGE_ABOVE_MAX, MODEL_MARKET_GAP_INTERIM, NO_PRICE, KELLY_*).
    #     Surfaces as a separate ``gate_pass_informational`` row that
    #     the JSON sink carries but the exit code IGNORES. Operator can
    #     review production-only filtering separately from regressions.
    if not replay.candidate_passed:
        rr = replay.rejection_reason or "UNKNOWN"
        is_informational = rr in _PROD_ONLY_REJECTION_REASONS or rr.startswith("REPLAY_ERROR:")
        out.fields.append(
            FieldDiff(
                field="gate_pass_informational" if is_informational else "gate_pass",
                audit_value="PASSED",
                replay_value=f"REJECTED:{rr}",
                abs_diff=None,
            )
        )
        # Surface underlying pooled diffs only on threshold markets (BETWEEN
        # rows in the CSV have prod_median_f/prod_pstdev_f blank by audit
        # convention — they don't drive a strike-distance gate).
        if is_threshold:
            for name, audit_str, replay_val in (
                ("pooled_median", audit_row.get("prod_median_f", ""), replay.pooled_median),
                (
                    "ensemble_std_pooled",
                    audit_row.get("prod_pstdev_f", ""),
                    replay.ensemble_std_pooled,
                ),
            ):
                fd = _diff_numeric(name, _parse_optional_float(audit_str), replay_val, _tol(name))
                if fd is not None:
                    out.fields.append(fd)
        return out

    # Numeric spec §4.5 fields. ``model_probability`` is checked TWICE —
    # once via the spec field name and once via ``emos_prob_raw`` (the
    # defensive cross-check). They alias to the same CSV column
    # (emos_prob_walkforward) and should produce identical diffs.
    # ``effective_edge`` is classified separately via
    # ``_classify_effective_edge`` (Option A tolerance pin).
    field_specs_numeric: list[tuple[str, str, float | None]] = [
        ("model_probability", "emos_prob_walkforward", replay.model_probability),
        ("market_price", "yes_mid", replay.market_price),
        ("emos_prob_raw", "emos_prob_walkforward", replay.emos_prob_raw),
        ("pnl_ask", "pnl_ask", replay.pnl_ask),
    ]
    # strike_abs_z + pooled inputs are only defined on threshold markets —
    # the CSV cells are blank for BETWEEN.
    if is_threshold:
        field_specs_numeric.extend(
            [
                ("strike_abs_z", "strike_abs_z", replay.strike_abs_z),
                ("pooled_median", "prod_median_f", replay.pooled_median),
                ("ensemble_std_pooled", "prod_pstdev_f", replay.ensemble_std_pooled),
            ]
        )

    for diff_name, csv_col, replay_val in field_specs_numeric:
        audit_val = _parse_optional_float(audit_row.get(csv_col, ""))
        fd = _diff_numeric(diff_name, audit_val, replay_val, _tol(diff_name))
        if fd is not None:
            out.fields.append(fd)

    # effective_edge: Option A tolerance pin. |Δ| ≤ 0.01 AND edge-floor
    # side preserved AND edge-ceiling side preserved AND Kelly bucket
    # preserved → informational; otherwise structural.
    _classification, ee_diff = _classify_effective_edge(
        audit_edge=_parse_optional_float(audit_row.get("effective_edge", "")),
        replay_edge=replay.effective_edge,
        replay_contracts=replay.contracts,
    )
    if ee_diff is not None:
        out.fields.append(ee_diff)

    # Categorical: side is YES/NO.
    side_diff = _diff_categorical("side", audit_row.get("side"), replay.side)
    if side_diff is not None:
        out.fields.append(side_diff)

    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_value(v: Any) -> str:
    if v is None:
        return "<none>"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.12g}"
    return str(v)


def _format_diff_row(diff: TradeDiff, fd: FieldDiff) -> str:
    return (
        f"  {diff.cell:>8}  {diff.snapshot_day}  {diff.market_ticker:<32}  "
        f"{fd.field:<22}  "
        f"audit={_format_value(fd.audit_value):<22}  "
        f"replay={_format_value(fd.replay_value):<22}  "
        f"|delta|={_format_value(fd.abs_diff)}"
    )


def _is_informational_field(field_name: str) -> bool:
    """True iff a field-diff is informational-only (does not gate the exit code).

    Informational field markers:
      * suffix ``_informational`` (e.g. ``effective_edge_informational``)
      * exact name ``gate_pass_informational`` (production-only gate rejection)
    """
    return field_name == "gate_pass_informational" or field_name.endswith("_informational")


def _is_structural_diff(td: TradeDiff) -> bool:
    """True iff the diff has at least one field that is NOT an
    informational-only marker. Replay-error rows are structural.
    """
    if td.error is not None:
        return True
    for fd in td.fields:
        if not _is_informational_field(fd.field):
            return True
    return False


def _emit_summary(
    *,
    n_trades: int,
    n_replay_errors: int,
    diffs: list[TradeDiff],
    tolerance: float,
    sha_match: bool,
    computed_sha: str,
    stream: Any,
) -> None:
    n_structural = sum(1 for d in diffs if _is_structural_diff(d))
    n_informational = sum(1 for d in diffs if not _is_structural_diff(d) and (d.fields or d.error))
    print(
        f"verify_intraday_replay: n_trades={n_trades} "
        f"n_structural_diffs={n_structural} "
        f"n_informational_diffs={n_informational} "
        f"n_replay_errors={n_replay_errors} tolerance={tolerance:g}",
        file=stream,
    )
    if not sha_match:
        print(
            f"verify_intraday_replay: GATE-INPUT SHA MISMATCH "
            f"(computed={computed_sha} expected={EXPECTED_PROD_STRIKE_INPUTS_SHA256}) "
            f"— production strike-distance inputs have drifted from v6",
            file=stream,
        )
    if n_structural == 0 and sha_match:
        print(
            "verify_intraday_replay: PASS — replay is bit-identical to v6 "
            "audit within tolerance and gate-input SHA matches "
            f"({n_informational} informational diff(s) from production-only gates)",
            file=stream,
        )
        return

    print(
        "verify_intraday_replay: FAIL — structural diff detected vs. v6 audit",
        file=stream,
    )
    print("", file=stream)
    print("Per-trade diffs:", file=stream)
    for d in diffs:
        if d.error is not None:
            print(
                f"  {d.cell:>8}  {d.snapshot_day}  {d.market_ticker:<32}  REPLAY_ERROR: {d.error}",
                file=stream,
            )
            continue
        for fd in d.fields:
            print(_format_diff_row(d, fd), file=stream)


def _write_json_sink(
    *,
    output_path: Path,
    run_date: str,
    n_trades: int,
    diffs: list[TradeDiff],
    n_replay_errors: int,
    computed_sha: str,
    sha_match: bool,
    exit_code: int,
) -> None:
    """Write ``data/research/intraday_replay_diff_<run_date>.json`` (plan
    M7 line 964 + spec §C line 297). Always emitted, regardless of
    pass/fail, so a scheduled monitor can consume it programmatically.
    """
    n_structural = sum(1 for d in diffs if _is_structural_diff(d))
    n_informational = sum(1 for d in diffs if not _is_structural_diff(d) and (d.fields or d.error))
    n_diffs = sum(1 for d in diffs if d.fields or d.error)
    payload_diffs: list[dict[str, Any]] = []
    payload_errors: list[dict[str, Any]] = []
    for d in diffs:
        if d.error is not None:
            payload_errors.append(
                {
                    "snapshot_day": d.snapshot_day,
                    "cell": d.cell,
                    "ticker": d.market_ticker,
                    "error": d.error,
                }
            )
            continue
        for fd in d.fields:
            payload_diffs.append(
                {
                    "snapshot_day": d.snapshot_day,
                    "cell": d.cell,
                    "ticker": d.market_ticker,
                    "field": fd.field,
                    "audit": fd.audit_value,
                    "replay": fd.replay_value,
                    "abs_delta": fd.abs_diff,
                    "informational": _is_informational_field(fd.field),
                }
            )

    payload = {
        "run_date": run_date,
        "n_trades": n_trades,
        "n_diffs": n_diffs,
        "n_structural_diffs": n_structural,
        "n_informational_diffs": n_informational,
        "n_replay_errors": n_replay_errors,
        "gate_input_sha256": computed_sha,
        "expected_sha256": EXPECTED_PROD_STRIKE_INPUTS_SHA256,
        "sha_match": sha_match,
        "exit_code": exit_code,
        "diffs": payload_diffs,
        "replay_errors": payload_errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI surface — matches plan M7 line 949."""
    p = argparse.ArgumentParser(
        description=(
            "Replay IntradayEmosHighStrategy.evaluate against v6 audit "
            "snapshot rows and assert bit-identical output (spec §4.5). "
            "Exit 0 on match, 1 on any diff, 2 on fixture/env error. "
            "Operator-invoked; exit code is the alert channel — no "
            "Telegram bridge fires (spec §11.2 v1.3 IMPORTANT #1)."
        ),
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help=(
            "Inclusive lower bound for snapshot_day (YYYY-MM-DD). Defaults "
            "to the earliest snapshot_day in the v6 fixture CSV."
        ),
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=(
            "Inclusive upper bound for snapshot_day (YYYY-MM-DD). Defaults "
            "to the latest snapshot_day in the v6 fixture CSV."
        ),
    )
    p.add_argument(
        "--cell",
        type=str,
        default="all",
        choices=["all", "MIA/high", "LAX/high"],
        help=(
            "Cell filter. 'all' (default) replays both v6-bound cells; "
            "'MIA/high' or 'LAX/high' restrict to one cell."
        ),
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=_DEFAULT_TOLERANCE,
        help=(
            f"Absolute numeric tolerance for bit-identity check (default: {_DEFAULT_TOLERANCE:g})."
        ),
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Reserved no-op for future diff-classification work; currently "
            "exits 1 on ANY diff regardless of this flag."
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output path for the per-row diff JSON. Defaults to "
            "data/research/intraday_replay_diff_<run_date>.json."
        ),
    )
    p.add_argument(
        "--snapshots-dir",
        type=Path,
        default=_DEFAULT_SNAPSHOTS_DIR,
        help=(
            f"Snapshot root directory (default: {_DEFAULT_SNAPSHOTS_DIR}). "
            "Overridable for hermetic tests."
        ),
    )
    p.add_argument(
        "--bias-path",
        type=Path,
        default=_DEFAULT_BIAS_PATH,
        help=(
            f"GFS bias-correction artifact (default: {_DEFAULT_BIAS_PATH}). "
            "Overridable for hermetic tests."
        ),
    )
    return p


def _resolve_output_path(
    *,
    cli_value: Path | None,
    run_date: str,
) -> Path:
    if cli_value is not None:
        return cli_value
    return _DEFAULT_OUTPUT_DIR / f"intraday_replay_diff_{run_date}.json"


async def _run_async(
    *,
    trades_csv: Path,
    coeffs_json: Path,
    strike_inputs_json: Path,
    snapshots_dir: Path,
    bias_path: Path,
    start_date: str | None,
    end_date: str | None,
    cell_filter: str,
    tolerance: float,
    output_json: Path | None,
    stream: Any,
) -> int:
    """Async core — separated from ``main`` so tests can drive the event
    loop manually.
    """
    # Fixture I/O.
    try:
        trades = _load_v6_trades(trades_csv)
        walkforward = _load_walkforward_coeffs_by_date(coeffs_json)
    except FileNotFoundError as exc:
        print(f"verify_intraday_replay: FIXTURE ERROR: {exc}", file=stream)
        return 2
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        print(
            f"verify_intraday_replay: FIXTURE PARSE ERROR: {exc!r}",
            file=stream,
        )
        return 2

    # Sanity: the strike-inputs JSON exists (we don't load it here — we
    # recompute the SHA from snapshots — but its absence indicates a
    # broken fixture set).
    if not strike_inputs_json.exists():
        print(
            f"verify_intraday_replay: FIXTURE ERROR: strike-inputs JSON "
            f"not found: {strike_inputs_json}",
            file=stream,
        )
        return 2

    # Load forecasts globally ONCE (same semantic as the v6 audit's
    # load_snapshot_data: last-writer-wins across runs so a market on
    # snapshot_day=A/run_id=B can use a forecast emitted from
    # snapshot_day=A/run_id=C if C is more recent). Per-row replay uses
    # _build_forecast_cache_from_global to project the relevant cell.
    forecasts_global, _settlements, _markets_all = load_snapshot_data(snapshots_dir)

    # Compute the production-faithful gate-input SHA and assert it matches
    # the pinned v6 value (spec §4.5 lines 764-767). Reuses
    # ``forecasts_global`` so we don't pay for the snapshot scan twice.
    bias_table_for_sha = BiasTable(bias_path)
    if not bias_table_for_sha.loaded:
        print(
            f"verify_intraday_replay: FIXTURE ERROR: bias table at "
            f"{bias_path} failed to load (required for v6 production-"
            f"faithful strike-distance gate inputs)",
            file=stream,
        )
        return 2
    # The v6 fixture file IS the pinned table — hash it directly. A
    # recompute over data/snapshots/ would false-alarm on every calendar
    # tick (post-freeze GHA re-runs mutate trailing-edge forecast_dates;
    # new forecast_dates accumulate). Snapshot mutation inside the v6
    # keyset is still detected via the per-row replay diff, where a
    # changed ensemble member surfaces as ``model_probability`` /
    # ``emos_prob_raw`` / ``pooled_median`` drift and flips exit to 1.
    computed_sha, _serialized = compute_gate_input_sha256(
        strike_inputs_path=strike_inputs_json,
    )
    sha_match = computed_sha == EXPECTED_PROD_STRIKE_INPUTS_SHA256

    # Filter v6 trades by window + cell.
    if start_date is None:
        start_date = min(r["snapshot_day"] for r in trades)
    if end_date is None:
        end_date = max(r["snapshot_day"] for r in trades)
    filtered_trades = [
        r
        for r in trades
        if start_date <= r["snapshot_day"] <= end_date
        and (cell_filter == "all" or r["cell"] == cell_filter)
    ]

    bias_table = bias_table_for_sha

    # Settings for the strategy. PAPER mode so the strategy's cal-floor
    # block applies (audit used min_calibrated_prob=0.10; shadow mode
    # would exempt the floor and create a divergence the audit didn't have).
    state_dir = snapshots_dir.parent / "_verify_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(state_dir=str(state_dir), _env_file=None)
    settings.execution.mode = ExecutionMode.PAPER

    diffs: list[TradeDiff] = []
    n_replay_errors = 0
    for audit_row in filtered_trades:
        try:
            replay = await replay_row(
                audit_row=audit_row,
                snapshots_dir=snapshots_dir,
                forecasts_global=forecasts_global,
                walkforward=walkforward,
                bias_table=bias_table,
                settings=settings,
            )
        except (ValueError, KeyError, AttributeError) as exc:
            n_replay_errors += 1
            diffs.append(
                TradeDiff(
                    cell=audit_row.get("cell", "?"),
                    market_ticker=audit_row.get("market_ticker", "?"),
                    snapshot_day=audit_row.get("snapshot_day", "?"),
                    error=repr(exc),
                )
            )
            continue
        diffs.append(diff_trade(audit_row, replay, tolerance=tolerance))

    _emit_summary(
        n_trades=len(filtered_trades),
        n_replay_errors=n_replay_errors,
        diffs=diffs,
        tolerance=tolerance,
        sha_match=sha_match,
        computed_sha=computed_sha,
        stream=stream,
    )

    has_structural = any(_is_structural_diff(d) for d in diffs)
    exit_code = 1 if (has_structural or not sha_match) else 0

    # Always emit JSON sink (plan M7 line 964).
    run_date_str = datetime.now().date().isoformat()
    output_path = _resolve_output_path(cli_value=output_json, run_date=run_date_str)
    try:
        _write_json_sink(
            output_path=output_path,
            run_date=run_date_str,
            n_trades=len(filtered_trades),
            diffs=diffs,
            n_replay_errors=n_replay_errors,
            computed_sha=computed_sha,
            sha_match=sha_match,
            exit_code=exit_code,
        )
    except OSError as exc:
        print(
            f"verify_intraday_replay: warning — failed to write JSON sink "
            f"at {output_path}: {exc!r}",
            file=stream,
        )

    return exit_code


def run(
    *,
    trades_csv: Path = _DEFAULT_TRADES_CSV,
    coeffs_json: Path = _DEFAULT_COEFFS_JSON,
    strike_inputs_json: Path = _DEFAULT_STRIKE_INPUTS_JSON,
    snapshots_dir: Path = _DEFAULT_SNAPSHOTS_DIR,
    bias_path: Path = _DEFAULT_BIAS_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
    cell_filter: str = "all",
    tolerance: float = _DEFAULT_TOLERANCE,
    output_json: Path | None = None,
    stream: Any = sys.stderr,
) -> int:
    """Sync entry point — wraps the async core for the CLI + test surface.

    Exit codes:
      * 0 — bit-identical replay vs. v6 audit AND gate-input SHA matches.
      * 1 — at least one diff (field, gate-pass, replay-error, or SHA).
      * 2 — fixture / environment error (file missing, JSON malformed,
        bias table fail).
    """
    return asyncio.run(
        _run_async(
            trades_csv=trades_csv,
            coeffs_json=coeffs_json,
            strike_inputs_json=strike_inputs_json,
            snapshots_dir=snapshots_dir,
            bias_path=bias_path,
            start_date=start_date,
            end_date=end_date,
            cell_filter=cell_filter,
            tolerance=tolerance,
            output_json=output_json,
            stream=stream,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return run(
        start_date=args.start_date,
        end_date=args.end_date,
        cell_filter=args.cell,
        tolerance=args.tolerance,
        output_json=args.output_json,
        snapshots_dir=args.snapshots_dir,
        bias_path=args.bias_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
