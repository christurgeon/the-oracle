"""Tests for ``scripts/verify_intraday_replay.py`` (M7).

Test layout:
  * ``test_replay_matches_v6_on_pinned_window`` — happy path against a
    small 3-day fixture window; assert structural diffs = 0 + SHA
    matches + exit 0.
  * ``test_replay_detects_diff_when_strategy_returns_wrong_prob`` — inject
    a perturbation into one trade's ``emos_prob_walkforward`` CSV cell;
    assert the verifier exits 1 with a ``model_probability`` /
    ``emos_prob_raw`` structural diff in the output.
  * ``test_replay_gate_input_sha_matches_v6`` — pure assertion that the
    computed gate-input SHA on production snapshots equals the pinned
    v6 hash (plan M7 line 975).
  * ``test_replay_emits_json_sink_on_every_run`` — JSON sink is written
    regardless of pass/fail status; schema matches the documented shape.
  * ``test_missing_trades_csv_exits_2`` — fixture-missing path.
  * ``test_missing_coeffs_json_exits_2`` — fixture-missing path.
  * ``test_missing_strike_inputs_json_exits_2`` — fixture-missing path.
  * ``test_diff_classification_informational_vs_structural`` — unit test
    for ``_is_structural_diff`` + ``_is_informational_field``.
  * ``test_format_diff_row_renders_known_shape`` — operator-facing format.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from python.analytics.audit_log import RejectionReason, ScanFunnel
from scripts.verify_intraday_replay import (
    EXPECTED_PROD_STRIKE_INPUTS_SHA256,
    FieldDiff,
    TradeDiff,
    _classify_effective_edge,
    _format_diff_row,
    _infer_rejection_reason,
    _is_informational_field,
    _is_structural_diff,
    compute_gate_input_sha256,
    run,
)

_REPO = Path(__file__).resolve().parent.parent
_V6_TRADES_CSV = _REPO / "data/research/workstream1_gate_audit_v6_trades.csv"
_V6_COEFFS_JSON = _REPO / "data/research/workstream1_gate_audit_v6_walkforward_coeffs.json"
_V6_STRIKE_INPUTS_JSON = _REPO / "data/research/workstream1_gate_audit_v6_prod_strike_inputs.json"
_SNAPSHOTS_DIR = _REPO / "data/snapshots"
_BIAS_PATH = _REPO / "data/gfs_bias/bias_correction.json"

# Tests that replay v6 fixture rows through the strategy need the local
# snapshot tree (forecasts + markets per snapshot_day/run_id). It is
# NOT tracked in git and not present on a fresh CI clone — skip when
# missing. See the test docstrings for the "how to populate locally"
# pointer.
_REQUIRES_SNAPSHOTS = pytest.mark.skipif(
    not _SNAPSHOTS_DIR.exists() or not any(_SNAPSHOTS_DIR.iterdir()),
    reason=(
        "requires data/snapshots/ — see test docstring. "
        "Populate by running "
        "`uv run python scripts/collect_daily_snapshot.py` locally for "
        "each forecast_date in the v6 window (2026-03-19 .. 2026-05-16)."
    ),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stderr_buf() -> io.StringIO:
    return io.StringIO()


def _copy_csv_with_perturbation(
    src: Path,
    dst: Path,
    *,
    target_row_filter: dict[str, str],
    column: str,
    delta: float,
) -> None:
    """Copy ``src`` -> ``dst`` and bump one column by ``delta`` on the FIRST
    row matching ``target_row_filter``. Other rows pass through untouched.
    """
    with src.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())

    bumped = False
    for row in rows:
        if bumped:
            break
        if all(row.get(k) == v for k, v in target_row_filter.items()):
            original = float(row[column])
            row[column] = repr(original + delta)
            bumped = True
    assert bumped, (
        f"no row matched target_row_filter={target_row_filter!r} — "
        "adjust the test fixture selector"
    )

    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# §4.5 LOAD-BEARING: gate-input SHA matches v6
# ---------------------------------------------------------------------------


def test_replay_gate_input_sha_matches_v6() -> None:
    """The verifier's gate-input SHA MUST equal the pinned v6 value.

    Plan M7 line 975. The SHA is computed by hashing the v6 pinned
    strike-inputs fixture
    (``data/research/workstream1_gate_audit_v6_prod_strike_inputs.json``)
    — this fixture file IS the authoritative pin. Production snapshot
    drift inside the v6 keyset is detected separately, via per-row
    replay diffs (``model_probability`` et al.) in
    ``test_replay_matches_v6_on_pinned_window``.

    Dependencies
    ------------
    * ``data/research/workstream1_gate_audit_v6_prod_strike_inputs.json``
      (checked into the repo; v6 freeze covers forecast_dates
      ``2026-03-19`` through ``2026-05-16``).

    This test has NO runtime dependency on ``data/snapshots/`` — only the
    fixture file. A SHA mismatch here means the fixture file itself was
    mutated; investigate before shipping any intraday cell.

    How to run locally
    ------------------
    ``uv run pytest tests/test_verify_intraday_replay.py -k sha_matches_v6``
    """
    computed_sha, serialized = compute_gate_input_sha256(
        strike_inputs_path=_V6_STRIKE_INPUTS_JSON,
    )
    assert computed_sha == EXPECTED_PROD_STRIKE_INPUTS_SHA256, (
        f"Gate-input SHA drift: computed={computed_sha} "
        f"expected={EXPECTED_PROD_STRIKE_INPUTS_SHA256}.\n"
        f"The v6 pinned strike-inputs fixture has been mutated. "
        f"Investigate before shipping any intraday cell — v6 evidence "
        f"no longer applies."
    )
    # read_bytes() symmetry: the returned text equals the on-disk bytes
    # decoded — guards against an accidental in-memory rewrite (e.g.
    # platform line-ending translation) that would silently invalidate
    # the SHA contract.
    on_disk = _V6_STRIKE_INPUTS_JSON.read_bytes().decode()
    assert serialized == on_disk, (
        "Gate-input serialization drift — the in-memory text no longer "
        "matches data/research/workstream1_gate_audit_v6_prod"
        "_strike_inputs.json. Check compute_gate_input_sha256."
    )


# ---------------------------------------------------------------------------
# Happy path: replay matches v6 on the pinned window
# ---------------------------------------------------------------------------


@_REQUIRES_SNAPSHOTS
def test_replay_matches_v6_on_pinned_window(
    stderr_buf: io.StringIO, tmp_path: Path
) -> None:
    """A small 3-day fixture window must replay with zero structural diffs.

    Per spec §4.5 the structural diff field set is
    ``(model_probability, side, strike_abs_z, gate_pass-on-audit-shared-
    gates, emos_prob_raw, pooled_median, ensemble_std_pooled, pnl_ask)``.
    Informational diffs (effective_edge inside the Option A tolerance
    envelope, YES_FLOOR/MAX_EDGE rejections) are expected and do NOT
    gate the exit code.

    Requires the local ``data/snapshots/`` tree populated for the v6
    window (forecast_dates 2026-03-19 .. 2026-05-16); skipped otherwise.
    """
    output_json = tmp_path / "intraday_replay_diff.json"
    rc = run(
        trades_csv=_V6_TRADES_CSV,
        coeffs_json=_V6_COEFFS_JSON,
        strike_inputs_json=_V6_STRIKE_INPUTS_JSON,
        snapshots_dir=_SNAPSHOTS_DIR,
        bias_path=_BIAS_PATH,
        # 3-day window inside the v6 fixture span.
        start_date="2026-04-06",
        end_date="2026-04-09",
        cell_filter="MIA/high",
        tolerance=1e-9,
        output_json=output_json,
        stream=stderr_buf,
    )
    output = stderr_buf.getvalue()
    assert rc == 0, (
        f"expected exit 0 (structural-diff-free), got {rc}; "
        f"stderr:\n{output}"
    )
    assert "n_structural_diffs=0" in output, output
    assert "PASS" in output, output
    # JSON sink emitted on every run.
    assert output_json.exists()
    payload = json.loads(output_json.read_text())
    assert payload["sha_match"] is True
    assert payload["exit_code"] == 0
    assert payload["n_structural_diffs"] == 0


# ---------------------------------------------------------------------------
# Diff detection: structural diff on EMOS posterior surfaces + exits 1
# ---------------------------------------------------------------------------


@_REQUIRES_SNAPSHOTS
def test_replay_detects_diff_when_strategy_returns_wrong_prob(
    tmp_path: Path, stderr_buf: io.StringIO
) -> None:
    """Inject +1e-6 into one trade's ``emos_prob_walkforward`` cell — the
    replay value (computed from production ``emos_probability``) will
    match the un-perturbed value, so the diff fires on ``model_probability``
    (structural) AND ``emos_prob_raw`` (defensive backbone). Confirms the
    verifier surfaces structural diffs and exits 1.

    Requires the local ``data/snapshots/`` tree; skipped otherwise.
    """
    perturbed_csv = tmp_path / "v6_trades_perturbed.csv"
    _copy_csv_with_perturbation(
        _V6_TRADES_CSV,
        perturbed_csv,
        target_row_filter={"cell": "MIA/high", "snapshot_day": "2026-04-06"},
        column="emos_prob_walkforward",
        delta=1e-6,  # well above the 1e-9 tolerance
    )

    output_json = tmp_path / "intraday_replay_diff.json"
    rc = run(
        trades_csv=perturbed_csv,
        coeffs_json=_V6_COEFFS_JSON,
        strike_inputs_json=_V6_STRIKE_INPUTS_JSON,
        snapshots_dir=_SNAPSHOTS_DIR,
        bias_path=_BIAS_PATH,
        start_date="2026-04-06",
        end_date="2026-04-06",
        cell_filter="MIA/high",
        tolerance=1e-9,
        output_json=output_json,
        stream=stderr_buf,
    )
    output = stderr_buf.getvalue()
    assert rc == 1, f"expected exit 1 (structural diff), got {rc}; stderr:\n{output}"
    assert "FAIL" in output
    # The diff row mentions the diff'd field — either model_probability
    # (spec §4.5) or emos_prob_raw (defensive backbone). Both alias to the
    # same CSV column so both fire.
    assert "model_probability" in output or "emos_prob_raw" in output, output
    payload = json.loads(output_json.read_text())
    assert payload["n_structural_diffs"] >= 1
    assert payload["exit_code"] == 1
    # The structural field is one of the two EMOS posterior aliases.
    fields = {d["field"] for d in payload["diffs"]}
    assert "model_probability" in fields or "emos_prob_raw" in fields, fields


# ---------------------------------------------------------------------------
# JSON sink emitted on every run
# ---------------------------------------------------------------------------


@_REQUIRES_SNAPSHOTS
def test_replay_emits_json_sink_on_every_run(
    tmp_path: Path, stderr_buf: io.StringIO
) -> None:
    """Plan M7 line 964 + spec §C line 297: the diff JSON is written
    regardless of pass/fail. Schema fields covered: ``run_date``,
    ``n_trades``, ``n_structural_diffs``, ``n_informational_diffs``,
    ``n_replay_errors``, ``gate_input_sha256``, ``expected_sha256``,
    ``sha_match``, ``exit_code``, ``diffs``, ``replay_errors``.

    Requires the local ``data/snapshots/`` tree; skipped otherwise.
    """
    output_json = tmp_path / "intraday_replay_diff.json"
    rc = run(
        trades_csv=_V6_TRADES_CSV,
        coeffs_json=_V6_COEFFS_JSON,
        strike_inputs_json=_V6_STRIKE_INPUTS_JSON,
        snapshots_dir=_SNAPSHOTS_DIR,
        bias_path=_BIAS_PATH,
        start_date="2026-04-06",
        end_date="2026-04-07",
        cell_filter="MIA/high",
        tolerance=1e-9,
        output_json=output_json,
        stream=stderr_buf,
    )
    assert rc in (0, 1)
    assert output_json.exists()
    payload = json.loads(output_json.read_text())
    expected_keys = {
        "run_date",
        "n_trades",
        "n_diffs",
        "n_structural_diffs",
        "n_informational_diffs",
        "n_replay_errors",
        "gate_input_sha256",
        "expected_sha256",
        "sha_match",
        "exit_code",
        "diffs",
        "replay_errors",
    }
    assert expected_keys.issubset(payload.keys()), (
        f"JSON sink missing keys: {expected_keys - set(payload.keys())}"
    )
    assert payload["expected_sha256"] == EXPECTED_PROD_STRIKE_INPUTS_SHA256
    # Every diff entry tags informational-or-not.
    for diff in payload["diffs"]:
        assert "informational" in diff, diff


# ---------------------------------------------------------------------------
# Fixture-missing paths
# ---------------------------------------------------------------------------


def test_missing_trades_csv_exits_2(
    tmp_path: Path, stderr_buf: io.StringIO
) -> None:
    rc = run(
        trades_csv=tmp_path / "does_not_exist.csv",
        coeffs_json=_V6_COEFFS_JSON,
        strike_inputs_json=_V6_STRIKE_INPUTS_JSON,
        snapshots_dir=_SNAPSHOTS_DIR,
        bias_path=_BIAS_PATH,
        tolerance=1e-9,
        stream=stderr_buf,
    )
    assert rc == 2
    assert "FIXTURE ERROR" in stderr_buf.getvalue()


def test_missing_coeffs_json_exits_2(
    tmp_path: Path, stderr_buf: io.StringIO
) -> None:
    rc = run(
        trades_csv=_V6_TRADES_CSV,
        coeffs_json=tmp_path / "does_not_exist.json",
        strike_inputs_json=_V6_STRIKE_INPUTS_JSON,
        snapshots_dir=_SNAPSHOTS_DIR,
        bias_path=_BIAS_PATH,
        tolerance=1e-9,
        stream=stderr_buf,
    )
    assert rc == 2
    assert "FIXTURE ERROR" in stderr_buf.getvalue()


def test_missing_strike_inputs_json_exits_2(
    tmp_path: Path, stderr_buf: io.StringIO
) -> None:
    rc = run(
        trades_csv=_V6_TRADES_CSV,
        coeffs_json=_V6_COEFFS_JSON,
        strike_inputs_json=tmp_path / "does_not_exist.json",
        snapshots_dir=_SNAPSHOTS_DIR,
        bias_path=_BIAS_PATH,
        tolerance=1e-9,
        stream=stderr_buf,
    )
    assert rc == 2
    assert "FIXTURE ERROR" in stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Diff classification (informational vs structural)
# ---------------------------------------------------------------------------


def test_diff_classification_informational_vs_structural() -> None:
    """_is_informational_field + _is_structural_diff route fields correctly.

    Spec §4.5: the exit code MUST be gated by structural diffs only — a
    field rename that accidentally drops the ``_informational`` suffix
    would convert a benign divergence into a hard fail.
    """
    assert _is_informational_field("gate_pass_informational") is True
    assert _is_informational_field("effective_edge_informational") is True
    assert _is_informational_field("model_probability") is False
    assert _is_informational_field("strike_abs_z") is False
    assert _is_informational_field("emos_prob_raw") is False

    # A diff that has ONLY an informational field is NOT structural.
    info_only = TradeDiff(
        cell="MIA/high",
        market_ticker="KXHIGHMIA-26APR06-B95",
        snapshot_day="2026-04-06",
        fields=[
            FieldDiff(
                field="effective_edge_informational",
                audit_value=0.5,
                replay_value=0.4,
                abs_diff=0.1,
            )
        ],
    )
    assert _is_structural_diff(info_only) is False

    # Mix of informational + structural is structural.
    mixed = TradeDiff(
        cell="MIA/high",
        market_ticker="KXHIGHMIA-26APR06-B95",
        snapshot_day="2026-04-06",
        fields=[
            FieldDiff(
                field="effective_edge_informational",
                audit_value=0.5,
                replay_value=0.4,
                abs_diff=0.1,
            ),
            FieldDiff(
                field="model_probability",
                audit_value=0.5,
                replay_value=0.4,
                abs_diff=0.1,
            ),
        ],
    )
    assert _is_structural_diff(mixed) is True

    # Replay-error row is always structural.
    errored = TradeDiff(
        cell="MIA/high",
        market_ticker="KXHIGHMIA-26APR06-B95",
        snapshot_day="2026-04-06",
        error="forecast missing",
    )
    assert _is_structural_diff(errored) is True


# ---------------------------------------------------------------------------
# Small unit tests
# ---------------------------------------------------------------------------


def test_format_diff_row_renders_known_shape() -> None:
    """Operator-facing surface — small format test so future
    field-additions don't accidentally shift column alignment in ways
    that break the stderr grep workflow.
    """
    diff = TradeDiff(
        cell="MIA/high",
        market_ticker="KXHIGHMIA-26APR05-B95",
        snapshot_day="2026-04-05",
    )
    fd = FieldDiff(
        field="emos_prob_raw",
        audit_value=0.123456789012,
        replay_value=0.123456789015,
        abs_diff=3e-12,
    )
    rendered = _format_diff_row(diff, fd)
    assert "MIA/high" in rendered
    assert "KXHIGHMIA-26APR05-B95" in rendered
    assert "emos_prob_raw" in rendered
    assert "audit=" in rendered
    assert "replay=" in rendered
    assert "|delta|=" in rendered


def test_format_diff_row_handles_none_sides() -> None:
    """One side ``None`` must render as ``<none>`` (audit-empty cell vs.
    replay-computed value) without raising.
    """
    diff = TradeDiff(
        cell="LAX/high",
        market_ticker="KXHIGHLAX-26APR06-G85",
        snapshot_day="2026-04-06",
    )
    fd = FieldDiff(
        field="strike_abs_z",
        audit_value=None,
        replay_value=2.5,
        abs_diff=None,
    )
    rendered = _format_diff_row(diff, fd)
    assert "<none>" in rendered
    assert "2.5" in rendered


# ---------------------------------------------------------------------------
# _classify_effective_edge — Option A tolerance envelope
# ---------------------------------------------------------------------------


def test_effective_edge_within_tolerance_informational() -> None:
    """|Δ|=0.005 with both rows inside the Kelly band → INFORMATIONAL.

    Spec §4.5 Option A: |Δ| ≤ 0.01 AND edge-floor side preserved AND
    Kelly bucket preserved → informational, does not gate exit code.
    """
    classification, fd = _classify_effective_edge(
        audit_edge=0.10,
        replay_edge=0.105,
        replay_contracts=42,
    )
    assert classification == "informational"
    assert fd is not None
    assert fd.field == "effective_edge_informational"


def test_effective_edge_tolerance_breach_structural() -> None:
    """|Δ|=0.05 (> 0.01 tolerance) → STRUCTURAL even if both rows are
    inside the Kelly band.
    """
    classification, fd = _classify_effective_edge(
        audit_edge=0.10,
        replay_edge=0.15,
        replay_contracts=42,
    )
    assert classification == "structural"
    assert fd is not None
    assert fd.field == "effective_edge"


def test_effective_edge_floor_flip_structural() -> None:
    """|Δ|=0.002 (within tolerance) but the edge-threshold floor side
    flipped (audit above 0.07, replay below) → STRUCTURAL.
    """
    classification, fd = _classify_effective_edge(
        audit_edge=0.071,
        replay_edge=0.069,
        replay_contracts=0,
    )
    assert classification == "structural"
    assert fd is not None
    assert fd.field == "effective_edge"


def test_effective_edge_kelly_bucket_flip_structural() -> None:
    """|Δ|=0.02 within tolerance but the Kelly bucket flipped (audit in
    [0.07, 0.30], replay above max_edge=0.30) → STRUCTURAL.
    """
    classification, fd = _classify_effective_edge(
        audit_edge=0.29,
        replay_edge=0.31,
        replay_contracts=42,
    )
    assert classification == "structural"
    assert fd is not None
    assert fd.field == "effective_edge"


# ---------------------------------------------------------------------------
# _infer_rejection_reason — parametric coverage so a new RejectionReason
# without a counter-walker entry fails loudly at test time.
# ---------------------------------------------------------------------------


# RejectionReasons that the verifier's funnel walker doesn't need to map —
# they correspond to scan-stage filters (PAST_DATE, VOLUME, OPEN_INTEREST,
# PRICE_GATE, SAME_DAY_BLOCKED) applied BEFORE per-market evaluate, plus
# Stage-2 depth-adjust outcomes (ORDERBOOK_THIN, FILL_EDGE_NEGATIVE,
# FILL_SLIPPAGE, EVENT_CAPPED) that evaluate_market_pure never reaches in
# replay mode, plus the terminal PASSED sentinel.
_REJECTION_REASONS_NOT_WALKED: frozenset[RejectionReason] = frozenset(
    {
        RejectionReason.PAST_DATE,
        RejectionReason.VOLUME,
        RejectionReason.OPEN_INTEREST,
        RejectionReason.PRICE_GATE,
        RejectionReason.SAME_DAY_BLOCKED,
        RejectionReason.ORDERBOOK_THIN,
        RejectionReason.FILL_EDGE_NEGATIVE,
        RejectionReason.FILL_SLIPPAGE,
        RejectionReason.EVENT_CAPPED,
        RejectionReason.PASSED,
    }
)

_REJECTION_REASONS_WALKED: list[RejectionReason] = [
    r for r in RejectionReason if r not in _REJECTION_REASONS_NOT_WALKED
]


@pytest.mark.parametrize("reason", _REJECTION_REASONS_WALKED)
def test_infer_rejection_reason_every_walked_reason(reason: RejectionReason) -> None:
    """For every RejectionReason the verifier expects to surface,
    incrementing its corresponding ScanFunnel counter MUST round-trip
    through _infer_rejection_reason. Catches the Fix A defect class —
    a new enum value with no counter-walker entry would return None.
    """
    from python.analytics.audit_log import funnel_attr_for

    attr = funnel_attr_for(reason)
    assert attr is not None, (
        f"RejectionReason.{reason.name} has no ScanFunnel counter mapping; "
        f"check python.analytics.audit_log._REASON_TO_FUNNEL_ATTR."
    )
    funnel = ScanFunnel()
    setattr(funnel, attr, 1)
    inferred = _infer_rejection_reason(funnel)
    assert inferred == reason.value, (
        f"RejectionReason.{reason.name} (counter {attr}) did not round-trip "
        f"through _infer_rejection_reason; got {inferred!r}. "
        f"Add ({attr!r}, RejectionReason.{reason.name}.value) to "
        f"verify_intraday_replay._infer_rejection_reason.counter_to_reason."
    )
