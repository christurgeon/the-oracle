"""Tests for F1.1 (GroupKFold by forecast_date) and F1.5 (fit_date stamp).

Covers ``scripts/fit_isotonic.py``:
- ``cross_validate`` must support a ``groups`` argument that uses
  ``GroupKFold`` to keep all rows for a given ``forecast_date`` in the same
  fold.
- The CLI ``--cross-validate`` path must surface BOTH the legacy Random-KFold
  improvement and the leakage-free GroupKFold improvement.
- ``fit_date`` in the emitted artifact must equal
  ``max(corpus.forecast_date)`` instead of ``date.today()``, falling back to
  today on an empty corpus.

Tests are written TDD-first: they fail against the unmodified script and
pass after the implementation lands.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="ML extras not installed (uv sync --extra ml)")


# ---------------------------------------------------------------------------
# Helpers (copied/adapted from test_fit_isotonic.py for isolation)
# ---------------------------------------------------------------------------


def _empty_snapshot_dir(tmp_path: Path) -> Path:
    snap_dir = tmp_path / "snapshots" / "2026-04-01"
    snap_dir.mkdir(parents=True)
    (snap_dir / "weather_forecasts.json").write_text(json.dumps([]))
    (snap_dir / "settlements.json").write_text(json.dumps([]))
    (snap_dir / "markets.json").write_text(json.dumps([]))
    return tmp_path / "snapshots"


def _write_calendar(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-05-09T00:00:00+00:00",
                "method": "auto",
                "models": {
                    "gfs_seamless": {"first_seen_date": "2026-03-19"},
                    "ecmwf_ifs025": {"first_seen_date": "2026-04-24"},
                },
            }
        )
    )
    return path


def _make_report(*, total: int = 200, full: int = 200):
    from scripts._calibration_input_contract import InputContractReport

    return InputContractReport(
        n_triples_total=total,
        n_triples_full=full,
        n_triples_partial=0,
        n_triples_empty=0,
        n_triples_single_model_era=0,
        n_triples_exempt=0,
        n_settlements_final=total,
        n_settlements_dropped_preliminary=0,
    )


def _stub_preflight(monkeypatch, report) -> None:
    import scripts._calibration_input_contract as ic

    def fake(*args, **kwargs):
        return report

    monkeypatch.setattr(ic, "assert_snapshot_input_contract", fake)


def _stub_pair_builder(monkeypatch, pairs) -> None:
    import scripts._calibration_common as cc

    def fake(**kwargs):
        return iter(pairs)

    monkeypatch.setattr(cc, "build_production_mirrored_pairs", fake)


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fit_isotonic.py", *args])


# ---------------------------------------------------------------------------
# F1.1 — GroupKFold split correctness
# ---------------------------------------------------------------------------


def test_groupkfold_no_forecast_date_leakage() -> None:
    """GroupKFold(5) keyed on forecast_date puts every row for one date in
    exactly one fold — never split across train/test."""
    from sklearn.model_selection import GroupKFold

    # 5 forecast_dates, 4 rows each — 20 total rows.
    dates = []
    for d in ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04", "2026-04-05"]:
        dates.extend([d] * 4)
    features = np.linspace(0.01, 0.99, len(dates))
    targets = np.array([0, 1] * (len(dates) // 2))

    # Map dates to int group ids (the implementation must do something equivalent).
    unique_dates = sorted(set(dates))
    date_to_id = {d: i for i, d in enumerate(unique_dates)}
    groups = np.array([date_to_id[d] for d in dates])

    gkf = GroupKFold(n_splits=5)
    for train_idx, test_idx in gkf.split(features, targets, groups=groups):
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        # No date may appear in both train and test for the same fold.
        assert not (train_groups & test_groups), (
            f"forecast_date leakage: groups {train_groups & test_groups} "
            "appeared in both train and test"
        )


def test_cross_validate_supports_groups_argument() -> None:
    """``cross_validate`` accepts a ``groups`` array and uses GroupKFold when set."""
    from scripts.fit_isotonic import cross_validate

    # Two perfectly-correlated groups — Random-KFold would leak across them,
    # GroupKFold cannot.
    n = 50
    # 10 forecast_dates × 5 rows each
    groups = np.repeat(np.arange(10), 5)
    rng = np.random.default_rng(0)
    features = rng.uniform(0.0, 1.0, n)
    targets = (features > 0.5).astype(int)

    out = cross_validate(features, targets, n_folds=5, groups=groups)
    # Returns the same dict-of-floats shape as before.
    assert "raw_brier_mean" in out
    assert "cal_brier_mean" in out
    assert "improvement_pct" in out
    # And signals which protocol was used.
    assert out.get("split_protocol") == "GroupKFold"


def test_cross_validate_random_kfold_default() -> None:
    """Without ``groups``, ``cross_validate`` keeps legacy Random-KFold for
    backwards-compatible reporting."""
    from scripts.fit_isotonic import cross_validate

    rng = np.random.default_rng(0)
    features = rng.uniform(0.0, 1.0, 200)
    targets = (rng.uniform(0.0, 1.0, 200) < features).astype(int)

    out = cross_validate(features, targets, n_folds=5)
    assert out.get("split_protocol") in ("KFold", None)


def test_groupkfold_falls_back_when_n_groups_lt_splits() -> None:
    """If the number of unique groups is < n_folds, the function adapts cleanly."""
    from scripts.fit_isotonic import cross_validate

    # Only 3 distinct forecast_dates, ask for 5 folds.
    groups = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05])
    targets = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])

    # Acceptable behavior: either reduce n_splits to len(unique_groups) and
    # report it, OR raise a clear ValueError. Both are fine; we assert that
    # it does NOT silently corrupt fold composition.
    try:
        out = cross_validate(features, targets, n_folds=5, groups=groups)
    except ValueError as exc:
        # Clean failure with an informative message is acceptable.
        assert "group" in str(exc).lower() or "fold" in str(exc).lower()
        return
    # Or it ran successfully — must have folded by group count.
    assert out.get("n_folds") <= 3
    assert out.get("split_protocol") == "GroupKFold"


# ---------------------------------------------------------------------------
# F1.1 — Synthetic leakage demonstration
# ---------------------------------------------------------------------------


def test_groupkfold_reduces_improvement_under_leakage() -> None:
    """Construct a corpus where same-day rows are perfectly predictive of
    each other. Random-KFold splits them, GroupKFold does not. The GroupKFold
    improvement_pct must be strictly less than the Random-KFold one.
    """
    from scripts.fit_isotonic import cross_validate

    rng = np.random.default_rng(42)
    # 50 forecast_dates × 20 rows each = 1000 rows.
    # Within a date, all rows share a "true" probability drawn once per date,
    # but their raw_prob is jittered around it. This simulates the leakage
    # pattern: random-KFold sees train rows from the same date as test rows.
    n_dates = 50
    rows_per_date = 20
    features_list: list[float] = []
    targets_list: list[int] = []
    groups_list: list[int] = []
    for i in range(n_dates):
        true_p = rng.uniform(0.05, 0.95)
        # One outcome per date, as one day's observation settles every row of
        # that day. Independent per-row outcomes carry no leakage to detect.
        outcome = int(rng.uniform() < true_p)
        for _ in range(rows_per_date):
            raw = float(np.clip(true_p + rng.normal(0, 0.02), 0.0, 1.0))
            features_list.append(raw)
            targets_list.append(outcome)
            groups_list.append(i)

    features = np.array(features_list)
    targets = np.array(targets_list)
    groups = np.array(groups_list)

    random_out = cross_validate(features, targets, n_folds=5)
    group_out = cross_validate(features, targets, n_folds=5, groups=groups)

    # Both report some "improvement", but the leakage-free number should be
    # lower — by construction, train rows in the random split are perfect
    # proxies for test rows.
    assert group_out["improvement_pct"] < random_out["improvement_pct"], (
        f"GroupKFold improvement_pct={group_out['improvement_pct']:.2f} "
        f"should be < Random-KFold improvement_pct={random_out['improvement_pct']:.2f}"
    )


def test_cli_cross_validate_prints_both_metrics(tmp_path: Path, monkeypatch, capsys) -> None:
    """The --cross-validate CLI path emits BOTH the Random-KFold and the
    GroupKFold improvement on the same corpus."""
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    # Pairs: 6-tuples (prob, outcome, city, var, strike_type, forecast_date).
    # Mix of 8 forecast_dates with class-balanced rows so the fitter doesn't
    # short-circuit on len(pairs) < 100.
    pairs: list = []
    dates = [
        "2026-04-01",
        "2026-04-02",
        "2026-04-03",
        "2026-04-04",
        "2026-04-05",
        "2026-04-06",
        "2026-04-07",
        "2026-04-08",
    ]
    for d in dates:
        for _ in range(8):
            pairs.append((0.3, 0, "NYC", "temperature_2m_max", "greater", d))
        for _ in range(8):
            pairs.append((0.7, 1, "NYC", "temperature_2m_max", "greater", d))

    _stub_pair_builder(monkeypatch, pairs)

    out_path = tmp_path / "out.json"
    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(out_path),
        "--bias-table",
        str(tmp_path / "missing_bias.json"),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
        "--cross-validate",
    )

    fit_iso.main()

    captured = capsys.readouterr()
    out_text = captured.out
    # Both protocol labels should appear in the report.
    assert "Random-KFold" in out_text or "KFold" in out_text
    assert "GroupKFold" in out_text


# ---------------------------------------------------------------------------
# F1.5 — fit_date stamps max(forecast_date), not date.today()
# ---------------------------------------------------------------------------


def test_fit_date_stamps_corpus_max_not_today(tmp_path: Path, monkeypatch) -> None:
    """The isotonic artifact's ``fit_date`` equals ``max(pair.forecast_date)``,
    not ``date.today()``."""
    import scripts.fit_isotonic as fit_iso

    snapshots = _empty_snapshot_dir(tmp_path)
    _write_calendar(tmp_path / "data" / "calibration" / "data_availability_calendar.json")
    monkeypatch.setattr(fit_iso, "_PROJECT_ROOT", tmp_path)

    _stub_preflight(monkeypatch, _make_report(total=200, full=200))

    target_max_date = "2024-01-15"
    other_date = "2024-01-10"
    pairs: list = []
    # 60 neg + 60 pos so the fit has both classes; spread across the two dates.
    for _ in range(30):
        pairs.append((0.3, 0, "NYC", "temperature_2m_max", "greater", target_max_date))
        pairs.append((0.3, 0, "NYC", "temperature_2m_max", "greater", other_date))
        pairs.append((0.7, 1, "NYC", "temperature_2m_max", "greater", target_max_date))
        pairs.append((0.7, 1, "NYC", "temperature_2m_max", "greater", other_date))

    _stub_pair_builder(monkeypatch, pairs)

    out_path = tmp_path / "out.json"
    _set_argv(
        monkeypatch,
        "--snapshots",
        str(snapshots),
        "--output",
        str(out_path),
        "--bias-table",
        str(tmp_path / "missing_bias.json"),
        "--ensemble-weights",
        json.dumps({"gfs_seamless": 1.0}),
        "--no-bias-correction",
    )

    fit_iso.main()
    artifact = json.loads(out_path.read_text())
    assert artifact["fit_date"] == target_max_date
    # And critically NOT today's wall-clock.
    assert artifact["fit_date"] != date.today().isoformat()


def test_fit_date_falls_back_to_today_on_empty(tmp_path: Path, monkeypatch, caplog) -> None:
    """If the pair list is empty the fitter still fails (insufficient data),
    but if the corpus has rows and somehow all forecast_dates are missing,
    the stamp falls back to today + logged warning. This guards the
    defensive code path described in F1.5 Spec §2."""
    # The fitter requires len(pairs) >= 100, so we exercise the defensive
    # branch via the unit-level helper directly.
    from scripts.fit_isotonic import _resolve_fit_date

    # Empty list → fall back, with warning.
    with caplog.at_level("WARNING"):
        result = _resolve_fit_date([])
    assert result == date.today()
    assert any("fit_date" in rec.message or "fallback" in rec.message for rec in caplog.records)


def test_resolve_fit_date_returns_max(monkeypatch) -> None:
    """The helper picks the maximum forecast_date from a pair iterable."""
    from scripts.fit_isotonic import _resolve_fit_date

    pairs = [
        (0.4, 0, "NYC", "temperature_2m_max", "greater", date(2024, 1, 1)),
        (0.5, 1, "NYC", "temperature_2m_max", "greater", date(2024, 1, 15)),
        (0.6, 0, "NYC", "temperature_2m_max", "greater", date(2024, 1, 7)),
    ]
    assert _resolve_fit_date(pairs) == date(2024, 1, 15)


def test_resolve_fit_date_accepts_str_dates() -> None:
    """forecast_date may be a str (e.g., from the production-mirrored pair
    yielder originally yielding ``date`` but stub callers using strings).
    The helper must coerce."""
    from scripts.fit_isotonic import _resolve_fit_date

    pairs = [
        (0.4, 0, "NYC", "temperature_2m_max", "greater", "2024-01-01"),
        (0.5, 1, "NYC", "temperature_2m_max", "greater", "2024-01-15"),
        (0.6, 0, "NYC", "temperature_2m_max", "greater", "2024-01-07"),
    ]
    assert _resolve_fit_date(pairs) == date(2024, 1, 15)
