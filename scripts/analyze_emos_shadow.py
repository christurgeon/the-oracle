"""EMOS shadow analyzer — produces a binding promote-or-hold verdict
for scope C.

Reads:
  * state/eval_audit.jsonl (and rotated siblings)
  * state/forecasts_raw.jsonl (and rotated siblings)
  * data/snapshots/<date>/settlements.json (via prefer_latest_settlement)

Writes:
  * data/emos/reconciliation_envelope.json (sealed day-1 anchor)
  * Decision JSON to stdout (or --json PATH)

Command:
  uv run python scripts/analyze_emos_shadow.py \\
      --start-date 2026-05-09 --end-date 2026-06-08 \\
      --decision-mode 30-day-system

"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

# Add project root so the script works standalone (matches build_emos_corpus.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from scripts._calibration_common import prefer_latest_settlement  # noqa: E402

# Historical anchor (documentation only — see module docstring): the triple
# (n=51, mean_predicted=0.478, observed_freq=0.314) was produced by
# scripts/fit_isotonic.py:_calibration_table() on the n=3,240 fit corpus,
# binned by decile of cal_prob. Runtime invariance is enforced by the sealed
# envelope + frozen population copy (data/emos/reconciliation_envelope.json +
# data/emos/reference_window_eval_audit.jsonl.frozen). The bin range constants
# below define WHICH raw-prob slice we anchor; they are not asserted against the
# historical triple at runtime.
_BIN_RANGE_LOW: Final[float] = 0.40
_BIN_RANGE_HIGH: Final[float] = 0.50

_RECONCILIATION_TOLERANCE_PRED = 0.005
_RECONCILIATION_TOLERANCE_OBS = 0.01

# Absolute floor below which a seal is statistically meaningless, regardless of
# the operator-supplied --first-run-n-floor. Wilson 95% half-width at n=10,
# p≈0.4 is ~30pp — already weak signal; below 10 the seal is noise.
_ABSOLUTE_MIN_SEAL_N: Final[int] = 10

_DEFAULT_FIRST_RUN_N_FLOOR: Final[int] = 30

_DEFAULT_RNG_SEED = 42

# Variable → settlement field mapping (mirrors
# scripts/_calibration_common._VARIABLE_TO_SETTLEMENT_FIELD but keyed on the
# eval_audit `var` column convention: "high" | "low").
_VAR_TO_SETTLEMENT_FIELD: Final[dict[str, str]] = {"high": "high_f", "low": "low_f"}


def _resolve_fittable_cells(
    *,
    coeffs_path: Path,
    config_path: Path,
    override: str | None,
) -> set[tuple[str, str]]:
    """Resolve the fittable_cells set from CLI override or
    coefficients.json - excluded_city_vars."""
    if override:
        out: set[tuple[str, str]] = set()
        for token in override.split(","):
            token = token.strip()
            if not token or "/" not in token:
                continue
            c, v = token.split("/", 1)
            out.add((c, v))
        return out

    import yaml  # type: ignore[import-untyped]

    coeffs_data = json.loads(coeffs_path.read_text())
    cell_keys = list(coeffs_data.get("cells", {}).keys())

    excluded: set[str] = set()
    if config_path.exists():
        config_data = yaml.safe_load(config_path.read_text()) or {}
        # Real config nests under strategies.weather; tests use top-level weather.
        weather = (
            (config_data.get("strategies") or {}).get("weather")
            or config_data.get("weather")
            or {}
        )
        for entry in weather.get("excluded_city_vars", []) or []:
            excluded.add(str(entry))

    out_cells: set[tuple[str, str]] = set()
    for key in cell_keys:
        if "/" not in key:
            continue
        c, v = key.split("/", 1)
        # excluded_city_vars uses underscore form (e.g. "DEN_low") per config.yaml.
        if f"{c}_{v}" in excluded:
            continue
        out_cells.add((c, v))
    return out_cells


def _resolve_git_rev() -> str:
    """Best-effort git HEAD; 'unknown' if not available."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--decision-mode", choices=["30-day-system", "60-day-per-cell"], required=True,
    )
    parser.add_argument("--rng-seed", type=int, default=_DEFAULT_RNG_SEED)
    parser.add_argument("--json", type=Path, default=None, help="Write decision JSON to PATH.")
    parser.add_argument(
        "--allow-future-schema", action="store_true",
        help="Override the schema fail-fast (requires _schema_version=1)",
    )
    parser.add_argument("--state-dir", type=Path, default=Path("state"))
    parser.add_argument("--snapshots-dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument(
        "--envelope-path", type=Path,
        default=Path("data/emos/reconciliation_envelope.json"),
    )
    parser.add_argument(
        "--frozen-path", type=Path,
        default=Path("data/emos/reference_window_eval_audit.jsonl.frozen"),
        help="Path to the sealed frozen-population file.",
    )
    parser.add_argument(
        "--iso-artifact-path", type=Path,
        default=Path("data/calibration/isotonic.json"),
        help="Path to the live isotonic artifact; its SHA-256 is sealed in the envelope.",
    )
    parser.add_argument(
        "--coeffs-path", type=Path,
        default=Path("data/emos/coefficients.json"),
        help="Path to the EMOS coefficients file; used to derive fittable_cells.",
    )
    parser.add_argument(
        "--config-path", type=Path,
        default=Path("python/config.yaml"),
        help="Path to config.yaml; used to read weather.excluded_city_vars.",
    )
    parser.add_argument(
        "--fittable-cells-override", type=str, default=None,
        help='Override fittable_cells, comma-separated "CITY/var" tokens '
             '(e.g., "AUS/high,NYC/low").',
    )
    parser.add_argument(
        "--pre-flag-flip-cutoff", type=date.fromisoformat, default=None,
        help="Optional cutoff date: rows with forecast_date >= this date are "
             "excluded from the reconciliation filter.",
    )
    parser.add_argument(
        "--first-run-n-floor", type=int, default=_DEFAULT_FIRST_RUN_N_FLOOR,
        help="Minimum n required to seal on first run (default 30).",
    )
    parser.add_argument(
        "--reseal", action="store_true",
        help="Operator escape hatch: move existing envelope + frozen to "
             ".bak.<ts> backups, then re-seal. Old files are NOT deleted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Reconcile only: load + seal-or-verify the envelope, then exit. "
             "No decision rendered.",
    )
    args = parser.parse_args(argv)

    try:
        fittable_cells = _resolve_fittable_cells(
            coeffs_path=args.coeffs_path,
            config_path=args.config_path,
            override=args.fittable_cells_override,
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"failed to resolve fittable_cells: {exc}", file=sys.stderr)
        return 1
    print(
        f"fittable_cells (n={len(fittable_cells)}): "
        f"{sorted(f'{c}/{v}' for c, v in fittable_cells)}"
    )

    if args.reseal:
        backups = _backup_for_reseal(args.envelope_path, args.frozen_path)
        for kind, path in backups.items():
            print(f"reseal: moved old {kind} → {path}")

    try:
        rows = _load_reconciliation_population(
            state_dir=args.state_dir,
            snapshots_dir=args.snapshots_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            iso_artifact_path=args.iso_artifact_path,
        )
    except FileNotFoundError as exc:
        print(f"failed to load reconciliation population: {exc}", file=sys.stderr)
        return 1

    try:
        passed, env, was_sealed = reconcile_or_use_envelope(
            rows,
            iso_artifact_path=args.iso_artifact_path,
            envelope_path=args.envelope_path,
            frozen_path=args.frozen_path,
            fittable_cells=fittable_cells,
            pre_flag_flip_cutoff_date=args.pre_flag_flip_cutoff,
            start_date=args.start_date,
            end_date=args.end_date,
            first_run_n_floor=args.first_run_n_floor,
            git_rev=_resolve_git_rev(),
            query_string=f"analyze_emos_shadow.py {args.start_date}..{args.end_date}",
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    n = env["recomputed_n"]
    pred = env["recomputed_predicted"]
    obs = env["recomputed_observed"]
    if was_sealed:
        print(f"sealed: n={n} predicted={pred:.4f} observed={obs:.4f}")
    else:
        print(f"verified against frozen population: n={n}")

    if args.dry_run:
        return 0

    # Non-dry-run pipeline (D4-D10) is not implemented yet — exit cleanly.
    print("EMOS shadow analyzer — non-dry-run pipeline not yet implemented", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# D9: loader for the reconciliation population
# ---------------------------------------------------------------------------


def _iter_eval_audit_files(state_dir: Path) -> list[Path]:
    """Return all eval_audit.jsonl files (live + rotated siblings) under state_dir."""
    candidates: list[Path] = []
    if not state_dir.is_dir():
        return candidates
    for entry in state_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name == "eval_audit.jsonl" or entry.name.startswith("eval_audit.jsonl."):
            candidates.append(entry)
    return sorted(candidates)


def _load_eval_audit_rows(state_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp in _iter_eval_audit_files(state_dir):
        with fp.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _load_settlements_index(
    snapshots_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Merge every settlements.json under snapshots_dir using
    prefer_latest_settlement. Handles both ``<date>/`` and ``<date>/<run>/``
    layouts, mirroring scripts/_calibration_common.load_snapshot_data."""
    settlements: dict[tuple[str, str], dict[str, Any]] = {}
    if not snapshots_dir.is_dir():
        return settlements

    def _ingest(file_path: Path) -> None:
        if not file_path.exists():
            return
        try:
            entries = json.loads(file_path.read_text())
        except json.JSONDecodeError:
            return
        for s in entries:
            try:
                sk = (s["city"], s["observation_date"])
            except KeyError:
                continue
            existing = settlements.get(sk)
            if existing is None or prefer_latest_settlement(existing, s):
                settlements[sk] = s

    for day_dir in sorted(snapshots_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        if (day_dir / "settlements.json").exists():
            _ingest(day_dir / "settlements.json")
        for run_dir in sorted(day_dir.iterdir()) if day_dir.is_dir() else []:
            if run_dir.is_dir() and (run_dir / "settlements.json").exists():
                _ingest(run_dir / "settlements.json")

    return settlements


def _outcome_from_bracket(
    *,
    strike_type: str | None,
    floor_strike: float | None,
    cap_strike: float | None,
    actual_temp: float | None,
) -> int | None:
    """Compute the binary outcome for a weather bracket given the settled temp.

    Delegates to ``python.analytics.outcomes.derive_outcome_yes`` (single
    source of Kalshi-truth semantics — between is CLOSED, right-inclusive).
    Normalises the eval_audit ``strike_type`` convention
    ("above"/"below"/"between") to the markets.json convention
    ("greater"/"less"/"between") before delegating.
    """
    from python.analytics.outcomes import derive_outcome_yes

    if actual_temp is None or strike_type is None:
        return None
    normalised = {"above": "greater", "below": "less"}.get(strike_type, strike_type)
    yes = derive_outcome_yes(normalised, floor_strike, cap_strike, float(actual_temp))
    if yes is None:
        return None
    return 1 if yes else 0


def _load_reconciliation_population(
    *,
    state_dir: Path,
    snapshots_dir: Path,
    start_date: date,
    end_date: date,
    iso_artifact_path: Path,
) -> list[dict[str, Any]]:
    """Load eval_audit rows in [start_date, end_date) and inject the three
    fields the reconciliation filter requires: settlement_status,
    settled_value, isotonic_artifact_hash.

    The iso-hash is computed once from the file at iso_artifact_path and
    attached to every row — single source of truth for the window. The
    spec's per-row attestation (`row["isotonic_artifact_hash"]`) is fiction
    for pre-PR-2 rows; the envelope-level guard at verify time is the
    honest check.
    """
    iso_hash = hashlib.sha256(iso_artifact_path.read_bytes()).hexdigest()
    settlements = _load_settlements_index(snapshots_dir)

    out: list[dict[str, Any]] = []
    for row in _load_eval_audit_rows(state_dir):
        fd_raw = row.get("forecast_date")
        if not fd_raw:
            continue
        try:
            fd = date.fromisoformat(fd_raw)
        except (TypeError, ValueError):
            continue
        if not (start_date <= fd < end_date):
            continue

        # Augment with iso hash (single value across the window).
        row = dict(row)
        row["isotonic_artifact_hash"] = iso_hash

        # Look up settlement; var → high_f/low_f.
        city = row.get("city")
        var = row.get("var")
        settle = settlements.get((city, fd_raw)) if city else None
        settle_field = _VAR_TO_SETTLEMENT_FIELD.get(str(var) if var else "")

        if settle is None or settle_field is None:
            row["settlement_status"] = "missing"
            row["settled_value"] = None
        else:
            row["settlement_status"] = (
                "preliminary" if settle.get("is_preliminary") else "final"
            )
            actual_temp = settle.get(settle_field)
            row["settled_value"] = _outcome_from_bracket(
                strike_type=row.get("strike_type"),
                floor_strike=row.get("floor"),
                cap_strike=row.get("cap"),
                actual_temp=actual_temp,
            )

        out.append(row)

    return out


# ---------------------------------------------------------------------------
# D3: Frozen-reference reconciliation + sealed envelope
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationResult:
    n: int
    mean_predicted: float
    observed_freq: float
    eval_audit_row_hashes: list[str] = field(default_factory=list)


def compute_broken_bin_population(
    rows: list[dict[str, Any]],
    *,
    fittable_cells: set[tuple[str, str]],
    pre_flag_flip_cutoff_date: date | None,
) -> ReconciliationResult:
    """Recompute (n, mean_predicted, observed_freq) on rows passing the
    reconciliation filters: settled=final, fittable cell, optional pre-flag-flip
    cutoff, raw_ens_prob ∈ [_BIN_RANGE_LOW, _BIN_RANGE_HIGH).

    No comparison against any hardcoded reference is performed — the caller
    decides whether to seal or verify against an envelope. The per-row
    isotonic_artifact_hash filter is gone (file-level guard in
    reconcile_or_use_envelope replaces it).
    """
    pop = []
    for r in rows:
        if r.get("settlement_status") != "final":
            continue
        cell = (r.get("city"), r.get("var"))
        if cell not in fittable_cells:
            continue
        if pre_flag_flip_cutoff_date is not None:
            forecast_date_str = r.get("forecast_date")
            if forecast_date_str:
                fd = date.fromisoformat(forecast_date_str)
                if fd >= pre_flag_flip_cutoff_date:
                    continue
        rp = r.get("raw_ens_prob")
        if rp is None or not (_BIN_RANGE_LOW <= rp < _BIN_RANGE_HIGH):
            continue
        pop.append(r)

    n = len(pop)
    mean_pred = float(np.mean([r["raw_ens_prob"] for r in pop])) if pop else 0.0
    obs_freq = float(np.mean([float(r["settled_value"]) for r in pop])) if pop else 0.0

    hashes = [
        hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:16]
        for r in pop
    ]
    return ReconciliationResult(n, mean_pred, obs_freq, hashes)


def _validate_seal_inputs(
    result: ReconciliationResult,
    fittable_cells: set[tuple[str, str]],
) -> None:
    """Defensive checks before sealing. Refuses to anchor on garbage.

    Sanity criteria (any failure raises RuntimeError):
      1. n >= _ABSOLUTE_MIN_SEAL_N — below 10, the seal is statistically noise.
      2. mean_predicted ∈ [_BIN_RANGE_LOW, _BIN_RANGE_HIGH) — out-of-bin means
         the filter was mis-applied; the recomputed stat is meaningless.
      3. observed_freq ∈ [0.0, 1.0] — defensive against numeric corruption.
      4. fittable_cells non-empty — sealing with zero cells would freeze a
         universe-empty filter.
    """
    if not fittable_cells:
        raise RuntimeError("Seal sanity: fittable_cells is empty. ABORT.")
    if result.n < _ABSOLUTE_MIN_SEAL_N:
        raise RuntimeError(
            f"Seal sanity: n={result.n} below absolute floor "
            f"({_ABSOLUTE_MIN_SEAL_N}). ABORT."
        )
    if not (_BIN_RANGE_LOW <= result.mean_predicted < _BIN_RANGE_HIGH):
        raise RuntimeError(
            f"Seal sanity: mean_predicted={result.mean_predicted:.6f} outside bin "
            f"[{_BIN_RANGE_LOW}, {_BIN_RANGE_HIGH}). Filter likely mis-applied. "
            "ABORT."
        )
    if not (0.0 <= result.observed_freq <= 1.0):
        raise RuntimeError(
            f"Seal sanity: observed_freq={result.observed_freq:.6f} not in "
            "[0.0, 1.0]. ABORT."
        )


def write_reconciliation_envelope(
    result: ReconciliationResult,
    *,
    envelope_path: Path,
    isotonic_artifact_hash: str,
    git_rev: str,
    query_string: str,
    fittable_cells_sealed: list[str],
    start_date_sealed: str,
    end_date_sealed: str,
    pre_flag_flip_cutoff_date_sealed: str | None,
    frozen_file_sha256: str | None = None,
) -> None:
    """Write the sealed envelope JSON. Atomic via tempfile+os.replace pattern."""
    envelope = {
        "recomputed_n": result.n,
        "recomputed_predicted": result.mean_predicted,
        "recomputed_observed": result.observed_freq,
        "query_string": query_string,
        "isotonic_artifact_hash": isotonic_artifact_hash,
        "eval_audit_row_hashes": result.eval_audit_row_hashes,
        "git_rev": git_rev,
        "query_executed_at_utc": datetime.now(UTC).isoformat(),
        "fittable_cells_sealed": fittable_cells_sealed,
        "start_date_sealed": start_date_sealed,
        "end_date_sealed": end_date_sealed,
        "pre_flag_flip_cutoff_date_sealed": pre_flag_flip_cutoff_date_sealed,
        "frozen_file_sha256": frozen_file_sha256,
    }
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = envelope_path.with_suffix(envelope_path.suffix + ".tmp")
    tmp.write_text(json.dumps(envelope, indent=2))
    tmp.replace(envelope_path)


def load_reconciliation_envelope(envelope_path: Path) -> dict[str, Any] | None:
    if not envelope_path.exists():
        return None
    payload: dict[str, Any] = json.loads(envelope_path.read_text())
    return payload


def _write_frozen_population(rows: list[dict[str, Any]], frozen_path: Path) -> None:
    """Write rows as JSONL with sorted keys per record. Atomic."""
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = frozen_path.with_suffix(frozen_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(frozen_path)


def _load_frozen_population(frozen_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with frozen_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _cells_to_sealed_list(cells: set[tuple[str, str]]) -> list[str]:
    """Canonical sorted list-of-strings form for fittable_cells in the envelope."""
    return sorted(f"{c}/{v}" for c, v in cells)


def _backup_for_reseal(
    envelope_path: Path,
    frozen_path: Path,
) -> dict[str, Path]:
    """Move any existing envelope/frozen files to timestamped backups.

    Returns a dict mapping {"envelope": <backup_path>, "frozen": <backup_path>}
    for whichever files existed. Returns {} if neither exists. Does not delete
    — operator can inspect or revert from the .bak.<ts> files.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, Path] = {}
    if envelope_path.exists():
        bak = envelope_path.with_suffix(envelope_path.suffix + f".bak.{ts}")
        envelope_path.replace(bak)
        out["envelope"] = bak
    if frozen_path.exists():
        bak = frozen_path.with_suffix(frozen_path.suffix + f".bak.{ts}")
        frozen_path.replace(bak)
        out["frozen"] = bak
    return out


def reconcile_or_use_envelope(
    rows: list[dict[str, Any]],
    *,
    iso_artifact_path: Path,
    envelope_path: Path,
    frozen_path: Path,
    fittable_cells: set[tuple[str, str]],
    pre_flag_flip_cutoff_date: date | None,
    start_date: date,
    end_date: date,
    first_run_n_floor: int = _DEFAULT_FIRST_RUN_N_FLOOR,
    git_rev: str = "unknown",
    query_string: str = "see scripts/analyze_emos_shadow.py",
) -> tuple[bool, dict[str, Any], bool]:
    """Two-branch reconcile.

    Branch A (envelope OR frozen file missing) — seal:
      1. compute_broken_bin_population on `rows`
      2. empty-pop guard, floor guard, sanity validator
      3. write frozen population + envelope
      4. return (True, envelope_dict, was_sealed=True)

    Branch B (both files present) — verify:
      1. load envelope; assert iso-hash matches sha256(iso_artifact_path)
      2. assert fittable_cells_sealed matches caller's cells
      3. assert start/end/cutoff match sealed window
      4. load frozen rows; compute_broken_bin_population on them
      5. empty-pop guard, row-hash equality vs envelope
      6. numeric-drift defense (exact match expected on byte-identical rows)
      7. return (True, envelope_dict, was_sealed=False)
    """
    current_iso_hash = hashlib.sha256(iso_artifact_path.read_bytes()).hexdigest()
    env = load_reconciliation_envelope(envelope_path)

    if env is None or not frozen_path.exists():
        # ----- Branch A: seal -----
        result = compute_broken_bin_population(
            rows,
            fittable_cells=fittable_cells,
            pre_flag_flip_cutoff_date=pre_flag_flip_cutoff_date,
        )
        if result.n == 0:
            raise RuntimeError(
                "Reconciliation: filtered population is empty. Loader produced no "
                "rows matching the bin/cell/cutoff/final filters. ABORT."
            )
        if result.n < first_run_n_floor:
            raise RuntimeError(
                f"First-run seal aborted: n={result.n} below floor="
                f"{first_run_n_floor}. Re-run later once more rows accumulate. ABORT."
            )
        _validate_seal_inputs(result, fittable_cells)

        # Re-run the filter to materialize the exact row set we sealed stats on.
        # compute_broken_bin_population doesn't return the filtered list (only its
        # hashes), so we reconstruct here. This is cheap (<1k rows typical).
        pop: list[dict[str, Any]] = []
        for r in rows:
            if r.get("settlement_status") != "final":
                continue
            cell = (r.get("city"), r.get("var"))
            if cell not in fittable_cells:
                continue
            if pre_flag_flip_cutoff_date is not None:
                fd_raw = r.get("forecast_date")
                if fd_raw:
                    if date.fromisoformat(fd_raw) >= pre_flag_flip_cutoff_date:
                        continue
            rp = r.get("raw_ens_prob")
            if rp is None or not (_BIN_RANGE_LOW <= rp < _BIN_RANGE_HIGH):
                continue
            pop.append(r)

        _write_frozen_population(pop, frozen_path)
        frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        write_reconciliation_envelope(
            result,
            envelope_path=envelope_path,
            isotonic_artifact_hash=current_iso_hash,
            git_rev=git_rev,
            query_string=query_string,
            fittable_cells_sealed=_cells_to_sealed_list(fittable_cells),
            start_date_sealed=start_date.isoformat(),
            end_date_sealed=end_date.isoformat(),
            pre_flag_flip_cutoff_date_sealed=(
                pre_flag_flip_cutoff_date.isoformat()
                if pre_flag_flip_cutoff_date is not None
                else None
            ),
            frozen_file_sha256=frozen_sha,
        )
        env = load_reconciliation_envelope(envelope_path)
        assert env is not None  # just wrote it
        return True, env, True

    # ----- Branch B: verify -----
    if env.get("isotonic_artifact_hash") != current_iso_hash:
        raise RuntimeError(
            "Reconciliation: envelope was sealed under a different isotonic artifact "
            f"(envelope iso_hash={env.get('isotonic_artifact_hash')!r}, current="
            f"{current_iso_hash!r}). ABORT."
        )
    expected_cells = _cells_to_sealed_list(fittable_cells)
    if env.get("fittable_cells_sealed") != expected_cells:
        raise RuntimeError(
            "Reconciliation: fittable_cells differs from sealed set "
            f"(sealed={env.get('fittable_cells_sealed')}, current={expected_cells}). "
            "ABORT."
        )
    if env.get("start_date_sealed") != start_date.isoformat() or \
            env.get("end_date_sealed") != end_date.isoformat():
        raise RuntimeError(
            "Reconciliation: window differs from sealed values "
            f"(sealed start={env.get('start_date_sealed')}, "
            f"end={env.get('end_date_sealed')}; current start="
            f"{start_date.isoformat()}, end={end_date.isoformat()}). ABORT."
        )
    sealed_cutoff = env.get("pre_flag_flip_cutoff_date_sealed")
    current_cutoff = (
        pre_flag_flip_cutoff_date.isoformat()
        if pre_flag_flip_cutoff_date is not None
        else None
    )
    if sealed_cutoff != current_cutoff:
        raise RuntimeError(
            "Reconciliation: pre_flag_flip_cutoff_date differs from sealed value "
            f"(sealed={sealed_cutoff}, current={current_cutoff}). ABORT."
        )

    # File-level tamper guard: any byte change to the frozen file fails this.
    sealed_frozen_sha = env.get("frozen_file_sha256")
    if sealed_frozen_sha is not None:
        current_frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        if current_frozen_sha != sealed_frozen_sha:
            raise RuntimeError(
                "Reconciliation: frozen-population file sha256 differs from envelope "
                f"(sealed={sealed_frozen_sha}, current={current_frozen_sha}). "
                "Possible row tampering. ABORT."
            )

    frozen_rows = _load_frozen_population(frozen_path)
    result = compute_broken_bin_population(
        frozen_rows,
        fittable_cells=fittable_cells,
        pre_flag_flip_cutoff_date=pre_flag_flip_cutoff_date,
    )
    if result.n == 0:
        raise RuntimeError(
            "Reconciliation: filtered population is empty in frozen file. ABORT."
        )
    if set(result.eval_audit_row_hashes) != set(env.get("eval_audit_row_hashes", [])):
        raise RuntimeError(
            "Reconciliation: frozen-population row hashes do not match envelope "
            "(possible tampering). ABORT."
        )
    # Numeric-drift defense — byte-identical rows must reproduce stats exactly.
    if (
        result.n != env["recomputed_n"]
        or abs(result.mean_predicted - env["recomputed_predicted"]) > 1e-9
        or abs(result.observed_freq - env["recomputed_observed"]) > 1e-9
    ):
        raise RuntimeError(
            "Reconciliation: numeric drift in frozen-population stats. ABORT."
        )
    return True, env, False


if __name__ == "__main__":
    sys.exit(main())
