"""Input-contract preflight + per-pair provenance for the calibration fitters.

Stdlib-only — no numpy / scipy / sklearn at module-import time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_TRIPLES_FOR_FIT",
    "CalibratorInputContractError",
    "DataAvailabilityCalendar",
    "InputContractReport",
    "assert_snapshot_input_contract",
    "build_data_availability_calendar",
    "record_pair_provenance",
]


CALENDAR_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1

MIN_TRIPLES_FOR_FIT = 100


class CalibratorInputContractError(Exception):
    """Hard contract failure — corpus is unsafe to fit against."""


@dataclass(frozen=True)
class DataAvailabilityCalendar:
    """First-appearance dates for each ensemble model in ``data/snapshots/``."""

    schema_version: int
    generated_at: str
    method: str  # "auto" | "manual"
    models: dict[str, date]  # model_name -> first_seen_date

    @classmethod
    def from_path(cls, p: Path) -> DataAvailabilityCalendar:
        """Parse a calendar JSON file. Raises ``CalibratorInputContractError`` on unknown schema."""
        raw = json.loads(p.read_text())
        sv = int(raw.get("schema_version", 0))
        if sv != CALENDAR_SCHEMA_VERSION:
            raise CalibratorInputContractError(
                f"data_availability_calendar.json has schema_version={sv}; "
                f"expected {CALENDAR_SCHEMA_VERSION}. Regenerate the calendar."
            )
        models_raw = raw.get("models", {}) or {}
        models: dict[str, date] = {}
        for name, entry in models_raw.items():
            d_str = entry.get("first_seen_date") if isinstance(entry, dict) else None
            if not d_str:
                continue
            models[str(name)] = date.fromisoformat(str(d_str))
        method = str(raw.get("method", "auto"))
        if method == "manual":
            logger.warning(
                "data_availability_calendar.json was loaded with method='manual' "
                "(path=%s) — auto-detect is the source of truth; verify the manual "
                "override is still intended.",
                p,
            )
        return cls(
            schema_version=sv,
            generated_at=str(raw.get("generated_at", "")),
            method=method,
            models=models,
        )

    def first_seen(self, model: str) -> date | None:
        """Return the first-seen date for ``model``, or ``None`` if not in the calendar."""
        return self.models.get(model)


def build_data_availability_calendar(snapshots_dir: Path) -> dict[str, Any]:
    """Walk snapshot directories and return a calendar dict mapping model -> first-seen date.

    ``forecast_date`` is read from each row, not the directory name — some
    directories hold rows with a forecast_date one or two days ahead.
    Rows with fewer than 25 members are ignored to avoid locking in a
    partial-fetch date prematurely.
    """
    earliest: dict[str, date] = {}

    def _walk_one(d: Path) -> None:
        fc_file = d / "weather_forecasts.json"
        if not fc_file.exists():
            return
        try:
            rows = json.loads(fc_file.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("build_calendar: could not read %s; skipping", fc_file)
            return
        for row in rows:
            model = row.get("model")
            if not model:
                continue
            members = row.get("temperatures") or row.get("ensemble_members") or []
            if len(members) < 25:
                continue
            fc_date_str = row.get("forecast_date")
            if not fc_date_str:
                continue
            try:
                fc_d = date.fromisoformat(str(fc_date_str))
            except ValueError:
                continue
            current = earliest.get(model)
            if current is None or fc_d < current:
                earliest[model] = fc_d

    for day_dir in sorted(snapshots_dir.iterdir()):
        if day_dir.name == "index.jsonl" or not day_dir.is_dir():
            continue
        if (day_dir / "weather_forecasts.json").exists():
            _walk_one(day_dir)
        else:
            for run_dir in sorted(day_dir.iterdir()):
                if run_dir.is_dir():
                    _walk_one(run_dir)

    return {
        "schema_version": CALENDAR_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "auto",
        "models": {
            name: {"first_seen_date": d.isoformat()} for name, d in sorted(earliest.items())
        },
    }


@dataclass(frozen=True)
class InputContractReport:
    """Structured diagnosis of snapshot/markets/settlements coverage."""

    n_triples_total: int
    n_triples_full: int
    n_triples_partial: int
    n_triples_empty: int
    n_triples_single_model_era: int
    n_triples_exempt: int
    n_settlements_final: int
    n_settlements_dropped_preliminary: int
    cells_with_partial_models: list[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]] = field(
        default_factory=list
    )
    calendar_first_seen_dates: dict[str, str] = field(default_factory=dict)

    def format_for_console(self) -> str:
        """Render a one-screen, human-readable summary."""
        lines = [
            "Calibrator input-contract report",
            "================================",
            f"  triples (city, fc_date, var) with market+settlement: {self.n_triples_total}",
            f"    full coverage      : {self.n_triples_full}",
            f"    partial coverage   : {self.n_triples_partial}",
            f"    empty (no fc rows) : {self.n_triples_empty}",
            f"    single-model era   : {self.n_triples_single_model_era}",
            f"    exempt (pre-models): {self.n_triples_exempt}",
            "",
            f"  settlements (final)        : {self.n_settlements_final}",
            f"  settlements (preliminary*) : {self.n_settlements_dropped_preliminary}",
            "    *dropped at load time; counted here for visibility only.",
            "",
            "  Calendar first-seen dates (all calendar entries):",
        ]
        if self.calendar_first_seen_dates:
            for name, d in sorted(self.calendar_first_seen_dates.items()):
                lines.append(f"    {name}: {d}")
        else:
            lines.append("    (none)")
        if self.cells_with_partial_models:
            lines.append("")
            n_partial_total = self.n_triples_partial
            n_shown = min(len(self.cells_with_partial_models), 20)
            if n_partial_total > n_shown:
                header = (
                    "  Top partial-coverage cells (city, fc_date, var: present / missing) "
                    f"— showing top {n_shown} of {n_partial_total}:"
                )
            else:
                header = "  Top partial-coverage cells (city, fc_date, var: present / missing):"
            lines.append(header)
            for entry in self.cells_with_partial_models[:20]:
                city, fc_d, var, present, missing = entry
                lines.append(
                    f"    {city} {fc_d} {var}: present={list(present)} missing={list(missing)}"
                )
        return "\n".join(lines)


def assert_snapshot_input_contract(
    forecasts: dict[tuple[str, str, str, str], dict[str, Any]],
    settlements: dict[tuple[str, str], dict[str, Any]],
    markets: list[dict[str, Any]],
    *,
    ensemble_weights: dict[str, float],
    calendar: DataAvailabilityCalendar,
    expected_min_members: int = 25,
) -> InputContractReport:
    """Bucket every (city, fc_date, daily_var) triple by model coverage.

    Pure — never raises on coverage findings. The one exception: if any key
    in ``ensemble_weights`` is absent from ``calendar.models``, raises
    ``CalibratorInputContractError`` — the coverage gate would otherwise be
    blind to a newly promoted model.

    Logs a WARN for models present in snapshots but absent from the calendar
    (operator cue to regenerate before the next refit).

    ``cells_with_partial_models`` is sorted worst-first (most models missing)
    and capped at 20.
    """
    unknown_in_weights = sorted(set(ensemble_weights) - set(calendar.models))
    if unknown_in_weights:
        raise CalibratorInputContractError(
            "ensemble_weights references model(s) not in the data-availability "
            f"calendar: {unknown_in_weights}. Regenerate via "
            "`scripts/_calibration_dry_run.py --regenerate-calendar` so the "
            "coverage gate knows when the model first appeared."
        )

    snapshot_models = {key[3] for key in forecasts.keys()}
    novel_in_snapshots = sorted(snapshot_models - set(calendar.models))
    for novel in novel_in_snapshots:
        logger.warning(
            "model %r present in snapshots but not in calendar — run "
            "`scripts/_calibration_dry_run.py --regenerate-calendar` before "
            "promoting the next refit",
            novel,
        )

    n_settlements_final = 0
    n_settlements_dropped_preliminary = 0
    for st in settlements.values():
        if st.get("is_preliminary"):
            n_settlements_dropped_preliminary += 1
        else:
            n_settlements_final += 1

    from scripts._calibration_common import VALID_STRIKE_TYPES

    cities_vars_in_markets: set[tuple[str, str]] = set()
    for mkt in markets:
        if mkt.get("strike_type") not in VALID_STRIKE_TYPES:
            continue
        c = mkt.get("city")
        v = mkt.get("daily_variable")
        if c and v:
            cities_vars_in_markets.add((c, v))

    candidate_triples: set[tuple[str, str, str]] = set()
    for city, fc_date_str, daily_var, _model in forecasts.keys():
        if (city, daily_var) in cities_vars_in_markets:
            candidate_triples.add((city, fc_date_str, daily_var))
    for city, fc_date_str in settlements.keys():
        for c, v in cities_vars_in_markets:
            if c == city:
                candidate_triples.add((city, fc_date_str, v))

    final_triples: list[tuple[str, str, str]] = []
    for city, fc_date_str, daily_var in sorted(candidate_triples):
        if (city, fc_date_str) in settlements:
            final_triples.append((city, fc_date_str, daily_var))

    n_full = 0
    n_partial = 0
    n_empty = 0
    n_single_model_era = 0
    n_exempt = 0
    partial_cells: list[tuple[str, str, str, tuple[str, ...], tuple[str, ...]]] = []

    for city, fc_date_str, daily_var in final_triples:
        try:
            fc_date_obj = date.fromisoformat(fc_date_str)
        except ValueError:
            # Skip malformed forecast_date strings — they cannot be graded
            # against the calendar.
            continue

        expected_models: set[str] = set()
        for m in ensemble_weights:
            first = calendar.first_seen(m)
            if first is not None and first <= fc_date_obj:
                expected_models.add(m)

        if len(expected_models) == 0:
            n_exempt += 1
            continue
        if len(expected_models) == 1:
            n_single_model_era += 1
            continue

        present_models: set[str] = set()
        for m in expected_models:
            fc = forecasts.get((city, fc_date_str, daily_var, m))
            if not fc:
                continue
            members = fc.get("temperatures") or fc.get("ensemble_members") or []
            if len(members) >= expected_min_members:
                present_models.add(m)

        if present_models == expected_models:
            n_full += 1
        elif present_models == set():
            n_empty += 1
        else:
            n_partial += 1
            missing = expected_models - present_models
            # Cap at 500 — enough to sort by severity and still surface the worst 20.
            if len(partial_cells) < 500:
                partial_cells.append(
                    (
                        city,
                        fc_date_str,
                        daily_var,
                        tuple(sorted(present_models)),
                        tuple(sorted(missing)),
                    )
                )

    partial_cells.sort(key=lambda c: (-len(c[4]), c[0], c[1]))

    n_total = n_full + n_partial + n_empty + n_single_model_era + n_exempt
    return InputContractReport(
        n_triples_total=n_total,
        n_triples_full=n_full,
        n_triples_partial=n_partial,
        n_triples_empty=n_empty,
        n_triples_single_model_era=n_single_model_era,
        n_triples_exempt=n_exempt,
        n_settlements_final=n_settlements_final,
        n_settlements_dropped_preliminary=n_settlements_dropped_preliminary,
        cells_with_partial_models=partial_cells[:20],
        calendar_first_seen_dates={
            name: d.isoformat() for name, d in sorted(calendar.models.items())
        },
    )


def record_pair_provenance(
    out_path: Path,
    *,
    ticker: str,
    city: str,
    forecast_date: date | str,
    daily_variable: str,
    strike_type: str,
    models_used: list[str],
    bias_correction_applied: bool,
    spread_inflation_factor: float,
    ensemble_weights_at_fit: dict[str, float],
    raw_prob: float,
    outcome: int,
) -> None:
    """Append a single provenance row to ``out_path`` (JSONL, schema_version=1).

    Uses ``os.open(O_WRONLY|O_APPEND|O_CREAT) + os.write`` — single os.write
    is atomic up to PIPE_BUF (~4 KB on Linux). Rows are well under that limit.
    """
    fc_iso = forecast_date.isoformat() if isinstance(forecast_date, date) else str(forecast_date)
    row = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "city": city,
        "forecast_date": fc_iso,
        "daily_variable": daily_variable,
        "strike_type": strike_type,
        "ticker": ticker,
        "models_used": list(models_used),
        "bias_correction_applied": bool(bias_correction_applied),
        "spread_inflation_factor": float(spread_inflation_factor),
        "ensemble_weights_at_fit": {k: float(v) for k, v in ensemble_weights_at_fit.items()},
        "raw_prob": float(raw_prob),
        "outcome": int(outcome),
    }
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
