"""Hermetic runtime EMOS coefficient loader + Gaussian-EMOS probability math.

Hermeticity invariant: imports limited to stdlib, numpy, scipy, and
python.data.emos itself. Enforced by tests/test_emos_runtime.py's
import-allowlist scan.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scipy.stats import norm

logger = logging.getLogger(__name__)


# Maps long Open-Meteo daily-variable names to the short canonical labels
# (`"high"` / `"low"`) that EmosTable._cell_key() requires. Lives here next to
# EmosTable's `_CANONICAL_VAR_LABELS` so the long↔short translation has a
# single source of truth — consumed by python.analytics.calibrator_harness
# (EmosDeployedReference + the harness's per-cell key builders).
LONG_TO_SHORT_VAR: dict[str, str] = {
    "temperature_2m_max": "high",
    "temperature_2m_min": "low",
}


@dataclass(frozen=True)
class EmosCoefficients:
    """Per-(city, var) Gaussian-EMOS coefficients.

    Predictive moments:
        μ(ens_mean) = a + b·ens_mean
        σ²(ens_std) = max(c + d·ens_std², 1e-12)

    Bounds (enforced at fit time): b ∈ [0.3, 1.5], c ≥ 1e-6, d ≥ 0.
    """
    a: float
    b: float
    c: float
    d: float
    n_train: int
    low_n: bool
    ci_warnings: list[str]
    leads_pooled: list[int]
    by_lead: dict[int, EmosCoefficients] | None = None
    ens_std_p99: float | None = None  # Optional — None on legacy artifacts


def emos_probability(
    coeffs: EmosCoefficients,
    ens_mean: float,
    ens_std: float,
    *,
    strike_type: str,
    floor: float | None,
    cap: float | None,
) -> tuple[float, float | None, float | None]:
    """Compute EMOS-implied probability for a bracket market.

    Returns (prob, cdf_at_floor, cdf_at_cap).

    For strike_type="between": both floor and cap REQUIRED (cdf_at_floor and
        cdf_at_cap both populated).
    For strike_type="greater": floor REQUIRED, cap MUST be None
        (cdf_at_cap returns None — missing-by-design, NOT NaN).
    For strike_type="less":    cap REQUIRED, floor MUST be None
        (cdf_at_floor returns None — missing-by-design, NOT NaN).

    Raises ValueError on shape violations.
    """
    if strike_type == "between":
        if floor is None or cap is None:
            raise ValueError("between strike requires both floor and cap")
    elif strike_type == "greater":
        if floor is None or cap is not None:
            raise ValueError("greater strike requires floor and forbids cap")
    elif strike_type == "less":
        if cap is None or floor is not None:
            raise ValueError("less strike requires cap and forbids floor")
    else:
        raise ValueError(f"unknown strike_type: {strike_type!r}")

    mu = coeffs.a + coeffs.b * ens_mean
    sigma2 = max(coeffs.c + coeffs.d * (ens_std ** 2), 1e-12)
    sigma = math.sqrt(sigma2)

    cdf_floor: float | None = None
    cdf_cap: float | None = None

    if floor is not None:
        cdf_floor = float(norm.cdf((floor - mu) / sigma))
    if cap is not None:
        cdf_cap = float(norm.cdf((cap - mu) / sigma))

    if strike_type == "between":
        # Validated above: floor + cap both non-None. Guard against
        # floating-point ordering: clamp to non-negative.
        assert cdf_floor is not None and cdf_cap is not None
        prob = max(cdf_cap - cdf_floor, 0.0)
    elif strike_type == "greater":
        # Validated above: floor not None, cap is None.
        assert cdf_floor is not None
        prob = 1.0 - cdf_floor
    else:  # less
        # Validated above: cap not None, floor is None.
        assert cdf_cap is not None
        prob = cdf_cap

    return float(prob), cdf_floor, cdf_cap


# Bounds enforced at load
_B_MIN, _B_MAX = 0.3, 1.5
_C_MIN = 1e-6
_D_MIN = 0.0

# Soft intercept tripwire (INFO log, does not fail smoke)
_INTERCEPT_INFO_THRESHOLD = 50.0


class _BoundsError(ValueError):
    """Raised when a coefficient violates b/c/d bounds. Distinguished from
    plain ValueError so the loader can route to ``bounds-error:`` vs
    ``schema-error:`` fallback_reason without string-matching the message."""


def _parse_cell(cell_key: str, raw: dict[str, Any]) -> EmosCoefficients:
    """Parse a single cell dict (flat or by_lead). Raises ValueError on
    schema/bounds violations."""

    if "by_lead" in raw:
        by_lead: dict[int, EmosCoefficients] = {}
        for lead_str, sub in raw["by_lead"].items():
            lead_int = int(lead_str)
            by_lead[lead_int] = _parse_flat_cell(f"{cell_key}/lead={lead_int}", sub)
        # Pick lead 0 as the canonical pooled — used by .get() if no by_lead match
        canonical = by_lead.get(0) or next(iter(by_lead.values()))
        return EmosCoefficients(
            a=canonical.a, b=canonical.b, c=canonical.c, d=canonical.d,
            n_train=canonical.n_train, low_n=canonical.low_n,
            ci_warnings=canonical.ci_warnings, leads_pooled=[],
            by_lead=by_lead, ens_std_p99=canonical.ens_std_p99,
        )
    return _parse_flat_cell(cell_key, raw)


def _parse_flat_cell(cell_key: str, raw: dict[str, Any]) -> EmosCoefficients:
    required = ("a", "b", "c", "d", "n_train")
    for k in required:
        if k not in raw:
            raise ValueError(f"{cell_key}: missing required field {k!r}")

    a = float(raw["a"])
    b = float(raw["b"])
    c = float(raw["c"])
    d = float(raw["d"])

    if not all(math.isfinite(v) for v in (a, b, c, d)):
        raise _BoundsError(f"{cell_key}: non-finite coefficient")
    if not (_B_MIN <= b <= _B_MAX):
        raise _BoundsError(f"{cell_key}: b={b} out of [{_B_MIN}, {_B_MAX}]")
    if c < _C_MIN:
        raise _BoundsError(f"{cell_key}: c={c} below {_C_MIN}")
    if d < _D_MIN:
        raise _BoundsError(f"{cell_key}: d={d} below {_D_MIN}")

    return EmosCoefficients(
        a=a, b=b, c=c, d=d,
        n_train=int(raw["n_train"]),
        low_n=bool(raw.get("low_n", False)),
        ci_warnings=list(raw.get("ci_warnings", [])),
        leads_pooled=list(raw.get("leads_pooled", [])),
        by_lead=None,
        ens_std_p99=(
            float(raw["ens_std_p99"]) if raw.get("ens_std_p99") is not None else None
        ),
    )


@dataclass
class EmosTable:
    """Hermetic loader for EMOS coefficient artifacts.

    Failure modes (file missing, parse error, schema violation, bounds
    violation, smoke failure) all yield an EmosTable with is_loaded=False
    and a populated fallback_reason. Strategy code uses the
    is_loaded guard before any get(...) call.
    """
    cells: dict[str, EmosCoefficients]
    artifact_hash: str | None
    artifact_signature: tuple[int, int, int] | None
    artifact_loaded_at: str | None
    fit_date: str | None
    training_window_days: int | None
    lead_handling: str | None
    schema_version: int | None
    n_skipped: int
    skipped_cells: dict[str, str]
    fallback_reason: str | None
    _path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        return self.fallback_reason is None and bool(self.cells)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "loaded_at": self.artifact_loaded_at,
            "fit_date": self.fit_date,
            "training_window_days": self.training_window_days,
            "lead_handling": self.lead_handling,
            "schema_version": self.schema_version,
            "n_cells": len(self.cells),
            "n_skipped": self.n_skipped,
            "skipped_cells": dict(self.skipped_cells),
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def _disabled(cls, path: Path | None, reason: str) -> EmosTable:
        return cls(
            cells={}, artifact_hash=None, artifact_signature=None,
            artifact_loaded_at=None, fit_date=None, training_window_days=None,
            lead_handling=None, schema_version=None,
            n_skipped=0, skipped_cells={},
            fallback_reason=reason, _path=path,
        )

    @classmethod
    def load(cls, path: Path | str) -> EmosTable:
        """Build-then-swap loader. Returns a disabled table on any failure;
        callers MUST check is_loaded before calling get()."""
        p = Path(path)

        if not p.exists():
            return cls._disabled(p, "file-missing")

        try:
            raw_bytes = p.read_bytes()
        except OSError as exc:
            return cls._disabled(p, f"io-error:{exc.errno}")

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return cls._disabled(p, f"parse-error:{exc.msg}")

        try:
            cells_raw = payload.get("cells")
            if not isinstance(cells_raw, dict):
                raise ValueError("cells must be a dict")
            cells: dict[str, EmosCoefficients] = {}
            inline_skipped: dict[str, str] = {}
            for key, cell_raw in cells_raw.items():
                # Cells flagged as skipped at fit time (e.g. low_n) are
                # surfaced in skipped_cells rather than parsed as coefficients.
                if isinstance(cell_raw, dict) and cell_raw.get("skipped"):
                    inline_skipped[key] = str(cell_raw.get("reason", "skipped"))
                    continue
                cells[key] = _parse_cell(key, cell_raw)
        except _BoundsError as exc:
            return cls._disabled(p, f"bounds-error:{exc}")
        except (ValueError, KeyError, TypeError) as exc:
            return cls._disabled(p, f"schema-error:{exc}")

        try:
            stat = p.stat()
            signature: tuple[int, int, int] | None = (
                stat.st_mtime_ns, stat.st_size, stat.st_ino,
            )
        except OSError:
            signature = None

        artifact_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Merge top-level skipped_cells (fitter R2-M3 amendment) with any
        # inline {"skipped": true} cells encountered while iterating cells.
        skipped_cells = dict(payload.get("skipped_cells", {}))
        skipped_cells.update(inline_skipped)
        n_skipped = int(payload.get("n_skipped", len(skipped_cells)))

        return cls(
            cells=cells,
            artifact_hash=artifact_hash,
            artifact_signature=signature,
            artifact_loaded_at=datetime.now(UTC).isoformat(),
            fit_date=payload.get("fit_date"),
            training_window_days=payload.get("training_window_days"),
            lead_handling=payload.get("lead_handling"),
            schema_version=payload.get("schema_version"),
            n_skipped=n_skipped,
            skipped_cells=skipped_cells,
            fallback_reason=None,
            _path=p,
        )

    _CANONICAL_VAR_LABELS = frozenset({"high", "low"})

    def _cell_key(self, city: str, var: str) -> str:
        if var not in self._CANONICAL_VAR_LABELS:
            raise KeyError(
                f"EmosTable: var must be 'high' or 'low' (canonical short label "
                f"matching corpus + eval_audit), got {var!r}"
            )
        return f"{city}/{var}"

    def get(
        self, *, city: str, var: str, lead: int,
    ) -> EmosCoefficients | None:
        """Return coefficients for (city, var, lead), or None if absent or
        lead is out of fit horizon. Raises KeyError on var shape mismatch."""
        key = self._cell_key(city, var)
        cell = self.cells.get(key)
        if cell is None:
            return None
        if lead < 0 or lead > 2:
            return None
        if cell.by_lead is not None:
            return cell.by_lead.get(lead)
        return cell  # pooled across leads

    def training_ens_std_p99(self, city: str, var: str) -> float | None:
        """99th percentile of training ens_std for the cell, or None if the
        artifact is legacy (pre-amendment, no field) or the cell is missing."""
        try:
            key = self._cell_key(city, var)
        except KeyError:
            return None
        cell = self.cells.get(key)
        return cell.ens_std_p99 if cell is not None else None

    def check_and_reload(self) -> EmosTable | None:
        """Per-scan hot-reload check.

        Cheap path: compare (mtime_ns, size, inode) tuple. If unchanged,
        return None immediately.

        Expensive path: tuple changed -> hash + parse + smoke. On full
        success, return a NEW EmosTable instance (caller reassigns its
        reference). On any failure, log and return None - the caller
        keeps the prior reference.

        IMPORTANT: This MUST return a new instance rather than mutate self
        in place. Callers (the strategy) implement a per-scan local-
        snapshot pattern (`emos_table_for_scan = self.emos_table`), and
        a mid-scan in-place mutation would propagate to the snapshot
        (every eval_audit row in one scan_id must carry the same
        emos_coeffs_artifact_hash). The instance
        returned here becomes the next scan's snapshot when the strategy
        reassigns `self.emos_table`; the prior scan's snapshot still
        points at the old instance.

        Returns the new EmosTable on swap, or None on no-change / failure.
        """
        if self._path is None or not self._path.exists():
            return None

        try:
            stat = self._path.stat()
            new_signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except OSError:
            return None

        if new_signature == self.artifact_signature:
            return None

        # Tuple changed - read + hash. If hash matches, content is the
        # same (e.g., a touch); we don't need to swap, but we want to
        # avoid re-checking the hash next scan. Caller can choose to
        # update its instance's signature; spec leaves this implicit -
        # for now we return None (no swap), accepting one extra hash
        # check per signature-changed-but-hash-stable event (rare).
        try:
            raw_bytes = self._path.read_bytes()
        except OSError as exc:
            logger.error(
                "EMOS hot-reload: read failed: %s - keeping prior table", exc
            )
            return None

        new_hash = hashlib.sha256(raw_bytes).hexdigest()
        if new_hash == self.artifact_hash:
            return None

        # Build candidate via EmosTable.load (which already does build-
        # then-swap atomicity at construction). Then smoke. On any
        # failure, return None.
        candidate = EmosTable.load(self._path)
        if not candidate.is_loaded:
            logger.error(
                "EMOS hot-reload: load failed (%s) - keeping prior table",
                candidate.fallback_reason,
            )
            return None
        try:
            candidate.smoke_test()  # may raise
        except Exception as exc:
            logger.error(
                "EMOS hot-reload: smoke failed: %s - keeping prior table", exc
            )
            return None

        return candidate

    def smoke_test(self) -> bool:
        """Compute emos_probability for each cell × strike × edge cases.
        Asserts finite + in [0,1]; for between, asserts cdf_floor ≤ cdf_cap.

        Soft sanity (INFO log only, does NOT fail): per cell, if |coeffs.a| > 50,
        emit an INFO log line. Threshold is a generous tripwire — observed fits
        cleared 50 (e.g. LAX/high a=30.77, AUS/low a=-30.75).

        Returns True on pass; raises on failure.
        """
        for cell_key, coeffs in self.cells.items():
            if abs(coeffs.a) > _INTERCEPT_INFO_THRESHOLD:
                logger.info(
                    "EMOS smoke sanity: cell %s has large intercept a=%.2f (>%.0f) — "
                    "informational only, not a failure",
                    cell_key, coeffs.a, _INTERCEPT_INFO_THRESHOLD,
                )

            test_inputs: list[
                tuple[str, float, float, float | None, float | None]
            ] = [
                ("between", 70.0, 3.0, 68.0, 72.0),
                ("greater", 70.0, 3.0, 70.0, None),
                ("less",    70.0, 3.0, None, 70.0),
                # Edge: floor==cap is allowed (probability ≈ 0)
                ("between", 70.0, 3.0, 70.0, 70.0),
            ]
            for strike, mean, std, fl, cp in test_inputs:
                prob, cdf_fl, cdf_cp = emos_probability(
                    coeffs, mean, std, strike_type=strike, floor=fl, cap=cp,
                )
                if not (math.isfinite(prob) and 0.0 <= prob <= 1.0):
                    raise RuntimeError(
                        f"smoke-error:{cell_key}:{strike}: prob={prob!r} not in [0,1]"
                    )
                if strike == "between" and cdf_fl is not None and cdf_cp is not None:
                    if cdf_fl > cdf_cp + 1e-12:
                        raise RuntimeError(
                            f"smoke-error:{cell_key}:{strike}: cdf_floor>{cdf_cp}"
                        )
        return True


@dataclass
class EmosTableHolder:
    """Mutable container so orchestrator-owned hot-reload propagates to all consumers.

    ``EmosTable.check_and_reload()`` returns a NEW instance — it does not mutate
    in place. Strategies that hold a direct ``EmosTable`` reference would
    therefore never see a rebind done by the orchestrator. The holder is the
    tiny shared mutable cell that bridges this gap: the orchestrator owns
    ``holder.table = holder.table.check_and_reload() or holder.table`` once per
    tick, and every strategy that received the same holder via constructor
    injection reads ``self._emos_holder.table`` on every access — guaranteed
    fresh.

    Strategies MUST NOT call ``check_and_reload`` themselves; ownership lives
    in the orchestrator's tick loop.

    CROSS-CONSUMER ALIASING: ``EmosTableHolder`` is a SHARED mutable container.
    The orchestrator owns the rebind. Setting ``weather.emos_table = X`` mutates
    the shared holder, which means intraday_emos_high (or any other consumer
    holding the same holder reference) will also see the new value on its next
    read. This is intentional — it is the mechanism by which one
    orchestrator-owned reload propagates atomically to every strategy that was
    handed the same holder at construction time. Tests that need an
    isolated-per-strategy holder must construct a SEPARATE
    ``EmosTableHolder`` per strategy.
    """

    table: EmosTable
