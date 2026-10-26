"""Probability calibration layer for weather ensemble forecasts.

Maps raw ensemble fractions (GFS+ECMWF blended probabilities after
bias-correction and spread inflation) to calibrated probabilities.

Four implementations share one ``Calibrator`` protocol:

* ``IdentityCalibrator`` — no-op, used when calibration is disabled.
* ``IsotonicCalibrator`` — monotonic piecewise-linear lookup; the
  Phase-A production calibrator (``data/calibration/isotonic.json``).
* ``PlattCalibrator`` — 2-parameter logistic (``a + b*raw``) fit via MLE;
  candidate to replace isotonic after shadow validation.
* ``PerTypePlattCalibrator`` — three independent Platt fits indexed by
  ``strike_type`` ∈ {greater, less, between}. Strike-type comes from
  ``_DIRECTION_TO_STRIKE_LABEL`` in ``python.strategies.weather_edge`` —
  the canonical translation of ``Direction`` enum values to the labels
  used in snapshot / settlement data.

Design notes
------------

* All loaders degrade to *identity* on malformed input, surfacing the
  reason via ``fallback_reason``. The orchestrator reads this on
  startup and alerts via ``/status`` so a silent mis-fit can't trade.
* ``calibrate(raw_prob, *, strike_type=None)`` is the unified entry
  point. Single-curve calibrators ignore ``strike_type``; the per-type
  variant requires it and rejects unknown values by load-time
  assertion — never silently falling back to identity on the hot path.
* No sklearn at runtime; Platt uses ``math.exp``, isotonic uses
  numpy-free linear interpolation over the fitted breakpoint table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Canonical strike-type labels used throughout the calibration pipeline,
# snapshot data, and settlement records. The ``Direction`` enum values
# ("above", "below", "between") must be translated via
# ``_DIRECTION_TO_STRIKE_LABEL`` in ``weather_edge.py`` before being
# passed to ``calibrate(strike_type=...)``.
STRIKE_TYPES: frozenset[str] = frozenset({"greater", "less", "between"})

StrikeType = Literal["greater", "less", "between"]

# Calibrated probabilities are clamped to this band to keep log-odds
# finite downstream (edge / Kelly math).
_PROB_EPS = 1e-6

# Minimum output spread between probe points raw=0.1 and raw=0.9.  A Platt
# calibrator whose spread falls below this is effectively constant — the fit
# has collapsed and is no worse than a prior.  Threshold is loose enough to
# accept a genuinely near-flat ``less`` fit (observed ~0.026) while catching a
# pathological ``a=15, b=0.05`` collapse (spread ≈ 0).
_MIN_PLATT_SPREAD = 0.01

# Probe points for the Platt spread-collapse check.
_PROBE_LO = 0.1
_PROBE_HI = 0.9

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ISOTONIC_PATH = _PROJECT_ROOT / "data" / "calibration" / "isotonic.json"
_DEFAULT_PLATT_PATH = _PROJECT_ROOT / "data" / "calibration" / "platt.json"
_DEFAULT_PER_TYPE_PLATT_PATH = _PROJECT_ROOT / "data" / "calibration" / "platt_per_strike_type.json"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Calibrator(Protocol):
    """Unified calibrator interface.

    Implementations must accept ``strike_type`` as a keyword-only
    argument even if they don't use it — the caller always passes it.
    """

    @property
    def is_loaded(self) -> bool: ...

    @property
    def fallback_reason(self) -> str | None: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def kind(self) -> str: ...

    @property
    def artifact_hash(self) -> str | None: ...

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float: ...


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityCalibrator:
    """Pass-through calibrator. Used when calibration is disabled."""

    kind: str = "identity"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def fallback_reason(self) -> str | None:
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        return {"kind": "identity"}

    @property
    def artifact_hash(self) -> str | None:
        return None

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        return raw_prob


# ---------------------------------------------------------------------------
# Shared artifact-loading machinery
# ---------------------------------------------------------------------------


def _load_json_artifact(path: Path) -> tuple[bytes, dict[str, Any]] | str:
    """Load a JSON calibration artifact.

    Returns ``(raw_bytes, parsed_dict)`` on success, or a ``fallback_reason``
    string on failure (file-missing / parse-error). The caller owns logging
    and schema validation; this helper owns only the file-missing → read →
    ``json.loads`` ladder and the bytes used for the artifact hash.
    """
    if not path.exists():
        return f"file-missing:{path}"
    try:
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes)
    except (json.JSONDecodeError, OSError) as exc:
        return f"parse-error:{type(exc).__name__}"
    return raw_bytes, data


def _validate_platt_coeffs(a: float, b: float) -> str | None:
    """Validate a Platt ``(a, b)`` coefficient pair.

    Returns a ``fallback_reason`` suffix string if the pair is invalid
    (non-finite, non-positive ``b``, or collapsed output spread), else
    ``None``. The reason strings match the historical PlattCalibrator
    messages byte-for-byte so downstream tests/alerts are unaffected.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return "non-finite-coeffs"
    if b <= 0:
        return f"non-positive-b={b:.4f}"
    spread = _stable_sigmoid(a + b * _PROBE_HI) - _stable_sigmoid(a + b * _PROBE_LO)
    if spread < _MIN_PLATT_SPREAD:
        return f"collapsed-spread={spread:.4f}"
    return None


class _BaseArtifactCalibrator:
    """Shared state + read-only properties for JSON-artifact calibrators.

    Subclasses set ``self._loaded`` / ``self._fallback_reason`` /
    ``self._metadata`` / ``self._artifact_hash`` during ``__init__``.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._fallback_reason: str | None = None
        self._metadata: dict[str, Any] = {}
        self._artifact_hash: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def artifact_hash(self) -> str | None:
        return self._artifact_hash


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------


def _validate_curve(breakpoints: list[float], values: list[float]) -> str | None:
    """Return a human-readable error string if the curve is malformed; None if OK."""
    if not breakpoints or not values:
        return "empty"
    if len(breakpoints) != len(values):
        return f"length-mismatch:{len(breakpoints)}!={len(values)}"
    if len(breakpoints) < 2:
        return "too-few-breakpoints"
    for i in range(1, len(breakpoints)):
        if breakpoints[i] < breakpoints[i - 1]:
            return f"non-monotonic-breakpoints-at-{i}"
        if values[i] < values[i - 1]:
            return f"non-monotonic-values-at-{i}"
    return None


class IsotonicCalibrator(_BaseArtifactCalibrator):
    """Isotonic regression calibrator backed by a JSON lookup table.

    Linear interpolation over pre-fitted ``(breakpoint, value)`` pairs.
    Falls back to identity on any loader error, surfacing the reason
    via ``fallback_reason``.
    """

    kind: str = "isotonic"

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._breakpoints: list[float] = []
        self._values: list[float] = []

        model_path = path or _DEFAULT_ISOTONIC_PATH
        loaded = _load_json_artifact(model_path)
        if isinstance(loaded, str):
            self._fallback_reason = loaded
            if loaded.startswith("file-missing:"):
                logger.warning(
                    "Isotonic model not found at %s — calibrator is in identity "
                    "fallback; trades will use raw ensemble fractions.",
                    model_path,
                )
            else:
                logger.warning(
                    "Failed to parse isotonic model at %s (%s) — identity fallback.",
                    model_path,
                    loaded,
                )
            return
        raw_bytes, data = loaded

        try:
            breakpoints = [float(x) for x in data["breakpoints"]]
            values = [float(x) for x in data["values"]]
        except (KeyError, TypeError, ValueError) as exc:
            self._fallback_reason = f"schema-error:{type(exc).__name__}"
            logger.warning(
                "Invalid isotonic model schema at %s (%s) — identity fallback.",
                model_path,
                exc,
            )
            return

        validation_error = _validate_curve(breakpoints, values)
        if validation_error is not None:
            self._fallback_reason = f"validation-error:{validation_error}"
            logger.warning(
                "Isotonic model at %s failed validation (%s) — identity fallback.",
                model_path,
                validation_error,
            )
            return

        self._breakpoints = breakpoints
        self._values = values
        self._metadata = data
        self._artifact_hash = hashlib.sha256(raw_bytes).hexdigest()
        self._loaded = True
        # Two artifact shapes are in use: the weather isotonic/Platt artifacts
        # carry ``fit_date`` at the top level, the favorite_bias g(mid) map nests
        # it under ``metadata``. Reading only the top level logged "fit ?" for the
        # live-money map, so the startup log could not identify which vintage was
        # serving. Mirrors ``audit_tripwires._read_fit_date``.
        metadata = data.get("metadata")
        fit_date = data.get("fit_date") or (
            metadata.get("fit_date") if isinstance(metadata, dict) else None
        )
        logger.info(
            "Loaded isotonic calibration from %s (%d breakpoints, schema v%s, "
            "fit %s, Brier %.4f → %.4f, artifact_hash=%s)",
            model_path,
            len(self._breakpoints),
            data.get("schema_version", 1),
            fit_date or "?",
            data.get("brier_raw", 0),
            data.get("brier_calibrated", 0),
            self._artifact_hash,
        )

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        """Map raw ensemble fraction to calibrated probability.

        ``strike_type`` is accepted for protocol compatibility but
        ignored by the single-curve isotonic implementation.
        """
        if not self._loaded:
            return raw_prob

        # Guard against NaN/Inf — comparisons with NaN return False, which
        # would silently fall through into the binary search and produce a
        # NaN result that poisons downstream Kelly math.
        if not math.isfinite(raw_prob):
            logger.warning(
                "IsotonicCalibrator.calibrate received non-finite raw_prob=%r — "
                "returning 0.5 (upstream bug?)",
                raw_prob,
            )
            return 0.5

        bp = self._breakpoints
        vl = self._values

        if raw_prob <= bp[0]:
            return vl[0]
        if raw_prob >= bp[-1]:
            return vl[-1]

        lo, hi = 0, len(bp) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if bp[mid] <= raw_prob:
                lo = mid
            else:
                hi = mid

        x0, x1 = bp[lo], bp[hi]
        y0, y1 = vl[lo], vl[hi]
        if x1 == x0:
            return y0
        t = (raw_prob - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------


def _stable_sigmoid(z: float) -> float:
    """Numerically stable logistic; never overflows for large |z|."""
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    # z < 0: exp(z) is in (0, 1), avoids exp(+big) overflow.
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _clamp_prob(p: float) -> float:
    """Clamp to [_PROB_EPS, 1 - _PROB_EPS] to keep downstream log-odds finite."""
    if p < _PROB_EPS:
        return _PROB_EPS
    if p > 1.0 - _PROB_EPS:
        return 1.0 - _PROB_EPS
    return p


def _apply_platt(raw_prob: float, a: float, b: float) -> float:
    """Compute ``sigmoid(a + b * raw)`` clamped to (0, 1)."""
    # Guard against non-finite inputs — a misfit ``a``/``b`` combined
    # with an out-of-band raw could in theory produce NaN; clamp first.
    if not math.isfinite(raw_prob):
        return 0.5
    z = a + b * raw_prob
    return _clamp_prob(_stable_sigmoid(z))


class PlattCalibrator(_BaseArtifactCalibrator):
    """Single-curve Platt (logistic) calibrator.

    Expects a JSON artifact with at minimum ``{"a": float, "b": float}``.
    Metadata fields (``fit_date``, ``n_pairs``, ``brier_raw``,
    ``brier_calibrated``, ...) are surfaced via ``metadata``.
    Falls back to identity on malformed input.
    """

    kind: str = "platt"

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._a: float = 0.0
        self._b: float = 1.0

        model_path = path or _DEFAULT_PLATT_PATH
        loaded = _load_json_artifact(model_path)
        if isinstance(loaded, str):
            self._fallback_reason = loaded
            if loaded.startswith("file-missing:"):
                logger.warning(
                    "Platt model not found at %s — calibrator is in identity fallback.",
                    model_path,
                )
            else:
                logger.warning(
                    "Failed to parse Platt model at %s (%s) — identity fallback.",
                    model_path,
                    loaded,
                )
            return
        raw_bytes, data = loaded

        try:
            a = float(data["a"])
            b = float(data["b"])
        except (KeyError, TypeError, ValueError) as exc:
            self._fallback_reason = f"schema-error:{type(exc).__name__}"
            logger.warning(
                "Invalid Platt model schema at %s (%s) — identity fallback.",
                model_path,
                exc,
            )
            return

        coeff_error = _validate_platt_coeffs(a, b)
        if coeff_error is not None:
            self._fallback_reason = f"validation-error:{coeff_error}"
            if coeff_error == "non-finite-coeffs":
                logger.warning(
                    "Platt model at %s has non-finite coefficients (a=%s, b=%s) — "
                    "identity fallback.",
                    model_path,
                    a,
                    b,
                )
            elif coeff_error.startswith("non-positive-b="):
                logger.error(
                    "Platt model at %s has b=%s (≤0) — inverted or flat calibration;"
                    " identity fallback.",
                    model_path,
                    b,
                )
            else:  # collapsed-spread
                spread = _stable_sigmoid(a + b * _PROBE_HI) - _stable_sigmoid(a + b * _PROBE_LO)
                logger.error(
                    "Platt model at %s output spread %.4f < %.4f (probe 0.1→0.9) — "
                    "calibrator is effectively constant; identity fallback.",
                    model_path,
                    spread,
                    _MIN_PLATT_SPREAD,
                )
            return

        self._a = a
        self._b = b
        self._metadata = data
        self._artifact_hash = hashlib.sha256(raw_bytes).hexdigest()
        self._loaded = True
        logger.info(
            "Loaded Platt calibration from %s (a=%.4f, b=%.4f, schema v%s, "
            "fit %s, Brier %.4f → %.4f, artifact_hash=%s)",
            model_path,
            a,
            b,
            data.get("schema_version", 1),
            data.get("fit_date", "?"),
            data.get("brier_raw", 0),
            data.get("brier_calibrated", 0),
            self._artifact_hash,
        )

    @property
    def coefficients(self) -> tuple[float, float]:
        return (self._a, self._b)

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        if not self._loaded:
            return raw_prob
        return _apply_platt(raw_prob, self._a, self._b)


# ---------------------------------------------------------------------------
# Per-strike-type Platt
# ---------------------------------------------------------------------------


class PerTypePlattCalibrator(_BaseArtifactCalibrator):
    """Three independent Platt fits, indexed by strike type.

    Expects a JSON artifact with the shape::

        {
          "schema_version": 2,
          "fit_date": "...",
          "per_type": {
            "greater": {"a": ..., "b": ..., "n": ...},
            "less":    {"a": ..., "b": ..., "n": ...},
            "between": {"a": ..., "b": ..., "n": ...}
          }
        }

    Load-time assertion: the ``per_type`` keys must be *exactly*
    ``{greater, less, between}``. Missing keys, extra keys, or
    non-finite coefficients trigger identity fallback — we never
    silently fall back on the hot path for an unknown ``strike_type``.

    At runtime, ``calibrate`` *requires* ``strike_type``. Passing
    ``None`` or an unknown label raises ``ValueError``; the caller is
    expected to pre-translate the ``Direction`` enum via the canonical
    ``_DIRECTION_TO_STRIKE_LABEL`` map in ``weather_edge.py``.
    """

    kind: str = "per_type_platt"

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._coeffs: dict[str, tuple[float, float]] = {}

        model_path = path or _DEFAULT_PER_TYPE_PLATT_PATH
        loaded = _load_json_artifact(model_path)
        if isinstance(loaded, str):
            self._fallback_reason = loaded
            if loaded.startswith("file-missing:"):
                logger.warning(
                    "Per-type Platt model not found at %s — calibrator is in identity fallback.",
                    model_path,
                )
            else:
                logger.warning(
                    "Failed to parse per-type Platt model at %s (%s) — identity fallback.",
                    model_path,
                    loaded,
                )
            return
        raw_bytes, data = loaded

        per_type = data.get("per_type") if isinstance(data, dict) else None
        if not isinstance(per_type, dict):
            self._fallback_reason = "schema-error:missing-per-type"
            logger.warning(
                "Per-type Platt model at %s missing 'per_type' object — identity fallback.",
                model_path,
            )
            return

        keys = set(per_type.keys())
        if keys != STRIKE_TYPES:
            self._fallback_reason = f"schema-error:strike-type-keys={sorted(keys)}"
            logger.error(
                "Per-type Platt model at %s has strike-type keys %s; expected %s. "
                "Identity fallback — refusing to silently mis-apply coefficients.",
                model_path,
                sorted(keys),
                sorted(STRIKE_TYPES),
            )
            return

        coeffs: dict[str, tuple[float, float]] = {}
        for stype in STRIKE_TYPES:
            entry = per_type[stype]
            try:
                a = float(entry["a"])
                b = float(entry["b"])
            except (KeyError, TypeError, ValueError) as exc:
                self._fallback_reason = f"schema-error:{stype}-coeffs:{type(exc).__name__}"
                logger.warning(
                    "Per-type Platt model at %s has invalid coefficients for "
                    "strike_type=%s (%s) — identity fallback.",
                    model_path,
                    stype,
                    exc,
                )
                return
            coeff_error = _validate_platt_coeffs(a, b)
            if coeff_error is not None:
                if coeff_error == "non-finite-coeffs":
                    self._fallback_reason = f"validation-error:{stype}-non-finite"
                    logger.warning(
                        "Per-type Platt model at %s has non-finite coefficients for "
                        "strike_type=%s (a=%s, b=%s) — identity fallback.",
                        model_path,
                        stype,
                        a,
                        b,
                    )
                elif coeff_error.startswith("non-positive-b="):
                    self._fallback_reason = f"validation-error:{stype}-non-positive-b={b:.4f}"
                    logger.error(
                        "Per-type Platt model at %s has b=%s (≤0) for strike_type=%s — "
                        "inverted or flat calibration; identity fallback.",
                        model_path,
                        b,
                        stype,
                    )
                else:  # collapsed-spread
                    spread = (
                        _stable_sigmoid(a + b * _PROBE_HI) - _stable_sigmoid(a + b * _PROBE_LO)
                    )
                    self._fallback_reason = (
                        f"validation-error:{stype}-collapsed-spread={spread:.4f}"
                    )
                    logger.error(
                        "Per-type Platt model at %s output spread %.4f < %.4f for "
                        "strike_type=%s (probe 0.1→0.9) — calibrator is effectively constant; "
                        "identity fallback.",
                        model_path,
                        spread,
                        _MIN_PLATT_SPREAD,
                        stype,
                    )
                return

            coeffs[stype] = (a, b)

        self._coeffs = coeffs
        self._metadata = data
        self._artifact_hash = hashlib.sha256(raw_bytes).hexdigest()
        self._loaded = True
        logger.info(
            "Loaded per-type Platt calibration from %s (schema v%s, fit %s, "
            "greater=(%.3f,%.3f) less=(%.3f,%.3f) between=(%.3f,%.3f), "
            "artifact_hash=%s)",
            model_path,
            data.get("schema_version", 1),
            data.get("fit_date", "?"),
            *coeffs["greater"],
            *coeffs["less"],
            *coeffs["between"],
            self._artifact_hash,
        )

    def calibrate(self, raw_prob: float, *, strike_type: str | None = None) -> float:
        if not self._loaded:
            return raw_prob
        if strike_type not in STRIKE_TYPES:
            # Loud failure: callers are contractually obligated to pass
            # a canonical label via _DIRECTION_TO_STRIKE_LABEL.
            raise ValueError(
                f"PerTypePlattCalibrator.calibrate: strike_type must be one of "
                f"{sorted(STRIKE_TYPES)}, got {strike_type!r}"
            )
        a, b = self._coeffs[strike_type]
        return _apply_platt(raw_prob, a, b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


CalibrationKind = Literal["none", "identity", "isotonic", "platt", "per_type_platt"]


def _kind_from_config(cfg: dict[str, Any], key: str) -> str | None:
    """Resolve the calibration kind for ``key`` from a strategy config dict.

    Supported forms:
      * Live:   ``calibration_type: <kind>`` (maps 1:1).
      * Shadow: ``shadow_calibrator: <kind>`` (separate namespace).

    Returns ``None`` when nothing is configured for ``key`` (caller
    interprets as "calibration disabled", i.e. identity).
    """
    if key == "calibrator":
        explicit = cfg.get("calibration_type")
        if explicit is not None:
            return str(explicit)
        return None
    if key == "shadow_calibrator":
        shadow = cfg.get("shadow_calibrator")
        if shadow is None:
            return None
        return str(shadow)
    raise ValueError(f"_kind_from_config: unknown key {key!r}")


def build_calibrator(
    kind: str | None,
    *,
    isotonic_path: Path | None = None,
    platt_path: Path | None = None,
    per_type_platt_path: Path | None = None,
) -> Calibrator:
    """Construct a calibrator by kind string.

    Unknown or ``None`` kinds return :class:`IdentityCalibrator`.
    """
    if kind is None or kind in ("none", "identity", "off", "disabled"):
        return IdentityCalibrator()
    if kind == "isotonic":
        return IsotonicCalibrator(path=isotonic_path)
    if kind == "platt":
        return PlattCalibrator(path=platt_path)
    if kind == "per_type_platt":
        return PerTypePlattCalibrator(path=per_type_platt_path)
    logger.warning(
        "Unknown calibration kind %r — falling back to identity.",
        kind,
    )
    return IdentityCalibrator()


def load_calibrator(
    config: dict[str, Any],
    *,
    key: Literal["calibrator", "shadow_calibrator"] = "calibrator",
    isotonic_path: Path | None = None,
    platt_path: Path | None = None,
    per_type_platt_path: Path | None = None,
) -> Calibrator:
    """Build the production (or shadow) calibrator from a strategy config.

    ``config`` is the weather strategy sub-dict (``strategies.weather``
    from ``config.yaml``). ``key`` selects the namespace:

    * ``"calibrator"`` (default) — the live calibrator applied to the
      raw model probability before edge / sizing. Resolves
      ``calibration_type`` → identity.
    * ``"shadow_calibrator"`` — a parallel calibrator whose output is
      logged for shadow validation but never used for trading. Only
      honored when ``shadow_calibrator`` is explicitly set.

    Returns :class:`IdentityCalibrator` if the selected calibrator is
    disabled or fails to load cleanly. Callers can inspect
    ``is_loaded`` / ``fallback_reason`` on the returned instance.
    """
    kind = _kind_from_config(config, key)
    cal = build_calibrator(
        kind,
        isotonic_path=isotonic_path,
        platt_path=platt_path,
        per_type_platt_path=per_type_platt_path,
    )
    # If the config asked for a real calibrator but the artifact is
    # missing/broken, degrade to identity so signal generation doesn't
    # crash — but keep the failure visible.
    if kind not in (None, "none", "identity", "off", "disabled") and not cal.is_loaded:
        logger.warning(
            "Calibrator %r for key=%r is in fallback (reason=%s) — "
            "signals will run on raw probabilities.",
            kind,
            key,
            getattr(cal, "fallback_reason", None),
        )
        return IdentityCalibrator()
    return cal


__all__ = [
    "STRIKE_TYPES",
    "StrikeType",
    "CalibrationKind",
    "Calibrator",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "PerTypePlattCalibrator",
    "build_calibrator",
    "load_calibrator",
]
