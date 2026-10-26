"""Append-only prediction log for live calibration monitoring.

Every signal that passes strategy gates is logged here — regardless of whether
risk-manager approves it or the exchange fills it. This gives calibration a
representative sample of our *model's* predictions, not just the ones that
happened to trade.

Concurrency model: **single-writer**. Only one orchestrator process should open
this log for writing at a time. Appends are done with a single ``write()`` call
of the fully-serialized JSON line (plus newline) so that other readers (Telegram
``/reliability`` queries, analysis scripts) never observe a torn line.

Schema versioning: each record carries ``schema_version``; bump when fields
change incompatibly.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from python.common import Side

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 7


@dataclass(frozen=True)
class PredictionRecord:
    """A single prediction emitted by a strategy."""

    timestamp: datetime  # signal generation time (must be tz-aware UTC)
    strategy: str
    market_ticker: str
    side: Side  # which side our signal favors
    city: str | None
    daily_variable: str | None  # "high" | "low" | None
    strike_type: str | None  # "greater" | "less" | "between"
    floor_strike: float | None
    cap_strike: float | None
    expiration_date: date | None  # market settlement date (ET)
    raw_ensemble_prob: float | None  # pre-isotonic fraction (weather only)
    calibrated_prob: float  # == signal.model_probability (YES-space)
    market_price: float  # YES-space midpoint at signal time
    exec_price: float | None  # side-denominated expected fill price
    confidence: float
    order_id: str | None = None  # filled post-execution if executed
    isotonic_fit_date: str | None = None  # lineage: which isotonic curve was active
    gfs_prob: float | None = None  # GFS-only bracket probability at signal time
    ecmwf_prob: float | None = None  # ECMWF-only bracket probability (None beyond 16d)
    ensemble_std_pooled: float | None = None  # pooled σ̂ across all models
    is_same_day_market: bool = False  # True iff signal originated from a same-day market
    # Schema v4 (Tier 2 C): which calibrator produced ``calibrated_prob``
    # on this row. One of {"isotonic", "emos_raw", "isotonic_fallback", "identity"}
    # for new rows. ``None`` for legacy v3 rows that pre-date this field; the
    # calibration monitor buckets those into kind="unknown" at score time.
    live_calibrator_kind: str | None = None
    # Schema v5: intraday EMOS calibration completeness fields. Populated by
    # intraday_emos_high; None for weather_edge rows (uses blended ensemble).
    ens_mean: float | None = None
    ens_std: float | None = None
    emos_prob_raw: float | None = None
    # Schema v6: intraday ASOS nowcast audit fields. Populated by
    # intraday_asos_high; None for all other strategies.
    asos_M: float | None = None  # noqa: N815  # ASOS observed temperature (°F)
    asos_obs_age_h: float | None = None  # age of ASOS observation (hours)
    mu_gain: float | None = None  # posterior mean shift vs prior (°F)
    sigma_gain: float | None = None  # posterior σ reduction factor
    climo_n: int | None = None  # number of climatology obs used
    climo_source: str | None = None  # climatology artifact identifier
    posterior_p_raw: float | None = None  # pre-calibration posterior probability
    boundary_distance_f: float | None = None  # distance to nearest NWS rounding boundary (°F)
    post_mean_f: float | None = None  # posterior mean temperature (°F)
    guard_outcome: str | None = None  # result of physical plausibility guard check
    prelim_drift_f: float | None = None  # preliminary NWS report drift magnitude (°F)
    # Schema v7: which climate-day frame produced this row's ASOS M / climatology.
    #   1 = legacy CIVIL (DST-aware ZoneInfo) frame — contaminated boundary hour
    #   2 = LST (fixed standard-time offset) frame — the corrected frame (§0a)
    # Absent on pre-v7 rows ⇒ deserializes to 1. ASOS-relevant strategies must set
    # this explicitly; non-ASOS strategies keep the default (their M-frame is moot).
    frame_version: int = 1
    schema_version: int = field(default=SCHEMA_VERSION)


def _serialize(rec: PredictionRecord) -> dict[str, Any]:
    d = asdict(rec)
    d["timestamp"] = rec.timestamp.astimezone(UTC).isoformat()
    d["side"] = str(rec.side.value) if isinstance(rec.side, Side) else str(rec.side)
    d["expiration_date"] = rec.expiration_date.isoformat() if rec.expiration_date else None
    return d


def _deserialize(obj: dict[str, Any]) -> PredictionRecord | None:
    try:
        ts = datetime.fromisoformat(obj["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        side_str = obj.get("side", "").upper() or "NONE"
        try:
            side = Side(side_str)
        except ValueError:
            side = Side.NONE
        exp_raw = obj.get("expiration_date")
        exp = date.fromisoformat(exp_raw) if exp_raw else None
        return PredictionRecord(
            timestamp=ts,
            strategy=str(obj.get("strategy", "")),
            market_ticker=str(obj.get("market_ticker", "")),
            side=side,
            city=obj.get("city"),
            daily_variable=obj.get("daily_variable"),
            strike_type=obj.get("strike_type"),
            floor_strike=obj.get("floor_strike"),
            cap_strike=obj.get("cap_strike"),
            expiration_date=exp,
            raw_ensemble_prob=obj.get("raw_ensemble_prob"),
            calibrated_prob=float(obj["calibrated_prob"]),
            market_price=float(obj["market_price"]),
            exec_price=obj.get("exec_price"),
            confidence=float(obj.get("confidence", 0.0)),
            order_id=obj.get("order_id"),
            isotonic_fit_date=obj.get("isotonic_fit_date"),
            gfs_prob=obj.get("gfs_prob"),
            ecmwf_prob=obj.get("ecmwf_prob"),
            ensemble_std_pooled=obj.get("ensemble_std_pooled"),
            is_same_day_market=bool(obj.get("is_same_day_market", False)),
            # Legacy v3 rows are missing this field → None, which
            # the monitor buckets as kind="unknown" at score time.
            live_calibrator_kind=obj.get("live_calibrator_kind"),
            ens_mean=obj.get("ens_mean"),
            ens_std=obj.get("ens_std"),
            emos_prob_raw=obj.get("emos_prob_raw"),
            asos_M=obj.get("asos_M"),  # noqa: N815
            asos_obs_age_h=obj.get("asos_obs_age_h"),
            mu_gain=obj.get("mu_gain"),
            sigma_gain=obj.get("sigma_gain"),
            climo_n=obj.get("climo_n"),
            climo_source=obj.get("climo_source"),
            posterior_p_raw=obj.get("posterior_p_raw"),
            boundary_distance_f=obj.get("boundary_distance_f"),
            post_mean_f=obj.get("post_mean_f"),
            guard_outcome=obj.get("guard_outcome"),
            prelim_drift_f=obj.get("prelim_drift_f"),
            # Pre-v7 rows lack frame_version → CIVIL frame (1). New rows persist 2.
            frame_version=int(obj.get("frame_version", 1)),
            schema_version=int(obj.get("schema_version", 1)),
        )
    except (KeyError, ValueError, TypeError):
        return None


class PredictionLog:
    """Append-only JSONL writer/reader for prediction records."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # In-process lock protects callers that share this instance from
        # interleaving across threads. Cross-process interleaving is the
        # caller's responsibility (we assume a single orchestrator process).
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, rec: PredictionRecord) -> None:
        if rec.timestamp.tzinfo is None:
            raise ValueError("PredictionRecord.timestamp must be timezone-aware")
        line = json.dumps(_serialize(rec), default=str, separators=(",", ":")) + "\n"
        payload = line.encode("utf-8")
        with self._lock:
            # O_APPEND guarantees the write is atomic for payloads up to
            # PIPE_BUF (~4 KB). Our records are ~500-800 B, well inside that.
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)

    def iter_all(self) -> Iterator[PredictionRecord]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("prediction_log: skipping malformed line")
                    continue
                rec = _deserialize(obj)
                if rec is not None:
                    yield rec

    def iter_since(self, cutoff: datetime) -> Iterator[PredictionRecord]:
        """Yield records with ``timestamp >= cutoff``. Cutoff must be tz-aware."""
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        for rec in self.iter_all():
            if rec.timestamp >= cutoff:
                yield rec


class EvalAuditRecordSource:
    """``PredictionRecord`` source backed by ``state/eval_audit.jsonl``.

    Duck-types ``PredictionLog.iter_all`` for the calibration monitor. The
    eval-audit stream is written at Stage-1 — for EVERY evaluated market,
    before depth-check, Kelly sizing, and allocation — so the calibration
    corpus it feeds is structurally immune to the $0-pod coupling that froze
    ``predictions.jsonl`` on 2026-06-16 (``strategy_allocations.weather_edge:
    0.0`` → every candidate rejected ``KELLY_ZERO`` → zero signals → zero
    prediction rows). It is also the stream already covered by the
    ``audit_tripwires`` staleness check, and its silence is unambiguous:
    no eval rows means the scan pipeline stopped.

    Row filter:

    * ``strategy`` must be ``"weather_edge"`` or absent. Rows written before
      the field landed (2026-07-04) carry no ``strategy`` key; treating absent
      as weather_edge is correct with one known, self-expiring exception —
      ``intraday_emos_high`` wrote unstamped eval rows via
      ``evaluate_market_pure`` during its 2026-05-29→06-07 shadow window, so
      MIA/high + LAX/high rows from that window are misattributed until they
      age out of the 30d window (~2026-07-07). ``intraday_asos_high`` builds
      its own rows and has always stamped its name.
    * ``calibrated_prob`` must be present — rows rejected before calibration
      (price gates, exclusions) carry ``None`` and are not scoreable.
    * a price is required (``mid``, falling back to ``exec_price``) to satisfy
      the ``PredictionRecord.market_price`` contract.

    Reads the active file plus rotated daily siblings
    (``eval_audit.jsonl.YYYY-MM-DD[.N]``), oldest first, so the monitor's 30d
    window survives rotation. Rotated siblings older than ``window_days``
    (by filename date — cheap, no parsing) are skipped: the monitor only
    scores 7d/30d windows, and the shared stream carries ~10⁵ rows/day, so an
    unbounded read would cost ~35s of parse per monitor tick. Retention is
    governed by ``strategies.weather.audit_log_retention_days`` — keep it
    above ``window_days``.
    """

    # Monitor scores at most a 30d window; +2 days of slack for timestamp-vs-
    # filename skew (a file rotated on day D contains rows from day D).
    _WINDOW_DAYS = 32

    def __init__(self, path: Path | str, window_days: int = _WINDOW_DAYS) -> None:
        self._path = Path(path)
        self._window_days = window_days

    @property
    def active_path(self) -> Path:
        """The live (unrotated) stream file — used for freshness probes."""
        return self._path

    def _files(self) -> list[Path]:
        cutoff = datetime.now(tz=UTC).date() - timedelta(days=self._window_days)
        rotated = []
        for p in sorted(self._path.parent.glob(self._path.name + ".*")):
            if p.name.endswith(".tmp"):
                continue
            # ``eval_audit.jsonl.2026-06-19`` / ``...2026-06-19.1`` — parse the
            # date segment; keep files whose date is unparseable (safe default).
            seg = p.name[len(self._path.name) + 1 :].split(".")[0]
            try:
                if date.fromisoformat(seg) < cutoff:
                    continue
            except ValueError:
                pass
            rotated.append(p)
        return rotated + ([self._path] if self._path.exists() else [])

    def iter_all(self) -> Iterator[PredictionRecord]:
        for path in self._files():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        # Cheap prefilter: the shared stream is dominated by
                        # other strategies' rows (intraday_asos writes ~10× the
                        # weather volume). A row that carries a ``strategy``
                        # key with some other value can be skipped without a
                        # JSON parse; anything ambiguous falls through to the
                        # real filter in ``_eval_row_to_record``.
                        if '"strategy"' in line and '"weather_edge"' not in line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "eval_audit_source: skipping malformed line in %s", path.name
                            )
                            continue
                        rec = _eval_row_to_record(obj)
                        if rec is not None:
                            yield rec
            except OSError:
                logger.warning("eval_audit_source: unreadable file %s", path, exc_info=True)

    def iter_since(self, cutoff: datetime) -> Iterator[PredictionRecord]:
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        for rec in self.iter_all():
            if rec.timestamp >= cutoff:
                yield rec


def _eval_row_to_record(obj: dict[str, Any]) -> PredictionRecord | None:
    """Adapt one eval-audit row to a ``PredictionRecord`` (or ``None`` to skip).

    Field mapping mirrors ``weather_edge._record_prediction``: eval rows carry
    ``forecast_date`` (the observation date), which is exactly what the legacy
    predictions path stored in ``expiration_date``.
    """
    strategy = obj.get("strategy")
    if strategy is not None and strategy != "weather_edge":
        return None
    calibrated = obj.get("calibrated_prob")
    # Eval rows stamp the raw ``Direction`` value ("above"/"below"/"between");
    # the calibration pipeline's canonical labels are "greater"/"less"/
    # "between" (what the legacy predictions path stored via
    # ``_DIRECTION_TO_STRIKE_LABEL`` and what ``derive_outcome_yes`` accepts —
    # unmapped, every directional row would be silently dropped on the NWS
    # settlement path). Translate here; unknown values pass through untouched.
    raw_strike_type = obj.get("strike_type")
    strike_type = {"above": "greater", "below": "less"}.get(
        str(raw_strike_type), raw_strike_type
    )
    if calibrated is None:
        return None
    price = obj.get("mid")
    if price is None:
        price = obj.get("exec_price")
    if price is None:
        return None
    ts_raw = obj.get("ts")
    ticker = obj.get("ticker")
    if not ts_raw or not ticker:
        return None
    try:
        ts = datetime.fromisoformat(str(ts_raw))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        fd_raw = obj.get("forecast_date")
        expiration = date.fromisoformat(str(fd_raw)) if fd_raw else None
        return PredictionRecord(
            timestamp=ts,
            strategy="weather_edge",
            market_ticker=str(ticker),
            side=Side.NONE,
            city=obj.get("city"),
            daily_variable=obj.get("var"),
            strike_type=strike_type,
            floor_strike=obj.get("floor"),
            cap_strike=obj.get("cap"),
            expiration_date=expiration,
            raw_ensemble_prob=obj.get("raw_ens_prob"),
            calibrated_prob=float(calibrated),
            market_price=float(price),
            exec_price=obj.get("exec_price"),
            confidence=0.0,
            live_calibrator_kind=obj.get("live_calibrator_kind"),
        )
    except (KeyError, ValueError, TypeError):
        return None
