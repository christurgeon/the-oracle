"""
Audit-row builders for the weather-edge strategy.

Pure helpers — no I/O beyond the audit-sink callback. Used by
``WeatherEdgeStrategy._build_candidate_audit_row`` on the class and by
``evaluate_market_pure`` in ``weather_evaluation``. All public symbols
are re-exported by ``weather_edge.py`` for back-compat with tests and
the backtest engine.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from python.analytics.audit_log import RejectionReason, reason_numeric
from python.strategies.weather_parsing import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
)

logger = logging.getLogger(__name__)


def _hours_to_settlement(market: dict[str, Any]) -> float | None:
    """Best-effort ``hours_to_settlement`` from Kalshi ``close_time``.

    Returns ``None`` when the field is missing or unparseable.
    """
    close_time = market.get("close_time")
    if not close_time:
        return None
    try:
        ts_raw = str(close_time).replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (ts - datetime.now(tz=UTC)).total_seconds() / 3600.0
    except (TypeError, ValueError):
        return None


def _audit_identity_fields(parsed: ParsedMarket, market: dict[str, Any]) -> dict[str, Any]:
    """Identity fields common to ``eval_audit`` and ``candidate_audit`` rows."""
    return {
        "ticker": parsed.ticker,
        "event": market.get("event_ticker", ""),
        "city": parsed.city,
        "var": parsed.temp_label,
        "forecast_date": parsed.forecast_date.isoformat(),
        "strike_type": parsed.strike_type.value,
        "floor": parsed.floor_strike,
        "cap": parsed.cap_strike,
        "hours_to_settlement": _hours_to_settlement(market),
    }


def _eval_audit_row(
    *,
    scan_id: str,
    sc: ScanCandidate,
    price_info: MarketPriceInfo | None,
    reason: RejectionReason,
    raw_ens_prob: float | None = None,
    calibrated_prob: float | None = None,
    shadow_prob: float | None = None,
    correction_f: float | None = None,
    ens_std: float | None = None,
    strike_abs_z: float | None = None,
    exec_price: float | None = None,
    raw_edge: float | None = None,
    effective_edge: float | None = None,
    fill_edge: float | None = None,
    fee: float | None = None,
    # EMOS scope-B columns — all nullable, all additive.
    emos_prob_raw: float | None = None,
    emos_prob_corrected: float | None = None,
    emos_mu_raw: float | None = None,
    emos_sigma_raw: float | None = None,
    emos_mu_corr: float | None = None,
    emos_sigma_corr: float | None = None,
    emos_cdf_at_floor_raw: float | None = None,
    emos_cdf_at_cap_raw: float | None = None,
    emos_cdf_at_floor_corr: float | None = None,
    emos_cdf_at_cap_corr: float | None = None,
    emos_warnings: list[str] | None = None,
    emos_lead: int | None = None,
    emos_disabled_reason: str | None = None,
    emos_coeffs_artifact_hash: str | None = None,
    emos_coeffs_loaded_at: str | None = None,
    isotonic_artifact_hash: str | None = None,
    gfs_bias_correction_f: float | None = None,
    # Which calibrator produced ``calibrated_prob`` (Stage A per-cell dispatcher):
    # "isotonic", "emos_raw", "isotonic_fallback" or "identity".
    live_calibrator_kind: str | None = None,
    # NBM-bracket-position veto (shadow-only). ``nbm_temperature`` is the NBM
    # point forecast (°F) at scan time; ``nbm_veto_fired`` is whether the
    # bracket-position rule disagreed with the model (vetoed if enabled, else
    # logged only). Both are ``None`` when the veto was not evaluated
    # (non-between strike, NBM unavailable under fail-open, or gate disabled).
    # Consumers MUST read them via ``.get()``: older rotated
    # ``eval_audit.jsonl.*`` files lack the keys and the row schema is unversioned.
    nbm_temperature: float | None = None,
    nbm_veto_fired: bool | None = None,
    # Which strategy evaluated this market. The eval stream is shared
    # (weather_edge + the intraday strategies write the same file), and the
    # calibration monitor's corpus source filters on this field — see
    # ``prediction_log.EvalAuditRecordSource``. Rows written before 2026-07-04
    # lack the key; consumers treat "absent" as weather_edge (correct except
    # for intraday_emos_high's unstamped 2026-05-29→06-07 shadow-window rows,
    # which age out of the monitor's 30d window by ~2026-07-07).
    strategy: str = "weather_edge",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``eval_audit.jsonl`` row. Unknown fields are ``None``.

    ``shadow_prob`` is the parallel-calibrator output for A/B logging; it
    is emitted on every row (``None`` when no shadow calibrator is
    configured) and *never* enters the live edge / sizing path.
    """
    row: dict[str, Any] = {
        "ts": datetime.now(tz=UTC).isoformat(),
        "scan_id": scan_id,
        "strategy": strategy,
        **_audit_identity_fields(sc.parsed, sc.market),
        "yes_bid": price_info.yes_bid if price_info else None,
        "yes_ask": price_info.yes_ask if price_info else None,
        "mid": price_info.midpoint if price_info else None,
        # Mirrors the candidate_audit schema (``_build_candidate_audit_row``) so
        # a consistent ``min_book_depth`` gate applies across streams.
        "yes_bid_size": price_info.yes_bid_size if price_info else None,
        "yes_ask_size": price_info.yes_ask_size if price_info else None,
        "volume": price_info.volume if price_info else None,
        "oi": price_info.open_interest if price_info else None,
        "raw_ens_prob": raw_ens_prob,
        "calibrated_prob": calibrated_prob,
        "shadow_prob": shadow_prob,
        "correction_f": correction_f,
        "ens_std": ens_std,
        "strike_abs_z": strike_abs_z,
        "exec_price": exec_price,
        "raw_edge": raw_edge,
        "effective_edge": effective_edge,
        "fill_edge": fill_edge,
        "fee": fee,
        "reason": reason.value,
        "reason_numeric": reason_numeric(reason),
        "emos_prob_raw": emos_prob_raw,
        "emos_prob_corrected": emos_prob_corrected,
        "emos_mu_raw": emos_mu_raw,
        "emos_sigma_raw": emos_sigma_raw,
        "emos_mu_corr": emos_mu_corr,
        "emos_sigma_corr": emos_sigma_corr,
        "emos_cdf_at_floor_raw": emos_cdf_at_floor_raw,
        "emos_cdf_at_cap_raw": emos_cdf_at_cap_raw,
        "emos_cdf_at_floor_corr": emos_cdf_at_floor_corr,
        "emos_cdf_at_cap_corr": emos_cdf_at_cap_corr,
        "emos_warnings": list(emos_warnings) if emos_warnings else [],
        "emos_lead": emos_lead,
        "emos_disabled_reason": emos_disabled_reason,
        "emos_coeffs_artifact_hash": emos_coeffs_artifact_hash,
        "emos_coeffs_loaded_at": emos_coeffs_loaded_at,
        "isotonic_artifact_hash": isotonic_artifact_hash,
        "gfs_bias_correction_f": gfs_bias_correction_f,
        "live_calibrator_kind": live_calibrator_kind,
        "nbm_temperature": nbm_temperature,
        "nbm_veto_fired": nbm_veto_fired,
        "bracket_window_version": 2,
    }
    if extra:
        row.update(extra)
    return row


def _log_emos_failure_once(
    dedup: dict[Any, Any] | None,
    scan_id: str | None,
    city: str,
    var: str,
    bucket: str,
    exc: Exception,
) -> None:
    """Tier-1 + tier-2 dedup helper for EMOS shadow failure logging.

    - Tier 1: (city, var, scan_id) — only one log per (city, var) within a
      single scan. Cleared at scan boundary by the strategy.
    - Tier 2: (city, var, bucket, utc_hour) — at most one ERROR per hour
      per failure bucket across scans.

    ``dedup=None`` (the backtest default) makes this a no-op so the pure
    backtest path never logs.
    """
    if dedup is None:
        return  # backtest path: never log
    tier1_key = (city, var, scan_id)
    if tier1_key in dedup:
        return
    dedup[tier1_key] = None
    utc_hour = datetime.now(UTC).strftime("%Y%m%d%H")
    tier2_key = (city, var, bucket, utc_hour)
    if tier2_key in dedup:
        return
    dedup[tier2_key] = None
    logger.error("EMOS shadow %s for %s/%s: %s", bucket, city, var, exc)
