#!/usr/bin/env bash
# scripts/post_flip_monitor.sh — pre-committed rollback triggers for the
# 2026-05-22 isotonic_v2 bounded-exposure pilot. Run on VPS or after
# --pull-vps locally. Operator interprets PASS/FAIL.
#
# IMPORTANT: daily_trade_report.py's .calibration.calibration_ratio is
# CUMULATIVE across the full trade log (n=224 pre-flip ratio = 2.43 as of
# flip date). It is NOT scoped by --date — only .settled[] is. T1 below
# reports the cumulative ratio for visibility AND a today-only loss count
# from .settled[] for the actual post-flip signal. Interpret with care.

set -uo pipefail

# Configurable via env: set LOCAL_ONLY=1 to skip --pull-vps (smoke testing).
PULL_VPS_FLAGS="${PULL_VPS_FLAGS:---pull-vps --refresh}"
if [[ "${LOCAL_ONLY:-0}" == "1" ]]; then
    PULL_VPS_FLAGS=""
fi

echo "================================================================"
echo "isotonic_v2 pilot — post-flip monitor"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PULL_VPS_FLAGS=\"$PULL_VPS_FLAGS\""
echo "================================================================"

# Trigger 1: today's loss rate vs cumulative ratio.
# Cumulative .calibration.calibration_ratio includes pre-flip identity-era
# data and will read ~2.43 indefinitely on a polluted corpus — operator
# must compare ACROSS DAYS (today's vs yesterday's) rather than against
# the absolute 2.0 threshold. Today-only n_losses from .settled[] is the
# clean post-flip signal.
# shellcheck disable=SC2086
uv run python scripts/daily_trade_report.py \
    --json /tmp/dtr.json \
    --date "$(date -u +%F)" \
    $PULL_VPS_FLAGS > /dev/null 2>&1 || echo "  (daily_trade_report.py returned non-zero — VPS unreachable or no trades)"
CUM_RATIO=$(jq -r '.calibration.calibration_ratio // "null"' /tmp/dtr.json 2>/dev/null || echo "null")
CUM_N=$(jq -r '.calibration.n_settled // 0' /tmp/dtr.json 2>/dev/null || echo "0")
TODAY_LOSSES=$(jq -r '[.settled[].losses[]?] | length' /tmp/dtr.json 2>/dev/null || echo "0")
TODAY_DAYS=$(jq -r '.settled | length' /tmp/dtr.json 2>/dev/null || echo "0")
echo ""
echo "[T1] cumulative calibration_ratio: $CUM_RATIO (n=$CUM_N; CUMULATIVE — includes pre-flip)"
echo "     today: $TODAY_LOSSES settled losses across $TODAY_DAYS day(s)"
echo "     FAIL if today's loss rate > 60% on n≥5 settled trades (per-day, not cumulative)."
echo "     SECONDARY: track day-over-day delta of cumulative ratio — if it RISES by >0.5"
echo "     in a single day post-flip, escalate even if absolute ratio is still elevated."

# Trigger 2: any single-trade loss above a fixed threshold (well above the Kelly bound at pilot exposure)
LOSSES_OVER_10=$(jq -r '[.settled[].losses[]? | select(.pnl < -10)] | length' /tmp/dtr.json 2>/dev/null || echo "0")
echo ""
echo "[T2] single-trade losses > \$10 (today): $LOSSES_OVER_10"
echo "     FAIL if > 0 (threshold sits well above the per-trade Kelly bound at pilot exposure)."

# Trigger 3: per-cell loss rate > 60% on n≥5 trades (today-only scope —
# daily_trade_report.py defaults --date to today and filters records to
# that single day; the "7d" label on the previous version was a misnomer).
# Reads .settled_per_cell from daily_trade_report's JSON, which aggregates
# wins + losses per (city, var) so true loss rate is computable in jq.
# shellcheck disable=SC2086
uv run python scripts/daily_trade_report.py --json /tmp/dtr_7d.json $PULL_VPS_FLAGS > /dev/null 2>&1 || true
echo ""
echo "[T3] per-cell loss rate (today; FAIL: loss_rate > 60% on n≥5 trades):"
jq -r '
  (.settled_per_cell // {}) | to_entries
  | if length == 0 then
      "     no settled trades today"
    else
      .[] | (.value.n_wins // 0) as $w | (.value.n_losses // 0) as $l | ($w + $l) as $n
            | "     \(.key): \($l)/\($n) losses (\((100 * $l / $n) | ceil)%)"
    end
' /tmp/dtr_7d.json 2>/dev/null || echo "     (per-cell aggregation: check schema)"

# Trigger 4: filters dead — 0 fires in 48h on n≥20 candidates.
# Reason codes are what weather_evaluation.py:_reject() actually emits
# (PLATEAU_BLOCK and CAL_FLOOR_BLOCK; see RejectionReason enum). Scope is
# the current eval_audit.jsonl only — rotated files contain pre-flip
# rows whose historical filter fires would mask a post-flip artifact
# misload. Rotation is UTC-midnight daily so the current file is the
# active post-flip window.
FILTER_FIRES=$(grep -hE '"reason":"(PLATEAU_BLOCK|CAL_FLOOR_BLOCK)"' state/eval_audit.jsonl 2>/dev/null | wc -l | tr -d ' ')
CAND_COUNT=$(wc -l < state/eval_audit.jsonl 2>/dev/null | tr -d ' ')
echo ""
echo "[T4] filter fires (PLATEAU_BLOCK + CAL_FLOOR_BLOCK): $FILTER_FIRES"
echo "     candidates seen (eval rows, current file): $CAND_COUNT"
echo "     FAIL if filter_fires=0 over 48h on ≥20 candidates (filters dead → wrong artifact loaded?)."

# Trigger 5: BETWEEN under-confidence — informational, NOT rollback.
# Replay 2026-05-22: iso BETWEEN mean_p ≈ 0.20 vs actual 0.36. Selection
# bias on Stage-1 high-edge filtering OR regime drift. Re-fit signal if
# the under-prediction is large and persistent.
echo ""
echo "[T5] BETWEEN under-confidence (informational, not rollback trigger):"
echo "     Re-fit signal: actual_win_rate > mean_predicted + 0.15 on n≥5 BETWEEN trades / 7d."
jq -r '
  (.settled // []) as $days
  | ($days | map(.losses[]? | select(.cap_strike != null and .floor_strike != null))) as $bt_losses
  | "     n_between_losses_7d=\($bt_losses | length) (between has both floor + cap strikes)"
' /tmp/dtr_7d.json 2>/dev/null || echo "     (BETWEEN aggregation: check schema)"

echo ""
echo "================================================================"
echo "Re-run via cron or manually at 6h / 12h / 18h / 24h post-flip."
echo "Pre-committed rollback triggers: T1 (day-over-day delta), T2, T3, T4."
echo "T5 is informational — informs whether to re-fit with selection-aware data."
echo "================================================================"
