#!/usr/bin/env bash
# check-heartbeat.sh — External heartbeat monitor for healthchecks.io
#
# Reads the heartbeat file written by the orchestrator every tick and pings
# a dead-man's-switch URL.  Intended to run via cron every 2 minutes.
#
# Usage (cron):
#   */2 * * * * oracle HEALTHCHECK_PING_URL="https://hc-ping.com/<uuid>" /opt/oracle/deploy/check-heartbeat.sh
set -euo pipefail

HEARTBEAT_FILE="${HEARTBEAT_FILE:-/opt/oracle/state/heartbeat}"
PING_URL="${HEALTHCHECK_PING_URL:-}"
MAX_AGE="${MAX_AGE:-300}"  # 5 minutes

_ping() { curl -fsS --retry 3 --max-time 10 "$@" >/dev/null 2>&1 || true; }

# No URL configured — exit silently
if [[ -z "$PING_URL" ]]; then
    exit 0
fi

# Heartbeat file missing
if [[ ! -f "$HEARTBEAT_FILE" ]]; then
    _ping "${PING_URL}/fail" -d "Heartbeat file missing: ${HEARTBEAT_FILE}"
    exit 1
fi

HEARTBEAT_TS=$(cat "$HEARTBEAT_FILE")
NOW=$(date +%s)
AGE=$(( NOW - ${HEARTBEAT_TS%.*} ))

if (( AGE > MAX_AGE )); then
    _ping "${PING_URL}/fail" -d "Heartbeat stale: ${AGE}s old (max ${MAX_AGE}s)"
    exit 1
else
    _ping "${PING_URL}"
fi
