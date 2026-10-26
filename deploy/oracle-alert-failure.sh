#!/usr/bin/env bash
# oracle-alert-failure.sh — Telegram alert for a failed systemd unit.
#
# Invoked by oracle-alert-failure@.service (template). $1 is the failing unit
# name with the .service suffix (e.g., "oracle-daily-report.service"), passed
# through OnFailure=oracle-alert-failure@%N.service.
#
# Plain text only (no parse_mode) — sidesteps Telegram MarkdownV2/HTML escaping
# pitfalls when journal output contains backticks, underscores, or angle
# brackets. Journal tail is line-bounded (-n 30). Message piped via stdin to
# curl with -d "text@-" to avoid shell quoting risks.
#
# Always exits 0 — a failed alert script must never itself fail in a way that
# could cascade (this script is itself called by an OnFailure= hook, so a
# nonzero exit could trigger further alert recursion).
set -euo pipefail

UNIT="${1:-unknown.service}"

TOKEN="${ORACLE_TELEGRAM_BOT_TOKEN:-}"
CHAT_ID="${ORACLE_TELEGRAM_OWNER_CHAT_ID:-}"

if [[ -z "$TOKEN" || -z "$CHAT_ID" ]]; then
    echo "oracle-alert-failure: missing telegram env vars; skipping alert for $UNIT" >&2
    exit 0
fi

tail_journal() {
    journalctl -u "$UNIT" --no-pager -n 30 --no-hostname 2>/dev/null || true
}

# Telegram sendMessage caps text at 4096 chars and 400-rejects oversize payloads.
# Byte-clamp leaves ~200B headroom for URL-encoding expansion of non-ASCII chars.
{
    printf '⚠️ %s failed\n\n' "$UNIT"
    tail_journal
} | head -c 3900 | curl -fsS --max-time 15 \
    --data-urlencode "chat_id=${CHAT_ID}" \
    --data-urlencode "text@-" \
    "https://api.telegram.org/bot${TOKEN}/sendMessage" \
    >/dev/null || \
    echo "oracle-alert-failure: curl exit=$? for $UNIT" >&2

exit 0
