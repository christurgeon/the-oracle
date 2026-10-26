#!/usr/bin/env bash
# update.sh — Deploy code updates to The Oracle
#
# Usage: sudo ./update.sh [branch-or-tag]
#        sudo ./update.sh main        (default)
#        sudo ./update.sh v1.2.0      (deploy a tagged release)
#
# Optional private ops checkout at $ORACLE_OPS (default /opt/oracle-ops): when
# present it is pulled first, supplies data/snapshots and CLAUDE.local.md via
# symlinks, and its config.live.sha256 pins the live config (see
# scripts/check_live_config.py). Each checkout may set its own deploy key with
# `git config core.sshCommand "ssh -i <key> -o IdentitiesOnly=yes"`.
set -euo pipefail

ORACLE_HOME="/opt/oracle"
ORACLE_OPS="${ORACLE_OPS:-/opt/oracle-ops}"
ORACLE_USER="oracle"
BRANCH="${1:-main}"

# Legacy single-key mode: only used when the code checkout has no core.sshCommand,
# because GIT_SSH_COMMAND would override every repo's own key.
DEPLOY_KEY="${DEPLOY_KEY:-/home/${SUDO_USER:-}/.ssh/github_deploy}"
if [[ -z "$(git -C "$ORACLE_HOME" config --get core.sshCommand || true)" && -f "$DEPLOY_KEY" ]]; then
    export GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes"
fi

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root (sudo $0)" >&2
    exit 1
fi

echo "=== Updating The Oracle (branch: $BRANCH) ==="

# ---------------------------------------------------------------------------
# 1. Pull latest code
# ---------------------------------------------------------------------------
echo "[1/6] Pulling code..."
cd "$ORACLE_HOME"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
chown -R "$ORACLE_USER:$ORACLE_USER" "$ORACLE_HOME"

# A running bash script keeps reading the copy it started with, so re-run once
# after the pull; otherwise changes to this script only apply on the next deploy.
if [[ -z "${ORACLE_UPDATE_REEXECED:-}" ]]; then
    export ORACLE_UPDATE_REEXECED=1
    exec "$ORACLE_HOME/deploy/update.sh" "$BRANCH"
fi

# Read by the KXRAIN observer as observer_version on every corpus row.
BUILD_SHA=$(git rev-parse --short HEAD)
install -d -o "$ORACLE_USER" -g "$ORACLE_USER" "$ORACLE_HOME/state"
printf '%s\n' "$BUILD_SHA" > "$ORACLE_HOME/state/.build_sha"
chown "$ORACLE_USER:$ORACLE_USER" "$ORACLE_HOME/state/.build_sha"
echo "  Code updated ($BUILD_SHA)"

PIN_ARGS=()
if [[ -d "$ORACLE_OPS/.git" ]]; then
    # The legacy GIT_SSH_COMMAND above would override the ops checkout's own key.
    if [[ -n "$(git -C "$ORACLE_OPS" config --get core.sshCommand || true)" ]]; then
        env -u GIT_SSH_COMMAND git -C "$ORACLE_OPS" pull --ff-only
    else
        git -C "$ORACLE_OPS" pull --ff-only
    fi
    chown -R "$ORACLE_USER:$ORACLE_USER" "$ORACLE_OPS"
    OPS_SHA=$(git -C "$ORACLE_OPS" rev-parse --short HEAD)
    printf '%s\n' "$OPS_SHA" > "$ORACLE_HOME/state/.ops_sha"
    chown "$ORACLE_USER:$ORACLE_USER" "$ORACLE_HOME/state/.ops_sha"
    # Recreated on every run: a rollback to a commit that still tracked
    # data/snapshots replaces the symlink with a directory.
    for rel in data/snapshots CLAUDE.local.md; do
        # Skip paths the code checkout still tracks (before they move to ops).
        if [[ -e "$ORACLE_OPS/$rel" ]] && [[ -z "$(git -C "$ORACLE_HOME" ls-files -- "$rel")" ]]; then
            if [[ -d "$ORACLE_HOME/$rel" && ! -L "$ORACLE_HOME/$rel" ]]; then
                rm -rf "${ORACLE_HOME:?}/$rel"
            fi
            ln -sfn "$ORACLE_OPS/$rel" "$ORACLE_HOME/$rel"
        fi
    done
    [[ -f "$ORACLE_OPS/config.live.sha256" ]] && PIN_ARGS=(--expect-hash "$ORACLE_OPS/config.live.sha256")
    echo "  Ops updated ($OPS_SHA)"
fi

# ---------------------------------------------------------------------------
# 2. Reinstall dependencies
# ---------------------------------------------------------------------------
echo "[2/6] Syncing dependencies..."
sudo -u "$ORACLE_USER" /usr/local/bin/uv sync --frozen --extra telegram
echo "  Dependencies synced (with telegram)"

# ---------------------------------------------------------------------------
# 3. Validate config before restart
# ---------------------------------------------------------------------------
echo "[3/6] Validating configuration..."
cd "$ORACLE_HOME"
sudo -u "$ORACLE_USER" /opt/oracle/.venv/bin/python scripts/check_live_config.py "${PIN_ARGS[@]}"

# ---------------------------------------------------------------------------
# 4. Ensure oracle user can execute Claude CLI (for the Telegram /agent command)
# ---------------------------------------------------------------------------
CLAUDE_BIN="${CLAUDE_BIN:-/home/${SUDO_USER:-}/.local/bin/claude}"
if [[ -L "$CLAUDE_BIN" ]]; then
    CLAUDE_TARGET=$(readlink -f "$CLAUDE_BIN")
    echo "[4/6] Fixing Claude CLI permissions..."
    chmod o+rx "$(dirname "$CLAUDE_TARGET")" "$CLAUDE_TARGET" 2>/dev/null || true
    echo "  $CLAUDE_TARGET is world-executable"
else
    echo "[4/6] Claude CLI not found at $CLAUDE_BIN — skipping"
fi

# ---------------------------------------------------------------------------
# 5. Update service file if changed
# ---------------------------------------------------------------------------
echo "[5/6] Checking service file..."
if ! diff -q "$ORACLE_HOME/deploy/oracle-bot.service" /etc/systemd/system/oracle-bot.service &>/dev/null; then
    cp "$ORACLE_HOME/deploy/oracle-bot.service" /etc/systemd/system/oracle-bot.service
    systemctl daemon-reload
    echo "  Service file updated and reloaded"
else
    echo "  Service file unchanged"
fi

# ---------------------------------------------------------------------------
# 6. Restart and verify
# ---------------------------------------------------------------------------
echo "[6/6] Restarting oracle-bot..."
systemctl restart oracle-bot

# Type=notify means systemctl restart blocks until READY=1 or timeout.
# Add a post-restart health check anyway.
sleep 5

if systemctl is-active --quiet oracle-bot; then
    echo "  Service is active"
else
    echo ""
    echo "  ERROR: Service is not active after restart!"
    systemctl status oracle-bot --no-pager
    exit 1
fi

# Check journal for errors in last 30 seconds
ERRORS=$(journalctl -u oracle-bot --since "30 seconds ago" -p err --no-pager -q 2>/dev/null || true)
if [[ -n "$ERRORS" ]]; then
    echo ""
    echo "  WARNING: Errors detected in journal:"
    echo "$ERRORS"
else
    echo "  No errors in journal"
fi

echo ""
echo "=== Update complete ==="
systemctl status oracle-bot --no-pager -l
