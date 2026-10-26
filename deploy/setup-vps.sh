#!/usr/bin/env bash
# setup-vps.sh — Idempotent VPS bootstrap for The Oracle
#
# Target: Ubuntu 24.04 LTS
# Run as: sudo REPO_URL=git@github.com:<owner>/<repo>.git ./setup-vps.sh
# REPO_URL is required only when /opt/oracle is not yet a git clone.
#
# This script is safe to re-run. Each section checks state before acting.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these before first run
# ---------------------------------------------------------------------------
ORACLE_HOME="/opt/oracle"
ORACLE_USER="oracle"
ORACLE_GROUP="oracle"
REPO_URL="${REPO_URL:-}"
PYTHON_VERSION="3.13"
UV_VERSION="0.6.6"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root (sudo $0)" >&2
    exit 1
fi

if [[ -z "$REPO_URL" && ! -d "$ORACLE_HOME/.git" ]]; then
    echo "ERROR: $ORACLE_HOME is not a git clone. Set REPO_URL (e.g. sudo REPO_URL=git@github.com:<owner>/<repo>.git $0)" >&2
    exit 1
fi

DEPLOY_USER="${SUDO_USER:-}"
if [[ -z "$DEPLOY_USER" ]]; then
    echo "ERROR: Run via sudo, not direct root login" >&2
    exit 1
fi

AUTH_KEYS="/home/${DEPLOY_USER}/.ssh/authorized_keys"
if [[ ! -s "$AUTH_KEYS" ]]; then
    echo "ERROR: No SSH authorized_keys found for $DEPLOY_USER" >&2
    echo "Add your SSH public key first: ssh-copy-id $DEPLOY_USER@<host>" >&2
    exit 1
fi

echo "=== The Oracle — VPS Setup ==="
echo "User: $DEPLOY_USER | Home: $ORACLE_HOME | Repo: ${REPO_URL:-existing clone}"
echo ""

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/14] Installing system packages..."
apt-get update -qq
apt-get install -y -qq curl git ufw fail2ban unattended-upgrades \
    ca-certificates gnupg lsb-release bc >/dev/null

# ---------------------------------------------------------------------------
# 2. SSH hardening
# ---------------------------------------------------------------------------
echo "[2/14] Hardening SSH..."
SSHD_CHANGED=false

if grep -qE '^\s*#?\s*PasswordAuthentication\s+yes' /etc/ssh/sshd_config 2>/dev/null; then
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    SSHD_CHANGED=true
fi

if grep -qE '^\s*#?\s*PermitRootLogin\s+(yes|prohibit-password)' /etc/ssh/sshd_config 2>/dev/null; then
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
    SSHD_CHANGED=true
fi

if $SSHD_CHANGED; then
    systemctl restart ssh || systemctl restart sshd
    echo "  SSH hardened: password auth disabled, root login disabled"
else
    echo "  SSH already hardened"
fi

# ---------------------------------------------------------------------------
# 3. Python 3.13
# ---------------------------------------------------------------------------
echo "[3/14] Installing Python ${PYTHON_VERSION}..."
if ! command -v "python${PYTHON_VERSION}" &>/dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev" >/dev/null
    echo "  Python ${PYTHON_VERSION} installed"
else
    echo "  Python ${PYTHON_VERSION} already installed"
fi

# ---------------------------------------------------------------------------
# 4. uv (pinned version)
# ---------------------------------------------------------------------------
echo "[4/14] Installing uv ${UV_VERSION}..."
CURRENT_UV=""
if command -v uv &>/dev/null; then
    CURRENT_UV=$(uv --version 2>/dev/null | head -1 || true)
fi

if [[ "$CURRENT_UV" != *"$UV_VERSION"* ]]; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
    # Move to system-wide location
    for bin in uv uvx; do
        if [[ -f "$HOME/.local/bin/$bin" ]]; then
            mv "$HOME/.local/bin/$bin" "/usr/local/bin/$bin"
        elif [[ -f "/root/.local/bin/$bin" ]]; then
            mv "/root/.local/bin/$bin" "/usr/local/bin/$bin"
        fi
    done
    echo "  uv ${UV_VERSION} installed to /usr/local/bin"
else
    echo "  uv ${UV_VERSION} already installed"
fi

# ---------------------------------------------------------------------------
# 5. Create oracle user
# ---------------------------------------------------------------------------
echo "[5/14] Creating oracle user..."
if ! id "$ORACLE_USER" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "$ORACLE_HOME" --create-home "$ORACLE_USER"
    echo "  User '$ORACLE_USER' created (shell: /usr/sbin/nologin)"
else
    echo "  User '$ORACLE_USER' already exists"
fi

# ---------------------------------------------------------------------------
# 6. Clone / update repo
# ---------------------------------------------------------------------------
echo "[6/14] Setting up repository..."
DEPLOY_KEY="/home/${DEPLOY_USER}/.ssh/github_deploy"
export GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes"

if [[ -d "$ORACLE_HOME/.git" ]]; then
    cd "$ORACLE_HOME"
    git pull --ff-only || {
        echo "  WARNING: git pull --ff-only failed (diverged?). Skipping."
    }
    chown -R "$ORACLE_USER:$ORACLE_GROUP" "$ORACLE_HOME"
    echo "  Repository updated"
else
    git clone "$REPO_URL" "$ORACLE_HOME"
    chown -R "$ORACLE_USER:$ORACLE_GROUP" "$ORACLE_HOME"
    echo "  Repository cloned to $ORACLE_HOME"
fi

# ---------------------------------------------------------------------------
# 7. Create .env from template
# ---------------------------------------------------------------------------
echo "[7/14] Setting up .env..."
ENV_FILE="$ORACLE_HOME/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ORACLE_HOME/.env.example" ]]; then
        cp "$ORACLE_HOME/.env.example" "$ENV_FILE"
    else
        touch "$ENV_FILE"
    fi
    chown "$ORACLE_USER:$ORACLE_GROUP" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "  .env created (chmod 600) — ACTION REQUIRED: fill in secrets"
else
    echo "  .env already exists"
    # Ensure permissions are correct even on re-run
    chown "$ORACLE_USER:$ORACLE_GROUP" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# 8. Private key directory
# ---------------------------------------------------------------------------
echo "[8/14] Setting up private key directory..."
KEYS_DIR="$ORACLE_HOME/keys"
mkdir -p "$KEYS_DIR"
chown "$ORACLE_USER:$ORACLE_GROUP" "$KEYS_DIR"
chmod 700 "$KEYS_DIR"

# ---------------------------------------------------------------------------
# 9. Install Python dependencies
# ---------------------------------------------------------------------------
echo "[9/14] Installing Python dependencies..."
cd "$ORACLE_HOME"
sudo -u "$ORACLE_USER" /usr/local/bin/uv sync --frozen --extra telegram
echo "  Dependencies installed (frozen lockfile, with telegram)"

# ---------------------------------------------------------------------------
# 10. Install systemd service
# ---------------------------------------------------------------------------
echo "[10/14] Installing systemd service..."
cp "$ORACLE_HOME/deploy/oracle-bot.service" /etc/systemd/system/oracle-bot.service
systemctl daemon-reload
systemctl enable oracle-bot
echo "  Service installed and enabled (NOT started)"

# ---------------------------------------------------------------------------
# 11. UFW firewall
# ---------------------------------------------------------------------------
echo "[11/14] Configuring firewall..."
ufw default deny incoming >/dev/null 2>&1
ufw default allow outgoing >/dev/null 2>&1
ufw allow ssh >/dev/null 2>&1
ufw --force enable >/dev/null 2>&1
echo "  UFW enabled: SSH only"

# ---------------------------------------------------------------------------
# 12. Journald log rotation
# ---------------------------------------------------------------------------
echo "[12/14] Configuring log rotation..."
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/oracle.conf << 'JOURNALD'
[Journal]
SystemMaxUse=500M
MaxFileSec=7day
JOURNALD
systemctl restart systemd-journald
echo "  Journald: 500MB cap, 7-day rotation"

# ---------------------------------------------------------------------------
# 13. Unattended upgrades (security patches only)
# ---------------------------------------------------------------------------
echo "[13/14] Configuring automatic security updates..."
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'UNATTENDED'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
UNATTENDED

cat > /etc/apt/apt.conf.d/20auto-upgrades << 'AUTOUPGRADES'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
AUTOUPGRADES
echo "  Unattended security upgrades enabled"

# ---------------------------------------------------------------------------
# 14. External heartbeat monitor cron
# ---------------------------------------------------------------------------
echo "[14/14] Installing heartbeat monitor..."
chmod +x "$ORACLE_HOME/deploy/check-heartbeat.sh"

cat > /etc/cron.d/oracle-heartbeat << CRON
# Ping healthchecks.io every 2 minutes based on bot heartbeat age
*/2 * * * * ${ORACLE_USER} HEALTHCHECK_PING_URL="" ${ORACLE_HOME}/deploy/check-heartbeat.sh
CRON
echo "  Heartbeat cron installed (every 2 min)"

# Ensure state directory exists with correct ownership
mkdir -p "$ORACLE_HOME/state"
chown "$ORACLE_USER:$ORACLE_GROUP" "$ORACLE_HOME/state"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "  Setup complete!"
echo "======================================================================"
echo ""
echo "  ACTION REQUIRED before starting the bot:"
echo ""
echo "  1. Edit secrets in .env:"
echo "     sudo -u $ORACLE_USER nano $ENV_FILE"
echo ""
echo "     Required variables:"
echo "       ORACLE_KALSHI_API_KEY=<your-key>"
echo "       ORACLE_KALSHI_PRIVATE_KEY_PATH=$KEYS_DIR/kalshi.pem"
echo "       ORACLE_EXECUTION_MODE=shadow"
echo "       ORACLE_BANKROLL=<amount>"
echo ""
echo "  2. Copy your Kalshi RSA private key:"
echo "     scp /path/to/kalshi.pem ${DEPLOY_USER}@<host>:/tmp/kalshi.pem"
echo "     sudo mv /tmp/kalshi.pem $KEYS_DIR/kalshi.pem"
echo "     sudo chown $ORACLE_USER:$ORACLE_GROUP $KEYS_DIR/kalshi.pem"
echo "     sudo chmod 0400 $KEYS_DIR/kalshi.pem"
echo ""
echo "  3. Set up heartbeat monitoring (free):"
echo "     a. Create a check at https://healthchecks.io (2-min period, 5-min grace)"
echo "     b. Edit /etc/cron.d/oracle-heartbeat and set HEALTHCHECK_PING_URL"
echo ""
echo "  4. Start the bot:"
echo "     sudo systemctl start oracle-bot"
echo "     sudo journalctl -u oracle-bot -f"
echo ""
echo "======================================================================"
