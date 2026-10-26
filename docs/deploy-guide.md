# Deploy Guide — The Oracle on a VPS

Step-by-step instructions for deploying The Oracle in shadow mode on a fresh VPS. Tested on a small VPS (2 vCPU / 2 GB RAM) with Ubuntu 24.04.

## Prerequisites

- A VPS provider account
- SSH key on your local machine (`~/.ssh/id_ed25519.pub`)
- Git access to the repo. The public repo clones over HTTPS. A deploy key (Step 3) is only needed for a private fork.
- Kalshi API key + RSA private key (PEM file)
- Telegram bot token + owner chat ID (from BotFather)

## Step 1: Provision VPS

1. Create a project in your provider's console
2. Add your `~/.ssh/id_ed25519.pub` as an SSH key
3. Create a server:
   - Location: a US region
   - Image: Ubuntu 24.04
   - Type: shared vCPU, 2 vCPU / 2 GB RAM
   - SSH Keys: select your key
4. Note the IPv4 address

## Step 2: Create Deploy User

```bash
ssh root@<VPS_IP>

# Create non-root user
adduser <admin>
usermod -aG sudo <admin>

# Copy SSH key
mkdir -p /home/<admin>/.ssh
cp ~/.ssh/authorized_keys /home/<admin>/.ssh/authorized_keys
chown -R <admin>:<admin> /home/<admin>/.ssh
chmod 700 /home/<admin>/.ssh
chmod 600 /home/<admin>/.ssh/authorized_keys
exit
```

Verify: `ssh <admin>@<VPS_IP>`

## Step 3: GitHub Deploy Key (private fork only)

Skip this step if you clone the public repo over HTTPS.

On the VPS as your deploy user:

```bash
ssh-keygen -t ed25519 -C "oracle-deploy" -f ~/.ssh/github_deploy -N ""
cat ~/.ssh/github_deploy.pub
```

Add the public key to GitHub: **Repo → Settings → Deploy keys → Add deploy key** (read-only).

Configure SSH to use it:

```bash
cat >> ~/.ssh/config << 'EOF'
Host github.com
    IdentityFile ~/.ssh/github_deploy
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Verify: `ssh -T git@github.com`

## Step 4: Run Setup Script

The setup script needs the repo cloned first (it expects to clone into `/opt/oracle`, but `useradd --create-home` creates that directory with skeleton files).

`<repo-url>` is `https://github.com/<owner>/<repo>.git` for the public repo, or
`git@github.com:<owner>/<repo>.git` for a private fork with the Step 3 deploy key.

```bash
# Clone repo to a temp location first
cd /tmp
git clone <repo-url>
cd <repo>

# Run setup — this will attempt to clone into /opt/oracle
# If it warns about the directory not being empty, clean it first:
sudo rm -f /opt/oracle/.bash_logout /opt/oracle/.bashrc /opt/oracle/.cloud-locale-test.skip
sudo git clone <repo-url> /opt/oracle
# Private fork over SSH instead:
# sudo GIT_SSH_COMMAND="ssh -i /home/<admin>/.ssh/github_deploy" git clone <repo-url> /opt/oracle
sudo chown -R oracle:oracle /opt/oracle

# Now run setup (idempotent — safe to re-run). /opt/oracle is already a clone, so REPO_URL
# is not needed. Without a clone, the script exits unless REPO_URL is set.
sudo ./deploy/setup-vps.sh
```

The script installs: Python 3.13, uv, UFW firewall (SSH only), SSH hardening, systemd service, journald log rotation, unattended security upgrades, heartbeat cron.

## Step 5: Copy Secrets

### Kalshi PEM key

From your local machine:

```bash
scp /path/to/kalshi.pem <admin>@<VPS_IP>:/tmp/kalshi.pem
```

On the VPS:

```bash
sudo mv /tmp/kalshi.pem /opt/oracle/keys/kalshi.pem
sudo chown oracle:oracle /opt/oracle/keys/kalshi.pem
sudo chmod 0400 /opt/oracle/keys/kalshi.pem
```

### Environment variables

```bash
sudo nano /opt/oracle/.env
```

Required contents:

```
ORACLE_KALSHI_API_KEY=<your-kalshi-api-key>
ORACLE_KALSHI_PRIVATE_KEY_PATH=/opt/oracle/keys/kalshi.pem
ORACLE_TELEGRAM_BOT_TOKEN=<your-bot-token>
ORACLE_TELEGRAM_OWNER_CHAT_ID=<your-chat-id>
ORACLE_BANKROLL=<your-amount>
ORACLE_HEALTHCHECK_PING_URL=<your-healthchecks-url>
```

Everything else (execution mode, risk params, strategy config) comes from `config.yaml` with good defaults. Shadow mode is the default.

**Important:** All env vars use the `ORACLE_` prefix. Nested config (strategies, risk, etc.) lives in `config.yaml`, not `.env`. See `python/settings.py` for the full field list.

### Install Telegram dependency

```bash
sudo -u oracle bash -c "cd /opt/oracle && /usr/local/bin/uv sync --frozen --extra telegram"
```

## Step 6: Set Up Healthchecks.io

1. Create a free account at [healthchecks.io](https://healthchecks.io/)
2. Add a check: name=`oracle-bot`, period=2 min, grace=5 min
3. Copy the ping URL
4. Add it to `.env` as `ORACLE_HEALTHCHECK_PING_URL=https://hc-ping.com/<uuid>`
5. Edit the cron job to use the same URL:

```bash
sudo nano /etc/cron.d/oracle-heartbeat
# Set HEALTHCHECK_PING_URL="https://hc-ping.com/<uuid>"
```

## Step 7: Start the Bot

```bash
sudo systemctl start oracle-bot
sudo journalctl -u oracle-bot -f
```

You should see:
- Kalshi client connected
- WebSocket connected
- Weather strategy scanning
- Telegram bot started
- "Starting main loop (poll every 60s)"

Check Telegram — you should receive "The Oracle started."

Send `/status` to your Telegram bot to verify bidirectional communication.

## Step 8: Claude Code for `/agent` (Optional)

The `/agent` Telegram command spawns a Claude Code CLI subprocess that can read the full codebase, bot state, and recent logs to provide AI-assisted debugging. With `--max-turns 10`, Claude Code can use its tools (Read, Grep, Glob) to trace through source code, check config files, and inspect state — not just answer from a pre-assembled prompt.

### Install Claude Code on the VPS

```bash
# As the deploy user (<admin>), not root
curl -fsSL https://claude.ai/install.sh | bash

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
claude --version
```

### Authenticate the oracle user via OAuth

Claude Code authenticates via OAuth tokens, which use your Pro/Max subscription — no per-token API billing. The `oracle` system user needs its own Claude login (tokens are tied to the Unix user).

```bash
# Log in as the oracle user and authenticate
sudo -u oracle /home/<admin>/.local/bin/claude login

# Trust the working directory (one-time)
sudo -u oracle /home/<admin>/.local/bin/claude -p "hi" --output-format text --max-turns 1
```

The login command prints a URL — open it in your browser to complete the OAuth flow. Credentials are stored in `/opt/oracle/.claude/`.

OAuth tokens auto-refresh but may expire after days/weeks on a headless server. If `/agent` starts failing with auth errors, re-run `sudo -u oracle /home/<admin>/.local/bin/claude login` to refresh.

### Configure

Add to `/opt/oracle/.env`:

```bash
sudo nano /opt/oracle/.env
```

```
ORACLE_CLAUDE_CODE_BIN=/home/<admin>/.local/bin/claude
```

`ORACLE_CLAUDE_CODE_BIN` is the full path to the Claude binary (installed under the deploy user). This bypasses PATH lookup since the systemd service has a minimal PATH.

### Permissions

The deploy script (`update.sh`) automatically fixes execute permissions on the Claude binary after each deploy. If Claude auto-updates, the new version binary may need `o+rx` — the deploy script handles this.

### Test

```bash
sudo /opt/oracle/deploy/update.sh
```

Then send `/agent` from Telegram.

### Usage

```
/agent                              — default: "what is the bot doing?"
/agent why no signals for Denver    — freeform query
/agent is the weather API healthy   — targeted question
```

Claude Code runs with `cwd=/opt/oracle`, so it can read source files, `config.yaml`, `state/`, and `data/` to trace issues through the code. The timeout is 15 minutes to allow for multi-turn analysis.

### Notes

- Never run `pytest` from `/opt/oracle`, by hand or through `/agent`. Tests resolve relative `state/` paths against the cwd, and on 2026-09-16 a run there wrote fixture rows into the live `state/per_day_cap_blocks.jsonl`. Runtime paths now derive from `settings.state_dir` and `tests/conftest.py` sandboxes `ORACLE_STATE_DIR`, but the run still shares the live checkout and its `.env`.
- `/agent` blocks other Telegram commands while running (up to 15 min). The kill switch via `state/STOP` file still works independently: `sudo -u oracle touch /opt/oracle/state/STOP`

## Step 9: Optional Private Ops Checkout

Keep live-only material (config overlay, data snapshots, local notes) in a separate private repo. `deploy/update.sh` uses it when `$ORACLE_OPS` (default `/opt/oracle-ops`) is a git checkout:

- It runs `git pull --ff-only` in the ops checkout and records its short SHA in `state/.ops_sha`.
- It symlinks `data/snapshots` and `CLAUDE.local.md` from the ops checkout into `/opt/oracle`. It skips any path the code checkout still tracks.
- If `$ORACLE_OPS/config.live.sha256` exists, it passes `--expect-hash` with that file to `scripts/check_live_config.py`. The check runs before the restart.

Set `ORACLE_CONFIG_OVERLAY` in `/opt/oracle/.env` to the overlay YAML in the ops checkout. `python/settings.py` merges it over `python/config.yaml`.

Each checkout can use its own deploy key:

```bash
git config core.sshCommand "ssh -i <key> -o IdentitiesOnly=yes"
```

`update.sh` uses the legacy `DEPLOY_KEY` only when the code checkout has no `core.sshCommand`.

## Day-to-Day Operations

### Check status
```bash
sudo systemctl status oracle-bot
sudo journalctl -u oracle-bot --since "10 minutes ago" --no-pager
```

### Deploy code updates
```bash
sudo /opt/oracle/deploy/update.sh
```

### Restart
```bash
sudo systemctl restart oracle-bot
```

### Stop
```bash
sudo systemctl stop oracle-bot
```

### Kill switch (emergency)
Send `/kill` to the Telegram bot, or:
```bash
sudo -u oracle touch /opt/oracle/state/STOP
```

### Telegram commands
| Command | Action |
|---------|--------|
| `/status` | Bankroll, P&L, exposure, mode |
| `/positions` | Open positions |
| `/trades` | Last 10 trades |
| `/signals` | Last 10 signals |
| `/risk` | Exposure vs limits |
| `/kill` | Emergency stop (the bot exits) |
| `/unkill` | Remove `state/STOP`. Not reachable after `/kill`, because the bot has stopped; clear a kill over SSH (operations runbook, Kill Switch) |
| `/enable <name>` | Re-enable a circuit-broken strategy |
| `/loglevel` | Show current log level + pending auto-revert |
| `/loglevel debug <1-30>m` | Temporarily set DEBUG; auto-reverts to baseline |
| `/loglevel info` | Revert to baseline immediately |
| `/agent` | AI diagnosis via Claude Code (optional query) |
