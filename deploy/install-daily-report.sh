#!/usr/bin/env bash
# install-daily-report.sh — Idempotent installer for the daily-report systemd
# cron and the reusable alert-on-failure template.
#
# Mirrors setup-vps.sh conventions: install -m for explicit modes, daemon-reload
# after unit changes, is-enabled --quiet guard before enable so re-runs are
# no-ops.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo install -m 0644 "$SCRIPT_DIR/oracle-daily-report.service" /etc/systemd/system/oracle-daily-report.service
sudo install -m 0644 "$SCRIPT_DIR/oracle-daily-report.timer" /etc/systemd/system/oracle-daily-report.timer
sudo install -m 0644 "$SCRIPT_DIR/oracle-alert-failure@.service" /etc/systemd/system/oracle-alert-failure@.service

# Wrapper scripts (run-daily-report.sh, oracle-alert-failure.sh) live in this
# directory already after `git pull` with +x tracked in git — no install step
# needed. Re-copying them to their own directory would error with "same file".

sudo systemctl daemon-reload
sudo systemctl is-enabled --quiet oracle-daily-report.timer || sudo systemctl enable oracle-daily-report.timer
# 'start' arms the schedule. The first report fires at 11:00 UTC;
# 'Persistent=true' may catch up immediately if a prior scheduled run was missed.
sudo systemctl start oracle-daily-report.timer

sudo systemctl list-timers oracle-daily-report.timer --no-pager
