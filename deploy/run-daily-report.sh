#!/usr/bin/env bash
# run-daily-report.sh — Wrapper for the daily trade report systemd unit.
#
# Builds the dated output path under state/reports/daily-report/ and invokes
# the report script via the project venv. Date logic lives here (not in
# ExecStart=) so the unit file doesn't need /bin/bash -c.
set -euo pipefail

cd /opt/oracle
mkdir -p state/reports/daily-report
out="state/reports/daily-report/$(date -u +%Y-%m-%d).json"
exec /opt/oracle/.venv/bin/python scripts/daily_trade_report.py --refresh --json "$out"
