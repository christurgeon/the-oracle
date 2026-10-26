# Operations Runbook

Procedures for monitoring, troubleshooting, and recovering The Oracle in production.

## Daily Monitoring

### Morning Check (~10 AM ET, after markets open)

1. **Telegram `/status`** — verify bot is running, no strategies disabled; check daily P&L
2. **Telegram `/positions`** — verify open position count is reasonable (< 15)
3. **Healthchecks.io** — confirm no missed pings overnight

### Evening Check (~6 PM ET, after weather settlements)

1. **Telegram `/status`** — review day's final P&L
2. **Telegram `/positions`** — should be near-zero after settlements
3. If positions remain open past settlement, investigate (see "Stuck Positions" below)

## Telegram Command Reference

All commands are owner-only — every handler checks `chat_id == ORACLE_TELEGRAM_OWNER_CHAT_ID` and silently ignores non-owner messages.

**Read / status:**
- `/status` — bankroll, P&L, drawdown, strategy status
- `/positions` — open positions
- `/trades` — recent executed trades
- `/signals` — recent signals, all statuses
- `/risk` — risk gauges and limits
- `/attribution [7d|30d|all]` — P&L decomposition by signal/city/side
- `/calibhealth` — calibration monitor health
- `/agent [query]` — AI-assisted diagnosis via Claude (optional freeform query)
- `/acks` — list current alert acknowledgements and suppression counts
- `/help` — command reference

**Control:**
- `/kill` — activate kill switch immediately; the bot stops
- `/unkill` — remove `state/STOP`. After `/kill` the bot has stopped and cannot receive it;
  clear a kill over SSH instead (see Kill Switch below)
- `/disable_reconciler` — pause the fills reconciler (process-local; survives until restart)
- `/enable_reconciler` — re-arm the fills reconciler
- `/enable <strategy>` — re-enable a circuit-broken strategy
- `/setexposure <N>` — set total-exposure cap (0<N≤100% or 0<N≤1 fraction)
- `/resetexposure` — revert to config baseline
- `/loglevel [debug <1-30>m|info]` — temporary log-level override with auto-revert; only `debug`/`info` allowed so WARNING-level alerts stay visible
- `/ack <code>` — record "reviewed, no action" for one alert code; only codes carrying an invalidating context (a fit date or artifact path) are acceptable — liveness and missing-artifact alerts cannot be acked, since they carry nothing that would make the finding "new" again and an ack would be a permanent mute

**Removing a mistaken acknowledgement:** there is no `/unack` command. To undo an `/ack`:
1. SSH to the VPS and stop the bot (`systemctl stop oracle-bot`).
2. Edit `state/alert_acks.json` and delete the mistaken code's entry from the `acked` object (leave `seen` and everything else untouched).
3. Restart the bot (`systemctl start oracle-bot`) — `_acked` is loaded from this file only at startup, so the edit has no effect until then.

Hand-editing this file is exactly the input `Orchestrator._ack_suppresses` must treat carefully: a `severity` value it doesn't recognize (missing, or not one of `INFO`/`WARNING`/`ERROR`/`CRITICAL`) is treated as no acknowledgement at all (fails OPEN), never as a silent mute — so a typo while hand-editing degrades to "alert re-arms," not "alert goes missing."

**Push alerts** (not commands): trade fills (batched per tick), circuit breaker trips, kill switch activation, strategy errors, startup/shutdown. Rate-limited (20 msg/min default); critical alerts bypass the limiter. All Telegram errors are swallowed — never crashes the orchestrator.

## Log Analysis

### Viewing Logs

```bash
# Live tail
journalctl -u oracle-bot -f

# Last hour
journalctl -u oracle-bot --since "1 hour ago"

# Errors only
journalctl -u oracle-bot --since "1 hour ago" | grep -iE "error|critical|exception"

# Signal generation activity
journalctl -u oracle-bot | grep "generated .* signal"

# Risk rejections
journalctl -u oracle-bot | grep "REJECTED"
```

### Healthy Log Patterns

- `Tick complete: N signals, M approved, ...` — every ~60s
- `Watchdog heartbeat loop active (ping every Ns)` — once at startup; the systemd watchdog pings themselves are not logged
- `state/heartbeat` — its timestamp updates every tick (no log line)
- `Strategy weather_edge generated N signal(s)` — signals found

### Red Flags

- `Tick failed — will retry next interval` — single occurrence is OK, consecutive is bad
- `CRITICAL: N consecutive tick failures` — bot is in trouble, check immediately
- `KILL SWITCH` — trading halted, investigate cause
- `STALE_DATA` — WebSocket disconnected or data source down
- `STRATEGY_DISABLED` — circuit breaker tripped
- `429` or `rate limit` — Kalshi API throttling, may need to reduce scan frequency
- `Failed to check fills` — order status unknown, check Kalshi dashboard manually
- `Kelly side flipped post-depth` — CRITICAL from `_apply_depth_adjustment`; a market's VWAP fill moved through model_prob, trade was correctly skipped; investigate if frequent

## Incident Response

### Bot Is Offline

1. Check systemd: `systemctl status oracle-bot`
2. If stopped, check why: `journalctl -u oracle-bot --since "30 min ago" | tail -50`
3. Common causes:
   - OOM kill: `dmesg | grep -i oom` — increase swap or reduce memory usage
   - Unhandled exception: check logs for traceback
   - Kill switch activated: check `state/STOP` file
4. Restart: `sudo systemctl restart oracle-bot`
5. Verify recovery: `journalctl -u oracle-bot -f` — watch for successful tick

### No Signals for Several Hours

1. Check strategy is enabled: Telegram `/status`
2. If disabled, re-enable: Telegram `/enable weather_edge`
3. Check if edge threshold is too high (all signals filtered):
   ```bash
   journalctl -u oracle-bot --since "2 hours ago" | grep "effective_edge.*below threshold"
   ```
4. Check data freshness — are forecasts being fetched?
   ```bash
   journalctl -u oracle-bot --since "1 hour ago" | grep -i "forecast\|ensemble"
   ```
5. Check if all markets are closed (weekend, holiday)
6. Turn on verbose logging briefly to see per-market skip reasons: Telegram `/loglevel debug 10m` (auto-reverts; `debug`/`info` only — WARNING+ is blocked so risk alerts stay loud)
7. **Check for orderbook depth-check exception.** All per-market orderbook calls run concurrently via `asyncio.gather`. If any single `_apply_depth_adjustment` call raises an unhandled exception, `gather` cancels in-flight peers and propagates the error up — the scan produces **0 signals** for that cycle rather than partial results. This looks identical to "no qualifying markets" unless you check for a traceback:
   ```bash
   journalctl -u oracle-bot --since "1 hour ago" | grep -iE "traceback|_apply_depth_adjustment|exception in"
   ```
   A one-off exception is harmless (next scan recovers automatically). Repeated failures indicate a bug in depth-check logic or a Kalshi API schema change — investigate before re-enabling.

### Circuit Breaker Tripped

1. Check which strategy and why: Telegram `/status` or `/risk`
2. Review recent trade performance: Telegram `/trades`
3. If caused by transient errors (not real losses), re-enable: `/enable <strategy>`
4. If caused by genuine losing streak, investigate before re-enabling:
   - Check if model calibration has drifted
   - Check if market conditions changed (new fee schedule, etc.)
   - Consider raising `edge_threshold` before re-enabling

### Stuck Positions (Not Settling)

1. Check Kalshi dashboard for the market's settlement status
2. If market has settled on Kalshi but bot hasn't processed it:
   - Restart the bot (reconciliation runs on startup in live/paper mode)
3. If the issue persists, check logs for settlement errors:
   ```bash
   journalctl -u oracle-bot | grep -i "settle"
   ```

### Bad P&L / Unexpected Losses

1. Pull latest data: `uv run python scripts/shadow_dashboard.py --pull-vps --refresh`
2. Check per-city breakdown for systematic issues
3. Compare bot's positions with Kalshi dashboard
4. Check if a specific city/variable is underperforming — consider adding to `excluded_city_vars`
5. Check if fee schedule changed: https://kalshi.com/fee-schedule

### Kalshi API Issues

1. **429 Rate Limits**: Bot has built-in backoff. If persistent:
   - Check if multiple instances are running: `pgrep -f orchestrator`
   - Reduce scan frequency in config
2. **5xx Errors**: Kalshi is having issues. Bot retries automatically for GET.
   - POST (order placement) does NOT retry on 5xx (to prevent double-orders)
   - Check Kalshi status page / Discord for outage info
3. **Auth Errors**: API key may have expired
   - Generate new key at kalshi.com
   - Update `.env` on VPS
   - Restart bot

### Data Source Outages

| Source | Symptom | Impact | Fallback |
|--------|---------|--------|----------|
| Open-Meteo | `RetryExhaustedError` in weather fetch | No new weather signals | Bot continues with stale forecasts; no trades generated if too stale |
| NWS CLI | Settlement data unavailable | Snapshot collection fails | Settlements still come from Kalshi API for P&L |
| Telegram | Bot stops responding | No alerts/commands | Trading continues; use SSH + journalctl |

## Stream monitoring: do not add `favorite_bias_audit.jsonl`

`favorite_bias_audit.jsonl` is deliberately absent from `audit_tripwires.STREAM_NAMES`. It is activity-gated behind the band filter and never rotates, so monitoring it would fire two permanent false alarms. Strategy liveness is covered by `_check_strategy_liveness` (`telegram.strategy_silent_alert_hours`). A per-scan heartbeat row would only shorten detection from ~30 h to minutes and was judged not worth touching the order-placing path.

## Deposits and withdrawals — you MUST re-anchor `peak_bankroll`

**Every deposit or withdrawal silently mis-states the drawdown and can halve every position.**

`RiskManager` computes `drawdown = 1 - bankroll / peak_bankroll` against an all-time high-water
mark of *equity*. Equity moves with capital flows, so **a withdrawal is indistinguishable from a
trading loss.** When `drawdown >= risk.max_drawdown_pct` (0.15), `check_signal` halves the
per-market cap via `risk.drawdown_size_reduction` (0.50) — on every trade, silently, until equity
recovers to `peak x 0.85`.

**A withdrawal without a re-anchor goes unnoticed.** `peak_bankroll` stays at the old high while
equity falls, so a large withdrawal reads as a large drawdown. Every position is then capped at
half size for weeks. Per-strategy sizing changes (`strategies.favorite_bias.*`) look like they
work but change nothing, because the clamp is the binding cap. Check filled dollars in
`state/trades.jsonl` after any sizing change.

### Procedure — run this after ANY transfer in or out

```bash
sudo systemctl stop oracle-bot
sudo cp /opt/oracle/state/risk_state.json /opt/oracle/state/risk_state.json.bak-$(date +%Y%m%d)
sudo python3 -c "import json,pathlib; p=pathlib.Path('/opt/oracle/state/risk_state.json'); \
  d=json.loads(p.read_text()); print('old peak:', d['peak_bankroll']); d['peak_bankroll']=0; \
  p.write_text(json.dumps(d))"
sudo systemctl start oracle-bot
journalctl -u oracle-bot -n 60 | grep -E "Hydrated risk state|Drawdown"
```

**Set it to `0`, never to an equity figure.** `hydrate_from_state` does
`max(state.peak_bankroll, self._bankroll)` and runs *after* Kalshi hydration, so `0` self-corrects
to live equity and drawdown becomes 0. A hardcoded number would be floored at the `$10,000` config
default (`bankroll` in `python/settings.py`) if that ordering ever changed — which would make the clamp *worse*, not better.

Expected after restart: `Hydrated risk state: peak=$<current equity>` and **no**
`Drawdown NN% >= 15.0% — capping position at $X` warnings. If that warning still fires, the reset
did not take.

Drawdown protection re-arms from the new peak — this re-anchors the mark, it does not disable it.

### After a withdrawal, also re-check the size-coupled knobs

Per-trade size is `risk.max_single_market_pct x effective_max_total_exposure_pct x bankroll`, and
the daily halt is `risk.max_daily_loss_pct x daily_start_bankroll`. Both scale with equity, so
they self-correct — but if you change per-trade size, `max_daily_loss_pct` must move with it or
the daily halt becomes the binding constraint on flow (a tight limit halts trading on a larger
share of days).

---

## Kill Switch

### Activating

- **Telegram**: `/kill` (preferred — confirms via Telegram)
- **SSH**: `sudo -u oracle touch /opt/oracle/state/STOP`
- **Automatic**: Bot creates STOP file after 20 consecutive tick failures

### Behaviour When Active

- All trading halts immediately
- Open resting orders are cancelled
- The bot exits with code 0, so systemd does not restart it
- Existing positions remain until settlement

### Deactivating

The bot is stopped, so Telegram `/unkill` cannot reach it. Use SSH:

1. `sudo rm /opt/oracle/state/STOP`
2. `sudo systemctl restart oracle-bot.service`

If `state/STOP` is still present, the bot halts again on its first tick.

At each start the bot logs `Kill switch inactive at startup` or `Kill switch ACTIVE at startup`.
  `scripts/pull_post_mortem_data.py` uses the "inactive" line to end a kill interval that a
  restart cleared. Do not change this text.

## Fills reconciler (trade-log fidelity)

### What it does

Periodic check that every Kalshi-confirmed fill made it into `state/trades.jsonl`. Backfills synthetic rows for any miss. Runs every 5 orchestrator ticks (~5 minutes) in live/paper mode when `fills_reconciler.enabled: true`.

### Reading the structured logs

Every tick emits a `reconciler_tick` log line with these fields (stable contract):

- `delta_count` — synthetic rows written this tick (0 = nominal)
- `pages_consumed`, `truncated_at_max_pages` — Kalshi pagination
- `consecutive_failures` — counter; 0 means last tick was clean
- `alert_state` — `clean` / `warning` / `critical` / `skipped`
- `last_safe_ts_unix_before` / `_after` — cursor advancement

Grep on the VPS:

```bash
ssh $ORACLE_VPS_HOST "journalctl -u oracle-bot.service --since '1h ago' | grep reconciler_tick | tail"
```

### Telegram alerts

Two distinct ladders — delta-driven (success-path) and failure-driven — share no cooldown state. A `CRITICAL` on one ladder cannot suppress the other.

Delta-driven (tick completed):
- `WARNING` on every non-clean tick where `delta > 0` and `delta ≤ alert_threshold_critical_delta`. Not cooldown-suppressed — sustained drift surfaces every tick.
- `CRITICAL` on truncation OR `delta > alert_threshold_critical_delta`. Within the 1h cooldown window the alert is downgraded to `WARNING` (still paged, just quieter). A clean tick clears the cooldown.

Failure-driven (tick raised):
- `WARNING` on the first failure and on every failure while `consecutive_failures < alert_threshold_consecutive`.
- `CRITICAL` once `consecutive_failures ≥ alert_threshold_consecutive` for a code-bug-class exception, with a 1h cooldown.
- `KalshiAPIError` always stays `WARNING` (API outages don't escalate).

### When to use `/disable_reconciler`

- The reconciler is firing false-positive duplicates.
- A Kalshi outage has lasted long enough that the WARNING flood is operationally annoying (rare — cooldown should handle this).
- A malformed fill (bad `side` or `count_fp`/`fee_cost`) is wedging the reconciler — it raises and aborts the tick every 5 minutes with no dedup token, so it never stops on its own. Confirm via `journalctl -u oracle-bot.service | grep reconciler_tick` showing repeated failures for the same `order_id`; the fix is upstream (contact Kalshi support or wait for a corrected fill) — `/disable_reconciler` only stops the repeated alert, it does not resolve the underlying data.

It's process-local; survives until orchestrator restart. For persistent disable, set `fills_reconciler.enabled: false` in `python/config.yaml` and redeploy.

### When to use `scripts/reconcile_fills.py`

Multi-day outage windows that exceed `startup_safety_window_hours = 24` (the periodic reconciler can't reach back further on its own). Always run with `--dry-run` first; `--apply` requires the lockfile (operator must `systemctl stop oracle-bot.service` first or the CLI will exit non-zero).

## Manual-close and cash-out classification

The FillsReconciler classifies every fill on `/portfolio/fills`. The v1 path routes
`SELL_MANUAL_CANDIDATE` through `_apply_manual_close`; the v1.1 path (see "Cash-out
classification" below) routes opposite-side `action="buy"` fills (Kalshi's "Cash Out"
button) through `_apply_cash_out`. Both paths share the same reconciler pipeline,
the same `/disable_reconciler` kill switch, and the same `FillsReconciler` lock
discipline — only the per-fill classification and apply-handler differ.

### Manual-close (v1) details

The FillsReconciler classifies every fill on `/portfolio/fills` and routes
the SELL_MANUAL_CANDIDATE outcome through `_apply_manual_close`.

**Flag (config.yaml under `fills_reconciler:`):**

```yaml
manual_close_enabled: false                  # shakedown default-off
manual_close_apply_limit_per_restart: 1      # canary; raise to 0 (unlimited) after first confirmed apply
shakedown_seen_oids_file: state/shakedown_seen_oids.json
```

**Verification grep recipes:**

```bash
# Watch shadow detections during shakedown
ssh $ORACLE_VPS_HOST \
  "journalctl -u oracle-bot.service --since '24h ago' | grep '\\[SHADOW\\] Manual close'"

# Per-tick classifier breakdown
ssh $ORACLE_VPS_HOST \
  "journalctl -u oracle-bot.service --since '24h ago' | grep reconciler_tick \
   | jq '.classifier_decision_counts, .apply_path_outcome_counts'"
```

**Rollback tiers:**

- Tier 1 (sub-second): `/disable_reconciler` Telegram command.
- Tier 2 (persistent): set `manual_close_enabled: false` in config.yaml, restart.
- Tier 3 (code revert): revert the change that introduced the feature, deploy, and restart `oracle-bot.service`.

If a SELL_MANUAL is incorrectly applied:
1. Append a compensating row `event="manual_close_reverted"` with sign-flipped
   fields and `metadata.reverts=<original_close_oid>`. (No CLI helper ships in
   v1; hand-write the one row.)
2. Rehydrate `_risk._positions` from Kalshi `/portfolio/positions` on restart.

**Long-term review triggers:**

- Signal: forensic audit after first 10 applied SELL_MANUAL events.
- Consider a per-`position_key` re-entry cooldown if re-entry loops are observed.

**Rollout playbook (Day-by-day after deploy):**

- **Day 0 (deploy):** Ship with `manual_close_enabled: false` (shakedown) and
  `manual_close_apply_limit_per_restart: 1` (canary). No state mutations possible.
- **Day 1-N (verify shadow detections):** Each time you manually close in the
  Kalshi UI, expect within ~30s a Telegram `[SHADOW] Manual close detected (NOT applied)`
  WARNING. Verify the `computed_pnl` matches your expectation. Log the event
  in your own operations log as `<date> | <ticker> | <computed_pnl> | confirmed`.
- **Graduate to canary (after n≥5 confirmed shadow detections + 14d floor):**
  Edit `python/config.yaml` → `manual_close_enabled: true`. SCP, restart. The
  apply-limit stays at 1 so only the FIRST manual close after the restart
  actually mutates state.
- **Canary verification (4-point checklist) on first applied close:**
  1. `grep manual_close state/trades.jsonl | tail -1` — verify the row's
     `realized_pnl`, `cost_basis_dollars`, `close_proceeds`, `exit_fees`.
  2. Telegram `/status` — verify bankroll delta equals the row's `realized_pnl`.
  3. Telegram `/positions` — verify the closed market is no longer listed.
  4. Telegram WARNING received with format `🔄 Manual close reconciled: ...`.
- **Unlimited (after canary passes):** Edit `python/config.yaml` →
  `manual_close_apply_limit_per_restart: 0`. SCP, restart. Subsequent closes
  apply immediately without per-restart caps.
- **Rollback at any tier:** `/disable_reconciler` (sub-second); config flag
  + restart (persistent); revert the change that introduced the feature (full revert).
  Hand-write a compensating `event="manual_close_reverted"` row if a wrong
  apply already wrote state (no CLI helper exists).

## Cash-out classification (v1.1)

The FillsReconciler classifies opposite-side `action="buy"` fills from
Kalshi's "Cash Out" button as `FillKind.BUY_CASH_OUT` and routes them
through `_apply_cash_out` (full or partial close). Defaults: feature
off, canary apply limit 1 per restart, P&L safety cap $500.

**Flag (`python/config.yaml` under `fills_reconciler:`):**

```yaml
cash_out_apply_enabled: false               # shakedown default-off
cash_out_apply_limit_per_restart: 1         # canary
cash_out_max_pnl_apply_dollars: 500.0       # catastrophic-bug detector
shakedown_seen_oids_cash_out_file: state/shakedown_seen_oids_cash_out.json
```

**Verification grep recipes:**

```bash
# Watch shadow detections during shakedown
ssh $ORACLE_VPS_HOST \
  "journalctl -u oracle-bot.service --since '24h ago' | grep '\\[SHADOW\\] Cash out'"

# Verify last applied cash-out
ssh $ORACLE_VPS_HOST \
  "grep '\"event\":\"cash_out\"' /opt/oracle/state/trades.jsonl | tail -1 | jq"

# List partial cash-outs (residual verification)
ssh $ORACLE_VPS_HOST \
  "grep '\"event\":\"cash_out\"' /opt/oracle/state/trades.jsonl | jq 'select(.close_type==\"partial\")'"
```

**Alert-class → script-mode mapping (with prerequisites):**

| Telegram alert | Apply path outcome | Script mode | Prerequisites |
|---|---|---|---|
| `Cash out P&L exceeded cap: ...` | `PNL_CAP_EXCEEDED_CASH_OUT` | `--apply` | Operator confirmed P&L is legitimate via manual recalc; reviewed the failing fill |
| `Cash out for {ticker} oid=...: position exists but no entry FILL rows` | `ZERO_COST_BASIS_CASH_OUT` | `--reset-dedup` | Missing FILL rows located + backfilled via `scripts/reconcile_fills.py` first |
| `Cash out contracts (...) exceeded local cost-basis contracts (...)` | `INVALID_PARTIAL_CASH_OUT` | `--reset-dedup` | Trade-log divergence root-caused + entry rows fixed manually |
| `Cash out reconciled: ...` (operator finds realized_pnl wrong on recalc) | `APPLIED_FULL_CASH_OUT` or `APPLIED_PARTIAL_CASH_OUT` | `--revert` | Apply-path bug confirmed via `cost_basis_for_position` manual run |
| `RACE_LOST_TO_SETTLEMENT` | (benign — auto-recoverable) | (no action) | None — `_settle_live` defensive guard handles it |

**`scripts/revert_cash_out.py` invocation recipes (OFFLINE-ONLY — bot must be stopped first):**

```bash
# CRITICAL: stop the bot before running the script. The script's pgrep
# check aborts if the orchestrator is detected.
sudo systemctl stop oracle-bot.service

# Mode 1: --reset-dedup (clears the oid from all five _co_*_oids sets).
# Use after backfilling missing trade-log rows (ZERO_COST_BASIS, INVALID_PARTIAL).
ssh $ORACLE_VPS_HOST \
  "cd /opt/oracle && python scripts/revert_cash_out.py --reset-dedup \
   --order-id <oid> --yes-i-am-sure"

# Mode 2: --revert (undo an applied cash_out).
# Writes a cash_out_reverted row + adjusts state/risk_state.json's
# daily_pnl + removes the matching strategy_trackers entry.
# Refuses if the oid already has a cash_out_reverted row (exit 9).
ssh $ORACLE_VPS_HOST \
  "cd /opt/oracle && python scripts/revert_cash_out.py --revert \
   --order-id <oid> --reason '<free text>' --yes-i-am-sure"

# Mode 3: --apply (hand-apply a fill the safety knob blocked).
# Re-fetches the fill from /portfolio/fills; computes realized_pnl via
# _apply_cash_out's formula; writes cash_out_force_applied row.
# Refuses if the oid already has a cash_out_force_applied row (exit 10).
# NOTE: v1.1 day-1 ship: _fetch_fill_for_oid is a stub. Use the hand-edit
# fallback below until KalshiClient wiring lands.
ssh $ORACLE_VPS_HOST \
  "cd /opt/oracle && python scripts/revert_cash_out.py --apply \
   --order-id <oid> --reason '<free text>' --yes-i-am-sure"

# Restart the bot.
sudo systemctl start oracle-bot.service
```

**Script exit codes:**

| Code | Meaning |
|------|---------|
| 0 | success |
| 2 | orchestrator running (pgrep aborted) |
| 3 | missing `--yes-i-am-sure` |
| 4 | `--state-dir` does not exist |
| 5 | `--revert` no `cash_out` row found for oid |
| 6 | `--apply` no fill found for oid |
| 7 | `--apply` cost basis is zero |
| 8 | `--apply` contracts overflow |
| 9 | `--revert` oid already reverted (double-revert refused) |
| 10 | `--apply` oid already force-applied (double-apply refused) |
| 11/12/13 | `risk_state.json` / `fills_reconciler.json` / `trades.jsonl` missing or unreadable |

**Crash-recovery for partial mutation:**

Both `--revert` and `--apply` write three things in sequence:
1. Append a row to `state/trades.jsonl`
2. Mutate `state/risk_state.json` (atomic write)
3. Mutate `state/fills_reconciler.json` (atomic write)
4. Best-effort Telegram POST

If the script SIGKILLs between steps 1 and 2 (rare; possible only via `kill -9` or kernel panic), the row is on disk but `daily_pnl` was not adjusted. The script's double-application guards (exit 9 / exit 10) refuse to re-apply. Recovery:
- For `--revert` after partial crash: read the orphan `cash_out_reverted` row to recover the original `realized_pnl`; manually subtract it from `state/risk_state.json:daily_pnl` and remove the matching tracker entry; restart bot.
- For `--apply` after partial crash: read the orphan `cash_out_force_applied` row; manually adjust `daily_pnl` + tracker; restart bot.

`SIGTERM` / `Ctrl-C` / normal exit are safe because the `with` block flushes the file before the next mutation.

**Fallback: hand-edit procedure (if `revert_cash_out.py` is broken at 3 AM):**

1. `sudo systemctl stop oracle-bot.service` — orchestrator MUST be stopped.
2. Append the appropriate `cash_out_reverted` row to `state/trades.jsonl`:
   ```json
   {
     "event": "cash_out_reverted",
     "order_id": "<new uuid>",
     "market_ticker": "<from original row>",
     "position_key": "<from original row>",
     "mode": "live",
     "timestamp": "<now iso>",
     "applied_at": "<now iso>",
     "reverts_order_id": "<original cash_out oid>",
     "closed_side": "<from original>",
     "hedge_side": "<from original>",
     "close_type": "<full|partial>",
     "contracts_offset": <integer>,
     "cost_basis_offset_dollars": <from original>,
     "cost_basis_offset_fees": <from original>,
     "cost_basis_residual_dollars": <from original>,
     "cost_basis_residual_fees": <from original>,
     "realized_pnl": <SIGN-FLIPPED from original>,
     "metadata": {
       "operator": "<env user>",
       "reason": "<free text>",
       "reverts": "<original cash_out oid>"
     }
   }
   ```
3. Hand-edit `state/risk_state.json`: subtract `realized_pnl` from `daily_pnl`. **Do NOT mutate `bankroll` or `positions`** — they are not in the persisted state file and self-heal from Kalshi on next hydrate.
4. Hand-edit `state/risk_state.json`'s `strategy_trackers["<strategy>"]["trades"]` list: find the last entry matching the wrong cash-out's pnl + won, remove it.
5. `sudo systemctl start oracle-bot.service`. On startup, `hydrate_state` reads the edited JSON; `hydrate_live_positions` re-syncs from Kalshi. The hand-edited values propagate.

**There is no Python-REPL-into-running-process fallback** — the architecture doesn't support attaching a REPL to a live orchestrator. The hand-edit of `state/risk_state.json` IS the canonical fallback.

**Rollback tiers:**

- Tier 1 (sub-second): `/disable_reconciler` Telegram command.
- Tier 2 (persistent): `cash_out_apply_enabled: false` in `python/config.yaml`, restart. **NOT bit-identical to v1.0** — classifier still produces `[SHADOW]` alerts.
- Tier 3 (bit-identical to v1.0): revert the change that introduced v1.1, deploy, and restart `oracle-bot.service`.

**Rollout playbook:**

- **Day 0:** ship with `cash_out_apply_enabled: false` and `cash_out_apply_limit_per_restart: 1`.
- **Days 1-N:** verify `[SHADOW]` Telegram alerts for each Cash Out you perform in the UI.
- **Graduate to canary** (after n≥3 confirmed shadow detections + 7d floor): set `cash_out_apply_enabled: true`. SCP, restart.
- **Canary verification:** 4-point checklist on first applied cash-out:
  - grep `cash_out` row in `state/trades.jsonl` matches the Cash Out event timing
  - `/status` shows realized_pnl delta
  - `/positions` shows the position popped (full) or reduced (partial)
  - Telegram `Cash out reconciled` WARNING received
- **Unlimited** (after canary passes): `cash_out_apply_limit_per_restart: 0`.

**Known v1.1 limitations:**

- Cold-start downtime cash-outs inherit v1's P&L attribution gap (the row is on disk but `_daily_pnl` is lost until next-day rollover; Kalshi-authoritative bankroll self-heals on hydrate).
- Partial cash-outs accept a documented win-rate bias on the per-strategy circuit breaker (the residual settles separately and produces a second tracker entry).
- `--apply` mode's `_fetch_fill_for_oid` is currently a stub. Use the hand-edit fallback for any `PNL_CAP_EXCEEDED_CASH_OUT` operator action until KalshiClient wiring lands.
- Row-written-then-risk-mutation-crashes loses `daily_pnl` attribution (same risk class as v1 manual-close). Bankroll self-heals from Kalshi; only `_daily_pnl` is permanently off until next-day rollover.

## Deployment

### Routine Update

```bash
ssh $ORACLE_VPS_HOST
cd /opt/oracle
./deploy/update.sh
```

### Rollback

```bash
ssh $ORACLE_VPS_HOST
cd /opt/oracle
git log --oneline -5          # find the commit to roll back to
git checkout <commit-hash>
sudo systemctl restart oracle-bot
```

### Config Changes

1. Edit config on VPS: `nano /opt/oracle/python/config.yaml`
2. Restart to pick up changes: `sudo systemctl restart oracle-bot`
3. Config is NOT hot-reloaded — restart is required

## Known Data Quirks

### Chicago NWS station (resolved 2026-03-20)

Kalshi's `KXHIGHCHI` / `KXLOWTCHI` markets settle on **Midway (KMDW)**, not O'Hare (KORD). Before 2026-03-20, `python/data/nws.py` fetched from ORD. Midway typically runs 1-3°F warmer (urban heat island), occasionally 2-5°F — enough to flip brackets.

- `weather.py` GFS coordinates were always correct (Midway); only NWS CLI settlement fetch was wrong.
- `scripts/calibrate.py` automatically excludes CHI settlements before 2026-03-20.
- `scripts/reconcile_settlements.py` caught 4 mismatches via live Kalshi comparison, all CHI, all explained by the station difference.
- All other cities (NYC/Central Park, MIA, LAX, DEN, AUS, SEA) have always matched between forecast station, NWS CLI station, and Kalshi settlement.

If a future city's settlement station is in doubt, cross-check via `reconcile_settlements.py` against ≥30 settled markets before trusting calibration on it.

## Known Baselines

Empirical baselines worth remembering before pitching new forecast-side strategies.

### Kalshi market is better-calibrated than raw GFS at T-1 evening

Measured 2026-05-20 against the snapshot corpus (60 usable days, ~4,700 lead=1 GFS rows joined with the evening-slot `markets.json` and NWS settlements):

| Quantity | Value |
|---|---|
| Pearson `corr(market_implied_prob, gfs_implied_prob)` | **0.11** full corpus; **0.15** ECMWF subset |
| Market Brier | **0.111** |
| Raw GFS Brier | **0.195** |
| `mean(realized − market)` | within ±0.03 across all GFS-prob quintiles |
| Per-cell Pearson range | −0.29 (DEN/low, MIA/high) to +0.32 (LAX/low, AUS/low) |

**Implication:** Kalshi prices have already absorbed the GFS-vs-realized bias. Any new strategy whose thesis is "market follows raw GFS so we fade the model" is structurally non-viable — pre-registered T2 test failed 2026-05-20 with Pearson 0.11 against a 0.70 threshold, and a direct fade backtest lost $0.035/event on n=782 (t=−2.47, p=0.014). Two cells (LAX/high, AUS/low) showed marginally positive ECMWF-subset fade returns but did not survive Bonferroni and collapsed back in the full corpus — classic in-sample selection; do not chase them without an *ex ante* mechanism.

The existing weather edge strategy operates against a counterparty that is at least as well-calibrated as our own raw model. Forecast-side alpha must therefore come from somewhere the market isn't already pricing — microstructure (timing, liquidity, settlement-rule edges like boundary-grazing), or signals the daily-cadence market hasn't absorbed (e.g., higher residual moments, intraday observations). Pure "improve GFS calibration" work has a bounded ceiling vs market and should be sized accordingly.

## One-Bet-Per-Event Cap Re-Evaluation

Shipped 2026-05-03 — caps weather signals to one trade per
`(city, var, forecast_date)` event, keeping the highest
`effective_edge` candidate.

The original plan was to re-evaluate at n>=80 live cap-active trades or on 2026-05-17,
whichever came first. That review ran on 2026-07-04 (see "Validation outcome" below).
Use the procedure below to repeat it.

### Procedure

1. **Pull fresh trades + settlements**

   ```bash
   uv run python scripts/daily_trade_report.py --pull-vps --refresh
   ```

2. **Loss-rate check** — target: `loss_rate <= 25%` on cap-active live trades.

3. **Calibration-ratio check** — target: `<= 2.5x` (avg model p(loss) vs actual loss rate).

4. **Adverse-selection check** — does the cap drop legs that would have won?

   For each event in `state/trades.jsonl` whose metadata has a non-empty
   `event_dropped_tickers`:
   a. Extract `(city, var, forecast_date)` from the kept signal's metadata.
   b. Read `data/snapshots/<forecast_date>/<latest_run>/settlements.json` for
      the actual NWS observation.
   c. For the kept ticker AND each dropped ticker, compute whether the
      actual temp landed in `[floor_strike, cap_strike]`.
   d. Count events where ANY dropped leg would have won AND the kept leg lost.

   **Threshold:** if >10% of multi-bet events show this pattern, the
   `effective_edge`-as-tie-break is wrong-headed and we need to revisit
   (likely candidates: distance-to-strike or neighbor-aware sizing).

   **DO NOT** use `scripts/fetch_shadow_settlements.py` for dropped tickers —
   that script reads `trades.jsonl` and dropped tickers were never traded.

### Disabling the cap (rollback)

Set in `python/config.yaml` under `strategies.weather`:

```yaml
    max_bets_per_day_per_event: null
```

Restart the orchestrator (`sudo systemctl restart oracle-bot`). The cap is bypassed entirely; all weather candidates that pass the other gates become signals. Note: this is NOT a revert to the pre-2026-05-21 per-scan cap; that path has been deleted. The per-scan cap was removed and cannot be restored by config.

### Loosening (operator-tunable)

Set in `python/config.yaml`:

```yaml
    max_bets_per_day_per_event: 2   # or higher
```

Restart. The cap allows up to N commits per `(city, var, forecast_date)` per UTC day. Current data (2026-05-20) does not support values >1; only use after a regime change is empirically confirmed.

### Resetting cap state mid-day

Cap state resets automatically at UTC midnight and on orchestrator restart (state hydrates from `state/trades.jsonl`). To force a reset mid-day, `sudo systemctl restart oracle-bot`.

### Falsification criteria (30-day post-deploy)

Revert to `max_bets_per_day_per_event: null` if ANY:

1. **ROI regression.** Per-day-cap-active live ROI on cap-fired events (n≥60) is lower than the per-scan baseline ROI on a matched-N comparison from 2026-05-03 → deploy date.
2. **SST regime flip.** Post-deploy bootstrap of SST-vs-Top-1 yields p(SST > Top-1) > 70%.
3. **Over-firing.** Cap drops > 60% of would-have-been-signal candidates over a 14-day rolling window.
4. **Calibration ratio regression.** On per-event-capped trades the `calibration_ratio` (model p(loss) vs actual loss rate) stays > 2.0× after 14 days, or climbs above the pre-deploy baseline.

**Validation outcome (author's deployment, run 2026-07-04, overdue from ~2026-06-20): UNDERPOWERED → KEEP-CAP (moot).**
Event count cleared the bar (109 cap-fired events, 87 in the live 05-21→06-07 window) but dollar
power did not: the realized live winner book on cap-fired events is 34 trades
(−24.7% ROI, statistically meaningless), and the idealized dropped-alt reconstruction (+12–13% ROI,
side recovered from `raw_ensemble_prob` vs `exec_price`, win rule 99.4% validated) ignores the
same-event correlation the cap exists to control. Criterion 3 NOT-FIRED (max 14-day drop share
59.3% vs the 60% trigger); criteria 2 and 4 not computable from the blocks file alone.
weather_edge went shadow-only 2026-06-07, freezing the live corpus permanently — the cap is
operationally moot (favorite_bias uses its own separate, non-logging `PerDayEventCap` with
`audit_writer=None`). Decision: keep `max_bets_per_day_per_event: 1`; reopen the test only if
weather_edge is un-shadowed. The pre-registered "delete the research docs" cleanup is deferred
until weather_edge's fate is decided, since the validation was underpowered rather than conclusive.

## EMOS operations (no cells enabled in shipped config)

The shipped config sets `weather.emos_enabled_cells: []`. Every tradeable cell uses the
isotonic calibrator (`weather.isotonic_path`, `isotonic_v3`) for live decisions. EMOS output
is logged in shadow only.

To put a cell on EMOS, add it to `weather.emos_enabled_cells` after it passes the per-cell
promotion gate (root `CLAUDE.md`, "Per-cell EMOS calibrator dispatch"). For an enabled cell,
`live_calibrator_kind == "emos_raw"` is recorded on every eval and candidate audit row, and
`model_probability` flows from `emos_prob_raw` into Kelly sizing.

**Rollback** (single config flip, bit-identical revert to all-isotonic): set
`weather.emos_enabled_cells: []` in `python/config.yaml` and restart. The
isotonic artifact and dispatcher stay in place; EMOS just stops being chosen.

**History (author's deployment).** The Stage A cutover (2026-05-11) started live EMOS dispatch.
NYC/low rolled back 2026-05-18 (live BSS_emos_vs_iso=−6.6%). LAX/high rolled back and was
re-excluded 2026-05-18 (ROI=−39.6% in 7 days; d=7.19 → EMOS σ=6.8°F structurally
incompatible with 1°F brackets; override revoked in `data/emos/overrides.json`). NYC/high and
CHI/low were demoted 2026-05-21 (the bracket-window fix exposed a mis-calibrated μ). MIA/high
and MIA/low rolled back to isotonic_v2 2026-05-24 per harness winner certification (LB +0.014,
+0.031, 5/5 folds; MIA/low live EMOS ROI was −37.2%). LAX/low, under an `unstable_d` override,
was the last EMOS cell. It was suspended to isotonic on 2026-06-05.

### EMOS_HALT kill switch
- Pause EMOS shadow + live without touching isotonic cells: `ssh $ORACLE_VPS_HOST 'touch /opt/oracle/state/EMOS_HALT'`. Live EMOS cells fall back to isotonic via the dispatcher's `isotonic_fallback` path; shadow logging on the remaining cells pauses.
- Resume: `ssh $ORACLE_VPS_HOST 'rm /opt/oracle/state/EMOS_HALT'`.
- Operators must allow up to one full scan-cycle (≤5 min under default `scan_interval_seconds: 300`) for noise to stop fully — the file check fires at `generate_signals()` start, not mid-call.

### Smoke-test alert tiering
- `weather.emos_shadow_log: false`: load/smoke failures log WARNING only, no Telegram.
- `weather.emos_shadow_log: true`: load/smoke failures log ERROR + Telegram (subject to per-(failure_bucket, utc_hour) dedup and 6/hour global cap).

### Calibration analysis: which audit fields are EMOS inputs?
On `eval_audit.jsonl` rows with `live_calibrator_kind == "emos_raw"`, the inputs the live calibrator actually saw are **`emos_mu_raw` and `emos_sigma_raw`** — not `raw_ens_prob` / `ens_std`. The latter two fields are computed from the bias-corrected, spread-inflated ensemble; the EMOS calibrator is fed the deep-copied pre-correction ensemble (`raw_member_sink`). For cells with a non-zero bias correction (e.g. NYC/low at `bias_corr_f = −0.246`) joining `raw_ens_prob` against `calibrated_prob` on an `emos_raw` row will mis-report what the live decision actually consumed. When bias correction is retired this asymmetry disappears.

### Coefficient artifact freshness
- Refit cadence: every gate evaluation (30-day, 60-day, contingency, hard-kill checkpoints) MUST run on coefficients fit against the latest available corpus.
- Run: `uv run python scripts/fit_emos.py`. Output: `data/emos/coefficients.json` + `data/emos/coefficients.diagnostics.json` + `data/emos/coefficients_history/<sha256>.json`.

### Frozen reference window backup (done 2026-05-09; historical)
This one-time step ran in the author's deployment on 2026-05-09, before the flip of
`weather.emos_shadow_log` to `true`. It is kept as a record. The steps were:
1. SSH to VPS: `ssh $ORACLE_VPS_HOST`.
2. Copy frozen window: `cp /opt/oracle/state/eval_audit.jsonl /opt/oracle/data/emos/reference_window_eval_audit.jsonl.frozen`.
3. Copy any rotated siblings covering [2026-05-01, 2026-05-08]: `cp /opt/oracle/state/eval_audit.jsonl.2026-05-* /opt/oracle/data/emos/`.
4. Verify size: `wc -l /opt/oracle/data/emos/reference_window_eval_audit.jsonl.frozen` (~few MB expected).
5. Then deploy with the flag flip.

At the time, `audit_log_retention_days` was 14. Without the copy, the day-1 dry-run
reconciliation could race the rotation cliff at ~2026-05-22, and the window could not be rebuilt.

### Files that survive audit-log retention
The following EMOS files MUST be included in any VPS backup/snapshot rotation:
- `data/emos/coefficients.json` (current)
- `data/emos/coefficients_history/<sha256>.json` (180-day archive)
- `data/emos/reconciliation_envelope.json` (sealed day-1 anchor; see analyzer docs)
- `data/emos/reference_window_eval_audit.jsonl.frozen` (frozen pre-flag-flip window)

The last two are produced locally (`scripts/analyze_emos_shadow.py` writes both on its first
run; the frozen file can also come from the manual copy above). Neither ships in this repository.

### Daily decision-window opening alerts (not implemented)
No job pings when a cell crosses n≥30, n≥50 or n≥150 settled outcomes. Run `uv run python scripts/analyze_emos_shadow.py --dry-run --start-date <D> --end-date <today>` weekly and read the per-cell n in the output to know when each gate becomes runnable.

## KXRAIN shadow observer (logging only, zero orders)

In-process observer for Kalshi's KXRAIN daily-rain markets. It builds a forward corpus that **cannot be
rebuilt**: resting book size is not in quote candles, and Kalshi 404s markets older than ~70 days. It places
no orders. Config: `kxrain_shadow` in `python/config.yaml`. Enable with `kxrain_shadow.enabled: true`; track status in your own work log.

### Enable / roll back

- **Before enabling:** `df -h /opt` must show ≥ 2.75 GiB free. Record `wc -l /opt/oracle/state/trades.jsonl`,
  and confirm `/opt/oracle/state/.build_sha` matches the deployed HEAD.
- **Stage 1:** `enabled: true`, `tick_interval_seconds: 1800` for 24 h. **Stage 2:** 600. Deploy each change
  with `deploy/update.sh`, which restarts the bot.
- **Roll back:** `enabled: false` + redeploy. No code revert, and no data is lost.
- `update.sh` re-runs itself once after `git pull`, so a change to the script applies in the same deploy.
  If `.build_sha` is ever missing or stale:
  `sudo -u oracle sh -c 'git -C /opt/oracle rev-parse --short HEAD > /opt/oracle/state/.build_sha'`, then restart.

### Files (all in `/opt/oracle/state/`)

| File | What |
|---|---|
| `kxrain_ticks.jsonl` | One row per tick attempt, including ticks with no city in the entry window. First file to appear. |
| `kxrain_decisions.jsonl` | One row per open market per in-window tick |
| `kxrain_book.jsonl`, `kxrain_trades.jsonl` | Order book and trade prints for in-band markets |
| `kxrain_settlements.jsonl`, `kxrain_settled_index.json` | Daily sweep at 14:00 UTC |
| `kxrain_ledger.json` | Virtual entries |

### Health checks

- **After enable (first hour):** `kxrain_ticks.jsonl` exists and grows by one row per tick. Each row has
  `observer_version` = the build SHA (not `"unknown"`) and the configured `tick_interval_seconds`.
  `trades.jsonl` line count is unchanged (zero orders).
- **Daily:** from ~15:00 UTC (10:00 LST in the east), `kxrain_decisions.jsonl` grows. After 14:00 UTC,
  `kxrain_settlements.jsonl` gains rows.
- **Weekly:** `du -sh /opt/oracle/state` (expect +135–170 MB/month), and
  `uv run python scripts/kxrain_backfill_settlements.py --dry-run --log-dir /opt/oracle/state`.
  A sweep outage raises no alert between day 28 and day 70, and after day 70 the lost settlements cannot
  be recovered. Run the backfill without `--dry-run` if dates are missing.

### Alerts (Telegram)

| Alert | Meaning / action |
|---|---|
| CRITICAL `KXRAIN task … died` | Observer or sweep task exited. Restart the bot. |
| CRITICAL `KXRAIN observer disabled … GiB free` | Disk guard tripped. It stays off for the process lifetime. Free space, restart. |
| WARNING `liveness: no decision row since …` | Cities are in window but no decision rows reached disk for 90 min. Check logs. |
| WARNING `consecutive failed ticks` | Ticks are failing. Check `journalctl` for `KxRainObserver`. |
| WARNING `listing came back incomplete` | `/markets` truncated or aborted. The census is suppressed while this lasts. |
| WARNING `settlement sweep anomaly` | The sweep may be dead. Run the backfill `--dry-run`. If it reports rows on disk that are NOT in the index, run it with `--reconcile` before the next 14:00 UTC sweep, or that sweep writes them again. |

## Daily trade report cron

Systemd timer that runs `scripts/daily_trade_report.py --refresh --json …` on the VPS daily at **11:00 UTC** (06:00 ET — Kalshi-quiet window between morning prep and US weather-market open). Outputs land under `/opt/oracle/state/reports/daily-report/{YYYY-MM-DD}.json`; files older than 90 days are pruned by `ExecStartPost=`. Failures route to Telegram via the reusable `oracle-alert-failure@.service` template — any future scheduled job can opt into the same alert path with a one-line `OnFailure=oracle-alert-failure@%N.service` directive.

Unit files (in-repo): `deploy/oracle-daily-report.service`, `deploy/oracle-daily-report.timer`, `deploy/oracle-alert-failure@.service`. Wrappers: `deploy/run-daily-report.sh`, `deploy/oracle-alert-failure.sh`. Installer: `deploy/install-daily-report.sh` (idempotent — safe to re-run).

### Schedule

- **Cadence:** daily at `11:00:00 UTC` (`OnCalendar=*-*-* 11:00:00 UTC`, `AccuracySec=1min`).
- **Post-boot:** `OnBootSec=10min` defers the first post-reboot run away from oracle-bot startup.
- **Catch-up:** `Persistent=true` triggers a missed run on next boot (e.g., if the VPS was off during 11:00 UTC).
- **Retention:** `*.json` reports older than 90 days are pruned by the unit's `ExecStartPost=find … -mtime +90 -delete`.

### Manual run (no schedule wait)

```bash
ssh $ORACLE_VPS_HOST
sudo systemctl start oracle-daily-report.service
sudo journalctl -u oracle-daily-report -f
ls -la /opt/oracle/state/reports/daily-report/
```

A successful run leaves a journal exit code of 0 and a dated JSON file. The unit is a `Type=oneshot`, so `systemctl status` will show `inactive (dead)` immediately after — that is the expected end state.

Note: this overwrites today's JSON. To preserve the original, copy it aside before re-running (e.g. `cp /opt/oracle/state/reports/daily-report/$(date -u +%Y-%m-%d).json /tmp/saved-report.json`).

### Failure-alert drill

Fire a synthetic failure to validate the Telegram path (no edits to real units required):

```bash
sudo systemd-run --unit=oracle-fake-fail.service \
    --property='OnFailure=oracle-alert-failure@%N.service' \
    /bin/false
```

Expected: within a few seconds, a Telegram message arrives with subject `⚠️ oracle-fake-fail.service failed` and the last 30 journal lines as the body. Confirms `%N` expansion, `EnvironmentFile=` propagation, and curl reachability. If no alert arrives, check `journalctl -u oracle-alert-failure@oracle-fake-fail` for the actual error.

### Cleanup (fully reversible)

```bash
sudo systemctl disable --now oracle-daily-report.timer
sudo systemctl stop oracle-daily-report.service 2>/dev/null || true
sudo rm /etc/systemd/system/oracle-daily-report.timer
sudo rm /etc/systemd/system/oracle-daily-report.service
sudo rm /etc/systemd/system/oracle-alert-failure@.service
sudo rm /opt/oracle/deploy/run-daily-report.sh /opt/oracle/deploy/oracle-alert-failure.sh
sudo systemctl daemon-reload
sudo systemctl reset-failed
# Optional: preserve JSON history for forensics, or remove:
sudo rm -rf /opt/oracle/state/reports/daily-report/
```

Then `git rm` the six `deploy/*` files (`oracle-daily-report.{service,timer}`, `oracle-alert-failure@.service`, `run-daily-report.sh`, `oracle-alert-failure.sh`, `install-daily-report.sh`) and revert the script patch + this section. No `.env` changes, no DB state to migrate.
