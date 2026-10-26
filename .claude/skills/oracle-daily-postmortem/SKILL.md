---
name: oracle-daily-postmortem
description: Use when investigating Oracle's trading performance for a specific past date — produces a comprehensive markdown post-mortem with per-trade narrative, pattern analysis, and systemic-only fix recommendations. Always for yesterday or earlier; never today.
---

# Oracle Daily Post-Mortem

Investigative process for analyzing a single past day's trading performance.
Produces a structured markdown report at `docs/post-mortems/YYYY-MM-DD.md` (gitignored)
with per-trade walkthroughs, pattern detection, and systemic-only fix recommendations.

## When to use

- Investigating a bad day's P&L.
- Understanding a notably positive day.
- Periodic deep audit (weekly or after notable events).

**Do NOT use for** routine morning briefings — `daily_trade_report.py` already serves that role.

## Hard rules

1. **Never analyze today or any future date.** Weather markets settle on the next-day NWS
   Daily Climate Report; today's high/low contracts have no settlement data until tomorrow
   morning. Earliest analyzable date: `today_utc - 1`. If the user asks for today,
   refuse and explain why.
2. **Never write a market-sentiment narrative.** The trade log captures market state at
   entry only — there is no settlement-time market-price snapshot anywhere in the system.
   Every paragraph in "Model performance vs outcome" must cite an entry-time data point
   we have. Do NOT invent settlement-time market psychology.
3. **Never recommend fixes for individual trades.** Only systemic patterns warrant
   recommendations. Every fix in the "Recommended Fixes" section must include a
   "Rejected as patch" line that names the trade-specific kludge you are NOT proposing.
   This rule applies ONLY to fixes — NOISE and INFORMATIONAL findings are observations
   only and do not require a "Rejected as patch" line.
4. **Never auto-commit the report.** Save to `docs/post-mortems/YYYY-MM-DD.md` (gitignored)
   and surface the path back to the user. They decide what to do with it.

## Workflow

### 1. Resolve and validate the date

- Parse the user's input. Default to `today_utc - 1` if unspecified or the user said "yesterday".
- Compute `today_utc = datetime.now(UTC).date()`.
- If target `>= today_utc`: refuse with explanation. Do NOT invoke any tooling.
- Confirm the resolved date back to the user before proceeding.
- Once confirmed, ensure output directories exist:
  `mkdir -p data/post_mortems/<D> docs/post-mortems`
  (the helper creates `data/post_mortems/<D>/` itself, but `daily_trade_report.py --json`
  in step 2 needs the directory to exist beforehand.)

### 2. Run data pulls (sequentially)

The two scripts must run in this order. Parallel execution races on `data/vps_trades.jsonl`
because SCP overwrites are not atomic on the destination side — a concurrent reader sees
a truncated JSONL.

```bash
uv run python scripts/daily_trade_report.py \
    --date <D> --pull-vps --refresh --mode all \
    --json data/post_mortems/<D>/daily_report.json

uv run python scripts/pull_post_mortem_data.py --date <D>
```

### 3. Read all artifacts

- `data/post_mortems/<D>/daily_report.json` — daily_trade_report output
- `data/post_mortems/<D>/bundle.json` — manifest with warnings
- `data/post_mortems/<D>/fills.json` — Kalshi fills
- `data/post_mortems/<D>/reconciliation.json` — local-vs-broker delta
- `data/post_mortems/<D>/operational_status.json` — VPS state for the date
- `data/post_mortems/<D>/signals/{eval_audit,candidate_audit,scan_funnel}.jsonl`

Surface every `bundle.json` warning to the user before continuing. If any pull failed,
ask the user whether to proceed with degraded data.

#### Authoritative sources & data gotchas (read before analyzing)

- **Per-order recorded P&L is the source of truth.** Use the per-order `pnl`/`won`/
  `market_result` from `data/shadow_settlements.full.json` (the `settled` block of
  `daily_report.json` mirrors it and carries both `wins[]` and `losses[]`). This is
  *recorded* settlement P&L.
- **`by_strategy` vs `settled`.** `by_strategy` reports recorded P&L including
  reconciler-synthetic rows and should reconcile exactly with `settled`. If they disagree
  (corpora from before 2026-06), trust `settled`.
- **Date axis = UTC fill timestamp.** Partition trades by `timestamp[:10]` (the fill date);
  `settled[].trade_date` is that fill date. A trade filled on day D for `forecast_date` D+1
  belongs to D. Do not assume the report grouped by settlement date.
- **`fills.json` field names** are: `count_fp` (size), `yes_price_dollars` /
  `no_price_dollars`, `fee_cost`, `created_time`, `outcome_side`, `action`. There is no
  `count` or `price` key — reading those returns None.
- **Null-safety on reconciler-synthetic rows.** Rows backfilled by FillsReconciler carry
  `model_probability=None` and `exec_price=None` (and, in corpora built before the
  ticker-backfill fix, empty `city`/`market_type`/`forecast_date`). Guard every numeric
  format with a None check; recover city/var/forecast_date from the ticker via
  `python.utils.ticker_dates.parse_ticker_city_var` / `parse_ticker_date` if blank.
- **Strategy split.** Threshold tickers (`...-T<NN>`) are favorite_bias; between tickers
  (`...-B<NN.5>`) are weather_edge. Favorite_bias is model-free, so it trades cities outside
  the 7-city weather model (BOS/PHIL/DEN/CHI) — that is expected, not an anomaly.

### 4. For each settled live trade, gather context

- Trade record (entry price, contracts, model_p, raw_p, ensemble_std, market_p, edge, side).
- Match to `eval_audit` row by `(ticker, ts close to fill timestamp)` for filter outcome,
  calibrator artifact, plateau check.
- Match to `candidate_audit` row by `(ticker, max ts < fill timestamp)` for
  `per_model_probs` — a dict of `{model: bracket_fraction}` showing per-model bracket-
  fraction disagreement at scan time (e.g., `{"gfs_seamless": 0.2, "ecmwf_ifs025": 0.5}`).
  This is a related signal to SPLIT MODEL but is NOT the formal rule.
- **For SPLIT MODEL retrospective detection** (the formal rule is
  `abs(gfs_mean - ecmwf_mean) >= 3.0`), ensemble **means** are required.
  candidate_audit does NOT carry mean fields — its schema has only bracket fractions.
  The only retrospective source for means is
  `data/snapshots/<forecast_date>/<latest_run>/weather_forecasts.json` under
  `gfs.ens_mean` and `ecmwf.ens_mean` per (city, var).
  If the snapshots row is missing or pruned for a trade, note "SPLIT MODEL retrospective
  unavailable" — do NOT assume "no SPLIT MODEL detected".
- Settlement: actual high/low, market resolution, P&L.
- Determine the model-vs-outcome story: was the model right; if edge failed to materialize,
  why (slippage, fees, plateau filter, depth, calibrator artifact at trade time).

### 5. For shadow trades

- All losses, full per-trade detail (same as Live walkthrough), ranked by P&L ascending,
  ties broken by fill timestamp ascending. If 0 losses: section reads "No shadow losses on this date."
- Top 5 wins by absolute P&L, one-line each. If 0 wins: "No shadow wins."

### 6. Run pattern checks

Apply the Pattern Thresholds. **Min-N gates are mandatory** — do not classify a metric
SYSTEMIC unless its min-N is met. Below min-N, the pattern is reported as raw observation only.

| Pattern | Min-N | NOISE | SYSTEMIC CANDIDATE |
|---|---|---|---|
| Loss clustering by city/side/var | Day total trades ≥ 6 OR ≥2 prior post-mortems exist | <3 cluster losses on this date AND not seen in priors | (Day trades ≥6 AND ≥3 cluster losses) OR (≥2 cluster losses across last 3 post-mortems) |
| SPLIT MODEL losses (retrospective via candidate_audit) | candidate_audit row joinable | (none) | Any SPLIT MODEL trade that lost |
| Boundary-grazing losses (settlement within 1°F of strike) | Day trades ≥ 4 | 1–2 with no historical pattern | (Day trades ≥4 AND ≥3) OR cumulative ≥5 across last 3 post-mortems |
| Calibration ratio (model p(loss) vs actual) | Live trades on day ≥ 10 | <1.5x | >2.0x |
| Plateau-window breach | (none — single occurrence sufficient) | (none) | Any breach |
| Reconciliation delta (after tolerance + shadow exclusion) | (none) | (none) | Any non-zero delta |
| Empty signal logs | Operational status confirms "nominal" | If status non-nominal: INFORMATIONAL | If status nominal AND ≥2 hours of expected scanning showed zero log writes |
| Wins clustering by city/side/var (model_p_at_entry < 0.55) | Day trades ≥ 6 | <3 cluster wins | (Day trades ≥6 AND ≥3 cluster wins where model_p_at_entry < 0.55) OR (≥2 cluster wins across last 3 post-mortems) |

### 7. Cold-start rule

Count distinct dates with at least one `.md` file in `docs/post-mortems/`, deduped by
leading `YYYY-MM-DD` (versioned reruns of the same date count as one).

**The report being written for `<D>` does NOT count itself.** Only OTHER distinct
prior dates with existing `.md` files count toward N. Example: if `docs/post-mortems/`
contains only `2026-05-04.md` and you are now writing a post-mortem for 2026-05-04,
then N=0 (no other prior dates exist).

If `N < 5`:
- Executive Summary `Framework maturity` line: `<COLD-START — N reports on file>`.
- Findings can ONLY be classified SYSTEMIC by an "always-systemic" pattern: plateau
  breach, reconciliation delta after tolerance, retrospective SPLIT MODEL loss.
- All other findings are NOISE or INFORMATIONAL with the note
  "Cross-date corroboration unavailable (cold-start)".
- A RAW OBSERVATIONS sub-block always lists every cluster regardless of label, so the
  operator can read raw counts.

If `N ≥ 5`: grep `docs/post-mortems/*.md` for prior occurrences before promoting to SYSTEMIC.

### 8. Draft fixes

Only systemic fixes. For each:

- **Issue:** one sentence
- **Evidence:** data points across this date + prior post-mortems
- **Proposed fix:** approach, not patch
- **Rejected as patch:** what NOT to do — single-trade tweaks, hard-coded blocks

If no systemic findings: "No systemic fixes warranted."

### 9. Write the report

Path: `docs/post-mortems/<date>.md`. Versioning: glob `<date>*.md`, find max `vN`,
write `<date>-v{N+1}.md` if collision. Never overwrite existing files in any mode.

If interactive: ask "Versioned write to `<date>-v{N+1}.md` is the default. Confirm,
abort, or specify a different suffix?"

If non-interactive: default to versioned overwrite without prompting; note the
versioning in the user-facing summary.

### 10. Final summary

3–4 sentences: P&L, headline finding, whether systemic issues were identified, path
to the report, any pull warnings.

## Report schema

Required sections (in order):

```markdown
# Post-Mortem: YYYY-MM-DD

**Date axis:** UTC fill timestamp
**Framework maturity:** N prior post-mortems on file [<COLD-START> if N < 5]
**Bot operational status:** [from operational_status.json `summary` field: `nominal` | `kill-switch` | `unit-down` | `deploy` | `mixed` | `unknown`]
**Settlement completeness:** N of M trades settled (P pending — rerun tomorrow morning UTC for full report)

## Executive Summary
- Live: N trades, $X.XX P&L, W-L, win rate
- Shadow: N trades, $X.XX P&L, W-L, win rate
- Calibration ratio (live, N≥10 only): X.XXx [or "N/A — sample too small"]
- Headline finding: <one sentence>

## Live Trades — Walkthrough
[Per-trade walkthrough as defined in step 4]

## Pending Live Trades
[List unsettled live trades; no walkthrough]

## Shadow Trades — Aggregate
[Per step 5]

## Signal Funnel
[From scan_funnel.jsonl]

## Patterns & Anomalies
[Each finding with NOISE / INFORMATIONAL / SYSTEMIC CANDIDATE label, plus a
RAW OBSERVATIONS sub-block during cold-start]

## Reconciliation: Trade Log vs Kalshi Fills
[N local non-shadow trades, M Kalshi fills, K matched. If clean: one line.]

## Systemic Issues Identified
[Or "No systemic issues identified — observations consistent with expected variance,
sample floor, or operator-explained outage."]

## Recommended Fixes
[Per step 8]

## Appendix — Data Sources
- `data/post_mortems/<date>/bundle.json` — manifest
- `data/post_mortems/<date>/fills.json` — Kalshi fills
- `data/post_mortems/<date>/reconciliation.json`
- `data/post_mortems/<date>/operational_status.json`
- `data/post_mortems/<date>/signals/*.jsonl`
- `data/post_mortems/<date>/daily_report.json`
```

## Forbidden

- Market-sentiment narrative (CD3): never write paragraphs about what "the market
  thought" or "buyers stepped back" or "selling pressure". Every claim must cite an
  entry-time data point we have.
- Fixes for individual trades. Every fix must be systemic (recurring or always-systemic
  pattern) AND must include a "Rejected as patch" line.
- Auto-commit. Save the file, surface the path, stop.
- Inflating the framework_maturity counter by counting versioned reruns of the same
  date as separate priors.
- Promoting a single occurrence of a non-always-systemic pattern to SYSTEMIC during
  cold-start (`N < 5` prior reports).
