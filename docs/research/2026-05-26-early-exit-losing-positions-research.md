---
title: Early-exit rule for losing binary positions — research & decision
date: 2026-05-26
type: RESEARCH + DECISION (data-driven; conclusion is "do not build")
owner: Claude Code (operator-directed)
status: COMPLETE — 2026-05-26. **VERDICT: the data does NOT support an early-exit / stop-loss rule.** Every price-drawdown threshold tested is net-negative vs hold-to-settlement on the live corpus. The one theoretically-supported variant (model-triggered exit) is not measurable with current logging and is not recommended for build. No code written.
data_window: live trades 2026-04-21 → 2026-05-22 (settled); fine-grained price paths 2026-05-07 → 2026-05-26 (eval_audit)
cohort_n: 166 settled live positions with reconstructed intraday price paths (101 win / 65 lose)
---

# Early-exit rule for losing binary positions — research & decision

## Question

The Oracle holds every Kalshi binary position to settlement ($0 or $1). Is there
a price/time threshold at which selling early — taking a partial loss — beats
holding a likely-zero settlement? Formally, an optimal-stopping problem: when does
`EV(sell now at current bid) > EV(hold to settlement)`?

**Answer: No.** On 166 real live positions with intraday price paths, *every*
drawdown-triggered stop-loss (−5¢ through −40¢) loses money versus holding,
by **−5.9¢ to −12.8¢ per contract** on a cohort whose hold-to-settlement P&L was only **+1.4¢ per contract**.
The mechanism is structural, not a sampling fluke, and it matches the published
theory. Details and the (small) caveats below.

---

## TL;DR for the operator

- **Do not add a price-based stop-loss.** It converts the cohort's +1.4¢/contract hold P&L
  into a −4.5¢ to −11.4¢/contract loss depending on threshold.
- **Why:** we almost always buy **NO at ~63–73¢** on weather brackets. The intraday
  price wobbles a lot (forecast noise), so **57% of eventual *winners* dip past
  −20¢** before settling in our favor. A stop cuts those recoverers. Meanwhile the
  *losers* gap straight to zero (median terminal exit value = **0.000**), so by the
  time a stop could fire it saves almost nothing.
- **At the moment a −20¢ stop fires, winners and losers are indistinguishable**
  (exit value 0.450 vs 0.410; both ~19–20h to settlement). The trigger has
  near-zero signal.
- The **only** theoretically-correct rule is *model-triggered* ("sell iff
  net_bid > current model P(win)"), not price-triggered. It is **not testable**
  today (the bot doesn't re-evaluate held positions — only 5/166 positions have any
  post-entry model probability logged) and is **not recommended** for build given
  the weak prior and high infra cost.

---

## 1. Theory frame (why a price stop is suspect a priori)

External literature review (citations at end) gives a sharp prior **before**
looking at our data:

1. **The EV rule for a {0,1} contract is one line.** Hold-EV per contract = your
   model probability of winning `q`. Sell-EV = `net_bid` (bid minus closing fee).
   **Sell iff `net_bid > q`.** A price *drop* only changes this if it changes `q` —
   i.e. the exit decision is fundamentally about your *model*, not the price.
2. **Optional Stopping Theorem.** If price is a martingale (`p_t ≈ q`), every hold
   is exactly zero-EV vs selling at mid — and crossing the spread + paying a fee
   makes early exit *strictly* EV-negative. Early exit adds EV only when you have
   *no or negative* edge at that moment.
3. **Kaminski & Lo (2007), "When do stop-loss rules stop losses?"** The stopping
   premium is **always negative under a random walk**; stops add value *only* under
   positive return autocorrelation (momentum). So the empirical question reduces to:
   **do adverse moves in our books carry momentum (informed) or mean-revert (noise)?**
4. **0DTE options practice** (tastytrade "Managing Losers or Managing Earlier"):
   adding a 2× stop on losers did **not** beat managing winners; stops realize
   losses that would have mean-reverted, and fill into the worst liquidity.
5. **Variance/ruin** is the *only* clean pro-stop argument — truncating a binary's
   guaranteed −100% tail can raise geometric growth even at flat/negative EV. But we
   already control this with fractional Kelly (0.075) and exposure caps.

The data below tests #3 directly on our corpus.

---

## 2. Data & method

**Spine (outcomes).** `data/shadow_settlements.full.json` — 364 settled positions,
entry 2026-04-21 → 2026-05-22 (290 live, 74 shadow; **346 NO / 18 YES**; net P&L
≈ **breakeven after fees** → gross ≈ breakeven-positive, fees are the drag). Each row
carries side, `order_price` (our-side cost basis), `won`, `contracts`, entry book
state, and `model_probability`.

**Price paths.** `state/eval_audit.jsonl` logs every scanned market every scan tick
(`ts, yes_bid, yes_ask, yes_ask_size, hours_to_settlement, calibrated_prob`). VPS
retains one rotated file per day **2026-05-07 → 2026-05-26**. Extraction:
`grep -Ff` on the 156 held tickers (exact `"ticker":"…"` match) across all daily
files → 214,102 rows → compacted to CSV and pulled. **91.8%** of rows carry valid
bid/ask; **median 1,338 price observations per position** (fine resolution, not
literal tick data but far more than adequate for max-adverse-excursion).

**Analyzable cohort.** The **166** settled positions whose entry (≥2026-05-07)
falls inside the price-path window: **101 won / 65 lost (60.8% base win rate);
153 NO / 13 YES; all live.** This is the freshest, most decision-relevant slice.

**Exit-value model.** For a **NO** holder, the realizable exit is selling NO at
`no_bid = 1 − yes_ask` (a taker hit consuming `yes_ask` liquidity); for **YES**,
`yes_bid`. Cost basis = `order_price`. Unrealized P&L/contract = `exit_value −
cost_basis`. **Max-Adverse-Excursion (MAE)** = the most-negative value over the
held path. Closing fee modeled with the Kalshi taker formula
`ceil(0.07 · C · P · (1−P) · 100)/100`. Settlement itself is fee-free, so the
exit branch carries an *extra* fee the hold branch does not.

> **Reproduction.** Grep pattern = exact `"ticker":"<TICKER>"`; exit-value formulas
> above; scripts lived at `/tmp/exit_research/{analyze,diag}.py` (ephemeral). Re-pull
> with the `scp` of `shadow_settlements.full.json` + the VPS `grep -Ff` recipe.

---

## 3. Empirical findings

### 3a. Win rate vs drawdown depth — adverse moves ARE informative, but bimodally

Cumulative: of positions that *ever* dipped at least X below cost basis, what
fraction still settled in our favor?

| MAE reached ≥ | n   | won | win % | 95% CI (Wilson) | avg exit value at trough |
|--------------:|----:|----:|------:|:----------------|-------------------------:|
| −5¢           | 147 | 87  | 59.2% | [51%, 67%]      | 0.224 |
| −10¢          | 139 | 84  | 60.4% | [52%, 68%]      | 0.224 |
| −15¢          | 133 | 78  | 58.6% | [50%, 67%]      | 0.207 |
| −20¢          | 112 | 58  | 51.8% | [43%, 61%]      | 0.153 |
| −25¢          | 100 | 47  | 47.0% | [38%, 57%]      | 0.116 |
| −30¢          |  89 | 36  | 40.4% | [31%, 51%]      | 0.085 |
| −40¢          |  73 | 20  | 27.4% | [18%, 39%]      | 0.039 |
| −50¢          |  61 | 10  | 16.4% | [9%, 28%]       | 0.016 |

Win rate **does** fall with drawdown depth (Glosten–Milgrom: price carries
information; the weather forecast firming into the bracket is real signal). But the
**non-cumulative** view reveals the trap:

| MAE bucket        | n  | won | win % | 95% CI    |
|:------------------|---:|----:|------:|:----------|
| ≤ −40¢            | 73 | 20  | 27.4% | [18%, 39%] |
| (−40¢, −30¢]      | 16 | 16  | **100%**  | [81%, 100%] |
| (−30¢, −20¢]      | 23 | 22  | **95.7%** | [79%, 99%] |
| (−20¢, −10¢]      | 27 | 26  | **96.3%** | [82%, 99%] |
| (−10¢, −3¢]       | 11 | 3   | 27.3% | [10%, 57%] |
| (−3¢, +)          | 16 | 14  | 87.5% | [64%, 97%] |

**The distribution is bimodal.** Positions that dip a *moderate* −10¢ to −30¢ and
no further recover to win **~96%** of the time (pure mean-reverting noise — exactly
the regime where Kaminski–Lo says a stop is *most* destructive). Only the *deep*
≥−40¢ cohort is mostly losers (27% win). A drawdown stop set anywhere ≤ −30¢
slaughters the 96%-win recoverers; set deeper, it barely catches losers (see 3c).

### 3b. At the trigger point, winners and losers are indistinguishable

First time a position crosses −20¢:

| group   | ever hit −20¢ | median hrs-to-settle at cross | median exit value at cross |
|:--------|:--------------|------------------------------:|---------------------------:|
| LOSERS  | 54/65 (83%)   | 18.8h                         | 0.410 |
| WINNERS | 58/101 (57%)  | 20.2h                         | 0.450 |

**A −20¢ drawdown happens to a majority of eventual winners**, at nearly the same
exit value and time-to-settlement as for losers. The trigger has almost no
discriminating power at the moment it fires.

And the losers give no early warning: their **terminal exit value is median 0.000
(p90 = 0.010)** — they gap toward zero — while **45 of 65 losers (69%) traded
*above* cost basis at some point intraday**. Most losers looked like winners before
the forecast resolved against us.

### 3c. Exit-rule P&L delta vs hold-to-settlement (the headline)

Rule simulated: when drawdown first ≤ −T, sell at that tick's exit value (taker,
Kalshi fee). "Saved" = P&L rescued on losers; "Forgone" = P&L surrendered on
winners that dipped past T and would have recovered. Per-contract, gross of the
(sunk, identical-in-both-branches) entry fee.

| threshold | # triggered | win-trig | lose-trig | saved (¢/contract) | forgone (¢/contract) | **NET Δ (¢/contract)** | no-liquidity trig |
|----------:|------------:|---------:|----------:|----------:|------------:|--------------:|------------------:|
| −5¢ | 147 | 87 | 60 | 11.6 | 20.8 | **−9.2** | 3 |
| −10¢ | 139 | 84 | 55 | 10.7 | 22.3 | **−11.5** | 2 |
| −15¢ | 133 | 78 | 55 | 9.7 | 22.5 | **−12.8** | 3 |
| −20¢ | 112 | 58 | 54 | 7.9 | 15.1 | **−7.2** | 3 |
| −25¢ | 100 | 47 | 53 | 7.2 | 14.4 | **−7.2** | 2 |
| −30¢ | 89 | 36 | 53 | 6.4 | 13.1 | **−6.7** | 1 |
| −40¢ | 73 | 20 | 53 | 3.7 | 9.7 | **−5.9** | 5 |

**Cohort hold-to-settlement gross P&L: +1.4¢ per contract** (166 positions, 5,871 contracts).

Every threshold is **net-negative**. The least-bad (−40¢) still converts +1.4¢ into
**−4.5¢** per contract; a "tight" −15¢ stop yields **−11.4¢**. The forgone-on-winners column
dominates saved-on-losers at every level. The `no-liquidity trig` column counts
triggers where the book had **zero** exit depth — those are modeled *optimistically*
at the quoted value, so the real result would be **worse**, not better.

**Perfect-hindsight oracle bound.** Selling *every* loser at its single best
lifetime price (impossible — requires hindsight — and fee-free) recovers at most
**18.4¢ per contract**. Any *realizable* rule must trigger on winners too, forgoing ~15.1¢ at
−20¢. The achievable ceiling sits far below the oracle, and we measured it at
−7.2¢. There is no room.

### 3d. The EV-correct (model-triggered) rule is not measurable today

The literature's correct rule is "sell iff `net_bid > model P(win)`" — re-evaluated
each tick. **Only 5 of 166 positions have *any* model-probability observation on
their post-entry path:** the bot logs `calibrated_prob` only when a market passes
the entry pre-filters, which a held position generally does not on later scans. So
this rule cannot be backtested on current data, and implementing it is not a config
flag — it requires re-running the model on open positions every tick (§5).

---

## 4. Decision

**No early-exit rule is adopted.** The data refutes every price/time drawdown stop:

- Net-negative at all thresholds (−5.9¢ to −12.8¢ per contract on a +1.4¢ cohort).
- Mechanism is structural (NO-heavy book bought at ~69¢ + mean-reverting weather
  forecast noise + thin post-fee edge + losers that gap to zero), so it will recur
  across regimes, not just this window.
- Theory-consistent: moderate adverse moves mean-revert (Kaminski–Lo: stops worst
  here); deep moves are informative but arrive as gaps the stop can't beat.
- The variance/ruin case fails too: by −40¢ the position is already worth ~4¢, so
  there is no tail left to truncate — the loss is realized before any stop can act.
  We also already hold tail risk down via 0.075 Kelly + exposure caps.

This mirrors the **SPX sunset** lesson — judge by realized tradeable P&L, not by an
appealing-sounding mechanism. Here the appealing mechanism ("cut losers") is
empirically and theoretically a P&L destroyer for *this* book.

---

## 5. Implementation sketch — IF ever revisited (do not build now)

For the record, the *only* variant with theoretical support, and what it would cost:

- **Trigger (model-triggered, not price-triggered):** on each scan, for every open
  position, recompute the calibrated model P(win) from the freshest ensemble/NWS
  data and the live book; flag exit iff `net_bid_now > model_P_win_now` (i.e. *our
  own model* now agrees the position is a loser relative to the exit price), gated
  on `yes_ask_size ≥ min_book_depth` so the exit is actually fillable.
- **Module placement:** a new open-position evaluation pass in
  `python/orchestrator.py` (today's loop only evaluates markets for *new entry* via
  `Strategy.generate_signals()`); the exit-edge math would reuse
  `weather_evaluation.evaluate_market_pure` + the Stage-2 orderbook/VWAP check in
  `weather_edge.py`; execution via a SELL path in `python/execution/engine.py`
  (currently buy-to-open only for weather). Risk accounting would route through
  `RiskManager.close_position` like the existing manual-close/cash-out reconciler.
- **Prerequisite (blocking):** **open-position model logging.** Add `calibrated_prob`
  / `emos_prob_raw` emission for *held* tickers to `eval_audit.jsonl` (today they're
  null on post-entry scans). Without ≥ ~8 weeks of this, the rule cannot even be
  validated — the 5/166 coverage above makes that explicit.
- **Recommendation:** **do not build.** Prior is weak (model tracks the market
  closely → divergences rare), infra cost is high (new open-position eval + sell
  path + logging), and exit liquidity in thin weather books is the documented 0DTE
  "$2 stop fills at $0.10" failure mode. If revisited, the *first* deliverable is
  the logging change + a measurement re-run — not an exit rule.

---

## 6. Limitations / honest n-flagging

- **Cohort n = 166** (65 losers). Headline cumulative thresholds rest on n = 61–147
  (solid); the *non-cumulative* MAE buckets in 3a are thin (n = 11–27, wide CIs) —
  treat the 96%-recovery and 100% cells as directional, not precise. The **NET-Δ
  signs in 3c are robust**: every threshold negative, forgone > saved by a wide
  margin, and the optimistic no-liquidity modeling only flatters the (still
  negative) result.
- **13 YES positions** — too few to analyze separately; cohort is NO-dominated,
  which matches the live book (95% NO).
- **Window:** live era 2026-05-07 → 2026-05-22 under isotonic_v2 + reduced-knob
  config. A materially different entry-price regime (e.g. buying favorites near 0.5)
  could shift magnitudes, but the mechanism is structural. The model-triggered rule
  (3d/5) is **untested**, not refuted.
- Pre-05-07 positions (≈198 of the 364, mostly shadow + early live) lack
  fine-grained paths and are excluded from the path analysis; including them via
  coarse daily snapshots would only add noise, not change the verdict.

---

## Citations (external research)

- Optional Stopping Theorem — https://en.wikipedia.org/wiki/Optional_stopping_theorem
- Kaminski & Lo, "When do stop-loss rules stop losses?" — https://www.sciencedirect.com/science/article/abs/pii/S138641811300030X
- tastytrade, "Managing Losers or Managing Earlier" — https://www.tastytrade.com/tt/shows/market-measures/episodes/managing-losers-or-managing-earlier-07-05-2016
- Glosten & Milgrom (1985), informed-flow / bid-ask — https://business.columbia.edu/faculty/research/bid-ask-and-transaction-prices-specialist-market-heterogeneously-informed-traders
- Kelly criterion / MacLean–Thorp–Ziemba "Good and bad properties of the Kelly criterion" — https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
- 0DTE stop-loss execution risk — https://0dte.com/what-is-spx-0dte-strategy
