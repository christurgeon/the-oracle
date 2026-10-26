# Weather-edge search dossier — what's dead and why (2026-05-30)

**Purpose.** A single consolidated record of the weather-strategy search so the
same ground is not re-tread. Outcome of this round: **no new deployable edge
found.** Kalshi daily-horizon weather books are near-efficient, and on the
underlying we are structurally *disadvantaged* (the market has better data than
our free sources). This is the honest verdict, not a pause.

This dossier consolidates the 2026-05-29/30 search; per-item detail lives in the
linked memories and research docs.

## Goal

"Discover a new weather-markets strategy we can prove edge (ideally
profitability) on and deploy soon." Pursued rigorously; came up null. Recording
*why* is the deliverable.

## The kill list

| Mechanism | Verdict | Evidence / why | Pointer |
|---|---|---|---|
| **Forecasting** (out-predict the bracket) | **Dead** | Phase 0 T2: market Brier 0.217 < our model 0.264 (significant). σ ≥ 1°F bracket width. | `edge_existence_audit.py` |
| **Between brackets** | **Dead** | Efficiently priced — realized ≈ price, CI [−0.017, +0.009]. | `market_reliability_audit.py` |
| **Favorite-longshot window recalibration** | **Dead (was an artifact)** | "+21.5% OOS" was a **look-ahead bug** (lead anchored on last-candle-ts, ~13h before true `expiration_ts`). Corrected at a true 24h lead: CI spans zero every window; −34% at 12h. The real reliability gap (~+8pp) is too small to beat fees+spread. | `2026-05-29-favorite-longshot-window-recalibration.md` (unpublished) |
| **Longshot-fade** (buy NO on cheap directional) | **Dead** | Fee-eaten — ROI ≈ 0, CI spans zero. | `favorite_longshot_robustness.py` |
| **Intraday momentum / order-book imbalance** | **Dead** | No orthogonal signal beyond price level; underpowered (n=1–7/cell). | `intraday_path_signal.py`, `microstructure_signal_probe.py` |
| **NBM-gap on greater/less** | **Dead** | Well-powered null (n=776, `expiration_ts`-anchored). Market Brier 0.071 vs NBM 0.199 — gap-selection = adverse selection (market ~3× sharper than NBM). | `2026-05-30-nbm-gap-feasibility-spike.md` |
| **Monthly rain accumulation-lock** | **Dead** | Books efficient (0/20 locked brackets lagged) **and** open-meteo precip can't determine settlement (3/20 false locks vs NWS gauge; market priced them right). | `rain_lock_probe.py` |

**Through-line.** On the *underlying*, the market consistently has better data
than we do (open-meteo reanalysis / GFS / NBM are all dominated by the market
price). On the *behavioral* side, the only real bias (favorite-longshot) is too
small to clear fees + the ~3¢ median spread at a tradeable lead. There is no
cheap prediction-or-lag edge left in this family.

## Methodology lessons (durable)

1. **Anchor candle backtests on `expiration_ts`, never on the last traded
   candle.** Kalshi weather books go quiet a median ~13h before settlement, so
   the two differ systematically and the gap masquerades as edge. This single
   bug produced a fully "significant," OOS-passing fake edge.
2. **"Edge grows with lead" is a red flag for a time-to-expiration confound,**
   not robustness. A real static mispricing should be ~lead-invariant.
3. **Adversarial subagent verification before any live-money change is
   mandatory** and earned its cost — the same day-clustered CIs looked
   significant until the anchor was fixed.
4. **Stop fishing for a forecast/lag edge in an efficient book.** Confirmed
   repeatedly; the unlock is structural (different market microstructure) or
   more forward data, not more cleverness on the same corpus.

## Reusable tooling built this round (kept)

- `scripts/backfill_candles_incremental.py` — resumable Kalshi candlestick
  backfill for any series → `data/research/candles_corpus.jsonl` (75-day, 6,768
  settled markets, 17 series; local artifact, regenerable, not committed for size).
- `scripts/favorite_longshot_verify.py` / `favorite_longshot_robustness.py` —
  reliability curve + fee-aware day-clustered strategy backtest + split-half
  walk-forward (now `expiration_ts`-anchored).
- `scripts/rain_lock_probe.py` — monthly-rain accumulation-lock tester
  (Kalshi candles + open-meteo archive precip).
- `scripts/backfill_eval_audit_results.py`, `market_reliability_audit.py`,
  `intraday_path_signal.py`, `microstructure_signal_probe.py` — labeling +
  microstructure probes.

## What is NOT dead (forward-data-gated; not pursued now)

Per the consolidated state, the only paths still open are **gated on forward
data or a bigger build**, deliberately not pursued in this round:

1. **Intraday / sub-12h latency** (e.g. `intraday_emos_high` shadow) — the one
   horizon not subject to the daily-forecasting-is-dead constraint; accruing
   shadow calibration, re-eval ~2026-06-26.
2. **Maker / liquidity-provision economics** (Whelan et al. 2026) — structurally
   different (4× maker-fee advantage + spread capture), not fee-killed, but a
   larger build and hard to backtest without order-book data.

> **Correction 2026-10-01:** weather series charge makers no fee (`fee_type: quadratic`; see the root `CLAUDE.md`, "Kalshi fee integration"). The maker advantage over taker is the full taker fee, not 4×.

## Bottom line

No new strategy is being deployed. `favorite_bias` continues as its existing
bounded pilot (its own n≥40 re-validation ~2026-06-23 stands — but note its
window is not statistically proven either). Revisit the search when forward data
accrues (the new precip markets, intraday shadow) or when committing to the
maker-economics build.
