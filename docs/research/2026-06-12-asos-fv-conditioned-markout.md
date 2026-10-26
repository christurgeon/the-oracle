# ASOS FV-conditioned passive-fill markout — invert-vs-shrink feasibility test

**Date:** 2026-06-12
**Author:** research (automated)
**Script:** `scripts/research/asos_fv_conditioned_markout.py` → `scripts/research/asos_fv_conditioned_markout.json`
**Question:** Does conditioning a passive resting limit fill on the ASOS intraday-nowcast
posterior (`posterior_p_raw`) **INVERT** the adverse-selection markout (make it non-negative),
or merely **SHRINK** it? Invert → lane is real, build a controller. Stays negative → lane dies,
zero capital spent.

---

## 1. Headline verdict: **INCONCLUSIVE, leaning NEGATIVE**

Conditioning on ASOS FV does **not robustly invert** the markout. The two pieces of apparent
good news both fail a basic robustness check:

1. The full-cohort numbers look mildly positive (always-YES baseline **+1.52¢**, guard=pass
   **+3.36¢**, pass & FV-favors-side≥5¢ **+4.36¢** maker-free), **but** removing the single
   day settled on **preliminary** NWS data (2026-06-11) collapses every one of these:
   always-YES baseline → **−2.86¢**, guard=pass → **+0.42¢**, pass&≥5¢ → **−0.50¢**, NO side →
   **−4.12¢**, ≥10¢ bucket → **−5.28¢**. The headline positivity is a **one-day artifact**, not
   a stable conditioning edge.
2. Even with 06-11 included, **every** day-clustered 95% CI straddles zero by a wide margin
   (≈±8–18¢). With only **6 day clusters** there is no statistical signal in either direction.

Critically, the "informed" FV-conditioning **does not beat the uninformed always-YES baseline**
at any threshold — it *under*performs it (baseline +1.52¢ vs informed +0.27¢/+0.03¢/−0.17¢ at
2/5/10¢). The FV staleness signal does not select profitable fills; it selects the **same or
worse** markout than ignoring FV entirely. That is the opposite of "invert."

The one real, robust split is the **boundary guard** (which is already live-suppressed): pass
+3.36¢ vs boundary −5.35¢. That confirms the existing boundary-distance guard is doing useful
work, but it is *not* an FV-conditioning result — it's a geometry filter the strategy already runs.

**Conclusion:** The FV-conditioned markout neither inverts nor cleanly shrinks the
adverse-selection cost in a way that survives the prelim-day robustness check. On the 5 clean
final-settled days the conditioned markout is **negative across the board**. The lane does not
clear the bar to build a passive-execution controller.

---

## 2. Staleness-bin decomposition (strategy-chosen side, optimistic fill)

`|posterior_p_raw − market_price|`, maker-free + maker-billed, day-clustered bootstrap 95% CI
(6 clusters → wide, reported honestly).

| bin    | n      | markout maker-free | markout maker-billed | CI (maker-free, day-clustered) |
|--------|--------|--------------------|----------------------|--------------------------------|
| 0–2¢   |  9,214 | +0.16¢             | −0.84¢               | [−2.39, +2.24]¢                |
| 2–5¢   |  8,978 | +1.52¢             | +0.52¢               | [−0.98, +3.34]¢                |
| 5–10¢  |  8,107 | +1.02¢             | +0.02¢               | [−3.01, +4.30]¢                |
| ≥10¢   | 39,836 | −0.17¢             | −1.17¢               | [−8.57, +12.69]¢               |

Observation: markout does **not** rise monotonically with staleness. The big ≥10¢ bucket
(60% of rows) is flat-to-negative, not the most positive — the opposite of a "stale book, FV
right" edge. Maker fees alone (~1¢/contract) push the inner buckets toward zero.

> **Correction 2026-10-01:** weather series charge makers no fee (`fee_type: quadratic`; see the root `CLAUDE.md`, "Kalshi fee integration"). The ~1¢/contract maker-fee drag above does not apply; markouts are about 1¢ better than stated.

---

## 3. Invert-vs-shrink comparison

**Uninformed baselines (ignore FV), full cohort:**

| rule                         | n      | maker-free | maker-billed | CI (free)        |
|------------------------------|--------|------------|--------------|------------------|
| Always-YES rest              | 66,135 | **+1.52¢** | +0.52¢       | [−0.75, +5.62]¢  |
| Book-mid-sign (rest cheap)   | 66,135 | −0.81¢     | −1.81¢       | [−7.73, +8.30]¢  |

**Informed (rest only when FV favors chosen side by ≥ thr), full cohort:**

| thr | n      | maker-free | maker-billed | CI (free)        |
|-----|--------|------------|--------------|------------------|
| 2¢  | 56,883 | +0.27¢     | −0.73¢       | [−6.20, +9.52]¢  |
| 5¢  | 47,942 | +0.03¢     | −0.97¢       | [−7.35, +11.04]¢ |
| 10¢ | 39,836 | −0.17¢     | −1.17¢       | [−8.55, +13.45]¢ |

**The invert-vs-shrink answer:** Neither. The informed subset is **worse** than the uninformed
always-YES baseline at every threshold (+1.52¢ → +0.27/+0.03/−0.17¢). FV conditioning does not
shrink a negative markout toward zero (the relevant baseline here isn't negative) nor does it
invert anything — it *erodes* the naive baseline. The FV staleness signal carries no positive
selection value in this cohort.

---

## 4. Edge vs miscalibration — the ≥10¢ bucket and the NO side

The ≥10¢-disagreement bucket (39,836 rows, **60%** of settleable) is the suspected
miscalibration zone. Full-cohort breakdown:

| slice           | n      | maker-free | CI (free)         |
|-----------------|--------|------------|-------------------|
| ≥10¢ all        | 39,836 | −0.17¢     | [−8.54, +12.69]¢  |
| ≥10¢ **YES**    | 15,421 | +3.84¢     | [−7.01, +19.30]¢  |
| ≥10¢ **NO**     | 24,415 | **−2.71¢** | [−9.70, +10.87]¢  |

**Finding:** The ≥10¢ bucket is **split by side, exactly as the known NO-bin miscalibration
risk predicts.** When the FV disagrees with the book by ≥10¢ and the strategy is on the **NO**
side, the markout is **negative** (−2.71¢, −5.28¢ excluding the prelim day) — i.e. the book was
right and our overconfident low-p NO posterior would have lost. The positive ≥10¢ YES number
(+3.84¢) is the mirror image of the same overconfidence (FV cheap → buy YES → wins more often)
but is one-day-fragile and CI-wide.

This confirms the binding open question from the unpublished tracker: **the ≥10¢ bucket is
dominated by FV miscalibration (NO-bin overconfidence), not genuine staleness edge.** A passive
controller resting on that signal would systematically rest on the wrong side of the large-
disagreement NO contracts.

**Side split (all settleable, chosen side):**

| side | n      | maker-free | maker-billed | CI (free)       |
|------|--------|------------|--------------|-----------------|
| YES  | 27,089 | +2.16¢     | +1.16¢       | [−6.96, +13.35]¢|
| NO   | 39,046 | **−1.07¢** | −2.07¢       | [−4.53, +6.03]¢ |

The NO side — 59% of rows and the cell the strategy most often chooses — is **net negative**
(−1.07¢ full cohort, −4.12¢ excluding 06-11). This is the known low-p NO miscalibration showing
up directly in markout.

**Guard split (boundary rows are live-suppressed):**

| guard    | n      | maker-free | CI (free)        |
|----------|--------|------------|------------------|
| pass     | 42,512 | +3.36¢     | [−3.08, +10.19]¢ |
| boundary | 23,623 | **−5.35¢** | [−11.07, +4.37]¢ |

The boundary guard correctly isolates the worst markout (−5.35¢). This is the strategy's
existing geometry filter working as designed — not new FV-conditioning value.

---

## 5. Power & caveats (read before acting)

- **6 day clusters, effectively 5 clean.** 2026-06-12 is unsettled (dropped: 12,662 rows /
  all 06-12 + LAX 06-11, 15,376 total dropped). 2026-06-11 settled on **preliminary** NWS CLI
  values for the 4 traded cities (AUS prelim=79°F and NYC prelim=91°F both look like outlier
  prelims that may revise). **The entire positive headline flips negative when 06-11 is
  excluded** — the result is not robust to one day. With 5–6 clusters every CI is ≈±8–18¢ wide;
  there is no statistical significance anywhere.
- **OPTIMISTIC fill model.** This simulation assumes our resting limit order *fills at the book
  price* on the chosen side. It contains **no adverse-selection mechanism** — it does not model
  the book moving against us before/at fill, queue position, or fade. This is why the uninformed
  baseline here is +1.52¢ rather than the −5/−6¢ live adverse-selection figure from
  an unpublished 2026-06-05 maker-liquidity note. **The real passive-execution markout will be
  materially worse than every number in this report.** Reading these as upper bounds: even the
  optimistic upper bound does not robustly invert, which strengthens the negative verdict.
- **Never-traded markets.** All 81,511 rows are `intraday_asos_high` *evaluations* in
  shadow-only mode — zero orders placed. No realized fills, no realized queue dynamics.
- **Settlement join:** 66,135 / 81,511 rows settled (81%). YES-region from `strike_type` +
  floor/cap; final NWS CLI value preferred over prelim. Drops are clean (today + LAX-no-CLI).

**Bottom line for the controller decision:** Under an *optimistic* fill model the FV-conditioned
markout does not invert, does not beat the naive always-YES baseline, and goes negative on the
NO side and the ≥10¢ bucket (the miscalibration zone) — and the only positive aggregate numbers
evaporate when a single prelim-settled day is removed. A realistic fill model can only be worse.
**Do not build the passive-execution controller on this signal.** The lane dies here; zero
capital spent.
