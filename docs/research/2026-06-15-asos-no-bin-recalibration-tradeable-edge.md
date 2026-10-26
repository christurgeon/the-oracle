# ASOS NO-bin recalibration — tradeable-edge verdict (offline analysis + local settled backtest)

**Date:** 2026-06-15
**Author:** research (automated routine)
**Script:** `scripts/research/research_2026_06_15_asos_no_bin_tradeable_edge.py`
**Question:** The μ+=G & σ-inflation-by-var(G) recalibration calibrates the
miscalibrated low-p NO bin out-of-sample (per the 2026-06-13 recut), but
σ-inflation erodes the sub-bracket sharpness that *is* the intraday edge. Does
the recalibrated posterior still produce **positive realized P&L net of fees**
on same-day KXHIGH, filled **at the ask**?

---

## 1. Headline verdict: **CONFIRM KILL — do NOT wire the σ-inflated recalibrator for a live flip.**

> **Update (2026-06-15, local settled backtest — the definitive half is now DONE).**
> The settled-P&L backtest this routine could not run (it lacked intraday `M`) was
> completed locally where the 1-minute ASOS corpus exists
> (`scripts/research/research_2026_06_15_asos_settled_backtest_local.py`; results
> `data/research/2026-06-15-asos-settled-backtest-local.json`). It replays the live
> decision (2000Z snapshot; AUS/CHI/NYC — MIA has no hour-20 climatology cell, LAX
> informational) over 2026-03-26→04-27 (1-min/snapshot overlap), reusing the production
> posterior, guards, side-selection and taker fees; fills at the ask; settles vs CLI truth.
>
> **Result — σ-inflation produces NO defensible edge:** recal net **slightly positive** over 61
> trades / 29 days, but the **day-clustered 95% CI [−$0.043, +$0.228] includes zero**
> (day t-test p≈0.22); 78% of the profit is 3 longshot wins; sample is below the
> 100-trade floor. Critically, σ-inflation does **not fix the bin** — the recalibrated
> low-p NO bin (model P(YES)≤0.05) still realizes YES **62.5%** of the time (raw: 30%).
> The only positive P&L is small-sample longshot variance, not edge. This upgrades
> LEAN KILL → **CONFIRM KILL**.
>
> *Caveat:* the local 1-min corpus ends 2026-04-27, so this validates a 33-day
> shoulder-season AUS/CHI/NYC slice, not the summer regime (a summer re-run needs the
> 1-min backfill extended). It bounds — but does not change — the conclusion: the bin is
> structurally miscalibrated and no fee-surviving cohort appears. The highest-leverage
> item remains a proper recalibrator on the climo low-p output, gated on forward-shadow
> accrual — **not** a live flip of `intraday_asos_high`.

Two parts, stated honestly:

1. **The *definitive* settled-P&L backtest cannot be executed in this
   environment.** It requires the intraday observed max-so-far `M` at the
   decision moment for every (city, day). `M` needs ASOS hourly observations,
   which are **both** gitignored (`*.csv`, `data/research/asos_1min/`; the
   prototype `asos_prototype_evaluations.parquet` is uncommitted) **and**
   unreachable — the IEM host `mesonet.agron.iastate.edu` and `api.weather.gov`
   are **not in this environment's network egress allowlist** (every outbound
   request, including `WebFetch`, returns `403 / "Host not in allowlist"`). The
   routine that was meant to deliver the offline half of the gate is blocked on
   data + egress, not on analysis. This is reported, not worked around — no `M`
   was fabricated.

2. **Everything reproducible from the committed repo leans decisively toward
   KILL.** The raw-posterior low-p NO bin is a *value destroyer* with a P&L
   bound derived from committed measured reliability (§5). The σ-inflation that
   fixes its calibration does so precisely by *removing the false confidence
   that generated those bets* (§4) — it deletes phantom edge rather than
   exposing real edge, and under the live boundary guard the surviving tradeable
   cohort collapses toward empty. No committed evidence shows residual
   net-of-fee edge after recalibration.

**Recommendation:** keep `intraday_asos_high` in shadow-only; do **not** invest
in wiring an ASOS NO-bin recalibrator + live flip. If the lane is to be given
one last look, the cheapest decisive experiment (§7) is to run the *already
committed* settled-P&L backtest where the ASOS corpus exists (the VPS, or with
IEM egress enabled) — it is otherwise fully built from committed snapshots.

---

## 2. Data inventory & reproducibility limits

| Input | Committed? | Source | Usable here? |
|-------|-----------|--------|--------------|
| CLI settlement truth (high_f, prelim flag) | ✅ | `data/snapshots/*/settlements.json` (84 files, 445 city-day, 436 FINAL) | ✅ rebuilt |
| Same-day KXHIGH market **ask** prices | ✅ | `data/snapshots/*/2000/markets.json` (`yes_ask_dollars`, geometry, expiry) | ✅ 405 clusters |
| Remaining-gain climatology (μ_gain, σ_gain) | ✅ | `data/asos/climatology.json` | ✅ |
| Recalibration G (CLI − hourly max) distribution | ✅ | `data/research/2026-06-10-asos-continuous-gap.json` (`G_value_counts_intF`) | ✅ exact mean/var |
| Prototype trade-cohort reliability | ✅ | `data/research/2026-06-05-asos-reliability.json` | ✅ P&L bound |
| **Intraday `M` (max-so-far at decision)** | ❌ | ASOS hourly obs (gitignored) / IEM (egress-blocked) | ❌ **blocker** |
| Forward shadow corpus (`predictions.jsonl`) | ❌ | live VPS `state/` (gitignored) | ❌ (out of scope) |

**Would-be backtest size (committed):** 2,430 same-day KXHIGH markets across
**405 (city, obs_date) decision clusters** at the 2000Z snapshot (81/city ×
MIA/AUS/CHI/NYC/LAX), **391 with CLI settlement**, spanning 2026-03-26 →
2026-06-14. The snapshot doubles as a clean, look-ahead-safe decision point: the
market ask, the `M` cutoff time (`snapshot_ts`), and the CLI settlement are all
anchored to it. The *only* missing join is `M`. The backtest is "one re-fetch
away."

---

## 3. Recalibration parameters confirmed (committed)

Recovered exactly from the committed `G_value_counts_intF` (whole-°F-quantized
G = CLI_high − hourly-METAR max; reproduces the 2026-06-13 recut's G-fit step —
e.g. NYC mean(G)=0.765 matches the recut doc):

| City | n | mean(G) °F | var(G) | sd(G) |
|------|---|-----------|--------|-------|
| MIA | 81 | +0.951 | 0.467 | 0.683 |
| AUS | 81 | +0.963 | 0.431 | 0.656 |
| CHI | 84 | +0.738 | 0.955 | 0.977 |
| NYC | 81 | +0.765 | 0.426 | 0.653 |
| LAX | 84 | +0.929 | 0.519 | 0.720 |

The recalibration is μ_gain += mean(G) (≈ +0.74…+0.96°F) and
σ_gain → √(σ_gain² + var(G)) (adds 0.43…0.96°F² in quadrature). The 2026-06-13
recut's OOS finding — that **μ-only worsens the bin and only μ+σ-inflation
calibrates it** — is consistent with these magnitudes; its held-out *reliability*
step is not re-runnable here (needs `M`), but its inputs reproduce bit-for-bit.

---

## 4. σ-inflation destroys the sharpness in exactly the traded zone

Using the committed climatology σ_gain and the committed posterior math
(`bracket_probability`, truncated-t ν=5) on the canonical post-peak grazer setup
(ABOVE 1°F bracket, `M` = floor−0.5°F):

| City | hour | σ_gain → σ_inflated | P(YES) raw → recal | Δ |
|------|------|---------------------|--------------------|---|
| NYC | 20Z | 0.72 → 0.97 (+35%) | 0.258 → 0.587 | **+0.329** |
| NYC | 22Z | 0.51 → 0.83 (+62%) | 0.121 → 0.529 | **+0.408** |
| CHI | 20Z | 0.62 → 1.16 (+87%) | 0.229 → 0.645 | **+0.416** |
| MIA | 18Z | 0.54 → 0.87 (+62%) | 0.151 → 0.639 | **+0.488** |
| LAX | 21Z | 0.55 → 0.91 (+64%) | 0.146 → 0.607 | **+0.460** |

In the afternoon decision hours the strategy actually trades, σ inflates **+35%
to +97%**, and near-strike setups that the raw posterior called "near-certain
NO" (P(YES) ≈ 0.12–0.26) become **coin-flips** (P(YES) ≈ 0.53–0.65). This is the
mechanism made quantitative: the recalibration's correctness for the NO bin and
its destruction of sub-bracket sharpness are *the same operation*. There is no
σ-inflation setting that keeps one without losing the other — var(G) is fixed by
the data.

---

## 5. P&L bound on the raw-posterior low-p NO bin (committed reliability)

From `data/research/2026-06-05-asos-reliability.json` (prototype trade cohort,
HIGH / 2000Z / {MIA,AUS,CHI,NYC}, 22 days), the binding bin:

> bin (0, 0.05]: **n=23**, predicted P(YES) = **0.016**, realized = **0.174**.

A NO contract bought at cost `c = 1 − yes_bid` has
`EV_gross(c) = (1 − realized) − c = 0.826 − c`.

- **Gross break-even NO cost:** c = 0.826 (net of the ~1¢ taker fee at c≈0.90: c ≈ **0.816**).
- The 7¢ edge filter, on a model that says P(YES)=0.016, permits NO fills at
  cost up to `c = (1−0.016) − 0.07 = 0.914`.
- ⟹ **Every NO fill with cost c ∈ (0.816, 0.914] is EV-negative**, with an
  expected loss up to **≈ 10¢/contract** at the aggressive end. Because this bin
  is where the raw model is *most confident* (and most willing to pay a high NO
  cost), it dominates the boundary-guarded cohort (23 of 36 rows) and drags
  realized P&L negative. This is the live-flip blocker, quantified from
  committed measured outcomes rather than asserted.

**What recalibration does to this bin:** it lifts predicted P(YES) from ~0.016
toward the realized ~0.13–0.17. The 7¢ filter then permits NO fills only up to
`c ≈ (1−0.13) − 0.07 = 0.80`, just **below** the net break-even (0.816). So
recalibration **removes** the losing NO bets — it does not convert them to
winners. And the markets that would still clear the filter (yes_bid ≥ ~0.20,
where the raw model had said 0.016) are essentially nonexistent in this cohort.
Net effect: the NO bin goes from "strongly negative" to "≈empty," not to
"positive."

The complementary high-p YES tail (raw bins 0.5–1.0, realized 0.71/0.75/1.0)
*is* well-calibrated and genuinely profitable in the **raw** cohort — but the
**live boundary guard** removes most of it (the guarded cohort retains only
n=1 in (0.7,0.9] and n=3 at 0.95+, the latter too deep-ITM to clear fees after
recalibration). So under the live config there is no profitable cohort left for
the recalibrated posterior to harvest.

---

## 6. Why this is decisive enough to lean KILL without the full backtest

The settled-P&L backtest would test "does net-of-fee edge survive σ-inflation?"
The committed evidence answers the *components* of that question directly:

1. The edge the raw strategy booked was concentrated in (a) a **miscalibrated
   low-p NO bin that is EV-negative** (§5, committed P&L bound) and (b) a
   **high-p YES tail the boundary guard removes** (§5).
2. σ-inflation **corrects (a) by deleting it** and does nothing to recover (b)
   (§4 — it only de-sharpens further).
3. Therefore the recalibrated, guarded strategy trades a near-empty book with no
   identified source of net-of-fee edge.

This is consistent with — and now mechanistically explains — every prior
finding: the FV-conditioned markout death (2026-06-12, "does not invert"), the
readiness "NOT YET" (2026-06-12), the continuous-gap null base rate for
intraday data-levers, and the prototype's own verdict ("raw-posterior
tradeability UNRESOLVED, leaning negative"). Intellectual honesty cuts toward
KILL here, and that is a *useful* outcome: it closes a lane that has consumed
multiple research cycles.

**Residual uncertainty (why "lean," not "confirmed"):** the §5 P&L bound is a
cohort-level bound, not a per-fill settled backtest, and it rests on the small
prototype cohort (22 days, n=23 in the binding bin). It is conceivable — though
unsupported by any committed evidence — that a per-fill backtest on the 391
committed clusters surfaces a narrow, well-calibrated YES sub-cohort that
survives both the guard and fees. The only way to rule that in or out is §7.

---

## 7. Cheapest next experiment — run the (now fully built) backtest on the VPS

The settled-P&L backtest is **implemented end-to-end** in the committed script
(`run_settled_backtest` / `run_backtest`), not stubbed. It runs the moment
intraday `M` is available. The script acquires `M` in priority order:

1. `data/research/asos/k{city}_asos.csv` — the recut corpus (present on the VPS);
2. live IEM fetch (auto-caches to that CSV) when `mesonet.agron.iastate.edu`
   is reachable;
3. neither → the backtest half is skipped with an explicit reason and only the
   committed-data half prints (the state in this egress-blocked routine sandbox).

On the VPS (CSV corpus present, or IEM in egress), simply run:

```bash
PYTHONPATH=. uv run python scripts/research/research_2026_06_15_asos_no_bin_tradeable_edge.py \
    --json data/research/2026-06-15-asos-tradeable-edge-backtest.json
# expose the full NO-bin cohort (drop the live min_calibrated_prob=0.11 floor):
PYTHONPATH=. uv run python scripts/research/research_2026_06_15_asos_no_bin_tradeable_edge.py --no-floor
```

What it does, faithful to the live path: for each 2000Z snapshot it reconstructs
`M` (LST-framed max-so-far up to `snapshot_ts`), looks up climatology at the
snapshot UTC hour, prices every same-day KXHIGH bracket under (a) raw and (b)
μ+G & σ-inflated posteriors, applies the live `AsosGuardConfig` defaults
(boundary ≥1°F, σ-floor 0.5°F, freshness ≤2h), the 7¢ edge filter and the 0.11
`min_calibrated_prob` floor, sizes with fractional Kelly (kelly 0.075, max 15
contracts), **fills at the ask**, charges the exact Kalshi taker fee
(`ceil(0.07·C·P·(1−P)·100)/100`), collapses to one bet per (city,day) event,
settles against committed CLI truth, and reports P&L / ROI / win-rate with
**day-clustered (city,date) bootstrap CIs and the low-p NO bin isolated**, for
raw vs recalibrated. G is fit look-ahead-clean on `obs_date < 2026-04-28`; only
`obs_date ≥ 2026-04-28` is traded (~half of the 405 clusters).

**Verdict rule:** recalibration KEEPS edge iff the RECAL variant shows P&L > 0
with a day-clustered per-contract CI **lower bound > 0** net of fees. If it
confirms negative/zero net edge (as §4–§6 predict), the KILL is final. If it
surprises with a positive, fee-surviving sub-cohort, *that* — not aggregate
Brier — is what would justify re-opening the forward-shadow gate.

> The script was exercised here against a synthetic ASOS fixture to confirm the
> backtest path runs end-to-end (sizing, fees, settlement, day-clustered CIs,
> NO-bin isolation, JSON export) without errors; the fixture produces no
> meaningful numbers and is not committed (`data/research/asos/*.csv` is
> gitignored). Real numbers come from the VPS run.

---

## 8. Reproduce

```bash
PYTHONPATH=. uv run python scripts/research/research_2026_06_15_asos_no_bin_tradeable_edge.py
```

Always prints the committed-data half (§1 G parameters, §2 σ-erosion table, §3
coverage, §4 P&L bound). It then attempts the settled-P&L backtest: in this
egress-blocked sandbox the IEM fetch returns `403 / not in allowlist` and the
backtest half is skipped with that reason; on the VPS it runs automatically
(§7). All §1–§5 numbers in this report are emitted from committed data only.
