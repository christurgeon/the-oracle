# Intraday settlement lock-in: mature-city falsification + new-city opportunity

**Date:** 2026-06-09
**Script:** `scripts/research_2026_06_09_lockin_backtest.py` (all five rules +
named cohorts are encoded in the script — every number below reproduces from
it alone)
**Artifact:** `data/research/2026-06-09-lockin-backtest.json`
**Status:** Mature-city lock-harvest **NO-GO** (3 of 5 rules strictly
negative; the best pre-named subgroup washes; zero harvestable premium
demonstrated). New-city (11-city expansion) immature-book probe **OPEN**.
**Review:** adversarially reviewed by backtesting-analyst 2026-06-09; this
revision incorporates all findings (reproducibility, slice honesty, LST frame
in the backtest itself, percentile indexing, penny-bet cohort, bias-mechanism
attribution).

## Hypothesis tested

Once the day's extremum is locked by realized observations (the daily low can
only fall, the daily high can only rise), some same-day KXHIGH/KXLOWT bracket
outcomes become deterministic or near-deterministic intraday, while market
quotes may still price residual uncertainty. Can that premium be harvested
fee-aware at executable quotes?

**Corpus:** 74 snapshot days with usable joins (4 slots/day) × 5 mature
cities (NYC, CHI, MIA, LAX, AUS), ASOS hourly METAR (IEM) extremum-so-far in
the **local-standard-time climate-day frame** joined against NWS CLI final
settlements, taker fees, day-clustered bootstrap CIs (B=2000).

## Verdict: on mature books, no rule clears fees

| Rule (pre-named in script) | n | win | ROI | day-clustered CI ($/contract) |
|---|---|---|---|---|
| CERT — deterministic side, premium ≥2¢ | 2 | 100% | +11.9% | books leave essentially zero premium on determined brackets |
| TAIL — YES on M-bracket, ask ≤0.93, all slots | 811 | 28.4% | −11.9% | **[−49.2, −10.0]** |
| TAIL `tail_nonpenny` (ask >0.05) | 415 | 55.4% | −8.5% | **[−41.1, −1.3]** |
| TAIL `tail_postext` (post-extremum hours) | 332 | 25.6% | −12.0% | **[−22.0, −0.2]** |
| TAIL `tail_postext_favorite` (post-ext, ask ≥0.5, margin ≥1°F) | 48 | 85.4% | +3.9% | [−3.9, +6.3] — **washes** |
| RESET — NO on re-set longshots (region ≤ M−2°F, bid ≥3¢) | 418 | 78.5% | −8.1% | **[−42.6, −15.2]** |
| LATE — late-evening LOW lock (LST ≥20h, ask ≤0.95) | 25 | 60% | −7.4% | [−4.5, +1.8] |
| OVNT — overnight determined-outcome | 36 open markets | — | — | **0 mispricings**; books at 0.99/0.01 |

Honest summary: the broad rules are strictly negative; the single best
pre-named subgroup (`tail_postext_favorite`) is point-positive at roughly fee
scale with a CI spanning zero — *not falsified as negative, but no
demonstrated edge*, and it is the slice most exposed to selection pressure.
The market's asks ≈ realized win rates throughout (well-calibrated lock-risk
pricing). When our hourly-obs signal disagreed with the market (sub-0.5 asks),
the market won ~91% of the time. **The mature-city intraday book is at least
as sharp as hourly METAR + climatology; there is nothing here to harvest at
taker prices.**

Caveats (apply to any reuse of these numbers): quotes are 4-slot snapshots,
not continuous; top-of-book only (no depth); multiple slots of the same
ticker-day are separate decision points (day-clustering mitigates but
`n_tickers` < `n` throughout); 3 NYC settlement dates missing from the join.
None of these plausibly flips a strictly-negative CI to materially positive.

## Load-bearing side findings

1. **NWS CLI climate day is midnight-to-midnight LOCAL STANDARD TIME**, not
   civil (DST) time. Verified two ways: (a) NYC 2026-03-27 — a 66 °F ob at
   00:51 EDT (= 23:51 LST 03-26) belongs to climate day 03-26; the civil
   frame mis-assigned it, manufacturing an "impossible" settlement (CLI high
   65 < civil-frame M 66); (b) **re-running the whole backtest in the LST
   frame eliminated all 10 CERT violations** (10 → 0) that the civil frame
   produced. **Production `python/data/asos.py` + `build_asos_climatology.py`
   use civil `ZoneInfo` dates** → `intraday_asos_high`'s runtime `M` and its
   climatology fit frame are contaminated by one boundary hour per day during
   DST. Fix scoped in the new-city spec §4.4 (standalone Phase 0).
   Note: a minority of the civil-frame violations (the CHI April/May cases)
   were hourly-vs-continuous sensor discrepancies, not frame errors — both
   failure modes are real; the LST rerun bounds the frame component.
2. **Hourly-METAR extremum understates the CLI extreme by ~1 °F at
   end-of-day** (LST frame, pooled): mean(final − M_hourly) at the 0200Z slot
   = **+1.03 °F** (high) / **−0.75 °F** (low); p95 = 2 °F. Mechanism is a mix
   of (a) CLI extremes coming from the continuous 1-minute sensor vs our
   hourly samples and (b) residual same-day drift for the latest-peaking
   cities at that slot — treat ~1 °F as the buffer floor and 2 °F as the
   conservative buffer for any near-certainty logic. IEM's **1-minute ASOS**
   archive closes the gap offline but is **next-day batch** (probed
   2026-06-09: same-day requests return empty) → unusable as a runtime
   signal; climatologies must be fit on hourly-frame M (train/serve parity),
   never on 1-minute M.
3. **LOW evening re-set risk is first-class, not a tail** (406 city-days,
   Mar–Jun): P(final ≥2 °F below the noon min-so-far) = 27% pooled (CHI 39%,
   NYC 31%, AUS 31%, LAX 21%, MIA 15%), collapsing to 5.9% pooled only after
   21:00 LST. The daily low frequently occurs at 23:xx LST. This vindicates
   the ASOS-high spec's LOW exclusion as a *posterior-shape* problem and
   quantifies what a future LOW-side climatology must model.
4. **Prelim CLI** (already fetched by `python/data/nws.py`) is the exact
   settlement instrument's own running extremum; per the 2026-05-22
   Hypothesis-E audit, LOW prelims captured at 2000Z are final 83% of the
   time. Useful as a future exact-M re-anchor after ~21Z.

## Why this isn't another "intraday is dead" entry

The no-go is **specific to mature books**. It is the control measurement for
the one surviving, unmeasured question on the 11 new cities (ATL, DAL, DC,
HOU, LV, MIN, NOLA, OKC, PHX, SATX, SFO — opened ~June 2026).

**Correction (2026-06-09, post-Opus-review):** an earlier draft claimed
new-city spreads of "2–16 ¢ vs 1.3–1.5 ¢ mature." That mature baseline was a
single-slot six-market artifact. The committed census
(`scripts/research_2026_06_09_newcity_spread_census.py`, fixed definition:
same-day, all 4 slots, mid ∈ [0.05, 0.95]) shows pooled new-city spreads
**identical** to pooled mature spreads on 2026-06-09 (avg 5.7 ¢ / median 2 ¢
both cohorts); SATX (16 ¢) and ATL (8 ¢) are individually wide, MIN (1.2 ¢)
is tighter than mature. **No width asymmetry is established** — and width
alone is anti-taker anyway (the premium must clear half-spread + fees). Real
same-day volume (1k–11k contracts by 14Z) and realtime IEM METAR for the new
stations are confirmed.

The open question is therefore **quote centering**: do new-city midpoints
deviate systematically from an observation-anchored posterior by more than
executable costs? That is cheaply measurable offline from accruing snapshots
— measurement program, pre-registered gates, and the conditional production
pathway live in the spec (v3).

## Do not re-pitch (mature-city scope)

- CERT/RESET/LATE/overnight lock harvest on NYC/CHI/MIA/LAX/AUS — measured
  above; strictly negative or zero-opportunity.
- "Buy the intraday locked favorite" on mature cities — best pre-named slice
  washes at fee scale (CI spans zero); the market is calibrated on lock risk.
- Re-slicing this corpus for a positive subgroup — 20+ implicit subgroups
  exist; anything found post-hoc needs a fresh forward window to count.
