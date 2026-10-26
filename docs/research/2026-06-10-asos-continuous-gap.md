# ASOS continuous-vs-sampled high gap — verdict memo

**Date:** 2026-06-10
**Artifact:** `data/research/2026-06-10-asos-continuous-gap.json`

---

## Verdict (per city)

| City | n_CLI_days | mean G (°F) | Statistical verdict | Economic gate | Final |
|------|-----------|------------|---------------------|---------------|-------|
| NYC  | 81        | 0.765       | GREEN               | PASS          | **GREEN** |
| CHI  | 84        | 0.738       | GREEN               | PASS          | **GREEN** |
| LAX  | 84        | 0.929       | GREEN               | PASS          | **GREEN** |
| MIA  | 81        | 0.951       | GREEN               | PASS          | **GREEN** |
| AUS  | 81        | 0.963       | PROXY_INVALID (1-min within-day gaps) | N/A | **PROXY_INVALID** |

**These GREEN ratings confirm: (1) a real, statistically stable settlement bias at the stated magnitudes, and (2) that a correction of this size moves bracket probabilities by economically meaningful amounts. They are NOT proof of tradeable edge.** Whether the corrected posterior would produce positive live ROI is decided by the existing forward per-bin reliability gate (`scripts/intraday_asos_shadow_report.py` + calibration-analyst sign-off), not by this probe.

---

## Headline finding

The NWS CLI daily high (the Kalshi settlement value) consistently exceeds the hourly-METAR-sampled max that `intraday_asos_high`'s climatology targets. Mean G across the four GREEN cities ranges from **0.74 to 0.95°F** (median uniformly 1.0°F in all five cities; p95 = 2.0°F for all cities, indicating the tail is meaningful).

This result is **robust** because it is computed solely from the hourly corpus and CLI settlements — it is independent of the 1-min data and survives across the full ~81–84 CLI-day windows.

**G is whole-degree-quantized, and that's reassuring.** Both CLI and `tmpf` are reported in whole °F, so G is a count of integer events, not a smooth sub-degree offset. For NYC: 29 days at +0°F, 42 at +1°F, 10 at +2°F (`G_value_counts_intF` in the artifact). The ~0.85°F mean is therefore "CLI is a full degree above the hourly max on ~52–64% of days" — a genuine whole-degree sampling/rounding miss, **not** a decimal-vs-integer comparison artifact. (CHI carries two forecast-bust outliers at +4 and +7°F; the GREEN rests on the mean/CI, not the tail.)

**Mechanism.** The remaining-gain climatology (`build_asos_climatology.py`) fits to `max(tmpf)` over hourly METAR rows as the day extremum. Since CLI uses the continuous 5-min-averaged internal record, the whole posterior is silently calibrated to a target that is systematically ~0.85°F below settlement. The bias lives in the **build target**, not in runtime `M`: train and serve both undershoot identically, so `M + mu_gain` estimates the *hourly* day max without self-cancellation but settles against the *continuous* CLI high. A runtime "`M + G`" patch would introduce new bias on the gain side; the fix is a build-target correction.

**Plausible connection to the live-flip blocker.** The miscalibrated low-p NO bin (predicted P(NO) ~0.02 vs realized ~0.17) means the strategy bets YES near-certain but loses 17% of the time. A build-target raised by G would reduce the remaining-gain `mu_gain` by roughly G, shrinking predicted P(YES) in near-strike setups — directionally the right correction. This mechanism is plausible but not confirmed; confirmation requires the forward per-bin corpus, not this probe.

---

## Economic gate

The spec requires that a `+mean(G)` correction (simulated via `bracket_prob_shift`, which raises runtime `M` as an upper bound) moves probability across at least one bracket boundary in the near-strike cohort. Results for representative setups (ABOVE 1°F bracket):

**ABOVE bracket, near-strike (floor = 90, M = 89.5, mu_gain = 0.25, sigma_gain = 0.8, nu = 5).** These numbers are reproducible from the artifact's `economic_gate_upper_bound.by_city` block (all params explicit):

| City | mean G | base P(YES) | corrected P(YES) | delta |
|------|--------|------------|-----------------|-------|
| NYC  | 0.765  | 0.318      | 0.823           | +0.505 |
| CHI  | 0.738  | 0.318      | 0.802           | +0.484 |
| LAX  | 0.929  | 0.318      | 0.948           | +0.630 |
| MIA  | 0.951  | 0.318      | 0.964           | +0.646 |

All four cities produce deltas well above the +0.03 materiality floor. Shifts are large because ~0.74–0.95°F is roughly 74–95% of a 1°F bracket width: a near-boundary market flips decisively when the effective target shifts by nearly a full bracket.

**BETWEEN bracket (1°F wide, M near bracket center):** deltas of magnitude 0.31–0.50 in the same direction (correction reduces YES probability on BETWEEN markets as the posterior shifts above the cap). These are symmetrically large.

**Important caveat on direction.** `bracket_prob_shift` raises runtime `M` by G, which is an upper bound on the actual build-target-correction effect. The production fix lowers `mu_gain` by approximately G rather than raising M — the math is analogous but not identical (sigma_gain is unchanged, so the posterior narrows less sharply). The qualitative conclusion — that G-scale corrections produce economically meaningful probability shifts — holds either way; the exact magnitude will depend on the posterior's shape at each cell.

All four GREEN cities pass the economic gate.

---

## Decomposition (lever A vs lever B)

The probe decomposes G into lever A (between-sample miss: `H₁min − M_hourly`, recoverable by a faster real-time feed) and lever B (continuous/quantization residual: `CLI − H₁min`, the irreducible floor a faster feed cannot recover).

**Critical caveat.** The raw 1-min max can **overshoot** the official 5-min-averaged CLI high, so `H₁min − CLI` (lever B) goes negative for CHI (−0.32°F) and MIA (−0.88°F) on the overlap cut. The clean "lever A recoverable / lever B irreducible floor" decomposition does not hold: a faster real-time feed could read **above** settlement, not just closer to it.

Per-city overlap results (n_overlap_days = 20–44):

| City | n_overlap | leverA_overlap | leverB_overlap | G_overlap |
|------|-----------|---------------|---------------|----------|
| NYC  | 43        | +0.77          | +0.14          | 0.91     |
| CHI  | 44        | +0.98          | −0.32          | 0.66     |
| LAX  | 40        | +0.80          | +0.25          | 1.05     |
| MIA  | 32        | +1.97          | −0.88          | 1.09     |

The identity `leverA_overlap + leverB_overlap = G_overlap` holds on the overlap cut; `G_overlap` does not equal `G_settlement_bias` (which spans the larger all-CLI-days sample). The 2yr leverA values use a different, larger day-sample and do not add up to the overlap G.

For NYC and LAX (where leverB > 0), lever A dominates in the decomposition, suggesting most of G is between-sample. For CHI and MIA, the 1-min overshoot makes lever B negative, meaning H₁min is already *above* CLI on those days — the gap G is entirely structural quantization/averaging, not recoverable by a faster feed.

**Consequence for a faster real-time feed:** it would NOT clearly help and could read above settlement on CHI/MIA days (leverage B negative). Low priority.

**Overlap is thin** (n = 20–44 per city). Treat the decomposition as directional, not precise.

---

## Power and coverage

- **CLI-day count:** 81–84 per city (snapshots since 2026-03-18, spanning **4 calendar months, Mar–Jun, warm-season only**). Leave-one-month-out stability was required for GREEN; with only 4 months — all warm-season — this is modest power, and **the bias is NOT validated for cold-season behavior**. A year-round re-run is the natural follow-up as snapshots accrue.
- **AUS PROXY_INVALID root cause:** AUS has n_overlap_days = 20 and the 1-min corpus shows within-day gaps (leverA_2yr mean = −1.53°F with stdev = 6.07 — clearly corrupted, likely missing afternoon rows). The headline G ≈ 0.96°F for AUS is real (computed from hourly + CLI, independent of 1-min), but the decomposition is not trusted. AUS is excluded from GREEN, not because its G is small, but because the 1-min proxy fails the §3.3 validation gate.
- **UNDERPOWERED ≠ NULL.** An underpowered result would require more CLI accrual and a re-run — no dead-end note would be written. All four GREEN cities cleared the power floor (n ≥ 30, bootstrap CI lower bound ≥ 0.3°F, leave-one-month-out stable). For AUS, if a corrected 1-min corpus becomes available, a fresh probe would likely deliver GREEN given G ≈ 0.96°F with the same mechanism.

---

## Base-rate caveat

Intraday data-lever ideas have a **null base rate** in this repo: the ASOS-prior repurposing panel (Idea A/B/C) failed, and the intraday lock-in mature-city probe falsified 5 pre-named rules on NYC/CHI/MIA/LAX/AUS. This probe is a **measurement**, not a deployed edge. The G finding is robust, but robust measurements of a real bias do not automatically translate to realized P&L. The live-flip gate is the forward per-bin reliability corpus, not this gate or aggregate Brier.

---

## Next step (if GREEN holds at live-flip time)

A **deferred, separately-gated follow-up** to correct the climatology build target. Concretely: refit `data/asos/climatology.json` with `day_extremum` replaced by a corrected target (CLI-calibrated, conditional on city and month), then validate through the existing forward per-bin reliability corpus via `scripts/intraday_asos_shadow_report.py`. Calibration-analyst sign-off required before any live change to `intraday_asos_high`.

This is **NOT** a direct live change. The correction goes to the build target (`scripts/build_asos_climatology.py`), not to runtime `M` or any config. A runtime "`M + G`" shortcut is explicitly rejected (it would re-introduce bias on the gain side).

---

## Free side-note (spec §8)

It is worth a one-line empirical check whether the live IEM feed already returns 5-min HFMETAR rows (`report_type=1`) — a costless cadence upgrade for lever A if already present. However, given the overshoot finding (lever B is negative for CHI/MIA, meaning 1-min already exceeds settlement on some days), a faster feed is **low priority** and could introduce new sources of error rather than reduce them.
