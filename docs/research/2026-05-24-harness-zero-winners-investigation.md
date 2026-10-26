# Per-cell calibrator harness — investigation of 13/14 do_not_trade verdict

**Date:** 2026-05-24
**Artifact under investigation:** `data/calibration/per_cell_winners.json` (harness 0.1.0, corpus 2026-03-19 → 2026-05-21, 62 settled days, hash `7053d282…`)
**Question:** is the harness right (production is leaking), or is it wrong (gates mis-calibrated)?

## Branch 1 — Per-cell failure-reason audit (from `per_cell_winners.json`)

The dominant failure mode is **FR-7.2 (ROI_point > 0)**. For every one of the 13 do_not_trade cells, `iso_v2_global.simulated_roi.mean` ranges from −0.16 to −1.07. The narrow CIs at ≈−1.069 (e.g. MIA/high, DEN/high, LAX/high) mean *every* selected validation trade lost (full bet + fees). M-1 (insufficient_data) is a secondary failure mode hitting only small-n candidates (LAX/low identity at n=28, all EMOS rows on the 7 cells where EMOS isn't fit). **Identity fails FR-7.1 by construction** (LogSkill is always 0 vs itself); on 10/13 cells identity also fails the n_clipped:>5% extreme-prob veto, but this is moot because LogSkill_LB > 0 was already unreachable. The verdict_message "identity beats both candidates" is misleading — identity isn't winning; nothing is winning.

## Branch 2 — Live-PnL vs harness verdict (`state/trades.jsonl` + `data/shadow_settlements.full.json`)

For the 6 currently-trading cells with sufficient live data (NYC/{high,low}, MIA/{high,low}, LAX/low, AUS/low), the harness agrees with live on 4 (NYC/low −76.5%, MIA/low −37.2%, LAX/low −11.9%, MIA/high ≈0%) but **disagrees on 2: NYC/high (+11.5% live ROI, n=51) and AUS/low (+14.4%, n=25)**, where iso_v2 harness ROI is −0.71 and −0.96 respectively. Among excluded cells, 4 would have been profitable in shadow: AUS/high (+43.1%), CHI/high (+37.0%), CHI/low (+27.2%), SEA/high (+15.0%) — all rated ≈−1.07 by the harness. **The harness's `simulate_stage1_roi` is systematically more pessimistic than reality.** Mechanism hypothesis: the simulator applies only the plateau, calibrated-floor, and Kelly filters; production additionally applies per-day event cap, spread-skill edge filter, orderbook depth check (VWAP fill recompute), volume/OI, and same-day gates — production trades a much smaller, selected subset of what the harness simulates.

## Branch 3 — Synthetic sensitivity test (`/tmp/harness_sensitivity_test.py`)

Custom `PerfectCalibrator` candidate with `p_true` known by construction, raw probs deliberately biased toward 0.5, market price matching raw (clean edge for the perfect candidate). Results scaling with corpus size:

| n_dates | n_val_total | n_val_traded | LogSkill_LB | fold_consistency | ROI_point | Verdict |
|---|---|---|---|---|---|---|
| 60 | 72 | 29 | −0.007 | 4/5 | insufficient_data (n<30) | do_not_trade |
| 200 | 240 | 86 | −0.010 | 3/5 | +0.062 (ci_lb=−0.005) | do_not_trade |
| **600** | **720** | **298** | **+0.004** | **4/5** | **+0.027** | **winner: perfect_calibrator** |

The harness picks the winner at n_dates≈600 but not below. Current per-cell corpus is n_val_total≈144 (well below the 720 needed). **The gate logic is correct; the corpus is ~5× too small to detect calibrator improvements at the 0.04 LogSkill / 3% ROI magnitude the synthetic test exhibits.**

## Recommendation

**Verdict is not trustworthy for Phase 1B wiring.** Two distinct problems:

1. **Simulator-production gap (Branch 2).** The simulated ROI is provably pessimistic vs live PnL on at least 2 currently-trading cells. Before any verdict gates production behavior, `simulate_stage1_roi` must replicate the production filter stack (per-day cap, spread-skill, orderbook depth, same-day, volume/OI), OR the harness must add a per-cell calibration check against `state/trades.jsonl` for cells with sufficient live data (n≥30 fills) — note this re-introduces the survivorship bias the spec explicitly excluded. **This is a Phase 1B prerequisite, not a Phase 1A bug.**

2. **Under-powered corpus (Branch 3).** Even with a perfect simulator, the 62-day corpus yields n_val_total≈144 per cell — ≈5× too small for the gate ladder to confidently identify a winner with magnitude similar to real calibrator improvements (LogSkill +0.04, ROI +3%). The verdict "do_not_trade" should be re-read as **"insufficient evidence to prefer iso_v2/EMOS over identity at current corpus size"** rather than "trading is leaking." The M-1 hypothesis (early-gate `len(rows) < 30` vs post-CV `n_val_traded < 30`) is real but secondary — fixing it would route a few more cells to `insufficient_n` honestly rather than masquerading as `do_not_trade`, but would not change the 13/14 headline.

**Concrete next steps before Phase 1B:**
- Add a `simulate_stage1_roi_vs_live` reconciliation test: pick a cell with ≥100 live fills, compare harness ROI to attribution-module ROI, fix the filter delta until they agree within 5%.
- Tighten FR-11 early-gate to `n_val_total >= 75` (per task-16 M-1) so the harness honestly reports `insufficient_n` rather than triggering the `insufficient_data` ROI sentinel inside `do_not_trade`.
- Re-run the harness after the bounded-exposure pilot collects another ~30 settled days (≈2026-06-25); revisit verdicts then. With 90+ days, NYC/high and AUS/low — the live-profitable cells the harness currently flags — are the priority diagnostic cases.

Do not propose tuning FR-7 thresholds blindly. The gates are doing what they were designed to do; the inputs they receive are the problem.
