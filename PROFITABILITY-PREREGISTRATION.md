# PROFITABILITY PRE-REGISTRATION -- the 2025 clean-split verdict

**Status:** FROZEN. This document and `backtest/ev_chain_constants.py` are ONE pre-registration in
two files, landed in ONE commit. This is the human-readable half; the constants module is the half
the code reads.

**What this document is for.** Phase 31 spends the single unburned 2025 NFL season exactly once, to
discharge the binding clean profitability verdict Phase 27 deferred and Phase 30 could not
discharge (`GATED-REFIT-READOUT.md` section 7b). Everything this phase publishes about 2025 is only
as good as one mechanical fact: that the rule commit strictly precedes the measurement commit, and
that the rule's content did not change in between. This document exists so that fact is checkable
by a reader, not merely asserted by an author.

**Editing after the measurement commit does not fix a bug -- it destroys the evidence.** Before the
CHECKPOINT 1 ratification, editing and re-committing this rule is legitimate. After the measurement
commit in Plan 31-14 it is not, and there is no honest repair path: a value that is wrong here is
wrong for the rest of the phase, and the only legitimate response is to say so in the readout.

**The ancestry anchor is recorded outside the files it witnesses.** Neither this document nor
`backtest/ev_chain_constants.py` records its own content hash. A file that must CONTAIN and exactly
REPRODUCE its own whole-file hash is self-referential: writing the hash changes the bytes the hash
was computed over, so no fixed point exists without a canonical exclusion rule nobody has defined.
Phase 30 already solved this -- `backtest/group_gate.py` RESOLVES the rule commit from git and never
transcribes it, and `tests/phase30_state.py` records the anchor in a LATER commit under its APPEND
PROTOCOL. Phase 31 does the same.

```
preregistration_anchor_recorded_in: tests/phase31_state.py
preregistration_anchor_slots: PRE_REGISTRATION_COMMIT, PRE_REGISTRATION_FILE_SHA256
preregistration_paths: PROFITABILITY-PREREGISTRATION.md, backtest/ev_chain_constants.py
```

`tests/unit/test_preregistration_ancestry.py` resolves the commit from git against those two paths,
asserts it equals the recorded slot, and recomputes each file's sha256 and compares. Plan 31-14
extends the same module with the assertion that this commit is a strict git ANCESTOR of the commit
that records the 2025 verdict artifact.

---

## 1. The windows, and the two discontinuities they cause

**Tune: 2021, 2022, 2023, 2024. Hold: 2025 only. Uniform across all three targets.**

This is D31-13, an owner decision that overrode the research recommendation of a per-target split.
The reasoning accepted: burnedness compromises EVALUATION, not FITTING, so leaving 2023-2024 unused
as a nuisance-parameter source would waste two seasons of estimation data out of ceremony.

**The first tune season's bias seed is 2018-2020.** 2021 has no prior tune season to estimate its
walk-forward bias from, so those three seasons -- the deployed artifacts' own train/validation
window -- seed ONLY the prior-season bias estimate. They are never candidates, never in the
frozen-SD fit, never in the threshold tuning, never in the trial selection.

**Game-type scope: playoffs EVERYWHERE.** The regular season and all four playoff types (`REG`,
`WC`, `DIV`, `CON`, `SB`) are ingested and carried in BOTH the tune window and the hold. The 2025
hold is therefore 285 games, not 272. This is D31-38, a post-research owner ruling that overrode a
recommended tune-regular-season-only split.

### The 2025 O/U result is NOT continuous with the Phase-27 and Phase-30 publications, for TWO reasons

Both are named here because one declared discontinuity with an unnamed second cause is the quiet
form of moving a goalpost.

1. **The tune window is WIDER (D31-13).** Phase 27 tuned O/U on 2021-2022 and held 2023-2024. This
   phase tunes on 2021-2024. The frozen residual SD, the prior-season bias and the selected EV floor
   `t` will all differ from the published Phase-27 and Phase-30 values.
2. **Playoff rows are now IN the tune population (D31-38).** The stored consensus rows were REG-only;
   the ingest admits playoffs, adding roughly 87 rows across 2018-2024 of which about 52 land inside
   the tune window and become new candidates for the frozen-SD fit, the prior-season bias and the
   EV-floor sweep. The tune population is therefore not byte-comparable to what Phase 27 and Phase 30
   fit on.

A consequence, not a cost: `regular_season_only` becomes a GENUINE robustness cut on both sides of
the split rather than the vacuous one Phase 27 disclosed.

**Nothing in this phase re-derives, re-publishes or re-reads a Phase-27 or Phase-30 figure as a
result.** The Phase-27 O/U reproduction may be published beside the 2025 result as CONTEXT ONLY.

---

## 2. The three narrowings, stated explicitly

An unstated narrowing is the quiet form of moving a goalpost. Each of these three refines a phrase
that was written earlier in this project and that would otherwise be read two ways.

**Narrowing 1: "2023-2024 stays burned" now means NOT EVALUATED, not NOT TOUCHED.** Under D31-13
those two seasons feed the frozen SD, the prior-season bias and the tune-side EV-floor sweep. No
figure is EVALUATED on them, no verdict is drawn from them, and no number from them is re-derived or
published as a result. What changed is that "not touched" was never the commitment the deferral
needed; "not evaluated" is.

**Narrowing 2: "that week's Friday 6 PM Eastern freeze" is refined to a PER-GAME freeze.** Staleness
is measured against the Friday 6 PM Eastern instant preceding EACH GAME'S OWN kickoff, not a single
weekly instant. The reason: a Thursday game's preceding Friday is seven days before the Friday
preceding that week's Sunday games, so under a per-week rule a correct Thursday snapshot is strictly
earlier than the week's freeze and would be suppressed every week as a pure calendar artifact. This
is D31-18, and it matches Phase 29's explicit `min(as_of, game_freeze)` ruling and
`get_synthetic_snapshot_ts`'s own semantics. The forward path is unaffected: the live Friday run
prices that week's Sunday and Monday games, whose per-game freeze IS that Friday.

**Narrowing 3: "the same apparatus" for WP means the same CONTRACT, not the same STEPS.** A
calibrated classifier has no residual to take a standard deviation of. Requiring WP to execute O/U's
literal steps would mean inventing a logit-space residual SD -- manufacturing a fitted parameter the
model does not need, on the target with the worst measured CLV. What WP owes is the same contract:
a pre-registered chain, a leakage fence, a calibration gate on tune data only, a registered fallback
that cannot fire silently, a real-price devig, a per-bet EV, and one EV floor. It delivers all of
those. It does not deliver a frozen residual SD, and it never claimed to.

---

## 3. The per-target chains

Each chain reads its window, its game-type scope, its bias policy, its eligibility rule and its
verdict vocabulary from `backtest/ev_chain_constants.py` and from nowhere else.

### 3.1 WP -- win probability

**Deployed artifact:** the WP model currently named in `artifacts/latest.json`
(`wp_20260824_113325` at the time of writing). Nothing is re-fit, re-tuned or swapped.

**Chain steps:**

1. Read the deployed artifact's isotonic-calibrated `P(home)` for each 2025 candidate game. This is
   the DEFAULT (D31-07).
2. Run the reliability and Brier calibration gate ON THE TUNE SPLIT ONLY -- 5 equal-count quantile
   bins, at least 20 observations per bin, a maximum per-bin `|mean_pred - realized|` of 0.10 (the
   frozen Phase-27 gate parameters, consumed unchanged).
3. IF and ONLY IF that gate FAILS, the REGISTERED FALLBACK fires: a prior-season
   probability-scale bias correction, walk-forward from strictly prior seasons. Its trigger is
   written into the trial registry's `fallback_trigger` field, so it can never activate silently,
   and a fired fallback ADDS one entry to the BH family (section 6).
4. Devig the REAL two-sided moneyline (`ml_home`, `ml_away`) with the existing
   `backtest.ou_ev_chain.devig`, reused verbatim -- no new devig implementation.
5. Compute per-bet EV as `p_side * payout - (1 - p_side)` with the devig-derived payout, through the
   existing `per_bet_ev`.
6. Admit iff per-bet EV is at or above the WP EV floor `t` (section 5).

**Eligibility rule:** NONE. The EV floor alone decides (D31-05).

**Bias policy:** none on the default path. The prior-season probability-scale correction is a
registered fallback and fires only on a gate failure.

**Why this inverts D27-07's structure:** there, prior-season bias subtraction is the default and
isotonic is the fallback, because the O/U model is not calibrated. The WP model is. Re-fitting
calibration on the tune split was rejected because the bet list would then be priced off a model
nobody runs, and the verdict would describe a model nobody runs.

### 3.2 ATS -- against the spread

**Deployed artifact:** `ats_20260605_220128`, RETAINED at the Phase-30 gate (it failed the paired
non-regression gate, `-0.212797`, `p = 0.0375`, and its incumbent was honestly kept).

**Chain steps:** a clone of the O/U totals converter with the ATS residual contract substituted for
the O/U one.

1. Read the deployed artifact's `model_spread` for each candidate game.
2. Bias-correct BEFORE the CDF: add the PRIOR-SEASON WALK-FORWARD mean residual, estimated by
   `estimate_prior_season_bias` from strictly-prior seasons only.
3. Convert to `P(home covers)` with a frozen residual standard deviation fit on BIAS-CORRECTED TUNE
   residuals only (`fit_frozen_residual_sd`), through the same normal-CDF step the O/U converter
   uses, clipped to the same disclosed `P_OVER_CLIP` band so an extreme z-score resolves to a bound
   rather than to NaN and the Kelly tail stays bounded.
4. Devig the real two-sided spread prices (`spread_ju_home`, `spread_ju_away`).
5. Per-bet EV through the same `per_bet_ev`.
6. Admit iff per-bet EV is at or above the ATS EV floor `t`.

**Eligibility rule:** NONE. The EV floor alone decides (D31-05).

**Bias policy: the ATS chain CARRIES a prior-season bias correction.** This answers a question the
phase context left open, and it is pre-registered here before any 2025 number exists.

The residual contract is `residual = actual home margin - predicted home spread`. Re-derived in
Plan 31-02 by scoring the DEPLOYED artifact through `score_deployed_artifacts("ats")` over canonical
gold, the tune-window POOLED bias is `+0.58937727047262922` over n = 1139 (sd `11.104272279463643`,
t `1.7912868714554835`, p `0.073512927737913666`). A POSITIVE pooled residual means the model
UNDER-predicts the home margin on the window as a whole, so adding the bias to the predicted spread
RAISES the cover probability -- **the OPPOSITE sign to the O/U contract**, where the bias is
negative and adding it pulls the total DOWN and LOWERS `P(over)`.

**The direction claim is POOLED, and one tune season is NEGATIVE.** The four re-derived per-season
means are:

| season | n | mean residual |
|---|---|---|
| 2021 | 285 | `+0.68147702779163399` |
| 2022 | 284 | **`-0.03539119799896865`** |
| 2023 | 285 | `+0.58219338768864415` |
| 2024 | 285 | `+1.1270376943443952` |
| **pooled** | **1139** | **`+0.58937727047262922`** |

2022 is negative and that is simply true. It does NOT weaken the chain, because the correction the
chain actually applies is the PER-SEASON WALK-FORWARD estimate, which reads only strictly-prior
seasons and therefore never assumes a constant sign. The pooled figure supports ONLY the magnitude
argument. Naming the negative season here is what stops a later reader discovering it and concluding
the direction had been overstated.

**The magnitude argument, which is why the correction is carried rather than omitted:** the implied
cover-probability shift `Phi(pooled_mean / pooled_sd) - 0.5` is `0.021164571224560169`, against a
flat -110 breakeven edge of `0.023809523809523836`. The bias is `0.88891199143152611` of the entire
house edge. Omitting it is not a neutral simplification.

**The research figures are superseded.** `31-RESEARCH.md` reported a pooled bias of `+0.714049`
(2022 at `+0.118713`) measured from `data/web_cache.duckdb`. That cache is STALE relative to the
Phase-25 re-fit. The cache figures MUST NOT be used anywhere in this phase.

### 3.3 O/U -- totals

**Deployed artifact:** `ou_20260326_163930`, the v1.0 pre-Elo model, RETAINED across both the
Phase-25 and Phase-30 gates.

**Chain steps:** the existing Phase-27 chain, executed on the Phase-31 window.

1. Read the deployed artifact's `model_total`.
2. Bias-correct BEFORE the CDF with `estimate_prior_season_bias` (the LOCKED D26-18 / D27-07
   residual contract: `residual = actual_total - model_total`, so an over-predicting model gives a
   NEGATIVE bias and adding it pulls the total DOWN).
3. `calibrated_p_over` with the frozen residual SD fit on bias-corrected TUNE residuals only, clipped
   to `P_OVER_CLIP`.
4. Devig the real two-sided total prices (`total_over_ju`, `total_under_ju`).
5. Per-bet EV, and admission at the O/U EV floor `t`.

**Eligibility rule: the frozen Phase-26/27 UNION, consumed unchanged (D31-06).** A candidate is
eligible if it is an UNDER pick OR the game is a high-total game, where "high total" is
`backtest.ou_divergence.HIGH_TOTAL_BOUNDARY_PREHOLD` -- the leakage-clean upper-tertile boundary
re-derived on pre-hold 2018-2022 data only (it lands at 48.0). The constant and the rule are READ,
never re-derived, so O/U's eligibility pre-registration is provable BY GIT ANCESTRY ALONE.
Re-deriving the boundary on 2018-2024 was rejected: it would re-derive a LOCKED-1 constant at exactly
the moment it could change the answer, and would contradict D30-13 three months after it was ruled.

**Bias policy:** the prior-season walk-forward correction, on the default path, per the LOCKED
Phase-27 contract.

---

## 4. The 2025 odds: source, ingest write contract, and what is excluded by construction

### 4.1 The source

**The 2025 odds source is the nflreadpy closing lines** (SPEC R2), ingested through
`scripts/ingest_historical_odds.py`.

**The partitioned silver odds store is NOT the source, and that is stated here before the run so no
one can argue the source was chosen after the fact.** The `2025-10-03 18:00:00-04:00` partition holds
real nine-book 2025 data. The OUM-06 allowlist (`backtest/ou_divergence._ALLOWED_SPORTSBOOKS`) is
`{consensus, draftkings}`, and the nine books present in that partition are `betmgm`, `betonlineag`,
`betrivers`, `betus`, `bovada`, `draftkings`, `fanduel`, `lowvig` and `mybookieag`. **`consensus`
does not appear in that partition at all, so the allowlist admits exactly ONE of the nine
(`draftkings`) and EIGHT are outside it.** Those eight are excluded BY CONSTRUCTION -- by never
sourcing 2025 from that store -- rather than discovered at run time by a rejection.

That count is MEASURED (Plan 31-02, `outputs/p31/odds_store_audit.json`). `31-RESEARCH.md` DEFECT-5
said "seven of the nine", which counted the allowlist's own size instead of the number of its entries
PRESENT in the partition. The measured figure is eight of nine, and the discrepancy is recorded here
rather than quietly reconciled.

### 4.2 What the historical juice ingest does (D31-15, D31-39)

**The historical four juice columns are PROMOTED from the partitioned silver store, not re-ingested
across the network.** D31-39 pre-registered a branch rule before the fact was known; Plan 31-02
measured the fact and the rule evaluated to PROMOTE. All 23 partition files are pyarrow per-write
GUIDs naming ONE logical table (silver `odds_snapshot`); every file carries all four juice columns
with real asymmetric prices (1,992 of 2,120 distinct `(game_id, sportsbook)` pairs carry a value
other than -110); and the rows agree with the flat table row-for-row on every shared value column
with zero mismatches. `31-RESEARCH.md` assumption A3 -- that the opaque filenames were
`ParquetManager` table-name hashes -- is FALSE, and is recorded as false.

### 4.3 The write contract

Every clause below is part of the frozen rule, not an implementation detail.

1. **Label.** Every row this phase writes carries `sportsbook = "consensus"`, the label the live
   silver rows already hold. `scripts/ingest_historical_odds.py`'s literal `"nflverse_closing"` is
   the DOCUMENTED LEGACY MISLABEL: it has been there since v1.0 Phase 02 (commit `b3f158d`), no row
   in live silver carries it, and no code in the repo writes the literal `"consensus"`. The OUM-06
   allowlist is NOT widened (D31-12 branch 1, settled by `31-RESEARCH.md` Priority 1).
2. **Juice columns are ADDITIVE.** `spread_ju_home`, `spread_ju_away`, `total_over_ju` and
   `total_under_ju` are added keyed on `game_id`. The stored `spread`, `total`, `ml_home` and
   `ml_away` are NEVER overwritten. The juice is used in the EV chain's devig ONLY; it reaches no
   model and no gold column, because `scripts/build_features.py` defines `market_feature_cols` as an
   explicit five-name allow-list that no juice-derived column is on.
3. **`snapshot_ts` is RE-DERIVED at ingest** through `get_synthetic_snapshot_ts`, from each game's
   own preceding Friday 6 PM Eastern instant, and stored as a TZ-AWARE datetime (D31-37). Every
   replay row is then exactly AT its freeze and therefore FRESH under the rule that at-freeze is
   fresh. The alternative -- exempting replay rows from the freshness check -- was REJECTED because
   it creates a SECOND freshness rule, and one registry rather than two lists is the 29-06 lesson
   this project already paid for. A type trap travels with this clause: the stored column is a
   string today, holding one fixed calendar date per season, and its single non-consensus row uses a
   different, space-separated UTC format. Any comparison must PARSE, never string-compare, and must
   be Eastern-anchored, not UTC-anchored.
4. **Game-type scope.** `REG`, `WC`, `DIV`, `CON` and `SB` are admitted in BOTH windows (D31-38).
   Preseason is excluded.
5. **Canonical keys, and the Rams resolution.** Every written `game_id` is constructed through
   `create_standard_game_id`, so the Rams key is the canonical `LA` that every other store uses. The
   stored table today carries the non-canonical `LAR` in 116 Rams `game_id`s, and `upsert_silver`
   keys on `game_id`, so `LAR` and `LA` are DIFFERENT KEYS. Measured by dry run in Plan 31-02: an
   `LA`-keyed re-ingest of 2024 took the table from 1,856 to 1,875 rows -- 19 rows ADDED, ZERO
   replaced, with the 116 `LAR` rows untouched. **The pre-registered resolution is to NORMALIZE the
   stored `LAR` keys to `LA` BEFORE any merge, not to replace the odds table outright.** Outright
   replacement would discard the accumulated juice history the PROMOTE branch depends on; key
   normalization is the minimal change that makes the upsert's own keying correct. A plain merge
   without it silently duplicates every Rams game.
6. **The one-row-per-key invariant.** After any ingest, exactly one row exists per
   `(game_id, sportsbook, snapshot_ts)` triple. A violation is a HARD FAILURE naming the offending
   keys, never a warning.
7. **The pre-ingest synthetic-id gate.** `assert_no_synthetic_game_ids` runs BEFORE any write. It
   rejects a `game_id` that fails `GAME_ID_PATTERN` and a well-formed `game_id` absent from
   `features_ou`. The provenance allowlist provably does not catch this case: the row
   `2025_W01_TEST@HOME` currently sitting in production silver has a legitimate sportsbook
   (`draftkings`) and an impossible `game_id`, and `assert_real_odds` ADMITS it.
8. **The completeness hard stop.** Zero rows, or fewer than 285 admitted 2025 games carrying a total,
   raises with the measured count in the message (SPEC R2).

**A note on DEFECT-4, so the row counts are not misread.** The 2024 transform yields 19 Rams rows
(17 `REG` plus 2 playoff) against 17 stored. Under D31-38 the two playoff rows are ADMITTED BY
DESIGN. They are not duplicates and are not evidence of a keying fault; the keying fault is clause 5
and is resolved there.

---

## 5. Sizing, admission, and the EV floor

**The EV floor `t` is selected per target by a sweep over the frozen `EV_FLOOR_GRID`
(`0.00, 0.01, 0.02, 0.03, 0.05`) ON THE TUNE SPLIT ONLY.** One scalar `t` per target is carried to
the hold. The sweep is RECORDED in the trial registry and is EXCLUDED from the BH family, because
those grid cells never touched 2025 (section 6).

**Admission: a bet is admitted iff its per-bet EV is GREATER THAN OR EQUAL TO `t`.** An EV exactly
at the floor is ADMITTED. The comparison is inclusive, stated here so it cannot be quietly flipped
in either direction after the counts are known.

**EV tier bands** are fixed absolute half-open `[lo, hi)` bands: low `[t, 0.03)`, medium
`[0.03, 0.05)`, high `[0.05, +inf)`. A boundary value falls in the HIGHER tier (D31-24). Per-bet EV
is unit-consistent across all three targets, which is why absolute bands are the only option that
keeps "high" meaning one thing.

**The weekly exposure cap is POOLED at 10 percent across all three targets** -- ONE cap over the
union of a week's bets, pro-rata scaled (D31-02). D27-10 pre-registered it as a per-week TOTAL
exposure cap, and the bankroll does not grow because targets were added. Ten percent per target
would have been a post-hoc tripling of a pre-registered ruin guard.

**The sizing pipeline order is LOCKED and unchanged: quarter-Kelly, then the 5 percent per-bet cap,
then de-weighting, then the 10 percent weekly cap.** Four steps, de-weighting third.

**The de-weighting formula (D31-03):**

```
factor = 1 / sqrt(max(same_side_group_size, same_game_group_size))
```

The stricter of the two groupings applies, so a bet correlated on both axes is penalised once rather
than twice. The same-GAME grouping is the addition. The three targets' side strings are disjoint
(`home`/`away`, `home_cover`/`away_cover`, `over`/`under`), so pooling a mixed week into a
side-only helper produces six groups that never mix: de-weighting would stay silently within-target
and the correlation that actually matters -- a WP `home` bet and an ATS `home_cover` bet on the SAME
game, close to one leveraged wager -- would go entirely uncaptured.

**The narrow, true claim about what de-weighting can do.** De-weighting cannot RAISE a week's TOTAL
exposure. It CAN raise an INDIVIDUAL stake, once the 10 percent weekly cap binds: de-weighting
shrinks the pre-cap total, which loosens the pro-rata scaling that the weekly cap then applies, and
a bet that was not itself de-weighted can end up larger than it would have been. This is stated in
its narrow form deliberately. An earlier draft of this claim said de-weighting could never increase
any stake, which is FALSE, and a worked case is recorded in `31-04-SUMMARY.md`. What matters for the
verdict is unchanged and is what the guard asserts: de-weighting runs AFTER EV admission, so it
cannot change WHICH bets are selected, and it cannot change the headline flat-stake ROI.

---

## 6. The multiplicity family, enumerated in advance

The Benjamini-Hochberg family is fixed HERE, before any 2025 number exists, so the denominator is a
pre-registration rather than an argument made afterwards (D31-33, D31-34).

**There is ONE registry artifact spanning all three targets.** A single BH family cannot be applied
across three independently written files.

**INCLUDED in the family (hold-side inferences only):**

- the PRIMARY 2025 profitability verdict for each of WP, ATS and O/U -- 3 entries;
- the robustness cut reported pass/fail for each target, counted ONCE per target -- 3 entries;
- a per-target CALIBRATION FALLBACK verdict, but ONLY IF that target's registered fallback actually
  fired -- 0 to 3 entries.

**EXCLUDED from the family:**

- **the tune-side `EV_FLOOR_GRID` sweep.** It is RECORDED in the registry for transparency and does
  NOT enter the family: the sweep runs entirely on the tune split and applies one selected `t` to
  the hold, so those grid cells never touched 2025. Counting every `(target, floor)` cell as 15
  would correct for fourteen tests never performed on the hold, making a real result harder to
  detect for ceremonial rather than statistical reasons.
- **every entry whose `entry_kind` is `control`.** The shifted-edge counterfactual pass (D31-08) has
  no p-value to correct, so counting it would statistically penalise the verdict for running a
  code-liveness check.

**The denominator is 6 with no fallback fired, and 7, 8 or 9 with one, two or three fired.**

**The robustness cut is counted ONCE, not twice, and here is why.** The frozen Phase-27
`ROBUSTNESS_CUTS` names two cuts, `regular_season_only` and `playoffs_excluded`. Measured in source
(`backtest/ou_monetization.py` lines 690-710), both keys are assigned the SAME
`_flat_roi_from_records(reg_season)` value from the SAME `week <= 18` filter -- they compute the
BYTE-IDENTICAL frame. One statistic reported twice is not two hypotheses. Both names are retained
for git-ancestry continuity with the Phase-27 record; only their family contribution is one.

**The trial registry schema** is the Phase-27 field tuple extended with `entry_kind` (taking
`inference` or `control`) and `fallback_trigger`. Registry row count and BH denominator are therefore
allowed to differ, visibly and by design: every read of 2025 is on the record, and the denominator
counts only actual inferences.

**The bootstrap configuration is the frozen Phase-27 one, consumed unchanged** (`BOOTSTRAP_B` 2000,
`BOOTSTRAP_SEED` 2704, percentile CI). **A disclosure travels with it:** a one-season hold gives at
most 22 season-week blocks against Phase 27's roughly 36, so the interval published here is WIDER
and the two intervals are NOT commensurable. That is a property of the design, disclosed before the
run, not a limitation discovered after it.

---

## 7. The ROI hypothesis test -- defined and frozen before any 2025 number exists

**No ROI hypothesis test exists in this repository today.** A verdict cannot rest on a statistic
invented after the numbers, so the test is registered here in full.

- `backtest/diagnose.py`'s `clv_significance` tests **CLV, not ROI**. Its own docstring says it
  mirrors the `ttest_1samp(clv, 0)` idiom.
- `backtest/ou_monetization.py`'s `_block_by_week_bootstrap_ci` returns a **percentile interval with
  NO p-value**.

**NULL.** `H0`: the POPULATION flat-stake ROI over the hold is LESS THAN OR EQUAL TO ZERO.

**STATISTIC.** The OBSERVED flat-stake ROI over the hold bets, computed by the same
`sum(payout_flat) / sum(flat_stake)` ratio the existing `_flat_roi_from_records` helper uses. No new
estimator is introduced.

**REFERENCE DISTRIBUTION.** The SAME block-by-week resamples the confidence interval already uses.
The `(season, week)` pair is the BLOCK. Blocks are resampled WITH REPLACEMENT from the HOLD bets
only, at the frozen `BOOTSTRAP_B` and `BOOTSTRAP_SEED`, and each replicate is then RECENTRED AT THE
NULL by subtracting the observed ROI. Resampling whole weeks preserves within-week correlation (a
half-point line move correlates the games on a slate); resampling individual bets would understate
the spread.

**ONE-SIDED ACHIEVED SIGNIFICANCE LEVEL.**

```
p = (1 + #{recentred replicates >= observed ROI}) / (BOOTSTRAP_B + 1)
```

**FINITE-SAMPLE RULE.** The plus-one in BOTH numerator and denominator makes `p` strictly positive
and never zero, so a run in which no recentred replicate reaches the observed ROI reports the
smallest attainable `p` rather than a `0.0` that would overstate the evidence.

**MINIMUM ATTAINABLE p-VALUE.** `1 / (BOOTSTRAP_B + 1)` = `1 / 2001` =
`0.0004997501249375312`.

**WHAT HAPPENS WHEN THAT MINIMUM EXCEEDS ALPHA.** If the minimum attainable `p` exceeds `alpha`, the
test CANNOT reach significance at ANY observed ROI, and a positive return MUST report
`INCONCLUSIVE_CLEAN` with that stated reason -- NEVER `PROFITABLE_CLEAN`. At the frozen
`BOOTSTRAP_B = 2000` the minimum is `0.0004997501249375312`, which is BELOW `alpha = 0.05`, so on
this configuration the test CAN reach significance. The rule is registered anyway, because a rule
that only exists once it binds is a rule chosen after the fact.

**RESOLUTION DISCLOSURE.** A one-season hold gives at most 22 season-week blocks, so the reference
distribution is COARSE: the attainable p-values form a discrete ladder in steps of
`1 / (BOOTSTRAP_B + 1)` over a resampling space of only 22 distinct blocks. That is a limitation of
the design, stated before the run rather than discovered after it.

### CLV SIGNIFICANCE IS REPORT-ONLY

**The CLV p-value is reported BESIDE the verdict and is FORBIDDEN from entering the multiplicity
family or from driving ANY verdict token.** The repository's existing significance primitive tests
CLV, not ROI. Reusing a CLV p-value as a profitability p-value would silently convert "significant
CLV" into "profitable" -- which is exactly the D25-14 and D26-09 trap this project has already been
caught by once, and exactly what this phase exists to prevent. Plan 31-12 adds the structural
assertion that the verdict-assignment path reads no CLV p-value at all, and the readout guard adds
the assertion that no sentence carrying a verdict token presents a CLV figure as its supporting
evidence.

---

## 8. The verdict vocabulary

The 2025 run may emit exactly one of five tokens per target. "No bets selected" is a FIRST-CLASS
OUTCOME that fills the same readout template slots as any other verdict.

| Token | Condition |
|---|---|
| `PROFITABLE_CLEAN` | at least one bet selected, observed flat-stake ROI strictly positive, and the pre-registered one-sided ROI p-value below alpha AFTER the BH correction |
| `UNPROFITABLE_CLEAN` | at least one bet selected and the observed flat-stake ROI negative or zero. A measured result, not a failure to measure |
| `INCONCLUSIVE_CLEAN` | at least one bet selected, observed ROI strictly positive, but the ROI p-value does not clear alpha after correction -- including the case where the minimum attainable p exceeds alpha, reported with that stated reason |
| `UNDISCHARGEABLE_NO_BETS` | the chain resolved, ran end to end on the 2025 frame, its positive control passed, and it selected ZERO bets. An explicitly defined PASS under SPEC R1 |
| `UNDISCHARGEABLE_NO_CHAIN` | the target's chain did not resolve on the 2025 frame at all -- a required input was absent, so no candidate could be priced. Distinct from NO_BETS: there the chain ran and declined, here it could not run |

A positive return that cannot be distinguished from zero is INCONCLUSIVE and is never called
PROFITABLE.

---

## 9. The one-shot enforcement

**"Runs exactly once" is enforced by a structural guard plus three blocking owner checkpoints**
(D31-16). D31-16 fixes the checkpoint count at THREE and explicitly declines a fourth: the readout
and `/bets` are signed off at the existing verify-work gate, and a fourth stop would duplicate a gate
that already runs.

### 9.1 The durable exclusive run ledger

**Path:** `config/profitability_2025_run_ledger.toml`. **States:** `armed`, `started`, `completed`,
`failed`.

**The ledger is created by an EXCLUSIVE file creation BEFORE the first 2025 read.** Refusing only
when the verdict artifact already exists is NOT crash-safe: the runner READS AND MEASURES 2025
BEFORE it writes the artifact, so a crash inside that window leaves no artifact and the next
invocation would be free to spend the single-use split again, with no record that it had already
been spent.

**Transitions.** `armed -> started` is written before the first 2025 read. `started -> completed` is
written after the verdict artifact lands. `started -> failed` is written on an error path.

**A ledger in state `started` or `failed` is NOT automatically rerunnable.** A new attempt requires
an OWNER RULING recorded in the ledger before the state may return to `armed`. Silently re-arming a
crashed attempt would spend the single clean split twice while leaving a record that says it ran
once.

**There is NO force flag,** and the runner additionally hard-refuses to overwrite an existing verdict
artifact.

### 9.2 The three checkpoints

1. **CHECKPOINT 1 -- pre-registration ratification.** Before anything runs. The owner reads this
   document and the constants module and states that this is the rule they intend to be bound by.
2. **CHECKPOINT 2 -- rebuild acceptance.** After the fingerprint ladder and before the verdict run.
3. **CHECKPOINT 3 -- the armed 2025 run.** Following a proxy rehearsal (section 9.3).

### 9.3 The rehearsal split is DISJOINT, and its result is discarded

**Tune 2021-2023, hold 2024.** DISJOINT on both sides.

You cannot dry-run on 2025 without seeing 2025, so the rehearsal proves the plumbing on an already
burned season. **Its result is DISCARDED and never reported, never published and never compared.**

**Why the obvious alternative is not available.** Overriding ONLY the hold to 2024, leaving the tune
window at the frozen 2021-2024, would make the hold season a TUNE season. The Phase-31 fence --
which mirrors the four checks in `_assert_fit_window` -- rejects a hold season consumed by the SD
fit or by the threshold tuning. The rehearsal would therefore either FAIL, or would only pass against
a fence weakened enough to permit leakage, in a phase whose entire value is temporal honesty. So the
rehearsal narrows the TUNE side too.

The rehearsal split is a separate named constant and is marked as NOT the pre-registered rule and as
never consumable by the armed run.

### 9.4 The counterfactual pass is INSIDE the one-shot run

Each chain's positive control is a SYNTHETIC unit control PLUS a SHIFTED-EDGE COUNTERFACTUAL PASS
over the ACTUAL 2025 frame, executed inside the SAME one-shot invocation and recorded in the verdict
artifact (D31-08).

The reason it must be on the real frame: a zero result has to be defensible on the 2025 frame
specifically, and a synthetic-only control passes even when a wiring fault -- a missing column, a
NaN, a silently empty merge -- produced a false zero. Running it inside the single invocation keeps
"exactly once" intact: ONE run, TWO recorded passes. The counterfactual is recorded with
`entry_kind = control` and does not enter the BH family.

---

## 10. The 2025 gold rebuild and its hard stops

**Only the 2025 slice is rebuilt. A 2021-2024 move is a HARD STOP.**

**The fingerprint ladder is three rungs, name-spaced `p31_` so it cannot overwrite the irreplaceable
Phase-30 record, and it refuses to run out of order** (a missing predecessor fingerprint raises
rather than silently starting mid-ladder).

- **Rung 0 -- the baseline.** The fingerprint of gold as it stands, before anything in this phase
  touches it.
- **Rung 1 -- a full rebuild on today's code with NO new odds,** fingerprinted. This establishes
  that the current code still reproduces the current gold. Without it, an unexpected move at rung 2
  would have two candidate causes and would be unattributable.
- **Rung 2 -- ingest the 2025 odds and run an incremental `--season 2025` MERGE,** fingerprinted, so
  every moved byte is attributable to the odds and to nothing else.

**The write mode is `replace_mode=False` (latest-wins on `game_id`, full history preserved).** SPEC
R2's phrase "rebuild only the 2025 slice in replace mode" is CORRECTED here: `save_feature_matrices`
selects its write mode from the build's scope, and its own docstring states that replace mode on a
scoped build "would write the slice AS the whole table in BOTH stores... destroying every season the
slice did not carry". The existing `_refuse_narrowing_incremental_write` guard protects the
194/195/194 width for free.

**The hard stop measures a STRICT 2021-2024 slice INCLUDING storage-level moves.** Adding 285 real
2025 rows can flip an `int64` to a `float64`, which genuinely rewrites the 2021-2024 bytes on disk
even when every value is equal. Each moved column carries a `move_kind` (`values`, `storage` or
`build_clock`) and a NON-EMPTY season attribution; a move reported with an empty season list is
exactly the case that would either trip the tripwire spuriously or slip past it, and it no longer
exists.

**The REGISTERED BUILD-CLOCK EXEMPTION.** `scripts/build_features.py:570` stamps
`datetime.now(UTC)` into `feature_timestamp` on every build, so "zero moved columns" is structurally
UNREACHABLE for a full rebuild. The reachable rung condition is therefore stated as: **zero NON-CLOCK
moves, with the moved set exactly EQUAL to the registered clock set.** The exemption is a
CLASSIFICATION, not a suppression -- a moved clock column stays in `columns_changed` and is merely
attributed elsewhere -- and an unregistered timestamp-LOOKING column that moves is asserted to land
in the non-clock set, so the classification cannot be defeated by naming a column `audit_timestamp`.

**`gate.seasons.holdout` is exactly `[2021, 2022, 2023, 2024]` and 2025 is NEVER added to it**
(D31-11). Adding 2025 would drop the single-use clean split into the deploy gate's own population and
burn it before the verdict run. A permanent guard asserts this, not a phase-local one.

**The gate baseline re-freeze runs UNCONDITIONALLY and its generated block must be BYTE-IDENTICAL to
the committed `[baseline.*]` section.** `compute_baseline` filters seasons 2021-2024, so the baseline
population cannot move on a successful run -- which makes SPEC R2's "re-freeze if gold moved" clause
dead text as literally written. Running it anyway turns a vacuous conditional into an INDEPENDENT
second proof of byte-identity, arriving through a different code path (the gate's own scoring loaders
and `per_season_clv`) than the fingerprint. **A difference IS the gold-moved signal and STOPS the
phase.**

**Neither of the two protected thresholds moves in this phase.** The gate's freshness tolerance
(`_FRESHNESS_TOL = 5e-3`, D30-DEFER-04) is NOT widened, and the O/U selection thresholds -- the EV
floor grid and the pre-hold high-total eligibility boundary -- are NOT lowered. A change to either
turns the suite red.

---

## 11. The forward-row grading contract

A forward row is the claim: "on this Friday, before these games were played, the system recommended
this bet at this size." Regenerating it from today's models would silently rewrite what was
recommended, and the tracker would then grade the system against recommendations it never made.

**A forward row's RECOMMENDATION facts are IMMUTABLE once that game's freeze has passed.** Before the
freeze the latest run wins, which is legitimate because the games have not been priced-and-frozen yet
and late odds can still land. A second Friday run after the freeze is a NO-OP.

**A forward row's GRADING facts transition EXACTLY ONCE, from `pending` to one of `win`, `loss` or
`push`.** They are the only mutable part of the row.

**Both halves are part of the frozen rule, not an implementation detail.** If the whole row were
immutable, a row could never be graded and the self-grading loop would be silently disabled -- which
is the failure this split exists to prevent. If the whole row were mutable, the recommendation could
be rewritten after the fact, which is the other failure.

**Replay rows are DERIVED and therefore fully regenerable** whenever inputs change. The split is by
provenance: `backtest_replay` regenerates, `forward` is recorded.

---

## 12. What the closing readout may restate

The R11 milestone-close readout POINTS AT the five existing readouts rather than restating their
numbers -- every restated number is a drift liability. But a blanket "restate no prior-document
number" rule is UNSATISFIABLE, because the same requirement separately demands several prior figures.
An unsatisfiable guard gets weakened during execution until it means nothing, which is how a guard
becomes decoration. So the rule is an ALLOWLIST.

**The readout may restate exactly these five, each printed beside its authoritative source:**

| Figure | Authoritative source |
|---|---|
| the per-target ABSOLUTE pooled CLV, with its t and p | `config/gate.toml` (the frozen baseline block) |
| the PREVIOUS and CURRENT ATS Kelly return figures | `31-10-SUMMARY.md` |
| the winner target's ZERO-STAKED bet count and ratio | `31-10-SUMMARY.md` |
| the standing QUARANTINED-REPRODUCTION count | `STATE-OF-SYSTEM.md` |
| every 2025 figure | `config/profitability_2025_verdict.toml` |

**Any other number in the readout is a violation, and a POINTER to the document that holds it is the
required alternative.**

**One number is FORBIDDEN outright:** `45.81043733209909`, the O/U `headline_clv`, must never appear
beside a closing-line-value label. It is the mean of the `probability_clv` column, which for O/U is
not a line-CLV at all, so publishing it as a CLV would publish a roughly 40x overstatement of a
quantity that does not exist. The readout states O/U's true `line_clv` mean instead.

**The readout's per-target template carries identical fields for all three targets:** the deployed
artifact; whether it was promoted or retained, and when; the absolute CLV-vs-market verdict; the 2025
verdict token; the number of bets selected; and a pointer to the readout holding the underlying
evidence. Omission is structurally impossible rather than something to remember.

**The readout must be able to say, without hedging, that a model is deployed, that its CLV against
the market is negative, and that the clean split selected no bets.** WP's absolute pooled CLV is
negative; the deploy gate is a NON-REGRESSION gate and never asserted a positive market edge. The
forbidden vocabulary is therefore SCOPED to hype words, not to honest negative findings.

---

## 13. What this pre-registration does NOT claim

- It does not claim any target is profitable. It fixes the rule by which that question will be
  answered once.
- It does not claim the deploy gate found a market edge. The gate is a non-regression gate.
- It does not claim the 2025 O/U result is comparable to the Phase-27 or Phase-30 publications. It
  declares two reasons why it is not.
- It does not claim the seven quarantined reproductions or the six deferred registers are closed.
  They remain open and are pointed at, not fixed.
- It does not claim any test can prove that no threshold was tuned after the numbers were seen. That
  clause is judgment-tier: no test can prove intent. What the tests prove is the commit order and the
  content hash. The rest is the owner's ratification, recorded at CHECKPOINT 1.
