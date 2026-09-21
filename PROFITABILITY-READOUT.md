# PROFITABILITY-READOUT.md -- v3.0 milestone-close per-target profitability readout

**Milestone:** v3.0 Accuracy & Profitability
**Phase:** 31 -- Ship the +EV Bet List & Profitability Readout
**Requirement:** PROD-04
**Authored:** 2026-09-06

```
pre_registration_commit: ee20773b58c3a59de2450d56c64992e240282820
measurement_commit: 01b246468f2f330c35d54e051be248f0f8994376
verdict_artifact: config/profitability_2025_verdict.toml
run_ledger: config/profitability_2025_run_ledger.toml
float_format: {:.17g}
```

> This is the honest close of v3.0. It states, per target, what is deployed, whether it was
> promoted or retained, what its absolute closing-line value against the market is, what the
> single-use 2025 clean split returned, and where the underlying evidence lives.
>
> **No target is `PROFITABLE_CLEAN`.** Two targets returned a positive raw flat-stake return that
> cannot be distinguished from zero, and one returned a measured loss. That is the result.
>
> ASCII only, no emoji (CLAUDE.md hard constraint). It is written BESIDE the five prior milestone
> documents and supersedes none of them.

---

## 0. How to read this document

### 0a. Closing-line value is NOT profitability, and this document never conflates them

Two different tests of two different quantities appear below, and they are labelled differently
every time they appear.

- A **CLV significance p-value** is the two-sided one-sample t-test that
  `backtest.diagnose.clv_significance` runs over a per-game closing-line-value array. It answers
  "is this model's line or price edge against the closing number distinguishable from zero". It is
  **REPORT-ONLY** in the 2025 chain: `clv_p_value_is_report_only = true` in the verdict artifact,
  it never entered the multiplicity family, and it is structurally incapable of driving a verdict
  token.
- A **ROI p-value** is the pre-registered one-sided achieved significance level from the
  null-recentred block-by-week bootstrap (`B = 2000`, `seed = 2704`). It answers "is this
  flat-stake return distinguishable from zero". **Every profitability statement in this document
  cites the ROI p-value by name.** No CLV figure is offered anywhere as evidence of profitability.

A reader who conflated the two would draw exactly the D25-14 and D26-09 conclusion this milestone
exists to refuse: a model with a good closing-line number and no money in it.

### 0b. The 2025 split was single-use, and it is spent

The 2025 season was the only unburned evaluation season in the project. The rule that graded it was
frozen in `PROFITABILITY-PREREGISTRATION.md` and `backtest/ev_chain_constants.py` before any 2025
number existed, at commit `ee20773`, and that commit is a strict git ancestor of the commit
recording the verdict. The run was armed by an owner ruling recorded verbatim in
`config/profitability_2025_run_ledger.toml`, executed once, and the ledger is now `completed`.
There is no force flag and no second attempt. **Nothing in this document can be re-run to check
it**; what can be checked is that every figure here matches the committed verdict artifact, which
is what `tests/unit/test_profitability_readout_md.py` does.

### 0c. What this document restates, and what it only points at

Every restated number is a future drift liability, so this readout POINTS AT the five prior
milestone documents rather than copying them. The exceptions are the five categories named in
`backtest.ev_chain_constants.READOUT_PERMITTED_FIGURES`, each printed beside its authoritative
source, plus the Phase-31 disclosure figures whose tracked witness is `tests/phase31_state.py`.
Any other number here would be a drift liability, and the guard rejects one.

---

## 1. The verdict, per target

Every target fills the SAME seven template slots. Omission is structurally impossible rather than
something to remember, and the drift guard asserts field presence directly. A target that selected
no bets would fill these slots too -- `UNDISCHARGEABLE_NO_BETS` is a first-class result in the
frozen vocabulary, not a gap.

Source for every 2025 figure below: `config/profitability_2025_verdict.toml`.

### 1a. Target `wp` (win probability)

- `deployed_artifact`: `wp_20260824_113325`
- `promoted_or_retained`: PROMOTED
- `promoted_or_retained_when`: 2026-08-24, at the Phase-30 per-target deploy gate (PROD-01). It
  replaced the Phase-25 re-fit, which had itself replaced the v1.0 pre-Elo model.
- `absolute_clv_verdict`: NEGATIVE. The frozen gate baseline records the deployed model's pooled
  absolute probability-CLV over the 2021-2024 holdout as mean `-0.03800034`, t `-15.52461299`,
  p `0.00000000`, ci95 [`-0.04280319`, `-0.03319749`] (source: the `[baseline.wp.pooled]` block of
  `config/gate.toml`). That is a significantly negative closing-line value against the market. The
  deploy gate is a NON-REGRESSION gate and never asserted a positive market edge -- see section 4.
- `verdict_token_2025`: **`INCONCLUSIVE_CLEAN`**
- `bets_selected`: 65 (positive control: 284 bets)
- `evidence_pointer`: `GATED-REFIT-READOUT.md` section 6 ("The per-target deploy outcome: one
  promotion, two refusals") for the promotion; `ACTIVATION-READOUT.md` section 3 ("Per-target CLV
  before/after") for the Phase-25 predecessor it replaced.

2025 measurement, from the committed verdict artifact:

| quantity | value |
|---|---|
| flat-stake ROI | `0.014408465573452605` |
| **ROI p-value, raw** | `0.41179410294852575` |
| **ROI p-value, BH-adjusted** | `0.671664167916042` |
| bootstrap ci95 | [`-0.23718108428651416`, `0.21400669330762565`] |
| weekly blocks | 20 |
| pushes / ungraded | 0 / 0 |
| robustness cut ROI (regular season only, playoffs excluded) | `0.0073879941712662201` |
| registered fallback fired | no |

**Reading:** a positive raw return whose pre-registered ROI p-value of `0.671664167916042` does not
clear alpha `0.050000000000000003`. A positive return that cannot be distinguished from zero is
INCONCLUSIVE and is never called profitable.

### 1b. Target `ats` (against the spread)

- `deployed_artifact`: `ats_20260605_220128`
- `promoted_or_retained`: RETAINED
- `promoted_or_retained_when`: RETAINED on 2026-08-24 when the Phase-30 gate REFUSED its candidate
  re-fit. The artifact itself was promoted earlier, on 2026-06-05, at the Phase-25 gate.
- `absolute_clv_verdict`: NOT DISTINGUISHABLE FROM ZERO, and negative in sign. The frozen gate
  baseline records pooled absolute line-CLV mean `-0.00149507`, t `-0.01222238`, p `0.99025044`,
  ci95 [`-0.24151011`, `0.23851997`] (source: the `[baseline.ats.pooled]` block of
  `config/gate.toml`). The CLV significance p-value there is far from alpha, so this is a
  no-detectable-edge reading rather than a significantly negative one.
- `verdict_token_2025`: **`UNPROFITABLE_CLEAN`**
- `bets_selected`: 161 (positive control: 285 bets)
- `evidence_pointer`: `GATED-REFIT-READOUT.md` section 6 for the Phase-30 refusal and the retention;
  `ACTIVATION-READOUT.md` section 3 for the Phase-25 activation that put this artifact in place.

2025 measurement, from the committed verdict artifact:

| quantity | value |
|---|---|
| flat-stake ROI | `-0.052760703432021903` |
| **ROI p-value, raw** | `0.76061969015492259` |
| **ROI p-value, BH-adjusted** | `0.76061969015492259` |
| bootstrap ci95 | [`-0.20312498771456264`, `0.0863424646731277`] |
| weekly blocks | 21 |
| pushes / ungraded | 7 / 0 |
| robustness cut ROI (regular season only, playoffs excluded) | `-0.05931816131756238` |
| registered fallback fired | no |

**Reading:** a measured LOSS. The token is `UNPROFITABLE_CLEAN` because the observed flat-stake
return is not strictly positive; that is a measured result, not a failure to measure. The
robustness cut is worse than the primary, not better, so the loss is not an artefact of the
playoff rows.

### 1c. Target `ou` (over / under)

- `deployed_artifact`: `ou_20260326_163930`
- `promoted_or_retained`: RETAINED. This is the v1.0 pre-Elo model, and it is the only target whose
  deployed artifact predates v2.0.
- `promoted_or_retained_when`: RETAINED on 2026-08-24 when the Phase-30 gate REFUSED its candidate
  re-fit, and RETAINED before that on 2026-06-06 when the Phase-25 gate refused an earlier one. Two
  refusals, both recorded.
- `absolute_clv_verdict`: POSITIVE and significant. The frozen gate baseline records pooled
  absolute line-CLV mean `1.09908061`, t `16.62961883`, p `0.00000000`, ci95 [`0.96939863`,
  `1.22876259`] (source: the `[baseline.ou.pooled]` block of `config/gate.toml`). This is the one
  target with a positive, significant closing-line value against the market -- and, as section 2
  says, it is also the target for which this project first measured that a positive CLV need not
  become money.
- `verdict_token_2025`: **`INCONCLUSIVE_CLEAN`**
- `bets_selected`: 41 (positive control: 285 bets)
- `evidence_pointer`: `GATED-REFIT-READOUT.md` section 7b ("The full re-run, beside the Phase-27
  publication"), which deferred the binding clean profitability verdict to this phase;
  `OU-DIVERGENCE-DIAGNOSIS.md` section 8 ("Verdict") for the Phase-26 go decision that scoped the
  sub-population this chain selects on.

2025 measurement, from the committed verdict artifact:

| quantity | value |
|---|---|
| flat-stake ROI | `0.032513056638189181` |
| **ROI p-value, raw** | `0.42128935532233885` |
| **ROI p-value, BH-adjusted** | `0.671664167916042` |
| bootstrap ci95 | [`-0.25091580596336027`, `0.28807994281619181`] |
| weekly blocks | 16 |
| pushes / ungraded | 2 / 0 |
| robustness cut ROI (regular season only, playoffs excluded) | `0.032513056638189181` |
| registered fallback fired | no |

**Reading:** a positive raw return whose pre-registered ROI p-value of `0.671664167916042` does not
clear alpha. INCONCLUSIVE. The robustness cut is byte-identical to the primary because no selected
O/U bet fell in a playoff week.

**This discharges the deferral.** `GATED-REFIT-READOUT.md` section 7b states that Phase 27 accepted
its provisional contaminated readout as a proceed signal and deferred the binding clean
profitability verdict, that Phase 30 could not discharge it because its hold was still the
partially burned 2023-2024, and that the binding clean verdict therefore carries to Phase 31. It is
discharged here, on the clean 2025 split, and the answer is `INCONCLUSIVE_CLEAN`.

---

## 2. The headline finding is the divergence, not the return

**The win-probability model shows strongly significant POSITIVE closing-line value on its own
selected 2025 bets while its return is indistinguishable from zero.**

CLV, measured over each target's selected 2025 bets and carried BESIDE the verdict as REPORT-ONLY
(source: `config/profitability_2025_verdict.toml`):

| target | CLV mean | CLV significance p | n bets |
|---|---|---|---|
| wp | `0.091811498994504193` | `6.8263173184833947e-34` | 65 |
| ats | `-3.4284412375657083` | `1.3743115504257695e-26` | 161 |
| ou | `-0.52473998651271914` | `0.19925181065524544` | 41 |

Each p-value in that table is a **CLV significance** p-value. None of them is evidence about
profitability, and none of them moved a verdict token.

Set beside section 1: `wp` has a CLV significance p of `6.8263173184833947e-34` on a positive mean,
and an ROI p-value of `0.671664167916042` on a return of `0.014408465573452605`. The model is
beating the closing number, decisively, on the games it chooses to bet -- and that is not turning
into a distinguishable return at this sample size.

This is the CLV-to-ROI divergence Phase 26 diagnosed for O/U (`OU-DIVERGENCE-DIAGNOSIS.md`), now
measured on a clean out-of-sample split for the win-probability target. **It is stated here as a
finding.** It is not a profitability claim, and it does not soften the spread target's measured
loss, which sits in the same table with a strongly significant NEGATIVE CLV alongside a negative
return.

The `ats` row deserves one sentence of its own: its CLV significance p is
`1.3743115504257695e-26` on a mean of `-3.4284412375657083`. The spread model is losing to the
closing number decisively AND losing money. Those two agree, which is the ordinary case; `wp` is
the interesting one precisely because they do not.

---

## 3. The multiplicity family, the registry, and what "clean" means

From the run block of `config/profitability_2025_verdict.toml`:

- `alpha` = `0.050000000000000003` (imported by identity from `backtest.diagnose.SIGNIFICANCE_ALPHA`;
  the comparison is strictly less than, so a value exactly at alpha is not significant).
- `correction_method` = benjamini-hochberg, `bh_denominator` = 6.
- `bh_denominator_rule`: hold-side inferences only, pooled across all three targets in ONE registry.
  Six members: a primary and a robustness entry for each of the three targets.
- `bh_fallbacks_fired` = 0. The registered calibration fallback was available and did not fire; the
  win-probability tune-side calibration gate passed, so the family is the no-fallback arm.
- `registry_rows` = 24, `excluded_control_entries` = 3, `excluded_tune_side_sweep_cells` = 15.
  24 = 6 family + 3 controls + 15 tune-side sweep cells. **The registry closes.** Every read of the
  hold is on the record; the denominator counts only actual hold-side inferences, because a
  tune-side sweep cell never touched 2025 and a counterfactual control has no p-value to correct.
- `roi_min_attainable_p` = `0.00049975012493753122` -- the floor a 2000-resample bootstrap can
  attain. No target came near it, so no verdict here rests on the bootstrap's resolution limit.
- All three positive controls PASSED (`control_passed = true` on every target), so a null verdict
  here is a null result rather than a broken chain.
- `robustness_cut_bh_count` = 1: the two declared cuts (`regular_season_only`,
  `playoffs_excluded`) produced byte-identical frames and are counted once.

---

## 4. What is deployed, and what the deploy gate does and does not assert

| target | deployed artifact | how it got there |
|---|---|---|
| wp | `wp_20260824_113325` | promoted 2026-08-24 (Phase 30) |
| ats | `ats_20260605_220128` | promoted 2026-06-05 (Phase 25), retained 2026-08-24 (Phase 30 refusal) |
| ou | `ou_20260326_163930` | v1.0 pre-Elo, retained twice (Phase 25 and Phase 30 refusals) |
| blend | `blend_dynamic_20260606_020635` | live for all three targets, unchanged since Phase 26 |

**Phase 31 deployed nothing.** No model was re-fit, no artifact promoted, no blend changed, and
`artifacts/latest.json` is byte-unchanged by this phase. The production swap surface was not
touched.

**The deploy gate is a NON-REGRESSION gate.** It asks whether a candidate is significantly WORSE
than the deployed incumbent on a paired per-game CLV delta. It has never asserted that any deployed
model has a positive edge against the market, and section 1a states the measured consequence
plainly: the deployed win-probability model's absolute pooled closing-line value is
`-0.03800034` with a CLV significance p of `0.00000000` (source: the `[baseline.wp.pooled]` block
of `config/gate.toml`) -- significantly negative. A model can pass this gate and still be behind
the market in absolute terms, and one does.

---

## 5. Declared discontinuities and narrowings

An unstated narrowing is the quiet form of moving a goalpost, so both of these are stated.

### 5a. The 2025 totals result is NOT continuous with the two prior publications, for TWO reasons

The O/U figure in section 1c cannot be compared like-for-like against the Phase-27 or Phase-30 O/U
publications in `GATED-REFIT-READOUT.md` section 7b. Two separate changes to the population are
responsible, and both must be named:

1. **The tune window was widened.** D31-13 made the tune window a uniform 2021-2024 for all three
   targets, overriding the recommended per-target split. The Phase-27 and Phase-30 chains fit on a
   narrower window.
2. **Playoff rows now sit in the tune population.** D31-38 made the odds population playoffs
   everywhere -- ingested and carried in BOTH the tune window and the hold. The 2025 hold is 285
   games, not 272, and the tune seasons gained playoff rows that the prior publications never saw.

One declared discontinuity with two causes. Neither is a defect; both change what the number is a
measurement of.

### 5b. "The 2023-2024 hold stays burned" now means NOT EVALUATED, not NOT TOUCHED

Under D31-13, 2023 and 2024 are inside the tune window. They feed the frozen residual SD, the
prior-season bias estimate and the EV-floor sweep. **No figure in this document is evaluated on
them and no verdict is drawn from them**, but they are no longer untouched, and the earlier phrasing
implied that they were. Burnedness compromises evaluation, not fitting; leaving two seasons unused
as a nuisance-parameter source would have wasted estimation data out of ceremony. The narrowing is
recorded here rather than left to be discovered by someone comparing phrasings across phases.

---

## 6. Disclosures the owner accepted, in full

Nine disclosures were put to the owner at CHECKPOINT 2 (2026-09-05) and CHECKPOINT 3 (the arming
ruling, 2026-09-05, recorded verbatim in `config/profitability_2025_run_ledger.toml`). Each carried
an obligation on this document. They are discharged below. The tracked machine-readable witness for
the CHECKPOINT-2 set is `tests/phase31_state.py`, because `.planning/` does not survive a fresh
checkout.

### 6a. 68 protected-window games were graded against a FABRICATED 0.0 market line

This is stated in those words deliberately. It is not "a data correction was applied".

Gold keys the Los Angeles Rams canonically as `LA`; silver had stored 116 of their odds rows under
`LAR`. Those rows were ORPHANS against gold, so those games fell through to
`_default_compressed_market_features` and gold's imputation wrote **a literal `0.0` market line**.
`scripts/build_features.py` then computed `target_ats = point_differential - snapshot_spread` and
`target_ou = total_points - snapshot_total` against that fabricated zero. **68 games inside the
protected 2021-2024 holdout -- 17 per season -- therefore carried a fabricated zero market line and
a correspondingly WRONG ATS and O/U label.** Across all seasons 136 games were affected, every one
of them a Rams game.

The ratified clause-5 `normalize_stored_game_ids` re-keying corrected them. The scoped 2025 build
could not reach 2018-2024, so the correction sat in silver until the full-scope rebuild of
2026-09-05, which is the build in which it landed. Eleven column-slots moved in the protected
slice: `snapshot_spread`, `snapshot_total` and `snapshot_ml_prob_home_fair` in all three matrices,
plus `target_ats` in `features_ats` and `target_ou` in `features_ou`.

**The corrected labels are what the 2025 verdict's baseline comparison now rests on.**

**The owner ACCEPTED this at CHECKPOINT 2 on 2026-09-05**, in these words: "The corrected labels
are the true ones, and grading a holdout against fabricated zero lines was never defensible." The
decision, the option space and the independent re-verification against live gold are recorded in
`tests/phase31_state.py` as `CHECKPOINT_2_DECISION` and `PROTECTED_SLICE_CORRECTION`. Two
independent instruments agree on the move: the Phase-31 attribution control and the Phase-30
`tests/integration/test_n01_resync_control.py`, which this phase did not write.

### 6b. The frozen gate baseline diverges from a re-score in 47 of 68 fields, and was deliberately NOT re-frozen

Running `scripts.freeze_gate_baseline` against current gold produces a block that differs from the
committed block in **47 of 68 fields**, with 21 identical. Before the full rebuild it was 44 of 68.
**All twelve per-season sample sizes are unchanged**, so the scored population did not move -- only
the values did. The pooled means, recorded in `tests/phase31_state.py` as
`GATE_BASELINE_DIVERGENCE`: `ats.pooled.mean` committed `-0.00149507` against regenerated
`+0.01362779`; `ou.pooled.mean` committed `1.09908061` against regenerated `1.09091962`;
`wp.pooled.mean` committed `-0.03800034` against regenerated `-0.03892902`.

**`config/gate.toml` is byte-unchanged and the baseline was NOT re-frozen.** Option 3 at
CHECKPOINT 2 -- accept and re-freeze -- was offered and declined.

**Why that is a disclosure rather than an omission.** The gate reads the COMMITTED baseline block,
never a re-score. The frozen block is therefore still the non-regression reference the verdict is
judged against, exactly as it was when the phase started. Re-freezing would move the reference to
match the data being judged, which is the one thing a non-regression baseline must never do. The
holdout stays exactly 2021-2024 and 2025 is absent from it, so the single clean split was still
unburned when it was spent.

A readout that reported the gate verdict without disclosing that a re-score of its baseline returns
different numbers would not be honest about what the verdict was measured against.

### 6c. The ratified ATS residual constants no longer re-derive

The frozen pre-registration fixes the ATS residual bias constants. Re-measuring the same deployed
artifact `ats_20260605_220128` over current gold no longer returns them. **Both sets are stated
here, because a readout that states only one does not discharge the obligation.** The live set is
recorded in `tests/phase31_state.py` as `ATS_RESIDUAL_LIVE_UPSTREAM_ONLY` and
`ATS_RESIDUAL_LIVE_AFTER_FULL_REBUILD`; the ratified set is recorded there as
`ATS_RESIDUAL_BY_SEASON` and in the frozen `backtest/ev_chain_constants.py` as
`ATS_RESIDUAL_BY_SEASON_P31`.

| season | RATIFIED (frozen; the rule the verdict was computed with) | LIVE, upstream cause only | LIVE now, after the full rebuild |
|---|---|---|---|
| 2021 | `0.68147702779163399` | `0.68443905065457022` | `0.63968358671194625` |
| 2022 | `-0.03539119799896865` | `-0.03170638450119697` | `0.046538869338765949` |
| 2023 | `0.58219338768864415` | `0.5905364753743797` | `0.62408175513867226` |
| 2024 | `1.1270376943443952` | `1.1275886151874275` | `1.1071011105258213` |
| **pooled** | **`0.58937727047262922`** | **`0.59326265763681096`** | **`0.60484106920061009`** |

**Every per-season sample size is unchanged across all three columns** -- 285 / 284 / 285 / 285,
pooled n 1139. The scored population is identical; only the values moved.

**Two separately-attributable causes, and conflating them would misreport this.**

1. **Upstream nflverse revision.** nflverse re-released play-by-play and depth-chart data between
   the ratifying measurement and 2026-09-05. Sixteen gold columns moved, as the Phase-30 control
   `tests/integration/test_n01_resync_control.py` independently reports against its own pre-resync
   digests. `data/upstream_pin.py` and `config/upstream_pin.json` stop this RECURRING; they cannot
   undo it, because the revision the constants were measured on is gone from upstream. This cause
   predates this phase.
2. **The ratified clause-5 key normalization reaching gold** -- the same correction section 6a
   describes. The full rebuild of 2026-09-05 is the first build to re-derive 2021-2024 since the
   normalization.

**What was NOT done:** no constant was re-derived, no baseline re-frozen, no tolerance widened, no
assertion weakened or deleted. The frozen constants ARE the rule, the chain reads them from the
frozen module, and the verdict in section 1 was computed with the ratified numbers. What was lost
is the RE-DERIVABILITY of those numbers, and that is a disclosure item rather than a tampering one.

**The pooled direction claim still holds** -- strictly positive, the model under-predicts -- on
every column, and is in fact stronger after the rebuild.

### 6d. The 2022 ATS residual SIGN FLIP contradicts the pre-registration's own prose

Called out separately because it is a statement of record that live gold no longer supports.

`PROFITABILITY-PREREGISTRATION.md` asserts in prose that the 2022 ATS residual season mean is
NEGATIVE, and the ratified constant is `-0.03539119799896865` (recorded in `tests/phase31_state.py`
as `ATS_RESIDUAL_BY_SEASON`). After the full rebuild the live re-score returns
`0.046538869338765949` (recorded there as `ATS_RESIDUAL_LIVE_AFTER_FULL_REBUILD`, with the whole
disclosure carried in `ATS_RESIDUAL_DRIFT_PROVENANCE`), so `measure_ats_residual_bias` now reports
`seasons_with_negative_mean == []` where it reported `[2022]`. The frozen document says the
negative thing in as many words, and it is no longer what a re-measurement shows.

The owner accepted this on the same basis as 6c: a disclosure, not a tampering. **The pooled
DIRECTION claim -- strictly positive -- still holds.** No gate verdict turns on the flip:
per-season sign was never gated, by design, precisely so a real negative season could be reported
rather than hidden behind a gate.

Six constant controls in `tests/integration/test_ingest_2025_odds.py` are marked
`xfail(strict=True)` with every assertion preserved byte for byte. `strict=True` is the point: if
the live values ever return to the ratified ones those become loud unexpected passes, forcing the
register entries CLOSED rather than quietly outlived.

### 6e. Phase 30's binding feature-group verdicts rest on superseded gold

Phase 30 (PROD-01) ruled `snap` KEPT, `situational` KEPT and `injury` DROPPED. **Every one of those
verdicts was measured on gold that still contained the fabricated 0.0 Rams market lines**, and
therefore on the same wrong ATS and O/U labels for those 68 protected-window games.

Re-running the committed Phase-28 signal-lift harness on the corrected gold returns a different
RULING, not merely a different point estimate: **DROP for all three groups**, each on a D-05 veto.
The situational-OU delta is `-0.3203552582994336` (source: `SIGNAL-LIFT-READOUT.md` section 0b,
which carries the dated 2026-09-05 drift record and the full grid; the 2026-06-29 anchor is left
standing beside it).

**Nothing was retrained, so `snap` and `situational` remain in the production models.** They are
included at TRAIN time in the deployed artifacts named in section 4. The verdicts are not shown to
be wrong; they are shown to rest on an input that has since been corrected.

**They are REVISITABLE without a gold rebuild**, on the same footing `injury`'s twelve columns
already have: physically present in gold, and re-rulable by a gate re-run on the gold that already
exists -- no rebuild, no re-ingest, no new data.

**Re-running the Phase-30 gate was considered and DECLINED** by the owner on 2026-09-05 as outside
Phase 31's scope. That is recorded so the absence of a gate re-run is not later read as an
oversight.

### 6f. ATS and O/U were priced on devigged real two-sided juice, and an earlier build priced them flat

Two ratified documents disagreed about the price. `PROFITABILITY-PREREGISTRATION.md` sections 3.2
and 3.3 step 4 say the ATS chain devigs the real two-sided spread prices (`spread_ju_home`,
`spread_ju_away`) and the O/U chain devigs the real two-sided total prices (`total_over_ju`,
`total_under_ju`). D31-04 called those two targets flat-quoted, and an earlier build priced both at
flat -110 on the selection path.

**The owner ruled on 2026-09-05 that the frozen pre-registration governs.** A reader comparing any
Phase-31 figure published before that date to one published after must be able to see that a price
convention changed between them, which is why this paragraph exists. D31-04's "flat-quoted"
characterisation is superseded for the selection path only; the rest of it stands.

**The realized split.** Zero bets in this verdict priced at the flat fallback. Every ATS and every
O/U bet was struck at `real_two_sided`. The win-probability target declares no `devig_method` at
all: it prices at the game's own two-sided moneyline (`ml_home` / `ml_away`) and never reaches a
fallback.

**How that split was established, stated plainly rather than implied.** No per-bet `devig_method`
value is persisted anywhere -- the decision records live only in memory during the run. The split
is therefore derived from the 2025 juice-coverage record written before the run
(`outputs/p31/p31_13b_devig_coverage_after.json`): coverage is COMPLETE for 2025, all four juice
columns present on every one of the 285 rows, so no selected bet could have lacked the juice and
the split follows deterministically. **This is a coverage fact about the silver store, not a second
read of the 2025 hold.** Re-running the selection to count the split directly would have been
exactly that second read, and it was not done.

**The historical promote that made this possible.** The pre-registration's section 4.2 promote of
the four juice columns from the partitioned store into the flat table had been ratified and never
run. It was executed on 2026-09-05 under a second owner ruling, before the hold was armed. Every
one of the 2140 stored odds rows now carries juice, where 285 did before. It had to land BEFORE the
run: with the devig ruling live but the promote missing, the EV floor would have been selected in
one pricing regime on the tune window and applied in another on the hold. Gold did not move --
demonstrated twice, by a byte-identical fingerprint document and by a content-digest boundary check
over every file under `data/`.

### 6g. Three register entries that were CLOSED, one line each

- **The stored spread sign convention was the OPPOSITE of the one the ATS chain assumed.** The
  stored `spread` is on the nflverse `spread_line` convention -- positive when the home team is
  favored. The owner ruled on 2026-09-04 and the fix landed the same day; the WP and O/U arms were
  never affected, and the ratified residual contract remains correct under the measured convention.
- **A live current-week ATS edge sign defect** in `models/blending.py` and
  `scripts/generate_current_week_predictions.py`, the same negation, was closed in this phase.
- **A whole-site 500** -- the base template read `cache_meta.last_updated` unguarded, so a
  cache_meta carrying rows but no `last_updated` took down every page -- was closed in this phase.

---

## 7. The betting page, and the spread Kelly figure that changed

### 7a. The previous spread Kelly return, why it was wrong, and what it reads now

**The previous figure was +1.37%** (source:
`.planning/phases/31-ship-the-ev-bet-list-profitability-readout/31-10-SUMMARY.md`, "The figures
this plan changes, for the R11 readout"). Removing a wrong number quietly would be its own
dishonesty, so it is stated here with its cause.

**Why it was wrong.** The legacy simulator branch handed Kelly a points-distance quantity,
`implied + abs(model_spread - closing_spread)`, in the argument slot that expects a probability.
The split it produced was perfectly separating in the wrong direction: Kelly staked only bets where
the model disagreed with the market by LESS than about half a point, and refused every bet with a
larger disagreement. 725 of 1073 spread bets were zero-staked and 348 staked -- a 67.6% zero-staked
share -- and the published +1.37% was computed over that inverted selection.

**What the code reads now.** The spread Kelly stake on the un-injected path is zero by design, with
the reason written at the branch, and a class-level guard now makes it structurally impossible to
pass a points-distance quantity where a Kelly probability is expected: a boundary value raises
rather than silently returning a tidy zero stake.

### 7b. What `/betting` actually serves today, which is NOT what the fix implies

This correction matters and it was measured for this document rather than inherited.

**`outputs/backtest/betting_simulation.csv` -- the ledger `/betting` reads -- has not been
regenerated since 2026-08-24 16:44, before Phase 31 began and before the Kelly fix landed.** Its
sha256 is `61f73c17ed1a118fd619a513fbf389d9378e9cbcc53a0b52e39cb331309624b1`. Measured directly
against that file: the spread rows still carry a large non-zero Kelly stake column and the page's
`roi_kelly` over them still computes to the pre-fix +1.37% published in
`.planning/phases/31-ship-the-ev-bet-list-profitability-readout/31-10-SUMMARY.md`. **It is the
totals rows, not the spread rows, whose Kelly column reads zero** -- and they read zero for a
different and older reason, the Phase-27 zeroing, not this phase's fix.

So the accurate statement is: **the code path was fixed, and the published artifact was never
regenerated, so `/betting` still serves the pre-fix spread Kelly figure.** A regeneration would move
it to zero. It has not been regenerated, deliberately -- D31-04 pins `/betting`'s published figures
and forbids moving them inside this phase.

One consequence worth naming, because it makes several claims in this phase weaker than they
sounded: the repeated "the artifact is unmoved, sha256 `61f73c17...`" verifications made during this
phase were comparing a file that nothing in the phase regenerates. They are true, and they prove
less than they appear to.

### 7c. The winner target's zero-staked ratio is a LEGITIMATE no-edge Kelly zero, not the same defect

888 of 1032 winner bets are zero-staked -- an 86.1% share, HIGHER than the spread target's 67.6%
(source: the same section of
`.planning/phases/31-ship-the-ev-bet-list-profitability-readout/31-10-SUMMARY.md`). A reader
comparing the two ratios could easily conclude a second target was missed. It was not.

Every winner `model_value` in that ledger is strictly inside the open unit interval, and the split
is clean at the calculator's own confidence threshold: the largest true edge among the zero-staked
winner bets is `0.019893` and the smallest among the staked ones is `0.020245`. Not one zero-staked
winner bet has an edge above the threshold. These are ordinary no-edge Kelly refusals on a genuine
probability.

### 7d. Three defects on the web surface that are pointed at, not fixed

- **One threshold pair applied to three incompatible units.** The edge band was de-duplicated into
  one source this phase and RENAMED -- not repaired. It maps the same two thresholds onto a
  probability delta (`wp_edge`), a fraction of the absolute spread (`ats_edge`) and a fraction of
  the market total (`ou_edge`). A "high" win-probability edge and a "high" spread edge are not
  comparable quantities. The collapse itself is behaviour-preserving, asserted value by value
  against a snapshot taken from both retired helpers before the change. The stored and rendered
  column names still say `confidence` although the value is an edge band.
- **The backtest-side `ats_edge` in `api/cache.py` carries the same sign defect that was fixed
  elsewhere in this phase**, and it is knowingly left in place. It feeds a band that `/` and
  `/betting` render and sort by, and D31-04 pins that published population. Correcting the sign
  would move the label on a large share of rows, which is a decision this phase did not have.
- **A bet-list row's `clv` is the DECISION-TIME model-edge CLV, never a forward freeze-vs-close
  CLV.** The forward metric is **not computable** from the current store: the silver odds table
  holds exactly one snapshot per game, and a freeze-vs-close CLV needs two observations of the same
  line. The value is carried honestly under its true meaning and the gap is registered. Fabricating
  a number there, or reporting the decision-time value under the forward label, would be precisely
  the mislabel this phase exists to prevent.

### 7e. One stylesheet gap, corrected from an earlier claim in this phase

An earlier report in this phase recorded two Tailwind utility classes as present in the compiled
stylesheet. Only one is. `max-w-3xl` is present; **`lg:grid-cols-7` is ABSENT** from
`web/static/css/tailwind-compiled.css`, so `/betting`'s KPI grid is still unstyled at the large
breakpoint. Pre-existing and out of scope to fix here, but it is not resolved and is not recorded
as resolved.

---

## 8. Five tests are deliberately RED on a shipped milestone

This is honesty of record, not an embarrassment to hide. Each of these is a tripwire that FIRED on
a real event the owner then accepted. A tripwire rewritten to accept the event it fired on stops
being a tripwire, so the assertions are preserved byte for byte and the red is carried instead.

| test | what it detects | why it stays red |
|---|---|---|
| `tests/integration/test_gate_baseline_byte_identity.py::TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::test_the_generated_block_equals_the_committed_block_byte_for_byte` | the committed gate baseline no longer equals a re-score | section 6b: the baseline was deliberately NOT re-frozen |
| `tests/unit/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields` | the same divergence, field by field | section 6b, same reason |
| `tests/integration/test_gold_rebuild_attribution.py::TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::test_no_NON_CLOCK_column_moved_in_a_protected_season` | a non-clock column moved inside the protected 2021-2024 window | section 6a: the fabricated-zero correction moved it, and the owner accepted the move |
| `tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical` (two cases) | the protected slice is not byte-identical to its pre-resync digests | section 6a: the same correction, reported by a Phase-30 instrument this phase did not write |

**The gate baseline predates the label correction, and the protected slice moved.** Those two facts
are what the five reds encode. Making them green would require either re-freezing the baseline
against the data it judges, or editing a control to accept the event it detected. Both were
available and both were refused.

---

## 9. What is still open

Pointed at, not fixed, and not claimed closed.

- **Seven quarantined reproductions remain OPEN**, and **six deferred registers remain OPEN**
  (source: `STATE-OF-SYSTEM.md`, the single consolidated open list). The suite's standing
  `7 xfailed` for the Phase-30 quarantine set is the mechanical proof; the whole-suite xfailed
  count is higher because this phase added the constant controls of section 6d on top of it.
- **The renamed edge band still applies one threshold pair to three incompatible units**
  (section 7d).
- **The prediction cache's swap is an availability gap**: it unlinks the destination immediately
  before renaming the new cache in, so a crash inside that window leaves no cache rather than a
  mixed one.
- **The gate's freshness tolerance is one absolute band across two different metrics.** It was not
  widened and must not be widened to make a drift go away.
- **The backtest-side `ats_edge` sign defect** and **the non-computable forward CLV** (section 7d).
- **Phase 30's `snap` and `situational` verdicts are revisitable** and have not been revisited
  (section 6e).
- **`lg:grid-cols-7` is missing from the compiled stylesheet** (section 7e).

---

## 10. What this readout is, and is not

**It is** the per-target close of v3.0: what is deployed, what was retained, the absolute
closing-line value against the market per target, and the single clean out-of-sample profitability
verdict per target under a rule frozen before any of the numbers existed.

**It is not** a profitability claim. No target is `PROFITABLE_CLEAN`. Two targets returned a raw
positive flat-stake return that the pre-registered ROI test cannot distinguish from zero, and one
returned a measured loss. The one strongly significant result in the whole document is a
closing-line-value result, and closing-line value is not profitability -- which is the finding of
section 2 and the reason section 0a exists.

**It is not** a statement that the system should be bet. Nothing here establishes an edge that
survives its own significance test.

**It does not supersede any prior document.** `ACTIVATION-READOUT.md`, `SIGNAL-LIFT-READOUT.md`,
`LINE-MOVEMENT-READOUT.md`, `OU-DIVERGENCE-DIAGNOSIS.md` and `GATED-REFIT-READOUT.md` stand exactly
as written. This readout points at them; their content hashes are pinned by
`tests/unit/test_profitability_readout_md.py` so that an overwrite fails a test rather than passing
unnoticed.

---

## Cross-references

- **`config/profitability_2025_verdict.toml`** -- the committed 2025 verdict. The single source of
  every 2025 figure in this document.
- **`config/profitability_2025_run_ledger.toml`** -- the durable exclusive one-shot ledger, holding
  the owner's arming ruling verbatim.
- **`PROFITABILITY-PREREGISTRATION.md`** and **`backtest/ev_chain_constants.py`** -- the frozen
  rule, in two files, anchored at commit `ee20773`.
- **`tests/phase31_state.py`** -- the tracked witness for the CHECKPOINT-2 decision, the protected
  slice correction, the gate-baseline divergence, the ATS residual drift, the pre-registration
  digests and the measurement commit.
- **`GATED-REFIT-READOUT.md`** -- Phase 30: the four-rung rebuild, the feature-group verdicts, the
  per-target deploy outcome (section 6), and the O/U deferral this document discharges (section 7b).
  Section 7c also records the O/U reading hazard: the backtest export's O/U `headline_clv` is not a
  CLV at all, and O/U's actual line-CLV mean is `+1.8954569`.
- **`ACTIVATION-READOUT.md`** -- Phase 25: the first gated activation, per-target CLV before and
  after (section 3), and the deployed/retained matrix (section 4).
- **`SIGNAL-LIFT-READOUT.md`** -- Phase 28: the feature-group screen, and section 0b's dated
  2026-09-05 drift record that section 6e above rests on.
- **`OU-DIVERGENCE-DIAGNOSIS.md`** -- Phase 26: the CLV-to-ROI divergence first measured, and the
  owner's go verdict (section 8).
- **`LINE-MOVEMENT-READOUT.md`** -- Phase 29: the line-movement screen and its DROP.
- **`STATE-OF-SYSTEM.md`** -- the single consolidated open list.
- **`METHODOLOGY.md`**, **`RUNBOOK.md`**, **`PIPELINE.md`**, **`AUTOMATION.md`**, **`README.md`** --
  the operator and project documents, reconciled to this end state.

<!-- old-rule-addendum-2026-09-15 -->

## Old-rule addendum (2026-09-15)

This section was added on 2026-09-15. Nothing above it has been changed: every number, table and
heading is exactly as it was first published.

Phase 33.2 found that the model inputs behind the results in this document were defective, in five
ways. Weather observed after each game stood in for the forecast that was actually available the day
before kickoff. Closing betting lines, which are only known at kickoff, were fed into the models as
inputs. Feature builders took their cutoff from one global clock instead of each game's own lock
time. The opponent adjustment never actually ran. Early-season placeholder values read as exactly
league average, with nothing to say they were placeholders.

This document stays in the record, unedited, because deleting it would be worse: it would hide what
was claimed and when. Read it as history, not as a measure of how well the system works.

**Built under the old rule on inputs later found defective; not evidence.** Only the 2026 season,
recorded live under the day-before 6 PM ET lock, counts (D33.2-07). See Phase 33.2.

What this covers in this document: the 2025 one-shot verdicts and the CLV-to-ROI divergence finding are both old-rule.
