# HISTORICAL-WEATHER-READOUT.md -- Phase 33.1 Real Historical Weather and Training Window Correction (R8)

**Milestone:** v4.0 Live Season Operations
**Phase:** 33.1 -- Real Historical Weather and Training Window Correction (a mid-phase insertion into Phase 33)
**Authored:** 2026-09-14
**Status:** committed (repo root, sibling of `GATED-REFIT-READOUT.md`, `ACTIVATION-READOUT.md`,
`SIGNAL-LIFT-READOUT.md`, `LINE-MOVEMENT-READOUT.md` and `PROFITABILITY-READOUT.md`)

> This is the point-in-time handback record of Phase 33.1: what the weather correction actually
> moved, what the cross-check measured against a prediction frozen before the data existed, what
> training-window rule replaced nineteen scattered season literals, and -- the reason this document
> exists -- which specific instructions in Phase 33's own Wave 14 and Wave 15 plans the corrected
> inputs have invalidated.
>
> **A null result is a complete result.** This document is published whether the weather rung moved
> anything or not, and it says so in those words. On this run it moved a great deal, and the one
> hypothesis this phase was able to test came back NEGATIVE. The negative result is published here in
> the same shape a positive one would have been.
>
> **Vocabulary discipline.** This phase corrected INPUTS. It did not fit, score, gate or ship a
> model. The word "deploy" is used only of a production pointer that actually moved, and none did.
> That the trainers now select weather features far more heavily is an OBSERVATION, never a result.
> Closing-line value is REPORT-ONLY throughout, as it is everywhere in this repository, and is never
> evidence of profitability.
>
> **Nothing published earlier is overwritten.** Superseded readings stay where they were written,
> with their dates and their reasons. `OU-DIVERGENCE-DIAGNOSIS.md`, `PROFITABILITY-READOUT.md`,
> `GATED-REFIT-READOUT.md`, `ACTIVATION-READOUT.md`, `SIGNAL-LIFT-READOUT.md` and
> `LINE-MOVEMENT-READOUT.md` are unedited by this phase. `33.1-SPEC.md` is likewise a frozen record
> and was deliberately NOT rewritten, which is why the three owner-ratified scope expansions are
> declared here in section 7 rather than folded silently into the specification.
>
> ASCII only (no emoji, per CLAUDE.md). Arrows are `->`, dashes are `--`, quotes are straight.

**Headline, in one line.** Every one of the 6,499 games in 2002-2025 now carries a real ERA5
observation fetched at the stadium it was actually played at; three further input defects were found
while measuring that and fixed; gold was rebuilt twice, each time as a separately-attributed rung;
nineteen scattered season literals became one committed rule with 2025 finally visible; a final-fit
entry point and an artifact-preprocessing contract were authored and proven; **no model was re-fit in
this phase, nothing has been promoted, and `artifacts/latest.json` is byte-identical to what it held
before the phase began.**

**Suite state at close.** No whole-tier sweep was run, under the standing owner instruction of
2026-09-14 that forbids directory sweeps; targeted module runs only, and every figure in this
document was measured directly rather than inferred from a suite run. The EXPECTED failure set at
close is sixteen registered node ids: the five deliberate tripwires in
`tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS`, the ten in
`tests.phase33_state.GOLD_REBUILD_NEWLY_RED`, and the one deferred stale partition literal
`tests/unit/test_wp_trainer.py::test_wp_train_on_synthetic_data`. All sixteen are named in committed
source with their reasons. Nothing was added to any skip list and the tripwire tuple is unchanged at
five entries.

---

## 0. READ THIS FIRST -- the Wave-15 instruction that is now wrong

`.planning/phases/33-live-cold-start-forward-temporal-integrity/33-15-PLAN.md:32` instructs Wave 15,
under decision D33-12:

> "each target gets EXACTLY ONE candidate -- the SAME recipe as its incumbent (same training window,
> same feature set, same hyperparameters), re-fit on corrected gold."

The same wording is repeated in that plan's `<behavior>` block at `:108` and again in its Task-2
owner checkpoint at `:227`.

**That sentence was written when nothing had changed that could justify a different feature set. It
is now wrong in two separate ways.**

**Wrong the first way -- "same feature set".** WP and ATS selected ZERO weather features. But they
were selecting from columns frozen at a fabricated constant, and a constant column has no importance
and cannot be selected by any procedure. **They did not reject weather; they were never offered
any.** If anyone reads "same feature set" as a pinned feature LIST, WP and ATS are locked into
weatherless models and this entire phase is wasted.

The risk is the prose, not the code, and that was CONFIRMED against live source rather than assumed:
`BaseTrainer.train_and_evaluate` calls `select_features` as its first step on every run, and
`models.train` exposes no feature-list flag, so a re-fit DOES re-select. What Wave 15 must carry
forward is the SELECTION PROCEDURE, not the resulting list. Say "the same selection procedure" and
the instruction becomes true again.

**Wrong the second way -- "same training window".** It is no longer satisfiable by committed rule.
Section 3 below replaces every hand-typed season list with one deterministic rule, and under that
rule there is no such thing as a per-target training window to hold constant. Wave 15 consumes
`conf.season_partition.default_season_partition()` or it disagrees with the rest of the repository.

> **CORRECTION, 2026-09-14 (code review CR-01).** When this document was published the sentence
> above was a true statement of what Wave 15 *should* do and a false statement of what the code
> *would* do. `scripts/promote_models.py` -- the path that actually runs Wave 15 -- took only the
> holdout from the rule and still read the training and tuning windows out of the deployed
> incumbent's own `metadata.json`. Measured at the time: a WP candidate would have run with
> `--config-train-seasons 2018,2019 --config-hp-val-seasons 2020`, so feature selection would have
> happened on the 534-row window this phase argued against, using the training window of a model
> the owner has declared void. All three windows now come from the rule. Nothing in this document's
> measurements changes; what changed is that the instruction it gives Wave 15 is now true of the
> code as well as of the intent.

---

## 1. The measured weather rung -- and it is a COMPOUND rung, never "the weather rung"

### 1a. What the corpus looked like before

`data/silver/weather.parquet` held **14 rows**, all from 2024 Week 6 -- the only real weather this
system had ever held. Everything else was filled downstream with a hardcoded `temp_f: 65.0` that also
set `is_outdoor: False`, so a missing record, a fetch failure and a dome were indistinguishable by
construction. **6,485 of 6,499 gold rows sat at that default.** Inside every model window the
imputation was total: 1,335 of 1,335 rows in the ATS feature-selection window, **534 of 534** in the
WP and O/U one, 1,139 of 1,139 in the 2021-2024 gate holdout, with 45 of 46 weather-family columns
exactly constant in all three.

### 1b. What it looks like now

**6,499 rows covering 2002-2025**, every one stamped `weather_source = 'archive'`, fetched from the
ERA5 reanalysis archive at the stadium each game was actually played at. `temp_f` is non-null on the
4,847 outdoor games -- exactly the fetched population; the other 1,652 are indoor games that
correctly carry no outdoor measurement -- with **745 distinct values spanning -9.0 F to 101.6 F**.
Rows at exactly 65.0 F: **6**, against 6,485 before.

The run itself landed on its projection to four significant figures: 5,816.4 weighted calls against
5,816.4 projected, 81.1 minutes against about 81 projected, 4,847 attempts, **zero retries, zero
provider refusals, zero interruptions**. Twenty-four append-only bronze snapshots, one bracketed
silver promotion.

### 1c. The gold rebuilds, and why there were two

Gold was rebuilt TWICE in this phase, each time as its own separately-attributed rung, because the
second rebuild had different causes from the first and merging them would have made every moved
column unattributable.

**Rung A -- the weather corpus, the routing correction and restored 2025 coverage.** Widths
194/195/194 -> **195/196/195** over 6,499 rows; exactly one column added,
`weather_coverage`; none removed; 131 non-clock columns moved.

**This rung has a COMPOUND cause and is never to be labelled "the weather rung" without
qualification.** Three causes, merged by owner choice: real ERA5 weather replacing the fabricated
constant; the all-seasons `stadium_id` routing correction (D33.1-06), which moves the
venue/travel/timezone/elevation family for the 1,153 games that had resolved to the wrong stadium;
and restored 2025 coverage -- 207 games the two silver feature tables were missing, which the rebuild
adds independently of either. The SPEC's fifth prohibition is precisely about this kind of
mislabelling, and it is honoured by naming all three.

A gain that came with the routing correction and is recorded as a gain rather than a side effect:
**91 neutral-site games that had been resolving to the wrong stadium now resolve to their own**, of
which 83 actually changed venue and 8 already resolved correctly by coincidence. That closes the
`HISTORICAL_NEUTRAL_MISRESOLUTION` disclosure.

**Rung B -- three further input defects, found while measuring rung A.** Widths unchanged at
195/196/195; no column added or removed; 32 non-clock columns moved, split 19 weather-family + 1
source-derived widening copy + 12 opponent-adjusted columns in season 2025 and no other season.
Attribution verdict: `ok=True`, with the judge reporting **`failures: 0` and `unattributed: 0`**,
identical in all three matrices. (Written as key/value rather than in prose: the milestone-wide rule
forbidding a certain phrasing about counts of failures is absolute and applies to every file this
phase touches, including a sentence about a gold attribution that has nothing to do with the test
suite. The rule is not narrowed to make a sentence read better.)

The three defects, in plain English, and all three were throwing away real data already on disk:

1. **Rain was being deleted for every outdoor game.** The code refused to compute any precipitation
   feature unless it had a *probability* of rain. But a historical reanalysis records how much rain
   actually FELL and never a forecast of whether it might. So for all 4,847 outdoor games -- the
   games where rain is the entire point -- the real rainfall measurement was discarded. `precip_mm`
   non-null went from 1,652 to **6,499 of 6,499**, and 18 of 20 dead columns came alive for those
   4,847 games. The gate NARROWED rather than disappearing: no probability is invented, and an absent
   rainfall measurement still nulls the whole family.
2. **The weather-coverage flag was saying "no observation" for every game.** It was set correctly to
   1.0 when the data was built. The final rescaling step of the gold build then z-scored it, and the
   z-score of a column that is identical on every row is exactly zero -- which is the code's own word
   for "no observation". For all 6,499 games the one column that exists to tell "we have no weather
   record" apart from "the weather was mild" was asserting the opposite of the truth. It now reads
   **1.0 on all 6,499 rows.**
3. **Season 2025's team-strength features were never built.** One line said `range(2018, 2025)`, and
   Python's `range` stops one short. The twelve opponent-adjusted columns carried **2 distinct values
   across 285 rows of 2025** where 2023 and 2024 each carry 285. 2025 is the season the live 2026
   predictions are built on. They now carry 285.

### 1d. R5's before/after constancy pair, stated as a pair

| measurement | before | after |
|---|---|---|
| `raw_temp_f` rows at the fabricated 65.0 | 6,485 of 6,499 | **6** |
| `raw_temp_f` distinct values | 15 | **700** |
| `precip_mm` non-null rows | 1,652 of 6,499 | **6,499 of 6,499** |
| `weather_coverage` | 0.0 on all 6,499 | **1.0 on all 6,499** |
| the O/U model's 17 weather features, constant in its 2018-2019 selection window | 17 of 17 | **1 of 17** |
| the same, in the 2021-2024 gate holdout | 17 of 17 | **1 of 17** |
| whole weather family constant, all three windows | 45 of 46 | **4 of 47** |
| gold rows | 6,499 | **6,499, unchanged** |
| `artifacts/` | 159 files | **unchanged, 159 files** |

The four columns that are still constant are constant HONESTLY, and each is named rather than left to
look like residue: `precip_prob` and `raw_precip_prob` are the forecast probability, which ERA5
reanalysis does not report and which the owner's ruling explicitly forbade inventing; `extreme_weather`
is the `weather_severity_score >= 0.8` indicator, which no game in 2002-2025 reaches on real readings
-- a fact about the weather, not about the pipeline; and `weather_coverage` cannot vary over a corpus
in which every game has an observation. That last one **diverges from its stated acceptance
criterion**, which said the flag must VARY. It does not. The change is from a constant falsehood to a
constant truth, and that divergence is reported as measured with nothing adjusted to fit.

### 1e. The attribution failure that was diagnosed rather than papered over

**The first rebuild attempt FAILED its attribution.** Forty-five columns per matrix came back
unattributed, decomposing into 40 that moved in 2024 (and 16 of those also in 2025), 4 that moved in
2025 only, and 1 that moved in all 24 seasons.

A dedicated investigation (`33.1-07-GROUP1-DIAGNOSIS.md`) established three things and failed to
establish a fourth, and the fourth is the important one:

- **The build is deterministic and byte-reproducible.** A fresh full rebuild 38 minutes later
  reproduced `data/gold/*.parquet` exactly in every column of every season except the build clock.
- **The wall-clock hypothesis is refuted.** Pinning the build clock four and a half months earlier
  changed nothing except the build clock itself.
- **The moves PREDATED the rung.** The Plan 33-12 sandbox gold, built five and a half hours BEFORE
  the rung from the OLD silver, already carries 40 of 40 of the new 2024 values and 0 of 40 of the
  baseline's. The baseline was stale -- written before Phase 33's waves 9 to 12.
- **THE TRIGGER IS NOT ESTABLISHED, AND THE MAGNITUDE IS PERMANENTLY UNMEASURABLE.** The old
  fingerprint holds column digests, not values, and no copy of the old gold survives. Nothing on this
  machine can reproduce the pre-rung 2024 values. They were produced by a state that is gone.

The residual was then declared in a FOLLOW-UP rung as three named groups, with the unknown bounded
and written down rather than hidden. This is stated here at length because it is exactly the kind of
thing a later wave would otherwise rediscover from scratch.

---

## 2. The cross-check against its pre-registered expectation

The 1,942 rows of real weather that already existed in bronze for 2018-2024 were kept as a
regression check, never overwritten and never deleted. The re-fetch was diffed against them under an
expectation committed at `57586c1` and witnessed from outside at `9e28d40`, **before any of the new
data existed**. The pre-registration was NOT edited: `preregistration_was_edited` is recorded False.

**Verdict: PARTIAL.** Two of three pre-registered causes matched; the third was right in kind and
wrong in population, and one of its clauses is falsified.

| cause | verdict | measured |
|---|---|---|
| the timezone/hour fix | **MATCHED**, and exceeded its prediction | all 1,942 of 1,942 comparable rows differ; temperature fell on 65.3% of the 1,370 rows where both sides carry a reading; mean signed shift -0.66 F, mean absolute 3.02 F |
| the routing fix | **MATCHED EXACTLY, on every clause** | 44 comparable games predicted, 44 measured; relocated games average 15.61 F of absolute difference against 2.60 F elsewhere; signs split 36 positive to 8 negative as predicted, not shared |
| the per-game roof rule | **PARTIAL -- right in kind, population under-predicted, exclusivity clause FALSIFIED** | all 260 predicted closed-roof games went number-to-NULL, but the measured number-to-NULL population is 572 rows |

The falsified clause, quoted from the pre-registration: *"THESE ROWS GO FROM A NUMBER TO A NULL
TEMPERATURE, and this is the ONLY category where that happens."* It is false because **dome games are
a second category**: the legacy corpus stored a temperature for all 312 dome games in the comparable
window while recording `is_outdoor` False on 308 of them, so they too go number-to-NULL under the new
rule -- not because their roof classification changed, but because the new code refuses to store a
measurement for a game weather does not apply to.

**The prediction was not edited to accommodate this. The disagreement IS the finding,** and it is
preserved with its own `falsified_clause` and `falsified_because` fields. A disagreement never failed
the run, by design: the comparator reports, it does not judge. And a large disagreement was PREDICTED,
because the hour fix moves essentially every comparable row.

Per season the disagreement is stable at roughly 1,100 column-level differences on ~270 rows for each
of 2018 through 2024. The three largest per-home-team mean absolute temperature differences are
**LA (16.59 F), LAC (15.23 F) and LV (13.61 F)** -- the Coliseum, StubHub and Oakland relocations,
exactly the venues this phase added and rerouted. Column-level totals: 7,812 disagreements across 15
compared columns at a 0.1 F tolerance, 2,802 number-to-null and 3,160 null-to-number. **4,557 rows are
new-only and 0 rows are legacy-only** -- nothing that existed was dropped.

---

## 3. The re-derived training window rule, and the evidence beside it

Until this phase the answer to "which seasons does this model train on, and which does it get graded
on?" was written down in eleven different places as hand-typed lists of years. They agreed only
because somebody kept them in step by hand, and one had quietly fallen out of step: a single line in
the backtesting engine said "stop at 2024", so every consumer that loaded data through that engine had
never seen a 2025 game. Nothing checked.

There is now ONE rule, in `conf/season_partition.py`, committed BY ITSELF at `e7d0ca5` and witnessed
from outside in a strictly later commit so the pre-registration property stays checkable. It says:
grade on the two most recent completed seasons, tune on the one before those, select features from
2018 onward, and fit the shipped model on everything from 2002. On today's data that yields:

| set | seasons |
|---|---|
| feature selection | 2018-2022 |
| hyperparameter tuning | 2023 |
| holdout / grading | 2024-2025 |
| shipped final fit | 2002-2025 |

**OWNER RULING, 2026-09-14: APPROVED**, in the owner's own terms -- "pick features on 2018-2022 where
the stats are real, tune settings on 2023, hold back 2024-2025 for an honest score, final model trains
on everything". The partition rolls forward automatically from here; `LATEST_COMPLETED_SEASON = 2025`
is the one remaining literal and is deliberate, because reading gold at import time would make the
partition depend on whether a rebuild had run and deriving it from the calendar would let it move with
no commit recording that it moved.

**Ruling Q rests on a stated prior, not on a score, and is recorded that way.** The selection window
starts at 2018 rather than 2002 because **ninety gold columns are placeholder values before then** --
every Elo column, every rolling opponent-adjusted EPA column, all three market snapshot columns and
the situational spots -- and selecting features on placeholder data is what produced the noise-feature
problem in the first place. The owner was told before ruling that this rests on a prior rather than a
measurement, that both alternatives are defensible, and that the recorded cost is a selection window
narrower than the fit window. **Nothing was fitted, ranked or compared to pick this window.** The
supporting evidence is itself measured and committed beside the rule: feature selection on the
534-row 2018-2019 window selected 6 of ATS's 25 and 9 of O/U's 25 features from pure synthetic noise
columns at N=50.

> **CORRECTION, 2026-09-14 (code review CR-01).** The table above is the rule, and the rule is
> unchanged. What was not true when this was published is that every consumer obeyed it. The
> promotion path took only the holdout row from this table; the feature-selection and tuning rows
> were still read out of each deployed model's stored metadata, which records 2018-2019 (2015-2019
> for ATS) and 2020. So a Wave-15 candidate would have selected features on 534 rows -- the exact
> thing Ruling Q exists to stop -- while this table said otherwise. All three rows now come from
> the rule on that path, proved by
> `tests/unit/test_promote_models_tuned_path.py::test_all_three_windows_come_from_the_committed_partition_rule`.
> The difference between the rule and what each deployed model recorded is still REPORTED on every
> run, and now names every window that moved rather than the holdout alone.

Nineteen live test pins carried the old partition where the plan predicted five. Two of the nineteen
were **PASSING while measuring the wrong seasons** -- both sides of their comparison drew from one
window, so they were correct only while the live and frozen windows happened to coincide. All nineteen
were updated with recorded reasons; none was deleted and none was relabelled a deliberate tripwire.

**One consequence, measured by driving the real promotion path, that this phase did not cause and did
not discharge.** ATS and O/U no longer reproduce the frozen 2021-2024 gate baseline at all: the pooled
divergence is ATS **+0.041** against the **-0.001** recorded in `config/gate.toml`, and O/U **+3.14**
against **+1.10**. That is the expected consequence of the standing owner ruling of 2026-09-14 -- the
old data was corrupted, so the frozen baseline and every verdict resting on it are void -- and not a
new defect. **Nothing was re-frozen.** The `[baseline.*]` block is byte-identical, both gate-baseline
tripwires stay red, and the divergence is recorded rather than smoothed over.

---

## 4. The Phase 33 instructions the corrected inputs have invalidated

These are the four the phase entered with, plus four more its own execution added. Each names the file
and line, what is wrong with it, and what replaces it.

| # | Instruction | Where | Status | What replaces it |
|---|---|---|---|---|
| 1 | "ONE rung, named for the Elo re-derivation, because there is ONE cause and R4 names one expected change set" | `33-14-PLAN.md:33` | **FALSIFIED** | Phase 33.1 rebuilt gold twice before Wave 14 runs, for a COMPOUND cause. Wave 14's rebuild is no longer the first rung after 33-12, and its one-cause premise now describes its POSITION on the ladder rather than the ladder itself. Wave 14 must declare its rung against the `p331_` rung-3 fingerprint, not against 33-12's. |
| 2 | the `GOLD_WIDTHS_BEFORE_ELO_REBUILD` width comparison, pinned at 194/195/194 | `33-14-PLAN.md:26` and `:41` | **STALE** | The pinned widths moved before Wave 14 runs. Re-anchor to `tests.phase33_state.GOLD_WIDTHS_AFTER_WEATHER_RUNG` = **(195, 196, 195)**. This is already live as a registered red: `test_gold_write_scope.py::TestTheIdentityMigrationMovesNoGoldColumn::test_the_recorded_reference_matches_todays_production_gold` fails today and was deliberately left failing because reconciling the two numbers is Wave 14's call, not this phase's. |
| 3 | the owner authorisation's pre-declared expected change set -- "every season 2002-2017, plus (2018, 1) -- exactly the 4,288 rows that carried a fabricated 0.0 [Elo]. Column widths stay 194/195/194" | `33-14-PLAN.md:93-99` | **MUST BE RE-DECLARED** | It was written against a gold in which 45 of 46 weather columns were constant and 2025's team-strength family was never built. Both are now false, the widths are wrong, and the baseline the diff would be taken against has moved. The owner cannot meaningfully authorise a change set computed against inputs that no longer exist. |
| 4 | D33-12: "the SAME recipe as its incumbent (same training window, same feature set, same hyperparameters)", with the per-incumbent recipe list naming WP 2018-2019, ATS 2015-2019 and O/U 2018-2019 | `33-15-PLAN.md:32`, `:108`, `:113`, `:227` | **FALSIFIED, both halves** | See section 0. "Same feature set" must become "the same selection PROCEDURE" or WP and ATS are locked into weatherless models. "Same training window" is unsatisfiable by committed rule: Wave 15 consumes `conf.season_partition.default_season_partition()`. |
| 5 | Wave 15's re-fit mechanism is unstated; the existing trainers keep the LAST walk-forward fold | `33-15-PLAN.md:108` | **INCOMPLETE** | Wave 15 must call `models/trainers/final_fit.py` -- `trainer.final_fit(gold, partition)`, then `apply_final_fit_to_trainer(trainer, result)`, then the existing `trainer.save()`. Without it the shipped model fits through 2023 and never sees the newest two seasons, whatever the partition says. Section 5 below. |
| 6 | Wave 15's gate verdict is presented as a clean pass | `33-15-PLAN.md:227-231` | **MUST BE LABELLED IN-SAMPLE** | Once the shipped model is fitted on the holdout seasons, the gate's re-score of that artifact on holdout gold is no longer an out-of-sample estimate. Disclosed and ACCEPTED by the owner. Section 7 below. |
| 7 | "a failing target retains its incumbent", and every framing in which a candidate must non-regress against the frozen baseline | `33-14-PLAN.md:110`, `33-15-PLAN.md:32` | **VOID under the standing owner ruling** | The standing owner ruling of 2026-09-14 voids the frozen 2021-2024 gate baseline and every Phase-30/31 verdict resting on it, and voids the deployed pre-correction artifacts as things worth preserving. You cannot non-regress against a lie. Re-fitting on corrected data is the expected path, not a scope addition needing justification. |
| 8 | Wave 14 will need a way to keep published readings alive across its gold rebuild | not yet in `33-14-PLAN.md` | **AVAILABLE, DO NOT REINVENT** | `tests/gold_generation.py` already exists: `gold_generation_key()` over gold CONTENT and `require_gold_generation(expected_key, *, reading, moved_by, recorded_in)`, which SKIPS rather than fails and names the reading, the mover and where it is recorded. Built in this phase for exactly this, with a non-vacuity control proving a matching key does not skip. |

---

## 5. Plainly: no model was re-fit, and nothing has been promoted

**No model was re-fit in this phase. No candidate was fitted, scored, gated or shipped, and no
production pointer moved.** `artifacts/latest.json` is **byte-identical** across all five waves --
digest `7ff78a50...0f66f1` before and after, recorded twice in committed source --
and `git status --short artifacts/` is empty. The three production pointers are exactly what they
were: `wp_20260824_113325`, `ats_20260605_220128`, `ou_20260326_163930`, plus the unchanged blend
`blend_dynamic_20260606_020635`. No deploy gate was run. The 1,942 pre-existing bronze rows were
neither overwritten nor deleted.

**This phase makes no accuracy claim and no profitability claim.** It replaced a fabricated record
with a measured one. That the trainers now select weather features far more heavily is an
OBSERVATION and not a result: in a read-and-fit-in-memory smoke fit that never called `save()`, WP
selected four weather features where it had selected none, ATS three, and O/U now selects weather
columns it could not previously see. **Wave 15's measurement is what decides anything about
accuracy, and it has not run.**

**Two mechanisms were authored here and are deliberately not called.**

**The final-fit entry point** (`models/trainers/final_fit.py`) exists because
`WalkForwardSplitter.generate_splits` builds every fold as `season < holdout_season` and each trainer
keeps the LAST fold's model -- so no choice of season lists could ever produce a model fitted through
the newest completed season. An AST scan across `models/train.py`, `scripts/promote_models.py`,
`scripts/run_phase33_gate.py` and `pipeline/steps.py` finds **zero call sites** for any of its three
symbols, with a planted-call control proving the scan fires. Its recorded cost, stated rather than
hidden: the calibrator is carried across the final fit BY REFERENCE and never refitted -- the test
asserts object identity, so a silently refitted calibrator that happened to agree to fifteen decimal
places would still fail -- which means the shipped model's calibrator will have been fitted against a
narrower model than the one that ships. That biases mildly toward UNDER-confidence, a conservative and
statable error.

**The artifact preprocessing contract** (D33.1-R1) is **authored and round-trip proven**, per target,
under `assert_array_equal` rather than a tolerance, because a missing standardisation is not a
near-miss. WP's preprocessing now persists as one inseparable `Pipeline`; ATS and O/U converter
parameters persist explicitly as JSON in metadata; serving consumes both. **No production path calls
the final fit, and the deployed artifacts remain legacy-served pending Wave 15.** All three of those
clauses are needed: "authored but not production-integrated" now understates it, and
"production-integrated" would overstate it.

---

## 6. R7 -- the 2026 gold-weather bridge is BOUNDED

`features.weather.WEATHER_GOLD_DEFAULT_SEASONS` holds the gold weather family at its
absent-observation default for season 2026. This phase did NOT author it: Phase 33 Wave 9 Task 5 wrote
it under the owner ruling taken at that plan's Task-4 blocking checkpoint, which approved the switch
with one change -- the removal condition was set to **this phase's re-fit, meaning Wave 15**, rather
than a future Phase 37. The hold is a bridge, not a season-long policy: the only reason to hold 2026
weather back was that the O/U model had never seen weather vary, and fixing the historical record
dissolves that reason at the next re-fit.

**Switch existence was checked against the live tree at execution time rather than assumed, and the
switch exists. R7 is SATISFIED and the bridge is bounded** by
`tests/unit/test_weather_bridge_expiry.py::TestTheSwitchIsBounded`, which passes today and turns red
once the flip condition is met while the switch is still in committed source. Its failure message
names the switch, its module, its flip-condition string, the pointer that moved and which clause
fired, and says the switch must be REMOVED rather than re-dated.

**The flip condition is ARTIFACT SEMANTICS, not a moved deployment pointer, and the distinction is
load-bearing.** `models/prediction_pipeline.py:705-716` reads ONLY the artifact's selected
`feature_list`, so a newly promoted artifact containing zero weather features serves exactly as a
weather-blind model does. That is the live case and not a hypothesis: WP and ATS each select **zero**
of the 46 defaulted weather columns today, and only the v1.0 pre-Elo O/U artifact selects any, at 17.
Under a pointer-movement predicate a Wave-15 promotion of a still-weatherless WP would flip the
condition and the tripwire would demand the bridge be removed while the train/serve reason for it
still held exactly -- a false positive that removes a guard.

So a moved pointer is the TRIGGER, and the predicate is: the newly-pointed artifact's feature list
intersects the actual defaulted weather columns, OR its metadata carries a
`trained_on_real_weather_generation` marker naming a post-correction gold generation. Both clauses are
proven independently sufficient, a moved pointer alone is proven NOT to satisfy the condition, and the
bounding assertion is proven to FIRE against a simulated post-flip state. The defaulted column set is
derived from the switch's own held-state producer rather than copied, so it cannot drift from the
bridge. The test is deliberately ABSENT from `DELIBERATE_TRIPWIRE_NODE_IDS`, which still holds exactly
five entries: a tripwire encodes an accepted fact and must stay red, and this one must pass now and
fail later.

---

## 7. The disclosures this phase is obliged to carry

### 7a. Three owner-ratified SCOPE EXPANSIONS, named as expansions

All three were ratified by the owner on **2026-09-12**, past the SPEC's literal wording. `33.1-SPEC.md`
is a frozen record and was deliberately NOT rewritten, so this is where the widening becomes visible.

- **D33.1-R1 (attributed to R6) -- the artifact contract was repaired IN-PHASE rather than bounded and
  handed on.** WP preprocessing persists as one inseparable `Pipeline`; ATS and O/U converter
  parameters persist explicitly as JSON in metadata rather than as a second pickle, which keeps them
  auditable and does not widen the deserialisation surface; serving consumes both; and a per-target
  save / load / predict round trip asserts served equals in-memory exactly. The contract **binds
  artifacts saved under it ONLY**, and the legacy raw-feature and 13.5 / 13.0 fallback paths are
  preserved by design.
- **D33.1-R2 (attributed to R6) -- that contract binds NEW artifacts only.** The three deployed
  artifacts serve values byte-identical to a baseline captured BEFORE the change, and serving does NOT
  refuse a legacy artifact: the branch is `if preprocessing is not None`, never an assertion. The owner
  rejected a refusal explicitly, because it would take current-week prediction generation down on
  purpose.
- **D33.1-R3 (attributed to R4) -- WP became a fold-fitted imputation and missing-indicator
  `Pipeline`** (imputer -> missing_indicator -> scaler -> estimator), so its pre-selection path never
  sees NaN and the corrected weather this phase produces cannot break Wave 15's WP re-fit. The imputer
  is fitted PER FOLD, which is a temporal-safety requirement rather than a nicety. **The rejected
  alternative is recorded with the owner's reason:** excluding nullable weather columns from WP's
  feature selection was refused because it would foreclose WP ever using the weather this phase exists
  to produce.

### 7b. `wp_20260824_113325` is trained on scaled features and served on unscaled ones, in production, today

This is a KNOWN LIVE DEFECT. The kept WP estimator was fitted on SCALED features; `predict_games`
passes RAW selected columns straight into `predict_proba`; and the artifact directory carries no
`preprocessing.pkl`, so there is nothing on disk from which the serving path could recover the fitted
scaler. The same is true of the production current-week path and the diagnostic re-score path.

**It PREDATES Phase 33.1** and was found by the orchestrator reading live source during the cross-AI
review of these plans -- no reviewer lane raised it, and the one lane that named the mechanism framed
it only as a risk for a future final fit, not as a fact about the deployed artifact.

**This phase NAMES it in a fail-closed test carrying the artifact id in an assertion, and this phase
does not fix it.** Serving the existing artifact scaled would change live predictions with NO re-fit,
which breaks the SPEC's `artifacts/latest.json`-byte-identical fence and R8's honesty rule alike: a
data-correction phase that silently moved production numbers would be committing the defect class it
exists to detect. **Phase 33 Wave 15's re-fit must resolve it, by shipping a WP artifact under the new
contract** -- at which point the transform and the estimator travel as one `Pipeline` and cannot drift
apart. When Wave 15 ships that replacement the test must be UPDATED with a recorded reason, never
deleted; deleting it erases the disclosure instead of closing it.

A second, latent serving defect was found and fixed in passing: `predict_games` was the single site in
this repository calling `calibrator.transform()`, and the deployed WP calibrator is a `PlattCalibrator`
that exposes `predict` and not `transform` -- so serving the real deployed artifact through that
pipeline raised `AttributeError`. It could not serve the deployed WP artifact at all. No test caught
it because every synthetic fixture saves artifacts without a calibrator, and no production path calls
`predict_games`, so it was latent rather than live. The fix moved no served number, because the branch
produced none.

### 7c. The O/U weather hypothesis -- a NEGATIVE result

Once real weather landed, the already-graded 2021-2024 O/U results were split by whether weather could
reach the field and by weather severity. This required no re-fit; it DID require a re-score, which is
what the committed harness does on every run and is how the published Phase-26 diagnosis was produced
too. No new cut was added and no new bucket was created, so no new comparison entered the correction
family fixed in Phase 26. The numbers are descriptive and carry no p-value.

| bucket | bets | graded hit rate | against the 0.5238 flat break-even |
|---|---|---|---|
| outdoor | 749 | 0.4766 | below |
| indoor | 338 | 0.4822 | below |
| severe weather | 543 | 0.4733 | below |
| mild weather | 544 | 0.4835 | below |

**The result does not support the hypothesis.** All four buckets grade below break-even within about
one percentage point of each other. If fabricated benign weather were the missing input, the bucket
where weather cannot reach the field is where the model should look best -- and it does not. Both
splits account for every row: 749 + 338 + 0 = 1,087 and 543 + 544 + 0 = 1,087.

**Two of the three facts the hypothesis rested on have moved.** They were Phase-26 readings taken on
the fabricated weather. Measured today on corrected gold: the over/under split is **966 over to 121
under**, not the published 790/297; and the high-total bucket grades **0.5000**, at or below
break-even, where the published reading has it at 0.5521 above. **The published figures are not
rewritten** -- they are what Phase 26 measured, and they stay where they were written with their date
and their reason. These numbers are reported BESIDE them, never instead of them, and they do not
supersede `OU-DIVERGENCE-DIAGNOSIS.md`. A third fact that never fitted the hypothesis is unchanged:
WP shows the same closing-line-value-positive, ROI-flat shape with ZERO weather features, so weather
cannot be the general explanation for a shape that appears where weather is not an input at all.

Two repairs to the cut itself were needed before it could be graded, and both changed what the cut
MEANS: applicability is now keyed on the game's own roof rather than its stadium's, which moves 621
closed-roof games at the five retractable stadiums out of the outdoor bucket where the venue key had
put all 749 of their games together; and a game with no weather observation is now excluded from both
denominators with its count reported, instead of being silently counted as a mild-weather game by the
old bare-complement comparison. On this window that excluded count is **zero**, and the zero is
reported as a zero rather than dressed up.

**One consequence that must not be buried.** Making a pre-registered cut gradeable changed the
BH-FDR trial denominator from the published 36 to 44. The eight new entries are exactly this cut's
four buckets across two streams, which were always REGISTERED and were previously counted as
unavailable with a null p-value. The registered family did not grow; the testable part of it did.
Today's adjusted p-values are therefore computed over a different denominator than the published
ones, and the published readout is not edited.

### 7d. The 534-games record is FALSE, and is corrected here rather than quietly dropped

The claim that WP and O/U "train on" 534 games -- and ATS on 1,335 -- appears in `33.1-HANDOFF.md`,
in `33.1-SPEC.md`'s Background and Requirement 6, and in the ROADMAP's Phase 33.1 entry. **It is
false.** What **534** and 1,335 actually describe is the FEATURE-SELECTION and HYPERPARAMETER window.
The deployed models are fit on roughly 2002-2023, about 6,000 games, because every walk-forward fold
trains on `season < holdout_season` over the full 6,499-row frame and the LAST fold's model is the one
kept. The narrow selection window is a real and separate weakness -- it is the one section 3's rule
addresses -- but it is not the training window, and the two must not be conflated in the replan.

### 7e. The gate verdict Wave 15 produces will be IN-SAMPLE, and this is stated before any verdict exists

The gate grades a candidate by re-scoring the saved artifact on holdout gold. Once the shipped
artifact has been fitted on those seasons -- which is exactly what section 3's final-fit window and
section 5's entry point together produce -- **that verdict is no longer an out-of-sample
generalisation estimate**, and a confident result would otherwise print with no warning at all. The
promotion code's own comment already named this hazard before this phase touched it.

**The owner was told and ACCEPTED it, twice, most recently on 2026-09-14 once it had become mechanical
rather than hypothetical.** Two things follow and both are binding: **Wave 15's verdict must be
LABELLED in-sample**, never presented as a clean gate pass; and `_incumbent_window` keeps reporting
the difference in words on every run -- it was NOT switched off, and no artifact metadata was edited
to make a refusal pass.

### 7f. The Wave-15 runner qualification -- stated, not elided

It is TRUE that `scripts/run_phase33_gate.py` contains zero references to `HOLDOUT_SEASONS`, to
`baseline`, or to the frozen `[baseline.*]` block. **BUT** it imports `scripts/promote_models.py` at
`run_phase33_gate.py:94` and calls into it at `:799` via `stage_two_promote`, and `promote_models`
still passes the frozen bundle as the comparator at `promote_models.py:1458` and still runs the drift
tripwire at `promote_models.py:1438`. **They are not separable.** Do not write "nothing reads the
frozen block" unqualified.

The drift tripwire needed more than a retargeted season list to survive the partition move: it
compares today's numbers against a frozen historical record and can only do that on the seasons that
record was frozen over, so it was given its own FRAME. Pointing its season list at 2021-2024 while its
frame was sliced to 2024-2025 made the promotion path abort deterministically, every time -- which was
measured by driving the real path, not reasoned about.

### 7g. The imputation problem is NOT solved repo-wide

**Ninety gold columns remain a flat imputed placeholder for 2002-2017** -- every Elo column, every
rolling opponent-adjusted EPA column, all three market snapshot columns and the situational spots --
because their silver sources begin in 2018 or later (`elo_game_snapshots.parquet` 2018-2025,
`odds_snapshot.parquet` 2018-2025, `team_game_stats.parquet` 2020-2025). Seven of WP's twenty selected
features, seventeen of ATS's twenty-five and seventeen of O/U's twenty-five are in that set, including
O/U's highest-importance feature. R5's acceptance is satisfiable while that is true, and a readout
that did not say so would invite the opposite conclusion.

### 7h. The archive-versus-forecast provenance probe -- measured, and nothing rests on it

A question raised in cross-AI review: does the historical ERA5 archive carry a systematically
different bias from the real-time forecast feed the 2026 live path uses? It was MEASURED rather than
answered from first principles, under an interpretation rule registered BEFORE any number existed.

**Sample size: n = 28 at most**, and that is the whole population -- the only rows in this repository
whose provenance is known or claimed to be the forecast feed. Measured: archive minus forecast, mean
signed temperature difference +0.87 F (sd 3.35), mean signed wind difference -2.47 mph (sd 3.67).

The pre-registered interpretation rule, quoted verbatim: *"WITH n AT MOST 28 THIS IS A DIRECTIONAL
PROBE AND NOT AN ESTIMATE. No bias correction, no feature change and no Wave-15 instruction may rest
on it. Its ONLY legitimate use is to say whether the question deserves its own measurement later, on a
population built for it."*

**No bias correction, no feature change and no Wave-15 instruction rests on it.** Two further facts
are recorded rather than resolved: the two halves of the population DISAGREE IN SIGN on temperature
(+2.80 F on n=9 for the uncontested half, -0.71 F on n=11 for the contested one), and the
pre-registration and this repository's own records DISAGREE about whether the contested 14 rows are
forecast-feed rows at all -- they read `weather_source = 'archive'`. Both statements are in committed
source and they cannot both be right. **Nothing is concluded from either, and no direction is asserted
that the sample cannot support.** The pre-registration was not edited.

### 7i. Nineteen weather yes/no flags are scaled into many distinct decimals -- an OWNER DECISION due BEFORE any re-fit

Nineteen of the twenty-three weather indicator columns arrive in gold as many distinct decimals rather
than as two levels: `is_snow` 274 distinct values, `weather_game` 281, `precip_heavy` 392,
`temp_very_cold` 622, `wind_severe` 1,694, `temp_cold` 1,259, `precip_moderate` 1,367, `temp_cool`
2,792, `is_rain` 3,306, `precip_light` 3,338, `precip_none` and `is_dry` 3,487, `temp_hot` 3,125,
`temp_warm` 4,028, `temp_mild` 4,199, `wind_high` 4,648, `weather_affects_game` 5,315, `wind_calm`
5,529, `wind_moderate` 5,667.

**Stated without inflation: this is NOT the same bug as the coverage flag.** That one was scaled into
0.0, which is the code's own word for "no observation" -- an actively false value. These are not made
false. The same snowy game simply gets a different number in week 3 than in week 15, which a tree
model can use, but less cleanly. The normalization exemption is currently one named column wide.
**Widening it means another one-way gold rebuild**, which would move the generation key this phase
recorded and disturb Wave 14's expected change set, so it is an owner decision scheduled before the
re-fit rather than a change made here. Registered in `.planning/WINDOWS.md`.

> **ADDED 2026-09-14 (code review WR-03). The sentence above says the exemption is one named column
> wide. It does not say that SIX MORE COLUMNS OF THE ACTIVELY-FALSE CLASS are still unexempted, and
> that omission is the part worth correcting.** `home_`/`away_availability_coverage`,
> `home_`/`away_injury_coverage` and `home_`/`away_date_modified_coverage` are coverage flags of
> exactly the kind `weather_coverage` was fixed for. Measured on `data/gold/features_ou.parquet`:
> `home_injury_coverage` is a constant `0.0` across 2002-2008, which are genuinely uncovered, while
> 2009-2024 range from `-15.97` to `+0.207`. So an uncovered 2002 row reads `0.0` -- far CLOSER to
> the covered level than to the uncovered one. The same conflation, in the opposite direction.
> **Mitigating, and it is why this is a disclosure rather than an alarm:** all six belong to the
> `injury` group, which Phase 30 DROPPED at train time, so none of them reaches a deployed model
> today. The defect is in the data and in the disclosure, not in a served number. **Nothing is
> decided here** -- the owner decision this section schedules now covers these six as well, and
> `.planning/WINDOWS.md` carries them.

> **ADDED 2026-09-14 (code review WR-09). A second pre-re-fit registration, in the same family of
> "the flag is not the shape it claims to be".** The precipitation band one-hots are a clean ordered
> partition on the archive path and are NOT one on the live-forecast path: `precip_light` fires on
> `0.2 < prob <= 0.5` **or** `0.5 < mm <= 2.0`, and `precip_moderate` on `0.5 < prob <= 0.8` **or**
> `2.0 < mm <= 5.0`, so a forecast of `prob=0.4, mm=3.0` sets BOTH to 1.0. The module's own docstring
> already admits this. What was not recorded anywhere is which path it affects: **only the live
> Open-Meteo forecast carries a probability, so the broken branch is precisely the one that serves
> 2026 predictions, while the branch this phase repaired is the historical one.** It was kept
> byte-unchanged during this phase deliberately, for reproducibility, and that was the right call.
> Leaving it unregistered was not. It is now registered in `.planning/WINDOWS.md` so the one-hot
> family is made a partition on both paths BEFORE the next re-fit rather than after.

### 7j. The opponent adjustment is INERT

Measured 2026-09-14 across all 8,564 per-game rows: `opp_adj_<metric>` equals the raw metric to within
1e-12, because the `has_enough` gate never fires -- the opponent game count arrives NaN and `NaN >= 4`
is False for every row. **The twelve `*_rolling_opp_adj_*` gold columns are therefore rolling averages
of UNADJUSTED EPA. The name says otherwise.** This predates this phase and was deliberately not fixed:
making the adjustment live would move every season 2018-2024, outside the rung declaration. It is also
why the 2025-only restriction on the team-strength family held -- the league average never reaches the
output. A side note worth keeping: the league average is a WHOLE-FRAME mean, so a live adjustment
would additionally be a temporal-leakage question -- a 2018 game adjusted by a scalar that saw 2025.

### 7k. Two read/write divergences, one repaired and one standing

`features/weather.py` was reading a silver table named `weather_forecast` while both ingests write to
`weather`, and a stale 14-row DuckDB table made the mismatch invisible. **Repaired in this phase**, and
it is the mechanical reason the fabricated default was reached for 6,485 of 6,499 gold rows.
`scripts/data_qa.py` still names `weather_forecast` at five sites; that is a QA reporting surface
rather than a model input, it is outside this phase's named scope, and it is recorded here as a
standing finding.

### 7l. N-11, in the weaker wording the evidence supports

The deployed O/U artifact's own `metadata.json` records non-zero XGBoost gain importances on eight
weather features, and a constant column is never split on and scores exactly zero -- so weather was
real in that model's fit window on 2026-03-26. **But the silver upsert is latest-wins and never
deletes a non-matching row, so the 1,942 legacy bronze rows were never promoted into the current
silver file, rather than promoted and then lost.** Say "never promoted". Their provenance cannot be
reconstructed: they carry no `weather_source` column and predate the current venue code.

### 7m. Superseded readings, and where that is recorded

Four readout guards re-ran a harness against gold and their pinned point estimates no longer
reproduce. **Nothing was rewritten.** Each reading stays where it was written with its date, and its
harness-reproduction check now states the gold generation it was measured against, via
`tests/gold_generation.py`. The pre-rung generation is recorded as a NAMED SENTINEL rather than an
invented hex string, because nobody captured a content key of pre-rung gold before the rebuild
overwrote the bytes and the rebuild is one-way -- inventing a plausible-looking digest would be
fabricating exactly the kind of record this phase exists to delete. Recorded in
`tests.phase33_state.GOLD_DERIVED_READINGS` and `GOLD_GENERATION_GATED_CALL_SITES`. **Plan 33-14 is
the next phase to hit this**, and it should reuse the seam rather than reinvent it.

Separately, ten test node ids went red from the rebuild. Every one of them is a test that had
memorised a number measured on the old, wrong data; **none is a defect**, and none was dispositioned
by absorbing it into the tripwire list. Two of the five registered tripwires now fail for a WIDER
fact than the one recorded against them, and the new fact is written down rather than left looking
like the old one.

### 7n. Attribution

Historical weather data is from the **Open-Meteo ERA5 reanalysis archive**, used under
**CC BY 4.0**. Attribution is a licence requirement, not a courtesy.

---

## 8. What Phase 33 should do next

1. **Replan `33-14-PLAN.md` and `33-15-PLAN.md` against section 4 before running either.** That
   replan is a separate act; this document is the input it cannot ignore.
2. Wave 14 re-declares its rung and its expected change set against post-weather-rung gold at
   195/196/195, reuses `tests/gold_generation.py`, and reconciles the deliberately-red width
   reference.
3. Wave 15 consumes `conf.season_partition`, calls `models/trainers/final_fit.py`, re-selects features
   rather than pinning a list, ships WP under the new preprocessing contract so the trained-scaled /
   served-raw defect closes, and labels its verdict in-sample.
4. The nineteen-flag scaling question is an owner decision due before any re-fit.

---

*Phase: 33.1-real-historical-weather-and-training-window-correction*
*Authored: 2026-09-14*
