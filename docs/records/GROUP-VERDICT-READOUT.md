# GROUP-VERDICT-READOUT.md -- the feature-group keep/drop verdict, before and after

**Phase 33.2, Plan 33.2-22. Owner decision D33.2-15. Measured 2026-09-23.**

This document publishes TWO verdicts side by side: the one Phase 30 reached in August 2026, and
the one the SAME frozen rule reaches today on corrected gold under an objective no betting line
enters. Both are shown in full so that a group whose verdict changed is visible, rather than
silently replaced.

**Built on re-measured past seasons; not clean evidence.** Only the 2026 season, recorded live
under the day-before 6 PM ET lock, counts (D33.2-07). See Phase 33.2. What follows selects which
inputs a model may be fitted on. It says nothing whatever about accuracy or profitability.

---

## 1. What a feature-group verdict is, in plain English

The model's input columns are grouped into families. Three of them are screened:

- **injury** -- 12 columns: quarterback-out flags, a backup-quality delta, roster availability
  fractions and their coverage flags.
- **snap** -- 20 columns: how consistent and how concentrated each team's snap counts have been.
- **situational** -- 6 columns: the look-ahead spot, the letdown spot, and coming off a bye.

A verdict answers one question per family: **may the next model fit use it at all?** A KEEP
family stays in the candidate pool. A DROP family is excluded at TRAIN time -- its columns remain
physically in gold, so a verdict is revisitable without rebuilding anything, but no model fitted
under that verdict can select from them.

The verdict is not advisory. `config/group_gate_verdict.toml`'s `excluded_groups` key is read
VERBATIM by `scripts/promote_models.py` at every re-fit. That is exactly why it had to be
re-taken: whatever it says on the day the corrected models are fitted is what shapes them.

**What a verdict does NOT establish.** Nothing about whether the system makes money, nothing
about accuracy against the market, and nothing about any individual column. It is a decision
about which families of inputs a fit may draw on, taken under a rule that was frozen before any
of these numbers existed.

---

## 2. The before and after

Nine cells: three groups, three targets. Every row carries BOTH verdicts, and a row whose two
verdicts differ is marked `FLIPPED`.

The two statistic columns measure DIFFERENT THINGS IN DIFFERENT UNITS and are not comparable
number-to-number. They are compared only through their VERDICTS. Section 3 says what each one is.

| Group | Target | P30 statistic (paired closing-line CLV lift) | P30 corrected p | P30 verdict | Re-measured statistic (paired out-of-sample outcome-loss improvement) | Re-measured corrected p | Re-measured verdict | Flip |
|-------|--------|---:|---:|---|---:|---:|---|---|
| injury | wp | +0.004297 | n/a (excluded) | DROP | -0.003171 | 0.721 | DROP | same |
| injury | ats | -0.295307 | 0.001585 | DROP | -0.030813 | 0.7541 | DROP | same |
| injury | ou | -0.156578 | 0.09541 | DROP | -0.088878 | 0.721 | DROP | same |
| snap | wp | +0.013158 | 2.76e-09 | KEEP | +0.001369 | n/a (excluded) | DROP | FLIPPED |
| snap | ats | -0.240678 | 0.0487 | KEEP | -0.139153 | 0.711 | DROP | FLIPPED |
| snap | ou | -0.227069 | 0.0487 | KEEP | -0.173082 | 0.711 | DROP | FLIPPED |
| situational | wp | +0.001806 | n/a (excluded) | KEEP | +0.000710 | n/a (excluded) | DROP | FLIPPED |
| situational | ats | -0.024307 | n/a (excluded) | KEEP | -0.063872 | 0.721 | DROP | FLIPPED |
| situational | ou | +0.361631 | 0.001585 | KEEP | -0.053285 | n/a (excluded) | DROP | FLIPPED |

`n/a (excluded)` means the cell was EXCLUDED from the multiple-comparison family under the
frozen rule, so it has no corrected p-value. In every case here the reason is the same
pre-registered one: no column of that group was selected by that target's own feature selection,
so the candidate model never saw the group and the delta is selection churn among the other
features, not this group's lift.

**The verdicts, per group:**

| Group | Phase 30 | Re-measured | Changed? |
|-------|----------|-------------|----------|
| injury | DROP | DROP | no |
| snap | KEEP | DROP | YES -- FLIPPED |
| situational | KEEP | DROP | YES -- FLIPPED |

`excluded_groups` therefore moves from `["injury"]` to `["injury", "situational", "snap"]`.

**The DROPs are not all the same kind of DROP, and the difference matters.** Phase 30's injury
DROP was a *significant negative*: the injury family measurably hurt ATS after correction. Every
re-measured DROP is the other arm of the same frozen rule -- **no positive point estimate on any
measured target**. NO CELL in the re-measurement was rejected by the Benjamini-Hochberg
correction in EITHER direction; the smallest corrected p-value in the family is 0.711. Read
plainly: on corrected gold, judged by each model's own out-of-sample loss, none of the three
families showed an improvement worth carrying, and none showed a significant harm either. "It
did not help" is the finding; "it hurt" is not.

The only two positive cells in the re-measurement -- snap/wp at +0.001369 and situational/wp at
+0.000710 -- are both cells where the group's own columns were NOT selected by that target's
feature selection, so the candidate model never saw the group and the frozen rule excludes them
from the family as selection churn rather than lift. That exclusion is the pre-registered rule
doing its job, not a convenient omission. Three cells are excluded on that ground
(snap/wp, situational/wp, situational/ou), which is why m = 6 of the nine.

**Sample and family sizes.** Phase 30 paired 1,019 games per cell with a BH denominator of
m = 6. The re-measurement pairs 570 games per cell, also with m = 6. The smaller sample is a
direct consequence of the honest window: the measured seasons are the derived partition's
holdout after the spent 2025 hold is removed (section 5), which is two seasons rather than four.

### A defect found while building this readout's guard, and closed

The screen's ATS and O/U legs **returned different answers at different OpenMP thread counts**,
by enough to move a verdict. Measured 2026-09-22: on a 12-thread run all three groups came back
DROP; under the 8-thread cap this repository's test sessions apply, all three came back
UNDETERMINED. The WP leg, a logistic regression, was bit-identical throughout. The mechanism is
feature SELECTION -- a fold's fitted importances shift by a hair with the floating-point
reduction order, which moves which columns clear the selector's threshold, which flips a cell
between MEASURED and EXCLUDED.

This repository met the same defect in 2026-09-12 and answered it by deleting four
harness-reproduction tests. That answer was not available here: this run WRITES a verdict that
shapes the corrected re-fit, so "it depends on the machine" is not a limitation to note, it is a
defect to close. **The re-measurement now pins its OpenMP thread count at 1** and records
`thread_limit = 1` in the written document. One is the only setting reproducible on any machine.
Verified: a 12-core plain process and an 8-thread test process now produce bit-identical
selections, deltas and verdicts.

The numbers in the table above are the PINNED ones. What this episode says about the measurement
itself is worth keeping in view: a paired delta between two separately-selected models is a
noisy instrument at this sample size, and the frozen rule's own exclusion for "no column of this
group was selected" exists precisely because of it.

---

## 3. The two objectives, and why the second one exists

**Phase 30's statistic** is a paired incremental **closing-line CLV lift**: for every game with a
closing price, how much closer to the closing line the candidate model's number was than the
baseline model's, averaged over the games both legs scored.

**The re-measured statistic** is a paired **out-of-sample outcome-loss improvement**:
`baseline_loss - candidate_loss` per game, where the loss is each model's OWN primary metric --
WP per-game log loss under the trainer's own probability clip, ATS and O/U per-game absolute
error. A positive number still means the group helped, which is why the frozen rule's KEEP and
DROP arms read the same sign they always did.

**Why the objective changed.** D33.2-03 removed every betting line from every model's inputs,
because for 2018-2025 those lines were CLOSING lines, which did not exist at each game's lock.
Choosing which feature families a model may be fitted on is a fit decision. Judging that choice
by a closing line would put the closing line straight back into the fit through the side door.
So the re-measurement uses no market line of any timing: the odds file is never read, every
trainer is called with no closing-odds frame, and a closing-line column reaching the screen
raises `ClosingLineInScreenError` by name before any leg runs.

The pre-registered rule did not have to move for this. `backtest/group_gate_constants.py` adds
only the multiplicity correction, the effect-size bar and the verdict thresholds, and its own
docstring states that it "re-derives no metric" -- the metric belongs to `backtest/signal_lift.py`.
Changing the objective therefore edited no frozen byte, and the module is byte-unchanged against
its own Phase-30 witness (`tests/unit/test_group_gate_constants_unedited.py`).

**The owned pre-lock line corpus was considered for this job and rejected**, with the reason
recorded rather than left to be rediscovered: the owned `odds_timeline` carries spreads and
totals but no moneyline of any kind, it covers 2020-2024 only, and a market line is the BLEND's
instrument (Plan 33.2-24), where the market enters by design and only after the model has
predicted.

---

## 4. The gold each verdict was measured on

| | Phase 30 | Re-measured |
|---|---|---|
| gold identity | `c3a1177423415ed34b7349ccb8d909c883460a1e5c76b5d2baaa409de1e444ac` | `484397642530db5b28c49d9234ecfb90e3860783f1b1b41766ab6151a1522597` |
| what that digest is | the ACCEPTED rung-4 fingerprint DOCUMENT (`tests.phase30_state.ACCEPTED_RUNG4_FINGERPRINT_SHA256`) | the gold CONTENT generation key (`tests.phase33_state.P332_20_CLEAN_BUILD_GOLD_GENERATION`) |
| shape | 194 / 195 / 194 columns over 6,499 rows | 188 / 188 / 187 columns over 6,516 rows |

**The two digests are different instruments and are not comparable to each other.** One
identifies an accepted fingerprint document; the other is a content digest over the three
matrices' bytes. Each is recorded so that its own verdict can be reproduced against the corpus it
actually ran on -- not so the two can be diffed.

The frozen rule module is the same in both: commit
`dc4d1c0c09ed3b4f5835c801e991aa945f23b479`, unedited.

---

## 5. The seasons each verdict was measured over

| | Selection window | hp-val | Measured (holdout) |
|---|---|---|---|
| Phase 30 | 2018-2019 | 2020 | 2021-2024 |
| Re-measured | 2002-2021 | 2022 | 2023-2024 |

**The re-measurement deliberately does NOT score 2025.** 2025 is the spent single-use hold of the
Phase-31 profitability pre-registration (`backtest.ev_chain_constants.HOLD_SEASONS_P31`); it was
spent once, on a one-shot ledgered run that cannot be repeated. Choosing feature groups by their
2025 score would turn that season into a selection criterion after the fact. The screen's split
config is DERIVED, never typed: the committed season-partition rule is applied to the gold's
completed seasons with the named hold removed, and gold is filtered to that config before any leg
runs, so no 2025 row reaches a trainer at all. Plan 33.2-23 excludes 2025 from its own
adjudication for the same reason, so the feature-group decision and the hyperparameter decision
rest on the same evidence.

The selection window widening from 2018 to 2002 is not this plan's doing: it is D33.2-14, made in
`conf/season_partition.py` at Plan 33.2-18, after the three coverage floors that held it at 2018
all dissolved.

---

## 6. Where the Phase-30 verdict came from, honestly

The Phase-30 column is not a baseline and nothing here is measured against it. It is recorded
because a verdict that is replaced without showing what it replaced is a verdict nobody can audit.

That verdict was derived on 2021-2024 gold that carried **fabricated Elo ratings, hindsight
weather and an inverted coverage flag**, and it chose groups by a **closing line**. The owner's
standing ruling of 2026-09-14 is that corrupted inputs void every model and every gate baseline
resting on them. SPEC R13 forbids comparing a corrected result against a pre-correction one. So
this section is a record of WHAT CHANGED, not a comparison that decides anything -- the
re-measured verdict stands on its own measurement, and would stand identically if the Phase-30
column were blank.

The Phase-30 document itself is untouched and recoverable byte-for-byte from git, at commit
`68eb6425eb379701bf1b9d1937feb8c520877394`. The two guards that read it as the ratified Phase-30
record now read it from that blob rather than from the working-tree file, so the overwrite this
plan performed did not disturb either of them.

---

## 7. What this readout does and does not say

**It says:** on corrected gold, judged by each model's own out-of-sample loss over 2023-2024 with
no market line of any timing, the same pre-registered rule carries none of the three screened
feature families into the corrected re-fit; and two of the three verdicts changed from what
Phase 30 recorded.

**It does not say:** that those families are useless, that the corrected models are better or
worse, that any of this is evidence about 2026, or anything at all about profitability. A DROP
here is "this family did not show an improvement worth carrying under this rule, on this gold,
over these two seasons". Their columns stay in gold and the verdict is revisitable.

**Built on re-measured past seasons; not clean evidence.** Only the 2026 season, recorded live
under the day-before 6 PM ET lock, counts (D33.2-07). See Phase 33.2.

---

*Generated by `python -m scripts.remeasure_group_verdict --apply`.*
*The live verdict it wrote is `config/group_gate_verdict.toml`; its witness is the*
*`P332_22_*` block in `tests/phase33_state.py`.*
