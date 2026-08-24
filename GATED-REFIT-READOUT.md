# GATED-REFIT-READOUT.md -- Phase 30 Re-fit on Widened Gold (Gated) (PROD-01)

**Milestone:** v3.0 Accuracy & Profitability
**Phase:** 30 -- Re-fit on Widened Gold (Gated)
**Authored:** 2026-08-24
**Status:** committed (repo root, sibling of `ACTIVATION-READOUT.md`, `SIGNAL-LIFT-READOUT.md`,
`LINE-MOVEMENT-READOUT.md` and `SELECTION-CENSUS.md`)

> This is the point-in-time record of the Phase-30 gated re-fit: what gold was rebuilt and why,
> what the frozen Stage-1 rule measured, which feature groups it kept and dropped, which targets
> the frozen Stage-2 gate promoted and which it REFUSED, what the O/U monetization chain returns
> against the end state, and what production points at now.
>
> **A null result is a complete result.** This document is published whether zero groups were kept
> and zero targets swapped or not, and it says so in those words. On this run two of three groups
> were kept and ONE of three targets swapped; the other two were refused by the gate and keep their
> incumbents, and the numbers behind those refusals are published here in the same shape as the
> promotion's.
>
> **Vocabulary discipline.** A group that was screened, measured UNDETERMINED, or carried into a
> candidate is NOT thereby a deployment. The word "deployed" is used ONLY of a target whose
> production pointer actually moved. UNDETERMINED and DROP are different findings and are never
> collapsed into one another. Every 2023-2024 monetization figure carries the fixed
> contaminated-readout label from `backtest/ou_monetization.py`, and the burned-holdout over-claim
> words are deliberately absent from this document.
>
> **Nothing published earlier is overwritten.** Superseded readings stay where they were written,
> with their dates and their reasons. `SIGNAL-LIFT-READOUT.md`, `LINE-MOVEMENT-READOUT.md`,
> `ACTIVATION-READOUT.md` and `OU-DIVERGENCE-DIAGNOSIS.md` are unedited by this plan.
>
> ASCII only (no emoji, per CLAUDE.md). Arrows are `->`, dashes are `--`, quotes are straight.

**Headline, in one line.** Gold was rebuilt in four attributed rungs to 194/195/194 columns at
6,499 rows; the frozen Stage-1 rule dropped `injury` and kept `snap` and `situational`; the frozen
Stage-2 non-regression gate promoted **WP** and REFUSED **ATS** and **O/U**, which retain their
incumbents byte-unchanged; the gate baseline was re-frozen twice; the dynamic blend was re-checked
against a copy and production was left alone.

**Suite state at close:** 2282 passed / 7 skipped / 7 xfailed / **0 failed**. It reconciles exactly
against the 2258 / 7 / 7 / 0 state this readout was written from: `+24` are this document's own
doc-drift guard, and the `7 xfailed` is unchanged -- which is the mechanical confirmation that the
seven quarantines in section 10c are still open.

---

## 1. The four-rung rebuild, and what each rung is allowed to have moved

Gold was rebuilt FOUR times, one named cause per rung, each judged mechanically by
`scripts/fingerprint_gold.py --attribute-rung N` against a signature declared before the rung ran.
Rebuilding once for four reasons would have made every moved column unattributable.

| Rung | Named cause | Build window (UTC, 2026-08-22) | wp / ats / ou | rows | added | removed | changed |
|---|---|---|---|---|---|---|---|
| 0 | the published pre-Phase-30 baseline (fingerprint captured, no build) | 02:26:00 | 209 / 210 / 209 | 6,263 | -- | -- | -- |
| 1 | CR-02 -- winsorization erasing a discrete indicator | 02:26:18 -> 02:45:41 (19m 23s) | 209 / 210 / 209 | 6,263 | 0 | 0 | `venue_high_altitude` + the build clock |
| 2 | WR-06 -- imputation medians and q01/q99 bounds refitted on strictly-prior seasons | 03:43:56 -> 04:01:55 (17m 59s) | 209 / 210 / 209 | 6,263 | 0 | 0 | 106 / 107 / 107 |
| 3 | the `line_movement` DROP (SPEC R3) | 14:23:00 -> 14:41:16 (18m 16s) | **194 / 195 / 194** | 6,263 | 0 | 15 / 15 / 15 | `home_win` (dtype only) + the build clock |
| 4 | the N-01 re-sync of the stale silver `games` DuckDB mirror | 16:26:50 -> 16:43:55 (17m 05s) | 194 / 195 / 194 | **6,499** | 0 | 0 | 193 / 194 / 193 |

A fifth full-history build (build B, 16:49:47 -> 17:06:57) is the SPEC R1 reproduction re-run, and
the gold standing on disk is build B's. It reproduces build A in every column digest and every
`column_meta` entry, differing in exactly the three `feature_timestamp` entries -- a per-build
`datetime.now(UTC)` clock, which must move on every rebuild -- and it re-judges to exit 0 on its own
account.

**Attribution verdicts, as run and after the judge's criterion was fixed.**

| Rung | Judge exit AS RUN | Judge exit after the D30-OWNER-08 criterion fix (Plan 30-18) | `unattributed` |
|---|---|---|---|
| 1 | **3** (FINDING, non-blocking) | **0** CLEAN | `[]` |
| 2 | **0** CLEAN | **0** CLEAN | `[]` |
| 3 | **1** (BLOCKING) | **0** CLEAN | `[]` |
| 4 | **0** CLEAN | **0** CLEAN | `[]` |

Rung 3's blocking exit was NOT the drop's fault and was not cleared by an override. Rung 3's
signature required an EMPTY changed set, and two columns changed: the per-build clock, which moves
on every rebuild and therefore made the criterion structurally unsatisfiable, and `home_win`
`float64 -> int32` in `features_wp` only -- a storage-only move, demonstrated by re-encoding the
`int32` values as `float64` and reproducing the rung-2 per-season digest in **24 of 24 seasons**,
with the column still equal to `home_score > away_score` in 6,263 of 6,263 rows. Plan 30-07's
executor REFUSED to edit the judge that was judging that very rung; the owner ruled the criterion
wrong rather than the drop; Plan 30-18 changed the RULE while the DATA it judges stayed
byte-identical (both fingerprints and all three gold parquets re-checked by sha256); and rung 3 was
re-judged ONCE, at exit 0, with `dtype_proof_failed: []` in all three matrices. Plan 30-07's
blocking verdict is retained beside the new one at
`outputs/fingerprints/attribution_rung3_blocked_30-07.json`.

**The uncached-live-upstream confound, stated plainly.** A full-history build reads `nflreadpy`
LIVE with no cache, so an upstream revision landing between two rungs would be indistinguishable
from that rung's own named cause by inspection alone. The timestamps above are published so the
hypothesis stays checkable: rungs 1 and 2 sit 35 minutes apart, the whole ladder spans
2026-08-22T02:26Z to 17:07Z, and `nflreadpy` was **0.1.5** throughout. Rungs 1-3 name upstream
revision as a candidate cause in the judge's own message; **rung 4 offers no upstream escape at
all**, so a single 2021-2024 move at rung 4 would have been a hard block. There was none.

**Every width in this section was counted empirically off the parquet.** The build's own summary
print reports `Features:` low by a constant (190/191/190 at rung 3, 189/190/189 at rung 4) and was
never used.

**One rung-2 finding that outlives its rung.** Rung 2's FIRST attempt attributed perfectly cleanly
-- `ok`, zero unattributed -- while having silently destroyed 18 columns, among them the entire
Phase-28 injury availability family and the entire Phase-29 line-movement family, through a
bound-fitting cascade and a degenerate `q01 == q99` clip. The judge could not see it, because rung
2's signature deliberately attributes every changed column. Only re-measuring per-column health
against the committed pre-Phase-30 fixture found it. Measurement beats attribution, and every later
rung was re-measured the same way: newly constant 0, newly all-NaN 0, newly constant within any
season 0, in all three matrices, at rungs 3 and 4.

---

## 2. The N-01 positive control (SPEC R2), and why it is about two different objects

The control is FAIL-CLOSED: it compares against a divergence measured BEFORE the re-sync, and after
the re-sync that number is unrecoverable. Its expected values therefore live in the git-TRACKED
`tests/phase30_state.py`, not under the gitignored `outputs/` tree, so the assertion still asserts
something on a checkout that never ran the phase. The capture also asserted `divergence > 0` at
capture time: a zero divergence would mean someone had already re-synced, leaving the control with
nothing to prove -- and a control that can be satisfied by doing nothing is not a control.

**Clause 1 -- the silver `games` DuckDB mirror.**

| | before | after |
|---|---|---|
| DuckDB rows | 6,292 | **6,499** |
| parquet rows | 6,499 | 6,499 |
| present only in parquet | 207 | **0** |
| present only in DuckDB | 0 | **0** |

`+207` equals the tracked `N01_DIVERGENCE_BEFORE` exactly. The re-sync dry run re-derived every
field of the original capture independently -- row counts, the per-season and per-week breakdown,
the sorted 207-id list and its sha256 -- so the expected delta is confirmed by a second computation
rather than transcribed from the first. All 207 rows are season 2025, weeks 6-22, and zero rows were
present only in DuckDB: N-01 is a mirror that fell BEHIND, not one that diverged in both directions.

**Clause 2 -- GOLD.**

| | before | after |
|---|---|---|
| gold 2025 rows (all three matrices) | 49 | **285** |
| gold 2025 weeks | 1-4 | **1-22** |

**Gold grew by 236. It did not grow by 207 and was never going to.** Gold's 2025 slice was behind
BOTH copies of silver `games`, because gold's 2025 coverage is bounded by the other silver sources
in the join, not by `games` alone. Writing the control as "gold grew by 207" would have failed, and
the phase would have been halted by its own control misreading its own subject. The separation is
now unreintroducible rather than merely currently-correct: a structural test walks the clause-2
class's own AST and fails if it ever references clause 1's constants.

**The 2021-2024 byte-identity result.** 583 per-column sha256 digests (194 + 195 + 194) over each
matrix's 2021-2024 rows, joined and ordered by `game_id`, through the ONE canonical byte encoding
every fingerprint document in this phase uses, compared by EXACT equality with no tolerance and no
float re-formatting on either side:

| matrix | columns | data columns moved |
|---|---|---|
| `features_wp` | 194 | **0** |
| `features_ats` | 195 | **0** |
| `features_ou` | 194 | **0** |

Exactly one column is excluded by name -- `feature_timestamp`, the per-build clock -- from a single
registry rather than an open-ended list, and the paired assertion is that the moved set **IS** the
clock. A genuine data column moving alongside it fails that test; a clock that did NOT move
(meaning gold was never rebuilt) fails it too.

---

## 3. The `line_movement` DROP: a decision, not an accident of the calendar

At rung 3 the fifteen Phase-29 `line_movement` columns left all three gold matrices. The removed set
is identical in all three and equal to the set DERIVED from `backtest.signal_lift.group_columns` --
computed independently of the judge and compared per matrix, which mattered, because the judge's own
partial-drop check was a subset relation that held by construction and could not fail. Plan 30-18
later replaced it with an EQUALITY check per matrix.

```
line_movement_coverage, opening_spread, opening_total, spread_abs_travel,
spread_drift, spread_drift_dir, spread_late_drift, spread_range, spread_reversals,
total_abs_travel, total_drift, total_drift_dir, total_late_drift, total_range,
total_reversals
```

**The structural reason, which is the whole of it.** The Odds API historical archive floor is
**2020-06-06**. The project's canonical feature-selection window is train **2018-2019**. Every
trainer selects its features on the train seasons ONLY and locks that set for the whole holdout
walk-forward. So across the entire canonical selection window all fifteen columns are exactly
constant, carry zero importance by construction, and **0 of 15 can EVER be selected** into a
candidate model no matter how informative the underlying signal is. A family that cannot enter a
model is not a candidate feature set; it is fifteen columns of width.

**The canonical window was RETAINED.** Phase 30 declined to adopt a covered selection window as its
binding configuration. The window was not mutated to rescue the family.

**The paid archive is intact and was never touched.** Read through `load_dataframe` against the
tracked manifest -- never through `git status data/`, which is vacuous under `.gitignore:22` and
returns empty whether the archive is intact or destroyed:

| | measured | tracked manifest |
|---|---|---|
| rows | **9,957** | 9,957 |
| distinct `(game_id, snapshot_ts)` pairs | **9,957** | 9,957 |
| per-season | 2020: 1,780 / 2021: 1,219 / 2022: 1,452 / 2023: 2,759 / 2024: 2,747 | identical |

A later phase that adopts a covered selection window can rebuild the family from it with no further
spend.

**The DROP does NOT rest on the Phase-29 confound tell.** `LINE-MOVEMENT-READOUT.md` section 6a --
the CR-03 annotation, appended 2026-08-22 BESIDE the Phase-29 record and over nothing -- states that
the tell as implemented checks literal name membership, that its "or anything collinear with it"
half was never built, and that on expanding-window z-scored gold the tell is STRUCTURALLY INERT:
`line_movement_coverage` carries 238 distinct values inside the selection window, Spearman `|rho|`
against the family runs 0.004 to 0.19 against 0.46 for an unrelated column, and any threshold low
enough to fire on the family fires on a dozen unrelated features first. The calendar fact above is
independent of the tell. If the tell had been implemented perfectly and had never fired, the DROP
would be identical; if it had fired on every cell, the DROP would still be identical.

**The Phase-29 screen harness was NOT removed, and now honestly reports the family as absent.**
`line_movement` stays registered in `backtest.signal_lift._GROUP_PREDICATE`, stays in
`features/validation.py`'s leakage keywords, and `features/line_movement.py` stays in the tree with
its builder working and its builder-level test unchanged and green. With the registration retained,
`group_columns` on post-drop gold returns an empty list, every screen's baseline leg automatically
excludes the family if it ever returns, and `python -m backtest.signal_lift --phase 29` is a
NOT-MEASURED run BY CONSTRUCTION rather than a number quietly returned from a group that is no
longer there. Removing the registration would re-arm the 29-06 baseline-composition trap for the
next phase that widens gold.

**Consequence for the Stage-1 grid.** `line_movement` is deliberately absent from `GRID_GROUPS` and
did not enter the BH family. It is the FULL registered set -- `['snap', 'injury', 'situational',
'line_movement']` -- that the Stage-1 baseline leg excludes, so every measured delta in section 5 is
incremental to the non-signal core. That pin was CHECKED against `ALL_REGISTERED_GROUPS` rather than
assumed: the 29-06 incident, in which a published cell moved from `+0.177334` to `-0.195371` purely
through baseline composition, is what makes checking rather than assuming load-bearing here.

---

## 4. The pre-registration: the rule, frozen before any number existed

The binding Stage-1 rule lives in `backtest/group_gate_constants.py` and was frozen in a single
commit that contains no measurement. Every value below is quoted from that module, not restated.

| Item | Frozen value |
|---|---|
| alpha | `ALPHA is backtest.diagnose.SIGNIFICANCE_ALPHA` -- bound by IDENTITY, so a stray float literal cannot pass as a second alpha. Its value is 0.05. |
| MDE | pre-registered as a FORMULA at 80% power, not as a number, so the bar survives the rebuild moving the underlying standard deviations and nothing can be shopped afterwards. 90% power was considered and rejected as widening the UNDETERMINED band without making any KEEP more trustworthy. |
| MDE power | `MDE_POWER = 0.80` |
| Correction | `CORRECTION_METHOD = "benjamini-hochberg"`, vendored as an INCLUSIVE step-up: the largest rank satisfying `p_(i) <= i*alpha/m` is rejected along with every rank below it. |
| BH family and denominator | `BH_DENOMINATOR = "full-grid"`, stated once and in one form -- the family is the FULL 9-cell grid (3 groups x 3 targets), and `m` is the count of MEASURED cells within it, equal to 9 when no cell is excluded. |
| Rank order | ascending raw p, ties broken on the `(group, target)` lexicographic key. The explicit secondary key is load-bearing: numpy's default argsort is unstable, the BH rejection set is tie-order-invariant but the reported RANK is not, and this document publishes ranks. |
| Verdict vocabulary | `KEEP` / `DROP` / `UNDETERMINED` / `NOT MEASURED`. UNDETERMINED resolves to DROP for the DEPLOY decision and is REPORTED as UNDETERMINED, never collapsed into DROP. NOT MEASURED is not a verdict about the group at all; it is a refusal to rule. |
| Verdict arm order | KEEP, DROP, UNDETERMINED, NOT MEASURED -- evaluated in that order, so a group rejected in BOTH directions is KEEP. This reverses Phase 28's D-05 rule and was ratified as a reversal, not adopted silently. |
| Measurement-exclusion rule | four pre-registered reasons, in order: zero columns of the group in gold; paired sample below the CLV sample-size floor; no usable p-value; and no column of the group selected by the target's own feature selection, so the candidate model never saw the group and the delta is selection churn, not lift. |
| Permitted fix-cycle levers | exactly ONE: "D25-05 feature-selection train-window widening (widen the SelectFromModel train window to the incumbent's own selection window, as Phase 25 did for ATS)". At most one documented candidate-side fix-cycle per failing target. |

**The exclusion rule was frozen because it DETERMINES the denominator.** Every excluded cell reduces
`m`, and a denominator chosen after seeing which cells were awkward would be the same rule-shopping
the correction exists to prevent.

**The anti-rule-shopping chain, machine-readable.** SPEC R4's claim is an ancestry relation between
two SPECIFIC commits -- not merely ancestry of the current head -- so both are published:

```
pre_registration_commit: dc4d1c0c09ed3b4f5835c801e991aa945f23b479
measurement_commit: 68eb6425eb379701bf1b9d1937feb8c520877394
```

The pre-registration commit is the LAST commit to touch `backtest/group_gate_constants.py`; it added
the rule, its sibling orchestrator and 45 contract tests, and contains no Stage-1 number. The
measurement commit ADDED `config/group_gate_verdict.toml` and touches **exactly one path**, so its
message's claim to be the measurement commit is checkable rather than merely asserted.
`git merge-base --is-ancestor dc4d1c0 68eb642` exits 0, and the two SHAs are NOT equal -- a rule and
the results it produced landing in one commit is not a pre-registration, it is only a claim of one.
`tests/unit/test_gated_refit_readout_md.py` resolves both from git and asserts exactly this
relation between exactly those two commits.

The frozen rule module was NOT edited at any point after the freeze. Two typos were found in
executing plans' own verification one-liners -- `'NOT_MEASURED'` with an underscore against the
frozen `"NOT MEASURED"` with a space, and `pre_registration_commit` with an underscore after `pre`
against the generator's emitted `preregistration_commit` -- and both are recorded as plan defects,
with the criteria they stood for asserted directly against the frozen constants instead. Editing the
pre-registration after the measurement destroys the evidence rather than fixing it.

---

## 5. The corrected 9-cell grid and the three group verdicts

Measured ONCE on the accepted rung-4 gold, and re-run once more with FULL structural equality on the
parsed JSON as the SPEC R4 determinism control (decision views equal: True; full result equal: True).

**The measurement's frame.**

| Fact | Value |
|---|---|
| Anchor | `BaseTrainer.train_and_evaluate(tune=False)` -- both legs untuned |
| Measure window | 2021-2024 holdout (train 2018-2019, hp-val 2020) |
| Paired n | 1,019 games in every cell (1,139 holdout games less 120 with no closing odds) |
| Baseline leg excludes | `['snap', 'injury', 'situational', 'line_movement']` -- the FULL registered set, checked not assumed |
| alpha / power / correction | 0.05 / 0.80 / benjamini-hochberg, full-grid family |
| **BH denominator actually used** | **m = 6**, not 9 -- three cells were excluded by the pre-registered no-selected-column rule |
| Gold | 6,499 rows; wp 194 / ats 195 / ou 194 columns |

### 5a. The grid

Every number is READ from the binding result document. It is a rendering, not a re-derivation.

| group | target | CLV column | paired n | raw delta | raw two-sided p | MDE | display q | BH rejected | cols present | cols selected | measurability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| injury | wp | `probability_clv` | 1019 | +0.004297 | 3.64587e-15 | 0.001508 | n/a | no | 12 | 0 | EXCLUDED -- no column of this group was selected by this target's own feature selection |
| injury | ats | `line_clv` | 1019 | -0.295307 | 0.000792709 | 0.246065 | 0.00158542 | **yes** | 12 | 1 | measured; in the BH family |
| injury | ou | `line_clv` | 1019 | -0.156578 | 0.0954125 | 0.263074 | 0.0954125 | no | 12 | 1 | measured; in the BH family |
| snap | wp | `probability_clv` | 1019 | +0.013158 | 4.59942e-10 | 0.005863 | 2.75965e-09 | **yes** | 20 | 6 | measured; in the BH family |
| snap | ats | `line_clv` | 1019 | -0.240678 | 0.0372646 | 0.323614 | 0.0486975 | **yes** | 20 | 6 | measured; in the BH family |
| snap | ou | `line_clv` | 1019 | -0.227069 | 0.0405813 | 0.310555 | 0.0486975 | **yes** | 20 | 7 | measured; in the BH family |
| situational | wp | `probability_clv` | 1019 | +0.001806 | 1.22918e-09 | 0.000826 | n/a | no | 6 | 0 | EXCLUDED -- no column of this group was selected by this target's own feature selection |
| situational | ats | `line_clv` | 1019 | -0.024307 | 0.763645 | 0.226628 | n/a | no | 6 | 0 | EXCLUDED -- no column of this group was selected by this target's own feature selection |
| situational | ou | `line_clv` | 1019 | +0.361631 | 0.000756312 | 0.300162 | 0.00158542 | **yes** | 6 | 3 | measured; in the BH family |

95% confidence intervals and BH ranks, same source:

| cell | ci95 lo | ci95 hi | BH rank |
|---|---|---|---|
| injury/wp | +0.00324132 | +0.00535185 | (unranked -- excluded) |
| injury/ats | -0.46749209 | -0.12312242 | 3 |
| injury/ou | -0.34066504 | +0.02750891 | 6 |
| snap/wp | +0.00905559 | +0.01726070 | 1 |
| snap/ats | -0.46712782 | -0.01422772 | 4 |
| snap/ou | -0.44438041 | -0.00975660 | 5 |
| situational/wp | +0.00122794 | +0.00238330 | (unranked -- excluded) |
| situational/ats | -0.18289075 | +0.13427579 | (unranked -- excluded) |
| situational/ou | +0.15159174 | +0.57166999 | 2 |

BH rank order over the m = 6 family: `snap/wp`, `situational/ou`, `injury/ats`, `snap/ats`,
`snap/ou`, `injury/ou`. The q-values above are DISPLAY values; the rejection flag is the decision.

### 5b. The three verdicts, in the three-valued vocabulary

| group | verdict | reason, quoted from the ratified verdict document |
|---|---|---|
| injury | **DROP** | DROP: significantly NEGATIVE after correction on `['ats']`; this group measurably hurt a target and is dropped, not silently retained. |
| snap | **KEEP** | KEEP: significantly positive after Benjamini-Hochberg correction on `['wp']` at alpha=0.05 with m=6 (full-grid family). Carried into the Stage-2 candidate feature set. |
| situational | **KEEP** | KEEP: significantly positive after Benjamini-Hochberg correction on `['ou']` at alpha=0.05 with m=6 (full-grid family). Carried into the Stage-2 candidate feature set. |

The Stage-2 exclusion list is therefore `["injury"]`. It was DERIVED from the ratified verdict
document by `scripts/promote_models._resolve_exclude_groups` with provenance string `verdict`, never
transcribed by hand.

**Nothing was UNDETERMINED and nothing was NOT MEASURED on this run**, so the exclusion list is a
pure DROP list. That is stated rather than passed over: the machinery for keeping UNDETERMINED
distinct from DROP exists, is tested, and simply had no occasion to fire. Had any group landed on
that arm, it would appear in this table as UNDETERMINED and would appear that way in the exclusion
list's header comment beside its own verdict word -- an UNDETERMINED group is excluded from the
Stage-2 candidate feature set for the same practical reason a DROP group is, and the record still
says which of the two happened.

**KEEP does not mean beneficial across the board, and `snap` is the case in point.** `snap` is KEEP
on the strength of its WP cell (BH rank 1) while being BH-rejected NEGATIVE on ATS (-0.2407, rank 4)
and O/U (-0.2271, rank 5). The frozen KEEP-before-DROP arm order resolves the split, and it was
frozen before any number existed; a rule resolving split evidence the other way was available and
was not chosen. Carrying `snap` into Stage 2 therefore carried a group with a measured, significant
negative line-CLV effect on two of three targets. Section 6 records what the per-target gate then
did about that.

### 5c. What the full-grid denominator actually cost, stated honestly

The SPEC's own framing is that with the full 9-cell family the smallest p in the grid must be at or
below roughly **0.00556** (`alpha/9`) for ANY rejection to occur -- which is the stricter of the two
available choices and was chosen for that reason. The REALIZED denominator was **m = 6**, because
three cells were excluded by the pre-registered no-selected-column rule, so the rank-1 bar was
actually `alpha/6 = 0.00833`. **The exclusions made the surviving family EASIER to reject in, not
harder.** The plan text that anticipated this run stated the m = 9 case; that framing was wrong in
the permissive direction, which is the direction that matters, and it is corrected here rather than
quietly restated.

**The single most significant cell in the whole grid was thrown out by the rule.** `injury/wp`
carries `p = 3.6e-15` at `+0.0043`, roughly three times its own MDE -- and ZERO of injury's twelve
columns were selected for WP, so the candidate model never saw the group. Had that cell been
admitted, injury's verdict would have flipped from DROP to KEEP on the strength of a number that
says nothing about injuries. This is the clearest demonstration in the phase of why the exclusion
rule had to be frozen before anyone saw a p-value.

**The feature-cap saturation mechanism, and where it does NOT apply.** Where the selected-feature
cap binds, adding one group's columns DISPLACES others, so a group's add-one-in delta can be
negative even when the group carries information -- the candidate is not "baseline plus the group",
it is "baseline minus whatever the group pushed out, plus the group". Ten of the twelve legs in this
grid do sit at their cap. Two do not, measured from the run's own per-leg selection records:

| leg | max_features | selected | at the cap? |
|---|---|---|---|
| baseline wp | 20 | 20 | yes |
| baseline ats | 25 | **19** | **no** |
| baseline ou | 25 | 25 | yes |
| injury wp / ats / ou | 20 / 25 / 25 | 20 / 25 / 25 | yes / yes / yes |
| snap wp / ats / ou | 20 / 25 / 25 | 20 / 25 / 25 | yes / yes / yes |
| situational wp / ats / ou | 20 / 25 / 25 | 20 / **20** / 25 | yes / **no** / yes |

The consequential one is `situational/ats`: it had five unused slots and still selected zero
situational columns, so for THAT cell the cap is not the mechanism at all -- those columns simply
fell below the importance threshold. Publishing displacement as a blanket explanation would be
publishing something the run's own logs contradict.

### 5d. How to read every per-group delta above (required by the D30-OWNER-13 ruling)

**Feature selection on the 534-row 2018-2019 training window admits synthetic noise columns over
real features. This was measured, not suspected.** The production selection path was run on the real
window with N synthetic unit-variance Gaussian columns appended, drawn independently of the target
and therefore carrying no information about it. The counts below are how many of the SELECTED
features were synthetic (WP selects 20; ATS and O/U select 25 each), recorded after the Plan 30-17
pre-filter, which does not and cannot address this:

| Target | N=10 | N=25 | N=50 |
|---|---|---|---|
| WP (of 20) | 0 | 0 | 1 |
| ATS (of 25) | 1 | 3 | 6 |
| O/U (of 25) | 3 | 2 | 9 |

At N=50, **6 of ATS's 25 and 9 of O/U's 25 selected features were pure noise** -- columns with no
relationship to the target beating real ones on gain importance. The named synthetic columns are
recorded per target in the selection-census run record.

**Therefore: the per-group deltas in section 5a are DIRECTIONAL EVIDENCE, not precise estimates.**
That is the ruling's wording and it is not softened here to "may be sensitive": noise columns were
measured being selected outright. No selection-rule change can fix it, which is why this is a
statement about how the numbers are read rather than a task in a plan -- a column with variance but
no target relationship is indistinguishable from a weak real signal at fit time, and excluding it
would require peeking at the target during selection, which is a different rule and a leakage hazard
of its own. `tests/unit/test_feature_selection_stability.py::test_a_noise_column_is_not_withheld`
pins that boundary rather than asserting past it. Widening the window, changing K and changing the
estimator were all explicitly out of scope.

**One comparability warning travels with the grid.** Plan 30-17 changed the selection pre-filter
before these binding measurements ran. WP's 20 features are unchanged by that change; ATS trains on
9 different features of 25 and O/U on 5 of 25. **Stage 1 is therefore NOT comparable to any
pre-30-17 selection for ATS or O/U.** The ratified rule, quoted rather than reconstructed:

> Fit a scoring model on the columns that **vary over the 2018-2019 training window**; keep the
> **top-K by importance** among those at or above the **mean** importance, K = 20 (WP) / 25 (ATS) /
> 25 (O/U). The cap is what binds -- 36 / 35 / 48 features clear the mean after the change, against
> caps of 20 / 25 / 25 -- so the threshold removes nothing and the effective rule is pure top-K. The
> selected set is now **exactly invariant** to how many zero-variance columns are in the frame, in
> both directions, and remains sensitive to columns that have variance but no signal.

---

## 6. The per-target deploy outcome: one promotion, two refusals

**All three targets get the same table shape, whatever their outcome.** A reader must be able to see
what a refused candidate actually scored, not merely that it was refused. Phase 25's O/U refusal is
the precedent for that: the value of that record is the numbers behind it.

The Stage-2 gate is a per-target NON-REGRESSION gate. It asks "is this candidate not worse than the
incumbent", not "is this candidate positive in absolute terms". All three baselines re-scored 1,087
games; the drift tripwire passed on every target, so the judge and the gold had not diverged since
the first re-freeze. Feature exclusion was applied live from the DERIVED list: 194/195/194 columns
in, 182/183/182 out, `n_cols_dropped = 12` per target.

The 2x2 summary of what production points at now:

| Target | Gate | Production pointer | Moved? |
|---|---|---|---|
| **WP** | **PASS** | `wp_20260824_113325` | **YES -- swapped** |
| ATS | **FAIL** | `ats_20260605_220128` | no -- incumbent retained |
| O/U | **FAIL** | `ou_20260326_163930` | no -- incumbent retained |
| blend | n/a (not gated here) | `blend_dynamic_20260606_020635` | no |

The armed run exited **1**. That is CORRECT reporting for a partial pass, not an error to suppress:
the exit code is driven by whether ANY gated target failed, and two did.

### 6a. WP -- PROMOTED (the phase's one production mutation)

| Metric | Incumbent `wp_20260605_215552` (the re-frozen judge) | Candidate `wp_20260824_113325` | Delta |
|---|---|---|---|
| pooled CLV mean (`probability_clv`) | -0.0441 | **-0.0380** | **+0.006094** paired, p = **0.0142** |
| per-season 2021 | -0.05013683 | -0.04337469 | PASS |
| per-season 2022 | -0.02560076 | -0.02069934 | PASS |
| per-season 2023 | -0.04643749 | -0.03899972 | PASS |
| per-season 2024 | -0.05413238 | -0.04886401 | PASS |
| accuracy | 0.66549605 | 0.67076383 | +0.0024 as gated |
| ECE | 0.05576979 | 0.04570643 | -0.0148 |
| Brier | 0.21813638 | 0.21757963 | -0.0008 |
| **Gate verdict** | -- | -- | **PASS -- all four per-season floors pass, secondary metrics improve** |

**The number a reader is most likely to mistake for the decision.** WP's ABSOLUTE pooled CLV is
-0.0380 (p = 3.07e-49) -- negative. WP still ships, because under `floor_mode = non_regression` the
absolute reading is a recorded readout and never the deploy decision. Removing a closing-line-value
leak is not the same thing as having a positive market edge, and this document keeps those two bars
apart exactly as `ACTIVATION-READOUT.md` does.

The promoted artifact is the artifact the gate scored, demonstrated three ways: the armed run reused
the exact gate-scored staging directories with no retrain; the armed run's full per-target 2x2 block
diffs EMPTY against the reviewed dry run's, to 4dp on every number; and all five files of
`artifacts/wp_20260824_113325` are sha256-identical to the pre-swap staging digests recorded before
the armed run was launched.

A positive side-finding worth recording: the promoted WP model's selected feature set CONTAINS
`home_/away_rolling_snap_share_*` and `away_snap_continuity`. That is direct evidence that the
ratified Stage-1 `snap` KEEP verdict actually reached the model now serving, rather than being a
verdict recorded and then lost between stages.

### 6b. The two REFUSED targets -- ATS and O/U, incumbents RETAINED

Both candidates were measurably worse than what is already serving, and the frozen gate said so
before either could ship. This is a deliberate, honest refusal in the D25-14 lineage -- the same
lineage in which Phase 25's O/U re-fit was refused to protect the stronger line-CLV of the model
already in place. It is not a failure and it is not a partial success dressed as one.

**ATS -- candidate `ats_20260824_113440`, REFUSED; `ats_20260605_220128` retained.**

| Metric | Incumbent `ats_20260605_220128` | Candidate `ats_20260824_113440` | Delta |
|---|---|---|---|
| pooled CLV mean (`line_clv`) | -0.0015 | **-0.2143** | **-0.212797** paired, p = **0.0375** |
| per-season floors | -- | -- | **2021 FAIL** (2022/2023/2024 pass) |
| MAE (secondary) | -- | -- | **+0.8178 against a maximum allowed 0.0** |
| **Gate verdict** | -- | -- | **FAIL -- retained** |

**O/U -- candidate `ou_20260824_113701`, REFUSED; `ou_20260326_163930` retained.**

| Metric | Incumbent `ou_20260326_163930` | Candidate `ou_20260824_113701` | Delta |
|---|---|---|---|
| pooled CLV mean (`line_clv`) | 1.0991 | **0.6121** | **-0.487007** paired, p = **9.24e-12** |
| per-season floors | -- | -- | **2022, 2023 and 2024 FAIL** (2021 passes) |
| MAE (secondary) | -- | -- | **-1.2290 -- the secondary IMPROVED** |
| **Gate verdict** | -- | -- | **FAIL -- retained** |

The O/U row is the D25-14 shape exactly: the candidate's point-error metric got better while its
line-CLV got substantially worse, and the gate weights line-CLV because that is the quantity the
system is trying to beat. A candidate that predicts totals more accurately and prices them worse is
not an improvement for this purpose.

**The refusals landed exactly where Stage 1 predicted, and that is the two-stage design working.**

| Stage-1 cell | Stage-1 measured delta | Stage-2 paired delta | same neighbourhood |
|---|---|---|---|
| `snap`/ats | -0.2407 | ATS -0.2128 | yes |
| `snap`/ou | -0.2271 | O/U -0.4870 | same sign, larger |
| `snap`/wp | **+0.0132** | WP **+0.0061** | same sign, positive |

The Stage-1 keep rule is a per-GROUP verdict rather than a per-cell one, so a group whose evidence
is directionally split is carried whole. The per-target gate is the mechanism that stops a group's
cost on one target riding into production on another target's benefit. Stage 1 measured the cost;
Stage 2 refused it. Two refusals here are the predicted outcome of a ratified rule, not a surprise
to debug.

**Nothing was loosened to rescue either candidate.** No threshold, band, tolerance or frozen value
in `config/gate.toml` was touched, and the phase-start threshold snapshot is green against the end
state: `floor_mode` still `non_regression`, `per_season_must_pass` still true,
`calibration_in_gate` still true, holdout still 2021-2024, both calibration bands untouched.

### 6c. Per-target diagnostics: what tuning actually did, and what the fix-cycle did not

**The Optuna search, recorded on three diagnostic facts.**

| Target | Study name | Storage path | NEW completed trials |
|---|---|---|---|
| wp | `wp_tuning_p30s2` | `outputs/optuna` | **100** |
| ats | `ats_tuning_p30s2` | `outputs/optuna` | **100** |
| ou | `ou_tuning_p30s2` | `outputs/optuna` | **100** |

None carries the v2.0 `{target}_tuning_v1` identity and none resolves under `data/`; the three v2.0
study files are byte- and mtime-identical across the whole phase. A fresh Stage-2 study tag was
opened before the binding run because all three `p30`-tagged studies stood at 100/100 trials and a
binding run on that tag would have added zero trials and raised the zero-new-trials guard at the
moment of the irreversible action. No study file was deleted.

**This document does NOT publish "the candidate's hyperparameters differed from the incumbent's" as
evidence that tuning occurred.** Cross-AI review established that hyperparameter inequality is
neither necessary nor sufficient for a real search: identical parameters can follow a genuine
search that reconverged, and different parameters can follow a resumed at-budget study that searched
nothing. The three facts above -- a study identity no earlier run wrote, a storage path outside
`data/`, and a NEW completed trial count -- are what actually distinguish the two.

**Resolved selection windows.** Each candidate's feature-selection window is derived from its own
incumbent's metadata rather than typed: WP 2018-2019, ATS 2015-2019, O/U 2018-2019.

**The single pre-registered fix-cycle was NOT spent, and deliberately so.**

| Target | Incumbent `train_seasons` | Candidate trained on | Lever available? |
|---|---|---|---|
| ATS | 2015, 2016, 2017, 2018, 2019 | 2015, 2016, 2017, 2018, 2019 | **No -- identical** |
| O/U | 2018, 2019 | 2018, 2019 | **No -- identical** |

The frozen lever list holds exactly one lever -- D25-05 feature-selection train-window widening to
the incumbent's own selection window -- and both failing candidates were ALREADY trained on their
incumbent's exact window, because the promotion seam derives that window from the incumbent's own
metadata, which is precisely what the lever would have done by hand. Re-running either target would
reproduce the same window and the same answer. **The fix-cycle is UNSPENT because the frozen lever
had no unspent move, NOT because it was overlooked.** That distinction is the whole point of
pre-registering the lever list: an unspent budget is only meaningful if the reason it went unspent
is on the record.

**Staged-artifact timestamps, confirming the promoted artifact is the reviewed one.** Confirmed
independently of the dry-run executor's report, against on-disk staged directory mtimes, against the
recorded staged versions, and a third time against what the staleness warning actually printed
during the armed run:

| Target | on-disk staged dir mtime | recorded version / embedded timestamp | printed by the armed run |
|---|---|---|---|
| wp | 2026-08-24 11:33:25 | `wp_20260824_113325` / 2026-08-24 11:33:25 | `wp_20260824_113325 (staged 2026-08-24 11:33:25)` |
| ats | 2026-08-24 11:34:40 | `ats_20260824_113440` / 2026-08-24 11:34:40 | `ats_20260824_113440 (staged 2026-08-24 11:34:40)` |
| ou | 2026-08-24 11:37:01 | `ou_20260824_113701` / 2026-08-24 11:37:01 | `ou_20260824_113701 (staged 2026-08-24 11:37:01)` |

All three match one for one across all three sources. The staleness warning was read, not
suppressed; it printed at its raised severity as designed.

**The end state, asserted rather than assumed.** The per-key manifest comparison is on the RAW LINE,
not the parsed value, so a re-serialization that reordered or reformatted a pointer would be caught
as well as an outright change:

| key | before | after | raw line identical |
|---|---|---|---|
| wp | `wp_20260605_215552` | **`wp_20260824_113325`** | no -- the authorised swap |
| ats | `ats_20260605_220128` | `ats_20260605_220128` | **yes** |
| ou | `ou_20260326_163930` | `ou_20260326_163930` | **yes** |
| blend | `blend_dynamic_20260606_020635` | `blend_dynamic_20260606_020635` | **yes** |

Also asserted: the passing target's directory now exists under the production artifacts root with
all five required files (a pointer swap alone does not relocate files, which was the Phase-25 armed
run's bug); the two failing targets' staged directories were NOT copied into production; all three
incumbent artifact directories are still present, since they are both the paired baseline and the
rollback target; and a second armed run against the same staged artifacts is a no-op, pinned
directly by test rather than inferred from the mechanisms that make it true.

---

## 7. The O/U monetization chain, re-run against the end state

### 7a. Why this ran at all, and what "unchanged" means here

The chain was re-run UNCONDITIONALLY. There is no nothing-changed branch: the runner SCORES the
serving O/U artifact ON gold, and gold was rebuilt four times, so even a fully retained O/U model
produces different residuals, a different frozen residual SD, a different swept EV floor and
different hold ROI numbers than the Phase-27 publication.

**O/U was RETAINED, not swapped.** The run loaded `ou_20260326_163930`, the same artifact Phase 25
retained after its own honest refusal, and the same one section 6b records this phase refusing to
replace.

**What was pre-registered here is the PROCEDURE, never the outputs.** The tune split (2021-2022),
the hold split (2023-2024), the EV-floor grid, the calibration method, the trial-registry schema and
the pre-hold high-total boundary are frozen. The residual standard deviation, the prior-season bias
and the EV floor are RE-FITTED and RE-SELECTED on the tune split against whatever O/U model is
serving, behind the existing leakage fence. Applying a frozen procedure to a legitimately new input
is not a forking path; it is how this repository has always pre-registered. **The EV floor was
deliberately NOT pinned at its Phase-27 value**, because that threshold was selected against a
different residual distribution and pinning it onto a new one would leave it inconsistent with the
standard deviation it is measured against.

**Two supporting facts, confirmed from the run's own output rather than assumed.**

1. The frozen prior-residual season tuple is `(2018, 2019, 2020)`. The serving O/U artifact's own
   metadata records `train_seasons = [2018, 2019]` and `hp_val_seasons = [2020]`. The tuple is
   exactly that artifact's train-plus-validation window, so the walk-forward bias seed for the first
   tune season remains leakage-clean. This holds because the O/U selection window stayed at its
   incumbent's -- the same fact that made the fix-cycle lever unavailable in section 6c.
2. The high-total boundary is market-derived (from closing totals) and therefore model-independent.
   The run's own fit-window assertion re-derived the pre-hold boundary at **48.0**, equal to the
   locked constant, and reports `fence_held: True`.

The modules were re-run, not modified: `git status --porcelain` on
`backtest/ou_monetization.py`, `backtest/ou_ev_chain.py` and `backtest/bet_selector.py` is empty.
The runner's own BH-FDR remains scipy's `false_discovery_control(method="bh")` and was left exactly
as it is -- it is reporting rather than a binding decision there, it is pre-registered as-is, and
harmonizing it onto the Phase-30 vendored step-up would be an unrequested change to a frozen module.

### 7b. The full re-run, beside the Phase-27 publication

**Every 2023-2024 figure in this section is a `provisional contaminated readout`
(`validation_type = PROVISIONAL_CONTAMINATED`).** The 2023-2024 holdout is in-sample contaminated
(D26-09 / D27-01). These are a `proceed signal` or a `redirect` in the module's own fixed
vocabulary, and nothing more.

| Metric | Phase 27 (2026-06-07) | **Phase 30 re-run (2026-08-24)** |
|---|---|---|
| O/U artifact scored | `ou_20260326_163930` | `ou_20260326_163930` -- RETAINED, not swapped |
| frozen EV-floor `t` (re-selected, not pinned) | 0.05 | **0.05** |
| frozen residual SD | 13.151514518753268 | **13.142272073989108** |
| pre-hold high-total boundary | 48.00 | 48.00 |
| BH-FDR trial registry | 5 tested of 5 forks | 5 tested of 5 forks |
| **HEADLINE provisional held-out flat-stake ROI** | **+0.0500** | **+0.004785** |
| n_bets / win_rate | 20 / 0.5500 | **19 / 0.5263** |
| within-holdout block-by-week percentile CI (B=2000, seed=2704, n_blocks=17) | [-0.2967, 0.4067] | **[-0.3636, 0.3788]** |
| Kelly ROI | -- | **+0.1332** |
| robustness cut: regular_season_only | +0.0500 | **+0.004785** |
| robustness cut: playoffs_excluded | +0.0500 | **+0.004785** |
| all applicable cuts positive | True | **True** |
| filtered (acceptance) ROI | +0.0500 | **+0.004785** |
| unfiltered cross-check ROI (n) | +0.0828 (45) | **+0.082828 (45)** |
| surviving high-total OVER count | 9 | **8** |
| surviving UNDER count | 11 | **11** |
| union ROI | +0.0500 | **+0.004785** |
| under-only ROI | +0.3884 | **+0.388430** |
| `union_not_below_under_only` | False | **False** |
| 8-pt-gap kelly_model_prob (is a probability <= 1) | 0.7398 (True) | **0.7406 (True)** |
| odds coverage | n_total 1139, n_with_line 1087, n_excluded 52, coverage 0.9543 | **identical: 1139 / 1087 / 52 / 0.9543459174714662** |

The 2020 robustness cut is N/A-for-hold in both runs: 2020 is not in the hold split, and a
2020-specific cut is never used to bolster a holdout claim.

**CLV is REPORT-ONLY in this chain and never gates a bet.** On the re-run the report-only
model-edge line CLV over the 19 selected bets has mean -1.209117, t = -1.662141, p = 0.113800,
ci95 [-2.737424, +0.319190]. The acceptance bar is graded simulator ROI on the held-out split, not
CLV magnitude.

**What moved, and the honest reading of it.** The under-only arm is unchanged: 11 bets at +0.3884
in both runs. Exactly ONE high-total OVER pick left the selection (9 -> 8), and the union ROI fell
from +0.0500 to +0.004785 while everything else in the chain stayed in place. So the entire headline
movement is one bet in a 19-bet sample, and the block-by-week confidence interval spans zero by a
wide margin in both runs. **The honest statement is that the headline point estimate is not
distinguishable from either its Phase-27 value or from zero at this sample size.** The Phase-27
watch item carries forward unchanged and is if anything sharper: the bankable edge concentrates in
the model's UNDER picks, and the high-total OVER arm drags the union down.

**Where the binding clean verdict now sits.** Phase 27 accepted +0.0500 as its `proceed signal` and
DEFERRED the binding clean profitability verdict to Phase 30. Phase 30 cannot discharge it: the hold
split is still 2023-2024, and 2023-2024 is still the partially burned holdout it was when the
deferral was written, so a Phase-30 reading of it is another provisional contaminated readout rather
than a clean out-of-sample verdict. **The binding clean profitability verdict therefore carries to
Phase 31.** The Phase-27 record is not overwritten -- its deferral stands as written, with this
paragraph beside it. The full re-run is published above precisely so Phase 31 can act on these
figures without re-deriving them, because a re-derivation there would be a second look at a
partially burned holdout.

### 7c. One O/U number that is not what it looks like

The backtest export reports an O/U `headline_clv` of **+45.81043733209909**. **That is NOT a CLV and
must not be read as one.** `BacktestResults.headline_clv` is the mean of the `probability_clv`
column for EVERY target, and for O/U that column is not a line-CLV at all. **O/U's actual `line_clv`
mean is +1.8954569**, which is the expected neighbourhood and is consistent with the pooled 1.0991
the gate works with. This is pre-existing and long known -- `tests/integration/test_backtest_comparison.py`
already uses a value in the +45.77 range as its realistic O/U fixture. It is stated here because
+45.81 is the number a reader hunting for an O/U headline will find first, and publishing it as a
result would be publishing a 40x overstatement of a quantity that does not exist.

A second reading hazard in the same family: `outputs/backtest/predictions_all.csv` is the UNBLENDED
population even under `--blend`, because blending happens after the per-season frames the exporter
reads. No number quoted from that CSV is a blended result.

---

## 8. The gate baseline was re-frozen TWICE, and the second time was a no-op for two targets

SPEC R7 requires the re-freeze be RUN and SHOWN, regardless of whether anything swapped. Both
re-freezes were performed by generator block-paste: the entire `[baseline.*]` section replaced in
ONE operation with the generator's printed output, nothing transcribed, merged or reformatted, and
the committed section asserted byte-identical to the generator's saved output.

**Why the FIRST re-freeze had to precede the gate.** The drift tripwire hard-asserts that a re-score
of the serving incumbents on the CURRENT gold reproduces every frozen baseline field before the
paired test is formed. The rebuild replaced the gold those values were measured on. Measured against
the pre-Phase-30 block, with a tripwire tolerance of 5e-3:

| Target | pooled drift vs the pre-Phase-30 block | would the tripwire have aborted? |
|---|---|---|
| WP | +0.00020226 | **No** -- inside the band |
| ATS | **+0.06695595** | **Yes** -- 13x the band; all four seasons outside, 2021 by +0.16675 |
| O/U | **-0.01046350** | **Yes** -- 2x the band; 2021 and 2023 outside |

The plan text that anticipated this predicted an abort for EVERY target. Measured: two of three. **WP
would have passed the tripwire and gone on to be gated against a baseline measured on gold that no
longer existed** -- the quieter and arguably worse failure, since it produces a verdict rather than
an error. The conclusion (re-freeze BEFORE the gate) is unchanged and better supported by that
correction than by the original framing.

**Why the SECOND re-freeze was required regardless of any swap, and what it demonstrated.** After
the promotion the frozen `[baseline.wp]` described the PREVIOUS serving WP artifact, and two
freshness tests were RED by construction. Both were closed by re-pointing the judge at the end
state, never by moving a tolerance: `_FRESHNESS_TOL` is still 5e-3 and still absolute, and
`_PARITY_TOL` is still 1e-9. **Only the five WP tables moved. The ATS and O/U tables are
byte-identical to the first re-freeze block** -- which IS the SPEC R7 no-op demonstration, delivered
per-target rather than phase-wide, and delivered exactly for the two targets the gate refused.

**The full per-target trajectory across all three freezes.**

Pooled:

| Target | metric | D25-11 (Phase 25) | first re-freeze (30-09) | **end-state re-freeze (30-12)** |
|---|---|---|---|---|
| WP | pooled CLV mean | -0.04429612 | -0.04409386 | **-0.03800034** |
| WP | t | -- | -14.72445755 | **-15.52461299** |
| WP | ci95 | -- | [-0.04996972, -0.03821801] | **[-0.04280319, -0.03319749]** |
| WP | accuracy | 0.66637401 | 0.66549605 | **0.67076383** |
| WP | ECE | 0.05633809 | 0.05576979 | **0.04570643** |
| WP | Brier | 0.21814079 | 0.21813638 | **0.21757963** |
| ATS | pooled CLV mean | -0.06845102 | -0.00149507 | -0.00149507 *(no-op)* |
| ATS | MAE | 8.45712175 | 8.58072427 | 8.58072427 *(no-op)* |
| O/U | pooled CLV mean | 1.10954411 | 1.09908061 | 1.09908061 *(no-op)* |
| O/U | MAE | 10.30555693 | 10.30191577 | 10.30191577 *(no-op)* |

Per-season CLV means:

| Target | season | D25-11 | first re-freeze | **end-state re-freeze** |
|---|---|---|---|---|
| WP | 2021 | -0.05001781 | -0.05013683 | **-0.04337469** |
| WP | 2022 | -0.02652157 | -0.02560076 | **-0.02069934** |
| WP | 2023 | -0.04628999 | -0.04643749 | **-0.03899972** |
| WP | 2024 | -0.05428974 | -0.05413238 | **-0.04886401** |
| ATS | 2021 | 0.35181612 | 0.51856560 | 0.51856560 |
| ATS | 2022 | -0.41936862 | -0.38671918 | -0.38671918 |
| ATS | 2023 | -0.67986029 | -0.63325109 | -0.63325109 |
| ATS | 2024 | 0.47231858 | 0.49400812 | 0.49400812 |
| O/U | 2021 | -1.21316813 | -1.22226360 | -1.22226360 |
| O/U | 2022 | 1.43181741 | 1.43459379 | 1.43459379 |
| O/U | 2023 | 2.56739651 | 2.53407518 | 2.53407518 |
| O/U | 2024 | 1.65331549 | 1.65115056 | 1.65115056 |

**Every per-season `n` is unchanged at 272 / 271 / 272 / 272 across all three targets and all three
freezes.** The holdout population never moved; only values did. The rebuild changed what was
measured, not who was measured.

**Both re-freezes were scoped mechanically, not by eye.** Every changed line in the config diff was
parsed out and mapped to its enclosing TOML table: 94 of 94 changed lines inside `[baseline.*]` on
the first re-freeze, 16 of 16 on the second, and zero under `[gate]`, `[gate.seasons]` or
`[gate.secondary]` either time. That control exists because a contiguous block-replace with an
off-by-one selection would silently absorb a neighbouring threshold line, which is precisely the
move this phase's own prohibition exists to stop.

---

## 9. The dynamic blend was re-checked, and production was left alone

**The hazard is real and historical.** The blend comparison tool's last step reads, mutates and
atomically rewrites the production manifest's `blend` key, and its report directory defaults inside
`data/baselines/`. The Phase-25 run did exactly that. Phase 30 therefore pointed the run at a full
copy of the production artifacts tree (48 entries) with its report directory redirected under
`outputs/`.

**Weights were LOADED from the existing artifact, never re-tuned.** The blender read
`blend_dynamic_20260606_020635`; the copy's newly written artifact carries identical static weights
(wp 0.5916990436565903 / ats 0.545553756120263 / ou 0.6040437178798241) and identical sigmoid
parameters for all three targets. Re-tuning would have been a second, ungated model change landing
in the same phase as the gated re-fit, which would have made the per-target before-and-after in
section 6 uninterpretable.

**The gating outcome against the newly serving models** (n = 1,087 per target):

| Target | static CLV | dynamic CLV | delta | gating verdict |
|---|---|---|---|---|
| WP | -0.0015932 | -0.0017204 | -0.0001272 (-8.0%) | FAIL |
| ATS | +0.0475976 | +0.0475696 | -0.0000281 (-0.1%) | FAIL |
| O/U | +1.1256887 | +1.1696746 | **+0.0439859 (+3.9%)** | PASS |

**A finding recorded and NOT acted on.** That gating would set `mode_by_target = {wp: static, ats:
static, ou: dynamic}`, whereas production serves `{wp: dynamic, ats: dynamic, ou: dynamic}`. Against
the newly serving models, dynamic blending no longer beats static for WP or ATS -- but both margins
are 1.3e-4 and 2.8e-5, so the honest reading is "indistinguishable", not "harmful". **That verdict
landed in the copy and was NOT applied to production.** Whether to act on it is a separate, gated
decision that belongs to a plan with a mandate to change the blend, and it would need the same
paired-significance treatment the model gate uses rather than a bare relative-delta rule.

**The isolation proof, taken from BOTH sides.**

| Check | Result |
|---|---|
| `artifacts/latest.json` sha256 across the run | **unchanged** -- equal to the recorded post-swap digest |
| production blend pointer | `blend_dynamic_20260606_020635` -> **unchanged** |
| new blend directory under `artifacts/` | **none** (3 blend directories before and after) |
| `data/baselines` tree (path-plus-content digest over all 21 entries) | **unchanged** |
| the rewrite actually happened | **yes** -- it landed in the throwaway copy |
| the copy's wp/ats/ou pointers | **equal production's**, so the comparison measured the serving models |
| `data/gold/features_{wp,ats,ou}.parquet` and `data/optuna/*_tuning_v1.db` sha256 | **all six byte-unchanged** |

The second side is deliberate. A test that only asserts "production did not change" passes just as
happily when the rewrite never happened -- a broken comparison, an early return, or a gating outcome
where no target passed. A NEGATIVE CONTROL in the same test module aims the same call at an
unprotected tree and asserts it MUST move it. Without that control the isolation assertion would
demonstrate nothing.

---

## 10. Scope notes, reading hazards, and what is still open

### 10a. What this phase did NOT do

- **The Friday orchestrator was NOT rewired.** The gated re-fit does not now run weekly. Nothing in
  `AUTOMATION.md`'s scheduled path was changed by this phase, and a reader should not assume the
  Stage-1 measurement or the Stage-2 gate is on a schedule. Both are operator-invoked.
- **No target was re-trained, re-tuned or re-promoted after the armed swap.** The manifest digest is
  byte-identical from the swap through the end of the phase.
- **No tolerance, band, threshold or frozen value was widened anywhere.** Three tests that went RED
  by construction when WP was promoted were closed by re-pointing the stale side at the end state,
  never by moving a number.

### 10b. Where the dropped group's columns actually are

The `injury` columns REMAIN PHYSICALLY IN GOLD and are excluded at TRAIN time, not at build time.
Gold is 194/195/194; the Stage-2 candidate trains on 182/183/182, with `n_cols_dropped = 12` per
target, applied live from the exclusion list DERIVED from the ratified verdict document. The same
would be true of an UNDETERMINED group's columns. Nothing was deleted from gold to implement the
Stage-1 DROP, so the verdict can be revisited by a later phase without another rebuild.

`line_movement` is the one family that DID leave gold, and section 3 records that as a separate
decision under SPEC R3 with its own structural grounds.

### 10c. Seven quarantined reproductions remain OPEN

Eight quarantined reproductions were opened during this phase. **Exactly ONE was closed** -- the v2.1
AUDIT-REPORT anchor reproduction, re-ratified after the backtest's tuned train was made genuine and
reproducible, with both readings measured byte-identically. **The other SEVEN remain OPEN**, and the
suite's standing `7 xfailed` is the mechanical confirmation:

| # | Quarantined reproduction | Class | Marker |
|---|---|---|---|
| 1 | `test_diag_diagnosis.py::test_backtest_numbers_match_audit_report` | anchor-vs-DATA | **CLOSED by this phase** |
| 2 | `test_ou_divergence.py::test_deployed_population` | anchor-vs-DATA | xfail(strict) -- OPEN |
| 3 | `test_ou_divergence.py::test_orchestrator_early_exit_is_skip_aware` | anchor-vs-DATA | xfail(strict) -- OPEN |
| 4 | `test_ou_divergence_diagnosis_md.py::test_doc_numbers_reproduce_from_harness` | anchor-vs-DATA | xfail(strict) -- OPEN |
| 5 | `test_line_movement_readout_md.py::test_canonical_ats_cell_reproduces_and_used_no_group_columns` | anchor-vs-HARNESS | xfail(strict) -- OPEN |
| 6 | `test_line_movement_readout_md.py::test_coverage_window_ats_cell_reproduces_and_used_group_columns` | anchor-vs-HARNESS | xfail(strict) -- OPEN |
| 7 | `test_line_movement_readout_md.py::test_headline_ats_cell_reproduces_from_the_covered_selection_window` | anchor-vs-HARNESS | xfail(strict) -- OPEN |
| 8 | `test_line_movement_readout_md.py::test_headline_confound_tell_is_applied_as_pre_registered` | anchor-vs-HARNESS | xfail(strict) -- OPEN |

Rows 5-8 additionally require rewriting `LINE-MOVEMENT-READOUT.md` section 4d, because the
pre-registered confound tell now FIRES and the headline finding's sign flips. That is an owner-grade
correction to a published readout, it must be made in ONE place, and it is a plan's worth of work
that was explicitly not folded into this phase. **Phase 30 does not close believing those seven were
handled.**

**A related second-order point worth keeping.** Reproducibility was FREE while the backtest's
"tuned" train was vacuous: every run returned the same stored parameters, so every number reproduced
perfectly. Making the search honest is what exposed the fragility -- and then fixing the study naming
is what made the reproducibility real rather than vacuous. The run-to-run anchor noise floor is back
to zero, for the right reason this time. The literal value of the backtest's study-name tag is
therefore load-bearing: any change to it silently re-brackets every trial through the pruner, so it
carries a named constant, a source citation and an exact-string test.

### 10d. Open registers, none of which this phase resolved

| Register | Status | One-line statement |
|---|---|---|
| D30-DEFER-04 | **OPEN** | The freshness tolerance is ONE absolute 5e-3 band across two different metrics -- roughly 11% of a WP `probability_clv` value and well under 1% of a typical O/U `line_clv` season value. It was NOT widened, and must not be widened to make a drift go away. |
| D30-DEFER-17 | **OPEN** (in part) | The gate-baseline generator's own EMITTED header still carries stale Phase-20 prose. Correcting it is a generator edit that would break the byte-identity control on every re-freeze and pollute the SPEC R7 no-op diff. The separate narrative prose above that header WAS reconciled, in a comment-only commit demonstrated so by parsed-config equality against HEAD. |
| D30-DEFER-22 | **OPEN** | The prediction cache's swap unlinks the destination immediately before renaming the new cache into place. A crash inside that window leaves NO cache rather than a mixed one -- not the failure SPEC R6 names, but a real availability gap. Not fixed. |
| D30-DEFER-23 | **OPEN** | The blend gating verdict now disagrees with what production serves (section 9). Recorded, not acted on. |
| D30-DEFER-24 | **OPEN** | `scripts/retrain_models.py` still resumes the legacy at-budget Optuna studies, so a "retrain" there searches zero trials and returns 2026-03-31 parameters. This is the same vacuity that was repaired for the backtest, on a caller the ruling did not name. Deliberately untouched, and NOT to be resolved by deleting the v2.0 study files. |
| D30-DEFER-25 | **OPEN** | `predictions_all.csv` is the UNBLENDED population even under `--blend`. Pre-existing, unchanged, and the reason section 7c warns against quoting a CSV number as a blended result. |

### 10e. What this phase is, and is not

It **is** a gated re-fit on rebuilt gold: four attributed rungs, one frozen Stage-1 rule measured
once, one frozen Stage-2 non-regression gate applied to three targets, one promotion, two refusals,
two baseline re-freezes and one blend re-check that touched nothing.

It is **not** a demonstration that the system has a positive market edge. WP's absolute pooled CLV
is still negative. It is **not** a clean out-of-sample profitability verdict on O/U -- section 7
states where that now sits. It is **not** a claim that the kept groups help every target; section 5b
records that `snap` measurably hurt the two targets whose candidates then failed the gate. And it is
**not** a claim that the four-rung rebuild is byte-reproducible in the literal sense; it reproduces
in the strongest form a per-build clock permits, and section 1 says which three entries differ and
why.

---

## 11. The Phase-30 state manifest

`outputs/`, `artifacts/`, `data/` and `.planning/` are ALL gitignored in this repository. Without a
tracked home, the values below would survive in no checkout that did not just run the phase -- and
several of them are not re-derivable at all, because the objects they describe no longer exist in
the state they were measured in.

There are exactly TWO tracked homes: the machine-readable `tests/phase30_state.py`, and this
section. **They publish the SAME values and must agree.** `tests/unit/test_gated_refit_readout_md.py`
asserts that they do, key by key, so the two records cannot drift apart. Section 4's
`pre_registration_commit:` and `measurement_commit:` marker lines are already two of these values,
published there in machine-readable form and repeated here for completeness.

| Constant in `tests/phase30_state.py` | Published value | What it pins |
|---|---|---|
| `N01_PARQUET_ROWS_BEFORE` | `6499` | silver `games` parquet rows, measured before the re-sync |
| `N01_DB_ROWS_BEFORE` | `6292` | silver `games` DuckDB rows, measured before the re-sync |
| `N01_DIVERGENCE_BEFORE` | `207` | the pre-re-sync divergence the SPEC R2 clause-1 control asserts against |
| `N01_MISSING_GAME_IDS_SHA256` | `40d871d75da81616744e95cb5fd82da77273f1e08c17f3cfa5e6f4c9a08bd9bd` | sha256 over the sorted, newline-joined list of the 207 game_ids present only in parquet |
| `GOLD_2025_ROWS_BEFORE_RESYNC` | `49` | gold's 2025 slice before the re-sync -- the clause-2 subject, a DIFFERENT object from clause 1 |
| `ODDS_TIMELINE_ROWS` | `9957` | rows in the paid `odds_timeline` archive |
| `ODDS_TIMELINE_PAIRS` | `9957` | distinct `(game_id, snapshot_ts)` pairs -- every row a distinct pair |
| `ODDS_TIMELINE_SEASON_COUNTS[2020]` | `1780` | per-season archive rows |
| `ODDS_TIMELINE_SEASON_COUNTS[2021]` | `1219` | per-season archive rows |
| `ODDS_TIMELINE_SEASON_COUNTS[2022]` | `1452` | per-season archive rows |
| `ODDS_TIMELINE_SEASON_COUNTS[2023]` | `2759` | per-season archive rows |
| `ODDS_TIMELINE_SEASON_COUNTS[2024]` | `2747` | per-season archive rows |
| `ODDS_TIMELINE_PAIR_LIST_SHA256` | `c4e313a788799b8bfbae5b85169c190101684b155b44e3ea28c9b2c4a0e82a2f` | sha256 over the sorted pair list -- counts can match while content has moved |
| `ACCEPTED_RUNG4_FINGERPRINT_SHA256` | `c3a1177423415ed34b7349ccb8d909c883460a1e5c76b5d2baaa409de1e444ac` | the accepted rung-4 fingerprint DOCUMENT -- the identity of the gold every binding number here was computed on |
| `PRE_REGISTRATION_COMMIT` | `dc4d1c0c09ed3b4f5835c801e991aa945f23b479` | the last commit to touch the frozen Stage-1 rule module |
| `MEASUREMENT_COMMIT` | `68eb6425eb379701bf1b9d1937feb8c520877394` | the single-file commit that added the ratified verdict document |
| `GROUP_VERDICT_FILE_SHA256` | `58da75ac62428a863ecbdaf2fb86078a12db44f3672286dbe89069a6ac7fa36a` | sha256 of `config/group_gate_verdict.toml`'s NEWLINE-NORMALIZED bytes |
| `MANIFEST_SHA256_BEFORE` | `9139e748b5c37167a048b08bba188c74beedbf71309d3ab37cbdd7ca359ebd11` | `artifacts/latest.json` immediately BEFORE the phase's one production write |
| `MANIFEST_SHA256_AFTER` | `7ff78a506b1cb06e206705c5900438a5388be64e963bcc8a39c7ed6a8d0f66f1` | `artifacts/latest.json` immediately AFTER it |
| `DATA_BASELINES_TREE_SHA256` | `49d23f82ff0cdf6b569f5abb5ea30c638d804e4f50946a72429cbdbd38518a29` | path-plus-content digest over the whole `data/baselines` tree, taken before the blend re-check |
| `DEPLOYED_BLEND_VERSION` | `blend_dynamic_20260606_020635` | the blend pointer, recorded separately from the manifest digest so a preserved pointer and a preserved digest are told apart |
| `SLICE_DIGESTS_2021_2024` | 583 digests -- `features_wp` 194, `features_ats` 195, `features_ou` 194 | the per-column 2021-2024 slice digests measured BEFORE the re-sync; the full map lives in the module, and its per-matrix counts are published here |

**Notes a reader needs to reconcile these on another machine.**

- `GROUP_VERDICT_FILE_SHA256` hashes bytes with every CRLF folded to LF. This repository has
  `core.autocrlf=true` and no `.gitattributes`, so the file is LF in the git blob and CRLF in a
  fresh Windows working tree; a raw-byte digest would have pinned a value true only on the machine
  that measured it. Normalized, it equals the sha256 of the committed blob on any platform.
- `MANIFEST_SHA256_BEFORE` / `_AFTER` are over RAW working-tree bytes, because `artifacts/latest.json`
  is gitignored and git never sees it, so the reproducible thing is the bytes the writer actually
  produced. Their LF-normalized equivalents are recorded in the module's own comment for
  reconciling a manifest written on a POSIX host.
- `MANIFEST_SHA256_BEFORE != MANIFEST_SHA256_AFTER` precisely BECAUSE a target swapped. Had zero
  targets passed the gate, those two constants would be EQUAL, and that equality would itself have
  been the record of a zero-swap phase.
- `ACCEPTED_RUNG4_FINGERPRINT_SHA256` identifies the accepted fingerprint DOCUMENT, not the bytes of
  the gold parquets. Two full-history builds on unchanged inputs produce gold identical in every
  column EXCEPT `feature_timestamp`, so no digest of gold itself could be stable across a rebuild,
  and the accepted document is the right thing to pin.

---

## Cross-references

| Document | What it holds that this one does not |
|---|---|
| `ACTIVATION-READOUT.md` | The Phase-25 activation record -- the per-target before-and-after this phase's section 6 follows, and the origin of the retained ATS and O/U incumbents. Unedited by Phase 30. |
| `SIGNAL-LIFT-READOUT.md` | The Phase-28 SCREEN of the same three feature groups, with its own dated drift record. A screen, not a gate ruling; this document is the gate ruling. Unedited by Phase 30. |
| `LINE-MOVEMENT-READOUT.md` | The Phase-29 line-movement record, plus the Phase-30 CR-03 annotation (section 6) that this document's section 3 points at. Unedited by this plan. |
| `OU-DIVERGENCE-DIAGNOSIS.md` | The Phase-26 CLV-to-ROI divergence diagnosis that scoped the Phase-27 monetization chain re-run in section 7. Unedited by Phase 30. |
| `SELECTION-CENSUS.md` | The full feature-selection census behind section 5d, including the ratified selection rule quoted verbatim. |
| `tests/phase30_state.py` | The machine-readable half of section 11's manifest, plus the 583 per-column 2021-2024 slice digests in full. |
| `config/group_gate_verdict.toml` | The ratified Stage-1 verdict as the generator emitted it, block-pasted and committed as the measurement commit. |
| `config/gate.toml` | The frozen judge -- thresholds, floors and the twice-re-frozen `[baseline.*]` block from section 8. |
