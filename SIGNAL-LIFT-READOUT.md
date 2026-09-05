# Signal-Lift Screen Readout (Phase 28, SIG-05)

**Status:** SCREEN closeout -- this is a CLV-lift SCREEN, not a build and not a gate ruling.
**Scope:** the three Phase-28 feature groups (injury / snap / situational) measured add-one-in
against the activated baseline feature set, per target (WP / ATS / OU).
**Window:** 2021-2024 walk-forward holdout on the Plan 28-06 widened gold (n = 1019 games with
closing odds per leg).
**Baseline composition:** gold MINUS every registered signal-group column
(`ALL_REGISTERED_GROUPS`). Every delta below is incremental to that non-signal core. Stated
explicitly because an add-one-in delta is meaningless without its baseline, and the omission of
this line is what let the drift in Section 0a go unnoticed.
**Measured:** 2026-06-29, against that date's gold. **The grid below is a dated snapshot
measurement, NOT a standing reproduction target** -- see Sections 0a and 0b. v3.0 deliberately
rebuilds gold, so re-running the harness today returns different point estimates. **On 2026-09-05
the RULINGS moved as well, for the first time: on corrected gold all three groups screen DROP.**
The 2026-06-29 grid stands as the Phase-28 record of what was measured on that date against that
gold; Section 0b records what the same harness returns now, and which reading is CURRENT.
**Reproducibility:** every load-bearing number below was reproducible from
`backtest/signal_lift.py` via `run_signal_lift_screen()` on 2026-06-29 gold. The doc-drift guard
`tests/unit/test_signal_lift_readout_md.py` asserts the permanent invariants -- required
sections, ASCII, the screen-not-deploy language, and that the harness's ruling is the ruling this
document records as CURRENT (Section 0b) -- and deliberately does NOT assert a frozen point
estimate against moving gold (D-20; see Sections 0a and 0b).

**Re-run command (owner verification):**

```
.venv\Scripts\python.exe -m backtest.signal_lift
```

---

## 0a. DRIFT RECORD -- the anchor is a snapshot, and here is what moved it

The numbers in Section 1 were measured on 2026-06-29. Re-running the same committed harness on
2026-08-17 gold returns **situational-OU +0.209526** against the recorded **+0.177334**. The
recorded numbers are left standing as the Phase-28 record; this section records the divergence
rather than overwriting them.

**Three causes, each measured rather than assumed:**

1. **Upstream data revision (the dominant cause).** nflreadpy revised 2018-2024 play-by-play
   between 2026-06-29 and 2026-08-16. A column-level diff of the 194 shared columns isolates
   exactly 16 moved columns, every one play-by-play-derived:
   `{home,away}_{off,def}_rolling_opp_adj_*` (max delta 0.198), `{home,away}_qb_adjustment`
   (max 0.039), `{home,away}_backup_quality_delta` (max 2.6e-4). Situational, snap and injury
   columns were byte-identical. The baseline leg re-fits on those columns, so every delta moved.

   | Leg | Gold input | Width | situational-OU |
   |---|---|---|---|
   | A | 2026-08-16 gold, as rebuilt | 209 | +0.239701 |
   | B | 2026-08-16 gold MINUS the 15 line-movement columns | 194 | +0.248666 |
   | C | pre-rebuild Phase-28 gold | 194 | +0.177334 (the recorded anchor) |

2. **Phase-29 input corrections (quick task 260817-dyp).** A live temporal leak was fixed (the
   per-game freeze was never capped at kickoff, so Friday-afternoon kickoffs admitted in-play
   lines), `games.kickoff_et` was normalized to one timezone contract, and the NFL season start
   now derives from Labor Day rather than the first Thursday. The rebuild moved exactly 16
   columns per matrix: 12 line-movement, 3 rest-days, and `feature_timestamp`.

3. **Baseline composition (found 2026-08-17).** The Phase-28 screen excluded the union over
   `GROUPS`, a deny-list of three names, which could not name Phase 29's 15 `line_movement`
   columns -- so they fell into the BASELINE leg. On 2026-08-16 gold this was nearly harmless
   (leg A vs leg B, 0.009), which is why the earlier attribution concluded the widening was not
   the cause. That conclusion was correct for its data and has since been overtaken: after cause
   2 stripped the in-play values out of those columns, the same comparison is worth **0.405**
   (-0.195371 with them in the baseline, +0.209526 with them pinned out) -- enough to flip
   situational-OU from a KEEP to a D-05 veto. Fixed in code: Phase 28 now pins its baseline to
   `ALL_REGISTERED_GROUPS`, so a group registered by any later phase is excluded automatically.

**The ruling was unchanged under every input measured AS OF 2026-08-17. It is no longer.** That
sentence originally read "the ruling is unchanged under every input measured", and asserted that
all three groups still screen KEEP and that the SIG-05 ruling ratified on 2026-06-29 therefore
stands. It was true for the inputs it was written against -- `keep=True` for situational in legs
A, B and C and under the pinned 2026-08-17 run -- and it is FALSE on the 2026-09-05 corrected
gold, where all three groups screen DROP. It is corrected here rather than deleted, so the record
shows both what was claimed and the date it stopped holding. **See Section 0b.**

**A caveat visible only in the current measurement.** Under the 2026-08-17 pinned run, injury-WP
and situational-WP report `grp_cols_used = 0` -- the WP selector locked no column from either
group, so no WP model saw them and those deltas are selection churn rather than lift.
Situational-WP nonetheless returns p = 0.00009 off a group the model never used, which is the
honest scale of this screen's noise floor and an argument for treating every cell here as a
carry decision rather than a result.

**Why the guard no longer asserts the point estimate.** The original assertion assumed re-running
the harness on today's gold reproduces a number measured months ago. That holds only while gold
is frozen, and v3.0 rebuilds gold by design, so the assumption is permanently false. The guard
keeps every invariant that should never move and asserts the recorded RULING still reproduces;
the point estimate is recorded here with its date instead.

---

## 0b. DRIFT RECORD -- 2026-09-05: the first input under which the RULING flips, not the estimate

Section 0a records point estimates moving while the KEEP/DROP rulings held. **This entry is
different in kind, and that is why it gets its own section: this is the FIRST input under which
the ruling itself flips.** Re-running the same committed harness, unchanged, on the 2026-09-05
corrected gold returns:

```
Group screened  group=situational  keep=False  measurable=True
reason=DROP: significantly-negative on ['ou'] (D-05 veto); dropped, not silently retained
situational-OU delta = -0.3203552582994336
```

against the recorded 2026-06-29 anchor of **+0.177334**. The 2026-06-29 numbers in Section 1 are
left standing as the Phase-28 record, exactly as Section 0a leaves its own; this section records
the divergence rather than overwriting it.

### The full grid on 2026-09-05 gold -- all three groups, measured rather than assumed

The whole 3x3 screen was re-run, not only the cell the doc-drift guard watches, because reporting
one flipped cell and leaving the other eight unstated would repeat exactly the omission Section 0a
exists to correct. n_paired = 1087 per cell (2021-2024 holdout; 52 of the window's 1,139 rows lack
closing odds).

| Group | Target | delta (mean) | t | p | D-05 veto |
|---|---|---|---|---|---|
| Injury      | WP  | -0.006596     | -12.641 | 2.9e-34 | **YES** |
| Injury      | ATS | -0.137880     | -1.511  | 0.13116 | no |
| Injury      | OU  | -0.007398     | -0.082  | 0.93485 | no |
| Snap        | WP  | +0.013064     | +7.505  | 1.3e-13 | no |
| Snap        | ATS | -0.499938     | -4.472  | 8.6e-06 | **YES** |
| Snap        | OU  | -0.255886     | -2.558  | 0.01068 | **YES** |
| Situational | WP  | -0.000174     | -0.608  | 0.54358 | no |
| Situational | ATS | +0.216227     | +2.169  | 0.03027 | no |
| Situational | OU  | **-0.320355** | -3.466  | 0.00055 | **YES** |

| Group | 2026-06-29 ruling | 2026-09-05 ruling | Why it moved |
|---|---|---|---|
| Injury      | KEEP | **DROP** | D-05 veto: significantly-negative on WP |
| Snap        | KEEP | **DROP** | D-05 veto: significantly-negative on ATS and OU |
| Situational | KEEP | **DROP** | D-05 veto: significantly-negative on OU |

**All three groups screen DROP on 2026-09-05 corrected gold.** The owner ruling of 2026-09-05
named the situational flip; the injury and snap flips were found by re-running the full grid and
are recorded here on the same principle.

### The cause, attributed by measurement rather than by assumption

**It is NOT a recurrence of the Section-0a baseline-composition bug.** The 2026-09-05 run logs
`baseline_excludes=['snap', 'injury', 'situational', 'line_movement']`, so the
`ALL_REGISTERED_GROUPS` pin from cause 3 above is live and Phase 29's fifteen `line_movement`
columns are correctly held out of the baseline leg. The baseline is composed as intended. That
possibility was checked first, precisely because it was the last cause.

**The cause is the LABELS.** The Plan 31-11 full gold rebuild of 2026-09-05 corrected a real data
defect. Gold keys the Rams canonically as `LA`; silver stored 116 of their odds rows as `LAR`.
Those rows were ORPHANS against gold, so those games fell through to the neutral default and
gold's imputation wrote a literal **0.0** market line. `scripts/build_features.py` then computed
`target_ats = point_differential - snapshot_spread` and `target_ou = total_points -
snapshot_total` against that fabricated zero. **68 games inside this screen's own 2021-2024
holdout were being graded against a FABRICATED 0.0 market line and carried a correspondingly
WRONG ATS and O/U label.** They now carry the real lines and the real labels. The correction is
registered as DEF-31-09 and bound as a disclosure obligation on `PROFITABILITY-READOUT.md`.

**The screen's own paired population corroborates that attribution, game for game.** The
2026-06-29 grid reports n = 1019 per cell; the 2026-09-05 grid reports n = 1087. The difference is
exactly **68**, and it is the same 68 games. The protected window holds 75 `LA`-involved rows
(21 / 17 / 18 / 19 by season -- the counts DEF-31-09 records independently), and 68 of them now
join closing odds where before they could not, because the key they were stored under did not
exist in gold. Games that cannot join closing odds are dropped from the CLV computation outright;
games that joined the fabricated zero contributed wrong labels to both legs. No `LAR`-keyed row
remains in gold, and only 2 rows in the whole 1,139-row protected window still sit at
`snapshot_spread == 0.0` -- neither of them a Rams game.

So the flip is neither noise nor a re-baselining artifact. This screen was previously measured
against partly fabricated labels over a paired population 68 games smaller, and it now is not.

### What follows from this, and what does NOT

- **The Phase-28 screen was a CARRY decision, never a gate ruling.** Section 5 said so when it was
  written and still does. A flipped screen changes what Phase 28 would have carried forward; it
  does not by itself un-carry anything already carried.
- **The BINDING ruling on these groups was Phase 30's, not this document's.** Phase 30's deploy
  gate KEPT `snap` and `situational` and DROPPED `injury`. Those verdicts were measured on gold
  that still held the fabricated Rams lines, so they rest on superseded data. That is registered
  as DEF-31-12, and it is REVISITABLE without a gold rebuild -- the same standing `injury`'s 12
  columns already have, physically present in gold and excluded at train time.
- **Nothing retrains, and no model artifact moves on account of this entry.** The production
  models -- `wp_20260824_113325`, `ats_20260605_220128`, `ou_20260326_163930`, and the dynamic
  blend `blend_dynamic_20260606_020635` -- are frozen and untouched by this reconciliation.
- **The 2025 profitability verdict is unaffected.** It runs those frozen artifacts, so a Phase-28
  screen ruling re-measured over 2021-2024 does not enter it.
- **Re-running Phase 30's binding gate was CONSIDERED and DECLINED** by the owner as outside
  Phase 31's scope. The declining is recorded in DEF-31-12 so it is not later mistaken for an
  oversight.

### Which reading of this document is current

The CURRENT ruling of this screen, on 2026-09-05 corrected gold, is **DROP for all three groups**
-- injury, snap and situational -- each on a D-05 veto. The 2026-06-29 grid in Section 1 and the
KEEP summary in Section 2 are the historical Phase-28 record of what was measured on that date
against that gold. They are NOT the current ruling and must not be quoted as one.

---

## 0. METHOD (the load-bearing correction, review #2 / #3)

The lift is measured by an IN-PROCESS per-season walk-forward add-one-in re-fit
(train <= Y-1, measure Y, for each holdout season Y) via
`BaseTrainer.train_and_evaluate(tune=False)` -- it is NOT whole-frame production-artifact
scoring. A whole-frame re-score of a single candidate trained through ~2023 would grade
2021-2023 IN-SAMPLE for that candidate; that is not the "train <= Y-1, measure Y per holdout
season" walk-forward SIG-05 requires, so every per-target delta below is OUT-OF-SAMPLE.

Per group G and target T:
- the BASELINE leg trains on the activated feature set with EVERY Phase-28 new column removed;
- the CANDIDATE leg adds ONLY group G's new columns (the add-one-in seam, D-02);
- both legs run the SAME walk-forward, so the per-game delta merged on `game_id` is PAIRED;
- the paired delta significance is `clv_significance` (the canonical primitive -- no bespoke
  t-test), read on the target's CLV column (`probability_clv` for WP, `line_clv` for ATS/OU).

The D25-11 re-frozen `config/gate.toml` post-activation baseline remains the DOCUMENTED reference
cross-check (D-04 paired intent preserved); it is not the lift anchor.

CLV unit note: the WP delta is in devigged win-probability units (small magnitude); the ATS/OU
deltas are in line units (spread / total points), so cross-target magnitudes are not comparable
-- only the sign and per-target significance matter for the keep/drop rule.

---

## 1. Per-target incremental-CLV lift grid (all three groups)

**HISTORICAL RECORD, measured 2026-06-29.** These numbers and rulings are left standing as the
Phase-28 record. They are NOT the current ruling -- on 2026-09-05 corrected gold all three groups
screen DROP. See Section 0b.

Each cell is the PAIRED add-one-in delta (candidate minus baseline) over n = 1019 holdout games.
`keep` = point-estimate delta > 0; `veto` = significantly-negative (mean < 0 AND p < 0.05).

### Injury group (coverage: injuries 2009+; measured 2021-2024)

| Target | n_paired | delta (mean) | t       | p        | keep  | veto  |
|--------|----------|--------------|---------|----------|-------|-------|
| WP     | 1019     | +0.001272    | +2.190  | 0.02876  | True  | False |
| ATS    | 1019     | +0.438649    | +4.916  | 0.00000  | True  | False |
| OU     | 1019     | +0.153966    | +1.736  | 0.08292  | True  | False |

**Decision: KEEP** -- positive point-estimate on all three targets, not significantly-negative on
any target. Carried to Phase 30 for the binding gate.

### Snap group (coverage: snaps 2013+; measured 2021-2024)

| Target | n_paired | delta (mean) | t       | p        | keep  | veto  |
|--------|----------|--------------|---------|----------|-------|-------|
| WP     | 1019     | +0.010884    | +5.204  | 0.00000  | True  | False |
| ATS    | 1019     | +0.144580    | +1.390  | 0.16494  | True  | False |
| OU     | 1019     | +0.046731    | +0.483  | 0.62909  | True  | False |

**Decision: KEEP** -- positive point-estimate on all three targets, not significantly-negative on
any target. Carried to Phase 30 for the binding gate.

### Situational group (the NEW look-ahead / letdown / off-bye spots; measured 2021-2024)

| Target | n_paired | delta (mean) | t       | p        | keep  | veto  |
|--------|----------|--------------|---------|----------|-------|-------|
| WP     | 1019     | +0.001622    | +3.881  | 0.00011  | True  | False |
| ATS    | 1019     | +0.214298    | +2.285  | 0.02254  | True  | False |
| OU     | 1019     | +0.177334    | +1.841  | 0.06586  | True  | False |

**Decision: KEEP** -- positive point-estimate on all three targets, not significantly-negative on
any. Carried to Phase 30 for the binding gate. See the weak / priced-in caveat in Section 3 --
this KEEP is a permissive screening signal, NOT evidence of a standing bet angle.

---

## 2. Keep/drop summary + multiplicity note

**HISTORICAL RECORD, measured 2026-06-29.** Superseded as the current ruling by Section 0b, where
all three groups screen DROP on 2026-09-05 corrected gold. Left standing as the Phase-28 record.

| Group       | Decision | Positive targets   | Vetoed targets | Carried to Phase 30? |
|-------------|----------|--------------------|----------------|----------------------|
| Injury      | KEEP     | WP, ATS, OU        | none           | Yes                  |
| Snap        | KEEP     | WP, ATS, OU        | none           | Yes                  |
| Situational | KEEP     | WP, ATS, OU        | none           | Yes                  |

Under the D-05 rule (keep if point-estimate > 0 on >= 1 target AND not significantly-negative on
any target), all three groups screen KEEP and are carried into widened gold for the Phase-30
binding gate. Every cell is positive this round, and no group is significantly-negative on any
target, so none is dropped.

**Multiplicity note (the 3x3 grid).** This is a 3x3 (group x target) screen reported RAW. The
per-target p-values are NOT multiple-comparison-corrected here -- the screen is a permissive
add-one-in filter, and the binding BH-FDR / p < 0.05 correction stays in the Phase-30 deploy
gate (D-05). Across nine cells, several clear a nominal alpha = 0.05 (injury-WP/ATS, snap-WP,
situational-WP/ATS) and some of that is chance alone; treat any single nominally-significant cell
(e.g. snap-WP p < 1e-6, injury-ATS p ~ 1e-6) as a screening signal to CARRY, not as a stand-alone
result. The corrected, binding judgment is Phase 30's job.

---

## 3. Situational caveat (SC3 / D-17) -- honest framing, regardless of screen outcome

The situational group screened KEEP above, but it MUST be read with the standing honesty caveat:
the new look-ahead / letdown / off-bye spots are WEAK and largely PRICED-IN. Short-week, rest,
travel, and bye information is among the most public, most-modeled situational context in the
market; by the closing line the books have already moved on it. These spots are NOT a standing
bet angle on their own. The nominally-significant situational cells (WP, p = 0.00011; ATS,
p = 0.02254) are among nine grid cells and are exactly the kind of result the multiplicity note
above says to carry, not to lean on. The screen KEEP means "worth measuring under the binding gate
in Phase 30", not "this is an edge".

---

## 4. Burned-holdout caveat (D-18e)

This screen measures CLV lift across WP / ATS / OU over the 2021-2024 walk-forward holdout. It is
NOT the O/U ROI gate that partially burned the 2023-2024 holdout in Phase 26 (D26-09). The two are
distinct measurements: this is a paired CLV-delta screen, not a betting-ROI acceptance test.
The burned-holdout concern is NOTED here for continuity but is NOT a blocker for this screen --
the binding O/U ROI judgment, with its holdout-burn accounting, remains a Phase-30 / Phase-32
concern.

---

## 5. Screen-not-deploy framing (D-01 / D-20)

Phase 28 SCREENS; Phase 30 runs the binding deploy gate. The KEEP decisions above mean the kept
groups are screened / carried to Phase 30 for the binding gate -- they are CARRIED into the
widened gold so the Phase-30 gate can judge them. Nothing here is shipped, and no group is
asserted to be an edge on the basis of this screen. Dropped groups (none this round) would be
documented and would never enter gold -- dropped, not silently retained. The honest one-line
summary: all three Phase-28 signal groups cleared the permissive add-one-in screen and are
carried to Phase 30 for the binding gate.
