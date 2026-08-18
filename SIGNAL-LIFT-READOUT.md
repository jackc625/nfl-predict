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
measurement, NOT a standing reproduction target** -- see Section 0a. v3.0 deliberately rebuilds
gold, so re-running the harness today returns different point estimates. The KEEP/DROP rulings
are unchanged under every input measured to date.
**Reproducibility:** every load-bearing number below was reproducible from
`backtest/signal_lift.py` via `run_signal_lift_screen()` on 2026-06-29 gold. The doc-drift guard
`tests/unit/test_signal_lift_readout_md.py` asserts the permanent invariants -- required
sections, ASCII, the screen-not-deploy language, and that the harness still returns the KEEP
ruling recorded here -- and deliberately does NOT assert a frozen point estimate against moving
gold (D-20; see Section 0a).

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

**The ruling is unchanged under every input measured.** `keep=True` for situational in legs A, B
and C, and under the pinned 2026-08-17 run. All three groups still screen KEEP. Only the point
estimates moved, so the SIG-05 ruling ratified on 2026-06-29 stands.

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
