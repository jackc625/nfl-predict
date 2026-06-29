# Signal-Lift Screen Readout (Phase 28, SIG-05)

**Status:** SCREEN closeout -- this is a CLV-lift SCREEN, not a build and not a gate ruling.
**Scope:** the three Phase-28 feature groups (injury / snap / situational) measured add-one-in
against the activated baseline feature set, per target (WP / ATS / OU).
**Window:** 2021-2024 walk-forward holdout on the Plan 28-06 widened gold (n = 1019 games with
closing odds per leg).
**Reproducibility:** every load-bearing number below is reproducible from
`backtest/signal_lift.py` via `run_signal_lift_screen()`. The doc-drift guard
`tests/unit/test_signal_lift_readout_md.py` runs the orchestrator and compares numbers against
this doc, so the doc cannot silently drift (D-20).

**Re-run command (owner verification):**

```
.venv\Scripts\python.exe -m backtest.signal_lift
```

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
