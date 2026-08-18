# Signal-Lift Screen Readout (Phase 28, SIG-05)

**Status:** SCREEN closeout -- this is a CLV-lift SCREEN, not a build and not a gate ruling.
**Scope:** the three Phase-28 feature groups (injury / snap / situational) measured add-one-in
against the activated baseline feature set, per target (WP / ATS / OU).
**Window:** 2021-2024 walk-forward holdout on the Plan 28-06 widened gold (n = 1019 games with
closing odds per leg).
**Baseline composition:** gold MINUS every registered signal-group column
(`ALL_REGISTERED_GROUPS` -- injury, snap, situational and the Phase-29 line_movement family).
Every delta below is incremental to that non-signal core. This line is load-bearing: an add-one-in
delta is only meaningful against a stated baseline, and omitting it here is what let the grid
below become unreproducible once a later phase widened gold (see the restatement note).
**Reproducibility:** every load-bearing number below is reproducible from
`backtest/signal_lift.py` via `run_signal_lift_screen()`. The doc-drift guard
`tests/unit/test_signal_lift_readout_md.py` runs the orchestrator through the same
`screen_kwargs_for_phase(28)` seam the command below uses and compares numbers against this doc,
so the doc, the command and the harness cannot drift apart (D-20).

**Re-run command (owner verification):**

```
.venv\Scripts\python.exe -m backtest.signal_lift
```

---

## 0a. RESTATEMENT (2026-08-17) -- what changed and why

The grid in Section 1 was restated. The original numbers are superseded, not merely refreshed,
and both causes are recorded here rather than quietly swapped:

1. **A baseline that grew underneath the measurement.** The Phase-28 screen excluded the union
   over `GROUPS`, a deny-list of three group names. It could not name Phase 29's fifteen
   `line_movement` columns, so those columns fell through into the BASELINE leg and every
   Phase-28 delta silently became "incremental to a baseline that now contains line movement".
   Re-running the situational-OU cell under that polluted baseline returned **-0.195371**
   (p = 0.04496) against a recorded **+0.177334** -- a D-05 veto where the doc recorded a KEEP,
   entirely from baseline composition. Phase 28 now PINS its baseline to `ALL_REGISTERED_GROUPS`,
   so a group registered by any later phase is excluded automatically.
2. **Corrected upstream inputs.** The NFL season start is now derived from Labor Day rather than
   the first Thursday, and kickoff timestamps were normalized to a single timezone contract.
   Both feed week and rest-day derivation, which is exactly what `look_ahead_spot`,
   `letdown_spot` and `off_bye` are built from. The original grid was therefore computed on
   inputs now known to be wrong; those numbers are not recoverable and should not be.

Re-running a pre-registered statistic once, on corrected inputs, under the unchanged D-05 rule is
a CORRECTION -- it is not a fresh look at the holdout and does not add to the multiplicity burden
in Section 2. What would be illegitimate is searching baseline compositions until one reads well,
which is why the composition is now pinned in code and stated at the top of this document.

**The keep/drop verdicts are unchanged: all three groups still screen KEEP.** Not one of the nine
numbers is unchanged, and one cell (snap-ATS) changed sign. The rule absorbed both without
changing a ruling.

---

## 0. METHOD (the load-bearing correction, review #2 / #3)

The lift is measured by an IN-PROCESS per-season walk-forward add-one-in re-fit
(train <= Y-1, measure Y, for each holdout season Y) via
`BaseTrainer.train_and_evaluate(tune=False)` -- it is NOT whole-frame production-artifact
scoring. A whole-frame re-score of a single candidate trained through ~2023 would grade
2021-2023 IN-SAMPLE for that candidate; that is not the "train <= Y-1, measure Y per holdout
season" walk-forward SIG-05 requires, so every per-target delta below is OUT-OF-SAMPLE.

Per group G and target T:
- the BASELINE leg trains on the activated feature set with EVERY registered signal-group column
  removed -- not just the three Phase-28 groups, but every group in `ALL_REGISTERED_GROUPS`, so
  that a family added by a later phase cannot leak into this baseline (see Section 0a);
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

| Target | n_paired | delta (mean) | t       | p        | keep  | veto  | grp_cols_used |
|--------|----------|--------------|---------|----------|-------|-------|---------------|
| WP     | 1019     | +0.001301    | +2.208  | 0.02750  | True  | False | 0/12          |
| ATS    | 1019     | +0.255399    | +2.579  | 0.01005  | True  | False | 3/12          |
| OU     | 1019     | +0.265034    | +2.642  | 0.00837  | True  | False | 3/12          |

**Decision: KEEP** -- positive point-estimate on all three targets, not significantly-negative on
any target. Carried to Phase 30 for the binding gate. **Read the WP cell as churn, not lift:**
`grp_cols_used` is 0/12, so the WP feature selector locked none of the injury columns and no WP
candidate model ever saw the group. Its +0.001301 is selection churn among the other features
(widening the candidate pool perturbs the fitted importances and so changes which features get
locked), and the KEEP rests on ATS and OU, where the group actually reached the model.

### Snap group (coverage: snaps 2013+; measured 2021-2024)

| Target | n_paired | delta (mean) | t       | p        | keep  | veto  | grp_cols_used |
|--------|----------|--------------|---------|----------|-------|-------|---------------|
| WP     | 1019     | +0.010812    | +5.158  | 0.00000  | True  | False | 6/20          |
| ATS    | 1019     | -0.031835    | -0.322  | 0.74731  | False | False | 4/20          |
| OU     | 1019     | +0.241895    | +2.666  | 0.00780  | True  | False | 3/20          |

**Decision: KEEP** -- positive point-estimate on WP and OU, not significantly-negative on any
target. Carried to Phase 30 for the binding gate. The ATS cell is NEGATIVE this round
(-0.031835) but nowhere near the veto threshold at p = 0.74731, which is the D-05 rule behaving
as designed: a point-estimate that lands slightly the wrong side of zero on one target, with no
evidence behind it, neither earns a KEEP by itself nor blocks one. Read it as "no measurable ATS
lift", not as a negative finding.

### Situational group (the NEW look-ahead / letdown / off-bye spots; measured 2021-2024)

| Target | n_paired | delta (mean) | t       | p        | keep  | veto  | grp_cols_used |
|--------|----------|--------------|---------|----------|-------|-------|---------------|
| WP     | 1019     | +0.001656    | +3.934  | 0.00009  | True  | False | 0/6           |
| ATS    | 1019     | +0.605618    | +7.067  | 0.00000  | True  | False | 2/6           |
| OU     | 1019     | +0.209526    | +2.086  | 0.03723  | True  | False | 3/6           |

**Decision: KEEP** -- positive point-estimate on all three targets, not significantly-negative on
any. Carried to Phase 30 for the binding gate. See the weak / priced-in caveat in Section 3 --
this KEEP is a permissive screening signal, NOT evidence of a standing bet angle. **The WP cell is
churn, not lift:** `grp_cols_used` is 0/6, so no situational column was locked by the WP selector
and no WP candidate model saw the group; its +0.001656 and p = 0.00009 measure selection churn,
NOT a situational effect, and a nominally significant number from a group the model never saw is
a warning about the screen's noise floor rather than a finding. The ATS cell (+0.605618,
p ~ 3e-12, off just 2 of 6 columns) is far larger than a genuinely weak, largely priced-in family
should produce against a public-information baseline, and Section 3 applies to it with MORE force
rather than less.

---

## 2. Keep/drop summary + multiplicity note

| Group       | Decision | Positive targets   | Vetoed targets | Carried to Phase 30? |
|-------------|----------|--------------------|----------------|----------------------|
| Injury      | KEEP     | WP, ATS, OU        | none           | Yes                  |
| Snap        | KEEP     | WP, OU             | none           | Yes                  |
| Situational | KEEP     | WP, ATS, OU        | none           | Yes                  |

Under the D-05 rule (keep if point-estimate > 0 on >= 1 target AND not significantly-negative on
any target), all three groups screen KEEP and are carried into widened gold for the Phase-30
binding gate. Eight of the nine cells are positive, the ninth (snap-ATS) is negative with no
evidence behind it (p = 0.74731), and no group is significantly-negative on any target, so none
is dropped.

**Multiplicity note (the 3x3 grid).** This is a 3x3 (group x target) screen reported RAW. The
per-target p-values are NOT multiple-comparison-corrected here -- the screen is a permissive
add-one-in filter, and the binding BH-FDR / p < 0.05 correction stays in the Phase-30 deploy
gate (D-05). This note carries MORE weight after the restatement, not less: **eight of the nine
cells now clear a nominal alpha = 0.05** (every cell except snap-ATS), where the original grid
had five. A screen in which almost every cell is nominally significant is a screen whose alpha is
doing very little filtering, and reading that as eight independent discoveries would be a
mistake. Treat any single nominally-significant cell -- including situational-ATS at p ~ 3e-12
and snap-WP at p ~ 3e-07 -- as a screening signal to CARRY, not as a stand-alone result.

**Two of those eight are churn, and they calibrate the rest.** Injury-WP and situational-WP both
report `grp_cols_used = 0`: the WP selector locked no column from either group, so no WP model
saw the group, yet both cells came back positive and situational-WP came back at p = 0.00009.
Whatever process produces a p = 0.00009 from a group the model never used is also operating on
the six cells where the group WAS used. That is the honest scale of this screen's noise floor,
and it is the strongest argument in this document for treating every cell here as a carry
decision rather than a result. The corrected, binding judgment is Phase 30's job.

---

## 3. Situational caveat (SC3 / D-17) -- honest framing, regardless of screen outcome

The situational group screened KEEP above, but it MUST be read with the standing honesty caveat:
the new look-ahead / letdown / off-bye spots are WEAK and largely PRICED-IN. Short-week, rest,
travel, and bye information is among the most public, most-modeled situational context in the
market; by the closing line the books have already moved on it. These spots are NOT a standing
bet angle on their own. All three situational cells are now nominally significant (WP,
p = 0.00009; ATS, p ~ 3e-12; OU, p = 0.03723), and that STRENGTHENS the caveat rather than
retiring it. A family this public, measured against a baseline built from the same public
information, should not out-predict the closing line by two thirds of a point on ATS. When a
weak, priced-in signal reports a large edge, the likelier explanations are the measurement and
the multiplicity, not a market oversight that has gone unnoticed. These cells are exactly the
kind of result the multiplicity note above says to carry, not to lean on. The screen KEEP means
"worth measuring under the binding gate in Phase 30", not "this is an edge".

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
