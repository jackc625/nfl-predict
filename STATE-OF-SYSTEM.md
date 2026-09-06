# STATE-OF-SYSTEM.md -- v2.1 State of the System (Phase 23)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 23 -- Documentation, Runbook & State-of-System
**Authored:** 2026-05-31

> This is the v2.1 capstone (DOC-03): the single consolidated answer to "what can I
> trust now, what did we fix this milestone, and what is still knowingly deferred?"
> It RECORDS the system's state; it does NOT fix anything. The Deferred section is the
> SINGLE registry of items otherwise scattered across five forensic docs -- each item
> is a one-line pointer to its source anchor, NOT a re-assertion of its detail and NOT
> a patch. Every restated number is a future drift liability, so this doc points at the
> harness-reproducible sources (`AUDIT-REPORT.md`, `MODEL-DIAGNOSIS.md`, `AUTOMATION.md`)
> rather than copying their figures.
>
> Status tags are ASCII `[PASS]` / `[CEILING]` / `[MIXED]` (no emoji, per CLAUDE.md).
>
> HARD BOUNDARY (carried from Phase 20 D-01, the milestone's namesake; D-07): v2.1 is a
> trust / audit / cleanup / diagnosis milestone -- NO model re-fit, NO canonical-gold
> rebuild, no new features/tuning/algorithm swaps, no deployment/hosting. This registry
> RECORDS deferred items; it does NOT fix them. Every item below that is deferred STAYS
> deferred in this phase.
>
> MAINTAINED FORWARD (2026-08-24, Phase 30). The v2.1 hard boundary above is HISTORICAL:
> v3.0 deliberately re-fits and rebuilds, with the per-target deploy gate as the safety
> rail (D-01 lifted). The sections authored 2026-05-31 are kept as the v2.1 reading, with
> their dates; where a v2.1 number has since been superseded it is relabelled in place
> rather than overwritten. The v3.0 additions live in their own dated sections. This
> remains the SINGLE consolidated open list.

---

## Trustworthy now

What v2.1's audit + automation + diagnosis work proved is correct. Each line is a pointer
to its source anchor; the numbers live in the harness-reproducible source, not here.

- **Every pipeline stage runs end-to-end on current data (AUDIT-01) [PASS].** See
  `AUDIT-REPORT.md` "Per-stage pass/fail (as-found)" -- all 7 PIPELINE.md stages [PASS],
  codified in `tests/integration/test_audit_stage_runner.py`. (The 7 is the v2.1-era stage
  count; Phase 25 inserted a Promote stage and `PIPELINE.md` documents 8 stages today. The
  AUDIT-01 verdict is a 2026-05-31 reading over the 7 stages that existed then.)
- **Data integrity + 32-team completeness + canonical mapping (AUDIT-02) [PASS].** See
  `AUDIT-REPORT.md` -- the 3 gold matrices are schema-clean, every on-disk team canonical,
  normalize hard-fails on unknowns. The v2.1 reading of the SHAPE was widths 156/157/156 over
  6263 rows, 2002-2025; that reading is superseded, not wrong-at-the-time. Gold stands at
  **194/195/194 over 6,499 rows, 2002-2025** after the Phase-30 four-rung rebuild
  (`GATED-REFIT-READOUT.md`). The AUDIT-02 verdict itself was not re-run in Phase 30.
- **Feature math + LeakageGate + temporal safety (AUDIT-03/04) [PASS].** See
  `AUDIT-REPORT.md` Wave-3 verdict; hand-traced by `tests/integration/test_audit_trace_*.py`
  (snapshot-then-update Elo, expanding-window normalization, build-time time-fence).
- **Data freshness verified (AUDIT-05) [PASS].** See `AUDIT-REPORT.md` "Data currency /
  freshness" -- 2024 source matches disk; the 2025 partial-season gap is documented (and
  carried as a Deferred item below).
- **Automation runs end-to-end + the scheduled path is verified (AUTO-01..04) [PASS].** See
  `AUTOMATION.md` Sections 6, 8, 9 -- the durable E2E orchestrator guard plus the
  owner-verified scheduled-task `Last Result 0`.
- **The model is honestly diagnosed (DIAG-01..05).** See `MODEL-DIAGNOSIS.md` -- per-target
  verdict WP `[CEILING]`, ATS `[CEILING]`, O/U `[MIXED]`; no methodology flaw on any target.

## Fixed this milestone

The correctness fixes landed during v2.1 (the FIX-01 cluster). Each points at its
`AUDIT-REPORT.md` anchor and its atomic commit; none re-fit a deployed artifact (D-01).

- **CR-01 model-validation-convention cluster** (ModelValidator / health / staleness /
  step_verify converged on `artifacts/latest.json`). See `AUDIT-REPORT.md` "Post-review
  correctness fixes" + the Phase 20-01 commit cluster.
- **Silver de-dup + idempotent writes** (the enabling reproducibility fix, D-13). See
  `AUDIT-REPORT.md` Wave-3 -- commit `25c364f`.
- **Gold-builder twin of the silver partitioned-append bug** (CR-01-gold). See
  `AUDIT-REPORT.md` "Post-review correctness fixes" table -- commit `0269a15`.
- **Market-anchor ET/UTC timezone bug** (snapshot cutoff localized to ET, WR-02). See
  `AUDIT-REPORT.md` "Post-review correctness fixes" table -- commit `dcc7883`.
- **Team-form `--current` data-loss guard** (WR-01). See `AUDIT-REPORT.md` "Post-review
  correctness fixes" table -- commit `8f08163`.
- **Single canonical gold rebuild ADOPTED** (D-02; every metric move explained by the
  clean-weather de-dup driver). See `AUDIT-REPORT.md` "Gold rebuild + metric moves".

## Deferred

The SINGLE consolidated registry of items knowingly deferred at the v2.1 close. Each is a
one-line pointer to its source doc + anchor -- RECORD ONLY, NOT a fix and NOT a
re-assertion. Every item here STAYS deferred in this milestone (D-07).

- See `MODEL-DIAGNOSIS.md` DIAG-05: the gated re-fit of WP/ATS on canonical gold was the
  recommended future model-improvement work -- it has since been EXECUTED in v3.0 Phase 25
  (WP + ATS activated through the per-target non-regression gate; O/U re-fit honestly
  refused and retained on v1.0). No longer deferred; the activation record is
  `ACTIVATION-READOUT.md`. v3.0 Phase 30 then ran the SAME gate again on rebuilt, widened
  gold: WP passed and was promoted, ATS and O/U both FAILED and their incumbents were
  RETAINED, so O/U has now been refused twice. See `GATED-REFIT-READOUT.md`.
- See `AUTOMATION.md` S10 ALERT-WIRING: email + Slack inert (console/log only); 3-part code
  remedy deferred.
- See `AUDIT-REPORT.md` F-LEAK-01: `home_margin` label-sibling excluded from the ATS
  feature_list; capture-only, not fixed.
- See `AUDIT-REPORT.md` F-ZERO-01: dead/bloat all-zero-column schema cleanup deferred.
- See `AUDIT-REPORT.md` F-VALIDATOR-01: run+harden the report-only validator; deferred.
- Legacy `train_*.py` converters coexist with `models/trainers/`; refactor deferred (see
  README.md Current Limitations).
- See `AUTOMATION.md` S5: per-run Friday log history (single-file overwrite today) is a
  deferred future enhancement.
- See `AUDIT-REPORT.md` + `AUTOMATION.md` S10: deferred robustness/hygiene catalog
  (broad-except, SQL string-build, etc.) -- one pointer; do not re-enumerate.

## Phase 30 (2026-08-24): what CLOSED

The v3.0 Phase-30 gated re-fit resolved four items this registry or its source docs carried,
plus two that were listed above and are now removed from the Deferred list. Recorded here so
the removals are attributable rather than silent. The full record is `GATED-REFIT-READOUT.md`.

- **The whole-frame bounds defect -- CLOSED.** Imputation medians and q01/q99 winsorization
  bounds were being fitted across the whole frame. They are now refitted on strictly-prior
  seasons (rebuild rung 2). See the residual below: this does NOT close the within-season
  case.
- **The un-rebuilt indicator fix -- CLOSED.** Winsorization was erasing the discrete
  `venue_high_altitude` indicator, and the fix had never been carried into gold. Rebuild
  rung 1 carried it in, attributed to that one named cause.
- **The stale silver `games` DuckDB mirror -- CLOSED.** The DuckDB copy had fallen 207 rows
  behind the parquet (all season 2025, weeks 6-22; zero rows the other way -- a mirror
  BEHIND, not one diverged in both directions). Re-synced at rung 4 under a fail-closed
  positive control whose expected values live in the git-tracked `tests/phase30_state.py`.
- **The line-movement disposition question -- CLOSED, as a DROP.** The fifteen Phase-29
  `line_movement` columns left all three gold matrices under SPEC R3. The ground is
  structural, not a judgement call about the signal: the Odds API historical archive floor is
  2020-06-06, the canonical feature-selection window is train 2018-2019, so across the entire
  selection window all fifteen columns are exactly constant and 0 of 15 can EVER be selected
  into a candidate. The canonical window was RETAINED rather than mutated to rescue the
  family, the paid archive (9,957 rows) is intact and untouched, and the screen harness stays
  registered so a future phase adopting a covered window can rebuild the family with no
  further spend.
- **The 2025 trailing-coverage gap -- CLOSED.** 2025 was ingested through about week 4 and
  frozen, with backfill deferred to the re-fit milestone. Gold's 2025 slice now carries 285
  rows over weeks 1-22. Note it grew by 236, NOT by the mirror's 207: gold's 2025 coverage is
  bounded by the other silver sources in the join, not by `games` alone, and the two are
  different objects.
- **The `--save` no-op argparse defect -- CLOSED.** `scripts/build_features.py` now defines a
  real `--no-save` (`action="store_false"`, `dest="save"`) beside an explaining comment, so a
  read-only build is possible. The bare `--save` still cannot turn saving off, which is why
  the off switch has to be passed rather than assumed.

## Phase 30 (2026-08-24): what stayed OPEN

Phase 30 does NOT close believing these were handled. Every one is recorded, not fixed.

**The two residuals this phase deliberately leaves open.** Both are consequences of choices
Phase 30 made, and neither is a defect discovered elsewhere:

- **The within-season lookahead that the prior-seasons-only bounds fix does not close.**
  Refitting imputation medians and winsorization bounds on strictly-prior seasons removes the
  cross-season leak. It does NOT establish that a value is fitted only on information
  available before its own kickoff WITHIN its season. The cross-season property is proven
  twice over; the within-season property was NOT measured, and no new analysis was authorised
  to close it. Stated as an accepted residual, not as a solved problem.
- **The dropped group's columns remain physically in gold, with nothing in gold saying so.**
  The Stage-1 DROP of `injury` is enforced at TRAIN time, not at build time: gold is
  194/195/194 and the candidate trains on 182/183/182, `n_cols_dropped = 12` per target,
  applied from the ratified verdict document. Nothing was deleted from gold, which is what
  makes the verdict revisitable without another rebuild -- but a reader inspecting gold sees
  twelve injury columns and no marker recording that they are excluded. The same would be
  true of an UNDETERMINED group's columns.

**Seven quarantined reproductions remain OPEN.** Eight were opened during Phase 30 and
exactly ONE was closed (the v2.1 AUDIT-REPORT anchor, re-ratified after the backtest's tuned
train was made genuine and reproducible). The suite's standing **`7 xfailed`** is the
mechanical confirmation that the other seven are still open -- if that count changes without
a deliberate re-ratification, something was unquarantined by accident. Four of the seven
additionally require an owner-grade correction to a published readout, because a
pre-registered confound tell now FIRES and a published finding's sign flips; that correction
must be made in ONE place and was explicitly not folded into Phase 30.

**Six deferred registers remain OPEN.** One line each; the detail is in
`GATED-REFIT-READOUT.md` section 10d.

- **D30-DEFER-04** -- the gate's freshness tolerance is ONE absolute 5e-3 band across two
  different metrics (roughly 11% of a WP probability-CLV value, well under 1% of a typical
  O/U line-CLV season value). It was NOT widened and must not be widened to make a drift go
  away.
- **D30-DEFER-17** -- the gate-baseline generator's EMITTED header still carries stale
  Phase-20 prose. Correcting it is a generator edit that would break the byte-identity
  control on every re-freeze. The separate narrative prose above that header WAS reconciled.
- **D30-DEFER-22** -- the prediction cache's swap unlinks the destination immediately before
  renaming the new cache in. A crash inside that window leaves NO cache rather than a mixed
  one: a real availability gap, not a corruption one. Not fixed.
- **D30-DEFER-23** -- the blend gating verdict, re-measured against the newly serving models,
  now disagrees with what production serves (it would prefer static for WP and ATS, by
  margins of 1.3e-4 and 2.8e-5 -- indistinguishable, not harmful). Recorded, NOT acted on;
  changing the blend is a separate gated decision.
- **D30-DEFER-24** -- `scripts/retrain_models.py` still resumes the legacy at-budget Optuna
  studies, so a "retrain" there searches zero trials and returns 2026-03-31 parameters. Same
  vacuity that was repaired for the backtest, on a caller the ruling did not name.
  Deliberately untouched, and NOT to be resolved by deleting the v2.0 study files.
- **D30-DEFER-25** -- `outputs/backtest/predictions_all.csv` is the UNBLENDED population even
  under `--blend`. Pre-existing and unchanged; no number quoted from that CSV is a blended
  result.

**One reading hazard worth carrying.** The backtest export reports an O/U `headline_clv` of
+45.81. That is NOT a CLV: `BacktestResults.headline_clv` is the mean of the `probability_clv`
column for every target, and for O/U that column is not a line-CLV at all. O/U's actual
`line_clv` mean is +1.8954569. The +45.81 figure is the one a reader hunting for an O/U
headline finds first, and publishing it as a result would overstate a quantity that does not
exist by roughly 40x.

**One observation this reconciliation surfaced, recorded rather than fixed.** Two silver side
tables did not move with the rung-4 re-sync: `data/silver/elo_rating_history.parquet` holds
6,263 rows and `data/silver/games_with_elo.parquet` holds 6,292, against
`data/silver/games.parquet` at 6,499. Gold is NOT affected -- its 2025 slice carries 285 rows
with zero nulls and 285 distinct values across every Elo column, checked directly -- so this
is a stale intermediate, not a gap in the matrices any model was fitted on. Why those two
tables were not refreshed was NOT established here, and no fix was attempted: Phase 30's data
is frozen and this is out of scope for a documentation reconciliation.

## Phase 31 (2026-09-06): the milestone close, and what it added to this list

The v3.0 Phase-31 productization phase shipped the weekly +EV bet list on a new `/bets` page and
spent the single unburned 2025 season on one pre-registered profitability measurement. **It
deployed no model:** nothing was re-fit or promoted, the dynamic blend was unchanged, and
`artifacts/latest.json` is byte-unchanged by the phase. The full record is
`PROFITABILITY-READOUT.md`.

**Nothing above is closed by Phase 31.** The seven quarantined reproductions and six deferred
registers recorded in the Phase-30 sections remain OPEN and their counts are carried forward
unchanged; the whole-suite xfailed count is HIGHER than the Phase-30 quarantine count because
Phase 31 added its own constant controls on top of it, which is an addition rather than a
change to that count. The verdict itself is a measurement, not a repair.

**Four items this phase ADDS to the open list, pointed at rather than fixed:**

- **The renamed edge band still applies ONE threshold pair to three incompatible units.**
  `utils/edge_tier.py` is now the single collapsed source of the band, replacing two
  byte-equivalent helpers -- a de-duplication asserted value by value against a 23-point snapshot
  taken from both retired helpers before the change. It was RENAMED, not repaired: the same two
  thresholds are still applied to a probability delta (`wp_edge`), a fraction of the absolute
  spread (`ats_edge`) and a fraction of the market total (`ou_edge`), so a "high" WP edge and a
  "high" ATS edge are not comparable quantities. Repairing it would move a published label on `/`
  and `/betting`, and no measurement shows new bands would be better. The stored and rendered
  column names still say `confidence` although the value is an edge band.
- **The backtest-side `ats_edge` in `api/cache.py` carries a sign defect** of the same shape that
  was fixed on the live current-week path this phase, knowingly left in place because it feeds a
  band that `/` and `/betting` render and sort by. Correcting it would move the label on a large
  share of a published population, which deserves its own scope and its own decision.
- **A bet-list row's `clv` is the DECISION-TIME model-edge CLV, never a forward freeze-vs-close
  CLV**, and the forward metric is NOT COMPUTABLE from the current store: the silver odds table
  holds exactly one snapshot per game, and a freeze-vs-close comparison needs two observations of
  the same line. The value is carried honestly under its true meaning and the gap is registered;
  closing it needs a second odds capture per game, not a change to that module.
- **`/betting`'s published ledger has not been regenerated since 2026-08-24**, before this phase
  began and before its spread-Kelly fix landed. The code path was corrected, the artifact was
  not, so the page still serves the pre-fix spread Kelly figure -- deliberately, because the
  phase's own scope rule pins `/betting`'s published figures. Measured, not inferred; the
  consequence is that "the artifact is unmoved" checks made during the phase were comparing a
  file nothing in the phase regenerates.

**Two items already on this list that Phase 31 touched without closing.** The cache-swap
availability gap (D30-DEFER-22) and the gate's single-band freshness tolerance (D30-DEFER-04)
are both still open exactly as recorded above. Neither was widened, narrowed or worked around.

**One stylesheet gap, corrected from an earlier claim.** `lg:grid-cols-7` is ABSENT from
`web/static/css/tailwind-compiled.css`, so `/betting`'s KPI grid is unstyled at the large
breakpoint. An earlier in-phase report recorded it as present; only `max-w-3xl` is. Pre-existing,
recorded rather than fixed.

**Five tests are DELIBERATELY RED and must stay red.** They are tripwires that fired on facts the
owner then accepted -- the gate baseline predating a label correction, and a protected-slice move
-- and a tripwire rewritten to accept the event it fired on stops being a tripwire. Which five,
and why each one, is in `PROFITABILITY-READOUT.md` section 8. An operator who runs the suite and
finds five reds should read that section before concluding anything is broken.

## Cross-references

- **`PROFITABILITY-READOUT.md`** -- the v3.0 Phase-31 milestone close: the per-target 2025
  clean-split profitability verdict, what is deployed and what was retained, the absolute
  closing-line value per target, the nine accepted disclosures, and the explanation of the
  five deliberately-red tests.
- **`MODEL-DIAGNOSIS.md`** -- the Phase 22 honest accuracy diagnosis (per-target verdict +
  the DIAG-05 production-vs-backtest mismatch and gated-re-fit recommendation this registry
  points to). The DOC-03 hard-required link.
- **`ACTIVATION-READOUT.md`** -- the v3.0 Phase-25 activation record: the gated re-fit that
  executed the DIAG-05 recommendation (WP + ATS activated, O/U retained on v1.0), with
  per-target CLV before/after and the pre/post manifest state.
- **`GATED-REFIT-READOUT.md`** -- the v3.0 Phase-30 gated re-fit record: the four attributed
  rebuild rungs, the pre-registered feature-group rule and its three verdicts (`injury`
  DROPPED, `snap` and `situational` KEPT), the per-target deploy outcome (WP promoted, ATS
  and O/U REFUSED and retained), the two gate-baseline re-freezes, the O/U monetization
  re-run, the blend re-check that changed nothing in production, and the full open list this
  registry summarizes above.
- **`AUDIT-REPORT.md`** -- the Phase 20 data & feature correctness audit (AUDIT-01..05 PASS,
  the FIX-01 fixed cluster, and the deferred F-* findings + 2025 gap).
- **`AUTOMATION.md`** -- the Phase 21 Friday-automation explanation (verified end-to-end +
  scheduled path; the ALERT-WIRING and per-run-log-history deferred items).
- **`PIPELINE.md`** -- the canonical 8-stage run sequence the audit and automation wrap
  (7 stages when this registry was authored; Phase 25 inserted Promote).
- **`RUNBOOK.md`** -- the operator runbook: how to perform each operation and tell whether a
  run succeeded.
- **`README.md`** -- the portfolio-facing system narrative and the single architecture
  diagram.
