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

---

## Trustworthy now

What v2.1's audit + automation + diagnosis work proved is correct. Each line is a pointer
to its source anchor; the numbers live in the harness-reproducible source, not here.

- **Every pipeline stage runs end-to-end on current data (AUDIT-01) [PASS].** See
  `AUDIT-REPORT.md` "Per-stage pass/fail (as-found)" -- all 7 PIPELINE.md stages [PASS],
  codified in `tests/integration/test_audit_stage_runner.py`.
- **Data integrity + 32-team completeness + canonical mapping (AUDIT-02) [PASS].** See
  `AUDIT-REPORT.md` -- the 3 gold matrices are schema-clean (widths 156/157/156, 6263 rows,
  2002-2025), every on-disk team canonical, normalize hard-fails on unknowns.
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

- See `MODEL-DIAGNOSIS.md` DIAG-05: gated re-fit of WP/ATS on canonical gold = a future
  model-improvement milestone.
- See `AUTOMATION.md` S10 ALERT-WIRING: email + Slack inert (console/log only); 3-part code
  remedy deferred.
- See `AUDIT-REPORT.md`: 2025 ingested through ~wk4 then frozen; backfill deferred to the
  re-fit milestone.
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

## Cross-references

- **`MODEL-DIAGNOSIS.md`** -- the Phase 22 honest accuracy diagnosis (per-target verdict +
  the DIAG-05 production-vs-backtest mismatch and gated-re-fit recommendation this registry
  points to). The DOC-03 hard-required link.
- **`AUDIT-REPORT.md`** -- the Phase 20 data & feature correctness audit (AUDIT-01..05 PASS,
  the FIX-01 fixed cluster, and the deferred F-* findings + 2025 gap).
- **`AUTOMATION.md`** -- the Phase 21 Friday-automation explanation (verified end-to-end +
  scheduled path; the ALERT-WIRING and per-run-log-history deferred items).
- **`PIPELINE.md`** -- the canonical 7-stage run sequence the audit and automation wrap.
- **`RUNBOOK.md`** -- the operator runbook: how to perform each operation and tell whether a
  run succeeded.
- **`README.md`** -- the portfolio-facing system narrative and the single architecture
  diagram.
