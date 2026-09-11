"""The git-TRACKED Phase-32 state manifest -- the measured pre-phase failure set.

WHY THIS MODULE EXISTS
----------------------
"The suite is green" is not a claim this repository can make, and pretending otherwise is
the failure mode this module exists to stop. Five of the six failures on this checkout are
DELIBERATE TRIPWIRES: they encode owner-accepted facts from Phase 30 and Phase 31 -- a
frozen gate baseline that diverges from a re-score and was deliberately NOT re-frozen, a
non-clock column that moved inside a protected season -- and they must STAY RED. A phase
that "fixed" them would be erasing the record, not repairing it.

The sixth is different in kind, and until this phase nobody had named it. Phase 32's own
RESEARCH and VALIDATION both left its identity as an explicit OPEN item. It is the pin's
2026 refusal, and it is the defect this phase is about.

A later plan that asserts against a failure set whose members were never enumerated is
asserting against a number, not against a fact. This module is the enumeration, frozen
where it survives a fresh checkout: ``outputs/`` is gitignored (``.gitignore:26``) and
``.planning/`` is gitignored too (``commit_docs`` is false), so neither a scratch document
nor a plan SUMMARY is a durable home. A tracked constants module under ``tests/`` is --
the same answer Phase 30 reached in ``tests/phase30_state.py`` and Phase 31 in
``tests/phase31_state.py``.

CONSTRAINTS ON THIS MODULE
--------------------------
Constants only. NO project imports, NO I/O and NO logic, so any test at any tier can
import it without cost or side effects. ASCII only. Every value below was MEASURED and
carries the plan, task and date that measured it.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later plan
that needs a new durable value appends a new slot here rather than inventing a second
home:

* Plan 32-01 Task 3 (this file's author) -- the pre-phase failure set, its partition into
  five deliberate tripwires plus the sixth red, the skipped-count divergence reason, the
  stated post-phase failure form, and the measured full-suite runtime.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The two baselines, and why they are NOT the same line.
# ---------------------------------------------------------------------------

# PROVENANCE: .planning/ROADMAP.md:191, the v3.0 milestone close. Recorded here because
# .planning/ is gitignored and the roadmap line does not survive a fresh checkout.
V30_CLOSE_BASELINE: str = "6 failed / 3603 passed / 7 skipped / 14 xfailed"

# MEASURED at Phase-32 plan time by one full `uv run pytest -q` on 2026-09-10. This is the
# line this checkout actually printed -- not the roadmap's, and not an inference from it.
PRE_PHASE_FAILURE_SET: str = "6 failed, 3602 passed, 8 skipped, 14 xfailed"

# The two baselines differ by exactly ONE test, and the cause is named in the suite's own
# evidence-backed-controls footer rather than guessed at.
SKIPPED_COUNT_DIVERGENCE_REASON: str = (
    "tests/unit/test_suppression_freshness.py::TestRejectionTaxonomy::"
    "test_every_reason_has_a_ui_spec_label moved passed -> skipped between the "
    "ROADMAP.md:191 v3.0 close baseline and this checkout: the suite reports it as an "
    "evidence-backed control that did NOT run here because its gitignored run record "
    "under outputs/ is absent. This is a pre-existing environment fact that predates "
    "Phase 32, so every assertion in this phase is made against the MEASURED 8 skipped "
    "and never against the roadmap's 7 -- asserting a count nobody just measured is the "
    "failure mode this constant exists to stop."
)

# ---------------------------------------------------------------------------
# The six failures, transcribed VERBATIM from the 2026-09-10 run's
# `short test summary info`, in the order that summary printed them.
# ---------------------------------------------------------------------------

PRE_PHASE_FAILING_NODE_IDS: tuple[str, ...] = (
    # 1. THE SIXTH RED -- the pin's own 2026 refusal. See SIXTH_RED_NODE_ID below.
    "tests/integration/test_audit_stage_runner.py::TestCurrentWeekIncrementalBuilders::test_build_team_form_current_path_runs",
    # 2. The frozen gate baseline diverges from a re-score in 47 of 68 fields and was
    #    deliberately NOT re-frozen (Phase 31, owner-accepted disclosure).
    "tests/integration/test_gate_baseline_byte_identity.py::TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::test_the_generated_block_equals_the_committed_block_byte_for_byte",
    # 3. A non-clock column moved inside a protected season during the Phase-30 rung-3
    #    full rebuild of the verdict population.
    "tests/integration/test_gold_rebuild_attribution.py::TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::test_no_NON_CLOCK_column_moved_in_a_protected_season",
    # 4. The n01 resync control: not every 2021-2024 data column reproduces its
    #    pre-resync digest exactly.
    "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_every_data_column_reproduces_its_pre_resync_digest_exactly",
    # 5. The n01 resync control's other half: the moved set is not exactly the build clock.
    "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_the_moved_set_is_exactly_the_build_clock",
    # 6. The same 47-of-68-field frozen-baseline divergence, caught at the promotion gate.
    "tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields",
)

# The ONE failure that is not a deliberate tripwire -- identified at Phase-32 plan time,
# after 32-RESEARCH.md and 32-VALIDATION.md both left it as an explicit OPEN item.
#
# From its own traceback, not from inference:
#   tests/integration/test_audit_stage_runner.py:261
#     -> scripts/build_team_form.py:106
#     -> features/team_form.py:663
#     -> features/team_form.py:123
#     -> data/upstream_pin.py:338
#   UpstreamPinMissing: The upstream pin does not cover pbp season(s) 2026.
#                       It covers 2001-2025.
#
# The captured stdout shows `seasons=[2025, 2026]` -- the exact [season-1, season] request
# shape PIN-02 protects, and the reason Plan 32-01 Task 2 partitions the request instead of
# sending the whole list back to nflverse.
SIXTH_RED_NODE_ID: str = (
    "tests/integration/test_audit_stage_runner.py::"
    "TestCurrentWeekIncrementalBuilders::test_build_team_form_current_path_runs"
)

# The other five. Each encodes an owner-accepted fact and MUST STAY RED: a phase that
# turned one of these green would have erased a disclosure, not fixed a defect.
DELIBERATE_TRIPWIRE_NODE_IDS: tuple[str, ...] = (
    # The frozen gate baseline diverges from a re-score in 47 of 68 fields; NOT re-frozen.
    "tests/integration/test_gate_baseline_byte_identity.py::TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::test_the_generated_block_equals_the_committed_block_byte_for_byte",
    # A non-clock column moved in a protected season during the rung-3 rebuild.
    "tests/integration/test_gold_rebuild_attribution.py::TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::test_no_NON_CLOCK_column_moved_in_a_protected_season",
    # n01 resync control: a 2021-2024 data column does not reproduce its pre-resync digest.
    "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_every_data_column_reproduces_its_pre_resync_digest_exactly",
    # n01 resync control: the moved set is wider than the build clock alone.
    "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_the_moved_set_is_exactly_the_build_clock",
    # The 47-of-68-field frozen-baseline divergence, seen from the promotion gate.
    "tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields",
)

# ---------------------------------------------------------------------------
# What this phase is allowed to end on.
# ---------------------------------------------------------------------------

# Form (b) of the two 32-VALIDATION.md permits, with its required reason recorded rather
# than hedged: Phase 32 makes the 2026 refusal CORRECT and ZONE-AWARE -- after this phase
# the same request refuses with a message naming the LIVE zone and
# `python -m scripts.capture_live_season` rather than pointing at the sealed tool -- but it
# does NOT make 2026 covered on an arbitrary checkout. The live capture's parquet bytes live
# under gitignored data/bronze/, so wherever those bytes are absent the refusal is still the
# correct answer and SIXTH_RED_NODE_ID is still correctly red. Clearing it for real is
# Phase 33's (COLD-01..09, the live cold start that runs the current-week builder end to
# end). <N> rises by the count of tests this phase adds; the 8 skipped and 14 xfailed must
# not move.
POST_PHASE_FAILURE_FORM: str = "6 failed / <N> passed / 8 skipped / 14 xfailed"

# MEASURED wall clock of the 2026-09-10 full-suite run (773.44 s), rounded UP. Fills the
# "UNMEASURED at plan time" Test Infrastructure row in 32-VALIDATION.md with a number
# somebody actually ran.
FULL_SUITE_RUNTIME_SECONDS: int = 774
