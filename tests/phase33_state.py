"""The git-TRACKED Phase-33 state manifest -- the MEASURED pre-phase failure set.

WHY THIS MODULE EXISTS
----------------------
Phase 33 is eighteen plans long and every one of them asserts something about the suite.
A plan that asserts against a failure set whose members were never enumerated is
asserting against a NUMBER, not against a fact -- and a number can be made to agree by
deleting a test. This module is the enumeration.

IT IS NOT PHASE 32'S. Phase 32 closed on ``5 failed, 3935 passed, 9 skipped, 14 xfailed``
(``tests/phase32_state.POST_PHASE_FAILURE_SET``, 3,963 collected). This checkout was
measured again at Phase-33 plan time and collects 4,029: the 66 extra tests are Phase
32's own Nyquist-validation additions, landed in commit ``6f6af82`` AFTER that manifest's
final slot was appended. Inheriting the 3,963 would have been asserting against the last
number somebody wrote down rather than against the suite that actually exists, which is
the exact failure these manifests are built to stop.

CONSTRAINTS ON THIS MODULE
--------------------------
Constants only. NO project imports, NO I/O and NO logic, so any test at any tier can
import it without cost or side effects. ASCII only. Every value below was MEASURED and
carries the plan, task and date that measured it.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later plan
that needs a new durable value appends a new slot here, under its own separator naming
the plan, the task and the measurement date, rather than inventing a second home:

* Plan 33-01 Task 1 (this file's author) -- the pre-phase failure set, the three-tier
  measurement that produced it, the five deliberate tripwires, the write-guard marker
  name, and the inventory of marked production writers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The pre-phase measurement.
#
# MEASURED 2026-09-11 by Plan 33-01 Task 1 on commit 6f6af82, IN THREE TIERS and
# never as one process: a single whole-suite `pytest -q` is killed for memory on
# this machine. The instrument is part of the measurement, so the split is
# recorded rather than hidden behind a single summed line.
#
#     uv run python -m pytest tests/unit -q
#     uv run python -m pytest tests/integration -q
#     uv run python -m pytest tests/api tests/test_*.py -q
#
# Phase 32's POST_PHASE_FAILURE_SET was measured under the SAME three-tier split,
# so the two lines are comparable in instrument as well as in kind.
# ---------------------------------------------------------------------------

PRE_PHASE_FAILURE_SET: str = "5 failed, 4001 passed, 9 skipped, 14 xfailed"

# 5 + 4001 + 9 + 14. Stated separately so the arithmetic is checkable by a test
# rather than by a reader: a collected count that does not close against the
# failure set means a tier was dropped from the sum.
PRE_PHASE_COLLECTED: int = 4029

# The three tier lines VERBATIM, in the order they were run, each with the wall
# clock that tier reported.
PRE_PHASE_TIER_LINES: tuple[str, ...] = (
    # tests/unit -- 242.19 s
    "2937 passed, 2 skipped, 5 xfailed",
    # tests/integration -- 512.97 s
    "5 failed, 697 passed, 7 skipped, 9 xfailed",
    # tests/api tests/test_*.py -- 116.38 s
    "367 passed",
)

# 242.19 + 512.97 + 116.38 = 871.54 s, rounded DOWN to the whole second the three
# tiers actually spent. Comparable in kind to phase32_state's 849 s, and under the
# SAME instrument -- both are three-tier sums, not single-process runs.
PRE_PHASE_TOTAL_SECONDS: int = 871

# ---------------------------------------------------------------------------
# The five reds that MUST STAY RED.
#
# Copied verbatim from tests/phase32_state.py:113. Each encodes an owner-accepted
# fact from Phase 30 or Phase 31; a phase that turned one of these green would
# have erased a disclosure, not fixed a defect. Two of them ARE the gate-baseline
# disclosure (the frozen baseline diverges from a re-score in 47 of 68 fields and
# was deliberately NOT re-frozen); two more live in the n01 resync control, which
# is also this plan's one legitimate production writer.
# ---------------------------------------------------------------------------

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
# COLD-05: the write guard's marker, and the complete inventory of tests that
# carry it.
# ---------------------------------------------------------------------------

# ONE name, used in ONE place twice: this is the string the guard looks up on an
# item AND the string registered in pyproject.toml's markers list. The two must be
# the same characters, and `--strict-markers` is deliberately OFF in this
# repository (turning it on would surface the unregistered `integration` marker as
# an error across some fifty modules), so a typo in either place would NOT raise --
# it would silently produce a test the guard never exempts, or worse, a marker the
# guard never finds on a test that legitimately writes. A single constant is the
# only mechanical defence available.
WRITES_PRODUCTION_STORE_MARKER: str = "writes_production_store"

# Every test in the suite that carries the marker, as (node_id, declared_paths).
# A live drift test in tests/unit/test_write_guard_marker_scope.py pins the
# collected set to this tuple in BOTH directions: an unrecorded new marker is a
# failure, and so is a marker that quietly disappeared.
#
# MEASURED 2026-09-11 by Plan 33-01 Task 1, by a bracketed three-tier run bisected
# across the 21 modules of tests/integration. This is the repository's ONLY
# legitimate production writer. It writes data/nfl_predictions.duckdb on EVERY run,
# and the write is a LOGICAL NO-OP: the test's own assertion is that the table's
# content digest is unchanged. A DuckDB `replace_mode` table rewrite nevertheless
# moves the FILE's bytes, which is what the content guard sees and why a
# path-scoped exemption -- rather than a sandbox -- is the bounded answer here.
#
# Neither TestEvery2021To2024ValueIsByteIdentical test is marked. Both are
# deliberate tripwires (see DELIBERATE_TRIPWIRE_NODE_IDS) and must keep zero write
# exemption.
MARKED_PRODUCTION_WRITERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "tests/integration/test_n01_resync_control.py::TestTheResyncIsIdempotent::test_a_second_apply_leaves_the_duckdb_table_byte_identical",
        ("data/nfl_predictions.duckdb",),
    ),
)
