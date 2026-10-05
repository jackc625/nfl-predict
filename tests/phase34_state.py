"""The git-TRACKED Phase-34 state manifest -- the MEASURED expected failure set.

WHY THIS MODULE EXISTS
----------------------
Phase 34 builds an append-only forward bet ledger across many plans, and every plan
that reports on the suite has to name the failures it EXPECTS, never a bare count. A
count can be made to agree by deleting a test; an enumerated, measured set cannot. This
module records that set once, before any Wave-2 plan lands, so later plans cite it by
name instead of re-deriving it from memory.

The five deliberate tripwires are NOT re-listed here. Their one home is
``tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS``; a second copy would drift, and the
copy a later gate read would silently win (see the docstring of
``tests/integration/test_phase33_expected_failure_set.py``). This module records only
that the set was RUN on the measured commit and how many of its members failed.

CONSTRAINTS ON THIS MODULE
--------------------------
Constants only. NO project imports, NO I/O and NO logic, so any test at any tier can
import it without cost or side effects. ASCII only. Every value below was MEASURED and
carries the plan, task and date that measured it.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later
correction is a NEW slot placed beside the old one, under its own separator naming the
plan, the task and the measurement date; the old slot stays as the record of what was
measured when it was written.

* Plan 34-02 Task 1 (this file's author) -- ``P34_EXPECTED_FAILURE_BASELINE``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plan 34-02 Task 1 -- the expected failure set, MEASURED 2026-10-05 (ET, read
# via PowerShell Get-Date) on HEAD c75119420f30a38592c59343a32abd02b398c3b5.
#
# Phase-34 commits already contained in that HEAD (all Plan 34-01; no Plan
# 34-03 commit had landed): 03a37d2, 9a00307, 860f9f5, c751194. None of the
# measured node ids exercises the bet-list schema widening or the book-ordering
# code the other Wave-1 plans touch.
#
# The three runs, each with its pytest summary line verbatim:
#
# 1. The five ids of tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS, passed
#    on the command line:
#        "5 failed, 34 warnings in 10.44s"  -- all five FAILED, as they must.
#
# 2. uv run python -m pytest tests/integration/test_ingest_2025_odds.py
#        tests/integration/test_ou_divergence.py -q -rxX
#        "90 passed, 1 skipped, 9 xfailed, 52 warnings in 53.98s"
#    The standing quarantine figure quoted in older records is "7 xfailed".
#    The MEASURED figure in these two modules is 9, and it is recorded as 9
#    rather than forced to 7: test_ingest_2025_odds.py carries six strict
#    xfails (DEF-31-06, four parametrized seasons plus two pooled checks) and
#    test_ou_divergence.py carries three (two D30-DEFER-08 anchors and the
#    DEF-31-07 inverted synthetic-stamp detector).
#    A grep of tests/ for a runtime `pytest.xfail(` call found none, and no
#    other module under tests/ carries an xfail marker.
#
# 3. The data-dependent nodes the owner's 2026-10-02 note listed as red on
#    master after 09e0dfd:
#        "14 passed, 32 warnings in 7.99s"
#    (the whole of test_bet_list_completeness.py was run, since the note did
#    not name its two nodes; they are TestTheStoredReplayRowsAreNeverBackfilled,
#    the two nodes commit c31b3a9 re-scoped to the stored replay rows).
#    ALL FOUR measured PASSED at this HEAD. They are recorded with that
#    outcome, not with the "red" the note predicted:
#      - test_health_stale_pipeline_log depends on the wall clock relative to
#        the last daily-run record, so it can flip between runs; d3b5210
#        (2026-09-24) changed how /health judges the automation.
#      - test_part_a_cache_predictions_equal_backtest_csv was re-scoped by
#        654d1cc (2026-10-03) to leave out the live-week cache rows.
#      - the two completeness nodes were re-scoped by c31b3a9 (2026-10-03)
#        to select provenance backtest_replay only.
#    The key keeps the plan's name `data_dependent_reds` so later plans find
#    it; the recorded outcomes say what was actually measured.
# ---------------------------------------------------------------------------

P34_EXPECTED_FAILURE_BASELINE: dict[str, object] = {
    "measured_on": "2026-10-05",
    "head": "c75119420f30a38592c59343a32abd02b398c3b5",
    "phase34_commits_in_head": ("03a37d2", "9a00307", "860f9f5", "c751194"),
    "tripwires": {
        "source": "tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS",
        "measured_failed": 5,
    },
    "xfail_quarantines": (
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2021]",
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2022]",
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2023]",
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2024]",
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_the_pooled_figure_matches",
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_the_negative_mean_season_set_is_still_exactly_2022",
        "tests/integration/test_ou_divergence.py::TestOuDivergence::test_deployed_population",
        "tests/integration/test_ou_divergence.py::TestOuDivergence::test_orchestrator_early_exit_is_skip_aware",
        "tests/integration/test_ou_divergence.py::TestTheSyntheticStampDetectorIsInverted::test_the_detector_reports_that_derivation_as_synthetic",
    ),
    "data_dependent_reds": (
        (
            "tests/api/test_health_endpoint.py::TestHealthDegraded::test_health_stale_pipeline_log",
            "passed",
        ),
        (
            "tests/integration/test_activation_parity.py::test_part_a_cache_predictions_equal_backtest_csv",
            "passed",
        ),
        (
            "tests/integration/test_bet_list_completeness.py::TestTheStoredReplayRowsAreNeverBackfilled::test_the_stored_artifact_is_all_replay_and_carries_no_observation_time",
            "passed",
        ),
        (
            "tests/integration/test_bet_list_completeness.py::TestTheStoredReplayRowsAreNeverBackfilled::test_reading_through_the_shim_does_not_write_the_column_back",
            "passed",
        ),
    ),
    "expected_to_change": (
        (
            "tests/integration/test_bet_list_completeness.py",
            "Plan 34-19's migration moves every 2026 forward row out of outputs/bet_list/. "
            "The plan predicted its two nodes would go green at cutover; they were already "
            "green when measured here (c31b3a9 re-scoped them to the replay rows), so the "
            "prediction is that they STAY green. Re-measure at cutover; do not assume.",
        ),
    ),
}
