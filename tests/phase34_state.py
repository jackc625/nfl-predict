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
* Plan 34-18 Task 3 -- ``P34_S4U_PUSH_SMOKE``.
* Plan 34-19 Task 3 -- ``P34_CUTOVER``.
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

# ---------------------------------------------------------------------------
# Plan 34-18 Task 3 -- the S4U push smoke, MEASURED 2026-10-05 (ET, read via
# PowerShell Get-Date; the S4U run itself was 2026-10-05T21:27:15Z to
# 21:27:19Z, i.e. 17:27 ET).
#
# Source: logs/ledger_push_smoke.json (written BY the S4U task) and the
# owner's `--readback` output, which printed "SMOKE_RESULT= PASS". Both ssh -T
# greetings named the REPOSITORY (deploy key), each with GitHub's normal
# no-shell exit 1. The smoke commit landed on refs/heads/ledger-anchor-smoke and
# matched the remote; the backup's first commit is now main on the private
# repository (by design, it stays). The public master read
# 664d8fbe5e230bf051b884e083b6de556518e875 before and after. `--cleanup`
# confirmed the smoke branch, the local smoke ref and both tasks gone, and a
# later read-only re-check (git ls-remote, gh api, schtasks /query) agreed.
#
# Registration (`--register`: schtasks /create of an S4U + HighestAvailable
# task) and `--cleanup` need an ELEVATED owner session; from a non-elevated
# session /create answers "ERROR: Access is denied.". The in-task pushes and
# the dummy-task create/export/delete ran UNELEVATED under S4U.
# ---------------------------------------------------------------------------

P34_S4U_PUSH_SMOKE: dict[str, object] = {
    "measured_on": "2026-10-05",
    "whoami": "jackslaptop\\jackc",
    "anchor_identity_ok": True,
    "backup_identity_ok": True,
    "anchor_push_ok": True,
    "backup_push_ok": True,
    "schtasks_from_s4u_ok": True,
    "smoke_commit": "23f56a7a684bf28c1f7179a6ef3ef6b98ad9f824",
    "backup_commit": "4dbaebaf2133439b3f87ab1b4204bd54bf7c52cf",
    "note": (
        "Registering and cleaning up the smoke task need an elevated owner session "
        "(schtasks /create of an S4U + HighestAvailable task is refused otherwise); "
        "the in-task pushes and the dummy-task probe ran unelevated under S4U."
    ),
}

# ---------------------------------------------------------------------------
# Plan 34-19 Task 3 -- the ledger cutover (go-live), MEASURED 2026-10-05 (ET,
# read via PowerShell Get-Date). Every value below is quoted from
# logs/34-19-golive-outputs.txt, which holds each step's output word for word.
#
# Order, as the plan requires: pre-migration copy (17:46:09 ET) -> migrate
# --apply (17:46:15) -> sync with the switch OFF (17:46:27; ANCHOR_PUSHED= True,
# BACKUP_PUSHED= True) -> remote read-backs (17:46:41) -> anchor ruleset +
# setup verify (17:46:49; SETUP_READBACK_MATCH= True) -> verify_ledger with the
# switch off (17:47:09; VERIFY_RESULT= PASS, REGRADE_OK= True over 17 settled
# rows) -> the one-file cutover commit (17:47:37) -> cache rebuild (17:47:47;
# 87 forward rows in the cache, none in outputs). The first anchor commit time
# (1791236790) precedes the cutover commit time (1791236857).
#
# The closing wake task was installed BY THE OWNER from an elevated terminal at
# 17:52 ET (an agent session cannot register an S4U task; see
# P34_S4U_PUSH_SMOKE). A later non-elevated `--verify-installed` agreed.
#
# Data-dependent nodes re-measured at 17:48:56 ET after the cache rebuild: the
# two completeness nodes and the activation-parity node PASSED; the health node
# FAILED ('ok' == 'degraded'). It reads the real daily-run log, which the
# 17:00 ET run of the same evening had just made fresh, so it fails for
# time-of-day reasons, not because of the cutover.
# ---------------------------------------------------------------------------

P34_CUTOVER: dict[str, object] = {
    "measured_on": "2026-10-05",
    "migrated_at_et": "2026-10-05 17:46:15",
    "cutover_committed_at_et": "2026-10-05 17:47:37",
    "cutover_commit": "300e44ea9b53facc8200cd056d4c23ac75b30d4f",
    "cutover_commit_files": ("forward_ledger/cutover.py",),
    "migrated_rows_by_week": ((2026, 3, 42), (2026, 4, 45)),
    "migrated_rows_total": 87,
    "migrated_status_counts": {"live": 25, "suppressed": 62},
    "migrated_rows_null_stamps": True,
    "outputs_replay_rows_after": 234,
    "forward_rows_in_outputs_after": 0,
    "ledger_path": "ledger/forward_2026.jsonl",
    "ledger_head_hash": "edadf04ea89bf5b64a81f0c9ffedde5d666111bafd7c5d9aaef313e13953a6a4",
    "ledger_entries": 87,
    "anchor_commit": "ec71c421b793b505ccfa70c475198199ce84436d",
    "anchor_content": (
        "head_hash=edadf04ea89bf5b64a81f0c9ffedde5d666111bafd7c5d9aaef313e13953a6a4",
        "row_count=87",
    ),
    "anchor_remote_equals_local": True,
    "anchor_ruleset_id": 24536696,
    "backup_head": "53b9d383525546d706bc6f157fd3e44e41d4863e",
    "backup_remote_equals_local": True,
    "public_master_unchanged": "664d8fbe5e230bf051b884e083b6de556518e875",
    "verify_ledger": {
        "VERIFY_RESULT": "PASS",
        "EXTERNAL_RESULT": "VERIFIED",
        "REMOTE_ANCHOR_BEHIND_BY": 0,
        "VERDICT_DECLARED": False,
        "REGRADE_CHECKED": 17,
        "REGRADE_OK": True,
    },
    "pre_migration_copy": {
        "dir": "outputs/bet_list_pre_migration",
        "bet_list.parquet": "4465d0586d6db45ea2743c2055c5b3a7e44c6a4ee7094db24ab2d33e24e63133",
        "bet_tracker.json": "feeb5572c96a9606501ea0414f27087246c30c422d9a26fa7cf61247965a7f09",
    },
    "cache_forward_rows": 87,
    "data_dependent_nodes": (
        (
            "tests/api/test_health_endpoint.py::TestHealthDegraded::test_health_stale_pipeline_log",
            "failed",
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
    "closing_task": {
        "name": "NFL_Predict_Closing",
        "installed_at_et": "2026-10-05 17:52",
        "installed_by": "owner, elevated terminal",
        "triggers": 7,
        "CLOSING_READBACK_MATCH": True,
        "READBACK_MATCH": True,
        "nfl_tasks": ("NFL_Predict_Closing", "NFL_Predict_Pipeline"),
    },
}
