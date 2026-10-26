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
* Plan 34-22 Task 3 -- ``P34_PREREGISTRATION_WITNESS``, ``P34_W``,
  ``P34_W_FIRST_LOCK_UTC``, ``P34_W_FIRST_DECISION_RUN_UTC``.
* Plan 34-23 Task 1 -- ``P34_FIRST_LIVE_READBACK``.
* Plan 38-07 Task 3 -- ``P38_EXPECTED_FAILURE_BASELINE``.
* Plan 38-12 Task 3 -- ``P38_FULL_SUITE_BASELINE``.
* Phase 38 code review fix CR-C01 -- ``P38_TRIPWIRE4_CRASH_CORRECTION``.
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

# ---------------------------------------------------------------------------
# Plan 34-22 Task 3 -- the Phase-34 pre-registration witness, MEASURED
# 2026-10-05 (ET, read via PowerShell Get-Date; step 0 at 18:38 ET).
#
# W was RE-COMPUTED immediately before the declaration commit with
# forward_ledger.verdict.compute_start_week over the live ledger (87 rows:
# week 3 = 42, week 4 = 45, none of week 5) and the recorded 2026 schedule, at
# Task 1's computation instant 2026-10-05T22:32:07Z: "W_RECOMPUTED= 5",
# "W_APPROVED= 5". Week 5's earliest game is 2026_W05_TB@DAL (Thu 2026-10-08
# 20:15 ET), so its first lock is Wed 2026-10-07 18:00 ET and its first
# decision run Wed 2026-10-07 17:00 ET. The owner approved the declaration
# text, W = 5 and the push at the Plan 34-22 checkpoint (~18:40 ET).
#
# Each commit below is `git log -1 --format=%H -- <path>` and touches ONLY that
# path; each digest is sha256 of the file's bytes with CRLF -> LF, equal for
# the working tree and the committed blob. The declaration was committed at
# 2026-10-05T18:38:59-04:00, alone, as c3a3399. This slot lands in a LATER
# commit than all three records, so no record carries its own witness.
# ---------------------------------------------------------------------------

P34_PREREGISTRATION_WITNESS: dict[str, dict[str, str]] = {
    "backtest/fill_conventions.py": {
        "commit": "b14471ca4fbffc9b32344111d44e04573625d115",
        "normalized_sha256": "4db6f84bf9db75d083f736ef20772119eeef58e053180ea5610be93fcef721b1",
    },
    "backtest/recipe_registry.py": {
        "commit": "c939862c5aaa6b4d278018b757c761ff21a4331d",
        "normalized_sha256": "742593bc5340610b9b34ae5ff5842218df843b7837ee8247c6300f3db15e88dc",
    },
    "backtest/verdict_scope_2026.py": {
        "commit": "c3a33993160769b2ab9a8c2cf7cd2a24ad19be10",
        "normalized_sha256": "c77eb858ff5248a0abcc79dda4b8bf40ca1d247f4ed1ff5a88fde3cb698c62e9",
    },
}

P34_W: int = 5

P34_W_FIRST_LOCK_UTC: str = "2026-10-07T22:00:00+00:00"

P34_W_FIRST_DECISION_RUN_UTC: str = "2026-10-07T21:00:00+00:00"

# ---------------------------------------------------------------------------
# Plan 34-23 Task 1 -- the first live ledger read-back, MEASURED 2026-10-12
# (ET, read via PowerShell; 00:07 to 00:10 ET Mon 2026-10-12), after week W = 5's
# LAST decision run: 2026_W05_BUF@LA kicks off Mon 2026-10-12 20:15 ET, locks Sun
# 2026-10-11 18:00 ET (utils.game_lock), so its decision run was the Sun 17:00
# ET scheduled run.
#
# Run log (logs/ledger_runs.jsonl) since the cutover, grouped by run_id: two
# scheduled runs appended rows and each has anchor_commit, anchor_push,
# backup_commit and backup_push events of its own run_id, all ok=true with the
# append's head_hash. No push has ever recorded ok=false. Console lines:
# "RUNS_WITH_APPENDS= ['48ec3a304bbf45ce98e1944e3c6bf7ac',
# '57d1b8ba56314c6996d429b28d5033e9']", "RUNS_MISSING_PUSH_OUTCOME= []".
#
# verify_ledger: VERIFY_RESULT= PASS, EXTERNAL_RESULT= VERIFIED,
# REMOTE_ANCHOR_BEHIND_BY= 0, BACKUP_BEHIND_BY= 0, VERDICT_DECLARED= True,
# VERDICT_START_WEEK= 5, VERDICT_STAMPS_OK= True, REGRADE_CHECKED= 26,
# REGRADE_OK= True. replay_ledger --week 5: REPLAY_PASS= 6, REPLAY_FAIL= 0.
#
# WEEK 5 IS INCOMPLETE: 13 of its 15 games have no ledger row. All 13 lock at
# Sat 2026-10-10 18:00 ET, so their one decision run was the Sat 17:00 ET
# scheduled run, which never fired: the Windows System log shows a user power-off
# (event 1074) at Fri 2026-10-09 23:33:05 ET and the next boot (event 6005) at Sun
# 2026-10-11 10:08:44 ET, and neither run log holds any record between
# 2026-10-09 00:05 UTC and 2026-10-11 16:50 UTC. WakeToRun wakes from sleep, not
# from power-off, and StartWhenAvailable is false, so the run was missed, not
# late. The games were NOT back-filled and nothing was hand-run (D33-37). Raised
# to the owner at the Plan 34-23 checkpoint. (The Fri 2026-10-09 17:00 ET run was
# also missed -- power-off 14:39:13 to 19:54:47 ET -- but no game locked on it.)
# ---------------------------------------------------------------------------

P34_FIRST_LIVE_READBACK: dict[str, object] = {
    "measured_on": "2026-10-12",
    "measured_at_et": "2026-10-12 00:07",
    "w": 5,
    "w_last_decision_run_et": "2026-10-11 17:00",
    "first_scheduled_appending_run": {
        "run_id": "48ec3a304bbf45ce98e1944e3c6bf7ac",
        "append_at_utc": "2026-10-07T21:16:28.619191+00:00",
        "append_at_et": "2026-10-07 17:16:28",
    },
    "appending_runs": (
        {
            "run_id": "48ec3a304bbf45ce98e1944e3c6bf7ac",
            "append_at_et": "2026-10-07 17:16:28",
            "appended_rows": 3,
            "head_hash": "0a7301bd45d8d5c524b1abd399c60cd326b9c6f2220b15bbe5c07e557850eecf",
            "entry_count": 90,
            "anchor_commit": ("ok", "20d7484639ae2774c9e1786d32a7169d3ae9a549"),
            "anchor_push": ("ok", "20d7484639ae2774c9e1786d32a7169d3ae9a549"),
            "backup_commit": ("ok", "164282652872fbe444c3ce6b67753f79cf3f85bf"),
            "backup_push": ("ok", "164282652872fbe444c3ce6b67753f79cf3f85bf"),
        },
        {
            "run_id": "57d1b8ba56314c6996d429b28d5033e9",
            "append_at_et": "2026-10-11 17:25:19",
            "appended_rows": 3,
            "head_hash": "aedc2fc9076174432ba5ee0afb69d9eccaa3ae910e6305557127744ccc0526b5",
            "entry_count": 93,
            "anchor_commit": ("ok", "0d1bfe663786e2178a3056dbaac8efb819610760"),
            "anchor_push": ("ok", "0d1bfe663786e2178a3056dbaac8efb819610760"),
            "backup_commit": ("ok", "98cf9ae263b16845b40cab93a12e20d7352a81fc"),
            "backup_push": ("ok", "98cf9ae263b16845b40cab93a12e20d7352a81fc"),
        },
    ),
    "runs_missing_push_outcome": (),
    "failed_pushes_ever": 0,
    "ledger_head_hash": "aedc2fc9076174432ba5ee0afb69d9eccaa3ae910e6305557127744ccc0526b5",
    "ledger_entries": 93,
    "anchor_local": "0d1bfe663786e2178a3056dbaac8efb819610760",
    "anchor_remote": "0d1bfe663786e2178a3056dbaac8efb819610760",
    "anchor_content": (
        "head_hash=aedc2fc9076174432ba5ee0afb69d9eccaa3ae910e6305557127744ccc0526b5",
        "row_count=93",
    ),
    "backup_local": "98cf9ae263b16845b40cab93a12e20d7352a81fc",
    "backup_remote": "98cf9ae263b16845b40cab93a12e20d7352a81fc",
    "public_master_remote": "d0d5d9502df7594dd723a519284f4ee5b9cc258f",
    "verify_ledger": {
        "VERIFY_RESULT": "PASS",
        "EXTERNAL_RESULT": "VERIFIED",
        "REMOTE_ANCHOR_BEHIND_BY": 0,
        "BACKUP_BEHIND_BY": 0,
        "VERDICT_DECLARED": True,
        "VERDICT_START_WEEK": 5,
        "VERDICT_STAMPS_OK": True,
        "VERDICT_LABELS_OK": True,
        "REGRADE_CHECKED": 26,
        "REGRADE_OK": True,
    },
    "expected_w_games": (
        "2026_W05_BAL@ATL",
        "2026_W05_BUF@LA",
        "2026_W05_CHI@GB",
        "2026_W05_CIN@MIA",
        "2026_W05_CLE@NYJ",
        "2026_W05_DEN@LAC",
        "2026_W05_DET@ARI",
        "2026_W05_HOU@TEN",
        "2026_W05_IND@PIT",
        "2026_W05_LV@NE",
        "2026_W05_MIN@NO",
        "2026_W05_NYG@WAS",
        "2026_W05_PHI@JAX",
        "2026_W05_SF@SEA",
        "2026_W05_TB@DAL",
    ),
    "ledger_w_games": ("2026_W05_BUF@LA", "2026_W05_TB@DAL"),
    "missing_w_games": (
        "2026_W05_BAL@ATL",
        "2026_W05_CHI@GB",
        "2026_W05_CIN@MIA",
        "2026_W05_CLE@NYJ",
        "2026_W05_DEN@LAC",
        "2026_W05_DET@ARI",
        "2026_W05_HOU@TEN",
        "2026_W05_IND@PIT",
        "2026_W05_LV@NE",
        "2026_W05_MIN@NO",
        "2026_W05_NYG@WAS",
        "2026_W05_PHI@JAX",
        "2026_W05_SF@SEA",
    ),
    "missing_w_games_cause": (
        "the Sat 2026-10-10 17:00 ET decision run never fired: machine powered off "
        "Fri 2026-10-09 23:33:05 ET to Sun 2026-10-11 10:08:44 ET (System events "
        "1074/6006, 6005); no daily_lock_runs.jsonl or ledger_runs.jsonl record for it"
    ),
    "w_games_without_3_targets": (),
    "w_rows": 6,
    "w_rows_verdict": 6,
    "w_rows_null_stamps": 0,
    "replay_week_w": {"REPLAY_PASS": 6, "REPLAY_NOT_REPLAYABLE": 0, "REPLAY_FAIL": 0},
    "rows_by_week": ((2026, 3, 42), (2026, 4, 45), (2026, 5, 6)),
    "null_stamp_rows_by_week": ((2026, 3, 42), (2026, 4, 45), (2026, 5, 0)),
}

# ---------------------------------------------------------------------------
# Plan 38-07 Task 3 -- the expected failure set after Phase 38 cleared the
# quarantines, MEASURED 2026-10-21 (ET, read via PowerShell Get-Date, 20:24) on
# HEAD 60e3e3fd83c59be165f9ae25a20b2a1154af2294.
#
# This slot SUPERSEDES P34_EXPECTED_FAILURE_BASELINE for the gate in
# tests/integration/test_phase34_expected_failure_set.py; that slot is left
# byte-unchanged as the record of what was measured on 2026-10-05.
#
# What changed since then: Plan 38-07 Task 2 FIXED the one real defect among
# the nine quarantines (DEF-31-07, the synthetic-stamp detector), and Tasks 1-2
# DELETED the other eight by owner ruling (six DEF-31-06 comparisons and two
# D30-DEFER-08 anchors, all of them reproductions of retired models that
# today's gold cannot score). Task 1 also deleted three dead-baseline
# duplicates of tripwires 1 and 5 (SPEC R4).
#
# The three runs, each with its pytest summary line verbatim:
#
# 1. The five ids of tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS, passed
#    on the command line:
#        "5 failed, 29 warnings in 10.98s"  -- all five FAILED, as they must.
#
# 2. uv run python -m pytest tests/integration/test_ingest_2025_odds.py
#        tests/integration/test_ou_divergence.py -q -p no:cacheprovider -rxX
#        "2 failed, 90 passed, 1 skipped, 47 warnings in 59.79s"
#    Zero xfails. The two failures are NOT quarantines and NOT caused by this
#    plan: both copy the production silver odds store into tmp_path, and that
#    store now holds 99 duplicated (game_id, sportsbook, snapshot_ts) keys,
#    all in 2026 live captures (for example 2026_W05_BUF@LA, a second reading
#    per book whose market_last_update falls after the game's lock). They are
#    recorded below under `unrelated_reds_measured` with that cause, and are
#    left for the owner; no test was skipped, marked or loosened for them.
#
# 3. The four data-dependent node ids of P34_EXPECTED_FAILURE_BASELINE:
#        "4 passed, 27 warnings in 5.59s"
#    All four measured PASSED, recorded as measured.
# ---------------------------------------------------------------------------

P38_EXPECTED_FAILURE_BASELINE: dict[str, object] = {
    "measured_on": "2026-10-21",
    "head": "60e3e3fd83c59be165f9ae25a20b2a1154af2294",
    "plan": "38-07",
    "tripwires": {
        "source": "tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS",
        "measured_failed": 5,
    },
    "xfail_quarantines": (),
    "closed_quarantines": {
        "fixed": (
            "tests/integration/test_ou_divergence.py::TestTheSyntheticStampDetectorIsInverted::test_the_detector_reports_that_derivation_as_synthetic",
        ),
        "deleted_by_owner_ruling": (
            "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2021]",
            "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2022]",
            "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2023]",
            "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_each_per_season_figure_matches[2024]",
            "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_the_pooled_figure_matches",
            "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_the_negative_mean_season_set_is_still_exactly_2022",
            "tests/integration/test_ou_divergence.py::TestOuDivergence::test_deployed_population",
            "tests/integration/test_ou_divergence.py::TestOuDivergence::test_orchestrator_early_exit_is_skip_aware",
        ),
    },
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
    "unrelated_reds_measured": (
        (
            "tests/integration/test_ingest_2025_odds.py::TestTheAdditiveJuiceColumns::test_the_stored_lines_are_byte_unchanged_by_a_merge_into_a_temporary_copy",
            "failed",
        ),
        (
            "tests/integration/test_ingest_2025_odds.py::TestTheWritePathInvariants::test_a_merge_into_a_temporary_copy_holds_every_invariant",
            "failed",
        ),
    ),
    "unrelated_reds_cause": (
        "the production silver odds store holds 99 duplicated (game_id, sportsbook, "
        "snapshot_ts) keys, all 2026 live captures (e.g. 2026_W05_BUF@LA: a second "
        "reading per book with market_last_update after the lock); both tests merge into a "
        "tmp_path copy of that store. Not caused by Plan 38-07; left for the owner."
    ),
    "deleted_dead_baseline_duplicates": (
        "tests/integration/test_drift_tripwire_sees_frozen_seasons.py::TestTheLiveFrameWouldAbort::test_passing_the_live_frame_RAISES_naming_a_zero_sample_size",
        "tests/integration/test_drift_tripwire_sees_frozen_seasons.py::TestTheLiveFrameWouldAbort::test_passing_the_frozen_frame_does_NOT_raise",
        "tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore",
    ),
    "retired_count_baseline": {
        "line": "6 failed / 3603 passed / 7 skipped / 14 xfailed",
        "source": "tests.phase32_state.V30_CLOSE_BASELINE",
        "retired_on": "2026-10-21",
        "reason": (
            "the v3.0 close count counted quarantines that no longer exist and predates "
            "five phases of tests; the gate now asserts an enumerated set -- exactly the "
            "five tripwires and zero xfails -- and the whole-suite baseline recorded at the "
            "Phase 38 close (three tier lines plus their sum, D-28) replaces it"
        ),
    },
}

# ---------------------------------------------------------------------------
# Plan 38-12 Task 3 -- the Phase 38 whole-suite baseline, MEASURED 2026-10-23
# (ET, read via PowerShell Get-Date) on HEAD
# f2b5b1ab2375dd3eca7eba88cfbd67e9a7c3b56d (equal to the run directory's
# head.txt before and after the run, and again at recording time).
#
# Run directory (outside the repository, never committed):
#     C:/Users/jackc/Code/p38-scratch/run2-f2b5b1ab2375-20261023-1409
# It holds head.txt and, per tier, <tier>.collect.txt, <tier>.xml and
# <tier>.log.
#
# The run was made as three SEQUENTIAL tier processes, not one `pytest tests`
# process, because a single-process whole suite is killed for memory on this
# machine -- the instrument of tests/phase33_state.py PRE_PHASE_TIER_LINES /
# PRE_PHASE_FAILURE_SET. It was ONE background Bash call from the repository
# root with NFL_PHASE38_FULL_RUN_DIR and NFL_PHASE33_FULL_TIER_VERDICT unset,
# the six commands in "commands" below joined by `;` (each tier's exit code and
# a UTC timestamp were appended to exits.txt / tier_end_utc.txt after it).
# Exit codes: unit 0, integration 1, api 0. No tier was killed.
#
# Judged by the opt-in node
# tests/integration/test_phase33_expected_failure_set.py::
# test_the_recorded_full_run_report_when_given with NFL_PHASE38_FULL_RUN_DIR
# set to the run directory: "1 passed, 27 warnings in 4.98s" -- across the
# union of the three tier reports the failing node ids are exactly
# tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS, nothing errors, nothing
# xfails, and every collected test has an outcome.
#
# This slot is the D-28 recording form (SPEC amendment 4, owner ruling
# 2026-10-21): the three tier lines verbatim plus their sum and the JUnit
# counts, which replaces the retired v3.0 close count recorded in
# P38_EXPECTED_FAILURE_BASELINE["retired_count_baseline"].
# ---------------------------------------------------------------------------

P38_FULL_SUITE_BASELINE: dict[str, object] = {
    "measured_on": "2026-10-23",
    "head": "f2b5b1ab2375dd3eca7eba88cfbd67e9a7c3b56d",
    "run_dir": "C:/Users/jackc/Code/p38-scratch/run2-f2b5b1ab2375-20261023-1409",
    "commands": (
        'uv run python -m pytest tests/unit --collect-only -q -p no:cacheprovider > "$R/unit.collect.txt" 2>&1',
        'timeout 3600 uv run python -m pytest tests/unit -q -rfE -p no:cacheprovider --junitxml="$R/unit.xml" > "$R/unit.log" 2>&1',
        'uv run python -m pytest tests/integration --collect-only -q -p no:cacheprovider > "$R/integration.collect.txt" 2>&1',
        'timeout 3600 uv run python -m pytest tests/integration -q -rfE -p no:cacheprovider --junitxml="$R/integration.xml" > "$R/integration.log" 2>&1',
        'uv run python -m pytest tests/api tests/test_*.py --collect-only -q -p no:cacheprovider > "$R/api.collect.txt" 2>&1',
        'timeout 3600 uv run python -m pytest tests/api tests/test_*.py -q -rfE -p no:cacheprovider --junitxml="$R/api.xml" > "$R/api.log" 2>&1',
    ),
    "tier_lines": {
        "unit": "8219 passed, 5 skipped, 6423 warnings in 817.34s (0:13:37)",
        "integration": "5 failed, 1767 passed, 59 skipped, 622 warnings in 1816.40s (0:30:16)",
        "api": "711 passed, 27 warnings in 299.01s (0:04:59)",
    },
    "summed_line": "5 failed, 10697 passed, 64 skipped, 7072 warnings",
    "junit": {
        "unit": {"tests": 8224, "failures": 0, "errors": 0, "skipped": 5},
        "integration": {"tests": 1831, "failures": 5, "errors": 0, "skipped": 59},
        "api": {"tests": 711, "failures": 0, "errors": 0, "skipped": 0},
        "union": {
            "tests": 10766,
            "failures": 5,
            "errors": 0,
            "skipped": 64,
            "xfailed": 0,
        },
    },
    "report_sha256": {
        "unit": "b7c10bb6769d55d68e45dd32002322f53ce65549e6acf096f926c4f5029616d1",
        "integration": "013b91c4f9905ece233790b0024b95e96d566d9e78087576e6b71c0bca3d5a0b",
        "api": "f947de0e506544fac56cc50fb230a0b58325bc1cd71be07271b8b45069179784",
    },
    "collected": {"unit": 8224, "integration": 1831, "api": 711},
    "failing_node_ids_source": "tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS",
    "started_et": "2026-10-23 14:10:17",
    "finished_et": "2026-10-23 14:59:57",
    "discovery_run_tier_lines": {
        "unit": "15 failed, 8206 passed, 5 skipped, 6423 warnings in 838.16s (0:13:58)",
        "integration": "6 failed, 1765 passed, 59 skipped, 622 warnings in 1804.66s (0:30:04)",
        "api": "711 passed, 27 warnings in 314.07s (0:05:14)",
    },
    "note": (
        "In the commands, $R is the run directory above. pytest's summary line omits "
        "categories whose count is zero, so the JUnit counts are the proof of 0 errors and "
        "0 xfailed (xfailed is counted from testcases skipped with type pytest.xfail). No "
        "tier log carried the COLD-05 session-end sweep section. By the owner's ruling "
        "D-28 (SPEC amendment 4, 2026-10-21) this baseline is the three tier lines plus "
        "their sum, not one pytest summary line: exactly the five deliberate tripwires "
        "fail, 0 errors, 0 xfailed. The run was started at 14:10 ET, outside the plan's "
        "Mon-Fri 08:00-13:00 ET start window, on the owner's explicit 'Start now "
        "(Recommended)' (about 14:09 ET); no scheduled writer ran before the 17:00 ET "
        "daily run and no kickoff fell before Sun 2026-10-25 09:30 ET. The discovery run "
        "(run 1, 2026-10-21, Plan 38-11) is kept for the record only; its lines are not "
        "the baseline."
    ),
}

# ---------------------------------------------------------------------------
# Phase 38 code review fix CR-C01 -- a CORRECTION to P38_FULL_SUITE_BASELINE,
# MEASURED 2026-10-25 (ET, read via PowerShell) on HEAD
# cd6514c4c4f9219fac7a48c510d1b2e8773a3b48 plus the CR-C01 edits committed with
# this slot (owner ruling 2026-10-23, "Repair it, keep it red").
#
# WHAT IT CORRECTS. P38_FULL_SUITE_BASELINE (and the 38-11 discovery run before
# it) recorded "exactly the five deliberate tripwires fail". In run 2 only four
# of them failed on their own check. Tripwire 4 -- index 3 of
# tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS -- CRASHED with the JUnit
# failure message "KeyError: 'away_def_rolling_avg_drive_start_yardline'"
# before it reached its assertion: commit fc1a5dd (33.2-14, 2026-09-21) dropped
# the *_def_rolling_avg_drive_start_yardline columns from gold, and the test
# indexed every pinned column. The gate counted any <failure> as the accepted
# red, so the crash passed as a disclosure. The old slot stays as the record of
# what was claimed; this slot is the record of what was true.
#
# THE FIX. The gate and the recorded-run judge now accept a tripwire's red only
# when its failure message is an assertion ("assert ..." or "AssertionError:
# ..."); any other message is "crashed". Tripwire 4 counts a pinned column that
# is missing from gold as moved and names it, so it reaches its check. It stays
# RED; nothing was loosened.
#
# THE RUN AFTER THE FIX. Exactly the five ids of
# tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS, and nothing else:
#     uv run python -m pytest <the five ids> -q -rfE -p no:cacheprovider
#         --junitxml=C:/Users/jackc/Code/p38-scratch/tripwires-after-crc01/tripwires.xml
#     "5 failed, 29 warnings in 12.81s"
# Every one of the five now fails on an assertion; "after_fix_failures" below
# quotes the start of each failure message, in the constant's order. No third
# full-suite run was made.
#
# RUN 2 IS STILL JUDGED. The opt-in node test_the_recorded_full_run_report_when_given
# accepts, for the run directory whose head.txt equals "run_head" below, exactly
# the one problem "TRIPWIRE NOT A FAILURE: <tripwire 4> [crashed]" and nothing
# else; it fails if that problem is absent too.
# ---------------------------------------------------------------------------

P38_TRIPWIRE4_CRASH_CORRECTION: dict[str, object] = {
    "measured_on": "2026-10-25",
    "corrects": "P38_FULL_SUITE_BASELINE",
    "run_head": "f2b5b1ab2375dd3eca7eba88cfbd67e9a7c3b56d",
    "run_dir": "C:/Users/jackc/Code/p38-scratch/run2-f2b5b1ab2375-20261023-1409",
    "tripwire_index": 3,
    "run_state": "crashed",
    "run_failure_message": "KeyError: 'away_def_rolling_avg_drive_start_yardline'",
    "after_fix_head": "cd6514c4c4f9219fac7a48c510d1b2e8773a3b48 plus the CR-C01 edits",
    "after_fix_line": "5 failed, 29 warnings in 12.81s",
    "after_fix_report_sha256": (
        "5356cb61b101911a21dd0098c0959879a5d082683d9c893854cf1e890220a456"
    ),
    "after_fix_failures": (
        "AssertionError: the REGENERATED gate baseline block is not byte-identical to "
        "the block committed in config/gate.toml.",
        "AssertionError: the FULL rebuild moved 11 non-clock column-slot(s) in the "
        "PROTECTED 2021-2024 window.",
        "AssertionError: features_ats's 2021-2024 slice gained 9 columns (weather_coverage "
        "and eight *_coverage columns) and lost 16 (target_ats among them).",
        "AssertionError: features_ats: 188 of its 2021-2024 columns moved, 16 of them no "
        "longer in gold at all (away_def_rolling_avg_drive_start_yardline among them), "
        "against an expected set of exactly ['feature_timestamp'].",
        "AssertionError: frozen baseline.wp.pooled.mean = -0.03800034 does not match a "
        "fresh re-score 0.01483587975541558 (tol 0.005).",
    ),
    "reason_drift_note": (
        "Tripwire 3's reason has also drifted from the Phase-33 record (which said the "
        "slice gained weather_coverage and lost none); it still fails on its assertion, "
        "and the drift is recorded here rather than re-anchored."
    ),
}
