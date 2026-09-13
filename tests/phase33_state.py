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


# ---------------------------------------------------------------------------
# What the armed guard OBSERVED about its own instrument.
#
# APPENDED by Plan 33-01 Task 3 on 2026-09-11. Nothing above this line was
# edited: a state manifest whose earlier slots move cannot be used to reconstruct
# what was believed when.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run python -m pytest tests/unit -q          325.26 s
#     uv run python -m pytest tests/integration -q   527.91 s
#     uv run python -m pytest tests/api tests/test_*.py -q   129.31 s
# ---------------------------------------------------------------------------

# How many times a content read hit the LOCKED-FILE path and had to close handles
# before it could hash the bytes. Per tier: 3 (unit) + 5 (integration) + 0 (api).
#
# Q-07 ANSWERED. The question was whether `digest_file`'s stat-signature fallback
# is a live degradation of the guard or a path that never fires. It fires -- eight
# times in one full pass, every one of them on `data/nfl_predictions.duckdb`. So
# the reviewer who called the fallback a HIGH weakness was describing a real
# exposure and not a hypothetical one, and D33-32's ruling (retry, then raise by
# name) is load-bearing rather than decorative. Under the shipped guard all eight
# RECOVERED: the retry read the bytes, and no verdict in any tier rested on a stat
# signature.
STAT_SIGNATURE_OBSERVATIONS: int = 8

# The largest number of `.duckdb.wal` siblings seen beside a tracked `.duckdb`
# store at any single sweep.
#
# THE RECORDED VALUE IS THE PRE-FIX MEASUREMENT, AND THAT IS DELIBERATE. The
# armed integration tier of 2026-09-11 20:20 observed ONE write-ahead-log sibling
# and, at session end, a full content sweep reporting `nfl_predictions.duckdb`
# moved with no per-test sweep having seen it. Those two facts are the same fact:
# a DuckDB write lands in the `.wal` sibling, the main file is not touched until
# the connection checkpoints, and `.wal` is not a tracked suffix -- so the
# declared write was invisible for the whole of the declaring test's window and
# surfaced later, attributed to nobody.
#
# The shipped guard now checkpoints a declared write inside the window of the test
# that declared it, so a re-measurement reads 0. Recording that 0 would erase the
# evidence this slot exists to preserve: the WAL was never absent, it is merely no
# longer allowed to outlive the test that created it.
#
# THIS IS THE EVIDENCE a later phase needs before widening TRACKED_SUFFIXES, which
# Plan 33-01 deliberately did NOT do -- widening the tracked set changes what every
# digest document taken under the narrower set means, including the
# `outputs/*_before.json` snapshots Plans 33-12, 33-13, 33-14 and 33-18 will take.
WAL_SIBLING_OBSERVATIONS: int = 1

# The COLLECTED count this plan adds, measured as the three-tier collected total
# after the plan (2980 + 726 + 367 = 4073) minus PRE_PHASE_COLLECTED (4029). Each
# parametrize case counts as the separate collected node it is.
#
# It closes exactly against the modules: 8 in
# tests/integration/test_data_boundary_guard_arming.py, 18 in
# tests/unit/test_write_guard_marker_scope.py, and 18 added to
# tests/unit/test_data_boundary_digest.py. That the arithmetic closes is the check
# that no test was quietly deleted to make a number look better.
#
# An IMMUTABLE per-plan counter, appended once and never edited. Plan 33-02 defines
# the protocol; Plan 33-18 sums the per-plan counters into TESTS_ADDED_BY_PHASE_33
# exactly once, at closure. No running aggregate is kept here.
TESTS_ADDED_33_01: int = 44


# ---------------------------------------------------------------------------
# THE PER-PLAN TEST-COUNT PROTOCOL.
#
# APPENDED by Plan 33-02 Task 1 on 2026-09-11. Nothing above this line was edited.
#
# WHY EIGHTEEN SLOTS AND NOT ONE RUNNING TOTAL. The obvious design is a single
# TESTS_ADDED_BY_PHASE_33, seeded at zero and incremented by each plan. That
# design EDITS ONE SLOT EIGHTEEN TIMES, which is exactly what the APPEND PROTOCOL
# at the head of this file forbids -- and it would be forbidden in the phase whose
# own audit record that protocol IS. The protocol cannot be the first thing the
# phase breaks.
#
# So there is one IMMUTABLE slot per plan, named for its plan. Plan 33-07 appends
# TESTS_ADDED_33_07 and touches nothing else. A reader can tell at a glance which
# plans have reported, and a hole in the sequence names the plan that finished
# without recording its count.
#
# THE UNIT IS COLLECTED NODES, NOT TEST FUNCTIONS (T-33-08c). pytest collects one
# node per `@pytest.mark.parametrize` case, so a parameterised test contributes as
# many to the arithmetic as it generates. A plan that counted FUNCTIONS would
# under-report by the parametrize multiplier and the closing equation would fail to
# close for a reason that has nothing to do with a deleted test. Each plan measures
# its slot by SUBTRACTING the previous three-tier collected total from its own --
# never by counting the functions it wrote.
#
# THE AGGREGATE IS PLAN 33-18'S, APPENDED ONCE, AT CLOSURE. It is deliberately
# ABSENT from this file until then. tests/unit/test_phase33_state_arithmetic.py
# asserts its absence while the parts are incomplete, asserts
# PRE_PHASE_COLLECTED + sum(parts) == POST_PHASE_COLLECTED once they are all
# present, and asserts TESTS_ADDED_BY_PHASE_33 == sum(parts) -- so the aggregate is
# a derived number with a check on it rather than a nineteenth claim nobody
# reconciles. tests/unit/test_phase33_state_append_once.py makes the whole protocol
# mechanical: an AST scan reports any name this file assigns twice.
# ---------------------------------------------------------------------------

PER_PLAN_TEST_COUNT_SLOTS: tuple[str, ...] = (
    "TESTS_ADDED_33_01",
    "TESTS_ADDED_33_02",
    "TESTS_ADDED_33_03",
    "TESTS_ADDED_33_04",
    "TESTS_ADDED_33_05",
    "TESTS_ADDED_33_06",
    "TESTS_ADDED_33_07",
    "TESTS_ADDED_33_08",
    "TESTS_ADDED_33_09",
    "TESTS_ADDED_33_10",
    "TESTS_ADDED_33_11",
    "TESTS_ADDED_33_12",
    "TESTS_ADDED_33_13",
    "TESTS_ADDED_33_14",
    "TESTS_ADDED_33_15",
    "TESTS_ADDED_33_16",
    "TESTS_ADDED_33_17",
    "TESTS_ADDED_33_18",
)


# ---------------------------------------------------------------------------
# TWO PHASE-WIDE PROHIBITIONS, recorded as the constants their scans read.
#
# APPENDED by Plan 33-02 Task 2 on 2026-09-11. Nothing above this line was edited.
# ---------------------------------------------------------------------------

# The commit this phase is measured FROM. Plan 33-01 began here; its SUMMARY records
# `git rev-list --count 6f6af82..HEAD` = 8. The scans that need the phase's own file
# list take `git diff --name-only PHASE_33_BASE_COMMIT..HEAD` rather than keeping a
# second, drift-prone list of file names: a base commit is one fact, and it either
# resolves or it does not.
PHASE_33_BASE_COMMIT: str = "6f6af82760bd3750aff4a0f2044f403a7113aa19"

# The phrasings no file in this phase may use about the test suite.
#
# THE HEALTHY FORM OF THIS SUITE'S SUMMARY LINE IS
#     5 failed, <N> passed, 9 skipped, 14 xfailed
# with the five named individually in DELIBERATE_TRIPWIRE_NODE_IDS above. Every one of
# them encodes a fact the owner accepted; a suite that reported a clean line would have
# erased five disclosures rather than fixed five defects. An assertion, comment,
# docstring or readout anywhere in this phase that says the suite reaches a failure
# count of zero is therefore itself a DEFECT, and it is the shape in which a record
# quietly stops being defended: the claim is written first, and the suite is made to
# agree with it afterwards.
#
# Matching is case-insensitive. THIS FILE and the scanning module are the only two
# files excluded, because RECORDING a phrase is not MAKING the claim -- and
# tests/unit/test_phase33_no_zero_failures_claim.py asserts that it never spells one
# of these literally, so its own exclusion is unnecessary as well as bounded.
FORBIDDEN_SUITE_CLAIM_PHRASES: tuple[str, ...] = (
    "zero failures",
    "0 failures",
    "no failures",
)

# The SELECTION / SIZING / REFUSAL surface over which closing-line value must never be
# read into a decision (D27-06, D27-12; T-33-09).
#
# PATHS RESOLVED AGAINST LIVE `git ls-files` on 2026-09-11, not from memory. An earlier
# draft of this plan named `backtest/kelly_criterion.py`, which does not exist in this
# repository; the real sizing module is `utils/kelly_criterion.py`. A path recorded here
# and absent from the tree is a RECORDING ERROR and the scan fails on it rather than
# visiting a shorter list and reporting green.
#
# `models/deploy_gate.py` is deliberately ABSENT -- see CLV_OUT_OF_SCOPE_MODULES.
CLV_REPORT_ONLY_MODULES: tuple[str, ...] = (
    "backtest/weekly_bet_list.py",
    "backtest/bet_selector.py",
    "utils/bet_selector.py",
    "utils/kelly_criterion.py",
    "scripts/generate_current_week_predictions.py",
)

# The CLV identifiers the scan treats as a read.
CLV_SYMBOLS: tuple[str, ...] = (
    "clv_delta_values",
    "clv_significance",
    "clv_non_regression_passes",
    "per_season_clv",
    "closing_line_value",
)

# Any identifier starting with this prefix is a CLV read as well, so a new
# `clv_whatever` needs no edit here to be seen.
CLV_SYMBOL_PREFIX: str = "clv_"

# The per-bet RECORD KEYS that hold a CLV value. These do NOT match CLV_SYMBOL_PREFIX
# and they are the dominant shape in this codebase: `bet_selector` writes `"clv"` onto
# each decision record and `weekly_bet_list` copies it onto the output row. The
# decision-position scan treats a subscript or `.get()` on one of these as a CLV value,
# so `if record["clv"] > 0: select(...)` is caught even though the key is not prefixed.
CLV_VALUE_KEYS: tuple[str, ...] = (
    "clv",
    "line_clv",
)

# EVERY CLV reference that exists in CLV_REPORT_ONLY_MODULES today, as
# (module, innermost_scope, kind, symbol). MEASURED 2026-09-11 by Plan 33-02 Task 2.
#
# WHY AN INVENTORY AND NOT "NONE AT ALL". The plan's first draft required that no
# scanned module reference a CLV symbol at all. Measured against live source that
# cannot pass, and must never be made to pass: `backtest/bet_selector.py` imports
# `clv_significance` and builds `SelectionResult.clv_report` -- AFTER the selection and
# sizing loops have run, over the bets already chosen, into a field nothing reads back.
# Deleting it would delete the REPORT and none of the risk. So the reference set is
# PINNED instead, and asserted in BOTH directions: an unpinned read fails by name (the
# default is REJECT), and a pinned site that no longer matches anything fails too,
# because an exemption that covers nothing is cover nobody is entitled to.
#
# The inventory is necessary and not sufficient. A pinned read must ALSO pass the
# decision-position scan, which forbids any CLV-tainted value from being branched on,
# compared, ranked by, or assigned into a sizing target -- including inside these five
# sites, which is where a future author would most naturally add the branch.
#
# The scope is a qualified name rather than a line number on purpose: a line number
# makes this tuple brittle against any edit above it, and the question it answers is
# WHERE IN THE DESIGN the read happens.
CLV_REPORT_ONLY_SITES: tuple[tuple[str, str, str, str], ...] = (
    # The report helper's import. `clv_significance` is the LOCKED mean/t/p/CI summary
    # from backtest.diagnose; importing it is how the report is rendered, not how a bet
    # is chosen.
    ("backtest/bet_selector.py", "<module>", "importfrom", "clv_significance"),
    # The report renderer itself: gathers the per-bet values from the ALREADY-SELECTED
    # bets and summarises them. Branches only on whether any values exist.
    ("backtest/bet_selector.py", "BetSelector._clv_report", "name", "clv_significance"),
    ("backtest/bet_selector.py", "BetSelector._clv_report", "name", "clv_values"),
    # `select()` calls the renderer AFTER `_admit_and_size_week` has run and passes the
    # result straight into the output dataclass. Nothing reads it back.
    ("backtest/bet_selector.py", "BetSelector.select", "name", "clv_report"),
    # The output field. REPORT-ONLY is stated in its own docstring.
    ("backtest/bet_selector.py", "SelectionResult", "name", "clv_report"),
)

# Modules where CLV is legitimately read and which are therefore OUTSIDE the scanned
# set, each with the reason recorded. An unexplained exclusion is indistinguishable
# from an oversight, which is why the reason travels with the path rather than living
# in a reviewer's memory.
CLV_OUT_OF_SCOPE_MODULES: tuple[tuple[str, str], ...] = (
    (
        "models/deploy_gate.py",
        "In the deploy gate CLV is the SUBJECT, not an input: clv_non_regression_passes, "
        "_pooled_floor_reasons and evaluate_target exist to judge a candidate model's "
        "closing-line value against the deployed incumbent's, which is the gate's whole "
        "definition as ruled in Phase 25 (D25-11), re-run in Phase 30 (PROD-01) and "
        "restated in Phase 31. Scanning it would flag the gate for doing the one thing "
        "it is for, and the only way to make that scan pass would be to weaken it. The "
        "gate decides which MODEL is deployed; it never decides which BET is taken, and "
        "that is the boundary this exclusion is drawn on.",
    ),
)


# ---------------------------------------------------------------------------
# The two PRE-EXISTING places a forbidden phrase legitimately appears.
#
# APPENDED by Plan 33-02 Task 2 on 2026-09-11, after MEASURING the scan against the
# phase's real diff rather than against what the plan assumed it would contain.
#
# Both lines are PROSE THAT NAMES THE FAILURE MODE IN ORDER TO CRITICISE IT, and both
# predate Phase 33 -- they are in the diff only because Plan 33-01 modified the files
# for unrelated reasons. `tests/conftest.py:33` explains why an aggregate skip note
# exists at all; `tests/unit/test_evidence_skip_visibility.py:13` is that note's own
# test module saying the same thing. Rewording somebody else's correct explanation to
# satisfy a new scan would be the scan editing the record, which is the inversion this
# whole phase exists to prevent.
#
# So they are PINNED, not excluded, and asserted in BOTH directions: a new line
# carrying a forbidden phrase fails by name (the default is REJECT), and a pinned line
# that no longer matches fails too, so the exemption cannot outlive the text it covers.
# The pin is the WHOLE STRIPPED LINE rather than a count, so an edit that turned the
# criticism into a claim would not inherit the cover.
FORBIDDEN_CLAIM_PHRASE_PINNED_MENTIONS: tuple[tuple[str, str], ...] = (
    (
        "tests/conftest.py",
        "# The problem is the AGGREGATE. A suite that skips them still reports zero "
        "failures,",
    ),
    (
        "tests/unit/test_evidence_skip_visibility.py",
        "central controls do not execute -- while the suite still reports zero failures.",
    ),
)


# ---------------------------------------------------------------------------
# THE MEASURED 2026 SCHEDULE FACTS the shared fixtures assert against.
#
# APPENDED by Plan 33-02 Task 3 on 2026-09-11. Nothing above this line was edited.
#
# MEASURED from the single live capture Plan 32-09 Task 2 took on 2026-09-11 at
# 11:02:52 UTC:
#     data/bronze/schedules_raw_bronze_2026_W01_20260911T110252.parquet
# That file lives under the gitignored production store and is READ, never written
# (T-33-11). tests/fixtures/season_2026.py derives every 2026 set from it exactly
# once, and these constants are what the derivations are checked against -- a
# derivation checked only against itself agrees with itself.
# ---------------------------------------------------------------------------

CAPTURED_SCHEDULE_ROWS: int = 272
CAPTURED_SCHEDULE_COLUMNS: int = 46

# How many of those 272 games already carry a score.
#
# THE CAPTURE IS MID-WEEK-1, NOT PRE-SEASON, and that is the cold start this phase is
# actually about. Exactly TWO games are graded in it:
#     2026_W01_NE@SEA  SEA 13 - NE 10
#     2026_W01_SF@LA   LA   7 - SF 27   (the Melbourne neutral-site game)
# The other 270 rows carry NA scores and an NA result.
#
# Recorded because "the 2026 season is unplayed" is the obvious assumption and it is
# WRONG BY TWO ROWS. A plan that asserts every score is NA would redden against real
# data; a plan that assumes any score is present would redden against 270 of them.
# Both mistakes are cheap to make and invisible until the assertion runs.
CAPTURED_SCHEDULE_GRADED_GAMES: int = 2

# Rows the feed marks `location == "Neutral"`. There is NO `neutral_site` column in
# the nflverse schedule feed at all, which is why scripts/ingest_games.py's
# `row.get("neutral_site", False)` reads False for all eight of these -- a
# PRE-EXISTING ingestion defect, owned by COLD-09, pinned as an asserted observation
# in tests/unit/test_season_2026_fixture_schema.py rather than silently fixed inside
# a fixtures plan.
NEUTRAL_SITE_GAME_COUNT_2026: int = 8

# The eight international venues, IN WEEK ORDER: weeks 1, 3, 4, 6, 7, 9, 10, 11.
#
# MUN01 IS NOT GER00 AND RIO00 IS NOT SAO00. The 2026 international venues are not
# the ones history used, and reaching for a prior season's stadium mapping is exactly
# how a cold start produces eight confidently wrong venue rows. Recorded so the
# mapping is a measured fact rather than an inference from 2024.
INTERNATIONAL_STADIUM_IDS: tuple[str, ...] = (
    "MEL00",  # week 1  -- Melbourne Cricket Ground, SF @ LA
    "RIO00",  # week 3  -- Maracana Stadium, BAL @ DAL
    "LON02",  # week 4  -- Tottenham Hotspur Stadium, IND @ WAS
    "LON00",  # week 6  -- Wembley Stadium, HOU @ JAX
    "PAR00",  # week 7  -- Stade de France, PIT @ NO
    "MAD01",  # week 9  -- Bernabeu, CIN @ ATL
    "MUN01",  # week 10 -- FC Bayern Munich Stadium, NE @ DET
    "MEX00",  # week 11 -- Estadio Banorte, MIN @ SF
)


# ---------------------------------------------------------------------------
# THIS PLAN'S OWN COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-02 at plan close on 2026-09-11, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED LAST, NOT IN TASK 1, AND THAT ORDERING IS FORCED BY THE PROTOCOL ITSELF.
# The plan's text places this slot in Task 1, but Tasks 2 and 3 each add tests, so a
# Task-1 value would have been wrong at plan close and could only have been made
# right by EDITING it -- the append-once violation this very plan exists to make
# mechanical. The slot is written once, from the measurement, at the end. This is the
# same shape Plan 33-01 used for its own measured slots (commit edb3e06).
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run python -m pytest tests/unit -q
#         3029 passed, 4 skipped, 5 xfailed            3038 collected   318.94 s
#     uv run python -m pytest tests/integration -q
#         5 failed, 713 passed, 8 skipped, 9 xfailed    735 collected   625.62 s
#     uv run python -m pytest tests/api tests/test_*.py -q
#         367 passed                                    367 collected   117.46 s
#     SUM: 5 failed, 4109 passed, 12 skipped, 14 xfailed        4140 collected
#
# 4140 - 4073 (the post-33-01 three-tier collected total) = 67.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better:
#      6  tests/unit/test_phase33_state_shape.py
#      5  tests/unit/test_phase33_state_append_once.py
#      8  tests/unit/test_phase33_state_arithmetic.py
#      8  tests/unit/test_phase33_no_zero_failures_claim.py
#     12  tests/unit/test_phase33_clv_report_only.py
#     19  tests/unit/test_season_2026_fixture_schema.py
#      9  tests/integration/test_phase33_expected_failure_set.py
#     --
#     67
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS. No tier reported a
# production-store boundary crossing. The skipped count moved 9 -> 12 and all three
# of the new skips are this plan's own PENDING guards, each stepping aside by name:
# the post-phase arithmetic (waiting on Plans 33-03 .. 33-18), the Plan-33-06 identity
# columns, and the opt-in whole-tier verdict.
# ---------------------------------------------------------------------------

TESTS_ADDED_33_02: int = 67


# ---------------------------------------------------------------------------
# THE ELO WRITE PATH: the split rule, and the TRUE blast radius of both verbs.
#
# APPENDED by Plan 33-03 Task 1 on 2026-09-11. Nothing above this line was edited.
#
# The split rule has ONE committed home the tests import rather than re-listing:
# a ROW table accumulates history (one row per game, keyed on game_id) and is
# UPSERTED by the weekly path; a STATE artifact is current-state by definition and
# is REPLACED. Getting the split wrong in either direction is a real failure --
# upserting a state artifact strands dead teams forever, and replacing a row table
# destroys 24 seasons of burn-in that three deployed models read through.
# ---------------------------------------------------------------------------

ELO_ROW_TABLE_NAMES: tuple[str, ...] = (
    "elo_game_snapshots",
    "games_with_elo",
    "elo_rating_history",
)

ELO_STATE_ARTIFACT_NAMES: tuple[str, ...] = (
    "elo_ratings_current",
    "elo_ratings",
)

# Every path the two Elo write verbs touch, INCLUDING THE SHARED DUCKDB.
#
# WHY THE DUCKDB IS NAMED, MEASURED AGAINST data/storage.py RATHER THAN ASSUMED.
# `save_dataframe` defaults to `save_to_db=True` AND `save_to_parquet=True`
# (data/storage.py:826-836), so every artifact it writes lands in BOTH stores. The
# full rebuild routes all four tables through it. The live append routes the three
# ROW tables through `upsert_silver`, which is PARQUET-ONLY -- and that asymmetry is
# precisely why `EloBuilder._upsert_row_table` re-reads the combined parquet and
# replaces the DuckDB copy from it: `load_dataframe(source="auto")` resolves DuckDB
# FIRST (data/storage.py:959-963), so a parquet-only upsert would leave the database
# serving the pre-append rows while the parquet said otherwise.
#
# The staged generation tree and the pointer are named too. They are not incidental
# temp files: the pointer is the ONE atomic publish surface, and a phase that
# declared only the live artifacts would be under-declaring what a crash can leave
# behind.
ELO_WRITE_SET_INCLUDING_DUCKDB: tuple[str, ...] = (
    "data/silver/elo_game_snapshots.parquet",
    "data/silver/games_with_elo.parquet",
    "data/silver/elo_rating_history.parquet",
    "data/silver/elo_ratings_current.parquet",
    "data/silver/elo_ratings.json",
    "data/silver/elo_generation.json",
    "data/silver/elo_generations/",
    "data/nfl_predictions.duckdb",
)


# ---------------------------------------------------------------------------
# PLAN 33-03'S OWN COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-03 at plan close on 2026-09-11, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 1, for the reason Plan 33-02 recorded when it did the
# same: all three tasks add tests, so a Task-1 value would have been wrong at plan close
# and could only have been made right by EDITING it -- the append-once violation the
# protocol exists to prevent.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run python -m pytest tests/unit -q
#         3061 passed, 4 skipped, 5 xfailed           3070 collected   311.13 s
#     uv run python -m pytest tests/integration -q
#         5 failed, 737 passed, 8 skipped, 9 xfailed   759 collected   519.76 s
#     uv run python -m pytest tests/api tests/test_*.py -q
#         367 passed                                   367 collected   117.86 s
#     SUM: 5 failed, 4165 passed, 12 skipped, 14 xfailed       4196 collected
#
# 4196 - 4140 (Plan 33-02's three-tier collected total) = 56.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better:
#     19  tests/unit/test_elo_write_entry_points.py            (new)
#     10  tests/unit/test_elo_live_append.py                   (new)
#      3  tests/unit/test_elo_correctness.py                   (23 -> 26)
#      8  tests/integration/test_elo_generation_atomicity.py   (new)
#      5  tests/integration/test_build_elo_persist_refusal.py  (new)
#      6  tests/integration/test_elo_live_append_isolation.py  (new)
#      5  tests/integration/test_elo_convergence.py            (2 -> 7)
#     --
#     56
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS, unchanged. The SKIPPED
# count did not move (12 -> 12): this plan added no pending guard and stepped aside from
# nothing. No tier reported a production-store boundary crossing
# (STAT_SIGNATURE_OBSERVATIONS 3 + 5 + 0, WAL_SIBLING_OBSERVATIONS 0 in every tier).
# ---------------------------------------------------------------------------

TESTS_ADDED_33_03: int = 56


# ---------------------------------------------------------------------------
# THE SNAPSHOT TABLE'S WIDTH, BEFORE AND AFTER `is_provisional`.
#
# APPENDED by Plan 33-04 Task 1 on 2026-09-12. Nothing above this line was edited.
#
# THE 11 IS A MEASUREMENT, NOT A READING OF THE SOURCE CONSTANT. It was taken from
# the PRE-PHASE production snapshot table -- data/silver/elo_game_snapshots.parquet,
# 2,227 rows over seasons 2018-2025 -- so the "before" is the width of the table
# that actually exists rather than the width of a tuple somebody could edit in the
# same commit that widens it.
#
# The 12th column is `is_provisional`: True on a row emitted for a scheduled-but-
# unplayed game by `EloBuilder.snapshot_upcoming_week`, False on every row either
# canonical writer produces. It is APPENDED to the existing eleven and never
# inserted among them -- reordering that table moves every column three deployed
# models read through.
#
# There is NO null third state. Both writers set the flag explicitly, and the ONE
# read seam (`features.elo_features.ensure_provisional_flag`) fills a pre-flag
# parquet with False rather than leaving a NaN that would be neither.
# ---------------------------------------------------------------------------

SNAPSHOT_COLUMN_COUNT_BEFORE: int = 11
SNAPSHOT_COLUMN_COUNT_AFTER: int = 12


# ---------------------------------------------------------------------------
# THE GOLD JOIN SUBSET, and the FOUR trainer gold-loading boundaries.
#
# APPENDED by Plan 33-04 Task 2 on 2026-09-12. Nothing above this line was edited.
# ---------------------------------------------------------------------------

# The columns `features/elo_features.build_features` takes from the snapshot frame
# before merging it onto games. READ OFF THE LIVE SOURCE (the explicit subset in
# `build_features`, immediately above the merge), not inferred from what gold
# happens to contain.
#
# `game_id` IS THE JOIN KEY, so the subset carries SIX direct numeric fields plus
# the key. Plan 33-18's value-by-value comparison depends on that count being
# right: comparing seven fields would compare the key against itself and report a
# match it never earned, and comparing five would silently skip one.
#
# THIS IS ALSO WHY `is_provisional` REACHES NO GOLD MATRIX -- it is not in this
# list, and the subset is explicit rather than a `drop`-based exclusion, so a NEW
# snapshot column is excluded from gold by DEFAULT. That is the NF-08 COLUMN claim.
# It does NOT cover the ROW claim: the FULL snapshots frame (all twelve columns,
# every row) is handed to `_add_momentum_features` and `_add_rank_features`
# afterwards, and `_add_rank_features` selects `week <= W`, so same-week
# provisional ROWS do reach the four rank/percentile columns. Deliberate for live
# serving, asserted in tests/unit/test_elo_gold_features.py.
#
# Two of these names are RENAMED by the merge: `home_elo_pre` -> `home_elo` and
# `away_elo_pre` -> `away_elo`. The list is the SNAPSHOT-side spelling, which is
# the side the join subset is written in.
ELO_GOLD_JOIN_SUBSET: tuple[str, ...] = (
    "game_id",
    "home_elo_pre",
    "away_elo_pre",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
    "elo_prob_home",
    "hfa_used",
)

# Every place a model training entry point loads a GOLD feature matrix, as
# (module, function, line).
#
# WHY FOUR AND NOT ONE (Codex HIGH, verified against live source in the 33-04
# replan). A guard placed inside `features/elo_features.py` protects NONE of these:
# every one of them reads gold DIRECTLY. Three call
# `load_dataframe("features_*", layer="gold")` inside a `try:`; the fourth calls
# `pd.read_parquet(features_path)` and bypasses `load_dataframe` entirely. So the
# refusal is wired at each boundary, immediately after the load and BEFORE any
# filtering -- a provisional row filtered out of sight would still be trained on in
# a different slice.
#
# THE LINE NUMBER IS PROVENANCE, NOT AN ASSERTION, and that is the same ruling
# CLV_REPORT_ONLY_SITES records above: a line number makes a pinned tuple brittle
# against any edit above it, and the question this tuple answers is WHICH ENTRY
# POINTS load gold. The drift test therefore resolves `(module, function)` and
# asserts the guard call is present in that function's source. A FIFTH trainer
# appearing without the guard is a drift failure; a trainer whose guard call moved
# down the file is not.
#
# MEASURED 2026-09-12 against the four files as Plan 33-04 Task 2 leaves them. The
# guard call is inserted BELOW each load, so these are also the pre-edit positions.
TRAINER_GOLD_LOAD_SITES: tuple[tuple[str, str, int], ...] = (
    ("models.train_wp", "main", 1144),
    ("models.train_ats", "main", 1163),
    ("models.train_ou", "main", 1423),
    ("models.train", "main", 632),
)


# ---------------------------------------------------------------------------
# PLAN 33-04'S OWN COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-04 at plan close on 2026-09-12, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 1, for the reason Plans 33-02 and 33-03 recorded
# when they did the same: both tasks add tests, so a Task-1 value would have been
# wrong at plan close and could only have been made right by EDITING it -- the
# append-once violation the protocol exists to prevent.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run python -m pytest tests/unit -q
#         3096 passed, 4 skipped, 5 xfailed          3105 collected   309.11 s
#     uv run python -m pytest tests/integration -q
#         5 failed, 743 passed, 8 skipped, 9 xfailed  765 collected   517.04 s
#     uv run python -m pytest tests/api tests/test_*.py -q
#         367 passed                                  367 collected   119.60 s
#     SUM: 5 failed, 4206 passed, 12 skipped, 14 xfailed       4237 collected
#
# 4237 - 4196 (Plan 33-03's three-tier collected total) = 41.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better:
#     13  tests/unit/test_snapshot_upcoming_week.py           (new)
#     20  tests/unit/test_provisional_training_refusal.py     (new)
#      2  tests/unit/test_elo_gold_features.py                (10 -> 12)
#      6  tests/integration/test_gold_snap_injury_columns.py  (12 -> 18)
#     --
#     41
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS, unchanged. The SKIPPED
# count did not move (12 -> 12): this plan added no pending guard and stepped aside
# from nothing. No tier reported a production-store boundary crossing
# (STAT_SIGNATURE_OBSERVATIONS 3 + 5 + 0, WAL_SIBLING_OBSERVATIONS 0 in every tier).
# ---------------------------------------------------------------------------

TESTS_ADDED_33_04: int = 41


# ---------------------------------------------------------------------------
# THE TWO INSTANTS THE DET @ BUF CONCESSION IS DATED AGAINST.
#
# APPENDED by Plan 33-05 Task 1 on 2026-09-12. Nothing above this line was edited.
#
# THE FIRST IS MEASURED FROM THE ONE FREEZE RULE, NOT TYPED FROM A CALENDAR:
#     get_synthetic_snapshot_ts("2026-09-17") -> 2026-09-11T22:00:00+00:00
# which is 2026-09-11 18:00 America/New_York -- the Friday 6 PM Eastern freeze
# preceding the week-2 Thursday game `2026_02_DET_BUF` (gameday 2026-09-17). That
# instant belongs to WEEK 1's Friday run, seven days before the week-2 Sunday
# slate's own freeze (2026-09-18T22:00:00+00:00), which is precisely why a
# WEEK-scoped fence would lose it and why D33-28 scopes selection by INSTANT.
#
# THE SECOND IS THE INSTANT THIS PHASE WAS PLANNED AT. It is recorded so the
# committed readout's concession is DATED rather than asserted: at plan time the
# week-2 Thursday freeze was roughly 100 minutes in the FUTURE, and Phase 33 could
# not be planned, executed and verified inside that window. So week 2's Thursday
# game is lost to the forward ledger regardless of what this plan does -- exactly
# as week 1 was conceded by D40-04 -- and the honest outcome is a MISSING row with
# a dated reason rather than a present one carrying a fabricated observation time.
# ---------------------------------------------------------------------------

WEEK_2_THURSDAY_FREEZE_UTC: str = "2026-09-11T22:00:00+00:00"
PLAN_TIME_UTC_INSTANT: str = "2026-09-11T20:19:33+00:00"


# ---------------------------------------------------------------------------
# THE BET-LIST SCHEMA BUMP, AND THE STORED ROWS IT DELIBERATELY DOES NOT TOUCH.
#
# APPENDED by Plan 33-05 Task 3 on 2026-09-12. Nothing above this line was edited.
#
# THE OWNER RULED ON 2026-09-12, selecting `decided-at-utc-only` (D33-06's own
# recommendation) at the Task-2 blocking-human decision checkpoint: ONE new column
# `decided_at_utc`, typed VARCHAR, holding an ISO-8601 string with an explicit UTC
# offset, placed in the IMMUTABLE half at INDEX 22 -- immediately after
# `validation_type` and immediately before `grading_status`, the last position of
# the immutable half. 22 + 6 = 28 becomes 23 + 6 = 29 in ONE locked order.
#
# WHY VARCHAR AND NOT TIMESTAMP: it matches the `snapshot_ts` / `freeze_ts` siblings
# it is COMPARED AGAINST, so the single strict parse helper landed in Task 1 serves
# ONE representation. `graded_at`'s TIMESTAMP type is not the sibling convention
# here; a second representation would need a second code path, which is the shape
# D33-27 refused.
#
# THE STORED ROWS ARE MEASURED, NOT ASSUMED. Read from
# `outputs/bet_list/bet_list.parquet` on 2026-09-12:
#     234 rows x 28 columns
#     provenance:      234 backtest_replay, 0 forward
#     grading_status:  144 pending, 51 win, 37 loss, 2 push
# Every one of the 234 is already PAST its freeze, so all 234 take NULL and are
# NEVER backfilled -- filling them from `snapshot_ts` would stamp an observation
# time nobody observed, which is this plan's own named prohibition.
#
# `freeze_ts` is stored as a tz-AWARE Eastern string on every one of those rows,
# which is exactly why `_is_frozen`'s naive branch is unreachable from real data and
# therefore safe to convert to a raise now (T-33-23).
# ---------------------------------------------------------------------------

BET_LIST_COLUMN_COUNT_BEFORE: int = 28
BET_LIST_COLUMN_COUNT_AFTER: int = 29
DECIDED_AT_COLUMN: str = "decided_at_utc"
BET_LIST_REPLAY_ROW_COUNT: int = 234


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-05 at plan close on 2026-09-12, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 1, for the reason Plans 33-02, 33-03 and 33-04
# recorded when they did the same: three tasks add tests, so a Task-1 value would
# have been wrong at plan close and could only have been made right by EDITING it
# -- the append-once violation the protocol exists to prevent.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run python -m pytest tests/unit -q
#         3167 passed, 4 skipped, 5 xfailed          3176 collected   308.55 s
#     uv run python -m pytest tests/integration -q
#         5 failed, 747 passed, 8 skipped, 9 xfailed  769 collected   522.76 s
#     uv run python -m pytest tests/api tests/test_*.py -q
#         370 passed                                  370 collected   117.23 s
#     SUM: 5 failed, 4284 passed, 12 skipped, 14 xfailed       4315 collected
#
# 4315 - 4237 (Plan 33-04's three-tier collected total) = 78.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better:
#     23  tests/unit/test_freeze_fence_binding.py            (new)
#     13  tests/unit/test_freeze_parse_single_source.py      (new)
#     12  tests/unit/test_selection_scoped_by_freeze_instant.py  (new)
#     22  tests/unit/test_decided_at_utc.py                  (new)
#      1  tests/unit/test_bet_list_schema.py                 (25 -> 26)
#      3  tests/api/test_cache_betting.py                    ( 9 -> 12)
#      4  tests/integration/test_bet_list_completeness.py    ( 8 -> 12)
#     --
#     78
#
# SIX FURTHER TEST MODULES WERE EDITED AND ADDED ZERO NODES, which is why they do
# not appear above and why the arithmetic still closes: test_weekly_bet_list.py,
# test_idempotency.py, test_bet_list_marker.py, test_bet_list_entry_point.py,
# test_friday_prediction_step.py and test_cold_start_bet_list_recovery.py all took
# FIXTURE changes only -- an observation stamp, or a pinned clock to match a pinned
# historical week. A fixture edit that adds no node is invisible to this equation by
# design; it is recorded in the SUMMARY's deviation 2 instead.
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS, unchanged. The SKIPPED
# count did not move (12 -> 12) and neither did the XFAILED (14 -> 14): this plan
# added no pending guard and stepped aside from nothing. No tier reported a
# production-store boundary crossing (STAT_SIGNATURE_OBSERVATIONS 3 + 5 + 0,
# WAL_SIBLING_OBSERVATIONS 0 in every tier).
# ---------------------------------------------------------------------------

TESTS_ADDED_33_05: int = 78


# ---------------------------------------------------------------------------
# THE EIGHT INTERNATIONAL VENUES, AS THE OWNER RATIFIED THEM.
#
# APPENDED by Plan 33-06 Task 1 on 2026-09-12. Nothing above this line was edited.
#
# THE OWNER RULED ON 2026-09-12 at the Task-1 blocking-human checkpoint, selecting
# "approve all forty as tabled", including both flagged minors:
#
#     Approved all forty values as tabled, including MEX00's elevation written as
#     the MEASURED 7365 (not CONTEXT's round 7350), `subtropical_highland` as a new
#     `climate_zone` token for Mexico City, and MEL00's `outdoor` roof_type on
#     structural-absence-plus-corroboration sourcing.
#
# The forty are the five cells the R11 acceptance is stated over -- latitude,
# longitude, elevation_ft, timezone, roof_type -- across the eight venues. The
# remaining descriptive cells (city, country, capacity, surface, climate_zone) were
# tabled and ratified alongside them and are recorded here too, so `data/venues.json`
# has ONE committed source rather than two half-sources that can disagree.
#
# NONE OF THESE IS INHERITED FROM THE FEED. `roof_type` in particular is entered
# EXPLICITLY for all eight: see FEED_ROOF_DISAGREEMENTS below, where the feed is
# WRONG on three of them.
#
# Field order, stated once and asserted by tests/unit/test_venues_json_international.py:
#     (stadium_id, venue_id, venue_name, city, country, latitude, longitude,
#      elevation_ft, roof_type, surface, capacity, climate_zone, timezone)
#
# `venue_name` MUST equal the feed's `stadium` string EXACTLY. Every one of the
# eight feed strings is PLAIN ASCII with NO diacritic -- "Maracana Stadium" not
# Maracana with a tilde, "Bernabeu" not Bernabeu with an acute -- because
# scripts/ingest_games._load_venue_lookup keys on the LOWERCASED NAME. A "corrected"
# spelling here would silently miss that third resolver and fall through to the
# nflverse roof map, re-introducing the three false domes on one surface while the
# other two resolvers are right. Partially repaired, silently, is the worst state.
#
# `home_teams` is empty for all eight: no NFL team calls any of them home, and an
# empty list is what makes the home-team resolvers decline them rather than claim
# them.
# ---------------------------------------------------------------------------

INTERNATIONAL_VENUE_FACTS: tuple[tuple[object, ...], ...] = (
    (
        "MEL00",
        "melbourne_cricket_ground",
        "Melbourne Cricket Ground",
        "Melbourne",
        "Australia",
        -37.8199,
        144.9834,
        43,
        "outdoor",
        "Matrix Turf",
        100024,
        "oceanic",
        "Australia/Melbourne",
    ),
    (
        "RIO00",
        "maracana_stadium",
        "Maracana Stadium",
        "Rio de Janeiro",
        "Brazil",
        -22.9122,
        -43.2303,
        49,
        "outdoor",
        "Matrix Turf",
        73139,
        "tropical",
        "America/Sao_Paulo",
    ),
    (
        "LON02",
        "tottenham_hotspur_stadium",
        "Tottenham Hotspur Stadium",
        "London",
        "United Kingdom",
        51.6044,
        -0.0664,
        43,
        "outdoor",
        "Grass",
        62850,
        "oceanic",
        "Europe/London",
    ),
    (
        "LON00",
        "wembley_stadium",
        "Wembley Stadium",
        "London",
        "United Kingdom",
        51.5556,
        -0.2794,
        154,
        "outdoor",
        "Grass",
        90000,
        "oceanic",
        "Europe/London",
    ),
    (
        "PAR00",
        "stade_de_france",
        "Stade de France",
        "Saint-Denis",
        "France",
        48.9244,
        2.3600,
        108,
        "outdoor",
        "Sport Turf",
        81338,
        "oceanic",
        "Europe/Paris",
    ),
    (
        "MAD01",
        "bernabeu",
        "Bernabeu",
        "Madrid",
        "Spain",
        40.4531,
        -3.6883,
        2349,
        "retractable",
        "FieldTurf",
        83186,
        "mediterranean",
        "Europe/Madrid",
    ),
    (
        "MUN01",
        "fc_bayern_munich_stadium",
        "FC Bayern Munich Stadium",
        "Munich",
        "Germany",
        48.2188,
        11.6248,
        1611,
        "outdoor",
        "FieldTurf",
        75024,
        "oceanic",
        "Europe/Berlin",
    ),
    (
        "MEX00",
        "estadio_banorte",
        "Estadio Banorte",
        "Mexico City",
        "Mexico",
        19.3031,
        -99.1506,
        7365,
        "outdoor",
        "Grass",
        87523,
        "subtropical_highland",
        "America/Mexico_City",
    ),
)

# THE FEED IS WRONG ON THREE ROOFS, AND THIS IS THE RECORD OF IT (D33-16).
#
# MEASURED from the eight neutral-site rows of
# data/bronze/schedules_raw_bronze_2026_W01_20260911T110252.parquet on 2026-09-12:
#
#     MEL00  Melbourne Cricket Ground   roof=dome      -> RATIFIED outdoor
#     RIO00  Maracana Stadium           roof=None      -> RATIFIED outdoor
#     LON02  Tottenham Hotspur Stadium  roof=outdoors  -> RATIFIED outdoor      (agrees)
#     LON00  Wembley Stadium            roof=outdoors  -> RATIFIED outdoor      (agrees)
#     PAR00  Stade de France            roof=dome      -> RATIFIED outdoor
#     MAD01  Bernabeu                   roof=None      -> RATIFIED retractable
#     MUN01  FC Bayern Munich Stadium   roof=dome      -> RATIFIED outdoor
#     MEX00  Estadio Banorte            roof=outdoors  -> RATIFIED outdoor      (agrees)
#
# A DISAGREEMENT is a venue where the feed carries a roof value that MAPS TO A
# DIFFERENT project roof than the ratified one. RIO00 and MAD01 carry NO value, so
# they are ABSENCES, not disagreements -- an absent value cannot contradict anything,
# and folding the two categories together would leave the recorder unable to tell a
# feed correction from a feed backfill. The three below are genuine contradictions:
# the feed says dome, _NFLVERSE_ROOF_MAP sends dome to indoor, and _is_outdoor_game
# sends indoor to a SKIP -- so an inherited value would zero the weather on three
# genuinely open-air games with no error raised.
#
# tests/unit/test_venues_json_international.py recomputes this set from the capture
# and asserts equality, so a future feed correction shows up as a DIFF against this
# tuple rather than as a silent flip.
FEED_ROOF_DISAGREEMENTS: tuple[str, ...] = ("MEL00", "PAR00", "MUN01")

# The feed's measured roof cell for each of the eight, recorded verbatim so the
# recorder above has something to be checked against rather than only itself. None
# means the cell is absent in the capture.
FEED_ROOF_VALUES_2026: tuple[tuple[str, str | None], ...] = (
    ("MEL00", "dome"),
    ("RIO00", None),
    ("LON02", "outdoors"),
    ("LON00", "outdoors"),
    ("PAR00", "dome"),
    ("MAD01", None),
    ("MUN01", "dome"),
    ("MEX00", "outdoors"),
)

# THE 91 HISTORICAL NEUTRAL-SITE GAMES THAT RESOLVED TO THE HOME TEAM'S OWN STADIUM
# FOR THE LIFE OF THIS PROJECT (D33-15), RE-DERIVED HERE RATHER THAN COPIED.
#
# The phase CONTEXT recorded these as measured but the researcher did not reproduce
# them, and a readout that publishes a wrong measured finding is the class this
# project treats as serious. All five were RE-DERIVED on 2026-09-12 from the pinned
# schedules (data.upstream_pin.load_schedules over 2002-2025, 6,499 rows) and every
# one reproduced EXACTLY -- there is no divergence to record.
#
#     91  rows carrying location == Neutral across 2002-2025
#     27  distinct stadium_id values among them
#      1  in 2002, rising to 8 in 2025
#
# Per-model TRAINING-row share, taken over each deployed artifact's own
# config.train_seasons (artifacts/latest.json -> wp_20260824_113325 [2018-2019],
# ats_20260605_220128 [2015-2019], ou_20260326_163930 [2018-2019]):
#
#     wp    10 /   534  = 1.87%
#     ats   25 / 1,335  = 1.87%
#     ou    10 /   534  = 1.87%
#
# against 2026's 8 / 272 = 2.94%. The forward season carries the defect at roughly
# 1.6x the rate the deployed models were trained under, which is the reason this is
# a disclosure and not a footnote. NOTHING IS REPAIRED HERE: history keeps its
# home_team resolution (D33-15 gates the new routing on season >= 2026), and the 91
# rows stay exactly as they have always been.
HISTORICAL_NEUTRAL_MISRESOLUTION: dict[str, object] = {
    "games": 91,
    "distinct_stadium_ids": 27,
    "first_season": 2002,
    "first_season_count": 1,
    "last_season": 2025,
    "last_season_count": 8,
    "train_share": {
        "wp": (10, 534),
        "ats": (25, 1335),
        "ou": (10, 534),
    },
    "forward_share_2026": (8, 272),
    "rederived_on": "2026-09-12",
    "diverged_from_context": False,
}


# ---------------------------------------------------------------------------
# THE THREE IDENTITY COLUMNS, AND THE SILVER WIDTH THEY MOVE.
#
# APPENDED by Plan 33-06 Task 3 on 2026-09-12. Nothing above this line was edited.
#
# THE MEASURED PRE-PHASE SILVER FRAME, read from data/silver/games.parquet on
# 2026-09-12: 6,499 rows over 2002-2025 with ZERO 2026 rows, 15 columns, and
#
#     every neutral_site cell reads False
#     every season_type cell reads the string Regular
#
# across every one of them -- including all 91 historical neutral-site games and
# every WC/DIV/CON/SB game ever played.
#
# THE CAUSE IS A DEFAULT THAT FIRES 100% OF THE TIME. The 46-column bronze schedule
# contains `location` and `stadium_id` and contains NEITHER `neutral_site` NOR
# `season_type`, so `row.get("season_type", "Regular")` and
# `row.get("neutral_site", False)` never read a feed value at all. A default that
# never loses is not a default; it is a constant nobody chose.
#
# THIS PLAN CHANGES THE CODE, NOT THE STORE. The silver re-ingest that backfills
# these three across all seasons is Plan 33-12's, deliberately separated so the gold
# rebuild's blast radius stays attributable to one cause. So 15 -> 18 is what the
# NEXT ingest will produce, not what data/silver/games.parquet holds today.
# ---------------------------------------------------------------------------

PLAN_33_06_IDENTITY_COLUMNS: tuple[str, ...] = (
    "stadium_id",
    "neutral_site",
    "season_type",
)

SILVER_GAMES_COLUMNS_BEFORE: int = 15
SILVER_GAMES_COLUMNS_AFTER: int = 18

# The five `game_type` values the feed uses and the two-value partition D33-17
# coarsens them to. Recorded so the COARSENING is visible as a deliberate choice
# rather than looking like a lossy duplicate of `game_type`: the season-close readout
# partitions regular-versus-post, and `utils/similar_games.py:444` already reads
# `season_type` with a `game_type` fallback and today receives a constant.
SEASON_TYPE_PARTITION: tuple[tuple[str, str], ...] = (
    ("REG", "Regular"),
    ("WC", "Postseason"),
    ("DIV", "Postseason"),
    ("CON", "Postseason"),
    ("SB", "Postseason"),
)


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-06 at plan close on 2026-09-12, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 1, for the reason Plans 33-02 through 33-05 each
# recorded when they did the same: two tasks add tests, so a Task-1 value would have
# been wrong at plan close and could only have been made right by EDITING it -- the
# append-once violation the protocol exists to prevent.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run python -m pytest tests/unit -q
#         3263 passed, 3 skipped, 5 xfailed          3271 collected   323.97 s
#     uv run python -m pytest tests/integration -q
#         5 failed, 766 passed, 8 skipped, 9 xfailed  788 collected   590.38 s
#     uv run python -m pytest tests/api tests/test_*.py -q
#         370 passed                                  370 collected   120.26 s
#     SUM: 5 failed, 4399 passed, 11 skipped, 14 xfailed       4429 collected
#
# 4429 - 4315 (Plan 33-05's three-tier collected total) = 114.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better:
#     36  tests/unit/test_venues_json_international.py       (new)
#     27  tests/unit/test_stadium_id_routing.py              (new)
#     26  tests/unit/test_games_identity_columns.py          (new)
#     14  tests/integration/test_neutral_site_venues_2026.py (new)
#      3  tests/unit/test_nflverse_roof_map_parity.py        (new)
#      5  tests/integration/test_gold_write_scope.py         (10 -> 15)
#      3  tests/unit/test_contextual_extensions.py           (24 -> 27)
#     --
#    114
#
# ONE FURTHER MODULE WAS EDITED AND ADDED ZERO NODES, which is why it does not
# appear above and why the arithmetic still closes:
# tests/unit/test_season_2026_fixture_schema.py stayed at 19. Plan 33-02's pinned
# defect observation was INVERTED IN PLACE -- renamed from
# test_the_production_transform_DROPS_the_neutral_site_fact to
# ..._CARRIES_..., with its final assertion flipped from "no row is neutral" to
# "exactly 8 rows are" -- rather than deleted. A rename plus an inverted assertion
# is one node before and one node after.
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS, unchanged, and that
# set matters more than usual for this plan: two of the five digest GOLD, and gold
# was rebuilt during this plan's execution after a test of this plan's own
# destroyed it (SUMMARY deviation 1). The failing NODE IDS are the same five and no
# previously-passing gold-reading test moved, which is the strongest available
# evidence that the rebuild reproduced what was lost -- it is not proof of byte
# identity, because the original was gone before it could be digested.
#
# The SKIPPED count moved 12 -> 11 and the single un-skip is deliberate: Plan
# 33-02's PENDING guard
# test_the_three_plan_33_06_identity_columns_reach_the_transformed_frame steps
# aside only while PLAN_33_06_IDENTITY_COLUMNS is absent from this file. Task 3
# appended it, so the guard now RUNS and passes. The XFAILED count did not move
# (14 -> 14). No tier reported a production-store boundary crossing in the final
# run (STAT_SIGNATURE_OBSERVATIONS 3 + 5 + 0, WAL_SIBLING_OBSERVATIONS 0 in every
# tier).
# ---------------------------------------------------------------------------

TESTS_ADDED_33_06: int = 114


# ---------------------------------------------------------------------------
# THE REQUIRED-ARTIFACT SET AND THE BOUNDARY EACH ONE IS CHECKED AT.
#
# APPENDED by Plan 33-07 Task 2 on 2026-09-12. Nothing above this line was edited.
#
# WHY THE PATHS ARE RECORDED HERE. `pipeline.steps._REQUIRED_ARTIFACTS` became
# `(path, phase_boundary, coverage_check)` TRIPLES in this task. A test that wants
# to assert the conversion neither dropped an artifact nor added one has to compare
# the path components against something, and re-typing six strings inside the test
# would be the second-list failure this repository has already paid for twice. The
# six below are the SAME six the gate required before the conversion, in the same
# order.
#
# WHY THE BOUNDARY MAP IS RECORDED HERE TOO. An artifact checked at the WRONG
# boundary does not fail loudly -- it produces a FALSE stale-artifact refusal (gold
# checked before the step that builds it) or a check that never runs (silver checked
# after everything that reads it). Neither shape announces itself, so the mapping
# needs a committed home where a later move is a drift failure rather than a quiet
# behaviour change.
#
# MEASURED 2026-09-12 from the built constant, not transcribed from the plan.
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_PATHS: tuple[str, ...] = (
    "data/silver/games.parquet",
    "data/silver/elo_game_snapshots.parquet",
    "data/silver/team_form_features.parquet",
    "data/gold/features_wp.parquet",
    "data/gold/features_ats.parquet",
    "data/gold/features_ou.parquet",
)

# `predictions` is the THIRD member of the closed boundary vocabulary and appears in
# NO row below. That is not an omission: the prediction artifact's filename embeds
# the season and the week, so it has no static path to declare and
# `step_verify_prediction_currency` resolves it per run.
ARTIFACT_PHASE_BOUNDARY_MAP: tuple[tuple[str, str], ...] = (
    ("data/silver/games.parquet", "data"),
    ("data/silver/elo_game_snapshots.parquet", "data"),
    ("data/silver/team_form_features.parquet", "data"),
    ("data/gold/features_wp.parquet", "gold"),
    ("data/gold/features_ats.parquet", "gold"),
    ("data/gold/features_ou.parquet", "gold"),
)


# ---------------------------------------------------------------------------
# THE REGISTRY THE LIVE-SEASON CAPTURE JOINS.
#
# APPENDED by Plan 33-07 Task 3 on 2026-09-12. Nothing above this line was edited.
#
# WHY THREE COUNTS AND NOT ONE. The registry grew by THREE steps in this one plan --
# Task 2 added `verify_gold_currency` and `verify_prediction_currency` at their own
# phase boundaries, and Task 3 added `capture_live_season` at the head of the DATA
# phase -- so a single after-count could be reached by two different insertions and
# would not distinguish them. The DATA-phase count is recorded separately because
# `--data-only` filters on the phase, so a step landing in the wrong phase changes
# what that mode runs without changing the total.
#
# EACH COUNT WAS MEASURED FROM THE BUILT REGISTRY, not copied from the docstring.
# The docstring is a string a human maintains; the registry is the object the
# orchestrator iterates, and this phase has already seen a step-count comment go
# stale. `REGISTRY_STEP_COUNT_BEFORE` is the pre-plan value recorded by the
# docstring AND by tests/unit/test_pipeline_orchestrator.py at commit 360c1c8.
#
#     uv run python -c "from pipeline.steps import build_step_registry, PipelinePhase;
#     r = build_step_registry(); print(len(r),
#     len([s for s in r if s.phase == PipelinePhase.DATA]))"
#     -> 22 9
# ---------------------------------------------------------------------------

REGISTRY_STEP_COUNT_BEFORE: int = 19
REGISTRY_STEP_COUNT_AFTER: int = 22
DATA_PHASE_STEP_COUNT_AFTER: int = 9


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-07 at plan close on 2026-09-12, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 2, for the reason Plans 33-02 through 33-06 each
# recorded when they did the same: three tasks add tests, so a Task-2 value would have
# been wrong at plan close and could only have been made right by EDITING it -- the
# append-once violation the protocol exists to prevent.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run pytest tests/unit -q
#         3305 passed, 3 skipped, 5 xfailed          3313 collected   318.26 s
#     uv run pytest tests/integration -q
#         5 failed, 767 passed, 8 skipped, 9 xfailed  789 collected   593.86 s
#     uv run pytest tests/api tests/test_*.py -q
#         370 passed                                  370 collected   130.54 s
#     SUM: 5 failed, 4442 passed, 11 skipped, 14 xfailed       4472 collected
#
# 4472 - 4429 (Plan 33-06's three-tier collected total) = 43.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better:
#     17  tests/unit/test_required_artifacts_currency.py     (new)
#      8  tests/unit/test_capture_step_season_week.py        (new)
#      6  tests/unit/test_weekly_decision_frame.py           (new)
#      4  tests/unit/test_empty_week_refusals.py             (new)
#      6  tests/unit/test_step_registry_order.py             (10 -> 16)
#      1  tests/unit/test_pipeline_model_validation.py       (13 -> 14)
#      1  tests/integration/test_friday_prediction_step.py   (4 -> 5)
#     --
#     43
#
# ONE FURTHER MODULE WAS EDITED AND ADDED ZERO NODES, which is why it does not appear
# above and why the arithmetic still closes:
# tests/unit/test_pipeline_orchestrator.py stayed at 28. Its step-count test was
# RENAMED (test_build_step_registry_returns_19_steps -> ..._22_steps) with its
# expectation moved deliberately, and two sibling expectations were updated in place.
# A rename plus a changed expectation is one node before and one node after.
#
# A PLAN-TEXT CORRECTION, recorded rather than quietly absorbed: Plan 33-07's Task 3
# read_first states that tests/unit/test_step_registry_order.py "EXISTS with 11 tests".
# Measured at the plan's base commit 360c1c8 it carries TEN. Six were added, not five,
# and the arithmetic above uses the measured number.
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS, unchanged. The SKIPPED
# count did not move (11 -> 11) and neither did XFAILED (14 -> 14). No tier reported a
# production-store boundary crossing (STAT_SIGNATURE_OBSERVATIONS 3 + 5 + 0,
# WAL_SIBLING_OBSERVATIONS 0 in every tier).
# ---------------------------------------------------------------------------

TESTS_ADDED_33_07: int = 43


# ---------------------------------------------------------------------------
# THE FROZEN GATE BASELINE BLOCK, PRESERVED AS A RECORD RATHER THAN USED AS A
# COMPARATOR.
#
# APPENDED by Plan 33-08 Task 2 on 2026-09-12. Nothing above this line was edited.
#
# THE OWNER RULED `live-rescore` ON 2026-09-12. The five SECONDARY comparator
# scalars stop reading `config/gate.toml`'s `[baseline.*]` block and start reading
# a LIVE PAIRED RE-SCORE of the deployed incumbent on the same gold the candidate
# was scored on -- the shape `_pooled_floor_reasons` has always had. The block
# below is therefore no longer the operative comparator; it is a HISTORICAL RECORD
# and it must stay BYTE-UNTOUCHED.
#
# WHY A DIGEST AND NOT A PROMISE. Two of the five deliberate tripwires ARE the
# gate-baseline disclosure. Re-freezing the block would make them reproduce --
# clearing a disclosure by making it pass -- so the phase needs a mechanical way to
# say the bytes did not move. `git diff` alone is not that instrument: a committed
# edit shows a clean working tree. A digest anchored here, compared by
# tests/unit/test_phase33_gate_toml_untouched.py, fails on a committed edit too.
#
# THE BOUNDS ARE MEASURED, NOT TRANSCRIBED. Plan 33-08's own text states the block
# spans lines (168, 265). The file carries 264 lines
# (`len(Path('config/gate.toml').read_text(encoding='utf-8').splitlines())`), so
# the upper bound overshoots the end of the file by one. The measured bound is
# recorded instead. The digest is identical either way -- a slice past the end is
# harmless -- which is exactly why the overshoot would never have announced itself.
#
# MEASURED 2026-09-12 on commit 549eb68. The span is the 97 lines from
# '[baseline.wp.pooled]' to the file's final 'n = 272', joined on LF and LF-terminated,
# then sha256'd:
#
#     uv run python -c "import hashlib, pathlib; s = pathlib.Path(
#     'config/gate.toml').read_text(encoding='utf-8').splitlines()[167:264]; print(
#     hashlib.sha256(chr(10).join(s).encode() + b'\n').hexdigest())"
#
# NEWLINE-NORMALIZED, per the idiom stated at
# tests/unit/test_preregistration_ancestry.py:32-39: this repository has
# core.autocrlf=true and no .gitattributes, so a tracked text file is LF in the git
# blob and CRLF in a Windows working tree. The value below was checked against
# `git cat-file blob HEAD:config/gate.toml` (LF, 0 CRLF pairs) and matches, so it
# reproduces on any checkout rather than only on the machine that measured it.
# ---------------------------------------------------------------------------

GATE_TOML_BASELINE_LINES: tuple[int, int] = (168, 264)
GATE_TOML_BASELINE_SHA256: str = (
    "0d5628da61593c73049c5f94e73df6f2626840f7740a2080e4418a18ff515d43"
)

# The pre-phase production manifest, VERBATIM from artifacts/latest.json at commit
# 549eb68. Recorded here so Plan 33-15's retained-incumbent assertions have a
# committed reference rather than re-reading the file they are trying to prove
# unchanged. FOUR entries, not three: `blend` shares the one production swap
# surface and is the entry a per-target promotion is most likely to move by
# accident (T-33-42).
INCUMBENT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("wp", "wp_20260824_113325"),
    ("ats", "ats_20260605_220128"),
    ("ou", "ou_20260326_163930"),
    ("blend", "blend_dynamic_20260606_020635"),
)

# FIVE, and the count is load-bearing. An earlier draft of Plan 33-08 said FOUR,
# and a completeness check written against four would have PASSED while one scalar
# went unchecked. The five are: WP accuracy, WP ECE, WP Brier, ATS MAE, O/U MAE.
SECONDARY_SCALAR_COUNT: int = 5

# The judge that renders a Phase-33 verdict. A permanent semantic change to
# deployment policy needs a name, because a verdict that does not say which judge
# produced it cannot be compared against a later one.
JUDGE_VERSION_PHASE33: str = "phase33-live-secondary-rescore-1"


# ---------------------------------------------------------------------------
# THE PRE-REGISTERED FIX-CYCLE ALLOWANCE, AND THE CLOSED VERDICT VOCABULARY.
#
# APPENDED by Plan 33-08 Task 3 on 2026-09-12. Nothing above this line was edited.
#
# ZERO, DECLARED BEFORE ANY PHASE-33 VERDICT EXISTS. Phase 30 pre-registered a
# single fix-cycle lever and it went UNSPENT for BOTH failing targets -- not
# because anybody overlooked it, but because it had no unspent move. Phase 33 says
# the same thing up front instead of discovering it by accident: the training
# window belongs to Phase 37's recipe, the feature groups were bindingly ruled in
# Phase 30 (snap KEEP / situational KEEP / injury DROP), and hyperparameter search
# is out of scope, so an allowance of one would have nothing legitimate to spend.
#
# D33-33's PRE-FLIGHT HEALTH CHECK DOES NOT CREATE A RETRY STATE. It answers "what
# if the environment breaks mid-run" by making the environment fail BEFORE any
# number exists, not by allowing a second look after one does. The rule itself is
# unchanged and absolute: no re-runnable failure category, no environmental-abort
# escape hatch, no post-scoring retry. That distinction is easy to erode on a later
# reading, which is why it is recorded here as well as in the runner's docstring.
#
# THE VOCABULARY HAS THREE MEMBERS AND THE THIRD IS NOT DECORATION.
# backtest/diagnose.py:249-285 DELIBERATELY returns null `t` and `p` below
# MIN_CLV_SAMPLE, and this phase's own success criterion says a zero-eligible-row
# target is REFUSED. A schema demanding non-null statistics everywhere would reject
# that legitimate refusal as malformed, and the only way to make such a record
# valid would be to fabricate a number. So PASS and FAIL require statistics;
# UNTESTABLE_REFUSAL permits nulls and requires a stated reason, always retains the
# incumbent, and is never a promotion.
# ---------------------------------------------------------------------------

FIX_CYCLE_ALLOWANCE: int = 0
GATE_VERDICT_STATES: tuple[str, ...] = ("PASS", "FAIL", "UNTESTABLE_REFUSAL")


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-08 at plan close on 2026-09-12, AFTER the three-tier
# measurement that produced it. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 2, for the reason Plans 33-02 through 33-07 each
# recorded when they did the same: two tasks add tests, so a Task-2 value would have
# been wrong at plan close and could only have been made right by EDITING it -- the
# append-once violation the protocol exists to prevent.
#
# MEASURED over all THREE tiers with the guard armed and NFL_GUARD_OBSERVATIONS=1:
#     uv run pytest tests/unit -q
#         3425 passed, 3 skipped, 5 xfailed          3433 collected   334.29 s
#     uv run pytest tests/integration -q
#         5 failed, 767 passed, 8 skipped, 9 xfailed  789 collected   620.39 s
#     uv run pytest tests/api tests/test_*.py -q
#         370 passed                                  370 collected   125.49 s
#     SUM: 5 failed, 4562 passed, 11 skipped, 14 xfailed       4592 collected
#
# 4592 - 4472 (Plan 33-07's three-tier collected total) = 120.
#
# It closes exactly against the modules, which is the check that no test was quietly
# deleted to make a number look better. All SEVEN are new; no existing module was
# edited by this plan, so there is no zero-node edit to reconcile:
#     38  tests/unit/test_deploy_gate_secondaries_live.py    (new, Task 2)
#      9  tests/unit/test_phase33_gate_toml_untouched.py     (new, Task 2)
#      8  tests/unit/test_deploy_gate_alpha_boundary.py      (new, Task 3)
#     18  tests/unit/test_deploy_gate_empty_pairs.py         (new, Task 3)
#      4  tests/unit/test_deploy_gate_order_invariance.py    (new, Task 3)
#     11  tests/unit/test_phase33_preregistration.py         (new, Task 3)
#     32  tests/unit/test_phase33_gate_runner.py             (new, Task 3(d))
#     --
#    120
#
# tests/phase33_gate_fixtures.py was also added and contributes ZERO nodes: it carries
# no `test_` prefix, so pytest never collects it. That is deliberate -- it is the shared
# FAIL-CLOSED sandbox builder three of the modules above depend on.
#
# The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS, unchanged. BOTH
# gate-baseline tripwires are still RED, which is the whole point of D33-11's refusal to
# re-freeze. The SKIPPED count did not move (11 -> 11) and neither did XFAILED
# (14 -> 14). No tier reported a production-store boundary crossing
# (STAT_SIGNATURE_OBSERVATIONS 3 + 5 + 0, WAL_SIBLING_OBSERVATIONS 0 in every tier) --
# identical to Plan 33-07's observation, so this plan added no new crossing.
# ---------------------------------------------------------------------------

TESTS_ADDED_33_08: int = 120


# ---------------------------------------------------------------------------
# THE DECLARED FORECAST HORIZON, AND THE SILVER WEATHER WIDTH IT ARRIVES AT.
#
# APPENDED by Plan 33-09 Task 1 on 2026-09-12. Nothing above this line was edited.
#
# THE HORIZON IS OURS, NOT THE PROVIDER'S (D33-26). Open-Meteo's forecast endpoint
# was PROBED on 2026-09-12 from this checkout and reported its own window in an
# error body rather than in prose:
#
#     GET https://api.open-meteo.com/v1/forecast?start_date=2026-09-28...
#     400 {"error":true,"reason":"Parameter 'start_date' is out of allowed range
#          from 2026-06-11 to 2026-09-27"}
#
# Measured on 2026-09-12 that is today+15, and +14 returned 24 non-null hours. The
# documented parameter is `forecast_days` (0-16, default 7), which counts today as
# day one and therefore reaches today+15 -- the two agree.
#
# So 14 sits INSIDE the provider's window by a day, deliberately. A horizon wider
# than the provider's is a promise we cannot keep; a horizon equal to it turns any
# provider narrowing into a silent empty response instead of a named refusal. And
# because the constant is OURS, the refusal is testable OFFLINE: no test in this
# plan needs the network to prove that a kickoff beyond the horizon is refused by
# name.
#
# BOUNDARY CONVENTION, stated once and asserted: a kickoff EXACTLY
# FORECAST_HORIZON_DAYS after `as_of_utc` is INSIDE. The comparison is a strict
# `>` between two timezone-aware INSTANTS, never between two calendar dates.
FORECAST_HORIZON_DAYS: int = 14

# The provider's own advertised window, recorded BESIDE ours rather than instead of
# it, so a future narrowing surfaces as a disagreement between two recorded numbers.
FORECAST_PROVIDER_HORIZON_DAYS: int = 15

# data/silver/weather.parquet as this plan found it, MEASURED 2026-09-12:
# 14 rows x 23 columns, all of them 2024 Week 6. Those 14 rows are the whole of
# the weather store -- the evidence in the plan's own objective that the ingest
# path has never run forward.
WEATHER_COLUMNS_BEFORE: int = 23
WEATHER_ROWS_BEFORE: int = 14


# ---------------------------------------------------------------------------
# `weather_source` PROVENANCE, AND THE 14-ROW PRODUCTION BACKFILL THAT ADDS IT.
#
# APPENDED by Plan 33-09 Task 2 on 2026-09-12, and the CHANGED-FILE DECLARATION
# below was appended BEFORE the backfill ran. Nothing above this line was edited.
#
# THE ORDER MATTERS AND IS THE POINT. A blast radius declared after the fact is
# not a declaration, it is a transcription of whatever happened. This slot was
# committed first, then `python -m tests.data_boundary snapshot data ...` was
# taken, then the backfill ran, then `verify` was run against the declaration. A
# file outside the set is a FINDING to report, never a reason to widen the set.
# ---------------------------------------------------------------------------

# 23 -> 24. The new column is `weather_source`.
WEATHER_COLUMNS_AFTER: int = 24

# The CLOSED vocabulary. Three members, and each one is a different provenance
# rather than a different degree of confidence:
#
#   archive             the Open-Meteo ARCHIVE (ERA5 reanalysis) endpoint --
#                       measured after the fact, the only source for 2002-2020.
#   forecast            the Open-Meteo FORECAST endpoint -- predicted before
#                       kickoff, which is the only thing that can answer for an
#                       unplayed game. Every live 2026 row is this.
#   historical_forecast the Historical Forecast API -- what the forecast SAID at
#                       the time, for a game that has since been played. Reaches
#                       back only to about 2021, which is why the archive endpoint
#                       cannot simply be replaced by it.
#
# Mirrors api/cache.BET_LIST_COLUMNS's single-source discipline: the tuple lives
# in scripts/ingest_weather.py and this slot pins it, so a silent edit to either
# is a test failure rather than a divergence nobody notices.
WEATHER_SOURCE_VOCABULARY: tuple[str, ...] = (
    "archive",
    "forecast",
    "historical_forecast",
)

# THE DECLARED BLAST RADIUS OF THE 14-ROW BACKFILL, appended BEFORE the run.
#
# ONE FILE, and the DuckDB half is deliberately NOT in it. The backfill writes
# through `data.storage.upsert_silver`, which ends at `_atomic_write_parquet` and
# never opens the database -- it is `save_dataframe` that defaults
# `save_to_db=True` and writes both halves, and this path does not call it. So
# `data/nfl_predictions.duckdb` is EXCLUDED as a positive statement about the
# write path, not as an oversight: if it moves, the write did not go where this
# declaration says it went, and that is a finding.
WEATHER_BACKFILL_EXPECTED_CHANGED_FILES: tuple[str, ...] = ("silver/weather.parquet",)


# ---------------------------------------------------------------------------
# WHAT THE 14-ROW BACKFILL ACTUALLY MOVED.
#
# APPENDED by Plan 33-09 Task 2 on 2026-09-12, AFTER the run. Nothing above this
# line was edited -- the declaration slot above stays exactly as it was written
# before the run, which is the only thing that makes it a declaration.
#
# THE BRACKET, in the order it was executed:
#
#   1. WEATHER_BACKFILL_EXPECTED_CHANGED_FILES appended and committed.
#   2. python -m tests.data_boundary snapshot data \
#          outputs/phase33_weather_backfill_before.json      (424 files digested)
#   3. python -m scripts.backfill_historical_weather --stamp-weather-source
#   4. python -m tests.data_boundary verify data \
#          outputs/phase33_weather_backfill_before.json
#
# THE VERIFY REPORTED EXACTLY ONE CHANGED FILE AND IT IS THE DECLARED ONE.
# Nothing was added and nothing was removed; the single rewritten path was
# silver/weather.parquet; and no comparison was left undecided (no MIXED key).
#
# `data/nfl_predictions.duckdb` did NOT move, which is the positive confirmation
# the declaration was making: the stamp writes through `upsert_silver` (parquet
# only) and never through `save_dataframe`, whose `save_to_db=True` default would
# have moved the database half as well.
#
# Fourteen rows in, fourteen rows out, 23 -> 24 columns, and all fourteen read
# `archive`. No weather VALUE changed -- only the column that says where each one
# came from.
WEATHER_BACKFILL_DIGEST_BEFORE: str = (
    "f592b7409cab8cfc0f98bf3d0c9a71b83efb3dcf3621238a504ecbb2422e1f7e"
)
WEATHER_BACKFILL_DIGEST_AFTER: str = (
    "3034b00b86ab95332cf1d385b736379b1b6796be2fa812acf3a4d1dcb9114d4d"
)


# ---------------------------------------------------------------------------
# THE GOLD WEATHER CONSTANCY, RE-DERIVED. MEASUREMENT ONLY.
#
# APPENDED by Plan 33-09 Task 3 on 2026-09-12. Nothing above this line was edited.
#
# WHY IT WAS RE-DERIVED RATHER THAN INHERITED. The claims "99.78% imputed" and
# "32 of 33 columns exactly constant, only venue_cold_climate varying" are the
# ENTIRE rationale for holding the gold weather family at its historical default
# for 2026 (D33-25). They were carried forward from an earlier phase without
# reproduction. A decision resting on an unreproduced number is a decision resting
# on the last thing somebody wrote down.
#
# READ-ONLY, AND PROVEN SO. The re-derivation was bracketed with
# `tests.data_boundary.digest_tree` on BOTH production roots. 424 files under
# `data/` and 159 under `artifacts/` were digested before and after; both
# comparisons came back clean. "It only reads" is a property here, not an
# intention.
#
# WHAT REPRODUCED, EXACTLY
# ------------------------
# The IMPUTED SHARE reproduces to the digit: 6,485 of 6,499 gold rows carry
# `raw_temp_f == 65.0`, the imputed default. That is 99.7846%, and the recorded
# claim was 99.78%.
#
# The SHAPE of the constancy claim reproduces exactly: in every one of the three
# populations, in all three gold matrices, there is exactly ONE varying weather
# column and it is `venue_cold_climate` -- precisely the column the recorded claim
# named.
#
# WHAT DIVERGED, RECORDED AS BOTH NUMBERS RATHER THAN REPLACED
# --------------------------------------------------------------
# The COUNT is 45 of 46, not 32 of 33. The re-derivation counts the whole weather
# family as it reaches gold -- all 45 columns `data/silver/weather_features.parquet`
# contributes, plus `venue_cold_climate` -- where the earlier claim counted a
# narrower set nobody wrote down. The two are the same finding at two widths, and
# the divergence is in the DENOMINATOR, not in the verdict. Both are recorded; the
# earlier figure is not overwritten.
#
# WHAT THE RE-DERIVATION FOUND THAT NOBODY HAD RECORDED
# ------------------------------------------------------
# TWO facts, and the second changes the size of the decision:
#
# 1. Inside all three populations the imputation is TOTAL, not merely dominant:
#    1335/1335, 534/534 and 1139/1139 rows sit at the 65.0 default. There is no
#    real weather anywhere in the ATS train window, the WP/OU train window, or the
#    2021-2024 gate holdout -- not 99.78% of it, all of it. The fourteen real rows
#    are 2024 Week 6, which falls in none of the three.
#
# 2. THE DEPLOYED WP AND ATS ARTIFACTS CONSUME NO WEATHER FEATURE AT ALL. Read from
#    each artifact's own `feature_list.json`: wp_20260824_113325 has 20 features and
#    0 are weather; ats_20260605_220128 has 25 and 0 are weather; only
#    ou_20260326_163930 has any, 17 of its 25. So the exposure this switch protects
#    against is SEVENTEEN columns entering ONE model, not thirty-three entering
#    three. All 17 are exactly constant in O/U's own 2018-2019 train window AND in
#    the 2021-2024 holdout, so the protection is real -- but it is narrower than the
#    decision was framed, and the owner was told so before ruling.
# ---------------------------------------------------------------------------

GOLD_WEATHER_CONSTANCY_MEASUREMENT: dict[str, object] = {
    "measured_on": "2026-09-12",
    "measured_by": "Plan 33-09 Task 3",
    "read_only_bracket": {
        "data_files_digested": 424,
        "artifacts_files_digested": 159,
        "data_unchanged": True,
        "artifacts_unchanged": True,
    },
    # 45 weather-family columns reach gold (all of weather_features.parquet except
    # `weather_condition`, which is dropped at build time), plus `venue_cold_climate`.
    "weather_columns_counted": 46,
    "populations": {
        # (rows, constant_columns, varying_columns) -- identical in all three gold
        # matrices, so one entry per population rather than three that agree.
        "ats_train_2015_2019": {
            "rows": 1335,
            "constant": 45,
            "varying": 1,
            "varying_columns": ("venue_cold_climate",),
        },
        "wp_ou_train_2018_2019": {
            "rows": 534,
            "constant": 45,
            "varying": 1,
            "varying_columns": ("venue_cold_climate",),
        },
        "gate_holdout_2021_2024": {
            "rows": 1139,
            "constant": 45,
            "varying": 1,
            "varying_columns": ("venue_cold_climate",),
        },
    },
    "raw_temp_f_imputed": {
        "default_value": 65.0,
        "rows_total": 6499,
        "rows_at_default": 6485,
        "share": 0.997846,
        # Within each population the imputation is TOTAL.
        "per_population_at_default": {
            "ats_train_2015_2019": (1335, 1335),
            "wp_ou_train_2018_2019": (534, 534),
            "gate_holdout_2021_2024": (1139, 1139),
        },
    },
    # What each DEPLOYED artifact actually consumes, from its own feature_list.json.
    "deployed_weather_feature_exposure": {
        "wp_20260824_113325": {"total_features": 20, "weather_features": 0},
        "ats_20260605_220128": {"total_features": 25, "weather_features": 0},
        "ou_20260326_163930": {"total_features": 25, "weather_features": 17},
    },
    "ou_weather_features_constant": {
        "ou_train_2018_2019": (17, 17),
        "gate_holdout_2021_2024": (17, 17),
        # Across the full 2002-2025 span 11 of the 17 DO vary, because the fourteen
        # real 2024 Week 6 rows and the 2025 partial rows are in that span. That is
        # the contrast that makes the windows' constancy a fact about the windows
        # rather than about the columns.
        "all_2002_2025": (6, 17),
    },
    "divergence_from_recorded_claims": {
        "imputed_share": "REPRODUCED. Recorded 99.78%; re-derived 99.7846% "
        "(6,485 of 6,499).",
        "constant_columns": "SHAPE REPRODUCED, COUNT DIVERGED. Recorded 32 of 33 "
        "with only venue_cold_climate varying; re-derived 45 of 46 with only "
        "venue_cold_climate varying. Same verdict, wider denominator. Both figures "
        "are recorded and the earlier one is not overwritten.",
        "blast_radius": "NOT PREVIOUSLY RECORDED, and it narrows the decision. The "
        "premise said 33 out-of-distribution columns would enter THREE deployed "
        "models. Measured: WP and ATS consume ZERO weather features, so it is 17 "
        "columns entering ONE model (the v1.0 pre-Elo O/U artifact).",
    },
}


# ---------------------------------------------------------------------------
# THE TWENTY-TWO HISTORICAL VENUES, AS THE OWNER RATIFIED THEM.
#
# APPENDED by Plan 33.1-01 Task 3 on 2026-09-12. Nothing above this line was edited.
#
# THE OWNER RULED ON 2026-09-12 at the Task-2 blocking checkpoint, selecting
# "approved with NYC00 -> (40.8122, -74.0769)":
#
#     Twenty-one rows ratified exactly as tabled. NYC00 Giants Stadium takes the
#     en.wikipedia article's own GeoHack coordinate instead of the Wikidata P625
#     value, because the Wikidata value sits 0.02 km -- twenty metres -- from the
#     already-ratified NYC01 MetLife record and is therefore the T-33.1-02
#     successor-contamination failure mode itself. The article coordinate is
#     0.29 km from MetLife and matches the demolished stadium's footprint.
#
# WHY THESE VALUES ARE A CONSTANT AND NOT AN EDIT TO data/venues.json.
# The governing rule is Plan 33-06's (33-06-SUMMARY.md:40): a reference value
# nothing in the repository can settle is RATIFIED by the owner, committed ONCE as
# a constant, and the data file is GENERATED from that constant -- so the file
# cannot drift from the record that authorised it. The values here were RESEARCHED
# and TABLED before any write, and ratified at a blocking checkpoint.
#
# PROVENANCE IS PER FIELD, NOT PER VENUE (Ruling C2). HISTORICAL_VENUE_FIELD_SOURCES
# below carries one row per (stadium_id, field) pair across all EIGHT non-identity
# fields -- 176 rows. A Wikidata P625 coordinate statement substantiates a latitude
# and a longitude and says NOTHING about what the playing surface was in 2003 or how
# many seats the building held, and BOTH of those fields feed live contextual
# features (features/contextual.py:345-356 and :653). The 88 `external` cells carry
# the revision-pinned strings the owner ratified; the 88 `derived` cells carry the
# citation the deriving function emitted.
#
# DUAL-SOURCE CORROBORATION, stronger than the plan required. Every Wikidata P625
# was cross-checked against its en.wikipedia article's own GeoHack coordinate: 21 of
# the 22 agree to within 0.27 km, most to 0.000 km, against an ERA5 grid cell of
# roughly 9-11 km. GER00 carries no GeoHack in its infobox and instead matches the
# already-ratified MUN01 record to all four decimals, which is correct -- they are
# the same building. NYC00 was the ONE material disagreement and is the ruling above.
#
# `home_teams` IS DELIBERATELY EMPTY on all 22. scripts/ingest_weather._get_venue_record
# (:621) returns the FIRST record whose `home_teams` contains the team, so a historical
# record claiming `LV` would shadow VEG00 for the live 2026 season. These records are
# reachable by `stadium_id` ONLY.
#
# RULING A, RECORDED HONESTLY AND WITH ITS OWN PREMISE CORRECTED. `elevation_ft` for
# these 22 is DERIVED from Open-Meteo's /v1/elevation -- a ~90 m DEM terrain value at
# the nearest grid cell -- so the 22 differ IN KIND from the 38 hand-sourced records.
# Ruling A justified that by asserting "no venue among the 22 sits within 1,500 ft of
# [the 3,000 ft venue_high_altitude] boundary -- the highest is PHO99 Sun Devil Stadium
# at roughly 1,150 ft". THAT CLAIM IS FALSE AS WRITTEN and is not papered over here:
# SAO00 Arena Corinthians is 2,562 ft (Sao Paulo sits at ~780 m), leaving 438 ft of
# margin, and GER00 is 1,611 ft. The VERDICT is unchanged -- all 22 give
# venue_high_altitude = 0.0 -- but the margin is 3.4x smaller than the ruling asserted.
# Corroboration that the DEM is sound anyway: the Sun Devil Stadium article states
# 1,160 ft AMSL against the derived 1,178 ft.
#
# `venue_name` IS THE MOST RECENT FEED `stadium` STRING, AND THE NAME PATH STAYS LOSSY
# (Ruling C). Twenty of these ids carry two to five feed names each -- OAK00 has five --
# so no single `venue_name` can satisfy features/contextual._get_venue_id_by_name for all
# of a venue's games. That is why routing moves to `stadium_id` in Plan 33.1-03. The
# constraint tests/phase33_state.py records for the international eight ("`venue_name`
# MUST equal the feed's `stadium` string EXACTLY") is simply unsatisfiable for OAK00.
#
# Field order, stated once and asserted by tests/unit/test_venues_json_historical.py:
#     (stadium_id, venue_id, venue_name, city, state, country, latitude, longitude,
#      elevation_ft, roof_type, surface, capacity, climate_zone, timezone)
#
# FOURTEEN fields, one more than INTERNATIONAL_VENUE_FACTS's thirteen, because five of
# the 22 are non-US and `state` must be recordable as the EMPTY STRING rather than
# guessed -- the same boundary 33-06-SUMMARY.md:70 records for the international eight.
# ---------------------------------------------------------------------------

HISTORICAL_STADIUM_IDS: tuple[str, ...] = (
    "OAK00",
    "NYC00",
    "SDG00",
    "ATL00",
    "STL00",
    "SFO00",
    "MIN00",
    "DAL99",
    "IND99",
    "PHO99",
    "LAX99",
    "LAX97",
    "MIN98",
    "PHI99",
    "CHI99",
    "BUF01",
    "BRG00",
    "SAN00",
    "LON01",
    "GER00",
    "FRA00",
    "SAO00",
)


HISTORICAL_VENUE_FACTS: tuple[tuple[object, ...], ...] = (
    (
        "OAK00",
        "ring_central_coliseum",
        "Ring Central Coliseum",
        "Oakland",
        "CA",
        "USA",
        37.7517,
        -122.2006,
        23,
        "outdoor",
        "Bermuda Grass",
        63122,
        "mediterranean",
        "America/Los_Angeles",
    ),
    (
        "NYC00",
        "giants_stadium",
        "Giants Stadium",
        "East Rutherford",
        "NJ",
        "USA",
        40.8122,
        -74.0769,
        7,
        "outdoor",
        "FieldTurf",
        80242,
        "humid_continental",
        "America/New_York",
    ),
    (
        "SDG00",
        "qualcomm_stadium",
        "Qualcomm Stadium",
        "San Diego",
        "CA",
        "USA",
        32.7831,
        -117.1194,
        59,
        "outdoor",
        "Bermuda Grass",
        70561,
        "mediterranean",
        "America/Los_Angeles",
    ),
    (
        "ATL00",
        "georgia_dome",
        "Georgia Dome",
        "Atlanta",
        "GA",
        "USA",
        33.7575,
        -84.4008,
        988,
        "indoor",
        "FieldTurf",
        71228,
        "humid_subtropical",
        "America/New_York",
    ),
    (
        "STL00",
        "edward_jones_dome",
        "Edward Jones Dome",
        "St. Louis",
        "MO",
        "USA",
        38.6328,
        -90.1886,
        492,
        "indoor",
        "AstroTurf",
        66000,
        "humid_continental",
        "America/Chicago",
    ),
    (
        "SFO00",
        "candlestick_park",
        "Candlestick Park",
        "San Francisco",
        "CA",
        "USA",
        37.7136,
        -122.3861,
        52,
        "outdoor",
        "Kentucky Bluegrass",
        69732,
        "mediterranean",
        "America/Los_Angeles",
    ),
    (
        "MIN00",
        "mall_of_america_field",
        "Mall of America Field",
        "Minneapolis",
        "MN",
        "USA",
        44.9739,
        -93.2581,
        830,
        "indoor",
        "FieldTurf",
        64121,
        "humid_continental",
        "America/Chicago",
    ),
    (
        "DAL99",
        "texas_stadium",
        "Texas Stadium",
        "Irving",
        "TX",
        "USA",
        32.8398,
        -96.9109,
        449,
        "outdoor",
        "RealGrass",
        65675,
        "humid_subtropical",
        "America/Chicago",
    ),
    (
        "IND99",
        "rca_dome",
        "RCA Dome",
        "Indianapolis",
        "IN",
        "USA",
        39.7637,
        -86.1633,
        761,
        "indoor",
        "FieldTurf",
        55506,
        "humid_continental",
        "America/Indiana/Indianapolis",
    ),
    (
        "PHO99",
        "sun_devil_stadium",
        "Sun Devil Stadium",
        "Tempe",
        "AZ",
        "USA",
        33.4264,
        -111.9325,
        1178,
        "outdoor",
        "Bermuda Grass",
        73379,
        "desert",
        "America/Phoenix",
    ),
    (
        "LAX99",
        "los_angeles_memorial_coliseum",
        "Los Angeles Memorial Coliseum",
        "Los Angeles",
        "CA",
        "USA",
        34.0142,
        -118.2878,
        194,
        "outdoor",
        "Bermuda Grass",
        77500,
        "mediterranean",
        "America/Los_Angeles",
    ),
    (
        "LAX97",
        "stubhub_center",
        "StubHub Center",
        "Carson",
        "CA",
        "USA",
        33.8644,
        -118.2611,
        75,
        "outdoor",
        "Bermuda Grass",
        27000,
        "mediterranean",
        "America/Los_Angeles",
    ),
    (
        "MIN98",
        "tcf_bank_stadium",
        "TCF Bank Stadium",
        "Minneapolis",
        "MN",
        "USA",
        44.9764,
        -93.2244,
        830,
        "outdoor",
        "FieldTurf",
        50805,
        "humid_continental",
        "America/Chicago",
    ),
    (
        "PHI99",
        "veterans_stadium",
        "Veterans Stadium",
        "Philadelphia",
        "PA",
        "USA",
        39.9067,
        -75.1711,
        20,
        "outdoor",
        "NexTurf",
        65352,
        "humid_continental",
        "America/New_York",
    ),
    (
        "CHI99",
        "memorial_stadium_champaign",
        "Memorial Stadium (Champaign)",
        "Champaign",
        "IL",
        "USA",
        40.0992,
        -88.2358,
        748,
        "outdoor",
        "AstroPlay",
        69249,
        "humid_continental",
        "America/Chicago",
    ),
    (
        "BUF01",
        "rogers_centre",
        "Rogers Centre",
        "Toronto",
        "",
        "Canada",
        43.6414,
        -79.3892,
        276,
        "indoor",
        "FieldTurf",
        53506,
        "humid_continental",
        "America/Toronto",
    ),
    (
        "BRG00",
        "tiger_stadium_lsu",
        "Tiger Stadium (LSU)",
        "Baton Rouge",
        "LA",
        "USA",
        30.4119,
        -91.1856,
        26,
        "outdoor",
        "Bermuda Grass",
        92400,
        "humid_subtropical",
        "America/Chicago",
    ),
    (
        "SAN00",
        "alamo_dome",
        "Alamo Dome",
        "San Antonio",
        "TX",
        "USA",
        29.4169,
        -98.4789,
        640,
        "indoor",
        "AstroTurf",
        65000,
        "humid_subtropical",
        "America/Chicago",
    ),
    (
        "LON01",
        "twickenham_stadium",
        "Twickenham Stadium",
        "London",
        "",
        "United Kingdom",
        51.4561,
        -0.3417,
        23,
        "outdoor",
        "Desso GrassMaster",
        75000,
        "oceanic",
        "Europe/London",
    ),
    (
        "GER00",
        "allianz_arena",
        "Allianz Arena",
        "Munich",
        "",
        "Germany",
        48.2188,
        11.6248,
        1611,
        "outdoor",
        "Hybrid Grass",
        75024,
        "oceanic",
        "Europe/Berlin",
    ),
    (
        "FRA00",
        "deutsche_bank_park",
        "Deutsche Bank Park",
        "Frankfurt",
        "",
        "Germany",
        50.0686,
        8.6453,
        407,
        "outdoor",
        "Grass",
        48000,
        "oceanic",
        "Europe/Berlin",
    ),
    (
        "SAO00",
        "arena_corinthians",
        "Arena Corinthians",
        "Sao Paulo",
        "",
        "Brazil",
        -23.5456,
        -46.474,
        2562,
        "outdoor",
        "Desso GrassMaster",
        47252,
        "tropical",
        "America/Sao_Paulo",
    ),
)


# The four fields that are genuinely EXTERNAL -- nothing in this repository can settle
# them, so each carries its own revision-pinned citation and each was ratified by the
# owner. Pinned test-side as well as production-side so the tuple in
# scripts/derive_historical_venue_facts.py and the coverage assertion in
# tests/unit/test_venues_json_historical.py cannot drift apart.
EXTERNALLY_SOURCED_VENUE_FIELDS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "surface",
    "capacity",
)

# (stadium_id, field, source) for all 22 records across all EIGHT non-identity fields --
# 176 rows. The 88 rows whose field is in EXTERNALLY_SOURCED_VENUE_FIELDS carry a
# revision-pinned external citation; the other 88 carry the derivation citation the
# deriving function emitted, which is reproducible rather than ratified.
#
# SIX VENUES CHANGED SURFACE OR CAPACITY DURING THEIR NFL LIFE. Where that happened the
# ERA PLURALITY was taken and the citation says so with the game counts, rather than a
# current value being written as though it had always been true.
HISTORICAL_VENUE_FIELD_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "OAK00",
        "latitude",
        "wikidata Q1147732 P625 coordinate location = (37.751666666667, -122.20055555556), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1147732&oldid=2541890918",
    ),
    (
        "OAK00",
        "longitude",
        "wikidata Q1147732 P625 coordinate location = (37.751666666667, -122.20055555556), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1147732&oldid=2541890918",
    ),
    (
        "OAK00",
        "surface",
        'en.wikipedia "Oakland Coliseum" infobox surface = "Tifway II Bermuda Grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=Oakland_Coliseum&oldid=1374295982',
    ),
    (
        "OAK00",
        "capacity",
        "wikidata Q1147732 P1083 seating capacity = 63122, qualified P641 sport = Q41323 American football; revision-pinned https://www.wikidata.org/w/index.php?title=Q1147732&oldid=2541890918",
    ),
    (
        "OAK00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=OAK00, roof=outdoors (single-valued, n=141) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "OAK00",
        "timezone",
        'open-meteo timezone=auto @ (37.7517, -122.2006) -> "America/Los_Angeles"',
    ),
    (
        "OAK00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (37.7517, -122.2006) -> 7.0 m -> 23 ft",
    ),
    (
        "OAK00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=SFO01 (Levi's Stadium) climate_zone=mediterranean",
    ),
    (
        "NYC00",
        "latitude",
        'en.wikipedia "Giants Stadium" infobox coordinates = {{coord|40|48|44|N|74|4|37|W}} = (40.81222, -74.07694); revision-pinned https://en.wikipedia.org/w/index.php?title=Giants_Stadium&oldid=1372182586 -- OWNER RULING 2026-09-12: the Wikidata Q375365 P625 value (40.813726, -74.074433) was REJECTED because it sits 0.02 km from the successor record, which is the T-33.1-02 successor-contamination failure mode; this coordinate is 0.29 km from the successor and matches the demolished footprint',
    ),
    (
        "NYC00",
        "longitude",
        'en.wikipedia "Giants Stadium" infobox coordinates = {{coord|40|48|44|N|74|4|37|W}} = (40.81222, -74.07694); revision-pinned https://en.wikipedia.org/w/index.php?title=Giants_Stadium&oldid=1372182586 -- OWNER RULING 2026-09-12: the Wikidata Q375365 P625 value (40.813726, -74.074433) was REJECTED because it sits 0.02 km from the successor record, which is the T-33.1-02 successor-contamination failure mode; this coordinate is 0.29 km from the successor and matches the demolished footprint',
    ),
    (
        "NYC00",
        "surface",
        'en.wikipedia "Giants Stadium" infobox surface = "AstroTurf (1976-1999) / Grass (2000-2002) / FieldTurf (2003-2009)"; FieldTurf covers 115 of the 132 games here (2003-2009), natural grass the other 17 (2002); revision-pinned https://en.wikipedia.org/w/index.php?title=Giants_Stadium&oldid=1372182586',
    ),
    (
        "NYC00",
        "capacity",
        'en.wikipedia "Giants Stadium" infobox capacity = "80,242"; revision-pinned https://en.wikipedia.org/w/index.php?title=Giants_Stadium&oldid=1372182586',
    ),
    (
        "NYC00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=NYC00, roof=outdoors (single-valued, n=132) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "NYC00",
        "timezone",
        'open-meteo timezone=auto @ (40.8122, -74.0769) -> "America/New_York"',
    ),
    (
        "NYC00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (40.8122, -74.0769) -> 2.0 m -> 7 ft",
    ),
    (
        "NYC00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=NYC01 (MetLife Stadium) climate_zone=humid_continental",
    ),
    (
        "SDG00",
        "latitude",
        "wikidata Q956072 P625 coordinate location = (32.783055555556, -117.11944444444), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q956072&oldid=2475782501",
    ),
    (
        "SDG00",
        "longitude",
        "wikidata Q956072 P625 coordinate location = (32.783055555556, -117.11944444444), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q956072&oldid=2475782501",
    ),
    (
        "SDG00",
        "surface",
        'en.wikipedia "San Diego Stadium" infobox surface = "Bandera Bermuda Grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=San_Diego_Stadium&oldid=1372228129',
    ),
    (
        "SDG00",
        "capacity",
        'en.wikipedia "San Diego Stadium" infobox capacity = "70,561 (Football, Chargers)"; revision-pinned https://en.wikipedia.org/w/index.php?title=San_Diego_Stadium&oldid=1372228129',
    ),
    (
        "SDG00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=SDG00, roof=outdoors (single-valued, n=125) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "SDG00",
        "timezone",
        'open-meteo timezone=auto @ (32.7831, -117.1194) -> "America/Los_Angeles"',
    ),
    (
        "SDG00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (32.7831, -117.1194) -> 18.0 m -> 59 ft",
    ),
    (
        "SDG00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=LAX01 (SoFi Stadium) climate_zone=mediterranean",
    ),
    (
        "ATL00",
        "latitude",
        "wikidata Q1058931 P625 coordinate location = (33.7575, -84.400833333333), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1058931&oldid=2501589476",
    ),
    (
        "ATL00",
        "longitude",
        "wikidata Q1058931 P625 coordinate location = (33.7575, -84.400833333333), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1058931&oldid=2501589476",
    ),
    (
        "ATL00",
        "surface",
        'en.wikipedia "Georgia Dome" infobox surface = "FieldTurf (2003-2017) / AstroTurf (1992-2002)"; FieldTurf covers 117 of the 125 games here, AstroTurf the other 8 (2002); revision-pinned https://en.wikipedia.org/w/index.php?title=Georgia_Dome&oldid=1372283248',
    ),
    (
        "ATL00",
        "capacity",
        'en.wikipedia "Georgia Dome" infobox capacity = "Football: 71,228"; revision-pinned https://en.wikipedia.org/w/index.php?title=Georgia_Dome&oldid=1372283248',
    ),
    (
        "ATL00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=ATL00, roof=dome (single-valued, n=125) -> NFLVERSE_ROOF_MAP -> indoor",
    ),
    (
        "ATL00",
        "timezone",
        'open-meteo timezone=auto @ (33.7575, -84.4008) -> "America/New_York"',
    ),
    (
        "ATL00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (33.7575, -84.4008) -> 301.0 m -> 988 ft",
    ),
    (
        "ATL00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=ATL97 (Mercedes-Benz Stadium) climate_zone=humid_subtropical",
    ),
    (
        "STL00",
        "latitude",
        "wikidata Q1292739 P625 coordinate location = (38.632777777778, -90.188611111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1292739&oldid=2540146594",
    ),
    (
        "STL00",
        "longitude",
        "wikidata Q1292739 P625 coordinate location = (38.632777777778, -90.188611111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1292739&oldid=2540146594",
    ),
    (
        "STL00",
        "surface",
        'en.wikipedia "The Dome at America\'s Center" infobox surface = "AstroTurf RootZone 3D3 (2025-) / AstroTurf GameDay Grass 3D (2010-2024) / FieldTurf (2005-2010) / AstroTurf (1995-2004)"; the AstroTurf family covers 72 of the 112 games here (2002-2004 and 2010-2015), FieldTurf the other 40; revision-pinned https://en.wikipedia.org/w/index.php?title=The_Dome_at_America%27s_Center&oldid=1372241665',
    ),
    (
        "STL00",
        "capacity",
        "wikidata Q1292739 P1083 seating capacity = 66000, the Rams-era football configuration (the article infobox states 67,277 for the present full-stadium configuration); revision-pinned https://www.wikidata.org/w/index.php?title=Q1292739&oldid=2540146594",
    ),
    (
        "STL00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=STL00, roof=dome (single-valued, n=112) -> NFLVERSE_ROOF_MAP -> indoor",
    ),
    (
        "STL00",
        "timezone",
        'open-meteo timezone=auto @ (38.6328, -90.1886) -> "America/Chicago"',
    ),
    (
        "STL00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (38.6328, -90.1886) -> 150.0 m -> 492 ft",
    ),
    (
        "STL00",
        "climate_zone",
        "operator entry, no same-metro sibling; consistent with the repository's other Midwest records KAN00, IND00 and CIN00, all humid_continental",
    ),
    (
        "SFO00",
        "latitude",
        "wikidata Q1033076 P625 coordinate location = (37.713611111111, -122.38611111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1033076&oldid=2540053846",
    ),
    (
        "SFO00",
        "longitude",
        "wikidata Q1033076 P625 coordinate location = (37.713611111111, -122.38611111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1033076&oldid=2540053846",
    ),
    (
        "SFO00",
        "surface",
        'en.wikipedia "Candlestick Park" infobox surface = "Bluegrass (1960-1969, 1979-2013) / AstroTurf (1970-1978)" -- natural bluegrass for every one of the 99 games here, written with the repository existing Kentucky Bluegrass token so it classifies as natural grass; revision-pinned https://en.wikipedia.org/w/index.php?title=Candlestick_Park&oldid=1373026906',
    ),
    (
        "SFO00",
        "capacity",
        'en.wikipedia "Candlestick Park" infobox capacity = "69,732", the 49ers media-guide football figure cited in the infobox; revision-pinned https://en.wikipedia.org/w/index.php?title=Candlestick_Park&oldid=1373026906',
    ),
    (
        "SFO00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=SFO00, roof=outdoors (single-valued, n=99) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "SFO00",
        "timezone",
        'open-meteo timezone=auto @ (37.7136, -122.3861) -> "America/Los_Angeles"',
    ),
    (
        "SFO00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (37.7136, -122.3861) -> 16.0 m -> 52 ft",
    ),
    (
        "SFO00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=SFO01 (Levi's Stadium) climate_zone=mediterranean",
    ),
    (
        "MIN00",
        "latitude",
        "wikidata Q1072186 P625 coordinate location = (44.973888888889, -93.258055555556), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1072186&oldid=2241607175",
    ),
    (
        "MIN00",
        "longitude",
        "wikidata Q1072186 P625 coordinate location = (44.973888888889, -93.258055555556), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1072186&oldid=2241607175",
    ),
    (
        "MIN00",
        "surface",
        'en.wikipedia "Hubert H. Humphrey Metrodome" infobox surface = "SuperTurf (1982-1986) / AstroTurf (1987-2003) / FieldTurf (2004-2010) / Sportexe Momentum Turf (2010) / UBU-Intensity Series-S5-M (2011-2013)"; FieldTurf covers 54 of the 95 games here; revision-pinned https://en.wikipedia.org/w/index.php?title=Hubert_H._Humphrey_Metrodome&oldid=1372680769',
    ),
    (
        "MIN00",
        "capacity",
        'en.wikipedia "Hubert H. Humphrey Metrodome" infobox capacity = "American football: 64,121"; revision-pinned https://en.wikipedia.org/w/index.php?title=Hubert_H._Humphrey_Metrodome&oldid=1372680769',
    ),
    (
        "MIN00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=MIN00, roof=dome (single-valued, n=95) -> NFLVERSE_ROOF_MAP -> indoor",
    ),
    (
        "MIN00",
        "timezone",
        'open-meteo timezone=auto @ (44.9739, -93.2581) -> "America/Chicago"',
    ),
    (
        "MIN00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (44.9739, -93.2581) -> 253.0 m -> 830 ft",
    ),
    (
        "MIN00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=MIN01 (U.S. Bank Stadium) climate_zone=humid_continental",
    ),
    (
        "DAL99",
        "latitude",
        "wikidata Q601596 P625 coordinate location = (32.839769444444, -96.910911111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q601596&oldid=2542836478",
    ),
    (
        "DAL99",
        "longitude",
        "wikidata Q601596 P625 coordinate location = (32.839769444444, -96.910911111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q601596&oldid=2542836478",
    ),
    (
        "DAL99",
        "surface",
        'en.wikipedia "Texas Stadium" infobox surface = "Artificial turf: Texas Turf (1971-1995) / AstroTurf (1996-2002) / RealGrass (2002-2008)"; RealGrass is listed UNDER Artificial turf and covers 2002-2008, all 57 games here; revision-pinned https://en.wikipedia.org/w/index.php?title=Texas_Stadium&oldid=1372241047',
    ),
    (
        "DAL99",
        "capacity",
        'en.wikipedia "Texas Stadium" infobox capacity = "65,675"; revision-pinned https://en.wikipedia.org/w/index.php?title=Texas_Stadium&oldid=1372241047',
    ),
    (
        "DAL99",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=DAL99, roof=outdoors (single-valued, n=57) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "DAL99",
        "timezone",
        'open-meteo timezone=auto @ (32.8398, -96.9109) -> "America/Chicago"',
    ),
    (
        "DAL99",
        "elevation_ft",
        "open-meteo /v1/elevation @ (32.8398, -96.9109) -> 137.0 m -> 449 ft",
    ),
    (
        "DAL99",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=DAL00 (AT&T Stadium) climate_zone=humid_subtropical",
    ),
    (
        "IND99",
        "latitude",
        "wikidata Q2092780 P625 coordinate location = (39.763658333333, -86.163319444444), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q2092780&oldid=2344969515",
    ),
    (
        "IND99",
        "longitude",
        "wikidata Q2092780 P625 coordinate location = (39.763658333333, -86.163319444444), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q2092780&oldid=2344969515",
    ),
    (
        "IND99",
        "surface",
        'en.wikipedia "RCA Dome" infobox surface = "AstroTurf (1984-2004) / FieldTurf (2005-2008)"; FieldTurf covers 28 of the 54 games here (2005-2007), AstroTurf the other 26; revision-pinned https://en.wikipedia.org/w/index.php?title=RCA_Dome&oldid=1372222990',
    ),
    (
        "IND99",
        "capacity",
        'en.wikipedia "RCA Dome" infobox capacity history = "56,127 (1999-2002) / 55,506 (2003-2005) / 55,531 (2006-2008)"; 55,506 covers 27 of the 54 games here; revision-pinned https://en.wikipedia.org/w/index.php?title=RCA_Dome&oldid=1372222990',
    ),
    (
        "IND99",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=IND99, roof=dome (single-valued, n=54) -> NFLVERSE_ROOF_MAP -> indoor",
    ),
    (
        "IND99",
        "timezone",
        'open-meteo timezone=auto @ (39.7637, -86.1633) -> "America/Indiana/Indianapolis"',
    ),
    (
        "IND99",
        "elevation_ft",
        "open-meteo /v1/elevation @ (39.7637, -86.1633) -> 232.0 m -> 761 ft",
    ),
    (
        "IND99",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=IND00 (Lucas Oil Stadium) climate_zone=humid_continental",
    ),
    (
        "PHO99",
        "latitude",
        "wikidata Q1849318 P625 coordinate location = (33.426388888889, -111.9325), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1849318&oldid=2540570507",
    ),
    (
        "PHO99",
        "longitude",
        "wikidata Q1849318 P625 coordinate location = (33.426388888889, -111.9325), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1849318&oldid=2540570507",
    ),
    (
        "PHO99",
        "surface",
        'en.wikipedia "Mountain America Stadium" infobox surface = "Bermuda grass"; the article body states "The natural grass playing surface"; revision-pinned https://en.wikipedia.org/w/index.php?title=Mountain_America_Stadium&oldid=1373448808',
    ),
    (
        "PHO99",
        "capacity",
        'en.wikipedia "Mountain America Stadium" article Capacity table = "1996-2003: 73,379 / 2004-2013: 71,706"; 73,379 covers 17 of the 32 games here; revision-pinned https://en.wikipedia.org/w/index.php?title=Mountain_America_Stadium&oldid=1373448808',
    ),
    (
        "PHO99",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=PHO99, roof=outdoors (single-valued, n=32) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "PHO99",
        "timezone",
        'open-meteo timezone=auto @ (33.4264, -111.9325) -> "America/Phoenix"',
    ),
    (
        "PHO99",
        "elevation_ft",
        "open-meteo /v1/elevation @ (33.4264, -111.9325) -> 359.0 m -> 1178 ft",
    ),
    (
        "PHO99",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=PHO00 (State Farm Stadium) climate_zone=desert",
    ),
    (
        "LAX99",
        "latitude",
        "wikidata Q849784 P625 coordinate location = (34.014167, -118.287778), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q849784&oldid=2541992842",
    ),
    (
        "LAX99",
        "longitude",
        "wikidata Q849784 P625 coordinate location = (34.014167, -118.287778), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q849784&oldid=2541992842",
    ),
    (
        "LAX99",
        "surface",
        'en.wikipedia "Los Angeles Memorial Coliseum" infobox surface = "Bermuda grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=Los_Angeles_Memorial_Coliseum&oldid=1374517166',
    ),
    (
        "LAX99",
        "capacity",
        'en.wikipedia "Los Angeles Memorial Coliseum" infobox capacity = "77,500 / 93,607 (pre-2018)"; 77,500 covers 16 of the 31 games here and 93,607 the other 15 -- both sit above the 75,000 band edge; revision-pinned https://en.wikipedia.org/w/index.php?title=Los_Angeles_Memorial_Coliseum&oldid=1374517166',
    ),
    (
        "LAX99",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=LAX99, roof=outdoors (single-valued, n=31) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "LAX99",
        "timezone",
        'open-meteo timezone=auto @ (34.0142, -118.2878) -> "America/Los_Angeles"',
    ),
    (
        "LAX99",
        "elevation_ft",
        "open-meteo /v1/elevation @ (34.0142, -118.2878) -> 59.0 m -> 194 ft",
    ),
    (
        "LAX99",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=LAX01 (SoFi Stadium) climate_zone=mediterranean",
    ),
    (
        "LAX97",
        "latitude",
        "wikidata Q200684 P625 coordinate location = (33.864444444444, -118.26111111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q200684&oldid=2532006428",
    ),
    (
        "LAX97",
        "longitude",
        "wikidata Q200684 P625 coordinate location = (33.864444444444, -118.26111111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q200684&oldid=2532006428",
    ),
    (
        "LAX97",
        "surface",
        'en.wikipedia "Dignity Health Sports Park" infobox surface = "Bandera Bermuda Grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=Dignity_Health_Sports_Park&oldid=1372080521',
    ),
    (
        "LAX97",
        "capacity",
        'en.wikipedia "Dignity Health Sports Park" infobox capacity = "27,000", the capacity for most games; revision-pinned https://en.wikipedia.org/w/index.php?title=Dignity_Health_Sports_Park&oldid=1372080521',
    ),
    (
        "LAX97",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=LAX97, roof=outdoors (single-valued, n=22) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "LAX97",
        "timezone",
        'open-meteo timezone=auto @ (33.8644, -118.2611) -> "America/Los_Angeles"',
    ),
    (
        "LAX97",
        "elevation_ft",
        "open-meteo /v1/elevation @ (33.8644, -118.2611) -> 23.0 m -> 75 ft",
    ),
    (
        "LAX97",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=LAX01 (SoFi Stadium) climate_zone=mediterranean",
    ),
    (
        "MIN98",
        "latitude",
        "wikidata Q3512039 P625 coordinate location = (44.976389, -93.224444), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q3512039&oldid=2540902485",
    ),
    (
        "MIN98",
        "longitude",
        "wikidata Q3512039 P625 coordinate location = (44.976389, -93.224444), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q3512039&oldid=2540902485",
    ),
    (
        "MIN98",
        "surface",
        'en.wikipedia "Huntington Bank Stadium" infobox surface = "FieldTurf Revolution"; revision-pinned https://en.wikipedia.org/w/index.php?title=Huntington_Bank_Stadium&oldid=1372204608',
    ),
    (
        "MIN98",
        "capacity",
        'en.wikipedia "Huntington Bank Stadium" infobox capacity = "50,805"; revision-pinned https://en.wikipedia.org/w/index.php?title=Huntington_Bank_Stadium&oldid=1372204608',
    ),
    (
        "MIN98",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=MIN98, roof=outdoors (single-valued, n=18) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "MIN98",
        "timezone",
        'open-meteo timezone=auto @ (44.9764, -93.2244) -> "America/Chicago"',
    ),
    (
        "MIN98",
        "elevation_ft",
        "open-meteo /v1/elevation @ (44.9764, -93.2244) -> 253.0 m -> 830 ft",
    ),
    (
        "MIN98",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=MIN01 (U.S. Bank Stadium) climate_zone=humid_continental",
    ),
    (
        "PHI99",
        "latitude",
        "wikidata Q1545870 P625 coordinate location = (39.906666666667, -75.171111111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1545870&oldid=2502313486",
    ),
    (
        "PHI99",
        "longitude",
        "wikidata Q1545870 P625 coordinate location = (39.906666666667, -75.171111111111), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1545870&oldid=2502313486",
    ),
    (
        "PHI99",
        "surface",
        'en.wikipedia "Veterans Stadium" infobox surface = "AstroTurf (1971-2001) / NexTurf (2001-2003)"; NexTurf covers 2002, all 10 games here; revision-pinned https://en.wikipedia.org/w/index.php?title=Veterans_Stadium&oldid=1372249012',
    ),
    (
        "PHI99",
        "capacity",
        'en.wikipedia "Veterans Stadium" infobox capacity = "Baseball: 61,831 / Football: 65,352"; revision-pinned https://en.wikipedia.org/w/index.php?title=Veterans_Stadium&oldid=1372249012',
    ),
    (
        "PHI99",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=PHI99, roof=outdoors (single-valued, n=10) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "PHI99",
        "timezone",
        'open-meteo timezone=auto @ (39.9067, -75.1711) -> "America/New_York"',
    ),
    (
        "PHI99",
        "elevation_ft",
        "open-meteo /v1/elevation @ (39.9067, -75.1711) -> 6.0 m -> 20 ft",
    ),
    (
        "PHI99",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=PHI00 (Lincoln Financial Field) climate_zone=humid_continental",
    ),
    (
        "CHI99",
        "latitude",
        "wikidata Q3305514 P625 coordinate location = (40.099166666667, -88.235833333333), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q3305514&oldid=2532027628",
    ),
    (
        "CHI99",
        "longitude",
        "wikidata Q3305514 P625 coordinate location = (40.099166666667, -88.235833333333), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q3305514&oldid=2532027628",
    ),
    (
        "CHI99",
        "surface",
        'en.wikipedia "Gies Memorial Stadium" infobox surface = "Grass (1923-1974) / AstroTurf (1975-2000) / AstroPlay (2001-2007) / FieldTurf (2008-)"; AstroPlay covers 2002, all 8 games here; revision-pinned https://en.wikipedia.org/w/index.php?title=Gies_Memorial_Stadium&oldid=1374437131',
    ),
    (
        "CHI99",
        "capacity",
        'en.wikipedia "Gies Memorial Stadium" infobox former capacity = "69,249 (2002-2006)", the era the 2002 Bears season falls in; revision-pinned https://en.wikipedia.org/w/index.php?title=Gies_Memorial_Stadium&oldid=1374437131',
    ),
    (
        "CHI99",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=CHI99, roof=outdoors (single-valued, n=8) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "CHI99",
        "timezone",
        'open-meteo timezone=auto @ (40.0992, -88.2358) -> "America/Chicago"',
    ),
    (
        "CHI99",
        "elevation_ft",
        "open-meteo /v1/elevation @ (40.0992, -88.2358) -> 228.0 m -> 748 ft",
    ),
    (
        "CHI99",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=CHI98 (Soldier Field) climate_zone=humid_continental",
    ),
    (
        "BUF01",
        "latitude",
        "wikidata Q76318 P625 coordinate location = (43.641388888889, -79.389166666667), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q76318&oldid=2539727922",
    ),
    (
        "BUF01",
        "longitude",
        "wikidata Q76318 P625 coordinate location = (43.641388888889, -79.389166666667), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q76318&oldid=2539727922",
    ),
    (
        "BUF01",
        "surface",
        'en.wikipedia "Rogers Centre" infobox surface = "AstroTurf (1989-2004) / FieldTurf (2005-2010) / AstroTurf GameDay Grass 3D (2010-2014)"; FieldTurf covers the 2008-2010 games and GameDay Grass 3D the 2011-2013 ones, three each -- both synthetic, so the grass/turf classification is the same either way; revision-pinned https://en.wikipedia.org/w/index.php?title=Rogers_Centre&oldid=1374122313',
    ),
    (
        "BUF01",
        "capacity",
        'en.wikipedia "Rogers Centre" article Seating capacity section, Football table = "53,506"; revision-pinned https://en.wikipedia.org/w/index.php?title=Rogers_Centre&oldid=1374122313',
    ),
    (
        "BUF01",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=BUF01, roof=dome (single-valued, n=6) -> NFLVERSE_ROOF_MAP -> indoor",
    ),
    (
        "BUF01",
        "timezone",
        'open-meteo timezone=auto @ (43.6414, -79.3892) -> "America/Toronto"',
    ),
    (
        "BUF01",
        "elevation_ft",
        "open-meteo /v1/elevation @ (43.6414, -79.3892) -> 84.0 m -> 276 ft",
    ),
    (
        "BUF01",
        "climate_zone",
        "operator entry, no same-metro sibling; Toronto sits across the lake from BUF00 Orchard Park, which is humid_continental",
    ),
    (
        "BRG00",
        "latitude",
        "wikidata Q1594708 P625 coordinate location = (30.411944, -91.185556), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1594708&oldid=2540539711",
    ),
    (
        "BRG00",
        "longitude",
        "wikidata Q1594708 P625 coordinate location = (30.411944, -91.185556), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1594708&oldid=2540539711",
    ),
    (
        "BRG00",
        "surface",
        'en.wikipedia "Tiger Stadium (Louisiana)" infobox surface = "Celebration Bermuda Grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=Tiger_Stadium_(Louisiana)&oldid=1373740250',
    ),
    (
        "BRG00",
        "capacity",
        'en.wikipedia "Tiger Stadium (Louisiana)" infobox capacity history = "92,400 (2005-10)", the era the 2005 games fall in; revision-pinned https://en.wikipedia.org/w/index.php?title=Tiger_Stadium_(Louisiana)&oldid=1373740250',
    ),
    (
        "BRG00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=BRG00, roof=outdoors (single-valued, n=4) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "BRG00",
        "timezone",
        'open-meteo timezone=auto @ (30.4119, -91.1856) -> "America/Chicago"',
    ),
    (
        "BRG00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (30.4119, -91.1856) -> 8.0 m -> 26 ft",
    ),
    (
        "BRG00",
        "climate_zone",
        "operator entry, no same-metro sibling; Baton Rouge follows the nearest existing Gulf record NOR00 New Orleans, which is humid_subtropical",
    ),
    (
        "SAN00",
        "latitude",
        "wikidata Q1618347 P625 coordinate location = (29.416944444444, -98.478888888889), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1618347&oldid=2540540662",
    ),
    (
        "SAN00",
        "longitude",
        "wikidata Q1618347 P625 coordinate location = (29.416944444444, -98.478888888889), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1618347&oldid=2540540662",
    ),
    (
        "SAN00",
        "surface",
        'en.wikipedia "Alamodome" infobox surface = "AstroTurf Magic Carpet II"; revision-pinned https://en.wikipedia.org/w/index.php?title=Alamodome&oldid=1372061422',
    ),
    (
        "SAN00",
        "capacity",
        "wikidata Q1618347 P1083 seating capacity = 65000, qualified P641 sport = Q41323 American football (the article infobox states 64,000 for the present configuration); revision-pinned https://www.wikidata.org/w/index.php?title=Q1618347&oldid=2540540662",
    ),
    (
        "SAN00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=SAN00, roof=dome (single-valued, n=3) -> NFLVERSE_ROOF_MAP -> indoor",
    ),
    (
        "SAN00",
        "timezone",
        'open-meteo timezone=auto @ (29.4169, -98.4789) -> "America/Chicago"',
    ),
    (
        "SAN00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (29.4169, -98.4789) -> 195.0 m -> 640 ft",
    ),
    (
        "SAN00",
        "climate_zone",
        "operator entry, no same-metro sibling; San Antonio follows the nearest existing Texas Gulf record HOU00 Houston, which is humid_subtropical",
    ),
    (
        "LON01",
        "latitude",
        "wikidata Q209725 P625 coordinate location = (51.456111111111, -0.34166666666667), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q209725&oldid=2506987837",
    ),
    (
        "LON01",
        "longitude",
        "wikidata Q209725 P625 coordinate location = (51.456111111111, -0.34166666666667), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q209725&oldid=2506987837",
    ),
    (
        "LON01",
        "surface",
        'en.wikipedia "Twickenham Stadium" infobox surface = "Desso GrassMaster"; revision-pinned https://en.wikipedia.org/w/index.php?title=Twickenham_Stadium&oldid=1372246421',
    ),
    (
        "LON01",
        "capacity",
        'en.wikipedia "Twickenham Stadium" infobox capacity = "82,000 (rugby) / 75,000 (American football) / 55,000 (concerts)"; the American-football figure is the configuration these games were played in; revision-pinned https://en.wikipedia.org/w/index.php?title=Twickenham_Stadium&oldid=1372246421',
    ),
    (
        "LON01",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=LON01, roof=outdoors (single-valued, n=3) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "LON01",
        "timezone",
        'open-meteo timezone=auto @ (51.4561, -0.3417) -> "Europe/London"',
    ),
    (
        "LON01",
        "elevation_ft",
        "open-meteo /v1/elevation @ (51.4561, -0.3417) -> 7.0 m -> 23 ft",
    ),
    (
        "LON01",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=LON00 (Wembley Stadium) climate_zone=oceanic",
    ),
    (
        "GER00",
        "latitude",
        "wikidata Q127429 P625 coordinate location = (48.218775, 11.624752777778), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q127429&oldid=2538614747",
    ),
    (
        "GER00",
        "longitude",
        "wikidata Q127429 P625 coordinate location = (48.218775, 11.624752777778), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q127429&oldid=2538614747",
    ),
    (
        "GER00",
        "surface",
        'en.wikipedia "Allianz Arena" infobox surface = "Hybrid grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=Allianz_Arena&oldid=1374158363',
    ),
    (
        "GER00",
        "capacity",
        "wikidata Q127429 P1083 seating capacity = 75024, preferred rank, qualified P580 start time = 2015, the domestic-match configuration and the era the 2022 and 2024 games fall in; revision-pinned https://www.wikidata.org/w/index.php?title=Q127429&oldid=2538614747",
    ),
    (
        "GER00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=GER00, roof=outdoors (single-valued, n=2) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "GER00",
        "timezone",
        'open-meteo timezone=auto @ (48.2188, 11.6248) -> "Europe/Berlin"',
    ),
    (
        "GER00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (48.2188, 11.6248) -> 491.0 m -> 1611 ft",
    ),
    (
        "GER00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=MUN01 (FC Bayern Munich Stadium) climate_zone=oceanic",
    ),
    (
        "FRA00",
        "latitude",
        "wikidata Q157273 P625 coordinate location = (50.068611, 8.645278), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q157273&oldid=2531932043",
    ),
    (
        "FRA00",
        "longitude",
        "wikidata Q157273 P625 coordinate location = (50.068611, 8.645278), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q157273&oldid=2531932043",
    ),
    (
        "FRA00",
        "surface",
        'en.wikipedia "Waldstadion (Frankfurt)" infobox surface = "Grass"; revision-pinned https://en.wikipedia.org/w/index.php?title=Waldstadion_(Frankfurt)&oldid=1372250182',
    ),
    (
        "FRA00",
        "capacity",
        'en.wikipedia "Waldstadion (Frankfurt)" infobox capacity = "American football: 48,000"; revision-pinned https://en.wikipedia.org/w/index.php?title=Waldstadion_(Frankfurt)&oldid=1372250182',
    ),
    (
        "FRA00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=FRA00, roof=outdoors (single-valued, n=2) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "FRA00",
        "timezone",
        'open-meteo timezone=auto @ (50.0686, 8.6453) -> "Europe/Berlin"',
    ),
    (
        "FRA00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (50.0686, 8.6453) -> 124.0 m -> 407 ft",
    ),
    (
        "FRA00",
        "climate_zone",
        "operator entry, no same-metro sibling; Frankfurt follows the other existing German record MUN01 Munich, which is oceanic",
    ),
    (
        "SAO00",
        "latitude",
        "wikidata Q1362236 P625 coordinate location = (-23.545555555556, -46.474), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1362236&oldid=2509659920",
    ),
    (
        "SAO00",
        "longitude",
        "wikidata Q1362236 P625 coordinate location = (-23.545555555556, -46.474), read this session; revision-pinned https://www.wikidata.org/w/index.php?title=Q1362236&oldid=2509659920",
    ),
    (
        "SAO00",
        "surface",
        'en.wikipedia "Arena Corinthians" infobox surface = "Perennial Ryegrass with Artificial Fibres (Desso GrassMaster)"; revision-pinned https://en.wikipedia.org/w/index.php?title=Arena_Corinthians&oldid=1374053041',
    ),
    (
        "SAO00",
        "capacity",
        "wikidata Q1362236 P1083 seating capacity = 47252, normal rank (the article infobox states 48,905 after a February 2025 increase, which postdates the 2024 game); revision-pinned https://www.wikidata.org/w/index.php?title=Q1362236&oldid=2509659920",
    ),
    (
        "SAO00",
        "roof_type",
        "pinned schedules, load_schedules(2002..2025), stadium_id=SAO00, roof=outdoors (single-valued, n=1) -> NFLVERSE_ROOF_MAP -> outdoor",
    ),
    (
        "SAO00",
        "timezone",
        'open-meteo timezone=auto @ (-23.5456, -46.474) -> "America/Sao_Paulo"',
    ),
    (
        "SAO00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (-23.5456, -46.474) -> 781.0 m -> 2562 ft",
    ),
    (
        "SAO00",
        "climate_zone",
        "same-metro sibling rule: data/venues.json stadium_id=RIO00 (Maracana Stadium) climate_zone=tropical",
    ),
)


# 38 existing records plus the 22 ratified above. tests/unit/test_venues_json_international
# .py's EXPECTED_TOTAL_RECORDS was UPDATED to match rather than forked: a new module
# asserting 60 while the old one asserts 38 is two answers to one question.
VENUE_RECORD_COUNT_AFTER: int = 60


# ---------------------------------------------------------------------------
# THE WEATHER COVERAGE FLAG, AND THE SILVER WEATHER WIDTH AFTER IT.
#
# APPENDED by Plan 33.1-02 Task 1 on 2026-09-12. Nothing above this line was
# edited.
#
# THE THIRD READING OF THE SAME WIDTH, and the three together are the column's
# whole history:
#
#     WEATHER_COLUMNS_BEFORE          = 23   (Plan 33-09 Task 1, pre-provenance)
#     WEATHER_COLUMNS_AFTER           = 24   (Plan 33-09 Task 2, + weather_source)
#     WEATHER_COLUMNS_AFTER_COVERAGE  = 25   (this slot,         + weather_coverage)
#
# MEASURED from `len(data.schemas.WeatherSchema.model_fields)` after the field was
# declared, not counted by hand off the source.
#
# WHY THE FIELD IS REQUIRED RATHER THAN DEFAULTED (Plan 33.1-02 Ruling E): a row
# that does not say whether it is covered is precisely the state the column exists
# to make impossible. It mirrors `is_outdoor`, which is required for the same
# reason and sits beside it in the schema.
#
# WHY IT HAD TO BE DECLARED IN THE SAME TASK THAT EMITS IT (RESEARCH P-6):
# `data/quality_gates.validate_bronze_to_silver` rebuilds every row as
# `schema_class(**row).model_dump()` and Pydantic v2 defaults to extra="ignore",
# so an emitted-but-undeclared column disappears between bronze and silver with no
# error at all. That already happened once, to the five Open-Meteo fields.
# ---------------------------------------------------------------------------

WEATHER_COLUMNS_AFTER_COVERAGE: int = 25
