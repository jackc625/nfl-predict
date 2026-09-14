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


# ---------------------------------------------------------------------------
# THE TRACER SEASON, AND THE SHAPE IT WAS ASSERTED AGAINST.
#
# APPENDED by Plan 33.1-02 Task 2 on 2026-09-12. Nothing above this line was
# edited.
#
# MEASURED from `data.upstream_pin.load_schedules([2016])` on 2026-09-12, joined
# against `data/silver/games.parquet`. No network call was made to measure any of
# it; the pin is sealed for 2016.
#
# WHY 2016 IS THE SLICE. It is the one season that exercises every seam of the
# corrected path AT ONCE, which is what makes a single season a tracer rather
# than a sample:
#
#   * FIVE of the 22 venues Plan 33.1-01 added are in it -- ATL00 (a dome, so
#     the no-call branch), LAX99, LON01 (non-US, so a NON-Eastern IANA zone),
#     OAK00 (a NON-neutral-site game that today misroutes to Allegiant, 650 km
#     away) and SDG00 (which today misroutes to SoFi).
#   * All four feed roof values appear, so both branches of the per-game roof
#     rule are taken.
#   * IND00 carries BOTH `open` and `closed` games, which is the ONLY shape that
#     can prove `is_outdoor` comes from the GAME rather than from the building.
#
# THE COUNTS CLOSE: 198 outdoors + 2 open = 200 fetched; 34 dome + 33 closed = 67
# written with no call; 200 + 67 = 267. A test that asserts the split without
# asserting the sum would pass on a frame that had quietly lost a game.
# ---------------------------------------------------------------------------

TRACER_SEASON: int = 2016

TRACER_SEASON_FACTS: dict[str, object] = {
    "season": 2016,
    "measured_on": "2026-09-12",
    "measured_from": "data.upstream_pin.load_schedules([2016])",
    # The corpus.
    "games": 267,
    "distinct_stadium_ids": 34,
    # The feed's own `roof` distribution. Zero nulls.
    "roof_outdoors": 198,
    "roof_dome": 34,
    "roof_closed": 33,
    "roof_open": 2,
    # The fetch split that follows from it: outdoors + open are fetched,
    # dome + closed are written without a network call.
    "games_fetched": 200,
    "games_written_without_a_call": 67,
    # The five Plan 33.1-01 venues 2016 exercises, with their game counts.
    "historical_venues_present": (
        ("ATL00", 10),
        ("LAX99", 7),
        ("LON01", 1),
        ("OAK00", 7),
        ("SDG00", 8),
    ),
    # The per-game-roof stadiums present in 2016. ATL97 is NOT among them.
    # Only IND00 carries two different roofs, and its two `open` games are the
    # season's only two.
    "per_game_roof_stadiums_present": ("DAL00", "HOU00", "IND00", "PHO00"),
    "stadium_with_both_roof_states": "IND00",
    "open_roof_game_ids": ("2016_W01_DET@IND", "2016_W05_CHI@IND"),
    # What the committed tracer run actually produced in its sandbox, so a later
    # reader can tell a reproduction from a re-derivation.
    "observed_silver_rows": 267,
    "observed_silver_columns": 25,
    "observed_distinct_temp_f": 182,
    "observed_temp_f_min": 5.0,
    "observed_temp_f_max": 90.8,
    "observed_weather_coverage_true": 267,
    "observed_weather_coverage_false": 0,
    "observed_seconds_per_fetch": 1.0,
}


# ---------------------------------------------------------------------------
# THE ROUTING CHANGE, MEASURED PER GAME AGAINST THE REAL PRIOR RULE.
#
# APPENDED by Plan 33.1-03 Task 2 on 2026-09-13. Nothing above this line was edited.
#
# WHAT WAS COMPARED (D33.1-09). For every one of the 6,499 pinned 2002-2025 games,
# the COORDINATES the OLD home-team rule resolves
# (scripts.ingest_weather.WeatherDataIngester._get_venue_record, CALLED rather than
# paraphrased, because the thing most worth catching is a prior behaviour that was
# not what anyone remembered) against the coordinates the NEW stadium_id rule
# resolves (features.contextual.resolve_venue_for_game).
#
# WHY COORDINATES AND NOT WEATHER VALUES. A coordinate comparison is exact and
# unconfounded. A weather-value comparison over the same games is confounded by
# D33.1-08's venue-local-hour fix, which lands in this same phase and moves a value
# on essentially every comparable row; a number that moved for two reasons at once
# cannot be attributed to either.
#
# RE-DERIVED, NOT TRANSCRIBED. 33.1-RESEARCH.md section 3.1 reported this diff on
# the same tree. It was recomputed here from the pinned feed rather than copied,
# and tests/integration/test_routing_coordinate_diff.py recomputes it again on
# every run and compares. Where a figure differs from the research, BOTH are
# recorded below and the divergence says which measurement produced which -- the
# idiom GOLD_WEATHER_CONSTANCY_MEASUREMENT's own divergence block established.
#
# NOTHING HERE IS A CLAIM ABOUT ACCURACY. It is a count of games whose resolved
# venue moved. No model was re-fit, no gold matrix has been rebuilt at these
# coordinates yet, and artifacts/latest.json is untouched.
# ---------------------------------------------------------------------------

ROUTING_COORDINATE_DIFF: dict[str, object] = {
    "measured_on": "2026-09-13",
    "measured_by": "Plan 33.1-03 Task 2",
    "measured_from": "data.upstream_pin.load_schedules(range(2002, 2026))",
    "old_rule": "scripts.ingest_weather.WeatherDataIngester._get_venue_record",
    "new_rule": "features.contextual.resolve_venue_for_game",
    "compared_on": ("latitude", "longitude"),
    "games_total": 6499,
    "games_changed": 1153,
    "distinct_stadium_ids_changed": 40,
    "games_naming_a_newly_added_id": 1082,
    "games_naming_an_already_known_id": 71,
    "neutral_site_games": 83,
    "weather_applicable_games": 737,
    # Every stadium_id whose games changed venue, not only the twelve largest:
    # a truncated breakdown could not be checked against the total above, and
    # the tail is where a misroute nobody predicted would sit.
    "per_stadium_id": {
        "OAK00": {
            "games": 141,
            "successor": "VEG00",
            "successor_games": 141,
            "successors": {"VEG00": 141},
            "neutral_site_games": 0,
            "weather_applicable_games": 141,
        },
        "NYC00": {
            "games": 132,
            "successor": "NYC01",
            "successor_games": 131,
            "successors": {"NOR00": 1, "NYC01": 131},
            "neutral_site_games": 0,
            "weather_applicable_games": 132,
        },
        "ATL00": {
            "games": 125,
            "successor": "ATL97",
            "successor_games": 125,
            "successors": {"ATL97": 125},
            "neutral_site_games": 0,
            "weather_applicable_games": 0,
        },
        "SDG00": {
            "games": 125,
            "successor": "LAX01",
            "successor_games": 124,
            "successors": {"LAX01": 124, "TAM00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 125,
        },
        "STL00": {
            "games": 112,
            "successor": "LAX01",
            "successor_games": 112,
            "successors": {"LAX01": 112},
            "neutral_site_games": 0,
            "weather_applicable_games": 0,
        },
        "SFO00": {
            "games": 99,
            "successor": "SFO01",
            "successor_games": 99,
            "successors": {"SFO01": 99},
            "neutral_site_games": 0,
            "weather_applicable_games": 99,
        },
        "MIN00": {
            "games": 95,
            "successor": "MIN01",
            "successor_games": 95,
            "successors": {"MIN01": 95},
            "neutral_site_games": 0,
            "weather_applicable_games": 0,
        },
        "DAL99": {
            "games": 57,
            "successor": "DAL00",
            "successor_games": 57,
            "successors": {"DAL00": 57},
            "neutral_site_games": 0,
            "weather_applicable_games": 57,
        },
        "IND99": {
            "games": 54,
            "successor": "IND00",
            "successor_games": 54,
            "successors": {"IND00": 54},
            "neutral_site_games": 0,
            "weather_applicable_games": 0,
        },
        "PHO99": {
            "games": 32,
            "successor": "PHO00",
            "successor_games": 31,
            "successors": {"LAX01": 1, "PHO00": 31},
            "neutral_site_games": 1,
            "weather_applicable_games": 32,
        },
        "LAX99": {
            "games": 31,
            "successor": "LAX01",
            "successor_games": 31,
            "successors": {"LAX01": 31},
            "neutral_site_games": 0,
            "weather_applicable_games": 31,
        },
        "LON00": {
            "games": 26,
            "successor": "JAX00",
            "successor_games": 10,
            "successors": {
                "ATL97": 1,
                "CIN00": 1,
                "JAX00": 10,
                "KAN00": 1,
                "LAX01": 3,
                "MIA00": 3,
                "MIN01": 1,
                "NOR00": 1,
                "SFO01": 1,
                "TAM00": 2,
                "VEG00": 2,
            },
            "neutral_site_games": 26,
            "weather_applicable_games": 26,
        },
        "LAX97": {
            "games": 22,
            "successor": "LAX01",
            "successor_games": 22,
            "successors": {"LAX01": 22},
            "neutral_site_games": 0,
            "weather_applicable_games": 22,
        },
        "MIN98": {
            "games": 18,
            "successor": "MIN01",
            "successor_games": 18,
            "successors": {"MIN01": 18},
            "neutral_site_games": 0,
            "weather_applicable_games": 18,
        },
        "LON02": {
            "games": 10,
            "successor": "ATL97",
            "successor_games": 1,
            "successors": {
                "ATL97": 1,
                "BUF00": 1,
                "CHI98": 1,
                "GNB00": 1,
                "JAX00": 1,
                "MIN01": 1,
                "NAS00": 1,
                "NOR00": 1,
                "TAM00": 1,
                "VEG00": 1,
            },
            "neutral_site_games": 10,
            "weather_applicable_games": 10,
        },
        "PHI99": {
            "games": 10,
            "successor": "PHI00",
            "successor_games": 10,
            "successors": {"PHI00": 10},
            "neutral_site_games": 0,
            "weather_applicable_games": 10,
        },
        "CHI99": {
            "games": 8,
            "successor": "CHI98",
            "successor_games": 8,
            "successors": {"CHI98": 8},
            "neutral_site_games": 0,
            "weather_applicable_games": 8,
        },
        "PHO00": {
            "games": 7,
            "successor": "SFO01",
            "successor_games": 3,
            "successors": {"BOS00": 1, "LAX01": 1, "PHI00": 1, "SEA00": 1, "SFO01": 3},
            "neutral_site_games": 7,
            "weather_applicable_games": 0,
        },
        "BUF01": {
            "games": 6,
            "successor": "BUF00",
            "successor_games": 6,
            "successors": {"BUF00": 6},
            "neutral_site_games": 2,
            "weather_applicable_games": 0,
        },
        "MEX00": {
            "games": 5,
            "successor": "PHO00",
            "successor_games": 2,
            "successors": {"LAX01": 1, "PHO00": 2, "VEG00": 2},
            "neutral_site_games": 5,
            "weather_applicable_games": 5,
        },
        "BRG00": {
            "games": 4,
            "successor": "NOR00",
            "successor_games": 4,
            "successors": {"NOR00": 4},
            "neutral_site_games": 0,
            "weather_applicable_games": 4,
        },
        "DET00": {
            "games": 4,
            "successor": "BUF00",
            "successor_games": 2,
            "successors": {"BUF00": 2, "MIN01": 1, "PIT00": 1},
            "neutral_site_games": 4,
            "weather_applicable_games": 0,
        },
        "LON01": {
            "games": 3,
            "successor": "LAX01",
            "successor_games": 2,
            "successors": {"CLE00": 1, "LAX01": 2},
            "neutral_site_games": 3,
            "weather_applicable_games": 3,
        },
        "MIA00": {
            "games": 3,
            "successor": "CHI98",
            "successor_games": 1,
            "successors": {"CHI98": 1, "IND00": 1, "KAN00": 1},
            "neutral_site_games": 3,
            "weather_applicable_games": 3,
        },
        "SAN00": {
            "games": 3,
            "successor": "NOR00",
            "successor_games": 3,
            "successors": {"NOR00": 3},
            "neutral_site_games": 0,
            "weather_applicable_games": 0,
        },
        "FRA00": {
            "games": 2,
            "successor": "BOS00",
            "successor_games": 1,
            "successors": {"BOS00": 1, "KAN00": 1},
            "neutral_site_games": 2,
            "weather_applicable_games": 2,
        },
        "GER00": {
            "games": 2,
            "successor": "CAR00",
            "successor_games": 1,
            "successors": {"CAR00": 1, "TAM00": 1},
            "neutral_site_games": 2,
            "weather_applicable_games": 2,
        },
        "HOU00": {
            "games": 2,
            "successor": "ATL97",
            "successor_games": 1,
            "successors": {"ATL97": 1, "BOS00": 1},
            "neutral_site_games": 2,
            "weather_applicable_games": 0,
        },
        "JAX00": {
            "games": 2,
            "successor": "NOR00",
            "successor_games": 1,
            "successors": {"NOR00": 1, "PHI00": 1},
            "neutral_site_games": 2,
            "weather_applicable_games": 2,
        },
        "NOR00": {
            "games": 2,
            "successor": "PHI00",
            "successor_games": 1,
            "successors": {"PHI00": 1, "SFO01": 1},
            "neutral_site_games": 2,
            "weather_applicable_games": 0,
        },
        "SFO01": {
            "games": 2,
            "successor": "BOS00",
            "successor_games": 1,
            "successors": {"BOS00": 1, "DEN00": 1},
            "neutral_site_games": 2,
            "weather_applicable_games": 2,
        },
        "ATL97": {
            "games": 1,
            "successor": "LAX01",
            "successor_games": 1,
            "successors": {"LAX01": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 0,
        },
        "DAL00": {
            "games": 1,
            "successor": "GNB00",
            "successor_games": 1,
            "successors": {"GNB00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 0,
        },
        "IND00": {
            "games": 1,
            "successor": "BOS00",
            "successor_games": 1,
            "successors": {"BOS00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 0,
        },
        "LAX01": {
            "games": 1,
            "successor": "CIN00",
            "successor_games": 1,
            "successors": {"CIN00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 0,
        },
        "MIN01": {
            "games": 1,
            "successor": "BOS00",
            "successor_games": 1,
            "successors": {"BOS00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 0,
        },
        "NYC01": {
            "games": 1,
            "successor": "DEN00",
            "successor_games": 1,
            "successors": {"DEN00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 1,
        },
        "SAO00": {
            "games": 1,
            "successor": "PHI00",
            "successor_games": 1,
            "successors": {"PHI00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 1,
        },
        "TAM00": {
            "games": 1,
            "successor": "PHO00",
            "successor_games": 1,
            "successors": {"PHO00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 1,
        },
        "VEG00": {
            "games": 1,
            "successor": "KAN00",
            "successor_games": 1,
            "successors": {"KAN00": 1},
            "neutral_site_games": 1,
            "weather_applicable_games": 0,
        },
    },
    "divergence_from_recorded_claims": {
        "the_five_way_split": "REPRODUCED EXACTLY. 33.1-RESEARCH.md section 3.1 "
        "recorded 1,153 changed / 1,082 naming a newly added id / 71 naming an "
        "already-known id / 83 carrying location == Neutral / 737 weather-"
        "applicable under R4's roof rule. All five re-derived to the same digit.",
        "per_stadium_counts": "SHAPE REPRODUCED, THREE COUNTS DIVERGED, AND THE "
        "TWO MEASUREMENTS ANSWER DIFFERENT QUESTIONS. RESEARCH's twelve-row table "
        "counts games per (true venue, successor) PAIR; this slot counts games per "
        "TRUE VENUE. Where a venue's games scattered across more than one "
        "successor the two differ by exactly the scattered games: NYC00 131 "
        "(pair) against 132 (venue), SDG00 124 against 125, PHO99 31 against 32. "
        "Each difference is one neutral-site game whose nominal home team was a "
        "different franchise. Both figures are recorded and neither is wrong.",
        "the_twelve_largest": "RE-CUT. Ranked by games per TRUE VENUE, LON00 (26 "
        "games) is larger than LAX97 (22) and MIN98 (18), so the twelve largest by "
        "that ranking are not the twelve RESEARCH tabled. LON00 is deliberately "
        "absent from the named-pair assertions in "
        "tests/integration/test_routing_coordinate_diff.py because it is a "
        "neutral-site venue whose old answers scatter across ELEVEN successors -- "
        '"the successor it used to resolve to" is not a well-formed claim about '
        "it. It is present in per_stadium_id above with all eleven.",
        "distinct_stadium_ids": "NOT PREVIOUSLY RECORDED. 40 of the 55 pinned "
        "stadium_id values had at least one game change venue. The 22 newly added "
        "historical records account for 1,082 of the 1,153; the remaining 71 are "
        "spread across 18 ALREADY-KNOWN ids, almost all of them neutral-site games "
        "at venues the file already carried.",
    },
}


# ---------------------------------------------------------------------------
# THE 91-GAME NEUTRAL-SITE MISRESOLUTION, REPAIRED.
#
# APPENDED by Plan 33.1-03 Task 2 on 2026-09-13. Nothing above this line was
# edited, and in particular HISTORICAL_NEUTRAL_MISRESOLUTION is BYTE-UNCHANGED.
#
# THIS IS A NEW SLOT, NOT A CORRECTION TO THE OLD ONE. The slot above is the
# record of a measured DEFECT and of the decision D33-15 took about it: history
# kept its home-team resolution, on purpose, because re-resolving it would move
# gold under three deployed models in the same change that fixed the forward
# season. That decision was correct when it was taken and the record of it is not
# rewritten. The append protocol is enforced mechanically by
# tests/unit/test_phase33_state_append_once.py, which reports any name this file
# assigns twice.
#
# WHAT CHANGED. The owner took the deferred repair at Phase 33.1 (D33.1-06), as
# part of retiring the routing gate entirely rather than as a separate act. All 91
# games now resolve to the stadium the feed says they were played at, and zero
# remain misrouted. The cost the earlier slot named is REAL and is now PAID
# deliberately: gold does move for those rows, and Plan 33.1-07 attributes that
# movement to the venue/travel/timezone/elevation cause family rather than to
# weather.
#
# THE TRAIN-ROW SHARES IN THE EARLIER SLOT ARE NOT RESTATED HERE. They describe
# what the deployed artifacts were fitted under, and those artifacts have not been
# re-fit -- this phase corrects inputs and promotes nothing.
# ---------------------------------------------------------------------------

HISTORICAL_NEUTRAL_MISRESOLUTION_REPAIRED: dict[str, object] = {
    "measured_on": "2026-09-13",
    "measured_by": "Plan 33.1-03 Task 2",
    "measured_from": "data.upstream_pin.load_schedules(range(2002, 2026))",
    # The population is the FEED's own `location` column, never silver's
    # `neutral_site`: every one of the 6,499 silver cells reads False because
    # `row.get("neutral_site", False)` never reads a feed value at all, and Phase
    # 33 Wave 12, which backfills it, has not run.
    "population_source": "the pinned feed's own `location` column",
    "games_disclosed": 91,
    "neutral_site_games_resolving_to_their_own_stadium": 91,
    "still_misrouted": 0,
    "repaired_by": "D33.1-06",
    "deferred_by": "D33-15",
    "supersedes": "HISTORICAL_NEUTRAL_MISRESOLUTION",
    # The repair is a SUBSET of the wider routing change, not a separate one: 83 of
    # the 91 changed venue, and the other 8 already resolved correctly by
    # coincidence -- their nominal home team's present-day stadium happened to be
    # the one the game was played at.
    "of_which_changed_venue": 83,
    "of_which_already_resolved_correctly": 8,
}


# ---------------------------------------------------------------------------
# RULING J -- WHAT EACH OF THE THREE D33.1-07 STATES PRODUCES, GROUP BY GROUP.
#
# APPENDED by Plan 33.1-04 Task 1 on 2026-09-13. Nothing above this line was
# edited.
#
# THE PRINCIPLE THE TABLE RESOLVES. D33.1-07 gives three states and one
# inheritance rule, and the inheritance rule and the dome rule appear to pull in
# opposite directions for the impact columns. The rule that resolves them: a
# column that answers "WHAT WAS THE WEATHER" is NULL when there was no
# measurement; a column that answers "HOW MUCH DID WEATHER AFFECT THIS GAME"
# carries its no-impact level when weather DOES NOT APPLY, and is NULL when
# weather applies but was not measured. D33.1-07 already applies exactly this
# reasoning when it keeps indoor wind and precipitation at a genuine 0.0.
#
# THE INTENDED CONSEQUENCE, STATED RATHER THAN DISCOVERED. 1,652 indoor games
# gain NaN across THIRTEEN columns -- the nine temperature columns plus the four
# temperature-derived impact and multiplier columns. That is a real part of the
# R5 rung's change set and Plan 33.1-07 pre-declares it. The seven composite
# columns are the ONLY group whose indoor value is unchanged from before this
# plan, and that is the same judgement D33.1-07 makes for wind and
# precipitation, applied to the columns that express impact rather than
# measurement.
#
# THIS IS A RULING, NOT A MEASUREMENT. Every other slot in this module records
# something counted off the repository. This one records a DECISION about what
# the code should produce, committed here so the three states cannot be
# re-litigated from memory by a later plan that reads only the code.
# ---------------------------------------------------------------------------

WEATHER_NULL_STATE_MATRIX: dict[str, object] = {
    "ruled_on": "2026-09-13",
    "ruled_by": "Plan 33.1-04 Task 1 (Ruling J, with the Codex HIGH amendment)",
    "decision": "D33.1-07",
    "requirements": ("R4", "R5"),
    # The three states, keyed by the (weather_coverage, is_outdoor) pair that
    # distinguishes them. Before the coverage flag existed all three collapsed
    # into one, because a missing record was written with is_outdoor False.
    "states": {
        "covered_indoor": {"weather_coverage": 1.0, "is_outdoor": False},
        "covered_outdoor_observed": {"weather_coverage": 1.0, "is_outdoor": True},
        "uncovered_outdoor_absent": {"weather_coverage": 0.0, "is_outdoor": True},
    },
    "column_groups": {
        "temperature": (
            "raw_temp_f",
            "temp_f",
            "apparent_temp_f",
            "temp_hot",
            "temp_warm",
            "temp_mild",
            "temp_cool",
            "temp_cold",
            "temp_very_cold",
        ),
        "temperature_impact": (
            "cold_impact_score",
            "heat_impact_score",
            "scoring_multiplier",
            "ball_handling_difficulty",
        ),
        "humidity": ("raw_humidity_pct",),
        "wind": (
            "raw_wind_mph",
            "wind_mph",
            "wind_calm",
            "wind_moderate",
            "wind_high",
            "wind_severe",
            "wind_impact_score",
            "kicking_difficulty",
            "passing_difficulty",
        ),
        "precipitation": (
            "raw_precip_mm",
            "raw_precip_prob",
            "precip_mm",
            "precip_prob",
            "precip_none",
            "precip_light",
            "precip_moderate",
            "precip_heavy",
            "is_snow",
            "is_rain",
            "is_dry",
            "precip_impact_score",
            "turnover_multiplier",
            "passing_efficiency",
        ),
        "composite": (
            "weather_severity_score",
            "home_weather_advantage",
            "defensive_advantage",
            "rushing_advantage",
            "scoring_reduction",
            "weather_game",
            "extreme_weather",
        ),
        "flags": ("weather_affects_game", "weather_coverage"),
    },
    # One row of Ruling J's table per group: what the group takes in each state.
    # "null" means NaN; "genuine" means a true statement about a covered game
    # rather than a stand-in; "measured" means the calculation ran on a reading.
    "table": {
        "temperature": {
            "covered_indoor": "null",
            "uncovered_outdoor_absent": "null",
            "covered_outdoor_observed": "measured",
        },
        "temperature_impact": {
            "covered_indoor": "null",
            "uncovered_outdoor_absent": "null",
            "covered_outdoor_observed": "measured",
        },
        "humidity": {
            "covered_indoor": "null",
            "uncovered_outdoor_absent": "null",
            "covered_outdoor_observed": "measured",
        },
        "wind": {
            "covered_indoor": "genuine calm",
            "uncovered_outdoor_absent": "null",
            "covered_outdoor_observed": "measured",
        },
        "precipitation": {
            "covered_indoor": "genuine dry",
            "uncovered_outdoor_absent": "null",
            "covered_outdoor_observed": "measured",
        },
        "composite": {
            "covered_indoor": "genuine 0.0",
            "uncovered_outdoor_absent": "null",
            "covered_outdoor_observed": "measured",
        },
        "flags": {
            "covered_indoor": "weather_affects_game 0.0, weather_coverage 1.0",
            "uncovered_outdoor_absent": (
                "weather_affects_game 1.0, weather_coverage 0.0"
            ),
            "covered_outdoor_observed": (
                "weather_affects_game 1.0, weather_coverage 1.0"
            ),
        },
    },
    "indoor_games_gaining_nan": 1652,
    "indoor_columns_gaining_nan": 13,
    # The Ruling J amendment (Codex 33.1-04 HIGH). The two indoor factories were
    # reached from TWO places: the indoor branch, where their values are true,
    # and an `except` handler, where they fabricated a whole family out of a
    # calculation error. The rename is what stops the next `except` branch being
    # wired to them by analogy.
    "renamed_indoor_factories": {
        "_default_wind_features": "_indoor_wind_features",
        "_default_precipitation_features": "_indoor_precipitation_features",
    },
    "deleted_factories": ("_default_temperature_features",),
    # Every family whose `except` handler now re-raises as
    # WeatherObservationError. FOUR, not the three the plan named: the composite
    # severity handler returned the whole family at 0.0 and is the same defect.
    "raising_families": ("temperature", "wind", "precipitation", "weather severity"),
    "exception_type": "features.weather.WeatherObservationError",
    "exception_base": "RuntimeError",
    "exception_base_reason": (
        "scripts/build_features._SOURCE_LOAD_ERRORS contains ValueError, "
        "KeyError and TypeError, so a refusal typed as any of those would be "
        "caught and converted into an empty frame -- the swallow this refusal "
        "exists to replace. ProvisionalSnapshotAsTrainingInputError is typed "
        "the same way for the same recorded reason."
    ),
}


# ---------------------------------------------------------------------------
# THE TWO MECHANISMS THAT WOULD HAVE PUT THE STAND-IN BACK, AND THE OPT-IN,
# PER-BUILDER SEAM THAT STOPS THEM.
#
# APPENDED by Plan 33.1-04 Task 2 on 2026-09-13. Nothing above this line was
# edited.
#
# WHY THIS SLOT EXISTS AT ALL. Plan 33.1-04 Task 1 stopped `features/weather.py`
# producing a numeric stand-in for an absent observation. Two mechanisms
# DOWNSTREAM of it would have re-introduced one, silently, after the deletion --
# and neither is named in the SPEC, the CONTEXT or the research. Recording them
# here is what stops the next reader concluding that deleting the factory was
# the whole fix.
#
# THE SEAM IS OPT-IN AND PER BUILDER. `preserve_missing_cols` defaults to empty,
# so every column outside the declared weather set keeps today's behaviour byte
# for byte, and the declared set is keyed by builder identity because the two
# weather builders do not emit the same columns (Ruling K1, Codex 33.1-04
# MEDIUM). The proof is a CONSUMPTION assertion captured at the real call sites,
# not a reading of the constant.
# ---------------------------------------------------------------------------

MISSING_PRESERVING_SEAM: dict[str, object] = {
    "recorded_on": "2026-09-13",
    "recorded_by": "Plan 33.1-04 Task 2",
    "requirement": "R5",
    "prohibition": (
        "MUST NOT replace the 65.0 constant with any other numeric stand-in "
        "for a missing observation, under any name -- a seasonal average or a "
        "venue mean is the same defect wearing a better label"
    ),
    "mechanisms": {
        "game_level_median": {
            "file": "scripts/build_features.py",
            "symbol": "FeatureMatrixBuilder._impute_game_level_features",
            "what_it_would_have_done": (
                "filled an absent weather observation with a prior-seasons "
                "median -- the seasonal average SPEC prohibition 1 names"
            ),
            "exempted_by": (
                "handle_missing_data_and_outliers skips BOTH imputers for a "
                "column in the active builder's preserving set"
            ),
        },
        "neutral_z_score": {
            "file": "features/normalization.py",
            "symbol": "expanding_normalize",
            "what_it_would_have_done": (
                "mapped the NaN to 0.0 -- the neutral z-score -- at the final "
                "normalized.fillna(0.0)"
            ),
            "exempted_by": (
                "the input NaN mask is captured BEFORE normalizing and "
                "restored after the fill, for named columns only"
            ),
            "why_the_fill_still_exists": (
                "TWO CAUSES WERE COLLAPSED INTO ONE FILL. The fallback is "
                "correct for a position whose STATISTIC was unavailable -- an "
                "early week below min_periods with no prior-season bootstrap. "
                "It is wrong for a position whose VALUE was absent. This "
                "separates them and changes nothing for the first case."
            ),
        },
    },
    "default_is_empty": True,
    "fail_closed": True,
    "fail_closed_rule": (
        "a merged weather frame whose entry for the builder about to run is "
        "absent, None or empty raises, naming BOTH the attribute and the "
        "builder key. An exemption that silently does nothing is worse than "
        "none, because it reads as a guarantee and behaves as a comment; one "
        "that is live for a single builder is worse still, because half the "
        "evidence says it works."
    ),
    "builder_keys": ("full", "compressed"),
    "preserved_set_size_by_builder": {"full": 47, "compressed": 4},
    "attribute": "scripts.build_features.FeatureMatrixBuilder.missing_preserving_columns",
    "sole_writer": "FeatureMatrixBuilder.record_missing_preserving_columns",
    "derived_not_hand_listed": True,
    # The winsorization pass is DELIBERATELY not exempted: a measured
    # temperature has genuine outliers. `weather_coverage` is already exempt
    # under the CR-02 discrete-indicator rule, which is why the coverage flag
    # cannot be clipped into a constant on a single-season build.
    "winsorization_exempted": False,
    "consumption_test": "tests/unit/test_missing_preserving_seam.py",
}


# ---------------------------------------------------------------------------
# WP'S NULLABLE-INPUT CONTRACT, AND THE ALTERNATIVE THE OWNER REJECTED.
#
# APPENDED by Plan 33.1-04 Task 4 on 2026-09-13. Nothing above this line was
# edited.
#
# THIS IS A SCOPE EXPANSION PAST THE SPEC'S LITERAL WORDING, and it is recorded
# here rather than left silent. The SPEC's R4 says an absent observation is
# written as NULL; it does not say what the WP trainer does when that NULL
# reaches feature selection. D33.1-R3 answers that, owner-ratified at replan
# time 2026-09-12 and attributed to R4 as nullable-input handling.
#
# THE MECHANISM, SO THE DECISION IS READABLE WITHOUT THE REVIEW THREAD.
# `WPTrainer.train_and_evaluate` calls `select_features` BEFORE anything is
# scaled; `BaseTrainer.select_features` fits `self._create_model(...)` over
# every informative candidate column; `informative_columns` withholds a column
# that does not VARY, and a NaN-bearing weather column DOES vary. WP's estimator
# was `sklearn.linear_model.LogisticRegression`, which rejects NaN. The
# currently-deployed `wp_20260824_113325` feature list containing zero weather
# features does NOT protect re-selection: selection re-runs on every call and
# `models.train` exposes no feature-list flag.
#
# THE REJECTION IS THE PART WORTH KEEPING. Excluding nullable weather from WP's
# selection would have worked and would have been less code. The owner refused
# it because it forecloses WP ever using the weather this phase exists to
# produce -- a decision about what the system is FOR, not about how to write it.
# ---------------------------------------------------------------------------

WP_NULLABLE_INPUT_CONTRACT: dict[str, object] = {
    "ruled_on": "2026-09-12",
    "recorded_on": "2026-09-13",
    "recorded_by": "Plan 33.1-04 Task 4",
    "decision": "D33.1-R3",
    "requirement": "R4",
    "owner_ratified": True,
    "scope_expansion_past_spec_wording": True,
    "composes_with": "D33.1-R1 (Plan 33.1-10 persists this same object)",
    "pipeline_steps": ("imputer", "missing_indicator", "scaler", "estimator"),
    "declared_in": "models.trainers.wp_trainer.WP_PIPELINE_STEP_NAMES",
    "imputer_strategy": "median",
    "imputer_strategy_reason": (
        "robust to the heavy tails a temperature or wind distribution has. The "
        "imputed VALUE is not asked to mean anything -- the indicator column "
        "beside it is what carries the information that the reading was absent."
    ),
    "missing_indicator_features": "all",
    "missing_indicator_features_reason": (
        "the emitted column set is then a deterministic function of the INPUT "
        "COLUMNS rather than of which rows happened to be null in the fit "
        "window. A data-dependent indicator set would make two re-fits produce "
        "two different feature spaces, which is the reproducibility constraint "
        "CLAUDE.md states."
    ),
    "missing_indicator_suffix": "_was_missing",
    "imputer_fitted_per_fold": True,
    "imputer_fitted_per_fold_is": "a TEMPORAL-SAFETY requirement, not a nicety",
    "imputer_fitted_per_fold_reason": (
        "an imputation statistic computed over the whole frame and applied "
        "inside a fold leaks the holdout's distribution into the training set "
        "-- the class of defect CLAUDE.md's walk-forward constraint forbids and "
        "that this milestone exists to detect. Pipeline.fit fits every step on "
        "the rows it is handed, and the fit site hands it that fold's "
        "pre-holdout rows."
    ),
    "per_fold_proof": (
        "tests/unit/test_wp_nullable_weather_inputs.py::"
        "TestTheImputerNeverCrossesAFoldBoundary::test_the_imputer_is_fitted_per_fold"
    ),
    "applies_to_pre_selection_path": True,
    "applies_to_pre_selection_path_via": (
        "models.trainers.base.BaseTrainer.select_features fits "
        "self._create_model(...), which for WP is now the Pipeline, so the "
        "selection path never hands NaN to a bare estimator"
    ),
    # The scaler moved INSIDE the estimator. Recorded because it is the property
    # D33.1-R1 exists to guarantee and the reason wp_20260824_113325 is
    # trained-scaled and served-raw today.
    "scaler_is_inseparable_from_the_estimator": True,
    "scaler_accessor": "WPTrainer.scaler (a property reading the fitted pipeline)",
    "ats_and_ou_unaffected": True,
    "ats_and_ou_unaffected_reason": (
        "both _create_model methods return an XGBoost model, which takes NaN "
        "natively, so the base.py selection change is behaviour-preserving"
    ),
    "rejected_alternative": {
        "option": "exclude nullable weather columns from WP's feature selection",
        "rejected_by": "the owner, at replan time 2026-09-12",
        "reason": (
            "it would foreclose WP ever using the weather this phase exists to produce"
        ),
    },
    "nothing_was_fitted_or_promoted": True,
    "nothing_was_fitted_or_promoted_note": (
        "this task changes how WP WOULD fit. Phase 33 Wave 15 is what fits it, "
        "and artifacts/latest.json is untouched."
    ),
}


# ---------------------------------------------------------------------------
# THE WP POOLED-ACCURACY ANCHOR, RE-RATIFIED BY OWNER RULING.
#
# APPENDED by Plan 33.1-04 on 2026-09-13, under the owner ruling of the same
# date. Nothing above this line was edited, and in particular
# WP_NULLABLE_INPUT_CONTRACT is BYTE-UNCHANGED.
#
# WHAT HAPPENED, IN ORDER, BECAUSE THE ORDER IS THE POINT. Plan 33.1-04's
# plan-level verification found
# tests/integration/test_diag_diagnosis.py::TestDiagDiagnosis::
# test_backtest_numbers_match_audit_report RED. The executor did NOT edit the
# anchor and did NOT widen the band: that test's own docstring reserves both
# branches to a deliberate act, so the executor re-measured TWICE and REPORTED A
# HALT. The owner then took the RE-RATIFY branch. A rule that has never actually
# been followed is not evidence that it works; this is the record that it was.
#
# THIS IS NOT A RESULT (SPEC R8, a hard constraint rather than a caveat). The
# reading moved UP. Phase 33.1 re-fit no model, promoted no model and left
# artifacts/latest.json byte-unchanged. A temporal-correctness defect was
# corrected and a number moved as a side effect of that correction. The model did
# not get better, and the accuracy question is not answerable until a re-fit on
# real weather has actually run -- none has. Any sentence a reader could quote as
# "Phase 33.1 improved accuracy" is a misreading of this slot.
#
# THE NODE IS NOT A TRIPWIRE. It is green again because the constant now records
# what the code produces, so it is deliberately ABSENT from
# DELIBERATE_TRIPWIRE_NODE_IDS, which stays at five members.
# ---------------------------------------------------------------------------

WP_ACCURACY_ANCHOR_RERATIFICATION: dict[str, object] = {
    "ruled_on": "2026-09-13",
    "ruled_by": "the owner, on Plan 33.1-04's reported halt",
    "recorded_by": "Plan 33.1-04",
    "node_id": (
        "tests/integration/test_diag_diagnosis.py::TestDiagDiagnosis::"
        "test_backtest_numbers_match_audit_report"
    ),
    "constant": "tests.integration.test_diag_diagnosis.ANCHOR_WP_ACCURACY",
    "superseded_value": 0.67691,
    "superseded_ratified_on": "2026-08-24",
    "superseded_ratified_by": "Plan 30-16 Task 3, under owner ruling D30-OWNER-02",
    "new_value": 0.6821773485513608,
    "drift": 0.0052673485513608,
    "band": 0.005,
    "drift_exceeded_the_band": True,
    "drift_exceeded_the_band_is_why": (
        "a drift inside the band would have passed silently and needed no "
        "ruling. This one did not, which is what sent it to the owner."
    ),
    # The reproducibility is what makes re-ratification defensible rather than a
    # guess about noise. Two independent runs, identical to every printed digit.
    "measured_times": 2,
    "measurements": (0.6821773485513608, 0.6821773485513608),
    "measurements_identical": True,
    "band_was_widened": False,
    "band_after": 0.005,
    "superseded_value_retained_in_source": True,
    "cause": "D33.1-R3",
    "cause_stated": (
        "D33.1-R3 changed WPTrainer into a fold-fitted imputation Pipeline, so "
        "the fill rule is fitted INSIDE each walk-forward fold rather than "
        "across the whole frame. run_diagnosis(run_backtest_half=True) drives "
        "backtest/engine.py -> WPTrainer.train_and_evaluate, which is the path "
        "that moved. A deliberate temporal-correctness change, not an "
        "unexplained wobble."
    ),
    "is_a_result": False,
    "spec_r8_note": (
        "The reading moved UP and that is NOT a result. No model was re-fit and "
        "none was promoted; artifacts/latest.json is byte-unchanged. A "
        "methodology defect was corrected and this number moved as a side "
        "effect. The accuracy question is not answerable until a re-fit on real "
        "weather has run, and none has."
    ),
    "added_to_deliberate_tripwires": False,
    "halt_was_reported_before_any_edit": True,
    "clv_anchor_changed": False,
    "clv_anchor_note": (
        "ANCHOR_HEADLINE_CLV_WP stays at its Plan 30-16 value. The accuracy "
        "assertion fires first, so the CLV reading was never reached while the "
        "test was red; it is re-confirmed by the green run rather than moved."
    ),
}


# ---------------------------------------------------------------------------
# THE COMPLETE LEGACY BRONZE WEATHER INVENTORY, THE CORPUS LOCK'S CONTRACT, AND
# THE THREE-WAY NULL-OBSERVATION CLASSIFICATION.
#
# APPENDED by Plan 33.1-05 Task 1 on 2026-09-13. Nothing above this line was
# edited.
#
# WHY THE INVENTORY IS PINNED BY CONTENT DIGEST AND NOT BY ROW COUNT. SPEC
# prohibition 6 says the 1,942 pre-existing bronze rows are neither overwritten
# nor deleted, and they are the ONLY evidence the routing fix can be
# regression-tested against. A row count would miss a rewrite that preserved the
# count -- which is exactly what a re-fetch under corrected routing would produce.
# The digest is `tests.data_boundary.digest_file`, so it degrades to a DECLARED
# stat signature on a locked file rather than lying about the instrument.
#
# THE 2025 FILE IS THE TRAP. `weather_raw_bronze_2025_W00_20260407T025733.parquet`
# is a WHOLE-SEASON `week=0` sentinel written by the current code, and it holds 78
# rows covering weeks 1 to 5 only, because `_load_games_data(2025, None)` read a
# silver `games` table that held 78 rows of 2025 at the time. A resume rule keyed
# on `_W00_` existence would call 2025 done and leave 207 games unfetched behind a
# green log. That is N-06.
#
# MEASURED 2026-09-13 by reading every one of the ten files.
# ---------------------------------------------------------------------------

LEGACY_WEATHER_BRONZE_INVENTORY: dict[str, object] = {
    "measured_on": "2026-09-13",
    "measured_by": "Plan 33.1-05 Task 1",
    "measured_from": "data/bronze/weather_raw_bronze_*.parquet",
    "digest_instrument": "tests.data_boundary.digest_file (sha256 of file bytes)",
    "legacy_total_rows": 1942,
    "grand_total_rows": 2048,
    "legacy_season_files": 7,
    "file_count": 10,
    "partial_2025_note": (
        "weather_raw_bronze_2025_W00_20260407T025733.parquet is a WHOLE-SEASON "
        "week=0 sentinel written by the current save_bronze_snapshot path. It "
        "holds 78 rows covering weeks 1 to 5 ONLY. A filename-based resume rule "
        "reads it as a completed season and leaves 207 games of 2025 unfetched "
        "(N-06). This is why the resume rule is a game-id COVERAGE test under a "
        "DISJOINT bronze table name."
    ),
    "disjointness": (
        "the new corpus writes under scripts.backfill_historical_weather."
        "BACKFILL_BRONZE_TABLE = 'weather_backfill', whose glob "
        "'weather_backfill_raw_bronze_*' matches none of the ten filenames below. "
        "SPEC prohibition 6 is satisfied STRUCTURALLY rather than by care, and the "
        "disjointness is asserted at import time rather than assumed."
    ),
    "files": {
        "weather_raw_bronze_2018_season.parquet": {
            "rows": 267,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "24684f32dd2bc4144f2356c605b301b5cb79bedb1b5872cd842616303ac81dd9"
            ),
        },
        "weather_raw_bronze_2019_season.parquet": {
            "rows": 267,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "4dda8debdb9eccf40cf8a6dc413966a6bee73c032c6aff986d9d2e64449cc3df"
            ),
        },
        "weather_raw_bronze_2020_season.parquet": {
            "rows": 269,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "bb414bb219e3489a2f875ef33bcaa09e6b1d3183be597e2d5da1ace3da7fe03a"
            ),
        },
        "weather_raw_bronze_2021_season.parquet": {
            "rows": 285,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "f2244e24d2d4dc5afa8cc0d255a2ac3f6df2229859415dbcd86a1debc9de7fb6"
            ),
        },
        "weather_raw_bronze_2022_season.parquet": {
            "rows": 284,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "770f3047e3c7dc6bd3388c69841ea466dfbfcb5c45d2ec036d4a6eb084875d0b"
            ),
        },
        "weather_raw_bronze_2023_season.parquet": {
            "rows": 285,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "6aff392185ff79ed69849327189fd7dc9326e0fcb958e4e19386c80861d951f8"
            ),
        },
        "weather_raw_bronze_2024_season.parquet": {
            "rows": 285,
            "columns": 17,
            "scheme": "legacy whole-season, no _W##_, no timestamp",
            "digest": (
                "19a37202de3374184da761ed4a083724d58969c914ea395ad04cbab15f30748b"
            ),
        },
        "weather_raw_bronze_2025_W05.parquet": {
            "rows": 14,
            "columns": 17,
            "scheme": "legacy week file, no timestamp",
            "digest": (
                "daf0f0a475576841d5826e32442971db25b95952485dffdd8a847c62e0aa7143"
            ),
        },
        "weather_raw_bronze_2024_W06_20260416T165923.parquet": {
            "rows": 14,
            "columns": 23,
            "scheme": "current save_bronze_snapshot",
            "digest": (
                "2031b8d9971e470271b9587b7cac55a12a5e3b069c4098f2ad7a2240dcd10b96"
            ),
        },
        "weather_raw_bronze_2025_W00_20260407T025733.parquet": {
            "rows": 78,
            "columns": 23,
            "scheme": "current save_bronze_snapshot, week=0 whole-season sentinel",
            "digest": (
                "6e7ea65f7acdcea5d74a69945b11f666369a75cb520b169c621bb6208ce67c36"
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# THE CORPUS LOCK'S CONTRACT (Ruling L3).
#
# WHY IT EXISTS AT ALL. `save_bronze_snapshot(..., exclusive=True)` is a FILENAME
# backstop: it stops two writers creating the SAME snapshot path and says nothing
# about two runs racing the corpus. `data.storage.upsert_silver`
# (data/storage.py:1110-1123) reads the existing parquet, filters it by key,
# concats and only then writes atomically -- the WRITE is atomic, the
# READ-MODIFY-WRITE is not. Two promoters each read the pre-state, and the second
# write silently discards the first's rows with no error and a digest bracket that
# still reports exactly one CHANGED path, exactly as declared.
#
# WHY THE STALE-LOCK RULE IS A REFUSAL RATHER THAN AN AGE HEURISTIC. A paced
# 81-minute run legitimately holds the lock for 81 minutes, so any age threshold
# short enough to be useful is short enough to break a healthy run, and breaking a
# healthy run mid-corpus is the one failure this phase cannot afford.
# ---------------------------------------------------------------------------

CORPUS_LOCK_CONTRACT: dict[str, object] = {
    "recorded_by": "Plan 33.1-05 Task 1",
    "recorded_on": "2026-09-13",
    "path": "<data root>/bronze/.weather_backfill.lock",
    "path_constant": "scripts.backfill_historical_weather.CORPUS_LOCK_PATH",
    "path_resolver": "scripts.backfill_historical_weather.corpus_lock_path",
    "primitive": (
        "open(path, 'xb') -- the CREATE is the exclusive step, across processes. "
        "The same primitive save_bronze_snapshot already relies on, applied at RUN "
        "scope instead of FILE scope."
    ),
    "lives_under_bronze_because": (
        "a lock that is not on the same filesystem as the thing it guards is "
        "guarding a different filesystem -- the reason _atomic_write_parquet uses "
        "a SIBLING tempfile rather than a system temp directory"
    ),
    "scope": (
        "corpus identity",
        "resume determination",
        "every season's fetch",
        "the silver promotion",
        "post-state verification (Plan 33.1-06's digest bracket runs INSIDE it)",
    ),
    "recorded_fields": (
        "pid",
        "acquired_at_utc",
        "corpus.table",
        "corpus.first_season",
        "corpus.last_season",
    ),
    "stale_lock_policy": "REFUSE, naming the recorded pid, the age and --force-unlock",
    "never_takes_over_because": (
        "a healthy paced run legitimately holds the lock for 81 minutes, so any age "
        "threshold short enough to be useful would break it mid-corpus"
    ),
    "exception_policy": "RETAINED on an exception inside the context",
    "exception_policy_is_why": (
        "threat T-33.1-31c is a lock that silently disappears, letting the next run "
        "proceed over a half-written corpus. An exception mid-corpus leaves the "
        "corpus in an unknown state, so the lock stays and the operator clears it "
        "deliberately with --force-unlock. The plan's Task-1 acceptance criteria and "
        "its test list both require this; one prose bullet in Ruling L3 said the "
        "opposite and was resolved in favour of the criteria and the threat."
    ),
    "proof": (
        "tests/integration/test_weather_corpus_lock.py -- a TWO-PROCESS test. A "
        "same-thread double-acquire proves nothing about the property, because the "
        "hazard is cross-process."
    ),
}


# ---------------------------------------------------------------------------
# THE THREE-WAY NULL-OBSERVATION CLASSIFICATION (Ruling L4).
#
# R4 explicitly makes a MISSING OBSERVATION recordable data, and the D33.1-07
# absent-observation state exists for exactly that case. A flat 1 percent bar would
# therefore convert legitimate ERA5 absence into a phase-stopping condition --
# refusing to write a season because the archive honestly has no reading for two
# games, which is the opposite of what this phase is for.
#
# THE CONTIGUOUS-RUN TEST IS THE DISCRIMINATING ONE. RESEARCH A2's unprobed
# assumption is that limit exhaustion returns a 400 carrying a `reason`; if the
# provider instead answers 200-with-null-arrays, the absences arrive CONSECUTIVELY,
# while genuine ERA5 gaps scatter across a season by geography and date. A 30
# percent scattered season and a 30 percent consecutive season are different facts
# and the gate must say which one it saw.
# ---------------------------------------------------------------------------

NULL_OBSERVATION_CLASSIFICATION: dict[str, object] = {
    "recorded_by": "Plan 33.1-05 Task 1",
    "recorded_on": "2026-09-13",
    "arms": ("CLEAN", "AMBIGUOUS", "OUTAGE_LIKE"),
    "isolated_null_fraction_ceiling": 0.01,
    "outage_null_fraction_floor": 0.25,
    "outage_contiguous_run": 8,
    "denominator": (
        "the games that were ACTUALLY FETCHED in the season, in fetch order. A dome "
        "was never asked, so including it would dilute the fraction and make the "
        "gate read a number it did not measure."
    ),
    "clean_is_written_with_no_override": True,
    "only_outage_like_stops_the_run": True,
    "r4_note": (
        "R4 makes an ISOLATED missing observation recordable data, so only the "
        "OUTAGE-LIKE arm is a phase-stopping condition. AMBIGUOUS refuses BY "
        "DEFAULT but names the exact --accept-null-fraction <season>=<fraction> "
        "flag carrying the OBSERVED fraction, so a season cannot be pre-authorised "
        "blind; a flag carrying a different fraction still refuses."
    ),
    "accepted_override_is_recorded_in": (
        "<data root>/bronze/.weather_backfill_overrides.jsonl, with the fraction, "
        "the longest contiguous run and the date -- an override nobody can find "
        "afterwards is indistinguishable from no gate at all"
    ),
    "replaced_constant": "MAX_NULL_OBSERVATION_FRACTION_PER_SEASON (never written)",
}


# ---------------------------------------------------------------------------
# THE ARCHIVE CALL BUDGET: what the run is PROJECTED to cost, beside what the
# tracer MEASURED.
#
# APPENDED by Plan 33.1-05 Task 2 on 2026-09-13. Nothing above this line was
# edited.
#
# THIS IS A ZERO-RETRY PROJECTION, and it says so in a key rather than only in
# prose. `fetch_game_weather` is wrapped in `@retry(stop=stop_after_attempt(3))`,
# so a retried game issues up to three requests and the provider counts three.
# Ruling L2 therefore debits the budget INSIDE the retried function, immediately
# before the request -- and the OBSERVED weighted total Plan 33.1-06 records
# INCLUDES retries. The gap between that observation and the projection below IS
# THE RETRY RATE. Comparing them without saying so is how a 10% retry rate reads
# as a budget-model error.
#
# THE 1.2 MULTIPLIER IS AN EXTRAPOLATION, logged as assumption A1. The vendor
# states that a request for more than ten weather variables counts as multiple
# API calls and gives ONE worked example (15 variables over 14 days = 1.5). The
# request here sends TWELVE variables for ONE day, so 12/10 = 1.2 is inferred
# from that single example rather than from a published formula. The archive
# response carries NO rate-limit header -- re-confirmed over the tracer's 200
# responses, nothing matching `rate`, `request` or `limit` -- so there is no
# observable counter to check the inference against. What is MEASURED below is
# REQUESTS ISSUED; the weight applied to them is not.
# ---------------------------------------------------------------------------

ARCHIVE_BUDGET_PLAN: dict[str, object] = {
    "recorded_by": "Plan 33.1-05 Task 2",
    "recorded_on": "2026-09-13",
    "limits": {
        "minutely": 600,
        "hourly": 5000,
        "daily": 10000,
        "monthly": 300000,
        "provenance": (
            "scraped VERBATIM from the vendor's pricing table "
            "(open-meteo.com/en/pricing) during 33.1-RESEARCH section 5.1 -- CITED, "
            "not inferred"
        ),
        "binding_limit": "hourly",
        "binding_limit_is_why": (
            "4,847 fetches at the 1.2 weight is about 5,816 weighted calls, which "
            "fits the 10,000/day cap with roughly 42% headroom and does NOT fit "
            "the 5,000/hour cap unless the run holds itself to an interval"
        ),
    },
    "weight": 1.2,
    "weight_provenance": (
        "EXTRAPOLATED from the vendor's single worked example (15 variables over 14 "
        "days = 1.5 calls). The rule that a request for more than ten weather "
        "variables counts as multiple API calls is CITED; the 12/10 = 1.2 figure "
        "for a twelve-variable single-day request is not. Assumption A1."
    ),
    "variables_requested": 12,
    "variables_read": 11,
    "unread_variable": "rain",
    "unread_variable_note": (
        "dropping `rain` would take the weight from 1.2 to 1.1 and the minimum run "
        "from 70 to 64 minutes. NOT taken (Ruling M): HOURLY_VARIABLES is shared "
        "with the live forecast path, so the change touches COLD-06's surface for a "
        "six-minute saving. Recorded as a known lever rather than an oversight."
    ),
    # The corpus split, measured from the pinned feed's own `roof` values.
    "games_total": 6499,
    "games_fetched": 4847,
    "games_without_a_call": 1652,
    # THE PROJECTION. Zero retries assumed, stated in a key.
    "projected_weighted_calls": 5816.4,
    "retries_assumed": 0,
    "retries_assumed_note": (
        "the OBSERVED weighted total Plan 33.1-06 records INCLUDES retries. The gap "
        "between the two IS the retry rate, not a model error."
    ),
    "interval_seconds": 1.0,
    "interval_derivation": (
        "5,816 weighted calls / 5,000 per hour = 1.164 hours minimum, i.e. about 70 "
        "minutes, i.e. a mean inter-ATTEMPT interval of at least 0.87 s. 1.0 s gives "
        "about 81 minutes with margin."
    ),
    "projected_seconds": 4847.0,
    "projected_wall_clock": "about 81 minutes",
    "projected_wall_clock_is_a_floor": True,
    "projected_wall_clock_is_a_floor_because": (
        "tenacity's exponential backoff now COMPOSES with the interval rather than "
        "replacing it, so a retried game takes longer than three intervals"
    ),
    # MEASURED, so the cited limits sit beside an observation (Plan 33.1-02's
    # tracer, 200 real requests over season 2016, recorded in COVERAGE.md).
    "measured_seconds_per_request": 0.134,
    "measured_seconds_per_request_basis": (
        "median of 8 sequential steady-state requests (0.128-0.139); the first "
        "request of a session cost 0.571 s for the TLS handshake, paid once"
    ),
    "measured_season": 2016,
    "measured_requests_issued": 200,
    "measured_mean_interval_seconds": 1.001,
    "measured_paced_season_seconds": 200.3,
    "measured_retries": 0,
    "unpaced_wall_clock_note": (
        "at 0.134 s per request the whole corpus would finish in about eleven "
        "minutes and present roughly 31,700 weighted calls inside that hour -- more "
        "than six times the hourly cap. The interval is what makes the run fit."
    ),
    "no_rate_limit_header": True,
    "no_rate_limit_header_note": (
        "a live header scan of a successful archive response returned nothing "
        "matching `rate`, `request` or `limit`, re-confirmed over the tracer's 200 "
        "responses. The Phase-29 x-requests-last pattern has NO analogue here, "
        "which is exactly why the run counts its own calls."
    ),
}


# ---------------------------------------------------------------------------
# THE WITNESS FOR PHASE 33.1'S WEATHER CROSS-CHECK PRE-REGISTRATION.
#
# APPENDED by Plan 33.1-05 Task 3 on 2026-09-13, in a SEPARATE, LATER commit than
# the pre-registration itself. THAT SEPARATION IS THE WHOLE POINT. Nothing above
# this line was edited.
#
# WHY THE ANCHOR LIVES HERE AND NOT INSIDE THE FILE IT WITNESSES (REVIEW-CIRCULAR).
# A document that must CONTAIN and exactly REPRODUCE its own whole-file hash is
# self-referential: writing the hash changes the bytes the hash was computed over,
# so no fixed point exists without a canonical exclusion rule nobody has defined. A
# test written against such a marker would have to be relaxed into meaninglessness.
# Phase 30 proved the outside-witness pattern and Phase 31 reused it; this is the
# third use, not a new idea.
#
# WHAT IT IS NOT. This does NOT extend Phase 31's pre-registration.
# `backtest.ev_chain_constants.PREREGISTRATION_PATHS` names exactly two paths and
# `test_both_preregistration_files_exist_and_are_tracked` asserts that count;
# adding to it would redefine a PUBLISHED pre-registration after the fact. Phase
# 33.1's ancestry check is a NEW class in tests/unit/test_preregistration_ancestry.py
# reusing that module's existing helpers (Ruling L).
#
# THE RELATION ASSERTED: the commit below is a STRICT ancestor of HEAD, and it is
# NOT the commit that records it. A rule and the numbers it produced landing in one
# commit is not a pre-registration; it is only a claim of one.
#
# THE DIGEST IS NEWLINE-NORMALIZED -- every CRLF folded to LF before hashing. This
# repository has core.autocrlf=true and no .gitattributes, so the file is LF in the
# git blob and CRLF in a fresh Windows working tree; a digest over raw working-tree
# bytes would pin a value that holds only on the machine that measured it. Verified
# equal by BOTH routes at append time: the normalized working-tree bytes and
# `git cat-file blob <commit>:<path>` produce the same value.
# ---------------------------------------------------------------------------

WEATHER_CROSSCHECK_PREREGISTRATION_COMMIT: str = (
    "57586c19125030153531ba55acc0fc6f9e417dde"
)

WEATHER_CROSSCHECK_PREREGISTRATION_FILE_SHA256: dict[str, str] = {
    "scripts/weather_crosscheck_constants.py": (
        "5825ed0343a11a61057db97f6823dd90ef9a6055eaaec2542c83fe61e9465219"
    ),
}

WEATHER_CROSSCHECK_PREREGISTRATION_PROVENANCE: dict[str, str] = {
    "plan": "33.1-05",
    "task": "Task 3: register the expected diff SHAPE, then witness it from outside",
    "date": "2026-09-13",
    "resolved_by": "git log -1 --format=%H -- scripts/weather_crosscheck_constants.py",
    "hash_basis": (
        "newline-normalized file bytes (CRLF folded to LF); equals the git blob "
        "sha256, verified by both routes at append time"
    ),
    "commit_contents": "exactly scripts/weather_crosscheck_constants.py and nothing else",
    "does_not_extend": (
        "backtest.ev_chain_constants.PREREGISTRATION_PATHS is UNTOUCHED -- that "
        "tuple defines Phase 31's pre-registration and a test asserts it holds "
        "exactly two paths (Ruling L)"
    ),
    "predicts": (
        "three causes with measured populations -- the hour fix (1,580 comparable "
        "measured rows of 1,942), the routing fix (737 weather-applicable, 44 inside "
        "the 2018-2024 comparable window) and the per-game roof rule (621 closed "
        "games, the only number-to-NULL category) -- plus two explicit NON-claims"
    ),
    "fourth_quantity": (
        "the archive-versus-forecast provenance probe is pre-registered as a "
        "MEASUREMENT with an explicit n and an interpretation rule, not as an "
        "expectation. No sign and no magnitude is predicted, because none is known."
    ),
}


# ---------------------------------------------------------------------------
# THE DECLARED BLAST RADIUS OF THE FULL-CORPUS SILVER PROMOTION.
#
# APPENDED by Plan 33.1-06 Task 2 on 2026-09-13, and COMMITTED BEFORE the first
# digest snapshot of the run was taken. Nothing above this line was edited.
#
# THE ORDER MATTERS AND IS THE POINT, exactly as it was for the 14-row backfill
# above (Plan 33-09). A blast radius declared after the fact is not a declaration,
# it is a transcription of whatever happened. This slot was committed first, then
# the corpus lock was acquired, then the pre-state digests were taken, then the
# corpus was fetched, then the promotion ran, then `verify` was run against this
# declaration. A FILE OUTSIDE THIS SET IS A FINDING TO REPORT, NEVER A REASON TO
# WIDEN THE SET.
#
# ONE FILE, and the DuckDB half is deliberately NOT in it. The promotion writes
# through `data.storage.upsert_silver`, which reads the existing parquet, filters
# it by key, concats and ends at `_atomic_write_parquet` -- it NEVER opens the
# database. It is `save_dataframe`, whose `save_to_db` defaults True, that writes
# both halves, and this path does not call it. So `data/nfl_predictions.duckdb` is
# EXCLUDED as a POSITIVE STATEMENT ABOUT THE WRITE PATH, not as an oversight: if
# it moves, the write did not go where this declaration says it went, and that is
# a finding.
#
# THE RUN HALF OF THE BRACKET IS A DIFFERENT SHAPE AND IS DECLARED SEPARATELY
# BELOW. Fetching 24 seasons legitimately ADDS 24 bronze files and CHANGES
# nothing; the promotion CHANGES exactly one file and ADDS nothing. Two steps,
# two declared shapes, so neither can absorb the other's surprise.
# ---------------------------------------------------------------------------

WEATHER_PROMOTION_EXPECTED_CHANGED_FILES: tuple[str, ...] = ("silver/weather.parquet",)

# The RUN half. `backfill_season` writes one timestamped bronze snapshot per
# season through `data.storage.save_bronze_snapshot`, which is parquet-only and
# append-only: it creates a NEW file and never opens the database either. So the
# fetch is declared as 24 ADDED paths, every one of them matching this prefix, and
# ZERO changed and ZERO removed. Silver must not move at all during the fetch --
# `--all-seasons` promotes nothing.
WEATHER_CORPUS_RUN_EXPECTED_ADDED_PREFIX: str = "bronze/weather_backfill_raw_bronze_"
WEATHER_CORPUS_RUN_EXPECTED_ADDED_COUNT: int = 24


# ---------------------------------------------------------------------------
# WHAT THE FULL-CORPUS RUN ACTUALLY DID.
#
# APPENDED by Plan 33.1-06 Task 2 on 2026-09-13, AFTER the run. Nothing above
# this line was edited -- the declaration slot above stays exactly as it was
# written before the run, which is the only thing that makes it a declaration.
#
# THE BRACKET, in the order it was executed, ALL OF IT INSIDE ONE LOCK
# ACQUISITION (Ruling L3, threat T-33.1-40c):
#
#   0. CorpusLock acquired -- pid 13128, 2026-09-14T00:23:33.585813+00:00 --
#      BEFORE the pre-state digest. A lock taken AFTER the digest protects
#      nothing: upsert_silver is read-filter-concat-write, so a second promoter
#      that read the same pre-state silently discards the first promoter's rows
#      while the bracket still reports exactly one CHANGED path, as declared.
#   1. data      -> outputs/phase331_corpus_before.json     (424 files digested)
#   2. artifacts -> outputs/phase331_artifacts_before.json  (159 files digested)
#   3. the 24-season fetch, promoting NOTHING
#   4. both trees re-digested, still inside the lock
#
# THE RUN HALF REPORTED EXACTLY THE DECLARED SHAPE: 24 ADDED paths, every one
# matching WEATHER_CORPUS_RUN_EXPECTED_ADDED_PREFIX, and ZERO changed, ZERO
# removed and ZERO undecided (no MIXED key). `artifacts/` did not move at all.
# `data/silver/weather.parquet` digested to its pre-run value on both sides --
# the fetch promoted nothing, which is what makes the promotion in Task 3 a
# separately-bracketed single write. `data/nfl_predictions.duckdb` did not move:
# save_bronze_snapshot is parquet-only and append-only.
#
# THE LOCK WAS ABSENT AFTER THE CLEAN EXIT, which is the other half of
# CORPUS_LOCK_CONTRACT: retained on an exception, removed on a clean one.
#
# ZERO RETRIES, AND THAT IS A MEASUREMENT RATHER THAN AN ASSUMPTION. The
# projection in ARCHIVE_BUDGET_PLAN is labelled `retries_assumed: 0`, and Ruling
# L2 put the debit INSIDE the retried fetch precisely so that the OBSERVED total
# would include retries if any occurred. Observed 5,816.4 weighted over 4,847
# attempts; projected 5,816.4 over 4,847 requests. The gap is exactly zero, so
# the retry rate is exactly zero -- 4,847 games, 4,847 attempts, no attempt
# repeated. Recorded as the measurement it is, BESIDE the projection rather than
# instead of it.
#
# THE 1.2 WEIGHT IS STILL UNVERIFIED, AND THIS RUN COULD NOT VERIFY IT
# (.planning/WINDOWS.md row 26, assumption A1). What was MEASURED is REQUESTS
# ISSUED: 4,847 game fetches plus ONE corpus-floor probe, 4,848 live requests in
# total. The 1.2 multiplier applied to them is an EXTRAPOLATION from a single
# published vendor example, the archive response carries no rate-limit header,
# and the provider never refused -- so the run supplies EVIDENCE THAT THE REAL
# WEIGHT IS NOT CATASTROPHICALLY HIGHER (a true weight above about 2.06 would
# have breached the 10,000/day cap at this volume) but it does NOT confirm 1.2.
# An unverified constant that happened not to bind is still unverified, and row
# 26 stays OPEN.
# ---------------------------------------------------------------------------

WEATHER_CORPUS_RUN: dict[str, object] = {
    "measured_on": "2026-09-13",
    "measured_by": "Plan 33.1-06 Task 2",
    "command": "uv run python -m scripts.backfill_historical_weather --bracket run",
    "owner_ruling": (
        "APPROVED 2026-09-13 about 20:20 EDT. The owner authorised the budget "
        "spend and the silver overwrite after being shown the dry run's zero "
        "requests, the 4,847/1,652 split, the 81-minute paced projection against "
        "the HOURLY 5,000 binding limit, and the 14-row pre-state. Assumption A2 "
        "was ACCEPTED with the per-season null-fraction gate as its control."
    ),
    # THE LOCK.
    "lock_path": "data/bronze/.weather_backfill.lock",
    "lock_pid": 13128,
    "lock_acquired_at_utc": "2026-09-14T00:23:33.585813+00:00",
    "lock_acquired_before_the_pre_state_digest": True,
    "lock_absent_after_clean_exit": True,
    # THE CLOCK.
    "started_utc": "2026-09-14T00:23:32Z",
    "first_fetch_utc": "2026-09-14T00:23:35.929271Z",
    "last_snapshot_utc": "2026-09-14T01:44:35.372929Z",
    "observed_wall_clock_seconds": 4863.4,
    "observed_wall_clock": "81.1 minutes",
    "projected_wall_clock": "about 81 minutes",
    "observed_seconds_per_attempt": 1.0026,
    "declared_interval_seconds": 1.0,
    # THE CORPUS.
    # Written out rather than built with tuple(range(...)): this module's own
    # contract is NO module-level calls, asserted by
    # tests/unit/test_phase33_state_shape.test_the_state_module_has_no_module_level_calls.
    "seasons_fetched": (
        2002,
        2003,
        2004,
        2005,
        2006,
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
    ),
    "seasons_still_missing_after": (),
    "games_total": 6499,
    "games_fetched": 4847,
    "games_without_a_call": 1652,
    "snapshots_written": 24,
    "snapshot_column_count": 25,
    "bronze_rows_written": 6499,
    # THE BUDGET -- observed BESIDE projected, never instead of it.
    "observed_weighted_calls": 5816.4,
    "projected_weighted_calls": 5816.4,
    "attempts": 4847,
    "retry_count": 0,
    "retry_count_is_how_derived": (
        "attempts (4,847) minus games fetched (4,847). The debit sits INSIDE the "
        "retried fetch (Ruling L2), so a retried game costs an extra attempt and "
        "the difference IS the retry rate."
    ),
    "corpus_floor_probe_requests": 1,
    "live_requests_total": 4848,
    "provider_refusals": 0,
    "rate_limit_headers_seen": 0,
    "weight_still_unverified": True,
    "weight_still_unverified_because": (
        "ARCHIVE_CALL_WEIGHT_PER_REQUEST = 1.2 is extrapolated from one published "
        "vendor example and the archive sends no rate-limit header, so this run "
        "measured REQUESTS ISSUED and not calls CHARGED. The provider never "
        "refused, which bounds the real weight below roughly 2.06 against the "
        "daily cap, but a constant that happened not to bind is still unverified. "
        "WINDOWS.md row 26 stays OPEN."
    ),
    # THE NULL-OBSERVATION GATE (Ruling L4). Every season, with its arm, below.
    "null_arms_observed": ("CLEAN",),
    "absent_observations_total": 0,
    "overrides_accepted": (),
    "ambiguous_seasons": (),
    "outage_like_seasons": (),
    "null_gate_note": (
        "ALL 24 SEASONS CLASSIFIED CLEAN AT A ZERO ABSENT-OBSERVATION FRACTION. "
        "The archive answered every one of the 4,847 outdoor games. So the gate "
        "never fired, the AMBIGUOUS and OUTAGE_LIKE arms were never exercised on "
        "live data, and no --accept-null-fraction override was requested or "
        "granted. Recorded plainly because 'the gate did not fire' and 'the gate "
        "works' are different statements, and only the first one was observed "
        "here; the arms are exercised by "
        "tests/unit/test_weather_backfill_resume.py."
    ),
    # RESUMABILITY. Recorded as a fact about this run rather than as a claim.
    "interruptions": 0,
    "resumed_from_season": None,
    "resumability_note": (
        "the run completed in ONE invocation and was never interrupted, so its "
        "resume path was NOT exercised here. R3's resumability is proven by "
        "tests/unit/test_weather_backfill_resume.py and by the coverage rule "
        "returning all 24 seasons before the run and the empty tuple after it -- "
        "not by this run, which had no interruption to recover from."
    ),
    "archive_floor_probe": {
        "date": "2002-09-05",
        "stadium_id": "NYC00",
        "timezone": "America/New_York",
        "hours": 24,
        "nulls": 0,
    },
    "per_season": (
        {
            "season": 2002,
            "games": 267,
            "fetched": 216,
            "written_without_a_call": 51,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2002_W00_20260914T002711.parquet",
        },
        {
            "season": 2003,
            "games": 267,
            "fetched": 211,
            "written_without_a_call": 56,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2003_W00_20260914T003043.parquet",
        },
        {
            "season": 2004,
            "games": 267,
            "fetched": 214,
            "written_without_a_call": 53,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2004_W00_20260914T003418.parquet",
        },
        {
            "season": 2005,
            "games": 267,
            "fetched": 219,
            "written_without_a_call": 48,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2005_W00_20260914T003757.parquet",
        },
        {
            "season": 2006,
            "games": 267,
            "fetched": 206,
            "written_without_a_call": 61,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2006_W00_20260914T004124.parquet",
        },
        {
            "season": 2007,
            "games": 267,
            "fetched": 207,
            "written_without_a_call": 60,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2007_W00_20260914T004451.parquet",
        },
        {
            "season": 2008,
            "games": 267,
            "fetched": 214,
            "written_without_a_call": 53,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2008_W00_20260914T004825.parquet",
        },
        {
            "season": 2009,
            "games": 267,
            "fetched": 198,
            "written_without_a_call": 69,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2009_W00_20260914T005143.parquet",
        },
        {
            "season": 2010,
            "games": 267,
            "fetched": 199,
            "written_without_a_call": 68,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2010_W00_20260914T005502.parquet",
        },
        {
            "season": 2011,
            "games": 267,
            "fetched": 196,
            "written_without_a_call": 71,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2011_W00_20260914T005819.parquet",
        },
        {
            "season": 2012,
            "games": 267,
            "fetched": 195,
            "written_without_a_call": 72,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2012_W00_20260914T010139.parquet",
        },
        {
            "season": 2013,
            "games": 267,
            "fetched": 196,
            "written_without_a_call": 71,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2013_W00_20260914T010456.parquet",
        },
        {
            "season": 2014,
            "games": 267,
            "fetched": 203,
            "written_without_a_call": 64,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2014_W00_20260914T010819.parquet",
        },
        {
            "season": 2015,
            "games": 267,
            "fetched": 205,
            "written_without_a_call": 62,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2015_W00_20260914T011144.parquet",
        },
        {
            "season": 2016,
            "games": 267,
            "fetched": 200,
            "written_without_a_call": 67,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2016_W00_20260914T011504.parquet",
        },
        {
            "season": 2017,
            "games": 267,
            "fetched": 205,
            "written_without_a_call": 62,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2017_W00_20260914T011829.parquet",
        },
        {
            "season": 2018,
            "games": 267,
            "fetched": 202,
            "written_without_a_call": 65,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2018_W00_20260914T012152.parquet",
        },
        {
            "season": 2019,
            "games": 267,
            "fetched": 206,
            "written_without_a_call": 61,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2019_W00_20260914T012518.parquet",
        },
        {
            "season": 2020,
            "games": 269,
            "fetched": 176,
            "written_without_a_call": 93,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2020_W00_20260914T012814.parquet",
        },
        {
            "season": 2021,
            "games": 285,
            "fetched": 202,
            "written_without_a_call": 83,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2021_W00_20260914T013136.parquet",
        },
        {
            "season": 2022,
            "games": 284,
            "fetched": 198,
            "written_without_a_call": 86,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2022_W00_20260914T013455.parquet",
        },
        {
            "season": 2023,
            "games": 285,
            "fetched": 199,
            "written_without_a_call": 86,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2023_W00_20260914T013814.parquet",
        },
        {
            "season": 2024,
            "games": 285,
            "fetched": 187,
            "written_without_a_call": 98,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2024_W00_20260914T014122.parquet",
        },
        {
            "season": 2025,
            "games": 285,
            "fetched": 193,
            "written_without_a_call": 92,
            "null_arm": "CLEAN",
            "absent_observations": 0,
            "null_fraction": 0.0,
            "longest_contiguous_run": 0,
            "override_accepted": False,
            "snapshot": "weather_backfill_raw_bronze_2025_W00_20260914T014435.parquet",
        },
    ),
}


# ---------------------------------------------------------------------------
# THE SINGLE BRACKETED SILVER PROMOTION, AND WHAT THE DIFF ACTUALLY LOOKED LIKE.
#
# APPENDED by Plan 33.1-06 Task 3 on 2026-09-13, AFTER the promotion and the
# cross-check. Nothing above this line was edited.
#
# THE BRACKET, in the order it was executed, ALL OF IT INSIDE ONE LOCK
# ACQUISITION separate from the run half's:
#
#   0. CorpusLock acquired -- pid 15016, 2026-09-14T01:48:44.604185+00:00 --
#      BEFORE the pre-state digest.
#   1. data      -> outputs/phase331_promote_before.json            (448 files)
#   2. artifacts -> outputs/phase331_promote_artifacts_before.json  (159 files)
#   3. promote_corpus_to_silver -- ONE upsert_silver over all 6,499 games, read
#      from the 24 bronze snapshots. ZERO network calls: 0 attempts, 0 weighted.
#   4. and 5. both trees re-digested, still inside the lock.
#
# THE DATA VERIFY REPORTED EXACTLY ONE CHANGED PATH AND IT IS THE DECLARED ONE.
# Zero added, zero removed, and no comparison left undecided (no MIXED key). The
# single rewritten path was silver/weather.parquet, which is exactly
# WEATHER_PROMOTION_EXPECTED_CHANGED_FILES as committed in 8cfc1cc BEFORE the run
# half's first snapshot. `artifacts/` did not move at all.
#
# `data/nfl_predictions.duckdb` did NOT move, which is the POSITIVE confirmation
# the declaration was making: promote_corpus_to_silver writes through
# upsert_silver (parquet only, ending at _atomic_write_parquet) and never through
# save_dataframe, whose save_to_db=True default would have moved the database
# half as well.
#
# 14 ROWS IN, 6,499 ROWS OUT. 23 -> 25 columns. 745 distinct temperatures
# spanning -9.0 F to 101.6 F, against a prior state of 6,485 gold rows at a
# single fabricated 65.0. 1,652 rows carry a NULL temperature and every one of
# them is an indoor game; every row is coverage-flagged True, because the archive
# answered for all 4,847 outdoor games and there is not one absent observation in
# the corpus.
# ---------------------------------------------------------------------------

WEATHER_CORPUS_DIGEST_BEFORE: str = (
    "3034b00b86ab95332cf1d385b736379b1b6796be2fa812acf3a4d1dcb9114d4d"
)
WEATHER_CORPUS_DIGEST_AFTER: str = (
    "53f11e8cee99a7f7cc6b6cb613319d23d9112eb05eed91244d346ad5a491543d"
)


# ---------------------------------------------------------------------------
# RULING X -- WHAT THE PROMOTION READ, AND WHAT IT PROVABLY DID NOT.
#
# "Silver holds 6,499 rows" is satisfied by the right bronze and by the wrong
# bronze. The seven legacy weather_raw_bronze_{2018..2024}_season.parquet files
# hold 1,942 rows at SEVENTEEN columns against this corpus's TWENTY-FIVE, so a
# promotion that accidentally globbed them would produce a silver frame mixing
# two schemas and two provenances -- and A ROW-COUNT MATCH CANNOT DETECT THAT.
# R2's acceptance is about provenance, not arithmetic.
#
# TWO INDEPENDENT INSTRUMENTS ON ONE PROPERTY: the filename set (asserted
# disjoint from LEGACY_WEATHER_BRONZE_INVENTORY's ten names, and the promotion
# RAISES if it is not) and the per-file column width (25 on every one of the 24,
# so a 17-column legacy file could not have contributed even if the filename
# check were wrong).
# ---------------------------------------------------------------------------

WEATHER_PROMOTION_INPUTS: dict[str, object] = {
    "measured_on": "2026-09-13",
    "measured_by": "Plan 33.1-06 Task 3",
    "bronze_tables_read": ("weather_backfill",),
    "bronze_glob": "weather_backfill_raw_bronze_*",
    "legacy_files_read": (),
    # The ten filenames the promotion is proven NOT to have read. Written out
    # rather than derived from LEGACY_WEATHER_BRONZE_INVENTORY with a
    # tuple(sorted(...)) call, because this module's contract is NO module-level
    # calls. tests/unit/test_weather_backfill_resume.py already asserts the two
    # lists agree, so the duplication cannot drift silently.
    "legacy_inventory_checked_against": (
        "weather_raw_bronze_2018_season.parquet",
        "weather_raw_bronze_2019_season.parquet",
        "weather_raw_bronze_2020_season.parquet",
        "weather_raw_bronze_2021_season.parquet",
        "weather_raw_bronze_2022_season.parquet",
        "weather_raw_bronze_2023_season.parquet",
        "weather_raw_bronze_2024_season.parquet",
        "weather_raw_bronze_2024_W06_20260416T165923.parquet",
        "weather_raw_bronze_2025_W00_20260407T025733.parquet",
        "weather_raw_bronze_2025_W05.parquet",
    ),
    "column_counts_read": (25,),
    "legacy_column_count": 17,
    "rows_promoted": 6499,
    "snapshots_promoted": 24,
    "network_calls": 0,
    "digest_instrument": (
        "sha256 of file bytes -- the same value tests.data_boundary.digest_file "
        "returns for a readable file, computed inside the script because "
        "production code does not import from the test tree"
    ),
    "bronze_files_read": {
        "weather_backfill_raw_bronze_2002_W00_20260914T002711.parquet": {
            "digest": (
                "eb17a7dcb77ebbd415fe69771e4e1e7a52e56d347b21614e918deefde2e4263d"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2003_W00_20260914T003043.parquet": {
            "digest": (
                "e3df310d1b3b8a246a23f2217d0b395fd4e6a71d0f55da3f5d9669d7323cc82e"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2004_W00_20260914T003418.parquet": {
            "digest": (
                "cb08f79520013d758bcd55533504437392f89e45f7a60ac06fb85b6ca7f1c8ee"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2005_W00_20260914T003757.parquet": {
            "digest": (
                "948761a16419b524e2fb668cf2671655a18798ac88c2619a19bdd5ce714f4a57"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2006_W00_20260914T004124.parquet": {
            "digest": (
                "f7d9054abdef0fcdfb19bd0173c7268f1bc8c67d40e67d12d776680e8c4bcf04"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2007_W00_20260914T004451.parquet": {
            "digest": (
                "a6f9cc0e4751cc161f9ff1aac79097474806ac06a82df86e1202de077c40742d"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2008_W00_20260914T004825.parquet": {
            "digest": (
                "ecb42e13dad1d74bbae8504f8fa3ca9aeba6b8fe3badaa2e43abea04de676e8e"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2009_W00_20260914T005143.parquet": {
            "digest": (
                "45acc7ca2e9ae0cc6e7b68501e84b40ef3794b8770e905cb965bbccdf75a5761"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2010_W00_20260914T005502.parquet": {
            "digest": (
                "543fd3cc463ade3d33b36539373958d9253818e98ce842b9a537dadb05693038"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2011_W00_20260914T005819.parquet": {
            "digest": (
                "88f6f367f7c488d740f779b0d1b21e5ed5343f84b6fb74dc0a256f088fa78d71"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2012_W00_20260914T010139.parquet": {
            "digest": (
                "382f38194862c5d58b8feb0d1f7761b4e181f9f229057333d335e90906c71848"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2013_W00_20260914T010456.parquet": {
            "digest": (
                "8367b5766c5f08956585d678ba67f1ad731ac8a6d8cb3ddc30e001db9b0a528e"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2014_W00_20260914T010819.parquet": {
            "digest": (
                "d7f00771af12e2b6e8f632e2c1b40a5b17e623557ec62ba307237f65ee467ad7"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2015_W00_20260914T011144.parquet": {
            "digest": (
                "dbe3aa01922548046192f84804b86419e5b9ffadb16f254241f4f6c2a0d1aee1"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2016_W00_20260914T011504.parquet": {
            "digest": (
                "111c292472d4eb79f28c4a0a7a4d9f3295fbd894234885c073d7c153badc7ac7"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2017_W00_20260914T011829.parquet": {
            "digest": (
                "1fa2968b9bd570281c67e94284d74711cbbba54854f73cf5fba46bdee2a32a3d"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2018_W00_20260914T012152.parquet": {
            "digest": (
                "bd4facc522c250f5e5f5e3c59aa308d7cd3240ba3b21b2cbc9432ca288446f3d"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2019_W00_20260914T012518.parquet": {
            "digest": (
                "362c6fac827bb8e81e70e1758c9107456ee127f41cbcc0971880b115d91d3982"
            ),
            "rows": 267,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2020_W00_20260914T012814.parquet": {
            "digest": (
                "74433d00da74b9835600bb171a37b6a2eb8dd09ade1b79c7d719cba80abf9de6"
            ),
            "rows": 269,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2021_W00_20260914T013136.parquet": {
            "digest": (
                "437a1cf317d9a60cb2fb9f4e8bd71a7c1f667c11414e18344d7d93bffdac0fb0"
            ),
            "rows": 285,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2022_W00_20260914T013455.parquet": {
            "digest": (
                "17fa3f788104f2de49eafaff2f81b6f704af9b45876610b5909758890784104a"
            ),
            "rows": 284,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2023_W00_20260914T013814.parquet": {
            "digest": (
                "ae477a53edd610f9c1510d8700cb1cd57eb37c94fc41667f1f1f2cc245331263"
            ),
            "rows": 285,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2024_W00_20260914T014122.parquet": {
            "digest": (
                "ba5c9ad3640586e1f3363ba7b7163f31989ed1622a0de612135c5aff66fd74c3"
            ),
            "rows": 285,
            "columns": 25,
        },
        "weather_backfill_raw_bronze_2025_W00_20260914T014435.parquet": {
            "digest": (
                "a7f1211fa32ede134eace24bad925d03d12521ab3325bd5683d649b331400c8b"
            ),
            "rows": 285,
            "columns": 25,
        },
    },
}


# ---------------------------------------------------------------------------
# THE MEASURED DIFF, JUDGED AGAINST A PREDICTION MADE BEFORE IT EXISTED.
#
# The prediction is scripts/weather_crosscheck_constants.EXPECTED_DIFF_SHAPE,
# committed ALONE in 57586c1 and witnessed from outside in 9e28d40, both strict
# ancestors of this measurement. THE PREDICTION WAS NOT EDITED, and it must not
# be: correcting a prediction so that it matches the answer turns a
# pre-registration into a description. Where the measurement disagrees with it,
# THE DISAGREEMENT IS THE FINDING and is recorded here as one.
#
# THE VERDICT IS **PARTIAL**. Two of the three predicted causes matched -- one of
# them exactly -- and the third matched in KIND but under-predicted its
# population and carried an exclusivity claim that the measurement FALSIFIES.
#
# 1. THE HOUR FIX -- MATCHED, and more strongly than predicted. The prediction
#    said "essentially every comparable row differs". ALL 1,942 of 1,942 do:
#    every single comparable row disagrees on at least one column. Temperature
#    fell on 65.3% of the 1,370 rows where both sides carry a reading, mean
#    signed -0.66 F, mean absolute 3.02 F. Wind speed fell by 3.74 mph on
#    average. The direction the prediction named is the direction observed.
#
# 2. THE ROUTING FIX -- MATCHED EXACTLY, on every clause. The prediction named
#    44 comparable games at OAK00, LAX99 and LAX97; the measurement finds exactly
#    44. It predicted "CATEGORICALLY DIFFERENT VALUES, not shifted ones" and the
#    largest differences in the diff: those 44 games average 15.61 F of absolute
#    temperature difference against 2.60 F for every other comparable game, six
#    times larger. It predicted the signs would NOT agree; they split 36 positive
#    to 8 negative. And the three largest per-home-team mean absolute differences
#    are LA (16.59), LAC (15.23) and LV (13.61) -- exactly the three relocated
#    franchises, ranked ahead of every other team by a factor of three.
#
# 3. THE PER-GAME ROOF RULE -- PARTIAL, and this is the finding. The prediction
#    was right that these rows go from a number to a NULL temperature, and right
#    about its own population: all 260 `closed` games in the comparable window
#    did go number-to-NULL, with none missing. But the measured population is
#    **572, not 260**. The extra 312 are DOME games.
#
#    WHY THE PREDICTION MISSED THEM, stated so the miss is understood rather than
#    explained away. The prediction reasoned from where the feed's per-game
#    `roof` DISAGREES with venues.json's `roof_type` about is_outdoor, and for a
#    dome the two agree -- so domes were outside the population it computed. What
#    it did not anticipate is that THE LEGACY CORPUS CARRIED A TEMPERATURE FOR
#    EVERY ONE OF ITS 1,942 ROWS, dome games included: the old code fetched and
#    stored a reading for all 312 dome games in the window while recording
#    is_outdoor False on 308 of them. Those rows go number-to-NULL now not
#    because the roof RULE changed for them but because the new code refuses to
#    store a measurement for a game weather does not apply to.
#
#    SO THE PREDICTION'S CLAUSE "this is the ONLY category where that happens" IS
#    FALSIFIED. There are two categories, not one. The measured number-to-NULL
#    set is EXACTLY the indoor population -- 260 closed plus 312 dome = 572 --
#    with no outdoor game in it and no indoor game outside it.
#
#    The direction of the finding happens to favour the phase: 572 meaningless
#    stored temperatures are gone rather than 260. That does not make the
#    prediction right, and it is recorded as a miss.
# ---------------------------------------------------------------------------

WEATHER_CROSSCHECK_MEASURED: dict[str, object] = {
    "measured_on": "2026-09-13",
    "measured_by": "Plan 33.1-06 Task 3",
    "command": "uv run python -m scripts.backfill_historical_weather --crosscheck",
    "process_exit_status": 0,
    "disagreement_failed_the_run": False,
    "preregistration_path": "scripts/weather_crosscheck_constants.py",
    "preregistration_commit": WEATHER_CROSSCHECK_PREREGISTRATION_COMMIT,
    "preregistration_was_edited": False,
    "tolerance_f": 0.1,
    "compared_columns": 15,
    # THE JOIN.
    "comparable_rows": 1942,
    "rows_new_only": 4557,
    "rows_legacy_only": 0,
    # ROW-LEVEL. The prediction is phrased about rows, so the judgement is too.
    "rows_differing": 1942,
    "rows_differing_share": 1.0,
    "rows_number_to_null": 572,
    "rows_number_to_null_any_column": 624,
    # COLUMN-LEVEL, as the shipped comparator reports it.
    "column_level_disagreements": 7812,
    "column_level_number_to_null": 2802,
    "column_level_null_to_number": 3160,
    # THE THREE PREDICTED CAUSES, measured.
    "hour_fix": {
        "verdict": "MATCHED",
        "mean_signed_temp_delta_f": -0.6642,
        "mean_abs_temp_delta_f": 3.0216,
        "temp_comparable_n": 1370,
        "share_of_rows_where_temperature_fell": 0.6533,
        "mean_signed_wind_delta_mph": -3.7351,
    },
    "routing_fix": {
        "verdict": "MATCHED EXACTLY",
        "predicted_comparable_window_size": 44,
        "measured_comparable_window_size": 44,
        "relocation_mean_abs_temp_delta_f": 15.6091,
        "everything_else_mean_abs_temp_delta_f": 2.6039,
        "signs": {"positive": 36, "negative": 8},
        "signs_note": (
            "the prediction said these rows would NOT share a sign, because a "
            "different place is not a warmer version of the same one. They do "
            "not: 36 positive against 8 negative."
        ),
    },
    "per_game_roof_rule": {
        "verdict": "PARTIAL -- right in kind, population under-predicted, "
        "exclusivity claim FALSIFIED",
        "predicted_comparable_window_size": 260,
        "measured_number_to_null_rows": 572,
        "predicted_population_fully_accounted_for": True,
        "closed_games_in_window": 260,
        "dome_games_in_window": 312,
        "open_games_in_window": 41,
        "outdoor_games_in_window": 1329,
        "number_to_null_set_equals_the_indoor_population": True,
        "legacy_rows_carrying_a_temperature": 1942,
        "legacy_dome_rows_carrying_a_temperature": 312,
        "falsified_clause": (
            "THESE ROWS GO FROM A NUMBER TO A NULL TEMPERATURE, and this is the "
            "ONLY category where that happens"
        ),
        "falsified_because": (
            "dome games are a SECOND category. The legacy corpus stored a "
            "temperature for all 312 dome games in the comparable window while "
            "recording is_outdoor False on 308 of them, so they too go "
            "number-to-NULL under the new rule -- not because their roof "
            "classification changed, but because the new code refuses to store a "
            "measurement for a game weather does not apply to."
        ),
    },
    # THE BREAKDOWNS THE ACCEPTANCE REQUIRES.
    "per_season": {
        2018: {"rows": 267, "agreed": 1436, "disagreed": 1143, "became_null": 351},
        2019: {"rows": 267, "agreed": 1431, "disagreed": 1154, "became_null": 330},
        2020: {"rows": 269, "agreed": 1483, "disagreed": 1028, "became_null": 429},
        2021: {"rows": 285, "agreed": 1589, "disagreed": 1116, "became_null": 405},
        2022: {"rows": 284, "agreed": 1573, "disagreed": 1115, "became_null": 417},
        2023: {"rows": 285, "agreed": 1549, "disagreed": 1147, "became_null": 414},
        2024: {"rows": 285, "agreed": 1555, "disagreed": 1109, "became_null": 456},
    },
    "ten_largest_per_home_team_mean_abs_temp_delta_f": (
        ("LA", 16.588, 17),
        ("LAC", 15.225, 16),
        ("LV", 13.606, 16),
        ("NO", 9.5, 2),
        ("DEN", 5.021, 58),
        ("MIN", 3.9, 1),
        ("JAX", 3.703, 59),
        ("TB", 3.317, 64),
        ("TEN", 3.118, 60),
        ("BAL", 3.006, 63),
    ),
    "per_home_team_note": (
        "(team, mean absolute temp delta in F, comparable rows). The full "
        "per-home-team breakdown over all 32 teams is printed by --crosscheck; "
        "the ten largest are recorded here. The top three are exactly the three "
        "relocated franchises."
    ),
    # THE VERDICT.
    "matched_expectation": "PARTIAL",
    "matched_expectation_reason": (
        "Two of the three pre-registered causes MATCHED and the third did not. "
        "The HOUR FIX matched and exceeded its prediction: all 1,942 of 1,942 "
        "comparable rows differ, temperature fell on 65.3 percent of the 1,370 "
        "rows where both sides carry a reading, and the mean signed shift is "
        "-0.66 F, the direction predicted. The ROUTING FIX matched EXACTLY on "
        "every clause: 44 comparable games as predicted, averaging 15.61 F of "
        "absolute difference against 2.60 F elsewhere, signs split 36 to 8 "
        "rather than shared, and the three largest per-home-team differences are "
        "LA, LAC and LV. The PER-GAME ROOF RULE was right in kind and about its "
        "own 260 closed games, every one of which went number-to-NULL, but the "
        "measured population is 572 rather than 260 and its clause 'this is the "
        "ONLY category where that happens' is FALSIFIED: 312 dome games are a "
        "second category, because the legacy corpus stored a temperature for "
        "them too. The pre-registration was NOT edited to accommodate this. The "
        "disagreement is the finding."
    ),
    # THE FOURTH REPORTED QUANTITY (pre-registered in Plan 33.1-05 Task 3).
    "archive_versus_forecast_probe": {
        "n": 20,
        "n_is_temperature_pairs": True,
        "n_wind": 28,
        "population_joined_on_game_id": 28,
        "archive_minus_forecast_temp_f_mean_signed": 0.87,
        "archive_minus_forecast_temp_f_sd": 3.3452,
        "archive_minus_forecast_wind_mph_mean_signed": -2.4714,
        "archive_minus_forecast_wind_mph_sd": 3.671,
        "per_half": {
            "2024_W06_silver_pre_promotion": {
                "rows_in_half": 14,
                "joined_on_game_id": 14,
                "temp_n": 11,
                "temp_mean_signed": -0.7091,
                "temp_sd": 2.3518,
                "wind_n": 14,
                "wind_mean_signed": -1.0,
                "wind_sd": 2.5926,
            },
            "2025_W05_duckdb_weather_forecast": {
                "rows_in_half": 14,
                "joined_on_game_id": 14,
                "temp_n": 9,
                "temp_mean_signed": 2.8,
                "temp_sd": 3.4706,
                "wind_n": 14,
                "wind_mean_signed": -3.9429,
                "wind_sd": 4.0748,
            },
        },
        "halves_disagree_in_sign_on_temperature": True,
        "excluded_population": (
            "the 1,942 legacy 2018-2024 bronze rows are EXCLUDED. They carry no "
            "weather_source and RESEARCH established that their provenance "
            "cannot be reconstructed, so including them would answer a DIFFERENT "
            "question while looking like this one."
        ),
        "interpretation_rule": (
            "WITH n AT MOST 28 THIS IS A DIRECTIONAL PROBE AND NOT AN ESTIMATE. "
            "No bias correction, no feature change and no Wave-15 instruction "
            "may rest on it. Its ONLY legitimate use is to say whether the "
            "question deserves its own measurement later, on a population built "
            "for it. A phase that measured 28 games and then acted on the sign "
            "would be doing exactly what this repository's pre-registration "
            "discipline exists to prevent. This rule is registered BEFORE any "
            "number exists so it cannot be chosen afterwards."
        ),
        "nothing_rests_on_it": True,
        # THE PROVENANCE CONTRADICTION, recorded rather than resolved.
        "provenance_finding": (
            "THE TWO HALVES OF THE PRE-REGISTERED POPULATION ARE NOT EQUALLY "
            "WELL-FOUNDED, and the pre-registration and this repository's own "
            "records DISAGREE about one of them. The probe spec calls the 14 "
            "2024_W06 rows in data/silver/weather.parquet rows 'whose provenance "
            "is KNOWN to be the forecast feed'. All 14 of them read "
            "weather_source = 'archive', stamped by Plan 33-09, whose own "
            "recorded reasoning is that 'every pre-existing row was produced by "
            "the archive endpoint ... and that ingest hit ARCHIVE_ENDPOINT_URL'. "
            "Both statements are in committed source and they cannot both be "
            "right. The 14 2025_W05 rows in the DuckDB weather_forecast table "
            "are NOT in dispute: their forecast_time is 2025-09-29, before those "
            "games were played, and they sit in a table named for the forecast "
            "feed. The registered n=20 figure is reported as registered; the "
            "per-half split is reported beside it so a later reader can see that "
            "the uncontested half alone gives +2.80 F on n=9 while the contested "
            "half gives -0.71 F on n=11 -- opposite signs. NOTHING is concluded "
            "from either. The pre-registration was NOT edited."
        ),
        "measured_by": "Plan 33.1-06 Task 3",
    },
}


# ---------------------------------------------------------------------------
# QUICK TASK 260913-w8x -- the measured after-state of two test-suite fixes.
#
# MEASURED 2026-09-14 on commit bc8bbec, in ONE process:
#
#     uv run python -m pytest tests -q
#
# The two fixes, both landed before this measurement was taken:
#
# QT-W8X-01 -- data/storage.DuckDBConnection.connect opens the database
# READ-ONLY unless a caller passes write=True. It previously connected with no
# read_only flag (read-WRITE) and held that handle as a module global for the
# whole of any process that touched load_dataframe, so a purely reading pytest
# session locked data/nfl_predictions.duckdb against every other opener,
# including a plain byte read of the file.
#
# QT-W8X-02 -- tests/conftest.production_store_write_guard moved from FUNCTION
# scope to MODULE scope. Attribution degrades from "which test wrote it" to
# "which test FILE wrote it"; the owner accepted that trade explicitly. The
# exemption is still a union of DECLARED PATHS, the session-end full content
# sweep is untouched, and the Phase 33 Wave 6 shape -- an unmarked test
# overwriting all three production gold matrices -- still fails the session,
# naming all three files and naming the module that wrote them.
#
# THE BASELINE THIS IS COMPARED AGAINST IS OLDER THAN THE TREE, AND THAT IS THE
# FIRST FINDING. The 994.40 s / 4852-passed line in the URGENT-fix-test-suite
# todo was measured on commit 5cfe621 on 2026-09-12. Phase 33.1 waves 5-6 and
# Plan 33-09 landed between then and this task, adding 247 collected tests.
# Every number below is reconciled against that intervening delta rather than
# against the stale line alone.
# ---------------------------------------------------------------------------

W8X_MEASURED_ON: str = "2026-09-14"
W8X_MEASURED_AT_COMMIT: str = "bc8bbec"
W8X_COMMAND: str = "uv run python -m pytest tests -q"

# The whole-suite line VERBATIM.
W8X_POST_TASK_FAILURE_SET: str = (
    "5 failed, 5118 passed, 12 skipped, 9 xfailed, 2913 warnings in 1381.52s"
)

# 5 + 5118 + 12 + 9. Independently confirmed by a --collect-only run reporting
# "5144 tests collected", so the summary line and the collection agree.
W8X_POST_TASK_COLLECTED: int = 5144

W8X_POST_TASK_TOTAL_SECONDS: float = 1381.52

# The failure set was compared MEMBER BY MEMBER against
# DELIBERATE_TRIPWIRE_NODE_IDS: both set differences are empty. Exactly the five
# deliberate tripwires, no sixth failure and no error.
W8X_FAILURE_SET_IS_EXACTLY_THE_TRIPWIRES: bool = True

# THE PASSED-COUNT ARITHMETIC, closed to the unit.
#
#   4852 (baseline, commit 5cfe621) + 246 (intervening) + 20 (this task) = 5118
#     11 (baseline skipped)         +   1 (intervening)                  =   12
#      5 failed and 9 xfailed are UNCHANGED.
#   4877 (baseline collected)       + 247 (intervening) + 20 (this task) = 5144
#
# The intervening +247 collected is +206 test FUNCTIONS across 16 modules (13 of
# them new) plus +41 from parametrised expansion in those same modules, measured
# by an AST census of every tests test module at 5cfe621, at 94769f2 and at
# HEAD. One of the new modules skips on this checkout, which is the +1 skip, so
# the intervening contribution to PASSED is 246 rather than 247.
W8X_NET_NEW_TESTS: int = 20
W8X_NET_NEW_TESTS_BY_MODULE: tuple[tuple[str, int], ...] = (
    # A new module: the read-only connection-mode proof.
    ("tests/unit/test_storage_connection_mode.py", 11),
    # TestTheGuardIsScopedToTheModule.
    ("tests/unit/test_write_guard_marker_scope.py", 5),
    # The generated mini-suite re-authored from one child module into five: the
    # module went from 8 collected tests to 12.
    ("tests/integration/test_data_boundary_guard_arming.py", 4),
)
W8X_INTERVENING_TESTS_NOT_FROM_THIS_TASK: int = 247
W8X_INTERVENING_SKIP: str = (
    "tests/integration/test_weather_corpus_coverage.py::"
    "TestTheThreeCoverageStatesStayApart::"
    "test_an_absent_observation_is_outdoor_covered_false_and_entirely_null"
)

# THE WALL CLOCK, AND WHY IT DOES NOT ANSWER THE QUESTION IT LOOKS LIKE IT
# ANSWERS. 1381.52 s against the 994.40 s baseline is +387.12 s, and the ~179 s
# saving this task predicted is NOT visible in it. The comparison is CONFOUNDED:
# the tree gained 247 tests between the two runs, among them the Phase-33.1
# weather rebuild and crosscheck integration modules, which are not cheap. This
# run therefore NEITHER SUPPORTS NOR REFUTES the ~179 s hypothesis. It is
# recorded as measured and is not spun.
W8X_WALL_CLOCK_DELTA_SECONDS: float = 387.12
W8X_WALL_CLOCK_COMPARISON_IS_CONFOUNDED: bool = True

# WHAT DOES ANSWER IT: an A/B run at THIS commit that changes ONE variable.
# 300 identical trivial tests across 10 modules, over the LIVE production roots,
# two repetitions per arm. The function-scope arm re-declares the pre-change
# guard verbatim over the same helpers; nothing in the repository was modified
# to take the measurement.
#
#   module scope   2.26 s, 2.24 s
#   function scope 12.20 s, 12.09 s
#
# 9.895 s over 290 additional sweeps = 34.1 ms per sweep, which independently
# reproduces the 36.7 ms the quick task RESEARCH measured a different way.
W8X_GUARD_AB_MODULE_SCOPE_SECONDS: tuple[float, ...] = (2.26, 2.24)
W8X_GUARD_AB_FUNCTION_SCOPE_SECONDS: tuple[float, ...] = (12.20, 12.09)
W8X_GUARD_PER_SWEEP_MILLISECONDS: float = 34.1

# MEASURED per-sweep cost times a COUNTED number of sweeps avoided (5144
# collected tests minus roughly 265 test modules). A projection, labelled as
# one, and never a measured whole-suite delta.
W8X_GUARD_PROJECTED_SUITE_SAVING_SECONDS: float = 166.5

# THE ATTRIBUTION TRADE, stated plainly.
W8X_ATTRIBUTION_TRADE: str = (
    "Per-TEST attribution was exchanged for the per-sweep cost above. The guard "
    "now names the FILE that wrote a production store, not the test inside it, "
    "and within a module that declares a path a DIFFERENT unmarked test writing "
    "that SAME path is permitted. The recovery is stated in the guard's own "
    "violation message: re-run that one file on its own and the window narrows "
    "to the tests in it. THE GUARANTEE IS UNCHANGED -- the exemption is still a "
    "union of declared PATHS and never a licence for the file, the session-end "
    "full content sweep is untouched and still content-based, and the Phase 33 "
    "Wave 6 shape still fails the session naming all three gold matrices."
)

# ---------------------------------------------------------------------------
# What the guard OBSERVED about its own instrument after QT-W8X-01.
#
#     NFL_GUARD_OBSERVATIONS=1 uv run python -m pytest tests/integration -q
#     5 failed, 850 passed, 9 skipped, 9 xfailed in 1063.48s
#
# THERE IS NO BEFORE-FIGURE FOR THIS TIER ON RECORD. Plan 33-01 Task 3(d)
# measured a different selection, so this is a first observation and NOT a
# comparison; nothing here should be read as a delta.
#
# STAT_SIGNATURE_OBSERVATIONS=0 means the locked-file read path did not fire
# ONCE across an armed integration tier. That is the QT-W8X-01 claim seen from
# the guard's side: with the production database opened read-only, its bytes
# stay readable, so no verdict in this tier had to reach for the weaker stat
# signature or refuse.
# ---------------------------------------------------------------------------

W8X_INTEGRATION_TIER_LINE: str = "5 failed, 850 passed, 9 skipped, 9 xfailed"
W8X_INTEGRATION_TIER_SECONDS: float = 1063.48
W8X_STAT_SIGNATURE_OBSERVATIONS: int = 0
W8X_WAL_SIBLING_OBSERVATIONS: int = 0


# ---------------------------------------------------------------------------
# THE ATS EDGE'S UNIT, AND THE BAND DISTRIBUTION IT PRODUCED BEFORE THE REPAIR.
#
# APPENDED by Plan 33-10 Task 1 on 2026-09-14. Nothing above this line was edited.
#
# R13 / D33-05 redefine ``ats_edge`` on a FIXED POINTS SCALE. What it was before is
# the reason the numbers below exist: ``(ats_prediction - market_spread) /
# |market_spread|``, a ratio whose denominator is a point count that approaches
# zero, banded by ``utils.edge_tier`` against the SAME 0.05 / 0.02 threshold pair
# that bands a WP probability. The unit is recorded here because Plan 33-16 freezes
# per-target thresholds derived from THIS edge's own distribution, and a threshold
# without a stated unit is a number nobody can check.
# ---------------------------------------------------------------------------

ATS_EDGE_SCALE: str = "points"  # signed model-minus-market home-margin difference

# The MEASURED pre-repair band distribution, RE-DERIVED by Plan 33-10 Task 1 on
# 2026-09-14 rather than copied from the plan text, read-only, over
# ``outputs/backtest/predictions_all.csv`` (1,139 games, seasons 2021-2024 -- the
# whole file, which carries no 2026 rows, so "the pre-2026 population" and "the file"
# are the same population here).
#
# THE DENOMINATOR IS NOT THE ROW COUNT. 52 of the 1,139 games carry no market line
# at all, so all three edges are NULL on them and ``edge_tier`` answers "low" for an
# absent edge. Banding those 52 as "low" would attribute a real band to a game that
# has no edge. The shares below are therefore over the 1,087 rows with a COMPUTABLE
# edge, and the row count is recorded beside them so the other denominator is always
# recoverable.
#
# RECONCILIATION AGAINST THE CARRIED FIGURES. Plan 33-10's own text carried
# wp 0.268/0.300/0.432, ats 0.028/0.046/0.926, ou 0.220/0.293/0.487. The re-derivation
# REPRODUCES all nine to the precision they were quoted at, under the 1,087-row
# denominator. Under the 1,139-row denominator they do NOT reproduce
# (ats 0.0720/0.0439/0.8841), which is how the denominator was identified. Both are
# recorded; neither overwrites the other.
ATS_BAND_SHARES_BEFORE: dict[str, dict[str, float]] = {
    "wp": {"low": 0.2677, "medium": 0.2999, "high": 0.4324},
    "ats": {"low": 0.0276, "medium": 0.0460, "high": 0.9264},
    "ou": {"low": 0.2199, "medium": 0.2935, "high": 0.4867},
}

# Rows with a computable edge -- the denominator of every share above.
ATS_BAND_SHARES_BEFORE_DENOMINATOR: int = 1087

# The same measurement over ALL rows, absent edges banded "low" by ``edge_tier``.
# Recorded so the two instruments stay distinguishable and Plan 33-16's "after" half
# can be taken with whichever one it declares.
ATS_BAND_SHARES_BEFORE_ALL_ROWS: dict[str, dict[str, float]] = {
    "wp": {"low": 0.3011, "medium": 0.2862, "high": 0.4126},
    "ats": {"low": 0.0720, "medium": 0.0439, "high": 0.8841},
    "ou": {"low": 0.2555, "medium": 0.2801, "high": 0.4644},
}
ATS_BAND_SHARES_BEFORE_ALL_ROWS_DENOMINATOR: int = 1139

# R13's headline case CANNOT BE FOUND IN THE DATA, measured on the same pass: zero of
# the 1,139 rows carry ``|market_spread| <= 0.5``. That is why the half-point proof is
# a CONSTRUCTED row (Plan 33-02's ``build_half_point_ats_frame``) and not a real one.
ATS_HALF_POINT_ROWS_IN_POPULATION: int = 0


# ---------------------------------------------------------------------------
# THE CROSS-COPY ATS EDGE PARITY TABLE.
#
# APPENDED by Plan 33-10 Task 2 on 2026-09-14. Nothing above this line was edited.
#
# ONE formula, TWO homes: ``api/cache.py``'s vectorized expression (feeding the
# ``predictions`` table, ``/`` and ``/betting``) and
# ``scripts/generate_current_week_predictions.compute_edges`` (feeding the
# current-week CSV). The cache derives its own edge from
# ``outputs/backtest/predictions_all.csv`` and NEVER reads the current-week file, so
# nothing in the running system would notice if one were repaired and the other left
# alone -- which is exactly what happened once already under DEF-31-03, and why this
# table exists. It lives HERE, committed once, so both test modules assert the same
# rows rather than each carrying a copy that can drift.
#
# Each entry is ``(market_spread, ats_prediction, expected_edge)`` in POINTS. The
# second copy spells the first field ``spread``; the value is the same quantity.
# ``None`` in the first field means NO STORED LINE and ``None`` in the third means NO
# EDGE -- the one surviving special branch, and it must be compared with explicit
# NaN handling rather than by an equality that silently skips it.
#
# THE PICK-EM ROW WITH A NON-ZERO PREDICTION IS THE POINT OF THE TABLE (D33-31). A
# pick-em whose prediction is also zero agrees under BOTH the repaired arithmetic and
# the old forced-zero branch, so a table carrying only that row would prove nothing
# about the ruling. The row below predicts the away side by four against a zero line
# and expects -4.0.
ATS_EDGE_PARITY_CASES: tuple[tuple[float | None, float, float | None], ...] = (
    (8.5, 12.0, 3.5),  # positive spread, model more bullish on home
    (-6.0, -2.5, 3.5),  # negative spread (road favourite), model less bearish
    (0.5, 2.5, 2.0),  # half-point line: the ratio scored 4.0 here
    (-0.5, -3.5, -3.0),  # half-point line, mirrored sign: the ratio scored -6.0
    (0.0, -4.0, -4.0),  # PICK-EM with a NON-ZERO prediction -- the D33-31 case
    (0.0, 0.0, 0.0),  # pick-em with a zero prediction: zero by arithmetic, not branch
    (None, 1.5, None),  # no stored line: NO edge, distinct from an edge of zero
    (-13.5, -7.0, 6.5),  # large favourite
    (11.0, 14.5, 3.5),  # large underdog
)


# ---------------------------------------------------------------------------
# PLAN 33-09'S COLLECTED-NODE COUNT, MEASURED AND FILLED BY PLAN 33-10.
#
# APPENDED by Plan 33-10 at plan close on 2026-09-14. Nothing above this line was
# edited. The slot is APPENDED, never inserted, so this block sits below
# TESTS_ADDED_33_08's even though its number is Plan 33-09's.
#
# WHY ANOTHER PLAN IS WRITING THIS SLOT. Plan 33-09 closed WITHOUT it, deliberately,
# recording in its SUMMARY that the protocol's prescribed measurement -- SUBTRACT THE
# PREVIOUS THREE-TIER COLLECTED TOTAL FROM YOUR OWN -- was "not computable at this
# plan's close", because Phase 33.1's waves 1-6 and the owner-directed test-
# infrastructure commits landed BETWEEN Plan 33-08's recorded 4,592 and Plan 33-09's
# own closing commits. That reasoning was right about the instrument it had. Leaving
# the hole was the honest choice at the time.
#
# The hole then BLOCKED Plan 33-10: appending TESTS_ADDED_33_10 over a gap turns
# tests/unit/test_phase33_state_arithmetic.py::
# test_the_slots_that_exist_so_far_are_a_contiguous_prefix RED. That test exists to
# force exactly this discovery -- its own docstring says a hole is "the term that
# would later be missing from the closing sum, discovered at phase close instead of
# at the plan that dropped it". It did its job one plan later, which is the earliest
# it could.
#
# WHAT MADE IT COMPUTABLE AFTER ALL. Plan 33-09's seven commits are not scattered:
# `git log --oneline 1ab8562..ed42df3` shows them in exactly TWO CONTIGUOUS BLOCKS
# with no foreign commit inside either.
#
#     BLOCK A  1ab8562 -> bc31982   bd326e1, d5cb966, 2333a53, 8c8ce05, bc31982
#     BLOCK B  40488ee -> ed42df3   d4466ab, ed42df3
#
# So the delta across each block IS Plan 33-09's contribution, with nothing else
# inside it to subtract. Measured on 2026-09-14 by whole-suite `--collect-only` in
# the MAIN TREE at each boundary -- collection runs no test, takes about five
# seconds, and counts the protocol's own unit (COLLECTED NODES, a parametrised case
# counting once per case):
#
#     uv run python -m pytest tests -q --collect-only
#         1ab8562   4592 collected     (Plan 33-09's base)
#         bc31982   4664 collected     BLOCK A = +72
#         40488ee   5144 collected     (Plan 33-09's Task-4 gate; 33.1 + w8x in between)
#         ed42df3   5164 collected     BLOCK B = +20
#         BLOCK A plus BLOCK B          92 nodes, this plan's total
#
# THREE INDEPENDENT CORROBORATIONS, none of them arranged by this plan:
#   * 4592 at 1ab8562 EQUALS Plan 33-08's recorded three-tier collected total to the
#     unit. A single-process `--collect-only` and the three-tier sum are therefore the
#     SAME INSTRUMENT for this unit, which is what licenses the comparison at all.
#   * 5144 at 40488ee EQUALS W8X_POST_TASK_COLLECTED, recorded above on 2026-09-14
#     from a different run by a different task.
#   * BLOCK B's +20 EQUALS the 20 nodes Plan 33-09's own SUMMARY attributes to
#     tests/unit/test_weather_gold_default_2026.py, the only test file Block B touches.
#
# AND IT CLEARS PLAN 33-09'S OWN FLOOR. That SUMMARY records a 74-node new-module
# subtotal and LABELS IT A FLOOR, because two modules it also extended were not
# separable from the 33.1 edits that followed. 92 >= 74, and the 18-node difference
# is those extensions plus tests/unit/test_audit_trace_epa_weather.py and
# tests/unit/test_quality_gates.py, which Block A also touched.
#
# A MEASUREMENT, NOT A RECONSTRUCTION. Nothing here is derived by counting test
# functions, which is the method Plan 33-09 rightly refused.
# ---------------------------------------------------------------------------

TESTS_ADDED_33_09: int = 92


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-10 at plan close on 2026-09-14, AFTER both tasks landed.
# Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 2, for the reason Plans 33-02 through 33-08 each
# recorded when they did the same: two tasks add tests, so a mid-plan value would
# have been wrong at close and could only have been made right by EDITING it -- the
# append-once violation the protocol exists to prevent.
#
# MEASURED the same way as the block above, by a whole-suite `--collect-only` BRACKET
# around this plan's own commits, in the main tree:
#
#     uv run python -m pytest tests -q --collect-only
#         ed42df3 (Plan 33-09's close, this plan's base)   5164 collected
#         this plan's close                                5191 collected
#         delta                                              27
#
# INDEPENDENTLY CONFIRMED per module, which is the check that no test was quietly
# deleted elsewhere to make the number look right. Both modules were EXTENDED, not
# created; neither is new:
#      8 -> 22   tests/unit/test_cache_ats_edge.py          (+14, Task 1)
#      7 -> 20   tests/unit/test_current_week_ats_edge.py   (+13, Task 2)
#     --
#     +27
#
# The two instruments agree exactly. No whole-suite RUN was taken (see this plan's
# SUMMARY for what was run and the residual risk); a collected-node count does not
# need one, and the standing five DELIBERATE_TRIPWIRE_NODE_IDS are untouched by both
# modules.
# ---------------------------------------------------------------------------

TESTS_ADDED_33_10: int = 27


# ---------------------------------------------------------------------------
# THE BYE-WEEK ROLLING-WINDOW AGEING FIXTURE.
#
# APPENDED by Plan 33-11 Task 1 on 2026-09-14. Nothing above this line was edited.
#
# WHAT IS PINNED HERE AND WHY. R12 claims a bye team's rolling window AGES rather
# than RESETS. The claim is checked on the real captured 2026 season, so the week,
# the teams, the control and the exact selected game_ids are committed constants
# rather than values a test re-derives from whatever the schedule happens to say on
# the day it runs. A test that derives its own expectation cannot disagree with the
# data; it can only disagree with itself.
#
# THE WEEK PAIR IS (11, 12) AND NOT (10, 11) -- A MEASURED CORRECTION.
# Plan 33-11's own prose asked for week 10 against week 11. That pairing does NOT
# express the claim, and the difference is not cosmetic.
# features/team_form._select_dynamic_window selects, for target week W, the games
# played in weeks 1..W-1 (plus a prior-season tail). A team whose bye is week 11
# played every week from 1 to 10, so:
#
#     window(10) = weeks 1..9    -> 9 games
#     window(11) = weeks 1..10   -> 10 games     <- the bye has not happened yet
#     window(12) = weeks 1..10   -> 10 games     <- the bye is now inside the window
#
# The bye first shows up in the window selected for week 12. Comparing 10 against 11
# measures an ordinary week of football -- the count grows by one, as it should --
# and would have been recorded as a proof of R12. Comparing 11 against 12 measures
# the bye: the selected set is identical while the target week advances.
#
# MEASURED 2026-09-14: across the pair (11, 12) all six bye teams move 10 -> 10
# (delta 0) and all twenty-six other teams move by exactly +1. The discrimination is
# total; no team is ambiguous.
BYE_WEEK_FIXTURE: int = 11

# The "after" half of the pair -- the first target week whose window holds the bye.
#
# IT HAPPENS TO EQUAL ZERO_BYE_WEEK BELOW, AND THAT IS A CALENDAR ACCIDENT, NOT A
# DESIGN. 2026 puts its byes in weeks 5-11 and 13-14, so the week after the richest
# bye week is also the one week in that range with no byes at all. The two constants
# are kept separate because they play different roles: this one is the second half of
# an ageing comparison, and the one below is a vacuity control.
BYE_WEEK_FIXTURE_AFTER: int = 12

# WEEK 12 HAS ZERO BYES AND THAT IS NOT AN ERROR (NF-11). A test parameterising over
# weeks 5-14 hits week 12 and finds nothing; an empty bye set there is a MEASURED
# fact about the 2026 calendar, recorded by tests/fixtures/season_2026.py as an
# explicit empty frozenset rather than as an absent key.
ZERO_BYE_WEEK: int = 12

# THE CONTROL TEAM, CHOSEN BY MEASUREMENT AND NOT BY EYE.
#
# The selection rule, applied to the captured 2026 schedule on 2026-09-14 and stated
# so that it can be re-run: the alphabetically first team that
#   (a) PLAYS in both week 11 and week 12 -- so it is the same week pair, and the
#       only difference between it and a bye team is the bye;
#   (b) has NOT yet had its bye by week 11 -- so its week-11 window count is 10,
#       IDENTICAL to all six bye teams. A control whose "before" count differs from
#       the subject's leaves a reader unable to say which difference did the work;
#   (c) is PAST THE EARLY-SEASON BOOTSTRAP REGIME, meaning its current-season game
#       count already exceeds max_prior_games (8), so its window draws ZERO
#       prior-season games and its stability is structural rather than lucky.
#
# Six teams satisfy all three: ARI, BAL, DAL, IND, LV and NYJ -- every team whose
# 2026 bye falls in week 13 or 14. ARI is the alphabetical first.
#
# ARI'S MEASURED NUMBERS: 10 games in the week-11 window, 11 in the week-12 window,
# gaining EXACTLY ONE -- 2026_11_ARI_KC -- and losing none. Its earliest selected
# game is week 1 at both target weeks, so it does not reset either. It is the same
# schedule shape as a bye team in every respect except that it played.
BYE_CONTROL_TEAM: str = "ARI"

# The single game ARI's window gains between week 11 and week 12. Pinned because
# "shifts by exactly one game" is a claim about WHICH game, not only about how many.
BYE_CONTROL_TEAM_GAME_GAINED: str = "2026_11_ARI_KC"

# THE EXACT SELECTED game_id SETS, MEASURED 2026-09-14 by driving
# features/team_form.TeamFormCalculator._select_dynamic_window over the captured
# 2026 schedule.
#
# THESE SETS ARE THE WEEK-11 WINDOW **AND** THE WEEK-12 WINDOW. That one recording
# serves both is not a shortcut -- it IS the finding. The window did not change
# across the bye; only the week did.
#
# ROW IDENTITIES, NOT COUNTS. A window holding ten games satisfies a count assertion
# no matter which ten it holds. These sets are what make the assertion say something.
BYE_WEEK_EXPECTED_GAME_IDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "ATL",
        (
            "2026_01_ATL_PIT",
            "2026_02_CAR_ATL",
            "2026_03_ATL_GB",
            "2026_04_ATL_NO",
            "2026_05_BAL_ATL",
            "2026_06_CHI_ATL",
            "2026_07_SF_ATL",
            "2026_08_ATL_TB",
            "2026_09_CIN_ATL",
            "2026_10_KC_ATL",
        ),
    ),
    (
        "CLE",
        (
            "2026_01_CLE_JAX",
            "2026_02_CLE_TB",
            "2026_03_CAR_CLE",
            "2026_04_PIT_CLE",
            "2026_05_CLE_NYJ",
            "2026_06_BAL_CLE",
            "2026_07_CLE_TEN",
            "2026_08_CLE_PIT",
            "2026_09_CLE_NO",
            "2026_10_HOU_CLE",
        ),
    ),
    (
        "GB",
        (
            "2026_01_GB_MIN",
            "2026_02_GB_NYJ",
            "2026_03_ATL_GB",
            "2026_04_GB_TB",
            "2026_05_CHI_GB",
            "2026_06_DAL_GB",
            "2026_07_GB_DET",
            "2026_08_CAR_GB",
            "2026_09_GB_NE",
            "2026_10_MIN_GB",
        ),
    ),
    (
        "LA",
        (
            "2026_01_SF_LA",
            "2026_02_NYG_LA",
            "2026_03_LA_DEN",
            "2026_04_LA_PHI",
            "2026_05_BUF_LA",
            "2026_06_ARI_LA",
            "2026_07_LA_LV",
            "2026_08_LAC_LA",
            "2026_09_LA_WAS",
            "2026_10_LA_ARI",
        ),
    ),
    (
        "NE",
        (
            "2026_01_NE_SEA",
            "2026_02_PIT_NE",
            "2026_03_NE_JAX",
            "2026_04_NE_BUF",
            "2026_05_LV_NE",
            "2026_06_NYJ_NE",
            "2026_07_NE_CHI",
            "2026_08_NE_MIA",
            "2026_09_GB_NE",
            "2026_10_NE_DET",
        ),
    ),
    (
        "SEA",
        (
            "2026_01_NE_SEA",
            "2026_02_SEA_ARI",
            "2026_03_SEA_WAS",
            "2026_04_LAC_SEA",
            "2026_05_SF_SEA",
            "2026_06_SEA_DEN",
            "2026_07_KC_SEA",
            "2026_08_CHI_SEA",
            "2026_09_ARI_SEA",
            "2026_10_SEA_LV",
        ),
    ),
)

# The SPAN either side of the pair, in weeks, MEASURED on the same pass. All six bye
# teams played in week 1, so all six read the same pair. Recorded as numbers because
# "the span grows" is checkable only against the values it grew between.
BYE_WEEK_SPAN_BEFORE: int = 10
BYE_WEEK_SPAN_AFTER: int = 11

# THE D33-18 OBSERVATION, RECORDED RATHER THAN MERELY ASSERTED -- AND CORRECTED.
#
# D33-18 asked that a finding be carried into the readout: "a bye team's window pulls
# in MORE prior-season data than its peers that week ... a second bootstrap-regime
# effect stacked on weeks 2-4". MEASURED, the finding is REAL but its window of
# applicability is NARROWER than the decision assumed, and at this plan's own fixture
# week the effect is exactly ZERO. Carrying the decision forward unchecked would have
# put a wrong number in the readout.
BYE_WINDOW_SPAN_OBSERVATION: str = """\
MEASURED 2026-09-14, Plan 33-11 Task 1, driving
features/team_form.TeamFormCalculator._select_dynamic_window (max_prior_games = 8)
over the captured 2026 schedule with the real 2025 season attached as the prior year.

THE EXTRA PRIOR-SEASON PULL IS EXACTLY ONE GAME, AND ONLY IN THE BOOTSTRAP REGIME.
The selector sets prior_count = max(0, 8 - current_season_count). A team that has
already had its bye carries one fewer current-season game than a peer that has not,
so it draws exactly one more prior-season game -- never two, and never a fraction:

    target week   6     7     8     9    10    11    12
    bye teams     4     3     2     1     0     0     0   prior-season games
    peers         3     2     1     0     0     0     0   prior-season games
    extra         1     1     1     1     0     0     0
    window size   8     8     8     8   8-9  9-10 10-11   total observations

AT WEEK 11 -- THIS PLAN'S OWN FIXTURE WEEK -- THE EXTRA PULL IS ZERO. By week 10
every team's current-season count has reached or passed max_prior_games, so
prior_count is 0 for bye team and peer alike and the window has stopped blending
seasons at all. The bye's whole effect at week 11 is that the window holds ten games
at week 11 and the SAME ten at week 12 while the target week advances: it ages, and
nothing is pulled forward to replace the missing game.

SO THE EFFECT IS NOT "STACKED ON WEEKS 2-4" -- IT SPANS TARGET WEEKS 6 TO 9, and it
is bounded above by one game. Weeks 2-4 carry a bootstrap effect of their own (a
window that is 5 to 7 parts prior season), but NO BYE HAS OCCURRED BY THEN: 2026's
earliest byes are week 5, so the earliest target week at which any team can show the
extra pull is 6.

THE ROSTER-CONTINUITY CAVEAT, STATED RATHER THAN LEFT IMPLICIT (Antigravity MEDIUM).
Every prior-season game a window pulls in -- in the first weeks of a season, or the
one extra game a post-bye team pulls in weeks 6-9 -- is evidence about a roster that
no longer exists in the same form. Offseason trades, the draft, free agency and
coaching changes all degrade the continuity assumption, and the selector applies NO
decay weight to distinguish a game played by last year's roster from one played by
this year's. Plan 33-11 deliberately does NOT add one: a decay parameter is a new
feature-engineering knob, which is a new predictive signal, which the SPEC excludes
phase-wide. The assumption is recorded here so that the readout states it, rather
than having a later reader discover it in the arithmetic.
"""


# ---------------------------------------------------------------------------
# THE POSTSEASON HALF OF R12 IS FIXTURE-BACKED, AND SAYING SO IS THE POINT.
#
# APPENDED by Plan 33-11 Task 2 on 2026-09-14. Nothing above this line was edited.
#
# T-33-54 names the risk this constant exists to close: a held-out constructed
# fixture presented as live evidence is how an unproven property comes to look
# proven. The label lives in three places on purpose -- the plan's must_haves mark
# the truth a BACKSTOP, tests/unit/test_postseason_partition.py's module docstring
# states it and asserts its own statement, and it is recorded here so a reader of
# the state manifest alone still learns it.
POSTSEASON_PARTITION_EVIDENCE_NOTE: str = """\
EVIDENCE CLASS: BACKSTOP (held-out constructed fixture). NOT live evidence.

WHY. All 272 rows of the single live 2026 capture
(schedules_raw_bronze_2026_W01_20260911T110252.parquet) are game_type REG in weeks 1
to 18. Weeks 19-22 are unseeded at capture time -- the bracket does not exist until
the regular season ends -- so there is no 2026 postseason row to measure. The proof
therefore runs on tests/fixtures/season_2026.WEEK_19_POSTSEASON_FIXTURE, the two
constructed wild-card rows Plan 33-02 built for the whole phase.

WHAT IS NEVERTHELESS REAL IN IT. The fixture is built in the FEED's column shape and
carries NO season_type of its own -- the feed has only game_type. The test runs it
through scripts.ingest_games.GameDataIngester.transform_schedule_data and compares
the stored value against _derive_season_type applied to the same rows, so the
Postseason value is PRODUCED by the real derivation rather than typed into a fixture.
A hand-typed literal would have made the module agree with itself forever, including
on the day the derivation changed -- which is the failure that let every postseason
game in this project be labelled a regular-season one in the first place.

WHAT WOULD UPGRADE IT TO LIVE EVIDENCE. One thing: a future capture carrying 2026
weeks 19-22. When one exists, load_captured_schedule returns rows whose game_type is
WC, DIV, CON or SB, the assertions re-point at them, and the constructed frame is
demoted to a shape check. Nothing else has to change, and no code has to move.

NOTE FOR PLAN 33-12. Silver's season_type is not yet real for historical seasons:
data/silver/games.parquet currently labels all 285 rows of 2025 -- including its 13
postseason games -- Regular. Plan 33-12 is the plan that makes the column true. This
plan asserts the DERIVATION and the PARTITION, both of which are correct today; it
deliberately asserts nothing about the historical stored values, which are not.
"""
