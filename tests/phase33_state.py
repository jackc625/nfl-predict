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
