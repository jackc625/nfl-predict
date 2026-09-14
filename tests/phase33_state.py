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


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-11 at plan close on 2026-09-14, AFTER both tasks landed.
# Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 1, for the reason Plans 33-02 through 33-10 each
# recorded when they did the same: two tasks add tests, so a mid-plan value would
# have been wrong at close and could only have been made right by EDITING it --
# the append-once violation the protocol exists to prevent.
#
# MEASURED by a whole-suite `--collect-only` BRACKET around this plan's own
# commits, in the main tree. Collection runs no test and takes about five seconds,
# and it counts the protocol's unit: COLLECTED NODES, a parametrised case counting
# once per generated node.
#
#     uv run python -m pytest tests -q --collect-only
#         07eaad4 (Plan 33-10's close, this plan's base)   5191 collected
#         9f0a408 (this plan's close)                      5246 collected
#         delta                                              55
#
# INDEPENDENTLY CONFIRMED per module, which is the check that no test was quietly
# deleted elsewhere to make the number look right. All three modules are NEW; this
# plan extended none:
#     38   tests/integration/test_bye_week_window_ageing.py    (Task 1)
#      9   tests/unit/test_bye_window_negative_control.py      (Task 2)
#      8   tests/unit/test_postseason_partition.py             (Task 2)
#     --
#     +55
#
# The two instruments agree exactly. 30 of the 38 integration nodes are the five
# parametrised properties generated once per bye team (5 x 6), which is why a count
# of test FUNCTIONS would have under-reported by 24 -- the multiplier the protocol's
# unit exists to capture.
#
# No whole-suite RUN was taken; see this plan's SUMMARY for what was run and the
# residual risk. A collected-node count does not need one, and none of the three new
# modules touches a DELIBERATE_TRIPWIRE_NODE_IDS member.
# ---------------------------------------------------------------------------

TESTS_ADDED_33_11: int = 55


# ---------------------------------------------------------------------------
# THE IDENTITY MIGRATION'S BLAST RADIUS, DECLARED BEFORE THE RUN.
#
# APPENDED by Plan 33-12 Task 1 on 2026-09-14, BEFORE a single byte of
# data/silver/games.parquet was rewritten, and committed in its own commit so the
# ordering is provable from git rather than asserted in prose. Declaring after the
# fact is not a declaration (T-33-61).
#
# DERIVED BY READING THE WRITE PATH, NOT BY ASSUMING IT:
#
#   scripts/ingest_games.GameDataIngester.ingest_games calls, in this order,
#     FIRST, save_bronze_snapshot with the schedule frame, the table name games,
#         season taken from seasons[0] and week 0.
#         data/storage.py:1135 -- writes ONE new timestamped parquet under
#         data/bronze/ per ingest_games() CALL, named
#         games_raw_bronze_{season}_W{week:02d}_{YYYYmmddTHHMMSS}.parquet.
#         It is season=seasons[0], so the file count follows the number of CLI
#         INVOCATIONS, not the number of seasons inside one invocation. This
#         migration therefore runs ONE INVOCATION PER SEASON -- twenty-four of
#         them -- so that each bronze file's embedded season is truthful. A
#         single --seasons 2002 ... 2025 call would have written ONE file
#         labelled 2002 holding all 6,499 rows, which is a misleading artifact.
#     THEN, upsert_silver with the validated frame and the table name games.
#         data/storage.py:1215 -- rewrites data/silver/games.parquet in place via
#         _atomic_write_parquet.
#
# THE SHARED DuckDB IS NOT IN THIS SET, AND THAT IS A READ FINDING RATHER THAN AN
# OMISSION. The Codex HIGH concern this plan carries is that
# data.storage.save_dataframe defaults save_to_db=True / save_to_parquet=True
# (data/storage.py:959-1063), so a silver write normally moves
# data/nfl_predictions.duckdb too. MEASURED BY READING upsert_silver
# (data/storage.py:1215-1263): it does NOT go through save_dataframe at all. It
# reads the existing parquet with pd.read_parquet, concatenates, normalises
# datetimes through a locally constructed ParquetManager, and calls
# _atomic_write_parquet. There is no get_db_connection call and no
# create_table_from_df call on that path. The DuckDB half is BYPASSED, so naming
# the file here would have declared a write that cannot happen -- and this phase's
# evidence model is that the declaration is EXACT, in both directions. If the
# bracket reports data/nfl_predictions.duckdb anyway, that is a finding about a
# second writer, not a reason to widen this tuple.
#
# THE FOURTH COLUMN THIS MIGRATION MOVES, DECLARED RATHER THAN DISCOVERED. The
# plan's prose says three identity columns. A dry run of the whole transform
# against the pinned schedules, taken BEFORE the migration and writing nothing,
# measured a FOURTH moved column: venue_roof, on 505 of 6,499 rows. It is not a
# surprise once located -- Plan 33-06 made _get_venue_roof_type consult the exact
# stadium_id key in data/venues.json FIRST (scripts/ingest_games.py:190-247) -- and
# it is the same repair reaching the same store, so it is inside this migration's
# declared cause rather than outside it. It is recorded here, before the run,
# because a value movement that feeds gold must never be found afterwards.
# See IDENTITY_MIGRATION_VENUE_ROOF_MOVEMENT below for the measured breakdown.
# ---------------------------------------------------------------------------

# POSIX-relative to the digest root `data`, matching tests/data_boundary.digest_tree's
# key form. The bronze entry is a PATTERN, not twenty-four guessed literals: the
# timestamp is minted at write time and cannot be known in advance.
IDENTITY_MIGRATION_EXPECTED_CHANGED_FILES: tuple[str, ...] = (
    # REWRITTEN. The upsert at scripts/ingest_games.py:342-343.
    "silver/games.parquet",
    # ADDED, one per CLI invocation. The snapshot at scripts/ingest_games.py:334-340.
    # <season> is 2002..2025; <ts> is a second-resolution UTC stamp.
    "bronze/games_raw_bronze_<season>_W00_<ts>.parquet",
)

# The number of ADDED bronze files the declaration above predicts: one per season,
# because the migration runs one CLI invocation per season.
IDENTITY_MIGRATION_EXPECTED_BRONZE_COUNT: int = 24

# The regular-expression form of the bronze pattern, so the verification can MATCH
# rather than eyeball. Kept beside the human-readable pattern deliberately: the
# prose entry is what a reader checks, this is what a test checks.
IDENTITY_MIGRATION_BRONZE_PATTERN: str = (
    r"^bronze/games_raw_bronze_(20(?:0[2-9]|1[0-9]|2[0-5]))_W00_\d{8}T\d{6}\.parquet$"
)

# EXPLICITLY NOT EXPECTED, and named so the absence is a claim rather than a gap.
IDENTITY_MIGRATION_EXPLICITLY_NOT_EXPECTED: tuple[str, ...] = (
    # upsert_silver bypasses save_dataframe entirely -- see the block above.
    "nfl_predictions.duckdb",
    # data/venues.json is Plan 33-06's artifact and is READ, never written, by the
    # ingest path. The snapshot for this bracket is taken AFTER that edit landed
    # (NF-06), so a REWRITTEN report here would mean something else touched it.
    "venues.json",
    # No gold matrix is written by an ingest. If one moves, the migration reached a
    # layer it has no business in.
    "gold/features_wp.parquet",
    "gold/features_ats.parquet",
    "gold/features_ou.parquet",
)

# ---------------------------------------------------------------------------
# WHICH IMMUTABLE UPSTREAM BYTES REPRODUCE EACH SEASON (Codex MEDIUM).
#
# A migration whose input cannot be named is not reproducible. Every season in
# 2002-2025 is covered by BOTH pinned datasets the ingest reads -- schedules (the
# identity source) and pbp (the score merge) -- so NO SEASON IS REFUSED. Coverage
# measured from config/upstream_pin.json on 2026-09-14:
#     schedules  1999-2025  (27 seasons)
#     pbp        2001-2025  (25 seasons)
# data.upstream_pin.SEALED_THROUGH_SEASON is 2025, so every row this migration
# writes comes from the SEALED, immutable zone. Nothing is fetched live.
#
# (season, dataset, path relative to data/, sha256 of those bytes)
# ---------------------------------------------------------------------------

IDENTITY_MIGRATION_PIN_MAPPING: tuple[tuple[int, str, str, str], ...] = (
    (
        2002,
        "schedules",
        "bronze/schedules_raw_bronze_2002_W00_20260905T044449.parquet",
        "be1bfdbdad8cf4057def76fd1208ba3b63eec0dc062ed0320a8f18118961685a",
    ),
    (
        2002,
        "pbp",
        "bronze/pbp_raw_bronze_2002_W00_20260905T044432.parquet",
        "64c61c84d88b632371be5f0bfe9c080c04b804cd8c47ddbbf684d8d923f5133e",
    ),
    (
        2003,
        "schedules",
        "bronze/schedules_raw_bronze_2003_W00_20260905T044449.parquet",
        "d8076921a22b75f39403f96cbbd34b479a55bfcd879f95944b1508b450d83679",
    ),
    (
        2003,
        "pbp",
        "bronze/pbp_raw_bronze_2003_W00_20260905T044433.parquet",
        "f381057320178272a1334f40f3a1b82cecbe4078bda9a0d9d7892cbaf536a2cf",
    ),
    (
        2004,
        "schedules",
        "bronze/schedules_raw_bronze_2004_W00_20260905T044449.parquet",
        "9e87caaabe65223a1982a939c4b598d25abe582ad1b535805f00d7e03e8c6b08",
    ),
    (
        2004,
        "pbp",
        "bronze/pbp_raw_bronze_2004_W00_20260905T044434.parquet",
        "6209519031f773cf9867997395986c7b015d65a81e82afb9f1c7b1a637c06562",
    ),
    (
        2005,
        "schedules",
        "bronze/schedules_raw_bronze_2005_W00_20260905T044449.parquet",
        "da69bed5b788e96739d006ee1162365a87d0051df9b16a97618b9a1aeaaa735f",
    ),
    (
        2005,
        "pbp",
        "bronze/pbp_raw_bronze_2005_W00_20260905T044434.parquet",
        "d70b21fe16eeef7e55d44d0ac1ec793c52d9ba0ee3098dfaf847b7a9e9e16f75",
    ),
    (
        2006,
        "schedules",
        "bronze/schedules_raw_bronze_2006_W00_20260905T044449.parquet",
        "a1445d57a19da2b685ebf36b532722173efb6e29f859321d8ca4ce1bca27044c",
    ),
    (
        2006,
        "pbp",
        "bronze/pbp_raw_bronze_2006_W00_20260905T044435.parquet",
        "07f258419224114542b86cf2ed337107266736c2372984a7057e2cabe36046a6",
    ),
    (
        2007,
        "schedules",
        "bronze/schedules_raw_bronze_2007_W00_20260905T044449.parquet",
        "2b5f81d44a644da36ee970b2b1a893e3eaea91642eda2c48e370ede4754f9210",
    ),
    (
        2007,
        "pbp",
        "bronze/pbp_raw_bronze_2007_W00_20260905T044436.parquet",
        "e62bd1f3f04a89e6853b9b5811bf6bd30b78a35320d1d577594b253aad1bd4e8",
    ),
    (
        2008,
        "schedules",
        "bronze/schedules_raw_bronze_2008_W00_20260905T044449.parquet",
        "f57d677eeff318cf592bcfd9da6d056972a3233540ffd6e84af1decc267ac08f",
    ),
    (
        2008,
        "pbp",
        "bronze/pbp_raw_bronze_2008_W00_20260905T044436.parquet",
        "41b724b22ea1479ce422fa6b5493c10bc8eeb898ed31f58617ca25fdb6635449",
    ),
    (
        2009,
        "schedules",
        "bronze/schedules_raw_bronze_2009_W00_20260905T044449.parquet",
        "2c4e705d1a772ea453e55d528b268cf638daf3b9e7382fdb206a0b5410f6b59b",
    ),
    (
        2009,
        "pbp",
        "bronze/pbp_raw_bronze_2009_W00_20260905T044437.parquet",
        "12eb6b4e689b72ca221be85c9d0e7cae5128bc471cbc389dda7b9cf83a584194",
    ),
    (
        2010,
        "schedules",
        "bronze/schedules_raw_bronze_2010_W00_20260905T044449.parquet",
        "720167aaa2f83cd188595c0252e99475165e2767b4cd22cd4b42cb0cf4cd9bb9",
    ),
    (
        2010,
        "pbp",
        "bronze/pbp_raw_bronze_2010_W00_20260905T044438.parquet",
        "567ce40efb787ff800f3636d1eb5cb472833e5d18a4bfdd133b2375b4c4a815a",
    ),
    (
        2011,
        "schedules",
        "bronze/schedules_raw_bronze_2011_W00_20260905T044449.parquet",
        "f8d52da91e095655476b897f24c38cef4986c232d5b10aa6c75236f7c1f73cfc",
    ),
    (
        2011,
        "pbp",
        "bronze/pbp_raw_bronze_2011_W00_20260905T044438.parquet",
        "9ad722a5ac90b520b8b0385db9550b0f524fad6872f81757fc2d7abc95743b52",
    ),
    (
        2012,
        "schedules",
        "bronze/schedules_raw_bronze_2012_W00_20260905T044449.parquet",
        "396b38cd3cfc6be7183d3e896d9cf3af2be74c21e85349ba6dc1127b7689bf75",
    ),
    (
        2012,
        "pbp",
        "bronze/pbp_raw_bronze_2012_W00_20260905T044439.parquet",
        "14ed7d17d6a177aea44a12a451926ae8670d4b29e53d405c0f86660c6e50cadf",
    ),
    (
        2013,
        "schedules",
        "bronze/schedules_raw_bronze_2013_W00_20260905T044449.parquet",
        "6ec41e9586018832bbb1a476a1fc234d24d42743c0e1d3b806b8d5ce6b435f4b",
    ),
    (
        2013,
        "pbp",
        "bronze/pbp_raw_bronze_2013_W00_20260905T044440.parquet",
        "d7648ecdf6fa20982025788f26915229e950643ad85c3311b8ee9a8524c4c74d",
    ),
    (
        2014,
        "schedules",
        "bronze/schedules_raw_bronze_2014_W00_20260905T044449.parquet",
        "0a5a7bcf957befff41841e9fb850730284dc4f5b946f8cc3780bacdbcefcd532",
    ),
    (
        2014,
        "pbp",
        "bronze/pbp_raw_bronze_2014_W00_20260905T044440.parquet",
        "53fa81f59086938fe50e957df8455b2d50ac4dfc55e49da78d21d03cf9174342",
    ),
    (
        2015,
        "schedules",
        "bronze/schedules_raw_bronze_2015_W00_20260905T044449.parquet",
        "cc232bd93359b5f20bdb8c50b7c9b09988c8c7722082a1060b8633742df961cf",
    ),
    (
        2015,
        "pbp",
        "bronze/pbp_raw_bronze_2015_W00_20260905T044441.parquet",
        "34577e8b53ce3355c2af800b77f8b867b3e4d88ced1d953d0dcb30b5cd77b3df",
    ),
    (
        2016,
        "schedules",
        "bronze/schedules_raw_bronze_2016_W00_20260905T044449.parquet",
        "c746580582985630edf789feccf0834c0ff986df00101e6051d918a4cb77e5c4",
    ),
    (
        2016,
        "pbp",
        "bronze/pbp_raw_bronze_2016_W00_20260905T044442.parquet",
        "1fc397b5a6fb3c80d4644fe6e6ac2456062ae0c4b2115ed59c5ca2cf18763d57",
    ),
    (
        2017,
        "schedules",
        "bronze/schedules_raw_bronze_2017_W00_20260905T044449.parquet",
        "99ef5bf88c7f11cc8207bdf12f108696b27c25e09147c65e89acc0919fc14327",
    ),
    (
        2017,
        "pbp",
        "bronze/pbp_raw_bronze_2017_W00_20260905T044443.parquet",
        "f06dd77c8fda477539949bcc6601c4ebb4585da9c231d1d53c443c231c60acb4",
    ),
    (
        2018,
        "schedules",
        "bronze/schedules_raw_bronze_2018_W00_20260905T044449.parquet",
        "0717edd443347b5f8db9aa16d80f54dbb4ec7bcb979e1c7f741c2b0dae1ee237",
    ),
    (
        2018,
        "pbp",
        "bronze/pbp_raw_bronze_2018_W00_20260905T044443.parquet",
        "3f1911453d42abfbf3c4aba4fc582bd52297712b62c68dab92550a8ba0eab67c",
    ),
    (
        2019,
        "schedules",
        "bronze/schedules_raw_bronze_2019_W00_20260905T044449.parquet",
        "b34d29acc4d4a1b9f780a421e3abb385566c5c61554921f185bd338530beb850",
    ),
    (
        2019,
        "pbp",
        "bronze/pbp_raw_bronze_2019_W00_20260905T044444.parquet",
        "d1b49b2251920b160f8d651c00b4ee4f38a6ce41237ae20c8b5c8f1881773742",
    ),
    (
        2020,
        "schedules",
        "bronze/schedules_raw_bronze_2020_W00_20260905T044449.parquet",
        "a5e0706516944849199f5ec9bb95cccf961fa124d3b0106c94b66e4a9b6e196d",
    ),
    (
        2020,
        "pbp",
        "bronze/pbp_raw_bronze_2020_W00_20260905T044445.parquet",
        "4d958443091a7ea9f6f819898b410a9ab80ddae1faa258270f77601c0076d14f",
    ),
    (
        2021,
        "schedules",
        "bronze/schedules_raw_bronze_2021_W00_20260905T044449.parquet",
        "4fb7e9e10cb10741cacb16f92f3401cfa74ab62c75884771fc7c3e1e6b5ee83d",
    ),
    (
        2021,
        "pbp",
        "bronze/pbp_raw_bronze_2021_W00_20260905T044445.parquet",
        "8b9a2ad3258eb69e914b78f9712a330855a83a184f7ed51b7e19914295b65f47",
    ),
    (
        2022,
        "schedules",
        "bronze/schedules_raw_bronze_2022_W00_20260905T044449.parquet",
        "b7728a683ce4c62a344ebfa1395072d627fbfaeced792d659a3dff699a56e3b4",
    ),
    (
        2022,
        "pbp",
        "bronze/pbp_raw_bronze_2022_W00_20260905T044446.parquet",
        "5c23f0fdcf9d6832dfab83b69cd94126a83ffed887d4d09e572ccf952cad7117",
    ),
    (
        2023,
        "schedules",
        "bronze/schedules_raw_bronze_2023_W00_20260905T044449.parquet",
        "c7a19c9bc92f60d8ca5c940536616389b204ed4a6c53b99803aa58fc17d59d18",
    ),
    (
        2023,
        "pbp",
        "bronze/pbp_raw_bronze_2023_W00_20260905T044447.parquet",
        "2f5b8326a440b09fd2dcfced2b0b6e477f6c6ee8fbb8c68cf9265302453eef12",
    ),
    (
        2024,
        "schedules",
        "bronze/schedules_raw_bronze_2024_W00_20260905T044449.parquet",
        "9c5279ba4c762b37f423597c8eb4eb0ca77791dc49fcaaebc213eadf2e9704e6",
    ),
    (
        2024,
        "pbp",
        "bronze/pbp_raw_bronze_2024_W00_20260905T044447.parquet",
        "c4875686c1baf25db35711c5e4f214ad209618cdcfb238054d83681da4e3dded",
    ),
    (
        2025,
        "schedules",
        "bronze/schedules_raw_bronze_2025_W00_20260905T044449.parquet",
        "b15e2ecffcd34a161d5337b549eaea2d91bad989890a981e7ed3dac9de0b5f00",
    ),
    (
        2025,
        "pbp",
        "bronze/pbp_raw_bronze_2025_W00_20260905T044448.parquet",
        "586d7cee6f3e526349a2c4145c7c6e304332793f6e800f669894079a6384715f",
    ),
)

# ---------------------------------------------------------------------------
# THE venue_roof MOVEMENT, MEASURED BEFORE THE MIGRATION RAN.
#
# 505 of 6,499 rows change roof classification, across 2002-2024 and ZERO rows of
# 2025. The 2025 slice already agrees with the new resolver, because it was
# re-ingested after Plan 33-06 landed; 2002-2024 was not. So the store currently
# carries TWO roof conventions at once, and this migration makes it carry one.
# That is a consistency repair, and stating it as one is more honest than calling
# 505 moved rows a side effect.
#
# Every transition is a venue whose NAME key missed in data/venues.json and fell
# through to the nflverse roof column, which maps both `dome` and `closed` to
# `indoor`. The stadium_id key does not miss.
# ---------------------------------------------------------------------------

IDENTITY_MIGRATION_VENUE_ROOF_MOVEMENT: tuple[tuple[str, str, str, str, int], ...] = (
    # (stadium_id, venue name as stored, before, after, rows)
    ("DAL00", "AT&T Stadium", "indoor", "retractable", 61),
    ("DAL00", "Cowboys Stadium", "indoor", "retractable", 28),
    ("HOU00", "NRG Stadium", "indoor", "retractable", 62),
    ("HOU00", "Reliant Stadium", "indoor", "retractable", 58),
    ("IND00", "Lucas Oil Stadium", "indoor", "retractable", 58),
    ("LAX01", "SoFi Stadium", "outdoor", "indoor", 87),
    ("PHO00", "State Farm Stadium", "indoor", "retractable", 62),
    ("PHO00", "University of Phoenix Stadium", "indoor", "retractable", 89),
)

IDENTITY_MIGRATION_VENUE_ROOF_ROWS_MOVED: int = 505

# ---------------------------------------------------------------------------
# THE PRE-MIGRATION DRY RUN, IN FULL.
#
# The whole transform was run in memory against the pinned schedules and diffed
# column by column against the live data/silver/games.parquet, writing nothing.
# Recorded because a migration that can say in advance exactly what it will move
# is a different object from one that reports afterwards what it did.
#
#     game_id set          IDENTICAL both ways, 6,499 rows, zero duplicates
#     season, week         0 rows differ
#     home_team, away_team 0 rows differ
#     venue                0 rows differ
#     kickoff_et           0 rows differ
#     home_score           0 rows differ
#     away_score           0 rows differ
#     result               0 rows differ
#     game_type            0 rows differ
#     venue_roof         505 rows differ   <- declared above
#     season_type        276 rows differ   REG 6223 / Postseason 276
#     neutral_site        91 rows differ
#     stadium_id         NEW column, 0 nulls, 55 distinct ids, all present in
#                        data/venues.json (60 records)
#
# home_score and away_score are already IDENTICAL to the pinned schedule feed, so
# the pbp merge this ingest still performs cannot move them. The merge is left ON
# anyway: the whole pinned pbp corpus is 30.9 MB over 23 columns, so running the
# tested default path costs nothing worth trading a deviation for.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# THE SILVER WIDTH THIS MIGRATION ACTUALLY PRODUCES: 16, NOT 18.
#
# SILVER_GAMES_COLUMNS_AFTER (18), appended by Plan 33-06 on 2026-09-12, is
# ARITHMETICALLY WRONG and is left exactly as written -- the manifest is
# append-only, and a slot is a record of what was believed when it was measured.
# The correction is appended beside it.
#
# THE ERROR IS 15 + 3. Only ONE of the three identity columns is NEW as a COLUMN.
# The measured pre-migration frame already carries `season_type` and
# `neutral_site` as columns; what it does not carry is a TRUE VALUE in either of
# them -- every cell reads 'Regular' and False respectively, which is the defect
# D33-15 describes. Adding `stadium_id` takes the frame from 15 to 16. The
# migration's real content is one new column and two columns that stop being
# constants, which is a bigger correction than a width change and a smaller number.
#
# MEASURED: validate_bronze_to_silver returns GameSchema.model_dump()'s 16 fields
# (data/schemas.py:26-77), and ingest_games overwrites created_at in place rather
# than adding a seventeenth.
# ---------------------------------------------------------------------------

SILVER_GAMES_COLUMNS_AFTER_MEASURED: int = 16

SILVER_GAMES_COLUMNS_AFTER_CORRECTION: str = """\
SILVER_GAMES_COLUMNS_AFTER is 18. The migration produces 16. The slot is not
edited; this is the correction appended beside it.

15 columns before, and TWO of the three identity columns were already among them:
season_type (constant 'Regular' on all 6,499 rows) and neutral_site (constant
False on all 6,499 rows). Only stadium_id is new as a column. 15 + 1 = 16.

Nothing downstream of the 18 was built on it: no test asserted the value, and the
only consumer was Plan 33-12's own verification, which is where the error surfaced
-- before the migration ran, not after.
"""


# ---------------------------------------------------------------------------
# WHAT THE MIGRATION ACTUALLY DID.
#
# APPENDED by Plan 33-12 Task 1 on 2026-09-14, AFTER the run, in a SECOND commit.
# Nothing above this line was edited -- in particular the declaration block is the
# one that was committed before the run, byte for byte.
#
# THE BRACKET. tests/data_boundary snapshot was taken over `data` (448 tracked
# files, ZERO stat signatures -- every baseline value is a content hash) after
# Plan 33-06's and Plan 33.1-01's data/venues.json edits had landed and been
# committed (NF-06), and re-read afterwards. The CLI exits 1 because the tree DID
# move; that is the point of a deliberate migration, and the acceptance test is
# that the moved set EQUALS the declaration, not that nothing moved.
#
#     REWRITTEN   silver/games.parquet                             1 file
#     ADDED       bronze/games_raw_bronze_<season>_W00_<ts>.parquet  24 files
#     REMOVED     none
#     MIXED       none        (no UNDECIDED comparison anywhere)
#     UNEXPLAINED none        (every moved key matches an entry or the pattern)
#
# Every member of IDENTITY_MIGRATION_EXPLICITLY_NOT_EXPECTED held: the shared
# data/nfl_predictions.duckdb did NOT move, confirming by measurement the read
# finding that upsert_silver bypasses save_dataframe; data/venues.json did not
# move; no gold matrix moved.
#
# THE CORROBORATION THAT THE INPUT WAS THE PINNED INPUT. Each of the 24 bronze
# snapshots the migration WROTE is BYTE-IDENTICAL to the sealed pinned schedule
# snapshot for that season recorded in IDENTITY_MIGRATION_PIN_MAPPING -- 24 of 24
# sha256 matches. The ingest therefore demonstrably read the immutable bytes the
# declaration named, rather than anything upstream happens to serve today. This is
# a stronger statement than "a pin covers the season": it is the same bytes out.
#
# THE RUN. Twenty-four invocations of the PUBLISHED command form
# (PIPELINE.md:29, RUNBOOK.md:140):
#     uv run python scripts/ingest_games.py --season <YEAR>
# for YEAR in 2002..2025, with the pbp score merge left ON (the tested default
# path). No parquet was hand-edited and nothing was written from a one-liner
# (T-33-60).
# ---------------------------------------------------------------------------

# The content sha256 of data/silver/games.parquet AFTER the migration, taken
# through tests.data_boundary.require_content_digest so it cannot be a stat
# signature (D33-32). Binary, so newlines are irrelevant.
SILVER_GAMES_DIGEST_AFTER_IDENTITY: str = (
    "3bc8cfa3e88d8e3218735502f20caf24fd583e67b11302bd867a3dc16f478f10"
)

# The digest the same file carried BEFORE, from the committed pre-run snapshot.
# Recorded beside the after value so the pair is checkable without the JSON.
SILVER_GAMES_DIGEST_BEFORE_IDENTITY: str = (
    "48901e5c24865896507fa1314a67a09325a164236a08ce267ad24707a98bc22f"
)

# Rows whose season_type is the postseason value, measured on the migrated store.
# Before the migration this was ZERO: season_type.unique() was ['Regular'] across
# all 6,499 rows, including every WC, DIV, CON and SB game ever played (D33-17).
POSTSEASON_ROW_COUNT: int = 276

# The full partition, so the count above is checkable rather than merely quoted.
SILVER_SEASON_TYPE_COUNTS_AFTER_IDENTITY: tuple[tuple[str, int], ...] = (
    ("Regular", 6223),
    ("Postseason", 276),
)

# Rows whose neutral_site is True on the migrated store. Before: zero, on a column
# that was a constant False. These are the same 91 games
# HISTORICAL_NEUTRAL_MISRESOLUTION enumerates -- the column now SAYS so, while the
# venue routing for those historical rows is deliberately NOT repaired here.
NEUTRAL_SITE_TRUE_ROW_COUNT: int = 91

# The migrated frame, measured rather than predicted.
SILVER_GAMES_ROWS_AFTER_IDENTITY: int = 6499
SILVER_GAMES_SEASONS_AFTER_IDENTITY: tuple[int, int] = (2002, 2025)

# Whole-store integrity checks taken at the same instant as the digest above.
# Recorded as a tuple of (check, value) so a later reader sees WHICH checks were
# run, not only that some were.
SILVER_GAMES_INTEGRITY_AFTER_IDENTITY: tuple[tuple[str, int], ...] = (
    ("rows", 6499),
    ("columns", 16),
    ("distinct_seasons", 24),
    ("duplicate_game_ids", 0),
    ("stadium_id_nulls", 0),
    ("primary_key_nulls", 0),
    ("season_type_game_type_disagreements", 0),
)

# The migrated roof distribution, the measured consequence of
# IDENTITY_MIGRATION_VENUE_ROOF_ROWS_MOVED. The BEFORE distribution is DERIVED by
# reversing the transition table rather than separately measured, because the file
# it would have been measured from no longer exists: outdoor 4806 (4719 + 87 that
# moved to indoor), indoor 1362 (1031 + 418 that moved to retractable - 87 that
# arrived from outdoor), retractable 331 (749 - 418). 4806 + 1362 + 331 = 6499.
SILVER_VENUE_ROOF_COUNTS_AFTER_IDENTITY: tuple[tuple[str, int], ...] = (
    ("outdoor", 4719),
    ("indoor", 1031),
    ("retractable", 749),
)


# ---------------------------------------------------------------------------
# THE PRE-ELO-REBUILD GOLD WIDTH REFERENCE (P6 / Q-01).
#
# APPENDED by Plan 33-12 Task 2 on 2026-09-14. Nothing above this line was edited.
#
# MEASURED through scripts/fingerprint_gold.fingerprint_gold -- the SAME instrument
# Plan 33-14 compares its rebuild with, so the two plans read the same numbers from
# the same tool rather than from two tools that happen to agree today.
#
# THE MEASUREMENT. A full-history gold build was driven from the MIGRATED
# 16-column data/silver/games.parquet into a sandbox lake, with the parquet
# manager injected onto the sandbox root and the DuckDB half disabled. It took
# 577 s and produced:
#
#     features_wp    6499 rows x 194 columns
#     features_ats   6499 rows x 195 columns
#     features_ou    6499 rows x 194 columns
#
# identical to the widths data/gold/ carries today, with a column SET identical to
# production's in all three matrices -- zero columns added, zero removed -- and
# none of stadium_id, neutral_site or season_type among them.
#
# SO THE MIGRATION IS SCHEMA-INERT AND VALUE-ACTIVE, AND BOTH HALVES ARE STATED.
# compare_fingerprints(production, sandbox) reports 132 of the 194/195/194 columns
# with MOVED VALUES in every matrix. That number is an UPPER BOUND on this
# migration's value effect and NOT an attribution to it: production gold predates
# the Phase-33.1 weather corrections and the Elo re-derivation as well, and
# separating those is exactly what running each as its own declared rung is for.
# What this measurement settles is the SCHEMA question, which is the one Plan
# 33-14's expected change set is anchored on.
#
# THE SANDBOX IS PROVEN TO BE THE SUBJECT (Codex HIGH). The committed instrument
# -- tests/integration/test_gold_write_scope.py, the full-history arm -- asserts
# each matrix file exists and carries an mtime at or after the build-start instant,
# brackets the whole build with digest_tree over the production data/ root and
# asserts identity, and records the three sandbox digests below. A build that
# silently fell back to the production root is therefore a boundary violation
# rather than a passing test.
#
# WHY THE COMMITTED ARM IS OPT-IN. 577 s against a suite that already runs about
# 23 minutes. It is run by name:
#     NFL_RUN_FULL_GOLD_BUILD=1 uv run python -m pytest
#         tests/integration/test_gold_write_scope.py -k full_history -q
# The always-on tests beside it check these recorded values against today's
# production gold, so the reference cannot go stale unnoticed.
# ---------------------------------------------------------------------------

GOLD_WIDTHS_BEFORE_ELO_REBUILD: tuple[int, int, int] = (194, 195, 194)

# WHICH FILES those widths were read from, rather than leaving it to be assumed.
# Content sha256, in the (wp, ats, ou) order of GOLD_WIDTHS_BEFORE_ELO_REBUILD.
# Each differs from the production gold digest of the same name, which is the
# anti-fallback control: a sandbox build that quietly read production gold would
# have produced three identical pairs.
SANDBOX_GOLD_DIGESTS_33_12: tuple[tuple[str, str], ...] = (
    (
        "features_wp.parquet",
        "c6842210e7968fd6ac4c85040cc0b54e67ce1f5c52af3d85edd1a26b3cd73ce3",
    ),
    (
        "features_ats.parquet",
        "8463bd20eb26ab78100e7930132a499579d2baa4374d26f9735d624d5f00bd40",
    ),
    (
        "features_ou.parquet",
        "7fb46b21b835742486283186e39a59d8f422693508d7855f2ece917e166f5a7d",
    ),
)

# The production gold digests the three above were compared against, recorded at
# the same instant so the comparison is reproducible from the manifest alone.
PRODUCTION_GOLD_DIGESTS_AT_33_12: tuple[tuple[str, str], ...] = (
    (
        "features_wp.parquet",
        "8b8a82b1e032b933d40eb9c8a3b7758d32ce82a3958ef5062cd1fd346064bf4f",
    ),
    (
        "features_ats.parquet",
        "362dbca9213d02a39a23b5430b26ad55ae9519ec55d36cb73abde8775a1fdd0b",
    ),
    (
        "features_ou.parquet",
        "453a83dc6b37080bdfefd2da438bfd1e98048fb425a3d31772330debb6fa7b2f",
    ),
)

# The bounded observation described above, recorded as a number so the readout can
# quote it rather than re-deriving it. Per matrix: columns whose VALUES moved
# between production gold and a fresh full-history build from today's inputs.
# NOT an attribution to this migration -- see the block above.
GOLD_COLUMNS_MOVED_PRODUCTION_VS_FRESH_BUILD: tuple[tuple[str, int, int], ...] = (
    # (matrix, columns whose values moved, total columns)
    ("features_wp", 132, 194),
    ("features_ats", 132, 195),
    ("features_ou", 132, 194),
)

# ---------------------------------------------------------------------------
# stadium_id IS LOAD-BEARING, NOT AN OPTIONAL EXTRA COLUMN.
#
# The plan's downstream-compatibility ask was that every reader of
# data/silver/games.parquet TOLERATE the added columns, including a null
# stadium_id. Driven against the real builders, two of the three are inert and the
# third is the opposite of tolerated, so the result is recorded split rather than
# averaged into a comfortable summary.
#
# season_type and neutral_site: dropping both changes not one output cell of the
# contextual or market-anchor builders.
#
# stadium_id: dropping it, or nulling one cell, raises UnknownStadiumError BY NAME.
# Since D33.1-06 the contextual builder routes EVERY game of EVERY season by its
# own stadium_id (features/contextual.py:1268, :1593), and Plan 33.1-04's
# owner-assigned fix made an unresolvable id a loud hard failure rather than an
# empty contextual frame. Refusing is the CORRECT behaviour: the alternative is
# silently resolving to the home team's stadium, which is the misresolution
# HISTORICAL_NEUTRAL_MISRESOLUTION measures on 91 games. The migrated store carries
# ZERO nulls, so the refusal is unreachable in practice -- unreachable by refusal
# rather than by luck, which is why it is asserted.
#
# THIS IS THE MECHANICAL REASON PLAN 33-12 HAD TO PRECEDE PLAN 33.1-07, and it is
# now measured rather than inferred from the sequencing document.
# ---------------------------------------------------------------------------

IDENTITY_COLUMNS_INERT_FOR_READERS: tuple[str, ...] = (
    "season_type",
    "neutral_site",
)

IDENTITY_COLUMNS_LOAD_BEARING_FOR_READERS: tuple[str, ...] = ("stadium_id",)


# ---------------------------------------------------------------------------
# THE SECOND COPY OF SILVER games, AND WHY A PARQUET-ONLY MIGRATION IS HALF DONE.
#
# APPENDED by Plan 33-12 on 2026-09-14, AFTER the migration and AFTER the
# verification run that found it. Nothing above this line was edited -- in
# particular IDENTITY_MIGRATION_EXPECTED_CHANGED_FILES and the read finding
# beside it are left exactly as declared before the run. They were CORRECT about
# what upsert_silver writes. What they did not foresee was the CONSEQUENCE.
#
# THE FINDING. There are TWO copies of silver `games`: the parquet the migration
# rewrote, and a base table inside the shared data/nfl_predictions.duckdb.
# data.storage.load_dataframe's DEFAULT source="auto" tries DuckDB FIRST
# (data/storage.py:1093-1099), so after a parquet-only write every ordinary
# reader -- build_features, build_contextual, data_qa, the prediction scripts --
# is served the STALE copy. On this migration that copy had no `stadium_id`
# column at all, which since D33.1-06 is a hard contextual-build failure.
#
# HOW IT SURFACED, recorded because the route matters. Not by foresight: by the
# idempotency control in tests/integration/test_n01_resync_control.py, whose
# "second apply" was in fact the FIRST apply after a divergence this migration
# had just created. It failed, correctly, and in failing it ran resync_games()
# and brought the two copies into agreement. The right end state by an accidental
# route, which is not the same as the right process, and saying so is the point
# of recording it here.
#
# THE BRACKET IS UNHARMED AND THE TWO EVENTS ARE NOT CONFLATED. The migration's
# own bracket was VERIFIED at 02:45 on 2026-09-14, before any test ran, and was
# exactly {silver/games.parquet REWRITTEN, 24 bronze ADDED}. The DuckDB moved
# LATER, during verification, and its mover is the repository's one declared
# production writer -- the node carrying
# @pytest.mark.writes_production_store(paths=["data/nfl_predictions.duckdb"]).
# A cumulative diff against the pre-migration snapshot therefore reports
# {silver/games.parquet, nfl_predictions.duckdb, 24 bronze} and every member of
# it is attributable by name.
#
# THE END STATE, MEASURED. `uv run python -m scripts.resync_games_duckdb` reports
# `parquet 6499  db 6499  divergence 0`, and the two copies' table_content_digest
# values are identical.
#
# THE GUARD THAT WAS MISSING. No test asserted that the two copies agree, which
# is why a parquet-only migration could look complete. Plan 33-12 adds it to
# tests/integration/test_data_completeness.py in the cheap always-on form.
#
# THE RUNBOOK CONSEQUENCE FOR EVERY LATER PLAN THAT REWRITES SILVER games --
# 33-13's Elo re-derivation and 33.1-07's rung both read it, and 33-18's readout
# reports on it:
#
#     uv run python -m scripts.resync_games_duckdb --apply
#
# must follow the write, and it moves data/nfl_predictions.duckdb, so it belongs
# in that plan's DECLARED changed-file set rather than arriving as a surprise.
# ---------------------------------------------------------------------------

SILVER_GAMES_HAS_TWO_COPIES: bool = True

SILVER_GAMES_DUCKDB_RESYNC_COMMAND: str = (
    "uv run python -m scripts.resync_games_duckdb --apply"
)

# The agreed content digest of BOTH copies after the re-sync, taken with
# scripts.resync_games_duckdb.table_content_digest -- a digest of the TABLE's
# content, not of either file's bytes, so it is comparable across the two stores.
SILVER_GAMES_TABLE_DIGEST_AFTER_RESYNC: str = (
    "f64bbd1be38b70f2e8e8cb15f1e48e792123bd46c488f9b283d74ed7c3f792b7"
)

# The cumulative moved set over the WHOLE plan -- the migration itself plus the
# verification run that re-synced the DuckDB. Recorded beside the migration's own
# declared set rather than replacing it, because the two answer different
# questions and merging them would hide which event moved what.
IDENTITY_MIGRATION_CUMULATIVE_CHANGED_FILES: tuple[tuple[str, str], ...] = (
    ("silver/games.parquet", "REWRITTEN by the migration, bracket verified 02:45"),
    (
        "bronze/games_raw_bronze_<season>_W00_<ts>.parquet",
        "24 ADDED by the migration, bracket verified 02:45",
    ),
    (
        "nfl_predictions.duckdb",
        "REWRITTEN LATER, during verification, by the declared production writer "
        "in tests/integration/test_n01_resync_control.py -- the N-01 re-sync",
    ),
)


# ---------------------------------------------------------------------------
# THIS PLAN'S COLLECTED-NODE COUNT.
#
# APPENDED by Plan 33-12 at plan close on 2026-09-14, AFTER both tasks and the
# deviation fixes landed. Nothing above this line was edited.
#
# APPENDED AT CLOSE, NOT IN TASK 1, for the reason Plans 33-02 through 33-11 each
# recorded when they did the same: both tasks add tests, so a Task-1 value would
# have been wrong at close and could only have been made right by EDITING it --
# the append-once violation the protocol exists to prevent. This plan is the case
# in point: the value after Task 2 was 31, and the deviation fix that closed the
# DuckDB finding added two more.
#
# MEASURED by a whole-suite `--collect-only` BRACKET around this plan's own
# commits. Collection runs no test and takes about five seconds, and it counts the
# protocol's unit: COLLECTED NODES, a parametrised case counting once per
# generated node.
#
#     uv run python -m pytest tests -q --collect-only
#         b01dd25 (Plan 33-11's close, this plan's base)   5246 collected
#         e27ffa8 (this plan's last code commit)           5279 collected
#         delta                                              33
#
# INDEPENDENTLY CONFIRMED per module against the SAME files at the base commit,
# which is the check that no test was quietly deleted elsewhere to make the number
# look right. All four modules already existed; this plan created none:
#
#     module                                            base   now   delta
#     tests/integration/test_gold_write_scope.py          15    26     +11
#     tests/integration/test_data_completeness.py         14    20      +6
#     tests/unit/test_games_identity_columns.py           26    42     +16
#     tests/integration/test_routing_coordinate_diff.py   43    43      +0
#                                                                      ---
#                                                                      +33
#
# The two instruments agree exactly. routing_coordinate_diff contributes ZERO
# because its Wave-12 tripwire was RESOLVED IN PLACE rather than removed or
# doubled: one node renamed, its body replaced with the recorded decision and a
# strictly stronger assertion.
#
# ONE OF THE 33 IS SKIPPED BY DEFAULT and that is deliberate: the full-history
# sandbox gold build costs 577 s measured. It is COLLECTED, which is the unit this
# number counts, and it is RUN by name with NFL_RUN_FULL_GOLD_BUILD=1 -- it was run
# once during this plan, passing in 608 s. A further group of nodes -- every one
# that asserts about the MIGRATED production store -- skips by pinned message on a
# checkout with no data/ directory, because data/ is gitignored and a hard failure
# there would report a fact about the checkout rather than about the data.
#
# No whole-suite RUN was taken; see this plan's SUMMARY for exactly what was run,
# what was not, and the residual risk. What WAS run is a 113-module affected set
# closing on the five DELIBERATE_TRIPWIRE_NODE_IDS and nothing else:
#
#     5 failed, 2101 passed, 11 skipped, 9 xfailed in 1108.77 s
#
# None of the four modules touches a DELIBERATE_TRIPWIRE_NODE_IDS member.
# ---------------------------------------------------------------------------

TESTS_ADDED_33_12: int = 33


# -------------------------------------------------------------------------
# THE PHASE-33.1 GOLD RUNG, DECLARED BEFORE ANYTHING WAS REBUILT.
#
# APPENDED by Plan 33.1-07 Task 1 on 2026-09-14, BEFORE the owner ruling of
# Task 2 and BEFORE the Task-3 rebuild. Nothing above this line was edited.
#
# WHY A DECLARATION AND NOT A RECORD. Every other slot in this module records
# what some run MEASURED. This one records what the rung PREDICTS, and it is
# committed first for exactly one reason: a signature written after the diff is
# seen is a transcription wearing a prediction's clothes (T-33.1-43). Task 3
# STOPS on a verdict it cannot satisfy rather than adjusting this.
#
# ALL THREE FAMILIES ARE ENUMERABLE OR SOURCE-DERIVED (Ruling N2). The first
# draft declared the third as "any column that moves because 207 rows gained
# coverage", which is a CAUSE STORY: it cannot be evaluated against a diff, so
# any moved column can be argued into it afterwards. This repository has the
# worked example -- scripts/fingerprint_gold._attribute_rung2 is a blanket
# predicate whose own comment records that it "cannot FAIL on a moved column",
# and that rung 2's first attempt "attributed perfectly cleanly -- ok, zero
# unattributed -- while having silently destroyed 18 columns".
#
# THE 207 IDS ARE A ONE-SHOT MEASUREMENT. They are exactly the games gold
# carries and the two silver feature tables lack; the rebuild CLOSES that gap,
# so afterwards the set cannot be re-derived. Measured here, before it ran, on
# the same argument p331_rung0.json rests on. RESEARCH section 10 recorded 207
# and the re-derivation measured 207 -- the two agree, so there is no
# divergence to record in the GOLD_WEATHER_CONSTANCY_MEASUREMENT idiom.
# -------------------------------------------------------------------------

PHASE331_RUNG_DECLARATION: dict[str, object] = {
    "declared_on": "2026-09-14",
    "declared_by": "Plan 33.1-07 Task 1",
    "committed_before_rebuild": True,
    "rung": 1,
    "rung_prefix": "p331_",
    "cause": (
        "COMPOUND (three causes, never 'the weather rung'): (1) real ERA5 "
        "weather replacing the fabricated 65.0F constant across 2002-2025 "
        "[R5/D33.1-07]; (2) the all-seasons stadium_id routing correction "
        "[D33.1-06], which moves the venue/travel/timezone/elevation "
        "family for the 1,153 games that resolved to the wrong stadium; "
        "and (3) restored 2025 coverage -- the 207 games the two silver "
        "feature tables were missing, which the rebuild adds "
        "independently of any weather or routing change"
    ),
    # The three families, and the MECHANISM each is expressed as. A value
    # outside these two mechanisms would be a cause story, which Ruling N2
    # forbids -- and a test asserts the set membership rather than trusting
    # the prose above.
    "declared_families": ("weather", "venue", "staleness_2025"),
    "family_mechanisms": {
        "weather": "source-derived constant",
        "venue": "source-derived constant",
        "staleness_2025": "row-scoped per-season-digest predicate",
    },
    "family_sources": {
        "weather": "features.weather.WEATHER_FEATURE_COLUMNS",
        "venue": (
            "scripts.fingerprint_gold.PHASE331_VENUE_FAMILY_COLUMNS, derived "
            "from the columns features.contextual.ContextualFeaturesCalculator"
            ".build_features actually emits, minus the merge keys"
        ),
        "staleness_2025": (
            "PHASE331_STALENESS_GAME_IDS restricted to the per-season digests "
            "outside 2025 -- a changed column outside the two column families "
            "is attributable only if it is byte-identical in every season "
            "except 2025"
        ),
    },
    # The predicted diff shape. ASSERTED by the rung, not described to it.
    "columns_added": ("weather_coverage",),
    "columns_removed": (),
    "rows": "unchanged",
    "rows_expected": 6499,
    "width_delta_per_matrix": 1,
    "widths_before": (194, 195, 194),
    "widths_predicted": (195, 196, 195),
    # Ruling N2, second half. An explanation SUPPLEMENTS the check and can
    # never substitute for it. Where an out-of-family move turns out to be
    # legitimate, the correct act is a NEW declared family in a follow-up
    # rung, not a footnote on this one.
    "ok_required_unconditionally": True,
    "staleness_seasons": (2025,),
    "staleness_game_id_count": 207,
    "staleness_game_id_count_recorded_by_research": 207,
    "staleness_measured_from": (
        "set(data/gold/features_ats.parquet.game_id) minus "
        "set(data/silver/weather_features.parquet.game_id), cross-checked "
        "against contextual_features.parquet -- the two agree exactly"
    ),
}

# The 207 ids themselves, measured 2026-09-14 BEFORE the rebuild. All 207 are
# season 2025: gold holds 285 rows of 2025 and both silver feature tables held
# 78, and 285 - 78 = 207 exactly.
PHASE331_STALENESS_GAME_IDS: tuple[str, ...] = (
    "2025_W06_ARI@IND",
    "2025_W06_BUF@ATL",
    "2025_W06_CHI@WAS",
    "2025_W06_CIN@GB",
    "2025_W06_CLE@PIT",
    "2025_W06_DAL@CAR",
    "2025_W06_DEN@NYJ",
    "2025_W06_DET@KC",
    "2025_W06_LA@BAL",
    "2025_W06_LAC@MIA",
    "2025_W06_NE@NO",
    "2025_W06_PHI@NYG",
    "2025_W06_SEA@JAX",
    "2025_W06_SF@TB",
    "2025_W06_TEN@LV",
    "2025_W07_ATL@SF",
    "2025_W07_CAR@NYJ",
    "2025_W07_GB@ARI",
    "2025_W07_HOU@SEA",
    "2025_W07_IND@LAC",
    "2025_W07_LA@JAX",
    "2025_W07_LV@KC",
    "2025_W07_MIA@CLE",
    "2025_W07_NE@TEN",
    "2025_W07_NO@CHI",
    "2025_W07_NYG@DEN",
    "2025_W07_PHI@MIN",
    "2025_W07_PIT@CIN",
    "2025_W07_TB@DET",
    "2025_W07_WAS@DAL",
    "2025_W08_BUF@CAR",
    "2025_W08_CHI@BAL",
    "2025_W08_CLE@NE",
    "2025_W08_DAL@DEN",
    "2025_W08_GB@PIT",
    "2025_W08_MIA@ATL",
    "2025_W08_MIN@LAC",
    "2025_W08_NYG@PHI",
    "2025_W08_NYJ@CIN",
    "2025_W08_SF@HOU",
    "2025_W08_TB@NO",
    "2025_W08_TEN@IND",
    "2025_W08_WAS@KC",
    "2025_W09_ARI@DAL",
    "2025_W09_ATL@NE",
    "2025_W09_BAL@MIA",
    "2025_W09_CAR@GB",
    "2025_W09_CHI@CIN",
    "2025_W09_DEN@HOU",
    "2025_W09_IND@PIT",
    "2025_W09_JAX@LV",
    "2025_W09_KC@BUF",
    "2025_W09_LAC@TEN",
    "2025_W09_MIN@DET",
    "2025_W09_NO@LA",
    "2025_W09_SEA@WAS",
    "2025_W09_SF@NYG",
    "2025_W10_ARI@SEA",
    "2025_W10_ATL@IND",
    "2025_W10_BAL@MIN",
    "2025_W10_BUF@MIA",
    "2025_W10_CLE@NYJ",
    "2025_W10_DET@WAS",
    "2025_W10_JAX@HOU",
    "2025_W10_LA@SF",
    "2025_W10_LV@DEN",
    "2025_W10_NE@TB",
    "2025_W10_NO@CAR",
    "2025_W10_NYG@CHI",
    "2025_W10_PHI@GB",
    "2025_W10_PIT@LAC",
    "2025_W11_BAL@CLE",
    "2025_W11_CAR@ATL",
    "2025_W11_CHI@MIN",
    "2025_W11_CIN@PIT",
    "2025_W11_DAL@LV",
    "2025_W11_DET@PHI",
    "2025_W11_GB@NYG",
    "2025_W11_HOU@TEN",
    "2025_W11_KC@DEN",
    "2025_W11_LAC@JAX",
    "2025_W11_NYJ@NE",
    "2025_W11_SEA@LA",
    "2025_W11_SF@ARI",
    "2025_W11_TB@BUF",
    "2025_W11_WAS@MIA",
    "2025_W12_ATL@NO",
    "2025_W12_BUF@HOU",
    "2025_W12_CAR@SF",
    "2025_W12_CLE@LV",
    "2025_W12_IND@KC",
    "2025_W12_JAX@ARI",
    "2025_W12_MIN@GB",
    "2025_W12_NE@CIN",
    "2025_W12_NYG@DET",
    "2025_W12_NYJ@BAL",
    "2025_W12_PHI@DAL",
    "2025_W12_PIT@CHI",
    "2025_W12_SEA@TEN",
    "2025_W12_TB@LA",
    "2025_W13_ARI@TB",
    "2025_W13_ATL@NYJ",
    "2025_W13_BUF@PIT",
    "2025_W13_CHI@PHI",
    "2025_W13_CIN@BAL",
    "2025_W13_DEN@WAS",
    "2025_W13_GB@DET",
    "2025_W13_HOU@IND",
    "2025_W13_JAX@TEN",
    "2025_W13_KC@DAL",
    "2025_W13_LA@CAR",
    "2025_W13_LV@LAC",
    "2025_W13_MIN@SEA",
    "2025_W13_NO@MIA",
    "2025_W13_NYG@NE",
    "2025_W13_SF@CLE",
    "2025_W14_CHI@GB",
    "2025_W14_CIN@BUF",
    "2025_W14_DAL@DET",
    "2025_W14_DEN@LV",
    "2025_W14_HOU@KC",
    "2025_W14_IND@JAX",
    "2025_W14_LA@ARI",
    "2025_W14_MIA@NYJ",
    "2025_W14_NO@TB",
    "2025_W14_PHI@LAC",
    "2025_W14_PIT@BAL",
    "2025_W14_SEA@ATL",
    "2025_W14_TEN@CLE",
    "2025_W14_WAS@MIN",
    "2025_W15_ARI@HOU",
    "2025_W15_ATL@TB",
    "2025_W15_BAL@CIN",
    "2025_W15_BUF@NE",
    "2025_W15_CAR@NO",
    "2025_W15_CLE@CHI",
    "2025_W15_DET@LA",
    "2025_W15_GB@DEN",
    "2025_W15_IND@SEA",
    "2025_W15_LAC@KC",
    "2025_W15_LV@PHI",
    "2025_W15_MIA@PIT",
    "2025_W15_MIN@DAL",
    "2025_W15_NYJ@JAX",
    "2025_W15_TEN@SF",
    "2025_W15_WAS@NYG",
    "2025_W16_ATL@ARI",
    "2025_W16_BUF@CLE",
    "2025_W16_CIN@MIA",
    "2025_W16_GB@CHI",
    "2025_W16_JAX@DEN",
    "2025_W16_KC@TEN",
    "2025_W16_LA@SEA",
    "2025_W16_LAC@DAL",
    "2025_W16_LV@HOU",
    "2025_W16_MIN@NYG",
    "2025_W16_NE@BAL",
    "2025_W16_NYJ@NO",
    "2025_W16_PHI@WAS",
    "2025_W16_PIT@DET",
    "2025_W16_SF@IND",
    "2025_W16_TB@CAR",
    "2025_W17_ARI@CIN",
    "2025_W17_BAL@GB",
    "2025_W17_CHI@SF",
    "2025_W17_DAL@WAS",
    "2025_W17_DEN@KC",
    "2025_W17_DET@MIN",
    "2025_W17_HOU@LAC",
    "2025_W17_JAX@IND",
    "2025_W17_LA@ATL",
    "2025_W17_NE@NYJ",
    "2025_W17_NO@TEN",
    "2025_W17_NYG@LV",
    "2025_W17_PHI@BUF",
    "2025_W17_PIT@CLE",
    "2025_W17_SEA@CAR",
    "2025_W17_TB@MIA",
    "2025_W18_ARI@LA",
    "2025_W18_BAL@PIT",
    "2025_W18_CAR@TB",
    "2025_W18_CLE@CIN",
    "2025_W18_DAL@NYG",
    "2025_W18_DET@CHI",
    "2025_W18_GB@MIN",
    "2025_W18_IND@HOU",
    "2025_W18_KC@LV",
    "2025_W18_LAC@DEN",
    "2025_W18_MIA@NE",
    "2025_W18_NO@ATL",
    "2025_W18_NYJ@BUF",
    "2025_W18_SEA@SF",
    "2025_W18_TEN@JAX",
    "2025_W18_WAS@PHI",
    "2025_W19_BUF@JAX",
    "2025_W19_GB@CHI",
    "2025_W19_HOU@PIT",
    "2025_W19_LA@CAR",
    "2025_W19_LAC@NE",
    "2025_W19_SF@PHI",
    "2025_W20_BUF@DEN",
    "2025_W20_HOU@NE",
    "2025_W20_LA@CHI",
    "2025_W20_SF@SEA",
    "2025_W21_LA@SEA",
    "2025_W21_NE@DEN",
    "2025_W22_SEA@NE",
)


# -------------------------------------------------------------------------
# THE OWNER RULING THAT AUTHORISED THE RUNG -- Plan 33.1-07 Task 2.
#
# APPENDED on 2026-09-14, AFTER Task 1 committed PHASE331_RUNG_DECLARATION and
# BEFORE Task 3 rebuilt anything. Nothing above this line was edited.
#
# Task 2 is a checkpoint:decision with reversibility rating="one-way":
# data/gold/ is gitignored and unbacked, the three matrices are overwritten in
# place, and the OLD gold is the state the three currently-deployed models were
# fitted and gate-baselined against. It cannot be reconstructed once the
# corrected silver has replaced its input. So the ruling is recorded here, as a
# constant, rather than living only in a SUMMARY: Task 3's own precondition
# reads "Task 2's owner ruling is recorded as approved", and a precondition
# that can only be satisfied by prose is a precondition nothing can check.
#
# WHAT THE OWNER WAS SHOWN AND ACCEPTED. All five what-to-check items were put
# to the owner and accepted ON THE RECORD. They are recorded here in the order
# they were put, because the fourth one in particular -- the ninety columns
# this rung does NOT fix -- is the one a later reader is most likely to assume
# away.
# -------------------------------------------------------------------------

PHASE331_RUNG_OWNER_RULING: dict[str, object] = {
    "ruled_on": "2026-09-14",
    "plan": "33.1-07",
    "task": "Task 2 (checkpoint:decision, reversibility one-way)",
    "verdict": "approved",
    "option_selected": "one-compound-rung",
    "option_declined": "three-rungs",
    "authorises": (
        "rebuilding data/silver/weather_features.parquet, "
        "data/silver/contextual_features.parquet and all three "
        "data/gold/features_*.parquet as ONE rung with the compound cause, "
        "against the change set already declared in committed source at "
        "77fe13c"
    ),
    # Ruling N2 survives the authorisation UNCHANGED. What the owner authorised
    # is the one-rung ATTRIBUTION -- not a promise that the change set turns out
    # to be exactly that narrow. A residual STOPS Task 3; it is never annotated
    # past, and the declared set is never widened after the diff is seen.
    "authorises_the_attribution_not_the_outcome": True,
    "ok_still_required_unconditionally": True,
    "remedy_for_a_legitimate_out_of_family_move": (
        "a NEW declared family in a follow-up rung, never a footnote on this one"
    ),
    "facts_shown_and_accepted": (
        (
            "1. The declared change set predicts ONE added column "
            "(weather_coverage), nothing removed, rows unchanged at 6,499, "
            "widths 194/195/194 -> 195/196/195. attribute_rung's ok must be "
            "True UNCONDITIONALLY; a residual STOPS Task 3 and is surfaced as "
            "a new checkpoint rather than annotated past."
        ),
        (
            "2. The third cause is NOT a weather change. Both silver feature "
            "tables hold 6,292 rows against gold's 6,499; the 207 missing "
            "games are all season 2025 and are committed as "
            "PHASE331_STALENESS_GAME_IDS. Including them in this rung is "
            "authorised."
        ),
        (
            "3. Thirteen columns become NaN for 1,652 indoor games (nine "
            "temperature plus four temperature-derived), per Ruling J and "
            "WEATHER_NULL_STATE_MATRIX. The seven composite columns keep a "
            "genuine 0.0. Authorised and understood."
        ),
        (
            "4. The 90 columns that stay a flat imputed constant for "
            "2002-2017 -- every Elo column, every rolling opponent-adjusted "
            "EPA column, all three market snapshot columns and the "
            "situational spots -- are NOT fixed by this rung and must NOT be "
            "described as fixed. The readout says so plainly."
        ),
        (
            "5. No model is re-fit, no gate runs, artifacts/latest.json stays "
            "byte-identical. Task 3's trainer smoke fit is "
            "read-and-fit-in-memory only: save() is never called and "
            "`git status --short artifacts/` is asserted empty afterwards."
        ),
    ),
}

# The SECOND ruling the owner made in the same sitting, on a Rule-3 blocker the
# checkpoint surfaced rather than on the rung itself.
#
# THE BLOCKER. Ruling N3's runbook names
# `uv run python scripts/build_contextual.py --all-seasons` and
# `uv run python scripts/build_weather.py --all-seasons` as the ONE command per
# artifact, and Task 3's own <verify> REJECTS a recorded command naming
# --season rather than --all-seasons. But neither script accepted the flag:
# both took only --season / --week / --save / --validate, and OMITTING --season
# is what produced the full historical build. The documented command exited 2.
#
# THE RULING: add the flag rather than record a different command. --all-seasons
# becomes an explicit alias for the full historical build in BOTH scripts,
# mirroring scripts/build_features.py:2133 and its recorded rationale -- the
# destructive/full mode should be ASKED FOR by name rather than reached by
# omission -- and PIPELINE.md is updated to match, so the runbook's "ONE exact
# command per artifact" is literally executable rather than a description.
#
# The alias is a NAMING alias and not new build logic: it is mutually exclusive
# with --season and leaves target_season None, which is byte-identically what
# omitting --season already did.
PHASE331_ALL_SEASONS_FLAG_RULING: dict[str, object] = {
    "ruled_on": "2026-09-14",
    "plan": "33.1-07",
    "raised_during": "Task 2 (checkpoint:decision)",
    "deviation_rule": "Rule 3 (blocking issue)",
    "verdict": "add the flag alias",
    "scripts_changed": (
        "scripts/build_contextual.py",
        "scripts/build_weather.py",
    ),
    "docs_changed": ("PIPELINE.md",),
    "mirrors": "scripts/build_features.py --all-seasons",
    "behaviour": (
        "a NAMING alias for the full historical build, mutually exclusive with "
        "--season; it leaves target_season None, which is exactly what "
        "omitting --season already did. No new build logic."
    ),
    "why_not_record_a_different_command": (
        "Ruling N3's point is that 'the rebuild' must be a reproducible ACT "
        "rather than a description, and Task 3's verify rejects a --season "
        "invocation because a scoped build MERGES latest-wins and is refused "
        "if it tries to change the schema. Recording a bare invocation instead "
        "would have left the full/destructive mode reachable only by omission, "
        "which is the exact shape build_features.py:2122-2129 already records "
        "as the mistake --all-seasons exists to prevent."
    ),
    "scope": (
        "AUTHORISED addition beyond the plan's files_modified list, for this "
        "purpose only; recorded in the SUMMARY as a Rule 3 deviation"
    ),
}


# -------------------------------------------------------------------------
# THE DECLARED BLAST RADIUS OF THE PHASE-33.1 GOLD RUNG.
#
# APPENDED by Plan 33.1-07 Task 3 on 2026-09-14, and COMMITTED BEFORE the
# first digest snapshot of the rung was taken. Nothing above this line was
# edited.
#
# THE ORDER IS THE POINT, as it was for Plan 33-09's 14-row backfill and Plan
# 33.1-06's full-corpus promotion above. A blast radius declared after the fact
# is not a declaration, it is a transcription of whatever happened. This slot
# was committed first, then p331_rung0.json was written, then the ladder
# pre-check ran, then the three builders ran, then `verify` was run against
# this declaration. A FILE OUTSIDE THIS SET IS A FINDING TO REPORT, NEVER A
# REASON TO WIDEN THE SET.
#
# THE DUCKDB HALF IS *IN* THIS SET, AND IT WAS *OUT* OF THE SILVER
# PROMOTION'S. That contrast is the reason this declaration was written fresh
# rather than copied, and a copied declaration would have hidden it.
#
#   * WEATHER_PROMOTION_EXPECTED_CHANGED_FILES (Plan 33.1-06) EXCLUDES
#     nfl_predictions.duckdb because that promotion writes through
#     `data.storage.upsert_silver`, which reads the existing parquet, filters,
#     concats and ends at `_atomic_write_parquet`. It never opens the database.
#   * THIS rung runs the two feature builders and the gold build, all of which
#     write through `data.storage.save_dataframe`, whose `save_to_db` defaults
#     True. Both halves move. `pipeline/steps.py:523-533`
#     (`step_build_weather_features`) is the same path.
#
# Two declarations differing because two WRITE PATHS differ is the point. If
# the database had NOT moved here, the write did not go where this declaration
# says it went, and that would itself be a finding.
#
# WHAT IS DELIBERATELY ABSENT: every bronze path. This rung fetches nothing --
# Plan 33.1-06 already fetched the corpus and promoted it to silver. A bronze
# file moving here would mean a builder reached for data it was supposed to
# read from silver, which is the same class of finding as the network guard
# firing.
# -------------------------------------------------------------------------

PHASE331_RUNG_EXPECTED_CHANGED_FILES: tuple[str, ...] = (
    "gold/features_ats.parquet",
    "gold/features_ou.parquet",
    "gold/features_wp.parquet",
    "nfl_predictions.duckdb",
    "silver/contextual_features.parquet",
    "silver/weather_features.parquet",
)


# -------------------------------------------------------------------------
# THE PHASE-33.1 FOLLOW-UP RUNG, DECLARED AFTER THE RESIDUAL WAS DIAGNOSED.
#
# APPENDED by Plan 33.1-07 Task 3 on 2026-09-14, AFTER the rebuild and AFTER
# the root-cause investigation the owner ordered at the rung's fail-closed
# checkpoint. Nothing above this line was edited.
#
# WHY THIS SLOT EXISTS AT ALL. Rung 1 returned ok=False with 45 unattributed
# columns, identical in all three matrices. Ruling N2 names exactly two
# legitimate responses -- STOP, or declare a NEW family in a FOLLOW-UP RUNG --
# and forbids the third, a footnote on rung 1. The owner ordered the residual
# diagnosed before anything was declared, so this is the second response taken
# on EVIDENCE rather than as a guess.
#
# WHAT IS DIFFERENT FROM A WIDENED DECLARATION, and it is the whole point:
# PHASE331_RUNG_DECLARATION above is BYTE-UNCHANGED, PHASE331_EXPECTED_SIGNATURE
# is byte-unchanged, and rung 1 STILL RETURNS ok=False on this diff. A test
# asserts that last property directly. A follow-up rung bounds the unknown and
# writes it down; a widened declaration would have hidden it inside a
# plausible-looking bucket and reported green.
#
# THE HONESTY THIS SLOT IS REQUIRED TO CARRY. Group 1's trigger is NOT
# ESTABLISHED and its magnitude is PERMANENTLY UNMEASURABLE. Both are recorded
# below as machine-readable values rather than as prose a later reader can skim
# past, and what the diagnosis ELIMINATED is recorded beside what it could not
# find -- because the value of naming a family here is that the unknown is
# bounded and legible, not that it looks explained.
# -------------------------------------------------------------------------

PHASE331_FOLLOWUP_RUNG_DECLARATION: dict[str, object] = {
    "declared_on": "2026-09-14",
    "declared_by": "Plan 33.1-07 Task 3, after the ordered root-cause diagnosis",
    "rung": 2,
    "rung_prefix": "p331_",
    "ruled_by_owner": (
        "close the attribution with a FOLLOW-UP RUNG declaring all three "
        "residual groups"
    ),
    "ruling_basis": (
        "Ruling N2's second legitimate move, taken on evidence rather than as a guess"
    ),
    # It rebuilt NOTHING. The same transition, re-judged under a second
    # declaration -- which is why no p331_rung2.json fingerprint document
    # exists. Writing one would assert a rebuild that did not happen.
    "judges_transition": (
        "p331_rung0.json -> p331_rung1.json (the SAME one rung 1 judged)"
    ),
    "no_new_rebuild": True,
    "fingerprint_document_written": None,
    "attribution_document_written": (
        "outputs/fingerprints/p331_rung2_attribution.json"
    ),
    "diagnosis_document": (
        ".planning/phases/33.1-real-historical-weather-and-training-window-"
        "correction/33.1-07-GROUP1-DIAGNOSIS.md"
    ),
    "debug_session_file": ".planning/debug/p331-group1-2024-drift.md",
    # Rung 1's declaration is untouched, and its verdict on this diff is
    # unchanged. Asserted by a test, not claimed here.
    "rung_1_declaration_byte_unchanged": True,
    "rung_1_still_returns_ok_false_on_this_diff": True,
    "ok_required_unconditionally": True,
    "residual_columns_declared": 45,
    "identical_across_all_three_matrices": True,
    "declared_families": (
        "carried_at_rung_1",
        "stale_baseline_2024",
        "prohibited_family_2025",
        "weather_widening",
    ),
    "family_mechanisms": {
        "carried_at_rung_1": "source-derived constant",
        "stale_baseline_2024": "enumerated names with a season restriction",
        "prohibited_family_2025": "enumerated names with a season restriction",
        "weather_widening": "source-derived constant",
    },
    "family_sources": {
        "carried_at_rung_1": (
            "rung 1's own three families, unchanged -- "
            "features.weather.WEATHER_FEATURE_COLUMNS, "
            "scripts.fingerprint_gold.PHASE331_VENUE_FAMILY_COLUMNS, and the "
            "row-scoped 2025 staleness predicate"
        ),
        "stale_baseline_2024": (
            "scripts.fingerprint_gold.PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS, "
            "restricted to PHASE331_FOLLOWUP_STALE_BASELINE_SEASONS"
        ),
        "prohibited_family_2025": (
            "scripts.fingerprint_gold.PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS, "
            "restricted to PHASE331_FOLLOWUP_PROHIBITED_2025_SEASONS"
        ),
        "weather_widening": (
            "scripts.fingerprint_gold.PHASE331_FOLLOWUP_WEATHER_WIDENING -- a "
            "mapping from each un-normalized copy to the "
            "WEATHER_FEATURE_COLUMNS member it copies, checked against the live "
            "registry at attribution time"
        ),
    },
    # ------------------------------------------------------------------
    # GROUP 1 -- THE STALE-BASELINE CARRY-FORWARD (40 columns)
    # ------------------------------------------------------------------
    "group_1": {
        "label": "stale-baseline carry-forward from Phase 33's waves 9-12",
        "columns": 40,
        "seasons_moved": "24 columns in 2024 only, 16 in 2024 and 2025",
        "what_they_are": (
            "12 opponent-adjusted rolling (all that exist), 26 silver-sourced "
            "team-form rolling, and both Elo momentum columns. Every "
            "non-degenerate team-form column moved; the 9 that did not are the "
            "degenerate defensive-side duplicates of offence-only metrics."
        ),
        # THE TWO FACTS THE RULING REQUIRES, RECORDED WITHOUT SOFTENING.
        "trigger": "NOT ESTABLISHED",
        "magnitude": "PERMANENTLY UNMEASURABLE",
        "trigger_detail": (
            "Localised to 2026-09-12 08:36 -> 2026-09-14 02:55, the window in "
            "which Phase 33's waves 9-12 ran. Inside that window every "
            "observable input and all builder code are byte-identical, and BOTH "
            "the pre-rung code and the current code produce the CURRENT 2024 "
            "values when run against today's data. The pre-rung values are not "
            "reproducible from any state that still exists."
        ),
        "magnitude_detail": (
            "p331_rung0.json holds per-season sha256 digests, never values, and "
            "no copy of the pre-rung gold survives -- an exhaustive search of "
            "the repository and the machine's temp tree found none. A digest can "
            "say 'different'; it can never say 'how different'. The sharp "
            "2020-2023-clean / 2024-moved boundary is an ARGUMENT against float "
            "noise (numpy and pandas unchanged since 2026-03-18), not a "
            "measurement, and it is not recorded as one."
        ),
        "predates_this_rung": (
            "The Plan 33-12 sandbox gold, built 02:55:48 on 2026-09-14 from the "
            "OLD 6,292-row weather silver -- five and a half hours BEFORE the "
            "rung -- already carries 40/40 of the new 2024 values and 0/40 of "
            "rung 0's. The move is not caused by the real weather, not by the "
            "stadium_id routing correction and not by the 2025 coverage "
            "restore. The rung inherited it."
        ),
        # What the investigation ELIMINATED, each by a controlled rebuild rather
        # than by argument. Recorded because a bounded unknown is only bounded
        # if the boundary is written down too.
        "eliminated_by_measurement": (
            "build nondeterminism -- a fresh full rebuild reproduced gold "
            "byte-for-byte in every column of every season except "
            "feature_timestamp",
            "the wall-clock as_of default -- pinning as_of_datetime 4.5 months "
            "earlier was byte-identical",
            "population-dependent normalisation -- read in code and refuted by "
            "measurement (40 of 181 normalised columns moved, and only in 2024)",
            "the Wave-12 identity migration -- reverting kickoff_et, "
            "venue_roof, season_type and neutral_site moves only the rest-days "
            "family",
            "the builder code -- the pre-rung tree at bc31982 run against "
            "today's data gives 40/40 match to CURRENT gold",
            "row order -- reversing season 2024's rows within each week moves "
            "146 columns, not 40",
            "the pinned upstream play-by-play -- all 76 "
            "config/upstream_pin.json entries re-hash clean",
            "the DuckDB/parquet two-copy divergence -- all three silver tables "
            "are row-order-and-value identical in both copies",
        ),
        "not_eliminated": (
            "that the 2026-09-12 build read something no longer on disk -- a "
            "silver table in a since-overwritten state, a different "
            "DATA_ROOT_PATH, or an environment that differed in a way nothing "
            "recorded. Untestable after the fact, which is the argument for the "
            "input-provenance manifest the diagnosis recommends."
        ),
        "why_2024_specifically": (
            "team-form silver covers only 2020-2025, so 2002-2019 are a "
            "structural 0.0 constant and CANNOT move; "
            "features/team_form.py:714 hardcodes range(2018, 2025), making 2024 "
            "the last opponent-adjusted season; and season 2025's normalisation "
            "bootstraps on season 2024, which is the 24-only / 16-both split."
        ),
        "season_restriction": (2024, 2025),
    },
    # ------------------------------------------------------------------
    # GROUP 2 -- THE PROHIBITION FIRING CORRECTLY (4 columns)
    # ------------------------------------------------------------------
    "group_2": {
        "label": (
            "the mislabelling prohibition firing correctly -- the guard "
            "working, not a defect"
        ),
        "columns": 4,
        "column_names": (
            "elo_prob_home",
            "home_def_rolling_red_zone_td_rate",
            "home_elo",
            "home_elo_uncertainty",
        ),
        "seasons_moved": "2025 only",
        "why_rung_1_refused_them": (
            "They satisfy rung 1's row-scoped staleness predicate exactly -- "
            "2025 and no other season -- and would have been attributed to the "
            "207-game coverage restore, except that they belong to PROHIBITED "
            "families (elo, and the bye-window team-form rolling predicate) and "
            "_attribute_phase331 checks the prohibition FIRST by design, so a "
            "prohibited column can never be absorbed by a declared family."
        ),
        "verdict": (
            "NOT a defect. Declaring these four does not open the prohibited "
            "families: every prohibited column NOT named here is still refused "
            "at the follow-up rung, and these four keep a season restriction of "
            "exactly 2025."
        ),
        "season_restriction": (2025,),
    },
    # ------------------------------------------------------------------
    # GROUP 3 -- A GENUINE WEATHER-FAMILY WIDENING (1 column)
    # ------------------------------------------------------------------
    "group_3": {
        "label": "genuine weather-family widening",
        "columns": 1,
        "column_names": ("raw_weather_severity",),
        "seasons_moved": "all 24 seasons, 2002-2025",
        "why_rung_1_refused_it": (
            "It is absent from features.weather.WEATHER_FEATURE_COLUMNS. "
            "scripts/build_features.py:1819 mints it as a verbatim "
            "un-normalized COPY of weather_severity_score after imputation and "
            "before normalisation, so it never passes through the registry rung "
            "1's family 1 is derived from."
        ),
        "verdict": (
            "It is weather-derived, and moving in every season is exactly what "
            "replacing a fabricated 65.0F constant with real ERA5 observations "
            "does to a weather column. Declared as a SOURCE-DERIVED mapping to "
            "the registered column it copies, so an entry whose source is not a "
            "weather column is refused."
        ),
        "source_column": "weather_severity_score",
    },
    # The verdict this rung actually returned, measured.
    "measured_verdict": {
        "ok": True,
        "blocking": False,
        "failures": 0,
        "attributed_per_matrix": 131,
        "unattributed_per_matrix": 0,
        "changed_by_family_per_matrix": {
            "carried_at_rung_1": 86,
            "stale_baseline_2024": 40,
            "prohibited_family_2025": 4,
            "weather_widening": 1,
        },
    },
    # What this rung does NOT claim, stated because a clean verdict is the most
    # likely thing to be over-read.
    "what_this_rung_does_not_claim": (
        "It does NOT claim the Group-1 trigger was found -- it was not, and the "
        "declaration says so in machine-readable form.",
        "It does NOT claim the current 2024 values are CORRECT. They are "
        "reproducible from today's inputs by both the current and the pre-rung "
        "code and the old ones are reproducible from nothing, which is an "
        "argument for them and not a proof; neither set was audited against "
        "ground truth.",
        "It does NOT re-open, defend or re-freeze any gate baseline. The "
        "standing owner ruling of 2026-09-14 voids the pre-correction artifacts "
        "and every verdict resting on the corrupted inputs, so there is nothing "
        "here to preserve and no non-regression comparison to protect.",
        "It rebuilt NOTHING, re-fit NOTHING and left artifacts/ byte-unchanged.",
    ),
}


# -------------------------------------------------------------------------
# WHAT THE PHASE-33.1 RUNG ACTUALLY MOVED -- Plan 33.1-07 Task 3.
#
# APPENDED on 2026-09-14, after the rebuild, after the ordered diagnosis and
# after the follow-up rung closed the attribution. Nothing above this line was
# edited -- in particular PHASE331_RUNG_DECLARATION, which is the PREDICTION
# this slot is the OBSERVATION of, is byte-unchanged. The two being separate
# names appended at different moments is what makes the pair readable as a
# prediction and its outcome rather than as one self-consistent story.
#
# READING `ok`. The attribution is CLOSED and `ok` is True -- but the number
# that would be easiest to misread is recorded beside it, not behind it:
# `rung_1_ok` is FALSE. Rung 1 refused 45 columns and still does. The True
# belongs to the FOLLOW-UP rung, which declared those 45 in three families
# after the owner ordered them diagnosed. Ruling N2's unconditional check is
# satisfied by a second declaration, never by annotating past the first, and
# both verdicts are on the record so a later reader can see which is which.
# -------------------------------------------------------------------------

PHASE331_RUNG_MEASURED: dict[str, object] = {
    "measured_on": "2026-09-14",
    "measured_by": "Plan 33.1-07 Task 3",
    # THE ATTRIBUTION'S FINAL STATE, and the intermediate one beside it.
    "ok": True,
    "ok_meaning": (
        "the attribution is CLOSED: every one of the 131 non-clock moved columns "
        "is attributed to a declared family across the two-rung ladder. It does "
        "NOT mean rung 1 passed -- see rung_1_ok."
    ),
    "rung_1_ok": False,
    "rung_1_blocking": True,
    "rung_1_failures": 135,
    "rung_1_unattributed_per_matrix": 45,
    "closed_by_followup_rung": True,
    "followup_rung": 2,
    "followup_ok": True,
    "followup_failures": 0,
    "followup_attribution_document": (
        "outputs/fingerprints/p331_rung2_attribution.json"
    ),
    "cause": (
        "COMPOUND (three causes, never 'the weather rung'): (1) real ERA5 "
        "weather replacing the fabricated 65.0F constant across 2002-2025 "
        "[R5/D33.1-07]; (2) the all-seasons stadium_id routing correction "
        "[D33.1-06], which moves the venue/travel/timezone/elevation family for "
        "the 1,153 games that resolved to the wrong stadium; and (3) restored "
        "2025 coverage -- the 207 games the two silver feature tables were "
        "missing, which the rebuild adds independently of any weather or "
        "routing change"
    ),
    # THE SHAPE. Every prediction in PHASE331_RUNG_DECLARATION held.
    "columns_added": ("weather_coverage",),
    "columns_removed": (),
    "widths_before": (194, 195, 194),
    "widths_after": (195, 196, 195),
    "rows_before": (6499, 6499, 6499),
    "rows_after": (6499, 6499, 6499),
    "coverage_flag_present_in_all_three": True,
    "silver_rows_before": 6292,
    "silver_rows_after": 6499,
    "silver_2025_rows_after": 285,
    "non_clock_moves": 131,
    "build_clock_moves": ("feature_timestamp",),
    "identical_residual_across_all_three_matrices": True,
    # THE PER-FAMILY SPLIT, per matrix. Rung 1's three families attributed 86;
    # the follow-up's three declared the 45 rung 1 refused.
    "changed_by_family": {
        "rung_1": {"weather": 45, "venue": 24, "staleness_2025": 17},
        "followup_rung_2": {
            "carried_at_rung_1": 86,
            "stale_baseline_2024": 40,
            "prohibited_family_2025": 4,
            "weather_widening": 1,
        },
    },
    # Ruling N2: this SUPPLEMENTS the check and never replaces it. It is
    # populated because the explanation is worth having -- and the check it
    # supplements was satisfied by a FOLLOW-UP RUNG, which is the remedy Ruling
    # N2 names, not by this text.
    "unattributed_with_reason": {
        "stale_baseline_2024": (
            "40 columns, 24 moving in 2024 only and 16 in 2024 and 2025. A "
            "STALE-BASELINE CARRY-FORWARD from Phase 33's waves 9-12: the Plan "
            "33-12 sandbox gold, built five and a half hours BEFORE this rung "
            "from the OLD silver, already carries 40/40 of the new 2024 values "
            "and 0/40 of rung 0's, so the move predates the rung and none of "
            "its three causes produced it. ITS TRIGGER IS NOT ESTABLISHED and "
            "ITS MAGNITUDE IS PERMANENTLY UNMEASURABLE -- see "
            "PHASE331_FOLLOWUP_RUNG_DECLARATION['group_1'] for the eight "
            "eliminations and the two unknowns, both stated without softening."
        ),
        "prohibited_family_2025": (
            "4 columns moving in 2025 only. The mislabelling prohibition firing "
            "CORRECTLY: they satisfy rung 1's staleness predicate but belong to "
            "prohibited families, and the prohibition takes precedence by "
            "design. The guard working, not a defect."
        ),
        "weather_widening": (
            "raw_weather_severity, moving in all 24 seasons -- a genuine "
            "weather-family widening. It is an un-normalized copy of "
            "weather_severity_score minted at "
            "scripts/build_features.py:1819, so it never passes through "
            "features.weather.WEATHER_FEATURE_COLUMNS and rung 1's family 1 "
            "could not reach it."
        ),
    },
    # THE RUNBOOK, as run. Ruling N3's ONE command per artifact, in order.
    "commands_run": (
        {
            "artifact": "data/silver/contextual_features.parquet",
            "command": "uv run python scripts/build_contextual.py --all-seasons",
            "wall_clock_seconds": 19.2,
        },
        {
            "artifact": "data/silver/weather_features.parquet",
            "command": "uv run python scripts/build_weather.py --all-seasons",
            "wall_clock_seconds": 6.4,
        },
        {
            "artifact": "data/gold/features_{wp,ats,ou}.parquet",
            "command": "uv run python scripts/build_features.py --all-seasons",
            "wall_clock_seconds": 563.0,
        },
    ),
    # Ruling N3's second half: R3's zero-network property proven on the REAL
    # bytes, not only in Plan 33.1-06's sandbox. The guard was proven live on a
    # deliberate probe BEFORE the builders ran, so a silent no-op guard could
    # not have passed for a denial.
    "rebuilt_with_network_denied": True,
    "network_guard_proven_on_a_deliberate_probe_first": True,
    # The digest bracket, closed. EXACTLY the declared six paths moved -- one
    # REWRITTEN section, no ADDED, no REMOVED, no MIXED.
    "declared_changed_files_verdict": "exact match, 6 of 6, REWRITTEN only",
    "artifacts_tree_unchanged": True,
    "artifacts_files_compared": 159,
    # THE TRAINER SMOKE FIT (step 7). READ-AND-FIT-IN-MEMORY: `save()` was never
    # called and `git status --short artifacts/` was empty afterwards. A build
    # can produce three valid parquet files while leaving a target untrainable,
    # and the rebuilt gold carries NaN-bearing weather columns.
    "trainer_smoke_feature_counts": {"wp": 20, "ats": 25, "ou": 25},
    "trainer_smoke_fitted_models": {
        "wp": "Pipeline",
        "ats": "XGBRegressor",
        "ou": "XGBRegressor",
    },
    "trainer_smoke_wrote_no_artifact": True,
    "trainer_smoke_partition": "TemporalSplitConfig.default() -- train 2018-2019, hp_val 2020, holdout 2021-2024",
    "trainer_smoke_tune": False,
    # AN OBSERVATION, NOT A TARGET. Whether WP or ATS now selects weather is
    # Wave 15's measurement and this phase predicts nothing about it. Recorded
    # because it is the first time WP -- a LogReg that consumed ZERO weather
    # features on the fabricated-constant gold -- selects any.
    "trainer_smoke_weather_columns_selected": {
        "wp": (
            "passing_difficulty",
            "temp_cold",
            "wind_high",
            "wind_severe",
        ),
        "ats": ("ball_handling_difficulty", "temp_mild", "wind_mph"),
        "ou": ("passing_difficulty", "temp_cold", "wind_high", "wind_mph"),
    },
    "trainer_smoke_weather_selection_is_an_observation_not_a_target": True,
    # WHAT THIS RUNG STILL DOES NOT DO.
    "no_model_refit": True,
    "no_gate_run": True,
    "latest_json_byte_identical": True,
}

# The counted widths after the rung, and the NAME of the one column that moved
# them. Both are appended here rather than left implicit in the integers,
# because the delta is pinned to the NAME in
# tests/unit/test_data_qa_gold_width.py: a build that added an unrelated column
# while omitting the flag satisfies +1 exactly as well as the right one does.
#
# GOLD_WIDTHS_BEFORE_ELO_REBUILD above is the BEFORE half and is byte-unchanged.
GOLD_WIDTHS_AFTER_WEATHER_RUNG: tuple[int, int, int] = (195, 196, 195)

WEATHER_COVERAGE_GOLD_COLUMN: str = "weather_coverage"


# -------------------------------------------------------------------------
# THE PHASE-33.1 RUNG 3 DECLARATION -- THREE INPUT CORRECTIONS, ONE REBUILD.
#
# APPENDED by Plan 33.1-07 Task 4 on 2026-09-14, and COMMITTED BEFORE the
# first digest snapshot of this rung was taken. Nothing above this line was
# edited: rungs 1 and 2 keep their own declarations, their own verdicts and
# their own residuals, which is the difference between a NEW rung and a
# widened one (Ruling N2).
#
# WHY THERE IS A THIRD RUNG. Task 4's job was to MEASURE what rung 1
# produced. The measurement found three defects in the INPUTS rung 1 -- and
# every rung before it -- had been consuming. All three DISCARD REAL DATA
# THAT IS SITTING ON DISK:
#
#   1. RAIN THAT FELL WAS DISCARDED FOR EVERY OUTDOOR GAME.
#      features/weather.calculate_precipitation_features refused whenever
#      precip_prob was absent. precip_prob is a FORECAST probability and the
#      corpus is ERA5 REANALYSIS, which reports what happened rather than
#      what was expected -- COVERAGE.md records the opt-out and
#      scripts/ingest_weather writes precip_prob: None for exactly that
#      reason. MEASURED in gold before the fix: raw_precip_mm non-null on
#      6,499 rows, precip_mm non-null on 1,652 -- the indoor games alone.
#      Twenty columns were NULL for all 4,847 outdoor games: the twelve
#      precipitation columns, and the seven composites plus is_dry that read
#      precip_impact_score. It also contradicted Plan 33.1-04's Ruling J,
#      which says those columns are MEASURED for a covered outdoor game.
#
#   2. THE COVERAGE FLAG WAS INERT. Silver weather_features carried
#      weather_coverage = 1.0 on all 6,499 rows; gold recorded 0.0 on all
#      6,499 -- the value _absent_observation_features writes to mean NO
#      OBSERVATION. The flag was not mis-written: normalize_combined_features
#      z-scored it, and the expanding std of a constant column is zero, so
#      every row collapsed onto the neutral 0.0. R5 requires a NaN null
#      observation ALONGSIDE a MEANINGFUL coverage column, and a column
#      asserting the opposite of the truth is not meaningful.
#
#   3. SEASON 2025'S TEAM STRENGTH WAS FAKE. features/team_form.py resolved
#      its per-game pool as range(2018, 2025), which stops at 2024. MEASURED
#      in gold: the twelve *_rolling_opp_adj_* columns carry 285 distinct
#      values across the 285 rows of 2023 and of 2024, and TWO across the 285
#      rows of 2025. 2025 is the season feeding the live 2026 predictions.
#
# THE OWNER'S RULING, 2026-09-14: fix all three, then ONE rebuild. The
# tradeoff they were given and ACCEPTED is recorded here rather than softened
# -- three causes in one rebuild cannot be separated afterwards. That
# separability mattered while the frozen pre-correction gate baseline was the
# thing being protected; the STANDING OWNER RULING of the same date voids
# that baseline ("you cannot non-regress against a lie"), so what one rebuild
# costs is separability between three corrections that all point the same
# way, and what it buys is ending the phase with inputs that are correct.
#
# THIS RUNG'S BASELINE IS NOT STALE, unlike rung 1's. p331_rung2.json is a
# fingerprint of gold as rung 1 left it -- freshly built, freshly attributed,
# and already the subject of eight controlled rebuilds by the Group-1
# diagnosis. Rung 1's baseline was five and a half hours old and carried the
# unexplained 2024 carry-forward the follow-up rung had to declare.
#
# WHY p331_rung2.json EXISTS AT ALL when rung 2 rebuilt nothing: gold at rung
# 2 IS gold at rung 1, so the document is a fresh MEASUREMENT of current gold
# rather than a copy, taken before this rung rebuilds anything.
# require_rung_ladder(dir, 3, "p331_") then finds rungs 0, 1 and 2 present.
# -------------------------------------------------------------------------

PHASE331_RUNG3_DECLARATION: dict[str, object] = {
    "declared_on": "2026-09-14",
    "declared_by": "Plan 33.1-07 Task 4",
    "committed_before_rebuild": True,
    "rung": 3,
    "rung_prefix": "p331_",
    "baseline_document": "p331_rung2.json",
    "baseline_is_a_fresh_measurement_not_a_copy": True,
    "cause": (
        "COMPOUND (three INPUT corrections, never 'the precipitation rung'): "
        "(1) precipitation derived from ERA5's measured precip_mm instead of "
        "being discarded for want of a forecast probability the archive never "
        "reports; (2) the weather_coverage flag preserved at its recorded "
        "level instead of z-scored to 0.0, the value that means NO "
        "OBSERVATION, on all 6,499 rows; and (3) season 2025's twelve "
        "opponent-adjusted team-strength columns built from real play-by-play "
        "instead of imputed, because the per-game season pool was the "
        "hardcoded range(2018, 2025) and stopped at 2024"
    ),
    # THE THREE FAMILIES AND THE MECHANISM EACH IS EXPRESSED AS. Ruling N2
    # forbids a CAUSE STORY -- a family that cannot be evaluated against a
    # diff -- and a machine check in the attribution tests asserts that every
    # value below is one of the three permitted mechanisms.
    "declared_families": ("weather", "weather_widening", "team_strength_2025"),
    "family_mechanisms": {
        "weather": "source-derived constant",
        "weather_widening": "source-derived constant",
        "team_strength_2025": "enumerated names with a season restriction",
    },
    # THE STRUCTURAL PREDICTION, and it is the OPPOSITE of rungs 1 and 2 in
    # the one way that matters. Those predicted a column ARRIVING. This rung
    # corrects what three existing families CONTAIN, so the shape must not
    # move at all -- and an ADDED column, including the coverage flag itself,
    # is undeclared here.
    "columns_added": "empty",
    "columns_removed": "empty",
    "rows": "unchanged at 6,499",
    "width": "unchanged at 195 / 196 / 195",
    # WHY THE TEAM-STRENGTH FAMILY IS RESTRICTED TO 2025, MEASURED RATHER
    # THAN ASSUMED. Adding 2025 to the per-game pool could in principle have
    # moved every season: features/opponent_adj.py computes league_avg_def as
    # a WHOLE-FRAME mean over the fetched rows, and the wider pool does move
    # that scalar (-0.003941 -> -0.002092). A READ-ONLY probe ran the REAL
    # adjustment stage under both pools BEFORE this declaration was written:
    #
    #     per-game rows compared  7,476   moved 0   max delta 0.0
    #     rolling  rows compared  8,064   moved 0   max delta 0.0
    #
    # ZERO rows moved outside 2025. The REASON is itself a finding, recorded
    # below rather than buried: the opponent adjustment is INERT.
    "team_strength_2025_probe": {
        "probe_was_read_only": True,
        "ran_before_the_declaration": True,
        "league_avg_def_2018_2024": -0.003940736770744526,
        "league_avg_def_2018_2025": -0.002092015581006465,
        "per_game_rows_compared": 7476,
        "per_game_rows_moved_outside_2025": 0,
        "rolling_rows_compared": 8064,
        "rolling_rows_moved_outside_2025": 0,
        "max_absolute_delta_outside_2025": 0.0,
    },
    # A PRE-EXISTING DEFECT, FOUND BY THE PROBE, DELIBERATELY NOT FIXED HERE.
    #
    # opp_adj_<metric> equals the RAW metric for all 8,564 per-game rows: the
    # has_enough gate in _adjust_per_game_epa never fires, so the league
    # average is never added and the "opponent-adjusted" columns are rolling
    # averages of UNADJUSTED EPA. It predates this plan and has nothing to do
    # with the three corrections above. Fixing it would move every season
    # 2018-2024 and blow straight past this declaration, which is exactly the
    # scope boundary that keeps a rung attributable. Recorded so the finding
    # is not lost, and carried into the SUMMARY's deferred items.
    "pre_existing_finding_opponent_adjustment_is_inert": {
        "measured_on": "2026-09-14",
        "rows_where_opp_adj_equals_raw": 8564,
        "rows_compared": 8564,
        "mechanism": (
            "features/opponent_adj._adjust_per_game_epa gates the adjustment "
            "on has_enough = game_count >= min_opponent_games, and the count "
            "arrives NaN from the opponent merge, so NaN >= 4 is False for "
            "every row and the raw value is kept"
        ),
        "in_scope_for_this_plan": False,
        "why_not": (
            "it predates this plan, it is unrelated to the three input "
            "corrections, and fixing it would move seasons 2018-2024 -- "
            "outside this rung's declared change set"
        ),
    },
    # Ruling N2, restated rather than inherited.
    "ok_required_unconditionally": True,
    "remedy_for_an_out_of_family_move": (
        "STOP and report. A legitimate out-of-family move is a NEW declared "
        "family in a follow-up rung, never a footnote on this one, and never "
        "an edit to this declaration after the diff has been seen"
    ),
    # WHAT THIS RUNG STILL DOES NOT DO. Unchanged from rung 1: Phase 33's
    # Wave 15 owns the re-fit, and nothing here touches a model.
    "no_model_refit": True,
    "no_gate_run": True,
}

# -------------------------------------------------------------------------
# THE DECLARED BLAST RADIUS OF RUNG 3.
#
# NARROWER THAN RUNG 1'S BY ONE PATH, and the difference is deliberate rather
# than an omission. Rung 1 rebuilt silver contextual_features because the
# stadium_id routing correction changed it. NONE of rung 3's three
# corrections touches the contextual builder, so re-running it would add an
# undeclared cause to the diff for no gain -- and contextual_features.parquet
# moving here would be a FINDING, not a formality.
#
# nfl_predictions.duckdb IS in the set, for the same reason it was in rung
# 1's: build_weather.py and build_features.py both write through
# data.storage.save_dataframe, whose save_to_db defaults True, so both halves
# move.
#
# A FILE OUTSIDE THIS SET IS A FINDING TO REPORT, NEVER A REASON TO WIDEN THE
# SET.
# -------------------------------------------------------------------------

PHASE331_RUNG3_EXPECTED_CHANGED_FILES: tuple[str, ...] = (
    "gold/features_ats.parquet",
    "gold/features_ou.parquet",
    "gold/features_wp.parquet",
    "nfl_predictions.duckdb",
    "silver/weather_features.parquet",
)

# The TWO commands this rung runs, in order, and the one it deliberately does
# NOT. Ruling N3's discipline: ONE exact command per rebuilt artifact, quoted
# from PIPELINE.md, with --all-seasons rather than --season because
# --all-seasons REPLACES the tables while a scoped build MERGES latest-wins.
PHASE331_RUNG3_RUNBOOK: tuple[dict[str, str], ...] = (
    {
        "artifact": "data/silver/weather_features.parquet",
        "command": "uv run python scripts/build_weather.py --all-seasons",
        "why": (
            "the precipitation correction lives in the FULL weather builder, "
            "which writes this table; gold reads it rather than re-deriving "
            "from silver weather"
        ),
    },
    {
        "artifact": "data/gold/features_{wp,ats,ou}.parquet",
        "command": "uv run python scripts/build_features.py --all-seasons",
        "why": (
            "the coverage-flag normalization exemption and the per-game "
            "season pool both live in the gold build"
        ),
    },
)

PHASE331_RUNG3_DELIBERATELY_NOT_RUN: dict[str, str] = {
    "uv run python scripts/build_contextual.py --all-seasons": (
        "none of the three input corrections touches the contextual builder. "
        "Re-running it would add an undeclared cause to the diff for no gain, "
        "and silver/contextual_features.parquet is deliberately ABSENT from "
        "PHASE331_RUNG3_EXPECTED_CHANGED_FILES so a move there is a FINDING"
    )
}


# -------------------------------------------------------------------------
# WHAT RUNG 3 ACTUALLY MOVED.
#
# APPENDED by Plan 33.1-07 Task 4 on 2026-09-14, AFTER the rebuild. Nothing
# above this line was edited -- PHASE331_RUNG3_DECLARATION is byte-unchanged,
# which is what makes it a prediction rather than a transcription.
#
# EVERY PREDICTION HELD. 32 non-clock columns moved, identical in all three
# matrices, and all 32 decomposed into the three declared families with ZERO
# unattributed. Nothing was added, nothing removed, rows unchanged at 6,499,
# widths unchanged at 195 / 196 / 195.
#
# THE DISCRIMINATION HELD TOO, and that is the part worth reading twice. The
# twelve opponent-adjusted columns are declared for 2025 AND NO OTHER SEASON,
# and the diff reports them moving in 2025 and no other season. Had the
# league-average scalar reached the output -- it moves when the pool widens --
# they would have moved in every season 2018-2024 as well, been UNATTRIBUTED,
# and blocked. The declaration would have refused the rebuild rather than
# absorbing it.
# -------------------------------------------------------------------------

PHASE331_RUNG3_MEASURED: dict[str, object] = {
    "measured_on": "2026-09-14",
    "measured_by": "Plan 33.1-07 Task 4",
    "rung": 3,
    "rung_prefix": "p331_",
    "ok": True,
    "blocking": False,
    "failures": 0,
    "unattributed_per_matrix": 0,
    "identical_residual_across_all_three_matrices": True,
    # The BASELINE this rung was judged against, and the proof it was not
    # stale: p331_rung2.json is BYTE-IDENTICAL to p331_rung1.json, because the
    # follow-up rung re-judged the rung0 -> rung1 transition and rebuilt
    # nothing. Gold had not moved between the two.
    "baseline_document": "p331_rung2.json",
    "baseline_sha256": (
        "8bc804b0bcce45b1323a88adca3c2063d38fcd8994d0f563ea7d1bb75e9e18ea"
    ),
    "baseline_is_byte_identical_to_p331_rung1": True,
    "ladder_precheck_passed_before_the_rebuild": True,
    "ladder_preserved_to_a_second_directory": "outputs/fingerprints/preserved/",
    # THE SHAPE. Every structural prediction held.
    "columns_added": (),
    "columns_removed": (),
    "widths_before": (195, 196, 195),
    "widths_after": (195, 196, 195),
    "rows_before": (6499, 6499, 6499),
    "rows_after": (6499, 6499, 6499),
    "non_clock_moves": 32,
    "build_clock_moves": ("feature_timestamp",),
    # THE PER-FAMILY SPLIT, identical in each of the three matrices.
    "changed_by_family": {
        "weather": 19,
        "weather_widening": 1,
        "team_strength_2025": 12,
    },
    "weather_family_columns_moved": (
        "defensive_advantage",
        "extreme_weather",
        "home_weather_advantage",
        "is_dry",
        "is_rain",
        "is_snow",
        "passing_efficiency",
        "precip_heavy",
        "precip_impact_score",
        "precip_light",
        "precip_mm",
        "precip_moderate",
        "precip_none",
        "rushing_advantage",
        "scoring_reduction",
        "turnover_multiplier",
        "weather_coverage",
        "weather_game",
        "weather_severity_score",
    ),
    # Ruling N2: `ok` was True UNCONDITIONALLY, with nothing to supplement.
    "unattributed_with_reason": {},
    # THE RUNBOOK, as run. TWO commands, not three: contextual_features was
    # deliberately NOT rebuilt (see PHASE331_RUNG3_DELIBERATELY_NOT_RUN) and
    # the digest bracket confirms it did not move.
    "commands_run": (
        {
            "artifact": "data/silver/weather_features.parquet",
            "command": "uv run python scripts/build_weather.py --all-seasons",
            "wall_clock_seconds": 6.3,
        },
        {
            "artifact": "data/gold/features_{wp,ats,ou}.parquet",
            "command": "uv run python scripts/build_features.py --all-seasons",
            "wall_clock_seconds": 591.5,
        },
    ),
    "rebuilt_with_network_denied": True,
    "network_guard_proven_on_a_deliberate_probe_first": True,
    "network_guard_layers_proven": (
        "scripts.backfill_historical_weather.fetch_game_weather -> "
        "ArchiveReachedUnderDenyNetwork",
        "scripts.backfill_historical_weather._probe_archive_day -> "
        "ArchiveReachedUnderDenyNetwork",
        "scripts.ingest_weather.fetch_game_forecast -> ForecastReachedUnderDenyNetwork",
        "socket.socket -> SocketOpenedUnderDenyNetwork",
    ),
    # THE DIGEST BRACKET, closed. EXACTLY the declared five paths moved -- all
    # REWRITTEN, none ADDED, none REMOVED, no MIXED key -- and
    # silver/contextual_features.parquet did NOT move, which is what the
    # declaration predicted for the builder it deliberately did not run.
    "declared_changed_files_verdict": "exact match, 5 of 5, REWRITTEN only",
    "contextual_features_did_not_move": True,
    "artifacts_tree_unchanged": True,
    "artifacts_files_compared": 159,
    # -----------------------------------------------------------------
    # THE THREE ACCEPTANCE MEASUREMENTS, each stated as a before/after pair
    # and each reported at the value MEASURED rather than the value hoped for.
    # -----------------------------------------------------------------
    "acceptance_precip_mm_non_null_rows": {
        "before": 1652,
        "after": 6499,
        "of": 6499,
        "note": (
            "1,652 was exactly the indoor games, which take the dome branch "
            "and never reached the gate. All 4,847 outdoor games now carry the "
            "measured rainfall"
        ),
    },
    "acceptance_outdoor_columns_recovered": {
        "outdoor_games": 4847,
        "columns_listed": 20,
        "columns_now_populated_for_outdoor_games": 18,
        "columns_still_null_for_outdoor_games": ("precip_prob", "raw_precip_prob"),
        "why_those_two_stay_null": (
            "they ARE the forecast probability. ERA5 reanalysis does not "
            "report one and COVERAGE.md records the opt-out, so the owner's "
            "ruling forbids inventing one. Two columns staying NULL is that "
            "ruling being honoured, not a shortfall -- and the value they "
            "would otherwise have carried is precisely the fabrication this "
            "phase exists to delete"
        ),
    },
    # A DIVERGENCE FROM THE STATED ACCEPTANCE NUMBER, RECORDED AS ONE.
    #
    # The acceptance criterion read "weather_coverage must VARY". It does not:
    # it reads 1.0 on all 6,499 rows. That expectation rested on a diagnosis
    # that the 1.0 path never fired. The 1.0 path DID fire -- silver carried
    # 1.0 on every row all along -- and what destroyed the flag was the z-score
    # of a constant column, which maps it to 0.0.
    #
    # So the column moved from a CONSTANT FALSEHOOD to a CONSTANT TRUTH. 0.0
    # means NO OBSERVATION and was wrong for all 6,499 games; 1.0 means
    # OBSERVED and is right for all 6,499, because the ERA5 backfill covered
    # the whole corpus (weather_source = 'archive' on 6,499 of 6,499 rows, and
    # zero rows carry an all-null core observation).
    #
    # A flag cannot vary over a population with no variation in it. What is
    # proven instead, by unit test rather than by gold's own variance, is that
    # the column WOULD read 0.0 for an absent observation and that the two
    # levels survive to gold distinguishable -- which is the property R5 needs
    # and the property a constant 0.0 could never have demonstrated.
    "acceptance_weather_coverage": {
        "expected_by_the_acceptance_criterion": "must VARY",
        "measured": {1.0: 6499},
        "diverges_from_the_stated_expectation": True,
        "why": (
            "every one of the 6,499 games HAS a real ERA5 observation "
            "(weather_source = 'archive' on all 6,499; zero rows carry an "
            "all-null core observation), so a coverage flag over this corpus "
            "is constant by construction. The change is from a constant "
            "FALSEHOOD (0.0, which means NO OBSERVATION) to a constant TRUTH"
        ),
        "variation_is_proven_by_test_instead": (
            "tests/unit/test_weather_coverage_flag_survives_normalization.py"
            "::TestTheFlagIsTheOneColumnItsOwnLevelsAreTheMeaningOf"
            "::test_a_varying_flag_keeps_both_of_its_levels_distinguishable"
        ),
    },
    "acceptance_opp_adj_distinct_values_in_2025": {
        "before": 2,
        "after": 285,
        "rows_in_2025": 285,
        "comparable_seasons": {"2023": 285, "2024": 285},
        "all_twelve_columns": True,
    },
    # THE TRAINER SMOKE FIT. READ-AND-FIT-IN-MEMORY, `save()` never called,
    # artifacts/ verified unchanged at 159 files afterwards.
    "trainer_smoke_feature_counts": {"wp": 20, "ats": 25, "ou": 25},
    "trainer_smoke_fitted_models": {
        "wp": "Pipeline",
        "ats": "XGBRegressor",
        "ou": "XGBRegressor",
    },
    "trainer_smoke_wrote_no_artifact": True,
    "trainer_smoke_tune": False,
    "trainer_smoke_partition": (
        "TemporalSplitConfig.default() -- train 2018-2019, hp_val 2020, "
        "holdout 2021-2024"
    ),
    # AN OBSERVATION, NOT A TARGET, and it is repeated here because the
    # temptation to read it as a result is exactly what the phase's own
    # prohibition R8 forbids. Whether a target selects weather is Wave 15's
    # measurement; this phase predicts nothing about it and claims no accuracy
    # improvement from it.
    "trainer_smoke_weather_columns_selected": {
        "wp": (
            "passing_difficulty",
            "passing_efficiency",
            "precip_mm",
            "scoring_reduction",
            "turnover_multiplier",
            "wind_high",
        ),
        "ats": (
            "apparent_temp_f",
            "kicking_difficulty",
            "precip_light",
            "scoring_reduction",
            "temp_f",
            "temp_mild",
            "weather_game",
        ),
        "ou": (
            "cold_impact_score",
            "home_weather_advantage",
            "is_snow",
            "kicking_difficulty",
            "passing_difficulty",
            "passing_efficiency",
            "precip_light",
            "precip_mm",
            "precip_moderate",
            "precip_none",
            "scoring_reduction",
            "turnover_multiplier",
            "weather_severity_score",
        ),
    },
    "trainer_smoke_weather_selection_is_an_observation_not_a_target": True,
    "trainer_smoke_withheld_zero_variance_columns": 45,
    # WHAT THIS RUNG STILL DOES NOT DO.
    "no_model_refit": True,
    "no_gate_run": True,
    "latest_json_byte_identical": True,
}


# -------------------------------------------------------------------------
# THE GOLD WEATHER CONSTANCY, AFTER. R5'S PAIR, SECOND HALF.
#
# APPENDED by Plan 33.1-07 Task 4 on 2026-09-14, after the rung-3 rebuild.
# GOLD_WEATHER_CONSTANCY_MEASUREMENT above is BYTE-UNCHANGED: R5's acceptance
# is a measured before/after PAIR, and a pair with one half overwritten is a
# single number wearing a pair's clothes.
#
# THE HEADLINE, in ordinary words: the weather columns in gold used to be the
# same number for every game in every window that mattered. They now vary,
# because they are measurements of real weather instead of one fabricated
# temperature repeated 6,485 times.
#
#   BEFORE   45 of 46 weather columns EXACTLY CONSTANT in all three windows
#   AFTER     4 of 47 constant; 43 vary
#
#   BEFORE   the O/U model's 17 weather features 17-of-17 constant in its own
#            2018-2019 train window AND in the 2021-2024 gate holdout
#   AFTER    1 of 17 in both
#
# THE FOUR THAT ARE STILL CONSTANT ARE CONSTANT FOR HONEST REASONS, and each
# is named rather than left as a residual:
#
#   precip_prob, raw_precip_prob -- the FORECAST probability. ERA5 reanalysis
#     reports what happened, never what was expected, so there is nothing to
#     put here and inventing one is what this phase exists to stop. These two
#     are NaN for every outdoor game and a genuine 0.0 for the 1,652 domes.
#   weather_coverage -- constant 1.0 because every one of the 6,499 games HAS
#     a real observation. A coverage flag cannot vary over a corpus with full
#     coverage; what changed is that it now says so. Before rung 3 it read 0.0
#     on all 6,499 rows, which is the value that means NO OBSERVATION.
#   extreme_weather -- the severity >= 0.8 indicator. No game in 2002-2025
#     reaches that threshold on real ERA5 readings. That is a fact about the
#     weather, not about the pipeline.
#
# AND THE ONE STILL CONSTANT AMONG THE O/U 17 IS raw_precip_prob -- the same
# forecast probability, for the same reason.
# -------------------------------------------------------------------------

GOLD_WEATHER_CONSTANCY_AFTER: dict[str, object] = {
    "measured_on": "2026-09-14",
    "measured_by": "Plan 33.1-07 Task 4",
    "supersedes": "GOLD_WEATHER_CONSTANCY_MEASUREMENT",
    "supersedes_note": (
        "SUPERSEDES means 'is the AFTER half of', never 'replaces'. The before "
        "slot is byte-unchanged and both halves are required to read R5's "
        "acceptance, which is a measured PAIR"
    ),
    "measured_after": "the Plan 33.1-07 rung-3 input-correction rebuild",
    # 47 here against the before slot's 46, and the difference is exactly one
    # named column -- see divergence_from_recorded_claims below.
    "weather_columns_counted": 47,
    "populations": {
        # Identical in all three gold matrices, so one entry per population
        # rather than three that agree -- the same shape the before slot uses.
        "ats_train_2015_2019": {
            "rows": 1335,
            "constant": 4,
            "varying": 43,
            "constant_columns": (
                "extreme_weather",
                "precip_prob",
                "raw_precip_prob",
                "weather_coverage",
            ),
        },
        "wp_ou_train_2018_2019": {
            "rows": 534,
            "constant": 4,
            "varying": 43,
            "constant_columns": (
                "extreme_weather",
                "precip_prob",
                "raw_precip_prob",
                "weather_coverage",
            ),
        },
        "gate_holdout_2021_2024": {
            "rows": 1139,
            "constant": 4,
            "varying": 43,
            "constant_columns": (
                "extreme_weather",
                "precip_prob",
                "raw_precip_prob",
                "weather_coverage",
            ),
        },
        "all_2002_2025": {
            "rows": 6499,
            "constant": 4,
            "varying": 43,
            "constant_columns": (
                "extreme_weather",
                "precip_prob",
                "raw_precip_prob",
                "weather_coverage",
            ),
        },
    },
    # Why each surviving constant is HONEST rather than a residual defect.
    "why_the_four_are_still_constant": {
        "precip_prob": (
            "the FORECAST probability. ERA5 reanalysis does not report one and "
            "COVERAGE.md records the opt-out, so it is NaN for every outdoor "
            "game and a genuine 0.0 for the 1,652 domes"
        ),
        "raw_precip_prob": "the same reading, un-normalized",
        "weather_coverage": (
            "constant 1.0 because all 6,499 games have a real observation "
            "(weather_source = 'archive' on 6,499 of 6,499). A coverage flag "
            "cannot vary over a fully-covered corpus; before rung 3 it read "
            "0.0 on every row, which is the value that means NO OBSERVATION"
        ),
        "extreme_weather": (
            "the weather_severity_score >= 0.8 indicator. No game in 2002-2025 "
            "reaches that threshold on real readings -- a fact about the "
            "weather, not about the pipeline"
        ),
    },
    # THE HEADLINE PAIR. Both halves printed by the failing message in
    # tests/integration/test_gold_weather_constancy_after.py, because R5's
    # acceptance is a pair rather than a bare after.
    "ou_weather_features_constant": {
        "ou_train_2018_2019": (1, 17),
        "gate_holdout_2021_2024": (1, 17),
        "all_2002_2025": (1, 17),
        "ats_train_2015_2019": (1, 17),
        "the_one_still_constant": ("raw_precip_prob",),
    },
    "ou_weather_features_constant_before": {
        "ou_train_2018_2019": (17, 17),
        "gate_holdout_2021_2024": (17, 17),
        "all_2002_2025": (6, 17),
    },
    "raw_temp_f_imputed": {
        "default_value": 65.0,
        "rows_total": 6499,
        "rows_at_default_before": 6485,
        "rows_at_default_after": 6,
        "distinct_values_after": 700,
        "non_null_after": 4847,
        "note": (
            "6 rows measuring exactly 65.0F is a handful of games that "
            "genuinely were that temperature, not a surviving default. The "
            "1,652 NULL rows are the domes, where there is no outdoor "
            "temperature to report"
        ),
    },
    # THE NO-STAND-IN PROPERTY, asserted POSITIVELY via per-column NaN counts
    # rather than by the absence of one literal. A test that only checked "no
    # cell equals 65.0" would pass against a column median-filled with a new
    # number, which is SPEC prohibition 1 wearing a better label.
    "indoor_games": 1652,
    "outdoor_games": 4847,
    "nan_counts": {
        "temperature_derived_columns": 13,
        "nan_per_temperature_derived_column": 1652,
        "identical_in_all_three_matrices": True,
        "composite_columns": 7,
        "nan_among_indoor_games_per_composite_column": 0,
        "composite_note": (
            "the seven composites keep a genuine 0.0 indoors -- 'weather "
            "reduced scoring by nothing' is a TRUE statement about a covered "
            "indoor game, not a stand-in. The two populations are asserted "
            "SEPARATELY so the distinction is proven rather than described"
        ),
    },
    "weather_coverage_value_counts": {"1.0": 6499},
    "gold_widths": (195, 196, 195),
    "gold_rows": (6499, 6499, 6499),
    "divergence_from_recorded_claims": {
        "weather_columns_counted": (
            "DENOMINATOR DIVERGED BY EXACTLY ONE NAMED COLUMN, and the column "
            "is weather_coverage. The before slot counted 46 on 2026-09-12, "
            "BEFORE the Plan 33.1-07 rung-1 rebuild added the coverage flag to "
            "silver weather_features; this slot counts 47 on the same rule "
            "afterwards. Same rule, one more column in the source it reads. "
            "Both figures are recorded and the earlier one is not overwritten."
        ),
        "ou_all_2002_2025": (
            "REPRODUCED AND IMPROVED. The before slot recorded 6 of 17 "
            "constant across the full span -- the fourteen real 2024 Week 6 "
            "rows were the only real weather anywhere. After: 1 of 17."
        ),
        "indoor_columns_gaining_nan": (
            "WEATHER_NULL_STATE_MATRIX records 13 (the temperature and "
            "temperature-impact groups). MEASURED: those 13 each carry exactly "
            "1,652 NaN, AND raw_humidity_pct carries 1,652 as well -- a "
            "FOURTEENTH column, which Ruling J's own table also marks 'null' "
            "for covered_indoor under its separate 'humidity' group. The "
            "recorded 13 counts two of the six groups; the measurement counts "
            "three. Both are recorded and neither is overwritten."
        ),
        "raw_humidity_pct_nan": 1652,
    },
    # WHAT THIS MEASUREMENT IS NOT (SPEC R8). It is a statement that the
    # weather columns now carry real measurements instead of one repeated
    # fabrication. It is NOT a claim that any model is more accurate. No model
    # was re-fit, no gate was run, and artifacts/latest.json is byte-unchanged.
    "is_not_an_accuracy_claim": True,
    "no_model_refit": True,
    "no_gate_run": True,
}

# ---------------------------------------------------------------------------
# WHAT THE RUNG-3 GOLD REBUILD TURNED RED -- Plan 33.1-08 Task 1.
#
# APPENDED on 2026-09-14, after Plan 33.1-07's rung-3 rebuild (commit cc5f0e7).
#
# THE INSTRUMENT IS NOT THE ONE THE PLAN ASKED FOR, AND THAT IS RECORDED RATHER
# THAN HIDDEN. Plan 33.1-08 Task 1 as written asked for three whole-tier runs
# (tests/unit, tests/integration, tests/api + tests/test_*.py) and their three
# summary lines verbatim. A STANDING OWNER INSTRUCTION dated 2026-09-14 -- later
# than the plan, and recorded as the first entry under `## Decisions` in
# .planning/STATE.md -- forbids exactly that: no bare pytest, no directory sweep,
# targeted modules only. The instruction outranks the plan because it is later
# and because it is the owner's.
#
# So the enumeration was taken over a MEASURED BLAST RADIUS instead of a blind
# sweep. The radius was derived mechanically, not guessed: every test module in
# the repository that reads data/gold/*.parquet or invokes a harness that scores
# gold (score_deployed_artifacts, run_ou_divergence_diagnosis,
# run_signal_lift_screen, run_backtest, load_gold, load_feature_matrix), plus the
# four modules that host the five registered tripwires, plus the two modules
# already identified as casualties by Plan 33.1-07. 44 modules, run as explicit
# module lists.
#
# A gold rebuild can only turn a test red through gold. A module that never reads
# gold and never scores against it cannot be a casualty of one. That is the
# argument for the radius, and it is the argument a reader should attack if they
# think this enumeration is incomplete.
#
# WHAT THIS MEASUREMENT DOES NOT ESTABLISH: any statement about the suite as a
# whole. The three tiers were not run. A tier line is not recorded here because
# one was not measured, and carrying a count over from an earlier run would be
# the exact defect PRE_PHASE_TIER_LINES exists to prevent.
# ---------------------------------------------------------------------------

# The summary lines of the runs that WERE made, verbatim, each beside the module
# list that produced it. The instrument is part of the measurement.
GOLD_REBUILD_TRIAGE_RUNS: tuple[tuple[str, str], ...] = (
    (
        "the five repo-root readout guards (ou_divergence_diagnosis_md, "
        "signal_lift_readout_md, line_movement_readout_md, gated_refit_readout_md, "
        "profitability_readout_md)",
        "107 passed, 31 warnings in 5.67s",
    ),
    (
        "test_activation_parity.py + test_temporal_display_columns.py "
        "(the predeclared parity module and the already-identified casualty)",
        "6 failed, 34 passed, 31 warnings in 5.99s",
    ),
    (
        "20 gold-reading unit modules (bet_list_entry_point, bet_list_schema, "
        "build_features_gold_write, build_features_wr06_bounds, "
        "data_boundary_digest, data_qa_duckdb_parquet, empty_week_refusals, "
        "evidence_skip_visibility, feature_selection_stability, group_gate, "
        "output_path_guards, pipeline_model_validation, "
        "provisional_training_refusal, required_artifacts_currency, "
        "storage_atomic_write, train_exclude_groups, weekly_decision_frame, "
        "write_guard_marker_scope, precipitation_from_measurement, "
        "team_form_per_game_season_coverage)",
        "443 passed, 162 warnings in 108.57s (0:01:48)",
    ),
    (
        "the three tripwire-host integration modules "
        "(gate_baseline_byte_identity, n01_resync_control, promote_models)",
        "4 failed, 42 passed, 57 warnings in 27.73s",
    ),
    (
        "the single gold_rebuild_attribution tripwire node, run alone to confirm "
        "it is still red for its recorded reason",
        "1 failed, 31 warnings in 8.77s",
    ),
    (
        "9 gold-reading integration and api modules "
        "(data_boundary_guard_arming, diag_diagnosis, gold_write_scope, "
        "ingest_2025_odds, lift_validation, ou_divergence, ou_monetization_roi, "
        "api/cache_game_context, api/cold_start_bet_list_recovery)",
        "4 failed, 182 passed, 8 skipped, 8 xfailed, 499 warnings in 297.26s (0:04:57)",
    ),
    (
        "9 modules that invoke a gold-scoring harness without naming a gold path "
        "(friday_prediction_step, audit_stage_runner, group_gate_determinism, "
        "signal_lift, backtest_tuning_identity, blend_backtest, elo_gold_features, "
        "promote_models_tuned_path, runbook_md)",
        "206 passed, 37 warnings in 89.26s (0:01:29)",
    ),
)

# Every node id that was red after the rung and is NOT one of the five registered
# deliberate tripwires, with its disposition.
#
# THE DISPOSITION VOCABULARY IS CLOSED, and deliberately does not include a
# fourth value meaning "absorbed": no id here was dispositioned by adding it to
# DELIBERATE_TRIPWIRE_NODE_IDS, which stays byte-unchanged at five entries.
#
#   SUPERSEDED         a point estimate or anchor MEASURED on pre-rung gold. The
#                      reading is not wrong about the gold it was taken on; it is
#                      no longer about the gold that exists.
#   DEFECT             a real breakage introduced by the rung.
#   EXPECTED-BY-DESIGN a control that asserts a state this phase deliberately
#                      changed, and that is doing exactly what it was built to do
#                      by saying so.
GOLD_REBUILD_NEWLY_RED: tuple[tuple[str, str], ...] = (
    (
        "tests/unit/test_temporal_display_columns.py::TestRealGold::test_the_wp_feature_set_is_free_of_nan",
        "SUPERSEDED",
    ),
    (
        "tests/unit/test_temporal_display_columns.py::TestRealGold::test_raw_humidity_pct_was_excluded_not_imputed",
        "SUPERSEDED",
    ),
    (
        "tests/unit/test_temporal_display_columns.py::TestRealGold::test_display_columns_are_not_constant_on_live_gold[features_wp]",
        "SUPERSEDED",
    ),
    (
        "tests/unit/test_temporal_display_columns.py::TestRealGold::test_display_columns_are_not_constant_on_live_gold[features_ats]",
        "SUPERSEDED",
    ),
    (
        "tests/unit/test_temporal_display_columns.py::TestRealGold::test_display_columns_are_not_constant_on_live_gold[features_ou]",
        "SUPERSEDED",
    ),
    (
        "tests/unit/test_temporal_display_columns.py::TestRealGold::test_only_raw_humidity_pct_carries_nulls_on_live_gold",
        "SUPERSEDED",
    ),
    (
        "tests/integration/test_diag_diagnosis.py::TestDiagDiagnosis::test_backtest_numbers_match_audit_report",
        "SUPERSEDED",
    ),
    (
        "tests/integration/test_ou_divergence.py::TestOuDivergence::test_bias_over_share",
        "SUPERSEDED",
    ),
    (
        "tests/integration/test_gold_write_scope.py::TestTheIdentityMigrationMovesNoGoldColumn::test_the_recorded_reference_matches_todays_production_gold",
        "EXPECTED-BY-DESIGN",
    ),
    (
        "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_the_negative_mean_season_set_is_still_exactly_2022",
        "EXPECTED-BY-DESIGN",
    ),
)

# One line per row above, saying WHY, with the measured numbers that decided it.
# Keyed by the same node id so a reader cannot read a disposition without its
# reason, and so a later plan can look one up without re-deriving it.
GOLD_REBUILD_NEWLY_RED_REASONS: dict[str, str] = {
    "tests/unit/test_temporal_display_columns.py::TestRealGold::test_the_wp_feature_set_is_free_of_nan": (
        "A Phase-30 anchor over live gold. Real weather brings real NULLs: the "
        "1,652 dome games have no outdoor temperature to report, so the "
        "temperature-derived columns are NaN there by construction. The anchor "
        "describes gold in which every weather cell carried the fabricated 65.0."
    ),
    "tests/unit/test_temporal_display_columns.py::TestRealGold::test_raw_humidity_pct_was_excluded_not_imputed": (
        "Same anchor family. raw_humidity_pct now carries 1,652 NaN rather than "
        "the single shape the Phase-30 anchor recorded."
    ),
    "tests/unit/test_temporal_display_columns.py::TestRealGold::test_display_columns_are_not_constant_on_live_gold[features_wp]": (
        "Anchored distinct-value counts measured in Phase 30. raw_temp_f read 15 "
        "distinct values then and reads 700 now. Already red BEFORE Plan "
        "33.1-07's session began -- it went stale at rung 1, the first rebuild to "
        "bring real weather to gold, not at rung 3."
    ),
    "tests/unit/test_temporal_display_columns.py::TestRealGold::test_display_columns_are_not_constant_on_live_gold[features_ats]": (
        "The features_ats parametrisation of the same anchor."
    ),
    "tests/unit/test_temporal_display_columns.py::TestRealGold::test_display_columns_are_not_constant_on_live_gold[features_ou]": (
        "The features_ou parametrisation of the same anchor."
    ),
    "tests/unit/test_temporal_display_columns.py::TestRealGold::test_only_raw_humidity_pct_carries_nulls_on_live_gold": (
        "The anchor names raw_humidity_pct as the ONLY null-bearing display "
        "column. Thirteen temperature-derived columns now carry 1,652 NaN each, "
        "for the domes. The module's own comments say a legitimate weather "
        "backfill must update these DELIBERATELY, which is why re-anchoring is "
        "not a side effect of some other plan's commit."
    ),
    "tests/integration/test_diag_diagnosis.py::TestDiagDiagnosis::test_backtest_numbers_match_audit_report": (
        "A point estimate pinned to moving gold, the D29-06-02 mistake. WP "
        "pooled accuracy measures 0.6769095697980685 against an AUDIT-REPORT.md "
        "anchor of 0.6821773485513608; the drift is 0.005267778753292318 against "
        "a tolerance of 0.005. It fails by 0.00027 -- it is a stale anchor, not a "
        "collapse. AUDIT-REPORT.md is a published reading and is NOT rewritten."
    ),
    "tests/integration/test_ou_divergence.py::TestOuDivergence::test_bias_over_share": (
        "The same class of failure, far larger. Pooled over-share measures "
        "0.8886844526218951 against the OU-DIVERGENCE-DIAGNOSIS.md anchor of "
        "0.727, a drift of 0.1617 against a tolerance of 0.005. The deployed O/U "
        "model consumes 17 weather features and 16 of them were a constant "
        "fabrication before the rung, so its picks moved. "
        "OU-DIVERGENCE-DIAGNOSIS.md is a published Phase-26 reading and is NOT "
        "rewritten."
    ),
    "tests/integration/test_gold_write_scope.py::TestTheIdentityMigrationMovesNoGoldColumn::test_the_recorded_reference_matches_todays_production_gold": (
        "Production gold is (195, 196, 195) wide; GOLD_WIDTHS_BEFORE_ELO_REBUILD "
        "records (194, 195, 194). The one-column delta is weather_coverage, added "
        "to silver weather_features by Plan 33.1-07's rung 1 and pinned by name "
        "in GOLD_WIDTHS_AFTER_WEATHER_RUNG. The reference is a PRE-ELO-REBUILD "
        "anchor owned by Plan 33-14, whose expected change set is built on it. "
        "The test's own message says not to adjust either number to match the "
        "other without deciding which is wrong. Deciding that is Plan 33-14's "
        "call, not this plan's, so the row is recorded and left red."
    ),
    "tests/integration/test_ingest_2025_odds.py::TestTheAppendedATSResidualConstantsStillHold::test_the_negative_mean_season_set_is_still_exactly_2022": (
        "An XPASS under strict=True: an xfail-marked disclosure tripwire that now "
        "PASSES. DEF-31-06 recorded that the live re-score had stopped "
        "reproducing the RATIFIED ATS residual constants, and the xfail carries "
        "the owner's instruction that a RETURN to the ratified values must fail "
        "loudly rather than pass unnoticed. It has returned: on corrected gold "
        "the negative-mean season set is exactly 2022 again, and this red line is "
        "the mechanism telling us so. The tripwire is working. What it now means "
        "is a question for whoever owns DEF-31-06, and it is deliberately NOT "
        "answered here."
    ),
}

# The five registered tripwires, each re-checked against its RECORDED reason
# rather than merely counted as still-red. Two of them are failing for a WIDER
# fact than the one recorded, and a tripwire that fails differently is a
# different fact -- so the new fact is written down.
GOLD_REBUILD_TRIPWIRES_RECHECKED: dict[str, str] = {
    "tests/integration/test_gate_baseline_byte_identity.py::TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::test_the_generated_block_equals_the_committed_block_byte_for_byte": (
        "Still red for its recorded reason: the frozen gate baseline diverges "
        "from a re-score and was deliberately not re-frozen."
    ),
    "tests/integration/test_gold_rebuild_attribution.py::TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::test_no_NON_CLOCK_column_moved_in_a_protected_season": (
        "Still red for its recorded reason: 11 non-clock column-slots moved in "
        "the protected 2021-2024 window (the snapshot_* market family across all "
        "three matrices, plus target_ats and target_ou). Unchanged in kind and in "
        "count from the fact Plan 33.1-07 recorded."
    ),
    "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_every_data_column_reproduces_its_pre_resync_digest_exactly": (
        "Still red, but for a WIDER fact. Recorded reason: a 2021-2024 data "
        "column does not reproduce its pre-resync digest. MEASURED NOW: the "
        "features_ats 2021-2024 slice GAINED a column -- weather_coverage -- and "
        "lost none. The failure is no longer only about values; the column SET "
        "moved too, by exactly the one column Plan 33.1-07's rung 1 added."
    ),
    "tests/integration/test_n01_resync_control.py::TestEvery2021To2024ValueIsByteIdentical::test_the_moved_set_is_exactly_the_build_clock": (
        "Still red, but for a MUCH wider fact. Recorded reason: the moved set is "
        "wider than the build clock alone. MEASURED NOW: roughly 110 columns "
        "moved in the features_ats 2021-2024 slice, against an expected set of "
        "exactly ['feature_timestamp']. The moved set is the entire weather "
        "family (temp_*, wind_*, precip_*, raw_*, the composites), the venue "
        "family, the opponent-adjusted family, the snapshot market family and "
        "target_ats. This is the rung-3 rebuild seen from the resync control's "
        "side and it is expected in kind, but the SET is far wider than when this "
        "tripwire was registered."
    ),
    "tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields": (
        "Still red for its recorded reason: the 47-of-68-field frozen-baseline "
        "divergence, seen from the promotion gate."
    ),
}

# The population Plan 33.1-08 Task 1 PREDECLARED as the expected newly-red set,
# and what was actually found there. Recorded because a triage checked against a
# prediction is worth more than one assembled from whatever happened -- and
# because the prediction was WRONG, which is itself the finding.
GOLD_REBUILD_PREDECLARED_POPULATION_OUTCOME: dict[str, object] = {
    "predicted": (
        "the harness-reproduction classes of the five repo-root readout guards, "
        "plus any test in tests/integration/test_activation_parity.py that pins a "
        "scored value rather than a shape"
    ),
    "found_red_in_that_population": 0,
    "why": (
        "FOUR of the five harness-reproduction classes NO LONGER EXIST. They were "
        "DELETED on 2026-09-12 by owner instruction -- two days before this "
        "measurement and after Plan 33.1-08 was written -- from "
        "test_ou_divergence_diagnosis_md.py, test_signal_lift_readout_md.py, "
        "test_line_movement_readout_md.py and test_gated_refit_readout_md.py. "
        "Each deletion comment names TWO causes, and names Plan 33.1-08 as the "
        "author of the successor design: the classes were not deterministic (a "
        "situational-OU delta measured 0.0074 / 0.4424 / 0.4784 at 4 / 1 / 8 BLAS "
        "threads), and generation-gating was already the agreed answer. The fifth "
        "guard, test_profitability_readout_md.py, never re-ran a harness against "
        "gold: it reads the committed verdict artifact. All 107 assertions across "
        "the five modules pass. test_activation_parity.py passes as well."
    ),
    "consequence": (
        "Every one of the ten rows in GOLD_REBUILD_NEWLY_RED is OUTSIDE the "
        "predeclared population, so by Plan 33.1-08 Task 1's own rule every one "
        "of them is a FINDING with its cause recorded, not a quiet DEFECT label. "
        "None of the ten was found where the plan expected to find it."
    ),
}

# WHAT THIS TRIAGE FOUND ZERO OF, stated positively so the absence is a result
# rather than an omission: not one of the ten rows is dispositioned DEFECT. The
# rung broke nothing that was working. It made stale a set of anchors that were
# measuring gold which no longer exists, and it tripped two controls that exist
# to announce exactly the kind of change it made.
GOLD_REBUILD_DEFECT_COUNT: int = 0

# ---------------------------------------------------------------------------
# THE GOLD GENERATION SEAM -- Plan 33.1-08 Task 2.
#
# APPENDED on 2026-09-14, after Plan 33.1-07's rung-3 rebuild (commit cc5f0e7).
#
# RULING P, and the rule it rests on. `GATED-REFIT-READOUT.md` states the
# governing rule in its own header: nothing published earlier is overwritten, and
# superseded readings stay where they were written, with their dates and their
# reasons. A gold rebuild makes a pinned point estimate false, and there are only
# three honest responses to that -- rewrite the readout (forbidden; it overwrites
# a published reading), delete the guard (forbidden; it removes the anti-rot
# control), or make the harness-reproduction half state which gold generation it
# was measured against and REFUSE TO COMPARE ACROSS GENERATIONS. The third is the
# seam, and it lives in tests/gold_generation.py.
#
# THE SPLIT THAT MAKES THIS SAFE. Only the harness-reproduction half is gated.
# Every document-level assertion -- the file is present at the repo root, the
# content is ASCII, the required sections are present, the forbidden phrases are
# absent -- keeps running unconditionally, because those are the assertions that
# actually guard the DOCUMENT and they do not read gold at all. Gating them would
# turn a real control into a no-op, which is the failure
# tests/unit/test_weather_archive_quarantined.py keeps a non-vacuity control for,
# and tests/unit/test_gold_generation.py::TestTheGateIsReal carries the
# equivalent control here: a MATCHING key must NOT skip.
#
# PLAN 33-14 IS THE NEXT CALLER. Phase 33's Wave 14 Elo rung will move gold
# again and will hit this wall in the identical shape. It should reuse this seam
# -- append a new generation constant, point the affected readings at it -- and
# not rediscover the problem or reach for one of the two forbidden responses.
# ---------------------------------------------------------------------------

# The live gold generation key MEASURED on 2026-09-14, after Plan 33.1-07's rung
# 3, by tests.gold_generation.gold_generation_key(). It is a sha256 over the
# CONTENT digests of the three gold matrices: a rebuild that changes only VALUES
# -- which is exactly what rung 3 did to 32 columns, changing no column name --
# changes this key, where a schema-shaped key would have missed it entirely.
GOLD_GENERATION_AFTER_WEATHER_RUNG: str = (
    "eea0882f4410d22af6e4b54d1c0929a28072325fff9b556a91ca12de900290d1"
)

# The sentinel used as the "expected" generation for every reading measured
# BEFORE the rung.
#
# WHY A SENTINEL AND NOT A REAL KEY. Nobody captured a content key of pre-rung
# gold before it was overwritten, and the bytes are gone -- the rebuild is
# one-way. Inventing a plausible-looking hex string here would be fabricating the
# very kind of record this phase exists to delete. The sentinel is NOT a hex
# digest, cannot equal any output of gold_generation_key(), and therefore makes
# every reading below skip unconditionally until somebody deliberately re-ratifies
# it against a generation they actually measured. That is the correct behaviour
# and it is arrived at honestly rather than by a number chosen to produce it.
GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED: str = (
    "uncaptured-pre-33.1-07-rung-3-gold"
)

# Each guarded reading, and the generation key it was MEASURED against.
#
# Every entry below reads the sentinel, because every one of them was measured on
# gold that no longer exists. None of the six readouts was edited to produce this
# table: the readings stay exactly where they were written.
GOLD_DERIVED_READINGS: tuple[tuple[str, str], ...] = (
    (
        "AUDIT-REPORT.md's WP pooled accuracy anchor of 0.6821773485513608, "
        "re-ratified by the owner on 2026-09-13 and re-asserted by "
        "tests/integration/test_diag_diagnosis.py::TestDiagDiagnosis::"
        "test_backtest_numbers_match_audit_report",
        GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    ),
    (
        "OU-DIVERGENCE-DIAGNOSIS.md's pooled model-over share anchor of 0.727, "
        "re-asserted by tests/integration/test_ou_divergence.py::TestOuDivergence::"
        "test_bias_over_share",
        GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    ),
    (
        "SIGNAL-LIFT-READOUT.md's situational-OU incremental-CLV delta of "
        "-0.3203552582994336, measured 2026-09-05. Its harness-reproduction class "
        "was DELETED on 2026-09-12 by owner instruction for non-determinism, so "
        "there is no live re-run to gate; the reading is recorded here so the "
        "generation it belongs to is not lost with the class",
        GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    ),
    (
        "LINE-MOVEMENT-READOUT.md's lift figures. Its harness-reproduction class "
        "was DELETED on 2026-09-12 by owner instruction, same cause",
        GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    ),
    (
        "GATED-REFIT-READOUT.md's per-target paired CLV figures -- WP +0.006094 "
        "p=0.0142, ATS -0.212797 p=0.0375, O/U -0.487007 p=9.24e-12. Its "
        "harness-reproduction class was DELETED on 2026-09-12 by owner "
        "instruction, same cause",
        GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    ),
    (
        "PROFITABILITY-READOUT.md's one-shot 2025 per-target ROI figures. These "
        "are NOT re-derived from gold by any guard: "
        "tests/unit/test_profitability_readout_md.py compares the document "
        "against the FROZEN committed verdict artifact, which is why that module "
        "did not redden and why it needs no gate. Recorded for completeness, so "
        "a later reader does not mistake its absence from the gated set for an "
        "oversight",
        GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
    ),
)

# The call sites that consume the seam, with what each one guards. A seam with no
# recorded consumers is a seam a later plan will duplicate.
#
# THE LIST IS SHORTER THAN PLAN 33.1-08 EXPECTED, AND WHY IS THE FINDING. The
# plan named the harness-reproduction classes of five readout guards. Four of
# those classes no longer exist -- deleted 2026-09-12, two days before this
# measurement -- and the fifth never read gold. So the seam is applied where the
# casualties ACTUALLY are: the two harness reproductions that genuinely reddened
# on rung-3 gold, both found by Task 1's triage and both dispositioned
# SUPERSEDED. Applying a gate to a class that does not exist would have been
# bookkeeping; applying it to the tests that actually broke is the work.
GOLD_GENERATION_GATED_CALL_SITES: tuple[tuple[str, str], ...] = (
    (
        "tests/integration/test_diag_diagnosis.py::TestDiagDiagnosis::"
        "test_backtest_numbers_match_audit_report",
        "WP pooled accuracy measured 0.6769095697980685 against the re-ratified "
        "anchor 0.6821773485513608, a drift of 0.005267778753292318 past a band "
        "of 0.005. The band is NOT widened and the anchor is NOT edited -- that "
        "test's own docstring forbids both, and records that the owner exercised "
        "the re-ratify route on 2026-09-13. Generation-gating is the third "
        "option: it sets the comparison aside without touching either number, "
        "and an owner re-ratification against a measured generation is what "
        "turns the check back on",
    ),
    (
        "tests/integration/test_ou_divergence.py::TestOuDivergence::"
        "test_bias_over_share",
        "Pooled over-share measured 0.8886844526218951 against the "
        "OU-DIVERGENCE-DIAGNOSIS.md anchor of 0.727, a drift of 0.1617 past a "
        "band of 0.005. The deployed O/U model consumes 17 weather features and "
        "16 of them were a constant fabrication before the rung, so its picks "
        "moved. The published Phase-26 reading is not rewritten",
    ),
)

# ---------------------------------------------------------------------------
# THE O/U WEATHER-CONDITIONS BREAKDOWN -- Plan 33.1-08 Task 4 (D33.1-11).
#
# MEASURED 2026-09-14 by running the COMMITTED run_ou_divergence_diagnosis()
# orchestrator once against post-rung gold, on commit cb39c19. Nothing was
# re-fit. data/ and artifacts/ were digest-bracketed around the run and both
# verified byte-unchanged (472 and 159 files).
#
# WHAT THIS IS. The dome-versus-outdoor and severe-versus-mild breakdown the
# phase context asked for, produced from the harness that was already committed,
# using the two cuts Phase 26 already registered. Those two cuts had never
# produced a graded number: their columns were degenerate on the old gold and the
# harness refused honestly rather than substituting. Correcting the weather made
# one of them gradeable, and re-keying applicability onto the game's own roof
# (Task 3) made the other one gradeable and correct.
#
# WHAT THIS IS NOT. It is not a new comparison, not a significance claim, not a
# gate input, and not a replacement for the Phase-26 readings. Those stay in
# OU-DIVERGENCE-DIAGNOSIS.md exactly where they were written, with their date.
#
# TWO SPLITS, NOT FOUR. Plan 33.1-08 describes four splits -- applicability plus
# three median bands (calm/windy, dry/wet, mild/severe). The harness has TWO. The
# plan's own Ruling O is the reason it must stay that way: CONTEXT's calm versus
# windy and dry versus wet map onto the existing severity split, and no new cut
# is added, because adding bands now would widen a correction family fixed in
# Phase 26 after the fact. Wind and precipitation already feed the severity score
# the single band splits on. Every split that exists is covered; none is added.
# ---------------------------------------------------------------------------

OU_WEATHER_CONDITIONS_BREAKDOWN: dict[str, object] = {
    "measured_on": "2026-09-14",
    "measured_by": "Plan 33.1-08 Task 4, on commit cb39c19",
    "harness": "backtest.ou_divergence.run_ou_divergence_diagnosis (committed, unchanged)",
    "window": "2021-2024 walk-forward holdout",
    "gold_generation_key": (
        "eea0882f4410d22af6e4b54d1c0929a28072325fff9b556a91ca12de900290d1"
    ),
    "deployed_ou_artifact": "ou_20260326_163930",
    "flat_breakeven_hit_rate": 0.5238095238095238,
    "coverage_count": 1087,
    # The RAW stream. Every hit rate below was identical in the BLENDED stream to
    # machine precision, and so was every other cut in the sweep (22 of 22 bucket
    # pairs). That identity is a PRE-EXISTING property of this harness, present
    # before Plan 33.1-08 touched it and not caused by the re-keying; it is
    # recorded here as an observation and deliberately NOT interpreted.
    "buckets": {
        "outdoor": {
            "n": 749,
            "n_graded": 749,
            "hit_rate": 0.4766355140186916,
            "line_clv_mean": 2.887171227082073,
            "vs_flat_breakeven": "below",
            "graded_edge_direction_by_season": (-1, -1, -1, 1),
        },
        "indoor": {
            "n": 338,
            "n_graded": 338,
            "hit_rate": 0.4822485207100592,
            "line_clv_mean": 3.687811394414958,
            "vs_flat_breakeven": "below",
            "graded_edge_direction_by_season": (-1, 0, -1, -1),
        },
        "severe_weather": {
            "n": 543,
            "n_graded": 543,
            "hit_rate": 0.4732965009208103,
            "line_clv_mean": 2.830641681537663,
            "vs_flat_breakeven": "below",
            "graded_edge_direction_by_season": (-1, -1, -1, 1),
        },
        "mild_weather": {
            "n": 544,
            "n_graded": 544,
            "hit_rate": 0.4834558823529412,
            "line_clv_mean": 3.4410534325767967,
            "vs_flat_breakeven": "below",
            "graded_edge_direction_by_season": (-1, -1, -1, -1),
        },
    },
    # Two splits, two counts. BOTH ARE ZERO, and the zero is reported rather than
    # dressed up: in this 2021-2024 window every game carries a weather
    # observation (silver weather_coverage is 1.0 on all 6,499 rows and the roof
    # flag has no nulls), so there was nothing to set aside. The exclusion
    # mechanism is proven by fixture in
    # tests/unit/test_ou_divergence_weather_cut.py, not by this population. A
    # mechanism that happens not to fire on today's data is still the difference
    # between a bucket that is a measurement and a bucket that is partly a gap --
    # the 2025 forward season is where it will start to matter.
    "excluded_missing_by_split": {
        "applicability_outdoor_vs_indoor": 0,
        "severity_severe_vs_mild": 0,
    },
    "split_accounting": {
        "applicability_outdoor_vs_indoor": "749 + 338 + 0 == 1087",
        "severity_severe_vs_mild": "543 + 544 + 0 == 1087",
    },
    "claims": (
        "It required NO re-fit, and it DID require a re-score. "
        "backtest.ou_divergence._deployed_ou_preds calls score_deployed_artifacts "
        "for the O/U target on every run where predictions are not supplied, so "
        "the deployed artifact is loaded and run over gold each time. That is how "
        "the published Phase-26 diagnosis was produced as well. A claim that it "
        "requires no re-score would be one the code does not support.",
        "No new cut was added and no new bucket was created, so no new comparison "
        "entered the Phase-26 correction family. The bucket names, the harness and "
        "the LOCKED BettingSimulator grading path are the ones already committed. "
        "The numbers above are DESCRIPTIVE and carry no p-value.",
        "They were measured on post-Phase-33.1 gold and DO NOT supersede the "
        "Phase-26 readings in OU-DIVERGENCE-DIAGNOSIS.md. Those were measured on "
        "gold whose weather family was a fabricated constant; they stay where they "
        "were written, with their date and their reason, and these numbers are "
        "reported beside them, never instead of them.",
        "Closing-line value remains REPORT-ONLY. The line_clv_mean figures above "
        "are disclosure, not evidence of profitability, and nothing here is an "
        "input to any gate.",
    ),
    # The two ways this cut's DEFINITION differs from the Phase-26 registered cut.
    # Its IDENTITY is preserved -- same harness, same LOCKED simulator, same bucket
    # names, no new trial -- but calling it the existing cut on better data without
    # stating these would be a mislabelling.
    "semantics": {
        "applicability_is_per_game_not_per_venue": (
            "Phase 26's cut split on the VENUE-level venue_outdoor, which encodes "
            "retractable as its own indicator and cannot distinguish an open-roof "
            "game from a closed-roof one at the same stadium. This cut splits on "
            "the game's own roof fact, read read-only from silver "
            "weather_features.weather_affects_game. Across the full 6,499-game "
            "population that moves 621 closed-roof games at the five retractable "
            "stadiums out of the outdoor bucket, where the venue key had put all "
            "749 of their games together."
        ),
        "a_missing_observation_is_excluded_from_the_denominator": (
            "Phase 26's severity band built mild as the bare complement of "
            "severity greater than the median. Every comparison against a missing "
            "value is False, so a game whose weather nobody observed was counted "
            "as a mild-weather game. Both sides of both splits are now built "
            "positively over observed rows and the set-aside count is reported. On "
            "this window the count is zero; the definition is different regardless."
        ),
    },
    "not_comparable_to_a_phase_26_run_of_the_same_cut": (
        "Stated plainly: these numbers are NOT comparable to a hypothetical "
        "Phase-26 run of the same cut. The applicability and coverage definitions "
        "changed, as recorded above, and the underlying weather data changed. A "
        "Phase-26 run never produced a graded result in any case -- the columns "
        "were degenerate and the harness refused, which is the state this phase "
        "removed."
    ),
    # A CONSEQUENCE that must not be buried. Making a pre-registered cut gradeable
    # changes the BH-FDR denominator, because eight entries that were recorded as
    # unavailable with a null p-value are now testable.
    "trial_denominator_note": (
        "OU-DIVERGENCE-DIAGNOSIS.md records n_trials = 36. Today's run reports 44. "
        "The eight new entries are exactly the weather cut's four buckets across "
        "the two streams, which were always REGISTERED and were previously counted "
        "as unavailable with a null p-value because the columns were degenerate. "
        "The registered family did not grow; the part of it that can be tested "
        "did. The arithmetic consequence is that every BH-adjusted p-value in "
        "today's run is computed over a 44-trial denominator rather than a "
        "36-trial one, so today's adjusted p-values are not the published ones. "
        "The published readout is NOT edited."
    ),
    # Both halves of the D33.1-11 hypothesis evidence, so a later readout can quote
    # rather than re-argue. The readout leads with neither half.
    "evidence_for_the_weather_hypothesis": (
        "PUBLISHED Phase 26, on pre-correction gold: the model picked over 790 "
        "times to under 297 and its over picks graded 0.4747, below breakeven.",
        "PUBLISHED Phase 26: a flat per-season bias subtraction FAILED to collapse "
        "the recorded bias, which is what a game-VARYING bias would predict.",
        "PUBLISHED Phase 26: the high-total bucket graded 0.5521, above breakeven, "
        "and high totals skew towards domes and fair conditions where a missing "
        "weather input costs nothing.",
    ),
    "evidence_against_the_weather_hypothesis": (
        "PROFITABILITY-READOUT.md: WP shows the same CLV-positive, ROI-flat shape "
        "with ZERO weather features -- +0.0918 at p 6.8e-34 beside an ROI p of "
        "0.67. Weather cannot be the general explanation for a shape that appears "
        "where weather is not an input at all.",
        "The 2021 under-heavy sign flip recorded in Phase 26 is not explained by "
        "the weather hypothesis.",
        "MEASURED TODAY, on corrected gold, and this one is new: two of the three "
        "FOR facts above are Phase-26 readings taken on fabricated weather, and "
        "they MOVED. The over/under split is now 966 over to 121 under, not "
        "790/297; and the high-total bucket now grades 0.5000, at or below "
        "breakeven, where it graded 0.5521 above it. The published figures are not "
        "rewritten -- they are what Phase 26 measured -- but a hypothesis resting "
        "on them is resting on numbers the correction moved.",
        "MEASURED TODAY: all four weather buckets grade BELOW the flat breakeven, "
        "between 0.4733 and 0.4835, a spread of about one percentage point. "
        "Neither split separates a winning population from a losing one. If "
        "weather were the missing input, the bucket where weather cannot reach the "
        "field is where the model should look best, and it does not.",
    ),
}


# ---------------------------------------------------------------------------
# THE WITNESS FOR PHASE 33.1'S SEASON PARTITION RULE (SPEC R6, D33.1-03).
#
# APPENDED by Plan 33.1-09 Task 1 on 2026-09-14, in a SEPARATE, LATER commit than
# the rule itself. THAT SEPARATION IS THE WHOLE POINT. Nothing above this line was
# edited.
#
# WHY THE ANCHOR LIVES HERE AND NOT INSIDE THE FILE IT WITNESSES (REVIEW-CIRCULAR).
# A file that must CONTAIN and exactly REPRODUCE its own whole-file hash is
# self-referential: writing the hash changes the bytes the hash was computed over,
# so no fixed point exists without a canonical exclusion rule nobody has defined.
# This is the FOURTH use of the outside-witness pattern in this repository, not a
# new idea: Phase 30 proved it on backtest/group_gate.py, Phase 31 reused it on the
# EV-chain pre-registration, and Plan 33.1-05 used it on the weather cross-check.
#
# WHAT IT IS NOT. This does NOT extend Phase 31's pre-registration.
# `backtest.ev_chain_constants.PREREGISTRATION_PATHS` names exactly two paths and a
# test asserts that count; adding to it would redefine a PUBLISHED pre-registration
# after the fact (Ruling L, applied again). Phase 33.1's rule gets its own class in
# tests/unit/test_preregistration_ancestry.py reusing that module's helpers.
#
# THE RELATION ASSERTED: the commit below is a STRICT ancestor of HEAD and of the
# commit that records it, and it touches the rule file and NOTHING ELSE. Task 1
# commits the module alone and Task 2 rewires the eleven consumers afterwards,
# precisely so that assertion can hold.
#
# THE PROHIBITION IS VACUOUS AND THE WITNESS EXISTS ANYWAY. Under D33.1-03 the rule
# is deterministic and scores nothing, so the SPEC's third prohibition ("MUST NOT
# select the training window after observing its scores") has nothing to bite on.
# The anchor is recorded so the property stays CHECKABLE if a future phase ever
# does score a window.
#
# THE DIGEST IS NEWLINE-NORMALIZED -- every CRLF folded to LF before hashing. This
# repository has core.autocrlf=true and no .gitattributes, so the file is LF in the
# git blob and CRLF in a fresh Windows working tree; a digest over raw working-tree
# bytes would pin a value that holds only on the machine that measured it. Verified
# equal by BOTH routes at append time: the normalized working-tree bytes and
# `git cat-file blob <commit>:<path>` produce the same value, and the blob carries
# zero CRLF pairs.
# ---------------------------------------------------------------------------

SEASON_PARTITION_RULE_COMMIT: str = "e7d0ca5b0997f7660acaa7e75420416c4b309dd7"

SEASON_PARTITION_RULE_FILE_SHA256: dict[str, str] = {
    "conf/season_partition.py": (
        "edd580beb0f7df06d4cebbcf10bdf3252ec2747aa7eabe51efa3032899b76acb"
    ),
}

SEASON_PARTITION_RULE_PROVENANCE: dict[str, str] = {
    "plan": "33.1-09",
    "task": "Task 1: write the rule and its evidence, then witness it from outside",
    "date": "2026-09-14",
    "resolved_by": "git log -1 --format=%H -- conf/season_partition.py",
    "hash_basis": (
        "newline-normalized file bytes (CRLF folded to LF); equals the git blob "
        "sha256, verified by both routes at append time"
    ),
    "commit_contents": "exactly conf/season_partition.py and nothing else",
    "does_not_extend": (
        "backtest.ev_chain_constants.PREREGISTRATION_PATHS is UNTOUCHED -- that "
        "tuple defines Phase 31's pre-registration and a test asserts it holds "
        "exactly two paths (Ruling L)"
    ),
    "the_rule": (
        "holdout = the HOLDOUT_SEASON_COUNT most recent completed seasons; hp_val "
        "= the HP_VAL_SEASON_COUNT seasons immediately before those; selection = "
        "SELECTION_WINDOW_FIRST_SEASON through the season before hp_val; final_fit "
        "= every completed season from CORPUS_FIRST_SEASON (D33.1-02)"
    ),
    "on_todays_data": (
        "selection 2018-2022, hp_val 2023, holdout 2024-2025, final fit 2002-2025"
    ),
    "why_it_is_vacuous": (
        "the rule is DETERMINISTIC and scores nothing -- no candidate window is "
        "fitted, ranked or compared -- so the SPEC's third prohibition is "
        "satisfied vacuously rather than by discipline. The anchor exists so the "
        "property stays checkable if a future phase ever does score a window."
    ),
    "the_one_remaining_literal": (
        "LATEST_COMPLETED_SEASON = 2025. Import-time consumers (conf.settings, "
        "TemporalSplitConfig.default, models/train.py argparse, "
        "backtest.engine.BacktestConfig) hold no frame; reading gold at import "
        "time would make the partition depend on whether a rebuild had run, and "
        "deriving it from the calendar would let it move with no commit recording "
        "that it moved. Bumping it is a deliberate one-line commit -- and it "
        "invalidates the digest above, which is the point."
    ),
}


# ---------------------------------------------------------------------------
# THE ELEVEN (NOW SIXTEEN) SITES THAT DEFINE THE SEASON PARTITION, AND HOW EACH
# ONE GETS ITS VALUE.
#
# APPENDED by Plan 33.1-09 Task 2 on 2026-09-14. Nothing above this line was
# edited.
#
# WHY A `mechanism` COLUMN (Ruling S2, Codex 33.1-09 MEDIUM). The plan's first
# draft said all eleven sites "derive" from conf.season_partition. That overstates
# the mechanism for exactly one of them: config/gate.toml is TOML, its
# gate.seasons.holdout is a literal list parsed by tomllib, and no mechanism exists
# by which a TOML file could import a Python rule. It is a GENERATED / PINNED
# MIRROR, kept honest by the agreement test rather than by construction, and
# scripts/sync_gate_holdout.py regenerates exactly that one line. In a milestone
# whose whole point is not overstating mechanisms, the record says which is which.
#
# WHY SIXTEEN ROWS FOR ELEVEN RESEARCH SITES. 33.1-RESEARCH.md section 11.1 lists
# eleven SITES; several of them carry more than one independently-evaluatable
# value (models/train.py has three argparse defaults, TemporalSplitConfig.default()
# has three season lists, backtest.engine.BacktestConfig has three fields). The
# rows below are the EVALUATABLE values, because that is what the agreement test
# can actually read. The research-row mapping is recorded per row so neither
# count has to be re-derived.
#
# ROW 16 IS THE WAVE-7 FOLD-IN. features/team_form.TEAM_FORM_PER_GAME_FIRST_SEASON
# was introduced by commit ad26b30 ("derive the per-game season pool from the data,
# not a literal") as a NAMED, test-pinned literal whose own comment says Plan
# 33.1-09 owns consolidating it here. Leaving it would have left two mechanisms
# side by side, which is the defect this plan exists to remove.
#
# EACH ROW IS (path, symbol, mechanism). `symbol` is an EVALUATABLE expression
# relative to its module, not a line number: line numbers rot and an evaluated
# value cannot.
# ---------------------------------------------------------------------------

SEASON_PARTITION_SITES: tuple[tuple[str, str, str], ...] = (
    # RESEARCH 11.1 row 1 -- the single-source gate constant.
    ("models/deploy_gate.py", "HOLDOUT_SEASONS", "derives"),
    # RESEARCH 11.1 row 3.
    ("conf/settings.py", "BacktestConfig().seasons", "derives"),
    # RESEARCH 11.1 row 4 -- three season lists on one default().
    ("models/temporal.py", "TemporalSplitConfig.default().train_seasons", "derives"),
    ("models/temporal.py", "TemporalSplitConfig.default().hp_val_seasons", "derives"),
    ("models/temporal.py", "TemporalSplitConfig.default().holdout_seasons", "derives"),
    # RESEARCH 11.1 row 5 -- the real source-side surrogate for "all three model
    # configs" (11.4: the artifacts' metadata.json is a RECORD and may not be
    # edited; these argparse defaults decide what a FUTURE run writes).
    ("models/train.py", "--config-train-seasons default", "derives"),
    ("models/train.py", "--config-hp-val-seasons default", "derives"),
    ("models/train.py", "--config-holdout-seasons default", "derives"),
    # RESEARCH 11.1 rows 6, 7 and 8. Row 7 (max_backtest_season) is the site
    # CONTEXT's four-site inventory never named and the strongest 2025-hider in the
    # repository: _load_features drops season > max_backtest_season, so EVERY
    # backtest and gate consumer loading gold through the engine had never seen a
    # 2025 row.
    ("backtest/engine.py", "BacktestConfig().holdout_seasons", "derives"),
    ("backtest/engine.py", "BacktestConfig().max_backtest_season", "derives"),
    ("backtest/engine.py", "BacktestConfig().first_data_season", "derives"),
    # RESEARCH 11.1 row 9 -- the two literals _load_gold_holdout slices on.
    ("scripts/promote_models.py", "_HOLDOUT_FIRST_SEASON", "derives"),
    ("scripts/promote_models.py", "_HOLDOUT_LAST_SEASON", "derives"),
    # RESEARCH 11.1 row 10 -- the ONE site that cannot derive. See the block
    # comment above, and scripts/sync_gate_holdout.py.
    ("config/gate.toml", "gate.seasons.holdout", "generated mirror"),
    # RESEARCH 11.1 row 11 -- the legacy runner. Its two BacktestConfig
    # constructions carried four more literals between them; both now go through
    # one helper that takes the engine's derived defaults.
    (
        "scripts/retrain_models.py",
        "_default_backtest_config().holdout_seasons",
        "derives",
    ),
    # THE WAVE-7 FOLD-IN (commit ad26b30). Not in RESEARCH 11.1 -- it did not exist
    # when that table was written.
    ("features/team_form.py", "TEAM_FORM_PER_GAME_FIRST_SEASON", "derives"),
)

# The partition those sixteen sites must agree on, on TODAY's data. Recorded as
# the AFTER state so a reader does not have to run the rule to know what it
# produced, and so the agreement test has an outside reference rather than only
# comparing the sites to each other (sixteen sites that all derive from one broken
# rule would agree perfectly).
SEASON_PARTITION_AFTER: dict[str, object] = {
    "selection": (2018, 2019, 2020, 2021, 2022),
    "hp_val": (2023,),
    "holdout": (2024, 2025),
    "final_fit_first": 2002,
    "final_fit_last": 2025,
    "final_fit_count": 24,
    "backtest_seasons_first": 2018,
    "backtest_seasons_last": 2025,
    "latest_completed_season": 2025,
    "before": (
        "selection 2018-2019 (WP/OU) or 2015-2019 (ATS), hp_val 2020, holdout "
        "2021-2024, and 2025 absent from every model config"
    ),
    "measured_on": "2026-09-14, Plan 33.1-09 Task 2",
}

# The frozen 2021-2024 set, PINNED from outside the module that declares it
# (models.deploy_gate.FROZEN_BASELINE_SEASONS). Ruling R put the constant in
# deploy_gate.py because validate_gate_config is production code and must not
# import from tests/; this is the test-side pin, which keeps the witness outside
# the witnessed file exactly as every other pinned constant here does.
#
# IT IS NOT THE LIVE PARTITION AND MUST NEVER BECOME IT. config/gate.toml's
# [baseline.*] block is a HISTORICAL RECORD of what Phase 30 measured, frozen over
# these four seasons, with an undischarged 47-of-68-field divergence D33.1-05
# deliberately preserves. validate_gate_config's season-key check is a SHAPE CHECK
# over that record -- per N-05 nothing in the deploy decision reads the block any
# more -- so it is retargeted here rather than being made to agree with a holdout
# that has moved.
FROZEN_BASELINE_SEASONS_PIN: tuple[int, ...] = (2021, 2022, 2023, 2024)


# ---------------------------------------------------------------------------
# THE LIVE PINS THAT WERE UPDATED WHEN THE PARTITION MOVED, AND WHY.
#
# APPENDED by Plan 33.1-09 Task 3 on 2026-09-14. Nothing above this line was
# edited.
#
# NONE OF THESE IS A DELIBERATE TRIPWIRE, AND NONE WAS DELETED. That distinction
# is the point of the slot. A control that stops being true has exactly three
# honest dispositions -- update it with a recorded reason, delete it with a
# recorded reason, or record it as deliberately red -- and only the first is
# correct here. These pinned the PRE-CORRECTION partition; the partition moved by
# owner decision (D33.1-01), so the guards move with it and keep guarding.
# `DELIBERATE_TRIPWIRE_NODE_IDS` is UNCHANGED at five entries and is asserted
# disjoint from this set: a pin quietly relabelled as a tripwire is a control
# neutered rather than updated, which is the failure this record exists to make
# visible.
#
# THE RECORDED REASON, ONCE, SO EACH ROW CAN BE SHORT. Phase 31 held 2025 back as
# the single unburned clean split and then SPENT it: the one-shot pre-registered
# verdict run happened, and PROFITABILITY-READOUT.md records that it cannot be
# repeated. So the condition the strongest of these guards set for ITSELF -- "a
# future milestone genuinely needing 2025 in the gate is a decision to record
# AFTER the Phase-31 verdict is published" -- is met. D33.1-01 is that decision,
# taken by the owner after the cost was stated twice.
#
# EACH ROW NAMES THE PARTITION IT NOW COVERS (Ruling R amendment, Codex 33.1-09
# LOW). A guard whose name no longer says which window it is about leaves a
# future reader unable to tell a correct 2025 from a regression, because 2025 in
# the LIVE partition is the requirement working and 2025 in the FROZEN baseline's
# key set would be a disclosure being erased. Every node id below carries
# `live_partition` or `frozen_baseline` in its own name, so the distinction
# survives without this file being consulted.
#
# ELEVEN ROWS, NOT FIVE. The plan's inventory named five live pins. Driving the
# modules turned up six more that pinned the same pre-correction partition and
# would have gone red for the same reason -- three walk-forward/temporal pins,
# the bundle-shape per-season key set, and two frozen-baseline population slices
# that were reading the LIVE constant because the two windows used to coincide.
# The last two are the quiet ones: they PASSED either way, because both sides of
# their comparison used one window, so they were measuring the wrong population
# without failing. They are recorded here rather than fixed silently.
#
# EACH ROW IS (node_id, reason, partition), where partition is `live_partition`
# or `frozen_baseline`.
# ---------------------------------------------------------------------------

UPDATED_HOLDOUT_PIN_TESTS: tuple[tuple[str, str, str], ...] = (
    (
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheLivePartitionIncludesTheSpentCleanSplit::"
        "test_the_holdout_constant_is_the_live_partition",
        "Asserted HOLDOUT_SEASONS was exactly (2021, 2022, 2023, 2024) with 2025 "
        "absent. The constant is now derived from conf.season_partition and the "
        "clean split is spent, so it asserts the live partition instead -- and "
        "additionally asserts FROZEN_BASELINE_SEASONS did NOT follow it.",
        "live_partition",
    ),
    (
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheLivePartitionIncludesTheSpentCleanSplit::"
        "test_the_committed_gate_configuration_mirrors_the_live_partition",
        "Asserted config/gate.toml's holdout was the four tune seasons with 2025 "
        "absent. That file is a GENERATED MIRROR of the rule (TOML cannot import "
        "it), so the test now evaluates the parsed value against the rule -- "
        "which is what catches a stale mirror whether or not anyone ran "
        "scripts/sync_gate_holdout.py.",
        "live_partition",
    ),
    (
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheLivePartitionIncludesTheSpentCleanSplit::"
        "test_the_scored_population_is_the_live_partition_and_reaches_2025",
        "NOT in the plan's five. Asserted that NO 2025 game reaches the scored "
        "population, via two mechanisms that were both literal-driven: "
        "_load_features' max_backtest_season filter and the holdout constant. "
        "Both are corrected, so it now asserts 2025 DOES reach it. The original "
        "docstring's warning -- read the parquet, not the loader, because the "
        "loader is itself one of the mechanisms -- is kept, since it is still "
        "true in the other direction.",
        "live_partition",
    ),
    (
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::"
        "test_the_frozen_baseline_extraction_is_not_vacuous",
        "NOT in the plan's five, and it never failed. It checked the extracted "
        "[baseline.*] block carries a per-season table for each of the LIVE "
        "holdout's seasons, which was right only while the two windows "
        "coincided. Retargeted to FROZEN_BASELINE_SEASONS, which is what that "
        "block's season tables are by definition.",
        "frozen_baseline",
    ),
    (
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheWidenedOddsTableDoesNotPerturbTheBaseline::"
        "test_dropping_the_juice_columns_changes_no_frozen_baseline_per_season_figure",
        "NOT in the plan's five, and it never failed either -- both sides of its "
        "comparison used one window, so it kept passing while sliced to the "
        "wrong population. It exists to make a byte-identity FAILURE "
        "unambiguous, and byte-identity is a property of the frozen block, so it "
        "now slices FROZEN_BASELINE_SEASONS.",
        "frozen_baseline",
    ),
    (
        "tests/unit/test_deploy_gate.py::"
        "test_gate_holdout_is_the_live_partition_and_now_contains_2025",
        "The strongest of the five. It asserted gate.seasons.holdout was exactly "
        "2021-2024 and stated its own release condition: a milestone needing "
        "2025 is a decision to record AFTER the Phase-31 verdict is published. "
        "It was published. The old sentence is quoted in the new docstring so "
        "the change is visible rather than silent.",
        "live_partition",
    ),
    (
        "tests/unit/test_deploy_gate.py::"
        "test_the_phase_start_snapshot_holdout_was_re_read_to_the_live_partition",
        "NEW, and the reason it is new rather than a rename: "
        "PHASE_START_GATE_SETTINGS['seasons.holdout'] was RE-READ from the new "
        "committed config in the same commit as the config edit, exactly as that "
        "snapshot's own comment instructs. The ten-key snapshot test keeps its "
        "name because nine keys are untouched; this guards the tenth and asserts "
        "the re-read value IS the live partition rather than a third number.",
        "live_partition",
    ),
    (
        "tests/unit/test_deploy_gate.py::"
        "test_bundle_delta_keys_pinned_in_builder_over_the_live_partition",
        "NOT in the plan's five. It pinned the per-season bundle keys to the "
        "literal set {2021, 2022, 2023, 2024} -- a copy of the partition living "
        "in a bundle-shape test. It now reads gate.HOLDOUT_SEASONS, since what "
        "is under test is that the keys COVER the scored window.",
        "live_partition",
    ),
    (
        "tests/unit/test_temporal_splits.py::test_default_config_is_the_live_partition",
        "Pinned TemporalSplitConfig.default() to the three literals [2018, 2019] "
        "/ [2020] / [2021..2024]. It now derives them from the rule and "
        "additionally asserts the rule's own output passes validate() -- SPEC "
        "R6's pairwise-disjoint and non-empty acceptance clauses.",
        "live_partition",
    ),
    (
        "tests/unit/test_temporal_splits.py::"
        "test_walk_forward_splits_cover_the_live_partition",
        "NOT in the plan's five. It asserted a literal 4 splits. The failure was "
        "instructive: its synthetic fixture stopped at 2024, and "
        "generate_splits SKIPS a holdout season with no test rows rather than "
        "raising, so a stale fixture silently changes how many folds a test "
        "sees. The fixture's span is now derived from the partition too.",
        "live_partition",
    ),
    (
        "tests/unit/test_temporal_splits.py::"
        "test_walk_forward_expanding_window_over_the_live_partition",
        "NOT in the plan's five. It named season 2023 and the literal train list "
        "[2018..2022]. The EXPANDING property is what it tests, so it now "
        "asserts that against the live partition's own last holdout season -- "
        "2025, the season that used to be invisible.",
        "live_partition",
    ),
)


# ---------------------------------------------------------------------------
# SIX MORE UPDATED PINS, FOUND BY MEASURING RATHER THAN BY READING.
#
# APPENDED by Plan 33.1-09 Task 3 on 2026-09-14, in the same task and the same
# commit as the eleven above. A SECOND slot rather than an edit to the first,
# because this file's APPEND PROTOCOL says a slot is appended once and not
# edited afterwards -- and because the way these were found is itself the
# record: the eleven came from the plan's inventory plus the three modules it
# named, and these six came from running the blast radius the changed symbols
# actually have.
#
# HOW THE RADIUS WAS DERIVED, so it can be re-derived rather than trusted: a
# grep over tests/ for every symbol this plan changed (HOLDOUT_SEASONS,
# TemporalSplitConfig, BacktestConfig, max_backtest_season, first_data_season,
# the two promote_models bounds, _incumbent_window, _drift_tripwire,
# _load_gold_holdout, TEAM_FORM_PER_GAME_FIRST_SEASON, gate.seasons,
# validate_gate_config, season_partition, the models/train argparse flags and
# build_parser) produced 21 unit modules, which were run as an explicit list.
#
# WHAT THAT RUN FOUND: 14 failures, of which SIX are recorded below and the
# other EIGHT were already dispositioned -- six in
# tests/unit/test_temporal_display_columns.py::TestRealGold, every one of them
# already in GOLD_REBUILD_NEWLY_RED from Plan 33.1-08, plus the two counted
# there under other modules. No suite-wide run was made and none is claimed.
#
# THE TWO SITES. `tests/unit/test_backtest_engine.py` pinned the engine's own
# defaults, INCLUDING an assertion that 2025 is filtered OUT -- the test-side
# statement of the very defect SPEC R6 removed.
# `tests/unit/test_promote_models_tuned_path.py` asserted that a differing
# incumbent holdout RAISES, which D33.1-04 deliberately turned into a report.
#
# Same dispositions as the eleven: updated with recorded reasons, none deleted,
# none added to DELIBERATE_TRIPWIRE_NODE_IDS, every node id naming the partition
# it covers.
# ---------------------------------------------------------------------------

UPDATED_HOLDOUT_PIN_TESTS_FOUND_BY_MEASUREMENT: tuple[tuple[str, str, str], ...] = (
    (
        "tests/unit/test_backtest_engine.py::TestBacktestConfig::"
        "test_default_holdout_seasons_are_the_live_partition",
        "Pinned BacktestConfig().holdout_seasons to the literal [2021, 2022, 2023, "
        "2024]. Derived from conf.season_partition now, because re-pinning a "
        "literal would make this file another declaration of the partition.",
        "live_partition",
    ),
    (
        "tests/unit/test_backtest_engine.py::TestBacktestConfig::"
        "test_default_max_backtest_season_is_the_live_partitions_latest_completed_season",
        "Pinned max_backtest_season to the literal 2024 -- the site that hid 2025 "
        "from every consumer loading gold through the engine. Derived from the "
        "partition's latest completed season, and additionally asserted >= 2025 so "
        "the specific regression cannot come back unnoticed.",
        "live_partition",
    ),
    (
        "tests/unit/test_backtest_engine.py::TestTheFutureSeasonFilter::"
        "test_filters_only_seasons_after_the_live_partition",
        "Asserted that 2025 was FILTERED OUT of _load_features -- the test-side "
        "statement of the defect. Now asserts 2025 is KEPT and that a season beyond "
        "the most recent completed one is still dropped, so the filter is proven to "
        "still bite rather than merely proven not to bite on 2025. Its class was "
        "renamed from TestFilters2025Data, which named a season instead of a "
        "behaviour.",
        "live_partition",
    ),
    (
        "tests/unit/test_promote_models_tuned_path.py::"
        "test_incumbent_window_derived_from_live_metadata_with_the_live_partition_holdout",
        "Asserted the derived holdout was '2021,2022,2023,2024'. Under D33.1-03 the "
        "holdout is superseded by the committed rule while train and hp_val still "
        "come from the incumbent's own metadata -- D30-12's per-target asymmetry fix "
        "is untouched, and the test now says which half comes from where.",
        "live_partition",
    ),
    (
        "tests/unit/test_promote_models_tuned_path.py::"
        "test_a_narrower_incumbent_holdout_is_reported_against_the_live_partition",
        "Asserted _incumbent_window RAISES on a differing holdout. D33.1-04 turned "
        "that refusal into a REPORT, because under D33.1-01 the condition is the "
        "chosen design rather than an anomaly -- raising would refuse every run. The "
        "load-bearing half of the old comment is what the test now asserts: the "
        "difference must NOT print with no warning. Its sibling "
        "test_a_wider_incumbent_holdout_is_reported_against_the_live_partition was "
        "updated identically, and a NEW control "
        "(test_an_incumbent_recording_the_live_partition_reports_NOTHING) keeps the "
        "report from being unconditional.",
        "live_partition",
    ),
    (
        "tests/unit/test_promote_models_tuned_path.py::"
        "test_every_live_incumbent_DIFFERS_from_the_live_partition_and_says_so",
        "Asserted all three live incumbents AGREED with the gate holdout, true only "
        "while the windows coincided. They record [2021..2024] and no re-fit has "
        "run, so under D33.1-04 the correct assertion is the reported DIFFERENCE -- "
        "agreement would mean a metadata.json was edited rather than a model "
        "re-fitted.",
        "live_partition",
    ),
)


# ---------------------------------------------------------------------------
# TWO TESTS THAT CARRIED A STALE PARTITION LITERAL BUT ARE NOT PARTITION GUARDS.
#
# Recorded SEPARATELY from UPDATED_HOLDOUT_PIN_TESTS, and the separation is the
# honest part rather than a filing convenience.
#
# The Ruling R amendment requires every UPDATED HOLDOUT PIN to carry
# `live_partition` or `frozen_baseline` in its own node id, because a GUARD whose
# name no longer says which window it covers leaves a future reader unable to tell
# a correct 2025 from a regression. These two are not guards of that kind. One is
# about merge-on-game_id pairing and the delta invariant; the other is about a
# Phase-29 diagnostic config being a separate object from the canonical one.
# Neither asserts a partition value any more -- both now DERIVE -- so renaming
# them to carry `live_partition` would MISDESCRIBE what they test in order to
# satisfy a naming rule written for something else.
#
# What they DID carry was a partition literal, which is why they went red. That is
# worth recording, and it is recorded here, in a slot whose name says what it
# holds.
# ---------------------------------------------------------------------------

PARTITION_LITERALS_REMOVED_FROM_NON_GUARD_TESTS: tuple[tuple[str, str, str], ...] = (
    (
        "tests/integration/test_promote_models.py::"
        "test_paired_delta_keys_populated_on_game_id",
        "Its synthetic frame hand-wrote seasons (2021, 2022, 2023, 2024) and read "
        "the per-season keys back by the same literals. _populate_paired_delta_keys "
        "rebuilds those dicts from deploy_gate.HOLDOUT_SEASONS, so the keys vanished "
        "and the test raised KeyError: 2021. The fixture now derives its seasons; "
        "what the test is about -- the merge-on-game_id pairing and the delta "
        "invariant -- does not depend on which seasons they are.",
        "live_partition",
    ),
    (
        "tests/integration/test_signal_lift.py::TestPhase29CliWiring::"
        "test_covered_selection_window_config_shape",
        "Asserted TemporalSplitConfig.default().train_seasons == [2018, 2019] to "
        "prove the Phase-29 diagnostic sibling had not mutated the canonical "
        "window. The PROPERTY is right and is kept; the literal was the partition "
        "written down again, so it is now asserted as a difference against the "
        "rule. The sibling's own [2018, 2019, 2020] window is DELIBERATELY left a "
        "literal: it is a frozen historical diagnostic window and must not roll "
        "forward with a rule it was never measured under.",
        "live_partition",
    ),
)


# ---------------------------------------------------------------------------
# Plan 33.1-10 Task 1 -- THE FOUR PERSISTED COMPONENTS OF THE FINAL FIT, each
# decided separately, with the rejected alternative and its cost.
#
# MEASURED/DECIDED 2026-09-14. Ruling S of `33.1-10-PLAN.md`, authored in
# `models/trainers/final_fit.py` as `FINAL_FIT_COMPONENT_POLICY` (the data form) and
# in that module's docstring (the prose form). Recorded HERE so Plan 33.1-11's
# readout can QUOTE the record rather than re-argue it, and so the decision survives
# independently of any one module's docstring.
#
# The load-bearing one is the CALIBRATOR. All three trainers fit a calibrator or a
# residual converter on hp-val PREDICTIONS from a model trained on the selection
# window only, OUTSIDE the tuning branch. The final fit INCLUDES those hp-val rows.
# Refitting the calibrator on them would make it in-sample: it would look perfectly
# calibrated while being useless, and WP's entire stated value is calibration. So it
# is carried over by reference and the test asserts object IDENTITY, not numeric
# agreement -- a silently refitted calibrator that happened to agree would still fail.
# ---------------------------------------------------------------------------

FINAL_FIT_COMPONENT_DECISIONS: dict[str, str] = {
    "model": (
        "REFIT on every completed season in SeasonPartition.final_fit. This is the "
        "point of D33.1-02 and the only reason the entry point exists: "
        "WalkForwardSplitter.generate_splits builds every fold as season < "
        "holdout_season and each trainer keeps the LAST fold's model, so the newest "
        "completed season never enters the shipped model under ANY choice of the "
        "three season lists."
    ),
    "preprocessing": (
        "REFIT on the SAME rows as the final model and PERSISTED WITH IT as one "
        "inseparable sklearn Pipeline (D33.1-R1). Under D33.1-R3 the scaler is the "
        "third step of the object whose fourth step is the estimator, so the two "
        "cannot be handed different row sets even deliberately. This is a DELIBERATE "
        "behaviour change from the pre-D33.1-R3 code, where the scaler was fitted on "
        "train_val_split.train_data while the model came from the last walk-forward "
        "fold -- the two had never shared a row set."
    ),
    "calibrator": (
        "CARRIED OVER unchanged from the walk-forward stage -- NOT refitted, and "
        "carried by REFERENCE so object identity is assertable. It was fitted on "
        "hp-val PREDICTIONS from a model trained on the selection window only, which "
        "is what makes it out-of-sample. Refitting it on rows the final model was "
        "fitted on would make it IN-SAMPLE: the reliability curve would look "
        "excellent and mean nothing. The carried-over pairing biases mildly toward "
        "UNDER-confidence, which is a conservative, statable error rather than a "
        "flattering, unstatable one."
    ),
    "feature_names": (
        "CARRIED OVER from the walk-forward stage's selection on the selection "
        "window. Selection stays where Ruling Q put it, which is what keeps a "
        "candidate comparable to its incumbent on the axis the deploy gate measures."
    ),
    "rejected_alternative": (
        "HOLDING hp_val OUT of the final fit. It keeps the calibrator honestly "
        "refittable, and it was NOT taken."
    ),
    "rejected_alternative_cost": (
        "It would leave the shipped model ONE SEASON SHORT of every completed season, "
        "which silently defeats D33.1-01 -- a locked owner decision, not a default to "
        "trade away in an implementer's judgement. The cost of the choice made "
        "INSTEAD is recorded rather than hidden: the shipped model's calibrator was "
        "fitted against a NARROWER model than the one that ships."
    ),
    "final_fit_seasons_metadata_key": "final_fit_seasons",
    "final_fit_components_metadata_key": "final_fit_components",
    "entry_point_module": "models/trainers/final_fit.py",
    "requirement": "R6",
    "decision": "D33.1-01 / D33.1-02 (Ruling S of 33.1-10-PLAN.md)",
}


# ---------------------------------------------------------------------------
# Plan 33.1-10 Task 2 -- THE PERSISTED-PREPROCESSING / CONVERTER-PARAMETER CONTRACT.
#
# DECIDED 2026-09-12 (D33.1-R1, owner-ratified at replan time), IMPLEMENTED and
# MEASURED 2026-09-14. A deliberate SCOPE EXPANSION past the SPEC's literal wording,
# attributed to R6.
#
# THE GAP IT CLOSES. `save_model_artifact` had no preprocessing parameter,
# `load_model_artifact` returned none, and the string `scaler` appeared nowhere in
# `models/artifacts.py` -- that ABSENCE was the contract. An estimator fitted on
# standardised features and served on unstandardised ones produces a plausible WRONG
# answer rather than an error, so nothing fails and nobody looks. Separately the ATS and
# O/U converters the trainers FIT were never persisted under any name, and serving
# rebuilt conversion from `metadata.get("residual_std", 13.5)` / `13.0`.
# ---------------------------------------------------------------------------

ARTIFACT_PREPROCESSING_CONTRACT: dict[str, object] = {
    "preprocessing_filename": "preprocessing.pkl",
    "converter_params_metadata_key": "converter_params",
    "converter_param_fields": (
        "distribution_type",
        "residual_std",
        "distribution_params",
    ),
    "binds": (
        "ARTIFACTS SAVED UNDER IT ONLY. An artifact directory carrying neither "
        "preprocessing.pkl nor a converter_params metadata key is served EXACTLY as it "
        "was before this contract existed: raw selected columns into model.predict_proba, "
        "converters rebuilt from residual_std with the 13.5 / 13.0 defaults. All three "
        "currently-deployed artifacts are in that class."
    ),
    "legacy_path_preserved": True,
    "legacy_path_removal_condition": (
        "The 13.5 / 13.0 fallbacks and the raw-feature branch MUST NOT be removed while "
        "any legacy artifact is deployed. Their absence would fail the D33.1-R2 "
        "byte-identity proof as loudly as their being the only path fails D33.1-R1."
    ),
    "serving_never_refuses_a_legacy_artifact": (
        "The branch is `if preprocessing is not None`, never an assertion. The owner "
        "rejected a refusal explicitly: it would take current-week prediction generation "
        "down on purpose."
    ),
    "wp_persisted_object": (
        "The four-step Pipeline from D33.1-R3 (imputer -> missing_indicator -> scaler -> "
        "estimator). A Pipeline rather than a bare scaler because it makes the transform "
        "and the estimator ONE object: there is no way to load the estimator without its "
        "scaler, which is a structural defence rather than a convention."
    ),
    "converter_form": (
        "JSON in metadata, NOT a second pickle. Three fields describe either converter "
        "completely and reconstruct it exactly; a JSON record is readable a year from now "
        "by a human auditing why a cover probability was what it was; and it does not "
        "widen the joblib.load deserialisation surface (threat T-33.1-65g)."
    ),
    "proof": (
        "tests/unit/test_artifact_preprocessing_roundtrip.py -- per-target fit / save / "
        "reload / serve, asserted under numpy.testing.assert_array_equal and never "
        "allclose, because a missing standardisation is not a near-miss."
    ),
    "requirement": "R6",
    "decision": "D33.1-R1",
    "owner_ratified": "2026-09-12",
    "measured": "2026-09-14",
}


# ---------------------------------------------------------------------------
# Plan 33.1-10 Task 2 -- THE PRE-CHANGE LEGACY SERVING BASELINE.
#
# CAPTURED 2026-09-14 at commit a916504, BEFORE the D33.1-R1 serving edits were made.
# THE ORDERING IS LOAD-BEARING and is the same argument `p331_rung0.json` and the
# cross-check pre-registration both rest on: a baseline captured AFTER the change is a
# transcription of the change, not a baseline.
#
# ONE FIX PRECEDES IT, and is named rather than folded in. `predict_games` was the single
# site in this repository calling `calibrator.transform()`; the deployed WP calibrator is
# a `models.calibrate.PlattCalibrator`, which exposes `predict` and not `transform`, so
# serving the real deployed artifact through this pipeline RAISED AttributeError. The
# legacy path could not be captured at all until that was repaired (commit a916504,
# deviation Rule 1). It moved no served number, because the branch produced none.
#
# THE FRAME IS REPRODUCIBLE FROM THIS RECORD ALONE, which is what lets the test rebuild it
# rather than store it: seed `baseline_seed`, `n_rows` rows, and for the i-th name in the
# SORTED UNION of the three artifacts' feature lists, the column is
# `numpy.random.default_rng(baseline_seed + i).standard_normal(n_rows)`. market_spread and
# market_total use the two declared offsets. If the rule and the capture ever disagreed,
# the pinned predictions below would not match -- so the record checks itself.
# ---------------------------------------------------------------------------

LEGACY_ARTIFACT_SERVING_BASELINE: dict[str, object] = {
    "captured_at_commit": "a916504",
    "captured_before": "the D33.1-R1 edits to models/prediction_pipeline.predict_games",
    "artifacts": (
        "wp_20260824_113325",
        "ats_20260605_220128",
        "ou_20260326_163930",
    ),
    "baseline_seed": 3311003,
    "n_rows": 24,
    "spread_seed_offset": 9001,
    "total_seed_offset": 9002,
    "feature_union_size": 59,
    "frame_rule": (
        "Columns are the SORTED UNION of the three artifacts' feature lists. Column i is "
        "numpy.random.default_rng(baseline_seed + i).standard_normal(n_rows). "
        "market_spread is default_rng(baseline_seed + spread_seed_offset)."
        "standard_normal(n_rows) * 3.0; market_total is 44.0 + "
        "default_rng(baseline_seed + total_seed_offset).standard_normal(n_rows) * 3.0."
    ),
    "series": {
        "wp_home_probability": (
            0.3286147393126939,
            0.6109543961053129,
            0.6487927358446811,
            0.5626405851466023,
            0.44536675216229055,
            0.9019047556248532,
            0.5419342245520633,
            0.5033821551788693,
            0.11406951270949264,
            0.6974731643786121,
            0.41105330748973246,
            0.46840896627918716,
            0.17990323557364132,
            0.40843179479206837,
            0.0955200513994877,
            0.24294428725978848,
            0.2039853954998342,
            0.16724392056144743,
            0.4273914735780277,
            0.40552539039427765,
            0.22306895889389258,
            0.6665598872833663,
            0.5380702221771154,
            0.5704338486441125,
        ),
        "predicted_margin": (
            -4.285736560821533,
            0.7871482372283936,
            7.209689617156982,
            5.820453643798828,
            1.9469223022460938,
            5.816115379333496,
            0.5511355400085449,
            -4.9769606590271,
            3.543321132659912,
            7.84668493270874,
            0.6631534695625305,
            1.1536474227905273,
            3.655007839202881,
            -3.0673835277557373,
            -6.8723344802856445,
            2.406721591949463,
            -3.3203885555267334,
            -7.446810722351074,
            -1.2644047737121582,
            0.9717382192611694,
            -3.8437631130218506,
            2.371617555618286,
            3.686084270477295,
            -1.9243866205215454,
        ),
        "ats_cover_probability": (
            0.3224630719248104,
            0.4420124465968941,
            0.6732770550019527,
            0.639198892597915,
            0.44618145122673936,
            0.7163957218043244,
            0.5884532578748258,
            0.45090959916708484,
            0.5415921923502203,
            0.7587776673954543,
            0.3773461029868883,
            0.5678568890439389,
            0.5672830668727291,
            0.25863084010207515,
            0.2900326387812584,
            0.70771534925464,
            0.3372958040886318,
            0.2084682356663039,
            0.5847564426144709,
            0.5124204318301468,
            0.23745560530484833,
            0.49031868099870834,
            0.6321992125145298,
            0.5073895950809308,
        ),
        "predicted_total": (
            45.17256164550781,
            50.32017517089844,
            47.07448196411133,
            47.53922653198242,
            46.673255920410156,
            49.27655792236328,
            42.86054992675781,
            44.797847747802734,
            48.031219482421875,
            41.60919189453125,
            41.86784744262695,
            42.83418655395508,
            47.089534759521484,
            49.62985610961914,
            49.481998443603516,
            41.44801330566406,
            43.46241760253906,
            46.09697723388672,
            49.61927795410156,
            46.5479736328125,
            40.07179260253906,
            49.23410415649414,
            50.85796356201172,
            46.93551254272461,
        ),
        "over_probability": (
            0.7437359007373157,
            0.7467092173200668,
            0.536625646075304,
            0.6026601915964309,
            0.5000988072242663,
            0.5863563437658266,
            0.5851333797878899,
            0.6268130418420721,
            0.6222508041318336,
            0.4651066450824819,
            0.36758212552860803,
            0.3660265564037658,
            0.5742289631432592,
            0.6685531440138526,
            0.7712854346251206,
            0.37993442406424294,
            0.4377376316528113,
            0.574475503010461,
            0.5415149927282727,
            0.38700986357336165,
            0.3559143938871967,
            0.47506542743370805,
            0.6182647615156507,
            0.4995363537512896,
        ),
        "under_probability": (
            0.25626409926268434,
            0.2532907826799332,
            0.46337435392469595,
            0.3973398084035691,
            0.49990119277573375,
            0.41364365623417343,
            0.41486662021211007,
            0.3731869581579279,
            0.37774919586816635,
            0.5348933549175181,
            0.632417874471392,
            0.6339734435962342,
            0.4257710368567408,
            0.3314468559861474,
            0.22871456537487944,
            0.6200655759357571,
            0.5622623683471887,
            0.425524496989539,
            0.45848500727172725,
            0.6129901364266384,
            0.6440856061128033,
            0.524934572566292,
            0.3817352384843493,
            0.5004636462487104,
        ),
    },
    "re_measured_after_the_change": (
        "All six series were re-measured after the D33.1-R1 serving edits and every one "
        "is IDENTICAL, which is the D33.1-R2 claim: the new contract binds new artifacts "
        "only and current predictions did not move."
    ),
    "requirement": "R6",
    "decision": "D33.1-R2",
}
