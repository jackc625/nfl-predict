"""The durable bet-list PAIR travels, and it has exactly two entry points (31-22, G-31-123b).

THE DEFECT THIS MODULE IS THE REGRESSION FOR
--------------------------------------------
The two artifacts ``/bets`` ultimately serves -- ``outputs/bet_list/bet_list.parquet`` and its
companion ``outputs/bet_list/bet_tracker.json`` -- had exactly ONE producer, buried at step 15 of
the 19-step Friday orchestrator, with no CLI, no module ``__main__``, no Makefile target, no
scheduler action and no documented manual stage. Worse, the two halves were written from two
DIFFERENT places: ``generate_weekly_bet_list`` wrote the parquet and
``pipeline/steps.py::step_generate_recommendations`` wrote the tracker JSON. So a caller of the
library function produced HALF the pair, and the realized-versus-expected tracker would sit
permanently empty while the page still looked correct.

WHY BOTH A BEHAVIOURAL AND A STRUCTURAL ASSERTION
-------------------------------------------------
The behavioural case proves the pair is written TODAY. It would pass again the day someone adds a
new caller that writes only the parquet, or moves the tracker write back out into a step -- which
is precisely the shape the original defect had. So this module ALSO pins, structurally, that there
are exactly two callers of the generator and exactly one production site that writes the tracker.
The structural style -- an ``ast`` walk over the production packages with an EXACT pinned set and a
stated reason why exactness matters in both directions -- is copied from
``tests/unit/test_one_recommendation_path.py``.

HERMETIC UNDER THE HARD DATA BOUNDARY
-------------------------------------
``build_weekly_candidates`` and ``load_frozen_chain_fit`` are the ONLY calls in
``generate_weekly_bet_list`` that read ``artifacts/``, ``data/gold/`` or ``data/silver/``, so
injecting those two is what keeps this module from touching the lake. Everything downstream of
them runs FOR REAL: the selector, the records-to-frame mapping, the artifact upsert, the grading
pass, and BOTH writes. Nothing here is stubbed out on the path under test.

Selectors (``-k``): pair, reader, no_stray_write, callers, one_write, not_vacuous, predicate,
wired.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from api.cache import BET_LIST_COLUMNS, bet_list_source_is_absent
from backtest.weekly_bet_list import (
    BET_LIST_ARTIFACT_NAME,
    BET_TRACKER_ARTIFACT_NAME,
    DEFAULT_BET_LIST_DIR,
    WeeklyChainFit,
    read_bet_list_cache_sources,
)

# ---------------------------------------------------------------------------
# The pinned structural facts
# ---------------------------------------------------------------------------

# The EXACT set of production modules permitted to CALL the weekly generator, count pinned.
# Exactness matters in BOTH directions. A third caller is a third producer of the durable pair and
# can drift from the other two -- which is the class of defect this whole plan closes. A DEPARTED
# caller is worse in a quieter way: ``scripts/generate_bet_list.py`` is the command the ``/bets``
# copy and this repository's operator documents now NAME, so losing it would leave the page
# instructing an operator to run something that is gone.
#
# THREE since Plan 33.2-27 moved the scheduled run to the daily lock-time cadence:
# ``pipeline/daily_steps.py`` is the SCHEDULED producer now (``scripts/daily_lock_pipeline.py``
# builds its registry), and ``pipeline/steps.py`` is the legacy Friday registry kept as a manual
# operator tool. Registered here by the 33.2 review, batch 3, rather than left to fail as an
# undeclared caller. It is still ONE library call: all three produce the pair through
# ``generate_weekly_bet_list`` itself, and the tracker write below stays in exactly one place.
_GENERATOR_CALLERS: frozenset[str] = frozenset(
    {
        "pipeline/daily_steps.py",
        "pipeline/steps.py",
        "scripts/generate_bet_list.py",
    }
)
_GENERATOR_NAME = "generate_weekly_bet_list"

# The tracker write must happen in exactly ONE production place, and that place must be inside the
# library function -- not in a caller. This is the assertion that stops the pair being split apart
# again by a future author who adds a step that "also" writes the tracker.
_TRACKER_WRITE_NAME = "write_bet_tracker_artifact"
_TRACKER_WRITE_MODULE = "backtest/weekly_bet_list.py"

# The predicate whose call site inside ``populate_cache`` is asserted below. A unit test of a
# helper nobody calls is the exact failure mode that assertion guards against.
_PREDICATE_NAME = "bet_list_source_is_absent"
_CACHE_MODULE = pathlib.Path("api") / "cache.py"
_POPULATE_FUNCTION = "populate_cache"

# The production packages walked. ``tests`` is excluded: this module itself names both symbols.
_PRODUCTION_PACKAGES: tuple[str, ...] = (
    "api",
    "backtest",
    "data",
    "features",
    "models",
    "pipeline",
    "ratings",
    "scripts",
    "utils",
    "web",
)


# ---------------------------------------------------------------------------
# The authored week -- fixed values, so every assertion below is deterministic
# ---------------------------------------------------------------------------

_SEASON = 2025
_WEEK = 1
_GAME_ONE = "2025_01_DET_KC"
_GAME_TWO = "2025_01_BAL_BUF"
# A Sunday. Its own preceding-Friday 6 PM ET freeze is the instant the freshness fence compares
# against, and the snapshot below is AT that freeze, which is FRESH (at-freeze is fresh, SPEC R6).
_GAMEDAY = "2025-09-07"
_SNAPSHOT_TS = "2025-09-05T18:00:00-04:00"
# The RUN INSTANT, injected rather than read from the wall clock (Phase 33, Plan 33-05 Task 3).
# A forward run is now refused by name if its own observation time falls at or after the game's
# freeze, and this authored week froze on 2025-09-05 -- so an entry-point test driven by the real
# clock would be refused, correctly, for asserting a pick made a year after the market closed.
# One minute before the freeze is the last instant a real Friday run legitimately has.
_RUN_INSTANT = datetime(2025, 9, 5, 17, 59, tzinfo=timezone(timedelta(hours=-4)))

_FITS: dict[str, WeeklyChainFit] = {
    "wp": WeeklyChainFit("wp", 0.0, None, {_SEASON: 0.0}),
    "ats": WeeklyChainFit("ats", 0.0, 11.5, {_SEASON: 0.0}),
    "ou": WeeklyChainFit("ou", 0.0, 13.0, {_SEASON: 0.0}),
}


def _candidate(game_id: str, target: str, **market: Any) -> dict[str, Any]:
    """One authored candidate row for ``target`` on ``game_id``.

    ``sportsbook`` is ``consensus`` and ``is_live`` is False so the selector's OUM-06 provenance
    hard-fail passes on real-shaped provenance rather than on an absent column.
    """
    row: dict[str, Any] = {
        "game_id": game_id,
        "season": _SEASON,
        "week": _WEEK,
        "target": target,
        "gameday": _GAMEDAY,
        "snapshot_ts": _SNAPSHOT_TS,
        "sportsbook": "consensus",
        "is_live": False,
    }
    row.update(market)
    return row


def _authored_candidates() -> pd.DataFrame:
    """Two games times the three registered targets, all of them sided and priced.

    The O/U rows are UNDER picks (model total below the market total), which is the one arm of the
    D27-04/05 sub-population UNION that does not depend on the high-total boundary.
    """
    rows: list[dict[str, Any]] = []
    for game_id in (_GAME_ONE, _GAME_TWO):
        rows += [
            _candidate(
                game_id,
                "ou",
                model_total=41.0,
                closing_total=47.5,
                total_over_ju=-110.0,
                total_under_ju=-110.0,
            ),
            _candidate(
                game_id,
                "ats",
                model_spread=7.0,
                closing_spread=3.0,
                spread_ju_home=-110.0,
                spread_ju_away=-110.0,
            ),
            _candidate(game_id, "wp", model_prob=0.72, ml_home=-150.0, ml_away=130.0),
        ]
    return pd.DataFrame(rows)


def _authored_schedule() -> pd.DataFrame:
    """The week's spine. Every row carries its OWN gameday, which the freeze fence requires."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": _SEASON,
                "week": _WEEK,
                "gameday": _GAMEDAY,
            }
            for game_id in (_GAME_ONE, _GAME_TWO)
        ]
    )


def _directory_state(path: pathlib.Path) -> Any:
    """A comparable snapshot of ``path``: None when absent, else per-entry name/size/mtime."""
    if not path.exists():
        return None
    return sorted(
        (entry.name, entry.stat().st_size, entry.stat().st_mtime_ns)
        for entry in path.iterdir()
    )


class _GeneratedWeek:
    """What one hermetic generator run produced, plus the pre-run default-directory snapshot."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        silver_dir: pathlib.Path,
        graded: pd.DataFrame,
        default_dir_before: Any,
    ) -> None:
        self.output_dir = output_dir
        self.silver_dir = silver_dir
        self.graded = graded
        self.default_dir_before = default_dir_before


@pytest.fixture()
def generated_week(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> _GeneratedWeek:
    """Run ``generate_weekly_bet_list`` for the authored week into ``tmp_path``.

    Both lake-reading seams are injected IN the ``backtest.weekly_bet_list`` namespace, which is
    where the function resolves them. ``gold_dir`` and ``silver_dir`` point at directories that do
    not exist, so the grading pass finds no realized labels and every row stays ``pending`` -- the
    genuine forward state, and the state that makes the tracker's not-measured branch the one under
    test.
    """
    import backtest.weekly_bet_list as wbl

    monkeypatch.setattr(wbl, "load_frozen_chain_fit", lambda path: dict(_FITS))
    monkeypatch.setattr(
        wbl,
        "build_weekly_candidates",
        lambda season, week, **kwargs: (_authored_candidates(), _authored_schedule()),
    )

    default_dir_before = _directory_state(DEFAULT_BET_LIST_DIR)
    output_dir = tmp_path / "bet_list"
    silver_dir = tmp_path / "silver"
    graded = wbl.generate_weekly_bet_list(
        _SEASON,
        _WEEK,
        output_dir=output_dir,
        artifacts_dir=tmp_path / "artifacts",
        gold_dir=tmp_path / "gold",
        silver_dir=silver_dir,
        chain_fit_path=tmp_path / "fit.json",
        now=_RUN_INSTANT,
    )
    return _GeneratedWeek(output_dir, silver_dir, graded, default_dir_before)


# ---------------------------------------------------------------------------
# 1. BEHAVIOURAL: one run leaves BOTH durable artifacts behind
# ---------------------------------------------------------------------------


def test_the_pair_bet_list_parquet_is_written_with_rows(
    generated_week: _GeneratedWeek,
) -> None:
    """The half that already worked, asserted so the tracker case below cannot pass vacuously."""
    path = generated_week.output_dir / BET_LIST_ARTIFACT_NAME
    assert path.is_file(), (
        f"{BET_LIST_ARTIFACT_NAME} was not written to the output directory"
    )

    stored = pd.read_parquet(path)
    assert not stored.empty, (
        "the bet list was written with ZERO rows, so the tracker assertions below would be "
        "measuring an empty week rather than the pair travelling"
    )
    assert list(stored.columns) == BET_LIST_COLUMNS


def test_the_pair_bet_tracker_json_lands_in_the_same_directory(
    generated_week: _GeneratedWeek,
) -> None:
    """THE CASE THAT FAILS AGAINST PRE-FIX CODE (31-22, T-31-114).

    Before this plan the tracker was written by ``step_generate_recommendations``, not by the
    library function, so a caller of ``generate_weekly_bet_list`` produced HALF the pair: the
    parquet landed and the tracker JSON never existed. Measured pre-fix: parquet present, tracker
    absent, and the shared reader returning bet rows beside ZERO tracker rows.
    """
    path = generated_week.output_dir / BET_TRACKER_ARTIFACT_NAME
    assert path.is_file(), (
        f"{BET_TRACKER_ARTIFACT_NAME} is ABSENT beside a written {BET_LIST_ARTIFACT_NAME}. The "
        "durable pair has been split: the tracker was written by the pipeline STEP rather than by "
        "generate_weekly_bet_list, so a caller of the library function produces half the pair and "
        "the realized-versus-expected tracker stays permanently empty while /bets looks correct."
    )

    records = json.loads(path.read_text(encoding="utf-8"))
    assert records, "the tracker artifact was written with no blocks in it"


def test_the_reader_both_cache_callers_use_returns_a_populated_pair(
    generated_week: _GeneratedWeek,
) -> None:
    """Asserted through the REAL reader, not by reading the two files directly.

    ``read_bet_list_cache_sources`` is the seam ``pipeline/steps.py::step_populate_web_cache`` and
    ``scripts/populate_cache.py`` both cross, so it is the only check that proves the artifacts
    reach the cache population step. A file that exists but does not survive the reader -- a
    schema mismatch, a filename the reader spells differently -- would still leave ``/bets`` empty.
    """
    sources = read_bet_list_cache_sources(
        generated_week.output_dir, generated_week.silver_dir
    )

    assert not sources.bet_list.empty, (
        "the shared reader returned NO bet rows, so the copy step would build a zero-row bet_list "
        "table and /bets would refuse the week"
    )
    assert not sources.tracker.empty, (
        "the shared reader returned NO tracker blocks. This is the state a split pair produces, "
        "and it is why plan 31-23's copy would still be naming an insufficient sequence."
    )


def test_no_stray_write_landed_in_the_default_bet_list_directory(
    generated_week: _GeneratedWeek,
) -> None:
    """The hard data boundary, asserted rather than trusted to the keyword argument.

    Every write in this module goes to ``tmp_path``. Asserting the DEFAULT directory is byte-for-
    byte as it was is what proves ``output_dir`` was honoured all the way down -- both writes take
    it, and a defaulted one in either would land in the repository's real ``outputs/bet_list/``.
    """
    assert (
        _directory_state(DEFAULT_BET_LIST_DIR) == generated_week.default_dir_before
    ), (
        f"{DEFAULT_BET_LIST_DIR.as_posix()} changed during a run directed at a temporary "
        "directory; some write is not honouring output_dir"
    )


# ---------------------------------------------------------------------------
# 2. STRUCTURAL: exactly two callers, and exactly one tracker write site
# ---------------------------------------------------------------------------


def _key(path: pathlib.Path) -> str:
    """POSIX-normalized repo-relative key so assertions are platform-independent."""
    return str(path).replace("\\", "/")


@pytest.fixture()
def production_files() -> list[pathlib.Path]:
    """Every ``*.py`` under the production packages, excluding ``__pycache__``.

    Fails fast when the walk finds nothing: a guard that scans an empty list passes VACUOUSLY.
    """
    files: list[pathlib.Path] = []
    for package in _PRODUCTION_PACKAGES:
        root = pathlib.Path(package)
        if not root.is_dir():
            continue
        files += [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, (
        "the production walk found no Python files; the guard would pass vacuously. "
        f"Packages searched: {list(_PRODUCTION_PACKAGES)}"
    )
    return sorted(files)


def _called_names(tree: ast.AST) -> set[str]:
    """Every function name CALLED anywhere under ``tree``, plain or attribute-qualified.

    Deliberately keyed on CALLS and not on imports: importing a symbol is not producing an
    artifact with it, and a module that imports the generator to re-export or type-check it is not
    a third producer.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _callers_of(production_files: list[pathlib.Path], name: str) -> set[str]:
    """The production modules that call ``name``."""
    callers: set[str] = set()
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if name in _called_names(tree):
            callers.add(_key(path))
    return callers


def test_the_callers_of_the_generator_are_exactly_the_intended_entry_points(
    production_files: list[pathlib.Path],
) -> None:
    """Exact in both directions -- see ``_GENERATOR_CALLERS`` for why each direction matters."""
    callers = _callers_of(production_files, _GENERATOR_NAME)

    assert callers == set(_GENERATOR_CALLERS), (
        f"the callers of {_GENERATOR_NAME!r} moved.\n"
        f"  expected: {sorted(_GENERATOR_CALLERS)}\n"
        f"  found:    {sorted(callers)}\n"
        "A NEW caller is a third producer of the durable pair and can drift from the other two. A "
        "MISSING one means an entry point the /bets recovery copy names has vanished."
    )


def test_the_tracker_is_written_from_exactly_one_place_and_it_is_the_library(
    production_files: list[pathlib.Path],
) -> None:
    """The assertion that the pair cannot be split apart again (T-31-114).

    One write site, and it is inside ``generate_weekly_bet_list``'s own module. A second site --
    in a step, in a script, anywhere -- is a caller that can produce a different artifact SET, and
    a site OUTSIDE the library means the write has moved back out of it.
    """
    writers = _callers_of(production_files, _TRACKER_WRITE_NAME)

    assert writers == {_TRACKER_WRITE_MODULE}, (
        f"{_TRACKER_WRITE_NAME!r} is called from {sorted(writers)}; it must be called from exactly "
        f"[{_TRACKER_WRITE_MODULE!r}]. A second site is how the durable pair gets split into two "
        "producers again -- the defect G-31-123b closed. A missing site means the tracker half is "
        "no longer written at all."
    )


def test_the_structural_walk_visited_a_nonzero_module_count(
    production_files: list[pathlib.Path],
) -> None:
    """A guard that visited nothing passes for the wrong reason."""
    scanned = len(production_files)
    assert scanned > 0, "the entry-point scan visited ZERO modules"
    # A floor, not an equality: the tree grows. Well below today's count and well above zero, so
    # it catches a walk that silently stopped recursing without breaking on every new module.
    assert scanned >= 50, (
        f"the entry-point scan visited only {scanned} module(s); the production tree is far "
        "larger, so the walk has stopped recursing"
    )


# ---------------------------------------------------------------------------
# 3. THE PREDICATE, and the proof that it is actually WIRED
# ---------------------------------------------------------------------------


def test_the_predicate_is_true_for_an_absent_source() -> None:
    """``None`` AND an empty frame both mean "nothing to load".

    The empty-frame arm is the one that matters: both production callers pass frames from
    ``read_bet_list_cache_sources``, which degrades an absent artifact to an EMPTY frame and never
    returns ``None``. A ``None``-only predicate is unreachable in production, which is exactly how
    the honest warning came to be dead code.
    """
    assert bet_list_source_is_absent(None) is True
    empty = pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))
    assert bet_list_source_is_absent(empty) is True, (
        "the predicate is False for an EMPTY frame carrying the full column set -- the shape a "
        "cold start actually produces, and the one case the pre-fix `is None` guard missed"
    )


def test_the_predicate_is_false_for_a_populated_source() -> None:
    """The control: the warning must not be always-on."""
    populated = pd.DataFrame([dict.fromkeys(BET_LIST_COLUMNS)])
    assert bet_list_source_is_absent(populated) is False


def test_the_predicate_is_wired_into_populate_cache() -> None:
    """A unit test of a helper nobody calls proves nothing (T-31-118).

    Parsed rather than monkeypatched: the question is whether the CALL EXISTS in the population
    path at all, and a source-level answer cannot be satisfied by a test double.
    """
    tree = ast.parse(_CACHE_MODULE.read_text(encoding="utf-8"))
    populate = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == _POPULATE_FUNCTION
        ),
        None,
    )
    assert populate is not None, (
        f"{_key(_CACHE_MODULE)} defines no function named {_POPULATE_FUNCTION!r}"
    )

    assert _PREDICATE_NAME in _called_names(populate), (
        f"{_POPULATE_FUNCTION} does not call {_PREDICATE_NAME!r}, so the widened empty-source "
        "warning is a helper nobody uses and the zero-row cache is still silent in production. "
        f"Called names: {sorted(_called_names(populate))}"
    )


# ---------------------------------------------------------------------------
# 5. The CLI's existing --chain-fit-path drives a no-floor record end to end (Plan 33.2-29)
# ---------------------------------------------------------------------------


def test_the_cli_refuses_a_no_floor_target_without_crashing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose ATS floor is ``null`` yields ZERO live ATS rows and every ATS game recorded."""
    import functools
    import sys

    import backtest.weekly_bet_list as wbl
    import scripts.generate_bet_list as cli

    record = {
        "tune_fit": {
            target: {
                "ev_floor_t": None if target == "ats" else 0.0,
                "frozen_sd": fit.frozen_sd,
                "season_bias_by_season": {str(_SEASON): 0.0},
            }
            for target, fit in _FITS.items()
        }
    }
    fit_path = tmp_path / "corrected_chain_fit.json"
    fit_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        wbl,
        "build_weekly_candidates",
        lambda season, week, **kwargs: (_authored_candidates(), _authored_schedule()),
    )
    # The run instant is injected (see _RUN_INSTANT); the CLI itself is driven for real.
    monkeypatch.setattr(
        cli,
        "generate_weekly_bet_list",
        functools.partial(wbl.generate_weekly_bet_list, now=_RUN_INSTANT),
    )
    output_dir = tmp_path / "bet_list"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_bet_list",
            "--season",
            str(_SEASON),
            "--week",
            str(_WEEK),
            "--output-dir",
            str(output_dir),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--gold-dir",
            str(tmp_path / "gold"),
            "--silver-dir",
            str(tmp_path / "silver"),
            "--chain-fit-path",
            str(fit_path),
        ],
    )

    cli.main()

    stored = pd.read_parquet(output_dir / BET_LIST_ARTIFACT_NAME)
    ats = stored[stored["target"] == "ats"]
    assert len(ats) == 2
    assert set(ats["status"]) == {"suppressed"}
    assert set(ats["rejection_reason"]) == {"no_honest_ev_floor"}
    assert (stored[stored["target"] != "ats"]["status"] == "live").any()
