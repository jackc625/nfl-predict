"""Follow the /bets recovery copy from a cold start and prove the page serves a list.

THE DEFECT THIS MODULE IS THE REGRESSION FOR (plan 31-23, G-31-123b)
--------------------------------------------------------------------
From a genuinely cold start the ``/bets`` cache-absent empty state told the reader to run
``uv run python scripts/populate_cache.py``. For the bet list that command is a pure COPY step: it
reads ``outputs/bet_list/`` through ``read_bet_list_cache_sources``, which by documented design
degrades an ABSENT source to a zero-row frame rather than raising. So from a cold checkout it
succeeded, created the ``bet_list`` table EMPTY, stamped no per-week populated-at marker, but DID
build the schedule-derived ``bet_week_freeze`` table -- and a freeze present beside an absent
marker is exactly the condition ``_is_bet_cache_stale`` treats as stale. The page therefore flipped
from the empty state into the withheld refusal, whose own recovery text named the SAME insufficient
command again. Re-running it could never break the loop.

THE COVERAGE GAP THAT LET IT SHIP
---------------------------------
Two assertions in ``tests/api/test_bets_page.py`` pinned ``"scripts/populate_cache.py" in body`` --
one on the cache-absent state, one on the refusal -- and the second's failure message read "the
refusal does not name the command that fixes it". Both asserted that a command STRING appeared in
the response body. That is a strictly weaker claim than the one the reader needs, which is that
FOLLOWING the named command reaches a served list. This module makes the stronger claim: it reads
the sequence OUT of the rendered page, runs it, and asserts the SERVED body carries the bet rows.

THE TEST BOUNDARY, STATED HONESTLY
----------------------------------
``api.cache.populate_cache`` does far more than the bet list: it loads feature importances from
``artifacts/``, and backtest predictions, metrics, simulation results, betting rows, predictions
and game context from ``outputs/backtest/`` and ``data/gold/``. None of that can run under this
phase's hard data boundary, which forbids reading or writing the protected lake. So this module
drives the BET-LIST HALF of the population through the same public writers ``populate_cache``
calls for those three frames, and case FOUR PINS that they are the same functions by walking
``populate_cache`` itself. Without that pin the substitution would be an assertion about the
production path made by a test that never touches it.

The two calls in ``generate_weekly_bet_list`` that read the lake -- ``build_weekly_candidates`` and
``load_frozen_chain_fit`` -- are injected in the ``backtest.weekly_bet_list`` namespace, the same
two seams and the same reason as ``tests/unit/test_bet_list_entry_point.py``. Everything downstream
runs FOR REAL: the selector, the frame mapping, the artifact upsert, the grading pass, both durable
writes, the shared reader, all four cache writers, the route and the template.

Selectors (``-k``): cold_state, follow, copy_only, writer_set.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import pathlib
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.cache import (
    CACHE_SCHEMA,
    materialize_available_bet_weeks,
    materialize_bet_list_with_marker,
    materialize_bet_tracker_blocks,
    materialize_bet_week_freeze,
)
from api.services import clear_cache
from backtest.weekly_bet_list import (
    DEFAULT_BET_LIST_DIR,
    WeeklyChainFit,
    read_bet_list_cache_sources,
)

# ---------------------------------------------------------------------------
# The two rendered strings that identify the states under test
# ---------------------------------------------------------------------------

# Transcribed from the design contract, exactly as ``tests/api/test_bets_page.py`` transcribes
# them. The refusal is HTML-escaped because it reaches the page through ``{{ recovery_text }}``.
_CACHE_ABSENT_HEADING = "Bet list not built yet"
_HARD_BLOCK_MESSAGE = "This week&#39;s list is withheld -- the cache is older than this week&#39;s line freeze"
# Unique to the LIVE row table: "Matchup" also heads the suppressed-candidates table, so it cannot
# distinguish a served list from a served disclosure.
_ROW_TABLE_HEADER = "Stake (units)"

# The two recovery commands, in the order the page must name them.
_GENERATE_SCRIPT = "scripts/generate_bet_list.py"
_POPULATE_SCRIPT = "scripts/populate_cache.py"
_SCRIPT_PATH_PATTERN = re.compile(r"scripts/[A-Za-z0-9_]+\.py")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CACHE_MODULE = _REPO_ROOT / "api" / "cache.py"
_POPULATE_FUNCTION = "populate_cache"

# The bet-list, schedule and tracker writers THIS module drives in stage two. Case FOUR asserts
# this is exactly the set ``populate_cache`` calls, so the hermetic substitution is checkable.
_DRIVEN_WRITERS: frozenset[str] = frozenset(
    {
        "materialize_bet_list_with_marker",
        "materialize_available_bet_weeks",
        "materialize_bet_week_freeze",
        "materialize_bet_tracker_blocks",
    }
)

# ---------------------------------------------------------------------------
# The authored week -- fixed values, so every assertion is deterministic
# ---------------------------------------------------------------------------

_SEASON = 2025
_WEEK = 1
_GAME_ONE = "2025_01_DET_KC"
_GAME_TWO = "2025_01_BAL_BUF"
# A Sunday, and a PAST one. Its own preceding-Friday 6 PM ET freeze is therefore long before the
# populated-at instant stamped below, so the fresh case is fresh for the real reason rather than
# because a fixture asserted it.
_GAMEDAY = "2025-09-07"
_KICKOFF_UTC = "2025-09-07T17:00:00Z"
# AT the freeze, which is fresh (SPEC R6) -- the selector's own freshness fence.
_SNAPSHOT_TS = "2025-09-05T18:00:00-04:00"

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
    """Two games times the three registered targets, all sided and priced.

    The O/U rows are UNDER picks (model total below the market total), the one arm of the
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


def _authored_selector_schedule() -> pd.DataFrame:
    """The week's spine as ``build_weekly_candidates`` returns it, with each game's own gameday."""
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


def _write_authored_silver(silver_dir: pathlib.Path) -> pathlib.Path:
    """Author the silver ``games.parquet`` that ``build_bet_week_schedule`` derives freezes from.

    A REAL schedule rather than a hand-written freeze table, because the freeze instant is the
    threshold the refusal compares against and the whole point of case THREE is that the copy step
    builds that table from the SCHEDULE, independently of whether any bet row exists. Writing the
    freeze by hand would author the very coupling the production code refuses to have.
    """
    silver_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": _SEASON,
                "week": _WEEK,
                "kickoff_et": pd.Timestamp(_KICKOFF_UTC),
            }
            for game_id in (_GAME_ONE, _GAME_TWO)
        ]
    ).to_parquet(silver_dir / "games.parquet", index=False)
    return silver_dir


# ---------------------------------------------------------------------------
# The cold state, and the two stages of the recovery
# ---------------------------------------------------------------------------


def _cold_cache(db_path: pathlib.Path, *, seed_navigation: bool) -> pathlib.Path:
    """Build the pre-Phase-31 cold state: every cache table, then DROP ``bet_list``.

    That DROP is the shape the diagnosis reproduced -- a cache built before the ``bet_list`` table
    existed -- and it is the same one ``test_state_one_cache_table_absent`` uses.

    ``seed_navigation`` exists because case ONE has no second stage. With no ``available_bet_weeks``
    rows ``_normalize_week`` resolves ``(None, None)`` and the page renders its no-current-week
    state instead of the cache-absent state, so case ONE must seed the week it then reads the copy
    from. Cases TWO and THREE seed NOTHING: their stage two writes both schedule-derived tables
    from the real schedule, exactly as ``populate_cache`` does, so seeding them would insert rows
    production would not have inserted twice.
    """
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        if seed_navigation:
            materialize_available_bet_weeks(
                conn,
                pd.DataFrame(
                    [{"game_id": _GAME_ONE, "season": _SEASON, "week": _WEEK}]
                ),
            )
        conn.execute("DROP TABLE bet_list")
    finally:
        conn.close()
    return db_path


def _stage_one_generate(
    output_dir: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> pd.DataFrame:
    """Run the GENERATION command's library entry point into ``output_dir``.

    This is what ``scripts/generate_bet_list.py`` does: resolve the week and delegate. Both
    lake-reading seams are injected in the namespace the function resolves them from;
    ``gold_dir`` points at a directory that does not exist, so the grading pass finds no realized
    labels and every selected row stays live -- the genuine forward state.
    """
    import backtest.weekly_bet_list as wbl

    monkeypatch.setattr(wbl, "load_frozen_chain_fit", lambda path: dict(_FITS))
    monkeypatch.setattr(
        wbl,
        "build_weekly_candidates",
        lambda season, week, **kwargs: (
            _authored_candidates(),
            _authored_selector_schedule(),
        ),
    )
    return wbl.generate_weekly_bet_list(
        _SEASON,
        _WEEK,
        output_dir=output_dir,
        artifacts_dir=tmp_path / "artifacts",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "gold",
        chain_fit_path=tmp_path / "fit.json",
    )


def _stage_two_populate(
    db_path: pathlib.Path,
    artifact_dir: pathlib.Path,
    silver_dir: pathlib.Path,
) -> None:
    """Run the COPY command's bet-list half against ``db_path``.

    The three frames come from ``read_bet_list_cache_sources`` -- the ONE seam both production
    cache-population callers cross -- and they are written through the four writers
    ``populate_cache`` calls for them, in the same order, under ONE populated-at instant resolved
    before the writes, exactly as ``populate_cache`` resolves it. Case FOUR pins the writer set.
    """
    sources = read_bet_list_cache_sources(artifact_dir, silver_dir)
    conn = duckdb.connect(str(db_path))
    try:
        now = datetime.now(tz=UTC)
        materialize_bet_list_with_marker(conn, sources.bet_list, populated_at=now)
        materialize_available_bet_weeks(conn, sources.schedule)
        materialize_bet_week_freeze(conn, sources.schedule)
        materialize_bet_tracker_blocks(conn, sources.tracker)
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [["last_updated", now.isoformat(), now]],
        )
    finally:
        conn.close()


@contextmanager
def _serving(db_path: pathlib.Path) -> Iterator[TestClient]:
    """Serve ``/bets`` out of ``db_path`` through the real route and template."""
    from api.dependencies import get_db
    from api.main import app

    clear_cache()
    conn = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()
        conn.close()
        clear_cache()


def _named_scripts(body: str) -> list[str]:
    """Every ``scripts/*.py`` path the rendered body names, deduplicated in DOCUMENT ORDER."""
    ordered: list[str] = []
    for match in _SCRIPT_PATH_PATTERN.findall(body):
        if match not in ordered:
            ordered.append(match)
    return ordered


def _instructional_paragraphs(body: str) -> list[str]:
    """Every rendered paragraph that names at least one ``scripts/*.py`` command.

    WHY THIS EXISTS SEPARATELY FROM ``_named_scripts``. That helper deduplicates across the whole
    body, which is right for asking "does the page name the sequence" and BLIND to "does every
    block that gives an instruction name a sequence that works". A third block naming only the
    copy step collapses into the first block's mention and disappears. G-31-123c was exactly that:
    two refusal blocks were corrected and a third kept telling the reader to re-run the command
    that produced the state they were in, with the whole-body assertion still green.

    Both ``_error_state.html`` and ``_empty_state.html`` render their instruction into a ``<p>``,
    so paragraphs are the block boundary the page actually has.
    """
    return [
        re.sub(r"<[^>]+>", "", para)
        for para in re.findall(r"<p\b[^>]*>(.*?)</p>", body, flags=re.S)
        if _SCRIPT_PATH_PATTERN.search(para)
    ]


def _module_of(script_path: str) -> ast.Module:
    """Parse a repo-relative script path."""
    return ast.parse((_REPO_ROOT / script_path).read_text(encoding="utf-8"))


def _has_main_guard(module: ast.Module) -> bool:
    """Whether the module carries a top-level ``if __name__ == "__main__":`` guard."""
    for node in module.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if isinstance(left, ast.Name) and left.id == "__name__":
            return True
    return False


def _default_dir_state() -> Any:
    """A comparable snapshot of the protected default artifact directory."""
    if not DEFAULT_BET_LIST_DIR.exists():
        return None
    return sorted(
        (entry.name, entry.stat().st_size, entry.stat().st_mtime_ns)
        for entry in DEFAULT_BET_LIST_DIR.iterdir()
    )


# ---------------------------------------------------------------------------
# ONE. The cold state names the whole sequence, and both named scripts are real
# ---------------------------------------------------------------------------


def test_cold_state_renders_the_cache_absent_copy(tmp_path: pathlib.Path) -> None:
    """The starting point: no ``bet_list`` table, so the page offers an action."""
    db_path = _cold_cache(tmp_path / "cold.duckdb", seed_navigation=True)

    with _serving(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    assert _CACHE_ABSENT_HEADING in response.text, (
        "the cold state did not render the cache-absent empty state, so the rest of this module "
        "would be following copy from some other state"
    )
    assert _ROW_TABLE_HEADER not in response.text, (
        "the cold state served a row table, which would make case TWO vacuous"
    )


def test_cold_state_names_exactly_the_two_scripts_in_order(
    tmp_path: pathlib.Path,
) -> None:
    """Read the sequence OUT of the page rather than hardcoding what is then followed.

    A test that hardcodes the sequence and runs it proves the SEQUENCE works. It does not prove
    the page names it, so copy that silently stopped naming the producer would still pass. Parsing
    the rendered body is what makes this module fail when the copy regresses.
    """
    db_path = _cold_cache(tmp_path / "cold.duckdb", seed_navigation=True)

    with _serving(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert _named_scripts(body) == [_GENERATE_SCRIPT, _POPULATE_SCRIPT], (
        "the cache-absent state does not name the generation script and then the population "
        f"script; it names {_named_scripts(body)}. Naming only the copy step is the shipped "
        "defect: from this exact state that command has nothing to copy."
    )


@pytest.mark.parametrize("script_path", [_GENERATE_SCRIPT, _POPULATE_SCRIPT])
def test_cold_state_named_script_is_a_runnable_entry_point(script_path: str) -> None:
    """Each named script exists, defines ``main``, and carries a ``__main__`` guard.

    "Runnable entry point" is CHECKED rather than assumed. A copy string naming a path that does
    not exist, or a module with no way to invoke it, is the same class of defect as a copy string
    naming a command that cannot reach its outcome.
    """
    path = _REPO_ROOT / script_path
    assert path.is_file(), (
        f"the /bets recovery copy names {script_path}, which is not on disk"
    )

    module = _module_of(script_path)
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert "main" in functions, f"{script_path} defines no top-level main()"
    assert _has_main_guard(module), (
        f'{script_path} carries no `if __name__ == "__main__":` guard, so the command the page '
        "tells the reader to run would do nothing"
    )


# ---------------------------------------------------------------------------
# TWO. Follow them, in the order the page names them, and the page serves a list
# ---------------------------------------------------------------------------


def test_follow_the_named_sequence_and_the_page_serves_the_row_table(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE TRUTH STATEMENT: following the on-screen instruction serves a list.

    Asserted on the SERVED BODY, not on the database. A row in ``bet_list`` that the route
    filters out, the template skips, or the freshness fence withholds is not a served list, and
    every one of those has been a real failure mode on this page.
    """
    default_before = _default_dir_state()
    db_path = _cold_cache(tmp_path / "cold.duckdb", seed_navigation=False)
    artifact_dir = tmp_path / "bet_list"
    silver_dir = _write_authored_silver(tmp_path / "silver")

    # Stage one: the generation command the page names FIRST.
    graded = _stage_one_generate(artifact_dir, tmp_path, monkeypatch)
    assert not graded.empty, (
        "stage one produced no rows, so stage two would copy nothing"
    )

    # Stage two: the population command the page names SECOND.
    _stage_two_populate(db_path, artifact_dir, silver_dir)

    with _serving(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _ROW_TABLE_HEADER in body, (
        "following the sequence the page itself names did NOT produce a served list. This is the "
        "G-31-123b truth statement and it is the claim the two prior string assertions could not "
        "make."
    )
    assert f"/games/{_GAME_ONE}" in body, (
        "the served row table names none of the authored games, so the header assertion above "
        "could be passing on a table with no rows in it"
    )
    assert _CACHE_ABSENT_HEADING not in body, (
        "the cache-absent empty state still renders after the rows were loaded"
    )
    assert _HARD_BLOCK_MESSAGE not in body, (
        "the page flipped into the withheld refusal after a COMPLETE recovery; the per-week "
        "populated-at marker was not stamped, or it did not postdate the week freeze"
    )
    assert _default_dir_state() == default_before, (
        f"{DEFAULT_BET_LIST_DIR.as_posix()} changed during a run directed at a temporary "
        "directory; some write is not honouring output_dir"
    )


# ---------------------------------------------------------------------------
# THREE. The negative control: the copy step ALONE does not get there
# ---------------------------------------------------------------------------


def test_copy_only_from_a_cold_state_does_not_serve_a_list(
    tmp_path: pathlib.Path,
) -> None:
    """Run ONLY stage two, against an EMPTY artifact directory. Exactly the reported state.

    This is what makes case TWO mean something: without it, case TWO proves a sequence works
    without establishing that the shorter sequence does not.

    The reported chain, reproduced: the reader runs the copy step, it reads an absent artifact
    through ``read_bet_list_cache_sources``, which degrades to a zero-row frame rather than
    raising, so it exits successfully; ``materialize_bet_list_with_marker`` returns early on an
    empty frame and stamps NOTHING; but ``bet_week_freeze`` is built from the SCHEDULE
    independently of the bet rows, so a freeze exists with no marker beside it -- which
    ``_is_bet_cache_stale`` treats as stale. The page therefore has no rows to withhold and
    withholds them anyway, which is the honest render of a failed insertion.
    """
    db_path = _cold_cache(tmp_path / "cold.duckdb", seed_navigation=False)
    empty_artifacts = tmp_path / "never_generated"
    silver_dir = _write_authored_silver(tmp_path / "silver")

    _stage_two_populate(db_path, empty_artifacts, silver_dir)

    with _serving(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _ROW_TABLE_HEADER not in body, (
        "the copy step alone served a row table, which would mean the generation command the copy "
        "now names is not actually required and this whole gap was misdiagnosed"
    )
    assert _HARD_BLOCK_MESSAGE in body, (
        "the copy step alone did not land on the withheld refusal. Anything else here is worse "
        "than the refusal: an empty week rendered as a plausible result is a claim about the "
        "models made from the absence of an insertion."
    )


def test_copy_only_refusal_names_both_commands_so_the_loop_is_broken(
    tmp_path: pathlib.Path,
) -> None:
    """THE LOOP: the state the copy step lands you in must point FORWARD, not back.

    Before this plan the refusal named ``scripts/populate_cache.py`` and nothing else -- the very
    command that had just put the reader here. Re-running it reproduced the same state. The
    refusal must now name the producer, so the reader has somewhere to go.
    """
    db_path = _cold_cache(tmp_path / "cold.duckdb", seed_navigation=False)
    silver_dir = _write_authored_silver(tmp_path / "silver")
    _stage_two_populate(db_path, tmp_path / "never_generated", silver_dir)

    with _serving(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    for paragraph in _instructional_paragraphs(body):
        assert _named_scripts(paragraph) == [_GENERATE_SCRIPT, _POPULATE_SCRIPT], (
            "a block that gives the reader a command names a sequence that cannot restore what it "
            f"refers to; it names {_named_scripts(paragraph)}. The offending block reads:\n"
            f"{paragraph.strip()[:400]}\n"
            "Every instructional block must name the producer before the copy step -- naming only "
            "the copy step is the loop, whichever block does it. The whole-body assertion below "
            "cannot catch this: it deduplicates, so a repeat of the copy step alone is invisible."
        )

    assert _named_scripts(body) == [_GENERATE_SCRIPT, _POPULATE_SCRIPT], (
        "the refusal a copy-step-only reader lands on does not name the generation script before "
        f"the population script; it names {_named_scripts(body)}. That is the loop: the only "
        "command offered is the one that produced this state."
    )


# ---------------------------------------------------------------------------
# FOUR. The substitution guard: this module drives the writers production drives
# ---------------------------------------------------------------------------


def test_writer_set_driven_here_equals_the_set_populate_cache_calls() -> None:
    """Walk ``populate_cache`` and pin its bet-list, schedule and tracker writers.

    Without this, stage two is an assertion about the production path made by a test that does not
    touch it -- ``populate_cache`` could start calling a different writer and this module would
    stay green while covering nothing. Exactness matters in both directions: an ADDED writer is
    one this module is not exercising, and a REMOVED one means it is exercising a path production
    abandoned.
    """
    module = ast.parse(_CACHE_MODULE.read_text(encoding="utf-8"))
    populate = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == _POPULATE_FUNCTION
        ),
        None,
    )
    assert populate is not None, (
        f"{_POPULATE_FUNCTION} was not found in {_CACHE_MODULE.name}; this guard has lost its "
        "subject and would otherwise pass vacuously"
    )

    called = {
        node.func.id
        for node in ast.walk(populate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("materialize_")
    }
    assert called == _DRIVEN_WRITERS, (
        f"{_POPULATE_FUNCTION} calls {sorted(called)} but this module drives "
        f"{sorted(_DRIVEN_WRITERS)}. The hermetic substitution has drifted from the production "
        "path, so every served-page assertion above is now about a path production does not take."
    )
