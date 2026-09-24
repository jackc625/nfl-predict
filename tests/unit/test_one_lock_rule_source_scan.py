"""ONE lock rule, many readers -- proven BY IDENTITY, then by shape (D33.2-01, Plan 33.2-02).

WHY IDENTITY AND NOT A SOURCE GREP
----------------------------------
A source scan can only show that a NAME appears in N files. It cannot show that the N names
resolve to the same object -- a module defining its own ``game_lock`` would satisfy the grep and
defeat the property entirely. So the PRIMARY assertion here replaces ``utils.game_lock.game_lock``
with a COUNTING DELEGATE and drives every declared production reader: if any reader reached a
different object, its call would not land on the shared counter and its delta would be zero.
The shape scan is the SECONDARY, structural control. This is the argument
``tests/unit/test_freeze_parse_single_source.py`` makes for the strict timestamp parser, carried
over to the lock rule itself.

WHY THE COUNTER'S SILENCE MEANS "NOT CALLED" AND NOT "NOT VISIBLE" -- THE BINDING CONTROL
----------------------------------------------------------------------------------------
``monkeypatch.setattr(utils.game_lock, "game_lock", stub)`` is observable only by a reader that
resolves the name at CALL time. A module-level ``from utils.game_lock import game_lock`` binds
the original function object once, at import, so that reader's calls never reach the counter --
and a reader invisible to the counter is indistinguishable from a reader that does not exist.
The fifth control below therefore asserts, over the parsed tree of every production module, that
no MODULE-level ``from utils.game_lock import`` names ``game_lock``, ``is_admissible`` or
``lock_frame``. The permitted spellings -- ``import utils.game_lock as lock_rule`` with attribute
access, or a FUNCTION-local import -- both resolve per call. ``features.provenance.build_lock_frame``
is a wrapper function rather than an alias for exactly this reason.

WHY THE SECOND SCAN MATCHES A SHAPE AND NOT A LITERAL
-----------------------------------------------------
A literal-string scan for a retired name is defeated by a rename. The derivations this plan
removed all had the same SHAPE: a function whose body computes a WEEKDAY OFFSET, pins a FIXED
HOUR OF DAY, and returns an instant. MEASURED 2026-09-21 against the pre-sweep commit ``40c4dce``,
exactly this shape flags the four function-level rival derivations Plan 33.2-02 deleted and
nothing else; against the current tree it flags nothing. This module names no retired symbol --
its planted violation is written as the shape, so the phase's retired-name scans stay green.

THE FIVE STRUCTURAL CONTROLS, EACH IN ITS OWN TEST
--------------------------------------------------
1. non-vacuity -- the declared reader set is non-empty and exactly the declared size;
2. the assertion -- every reader's call delta is positive in one run;
3. a planted violation -- a local weekday-offset-and-fixed-hour function IS flagged;
4. no false positive -- a function that calls ``game_lock`` is NOT flagged;
5. the binding control -- no module-level function import of the rule anywhere in production.

Run this module:  uv run pytest tests/unit/test_one_lock_rule_source_scan.py -q

Sandboxed: every frame is built in memory or under ``tmp_path``; no production store is read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import utils.game_lock as lock_rule

REPO_ROOT = Path(__file__).resolve().parents[2]

# The production roots every scan below covers.
PRODUCTION_ROOTS: tuple[str, ...] = (
    "scripts",
    "features",
    "backtest",
    "models",
    "utils",
    "conf",
    "api",
    "pipeline",
    "data",
    "ratings",
)

# The rule's three public callables a module must never bind at module level.
GUARDED_NAMES: frozenset[str] = frozenset({"game_lock", "is_admissible", "lock_frame"})

_SUNDAY_GAMEDAY = "2026-09-20"
_SUNDAY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00Z")


class _CountingLock:
    """A delegate that COUNTS and then calls the real rule, so every downstream value holds."""

    def __init__(self, real: Callable[..., datetime]) -> None:
        self._real = real
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> datetime:
        self.calls += 1
        return self._real(*args, **kwargs)


@pytest.fixture
def counting_lock(monkeypatch: pytest.MonkeyPatch) -> _CountingLock:
    stub = _CountingLock(lock_rule.game_lock)
    monkeypatch.setattr(lock_rule, "game_lock", stub)
    return stub


# ---------------------------------------------------------------------------
# The declared production readers, each driven through its real entry point.
# ---------------------------------------------------------------------------


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_W02_CAR@ATL", "2026_W02_JAX@DEN"],
            "season": [2026, 2026],
            "week": [2, 2],
            "home_team": ["ATL", "DEN"],
            "away_team": ["CAR", "JAX"],
            "kickoff_et": [_SUNDAY_KICKOFF, _SUNDAY_KICKOFF],
        }
    )


def _drive_gameday_lock(_tmp: Path) -> None:
    from scripts.ingest_historical_odds import gameday_lock

    gameday_lock(_SUNDAY_GAMEDAY)


def _drive_gameday_lock_admissibility(_tmp: Path) -> None:
    """Was: ``_drive_is_admissible_at_lock``, which drove a SECOND admissibility helper.

    ``scripts.ingest_historical_odds.is_admissible_at_lock`` was DELETED by Plan 33.2-20:
    it never acquired a production caller (Plans 33.2-13 and 33.2-14 each recorded why they
    reached ``utils.game_lock`` directly instead), and a second admissibility entry point in
    ``scripts/`` is exactly the two-answers shape this module exists to forbid.

    The reader this drives is unchanged in substance: ``gameday_lock`` hands an Eastern
    gameday to the ONE rule, and the verdict is taken from ``utils.game_lock.is_admissible``
    against the lock that rule returns.
    """
    from scripts.ingest_historical_odds import gameday_lock
    from utils.game_lock import is_admissible

    is_admissible("2026-09-18T18:00:00-04:00", gameday_lock(_SUNDAY_GAMEDAY))


def _drive_historical_odds_stamp(_tmp: Path) -> None:
    from scripts.ingest_historical_odds import transform_nfl_odds_with_counts

    row = {
        "season": 2024,
        "week": 1,
        "gameday": "2024-09-08",
        "gametime": "13:00",
        "home_team": "KC",
        "away_team": "BAL",
        "game_type": "REG",
        "spread_line": -3.0,
        "total_line": 46.5,
        "home_moneyline": -150,
        "away_moneyline": 130,
        "home_spread_odds": -110,
        "away_spread_odds": -110,
        "over_odds": -110,
        "under_odds": -110,
    }
    report = transform_nfl_odds_with_counts(pd.DataFrame([row]))
    assert report.admitted == 1


def _drive_bet_selector(_tmp: Path) -> None:
    from backtest.bet_selector import _freshness_context

    _freshness_context(
        {"gameday": _SUNDAY_GAMEDAY, "snapshot_ts": "2026-09-18T18:00:00-04:00"}
    )


def _drive_selection_fence(_tmp: Path) -> None:
    from backtest.weekly_bet_list import select_games_for_decision_instant
    from scripts.ingest_historical_odds import gameday_lock

    schedule = pd.DataFrame(
        [{"game_id": "g", "season": 2026, "week": 2, "gameday": _SUNDAY_GAMEDAY}]
    )
    instant = gameday_lock(_SUNDAY_GAMEDAY)
    select_games_for_decision_instant(schedule, instant, decided_at=instant)


def _drive_bet_week_schedule(tmp: Path) -> None:
    from backtest.weekly_bet_list import build_bet_week_schedule

    silver = tmp / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    _games()[["game_id", "season", "week", "kickoff_et"]].to_parquet(
        silver / "games.parquet", index=False
    )
    assert not build_bet_week_schedule(silver).empty


def _drive_line_movement(_tmp: Path) -> None:
    from features.line_movement import LineMovementBuilder

    LineMovementBuilder._game_lock(_SUNDAY_KICKOFF.to_pydatetime())


def _drive_market_anchors(_tmp: Path) -> None:
    from features.market_anchors import MarketAnchorFeaturesCalculator

    odds = pd.DataFrame(
        {
            "game_id": ["2026_W02_CAR@ATL"],
            "sportsbook": ["consensus"],
            "snapshot_ts": ["2026-09-18T18:00:00-04:00"],
            "total": [44.5],
        }
    )
    MarketAnchorFeaturesCalculator().select_snapshot_lines_at_lock(odds, _games())


def _drive_live_odds_ingest(_tmp: Path) -> None:
    from scripts.ingest_odds import OddsDataIngester

    class _EmptyBoard:
        def get_nfl_odds(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

        def close(self) -> None:
            return None

    ingester = OddsDataIngester.__new__(OddsDataIngester)
    ingester.api_client = _EmptyBoard()
    ingester.sportsbook_priority = []
    # An empty board writes nothing and FAILS by name (33.2 review C1 WR-08); the locks
    # are derived before the request either way.
    from utils import DataIngestionError

    try:
        ingester.ingest_odds(season=2026, week=2, schedule=_games())
    except DataIngestionError as refusal:
        assert "no odds row was captured" in str(refusal)
    else:
        raise AssertionError("an empty odds board was reported as a success")


def _drive_timeline_locks(_tmp: Path) -> None:
    from scripts.ingest_odds_timeline import game_lock_instants

    game_lock_instants(_games())


def _drive_provenance_lock_frame(_tmp: Path) -> None:
    from features.provenance import build_lock_frame

    build_lock_frame(_games())


READERS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("scripts.ingest_historical_odds.gameday_lock", _drive_gameday_lock),
    (
        "scripts.ingest_historical_odds.gameday_lock -> utils.game_lock.is_admissible",
        _drive_gameday_lock_admissibility,
    ),
    (
        "scripts.ingest_historical_odds.transform_nfl_odds_with_counts",
        _drive_historical_odds_stamp,
    ),
    ("backtest.bet_selector._freshness_context", _drive_bet_selector),
    (
        "backtest.weekly_bet_list.select_games_for_decision_instant",
        _drive_selection_fence,
    ),
    ("backtest.weekly_bet_list.build_bet_week_schedule", _drive_bet_week_schedule),
    ("features.line_movement.LineMovementBuilder._game_lock", _drive_line_movement),
    (
        "features.market_anchors.MarketAnchorFeaturesCalculator."
        "select_snapshot_lines_at_lock",
        _drive_market_anchors,
    ),
    ("scripts.ingest_odds.OddsDataIngester.ingest_odds", _drive_live_odds_ingest),
    ("scripts.ingest_odds_timeline.game_lock_instants", _drive_timeline_locks),
    ("features.provenance.build_lock_frame", _drive_provenance_lock_frame),
)

# The declared reader count. An edit that quietly shrinks READERS fails the non-vacuity control.
DECLARED_READER_COUNT = 11


class TestEveryReaderResolvesToTheOneRule:
    """Controls 1 and 2: the reader set is what it says, and every reader reaches the rule."""

    def test_control_1_the_reader_set_is_non_empty_and_exactly_declared(self) -> None:
        assert len(READERS) == DECLARED_READER_COUNT
        assert len({name for name, _ in READERS}) == DECLARED_READER_COUNT

    def test_control_2_every_reader_increments_the_one_counter(
        self, counting_lock: _CountingLock, tmp_path: Path
    ) -> None:
        deltas: dict[str, int] = {}
        for name, drive in READERS:
            before = counting_lock.calls
            drive(tmp_path)
            deltas[name] = counting_lock.calls - before

        silent = sorted(name for name, delta in deltas.items() if delta <= 0)
        assert silent == [], (
            f"these readers never reached utils.game_lock.game_lock: {silent}. Either the "
            "reader derives its own lock (a second rule) or it bound the function at module "
            "level and the counter cannot see it -- the binding control below says which."
        )


# ---------------------------------------------------------------------------
# The shape scan.
# ---------------------------------------------------------------------------

_HOUR_NAME = re.compile(r"HOUR", re.IGNORECASE)


def _computes_a_weekday_offset(function: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"weekday", "isoweekday"}
        for node in ast.walk(function)
    )


def _pins_a_fixed_hour(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "hour" and isinstance(
                    keyword.value, ast.Constant | ast.Name | ast.Attribute
                ):
                    return True
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name == "datetime" and len(node.args) >= 4:
                return True
            if name == "time" and node.args:
                return True
        if (
            isinstance(node, ast.Name)
            and node.id.isupper()
            and _HOUR_NAME.search(node.id)
        ):
            return True
    return False


def _returns_a_value(function: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Return) and node.value is not None
        for node in ast.walk(function)
    )


def rival_derivations(tree: ast.AST) -> list[str]:
    """Functions in *tree* with the retired rule's SHAPE: weekday offset + fixed hour + return."""
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and _computes_a_weekday_offset(node)
        and _pins_a_fixed_hour(node)
        and _returns_a_value(node)
    ]


def _production_trees() -> list[tuple[Path, ast.Module]]:
    files = [
        path for root in PRODUCTION_ROOTS for path in (REPO_ROOT / root).rglob("*.py")
    ]
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in files]


# A planted second rule, written as the SHAPE only -- it names no retired symbol.
_PLANTED_RIVAL = """
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

CUTOFF_HOUR = 18

def my_own_cutoff(day):
    back = (day.weekday() - 4) % 7
    anchor = day - timedelta(days=back)
    return datetime(anchor.year, anchor.month, anchor.day, CUTOFF_HOUR, 0,
                    tzinfo=ZoneInfo("America/New_York"))
"""

_HONEST_READER = """
import utils.game_lock as lock_rule

def my_lock(kickoff):
    return lock_rule.game_lock(kickoff)
"""


class TestNoSecondDerivationSurvivesInProduction:
    """The shape scan, with its planted violation and its no-false-positive control."""

    def test_the_scan_covers_the_production_tree(self) -> None:
        trees = _production_trees()
        assert len(trees) > 100, f"only {len(trees)} production modules were parsed"

    def test_no_production_function_has_the_retired_shape(self) -> None:
        hits = sorted(
            f"{path.relative_to(REPO_ROOT).as_posix()}::{name}"
            for path, tree in _production_trees()
            for name in rival_derivations(tree)
        )
        assert hits == [], (
            f"production code computes a weekday offset against a fixed hour and returns an "
            f"instant in {hits}: that is how a second lock rule gets written. The lock comes "
            "from utils.game_lock (D33.2-01)."
        )

    def test_control_3_a_planted_rival_derivation_is_flagged(self) -> None:
        assert rival_derivations(ast.parse(_PLANTED_RIVAL)) == ["my_own_cutoff"]

    def test_control_4_a_function_calling_the_rule_is_not_flagged(self) -> None:
        assert rival_derivations(ast.parse(_HONEST_READER)) == []


# ---------------------------------------------------------------------------
# The binding control.
# ---------------------------------------------------------------------------


def module_level_rule_bindings(tree: ast.Module) -> list[str]:
    """Names of the rule bound at MODULE level by ``from utils.game_lock import ...``.

    Only top-level statements are inspected: a FUNCTION-local ``from utils.game_lock import``
    re-executes on every call and is therefore observable by the counting delegate, so it is a
    correct spelling and must not be flagged.
    """
    bound: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "") == "utils.game_lock"
        ):
            bound.extend(
                alias.name for alias in node.names if alias.name in GUARDED_NAMES
            )
    return bound


class TestTheCounterCanSeeEveryReader:
    """Control 5: no production module holds its own reference to the rule's functions."""

    def test_control_5_no_module_level_function_binding_of_the_rule(self) -> None:
        trees = _production_trees()
        hits = sorted(
            f"{path.relative_to(REPO_ROOT).as_posix()}: {names}"
            for path, tree in trees
            if (names := module_level_rule_bindings(tree))
        )
        assert len(GUARDED_NAMES) == 3
        assert hits == [], (
            f"{hits} bind the lock rule's functions at module level, so the identity "
            "delegate cannot see their calls and a silent counter would mean 'not visible' "
            "rather than 'not called'. Use `import utils.game_lock as lock_rule` or a "
            "function-local import."
        )

    def test_the_binding_scan_flags_a_planted_module_level_import(self) -> None:
        planted = ast.parse("from utils.game_lock import game_lock, is_admissible\n")
        assert module_level_rule_bindings(planted) == ["game_lock", "is_admissible"]

    def test_the_binding_scan_allows_a_function_local_import(self) -> None:
        local = ast.parse(
            "def reader(kickoff):\n"
            "    from utils.game_lock import game_lock\n"
            "    return game_lock(kickoff)\n"
        )
        assert module_level_rule_bindings(local) == []


# ---------------------------------------------------------------------------
# CLAUDE.md states the rule that is actually in force.
# ---------------------------------------------------------------------------


def _key_constraints_block() -> str:
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*Key constraints:\*\*(.*?)(?=\n## |\Z)", text, re.S)
    return match.group(1) if match else ""


class TestClaudeMdStatesTheDayBeforeLock:
    """The project instructions name the rule in force, not the retired one.

    Scoped to the **Key constraints** block, never the whole file: the project overview
    legitimately narrates the historical Friday orchestrator.
    """

    def test_the_block_is_found(self) -> None:
        assert _key_constraints_block(), "the Key constraints block was not found"

    def test_the_block_states_the_day_before_lock(self) -> None:
        block = _key_constraints_block()
        assert "day before its kickoff" in block
        assert "Friday 6 PM" not in block

    def test_claude_md_is_ascii(self) -> None:
        assert (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8").isascii()


def test_the_rule_module_is_the_one_under_test() -> None:
    """Sanity: the delegate above patches the real rule module, not a copy."""
    assert lock_rule.__name__ == "utils.game_lock"
    assert lock_rule.game_lock(datetime(2026, 9, 20, 17, 0, tzinfo=UTC)) == datetime(
        2026, 9, 19, 18, 0, tzinfo=lock_rule.ET
    )
