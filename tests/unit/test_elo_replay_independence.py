"""The Elo replay is independent of the builder, and it recomputes what it claims to.

WHAT IS BEING GUARDED
---------------------
``audit/elo_replay.py`` is the CONTENT evidence for the Elo source's information times.
The provenance Plan 33.2-01 wrote for Elo is RULE-derived: it asserts the rule the builder
already follows. A rule checked with the same rule proves the rule, not the content
(RESEARCH pitfall P3). The replay recomputes every pre-game rating from only the results
that ended at or before that game's lock and compares it with ``==`` against silver
``elo_game_snapshots``.

That comparison is worthless if the replay reaches the thing it checks. A replay that
imported ``scripts.build_elo``'s chain would agree with it on every row and prove nothing.
So this module has two halves:

1. BEHAVIOUR on small in-memory fixtures: the replay matches the canonical chain, the
   at-lock boundary is inclusive, a missing snapshot row fails by name, a mismatch names
   its game and column, vacuous rows are excluded from the evidence count, tied kickoffs
   commute, and every permitted per-game formula actually fires.
2. AN AST SOURCE SCAN over ``audit/elo_replay.py`` across three forbidden families --
   imports, legacy table names, and forbidden ``ratings.elo`` calls -- with the four
   structural controls (non-vacuity, the assertion, a planted violation per name, and
   no false positive).

MATCH MODE FOR TABLE NAMES: EQUALITY, NEVER SUBSTRING. A table name is flagged when an
``ast.Constant`` string's VALUE EQUALS it. The replay's own docstring must name every
forbidden table and call (Plan 33.2-04 Task 1), and a module docstring is itself one
``ast.Constant``. Under substring matching the scan would flag the module it exists to
clear; under equality the docstring's value is the whole docstring and equals no name.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# The synthetic fixture: two seasons, six teams, three weeks each, with a Thursday
# game, a Monday game, two tied 13:00 ET kickoffs every week, and one unplayed
# week at the end of the second season (the provisional week).
# ---------------------------------------------------------------------------

_WEEK_SUNDAYS: dict[int, datetime] = {
    2002: datetime(2002, 9, 8),
    2003: datetime(2003, 9, 7),
}

# (week, day offset from that week's Sunday, ET hour, ET minute, home, away,
#  home score, away score). Offsets: -3 = Thursday, 0 = Sunday, +1 = Monday.
_SCHEDULE: tuple[tuple[int, int, int, int, str, str, int, int], ...] = (
    (1, 0, 13, 0, "BUF", "MIA", 24, 17),
    (1, 0, 13, 0, "NE", "NYJ", 20, 23),
    (1, 0, 16, 25, "DAL", "PHI", 31, 10),
    (2, -3, 20, 20, "MIA", "NE", 14, 13),
    (2, 0, 13, 0, "BUF", "DAL", 7, 28),
    (2, 0, 13, 0, "NYJ", "PHI", 21, 21),
    (3, 0, 13, 0, "NE", "BUF", 35, 3),
    (3, 0, 13, 0, "PHI", "MIA", 17, 20),
    (3, 1, 20, 15, "NYJ", "DAL", 27, 24),
)


def _kickoff_utc(season: int, week: int, day_offset: int, hour: int, minute: int):
    sunday = _WEEK_SUNDAYS[season] + timedelta(days=7 * (week - 1) + day_offset)
    local = datetime.combine(sunday.date(), time(hour, minute), tzinfo=ET)
    return pd.Timestamp(local).tz_convert("UTC")


def synthetic_games(*, unplayed_final_week: bool = True) -> pd.DataFrame:
    """Silver-``games``-shaped rows for 2002 and 2003.

    2003 week 3 is left UNPLAYED (null scores) when *unplayed_final_week* is True, so the
    fixture carries a provisional week exactly like the live 2026 table does.
    """
    rows: list[dict[str, object]] = []
    for season in (2002, 2003):
        for week, offset, hour, minute, home, away, hs, as_ in _SCHEDULE:
            if season == 2003:
                # Reverse the home side so 2003 is not a copy of 2002.
                home, away, hs, as_ = away, home, as_ + 3, hs
            unplayed = unplayed_final_week and season == 2003 and week == 3
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_{away}@{home}",
                    "season": season,
                    "week": week,
                    "kickoff_et": _kickoff_utc(season, week, offset, hour, minute),
                    "home_team": home,
                    "away_team": away,
                    "home_score": float("nan") if unplayed else float(hs),
                    "away_score": float("nan") if unplayed else float(as_),
                }
            )
    frame = pd.DataFrame(rows)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def canonical_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """The CANONICAL chain's snapshots for *games* -- the thing the replay is checked against.

    Tests may import the builder; only ``audit/elo_replay.py`` may not. Running the real
    ``_process_chain`` and ``snapshot_upcoming_week`` here is what makes the fixture a
    cross-check of two independent derivations rather than of the replay against itself.
    """
    from scripts.build_elo import EloBuilder

    builder = EloBuilder()
    seasons = sorted(int(s) for s in games["season"].unique())
    rows, _ = builder._process_chain(seasons, games=games, learn_from=games)
    real = pd.DataFrame(rows)
    unplayed = games[games["home_score"].isna() | games["away_score"].isna()]
    frames = [real]
    weeks = pd.DataFrame(unplayed[["season", "week"]]).drop_duplicates()
    for season, week in zip(weeks["season"], weeks["week"], strict=True):
        frames.append(
            builder.snapshot_upcoming_week(int(season), int(week), games=games)
        )
    return pd.concat(frames, ignore_index=True)


AFTER_EVERYTHING = datetime(2004, 1, 1, tzinfo=UTC)


def _replay(**kwargs):
    from audit.elo_replay import replay

    kwargs.setdefault("as_of", AFTER_EVERYTHING)
    return replay(**kwargs)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestTheReplayMatchesTheCanonicalChain:
    def test_zero_mismatches_on_every_row_including_the_provisional_week(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)

        assert result.mismatches.empty, result.mismatches.to_string()
        assert result.missing_snapshot_rows == ()
        assert result.orphan_snapshot_rows == ()
        assert result.compared > 0
        assert result.compared + result.vacuous == len(snapshots)
        assert result.ok

    def test_the_provisional_rows_are_compared_not_skipped(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        provisional_ids = set(snapshots.loc[snapshots["is_provisional"], "game_id"])
        assert len(provisional_ids) == 3
        result = _replay(seasons=[2003], games_df=games, snapshots_df=snapshots)
        assert provisional_ids <= set(result.replayed["game_id"])

    def test_vacuous_rows_are_the_first_season_week_one_rows(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)
        week_one_2002 = snapshots[
            (snapshots["season"] == 2002) & (snapshots["week"] == 1)
        ]

        assert result.vacuous == len(week_one_2002) == 3
        assert result.compared == len(snapshots) - 3
        # 2002 hfa_used is the initial value on every 2002 row: reported as vacuous CELLS
        # in the rows that stay in the evidence count.
        assert result.vacuous_cells == int((snapshots["season"] == 2002).sum()) - 3

    def test_a_season_filter_still_replays_from_the_chain_start(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(seasons=[2003], games_df=games, snapshots_df=snapshots)
        assert result.mismatches.empty, result.mismatches.to_string()
        assert result.vacuous == 0
        assert result.compared == int((snapshots["season"] == 2003).sum())


class TestTheLockBoundary:
    """A game ending exactly AT another game's lock is applied; one second later is not."""

    @staticmethod
    def _games_with_prior_end_at(offset: timedelta) -> pd.DataFrame:
        """DAL's week-1 game is moved so it ENDS at (DAL's week-2 lock + *offset*).

        DAL's week-1 opponent (PHI) next plays on the Sunday, after that lock, so moving
        the game changes nothing about either team's state except whether it is applied.
        """
        from utils.game_lock import game_lock

        games = synthetic_games()
        target = games["game_id"] == "2002_W02_DAL@BUF"
        prior = games["game_id"] == "2002_W01_PHI@DAL"
        lock = game_lock(games.loc[target, "kickoff_et"].iloc[0])
        end = pd.Timestamp(lock).tz_convert("UTC") + offset
        games.loc[prior, "kickoff_et"] = end - timedelta(hours=4)
        return games

    def _dal_week_two_pre(self, games: pd.DataFrame) -> float:
        result = _replay(
            seasons=[2002],
            games_df=games,
            snapshots_df=canonical_snapshots(synthetic_games()),
            duration_hours=4.0,
        )
        row = result.replayed.set_index("game_id").loc["2002_W02_DAL@BUF"]
        return float(row["away_elo_pre"])

    def test_at_lock_is_applied(self) -> None:
        at_lock = self._dal_week_two_pre(self._games_with_prior_end_at(timedelta(0)))
        reference = self._dal_week_two_pre(synthetic_games())
        assert at_lock == reference
        assert at_lock != 1500.0

    def test_one_second_after_the_lock_is_not_applied(self) -> None:
        late = self._dal_week_two_pre(
            self._games_with_prior_end_at(timedelta(seconds=1))
        )
        at_lock = self._dal_week_two_pre(self._games_with_prior_end_at(timedelta(0)))
        assert late != at_lock
        assert late == 1500.0


class TestTheFailureReports:
    def test_a_missing_snapshot_row_is_named_and_fails(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        dropped = "2003_W03_NE@BUF"
        assert dropped in set(snapshots["game_id"])
        thinned = snapshots[snapshots["game_id"] != dropped]

        result = _replay(seasons=[2003], games_df=games, snapshots_df=thinned)

        assert result.missing_snapshot_rows == (dropped,)
        assert not result.ok

    def test_an_unplayed_game_whose_lock_has_not_passed_is_not_required(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        thinned = snapshots[~snapshots["is_provisional"]]
        before_week_three = datetime(2003, 9, 15, tzinfo=UTC)

        result = _replay(
            seasons=[2003],
            games_df=games,
            snapshots_df=thinned,
            as_of=before_week_three,
        )

        assert result.missing_snapshot_rows == ()
        assert result.ok

    def test_an_orphan_snapshot_row_is_named_and_fails(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        orphan = snapshots.iloc[[5]].copy()
        orphan["game_id"] = "2003_W09_XXX@YYY"
        orphan["season"] = 2003
        planted = pd.concat([snapshots, orphan], ignore_index=True)

        result = _replay(seasons=[2003], games_df=games, snapshots_df=planted)

        assert result.orphan_snapshot_rows == ("2003_W09_XXX@YYY",)
        assert not result.ok

    def test_a_mismatch_names_the_game_the_column_and_both_values(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games).copy()
        target = snapshots["game_id"] == "2002_W03_BUF@NE"
        stored = float(snapshots.loc[target, "away_elo_uncertainty"].iloc[0])
        snapshots.loc[target, "away_elo_uncertainty"] = stored + 1e-9

        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)

        assert len(result.mismatches) == 1
        row = result.mismatches.iloc[0]
        assert row["game_id"] == "2002_W03_BUF@NE"
        assert row["column"] == "away_elo_uncertainty"
        assert row["replayed"] == stored
        assert row["stored"] == stored + 1e-9
        assert not result.ok

    def test_an_empty_snapshot_source_is_refused_not_compared(self) -> None:
        from audit.elo_replay import EmptySnapshotSourceError

        games = synthetic_games()
        empty = canonical_snapshots(games).iloc[0:0]
        with pytest.raises(EmptySnapshotSourceError):
            _replay(seasons=[2002], games_df=games, snapshots_df=empty)


class TestTheDeclaredConstantsAreNotLoadBearing:
    @pytest.mark.parametrize("hours", [0.0, 4.0, 8.0])
    def test_duration_does_not_change_any_replayed_value(self, hours: float) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(
            seasons=[2002, 2003],
            games_df=games,
            snapshots_df=snapshots,
            duration_hours=hours,
        )
        assert result.mismatches.empty, result.mismatches.to_string()

    def test_permuting_tied_kickoffs_changes_no_replayed_value(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        forward = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)
        reversed_rank = {
            gid: rank for rank, gid in enumerate(sorted(games["game_id"], reverse=True))
        }
        backward = _replay(
            seasons=[2002, 2003],
            games_df=games.iloc[::-1].reset_index(drop=True),
            snapshots_df=snapshots,
            tie_rank=reversed_rank,
        )

        assert forward.tied_kickoff_groups > 0
        pd.testing.assert_frame_equal(
            forward.replayed.sort_values("game_id").reset_index(drop=True),
            backward.replayed.sort_values("game_id").reset_index(drop=True),
            check_exact=True,
        )


# ---------------------------------------------------------------------------
# The independence source scan (Plan 33.2-04 Task 2)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# DECLARED, never globbed: a future audit module can neither join this scan by accident
# nor slip out of it silently. test_the_scanned_module_list_is_declared_not_globbed
# asserts both the value and the literal-tuple shape of this assignment.
SCANNED_MODULES: tuple[str, ...] = ("audit/elo_replay.py",)

# Family 1: imports, matched on ast.Import / ast.ImportFrom anywhere in the module,
# including a lazy import inside a function body.
FORBIDDEN_IMPORTS: tuple[str, ...] = ("scripts.build_elo",)

# Family 2: the legacy-pass tables, matched on ast.Constant string VALUES by EQUALITY.
FORBIDDEN_TABLE_NAMES: tuple[str, ...] = (
    "games_with_elo",
    "elo_rating_history",
    "elo_ratings_current",
    "elo_ratings.json",
)

# Family 3: the ratings.elo members that hand ordering back to the library, write a
# production file, or read the legacy state file -- matched on ast.Call by attribute or
# name.
FORBIDDEN_CALLS: tuple[str, ...] = (
    "process_season_chronologically",
    "save_ratings",
    "load_ratings",
)

ALL_FORBIDDEN: tuple[str, ...] = (
    *FORBIDDEN_IMPORTS,
    *FORBIDDEN_TABLE_NAMES,
    *FORBIDDEN_CALLS,
)

# One planted violation per forbidden name, written by hand. A name added to a family
# without a control here fails test_every_forbidden_name_has_a_planted_control.
PLANTED_FRAGMENTS: dict[str, str] = {
    "scripts.build_elo": (
        "def lazy():\n    from scripts.build_elo import EloBuilder\n    return EloBuilder\n"
    ),
    "games_with_elo": 'TABLE = "games_with_elo"\n',
    "elo_rating_history": 'TABLE = "elo_rating_history"\n',
    "elo_ratings_current": 'TABLE = "elo_ratings_current"\n',
    "elo_ratings.json": 'PATH = "elo_ratings.json"\n',
    "process_season_chronologically": (
        "def run(system, games):\n"
        "    return system.process_season_chronologically(games, 2002)\n"
    ),
    "save_ratings": "def run(system):\n    system.save_ratings()\n",
    "load_ratings": "def run(system):\n    system.load_ratings()\n",
}

PERMITTED_FORMULAS: tuple[str, ...] = (
    "get_or_create_rating",
    "apply_season_carryover",
    "update_ratings",
    "learn_home_field_advantage",
    "predict_game",
)


def _dotted_imports(node: ast.AST) -> list[str]:
    """Every fully dotted module name an import node can bind."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
    return []


def _called_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def scan_source(source: str) -> list[str]:
    """Every forbidden reference in *source*, as ``family:name:line`` strings.

    Read off the PARSED tree, never the text: comments are absent from the AST, so the
    replay's mandated comment naming ``scripts.build_elo`` cannot trip the scan.

    MATCH MODE:
    * imports -- an ``ast.Import`` / ``ast.ImportFrom`` whose dotted name equals, or is a
      submodule of, a forbidden module (``from scripts import build_elo`` included);
    * table names -- an ``ast.Constant`` string whose VALUE EQUALS a forbidden table name.
      EQUALITY, NEVER SUBSTRING: a module docstring is one ``ast.Constant`` whose value is
      the whole docstring, so the docstring the replay is REQUIRED to carry, naming every
      forbidden table, is not flagged -- and a bare ``"games_with_elo"`` literal is;
    * calls -- an ``ast.Call`` whose function attribute or name equals a forbidden call.

    A string constant EQUAL to a forbidden module or call name is also flagged, so the
    indirections ``importlib.import_module("scripts.build_elo")`` and
    ``getattr(system, "save_ratings")`` are caught by the same equality rule.
    """
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        line = getattr(node, "lineno", 0)
        for dotted in _dotted_imports(node):
            for forbidden in FORBIDDEN_IMPORTS:
                if dotted == forbidden or dotted.startswith(f"{forbidden}."):
                    hits.append(f"import:{forbidden}:{line}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            hits.extend(
                f"constant:{forbidden}:{line}"
                for forbidden in ALL_FORBIDDEN
                if node.value == forbidden
            )
        if isinstance(node, ast.Call):
            called = _called_name(node)
            if called in FORBIDDEN_CALLS:
                hits.append(f"call:{called}:{line}")
    return sorted(set(hits))


def scan_module(relative_path: str) -> list[str]:
    return scan_source((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


class TestTheIndependenceScan:
    """Four controls: non-vacuity, the assertion, a planted violation, no false positive."""

    def test_the_forbidden_families_are_non_empty_and_complete(self) -> None:
        """NON-VACUITY: a truncated list would pass the module while checking less."""
        assert len(FORBIDDEN_IMPORTS) == 1
        assert len(FORBIDDEN_TABLE_NAMES) == 4
        assert len(FORBIDDEN_CALLS) == 3
        total = (
            len(FORBIDDEN_IMPORTS) + len(FORBIDDEN_TABLE_NAMES) + len(FORBIDDEN_CALLS)
        )
        assert total == 8
        for relative in SCANNED_MODULES:
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            assert sum(1 for _ in ast.walk(ast.parse(source))) > 100, relative

    @pytest.mark.parametrize("relative", SCANNED_MODULES)
    def test_the_replay_is_clean_on_all_three_families(self, relative: str) -> None:
        """THE ASSERTION."""
        assert scan_module(relative) == []

    def test_every_forbidden_name_has_a_planted_control(self) -> None:
        assert set(PLANTED_FRAGMENTS) == set(ALL_FORBIDDEN)
        assert len(PLANTED_FRAGMENTS) == len(ALL_FORBIDDEN) == 8

    @pytest.mark.parametrize("name", ALL_FORBIDDEN)
    def test_a_planted_violation_is_flagged(self, name: str) -> None:
        """PLANTED VIOLATION: the scan fires on each forbidden name, not only finds none."""
        hits = scan_source(PLANTED_FRAGMENTS[name])
        assert any(f":{name}:" in hit for hit in hits), (name, hits)

    @pytest.mark.parametrize(
        "source",
        [
            "import scripts.build_elo\n",
            "from scripts.build_elo import ELO_SNAPSHOT_COLUMNS\n",
            "from scripts import build_elo\n",
            'import importlib\nimportlib.import_module("scripts.build_elo")\n',
        ],
    )
    def test_every_spelling_of_the_builder_import_is_flagged(self, source: str) -> None:
        assert any("scripts.build_elo" in hit for hit in scan_source(source)), source

    def test_the_permitted_per_game_formulas_are_not_flagged(self) -> None:
        """NO FALSE POSITIVE (a): the legitimate shape the replay actually uses."""
        source = (
            "from ratings.elo import EloRatingSystem\n"
            "def run(games):\n"
            "    system = EloRatingSystem()\n"
            "    system.apply_season_carryover(2003)\n"
            "    system.learn_home_field_advantage(games, 2003)\n"
            '    system.get_or_create_rating("BUF", 2003)\n'
            '    system.update_ratings("BUF", "MIA", 24, 17, 2003, None)\n'
            '    return system.predict_game("BUF", "MIA", 2003)\n'
        )
        assert scan_source(source) == []

    def test_a_docstring_naming_every_forbidden_symbol_is_not_flagged_by_equality(
        self,
    ) -> None:
        """NO FALSE POSITIVE (b): the exact shape the replay's own docstring must take.

        Under substring matching this module would be flagged; under equality it is not.
        A later switch to substring matching fails HERE rather than silently flagging the
        module the scan exists to clear.
        """
        names = ", ".join(ALL_FORBIDDEN)
        source = f'"""This module may not touch any of: {names}."""\nVALUE = 1\n'
        for name in ALL_FORBIDDEN:
            assert name in source
        assert scan_source(source) == []

    def test_the_scanned_module_list_is_declared_not_globbed(self) -> None:
        assert SCANNED_MODULES == ("audit/elo_replay.py",)
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        declared = [
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SCANNED_MODULES"
        ]
        assert len(declared) == 1
        literal = declared[0]
        assert isinstance(literal, ast.Tuple)
        assert all(isinstance(element, ast.Constant) for element in literal.elts)


class TestTheReplayCallsTheRealFormulas:
    """A structural scan proves a call site EXISTS; this proves each one RUNS."""

    @staticmethod
    def unfired(counters: dict[str, int]) -> list[str]:
        """The permitted formulas a run never called, in declaration order."""
        return [name for name in PERMITTED_FORMULAS if counters.get(name, 0) == 0]

    @staticmethod
    def _spy_on(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Wrap each permitted method with a counter that delegates to the original."""
        from ratings.elo import EloRatingSystem

        counters = dict.fromkeys(PERMITTED_FORMULAS, 0)

        def counting(name: str, original):
            def wrapper(self, *args, **kwargs):
                counters[name] += 1
                return original(self, *args, **kwargs)

            return wrapper

        for name in PERMITTED_FORMULAS:
            original = getattr(EloRatingSystem, name)
            monkeypatch.setattr(EloRatingSystem, name, counting(name, original))
        return counters

    def test_replay_invokes_every_permitted_formula(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)  # built BEFORE the spies go on
        counters = self._spy_on(monkeypatch)

        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)

        assert result.ok
        assert self.unfired(counters) == [], counters

    def test_a_planted_local_arithmetic_replay_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PLANTED CONTROL: computing one update by hand leaves ``update_ratings`` unfired."""
        from ratings.elo import EloRatingSystem

        counters = self._spy_on(monkeypatch)

        def stand_in_replay(games: pd.DataFrame) -> float:
            system = EloRatingSystem()
            system.apply_season_carryover(2002)
            system.learn_home_field_advantage(games, 2002)
            home = system.get_or_create_rating("BUF", 2002)
            away = system.get_or_create_rating("MIA", 2002)
            expected = system.predict_game("BUF", "MIA", 2002)["home_win_prob"]
            home.rating += 20.0 * (1.0 - expected)  # the update, done locally
            away.rating -= 20.0 * (1.0 - expected)
            return home.rating

        stand_in_replay(synthetic_games())

        assert self.unfired(counters) == ["update_ratings"]


def test_compared_columns_match_the_gold_join_subset() -> None:
    """The TEST owns the link to the test-owned constant; the audit module imports none."""
    from audit.elo_replay import COMPARED_COLUMNS
    from tests.phase33_state import ELO_GOLD_JOIN_SUBSET

    assert tuple(c for c in ELO_GOLD_JOIN_SUBSET if c != "game_id") == COMPARED_COLUMNS
    assert len(COMPARED_COLUMNS) == 6


_SUBPROCESS_REPLAY = """
import sys
import pandas as pd
from audit.elo_replay import replay
games = pd.DataFrame({
    "game_id": ["2002_W01_MIA@BUF"], "season": [2002], "week": [1],
    "kickoff_et": pd.to_datetime(["2002-09-08T17:00:00Z"]),
    "home_team": ["BUF"], "away_team": ["MIA"],
    "home_score": [24.0], "away_score": [17.0],
})
snapshots = pd.DataFrame({
    "game_id": ["2002_W01_MIA@BUF"], "season": [2002], "week": [1],
    "home_elo_pre": [1500.0], "away_elo_pre": [1500.0],
    "home_elo_uncertainty": [350.0], "away_elo_uncertainty": [350.0],
    "elo_prob_home": [0.5], "hfa_used": [48.0],
})
replay([2002], games_df=games, snapshots_df=snapshots)
print("LOADED=", sorted(m for m in sys.modules if "build_elo" in m))
"""


def test_running_the_replay_never_loads_the_builder_module() -> None:
    """RUNTIME half of the import family: a real replay leaves scripts.build_elo unloaded.

    Run in a fresh interpreter, because this test process has already imported the builder
    to produce the canonical fixture.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_REPLAY],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "LOADED= []" in completed.stdout, completed.stdout + completed.stderr
