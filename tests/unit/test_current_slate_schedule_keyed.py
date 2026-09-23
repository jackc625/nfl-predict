"""The current NFL slate is read from the RECORDED SCHEDULE, never from calendar arithmetic.

Step 24c of Plan 33.2-24 (orchestrator-assigned, owner standing rule: a known bug is fixed at
its root in the same work). Step 24b found and MEASURED the defect while sweeping siblings of the
odds-timeline week count: ``utils.date_utils.get_current_nfl_week`` counted weeks from a COMPUTED
Thursday-after-Labor-Day, so over silver ``games`` 2018-2026 it disagreed with the schedule on 18
games at their own lock instant or kickoff -- every season opener at its lock (a lock before the
computed Thursday returned the PREVIOUS season's week 18), the whole Wednesday 2026 opener, the
2020/2021 Tuesday reschedules and the 17-week-era Super Bowls (a bye week the count cannot see).

Its failure mode is an OMITTED capture or prediction: the resolved slate does not contain the game.
It is the live resolver behind the Friday orchestrator, the staleness gate, the trainers' ``--week``
default, ``generate_bet_list``, ``data_qa`` and several ingests, so every one of them inherited it.

WHAT IS PINNED HERE
-------------------
* Each of the 23 measured (game, instant) disagreements resolves to that game's own week, through
  the PUBLIC ``get_current_nfl_week`` -- and a control proves the retired calendar count gets every
  one of them wrong, so the list is the defect and not a list of cases that always passed.
* The agreement sweep: at EVERY 2018-2026 game's lock instant and kickoff instant, the resolver
  returns that game's own ``(season, week)``.
* The branches where the schedule cannot name a slate, each explicit and each tested on a
  synthetic schedule: the offseason (the retired calendar value, the documented pre-season
  contract, used ONLY when the schedule itself says no slate is open), a season underway with no
  recorded schedule (a named refusal), the playoff gap before the next round is recorded, a
  truncated schedule, an empty or unreadable one, a naive instant, a null kickoff and a day that
  names two slates.
* The callers that decide "in season or not" -- the Friday offseason no-op (Phase 21 D-06) and the
  staleness season gate -- key on the same resolver, and the offseason no-op still exits 0 without
  constructing the orchestrator.
* The three trainers' ``--week`` default: it was a TUPLE compared against the ``week`` column.

Targeted tests only; this module reads production silver ``games`` read-only and writes nothing.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import pytest

from utils.date_utils import (
    ET,
    NFL_REGULAR_SEASON_WEEKS,
    NFL_TOTAL_WEEKS,
    get_current_nfl_week,
    kickoff_wall_clock_et,
)
from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_GAMES = REPO_ROOT / "data" / "silver" / "games.parquet"
SWEEP_SEASONS = range(2018, 2027)

#: The 18 games step 24b measured, as the 23 (game, instant) pairs on which the retired
#: calendar count disagreed with the schedule (re-measured 2026-09-23 before this step wrote
#: anything). ``lock`` is the game's own D33.2-01 lock; ``kickoff`` its kickoff instant.
MEASURED_DISAGREEMENTS: tuple[tuple[str, str], ...] = (
    ("2018_W01_ATL@PHI", "lock"),
    ("2018_W21_NE@LA", "lock"),
    ("2018_W21_NE@LA", "kickoff"),
    ("2019_W01_GB@CHI", "lock"),
    ("2019_W21_SF@KC", "lock"),
    ("2019_W21_SF@KC", "kickoff"),
    ("2020_W01_HOU@KC", "lock"),
    ("2020_W05_BUF@TEN", "kickoff"),
    ("2020_W12_BAL@PIT", "lock"),
    ("2020_W12_BAL@PIT", "kickoff"),
    ("2020_W13_DAL@BAL", "kickoff"),
    ("2020_W21_KC@TB", "lock"),
    ("2020_W21_KC@TB", "kickoff"),
    ("2021_W01_DAL@TB", "lock"),
    ("2021_W15_SEA@LA", "kickoff"),
    ("2021_W15_WAS@PHI", "kickoff"),
    ("2022_W01_BUF@LA", "lock"),
    ("2023_W01_DET@KC", "lock"),
    ("2024_W01_BAL@KC", "lock"),
    ("2025_W01_DAL@PHI", "lock"),
    ("2026_W01_NE@SEA", "lock"),
    ("2026_W01_NE@SEA", "kickoff"),
    ("2026_W01_SF@LA", "lock"),
)
MEASURED_GAME_COUNT = 18


def _slate() -> ModuleType:
    """The schedule-keyed resolver module, imported at CALL time.

    Imported here rather than at module scope so the module still COLLECTS before the resolver
    exists, and the public-API tests fail on their assertions rather than on an import error.
    """
    return importlib.import_module("utils.current_slate")


@pytest.fixture(scope="module")
def production_schedule() -> pd.DataFrame:
    """Production silver ``games``, read-only, with each game's lock and kickoff instants."""
    assert PRODUCTION_GAMES.exists(), (
        f"{PRODUCTION_GAMES} is missing: the agreement sweep measures the real schedule and "
        "refuses to pass vacuously without it"
    )
    games = pd.read_parquet(
        PRODUCTION_GAMES, columns=["game_id", "season", "week", "kickoff_et"]
    )
    games["lock"] = [game_lock(v) for v in games["kickoff_et"]]
    games["kickoff"] = [kickoff_wall_clock_et(v) for v in games["kickoff_et"]]
    return games.set_index("game_id", drop=False)


def _et(text: str) -> datetime:
    """An ET-aware instant from ``YYYY-MM-DD HH:MM``."""
    return datetime.fromisoformat(text).replace(tzinfo=ET)


def _schedule(*rows: tuple[str, int, int, str, str]) -> pd.DataFrame:
    """A synthetic silver ``games`` frame: (game_id, season, week, 'YYYY-MM-DD HH:MM' ET, type)."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "kickoff_et": pd.Timestamp(kickoff, tz=ET),
                "game_type": game_type,
            }
            for game_id, season, week, kickoff, game_type in rows
        ]
    )


# A complete 2039 season's tail (its Super Bowl on 2040-02-10) and a 2040 season whose opener is
# THURSDAY 2040-09-06 -- which is also the computed Thursday after Labor Day (Mon 2040-09-03), so
# the calendar and the schedule agree on the opener and every difference below is the rule's.
SEASON_2039_TAIL = (
    ("2039_W18_A@B", 2039, 18, "2040-01-06 13:00", "REG"),
    ("2039_W19_C@D", 2039, 19, "2040-01-12 16:30", "WC"),
    ("2039_W20_E@F", 2039, 20, "2040-01-19 16:30", "DIV"),
    ("2039_W21_G@H", 2039, 21, "2040-01-27 15:00", "CON"),
    ("2039_W22_I@J", 2039, 22, "2040-02-10 18:30", "SB"),
)
SEASON_2040_HEAD = (
    ("2040_W01_A@B", 2040, 1, "2040-09-06 20:20", "REG"),
    ("2040_W01_C@D", 2040, 1, "2040-09-09 13:00", "REG"),
    ("2040_W01_E@F", 2040, 1, "2040-09-10 20:15", "REG"),
    ("2040_W02_G@H", 2040, 2, "2040-09-13 20:15", "REG"),
    ("2040_W02_I@J", 2040, 2, "2040-09-16 13:00", "REG"),
)


# ---------------------------------------------------------------------------
# The measured defect, through the public API
# ---------------------------------------------------------------------------


class TestTheMeasuredDisagreementsResolveToTheSchedulesWeek:
    """Every one of the 23 measured pairs now resolves to the game's own week."""

    @pytest.mark.parametrize(
        ("game_id", "instant_kind"),
        MEASURED_DISAGREEMENTS,
        ids=[f"{g}-{k}" for g, k in MEASURED_DISAGREEMENTS],
    )
    def test_the_pair_resolves_to_its_own_week(
        self, production_schedule: pd.DataFrame, game_id: str, instant_kind: str
    ) -> None:
        game = production_schedule.loc[game_id]
        expected = (int(game["season"]), int(game["week"]))

        assert get_current_nfl_week(game[instant_kind]) == expected, (
            f"{game_id} at its {instant_kind} ({game[instant_kind]}) must resolve to its own "
            f"week {expected}; the retired calendar count put it in another slate, so the run "
            "that should have captured or predicted it would have omitted it"
        )

    def test_the_list_is_the_measured_eighteen_games(
        self, production_schedule: pd.DataFrame
    ) -> None:
        """Non-vacuity: 18 distinct games, every one of them in the production schedule."""
        ids = {game_id for game_id, _kind in MEASURED_DISAGREEMENTS}
        assert len(ids) == MEASURED_GAME_COUNT
        assert ids <= set(production_schedule.index)

    def test_the_retired_calendar_count_gets_every_pair_wrong(
        self, production_schedule: pd.DataFrame
    ) -> None:
        """Control: the list is the defect, not a set of cases that always passed."""
        retired = _slate().retired_calendar_week
        agreeing = [
            (game_id, kind)
            for game_id, kind in MEASURED_DISAGREEMENTS
            if retired(production_schedule.loc[game_id, kind])
            == (
                int(production_schedule.loc[game_id, "season"]),
                int(production_schedule.loc[game_id, "week"]),
            )
        ]
        assert agreeing == []


class TestTheAgreementSweep:
    """At every 2018-2026 game's lock and kickoff, the resolver names that game's own slate."""

    def test_every_game_resolves_to_its_own_week_at_its_lock_and_its_kickoff(
        self, production_schedule: pd.DataFrame
    ) -> None:
        swept = production_schedule[production_schedule["season"].isin(SWEEP_SEASONS)]
        disagreements = [
            (row.game_id, kind, got)
            for row in swept.itertuples(index=False)
            for kind in ("lock", "kickoff")
            if (got := get_current_nfl_week(getattr(row, kind)))
            != (int(row.season), int(row.week))
        ]
        # Non-vacuity: every season in the window is present and the sweep is not a handful.
        assert set(swept["season"]) == set(SWEEP_SEASONS)
        assert (
            len(swept) > 2_400
        )  # 2,499 on 2026-09-23 (2026 postseason not yet scheduled)
        assert disagreements == []


# ---------------------------------------------------------------------------
# The public contract every caller relies on
# ---------------------------------------------------------------------------


class TestThePublicContractIsPreserved:
    def test_the_signature_is_unchanged(self) -> None:
        signature = inspect.signature(get_current_nfl_week)
        assert list(signature.parameters) == ["now"]
        assert signature.parameters["now"].default is None

    def test_it_delegates_to_the_one_schedule_keyed_resolver(self) -> None:
        resolve = _slate().resolve_current_slate
        for instant in (_et("2025-10-17 18:00"), _et("2026-09-08 12:00")):
            assert get_current_nfl_week(instant) == resolve(instant).as_tuple()

    def test_the_default_instant_still_works_with_no_argument(self) -> None:
        season, week = get_current_nfl_week()
        assert 1 <= week <= NFL_TOTAL_WEEKS
        assert 2000 < season < 2100

    def test_importing_date_utils_reads_nothing_and_loads_no_resolver(self) -> None:
        """The resolver is imported lazily, so ``utils.date_utils`` stays cheap to import."""
        probe = (
            "import sys, utils.date_utils; print('utils.current_slate' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# Every branch, on a synthetic schedule
# ---------------------------------------------------------------------------


class TestInSeasonTransitions:
    def test_monday_night_stays_in_its_week_and_tuesday_moves_on(self) -> None:
        resolve = _slate().resolve_current_slate
        frame = _schedule(*SEASON_2039_TAIL, *SEASON_2040_HEAD)

        monday = resolve(_et("2040-09-10 23:30"), schedule=frame)
        tuesday = resolve(_et("2040-09-11 00:01"), schedule=frame)

        assert (monday.season, monday.week, monday.in_season) == (2040, 1, True)
        assert (tuesday.season, tuesday.week, tuesday.in_season) == (2040, 2, True)
        assert tuesday.basis == _slate().BASIS_SCHEDULED_SLATE

    def test_a_tuesday_reschedule_resolves_to_its_own_week_on_its_own_day(self) -> None:
        resolve = _slate().resolve_current_slate
        moved = ("2040_W01_K@L", 2040, 1, "2040-09-11 19:00", "REG")
        frame = _schedule(*SEASON_2039_TAIL, *SEASON_2040_HEAD, moved)

        assert resolve(_et("2040-09-11 19:00"), schedule=frame).as_tuple() == (2040, 1)
        assert resolve(_et("2040-09-12 09:00"), schedule=frame).as_tuple() == (2040, 2)


class TestTheOpenersLockDay:
    def test_the_openers_lock_day_is_week_one(self) -> None:
        resolve = _slate().resolve_current_slate
        frame = _schedule(*SEASON_2039_TAIL, *SEASON_2040_HEAD)

        at_lock = resolve(_et("2040-09-05 18:00"), schedule=frame)

        assert (at_lock.season, at_lock.week, at_lock.in_season) == (2040, 1, True)

    def test_the_day_before_the_openers_lock_day_is_still_the_offseason(self) -> None:
        slate = _slate()
        frame = _schedule(*SEASON_2039_TAIL, *SEASON_2040_HEAD)

        eve = slate.resolve_current_slate(_et("2040-09-04 23:59"), schedule=frame)

        # The documented pre-season contract: the previous season's last regular week.
        assert (eve.season, eve.week) == (2039, NFL_REGULAR_SEASON_WEEKS)
        assert eve.in_season is False
        assert eve.basis == slate.BASIS_OFFSEASON

    def test_a_wednesday_opener_locks_on_tuesday_through_the_one_lock_rule(
        self,
    ) -> None:
        resolve = _slate().resolve_current_slate
        wednesday_opener = ("2040_W01_W@X", 2040, 1, "2040-09-05 20:20", "REG")
        frame = _schedule(*SEASON_2039_TAIL, wednesday_opener, *SEASON_2040_HEAD[1:])

        lock_day = game_lock(pd.Timestamp("2040-09-05 20:20", tz=ET)).date()
        on_lock_day = resolve(_et("2040-09-04 12:00"), schedule=frame)

        assert lock_day.isoformat() == "2040-09-04"
        assert on_lock_day.as_tuple() == (2040, 1)
        assert on_lock_day.in_season is True

    def test_a_late_opener_never_lets_the_calendar_open_the_season_early(self) -> None:
        """The contradiction guard: the calendar thinks the season began, the schedule says not."""
        slate = _slate()
        late_opener = ("2040_W01_L@M", 2040, 1, "2040-09-13 20:20", "REG")
        frame = _schedule(*SEASON_2039_TAIL, late_opener)
        instant = _et("2040-09-08 12:00")

        assert slate.retired_calendar_week(instant) == (2040, 1)
        resolved = slate.resolve_current_slate(instant, schedule=frame)
        assert resolved.as_tuple() == (2039, NFL_REGULAR_SEASON_WEEKS)
        assert resolved.in_season is False

    def test_a_season_whose_first_recorded_week_is_not_one_is_refused(self) -> None:
        slate = _slate()
        frame = _schedule(*SEASON_2039_TAIL, *SEASON_2040_HEAD[3:])

        with pytest.raises(slate.ScheduleIncompleteError, match="2040"):
            slate.resolve_current_slate(_et("2040-09-12 12:00"), schedule=frame)


class TestTheOffseason:
    def test_deep_offseason_with_no_next_schedule_is_the_documented_offseason_value(
        self,
    ) -> None:
        slate = _slate()
        frame = _schedule(*SEASON_2039_TAIL)
        instant = _et("2040-03-15 12:00")

        resolved = slate.resolve_current_slate(instant, schedule=frame)

        assert resolved.as_tuple() == slate.retired_calendar_week(instant)
        assert resolved.as_tuple() == (2039, NFL_TOTAL_WEEKS)
        assert resolved.in_season is False
        assert resolved.basis == slate.BASIS_OFFSEASON

    def test_a_season_at_hand_with_no_recorded_schedule_is_refused_by_name(
        self,
    ) -> None:
        """Never the silent calendar: from August the next schedule must be on record."""
        slate = _slate()
        frame = _schedule(*SEASON_2039_TAIL)

        with pytest.raises(slate.ScheduleNotRecordedError, match="2040") as refusal:
            slate.resolve_current_slate(_et("2040-08-15 12:00"), schedule=frame)
        assert "ingest_games --season 2040" in str(refusal.value)


class TestThePlayoffGap:
    def test_the_next_round_before_it_is_recorded(self) -> None:
        slate = _slate()
        through_wild_card = _schedule(*SEASON_2039_TAIL[:2])

        resolved = slate.resolve_current_slate(
            _et("2040-01-15 12:00"), schedule=through_wild_card
        )

        assert resolved.as_tuple() == (2039, 20)
        assert resolved.in_season is True
        assert resolved.basis == slate.BASIS_NEXT_UNRECORDED_ROUND

    def test_the_wild_card_round_after_the_final_regular_week(self) -> None:
        resolve = _slate().resolve_current_slate
        through_week_18 = _schedule(SEASON_2039_TAIL[0])

        assert resolve(
            _et("2040-01-09 12:00"), schedule=through_week_18
        ).as_tuple() == (2039, 19)

    def test_a_store_stale_by_more_than_one_round_is_refused(self) -> None:
        slate = _slate()
        through_wild_card = _schedule(*SEASON_2039_TAIL[:2])

        with pytest.raises(slate.ScheduleIncompleteError, match="2039"):
            slate.resolve_current_slate(
                _et("2040-01-30 12:00"), schedule=through_wild_card
            )

    def test_a_truncated_regular_season_is_refused(self) -> None:
        slate = _slate()
        truncated = _schedule(("2039_W10_A@B", 2039, 10, "2039-11-10 13:00", "REG"))

        with pytest.raises(slate.ScheduleIncompleteError, match="2039"):
            slate.resolve_current_slate(_et("2039-11-15 12:00"), schedule=truncated)


class TestRefusals:
    def test_a_naive_instant_is_refused(self) -> None:
        resolve = _slate().resolve_current_slate
        frame = _schedule(*SEASON_2039_TAIL, *SEASON_2040_HEAD)

        with pytest.raises(ValueError, match="timezone"):
            resolve(datetime(2040, 9, 10, 12, 0), schedule=frame)

    def test_an_empty_schedule_is_refused(self) -> None:
        slate = _slate()
        with pytest.raises(slate.ScheduleUnavailableError):
            slate.resolve_current_slate(_et("2040-09-10 12:00"), schedule=_schedule())

    def test_an_unreadable_schedule_is_refused(self, tmp_path: Path) -> None:
        slate = _slate()
        with pytest.raises(slate.ScheduleUnavailableError, match="missing"):
            slate.load_recorded_schedule(tmp_path / "missing" / "games.parquet")

    def test_a_null_kickoff_is_refused_by_name(self) -> None:
        from utils.game_lock import MissingKickoffError

        frame = _schedule(*SEASON_2040_HEAD)
        frame.loc[1, "kickoff_et"] = pd.NaT
        with pytest.raises(MissingKickoffError, match="2040_W01_C@D"):
            _slate().resolve_current_slate(_et("2040-09-08 12:00"), schedule=frame)

    def test_a_day_that_names_two_slates_is_refused(self) -> None:
        slate = _slate()
        clash = ("2040_W02_Z@Y", 2040, 2, "2040-09-10 13:00", "REG")
        frame = _schedule(*SEASON_2040_HEAD, clash)

        with pytest.raises(slate.AmbiguousSlateError, match="2040-09-10"):
            slate.resolve_current_slate(_et("2040-09-08 12:00"), schedule=frame)


# ---------------------------------------------------------------------------
# The callers that decide "in season or not"
# ---------------------------------------------------------------------------


class TestTheFridayOffseasonNoOp:
    """Phase 21 D-06, keyed on the schedule: the offseason still exits 0, silently."""

    def test_deep_offseason_is_offseason(self) -> None:
        from scripts.friday_pipeline import _is_offseason

        assert _is_offseason(_et("2026-05-29 12:00")) is True

    def test_the_2026_openers_lock_day_is_in_season(self) -> None:
        """The retired window opened on the computed Thursday, Sept 10, after this lock."""
        from scripts.friday_pipeline import _is_offseason

        assert _is_offseason(_et("2026-09-08 12:00")) is False

    def test_super_bowl_week_is_in_season(self) -> None:
        """The retired 22-week window closed on 2026-02-05, before the 2026-02-08 Super Bowl."""
        from scripts.friday_pipeline import _is_offseason

        assert _is_offseason(_et("2026-02-06 12:00")) is False

    def test_an_unforced_offseason_run_is_still_a_clean_no_op(self) -> None:
        from scripts.friday_pipeline import main

        with (
            patch("sys.argv", ["friday_pipeline.py"]),
            patch("scripts.friday_pipeline.datetime") as mock_dt,
            patch("pipeline.orchestrator.FridayPipeline") as mock_pipeline_cls,
        ):
            mock_dt.now.return_value = _et("2026-05-29 12:00")
            mock_dt.side_effect = datetime
            assert main() == 0

        mock_pipeline_cls.assert_not_called()

    def test_an_unresolvable_schedule_fails_the_run_by_name_before_the_orchestrator(
        self,
    ) -> None:
        slate = _slate()
        from scripts.friday_pipeline import main

        refusal = slate.ScheduleNotRecordedError("the 2027 schedule is not recorded")
        with (
            patch("sys.argv", ["friday_pipeline.py"]),
            patch("scripts.friday_pipeline.resolve_current_slate", side_effect=refusal),
            patch("pipeline.orchestrator.FridayPipeline") as mock_pipeline_cls,
        ):
            assert main() == 1

        mock_pipeline_cls.assert_not_called()


class TestTheStalenessSeasonGate:
    def _gate_result(self, instant: datetime) -> bool:
        from pipeline.staleness import StalenessGate

        with patch("pipeline.staleness.datetime") as mock_dt:
            mock_dt.now.return_value = instant
            mock_dt.side_effect = datetime
            return StalenessGate(season=instant.year, week=1).check_season().passed

    def test_the_openers_lock_day_passes_the_season_gate(self) -> None:
        assert self._gate_result(_et("2026-09-08 12:00")) is True

    def test_deep_offseason_fails_the_season_gate(self) -> None:
        assert self._gate_result(_et("2026-05-29 12:00")) is False


class TestTheTrainersWeekDefault:
    """The CLI default was ``get_current_nfl_week()`` -- a TUPLE -- compared to the week column."""

    TRAINERS = ("models/train_wp.py", "models/train_ats.py", "models/train_ou.py")

    def test_the_default_is_one_resolution_with_an_integer_week(self) -> None:
        season, week = _slate().cli_target_season_week(None, None)
        assert (season, week) == get_current_nfl_week()
        assert isinstance(week, int)

    def test_all_weeks_and_explicit_values_pass_through(self) -> None:
        target = _slate().cli_target_season_week
        assert target(2024, "5") == (2024, 5)
        assert target(2024, "all") == (2024, None)
        assert target(None, "all")[1] is None

    def test_explicit_values_never_consult_the_schedule(self) -> None:
        slate = _slate()
        with patch.object(
            slate, "resolve_current_slate", side_effect=AssertionError("resolved")
        ):
            assert slate.cli_target_season_week(2024, "7") == (2024, 7)

    @pytest.mark.parametrize("trainer", TRAINERS)
    def test_each_trainer_resolves_through_the_shared_default(
        self, trainer: str
    ) -> None:
        tree = ast.parse((REPO_ROOT / trainer).read_text(encoding="utf-8"))
        main = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        names = {n.id for n in ast.walk(main) if isinstance(n, ast.Name)} | {
            a.name
            for n in ast.walk(main)
            if isinstance(n, ast.ImportFrom)
            for a in n.names
        }
        assert "cli_target_season_week" in names
        assert "get_current_nfl_season" not in names
        assert "get_current_nfl_week" not in names
