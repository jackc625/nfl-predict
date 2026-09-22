"""THE shared reveal-test harness for every builder fenced at its game's own lock.

Plan 33.2-13 Task 3 (SPEC R5 / R2). A lock fence that admits everything looks exactly like
a lock fence that works until something is PLANTED against it, and a provenance derived
with the same rule the builder used proves the rule, not the content (RESEARCH P3). So each
builder registered here is driven through the same three assertions:

1. a row planted one second AFTER the target game's lock leaves the produced feature
   values byte-identical to the build without it;
2. the same row planted exactly AT the lock changes at least one value (the positive
   control -- it is what makes assertion 1 mean "fenced" rather than "inert fixture");
3. when the planted row's information time is reported honestly, the information-time
   gate REFUSES the frame and names the game.

ONE HARNESS, NOT SEVEN. ``REVEAL_CASES`` was seeded by Plan 33.2-13 with ``injury`` and
``qb``. Plan 33.2-14 adds ``contextual``, ``snaps``, ``team_form`` and ``weather`` to the
same tuple and raises ``DECLARED_REVEAL_CASE_COUNT``; a builder silently leaving the list
fails the declared-length control. The ``weather`` case is a REGRESSION CHECK of Plan
33.2-12's fence (rung 4 owns it), not a second owner of it. The contextual, snap and
team-form plants are a SYNTHETIC boundary: a prior game rescheduled to end exactly at, or
one second after, the target's lock -- no ordinary schedule puts a team's previous game on
that boundary.

Structural controls (the ``tests/unit/test_freeze_parse_single_source.py`` shape): the
parameter list has an asserted length; a synthetic builder that IGNORES the cutoff is
flagged; a synthetic builder that honours it is not.

The same module proves the gate's refusals survive the PRODUCTION ORCHESTRATION.
``scripts/build_features.py`` turns every member of ``_SOURCE_LOAD_ERRORS`` into an EMPTY
source frame, so the refusals are excluded BY TYPE (asserted member by member, on the
``tests/unit/test_provisional_training_refusal.py`` pattern) and then driven through
``load_all_feature_sources`` and ``generate_feature_matrices`` to show they reach the
caller. Nothing here writes ``data/`` or ``artifacts/``: every write lands under
``tmp_path``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.contextual import ContextualFeaturesCalculator
from features.injury import InjuryBuilder
from features.provenance import (
    DECLARED_GAME_DURATION,
    InformationBasis,
    InformationTimeGate,
    InformationTimeViolation,
    ProvenanceCoverageError,
    SourceCheckState,
    UndatedSourceError,
)
from features.qb_tracking import QBTracker
from features.snaps import SnapCountBuilder
from features.team_form import TeamFormCalculator, team_game_schedule
from features.weather import WeatherFeaturesCalculator

ET = ZoneInfo("America/New_York")
_ONE_SECOND = timedelta(seconds=1)
_AS_OF = datetime(2030, 1, 1, tzinfo=ET)


# ---------------------------------------------------------------------------
# The scenario contract every reveal case supplies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RevealScenario:
    """One builder, one target game, and a way to plant a row at a chosen instant.

    Attributes:
        source_name: The ``feature_sources`` registry key the gate names.
        games: The games frame the build runs over.
        game_id: The target game.
        feature_columns: The produced columns the reveal compares.
        build: ``planted_at -> (source frame, provenance, signature)``; ``None`` plants
            nothing.
    """

    source_name: str
    games: pd.DataFrame
    game_id: str
    feature_columns: tuple[str, ...]
    build: Callable[
        [pd.Timestamp | None],
        tuple[pd.DataFrame, pd.DataFrame, dict[str, float | None]],
    ]

    @property
    def lock(self) -> pd.Timestamp:
        return pd.Timestamp(lock_rule.lock_frame(self.games)[self.game_id])

    def values(self, planted_at: pd.Timestamp | None) -> pd.DataFrame:
        frame, _, _ = self.build(planted_at)
        return (
            frame.loc[
                frame["game_id"] == self.game_id, ["game_id", *self.feature_columns]
            ]
            .sort_values(["game_id", *self.feature_columns])
            .reset_index(drop=True)
        )


class RevealCase(NamedTuple):
    """``(name, scenario factory, feature column)``: the harness's parameter shape."""

    name: str
    scenario: Callable[[], RevealScenario]
    feature_column: str


def reveal_violations(scenario: RevealScenario) -> list[str]:
    """What is wrong with *scenario*'s fence, or ``[]`` when it is load-bearing.

    The harness's one judgement, shared by every registered builder and by the two
    structural controls, so the controls exercise exactly the code the cases rely on.
    """
    problems: list[str] = []
    baseline = scenario.values(None)
    late = scenario.values(scenario.lock + _ONE_SECOND)
    at_lock = scenario.values(scenario.lock)
    if not late.equals(baseline):
        problems.append("a row planted at lock+1s changed the produced values")
    if at_lock.equals(baseline):
        problems.append(
            "the same row planted AT the lock changed nothing: the fence is not "
            "load-bearing (or the planted row reaches no feature)"
        )
    return problems


def honest_late_provenance(scenario: RevealScenario) -> pd.DataFrame:
    """The provenance a builder that USED the lock+1s row would have to report."""
    _, provenance, _ = scenario.build(scenario.lock)
    provenance = provenance.copy()
    target = provenance["game_id"] == scenario.game_id
    provenance["information_time"] = provenance["information_time"].astype(object)
    provenance.loc[target, "basis"] = InformationBasis.PER_ROW.value
    provenance.loc[target, "information_time"] = scenario.lock + _ONE_SECOND
    return provenance


# ---------------------------------------------------------------------------
# Case 1: injury -- a QB1 "Out" report planted against a Thursday game
# ---------------------------------------------------------------------------

_KC_QB1 = "00-0033873"
_KC_QB2 = "00-0000002"


def _empty_snap_builder(schedule_df: pd.DataFrame | None = None) -> SnapCountBuilder:
    """A SnapCountBuilder with one week-1 row, so week-1 targets have no prior share.

    *schedule_df* keeps it hermetic: the snap window is timed against a schedule and
    locked at each target's lock (Plan 33.2-14), and without one the builder would read
    silver ``games``. The orchestration sandbox leaves it ``None`` on purpose -- its
    silver IS the sandbox.
    """
    return SnapCountBuilder(
        schedule_df=schedule_df,
        snaps_df=pd.DataFrame(
            {
                "game_id": ["2023_01_BUF_KC"],
                "pfr_player_id": ["kc_qb"],
                "player": ["kc_qb"],
                "position": ["QB"],
                "team": ["KC"],
                "opponent": ["BUF"],
                "season": [2023],
                "week": [1],
                "offense_snaps": [70.0],
                "offense_pct": [1.0],
                "defense_snaps": [0.0],
                "defense_pct": [0.0],
                "st_snaps": [0.0],
                "st_pct": [0.0],
            }
        ),
    )


def _injury_scenario() -> RevealScenario:
    games = pd.DataFrame(
        [
            {
                "game_id": "2023_W01_BUF@KC",
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "kickoff_et": pd.Timestamp(datetime(2023, 9, 7, 20, 20, tzinfo=ET)),
            }
        ]
    )
    games["kickoff_et"] = pd.to_datetime(games["kickoff_et"], utc=True)
    lock = pd.Timestamp(lock_rule.lock_frame(games).iloc[0])
    depth = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "club_code": ["KC", "KC"],
            "position": ["QB", "QB"],
            "depth_team": ["1", "2"],
            "gsis_id": [_KC_QB1, _KC_QB2],
            "full_name": ["QB1", "QB2"],
        }
    )

    def build(planted_at: pd.Timestamp | None):
        rows = [
            {
                "season": 2023,
                "week": 1,
                "team": "KC",
                "gsis_id": _KC_QB1,
                "position": "QB",
                "report_status": "Questionable",
                "date_modified": lock - timedelta(days=2),
            }
        ]
        if planted_at is not None:
            rows.append(
                {**rows[0], "report_status": "Out", "date_modified": planted_at}
            )
        injuries = pd.DataFrame(rows)
        injuries["date_modified"] = pd.to_datetime(injuries["date_modified"], utc=True)
        builder = InjuryBuilder(
            snap_builder=_empty_snap_builder(schedule_df=games),
            injuries_df=injuries,
            depth_charts_df=depth,
            pbp_df=pd.DataFrame(),
        )
        frame = builder.build_features(games, _AS_OF)
        return (
            frame,
            builder.information_times(games),
            dict(builder.no_information_signature()),
        )

    return RevealScenario(
        source_name="injury",
        games=games,
        game_id="2023_W01_BUF@KC",
        feature_columns=("home_qb_out_flag",),
        build=build,
    )


# ---------------------------------------------------------------------------
# Case 2: qb -- a 2025 depth-chart snapshot naming a different QB1
# ---------------------------------------------------------------------------

_A, _B, _OPP = "00-000000A", "00-000000B", "00-000000C"


def _qb_plays(game, week, home, away, team, passer, epa, n=20):
    return [
        {
            "game_id": game,
            "season": 2025,
            "week": week,
            "home_team": home,
            "away_team": away,
            "posteam": team,
            "passer_player_id": passer,
            "qb_epa": epa + i * 0.001,
            "cpoe": epa * 10 + i * 0.01,
        }
        for i in range(n)
    ]


def _qb_scenario() -> RevealScenario:
    kickoffs = [datetime(2025, 9, day, 13, 0, tzinfo=ET) for day in (7, 14, 21)]
    games = pd.DataFrame(
        {
            "game_id": ["2025_W01_BUF@KC", "2025_W02_KC@BUF", "2025_W03_BUF@KC"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "home_team": ["KC", "BUF", "KC"],
            "away_team": ["BUF", "KC", "BUF"],
            "kickoff_et": pd.to_datetime(kickoffs, utc=True),
        }
    )
    lock = pd.Timestamp(lock_rule.lock_frame(games)["2025_W03_BUF@KC"])
    pbp = pd.DataFrame(
        _qb_plays("2025_01_BUF_KC", 1, "KC", "BUF", "KC", _A, 0.4)
        + _qb_plays("2025_01_BUF_KC", 1, "KC", "BUF", "BUF", _OPP, 0.1)
        + _qb_plays("2025_02_KC_BUF", 2, "BUF", "KC", "KC", _A, 0.35)
        + _qb_plays("2025_02_KC_BUF", 2, "BUF", "KC", "KC", _B, -0.3, n=10)
        + _qb_plays("2025_02_KC_BUF", 2, "BUF", "KC", "BUF", _OPP, 0.1)
    )

    def snapshot(published: pd.Timestamp, kc_qb1: str) -> list[dict]:
        other = _B if kc_qb1 == _A else _A
        return [
            {"dt": published, "club_code": "KC", "position": "QB", "depth_team": "1",
             "gsis_id": kc_qb1, "full_name": kc_qb1, "season": 2025},
            {"dt": published, "club_code": "KC", "position": "QB", "depth_team": "2",
             "gsis_id": other, "full_name": other, "season": 2025},
            {"dt": published, "club_code": "BUF", "position": "QB", "depth_team": "1",
             "gsis_id": _OPP, "full_name": _OPP, "season": 2025},
        ]  # fmt: skip

    def build(planted_at: pd.Timestamp | None):
        rows = snapshot(lock - timedelta(days=2), _A)
        if planted_at is not None:
            rows += snapshot(planted_at, _B)
        depth = pd.DataFrame(rows)
        depth["dt"] = pd.to_datetime(depth["dt"], utc=True)
        tracker = QBTracker()
        tracker._depth_chart_cache = {2025: depth}
        tracker._pbp_cache = {2024: pbp.iloc[0:0], 2025: pbp}
        frame = tracker.build_features(games, _AS_OF)
        return (
            frame,
            tracker.information_times(games),
            dict(tracker.no_information_signature()),
        )

    return RevealScenario(
        source_name="qb_tracking",
        games=games,
        game_id="2025_W03_BUF@KC",
        feature_columns=("team", "qb_adjustment"),
        build=build,
    )


# ---------------------------------------------------------------------------
# Plan 33.2-14's cases share one synthetic week: KC hosts BUF in week 3 (a Sunday), after
# a week-1 meeting that is always there. The PLANT is a week-2 KC game rescheduled so that
# it ENDS at the instant under test -- the shape of a postponed game -- and every builder
# must read it exactly when its end is at or before the week-3 lock.
# ---------------------------------------------------------------------------

_W3_ID = "2023_W03_BUF@KC"
_W3_KICKOFF = pd.Timestamp("2023-09-24 17:00", tz="UTC")  # Sunday 13:00 ET


def _week(game_id: str, week: int, home: str, away: str, kickoff) -> dict:
    return {
        "game_id": game_id,
        "season": 2023,
        "week": week,
        "home_team": home,
        "away_team": away,
        "kickoff_et": pd.Timestamp(kickoff),
        "stadium_id": "KAN00" if home == "KC" else "BUF00",
        "neutral_site": False,
    }


_W1 = _week(
    "2023_W01_BUF@KC", 1, "KC", "BUF", pd.Timestamp("2023-09-10 17:00", tz="UTC")
)
_W3 = _week(_W3_ID, 3, "KC", "BUF", _W3_KICKOFF)
_W3_GAMES = pd.DataFrame([_W3])


def _planted_week_two(planted_at: pd.Timestamp) -> dict:
    """A week-2 KC game (away at BUF) rescheduled to END at *planted_at*."""
    kickoff = (planted_at - DECLARED_GAME_DURATION).tz_convert("UTC")
    return _week("2023_W02_KC@BUF", 2, "BUF", "KC", kickoff)


def _schedule(planted_at: pd.Timestamp | None) -> pd.DataFrame:
    rows = (
        [_W1, _W3] if planted_at is None else [_W1, _planted_week_two(planted_at), _W3]
    )
    return pd.DataFrame(rows)


def _only_the_target(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["game_id"] == _W3_ID].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Case 3: contextual -- the rest window (and the letdown) read a prior game at the lock
# ---------------------------------------------------------------------------


def _contextual_scenario() -> RevealScenario:
    def build(planted_at: pd.Timestamp | None):
        calculator = ContextualFeaturesCalculator()
        # The spot flags reload the silver schedule, which holds none of these games.
        calculator._load_full_season_schedule = lambda seasons: pd.DataFrame()
        games = _schedule(planted_at)
        frame = calculator.build_features(games, _AS_OF)
        return (
            _only_the_target(frame),
            _only_the_target(calculator.information_times(games)),
            dict(calculator.no_information_signature()),
        )

    return RevealScenario(
        source_name="contextual",
        games=_W3_GAMES,
        game_id=_W3_ID,
        feature_columns=("home_rest_days", "rest_advantage"),
        build=build,
    )


# ---------------------------------------------------------------------------
# Case 4: snaps -- the snap window admits a team-game once it ENDED at the lock
# ---------------------------------------------------------------------------

_SNAP_POSITIONS = (("QB", 70.0), ("RB", 40.0), ("WR", 60.0), ("T", 70.0))


def _snap_rows(game_id: str, week: int, team: str, opponent: str, scale: float):
    return [
        {
            "game_id": game_id,
            "pfr_player_id": f"{team.lower()}_{position.lower()}",
            "player": f"{team} {position}",
            "position": position,
            "team": team,
            "opponent": opponent,
            "season": 2023,
            "week": week,
            "offense_snaps": snaps * (scale if position == "QB" else 1.0),
            "offense_pct": 1.0,
            "defense_snaps": 0.0,
            "defense_pct": 0.0,
            "st_snaps": 0.0,
            "st_pct": 0.0,
        }
        for position, snaps in _SNAP_POSITIONS
    ]


def _snaps_scenario() -> RevealScenario:
    def build(planted_at: pd.Timestamp | None):
        rows = _snap_rows("2023_01_BUF_KC", 1, "KC", "BUF", 1.0) + _snap_rows(
            "2023_01_BUF_KC", 1, "BUF", "KC", 1.0
        )
        if planted_at is not None:
            # A QB-heavy week-2 snap distribution: it moves KC's concentration if read.
            rows += _snap_rows("2023_02_KC_BUF", 2, "KC", "BUF", 3.0)
        builder = SnapCountBuilder(
            snaps_df=pd.DataFrame(rows), schedule_df=_schedule(planted_at)
        )
        frame = builder.build_features(_W3_GAMES, _AS_OF)
        return (
            frame,
            builder.information_times(_W3_GAMES),
            dict(builder.no_information_signature()),
        )

    return RevealScenario(
        source_name="snaps",
        games=_W3_GAMES,
        game_id=_W3_ID,
        feature_columns=("home_snap_concentration",),
        build=build,
    )


# ---------------------------------------------------------------------------
# Case 5: team_form -- the rolling window admits a team-game once it ENDED at the lock
# ---------------------------------------------------------------------------


def _team_stats_row(week: int, team: str, side: str, epa: float) -> dict:
    return {
        "game_id": f"2023_{week:02d}_{team}",
        "season": 2023,
        "week": week,
        "team": team,
        "side": side,
        "epa_per_play": epa,
        "pass_epa_per_play": epa,
        "rush_epa_per_play": epa,
        "success_rate": 0.45,
        "pass_success_rate": 0.45,
        "rush_success_rate": 0.45,
        "neutral_pass_rate": 0.6 if side == "offense" else float("nan"),
        "red_zone_td_rate": 0.5,
        "third_down_conversion_rate": 0.4,
        "team_cpoe": float("nan"),
        "avg_drive_start_yardline": float("nan"),
        "neutral_pace": float("nan"),
    }


def _team_form_scenario() -> RevealScenario:
    def build(planted_at: pd.Timestamp | None):
        stats = [
            _team_stats_row(1, team, side, 0.10)
            for team in ("KC", "BUF")
            for side in ("offense", "defense")
        ]
        if planted_at is not None:
            stats += [_team_stats_row(2, "KC", "offense", 0.40)]
        schedule_games = _schedule(planted_at)
        calculator = TeamFormCalculator(schedule_df=schedule_games)
        rolling = calculator.calculate_rolling_averages(
            pd.DataFrame(stats),
            2023,
            3,
            schedule=team_game_schedule(schedule_games),
        )
        kc_offense = rolling.loc[
            (rolling["team"] == "KC") & (rolling["side"] == "offense")
        ].iloc[0]
        # The per-game frame gold reads (scripts/build_features lays it out the same way).
        frame = pd.DataFrame(
            {
                "game_id": [_W3_ID],
                "home_off_rolling_epa_per_play": [kc_offense["rolling_epa_per_play"]],
            }
        )
        calculator._form_df = rolling
        return (
            frame,
            calculator.information_times(_W3_GAMES),
            dict(calculator.no_information_signature()),
        )

    return RevealScenario(
        source_name="team_form",
        games=_W3_GAMES,
        game_id=_W3_ID,
        feature_columns=("home_off_rolling_epa_per_play",),
        build=build,
    )


# ---------------------------------------------------------------------------
# Case 6: weather -- a REGRESSION CHECK of rung 4's one fence (Plan 33.2-12 owns it)
# ---------------------------------------------------------------------------

_WEATHER_BUILD_INSTANT = datetime(2030, 1, 1, tzinfo=ET)


def _bulletin(issued: pd.Timestamp, temp: float) -> dict:
    return {
        "game_id": _W3_ID,
        "forecast_time": issued,
        "forecast_issue_time": issued,
        "weather_source": "historical_forecast",
        "is_outdoor": True,
        "weather_coverage": True,
        "temp_f": temp,
        "wind_mph": 8.0,
        "humidity_pct": 60.0,
        "precip_prob": 0.1,
        "precip_mm": None,
        "mos_precip_level": 0,
    }


def _weather_scenario() -> RevealScenario:
    lock = pd.Timestamp(lock_rule.lock_frame(_W3_GAMES)[_W3_ID])

    def build(planted_at: pd.Timestamp | None):
        rows = [_bulletin((lock - timedelta(hours=10)).tz_convert("UTC"), 50.0)]
        if planted_at is not None:
            rows.append(_bulletin(planted_at.tz_convert("UTC"), 70.0))
        weather = pd.DataFrame(rows)
        calculator = WeatherFeaturesCalculator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            frame = calculator.build_weather_features(
                _W3_GAMES, weather_df=weather, build_instant=_WEATHER_BUILD_INSTANT
            )
        return (
            frame,
            calculator.information_times(
                _W3_GAMES, weather_df=weather, build_instant=_WEATHER_BUILD_INSTANT
            ),
            dict(calculator.no_information_signature()),
        )

    return RevealScenario(
        source_name="weather",
        games=_W3_GAMES,
        game_id=_W3_ID,
        feature_columns=("temp_f",),
        build=build,
    )


# ---------------------------------------------------------------------------
# THE PARAMETER LIST. Plan 33.2-13 seeded it with injury and qb; Plan 33.2-14 appends
# contextual, snaps, team_form and weather. The market_anchors case joins with the odds
# selection's owner ruling (see the plan's checkpoint).
# ---------------------------------------------------------------------------

REVEAL_CASES: tuple[RevealCase, ...] = (
    RevealCase("injury", _injury_scenario, "home_qb_out_flag"),
    RevealCase("qb", _qb_scenario, "qb_adjustment"),
    RevealCase("contextual", _contextual_scenario, "home_rest_days"),
    RevealCase("snaps", _snaps_scenario, "home_snap_concentration"),
    RevealCase("team_form", _team_form_scenario, "home_off_rolling_epa_per_play"),
    RevealCase("weather", _weather_scenario, "temp_f"),
)
DECLARED_REVEAL_CASE_COUNT = 6


class TestTheHarnessIsNotVacuous:
    def test_the_parameter_list_has_its_declared_length(self) -> None:
        assert len(REVEAL_CASES) == DECLARED_REVEAL_CASE_COUNT
        assert len({case.name for case in REVEAL_CASES}) == DECLARED_REVEAL_CASE_COUNT

    def test_every_case_names_a_column_its_scenario_compares(self) -> None:
        for case in REVEAL_CASES:
            assert case.feature_column in case.scenario().feature_columns


@pytest.mark.parametrize("case", REVEAL_CASES, ids=lambda case: case.name)
class TestEveryRegisteredBuilderIsFencedAtItsLock:
    def test_a_row_planted_one_second_after_the_lock_changes_nothing(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        pd.testing.assert_frame_equal(
            scenario.values(scenario.lock + _ONE_SECOND),
            scenario.values(None),
            check_exact=True,
        )

    def test_the_same_row_planted_at_the_lock_changes_a_value(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        assert not scenario.values(scenario.lock).equals(scenario.values(None))

    def test_the_harness_judgement_passes(self, case: RevealCase) -> None:
        assert reveal_violations(case.scenario()) == []

    def test_the_gate_accepts_what_the_builder_honestly_reports(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        frame, provenance, signature = scenario.build(scenario.lock + _ONE_SECOND)
        state = InformationTimeGate().check(
            scenario.source_name,
            frame,
            provenance,
            lock_rule.lock_frame(scenario.games),
            no_information_signature=signature,
        )
        assert state is SourceCheckState.CHECKED

    def test_the_gate_refuses_the_planted_row_reported_honestly(
        self, case: RevealCase
    ) -> None:
        scenario = case.scenario()
        frame, _, signature = scenario.build(scenario.lock)
        with pytest.raises(InformationTimeViolation) as refused:
            InformationTimeGate().check(
                scenario.source_name,
                frame,
                honest_late_provenance(scenario),
                lock_rule.lock_frame(scenario.games),
                no_information_signature=signature,
            )
        assert refused.value.details["game_ids"] == [scenario.game_id]
        assert refused.value.details["violation_type"] == "information_time"


# ---------------------------------------------------------------------------
# Structural controls: the harness flags a cutoff-ignoring builder, not a compliant one
# ---------------------------------------------------------------------------


def _counting_scenario(*, honours_the_lock: bool) -> RevealScenario:
    """A synthetic builder whose one feature counts the rows it admitted."""
    games = pd.DataFrame(
        {
            "game_id": ["2024_W05_NE@NYJ"],
            "season": [2024],
            "week": [5],
            "home_team": ["NYJ"],
            "away_team": ["NE"],
            "kickoff_et": pd.to_datetime(
                [datetime(2024, 10, 6, 13, 0, tzinfo=ET)], utc=True
            ),
        }
    )
    lock = pd.Timestamp(lock_rule.lock_frame(games).iloc[0])

    def build(planted_at: pd.Timestamp | None):
        times = [lock - timedelta(days=1)]
        if planted_at is not None:
            times.append(planted_at)
        admitted = (
            [t for t in times if lock_rule.is_admissible(t, lock)]
            if honours_the_lock
            else times
        )
        frame = pd.DataFrame(
            {"game_id": ["2024_W05_NE@NYJ"], "rows_used": [float(len(admitted))]}
        )
        provenance = pd.DataFrame(
            {
                "game_id": ["2024_W05_NE@NYJ"],
                "basis": [InformationBasis.PER_ROW.value],
                "information_time": [max(admitted)],
            }
        )
        return frame, provenance, {"rows_used": 0.0}

    return RevealScenario(
        source_name="synthetic",
        games=games,
        game_id="2024_W05_NE@NYJ",
        feature_columns=("rows_used",),
        build=build,
    )


class TestTheHarnessControls:
    def test_control_a_builder_that_ignores_the_cutoff_is_flagged(self) -> None:
        problems = reveal_violations(_counting_scenario(honours_the_lock=False))
        assert problems == ["a row planted at lock+1s changed the produced values"]

    def test_control_a_builder_that_honours_the_cutoff_is_not_flagged(self) -> None:
        assert reveal_violations(_counting_scenario(honours_the_lock=True)) == []


# ---------------------------------------------------------------------------
# The refusals survive the production orchestration
# ---------------------------------------------------------------------------

_REFUSALS: dict[str, type[Exception]] = {
    "InformationTimeViolation": InformationTimeViolation,
    "UndatedSourceError": UndatedSourceError,
    "ProvenanceCoverageError": ProvenanceCoverageError,
}


class TestTheRefusalsAreExcludedFromTheSwallowTupleByType:
    def test_the_swallow_tuple_is_not_empty(self) -> None:
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        assert len(_SOURCE_LOAD_ERRORS) > 0, (
            "an empty tuple would make every issubclass check below pass vacuously"
        )

    @pytest.mark.parametrize("refusal_name", sorted(_REFUSALS))
    def test_no_refusal_is_a_member_or_a_subclass_of_a_member(
        self, refusal_name: str
    ) -> None:
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        refusal = _REFUSALS[refusal_name]
        assert refusal not in _SOURCE_LOAD_ERRORS
        for caught in _SOURCE_LOAD_ERRORS:
            assert not issubclass(refusal, caught), (
                f"{refusal_name} is a subclass of {caught.__name__}, which is in "
                "_SOURCE_LOAD_ERRORS -- the injury guard would turn the refusal into an "
                "empty injury frame and the build would exit green"
            )


def _sandbox_builder(monkeypatch, tmp_path):
    """A FeatureMatrixBuilder over a sandboxed silver ``games`` table (one 2026 week)."""
    from data.storage import save_dataframe
    from scripts.build_features import FeatureMatrixBuilder
    from tests.fixtures.elo_sandbox import (
        make_season_games,
        redirect_storage_to_sandbox,
    )

    sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
    games = make_season_games(2026, weeks=1)
    save_dataframe(games, "games", layer="silver", replace_mode=True)
    builder = FeatureMatrixBuilder()
    # The QB source is not under test here and reads pinned upstream play-by-play;
    # an empty frame is reported EMPTY_UNCHECKED by the gate and checks nothing.
    monkeypatch.setattr(
        builder.qb_tracker,
        "build_features",
        lambda *_a, **_k: pd.DataFrame(columns=["game_id", "team", "qb_adjustment"]),
    )
    # The injury builder draws its availability weights from the snap builder, and the
    # sandbox has no snap table: without this frame the injury source would fail to load
    # and become EMPTY, and the build would run on past Stage 1.
    builder.snap_builder._snaps_df = _empty_snap_builder()._snaps_df
    # LeakageGate.write_diagnostic_report defaults to the PRODUCTION outputs/diagnostics
    # tree. Any refusal this module provokes past Stage 1 must write under tmp_path.
    real_writer = builder.leakage_gate.write_diagnostic_report
    monkeypatch.setattr(
        builder.leakage_gate,
        "write_diagnostic_report",
        lambda violation, output_dir=None: real_writer(
            violation, output_dir=str(tmp_path / "diagnostics")
        ),
    )
    return sandbox, games, builder


def _injured_week(games: pd.DataFrame) -> pd.DataFrame:
    """One dated, pre-lock report per game so the injury frame carries real rows."""
    locks = lock_rule.lock_frame(games)
    frame = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "team": game["home_team"],
                "gsis_id": f"00-00000{index:02d}",
                "position": "WR",
                "report_status": "Out",
                "date_modified": locks[game["game_id"]] - timedelta(days=1),
            }
            for index, game in enumerate(games.to_dict("records"))
        ]
    )
    frame["date_modified"] = pd.to_datetime(frame["date_modified"], utc=True)
    return frame


def builder_games(builder) -> pd.DataFrame:
    """The sandbox's silver games, read through the builder's own storage."""
    del builder
    from data.storage import load_dataframe

    return load_dataframe("games", layer="silver")


def _raise_planted_refusal(source: str, game_id: str):
    def _refuse(*_a, **_k):
        raise InformationTimeViolation(
            f"planted: a {source} value built from information after its game's lock",
            {
                "source": source,
                "game_ids": [game_id],
                "violation_type": "information_time",
            },
        )

    return _refuse


#: Where each registry key's load path runs inside load_all_feature_sources' swallow
#: handlers: (object attribute path on the builder, attribute to replace). team_form and
#: weather are read from silver rather than built, so their seam is the load step itself.
_SOURCE_SEAMS: dict[str, tuple[str, str]] = {
    "injury": ("injury_builder", "build_features"),
    "qb_tracking": ("qb_tracker", "build_features"),
    "contextual": ("contextual_calc", "build_features"),
    "snaps": ("snap_builder", "build_features"),
    "team_form": ("", "_team_form_per_game"),
    "weather": ("", "load_dataframe"),
}


class TestEveryRefusalSurvivesLoadAllFeatureSources:
    """End-to-end partner of the member-by-member type exclusion above (Plan 33.2-14).

    Every optional source in ``load_all_feature_sources`` is wrapped in ``except
    _SOURCE_LOAD_ERRORS``, which turns a failure into an EMPTY frame. A refusal raised from
    inside any source's load path -- here, a planted post-lock information time -- must
    reach the CALLER instead. The control shows the swallow path is live for a tuple member.
    """

    def test_every_registry_key_has_a_seam(self) -> None:
        from scripts.build_features import FEATURE_SOURCE_KEYS

        # market's seam joins with its reveal case (see the parameter list above).
        assert set(_SOURCE_SEAMS) == set(FEATURE_SOURCE_KEYS) - {"elo", "market"}

    @staticmethod
    def _plant(builder, monkeypatch, source: str, raiser) -> None:
        owner_name, attribute = _SOURCE_SEAMS[source]
        if source == "weather":
            import scripts.build_features as build_module

            real = build_module.load_dataframe

            def _load(table, *args, **kwargs):
                if table == "weather_features":
                    return raiser()
                return real(table, *args, **kwargs)

            monkeypatch.setattr(build_module, "load_dataframe", _load)
            return
        if source == "team_form":
            # The layout step is reached only when the silver table loads, so the sandbox
            # carries a minimal one (a week-1 row per team).
            from data.storage import save_dataframe

            teams = sorted(
                set(builder_games(builder)["home_team"])
                | set(builder_games(builder)["away_team"])
            )
            save_dataframe(
                pd.DataFrame(
                    {
                        "team": teams,
                        "side": ["offense"] * len(teams),
                        "target_season": [2026] * len(teams),
                        "target_week": [1] * len(teams),
                        "rolling_epa_per_play": [0.1] * len(teams),
                    }
                ),
                "team_form_features",
                layer="silver",
                replace_mode=True,
            )
        owner = getattr(builder, owner_name) if owner_name else builder
        monkeypatch.setattr(owner, attribute, raiser)

    @pytest.mark.parametrize("source", sorted(_SOURCE_SEAMS))
    def test_a_planted_refusal_reaches_the_caller(
        self, source: str, tmp_path, monkeypatch
    ) -> None:
        _, games, builder = _sandbox_builder(monkeypatch, tmp_path)
        target = str(games["game_id"].iloc[0])
        self._plant(
            builder, monkeypatch, source, _raise_planted_refusal(source, target)
        )
        with pytest.raises(InformationTimeViolation) as refused:
            builder.load_all_feature_sources(target_season=2026)
        assert refused.value.details["source"] == source
        assert refused.value.details["game_ids"] == [target]

    @pytest.mark.parametrize("source", sorted(_SOURCE_SEAMS))
    def test_control_an_ordinary_error_on_the_same_seam_becomes_an_empty_source(
        self, source: str, tmp_path, monkeypatch
    ) -> None:
        _, _, builder = _sandbox_builder(monkeypatch, tmp_path)

        def _ordinary(*_a, **_k):
            raise ValueError("an ordinary source-load failure")

        self._plant(builder, monkeypatch, source, _ordinary)
        sources = builder.load_all_feature_sources(target_season=2026)
        assert len(sources[source]) == 0


class TestTheRefusalReachesTheCaller:
    def test_control_an_ordinary_injury_load_error_becomes_an_empty_frame(
        self, tmp_path, monkeypatch
    ) -> None:
        """The CONTROL: the swallow path exists and fires for a tuple member."""
        _, _, builder = _sandbox_builder(monkeypatch, tmp_path)

        def _raise(*_a, **_k):
            raise ValueError("an ordinary source-load failure")

        monkeypatch.setattr(builder.injury_builder, "build_features", _raise)
        sources = builder.load_all_feature_sources(target_season=2026)
        assert len(sources["injury"]) == 0

    def test_a_refusal_from_the_injury_builder_escapes_load_all_feature_sources(
        self, tmp_path, monkeypatch
    ) -> None:
        _, _, builder = _sandbox_builder(monkeypatch, tmp_path)

        def _refuse(*_a, **_k):
            raise InformationTimeViolation(
                "planted: an injury value built after its lock",
                {"source": "injury", "game_ids": ["2026_W01_MIA@BUF"]},
            )

        monkeypatch.setattr(builder.injury_builder, "build_features", _refuse)
        with pytest.raises(InformationTimeViolation):
            builder.load_all_feature_sources(target_season=2026)

    def test_a_post_lock_injury_time_stops_the_build_and_writes_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """The production path: the real injury builder's frame, a provenance that reports
        one game one second after its lock, and the real Stage-1 gate inside
        ``generate_feature_matrices``. The refusal must reach the caller -- not an empty
        injury frame and a green exit."""
        sandbox, games, builder = _sandbox_builder(monkeypatch, tmp_path)
        builder.injury_builder._injuries_df = _injured_week(games)
        target = str(games["game_id"].iloc[0])
        late = pd.Timestamp(lock_rule.lock_frame(games)[target]) + _ONE_SECOND
        honest = builder.injury_builder.information_times

        def _late_provenance(games_df, **kwargs):
            provenance = honest(games_df, **kwargs).copy()
            provenance["information_time"] = provenance["information_time"].astype(
                object
            )
            hit = provenance["game_id"] == target
            provenance.loc[hit, "basis"] = InformationBasis.PER_ROW.value
            provenance.loc[hit, "information_time"] = late
            return provenance

        monkeypatch.setattr(
            builder.injury_builder, "information_times", _late_provenance
        )
        with pytest.raises(InformationTimeViolation) as refused:
            builder.generate_feature_matrices(target_season=2026)

        assert refused.value.details["source"] == "injury"
        assert refused.value.details["game_ids"] == [target]
        assert not list((sandbox / "gold").glob("*.parquet")), "nothing may be written"
