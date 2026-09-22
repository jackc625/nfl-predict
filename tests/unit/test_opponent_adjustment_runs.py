"""The opponent adjustment actually RUNS, and says so when it cannot (Plan 33.2-16, SPEC R9 / R2).

THE DEFECT THIS MODULE PINS SHUT. ``OpponentAdjuster._add_opponent_column`` looked the
play-by-play ``game_id`` (``2023_01_ARI_WAS``) up in silver ``games`` (``2023_W01_ARI@WAS``).
No id ever matched, so every ``opponent`` was ``""``, the lagged merge found nothing,
``game_count`` was NaN, ``NaN >= 4`` was False, and the RAW value was kept under the adjusted
name: 0 of 1,088 rows adjusted for 2023, and gold's twelve ``*_rolling_opp_adj_*`` columns were
plain EPA wearing an adjusted name. And the league averages it subtracted were means over the
WHOLE loaded frame, games after the one being adjusted included.

What is asserted here:

* opponents resolve through ONE canonical mapping (``utils.game_id_utils.
  convert_legacy_game_id``), and an unresolvable id RAISES ``OpponentResolutionError`` -- a type
  neither swallow tuple in ``scripts/build_features.py`` catches;
* the minimum-history boundary is inclusive, and a row below it carries NaN plus an explicit
  coverage flag, never raw EPA under an adjusted name;
* a team's first game is the flagged unknown (``basis="no_information"``);
* the degradation tripwire: a frame in which every adjusted value equals its raw EPA FAILS;
* the AST id scan with its four structural controls (the planted parser written in BOTH quote
  spellings);
* 2023 on real data: at least 832 adjusted rows, and EXACTLY the rows meeting the
  minimum-history rule.

Nothing here writes ``data/`` or ``outputs/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import features.opponent_adj as opponent_module
from features.opponent_adj import (
    OPP_ADJ_COVERAGE_COLUMN,
    OpponentAdjuster,
    OpponentResolutionError,
    opponent_adjusted_gold_columns,
)
from features.provenance import PROVENANCE_COLUMNS, InformationBasis

ET = ZoneInfo("America/New_York")
_AS_OF = datetime(2030, 1, 1, tzinfo=ET)
_METRICS = ("epa_per_play", "pass_epa_per_play", "rush_epa_per_play")
_ROLLING = (
    "rolling_opp_adj_epa_per_play",
    "rolling_opp_adj_pass_epa",
    "rolling_opp_adj_rush_epa",
)

# ---------------------------------------------------------------------------
# A synthetic season on REAL abbreviations: the canonical converter normalizes team names and
# refuses an unknown one, so the fixture must speak the same language production does.
# ---------------------------------------------------------------------------

_TEAMS = ("KC", "BUF", "MIA", "NYJ", "NE", "DEN")

#: (home, away) per week -- every team plays every week, so before week w each team has
#: played exactly w-1 games (the boundary arithmetic below relies on it).
_MATCHUPS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("KC", "BUF"), ("MIA", "NYJ"), ("NE", "DEN")),
    (("BUF", "MIA"), ("NYJ", "NE"), ("DEN", "KC")),
    (("KC", "NYJ"), ("BUF", "NE"), ("MIA", "DEN")),
    (("NYJ", "DEN"), ("NE", "KC"), ("BUF", "MIA")),
    (("KC", "MIA"), ("BUF", "DEN"), ("NYJ", "NE")),
    (("MIA", "NE"), ("NYJ", "KC"), ("DEN", "BUF")),
    (("KC", "DEN"), ("BUF", "NYJ"), ("MIA", "NE")),
    (("NE", "BUF"), ("DEN", "MIA"), ("NYJ", "KC")),
)

_OFF_EPA = {"KC": 0.10, "BUF": -0.05, "MIA": 0.05, "NYJ": 0.03, "NE": 0.08, "DEN": 0.00}
_DEF_EPA = {
    "KC": -0.05,
    "BUF": 0.15,
    "MIA": 0.10,
    "NYJ": 0.12,
    "NE": -0.03,
    "DEN": 0.05,
}

_SEASON_START = datetime(2023, 9, 10, 13, 0, tzinfo=ET)  # a Sunday


def synthetic_schedule(weeks: int = len(_MATCHUPS)) -> pd.DataFrame:
    """Silver ``games`` shape: canonical ids and tz-aware kickoffs, one Sunday per week."""
    rows = []
    for index, week in enumerate(_MATCHUPS[:weeks], start=1):
        kickoff = _SEASON_START + timedelta(weeks=index - 1)
        for home, away in week:
            rows.append(
                {
                    "game_id": f"2023_W{index:02d}_{away}@{home}",
                    "season": 2023,
                    "week": index,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": pd.Timestamp(kickoff),
                }
            )
    frame = pd.DataFrame(rows)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def synthetic_stats(schedule: pd.DataFrame) -> pd.DataFrame:
    """Per-game team stats in the PLAY-BY-PLAY id form, exactly as TeamFormCalculator emits."""
    rows = []
    for game in schedule.to_dict("records"):
        pbp_id = f"{game['season']}_{game['week']:02d}_{game['away_team']}_{game['home_team']}"
        for team in (game["home_team"], game["away_team"]):
            for side, epa in (("offense", _OFF_EPA[team]), ("defense", _DEF_EPA[team])):
                rows.append(
                    {
                        "game_id": pbp_id,
                        "season": game["season"],
                        "week": game["week"],
                        "team": team,
                        "side": side,
                        "epa_per_play": epa,
                        "pass_epa_per_play": epa + 0.02,
                        "rush_epa_per_play": epa - 0.02,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def schedule() -> pd.DataFrame:
    return synthetic_schedule()


@pytest.fixture
def stats(schedule: pd.DataFrame) -> pd.DataFrame:
    return synthetic_stats(schedule)


@pytest.fixture
def adjuster(schedule: pd.DataFrame) -> OpponentAdjuster:
    return OpponentAdjuster(window=10, min_opponent_games=4, schedule_df=schedule)


# ---------------------------------------------------------------------------
# 1. Resolution through the ONE canonical mapping, and the loud miss
# ---------------------------------------------------------------------------


class TestOpponentsResolveCanonically:
    def test_every_play_by_play_id_resolves_to_its_scheduled_game(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame, schedule: pd.DataFrame
    ) -> None:
        resolved = adjuster.resolve_team_games(stats)
        assert len(resolved) == len(stats)
        assert set(resolved["schedule_game_id"]) == set(schedule["game_id"])
        assert (resolved["opponent"] != "").all()
        assert resolved["opponent"].notna().all()
        first = resolved.loc[
            (resolved["game_id"] == "2023_01_BUF_KC") & (resolved["team"] == "KC")
        ].iloc[0]
        assert first["schedule_game_id"] == "2023_W01_BUF@KC"
        assert first["opponent"] == "BUF"

    @pytest.mark.parametrize(
        "bad_id",
        [
            "2023_01_BUF",  # not the four-part play-by-play form
            "2023_01_XXX_KC",  # an unknown team
            "2023_30_BUF_KC",  # a week past 22
            "2023_09_BUF_KC",  # well formed, but no such game on the schedule
        ],
    )
    def test_an_unresolvable_id_raises_by_name_and_carries_the_id(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame, bad_id: str
    ) -> None:
        planted = stats.copy()
        planted.loc[planted.index[0], "game_id"] = bad_id
        with pytest.raises(OpponentResolutionError) as refused:
            adjuster.resolve_team_games(planted)
        assert bad_id in str(refused.value)

    def test_a_team_that_did_not_play_in_the_game_is_refused(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        planted = stats.copy()
        planted.loc[planted.index[0], "team"] = "NE"  # 2023_01_BUF_KC is KC v BUF
        with pytest.raises(OpponentResolutionError):
            adjuster.resolve_team_games(planted)


class TestTheRefusalCannotBeSwallowed:
    """Member by member, on the ``tests/unit/test_provisional_training_refusal.py`` pattern."""

    def test_it_is_a_runtime_error(self) -> None:
        assert issubclass(OpponentResolutionError, RuntimeError)

    def test_the_source_tuple_is_not_empty(self) -> None:
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        assert len(_SOURCE_LOAD_ERRORS) > 0

    def test_no_member_of_the_source_tuple_catches_it(self) -> None:
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        for caught in _SOURCE_LOAD_ERRORS:
            assert not issubclass(OpponentResolutionError, caught), caught.__name__

    @pytest.mark.parametrize("caught", [ValueError, KeyError, TypeError])
    def test_no_member_of_the_merge_handler_catches_it(self, caught) -> None:
        assert not issubclass(OpponentResolutionError, caught)


# ---------------------------------------------------------------------------
# 2. The inclusive boundary, and a flag instead of raw EPA wearing an adjusted name
# ---------------------------------------------------------------------------


class TestTheMinimumHistoryBoundary:
    def test_exactly_the_minimum_history_is_adjusted(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        per_game = adjuster.per_game_adjusted(stats)
        week5 = per_game[per_game["week"] == 5]  # every opponent has played exactly 4
        assert (week5["opponent_prior_games"] == 4).all()
        assert week5["adjusted"].all()
        for metric in _METRICS:
            assert week5[f"opp_adj_{metric}"].notna().all()

    def test_one_below_the_minimum_carries_nan_and_a_false_flag(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        per_game = adjuster.per_game_adjusted(stats)
        week4 = per_game[per_game["week"] == 4]  # every opponent has played exactly 3
        assert (week4["opponent_prior_games"] == 3).all()
        assert not week4["adjusted"].any()
        for metric in _METRICS:
            assert week4[f"opp_adj_{metric}"].isna().all()

    def test_no_adjusted_column_carries_raw_epa_for_an_unadjusted_row(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        per_game = adjuster.per_game_adjusted(stats)
        unadjusted = per_game[~per_game["adjusted"]]
        assert len(unadjusted) > 0
        for metric in _METRICS:
            assert unadjusted[f"opp_adj_{metric}"].isna().all()

    def test_a_rolling_row_with_no_adjusted_game_is_nan_with_the_flag_false(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame, schedule: pd.DataFrame
    ) -> None:
        rolling = adjuster.build_features(
            schedule, _AS_OF, target_season=2023, target_week=3, team_game_stats=stats
        )
        assert len(rolling) > 0
        assert (rolling[OPP_ADJ_COVERAGE_COLUMN] == 0.0).all()
        for column in _ROLLING:
            assert rolling[column].isna().all()

    def test_a_rolling_row_with_adjusted_history_carries_values_and_the_flag_true(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame, schedule: pd.DataFrame
    ) -> None:
        rolling = adjuster.build_features(
            schedule, _AS_OF, target_season=2023, target_week=7, team_game_stats=stats
        )
        assert len(rolling) == 2 * len(_TEAMS)
        assert (rolling[OPP_ADJ_COVERAGE_COLUMN] == 1.0).all()
        for column in _ROLLING:
            assert rolling[column].notna().all()


class TestTheFlagName:
    def test_it_survives_both_merge_filters_by_construction(self) -> None:
        assert OPP_ADJ_COVERAGE_COLUMN == "rolling_opp_adj_coverage"
        assert OPP_ADJ_COVERAGE_COLUMN.startswith("rolling_")
        assert "opp_adj" in OPP_ADJ_COVERAGE_COLUMN

    def test_the_gold_columns_are_twelve_values_and_four_flags(self) -> None:
        values, flags = opponent_adjusted_gold_columns()
        assert len(values) == 12
        assert sorted(flags) == sorted(
            f"{p}_{s}_{OPP_ADJ_COVERAGE_COLUMN}"
            for p in ("home", "away")
            for s in ("off", "def")
        )


# ---------------------------------------------------------------------------
# 3. A team's first game is the flagged unknown, and the provenance says so
# ---------------------------------------------------------------------------


class TestTheFirstGameIsTheFlaggedUnknown:
    def test_the_signature_is_not_empty_and_declares_nan_values_and_false_flags(
        self, adjuster: OpponentAdjuster
    ) -> None:
        signature = dict(adjuster.no_information_signature())
        values, flags = opponent_adjusted_gold_columns()
        assert signature, "an empty signature would make no_information an exemption"
        assert set(signature) == set(values) | set(flags)
        assert all(signature[column] is None for column in values)
        assert all(signature[column] == 0.0 for column in flags)

    def test_week_one_reports_no_information_and_a_later_game_is_dated(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame, schedule: pd.DataFrame
    ) -> None:
        adjuster.build_features(schedule, _AS_OF, team_game_stats=stats)
        provenance = adjuster.information_times(schedule)
        assert list(provenance.columns) == list(PROVENANCE_COLUMNS)
        assert set(provenance["game_id"]) == set(schedule["game_id"])
        by_game = provenance.set_index("game_id")
        week1 = schedule.loc[schedule["week"] == 1, "game_id"]
        for game_id in week1:
            assert (
                by_game.loc[game_id, "basis"] == InformationBasis.NO_INFORMATION.value
            )
            assert pd.isna(by_game.loc[game_id, "information_time"])
        week7 = schedule.loc[schedule["week"] == 7, "game_id"]
        for game_id in week7:
            assert by_game.loc[game_id, "basis"] == InformationBasis.PER_ROW.value
            assert pd.notna(by_game.loc[game_id, "information_time"])


# ---------------------------------------------------------------------------
# 4. The degradation tripwire: raw EPA wearing an adjusted name FAILS
# ---------------------------------------------------------------------------


def degradation_findings(per_game: pd.DataFrame) -> list[str]:
    """Why *per_game* looks like the silent raw-EPA fall-through, or ``[]``.

    The defect this names: every "adjusted" value equal to its raw EPA -- what gold carried
    before Plan 33.2-16, because no opponent id ever resolved and the raw value was kept under
    the adjusted name.
    """
    findings: list[str] = []
    for metric in _METRICS:
        adjusted = per_game[f"opp_adj_{metric}"]
        raw = per_game[metric]
        present = adjusted.notna()
        if not present.any():
            findings.append(f"opp_adj_{metric}: no row was adjusted at all")
        elif np.allclose(adjusted[present], raw[present], rtol=0.0, atol=0.0):
            findings.append(
                f"opp_adj_{metric}: every adjusted value equals its raw EPA -- the silent "
                "fall-through to raw EPA under an adjusted name has returned"
            )
    return findings


class TestTheDegradationTripwire:
    def test_the_real_adjustment_is_not_raw_epa(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        findings = degradation_findings(adjuster.per_game_adjusted(stats))
        assert findings == [], (
            "the opponent adjustment has degraded back to raw EPA under an adjusted name: "
            f"{findings}"
        )

    def test_control_a_frame_whose_adjusted_values_equal_raw_epa_is_flagged(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        per_game = adjuster.per_game_adjusted(stats)
        for metric in _METRICS:
            per_game[f"opp_adj_{metric}"] = per_game[metric]
        findings = degradation_findings(per_game)
        assert len(findings) == len(_METRICS)
        assert all("raw EPA" in finding for finding in findings)

    def test_control_a_frame_with_nothing_adjusted_is_flagged(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        per_game = adjuster.per_game_adjusted(stats)
        for metric in _METRICS:
            per_game[f"opp_adj_{metric}"] = np.nan
        assert len(degradation_findings(per_game)) == len(_METRICS)


# ---------------------------------------------------------------------------
# 5. The id scan: no second hand-written game id format (AST, four controls)
# ---------------------------------------------------------------------------

_ID_SPLITTERS = frozenset({"split", "rsplit", "partition", "rpartition"})


def call_names(tree: ast.AST) -> set[str]:
    """Every called name in *tree* (attribute or bare), the scanner's resolved call set."""
    return {
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }


def second_parser_lines(tree: ast.AST) -> list[int]:
    """Lines of every ``.split('_')``-family call: a hand-written game id parser.

    An ``ast.Call`` test, so it is quote-agnostic BY CONSTRUCTION: ``split("_")`` and
    ``split('_')`` parse to the same node, which a regex over source text does not see.
    """
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _ID_SPLITTERS
        and any(isinstance(arg, ast.Constant) and arg.value == "_" for arg in node.args)
    )


class TestNoSecondGameIdFormat:
    @staticmethod
    def _module_tree() -> ast.Module:
        return ast.parse(inspect.getsource(opponent_module))

    def test_control_non_vacuity_the_scanner_resolves_a_call_set(self) -> None:
        calls = call_names(self._module_tree())
        assert len(calls) > 10
        assert "convert_legacy_game_id" in calls

    def test_the_module_contains_no_second_parser(self) -> None:
        assert second_parser_lines(self._module_tree()) == []

    @pytest.mark.parametrize(
        "planted",
        [
            'def f(gid):\n    return gid.split("_")\n',
            "def f(gid):\n    return gid.split('_')\n",
        ],
        ids=["double-quoted", "single-quoted"],
    )
    def test_control_a_planted_parser_is_flagged_in_either_quote_spelling(
        self, planted: str
    ) -> None:
        assert second_parser_lines(ast.parse(planted)) == [2]

    def test_control_the_canonical_converter_is_not_flagged(self) -> None:
        planted = (
            "from utils.game_id_utils import convert_legacy_game_id\n"
            "def f(gid):\n    return convert_legacy_game_id(gid)\n"
        )
        tree = ast.parse(planted)
        assert second_parser_lines(tree) == []
        assert "convert_legacy_game_id" in call_names(tree)


# ---------------------------------------------------------------------------
# 6. 2023 on real data: >= 832 adjusted rows, and exactly the minimum-history-eligible ones
# ---------------------------------------------------------------------------

_SILVER_GAMES = Path("data/silver/games.parquet")


@pytest.fixture(scope="module")
def real_2023() -> pd.DataFrame:
    """The 2022-2023 per-game pool (what a --season 2023 build reads), adjusted. READ-ONLY."""
    if not _SILVER_GAMES.is_file():
        pytest.skip("silver games is absent (data/ is gitignored)")
    from features.team_form import TeamFormCalculator

    stats = TeamFormCalculator().get_per_game_stats(_AS_OF, target_season=2023)
    return OpponentAdjuster().per_game_adjusted(stats)


def independent_eligibility(per_game: pd.DataFrame, min_games: int) -> pd.Series:
    """Opponent history counted WITHOUT the adjuster's machinery: the opponent's games on the
    opposite side that ENDED at or before this row's lock (every one is a real game, so the
    window cap never binds below ``min_games``)."""
    opposite = {"offense": "defense", "defense": "offense"}
    ends = {
        (team, side): np.sort(group["_end"].to_numpy())
        for (team, side), group in per_game.groupby(["team", "side"])
    }
    counts = [
        int(np.searchsorted(ends[(opponent, opposite[side])], lock, side="right"))
        for opponent, side, lock in zip(
            per_game["opponent"], per_game["side"], per_game["_lock"], strict=True
        )
    ]
    return pd.Series(counts, index=per_game.index) >= min_games


class TestTwentyTwentyThreeActuallyAdjusts:
    def test_at_least_832_rows_are_adjusted(self, real_2023: pd.DataFrame) -> None:
        season = real_2023[real_2023["season"] == 2023]
        assert len(season) == 1088
        assert int(season["adjusted"].sum()) >= 832

    def test_the_adjusted_rows_are_exactly_the_eligible_rows(
        self, real_2023: pd.DataFrame
    ) -> None:
        eligible = independent_eligibility(
            real_2023, OpponentAdjuster().min_opponent_games
        )
        season = real_2023["season"] == 2023
        assert eligible[season].sum() == real_2023.loc[season, "adjusted"].sum()
        mismatched = real_2023.loc[season & (eligible != real_2023["adjusted"])]
        assert mismatched.empty, mismatched[["game_id", "team", "side"]].head()

    def test_the_2023_adjustment_is_not_raw_epa(self, real_2023: pd.DataFrame) -> None:
        season = real_2023[real_2023["season"] == 2023]
        assert degradation_findings(season) == []


# ===========================================================================
# PLAN 33.2-16 TASK 2: per-lock league averages, the merge coupling, and the gate wired at
# the post-Stage-1 merge site.
# ===========================================================================

_ONE_SECOND = pd.Timedelta(seconds=1)


def _lock_of(schedule: pd.DataFrame, game_id: str) -> pd.Timestamp:
    import utils.game_lock as lock_rule

    return pd.Timestamp(lock_rule.lock_frame(schedule)[game_id])


def synthetic_stats_for(games: pd.DataFrame, *, epa: float) -> pd.DataFrame:
    """Per-game stats with ONE EPA value for every team and side (a plant's rows)."""
    rows = []
    for game in games.to_dict("records"):
        pbp_id = f"{game['season']}_{game['week']:02d}_{game['away_team']}_{game['home_team']}"
        for team in (game["home_team"], game["away_team"]):
            for side in ("offense", "defense"):
                rows.append(
                    {
                        "game_id": pbp_id,
                        "season": game["season"],
                        "week": game["week"],
                        "team": team,
                        "side": side,
                        "epa_per_play": epa,
                        "pass_epa_per_play": epa,
                        "rush_epa_per_play": epa,
                    }
                )
    return pd.DataFrame(rows)


def _with_planted_game(
    schedule: pd.DataFrame,
    stats: pd.DataFrame,
    ends_at: pd.Timestamp,
    *,
    week: int,
    season: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """*schedule* and *stats* plus one SEA@SF game that ENDS at *ends_at*.

    Neither team plays anyone in the fixture, so the plant can reach a target row ONLY
    through the league average -- which is exactly what the reveal isolates.
    """
    from features.provenance import DECLARED_GAME_DURATION

    kickoff = (pd.Timestamp(ends_at) - DECLARED_GAME_DURATION).tz_convert("UTC")
    game = {
        "game_id": f"{season}_W{week:02d}_SEA@SF",
        "season": season,
        "week": week,
        "home_team": "SF",
        "away_team": "SEA",
        "kickoff_et": kickoff,
    }
    planted_schedule = pd.concat([schedule, pd.DataFrame([game])], ignore_index=True)
    planted_schedule["kickoff_et"] = pd.to_datetime(
        planted_schedule["kickoff_et"], utc=True
    )
    planted_stats = pd.concat(
        [stats, synthetic_stats_for(pd.DataFrame([game]), epa=0.9)], ignore_index=True
    )
    return planted_schedule, planted_stats


def _target_values(
    schedule: pd.DataFrame, stats: pd.DataFrame, game_id: str
) -> pd.DataFrame:
    """The target game's per-game adjusted values, from a fresh adjuster over *schedule*."""
    per_game = OpponentAdjuster(schedule_df=schedule).per_game_adjusted(stats)
    columns = ["team", "side", *(f"opp_adj_{m}" for m in _METRICS)]
    target = per_game.loc[per_game["schedule_game_id"] == game_id, columns]
    return target.sort_values(["team", "side"]).reset_index(drop=True)


def _two_season_schedule() -> pd.DataFrame:
    """2022 (six weeks, so week-1 opponents carry history) then 2023 week 1."""
    early = synthetic_schedule(weeks=6)
    prior = early.assign(
        season=2022,
        game_id=early["game_id"].str.replace("2023_", "2022_", regex=False),
        kickoff_et=early["kickoff_et"] - pd.Timedelta(weeks=52),
    )
    opener = synthetic_schedule(weeks=1)
    return pd.concat([prior, opener], ignore_index=True)


class TestTheLatestInformationIsReported:
    def test_a_dated_game_reports_the_latest_game_end_it_read(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame, schedule: pd.DataFrame
    ) -> None:
        """The week-7 game read through week 6: its time is the END of the week-6 game.

        Under the retired whole-frame league average this reported the week-8 game's end --
        a mean over the whole loaded frame reads games after the one it adjusts, and an
        honest provenance says so. The information-time gate would refuse it.
        """
        from features.provenance import DECLARED_GAME_DURATION

        adjuster.build_features(schedule, _AS_OF, team_game_stats=stats)
        provenance = adjuster.information_times(schedule).set_index("game_id")
        week6_kickoff = pd.Timestamp(
            schedule.loc[schedule["week"] == 6, "kickoff_et"].iloc[0]
        )
        game_id = schedule.loc[schedule["week"] == 7, "game_id"].iloc[0]
        assert pd.Timestamp(provenance.loc[game_id, "information_time"]) == (
            week6_kickoff + DECLARED_GAME_DURATION
        )


class TestThePerLockLeagueAverage:
    def test_it_is_the_mean_of_the_levels_known_at_each_lock(
        self, adjuster: OpponentAdjuster, stats: pd.DataFrame
    ) -> None:
        from features.opponent_adj import _utc_ns

        timed = adjuster.resolve_team_games(stats)
        states = adjuster.side_states(timed, "defense")
        locks = timed["_lock"].drop_duplicates().sort_values().reset_index(drop=True)
        averages, _ = adjuster.league_average_at(states, _utc_ns(locks))["epa_per_play"]
        for lock, average in zip(_utc_ns(locks), averages, strict=True):
            known = states.loc[states["_end_ns"] <= lock, "epa_per_play"]
            if len(known) == 0:
                assert np.isnan(average)
            else:
                assert average == pytest.approx(float(known.mean()), abs=1e-15)
        assert len(set(np.round(averages[~np.isnan(averages)], 12))) > 1, (
            "a single value at every lock is the retired whole-frame mean"
        )

    def test_mid_season_a_post_lock_game_moves_no_league_average(
        self, stats: pd.DataFrame, schedule: pd.DataFrame
    ) -> None:
        target = schedule.loc[schedule["week"] == 6, "game_id"].iloc[0]
        lock = _lock_of(schedule, target)
        baseline = _target_values(schedule, stats, target)
        late = _target_values(
            *_with_planted_game(schedule, stats, lock + _ONE_SECOND, week=6), target
        )
        at_lock = _target_values(
            *_with_planted_game(schedule, stats, lock, week=6), target
        )
        assert baseline[[f"opp_adj_{m}" for m in _METRICS]].notna().all().all()
        pd.testing.assert_frame_equal(late, baseline, check_exact=True)
        assert not at_lock.equals(baseline), (
            "the positive control: the same game AT the lock must move the average"
        )

    def test_in_week_one_a_post_lock_game_moves_no_league_average(self) -> None:
        schedule = _two_season_schedule()
        stats = synthetic_stats(schedule)
        target = "2023_W01_BUF@KC"
        lock = _lock_of(schedule, target)
        baseline = _target_values(schedule, stats, target)
        late = _target_values(
            *_with_planted_game(schedule, stats, lock + _ONE_SECOND, week=1), target
        )
        at_lock = _target_values(
            *_with_planted_game(schedule, stats, lock, week=1), target
        )
        assert baseline[[f"opp_adj_{m}" for m in _METRICS]].notna().all().all()
        pd.testing.assert_frame_equal(late, baseline, check_exact=True)
        assert not at_lock.equals(baseline)

    def test_the_week_one_average_comes_from_the_prior_season(self) -> None:
        from features.opponent_adj import _utc_ns

        schedule = _two_season_schedule()
        adjuster = OpponentAdjuster(schedule_df=schedule)
        timed = adjuster.resolve_team_games(synthetic_stats(schedule))
        states = adjuster.side_states(timed, "defense")
        lock = _lock_of(schedule, "2023_W01_BUF@KC")
        averages, latest = adjuster.league_average_at(
            states, _utc_ns(pd.Series([lock]))
        )["epa_per_play"]
        prior = timed.loc[(timed["season"] == 2022) & (timed["side"] == "defense")]
        prior_states = states.loc[states["_end_ns"].isin(_utc_ns(prior["_end"]))]
        assert len(prior_states) == len(
            states.loc[states["_end_ns"] <= _utc_ns(pd.Series([lock]))[0]]
        )
        assert not np.isnan(averages[0])
        assert averages[0] != 0.0
        assert averages[0] == pytest.approx(float(prior_states["epa_per_play"].mean()))
        assert int(latest[0]) == int(_utc_ns(prior["_end"]).max())
        week1 = adjuster.per_game_adjusted(synthetic_stats(schedule))
        opener = week1.loc[week1["schedule_game_id"] == "2023_W01_BUF@KC"]
        assert opener["adjusted"].all(), "the prior season supplies week 1"


class TestTheCoverageFlagSurvivesTheRealMerge:
    """The flag reaches gold only through BOTH real filters in scripts/build_features.py."""

    @staticmethod
    def _games() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": ["2023_W07_BUF@KC"],
                "season": [2023],
                "week": [7],
                "home_team": ["KC"],
                "away_team": ["BUF"],
            }
        )

    @staticmethod
    def _adjusted(extra: dict[str, float]) -> pd.DataFrame:
        rows = []
        for team in ("KC", "BUF"):
            for side in ("offense", "defense"):
                rows.append(
                    {
                        "team": team,
                        "side": side,
                        "target_season": 2023,
                        "target_week": 7,
                        "games_used": 6,
                        **dict.fromkeys(_ROLLING, 0.05),
                        **extra,
                    }
                )
        return pd.DataFrame(rows)

    def test_all_four_prefixed_flags_come_out(self) -> None:
        from scripts.build_features import FeatureMatrixBuilder

        laid_out = FeatureMatrixBuilder._lay_out_opponent_adjusted(
            self._adjusted({OPP_ADJ_COVERAGE_COLUMN: 1.0}), self._games()
        )
        _, flags = opponent_adjusted_gold_columns()
        for flag in flags:
            assert flag in laid_out.columns, flag
            assert (laid_out[flag] == 1.0).all()

    def test_control_a_flag_without_the_rolling_prefix_does_not_survive(self) -> None:
        from scripts.build_features import FeatureMatrixBuilder

        laid_out = FeatureMatrixBuilder._lay_out_opponent_adjusted(
            self._adjusted({"opp_adj_coverage": 1.0}), self._games()
        )
        assert not [c for c in laid_out.columns if "opp_adj_coverage" in c], (
            "a flag named opp_adj_coverage passes the opp_adj filter but NOT "
            "_get_team_features' rolling_ copy, so it must not reach the merged frame"
        )

    def test_control_a_rolling_column_without_opp_adj_does_not_survive(self) -> None:
        from scripts.build_features import FeatureMatrixBuilder

        laid_out = FeatureMatrixBuilder._lay_out_opponent_adjusted(
            self._adjusted({"rolling_success_rate": 0.4}), self._games()
        )
        assert not [c for c in laid_out.columns if "success_rate" in c]


class TestTheFamilyIsNoLongerAPostStage1Gap:
    def test_the_post_stage1_set_is_empty(self) -> None:
        from scripts.build_features import POST_STAGE1_SOURCES

        assert POST_STAGE1_SOURCES == ()

    def test_the_source_name_is_not_a_registry_key(self) -> None:
        from features.opponent_adj import OPP_ADJ_SOURCE_NAME
        from scripts.build_features import FEATURE_SOURCE_KEYS

        assert OPP_ADJ_SOURCE_NAME == "opponent_adj"
        assert OPP_ADJ_SOURCE_NAME not in FEATURE_SOURCE_KEYS


# ---------------------------------------------------------------------------
# The production path: the REAL generate_feature_matrices over a sandbox lake
# ---------------------------------------------------------------------------


def _canonical_sandbox_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """The sandbox games under their canonical ids (the sandbox's own ids are opaque)."""
    return games.assign(
        game_id=[
            f"{s}_W{w:02d}_{a}@{h}"
            for s, w, a, h in zip(
                games["season"],
                games["week"],
                games["away_team"],
                games["home_team"],
                strict=True,
            )
        ]
    )


def _sandbox_build(monkeypatch, *, stats_editor=None, provenance_editor=None):
    """Run the REAL generate_feature_matrices over the gold-write-scope sandbox sources.

    Only ``load_all_feature_sources`` and the per-game stats are replaced (exactly the
    ``tests/integration/test_gold_write_scope.py`` shape); the opponent adjuster, its merge
    and the information-time gate at the merge site are the production code.
    """
    from datetime import UTC

    import data.storage as storage_mod
    from scripts.build_features import FeatureMatrixBuilder
    from tests.integration.test_gold_write_scope import (
        _honest_elo_snapshots,
        _sandbox_games,
        _sandbox_silver_odds,
        _sandbox_silver_weather,
        _sandbox_sources,
        _sandbox_team_form,
    )

    base = Path(storage_mod._parquet_manager.base_path).resolve()
    repo_data = (Path(__file__).resolve().parents[2] / "data").resolve()
    assert base != repo_data and repo_data not in base.parents, (
        "REFUSING to build against the production lake; request `gold_lake`"
    )
    seeded = _sandbox_games()
    storage_mod._parquet_manager.save(seeded, "silver/games.parquet")
    storage_mod._parquet_manager.save(
        _honest_elo_snapshots(seeded), "silver/elo_game_snapshots.parquet"
    )
    storage_mod._parquet_manager.save(
        _sandbox_silver_weather(seeded), "silver/weather.parquet"
    )
    storage_mod._parquet_manager.save(
        _sandbox_silver_odds(seeded), "silver/odds_snapshot.parquet"
    )
    storage_mod._parquet_manager.save(
        _sandbox_team_form(seeded), "silver/team_form_features.parquet"
    )

    builder = FeatureMatrixBuilder()
    sources = _sandbox_sources()
    sources["team_form"] = builder._team_form_per_game(
        sources["team_form"], sources["games"]
    )
    schedule = _canonical_sandbox_schedule(sources["games"])
    builder.opponent_adj = OpponentAdjuster(
        window=10, min_opponent_games=4, schedule_df=schedule
    )
    stats = synthetic_stats_for(schedule, epa=0.0)
    stats["epa_per_play"] = np.linspace(-0.2, 0.2, len(stats))
    stats["pass_epa_per_play"] = stats["epa_per_play"] + 0.01
    stats["rush_epa_per_play"] = stats["epa_per_play"] - 0.01
    if stats_editor is not None:
        stats = stats_editor(stats)
    monkeypatch.setattr(builder, "load_all_feature_sources", lambda *a, **k: sources)
    monkeypatch.setattr(
        builder.team_form_calc, "get_per_game_stats", lambda *a, **k: stats
    )
    if provenance_editor is not None:
        honest = builder.opponent_adj.information_times
        monkeypatch.setattr(
            builder.opponent_adj,
            "information_times",
            lambda games_df, **kw: provenance_editor(honest(games_df, **kw), games_df),
        )
    matrices = builder.generate_feature_matrices(
        as_of_datetime=datetime(2030, 1, 1, tzinfo=UTC)
    )
    return builder, matrices


@pytest.fixture
def sandbox_lake(tmp_path, monkeypatch):
    """The gold-write-scope sandbox lake (parquet only, never the production data root)."""
    import data.storage as storage_mod
    import features.elo_features as elo_features_mod
    import scripts.build_features as bf_mod
    from data.storage import ParquetManager

    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(tmp_path)))
    real_save = storage_mod.save_dataframe
    real_load = storage_mod.load_dataframe

    def _parquet_only(*args, **kwargs):
        kwargs["save_to_db"] = False
        return real_save(*args, **kwargs)

    def _parquet_load(table_name, layer="silver", source="auto", **kwargs):
        return real_load(table_name, layer=layer, source="parquet", **kwargs)

    monkeypatch.setattr(bf_mod, "save_dataframe", _parquet_only)
    monkeypatch.setattr(storage_mod, "save_dataframe", _parquet_only)
    monkeypatch.setattr(bf_mod, "load_dataframe", _parquet_load)
    monkeypatch.setattr(elo_features_mod, "load_dataframe", _parquet_load)
    return tmp_path


class TestTheGateIsWiredAtTheMergeSite:
    def test_a_clean_build_checks_the_family_and_lands_the_four_flags(
        self, sandbox_lake, monkeypatch
    ) -> None:
        builder, matrices = _sandbox_build(monkeypatch)
        report = builder.information_time_coverage
        assert report is not None
        assert "opponent_adj" in report.checked_sources
        assert report.post_stage1_sources == ()
        values, flags = opponent_adjusted_gold_columns()
        for target in ("wp", "ats", "ou"):
            for column in (*values, *flags):
                assert column in matrices[target].columns, (target, column)
            assert (matrices[target][list(flags)].isin([0.0, 1.0])).all().all()
            assert (matrices[target][list(flags)] == 1.0).any().any()
            assert (matrices[target][list(flags)] == 0.0).any().any()

    def test_a_post_lock_information_time_stops_the_build_naming_the_game(
        self, sandbox_lake, monkeypatch
    ) -> None:
        planted: dict[str, str] = {}

        def _late(provenance: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
            import utils.game_lock as lock_rule

            dated = provenance.loc[provenance["basis"] == "per_row", "game_id"]
            target = str(dated.iloc[0])
            planted["game_id"] = target
            late = pd.Timestamp(lock_rule.lock_frame(games_df)[target]) + _ONE_SECOND
            provenance = provenance.copy()
            provenance["information_time"] = provenance["information_time"].astype(
                object
            )
            provenance.loc[provenance["game_id"] == target, "information_time"] = late
            return provenance

        from features.provenance import InformationTimeViolation

        with pytest.raises(InformationTimeViolation) as refused:
            _sandbox_build(monkeypatch, provenance_editor=_late)
        assert refused.value.details["source"] == "opponent_adj"
        assert refused.value.details["game_ids"] == [planted["game_id"]]
        assert not list((sandbox_lake / "gold").glob("*.parquet"))

    def test_a_malformed_play_by_play_id_aborts_the_build(
        self, sandbox_lake, monkeypatch
    ) -> None:
        def _malform(stats: pd.DataFrame) -> pd.DataFrame:
            stats = stats.copy()
            stats.loc[stats.index[0], "game_id"] = "2023_01_XXX_KC"
            return stats

        with pytest.raises(OpponentResolutionError) as refused:
            _sandbox_build(monkeypatch, stats_editor=_malform)
        assert "2023_01_XXX_KC" in str(refused.value)
        assert not list((sandbox_lake / "gold").glob("*.parquet"))
