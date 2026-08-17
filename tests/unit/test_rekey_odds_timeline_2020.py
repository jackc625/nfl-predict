"""Unit tests for the 2020 odds_timeline re-key (quick task 260816-u0e).

Two things are pinned here, both on synthetic data so the tests never touch the
paid archive:

1. The FROZEN REPLAY of the pre-``45bff24`` season-start rule. If this replay
   ever stops reproducing the buggy dates, the re-key map stops matching the keys
   actually stored on disk, and the recovery silently degrades from an exact
   re-derivation to a guess.
2. The three-step resolution order in :func:`build_rekey_map`, including the
   fixed-point branch that makes a second ``--apply`` run a no-op and the
   hard-fail on an unmapped id.
"""

from datetime import datetime

import pandas as pd
import pytest

from scripts.rekey_odds_timeline_2020 import (
    RekeyInvariantError,
    build_rekey_map,
    kickoff_as_et,
    old_rule_season_start,
    old_rule_week,
)
from utils.date_utils import ET, get_nfl_season_start

# The pre-45bff24 rule's output, verified against `git show 45bff24`. Only 2020
# (and future 2026) diverge from the corrected rule.
OLD_RULE_SEASON_STARTS = {
    2018: "2018-09-06",
    2019: "2019-09-05",
    2020: "2020-09-03",
    2021: "2021-09-09",
    2022: "2022-09-08",
    2023: "2023-09-07",
    2024: "2024-09-05",
}

DIVERGENT_SEASONS = (2020, 2026)


class TestOldRuleSeasonStart:
    """The frozen replay of the fixed bug."""

    @pytest.mark.parametrize(("season", "expected"), OLD_RULE_SEASON_STARTS.items())
    def test_replays_the_pre_fix_rule(self, season: int, expected: str) -> None:
        assert old_rule_season_start(season).date().isoformat() == expected

    def test_2020_is_exactly_seven_days_before_the_corrected_opener(self) -> None:
        old = old_rule_season_start(2020)
        new = get_nfl_season_start(2020)
        assert new.date().isoformat() == "2020-09-10"
        assert (new - old).days == 7

    @pytest.mark.parametrize("season", sorted(OLD_RULE_SEASON_STARTS))
    def test_diverges_from_the_live_rule_only_for_2020(self, season: int) -> None:
        same = old_rule_season_start(season) == get_nfl_season_start(season)
        assert same is (season not in DIVERGENT_SEASONS)

    def test_2026_also_diverges(self) -> None:
        assert old_rule_season_start(2026).date().isoformat() == "2026-09-03"
        assert get_nfl_season_start(2026).date().isoformat() == "2026-09-10"

    def test_the_day_below_three_clause_is_load_bearing(self) -> None:
        """Without the ``day < 3`` bump, 2021 and 2022 would appear affected.

        Both years' first September Thursday falls before the 3rd (Sept 2 in
        2021, Sept 1 in 2022), which the OLD NFL_SEASON_START_DAY_RANGE lower
        bound of 3 pushed forward by a week.
        """
        first_thursday_day = {2021: 2, 2022: 1}
        for season, day in first_thursday_day.items():
            assert old_rule_season_start(season).day == day + 7


class TestOldRuleWeek:
    """The stored week numbers the replay has to reproduce."""

    def test_week_one_thursday_opener_lands_in_week_two(self) -> None:
        # 2020_W01_HOU@KC kicked off 2020-09-10 20:20 ET and was stored as W02.
        kickoff = datetime(2020, 9, 10, 20, 20, tzinfo=ET)
        assert old_rule_week(kickoff, 2020) == 2

    def test_week_is_clamped_at_one(self) -> None:
        kickoff = datetime(2020, 9, 1, 13, 0, tzinfo=ET)
        assert old_rule_week(kickoff, 2020) == 1

    def test_week_is_clamped_at_twenty_two(self) -> None:
        kickoff = datetime(2021, 6, 1, 13, 0, tzinfo=ET)
        assert old_rule_week(kickoff, 2020) == 22


class TestKickoffAsEt:
    """games.kickoff_et carries an ET wall clock; it is re-attached, not shifted."""

    def test_wall_clock_is_preserved(self) -> None:
        value = pd.Timestamp("2020-09-13 13:00:00+00:00")
        converted = kickoff_as_et(value)
        assert (converted.hour, converted.day) == (13, 13)
        assert converted.tzinfo is ET

    def test_naive_input_is_localized(self) -> None:
        converted = kickoff_as_et(pd.Timestamp("2020-11-01 20:20:00"))
        assert (converted.hour, converted.day) == (20, 1)
        assert converted.tzinfo is ET


def _games_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _reg_game(game_id: str, kickoff: str, away: str, home: str) -> dict:
    return {
        "game_id": game_id,
        "season": 2020,
        "game_type": "REG",
        "kickoff_et": pd.Timestamp(kickoff),
        "away_team": away,
        "home_team": home,
    }


class TestBuildRekeyMap:
    """The three-step resolution order, on synthetic frames."""

    def test_old_rule_inverse_branch(self) -> None:
        games = _games_frame(
            [_reg_game("2020_W01_HOU@KC", "2020-09-10 20:20:00", "HOU", "KC")]
        )
        mapping = build_rekey_map(games, ["2020_W02_HOU@KC"])
        assert mapping == {"2020_W02_HOU@KC": "2020_W01_HOU@KC"}

    def test_fixed_point_branch(self) -> None:
        """A stored id that is already a true games id maps to itself."""
        games = _games_frame(
            [_reg_game("2020_W01_HOU@KC", "2020-09-10 20:20:00", "HOU", "KC")]
        )
        mapping = build_rekey_map(games, ["2020_W01_HOU@KC"])
        assert mapping == {"2020_W01_HOU@KC": "2020_W01_HOU@KC"}

    def test_matchup_fallback_branch(self) -> None:
        """An early-September board listing whose date later moved."""
        games = _games_frame(
            [_reg_game("2020_W07_PIT@TEN", "2020-10-25 13:00:00", "PIT", "TEN")]
        )
        # The old rule would have stored this game as W08, so W05 is reachable
        # only through the (away, home) matchup map.
        mapping = build_rekey_map(games, ["2020_W05_PIT@TEN"])
        assert mapping == {"2020_W05_PIT@TEN": "2020_W07_PIT@TEN"}

    def test_resolution_order_prefers_the_fixed_point(self) -> None:
        """A true id is never re-shifted, even when it is also an old-rule key."""
        games = _games_frame(
            [
                _reg_game("2020_W02_HOU@KC", "2020-09-17 20:20:00", "HOU", "KC"),
                _reg_game("2020_W01_SEA@ATL", "2020-09-13 13:00:00", "SEA", "ATL"),
            ]
        )
        mapping = build_rekey_map(games, ["2020_W02_HOU@KC"])
        assert mapping["2020_W02_HOU@KC"] == "2020_W02_HOU@KC"

    def test_unmapped_id_raises(self) -> None:
        games = _games_frame(
            [_reg_game("2020_W01_HOU@KC", "2020-09-10 20:20:00", "HOU", "KC")]
        )
        with pytest.raises(RekeyInvariantError, match="could not be resolved"):
            build_rekey_map(games, ["2020_W03_SEA@ATL"])

    def test_duplicate_matchup_raises(self) -> None:
        games = _games_frame(
            [
                _reg_game("2020_W01_HOU@KC", "2020-09-10 20:20:00", "HOU", "KC"),
                _reg_game("2020_W09_HOU@KC", "2020-11-05 20:20:00", "HOU", "KC"),
            ]
        )
        with pytest.raises(RekeyInvariantError, match="ambiguous"):
            build_rekey_map(games, ["2020_W02_HOU@KC"])

    def test_playoff_rematches_do_not_break_the_matchup_map(self) -> None:
        """Only REG games feed the matchup map, so a postseason rematch is safe."""
        games = _games_frame(
            [
                _reg_game("2020_W01_HOU@KC", "2020-09-10 20:20:00", "HOU", "KC"),
                {
                    "game_id": "2020_W20_HOU@KC",
                    "season": 2020,
                    "game_type": "POST",
                    "kickoff_et": pd.Timestamp("2021-01-17 15:05:00"),
                    "away_team": "HOU",
                    "home_team": "KC",
                },
            ]
        )
        mapping = build_rekey_map(games, ["2020_W02_HOU@KC"])
        assert mapping == {"2020_W02_HOU@KC": "2020_W01_HOU@KC"}

    def test_map_is_idempotent(self) -> None:
        """Applying the map to its own output is the identity."""
        games = _games_frame(
            [
                _reg_game("2020_W01_HOU@KC", "2020-09-10 20:20:00", "HOU", "KC"),
                _reg_game("2020_W01_SEA@ATL", "2020-09-13 13:00:00", "SEA", "ATL"),
            ]
        )
        stored = ["2020_W02_HOU@KC", "2020_W02_SEA@ATL"]
        first = build_rekey_map(games, stored)
        second = build_rekey_map(games, sorted(set(first.values())))
        assert {second[value] for value in first.values()} == set(first.values())
        assert all(second[key] == key for key in second)
