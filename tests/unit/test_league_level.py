"""League scoring level: the kickoff-plus-duration end fence and the prior-season blend."""

from __future__ import annotations

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.league_level import LEVEL_PRIOR_GAMES, league_points_level

ET = "America/New_York"


def _game(game_id, season, kickoff_et, home_score=None, away_score=None):
    return {
        "game_id": game_id,
        "season": season,
        "kickoff_et": pd.Timestamp(kickoff_et, tz=ET),
        "home_score": home_score,
        "away_score": away_score,
    }


def _prior_season() -> list[dict]:
    # 2024 mean total: (40 + 48) / 2 = 44.0
    return [
        _game("2024_A", 2024, "2024-10-06 13:00", 20, 20),
        _game("2024_B", 2024, "2024-10-13 13:00", 26, 22),
    ]


def test_level_blends_current_season_with_the_prior_season_mean():
    games = pd.DataFrame(
        [
            *_prior_season(),
            _game("2025_A", 2025, "2025-11-02 13:00", 28, 22),  # total 50
            _game("2025_B", 2025, "2025-11-02 13:00", 20, 20),  # total 40
            _game("2025_G", 2025, "2025-11-09 13:00"),  # target, unplayed
        ]
    )
    out = league_points_level(games).set_index("game_id")
    assert LEVEL_PRIOR_GAMES == 32.0
    assert out.loc["2025_G", "league_points_level"] == pytest.approx(
        (90 + 32 * 44.0) / (2 + 32)
    )


def test_a_game_kicked_off_but_not_ended_by_the_lock_is_not_read():
    # Sunday 16:25 ET game ends 20:25 ET, after the Sunday 18:00 ET lock of the Monday target.
    games = pd.DataFrame(
        [
            *_prior_season(),
            _game("2025_A", 2025, "2025-11-02 13:00", 28, 22),  # total 50, ends 17:00
            _game("2025_B", 2025, "2025-11-02 13:00", 20, 20),  # total 40, ends 17:00
            _game(
                "2025_LATE", 2025, "2025-11-02 16:25", 30, 30
            ),  # total 60, ends 20:25
            _game("2025_MON", 2025, "2025-11-03 20:15"),  # target, unplayed
        ]
    )
    out = league_points_level(games).set_index("game_id")
    assert out.loc["2025_MON", "league_points_level"] == pytest.approx(
        (90 + 32 * 44.0) / 34
    )


def test_a_game_ending_exactly_at_the_lock_is_read_and_one_minute_later_is_not():
    # Saturday 14:00 ET game ends exactly at the Saturday 18:00 ET lock of the Sunday
    # target; the Saturday 14:01 ET game ends one minute after it.
    games = pd.DataFrame(
        [
            *_prior_season(),
            _game(
                "2025_EDGE", 2025, "2025-11-08 14:00", 28, 22
            ),  # total 50, ends 18:00
            _game(
                "2025_LATE", 2025, "2025-11-08 14:01", 40, 30
            ),  # total 70, ends 18:01
            _game("2025_SUN", 2025, "2025-11-09 13:00"),  # target, unplayed
        ]
    )
    out = league_points_level(games).set_index("game_id")
    assert out.loc["2025_SUN", "league_points_level"] == pytest.approx(
        (50 + 32 * 44.0) / 33
    )


def test_without_a_previous_season_the_level_is_the_current_season_mean():
    games = pd.DataFrame(
        [
            _game("2002_A", 2002, "2002-09-08 13:00", 28, 22),  # total 50
            _game("2002_B", 2002, "2002-09-08 13:00", 20, 20),  # total 40
            _game("2002_G", 2002, "2002-09-15 13:00"),  # target, unplayed
        ]
    )
    out = league_points_level(games).set_index("game_id")
    assert out.loc["2002_G", "league_points_level"] == pytest.approx(45.0)


def test_nothing_ended_before_the_lock_gives_nan_level_and_no_information_time():
    games = pd.DataFrame(
        [
            _game("2003_G", 2003, "2003-09-07 13:00"),  # first game: nothing read
        ]
    )
    out = league_points_level(games).set_index("game_id")
    assert pd.isna(out.loc["2003_G", "league_points_level"])
    assert pd.isna(out.loc["2003_G", "information_time"])


def test_information_time_is_the_latest_end_read():
    games = pd.DataFrame(
        [
            *_prior_season(),
            _game("2025_EARLY", 2025, "2025-11-08 13:00", 20, 20),  # ends 17:00
            _game("2025_EDGE", 2025, "2025-11-08 14:00", 28, 22),  # ends 18:00
            _game("2025_LATE", 2025, "2025-11-08 14:01", 40, 30),  # ends 18:01, unread
            _game("2025_SUN", 2025, "2025-11-09 13:00"),  # target, unplayed
        ]
    )
    out = league_points_level(games).set_index("game_id")
    expected = pd.Timestamp("2025-11-08 14:00", tz=ET) + pd.Timedelta(hours=4)
    assert out.loc["2025_SUN", "information_time"] == expected


def _early_week_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            *_prior_season(),
            _game("2025_THU", 2025, "2025-11-06 20:15", 28, 22),  # total 50
            _game("2025_SUN", 2025, "2025-11-09 13:00"),  # target, unplayed
        ]
    )


def test_a_supplied_earlier_time_leaves_out_a_game_ended_after_it():
    games = _early_week_games()
    locks = lock_rule.lock_frame(games)
    locks["2025_SUN"] = pd.Timestamp("2025-11-04 11:55", tz=ET)
    default = league_points_level(games).set_index("game_id")
    moved = league_points_level(games, locks=locks).set_index("game_id")
    assert default.loc["2025_SUN", "league_points_level"] == pytest.approx(
        (50 + 32 * 44.0) / 33
    )
    assert moved.loc["2025_SUN", "league_points_level"] == pytest.approx(44.0)


def test_supplied_locks_must_name_every_game():
    games = _early_week_games()
    locks = lock_rule.lock_frame(games)
    with pytest.raises(ValueError, match="2025_SUN"):
        league_points_level(games, locks=locks.drop("2025_SUN"))


def test_supplied_locks_must_not_name_a_game_twice():
    # A repeated id once shifted later games onto a neighbour's lock, so a game could
    # read games that ended after its own supplied instant.
    games = _early_week_games()
    locks = lock_rule.lock_frame(games)
    locks["2025_SUN"] = pd.Timestamp("2025-11-04 11:55", tz=ET)
    repeated = pd.concat([locks.loc[["2024_A"]], locks])
    with pytest.raises(ValueError, match=r"more than once.*2024_A"):
        league_points_level(games, locks=repeated)


def test_supplied_locks_must_give_every_game_an_instant():
    # A missing instant once sorted after every game end, so the game read its season.
    games = _early_week_games()
    locks = lock_rule.lock_frame(games)
    locks["2025_SUN"] = pd.NaT
    with pytest.raises(ValueError, match=r"no instant.*2025_SUN"):
        league_points_level(games, locks=locks)


def test_supplied_locks_are_matched_by_game_id_not_by_position():
    games = _early_week_games()
    locks = lock_rule.lock_frame(games)
    extra = pd.Series(
        [pd.Timestamp("2025-12-01 18:00", tz=ET)], index=pd.Index(["2025_X"])
    )
    shuffled = pd.concat([locks.iloc[::-1], extra])
    pd.testing.assert_frame_equal(
        league_points_level(games, locks=shuffled), league_points_level(games)
    )
