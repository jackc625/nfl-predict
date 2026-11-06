"""League scoring level: the kickoff-plus-duration end fence and the prior-season blend."""

from __future__ import annotations

import pandas as pd
import pytest

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
