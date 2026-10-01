"""Unit tests for api.presentation (redesign Task 2).

Pins review-focus items 2 and 3: near-black team colours and unknown teams must still render a
readable block, and odd kickoffs (an international morning game, a Saturday slate, a Christmas
weekday, a Monday doubleheader, no date at all) must land in a sensible, labelled group.

ASCII only, no emoji (CLAUDE.md). The window separator is U+00B7, written as an escape.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from api.presentation import (
    MIN_BLOCK_CONTRAST,
    PANEL_HEX,
    TEXT_DARK,
    TEXT_LIGHT,
    TIME_TBD,
    UNKNOWN_BLOCK,
    TeamColors,
    contrast_ratio,
    decorate_game,
    group_games_by_window,
    kickoff_label,
    kickoff_window,
    team_block_colors,
    team_nickname,
)
from utils.team_data import NFL_TEAMS_DATA

DOT = "·"


class TestTeamBlockColors:
    def test_a_readable_primary_is_kept(self) -> None:
        assert team_block_colors("KC") == TeamColors("#E31837", TEXT_LIGHT)

    def test_dark_but_distinct_primaries_keep_their_identity(self) -> None:
        assert team_block_colors("NYG").bg == "#0B2265"
        assert team_block_colors("BAL").bg == "#241773"

    @pytest.mark.parametrize(
        ("abbr", "expected"),
        [
            ("CHI", TeamColors("#C83803", TEXT_LIGHT)),
            ("HOU", TeamColors("#A71930", TEXT_LIGHT)),
            ("NE", TeamColors("#C60C30", TEXT_LIGHT)),
            ("SEA", TeamColors("#69BE28", TEXT_DARK)),
            ("TEN", TeamColors("#4B92DB", TEXT_DARK)),
            ("CLE", TeamColors("#FF3C00", TEXT_DARK)),
        ],
    )
    def test_a_near_black_primary_falls_back_to_the_secondary(
        self, abbr: str, expected: TeamColors
    ) -> None:
        assert team_block_colors(abbr) == expected

    def test_a_pure_black_primary_is_never_a_block(self) -> None:
        assert team_block_colors("LV") == TeamColors("#A5ACAF", TEXT_DARK)

    def test_aliases_resolve_to_the_same_team(self) -> None:
        assert team_block_colors("LAR") == team_block_colors("LA")
        assert team_block_colors("OAK") == team_block_colors("LV")
        assert team_block_colors("kc") == team_block_colors("KC")

    @pytest.mark.parametrize("abbr", [None, "", "XYZ"])
    def test_an_unknown_team_gets_a_neutral_block(self, abbr: str | None) -> None:
        assert team_block_colors(abbr) == TeamColors(UNKNOWN_BLOCK, TEXT_LIGHT)

    @pytest.mark.parametrize("abbr", sorted(NFL_TEAMS_DATA))
    def test_every_team_block_reads_on_the_panel(self, abbr: str) -> None:
        colors = team_block_colors(abbr)
        assert colors.bg not in {"#000000", "#FFFFFF"}
        assert contrast_ratio(colors.bg, PANEL_HEX) >= MIN_BLOCK_CONTRAST
        # The text on the block: the better of white and ink is never below 4.4 for these colours.
        assert contrast_ratio(colors.fg, colors.bg) >= 4.4


class TestTeamNickname:
    @pytest.mark.parametrize(
        ("abbr", "expected"),
        [
            ("KC", "Chiefs"),
            ("SF", "49ers"),
            ("WAS", "Commanders"),
            ("LAR", "Rams"),
            ("XYZ", "XYZ"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_nickname(self, abbr: str | None, expected: str) -> None:
        assert team_nickname(abbr) == expected


class TestKickoffLabel:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime(2026, 9, 27, 13, 0), "Sun 1:00 PM ET"),
            ("2026-09-27 16:25:00", "Sun 4:25 PM ET"),
            (datetime(2026, 9, 28, 0, 20, tzinfo=UTC), "Sun 8:20 PM ET"),
            ("2026-09-27T17:00:00+00:00", "Sun 1:00 PM ET"),
            (datetime(2026, 9, 27, 9, 30), "Sun 9:30 AM ET"),
            ("2024-09-10", "Tue Sep 10"),
            (datetime(2024, 9, 10), "Tue Sep 10"),
            (date(2024, 9, 10), "Tue Sep 10"),
            (None, TIME_TBD),
            ("", TIME_TBD),
            ("not a date", TIME_TBD),
        ],
    )
    def test_label(self, value: datetime | date | str | None, expected: str) -> None:
        assert kickoff_label(value) == expected

    def test_a_pandas_not_a_time_is_time_tbd(self) -> None:
        assert kickoff_label(pd.NaT) == TIME_TBD  # pyright: ignore[reportArgumentType]


class TestKickoffWindow:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime(2026, 9, 24, 20, 15), "Thursday Night"),
            (datetime(2026, 9, 27, 9, 30), f"Sunday {DOT} 9:30 AM ET"),
            (datetime(2026, 9, 27, 13, 0), f"Sunday {DOT} 1:00 PM ET"),
            (datetime(2026, 9, 27, 16, 25), f"Sunday {DOT} 4:25 PM ET"),
            (datetime(2026, 9, 27, 20, 20), "Sunday Night"),
            (datetime(2026, 9, 28, 19, 15), "Monday Night"),
            ("2026-09-27", "Sunday"),
            (None, TIME_TBD),
        ],
    )
    def test_window(self, value: datetime | str | None, expected: str) -> None:
        assert kickoff_window(value) == expected


def _games(*kickoffs: datetime | None) -> list[dict]:
    return [{"game_id": f"g{i}", "game_date": k} for i, k in enumerate(kickoffs)]


class TestGroupGamesByWindow:
    def test_a_full_week_groups_in_first_appearance_order(self) -> None:
        games = _games(
            datetime(2026, 9, 24, 20, 15),
            datetime(2026, 9, 27, 9, 30),
            datetime(2026, 9, 27, 13, 0),
            datetime(2026, 9, 27, 13, 0),
            datetime(2026, 9, 27, 16, 5),
            datetime(2026, 9, 27, 16, 25),
            datetime(2026, 9, 27, 20, 20),
            datetime(2026, 9, 28, 19, 15),
            datetime(2026, 9, 28, 20, 15),
        )
        groups = group_games_by_window(games)
        assert [g["label"] for g in groups] == [
            "Thursday Night",
            f"Sunday {DOT} 9:30 AM ET",
            f"Sunday {DOT} 1:00 PM ET",
            f"Sunday {DOT} 4:05 / 4:25 PM ET",
            "Sunday Night",
            "Monday Night",
        ]
        assert [len(g["games"]) for g in groups] == [1, 1, 2, 2, 1, 2]
        # Input order is kept inside a group.
        assert [g["game_id"] for g in groups[5]["games"]] == ["g7", "g8"]

    def test_a_saturday_late_season_slate(self) -> None:
        groups = group_games_by_window(
            _games(
                datetime(2025, 12, 20, 16, 30),
                datetime(2025, 12, 20, 20, 0),
                datetime(2025, 12, 21, 13, 0),
            )
        )
        assert [g["label"] for g in groups] == [
            f"Saturday {DOT} 4:30 PM ET",
            "Saturday Night",
            f"Sunday {DOT} 1:00 PM ET",
        ]

    def test_a_christmas_day_weekday(self) -> None:
        groups = group_games_by_window(
            _games(datetime(2024, 12, 25, 13, 0), datetime(2024, 12, 25, 16, 30))
        )
        assert [g["label"] for g in groups] == [
            f"Wednesday {DOT} 1:00 PM ET",
            f"Wednesday {DOT} 4:30 PM ET",
        ]

    def test_a_game_with_no_date_lands_in_time_tbd(self) -> None:
        groups = group_games_by_window(_games(datetime(2026, 9, 27, 13, 0), None))
        assert [g["label"] for g in groups] == [f"Sunday {DOT} 1:00 PM ET", TIME_TBD]
        assert groups[1]["games"][0]["game_id"] == "g1"

    def test_an_empty_week_has_no_groups(self) -> None:
        assert group_games_by_window([]) == []


class TestDecorateGame:
    def test_it_adds_the_presentation_fields_without_mutating_the_input(self) -> None:
        game = {
            "game_id": "2026_W03_KC@MIA",
            "away_team": "KC",
            "home_team": "MIA",
            "game_date": datetime(2026, 9, 27, 13, 0),
            "wp_prob": 0.412,
        }
        decorated = decorate_game(game)
        assert decorated is not game
        assert "away_color" not in game
        assert decorated["wp_prob"] == 0.412
        assert decorated["away_color"] == "#E31837"
        assert decorated["away_fg"] == TEXT_LIGHT
        assert decorated["home_color"] == "#008E97"
        assert decorated["home_fg"] == TEXT_DARK
        assert decorated["away_name"] == "Chiefs"
        assert decorated["home_name"] == "Dolphins"
        assert decorated["kickoff_label"] == "Sun 1:00 PM ET"
        assert decorated["window_label"] == f"Sunday {DOT} 1:00 PM ET"

    def test_an_unknown_team_and_missing_date_still_decorate(self) -> None:
        decorated = decorate_game({"away_team": "XYZ", "home_team": None})
        assert decorated["away_color"] == UNKNOWN_BLOCK
        assert decorated["away_name"] == "XYZ"
        assert decorated["home_name"] == ""
        assert decorated["kickoff_label"] == TIME_TBD
        assert decorated["window_label"] == TIME_TBD


def test_the_helpers_are_jinja_globals() -> None:
    from api.dependencies import templates

    rendered = templates.env.from_string(
        "{{ team_colors('KC').bg }} {{ team_colors('KC').fg }} {{ team_nickname('KC') }}"
    ).render()
    assert rendered == "#E31837 #FFFFFF Chiefs"
