"""Tests for the windowed, prior-regressed opponent-adjusted EPA rolling step."""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from features.opponent_adj import OPP_ADJ_COVERAGE_COLUMN, OpponentAdjuster
from features.opponent_adj_windowed import WindowedOpponentAdjuster


def _ts(text: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(text, tz="UTC"))


def _days(count: int) -> pd.Timedelta:
    return cast(pd.Timedelta, pd.Timedelta(days=count))


_METRICS = ("epa_per_play", "pass_epa_per_play", "rush_epa_per_play")
_HOUR = cast(pd.Timedelta, pd.Timedelta(hours=1))
_ONE_SECOND = cast(pd.Timedelta, pd.Timedelta(seconds=1))
_LOCK = _ts("2024-10-12 22:00")
_OUTPUT_COLUMNS = [
    "team",
    "side",
    "target_season",
    "target_week",
    "games_used",
    "rolling_opp_adj_epa_per_play",
    "rolling_opp_adj_pass_epa",
    "rolling_opp_adj_rush_epa",
    OPP_ADJ_COVERAGE_COLUMN,
    "latest_information_end",
]


def _game(
    season: int,
    week: int,
    end: pd.Timestamp,
    value: float,
    *,
    adjusted: bool = True,
) -> dict:
    """One adjusted-frame row for KC offense, shaped as ``per_game_adjusted`` returns."""
    row = {
        "team": "KC",
        "side": "offense",
        "season": season,
        "week": week,
        "_end": end,
        "adjusted": adjusted,
        "information_end": end + _HOUR if adjusted else pd.NaT,
    }
    for metric in _METRICS:
        row[f"opp_adj_{metric}"] = value if adjusted else np.nan
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["_end"] = pd.to_datetime(frame["_end"], utc=True)
    frame["information_end"] = pd.to_datetime(frame["information_end"], utc=True)
    return frame


def _targets(season: int = 2024, week: int = 3, lock: pd.Timestamp = _LOCK):
    """One KC target team-game, shaped as ``_target_team_games`` returns."""
    return pd.DataFrame(
        {
            "game_id": ["2024_W03_KC@LV"],
            "team": ["KC"],
            "season": [season],
            "week": [week],
            "lock": [lock],
        }
    )


def _two_season_history() -> list[dict]:
    return [
        _game(2023, 1, _ts("2023-09-10"), 0.2),
        _game(2023, 2, _ts("2023-09-17"), 0.2),
        _game(2024, 1, _LOCK - _days(14), 0.4),
        _game(2024, 2, _LOCK - _days(7), 0.6),
    ]


def _row(out: pd.DataFrame) -> pd.Series:
    return out.loc[(out["team"] == "KC") & (out["side"] == "offense")].iloc[0]


def test_decayed_current_season_blended_with_regressed_prior():
    # prior season: 0.2, 0.2 -> prior 0.5 * 0.2 = 0.1; 2024: 0.4 then 0.6 -> weights 0.9, 1.0
    adjusted = _frame(_two_season_history())

    out = WindowedOpponentAdjuster()._rolling_for_targets(adjusted, _targets())

    row = _row(out)
    expected = (0.9 * 0.4 + 1.0 * 0.6 + 3.0 * 0.1) / (1.9 + 3.0)
    for column in (
        "rolling_opp_adj_epa_per_play",
        "rolling_opp_adj_pass_epa",
        "rolling_opp_adj_rush_epa",
    ):
        assert row[column] == pytest.approx(expected)
    assert row["rolling_opp_adj_coverage"] == 1.0
    assert row["games_used"] == 4
    assert row["latest_information_end"] == _LOCK - _days(7) + _HOUR


def test_games_ending_after_the_lock_and_older_seasons_are_ignored():
    after_lock = _game(2024, 3, _LOCK + _ONE_SECOND, 9.0)
    too_old = _game(2022, 5, _ts("2022-10-09"), -9.0)
    adjusted = _frame([*_two_season_history(), after_lock, too_old])

    out = WindowedOpponentAdjuster()._rolling_for_targets(adjusted, _targets())

    assert out["rolling_opp_adj_epa_per_play"].iloc[0] == pytest.approx(
        0.25714, abs=1e-4
    )
    assert out["games_used"].iloc[0] == 4


def test_a_game_ending_exactly_at_the_lock_counts():
    at_lock = _game(2024, 3, _LOCK, 1.0)
    adjusted = _frame([at_lock])

    out = WindowedOpponentAdjuster()._rolling_for_targets(adjusted, _targets())

    assert out["rolling_opp_adj_epa_per_play"].iloc[0] == pytest.approx(
        1.0 / (1.0 + 3.0)
    )


def test_week_one_uses_the_regressed_prior_only():
    prior_season_only = _frame(_two_season_history()[:2])
    week_one = _targets(week=1, lock=_ts("2024-09-05"))

    out = WindowedOpponentAdjuster()._rolling_for_targets(prior_season_only, week_one)

    assert out["rolling_opp_adj_epa_per_play"].iloc[0] == pytest.approx(0.1)
    assert out["rolling_opp_adj_coverage"].iloc[0] == 1.0


def test_history_but_no_adjusted_game_gives_nan_and_flag_zero():
    unadjusted_only = _frame([_game(2024, 1, _LOCK - _days(7), 0.5, adjusted=False)])

    out = WindowedOpponentAdjuster()._rolling_for_targets(unadjusted_only, _targets())

    assert np.isnan(out["rolling_opp_adj_epa_per_play"].iloc[0])
    assert out["rolling_opp_adj_coverage"].iloc[0] == 0.0
    assert pd.isna(out["latest_information_end"].iloc[0])
    assert out["games_used"].iloc[0] == 1


def test_no_history_at_all_gives_no_row_and_schema_matches_the_parent():
    only_old = _frame([_game(2022, 5, _ts("2022-10-09"), -9.0)])
    full = _frame(_two_season_history())

    assert (
        len(WindowedOpponentAdjuster()._rolling_for_targets(only_old, _targets())) == 0
    )
    windowed = WindowedOpponentAdjuster()._rolling_for_targets(full, _targets())
    parent = OpponentAdjuster()._rolling_for_targets(full, _targets())
    assert list(windowed.columns) == _OUTPUT_COLUMNS
    assert list(windowed.columns) == list(parent.columns)
