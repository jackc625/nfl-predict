"""Tests for features.qb_early: a QB tracker that never reads an untimed depth chart."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from features.injury import REPLACEMENT_LEVEL_QB_QUALITY
from features.qb_early import QBTrackerEarly
from features.qb_tracking import DepthChartQBs, DepthChartSchemaError
from features.qb_v2 import QBTrackerV2

_ET = ZoneInfo("America/New_York")
LOCK = pd.Timestamp("2024-09-28 18:00", tz="America/New_York")
_STARTER = "00-0033873"
_UNSEEN = "00-0099999"


def _chart_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": 2024,
        "week": 4,
        "club_code": "KC",
        "position": "QB",
        "depth_team": "1",
        "gsis_id": _UNSEEN,
    }
    row.update(overrides)
    return row


def test_a_week_keyed_chart_is_never_read() -> None:
    chart = pd.DataFrame([_chart_row()])
    result = QBTrackerEarly().resolve_depth_chart_qbs(chart, 2024, 4, "KC", LOCK)
    assert result == DepthChartQBs(None, None, None)


def test_a_timed_chart_is_read_only_up_to_the_lock() -> None:
    timed = pd.DataFrame(
        [
            {
                "dt": "2025-09-28T16:55:00Z",  # LOCK - 1 h (LOCK is 22:00Z in EDT)
                "club_code": "KC",
                "position": "QB",
                "depth_team": "1",
                "gsis_id": "00-000000A",
            },
            {
                "dt": "2025-09-28T22:00:01Z",  # one second after the lock
                "club_code": "KC",
                "position": "QB",
                "depth_team": "1",
                "gsis_id": "00-000000B",
            },
        ]
    )
    lock = pd.Timestamp("2025-09-28 18:00", tz="America/New_York")
    result = QBTrackerEarly().resolve_depth_chart_qbs(timed, 2025, 4, "KC", lock)
    assert result.qb1 == "00-000000A"
    assert result.published_at is not None


def test_a_chart_with_neither_schema_still_raises() -> None:
    chart = pd.DataFrame([_chart_row()]).drop(columns=["week"])
    with pytest.raises(DepthChartSchemaError):
        QBTrackerEarly().resolve_depth_chart_qbs(chart, 2024, 4, "KC", LOCK)


def _games() -> pd.DataFrame:
    rows = [
        ("2024_01_KC_BUF", 1, "KC", "BUF", datetime(2024, 9, 5, 20, 20, tzinfo=_ET)),
        ("2024_02_KC_DAL", 2, "KC", "DAL", datetime(2024, 9, 12, 13, 0, tzinfo=_ET)),
        ("2024_03_KC_DEN", 3, "KC", "DEN", datetime(2024, 9, 19, 13, 0, tzinfo=_ET)),
        ("2024_04_KC_LA", 4, "KC", "LA", datetime(2024, 9, 26, 13, 0, tzinfo=_ET)),
    ]
    return pd.DataFrame(
        rows, columns=["game_id", "week", "home_team", "away_team", "kickoff_et"]
    ).assign(season=2024)


def _pbp() -> pd.DataFrame:
    plays = []
    for game_id, week, away in (
        ("2024_01_KC_BUF", 1, "BUF"),
        ("2024_02_KC_DAL", 2, "DAL"),
        ("2024_03_KC_DEN", 3, "DEN"),
    ):
        for i in range(20):
            plays.append(
                {
                    "game_id": game_id,
                    "season": 2024,
                    "week": week,
                    "home_team": "KC",
                    "away_team": away,
                    "posteam": "KC",
                    "passer_player_id": _STARTER,
                    "qb_epa": 0.3 + i * 0.01 + week * 0.02,
                    "cpoe": 5.0 + i * 0.1,
                    "play_id": week * 100 + i,
                }
            )
    return pd.DataFrame(plays)


def _chart(week4_qb1: str) -> pd.DataFrame:
    rows = [
        _chart_row(week=week, gsis_id=_STARTER if week < 4 else week4_qb1)
        for week in (1, 2, 3, 4)
    ]
    return pd.DataFrame(rows)


def _kc_week4_value(tracker: QBTrackerV2, chart: pd.DataFrame) -> float:
    tracker._depth_chart_cache = {2024: chart}
    tracker._pbp_cache = {2024: _pbp()}
    games = _games()
    # An early frame: Tuesday after week 3 ended, long before the day-before lock.
    early = pd.Timestamp("2024-09-24 11:55", tz="America/New_York")
    lock_frame = pd.Series(dict.fromkeys(games["game_id"], early), name="lock")
    result = tracker.build_features(
        games,
        as_of_datetime=datetime(2024, 9, 24, 12, 0),
        target_season=2024,
        target_week=4,
        lock_frame=lock_frame,
    )
    row = result.loc[(result["game_id"] == "2024_04_KC_LA") & (result["team"] == "KC")]
    return float(row["qb_adjustment"].iloc[0])


def test_the_last_game_passer_replaces_a_chart_backup() -> None:
    starter = _kc_week4_value(QBTrackerV2(), _chart(_STARTER))
    planted = _chart(_UNSEEN)
    assert _kc_week4_value(QBTrackerV2(), planted) == REPLACEMENT_LEVEL_QB_QUALITY
    assert starter != REPLACEMENT_LEVEL_QB_QUALITY
    assert _kc_week4_value(QBTrackerEarly(), planted) == pytest.approx(starter)
