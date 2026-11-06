"""Tests for Study 1b's inputs: the v2 input rows as known at each game's early capture.

Every frame is built here by hand and every gold matrix is written under ``tmp_path``;
nothing reads the live lake. The QB tracker is stubbed (it would load pinned play-by-play),
and so are the two families where a test needs to plant a late information time. Instants
are written in ET (``et``); the module compares instants, so an equality is an instant's.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import utils.game_lock as lock_rule
from backtest import study_1b_inputs
from backtest.study_1b_inputs import (
    EARLY_REPLACED_COLUMNS,
    NO_GOLD_ROW,
    NOT_SETTLED,
    build_early_inputs,
    early_lock_frame,
    replace_early_columns,
    settled_by_early_time,
)
from scripts.build_gold_v2 import add_v2_derived_columns

_ET = "America/New_York"

# 2024 week 2: two Sunday games, locked Saturday 2024-09-14 18:00 ET.
SUN = "2024_W02_LV@BAL"
OTHER = "2024_W02_NYG@WAS"

# 2020 weeks 12-13: KC's week-12 game moved to a Wednesday (ends Thursday 00:00 ET).
KC_WED = "2020_W12_KC@PIT"
LV_ATL = "2020_W12_LV@ATL"
NO_DEN = "2020_W12_NO@DEN"
KC_GAME = "2020_W13_DEN@KC"
OTHER_GAME = "2020_W13_NO@ATL"
NO_GOLD_GAME = "2020_W13_LV@NYJ"
THU_W13 = "2020_W13_DAL@BAL"
EARLY_W13 = "2020-12-01 11:55"

TEAM_QB: dict[str, float] = {"ATL": 1.2, "NO": -0.5, "LV": 0.3, "NYJ": 0.1}


def et(text: str) -> pd.Timestamp:
    """*text* as an ET wall-clock instant."""
    return cast(pd.Timestamp, pd.Timestamp(text, tz=_ET))


def _games(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Silver-shaped games from ``(game_id, ET kickoff, home_score, away_score)``.

    Season, week and both teams are read off the id (``2020_W12_KC@PIT``); the kickoff
    is stored in UTC, as silver stores it.
    """
    records = []
    for game_id, kickoff, home_score, away_score in rows:
        season, week, teams = game_id.split("_")
        away, home = teams.split("@")
        records.append(
            {
                "game_id": game_id,
                "season": int(season),
                "week": int(week.removeprefix("W")),
                "home_team": home,
                "away_team": away,
                "kickoff_et": et(kickoff).tz_convert("UTC"),
                "neutral_site": False,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
    return pd.DataFrame.from_records(records)


def _week_2_2024() -> pd.DataFrame:
    return _games(
        [(SUN, "2024-09-15 13:00", 20.0, 17.0), (OTHER, "2024-09-15 13:00", 21.0, 18.0)]
    )


def _weeks_12_13_2020() -> pd.DataFrame:
    return _games(
        [
            (LV_ATL, "2020-11-29 13:00", 17.0, 13.0),  # 30 points, ends 17:00
            (NO_DEN, "2020-11-29 16:25", 10.0, 30.0),  # 40 points, ends 20:25
            (KC_WED, "2020-12-02 20:00", 30.0, 30.0),  # 60 points, ends Thu 00:00
            (THU_W13, "2020-12-03 20:20", 24.0, 24.0),  # after the early time
            (KC_GAME, "2020-12-06 13:00", 22.0, 16.0),
            (OTHER_GAME, "2020-12-06 13:00", 16.0, 21.0),
            (NO_GOLD_GAME, "2020-12-06 13:00", 28.0, 31.0),
        ]
    )


def _gold_row(game_id: str) -> dict[str, Any]:
    """One gold row before its derived columns: identity, per-side inputs, stale values."""
    season, week, _ = game_id.split("_")
    row: dict[str, Any] = {
        "game_id": game_id,
        "season": int(season),
        "week": int(week.removeprefix("W")),
        "elo_diff": 87.0,
        "hfa_used": 48.0,
        "wind_mph": 14.0,
        "home_win": 1.0,
        "home_qb_adjustment": 9.9,
        "away_qb_adjustment": 9.0,
    }
    for side, success, pace in (("home", 0.47, 28.0), ("away", 0.43, 30.0)):
        row[f"{side}_mkt_rating"] = 9.9
        for unit in ("off", "def"):
            row[f"{side}_{unit}_rolling_success_rate"] = success
            row[f"{side}_{unit}_rolling_opp_adj_epa_per_play"] = success - 0.4
        row[f"{side}_off_rolling_neutral_pace"] = pace
    row.update(
        mkt_rating_diff=9.9,
        mkt_total_sum=99.0,
        mkt_rating_coverage=0.0,
        league_points_level=99.0,
    )
    return row


def _gold(ids: list[str]) -> pd.DataFrame:
    """A gold matrix of *ids*, its derived columns computed as the v2 gold build does."""
    return add_v2_derived_columns(
        pd.DataFrame.from_records([_gold_row(g) for g in ids])
    )


def _write_gold(directory: Path, ids_by_target: Mapping[str, list[str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for target, ids in ids_by_target.items():
        _gold(ids).to_parquet(directory / f"features_{target}.parquet", index=False)
    return directory


def _qb_rows(games_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """What the tracker returns for one week: one row per team per game."""
    week_games = games_df.loc[
        (games_df["season"] == season) & (games_df["week"] == week)
    ]
    rows = [
        {"game_id": game_id, "team": team, "qb_adjustment": TEAM_QB.get(team, 0.0)}
        for game_id, home, away in zip(
            week_games["game_id"],
            week_games["home_team"],
            week_games["away_team"],
            strict=True,
        )
        for team in (home, away)
    ]
    return pd.DataFrame.from_records(rows)


class _QBStub:
    """Stands in for ``QBTrackerEarly.build_features`` and records each call's locks."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, pd.Series]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = self

        def build_features(
            _tracker: object,
            games_df: pd.DataFrame,
            as_of_datetime: datetime,
            *,
            target_season: int,
            target_week: int,
            lock_frame: pd.Series,
        ) -> pd.DataFrame:
            stub.calls.append((target_season, target_week, lock_frame))
            return _qb_rows(games_df, target_season, target_week)

        monkeypatch.setattr(
            study_1b_inputs.QBTrackerEarly, "build_features", build_features
        )


def test_early_lock_frame_replaces_only_early_timed_games() -> None:
    games = _week_2_2024()
    locks = early_lock_frame(games, pd.Series({SUN: et("2024-09-10 11:55")}))
    assert locks[SUN] == et("2024-09-10 11:55")
    assert locks[OTHER] == lock_rule.lock_frame(games)[OTHER]


def test_an_early_time_after_the_lock_is_refused() -> None:
    games = _week_2_2024()
    with pytest.raises(ValueError, match="after"):
        early_lock_frame(games, pd.Series({SUN: et("2024-09-14 18:01")}))


def test_an_early_time_for_a_game_not_in_games_is_refused() -> None:
    with pytest.raises(ValueError, match="2024_W02_KC@CIN"):
        early_lock_frame(
            _week_2_2024(), pd.Series({"2024_W02_KC@CIN": et("2024-09-10 11:55")})
        )


def test_a_team_whose_previous_game_ends_after_the_early_time_is_not_settled() -> None:
    # Review Focus 5. KC's last game kicked off Wed 2020-12-02 20:00 ET and ends Thu
    # 00:00, after the Tue 11:55 early time of its Sunday game. NO and ATL last played
    # Sun 11-29. LV and ATL, in week 12, have no earlier game here at all.
    games = _weeks_12_13_2020()
    early_times = pd.Series(
        {
            KC_GAME: et(EARLY_W13),
            OTHER_GAME: et(EARLY_W13),
            LV_ATL: et("2020-11-24 11:55"),
        }
    )
    out = settled_by_early_time(games, early_times)
    assert out.dtype == bool
    settled = out.to_dict()
    assert not settled[KC_GAME] and settled[OTHER_GAME]
    assert settled[LV_ATL]


def test_replaced_columns_come_from_the_early_families_and_wind_is_unknown() -> None:
    games = _weeks_12_13_2020()
    gold = _gold([OTHER_GAME])
    qb = pd.DataFrame(
        {
            "game_id": [OTHER_GAME, OTHER_GAME],
            "team": ["ATL", "NO"],
            "qb_adjustment": [1.2, -0.5],
        }
    )
    ratings = pd.DataFrame(
        {
            "game_id": [OTHER_GAME],
            "home_mkt_rating": [1.5],
            "away_mkt_rating": [-1.0],
            "mkt_rating_diff": [2.5],
            "mkt_total_sum": [45.5],
            "mkt_rating_coverage": [1.0],
            "information_time": [et("2020-11-29 16:25")],
        }
    )
    level = pd.DataFrame(
        {
            "game_id": [OTHER_GAME],
            "league_points_level": [46.0],
            "information_time": [et("2020-11-29 20:25")],
        }
    )
    out = replace_early_columns(gold, games, qb, ratings, level)
    assert out.loc[0, "qb_value_diff"] == pytest.approx(1.7)
    assert (
        out.loc[0, "mkt_rating_diff"] == 2.5
        and out.loc[0, "league_points_level"] == 46.0
    )
    assert np.isnan(out.loc[0, "wind_mph"]) and out.loc[0, "elo_diff"] == 87.0
    assert list(out.columns) == list(gold.columns)
    kept = [
        c for c in gold.columns if c not in (*EARLY_REPLACED_COLUMNS, "qb_value_diff")
    ]
    pd.testing.assert_frame_equal(out[kept], gold[kept])


def test_build_refuses_a_family_read_after_the_early_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    games = _weeks_12_13_2020()
    early_times = pd.Series({OTHER_GAME: et(EARLY_W13)})
    gold_dir = _write_gold(
        tmp_path / "gold", {target: [OTHER_GAME] for target in ("wp", "ats", "ou")}
    )
    _QBStub().install(monkeypatch)

    def late_ratings(
        scope: pd.DataFrame, nflverse: pd.DataFrame, *, locks: pd.Series
    ) -> pd.DataFrame:
        late = et(EARLY_W13) + pd.Timedelta(seconds=1)
        return pd.DataFrame(
            {
                "game_id": scope["game_id"],
                "home_mkt_rating": 1.0,
                "away_mkt_rating": 0.0,
                "mkt_rating_diff": 1.0,
                "mkt_total_sum": 44.0,
                "mkt_rating_coverage": 1.0,
                "information_time": late,
            }
        )

    def level(scope: pd.DataFrame, *, locks: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": scope["game_id"],
                "league_points_level": 44.0,
                "information_time": et("2020-11-29 20:25"),
            }
        )

    monkeypatch.setattr(study_1b_inputs, "market_ratings", late_ratings)
    monkeypatch.setattr(study_1b_inputs, "league_points_level", level)
    with pytest.raises(RuntimeError, match="market_rating"):
        build_early_inputs(gold_dir, games, pd.DataFrame(), early_times)


def _nflverse(games: pd.DataFrame) -> pd.DataFrame:
    """An nflverse-shaped schedule carrying a line for every game."""
    return pd.DataFrame(
        {
            "season": games["season"],
            "week": games["week"],
            "home_team": games["home_team"],
            "away_team": games["away_team"],
            "spread_line": 3.0,
            "total_line": 45.0,
        }
    )


def test_build_reads_every_family_at_the_early_time_and_counts_what_it_leaves_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    games = _weeks_12_13_2020()
    early_times = pd.Series(
        {
            KC_GAME: et(EARLY_W13),
            OTHER_GAME: et(EARLY_W13),
            NO_GOLD_GAME: et(EARLY_W13),
        }
    )
    every = [KC_GAME, OTHER_GAME, NO_GOLD_GAME]
    gold_dir = _write_gold(
        tmp_path / "gold",
        {"wp": every, "ats": every, "ou": [KC_GAME, OTHER_GAME]},
    )
    qb_stub = _QBStub()
    qb_stub.install(monkeypatch)

    out = build_early_inputs(gold_dir, games, _nflverse(games), early_times)

    # QB ran once, for the one (season, week) holding an early-timed game, at the early locks.
    assert [(season, week) for season, week, _ in qb_stub.calls] == [(2020, 13)]
    passed_locks = qb_stub.calls[0][2]
    assert passed_locks[OTHER_GAME] == et(EARLY_W13)
    assert passed_locks[THU_W13] == lock_rule.lock_frame(games)[THU_W13]

    reasons = dict(zip(out.excluded["game_id"], out.excluded["reason"], strict=True))
    assert reasons == {KC_GAME: NOT_SETTLED, NO_GOLD_GAME: NO_GOLD_ROW}
    for target in ("wp", "ats", "ou"):
        rows = out.rows[target]
        assert rows["game_id"].tolist() == [OTHER_GAME]
        # Only the two Sunday 11-29 games had ended: (30 + 40) / 2. KC's Wednesday
        # game (60 points) and the Thursday game would move it.
        assert rows.loc[0, "league_points_level"] == pytest.approx(35.0)
        assert rows.loc[0, "mkt_rating_coverage"] == 1.0
        assert rows.loc[0, "qb_value_diff"] == pytest.approx(1.2 - (-0.5))
        assert np.isnan(rows.loc[0, "wind_mph"])
