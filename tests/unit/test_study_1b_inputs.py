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
    SITE_NOT_KNOWN_AT_EARLY_CAPTURE,
    EarlyInputs,
    build_early_inputs,
    early_lock_frame,
    replace_early_columns,
    settled_by_early_time,
)
from models.v2_recipe import V2_INPUTS, V2_TARGETS
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


def _games(
    rows: list[tuple[str, str, float, float]],
    roofs: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    """Silver-shaped games from ``(game_id, ET kickoff, home_score, away_score)``.

    Season, week and both teams are read off the id (``2020_W12_KC@PIT``); the kickoff
    is stored in UTC, as silver stores it. ``venue_roof`` is ``outdoor`` unless *roofs*
    names the game. Every game is a regular-season game at its home team's stadium.
    """
    roofs = roofs or {}
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
                "venue_roof": roofs.get(game_id, "outdoor"),
                "game_type": "REG",
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


def _weeks_12_13_2020(roofs: Mapping[str, str | None] | None = None) -> pd.DataFrame:
    return _games(
        [
            (LV_ATL, "2020-11-29 13:00", 17.0, 13.0),  # 30 points, ends 17:00
            (NO_DEN, "2020-11-29 16:25", 10.0, 30.0),  # 40 points, ends 20:25
            (KC_WED, "2020-12-02 20:00", 30.0, 30.0),  # 60 points, ends Thu 00:00
            (THU_W13, "2020-12-03 20:20", 24.0, 24.0),  # after the early time
            (KC_GAME, "2020-12-06 13:00", 22.0, 16.0),
            (OTHER_GAME, "2020-12-06 13:00", 16.0, 21.0),
            (NO_GOLD_GAME, "2020-12-06 13:00", 28.0, 31.0),
        ],
        roofs,
    )


#: The stale gold QB values of a fixture game, (home, away).
STALE_QB = (9.9, 9.0)


def _range_game(season: int) -> str:
    """The extra game every fixture gold season holds, its stale QB values negated.

    It opens the season's gold QB range to [-9.9, 9.9] (home) and [-9.0, 9.0] (away), as a
    real season's spread of QB values encloses the early ones.
    """
    return f"{season}_W01_QBL@QBH"


def _gold_row(game_id: str, qb: tuple[float, float] = STALE_QB) -> dict[str, Any]:
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
        "home_qb_adjustment": qb[0],
        "away_qb_adjustment": qb[1],
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


def _gold(
    ids: list[str], qb: Mapping[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """A gold matrix of *ids* plus each season's ``_range_game``, derived columns included.

    *qb* overrides a game's stale ``(home, away)`` QB values. The derived columns are
    computed as the v2 gold build does.
    """
    qb = qb or {}
    seasons = sorted({int(game_id.split("_")[0]) for game_id in ids})
    records = [_gold_row(g, qb.get(g, STALE_QB)) for g in ids] + [
        _gold_row(_range_game(season), (-STALE_QB[0], -STALE_QB[1]))
        for season in seasons
    ]
    return add_v2_derived_columns(pd.DataFrame.from_records(records))


def _write_gold(directory: Path, ids_by_target: Mapping[str, list[str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for target, ids in ids_by_target.items():
        _gold(ids).to_parquet(directory / f"features_{target}.parquet", index=False)
    return directory


def _qb_rows(
    week_games: pd.DataFrame, team_qb: Mapping[str, float] = TEAM_QB
) -> pd.DataFrame:
    """What the tracker returns for *week_games*: one row per team per game."""
    rows = [
        {"game_id": game_id, "team": team, "qb_adjustment": team_qb.get(team, 0.0)}
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

    def __init__(self, team_qb: Mapping[str, float] = TEAM_QB) -> None:
        self.team_qb = team_qb
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
            week_games = games_df.loc[
                (games_df["season"] == target_season)
                & (games_df["week"] == target_week)
            ]
            return _qb_rows(week_games, stub.team_qb)

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
    gold_row = gold.loc[gold["game_id"] == OTHER_GAME].reset_index(drop=True)
    pd.testing.assert_frame_equal(out[kept], gold_row[kept])


def _early_families(
    games_df: pd.DataFrame, ids: list[str], team_qb: Mapping[str, float] = TEAM_QB
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(qb, ratings, level)`` for *ids*, as the early families would return them."""
    qb = _qb_rows(games_df.loc[games_df["game_id"].isin(ids)], team_qb)
    ratings = pd.DataFrame(
        {
            "game_id": ids,
            "home_mkt_rating": 1.5,
            "away_mkt_rating": -1.0,
            "mkt_rating_diff": 2.5,
            "mkt_total_sum": 45.5,
            "mkt_rating_coverage": 1.0,
            "information_time": et("2020-11-29 16:25"),
        }
    )
    level = pd.DataFrame(
        {
            "game_id": ids,
            "league_points_level": 46.0,
            "information_time": et("2020-11-29 20:25"),
        }
    )
    return qb, ratings, level


def test_a_rebuilt_qb_value_outside_its_seasons_gold_range_is_clipped_to_it() -> None:
    # The 2020 gold range is [-9.9, 9.9] home and [-9.0, 9.0] away. The 2019 game widens
    # 2019's range only, so it must not reach the 2020 game's clip.
    prior_season_game = "2019_W17_ATL@NO"
    games = _weeks_12_13_2020()
    gold = _gold([OTHER_GAME, prior_season_game], qb={prior_season_game: (20.0, -20.0)})
    qb, ratings, level = _early_families(
        games, [OTHER_GAME], {"ATL": 12.0, "NO": -12.0}
    )
    out = replace_early_columns(gold, games, qb, ratings, level)
    assert out.loc[0, "home_qb_adjustment"] == pytest.approx(9.9)
    assert out.loc[0, "away_qb_adjustment"] == pytest.approx(-9.0)
    assert out.loc[0, "qb_value_diff"] == pytest.approx(9.9 - (-9.0))
    kept = [
        c for c in gold.columns if c not in (*EARLY_REPLACED_COLUMNS, "qb_value_diff")
    ]
    gold_row = gold.loc[gold["game_id"] == OTHER_GAME].reset_index(drop=True)
    pd.testing.assert_frame_equal(out[kept], gold_row[kept])


def test_an_indoor_game_keeps_its_gold_wind_and_every_other_roof_is_unknown() -> None:
    ids = [OTHER_GAME, KC_GAME, NO_GOLD_GAME, THU_W13]
    games = _weeks_12_13_2020(
        {
            OTHER_GAME: "indoor",
            KC_GAME: "outdoor",
            NO_GOLD_GAME: "retractable",
            THU_W13: None,
        }
    )
    gold = _gold(ids)
    gold.loc[gold["game_id"] == OTHER_GAME, "wind_mph"] = 0.0
    out = replace_early_columns(gold, games, *_early_families(games, ids))
    wind = dict(zip(out["game_id"], out["wind_mph"], strict=True))
    assert wind[OTHER_GAME] == 0.0
    assert np.isnan(wind[KC_GAME]) and np.isnan(wind[NO_GOLD_GAME])
    assert np.isnan(wind[THU_W13])


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
    assert out.qb_values_clipped == 0
    for target in V2_TARGETS:
        assert out.rows[target]["game_id"].tolist() == [OTHER_GAME]
    for target in ("wp", "ats"):
        rows = out.rows[target]
        assert rows.loc[0, "qb_value_diff"] == pytest.approx(1.2 - (-0.5))
        # Both teams were rated at the early capture.
        assert not np.isnan(rows.loc[0, "mkt_rating_diff"])
    ou = out.rows["ou"]
    # Only the two Sunday 11-29 games had ended: (30 + 40) / 2. KC's Wednesday
    # game (60 points) and the Thursday game would move it.
    assert ou.loc[0, "league_points_level"] == pytest.approx(35.0)
    assert not np.isnan(ou.loc[0, "mkt_total_sum"])
    assert np.isnan(ou.loc[0, "wind_mph"])


def test_a_neutral_site_game_is_left_out_unless_it_is_a_super_bowl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Silver records the site a game was played at, not when it was announced, so a
    # relocation announced after the early capture (2022_W11_CLE@BUF, moved Thursday)
    # would reach the kept hfa_used and indoor wind. A Super Bowl's site is fixed years
    # ahead. A missing flag reads as not neutral.
    super_bowl = "2020_W21_KC@TB"
    games = pd.concat(
        [
            _weeks_12_13_2020(),
            _games([(super_bowl, "2021-02-07 18:30", 31.0, 9.0)]),
        ],
        ignore_index=True,
    )
    games["neutral_site"] = games["neutral_site"].astype("boolean")
    games.loc[games["game_id"] == OTHER_GAME, "neutral_site"] = True
    games.loc[games["game_id"] == THU_W13, "neutral_site"] = pd.NA
    games.loc[games["game_id"] == super_bowl, "neutral_site"] = True
    games.loc[games["game_id"] == super_bowl, "game_type"] = "SB"
    studied = [OTHER_GAME, THU_W13, super_bowl]
    gold_dir = _write_gold(tmp_path / "gold", dict.fromkeys(V2_TARGETS, studied))
    _QBStub().install(monkeypatch)
    early_times = pd.Series(
        {
            OTHER_GAME: et(EARLY_W13),
            THU_W13: et(EARLY_W13),
            super_bowl: et("2021-02-02 11:55"),
        }
    )

    out = build_early_inputs(gold_dir, games, _nflverse(games), early_times)

    reasons = dict(zip(out.excluded["game_id"], out.excluded["reason"], strict=True))
    assert reasons == {OTHER_GAME: SITE_NOT_KNOWN_AT_EARLY_CAPTURE}
    for target in V2_TARGETS:
        assert out.rows[target]["game_id"].tolist() == [THU_W13, super_bowl]


def _build_one_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    team_qb: Mapping[str, float] = TEAM_QB,
) -> EarlyInputs:
    """``build_early_inputs`` for OTHER_GAME alone, the QB tracker stubbed with *team_qb*."""
    games = _weeks_12_13_2020()
    gold_dir = _write_gold(
        tmp_path / "gold", {target: [OTHER_GAME] for target in V2_TARGETS}
    )
    _QBStub(team_qb).install(monkeypatch)
    early_times = pd.Series({OTHER_GAME: et(EARLY_W13)})
    return build_early_inputs(gold_dir, games, _nflverse(games), early_times)


def test_build_counts_the_early_qb_values_the_clip_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ATL (home) at 12.0 is above the 2020 gold max of 9.9; NO (away) at -0.5 is inside.
    out = _build_one_game(monkeypatch, tmp_path, {**TEAM_QB, "ATL": 12.0})
    # One early value moved, counted once although each of the three gold matrices clipped it.
    assert out.qb_values_clipped == 1
    for target in ("wp", "ats"):
        assert out.rows[target].loc[0, "qb_value_diff"] == pytest.approx(9.9 - (-0.5))


def test_build_rows_hold_only_the_game_keys_and_the_targets_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _build_one_game(monkeypatch, tmp_path)
    for target in V2_TARGETS:
        expected = ["game_id", "season", "week", *V2_INPUTS[target]]
        assert list(out.rows[target].columns) == expected
