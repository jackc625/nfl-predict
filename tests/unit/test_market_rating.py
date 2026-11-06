"""Market-implied ratings: recovery, neutral sites, the kickoff fence, relocation."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.market_rating import (
    fit_spread_ratings,
    fit_total_levels,
    market_rating_no_information_signature,
    market_ratings,
)

TRUTH = {"A": 3.0, "B": 1.0, "C": -1.0, "D": -3.0}
HFA = 2.0


def _fit_lines(neutral_every: int = 0) -> pd.DataFrame:
    rows = []
    for i, (home, away) in enumerate(itertools.permutations(TRUTH, 2)):
        neutral = bool(neutral_every) and i % neutral_every == 0
        spread = (0.0 if neutral else HFA) + TRUTH[home] - TRUTH[away]
        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "spread_line": spread,
                "total_line": 44.0 + TRUTH[home] * 0.1 + TRUTH[away] * 0.1,
                "neutral": neutral,
                "weight": 1.0,
            }
        )
    return pd.concat([pd.DataFrame(rows)] * 6, ignore_index=True)


def test_ratings_recover_known_strengths_and_home_field():
    ratings, hfa = fit_spread_ratings(_fit_lines())
    assert ratings["A"] - ratings["D"] == pytest.approx(6.0, abs=0.15)
    assert hfa == pytest.approx(2.0, abs=0.15)
    levels, mu = fit_total_levels(_fit_lines())
    assert mu == pytest.approx(44.0, abs=0.5)
    assert levels["A"] > levels["D"]


def test_neutral_site_games_carry_no_home_field():
    _, hfa = fit_spread_ratings(_fit_lines(neutral_every=3))
    assert hfa == pytest.approx(2.0, abs=0.15)


def _game(season, week, away, home, kickoff_utc, neutral=False):
    return {
        "game_id": f"{season}_W{week:02d}_{away}@{home}",
        "season": season,
        "week": week,
        "kickoff_et": pd.Timestamp(kickoff_utc, tz="UTC")
        if isinstance(kickoff_utc, str)
        else kickoff_utc,
        "home_team": home,
        "away_team": away,
        "neutral_site": neutral,
    }


def _nfl(season, week, away, home, spread, total=44.0):
    return {
        "season": season,
        "week": week,
        "away_team": away,
        "home_team": home,
        "spread_line": spread,
        "total_line": total,
    }


def test_only_lines_kicked_off_by_the_lock_are_read():
    target = _game(2024, 3, "BUF", "KC", "2024-09-22 17:00")
    lock = lock_rule.lock_frame(pd.DataFrame([target])).iloc[0]
    lock_utc = lock.tz_convert("UTC")
    early = _game(2024, 1, "BUF", "DEN", "2024-09-08 17:00")
    at_lock = _game(2024, 2, "KC", "DEN", lock_utc)
    later = _game(2024, 2, "BUF", "MIA", lock_utc + pd.Timedelta(seconds=1))
    games = pd.DataFrame([early, at_lock, later, target])

    def frame(later_spread, own_spread):
        return pd.DataFrame(
            [
                _nfl(2024, 1, "BUF", "DEN", 3.0),
                _nfl(2024, 2, "KC", "DEN", -2.0),
                _nfl(2024, 2, "BUF", "MIA", later_spread),
                _nfl(2024, 3, "BUF", "KC", own_spread),
            ]
        )

    base = market_ratings(games, frame(4.0, 3.0)).set_index("game_id")
    moved = market_ratings(games, frame(50.0, 50.0)).set_index("game_id")
    gid = target["game_id"]
    assert not np.isnan(base.loc[gid, "mkt_rating_diff"])
    assert base.loc[gid, "mkt_rating_diff"] == moved.loc[gid, "mkt_rating_diff"]
    assert base.loc[gid, "information_time"] <= lock
    assert base.loc[gid, "information_time"] == lock_utc


def test_relocated_team_keeps_its_prior_season_rating():
    games = pd.DataFrame(
        [
            _game(2016, 5, "KC", "LAC", "2016-10-09 17:00"),
            _game(2017, 1, "SEA", "LAC", "2017-09-10 17:00"),
        ]
    )
    nfl = pd.DataFrame(
        [_nfl(2016, 5, "KC", "SD", 3.0), _nfl(2017, 1, "SEA", "LAC", 1.0)]
    )
    out = market_ratings(games, nfl).set_index("game_id")
    assert not np.isnan(out.loc["2017_W01_SEA@LAC", "home_mkt_rating"])


def test_team_with_no_history_is_blank_with_coverage_zero():
    games = pd.DataFrame(
        [
            _game(2001, 4, "KC", "DEN", "2001-10-07 17:00"),
            _game(2002, 1, "DAL", "HOU", "2002-09-08 17:00"),
        ]
    )
    nfl = pd.DataFrame(
        [_nfl(2001, 4, "KC", "DEN", 3.0), _nfl(2002, 1, "DAL", "HOU", -3.0)]
    )
    out = market_ratings(games, nfl).set_index("game_id")
    gid = "2002_W01_DAL@HOU"
    assert np.isnan(out.loc[gid, "home_mkt_rating"])
    assert out.loc[gid, "mkt_rating_coverage"] == 0.0


def test_no_information_signature_shape():
    sig = market_rating_no_information_signature()
    assert sig["mkt_rating_coverage"] == 0.0
    assert sig["home_mkt_rating"] is None


def _early_week_slate():
    """Weeks 1-2 played, a Thursday game of week 3, and the Sunday target BUF@KC."""
    target = _game(2024, 3, "BUF", "KC", "2024-09-22 17:00")
    thursday = _game(2024, 3, "NYJ", "NE", "2024-09-20 00:15")  # Thu 20:15 ET
    games = pd.DataFrame(
        [
            _game(2024, 1, "BUF", "DEN", "2024-09-08 17:00"),
            _game(2024, 2, "KC", "DEN", "2024-09-15 17:00"),
            thursday,
            target,
        ]
    )
    nfl = pd.DataFrame(
        [
            _nfl(2024, 1, "BUF", "DEN", 3.0),
            _nfl(2024, 2, "KC", "DEN", -2.0),
            _nfl(2024, 3, "NYJ", "NE", -6.0),
            _nfl(2024, 3, "BUF", "KC", 3.0),
        ]
    )
    return games, nfl, target["game_id"], thursday["kickoff_et"]


def test_a_supplied_earlier_time_leaves_out_a_game_kicked_off_after_it():
    games, nfl, gid, thursday_kickoff_utc = _early_week_slate()
    early = pd.Timestamp("2024-09-17 11:55", tz="America/New_York")
    locks = lock_rule.lock_frame(games)
    locks[gid] = early
    default = market_ratings(games, nfl).set_index("game_id")
    moved = market_ratings(games, nfl, locks=locks).set_index("game_id")
    assert default.loc[gid, "information_time"] == thursday_kickoff_utc
    assert moved.loc[gid, "information_time"] <= early
    assert moved.loc[gid, "information_time"] < thursday_kickoff_utc
    assert moved.loc[gid, "mkt_rating_diff"] != default.loc[gid, "mkt_rating_diff"]


def test_the_default_locks_match_the_lock_rule_frame():
    games, nfl, _, _ = _early_week_slate()
    explicit = market_ratings(games, nfl, locks=lock_rule.lock_frame(games))
    pd.testing.assert_frame_equal(explicit, market_ratings(games, nfl))


def test_supplied_locks_must_name_every_game():
    games, nfl, gid, _ = _early_week_slate()
    locks = lock_rule.lock_frame(games)
    with pytest.raises(ValueError, match="2024_W03_BUF@KC"):
        market_ratings(games, nfl, locks=locks.drop(gid))


def test_supplied_locks_must_not_name_a_game_twice():
    games, nfl, gid, _ = _early_week_slate()
    locks = lock_rule.lock_frame(games)
    repeated = pd.concat([locks, locks.loc[[gid]]])
    with pytest.raises(ValueError, match=r"more than once.*2024_W03_BUF@KC"):
        market_ratings(games, nfl, locks=repeated)


def test_supplied_locks_must_give_every_game_an_instant():
    games, nfl, gid, _ = _early_week_slate()
    locks = lock_rule.lock_frame(games)
    locks[gid] = pd.NaT
    with pytest.raises(ValueError, match=r"no instant.*2024_W03_BUF@KC"):
        market_ratings(games, nfl, locks=locks)
