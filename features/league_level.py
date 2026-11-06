"""League-wide scoring level built only from games that had ended by each target's lock.

For a target game G with lock L (``utils.game_lock``), the level reads the combined points
(home + away) of every scored game whose END (kickoff + ``DECLARED_GAME_DURATION``) is at
or before L. Current-season games contribute their points; the prior season contributes
its mean, which acts as a shrinkage prior worth ``LEVEL_PRIOR_GAMES`` games:

    level = (sum(points) + LEVEL_PRIOR_GAMES * prior_mean) / (n + LEVEL_PRIOR_GAMES)

The prior-season mean is fenced by the lock too: it averages only the prior-season games
that had ended by L.

With no previous season the prior drops out and the level is the current-season mean
(NaN when no current-season game has ended). The target game is never read: it cannot end
before its own lock. Kickoffs come from silver ``games.kickoff_et`` as true instants and
are compared in UTC; a game with no kickoff has no lock and is left out of the output.

The scored games are sorted by end once, and each season keeps cumulative point sums, so
each target costs one binary search rather than a scan of the whole pool.

ASCII only, no emoji.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from features.provenance import DECLARED_GAME_DURATION

LEVEL_PRIOR_GAMES = 32.0

_OUTPUT_COLUMNS = ["game_id", "league_points_level", "information_time"]


@dataclass(frozen=True)
class _SeasonEnds:
    """One season's scored games: sorted UTC end instants and cumulative points.

    ``cum_points[k]`` is the points of the first ``k`` games by end, so the sum of the
    games that ended at or before a lock is one lookup after a binary search.
    """

    ends: pd.DatetimeIndex
    cum_points: np.ndarray

    def read(
        self, locks: pd.DatetimeIndex
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Count, points sum and latest end of the games ended at or before each lock.

        ``side="right"`` makes a game ending exactly at the lock count (at-lock is
        admissible). The latest end is NaT where no game was read.
        """
        counts = self.ends.searchsorted(locks, side="right")
        totals = self.cum_points[counts]
        if len(self.ends) == 0:
            no_end = pd.DatetimeIndex([pd.NaT] * len(locks), tz="UTC")
            return counts.astype(float), totals, no_end
        last_end = pd.DatetimeIndex(self.ends[np.maximum(counts - 1, 0)])
        return (
            counts.astype(float),
            totals,
            pd.DatetimeIndex(last_end.where(counts > 0)),
        )


_EMPTY_SEASON = _SeasonEnds(
    ends=pd.DatetimeIndex([], tz="UTC"), cum_points=np.zeros(1, dtype=float)
)


def _scored_games_with_end(games: pd.DataFrame) -> pd.DataFrame:
    """Scored games with their UTC end instant and combined points, sorted by end.

    Unplayed games (null scores) are dropped here, so they are never read as inputs.
    """
    played = games.dropna(subset=["home_score", "away_score"])
    end = pd.to_datetime(played["kickoff_et"], utc=True) + DECLARED_GAME_DURATION
    pool = pd.DataFrame(
        {
            "season": played["season"].astype(int).to_numpy(),
            "end": end.to_numpy(),
            "points": (played["home_score"] + played["away_score"]).to_numpy(
                dtype=float
            ),
        }
    )
    return pool.sort_values("end", kind="stable").reset_index(drop=True)


def _index_by_season(pool: pd.DataFrame) -> dict[int, _SeasonEnds]:
    """One ``_SeasonEnds`` per season present in the (end-sorted) pool."""
    indexes: dict[int, _SeasonEnds] = {}
    for season, group in pool.groupby("season", sort=False):
        ends = pd.DatetimeIndex(group["end"])
        cum_points = np.concatenate([[0.0], np.cumsum(group["points"].to_numpy())])
        indexes[int(str(season))] = _SeasonEnds(ends=ends, cum_points=cum_points)
    return indexes


def _season_levels(
    indexes: dict[int, _SeasonEnds], season: int, locks: pd.DatetimeIndex
) -> pd.DataFrame:
    """League level and latest end read for every target of one season."""
    cur_n, cur_sum, cur_last = indexes.get(season, _EMPTY_SEASON).read(locks)
    prior_n, prior_sum, prior_last = indexes.get(season - 1, _EMPTY_SEASON).read(locks)

    has_prior = prior_n > 0
    prior_mean = np.divide(
        prior_sum, prior_n, out=np.zeros_like(prior_sum), where=has_prior
    )
    blended = (cur_sum + LEVEL_PRIOR_GAMES * prior_mean) / (cur_n + LEVEL_PRIOR_GAMES)
    current_mean = np.divide(
        cur_sum, cur_n, out=np.full_like(cur_sum, np.nan), where=cur_n > 0
    )
    level = np.where(has_prior, blended, current_mean)
    info_time = pd.DataFrame({"current": cur_last, "prior": prior_last}).max(axis=1)
    return pd.DataFrame(
        {"league_points_level": level, "information_time": info_time.to_numpy()}
    )


def league_points_level(
    games: pd.DataFrame, *, locks: pd.Series | None = None
) -> pd.DataFrame:
    """One row per silver game with a kickoff: the league level its lock admits.

    Args:
        games: Silver ``games`` rows with ``game_id, season, kickoff_et, home_score,
            away_score``. ``kickoff_et`` must be timezone-aware.
        locks: Optional ``game_id``-indexed series of tz-aware instants replacing each
            game's lock (for example an early line capture). It must name every game with
            a kickoff, else ``ValueError`` names the first ten missing ids. ``None`` uses
            the day-before lock rule.

    Returns:
        A frame with ``game_id, league_points_level, information_time``. The level is NaN
        when no game is admissible; ``information_time`` is the latest game end read
        (NaT when nothing was read).
    """
    with_kickoff = games.loc[games["kickoff_et"].notna()].reset_index(drop=True)
    if locks is None:
        game_locks = lock_rule.lock_frame(with_kickoff)
    else:
        missing = [g for g in with_kickoff["game_id"] if g not in locks.index]
        if missing:
            raise ValueError(
                f"locks must name every game with a kickoff; {len(missing)} missing, "
                f"first ten: {missing[:10]}"
            )
        game_locks = locks.loc[with_kickoff["game_id"]]
    lock_index = pd.DatetimeIndex(game_locks).tz_convert("UTC")
    indexes = _index_by_season(_scored_games_with_end(with_kickoff))
    seasons = with_kickoff["season"].astype(int).to_numpy()

    pieces = []
    for season in np.unique(seasons):
        positions = np.flatnonzero(seasons == season)
        piece = _season_levels(indexes, int(season), lock_index[positions])
        piece.index = positions
        pieces.append(piece)
    levels = pd.concat(pieces).sort_index()

    out = pd.DataFrame(
        {
            "game_id": with_kickoff["game_id"].to_numpy(),
            "league_points_level": levels["league_points_level"].to_numpy(dtype=float),
            "information_time": pd.to_datetime(levels["information_time"], utc=True),
        }
    )
    return out.loc[:, _OUTPUT_COLUMNS]
