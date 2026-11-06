"""League-wide scoring level built only from games that had ended by each target's lock.

For a target game G with lock L (``utils.game_lock``), the level reads the combined points
(home + away) of every scored game whose END (kickoff + ``DECLARED_GAME_DURATION``) is at
or before L. Current-season games contribute their points; the prior season contributes
its mean, which acts as a shrinkage prior worth ``LEVEL_PRIOR_GAMES`` games:

    level = (sum(points) + LEVEL_PRIOR_GAMES * prior_mean) / (n + LEVEL_PRIOR_GAMES)

With no previous season the prior drops out and the level is the current-season mean
(NaN when no current-season game has ended). The target game is never read: it cannot end
before its own lock. Kickoffs come from silver ``games.kickoff_et`` as true instants and
are compared in UTC; a game with no kickoff has no lock and is left out of the output.

ASCII only, no emoji.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

import utils.game_lock as lock_rule
from features.provenance import DECLARED_GAME_DURATION

LEVEL_PRIOR_GAMES = 32.0

_OUTPUT_COLUMNS = ["game_id", "league_points_level", "information_time"]


def _scored_games_with_end(games: pd.DataFrame) -> pd.DataFrame:
    """Scored games with their UTC end instant and combined points.

    Unplayed games (null scores) are dropped here, so they are never read as inputs.
    """
    played = games.dropna(subset=["home_score", "away_score"])
    end = pd.to_datetime(played["kickoff_et"], utc=True) + DECLARED_GAME_DURATION
    return pd.DataFrame(
        {
            "season": played["season"].astype(int).to_numpy(),
            "end": end.to_numpy(),
            "points": (played["home_score"] + played["away_score"]).to_numpy(
                dtype=float
            ),
        }
    )


def _level_for_lock(
    pool: pd.DataFrame, season: int, lock_utc: datetime
) -> tuple[float, pd.Timestamp | None]:
    """League level and latest end read (None when nothing was read)."""
    ended = pool.loc[pool["end"] <= lock_utc]
    current = ended.loc[ended["season"] == season]
    prior = ended.loc[ended["season"] == season - 1]
    # The target itself never reaches this point: its end is after its own lock.
    read = pd.concat([current, prior])
    info_time = None if read.empty else max(read["end"])

    n_current = len(current)
    if prior.empty:
        if n_current == 0:
            return float("nan"), info_time
        return float(current["points"].mean()), info_time

    prior_mean = float(prior["points"].mean())
    current_total = float(current["points"].sum())
    level = (current_total + LEVEL_PRIOR_GAMES * prior_mean) / (
        n_current + LEVEL_PRIOR_GAMES
    )
    return level, info_time


def league_points_level(games: pd.DataFrame) -> pd.DataFrame:
    """One row per silver game with a kickoff: the league level its lock admits.

    Args:
        games: Silver ``games`` rows with ``game_id, season, kickoff_et, home_score,
            away_score``. ``kickoff_et`` must be timezone-aware.

    Returns:
        A frame with ``game_id, league_points_level, information_time``. The level is NaN
        when no game is admissible; ``information_time`` is the latest game end read
        (NaT when nothing was read).
    """
    with_kickoff = games.loc[games["kickoff_et"].notna()].reset_index(drop=True)
    locks_et = lock_rule.lock_frame(with_kickoff)
    locks_utc = locks_et.dt.tz_convert("UTC").dt.to_pydatetime()
    pool = _scored_games_with_end(with_kickoff)

    records: list[dict[str, object]] = []
    targets = zip(with_kickoff.itertuples(index=False), locks_utc, strict=True)
    for row, lock_utc in targets:
        level, info_time = _level_for_lock(pool, int(row.season), lock_utc)
        records.append(
            {
                "game_id": row.game_id,
                "league_points_level": level,
                "information_time": info_time,
            }
        )
    out = pd.DataFrame.from_records(records, columns=_OUTPUT_COLUMNS)
    out["league_points_level"] = out["league_points_level"].astype(float)
    out["information_time"] = pd.to_datetime(out["information_time"], utc=True)
    return out
