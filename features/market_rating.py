"""Market-implied team ratings built only from lines of games already kicked off.

For a target game G with lock L (``utils.game_lock``), a rating reads the point-spread and
total lines of games P with ``kickoff_P <= L`` in G's season and the season before. The
target's own line is never read: G kicks off after its own lock. In nflverse a game that
has not been played carries its CURRENT line, so the kickoff fence is what keeps a live
line out. Kickoffs come from silver ``games.kickoff_et`` (a true instant), never from
nflverse ``gameday`` / ``gametime``.

Team codes are canonicalised with ``utils.team_data.normalize_team_abbreviation``, which
already maps relocated franchises onto one code (OAK->LV, SD->LAC, STL->LA), so a
relocated team keeps its prior-season rating.

ASCII only, no emoji.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from utils.game_id_utils import create_standard_game_id
from utils.team_data import normalize_team_abbreviation

RATING_PRIOR_SEASON_WEIGHT = 0.35
RATING_RIDGE = 1.0

_MIN_SEASON = 2000
_OUTPUT_COLUMNS = [
    "game_id",
    "home_mkt_rating",
    "away_mkt_rating",
    "mkt_rating_diff",
    "mkt_total_sum",
    "mkt_rating_coverage",
    "information_time",
]


def _solve_ridge(
    design: np.ndarray, target: np.ndarray, weight: np.ndarray, ridge_columns: int
) -> np.ndarray:
    """Weighted least squares; the ridge touches only the LAST ``ridge_columns`` columns."""
    root_w = np.sqrt(weight)[:, None]
    n_cols = design.shape[1]
    penalty = np.zeros((ridge_columns, n_cols))
    penalty[:, n_cols - ridge_columns :] = np.sqrt(RATING_RIDGE) * np.eye(ridge_columns)
    stacked_x = np.vstack([design * root_w, penalty])
    stacked_y = np.concatenate([target * root_w[:, 0], np.zeros(ridge_columns)])
    solution, *_ = np.linalg.lstsq(stacked_x, stacked_y, rcond=None)
    return solution


def _team_index(lines: pd.DataFrame) -> dict[str, int]:
    teams = sorted(set(lines["home_team"]) | set(lines["away_team"]))
    return {team: i for i, team in enumerate(teams)}


def fit_spread_ratings(lines: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Fit ``spread_line = hfa * (1 - neutral) + r[home] - r[away]``.

    Returns ``(ratings, hfa)``. The ridge shrinks ratings toward 0, never hfa.
    """
    rows = lines.dropna(subset=["spread_line"])
    index = _team_index(rows)
    n = len(rows)
    design = np.zeros((n, 1 + len(index)))
    design[:, 0] = 1.0 - rows["neutral"].astype(float).to_numpy()
    arange = np.arange(n)
    design[arange, 1 + np.array([index[t] for t in rows["home_team"]], dtype=int)] = 1.0
    design[arange, 1 + np.array([index[t] for t in rows["away_team"]], dtype=int)] -= (
        1.0
    )
    solution = _solve_ridge(
        design,
        rows["spread_line"].to_numpy(dtype=float),
        rows["weight"].to_numpy(dtype=float),
        len(index),
    )
    return {t: float(solution[1 + i]) for t, i in index.items()}, float(solution[0])


def fit_total_levels(lines: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Fit ``total_line = mu + t[home] + t[away]``; returns ``(levels, mu)``."""
    rows = lines.dropna(subset=["total_line"])
    index = _team_index(rows)
    n = len(rows)
    design = np.zeros((n, 1 + len(index)))
    design[:, 0] = 1.0
    arange = np.arange(n)
    design[arange, 1 + np.array([index[t] for t in rows["home_team"]], dtype=int)] += (
        1.0
    )
    design[arange, 1 + np.array([index[t] for t in rows["away_team"]], dtype=int)] += (
        1.0
    )
    solution = _solve_ridge(
        design,
        rows["total_line"].to_numpy(dtype=float),
        rows["weight"].to_numpy(dtype=float),
        len(index),
    )
    return {t: float(solution[1 + i]) for t, i in index.items()}, float(solution[0])


def _silver_id(row: pd.Series) -> str | None:
    if int(row["season"]) < _MIN_SEASON:
        return None
    try:
        return create_standard_game_id(
            int(row["season"]),
            int(row["week"]),
            str(row["away_team"]),
            str(row["home_team"]),
        )
    except ValueError:
        return None


def _line_pool(games: pd.DataFrame, nflverse: pd.DataFrame) -> pd.DataFrame:
    """Every nflverse line joined to its silver kickoff, with canonical team codes."""
    schedule = nflverse.loc[
        :, ["season", "week", "home_team", "away_team", "spread_line", "total_line"]
    ].copy()
    schedule["game_id"] = schedule.apply(_silver_id, axis=1)
    schedule = schedule.dropna(subset=["game_id"]).drop(
        columns=["season", "week", "home_team", "away_team"]
    )
    silver = games.loc[
        :, ["game_id", "season", "kickoff_et", "home_team", "away_team", "neutral_site"]
    ].rename(columns={"neutral_site": "neutral"})
    pool = silver.merge(schedule, on="game_id", how="inner")
    pool = pool.dropna(subset=["kickoff_et"])
    pool["kickoff_et"] = pd.to_datetime(pool["kickoff_et"], utc=True)
    pool["neutral"] = pool["neutral"].fillna(False).astype(bool)
    pool["home_team"] = pool["home_team"].map(normalize_team_abbreviation)
    pool["away_team"] = pool["away_team"].map(normalize_team_abbreviation)
    return pool.reset_index(drop=True)


def _solve_for_lock(
    pool: pd.DataFrame, season: int, lock: pd.Timestamp
) -> tuple[dict[str, float], dict[str, float], float, Any]:
    """Ratings, total levels, mu and the latest kickoff read, for one (season, lock)."""
    in_window = pool["season"].isin([season - 1, season]) & (pool["kickoff_et"] <= lock)
    window = pd.DataFrame(pool.loc[in_window]).copy()
    if window.empty:
        return {}, {}, float("nan"), pd.NaT
    window["weight"] = np.where(
        window["season"] == season, 1.0, RATING_PRIOR_SEASON_WEIGHT
    )
    ratings, _ = fit_spread_ratings(window)
    levels, mu = fit_total_levels(window)
    return ratings, levels, mu, window["kickoff_et"].max()


def market_ratings(games: pd.DataFrame, nflverse: pd.DataFrame) -> pd.DataFrame:
    """One row per silver game: ratings from lines of games kicked off by its lock."""
    pool = _line_pool(games, nflverse)
    locks = lock_rule.lock_frame(games)
    targets = games.loc[:, ["game_id", "season", "home_team", "away_team"]].copy()
    targets["lock"] = pd.to_datetime(targets["game_id"].map(locks), utc=True)
    targets["home_team"] = targets["home_team"].map(normalize_team_abbreviation)
    targets["away_team"] = targets["away_team"].map(normalize_team_abbreviation)

    records: list[dict[str, object]] = []
    for (season, lock), group in targets.groupby(["season", "lock"], sort=False):
        ratings, levels, mu, info_time = _solve_for_lock(pool, int(season), lock)
        for row in group.itertuples(index=False):
            home_r = ratings.get(row.home_team, np.nan)
            away_r = ratings.get(row.away_team, np.nan)
            both = not (np.isnan(home_r) or np.isnan(away_r))
            home_t = levels.get(row.home_team, np.nan)
            away_t = levels.get(row.away_team, np.nan)
            records.append(
                {
                    "game_id": row.game_id,
                    "home_mkt_rating": home_r,
                    "away_mkt_rating": away_r,
                    "mkt_rating_diff": home_r - away_r,
                    "mkt_total_sum": mu + home_t + away_t,
                    "mkt_rating_coverage": 1.0 if both else 0.0,
                    "information_time": info_time,
                }
            )
    out = pd.DataFrame.from_records(records, columns=_OUTPUT_COLUMNS)
    return out.set_index("game_id").loc[games["game_id"]].reset_index()


def market_rating_no_information_signature() -> dict[str, float | None]:
    """Column values of a game with no rating information (NaN columns are None)."""
    return {
        "home_mkt_rating": None,
        "away_mkt_rating": None,
        "mkt_rating_diff": None,
        "mkt_total_sum": None,
        "mkt_rating_coverage": 0.0,
    }
