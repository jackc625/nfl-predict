"""Windowed opponent-adjusted EPA with a regressed prior (model v2).

The parent :class:`features.opponent_adj.OpponentAdjuster` weights a team's WHOLE history
``[1, 2, ..., N]``, so a week-1 value is dominated by seasons-old games. This subclass
overrides only the rolling step with a windowed rule and keeps the parent's inputs and output
schema column for column, so a gold builder can swap it in unchanged.

The rule, per (team, side, target) and per metric:

* Current-season adjusted games that ENDED at or before the target's lock get weights
  ``DECAY ** k`` (``k = 0`` for the most recent).
* The prior is ``PRIOR_REGRESSION`` times the team's previous-season value (the same decayed
  mean over all of last season's adjusted games).
* ``value = (sum(w * x) + PRIOR_GAMES * prior) / (sum(w) + PRIOR_GAMES)``. With no current
  season game the value is the prior; with neither it is NaN and the coverage flag is 0.0.
* Seasons older than the previous one are never read.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from features.opponent_adj import (
    _EPA_METRICS,
    _ROLLING_NAMES,
    OPP_ADJ_COVERAGE_COLUMN,
    OpponentAdjuster,
    _from_utc_ns,
    _utc_ns,
)

DECAY: float = 0.9
PRIOR_GAMES: float = 3.0
PRIOR_REGRESSION: float = 0.5


def _decayed_weights(count: int) -> np.ndarray:
    """``DECAY ** k`` for *count* chronologically ordered games (newest has ``k = 0``)."""
    return DECAY ** np.arange(count - 1, -1, -1, dtype=float)


class WindowedOpponentAdjuster(OpponentAdjuster):
    """:class:`OpponentAdjuster` whose rolling feature is a decayed, prior-regressed mean."""

    def _rolling_for_targets(
        self, adjusted: pd.DataFrame, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """The rolling rows for each target team-game (the parent's output schema).

        ``games_used`` counts the games of the target's season and the previous season that
        ENDED at or before the lock (adjusted or not). A target with none of them gets no
        row; one with games but no adjusted game gets NaN values, flag 0.0 and a NaT
        ``latest_information_end``.
        """
        rows: list[dict[str, Any]] = []
        for key, group in adjusted.groupby(["team", "side"], sort=True):
            team, side = cast(tuple[str, str], key)
            team_targets = targets.loc[targets["team"] == team]
            if len(team_targets) == 0:
                continue
            history = cast(pd.DataFrame, group).sort_values(["_end", "season", "week"])
            ends_ns = _utc_ns(cast(pd.Series, history["_end"]))
            seasons = history["season"].to_numpy(dtype=int)
            mask = history["adjusted"].to_numpy(dtype=bool)
            values = {
                metric: history[f"opp_adj_{metric}"].to_numpy(dtype=float)
                for metric in _EPA_METRICS
            }
            info_ns = _utc_ns(
                cast(pd.Series, history["information_end"]).fillna(
                    pd.Timestamp(0, tz="UTC")
                )
            )
            lock_ns = _utc_ns(team_targets["lock"])
            for target, lock in zip(
                team_targets.to_dict("records"), lock_ns, strict=True
            ):
                season = int(target["season"])
                ended = ends_ns <= lock
                current = ended & (seasons == season)
                previous = ended & (seasons == season - 1)
                games_used = int((current | previous).sum())
                if games_used == 0:
                    continue
                row: dict[str, Any] = {
                    "team": team,
                    "side": side,
                    "target_season": season,
                    "target_week": int(target["week"]),
                    "games_used": games_used,
                }
                current_read = np.flatnonzero(current & mask)
                previous_read = np.flatnonzero(previous & mask)
                covered = len(current_read) + len(previous_read) > 0
                for metric in _EPA_METRICS:
                    row[_ROLLING_NAMES[metric]] = (
                        self._blend(
                            values[metric][current_read], values[metric][previous_read]
                        )
                        if covered
                        else np.nan
                    )
                row[OPP_ADJ_COVERAGE_COLUMN] = 1.0 if covered else 0.0
                row["latest_information_end"] = (
                    _from_utc_ns(
                        int(
                            info_ns[np.concatenate([current_read, previous_read])].max()
                        )
                    )
                    if covered
                    else pd.NaT
                )
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _blend(current: np.ndarray, previous: np.ndarray) -> float:
        """One metric's value from chronologically ordered current and previous values."""
        prior = 0.0
        if len(previous) > 0:
            weights = _decayed_weights(len(previous))
            prior = PRIOR_REGRESSION * float(np.sum(weights * previous) / weights.sum())
        weights = _decayed_weights(len(current))
        numerator = float(np.sum(weights * current)) + PRIOR_GAMES * prior
        return numerator / (float(weights.sum()) + PRIOR_GAMES)
