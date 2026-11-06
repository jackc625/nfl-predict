"""Early-week QB tracker: an untimed depth chart is never read.

The 2002-2024 depth charts are keyed by week and carry no publication time, so nothing
proves a chart was filed before an early lock (a Tuesday line capture). ``QBTrackerEarly``
therefore skips them: ``_game_inputs`` then falls back to the team's latest primary passer
among games that ended by the lock it was handed. The caller (Study 1b) supplies the early
lock frame through ``build_features(..., lock_frame=...)``; this class only decides which
depth charts may be read at it. A 2025+ chart carries a real ``dt`` and is read at or
before the lock as usual.
"""

from typing import Any

import pandas as pd

from features.qb_tracking import DepthChartQBs
from features.qb_v2 import QBTrackerV2


class QBTrackerEarly(QBTrackerV2):
    """``QBTrackerV2`` that treats a week-keyed (untimed) depth chart as unknown."""

    def resolve_depth_chart_qbs(
        self,
        depth_charts: pd.DataFrame,
        season: int,
        week: int,
        team: str,
        lock: Any,
    ) -> DepthChartQBs:
        """A team's QB1 / QB2 from a TIMED chart only; a week-keyed chart names nobody.

        A non-empty frame with a ``week`` column and no ``dt`` column is the 2002-2024
        schema, which has no publication time: it returns ``DepthChartQBs(None, None,
        None)``. Every other frame goes to the parent, so a ``dt`` chart is read at or
        before the lock and a frame with neither column still raises
        ``DepthChartSchemaError``.
        """
        if (
            depth_charts is not None
            and len(depth_charts) > 0
            and "week" in depth_charts.columns
            and "dt" not in depth_charts.columns
        ):
            return DepthChartQBs(None, None, None)
        return super().resolve_depth_chart_qbs(depth_charts, season, week, team, lock)
