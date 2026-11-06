"""v2 QB tracker: an unknown quarterback is valued at replacement level, not league average.

The live ``QBTracker`` gives a team whose starter is unknown the league-average 0.0. A
team with no identifiable quarterback is, in practice, starting a below-average backup,
so the v2 tracker values that case at ``REPLACEMENT_LEVEL_QB_QUALITY`` instead. Only the
two no-information paths change; a known quarterback keeps his rolling quality.
"""

from collections.abc import Mapping

import pandas as pd

from features.injury import REPLACEMENT_LEVEL_QB_QUALITY
from features.qb_tracking import QBTracker


class QBTrackerV2(QBTracker):
    """``QBTracker`` with a replacement-level value for unknown quarterbacks."""

    @staticmethod
    def _quality_of(rolling: pd.DataFrame, qb_id: str | None) -> float:
        """The QB's rolling quality, or replacement level with no id or no history."""
        if qb_id is None or len(rolling) == 0:
            return REPLACEMENT_LEVEL_QB_QUALITY
        match = rolling.loc[rolling["passer_player_id"] == qb_id]
        if len(match) == 0:
            return REPLACEMENT_LEVEL_QB_QUALITY
        return float(match.iloc[0]["qb_quality"])

    def no_information_signature(self) -> Mapping[str, float | None]:
        """A game with no dated input is valued at replacement level, not league average."""
        return {"qb_adjustment": REPLACEMENT_LEVEL_QB_QUALITY}
