"""FeatureBuilder Protocol definition.

Defines the structural type contract that all feature builders must satisfy.
The key requirement is the `as_of_datetime` parameter, which serves as the
time-fence: builders MUST NOT use any data with timestamps after this cutoff.

Uses typing.Protocol for structural subtyping -- existing classes satisfy
the protocol if they have matching method signatures, without needing to
change their inheritance hierarchy. Pyright catches violations at static
analysis time.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureBuilder(Protocol):
    """Protocol that all feature builders must satisfy.

    The as_of_datetime parameter is the time-fence: builders MUST NOT
    use any data with timestamps after this cutoff. This enforces
    temporal correctness and prevents data leakage.
    """

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build features using only data available before as_of_datetime.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.
            target_season: Optional season to filter features for.
            target_week: Optional week to filter features for.

        Returns:
            DataFrame with computed features, one row per team per game.
        """
        ...

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get features for a single game using data before as_of_datetime.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        ...
