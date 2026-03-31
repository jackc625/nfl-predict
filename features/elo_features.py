"""Elo Feature Builder

This module creates Elo-based features for game prediction models by
looking up pre-computed per-game Elo snapshots from the silver layer.

The snapshots are produced by scripts/build_elo.py which processes all
games chronologically, recording each team's pre-game Elo BEFORE the
game result is applied. This eliminates batch leakage where end-of-season
ratings were previously assigned to every game.

Conforms to the FeatureBuilder Protocol with mandatory as_of_datetime
parameter for time-fence enforcement.
"""

from datetime import datetime

import pandas as pd

from data.storage import load_dataframe
from features.protocol import FeatureBuilder  # noqa: F401 (documents conformance)
from utils import get_logger

logger = get_logger(__name__)

# Elo feature column names -- unchanged from v1.0 (per D-17)
ELO_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_prob_home",
    "elo_prob_away",
    "hfa_used",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
]


class EloFeatureBuilder:
    """Build Elo-based features from pre-computed snapshots.

    Satisfies the FeatureBuilder Protocol via structural subtyping.
    Features are derived from per-game Elo snapshots stored in the silver
    layer (elo_game_snapshots), NOT recomputed on the fly.

    The as_of_datetime parameter is accepted for protocol conformance but
    filtering is handled by the snapshot join -- each game's snapshot already
    contains only pre-game information by construction.
    """

    def __init__(self) -> None:
        """Initialize Elo feature builder with empty snapshot cache."""
        self._snapshots_df: pd.DataFrame | None = None

    def _load_snapshots(self) -> pd.DataFrame:
        """Load per-game Elo snapshots from silver layer.

        Loads elo_game_snapshots produced by scripts/build_elo.py and
        caches the result for repeated calls within the same session.

        Returns:
            DataFrame with per-game pre-game Elo snapshots.

        Raises:
            ValueError: If snapshots are not found in the silver layer.
        """
        if self._snapshots_df is not None:
            return self._snapshots_df

        logger.info("Loading Elo game snapshots from silver layer")
        self._snapshots_df = load_dataframe("elo_game_snapshots", layer="silver")
        logger.info(
            "Loaded Elo snapshots",
            total_snapshots=len(self._snapshots_df),
            seasons=sorted(self._snapshots_df["season"].unique().tolist())
            if len(self._snapshots_df) > 0
            else [],
        )
        return self._snapshots_df

    # ---- Protocol-conforming methods ----

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build Elo features by joining games with pre-computed snapshots.

        Looks up per-game Elo snapshots from the silver layer and joins
        them to the input games DataFrame. Each snapshot contains the
        pre-game Elo ratings (computed from all prior games only).

        Args:
            games_df: DataFrame with game records (must have game_id column).
            as_of_datetime: Time-fence cutoff (accepted for protocol
                conformance; filtering is inherent in snapshot construction).
            target_season: Optional filter to a specific season.
            target_week: Optional filter to a specific week.

        Returns:
            DataFrame with Elo feature columns added.
        """
        logger.info(
            "Building Elo features from snapshots",
            games=len(games_df),
            as_of_datetime=str(as_of_datetime),
        )

        # Apply optional filters
        filtered_df = games_df.copy()
        if target_season is not None:
            filtered_df = filtered_df[filtered_df["season"] == target_season]
        if target_week is not None:
            filtered_df = filtered_df[filtered_df["week"] == target_week]

        # Load pre-computed snapshots
        snapshots = self._load_snapshots()

        # Select only the columns we need from snapshots for the join
        snapshot_cols = [
            "game_id",
            "home_elo_pre",
            "away_elo_pre",
            "home_elo_uncertainty",
            "away_elo_uncertainty",
            "elo_prob_home",
            "hfa_used",
        ]
        # Only keep columns that exist in the snapshots
        available_cols = [c for c in snapshot_cols if c in snapshots.columns]
        snapshot_subset = snapshots[available_cols]

        # Merge snapshots onto games by game_id (left join to keep all games)
        merged = filtered_df.merge(snapshot_subset, on="game_id", how="left")

        # Rename snapshot columns to feature names
        merged = merged.rename(columns={
            "home_elo_pre": "home_elo",
            "away_elo_pre": "away_elo",
        })

        # Compute derived columns
        merged["elo_diff"] = merged["home_elo"] - merged["away_elo"]
        merged["elo_prob_away"] = 1.0 - merged["elo_prob_home"]

        # Log any games without snapshots (future games or missing data)
        missing_count = merged["home_elo"].isna().sum()
        if missing_count > 0:
            logger.warning(
                "Games without Elo snapshots (future games or missing data)",
                missing_count=missing_count,
                total_games=len(merged),
            )

        logger.info("Built Elo features from snapshots", games=len(merged))
        return merged

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get Elo features for a single game from pre-computed snapshots.

        Looks up the game's pre-game Elo snapshot from the silver layer.

        Args:
            game_id: Unique game identifier (e.g., '2024_01_BUF_MIA').
            as_of_datetime: Time-fence cutoff (accepted for protocol
                conformance).

        Returns:
            Dictionary mapping feature names to values.

        Raises:
            ValueError: If game_id is not found in snapshots.
        """
        snapshots = self._load_snapshots()
        game_snap = snapshots[snapshots["game_id"] == game_id]

        if len(game_snap) == 0:
            raise ValueError(
                f"No Elo snapshot found for game_id={game_id}. "
                "Ensure build_elo.py has been run with --all-seasons."
            )

        snap = game_snap.iloc[0]
        return {
            "home_elo": float(snap["home_elo_pre"]),
            "away_elo": float(snap["away_elo_pre"]),
            "elo_diff": float(snap["home_elo_pre"] - snap["away_elo_pre"]),
            "elo_prob_home": float(snap["elo_prob_home"]),
            "elo_prob_away": float(1.0 - snap["elo_prob_home"]),
            "hfa_used": float(snap["hfa_used"]),
            "home_elo_uncertainty": float(snap["home_elo_uncertainty"]),
            "away_elo_uncertainty": float(snap["away_elo_uncertainty"]),
        }
