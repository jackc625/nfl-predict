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

import numpy as np
import pandas as pd

from data.storage import load_dataframe
from features.protocol import FeatureBuilder  # noqa: F401 (documents conformance)
from utils import get_logger

logger = get_logger(__name__)

# Elo feature column names -- extended with momentum, rank, percentile
ELO_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_prob_home",
    "elo_prob_away",
    "hfa_used",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
    "home_elo_momentum",
    "away_elo_momentum",
    "home_elo_rank",
    "away_elo_rank",
    "home_elo_percentile",
    "away_elo_percentile",
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

    # ---- Derived feature methods ----

    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        snapshots: pd.DataFrame,
        lookback: int = 4,
    ) -> pd.DataFrame:
        """Add Elo momentum features for home and away teams.

        Momentum measures the Elo change per game over a rolling window.
        For each team, it looks at the last `lookback` games (or fewer if
        the team has played fewer games in the season) and computes:
            momentum = (newest_elo - oldest_elo) / window_size

        Returns NaN for a team's first game of the season (no prior games).

        Args:
            df: DataFrame with games (must have game_id, season, week,
                home_team, away_team columns and Elo columns from snapshot merge).
            snapshots: Full elo_game_snapshots DataFrame.
            lookback: Number of prior games for momentum window (default 4).

        Returns:
            DataFrame with home_elo_momentum and away_elo_momentum added.
        """
        home_momentum = []
        away_momentum = []

        for _, game in df.iterrows():
            season = game["season"]
            week = game["week"]

            for team, elo_list in [
                (game["home_team"], home_momentum),
                (game["away_team"], away_momentum),
            ]:
                # Find all prior games for this team in this season
                team_home = snapshots[
                    (snapshots["season"] == season)
                    & (snapshots["week"] < week)
                    & (snapshots["home_team"] == team)
                ][["week", "home_elo_pre"]].rename(
                    columns={"home_elo_pre": "team_elo"}
                )
                team_away = snapshots[
                    (snapshots["season"] == season)
                    & (snapshots["week"] < week)
                    & (snapshots["away_team"] == team)
                ][["week", "away_elo_pre"]].rename(
                    columns={"away_elo_pre": "team_elo"}
                )
                prior_games = pd.concat(
                    [team_home, team_away], ignore_index=True
                ).sort_values("week")

                if len(prior_games) == 0:
                    elo_list.append(np.nan)
                else:
                    window = prior_games.tail(lookback)
                    oldest_elo = window.iloc[0]["team_elo"]
                    newest_elo = window.iloc[-1]["team_elo"]
                    momentum = (newest_elo - oldest_elo) / len(window)
                    elo_list.append(momentum)

        df = df.copy()
        df["home_elo_momentum"] = home_momentum
        df["away_elo_momentum"] = away_momentum
        return df

    def _add_rank_features(
        self,
        df: pd.DataFrame,
        snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add Elo rank and percentile features for home and away teams.

        For each game, determines the latest pre-game Elo for all teams
        as of that game's week, then ranks them 1-32 (1 = highest Elo).
        Percentile = (32 - rank + 1) / 32, so rank 1 = 1.0, rank 32 ~= 0.03.

        Args:
            df: DataFrame with games (must have game_id, season, week,
                home_team, away_team columns).
            snapshots: Full elo_game_snapshots DataFrame.

        Returns:
            DataFrame with home_elo_rank, away_elo_rank,
            home_elo_percentile, away_elo_percentile added.
        """
        home_ranks = []
        away_ranks = []
        home_pcts = []
        away_pcts = []

        # Cache rankings by (season, week) to avoid recomputation
        rank_cache: dict[tuple[int, int], dict[str, int]] = {}

        for _, game in df.iterrows():
            season = game["season"]
            week = game["week"]
            cache_key = (season, week)

            if cache_key not in rank_cache:
                # Build a mapping of team -> latest pre-game Elo as of this week
                season_snaps = snapshots[snapshots["season"] == season]
                # Include current week's games (pre-game Elo is captured BEFORE
                # the game, so week N's snapshot is valid for ranking at week N)
                week_snaps = season_snaps[season_snaps["week"] <= week]

                team_elos: dict[str, float] = {}

                # Process home teams
                for _, snap in week_snaps.iterrows():
                    home_t = snap["home_team"]
                    away_t = snap["away_team"]
                    snap_week = snap["week"]

                    # Keep the latest week's Elo for each team
                    if home_t not in team_elos or snap_week >= team_elos.get(
                        f"_week_{home_t}", -1
                    ):
                        team_elos[home_t] = snap["home_elo_pre"]
                        team_elos[f"_week_{home_t}"] = snap_week

                    if away_t not in team_elos or snap_week >= team_elos.get(
                        f"_week_{away_t}", -1
                    ):
                        team_elos[away_t] = snap["away_elo_pre"]
                        team_elos[f"_week_{away_t}"] = snap_week

                # Remove internal tracking keys
                clean_elos = {
                    k: v for k, v in team_elos.items() if not k.startswith("_week_")
                }

                # Rank: 1 = highest Elo
                sorted_teams = sorted(
                    clean_elos.items(), key=lambda x: x[1], reverse=True
                )
                n_teams = len(sorted_teams)
                team_rank = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
                rank_cache[cache_key] = team_rank

            team_rank = rank_cache[cache_key]
            n_teams = len(team_rank)

            home_r = team_rank.get(game["home_team"], n_teams)
            away_r = team_rank.get(game["away_team"], n_teams)

            home_ranks.append(home_r)
            away_ranks.append(away_r)
            home_pcts.append((n_teams - home_r + 1) / n_teams)
            away_pcts.append((n_teams - away_r + 1) / n_teams)

        df = df.copy()
        df["home_elo_rank"] = home_ranks
        df["away_elo_rank"] = away_ranks
        df["home_elo_percentile"] = home_pcts
        df["away_elo_percentile"] = away_pcts
        return df

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

        # Add derived features: momentum and rank/percentile
        merged = self._add_momentum_features(merged, snapshots)
        merged = self._add_rank_features(merged, snapshots)

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
        season = int(snap["season"])
        week = int(snap["week"])

        # Base features from snapshot
        features = {
            "home_elo": float(snap["home_elo_pre"]),
            "away_elo": float(snap["away_elo_pre"]),
            "elo_diff": float(snap["home_elo_pre"] - snap["away_elo_pre"]),
            "elo_prob_home": float(snap["elo_prob_home"]),
            "elo_prob_away": float(1.0 - snap["elo_prob_home"]),
            "hfa_used": float(snap["hfa_used"]),
            "home_elo_uncertainty": float(snap["home_elo_uncertainty"]),
            "away_elo_uncertainty": float(snap["away_elo_uncertainty"]),
        }

        # Compute momentum for home and away teams
        home_team = snap["home_team"]
        away_team = snap["away_team"]
        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            team_home = snapshots[
                (snapshots["season"] == season)
                & (snapshots["week"] < week)
                & (snapshots["home_team"] == team)
            ][["week", "home_elo_pre"]].rename(columns={"home_elo_pre": "team_elo"})
            team_away = snapshots[
                (snapshots["season"] == season)
                & (snapshots["week"] < week)
                & (snapshots["away_team"] == team)
            ][["week", "away_elo_pre"]].rename(columns={"away_elo_pre": "team_elo"})
            prior_games = pd.concat(
                [team_home, team_away], ignore_index=True
            ).sort_values("week")

            if len(prior_games) == 0:
                features[f"{prefix}_elo_momentum"] = float("nan")
            else:
                window = prior_games.tail(4)
                oldest_elo = window.iloc[0]["team_elo"]
                newest_elo = window.iloc[-1]["team_elo"]
                features[f"{prefix}_elo_momentum"] = float(
                    (newest_elo - oldest_elo) / len(window)
                )

        # Compute rank and percentile
        season_snaps = snapshots[
            (snapshots["season"] == season) & (snapshots["week"] <= week)
        ]
        team_elos: dict[str, float] = {}
        for _, s in season_snaps.iterrows():
            team_elos[s["home_team"]] = s["home_elo_pre"]
            team_elos[s["away_team"]] = s["away_elo_pre"]

        sorted_teams = sorted(team_elos.items(), key=lambda x: x[1], reverse=True)
        n_teams = len(sorted_teams)
        team_rank = {t: r + 1 for r, (t, _) in enumerate(sorted_teams)}

        for team, prefix in [(home_team, "home"), (away_team, "away")]:
            rank = team_rank.get(team, n_teams)
            features[f"{prefix}_elo_rank"] = float(rank)
            features[f"{prefix}_elo_percentile"] = float(
                (n_teams - rank + 1) / n_teams
            )

        return features
