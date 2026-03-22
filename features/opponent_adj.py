"""Opponent-Adjusted EPA Feature (FEAT-14).

Single-pass opponent strength adjustment for EPA metrics, following the
Open Source Football methodology:
https://opensourcefootball.com/posts/2020-08-20-adjusting-epa-for-strenght-of-opponent/

This is a post-processor: it takes per-game EPA stats (from TeamFormCalculator)
and produces opponent-adjusted rolling EPA values that replace the raw versions.

Algorithm:
For each team-game:
1. Get the team's raw EPA/play for that game
2. Look up each opponent's lagged rolling defensive EPA/play
   (what was the opponent's defense allowing, on average, BEFORE this game?)
3. Compute adjustment = league_avg_def_epa - opponent_lagged_def_epa
   (positive if opponent has worse-than-average defense)
4. adjusted_epa = raw_epa + adjustment

The adjustment is bidirectional:
- Offensive EPA is adjusted for opponent defensive quality
- Defensive EPA is adjusted for opponent offensive quality

Key constraints:
- Opponent metrics are lagged by 1 week (cannot use current-game data)
- Below min_opponent_games in the lag window, adjustment = 0 (raw EPA used)
- as_of_datetime time-fence is respected (no future data)
"""

from datetime import datetime

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

# EPA metrics to adjust (3 per side = 6 total)
_EPA_METRICS = ["epa_per_play", "pass_epa_per_play", "rush_epa_per_play"]

# Rolling output column names for each metric
_ROLLING_NAMES = {
    "epa_per_play": "rolling_opp_adj_epa_per_play",
    "pass_epa_per_play": "rolling_opp_adj_pass_epa",
    "rush_epa_per_play": "rolling_opp_adj_rush_epa",
}


class OpponentAdjuster:
    """Single-pass opponent strength adjustment for EPA metrics.

    Adjusts a team's EPA by comparing their opponents' quality to the
    league average. Uses Open Source Football methodology:
    https://opensourcefootball.com/posts/2020-08-20-adjusting-epa-for-strenght-of-opponent/

    This is a post-processor: it takes per-game EPA stats (from
    TeamFormCalculator) and produces opponent-adjusted rolling EPA values
    that replace the raw versions.

    Conforms to the FeatureBuilder Protocol via build_features() and
    get_features_for_game().
    """

    def __init__(self, window: int = 10, min_opponent_games: int = 4):
        """Initialize opponent adjuster.

        Args:
            window: Number of games for opponent rolling average (default 10).
            min_opponent_games: Minimum opponent games before applying
                adjustment. Below this threshold, raw EPA is returned
                (default 4).
        """
        self.window = window
        self.min_opponent_games = min_opponent_games

    # ------------------------------------------------------------------
    # Core adjustment logic
    # ------------------------------------------------------------------

    def _compute_lagged_opponent_rolling(
        self,
        team_game_stats: pd.DataFrame,
        metric: str,
        side: str,
    ) -> pd.DataFrame:
        """Compute lagged rolling average of each team's metric for a given side.

        For each team, computes a rolling average of their per-game metric
        (e.g. defensive EPA/play), then shifts by 1 to lag it. This gives
        us "what was this team's defense allowing, on average, BEFORE the
        current game?"

        Args:
            team_game_stats: Per-game stats with columns: team, season, week,
                side, and the metric column.
            metric: Column name to compute rolling average for (e.g.
                'epa_per_play').
            side: Which side to compute for ('offense' or 'defense').

        Returns:
            DataFrame with columns: team, season, week, and the lagged
            rolling metric column (named f'lagged_{side}_{metric}').
        """
        side_stats = team_game_stats[team_game_stats["side"] == side].copy()
        side_stats = side_stats.sort_values(["team", "season", "week"])

        lagged_col = f"lagged_{side}_{metric}"
        game_count_col = f"game_count_{side}_{metric}"

        def _rolling_with_lag(group: pd.DataFrame) -> pd.DataFrame:
            values = group[metric].values.astype(float)
            n = len(values)
            lagged = np.full(n, np.nan)
            counts = np.full(n, 0, dtype=int)

            for i in range(n):
                # Use games before this one (indices 0..i-1)
                if i == 0:
                    # No prior games, no lagged value
                    continue
                start = max(0, i - self.window)
                window_values = values[start:i]
                counts[i] = len(window_values)
                if len(window_values) > 0:
                    lagged[i] = np.nanmean(window_values)

            group = group.copy()
            group[lagged_col] = lagged
            group[game_count_col] = counts
            return group

        # Compute lagged rolling per team; iterate instead of groupby.apply
        # to avoid FutureWarning about grouping columns
        parts = []
        for _team, grp in side_stats.groupby("team", group_keys=False):
            parts.append(_rolling_with_lag(grp))
        result = pd.concat(parts, ignore_index=True) if parts else side_stats.copy()

        return result[["team", "season", "week", lagged_col, game_count_col]]

    def _adjust_per_game_epa(
        self,
        team_game_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply opponent strength adjustment to per-game EPA metrics.

        For each team-game:
        - Offensive EPA is adjusted using opponent's lagged defensive metrics
        - Defensive EPA is adjusted using opponent's lagged offensive metrics

        Args:
            team_game_stats: Per-game stats DataFrame with columns: team,
                opponent, season, week, side, epa_per_play, pass_epa_per_play,
                rush_epa_per_play.

        Returns:
            DataFrame with adjusted per-game EPA columns added.
        """
        result = team_game_stats.copy()

        for metric in _EPA_METRICS:
            adj_col = f"opp_adj_{metric}"

            # Compute lagged rolling defensive EPA for all teams
            # (used to adjust offensive EPA)
            def_lagged = self._compute_lagged_opponent_rolling(
                team_game_stats, metric, "defense"
            )
            def_lagged_col = f"lagged_defense_{metric}"
            def_count_col = f"game_count_defense_{metric}"

            # Compute lagged rolling offensive EPA for all teams
            # (used to adjust defensive EPA)
            off_lagged = self._compute_lagged_opponent_rolling(
                team_game_stats, metric, "offense"
            )
            off_lagged_col = f"lagged_offense_{metric}"
            off_count_col = f"game_count_offense_{metric}"

            # League average of lagged defensive metric (across all teams)
            league_avg_def = def_lagged[def_lagged_col].mean()
            # League average of lagged offensive metric
            league_avg_off = off_lagged[off_lagged_col].mean()

            # Initialize adjusted column with raw values
            result[adj_col] = result[metric].astype(float)

            # ---- Adjust offensive EPA for opponent defensive quality ----
            off_mask = result["side"] == "offense"
            off_rows = result[off_mask].copy()

            # Look up each opponent's lagged defensive metric
            # Merge on opponent = def_lagged.team, same season/week
            off_merged = off_rows.merge(
                def_lagged.rename(columns={"team": "opponent"}),
                on=["opponent", "season", "week"],
                how="left",
            )

            # Compute adjustment: league_avg_def - opponent_lagged_def
            # Positive if opponent has worse-than-average defense
            off_adjustment = league_avg_def - off_merged[def_lagged_col]
            off_game_count = off_merged[def_count_col]

            # Apply adjustment only when opponent has enough games
            has_enough = off_game_count >= self.min_opponent_games
            off_adjusted = off_merged[metric].astype(float).copy()
            off_adjusted[has_enough] = (
                off_merged.loc[has_enough, metric].astype(float)
                + off_adjustment[has_enough]
            )
            # Where not enough games or NaN, keep raw value
            off_adjusted[~has_enough | off_adjustment.isna()] = (
                off_merged.loc[~has_enough | off_adjustment.isna(), metric]
                .astype(float)
            )

            result.loc[off_mask, adj_col] = off_adjusted.values

            # ---- Adjust defensive EPA for opponent offensive quality ----
            def_mask = result["side"] == "defense"
            def_rows = result[def_mask].copy()

            # Look up each opponent's lagged offensive metric
            def_merged = def_rows.merge(
                off_lagged.rename(columns={"team": "opponent"}),
                on=["opponent", "season", "week"],
                how="left",
            )

            # Compute adjustment: league_avg_off - opponent_lagged_off
            # Positive if opponent has worse-than-average offense
            def_adjustment = league_avg_off - def_merged[off_lagged_col]
            def_game_count = def_merged[off_count_col]

            has_enough_def = def_game_count >= self.min_opponent_games
            def_adjusted = def_merged[metric].astype(float).copy()
            def_adjusted[has_enough_def] = (
                def_merged.loc[has_enough_def, metric].astype(float)
                + def_adjustment[has_enough_def]
            )
            def_adjusted[~has_enough_def | def_adjustment.isna()] = (
                def_merged.loc[~has_enough_def | def_adjustment.isna(), metric]
                .astype(float)
            )

            result.loc[def_mask, adj_col] = def_adjusted.values

        return result

    def _compute_rolling_adjusted(
        self,
        adjusted_stats: pd.DataFrame,
        target_season: int,
        target_week: int,
    ) -> pd.DataFrame:
        """Compute rolling averages of adjusted per-game EPA metrics.

        Uses recency-weighted expanding window (same approach as
        TeamFormCalculator) to compute rolling averages of the adjusted
        EPA values.

        Args:
            adjusted_stats: Per-game stats with opp_adj_* columns.
            target_season: Season to compute rolling averages for.
            target_week: Week to compute rolling averages for.

        Returns:
            DataFrame with rolling_opp_adj_* columns, one row per team per side.
        """
        # Filter to games before target week (no data leakage)
        historical = adjusted_stats[
            (adjusted_stats["season"] < target_season)
            | (
                (adjusted_stats["season"] == target_season)
                & (adjusted_stats["week"] < target_week)
            )
        ].copy()

        if len(historical) == 0:
            logger.warning("No historical games for rolling adjusted EPA")
            return pd.DataFrame()

        historical = historical.sort_values(["season", "week"])

        rolling_rows = []

        for (team, side), group in historical.groupby(["team", "side"]):
            if len(group) == 0:
                continue

            # Use recency weighting [1, 2, ..., N]
            n = len(group)
            weights = np.arange(1, n + 1, dtype=float)
            weights = weights / weights.sum()

            row_data = {
                "team": team,
                "side": side,
                "target_season": target_season,
                "target_week": target_week,
                "games_used": n,
            }

            for metric in _EPA_METRICS:
                adj_col = f"opp_adj_{metric}"
                rolling_col = _ROLLING_NAMES[metric]

                values = group[adj_col].values.astype(float)
                valid_mask = ~np.isnan(values)

                if valid_mask.sum() > 0:
                    valid_values = values[valid_mask]
                    valid_weights = weights[valid_mask]
                    valid_weights = valid_weights / valid_weights.sum()
                    row_data[rolling_col] = float(
                        np.average(valid_values, weights=valid_weights)
                    )
                else:
                    row_data[rolling_col] = np.nan

            rolling_rows.append(row_data)

        return pd.DataFrame(rolling_rows)

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
        team_game_stats: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build opponent-adjusted EPA features.

        Conforms to the FeatureBuilder Protocol. Takes either pre-computed
        team_game_stats or the games schedule (to look up opponents), and
        produces rolling opponent-adjusted EPA metrics.

        Args:
            games_df: DataFrame of games with home_team, away_team, season,
                week columns.
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.
            target_season: Season to calculate features for.
            target_week: Week to calculate features for.
            team_game_stats: Pre-computed per-game stats from
                TeamFormCalculator. If None, will be derived from games_df
                schedule (requires opponent column).

        Returns:
            DataFrame with rolling opponent-adjusted EPA features.
        """
        logger.info(
            "Building opponent-adjusted EPA features",
            as_of_datetime=str(as_of_datetime),
            target_season=target_season,
            target_week=target_week,
        )

        if team_game_stats is None:
            msg = (
                "team_game_stats must be provided. OpponentAdjuster is a "
                "post-processor that requires pre-computed per-game stats "
                "from TeamFormCalculator."
            )
            raise ValueError(msg)

        stats = team_game_stats.copy()

        # Ensure opponent column exists
        if "opponent" not in stats.columns:
            stats = self._add_opponent_column(stats, games_df)

        # Apply as_of_datetime filter if kickoff column exists
        if "kickoff_et" in stats.columns:
            stats = stats[stats["kickoff_et"] < as_of_datetime]

        # Filter to only relevant seasons (target + prior)
        if target_season is not None:
            stats = stats[
                stats["season"].isin([target_season - 1, target_season])
            ]

        # Apply per-game opponent adjustment
        adjusted = self._adjust_per_game_epa(stats)

        # Compute rolling averages of adjusted metrics
        if target_season is not None and target_week is not None:
            return self._compute_rolling_adjusted(
                adjusted, target_season, target_week
            )

        # Fall back: compute for all weeks in all present seasons
        all_seasons = sorted(adjusted["season"].unique())
        all_rolling = []
        for season in all_seasons:
            max_week = int(
                adjusted[adjusted["season"] == season]["week"].max()
            )
            for week in range(2, max_week + 2):
                rolling_df = self._compute_rolling_adjusted(
                    adjusted, season, week
                )
                if len(rolling_df) > 0:
                    all_rolling.append(rolling_df)

        if all_rolling:
            return pd.concat(all_rolling, ignore_index=True)
        return pd.DataFrame()

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get opponent-adjusted EPA features for a single game.

        Conforms to the FeatureBuilder Protocol. Returns a dictionary
        mapping feature names to values for the specified game.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        # This method requires access to stored features or a full
        # computation pipeline. For now, return empty dict -- the primary
        # interface is build_features().
        logger.warning(
            "get_features_for_game not yet implemented for OpponentAdjuster",
            game_id=game_id,
        )
        return {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _add_opponent_column(
        stats: pd.DataFrame,
        games_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add opponent column to team game stats using the schedule.

        For each team-game row, looks up who the opponent was from the
        games schedule (home_team/away_team).

        Args:
            stats: Team game stats DataFrame with game_id, team columns.
            games_df: Games schedule with game_id, home_team, away_team.

        Returns:
            Stats DataFrame with 'opponent' column added.
        """
        # Build game -> (home, away) lookup
        game_teams = games_df.set_index("game_id")[
            ["home_team", "away_team"]
        ].to_dict("index")

        def _get_opponent(row: pd.Series) -> str:
            game_info = game_teams.get(row["game_id"])
            if game_info is None:
                return ""
            if row["team"] == game_info["home_team"]:
                return game_info["away_team"]
            return game_info["home_team"]

        stats = stats.copy()
        stats["opponent"] = stats.apply(_get_opponent, axis=1)
        return stats
