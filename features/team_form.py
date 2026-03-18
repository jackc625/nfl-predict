"""
Team Form Metrics Calculator

This module calculates rolling team performance metrics from play-by-play data:
- Rolling 4-week EPA/play calculations (offense/defense)
- Success rate metrics (offense/defense)
- Neutral situation pass rate calculations
- Rest days since last game
- Ensures no data leakage (only past games used)

Key metrics:
- EPA/play: Expected Points Added per play (nfl_data_py provides this)
- Success rate: Percentage of plays that increase win probability
- Neutral situations: Down 1-2, distance 5+ yards, not in red zone
- Pass rate: Tendency to pass vs run in neutral situations
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe, save_dataframe
from utils import get_logger

logger = get_logger(__name__)


class TeamFormCalculator:
    """
    Calculate rolling team form metrics from play-by-play data.

    Features calculated:
    - Offensive EPA/play (overall, pass, rush)
    - Defensive EPA/play (overall, pass, rush)
    - Offensive success rate
    - Defensive success rate
    - Neutral situation pass rate
    - Red zone efficiency
    - Third down conversion rates
    - Rest days since last game
    """

    def __init__(self, rolling_weeks: int = 4):
        """
        Initialize team form calculator.

        Args:
            rolling_weeks: Number of weeks to use for rolling averages
        """
        self.rolling_weeks = rolling_weeks
        self.settings = get_settings()

        # Team name mapping for consistency with our game data
        self.team_mapping = self._build_team_mapping()

    def _build_team_mapping(self) -> dict[str, str]:
        """Build mapping from nfl_data_py team names to our canonical abbreviations."""
        return {
            # Most should be the same, but handle any differences
            "LA": "LAR",  # Los Angeles Rams
            "LV": "LV",  # Las Vegas Raiders
            "GB": "GB",  # Green Bay Packers
            # Add any other mappings needed
        }

    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to our canonical format."""
        if not team or pd.isna(team):
            return team
        team_upper = str(team).upper().strip()
        return self.team_mapping.get(team_upper, team_upper)

    def fetch_pbp_data(self, seasons: list[int]) -> pd.DataFrame:
        """
        Fetch play-by-play data for specified seasons.

        Args:
            seasons: List of seasons to fetch

        Returns:
            DataFrame with play-by-play data
        """
        try:
            logger.info("Fetching play-by-play data", seasons=seasons)

            # Import play-by-play data (this is the expensive call)
            pbp_df = nfl.import_pbp_data(seasons, include_participation=False)

            # Normalize team names
            pbp_df["posteam"] = pbp_df["posteam"].apply(self._normalize_team_name)
            pbp_df["defteam"] = pbp_df["defteam"].apply(self._normalize_team_name)

            # Filter to regular season only (weeks 1-18)
            pbp_df = pbp_df[pbp_df["week"].between(1, 18)]

            # Filter to meaningful plays (exclude special teams, penalties, etc.)
            meaningful_plays = pbp_df[
                (pbp_df["play_type"].isin(["pass", "run"]))
                & (pbp_df["posteam"].notna())
                & (pbp_df["defteam"].notna())
                & (pbp_df["epa"].notna())
            ].copy()

            logger.info(
                "Fetched play-by-play data",
                total_plays=len(pbp_df),
                meaningful_plays=len(meaningful_plays),
                seasons=seasons,
            )

            return meaningful_plays

        except Exception as e:
            logger.error(
                "Failed to fetch play-by-play data", seasons=seasons, error=str(e)
            )
            raise

    def _identify_neutral_situations(self, pbp_df: pd.DataFrame) -> pd.Series:
        """
        Identify neutral game situations for pass rate analysis.

        Neutral situation criteria:
        - Down 1 or 2
        - 5+ yards to go
        - Not in red zone (>20 yards from goal)
        - Score differential within 14 points
        - Not in final 2 minutes of half

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            Boolean series indicating neutral situations
        """
        return (
            (pbp_df["down"].isin([1, 2]))
            & (pbp_df["ydstogo"] >= 5)
            & (pbp_df["yardline_100"] > 20)
            & (pbp_df["score_differential"].abs() <= 14)
            & (pbp_df["half_seconds_remaining"] > 120)
        )

    def calculate_team_game_stats(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team-level statistics for each game from play-by-play data.

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            DataFrame with team-game level statistics
        """
        logger.info("Calculating team-game statistics", plays=len(pbp_df))

        # Add neutral situation indicator
        pbp_df = pbp_df.copy()
        pbp_df["neutral_situation"] = self._identify_neutral_situations(pbp_df)

        # Separate offensive and defensive stats
        offense_stats = []
        defense_stats = []

        # Group by game and team (offensive perspective)
        for (game_id, season, week, team), group in pbp_df.groupby(
            ["game_id", "season", "week", "posteam"]
        ):
            if team is None or pd.isna(team):
                continue

            plays = len(group)
            if plays == 0:
                continue

            # Basic EPA metrics
            total_epa = group["epa"].sum()
            epa_per_play = group["epa"].mean()

            # Pass vs Run EPA
            pass_plays = group[group["play_type"] == "pass"]
            rush_plays = group[group["play_type"] == "run"]

            pass_epa_per_play = pass_plays["epa"].mean() if len(pass_plays) > 0 else 0
            rush_epa_per_play = rush_plays["epa"].mean() if len(rush_plays) > 0 else 0

            # Success rate
            success_rate = group["success"].mean()
            pass_success_rate = (
                pass_plays["success"].mean() if len(pass_plays) > 0 else 0
            )
            rush_success_rate = (
                rush_plays["success"].mean() if len(rush_plays) > 0 else 0
            )

            # Neutral situation metrics
            neutral_plays = group[group["neutral_situation"]]
            neutral_pass_rate = 0
            if len(neutral_plays) > 0:
                neutral_pass_rate = (neutral_plays["play_type"] == "pass").mean()

            # Red zone metrics (within 20 yards of goal)
            red_zone_plays = group[group["yardline_100"] <= 20]
            red_zone_td_rate = 0
            if len(red_zone_plays) > 0:
                red_zone_td_rate = red_zone_plays["touchdown"].mean()

            # Third down metrics
            third_down_plays = group[group["down"] == 3]
            third_down_conversion_rate = 0
            if len(third_down_plays) > 0:
                third_down_conversion_rate = third_down_plays["first_down"].mean()

            offense_stats.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "side": "offense",
                    "plays": plays,
                    "total_epa": total_epa,
                    "epa_per_play": epa_per_play,
                    "pass_epa_per_play": pass_epa_per_play,
                    "rush_epa_per_play": rush_epa_per_play,
                    "success_rate": success_rate,
                    "pass_success_rate": pass_success_rate,
                    "rush_success_rate": rush_success_rate,
                    "neutral_pass_rate": neutral_pass_rate,
                    "red_zone_td_rate": red_zone_td_rate,
                    "third_down_conversion_rate": third_down_conversion_rate,
                    "pass_attempts": len(pass_plays),
                    "rush_attempts": len(rush_plays),
                }
            )

        # Group by game and team (defensive perspective)
        for (game_id, season, week, team), group in pbp_df.groupby(
            ["game_id", "season", "week", "defteam"]
        ):
            if team is None or pd.isna(team):
                continue

            plays = len(group)
            if plays == 0:
                continue

            # Defensive EPA (opponent's EPA against this defense)
            total_epa_allowed = group["epa"].sum()
            epa_per_play_allowed = group["epa"].mean()

            # Pass vs Run EPA allowed
            pass_plays = group[group["play_type"] == "pass"]
            rush_plays = group[group["play_type"] == "run"]

            pass_epa_per_play_allowed = (
                pass_plays["epa"].mean() if len(pass_plays) > 0 else 0
            )
            rush_epa_per_play_allowed = (
                rush_plays["epa"].mean() if len(rush_plays) > 0 else 0
            )

            # Success rate allowed
            success_rate_allowed = group["success"].mean()
            pass_success_rate_allowed = (
                pass_plays["success"].mean() if len(pass_plays) > 0 else 0
            )
            rush_success_rate_allowed = (
                rush_plays["success"].mean() if len(rush_plays) > 0 else 0
            )

            # Red zone defense
            red_zone_plays = group[group["yardline_100"] <= 20]
            red_zone_td_rate_allowed = 0
            if len(red_zone_plays) > 0:
                red_zone_td_rate_allowed = red_zone_plays["touchdown"].mean()

            # Third down defense
            third_down_plays = group[group["down"] == 3]
            third_down_conversion_rate_allowed = 0
            if len(third_down_plays) > 0:
                third_down_conversion_rate_allowed = third_down_plays[
                    "first_down"
                ].mean()

            defense_stats.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "side": "defense",
                    "plays": plays,
                    "total_epa": total_epa_allowed,
                    "epa_per_play": epa_per_play_allowed,
                    "pass_epa_per_play": pass_epa_per_play_allowed,
                    "rush_epa_per_play": rush_epa_per_play_allowed,
                    "success_rate": success_rate_allowed,
                    "pass_success_rate": pass_success_rate_allowed,
                    "rush_success_rate": rush_success_rate_allowed,
                    "neutral_pass_rate": np.nan,  # Not applicable for defense
                    "red_zone_td_rate": red_zone_td_rate_allowed,
                    "third_down_conversion_rate": third_down_conversion_rate_allowed,
                    "pass_attempts": len(pass_plays),
                    "rush_attempts": len(rush_plays),
                }
            )

        # Combine offense and defense stats
        all_stats = offense_stats + defense_stats
        stats_df = pd.DataFrame(all_stats)

        logger.info(
            "Calculated team-game statistics",
            total_records=len(stats_df),
            offense_records=len(offense_stats),
            defense_records=len(defense_stats),
        )

        return stats_df

    def calculate_rolling_averages(
        self, team_stats_df: pd.DataFrame, target_season: int, target_week: int
    ) -> pd.DataFrame:
        """
        Calculate rolling averages for each team up to a specific point in time.

        Args:
            team_stats_df: Team-game statistics DataFrame
            target_season: Season to calculate rolling averages for
            target_week: Week to calculate rolling averages for

        Returns:
            DataFrame with rolling averages for each team
        """
        logger.info(
            "Calculating rolling averages",
            target_season=target_season,
            target_week=target_week,
            rolling_weeks=self.rolling_weeks,
        )

        # Filter to games before target week (no data leakage)
        historical_games = team_stats_df[
            (team_stats_df["season"] < target_season)
            | (
                (team_stats_df["season"] == target_season)
                & (team_stats_df["week"] < target_week)
            )
        ].copy()

        if len(historical_games) == 0:
            logger.warning("No historical games found for rolling averages")
            return pd.DataFrame()

        # Sort chronologically
        historical_games = historical_games.sort_values(["season", "week"])

        rolling_stats = []

        # Calculate rolling averages for each team and side
        for (team, side), group in historical_games.groupby(["team", "side"]):
            # Get the most recent N weeks of games
            recent_games = group.tail(self.rolling_weeks)

            if len(recent_games) == 0:
                continue

            # Calculate weighted averages (more recent games weighted higher)
            weights = np.arange(1, len(recent_games) + 1)  # 1, 2, 3, 4 for 4 games
            weights = weights / weights.sum()

            # Calculate weighted averages for key metrics
            avg_stats = {}
            for metric in [
                "epa_per_play",
                "pass_epa_per_play",
                "rush_epa_per_play",
                "success_rate",
                "pass_success_rate",
                "rush_success_rate",
                "neutral_pass_rate",
                "red_zone_td_rate",
                "third_down_conversion_rate",
            ]:
                values = recent_games[metric].values
                if metric == "neutral_pass_rate" and side == "defense":
                    avg_stats[f"rolling_{metric}"] = np.nan
                else:
                    # Handle NaN values
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() > 0:
                        valid_values = values[valid_mask]
                        valid_weights = weights[valid_mask]
                        valid_weights = valid_weights / valid_weights.sum()
                        avg_stats[f"rolling_{metric}"] = np.average(
                            valid_values, weights=valid_weights
                        )
                    else:
                        avg_stats[f"rolling_{metric}"] = np.nan

            rolling_stats.append(
                {
                    "team": team,
                    "side": side,
                    "target_season": target_season,
                    "target_week": target_week,
                    "games_used": len(recent_games),
                    "weeks_span": (
                        recent_games["season"].max() - recent_games["season"].min()
                    )
                    * 18
                    + (recent_games["week"].max() - recent_games["week"].min()),
                    **avg_stats,
                }
            )

        rolling_df = pd.DataFrame(rolling_stats)

        logger.info(
            "Calculated rolling averages",
            teams=len(rolling_df["team"].unique()) if len(rolling_df) > 0 else 0,
            records=len(rolling_df),
        )

        return rolling_df

    def calculate_rest_days(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rest days for each team since their last game.

        Args:
            games_df: Games DataFrame with kickoff times

        Returns:
            DataFrame with rest days added
        """
        games_with_rest = []

        for team in games_df[["home_team", "away_team"]].values.flatten():
            if pd.isna(team):
                continue

            # Get all games for this team, sorted chronologically
            team_games = (
                games_df[
                    (games_df["home_team"] == team) | (games_df["away_team"] == team)
                ]
                .sort_values("kickoff_et")
                .copy()
            )

            team_games["rest_days"] = np.nan

            for i in range(1, len(team_games)):
                current_game = team_games.iloc[i]
                previous_game = team_games.iloc[i - 1]

                rest_days = (
                    current_game["kickoff_et"] - previous_game["kickoff_et"]
                ).days
                team_games.iloc[i, team_games.columns.get_loc("rest_days")] = rest_days

            # Set first game of season to standard rest (7 days)
            if len(team_games) > 0:
                team_games.iloc[0, team_games.columns.get_loc("rest_days")] = 7

            games_with_rest.append(team_games)

        if games_with_rest:
            result_df = pd.concat(games_with_rest).drop_duplicates(subset=["game_id"])
            return result_df.sort_values(["season", "week", "kickoff_et"])
        return games_df.copy()

    def build_team_form_features(
        self,
        seasons: list[int],
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """
        Build complete team form features for specified seasons.

        Args:
            seasons: Seasons to process play-by-play data for
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for

        Returns:
            DataFrame with team form features
        """
        logger.info(
            "Building team form features",
            seasons=seasons,
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Fetch play-by-play data
            pbp_df = self.fetch_pbp_data(seasons)

            # Calculate team-game statistics
            team_stats_df = self.calculate_team_game_stats(pbp_df)

            # Save team statistics to silver layer
            save_dataframe(
                team_stats_df,
                "team_game_stats",
                layer="silver",
                partition_cols=["season"],
            )

            # If specific target provided, calculate rolling averages for that point
            if target_season and target_week:
                rolling_df = self.calculate_rolling_averages(
                    team_stats_df, target_season, target_week
                )
                return rolling_df

            # Otherwise, calculate rolling averages for all weeks
            all_rolling_stats = []

            for season in seasons:
                for week in range(1, 19):  # Weeks 1-18
                    rolling_df = self.calculate_rolling_averages(
                        team_stats_df, season, week
                    )
                    if len(rolling_df) > 0:
                        all_rolling_stats.append(rolling_df)

            if all_rolling_stats:
                final_df = pd.concat(all_rolling_stats, ignore_index=True)

                # Save rolling team form features
                save_dataframe(
                    final_df,
                    "team_form_features",
                    layer="silver",
                    partition_cols=["target_season"],
                )

                logger.info(
                    "Built team form features",
                    total_records=len(final_df),
                    seasons=seasons,
                )

                return final_df
            return pd.DataFrame()

        except Exception as e:
            logger.error(
                "Failed to build team form features", seasons=seasons, error=str(e)
            )
            raise

    def get_team_form_for_game(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> dict[str, dict[str, float]]:
        """
        Get team form features for a specific game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dictionary with form features for both teams
        """
        try:
            # Load team form features
            form_df = load_dataframe("team_form_features", layer="silver")

            # Filter to the specific season/week
            week_form = form_df[
                (form_df["target_season"] == season) & (form_df["target_week"] == week)
            ]

            result = {}

            for team in [home_team, away_team]:
                result[team] = {}

                # Get offensive and defensive form
                team_offense = week_form[
                    (week_form["team"] == team) & (week_form["side"] == "offense")
                ]
                team_defense = week_form[
                    (week_form["team"] == team) & (week_form["side"] == "defense")
                ]

                # Combine offense and defense metrics
                if len(team_offense) > 0:
                    offense_row = team_offense.iloc[0]
                    for col in offense_row.index:
                        if col.startswith("rolling_"):
                            result[team][f"off_{col}"] = offense_row[col]

                if len(team_defense) > 0:
                    defense_row = team_defense.iloc[0]
                    for col in defense_row.index:
                        if col.startswith("rolling_"):
                            result[team][f"def_{col}"] = defense_row[col]

            return result

        except Exception as e:
            logger.error(
                "Failed to get team form for game",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def validate_form_features(self, form_df: pd.DataFrame) -> bool:
        """
        Validate team form features for data quality.

        Args:
            form_df: Team form features DataFrame

        Returns:
            True if validation passes
        """
        if len(form_df) == 0:
            logger.error("No team form features found")
            return False

        # Check for required columns
        required_cols = [
            "team",
            "side",
            "target_season",
            "target_week",
            "rolling_epa_per_play",
        ]
        missing_cols = set(required_cols) - set(form_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check for reasonable EPA ranges
        epa_cols = [col for col in form_df.columns if "epa_per_play" in col]
        for col in epa_cols:
            values = form_df[col].dropna()
            if len(values) > 0:
                if values.min() < -2.0 or values.max() > 2.0:
                    logger.warning(
                        f"EPA values outside reasonable range for {col}",
                        min_value=values.min(),
                        max_value=values.max(),
                    )

        # Check success rate ranges (should be 0-1)
        success_cols = [col for col in form_df.columns if "success_rate" in col]
        for col in success_cols:
            values = form_df[col].dropna()
            if len(values) > 0 and (values.min() < 0 or values.max() > 1):
                logger.warning(
                    f"Success rate outside 0-1 range for {col}",
                    min_value=values.min(),
                    max_value=values.max(),
                )

        logger.info(
            "Team form features validation completed",
            records=len(form_df),
            teams=len(form_df["team"].unique()),
            seasons=sorted(form_df["target_season"].unique()),
        )

        return True
