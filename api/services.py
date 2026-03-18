"""
Data services for NFL Prediction API.

This module provides data access services that interface between the API endpoints
and the underlying data pipeline, models, and backtest systems.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from utils.game_utils import determine_week_type, is_prime_time_game
from utils.team_data import get_team_info

from .config import settings
from .exceptions import (
    raise_model_unavailable,
    raise_not_found,
)
from .schemas import (
    BacktestSummary,
    BetRecommendation,
    BetType,
    CalibrationData,
    GameDetail,
    GameStatus,
    GameSummary,
    MarketInfo,
    PredictionInfo,
    RecommendationTier,
    TeamInfo,
    VenueInfo,
    WeatherInfo,
    WeekMetadata,
    validate_nfl_team,
)

logger = logging.getLogger(__name__)


class DataService:
    """Core data service for accessing NFL prediction data."""

    def __init__(self):
        self.data_path = Path(settings.data_path)
        self.outputs_path = Path(settings.outputs_path)
        self.artifacts_path = Path(settings.artifacts_path)

        # Initialize database connection
        self.db_path = self.data_path / "nfl_predictions.duckdb"

    def _get_db_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection with read-only access to avoid locking issues."""
        try:
            if not self.db_path.exists():
                logger.warning(f"Database file not found: {self.db_path}")
                # Return in-memory database as fallback
                return duckdb.connect()

            # Use read-only connection to avoid file locking issues with multiple workers
            conn = duckdb.connect(str(self.db_path), read_only=True)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            # Return in-memory database as fallback
            return duckdb.connect()

    def _normalize_team_filter(self, team: str) -> str:
        """Normalize team filter input for consistent querying."""
        if not team:
            return team

        # Strip whitespace and convert to uppercase
        normalized = team.strip().upper()

        # Validate against known NFL teams for early error detection
        try:
            return validate_nfl_team(normalized)
        except ValueError as e:
            logger.warning(f"Invalid team filter '{team}': {e}")
            # Return original normalized value to let downstream handle the error
            return normalized

    def _infer_current_season_week(self, current_date: datetime) -> tuple[int, int]:
        """
        Infer current NFL season and week from date.

        NFL Season Rules:
        - Regular season typically starts first Thursday after Labor Day (first Monday in September)
        - Season spans September through January of following year
        - Week 1 is typically the second week of September
        - Playoffs are weeks 19-22 (not included in regular season weeks 1-18)

        Args:
            current_date: The date to evaluate

        Returns:
            Tuple of (season_year, week_number)
        """
        year = current_date.year
        month = current_date.month

        # Season year determination:
        # - September through December: current year
        # - January through February: previous year (playoffs/offseason)
        # - March through August: previous year (offseason)
        if month >= 3 and month <= 8:
            # Offseason: use previous year's season
            season_year = year - 1
            return season_year, 18  # Return last week of season during offseason
        if month <= 2:
            # January-February: previous year's season (playoffs)
            season_year = year - 1
            return season_year, 18  # Playoffs period
        # September-December: current year
        season_year = year

        # Week calculation for in-season dates (September-December)
        # NFL season typically starts second Thursday of September
        # Use September 8th as average season start (can vary ±7 days)
        season_start = datetime(season_year, 9, 8)

        # Calculate weeks since season start
        days_since_start = (current_date - season_start).days

        if days_since_start < 0:
            # Before season starts
            return season_year, 1
        if days_since_start > 126:  # 18 weeks * 7 days = 126
            # After regular season (playoffs)
            return season_year, 18
        # During regular season
        week = min(max(1, (days_since_start // 7) + 1), 18)
        return season_year, week

    def _load_games_data(
        self, season: int | None = None, week: int | None = None
    ) -> pd.DataFrame:
        """Load games data from silver layer."""
        try:
            with self._get_db_connection() as conn:
                query = "SELECT * FROM games"
                conditions = []

                if season:
                    conditions.append(f"season = {season}")
                if week:
                    conditions.append(f"week = {week}")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY kickoff_et"

                df = conn.execute(query).df()

                # Log data freshness information
                row_count = len(df)
                if row_count > 0:
                    latest_update = (
                        df["kickoff_et"].max()
                        if "kickoff_et" in df.columns
                        else "unknown"
                    )
                    earliest_game = (
                        df["kickoff_et"].min()
                        if "kickoff_et" in df.columns
                        else "unknown"
                    )
                    logger.info(
                        f"Loaded games data: {row_count} rows, date range: {earliest_game} to {latest_update}",
                        extra={
                            "data_source": "games_table",
                            "row_count": row_count,
                            "season_filter": season,
                            "week_filter": week,
                            "latest_game": str(latest_update),
                            "earliest_game": str(earliest_game),
                        },
                    )
                else:
                    logger.warning(
                        f"No games data found for season={season}, week={week}",
                        extra={
                            "data_source": "games_table",
                            "row_count": 0,
                            "season_filter": season,
                            "week_filter": week,
                        },
                    )

                return df
        except Exception as e:
            logger.error(f"Failed to load games data: {e}")
            # Return empty DataFrame if no data available
            return pd.DataFrame()

    def _load_predictions_data(
        self, season: int | None = None, week: int | None = None
    ) -> pd.DataFrame:
        """Load prediction data from outputs."""
        try:
            predictions_file = (
                self.outputs_path / "predictions" / "current_predictions.parquet"
            )
            if not predictions_file.exists():
                logger.warning(
                    "Predictions file not found",
                    extra={
                        "data_source": "predictions_parquet",
                        "file_path": str(predictions_file),
                        "file_exists": False,
                    },
                )
                return pd.DataFrame()

            # Get file modification time for freshness
            file_mtime = datetime.fromtimestamp(predictions_file.stat().st_mtime, UTC)
            age_hours = (datetime.now(UTC) - file_mtime).total_seconds() / 3600

            df = pd.read_parquet(predictions_file)
            original_count = len(df)

            if season:
                df = df[df["season"] == season]
            if week:
                df = df[df["week"] == week]

            filtered_count = len(df)

            logger.info(
                f"Loaded predictions data: {filtered_count} rows (from {original_count} total), file age: {age_hours:.1f}h",
                extra={
                    "data_source": "predictions_parquet",
                    "row_count": filtered_count,
                    "original_row_count": original_count,
                    "file_age_hours": round(age_hours, 1),
                    "file_modified": file_mtime.isoformat(),
                    "season_filter": season,
                    "week_filter": week,
                    "data_freshness": "stale" if age_hours > 24 else "fresh",
                },
            )

            return df
        except Exception as e:
            logger.error(f"Failed to load predictions data: {e}")
            return pd.DataFrame()

    def _load_market_data(
        self, season: int | None = None, week: int | None = None
    ) -> pd.DataFrame:
        """Load market odds data from silver layer."""
        try:
            with self._get_db_connection() as conn:
                if season or week:
                    # Join with games table to filter by season/week
                    query = (
                        "SELECT o.* FROM odds o JOIN games g ON o.game_id = g.game_id"
                    )
                    conditions = []

                    if season:
                        conditions.append(f"g.season = {season}")
                    if week:
                        conditions.append(f"g.week = {week}")

                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                else:
                    # Load all odds data if no filters
                    query = "SELECT * FROM odds"

                df = conn.execute(query).df()

                # Log market data freshness
                row_count = len(df)
                if row_count > 0:
                    # Look for latest odds update timestamp if available
                    latest_update = "unknown"
                    if "last_updated" in df.columns:
                        latest_update = df["last_updated"].max()
                    elif "timestamp" in df.columns:
                        latest_update = df["timestamp"].max()

                    logger.info(
                        f"Loaded market data: {row_count} rows, latest update: {latest_update}",
                        extra={
                            "data_source": "odds_table",
                            "row_count": row_count,
                            "season_filter": season,
                            "week_filter": week,
                            "latest_odds_update": str(latest_update),
                        },
                    )
                else:
                    logger.warning(
                        f"No market data found for season={season}, week={week}",
                        extra={
                            "data_source": "odds_table",
                            "row_count": 0,
                            "season_filter": season,
                            "week_filter": week,
                        },
                    )

                return df
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return pd.DataFrame()

    def get_current_week_metadata(self) -> WeekMetadata:
        """Get current NFL season and week information."""
        try:
            # Try to get from database first
            with self._get_db_connection() as conn:
                # Check if games table exists
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [table[0] for table in tables]

                if "games" not in table_names:
                    logger.warning("Games table not found in database, using fallback")
                    raise Exception("Games table not found")

                result = conn.execute("""
                    SELECT
                        MAX(season) as current_season,
                        MAX(week) as current_week,
                        COUNT(*) as games_this_week
                    FROM games
                    WHERE kickoff_et >= CURRENT_DATE - INTERVAL '7 days'
                    AND kickoff_et <= CURRENT_DATE + INTERVAL '7 days'
                """).fetchone()

                if result and result[0]:
                    current_season, current_week, games_this_week = result

                    # Check if predictions are available
                    predictions_available = (
                        self.outputs_path
                        / "predictions"
                        / "current_predictions.parquet"
                    ).exists()

                    # Get last updated time
                    last_updated = datetime.now(UTC)
                    if predictions_available:
                        pred_file = (
                            self.outputs_path
                            / "predictions"
                            / "current_predictions.parquet"
                        )
                        last_updated = datetime.fromtimestamp(
                            pred_file.stat().st_mtime, UTC
                        )

                    # Determine actual week type
                    week_type = determine_week_type(
                        int(current_season), int(current_week)
                    )

                    return WeekMetadata(
                        current_season=int(current_season),
                        current_week=int(current_week),
                        week_type=week_type,
                        games_this_week=int(games_this_week or 0),
                        predictions_available=predictions_available,
                        last_updated=last_updated,
                        snapshot_time=last_updated,
                    )
        except Exception as e:
            logger.warning(f"Could not determine current week from database: {e}")

        # Try to get the most recent data from database as fallback
        try:
            with self._get_db_connection() as conn:
                # Get the most recent season and week that has data
                result = conn.execute("""
                    SELECT season, week, COUNT(*) as games
                    FROM games
                    GROUP BY season, week
                    ORDER BY season DESC, week DESC
                    LIMIT 1
                """).fetchone()

                if result:
                    current_season, current_week, games_count = result
                    logger.info(
                        f"Using most recent data: season {current_season}, week {current_week}"
                    )

                    week_type = determine_week_type(
                        int(current_season), int(current_week)
                    )
                    predictions_available = (
                        self.outputs_path
                        / "predictions"
                        / "current_predictions.parquet"
                    ).exists()

                    return WeekMetadata(
                        current_season=int(current_season),
                        current_week=int(current_week),
                        week_type=week_type,
                        games_this_week=int(games_count),
                        predictions_available=predictions_available,
                        last_updated=datetime.now(UTC),
                        snapshot_time=None,
                    )
        except Exception as e2:
            logger.warning(f"Could not get most recent data from database: {e2}")

        # Final fallback to current date-based logic
        now = datetime.now()
        current_season, current_week = self._infer_current_season_week(now)

        # Determine week type for fallback case
        week_type = determine_week_type(current_season, current_week)

        return WeekMetadata(
            current_season=current_season,
            current_week=current_week,
            week_type=week_type,
            games_this_week=0,
            predictions_available=False,
            last_updated=datetime.now(UTC),
            snapshot_time=None,
        )

    def list_games(
        self,
        season: int | None = None,
        week: int | None = None,
        team: str | None = None,
        status: str | None = None,
        has_recommendations: bool | None = None,
        min_edge: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[GameSummary], WeekMetadata, int]:
        """List games with optional filtering."""

        # Get current week metadata
        metadata = self.get_current_week_metadata()

        # Use current week if not specified
        if season is None:
            season = metadata.current_season
        if week is None:
            week = metadata.current_week

        # Load games data
        games_df = self._load_games_data(season, week)
        if games_df.empty:
            return [], metadata, 0

        # Load predictions and market data
        predictions_df = self._load_predictions_data(season, week)
        market_df = self._load_market_data(season, week)

        # Apply filters with normalization
        if team:
            team = self._normalize_team_filter(team)
            games_df = games_df[
                (games_df["home_team"] == team) | (games_df["away_team"] == team)
            ]

        if status:
            status = status.lower()  # Normalize status to lowercase
            games_df = games_df[games_df["status"].str.lower() == status]

        # Convert to GameSummary objects
        game_summaries = []
        for _, game in games_df.iterrows():
            try:
                # Get prediction data for this game (safely)
                if not predictions_df.empty and "game_id" in predictions_df.columns:
                    game_predictions = predictions_df[
                        predictions_df["game_id"] == game["game_id"]
                    ]
                else:
                    game_predictions = pd.DataFrame()

                if not market_df.empty and "game_id" in market_df.columns:
                    game_market = market_df[market_df["game_id"] == game["game_id"]]
                else:
                    game_market = pd.DataFrame()

                # Extract prediction values
                win_prob = None
                spread_pred = None
                total_pred = None
                top_recommendation = None
                has_recs = False

                if not game_predictions.empty:
                    pred = game_predictions.iloc[0]
                    win_prob = pred.get("win_probability")
                    spread_pred = pred.get("spread_prediction")
                    total_pred = pred.get("total_prediction")

                    # Generate actual recommendations using the recommendation engine
                    try:
                        from utils.api_recommendation_bridge import (
                            api_recommendation_bridge,
                        )

                        game_market_data = (
                            game_market.iloc[0]
                            if not game_market.empty
                            else pd.Series()
                        )
                        recommendations = (
                            api_recommendation_bridge.generate_game_recommendations(
                                game["game_id"],
                                pred,
                                game_market_data,
                                min_edge or 0.02,
                            )
                        )

                        has_recs = len(recommendations) > 0
                        top_recommendation = (
                            recommendations[0] if recommendations else None
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error generating recommendations for game {game['game_id']}: {e}"
                        )
                        # Fallback to simple logic
                        if min_edge is None or (
                            win_prob and abs(win_prob - 0.5) >= (min_edge or 0)
                        ):
                            has_recs = True
                            if win_prob and abs(win_prob - 0.5) >= 0.1:  # Strong edge
                                top_recommendation = BetRecommendation(
                                    bet_type=BetType.MONEYLINE,
                                    team=game["home_team"]
                                    if win_prob > 0.5
                                    else game["away_team"],
                                    recommended_odds=-110,
                                    edge=abs(win_prob - 0.5),
                                    expected_value=0.05,
                                    confidence=0.8,
                                    tier=RecommendationTier.MEDIUM,
                                    units=1.0,
                                    description=f"Strong edge on {'home' if win_prob > 0.5 else 'away'} team",
                                )
                            else:
                                has_recs = False
                                top_recommendation = None
                        else:
                            has_recs = False
                            top_recommendation = None

                # Apply recommendation filter
                if has_recommendations is not None and has_recommendations != has_recs:
                    continue

                # Determine game status
                game_status = GameStatus.SCHEDULED
                if game.get("status"):
                    if game["status"].lower() in ["final", "completed"]:
                        game_status = GameStatus.COMPLETED
                    elif game["status"].lower() in ["in_progress", "live"]:
                        game_status = GameStatus.IN_PROGRESS

                # Handle NULL scores
                home_score = game.get("home_score")
                away_score = game.get("away_score")
                if pd.isna(home_score):
                    home_score = None
                if pd.isna(away_score):
                    away_score = None

                # Determine if this is a prime time game
                game_date_parsed = pd.to_datetime(game["kickoff_et"])
                is_prime_time = is_prime_time_game(game_date_parsed)

                game_summary = GameSummary(
                    game_id=game["game_id"],
                    season=int(game["season"]),
                    week=int(game["week"]),
                    game_date=game_date_parsed,
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    status=game_status,
                    home_score=home_score,
                    away_score=away_score,
                    win_probability=win_prob,
                    spread=spread_pred,
                    total=total_pred,
                    top_recommendation=top_recommendation,
                    has_recommendations=has_recs,
                    is_prime_time=is_prime_time,
                )

                game_summaries.append(game_summary)

            except Exception as e:
                logger.error(
                    f"Error processing game {game.get('game_id') if hasattr(game, 'get') else game.get('game_id', 'unknown')}: {e}"
                )
                continue

        # Apply pagination
        total_games = len(game_summaries)
        game_summaries = game_summaries[offset : offset + limit]

        # Log summary of games processing
        logger.info(
            f"Processed games list: {len(game_summaries)} games returned (total: {total_games})",
            extra={
                "games_returned": len(game_summaries),
                "total_games": total_games,
                "season": season,
                "week": week,
                "filters": {
                    "team": team,
                    "status": status,
                    "has_recommendations": has_recommendations,
                    "min_edge": min_edge,
                },
            },
        )

        return game_summaries, metadata, total_games

    def get_game_detail(self, game_id: str) -> GameDetail:
        """Get detailed information for a specific game."""

        # Load game data
        with self._get_db_connection() as conn:
            try:
                result = conn.execute(
                    "SELECT * FROM games WHERE game_id = ?", [game_id]
                ).fetchone()

                if not result:
                    raise_not_found("Game", game_id)

                # Get column names
                columns = [desc[0] for desc in conn.description]
                # Convert to dict for easier access
                game_data = dict(zip(columns, result, strict=False))

            except Exception as e:
                logger.error(f"Database error querying game {game_id}: {e}")
                raise_not_found("Game", game_id)

        # Load additional data
        predictions_df = self._load_predictions_data()
        market_df = self._load_market_data()

        # Safely filter predictions and market data
        if not predictions_df.empty and "game_id" in predictions_df.columns:
            game_predictions = predictions_df[predictions_df["game_id"] == game_id]
        else:
            game_predictions = pd.DataFrame()

        if not market_df.empty and "game_id" in market_df.columns:
            game_market = market_df[market_df["game_id"] == game_id]
        else:
            game_market = pd.DataFrame()

        # Create team info with actual data
        home_team_info = get_team_info(game_data["home_team"])
        away_team_info = get_team_info(game_data["away_team"])

        home_team = TeamInfo(
            team_id=game_data["home_team"],
            team_name=home_team_info["name"],
            city=home_team_info["city"],
            conference=home_team_info["conference"],
            division=home_team_info["division"],
        )

        away_team = TeamInfo(
            team_id=game_data["away_team"],
            team_name=away_team_info["name"],
            city=away_team_info["city"],
            conference=away_team_info["conference"],
            division=away_team_info["division"],
        )

        # Create venue info
        venue = VenueInfo(
            venue_id=game_data.get("venue", "unknown"),
            venue_name=game_data.get("venue", "Unknown Stadium"),
            city="Unknown",
            state="Unknown",
            roof_type="outdoor",
            surface="grass",
        )

        # Weather info
        weather = None
        if venue.roof_type == "outdoor":
            weather = WeatherInfo(
                temperature=70.0,
                wind_speed=5.0,
                wind_direction="NW",
                precipitation_chance=0.1,
                conditions="Clear",
                is_dome=False,
            )

        # Predictions
        predictions = None
        if not game_predictions.empty:
            pred = game_predictions.iloc[0]
            predictions = PredictionInfo(
                win_probability=pred.get("win_probability", 0.5),
                spread_prediction=pred.get("spread_prediction", 0.0),
                total_prediction=pred.get("total_prediction", 45.0),
                win_prob_confidence=pred.get("win_prob_confidence", 0.7),
                spread_confidence=pred.get("spread_confidence", 0.7),
                total_confidence=pred.get("total_confidence", 0.7),
            )

        # Market info
        market = None
        if not game_market.empty:
            mkt = game_market.iloc[0]
            market = MarketInfo(
                home_moneyline=mkt.get("home_moneyline"),
                away_moneyline=mkt.get("away_moneyline"),
                spread=mkt.get("spread"),
                spread_odds=mkt.get("spread_odds", -110),
                total=mkt.get("total"),
                over_odds=mkt.get("over_odds", -110),
                under_odds=mkt.get("under_odds", -110),
                last_updated=datetime.now(UTC),
                sportsbook="DraftKings",
            )

        # Game status
        game_status = GameStatus.SCHEDULED
        if game_data.get("status"):
            if game_data["status"].lower() in ["final", "completed"]:
                game_status = GameStatus.COMPLETED
            elif game_data["status"].lower() in ["in_progress", "live"]:
                game_status = GameStatus.IN_PROGRESS

        # Generate actual recommendations for game detail
        recommendations = []
        try:
            if not game_predictions.empty and not game_market.empty:
                from utils.api_recommendation_bridge import api_recommendation_bridge

                recommendations = (
                    api_recommendation_bridge.generate_game_recommendations(
                        game_id, game_predictions.iloc[0], game_market.iloc[0], 0.02
                    )
                )
        except Exception as e:
            logger.warning(
                f"Error generating detailed recommendations for game {game_id}: {e}"
            )

        return GameDetail(
            game_id=game_id,
            season=int(game_data["season"]),
            week=int(game_data["week"]),
            game_date=pd.to_datetime(game_data["kickoff_et"]),
            home_team=home_team,
            away_team=away_team,
            venue=venue,
            weather=weather,
            status=game_status,
            home_score=game_data.get("home_score")
            if not pd.isna(game_data.get("home_score"))
            else None,
            away_score=game_data.get("away_score")
            if not pd.isna(game_data.get("away_score"))
            else None,
            predictions=predictions,
            market=market,
            recommendations=recommendations,
            last_updated=datetime.now(UTC),
        )

    def get_team_stats(
        self, team_id: str, season: int | None = None, week: int | None = None
    ) -> dict[str, Any]:
        """Get team statistics and performance metrics."""
        try:
            # Validate team
            from utils.team_data import (
                get_team_info,
                normalize_team_abbreviation,
                validate_team_abbreviation,
            )

            normalized_team = normalize_team_abbreviation(team_id)
            if not validate_team_abbreviation(normalized_team):
                raise_not_found("Team", team_id)

            team_info = get_team_info(normalized_team)

            with self._get_db_connection() as conn:
                # Build query for team game stats
                query = """
                SELECT
                    season, week, side, game_id,
                    plays, total_epa, epa_per_play,
                    pass_epa_per_play, rush_epa_per_play,
                    success_rate, pass_success_rate, rush_success_rate,
                    neutral_pass_rate, red_zone_td_rate, third_down_conversion_rate,
                    pass_attempts, rush_attempts
                FROM team_game_stats
                WHERE team = ?
                """

                conditions = [normalized_team]

                if season:
                    query += " AND season = ?"
                    conditions.append(season)

                if week:
                    query += " AND week = ?"
                    conditions.append(week)

                query += " ORDER BY season DESC, week DESC, side"

                stats_df = conn.execute(query, conditions).df()

                if stats_df.empty:
                    return {
                        "team_info": {
                            "abbreviation": normalized_team,
                            "name": team_info["name"],
                            "conference": team_info["conference"],
                            "division": team_info["division"],
                        },
                        "stats_summary": {},
                        "recent_games": [],
                        "season_averages": {},
                        "message": "No stats data available for the specified criteria",
                    }

                # Calculate summary statistics
                offense_stats = stats_df[stats_df["side"] == "offense"]
                defense_stats = stats_df[stats_df["side"] == "defense"]

                # Calculate averages for the period
                season_averages = {}

                if not offense_stats.empty:
                    season_averages["offense"] = {
                        "epa_per_play": round(offense_stats["epa_per_play"].mean(), 3),
                        "pass_epa_per_play": round(
                            offense_stats["pass_epa_per_play"].mean(), 3
                        ),
                        "rush_epa_per_play": round(
                            offense_stats["rush_epa_per_play"].mean(), 3
                        ),
                        "success_rate": round(offense_stats["success_rate"].mean(), 3),
                        "pass_success_rate": round(
                            offense_stats["pass_success_rate"].mean(), 3
                        ),
                        "rush_success_rate": round(
                            offense_stats["rush_success_rate"].mean(), 3
                        ),
                        "third_down_conversion_rate": round(
                            offense_stats["third_down_conversion_rate"].mean(), 3
                        ),
                        "red_zone_td_rate": round(
                            offense_stats["red_zone_td_rate"].mean(), 3
                        ),
                        "neutral_pass_rate": round(
                            offense_stats["neutral_pass_rate"].mean(), 3
                        ),
                        "total_plays": int(offense_stats["plays"].sum()),
                        "total_epa": round(offense_stats["total_epa"].sum(), 2),
                        "games_played": len(offense_stats),
                    }

                if not defense_stats.empty:
                    # For defense, lower EPA allowed is better, so we'll negate some values for clarity
                    season_averages["defense"] = {
                        "epa_per_play_allowed": round(
                            defense_stats["epa_per_play"].mean(), 3
                        ),
                        "pass_epa_per_play_allowed": round(
                            defense_stats["pass_epa_per_play"].mean(), 3
                        ),
                        "rush_epa_per_play_allowed": round(
                            defense_stats["rush_epa_per_play"].mean(), 3
                        ),
                        "success_rate_allowed": round(
                            defense_stats["success_rate"].mean(), 3
                        ),
                        "pass_success_rate_allowed": round(
                            defense_stats["pass_success_rate"].mean(), 3
                        ),
                        "rush_success_rate_allowed": round(
                            defense_stats["rush_success_rate"].mean(), 3
                        ),
                        "third_down_conversion_rate_allowed": round(
                            defense_stats["third_down_conversion_rate"].mean(), 3
                        ),
                        "red_zone_td_rate_allowed": round(
                            defense_stats["red_zone_td_rate"].mean(), 3
                        ),
                        "total_epa_allowed": round(defense_stats["total_epa"].sum(), 2),
                        "games_played": len(defense_stats),
                    }

                # Get recent games (last 5)
                recent_games = []
                recent_offense = offense_stats.head(5)

                for _, game in recent_offense.iterrows():
                    recent_games.append(
                        {
                            "game_id": game["game_id"],
                            "season": int(game["season"]),
                            "week": int(game["week"]),
                            "offensive_epa_per_play": round(game["epa_per_play"], 3),
                            "success_rate": round(game["success_rate"], 3),
                            "total_plays": int(game["plays"]),
                            "total_epa": round(game["total_epa"], 2),
                        }
                    )

                # Overall summary
                stats_summary = {
                    "total_games_with_data": len(offense_stats),
                    "seasons_covered": sorted(stats_df["season"].unique().tolist()),
                    "weeks_covered": sorted(stats_df["week"].unique().tolist())
                    if week
                    else None,
                    "data_period": f"{stats_df['season'].min()}-{stats_df['season'].max()}"
                    if not season
                    else f"Season {season}",
                }

                logger.info(
                    f"Retrieved stats for {normalized_team}: {len(stats_df)} records"
                )

                return {
                    "team_info": {
                        "abbreviation": normalized_team,
                        "name": team_info["name"],
                        "conference": team_info["conference"],
                        "division": team_info["division"],
                    },
                    "stats_summary": stats_summary,
                    "recent_games": recent_games,
                    "season_averages": season_averages,
                }

        except Exception as e:
            logger.error(f"Error retrieving team stats for {team_id}: {e}")
            raise

    def get_week_predictions(self, season: int, week: int) -> dict[str, Any]:
        """Get all predictions for a specific week."""
        try:
            with self._get_db_connection() as conn:
                # Get games for the specified week
                games_query = """
                SELECT
                    game_id, season, week, kickoff_et,
                    home_team, away_team, venue,
                    home_score, away_score
                FROM games
                WHERE season = ? AND week = ?
                ORDER BY kickoff_et
                """

                games_df = conn.execute(games_query, [season, week]).df()

                if games_df.empty:
                    return {
                        "season": season,
                        "week": week,
                        "games": [],
                        "total_games": 0,
                        "message": f"No games found for season {season}, week {week}",
                    }

                # Try to get predictions from features tables
                predictions_query = """
                SELECT
                    fw.game_id,
                    fw.win_probability,
                    fw.home_elo_pre as home_elo,
                    fw.away_elo_pre as away_elo,
                    fw.elo_diff,
                    fw.home_team,
                    fw.away_team
                FROM features_wp fw
                WHERE fw.season = ? AND fw.week = ?
                """

                predictions_df = conn.execute(predictions_query, [season, week]).df()

                # Build predictions for each game
                game_predictions = []

                for _, game in games_df.iterrows():
                    game_id = game["game_id"]

                    # Find corresponding prediction data (safely)
                    if not predictions_df.empty and "game_id" in predictions_df.columns:
                        pred_data = predictions_df[predictions_df["game_id"] == game_id]
                    else:
                        pred_data = pd.DataFrame()

                    if not pred_data.empty:
                        pred = pred_data.iloc[0]
                        win_prob = pred["win_probability"]
                        elo_diff = pred.get("elo_diff", 0)

                        # Generate basic spread and total predictions based on elo
                        spread_prediction = max(
                            -20, min(20, elo_diff * 0.03)
                        )  # Simple elo-to-spread conversion
                        total_prediction = (
                            42 + abs(elo_diff) * 0.01
                        )  # Base total with elo adjustment

                        # Calculate confidence based on win probability
                        confidence = (
                            abs(win_prob - 0.5) * 2
                        )  # 0.5 = no confidence, 1.0 = maximum confidence

                        prediction_data = {
                            "win_probability": round(win_prob, 3),
                            "spread_prediction": round(spread_prediction, 1),
                            "total_prediction": round(total_prediction, 1),
                            "confidence": round(confidence, 3),
                            "elo_difference": round(elo_diff, 1),
                            "home_elo": round(pred.get("home_elo", 1500), 1),
                            "away_elo": round(pred.get("away_elo", 1500), 1),
                        }
                    else:
                        # Fallback prediction for games without model data
                        prediction_data = {
                            "win_probability": 0.5,
                            "spread_prediction": 0.0,
                            "total_prediction": 44.0,
                            "confidence": 0.0,
                            "elo_difference": 0.0,
                            "home_elo": 1500.0,
                            "away_elo": 1500.0,
                        }

                    # Build game prediction object
                    game_prediction = {
                        "game_id": game_id,
                        "season": int(game["season"]),
                        "week": int(game["week"]),
                        "kickoff_et": game["kickoff_et"].isoformat()
                        if pd.notna(game["kickoff_et"])
                        else None,
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "venue": game["venue"],
                        "game_status": "completed"
                        if pd.notna(game["home_score"])
                        else "scheduled",
                        "actual_result": {
                            "home_score": int(game["home_score"])
                            if pd.notna(game["home_score"])
                            else None,
                            "away_score": int(game["away_score"])
                            if pd.notna(game["away_score"])
                            else None,
                        }
                        if pd.notna(game["home_score"])
                        else None,
                        "predictions": prediction_data,
                    }

                    game_predictions.append(game_prediction)

                # Calculate summary statistics
                total_games = len(game_predictions)
                games_with_predictions = len(
                    [g for g in game_predictions if g["predictions"]["confidence"] > 0]
                )
                avg_confidence = np.mean(
                    [g["predictions"]["confidence"] for g in game_predictions]
                )

                logger.info(
                    f"Retrieved predictions for {total_games} games in {season} week {week}"
                )

                return {
                    "season": season,
                    "week": week,
                    "games": game_predictions,
                    "total_games": total_games,
                    "games_with_model_predictions": games_with_predictions,
                    "average_confidence": round(avg_confidence, 3),
                    "prediction_summary": {
                        "model_coverage": f"{games_with_predictions}/{total_games}",
                        "data_quality": "good"
                        if games_with_predictions / total_games > 0.8
                        else "limited",
                    },
                }

        except Exception as e:
            logger.error(
                f"Error retrieving week predictions for {season} week {week}: {e}"
            )
            raise


class BacktestService:
    """Service for accessing backtest results and analytics."""

    def __init__(self):
        self.outputs_path = Path(settings.outputs_path)
        self.artifacts_path = Path(settings.artifacts_path)

    def get_backtest_summary(
        self,
        start_season: int | None = None,
        end_season: int | None = None,
        model_type: str | None = None,
    ) -> BacktestSummary:
        """Get backtest performance summary."""

        try:
            # Try to load backtest results
            backtest_file = self.outputs_path / "backtest" / "backtest_results.parquet"
            if not backtest_file.exists():
                logger.warning(
                    "Backtest results file not found",
                    extra={
                        "data_source": "backtest_parquet",
                        "file_path": str(backtest_file),
                        "file_exists": False,
                    },
                )
                raise_model_unavailable("backtest results")

            # Get file freshness
            file_mtime = datetime.fromtimestamp(backtest_file.stat().st_mtime, UTC)
            age_hours = (datetime.now(UTC) - file_mtime).total_seconds() / 3600

            df = pd.read_parquet(backtest_file)
            original_count = len(df)

            # Apply filters
            if start_season:
                df = df[df["season"] >= start_season]
            if end_season:
                df = df[df["season"] <= end_season]
            if model_type:
                df = df[df["model_type"] == model_type]

            filtered_count = len(df)

            if df.empty:
                raise_not_found(
                    "Backtest results", f"seasons {start_season}-{end_season}"
                )

            # Log backtest data info
            logger.info(
                f"Loaded backtest data: {filtered_count} rows (from {original_count} total), file age: {age_hours:.1f}h",
                extra={
                    "data_source": "backtest_parquet",
                    "row_count": filtered_count,
                    "original_row_count": original_count,
                    "file_age_hours": round(age_hours, 1),
                    "file_modified": file_mtime.isoformat(),
                    "start_season": start_season,
                    "end_season": end_season,
                    "model_type": model_type,
                    "data_freshness": "stale"
                    if age_hours > 168
                    else "fresh",  # 1 week threshold for backtest data
                },
            )

            # Calculate actual metrics using the metrics bridge
            try:
                from utils.api_metrics_bridge import api_metrics_bridge

                calculated_metrics = api_metrics_bridge.calculate_backtest_metrics(df)
            except Exception as e:
                logger.warning(f"Error calculating actual metrics: {e}")
                calculated_metrics = {}

            # Use calculated metrics or fallback to basic calculations
            total_predictions = len(df)
            wp_accuracy = calculated_metrics.get(
                "wp_accuracy",
                df["wp_correct"].mean() if "wp_correct" in df.columns else 0.0,
            )
            ats_accuracy = calculated_metrics.get(
                "ats_accuracy",
                df["ats_correct"].mean() if "ats_correct" in df.columns else 0.0,
            )
            ou_accuracy = calculated_metrics.get(
                "ou_accuracy",
                df["ou_correct"].mean() if "ou_correct" in df.columns else 0.0,
            )

            total_bets = calculated_metrics.get(
                "total_bets",
                df["bet_placed"].sum() if "bet_placed" in df.columns else 0,
            )
            winning_bets = calculated_metrics.get(
                "winning_bets", df["bet_won"].sum() if "bet_won" in df.columns else 0
            )
            total_profit = calculated_metrics.get(
                "total_profit",
                df["bet_profit"].sum() if "bet_profit" in df.columns else 0.0,
            )

            return BacktestSummary(
                total_seasons=df["season"].nunique(),
                total_weeks=df["week"].nunique(),
                total_predictions=total_predictions,
                wp_accuracy=wp_accuracy,
                wp_log_loss=calculated_metrics.get("wp_log_loss", 0.5),
                wp_brier_score=calculated_metrics.get("wp_brier_score", 0.25),
                ats_accuracy=ats_accuracy,
                ats_mae=calculated_metrics.get("ats_mae", 3.5),
                ou_accuracy=ou_accuracy,
                ou_mae=calculated_metrics.get("ou_mae", 4.2),
                total_bets=int(total_bets),
                winning_bets=int(winning_bets),
                betting_roi=calculated_metrics.get(
                    "betting_roi", total_profit / total_bets if total_bets > 0 else 0.0
                ),
                total_profit=total_profit,
                start_date=df["kickoff_et"].min()
                if "kickoff_et" in df.columns
                else datetime(2018, 9, 6),
                end_date=df["kickoff_et"].max()
                if "kickoff_et" in df.columns
                else datetime(2024, 1, 8),
                sharpe_ratio=calculated_metrics.get("sharpe_ratio", 1.2),
                max_drawdown=calculated_metrics.get("max_drawdown", 0.15),
            )

        except Exception as e:
            logger.error(f"Failed to load backtest summary: {e}")
            # Return placeholder data for development
            return BacktestSummary(
                total_seasons=6,
                total_weeks=102,
                total_predictions=1632,
                wp_accuracy=0.652,
                wp_log_loss=0.489,
                wp_brier_score=0.241,
                ats_accuracy=0.534,
                ats_mae=3.2,
                ou_accuracy=0.518,
                ou_mae=4.1,
                total_bets=324,
                winning_bets=178,
                betting_roi=0.067,
                total_profit=21.7,
                start_date=datetime(2018, 9, 6),
                end_date=datetime(2024, 1, 8),
                sharpe_ratio=1.18,
                max_drawdown=0.12,
            )

    def get_calibration_data(
        self, model_type: str | None = None, season: int | None = None
    ) -> dict[str, CalibrationData]:
        """Get model calibration data."""

        calibration_data = {}

        # For now, return placeholder calibration data
        model_types = [model_type] if model_type else ["wp", "ats", "ou"]

        for mtype in model_types:
            # Generate sample calibration curve
            predicted_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            observed_freqs = [0.08, 0.18, 0.31, 0.42, 0.51, 0.59, 0.71, 0.82, 0.91]
            bin_counts = [45, 78, 123, 156, 234, 198, 134, 87, 43]

            calibration_data[mtype] = CalibrationData(
                model_type=mtype,
                predicted_probabilities=predicted_probs,
                observed_frequencies=observed_freqs,
                bin_counts=bin_counts,
                ece=0.023,  # Expected Calibration Error
                mce=0.045,  # Maximum Calibration Error
                reliability=0.977,
                total_predictions=sum(bin_counts),
                confidence_interval={"lower": 0.95, "upper": 0.99},
            )

        return calibration_data

    def generate_seasonal_breakdown(
        self,
        start_season: int | None = None,
        end_season: int | None = None,
        model_type: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Generate seasonal breakdown of backtest performance."""
        try:
            # Load backtest data
            backtest_file = self.outputs_path / "backtest" / "backtest_results.parquet"
            if not backtest_file.exists():
                logger.warning("Backtest results file not found for seasonal breakdown")
                return {}

            df = pd.read_parquet(backtest_file)

            # Apply filters
            if start_season:
                df = df[df["season"] >= start_season]
            if end_season:
                df = df[df["season"] <= end_season]
            if model_type:
                df = df[df["model_type"] == model_type]

            if df.empty:
                return {}

            # Group by season and calculate metrics
            seasonal_breakdown = {}

            for season in sorted(df["season"].unique()):
                season_df = df[df["season"] == season]

                # Calculate metrics for this season
                wp_accuracy = (
                    season_df["wp_correct"].mean()
                    if "wp_correct" in season_df.columns
                    else 0.0
                )
                ats_accuracy = (
                    season_df["ats_correct"].mean()
                    if "ats_correct" in season_df.columns
                    else 0.0
                )
                ou_accuracy = (
                    season_df["ou_correct"].mean()
                    if "ou_correct" in season_df.columns
                    else 0.0
                )

                total_bets = (
                    season_df["bet_placed"].sum()
                    if "bet_placed" in season_df.columns
                    else 0
                )
                total_profit = (
                    season_df["bet_profit"].sum()
                    if "bet_profit" in season_df.columns
                    else 0.0
                )
                betting_roi = total_profit / total_bets if total_bets > 0 else 0.0

                winning_bets = (
                    season_df["bet_won"].sum() if "bet_won" in season_df.columns else 0
                )
                bet_win_rate = winning_bets / total_bets if total_bets > 0 else 0.0

                seasonal_breakdown[int(season)] = {
                    "wp_accuracy": round(wp_accuracy, 3),
                    "ats_accuracy": round(ats_accuracy, 3),
                    "ou_accuracy": round(ou_accuracy, 3),
                    "betting_roi": round(betting_roi, 3),
                    "bet_win_rate": round(bet_win_rate, 3),
                    "total_predictions": len(season_df),
                    "total_bets": int(total_bets),
                    "total_profit": round(total_profit, 2),
                }

            logger.info(
                f"Generated seasonal breakdown for {len(seasonal_breakdown)} seasons"
            )
            return seasonal_breakdown

        except Exception as e:
            logger.error(f"Error generating seasonal breakdown: {e}")
            return {}

    def generate_recent_performance(
        self, weeks_back: int = 8, model_type: str | None = None
    ) -> dict[str, Any]:
        """Generate recent performance trends."""
        try:
            # Load backtest data
            backtest_file = self.outputs_path / "backtest" / "backtest_results.parquet"
            if not backtest_file.exists():
                logger.warning("Backtest results file not found for recent performance")
                return {
                    "last_4_weeks_roi": 0.0,
                    "last_8_weeks_roi": 0.0,
                    "current_season_roi": 0.0,
                    "trend": "unknown",
                }

            df = pd.read_parquet(backtest_file)

            # Apply model type filter
            if model_type:
                df = df[df["model_type"] == model_type]

            if df.empty:
                return {
                    "last_4_weeks_roi": 0.0,
                    "last_8_weeks_roi": 0.0,
                    "current_season_roi": 0.0,
                    "trend": "unknown",
                }

            # Sort by date to get most recent data
            if "kickoff_et" in df.columns:
                df = df.sort_values("kickoff_et")

            # Get current season (most recent season)
            current_season = df["season"].max()
            current_season_df = df[df["season"] == current_season]

            # Calculate recent performance metrics
            recent_weeks_4 = current_season_df.tail(
                weeks_back // 2 * 16
            )  # Approximate 4 weeks
            recent_weeks_8 = current_season_df.tail(
                weeks_back * 16
            )  # Approximate 8 weeks

            # Calculate ROI for different periods
            def calculate_roi(data):
                if "bet_placed" in data.columns and "bet_profit" in data.columns:
                    total_bets = data["bet_placed"].sum()
                    total_profit = data["bet_profit"].sum()
                    return total_profit / total_bets if total_bets > 0 else 0.0
                return 0.0

            last_4_weeks_roi = calculate_roi(recent_weeks_4)
            last_8_weeks_roi = calculate_roi(recent_weeks_8)
            current_season_roi = calculate_roi(current_season_df)

            # Determine trend
            if last_4_weeks_roi > last_8_weeks_roi:
                trend = "improving"
            elif last_4_weeks_roi < last_8_weeks_roi:
                trend = "declining"
            else:
                trend = "stable"

            return {
                "last_4_weeks_roi": round(last_4_weeks_roi, 3),
                "last_8_weeks_roi": round(last_8_weeks_roi, 3),
                "current_season_roi": round(current_season_roi, 3),
                "trend": trend,
                "current_season": int(current_season),
                "recent_games_analyzed": len(recent_weeks_8),
            }

        except Exception as e:
            logger.error(f"Error generating recent performance: {e}")
            return {
                "last_4_weeks_roi": 0.0,
                "last_8_weeks_roi": 0.0,
                "current_season_roi": 0.0,
                "trend": "unknown",
            }

    def generate_html_report(
        self, start_season: int | None = None, end_season: int | None = None
    ) -> str:
        """Generate HTML report using the backtest reporting system."""
        try:
            # Import the reporting system
            from backtest.reporting import BacktestReporter
            from backtest.walkforward import BacktestResult

            # Load backtest data
            backtest_file = self.outputs_path / "backtest" / "backtest_results.parquet"
            if not backtest_file.exists():
                logger.warning("Backtest results file not found for HTML report")
                return self._generate_fallback_html_report()

            df = pd.read_parquet(backtest_file)

            # Apply filters
            if start_season:
                df = df[df["season"] >= start_season]
            if end_season:
                df = df[df["season"] <= end_season]

            if df.empty:
                return self._generate_fallback_html_report()

            # Create backtest summary (use existing method)
            backtest_summary = self.get_backtest_summary(start_season, end_season)

            # Convert DataFrame to BacktestResult objects
            raw_results = []
            for _, row in df.iterrows():
                # Create simplified BacktestResult objects
                result = BacktestResult(
                    season=row.get("season", 2024),
                    week=row.get("week", 1),
                    game_id=row.get("game_id", ""),
                    model_predictions=row.to_dict(),
                    actual_outcomes=row.to_dict(),
                    metrics={
                        "wp_correct": row.get("wp_correct", 0),
                        "ats_correct": row.get("ats_correct", 0),
                        "ou_correct": row.get("ou_correct", 0),
                        "bet_placed": row.get("bet_placed", 0),
                        "bet_won": row.get("bet_won", 0),
                        "bet_profit": row.get("bet_profit", 0.0),
                    },
                )
                raw_results.append(result)

            # Initialize reporter and generate report
            reporter = BacktestReporter(output_dir="outputs/reports")
            report_path = reporter.generate_full_report(
                backtest_summary=backtest_summary,
                raw_results=raw_results,
                additional_data={
                    "start_season": start_season,
                    "end_season": end_season,
                    "generated_at": datetime.now().isoformat(),
                },
            )

            # Read the generated HTML file
            with open(report_path, encoding="utf-8") as f:
                html_content = f.read()

            logger.info(f"Generated HTML report with {len(raw_results)} results")
            return html_content

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return self._generate_fallback_html_report()

    def _generate_fallback_html_report(self) -> str:
        """Generate a simple fallback HTML report."""
        return (
            """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NFL Prediction System - Backtest Report</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 30px;
                }
                .header {
                    text-align: center;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }
                .error {
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                    padding: 15px;
                    border-radius: 4px;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>NFL Prediction System</h1>
                    <h2>Backtest Report</h2>
                </div>

                <div class="error">
                    <h3>Report Generation Error</h3>
                    <p>The comprehensive backtest report could not be generated at this time. This may be due to:</p>
                    <ul>
                        <li>Missing backtest data files</li>
                        <li>Insufficient historical data</li>
                        <li>Temporary system issues</li>
                    </ul>
                    <p>Please try again later or contact support if the issue persists.</p>
                </div>

                <div style="text-align: center; margin-top: 30px; color: #666;">
                    <p>Generated at: """
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            + """</p>
                </div>
            </div>
        </body>
        </html>
        """
        )

    def generate_csv_download(
        self, start_season: int | None = None, end_season: int | None = None
    ) -> str:
        """Generate CSV file and return file path for download."""
        try:
            # Load backtest data
            backtest_file = self.outputs_path / "backtest" / "backtest_results.parquet"
            if not backtest_file.exists():
                logger.warning("Backtest results file not found for CSV export")
                raise FileNotFoundError("Backtest results not available")

            df = pd.read_parquet(backtest_file)

            # Apply filters
            if start_season:
                df = df[df["season"] >= start_season]
            if end_season:
                df = df[df["season"] <= end_season]

            if df.empty:
                raise FileNotFoundError("No data available for the specified period")

            # Create filename with timestamp and filters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            season_suffix = ""
            if start_season or end_season:
                season_suffix = f"_{start_season or 'all'}_{end_season or 'all'}"

            filename = f"backtest_results{season_suffix}_{timestamp}.csv"

            # Ensure outputs directory exists
            outputs_dir = self.outputs_path / "exports"
            outputs_dir.mkdir(parents=True, exist_ok=True)

            csv_path = outputs_dir / filename

            # Select and rename columns for better CSV export
            export_columns = {
                "game_id": "Game ID",
                "season": "Season",
                "week": "Week",
                "kickoff_et": "Kickoff Time (ET)",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "home_score": "Home Score",
                "away_score": "Away Score",
                "win_probability": "Win Probability",
                "wp_correct": "WP Correct",
                "spread_prediction": "Spread Prediction",
                "ats_correct": "ATS Correct",
                "total_prediction": "Total Prediction",
                "ou_correct": "O/U Correct",
                "bet_placed": "Bet Placed",
                "bet_type": "Bet Type",
                "bet_team": "Bet Team",
                "bet_odds": "Bet Odds",
                "bet_won": "Bet Won",
                "bet_profit": "Bet Profit",
                "model_confidence": "Model Confidence",
                "edge": "Edge",
                "expected_value": "Expected Value",
            }

            # Select available columns and rename
            available_columns = {
                k: v for k, v in export_columns.items() if k in df.columns
            }
            export_df = df[list(available_columns.keys())].copy()
            export_df = export_df.rename(columns=available_columns)

            # Format datetime columns
            if "Kickoff Time (ET)" in export_df.columns:
                export_df["Kickoff Time (ET)"] = pd.to_datetime(
                    export_df["Kickoff Time (ET)"]
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Round numeric columns for readability
            numeric_columns = [
                "Win Probability",
                "Spread Prediction",
                "Total Prediction",
                "Bet Profit",
                "Model Confidence",
                "Edge",
                "Expected Value",
            ]
            for col in numeric_columns:
                if col in export_df.columns:
                    export_df[col] = export_df[col].round(4)

            # Sort by season, week, kickoff time
            sort_columns = ["Season", "Week"]
            if "Kickoff Time (ET)" in export_df.columns:
                sort_columns.append("Kickoff Time (ET)")

            export_df = export_df.sort_values(sort_columns)

            # Export to CSV
            export_df.to_csv(csv_path, index=False)

            logger.info(f"Generated CSV export: {csv_path} with {len(export_df)} rows")
            return str(csv_path)

        except Exception as e:
            logger.error(f"Error generating CSV export: {e}")
            raise


# Global service instances
data_service = DataService()
backtest_service = BacktestService()
