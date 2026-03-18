"""
Similar Games Engine.

This module finds similar games based on multiple criteria including:
- Team matchups (same teams, conference matchups, division matchups)
- Venue characteristics (same venue, similar roof type, surface, capacity)
- Weather conditions (temperature, wind, precipitation patterns)
- Game context (prime time, playoff games, season timing)
- Historical performance (Elo ratings, scoring patterns)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimilarGameCriteria:
    """Criteria for finding similar games."""

    max_results: int = 10
    min_similarity_score: float = 0.3
    max_seasons_back: int = 5
    weight_teams: float = 0.25
    weight_venue: float = 0.20
    weight_weather: float = 0.25
    weight_context: float = 0.15
    weight_elo: float = 0.15


@dataclass
class SimilarGame:
    """A similar game with similarity metadata."""

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    venue: str
    game_date: datetime
    similarity_score: float
    similarity_breakdown: dict[str, float]
    home_score: int | None = None
    away_score: int | None = None


class SimilarGamesEngine:
    """Engine for finding similar NFL games based on multiple criteria."""

    def __init__(self, db_path: str | None = None):
        """Initialize the similar games engine."""
        if db_path is None:
            # Default to project data directory
            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "nfl_predictions.duckdb"

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            logger.warning(f"Database not found at {self.db_path}")

    def find_similar_games(
        self, target_game_id: str, criteria: SimilarGameCriteria | None = None
    ) -> list[SimilarGame]:
        """
        Find games similar to the target game.

        Args:
            target_game_id: The game to find similar games for
            criteria: Similarity criteria and weights

        Returns:
            List of similar games sorted by similarity score
        """
        if criteria is None:
            criteria = SimilarGameCriteria()

        try:
            with duckdb.connect(str(self.db_path)) as conn:
                # Get target game details
                target_game = self._get_game_details(conn, target_game_id)
                if target_game is None:
                    logger.warning(f"Target game {target_game_id} not found")
                    return []

                # Find candidate games (excluding the target game)
                candidates = self._get_candidate_games(conn, target_game, criteria)
                if candidates.empty:
                    logger.info("No candidate games found")
                    return []

                # Calculate similarity scores for each candidate
                similarities = []
                for _, candidate in candidates.iterrows():
                    similarity = self._calculate_similarity(
                        target_game, candidate, criteria
                    )
                    if similarity["total_score"] >= criteria.min_similarity_score:
                        similarities.append(
                            {
                                "game_id": candidate["game_id"],
                                "season": candidate["season"],
                                "week": candidate["week"],
                                "home_team": candidate["home_team"],
                                "away_team": candidate["away_team"],
                                "venue": candidate["venue"],
                                "game_date": candidate["kickoff_et"],
                                "home_score": candidate.get("home_score"),
                                "away_score": candidate.get("away_score"),
                                "similarity_score": similarity["total_score"],
                                "similarity_breakdown": similarity["breakdown"],
                            }
                        )

                # Sort by similarity score and return top results
                similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

                similar_games = []
                for sim in similarities[: criteria.max_results]:
                    similar_games.append(
                        SimilarGame(
                            game_id=sim["game_id"],
                            season=sim["season"],
                            week=sim["week"],
                            home_team=sim["home_team"],
                            away_team=sim["away_team"],
                            venue=sim["venue"],
                            game_date=sim["game_date"],
                            similarity_score=sim["similarity_score"],
                            similarity_breakdown=sim["similarity_breakdown"],
                            home_score=sim["home_score"],
                            away_score=sim["away_score"],
                        )
                    )

                logger.info(
                    f"Found {len(similar_games)} similar games for {target_game_id} "
                    f"(top score: {similar_games[0].similarity_score:.3f})"
                    if similar_games
                    else f"No similar games found for {target_game_id}"
                )

                return similar_games

        except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
            logger.error(f"Error finding similar games for {target_game_id}: {e}")
            return []

    def _get_game_details(
        self, conn: duckdb.DuckDBPyConnection, game_id: str
    ) -> pd.Series | None:
        """Get comprehensive details for a target game."""
        query = """
        SELECT
            g.*,
            wf.temp_f, wf.wind_mph, wf.precip_mm, wf.weather_severity_score,
            wf.weather_condition, wf.weather_affects_game,
            cf.venue_outdoor, cf.venue_indoor, cf.venue_retractable,
            cf.venue_elevation_ft, cf.venue_capacity,
            cf.thursday_game, cf.monday_game, cf.saturday_game,
            v.roof_type, v.surface, v.capacity as venue_capacity_actual,
            v.climate_zone, v.elevation_ft as venue_elevation_actual,
            eh.home_rating_pre, eh.away_rating_pre
        FROM games g
        LEFT JOIN weather_features wf ON g.game_id = wf.game_id
        LEFT JOIN contextual_features cf ON g.game_id = cf.game_id
        LEFT JOIN venues v ON g.venue = v.venue_name
        LEFT JOIN elo_rating_history eh ON g.game_id = eh.game_id
        WHERE g.game_id = ?
        """

        result = conn.execute(query, [game_id]).df()
        if result.empty:
            return None

        return result.iloc[0]

    def _get_candidate_games(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_game: pd.Series,
        criteria: SimilarGameCriteria,
    ) -> pd.DataFrame:
        """Get candidate games for similarity comparison."""
        # Calculate season range
        target_season = target_game["season"]
        min_season = max(
            2018, target_season - criteria.max_seasons_back
        )  # Data availability limit

        query = """
        SELECT
            g.*,
            wf.temp_f, wf.wind_mph, wf.precip_mm, wf.weather_severity_score,
            wf.weather_condition, wf.weather_affects_game,
            cf.venue_outdoor, cf.venue_indoor, cf.venue_retractable,
            cf.venue_elevation_ft, cf.venue_capacity,
            cf.thursday_game, cf.monday_game, cf.saturday_game,
            v.roof_type, v.surface, v.capacity as venue_capacity_actual,
            v.climate_zone, v.elevation_ft as venue_elevation_actual,
            eh.home_rating_pre, eh.away_rating_pre
        FROM games g
        LEFT JOIN weather_features wf ON g.game_id = wf.game_id
        LEFT JOIN contextual_features cf ON g.game_id = cf.game_id
        LEFT JOIN venues v ON g.venue = v.venue_name
        LEFT JOIN elo_rating_history eh ON g.game_id = eh.game_id
        WHERE g.season >= ?
        AND g.game_id != ?
        AND g.home_score IS NOT NULL  -- Only completed games
        ORDER BY g.season DESC, g.week DESC
        """

        return conn.execute(query, [min_season, target_game["game_id"]]).df()

    def _calculate_similarity(
        self, target: pd.Series, candidate: pd.Series, criteria: SimilarGameCriteria
    ) -> dict[str, Any]:
        """Calculate similarity score between target and candidate game."""
        breakdown = {}

        # Team similarity
        team_score = self._calculate_team_similarity(target, candidate)
        breakdown["teams"] = team_score

        # Venue similarity
        venue_score = self._calculate_venue_similarity(target, candidate)
        breakdown["venue"] = venue_score

        # Weather similarity
        weather_score = self._calculate_weather_similarity(target, candidate)
        breakdown["weather"] = weather_score

        # Context similarity (game timing, type)
        context_score = self._calculate_context_similarity(target, candidate)
        breakdown["context"] = context_score

        # Elo rating similarity
        elo_score = self._calculate_elo_similarity(target, candidate)
        breakdown["elo"] = elo_score

        # Calculate weighted total
        total_score = (
            team_score * criteria.weight_teams
            + venue_score * criteria.weight_venue
            + weather_score * criteria.weight_weather
            + context_score * criteria.weight_context
            + elo_score * criteria.weight_elo
        )

        return {"total_score": total_score, "breakdown": breakdown}

    def _calculate_team_similarity(
        self, target: pd.Series, candidate: pd.Series
    ) -> float:
        """Calculate team matchup similarity."""
        target_home, target_away = target["home_team"], target["away_team"]
        candidate_home, candidate_away = candidate["home_team"], candidate["away_team"]

        # Exact same matchup (any venue)
        if (target_home == candidate_home and target_away == candidate_away) or (
            target_home == candidate_away and target_away == candidate_home
        ):
            return 1.0

        # Same teams, different venues
        target_teams = {target_home, target_away}
        candidate_teams = {candidate_home, candidate_away}
        if target_teams == candidate_teams:
            return 0.8

        # One team in common
        common_teams = target_teams.intersection(candidate_teams)
        if len(common_teams) == 1:
            return 0.4

        # Conference/division similarity would require team metadata
        # For now, different teams get base score
        return 0.1

    def _calculate_venue_similarity(
        self, target: pd.Series, candidate: pd.Series
    ) -> float:
        """Calculate venue similarity."""
        # Same venue
        if target.get("venue") == candidate.get("venue"):
            return 1.0

        score = 0.0

        # Roof type similarity
        target_roof = target.get("roof_type", target.get("venue_roof"))
        candidate_roof = candidate.get("roof_type", candidate.get("venue_roof"))

        if pd.notna(target_roof) and pd.notna(candidate_roof):
            if target_roof == candidate_roof:
                score += 0.3
            elif (
                target_roof in ["dome", "closed"]
                and candidate_roof
                in [
                    "dome",
                    "closed",
                ]
            ) or (target_roof == "outdoors" and candidate_roof == "outdoors"):
                score += 0.2

        # Surface similarity
        target_surface = target.get("surface")
        candidate_surface = candidate.get("surface")
        if pd.notna(target_surface) and pd.notna(candidate_surface):
            if target_surface == candidate_surface:
                score += 0.2

        # Capacity similarity
        target_capacity = target.get(
            "venue_capacity_actual", target.get("venue_capacity")
        )
        candidate_capacity = candidate.get(
            "venue_capacity_actual", candidate.get("venue_capacity")
        )

        if pd.notna(target_capacity) and pd.notna(candidate_capacity):
            capacity_diff = abs(target_capacity - candidate_capacity)
            capacity_similarity = max(
                0, 1 - capacity_diff / 30000
            )  # Normalize by ~30k seats
            score += capacity_similarity * 0.2

        # Elevation similarity
        target_elevation = target.get(
            "venue_elevation_actual", target.get("venue_elevation_ft")
        )
        candidate_elevation = candidate.get(
            "venue_elevation_actual", candidate.get("venue_elevation_ft")
        )

        if pd.notna(target_elevation) and pd.notna(candidate_elevation):
            elevation_diff = abs(target_elevation - candidate_elevation)
            elevation_similarity = max(
                0, 1 - elevation_diff / 5000
            )  # Normalize by 5k feet
            score += elevation_similarity * 0.3

        return min(score, 1.0)

    def _calculate_weather_similarity(
        self, target: pd.Series, candidate: pd.Series
    ) -> float:
        """Calculate weather condition similarity."""
        score = 0.0

        # Temperature similarity
        target_temp = target.get("temp_f")
        candidate_temp = candidate.get("temp_f")

        if pd.notna(target_temp) and pd.notna(candidate_temp):
            temp_diff = abs(target_temp - candidate_temp)
            temp_similarity = max(0, 1 - temp_diff / 40)  # Normalize by 40 degrees
            score += temp_similarity * 0.4

        # Wind similarity
        target_wind = target.get("wind_mph")
        candidate_wind = candidate.get("wind_mph")

        if pd.notna(target_wind) and pd.notna(candidate_wind):
            wind_diff = abs(target_wind - candidate_wind)
            wind_similarity = max(0, 1 - wind_diff / 20)  # Normalize by 20 mph
            score += wind_similarity * 0.3

        # Precipitation similarity
        target_precip = target.get("precip_mm", 0)
        candidate_precip = candidate.get("precip_mm", 0)

        if pd.notna(target_precip) and pd.notna(candidate_precip):
            # Both dry
            if target_precip < 1 and candidate_precip < 1:
                score += 0.2
            # Both wet
            elif target_precip >= 1 and candidate_precip >= 1:
                precip_diff = abs(target_precip - candidate_precip)
                precip_similarity = max(0, 1 - precip_diff / 10)  # Normalize by 10mm
                score += precip_similarity * 0.2

        # Weather severity similarity
        target_severity = target.get("weather_severity_score")
        candidate_severity = candidate.get("weather_severity_score")

        if pd.notna(target_severity) and pd.notna(candidate_severity):
            severity_diff = abs(target_severity - candidate_severity)
            severity_similarity = max(0, 1 - severity_diff)
            score += severity_similarity * 0.1

        return min(score, 1.0)

    def _calculate_context_similarity(
        self, target: pd.Series, candidate: pd.Series
    ) -> float:
        """Calculate game context similarity."""
        score = 0.0

        # Game day similarity
        target_thursday = target.get("thursday_game", 0)
        target_monday = target.get("monday_game", 0)
        target_saturday = target.get("saturday_game", 0)

        candidate_thursday = candidate.get("thursday_game", 0)
        candidate_monday = candidate.get("monday_game", 0)
        candidate_saturday = candidate.get("saturday_game", 0)

        # Prime time games
        if (
            (target_thursday and candidate_thursday)
            or (target_monday and candidate_monday)
            or (target_saturday and candidate_saturday)
        ):
            score += 0.3
        # Regular Sunday games
        elif not any([target_thursday, target_monday, target_saturday]) and not any(
            [candidate_thursday, candidate_monday, candidate_saturday]
        ):
            score += 0.2

        # Season timing similarity (early, mid, late season)
        target_week = target.get("week", 0)
        candidate_week = candidate.get("week", 0)

        if target_week and candidate_week:
            week_diff = abs(target_week - candidate_week)
            if week_diff <= 2:
                score += 0.3
            elif week_diff <= 4:
                score += 0.2
            elif week_diff <= 8:
                score += 0.1

        # Season type similarity
        target_type = target.get("season_type", target.get("game_type"))
        candidate_type = candidate.get("season_type", candidate.get("game_type"))

        if target_type == candidate_type:
            score += 0.4

        return min(score, 1.0)

    def _calculate_elo_similarity(
        self, target: pd.Series, candidate: pd.Series
    ) -> float:
        """Calculate Elo rating similarity."""
        target_home_elo = target.get("home_rating_pre")
        target_away_elo = target.get("away_rating_pre")
        candidate_home_elo = candidate.get("home_rating_pre")
        candidate_away_elo = candidate.get("away_rating_pre")

        if not all(
            pd.notna(
                [
                    target_home_elo,
                    target_away_elo,
                    candidate_home_elo,
                    candidate_away_elo,
                ]
            )
        ):
            return 0.5  # Neutral score if Elo data missing

        # Calculate Elo differences
        target_elo_diff = target_home_elo - target_away_elo
        candidate_elo_diff = candidate_home_elo - candidate_away_elo

        # Similarity based on how close the Elo differences are
        elo_diff_similarity = max(
            0, 1 - abs(target_elo_diff - candidate_elo_diff) / 200
        )

        # Also consider absolute Elo levels (strength of teams)
        target_avg_elo = (target_home_elo + target_away_elo) / 2
        candidate_avg_elo = (candidate_home_elo + candidate_away_elo) / 2

        avg_elo_similarity = max(0, 1 - abs(target_avg_elo - candidate_avg_elo) / 200)

        # Weighted combination
        return elo_diff_similarity * 0.7 + avg_elo_similarity * 0.3


# Global instance for easy access
_similar_games_engine = None


def get_similar_games_engine() -> SimilarGamesEngine:
    """Get global similar games engine instance."""
    global _similar_games_engine
    if _similar_games_engine is None:
        _similar_games_engine = SimilarGamesEngine()
    return _similar_games_engine
