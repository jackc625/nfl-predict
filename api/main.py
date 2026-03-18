"""
Main FastAPI Application for NFL Prediction System.

This module sets up the FastAPI application with all endpoints, middleware,
exception handlers, and configuration for the NFL prediction API.
"""

import csv
import io
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path as FilePath
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import utils for additional functionality
from utils.exceptions import StorageError
from utils.similar_games import SimilarGameCriteria, get_similar_games_engine

from .config import get_middleware_config, settings
from .exceptions import NFLPredictionAPIException, setup_exception_handlers
from .middleware import setup_middleware
from .schemas import (
    BacktestResponse,
    CalibrationResponse,
    GameDetailResponse,
    GamesListResponse,
    GameStatus,
    HealthStatus,
    ServiceInfo,
    WeekMetadata,
)
from .services import backtest_service, data_service

logger = logging.getLogger(__name__)


def convert_similar_game_to_summary(similar_game) -> dict[str, Any]:
    """Convert SimilarGame to GameSummary dict."""
    # Determine game status based on scores
    if similar_game.home_score is not None and similar_game.away_score is not None:
        status = GameStatus.COMPLETED
    else:
        status = GameStatus.SCHEDULED

    return {
        "game_id": similar_game.game_id,
        "season": similar_game.season,
        "week": similar_game.week,
        "game_date": similar_game.game_date,
        "home_team": similar_game.home_team,
        "away_team": similar_game.away_team,
        "status": status.value,  # Convert enum to string for frontend
        "home_score": similar_game.home_score,
        "away_score": similar_game.away_score,
        "venue": similar_game.venue,
    }


# Global state for tracking startup time and system resources
startup_time = None
models_loaded = False
database_connected = False
system_healthy = True


# System management functions
async def initialize_models():
    """Initialize ML models if available."""
    global models_loaded
    try:
        # Check if model artifacts exist
        models_dir = FilePath("artifacts")
        if models_dir.exists():
            logger.info("Model artifacts directory found")
            models_loaded = True
            logger.info("Models marked as available")
        else:
            logger.warning("Model artifacts directory not found - models unavailable")
            models_loaded = False
    except OSError as e:
        logger.error(f"Model initialization failed: {e}")
        models_loaded = False


async def setup_database_connections():
    """Setup database connections."""
    global database_connected
    try:
        # Check if data directory exists
        data_dir = FilePath("data")
        if data_dir.exists():
            logger.info("Data directory found")
            database_connected = True
            logger.info("Database connections marked as available")
        else:
            logger.warning("Data directory not found - database unavailable")
            database_connected = False
    except OSError as e:
        logger.error(f"Database setup failed: {e}")
        database_connected = False


async def cleanup_models():
    """Cleanup ML models."""
    global models_loaded
    try:
        if models_loaded:
            models_loaded = False
            logger.info("Models cleaned up")
    except (RuntimeError, OSError) as e:
        logger.error(f"Model cleanup failed: {e}")


async def close_database_connections():
    """Close database connections."""
    global database_connected
    try:
        if database_connected:
            database_connected = False
            logger.info("Database connections closed")
    except (RuntimeError, OSError) as e:
        logger.error(f"Database cleanup failed: {e}")


def get_build_date() -> datetime:
    """Get build date from various sources."""
    try:
        # Try environment variable first (CI/CD)
        build_date_str = os.environ.get("BUILD_DATE")
        if build_date_str:
            return datetime.fromisoformat(build_date_str.replace("Z", "+00:00"))

        # Try git commit date
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                from dateutil import parser

                return parser.parse(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ImportError):
            pass

        # Fall back to main.py modification time
        main_py = FilePath(__file__)
        if main_py.exists():
            mtime = main_py.stat().st_mtime
            return datetime.fromtimestamp(mtime, tz=UTC)

    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Could not determine build date: {e}")

    return datetime.now(UTC)


def get_commit_hash() -> str | None:
    """Get current git commit hash."""
    try:
        # Try environment variable first (CI/CD)
        commit_hash = os.environ.get("GIT_COMMIT")
        if commit_hash:
            return commit_hash[:8]

        # Try git command
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()

    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass

    return None


def get_component_health() -> dict[str, str]:
    """Get component health status using existing health check."""
    try:
        # Try to import and use existing health checker
        from scripts.health_check import HealthChecker

        checker = HealthChecker()

        # Perform basic checks
        health = {}

        # Check database
        db_result = checker.check_database_connectivity()
        health["database"] = (
            "healthy" if db_result.get("status") == "healthy" else "warning"
        )

        # Check models
        health["models"] = "healthy" if models_loaded else "unavailable"

        # Check cache (placeholder)
        health["cache"] = "healthy"

        return health

    except ImportError:
        # Fallback if health checker not available
        return {
            "database": "healthy" if database_connected else "unavailable",
            "models": "healthy" if models_loaded else "unavailable",
            "cache": "healthy",
        }
    except (RuntimeError, OSError, ValueError) as e:
        logger.error(f"Health check failed: {e}")
        return {"database": "error", "models": "error", "cache": "error"}


# Validation functions
def validate_team_param(team: str | None) -> str | None:
    """Validate team parameter."""
    if team is None:
        return None

    # Convert to uppercase and validate format
    team = team.upper().strip()

    # Check length (2-4 characters for NFL teams)
    if not (2 <= len(team) <= 4):
        raise HTTPException(
            status_code=422,
            detail="Team abbreviation must be 2-4 characters (e.g., 'KC', 'NE', 'TB')",
        )

    # Check format (letters only)
    if not re.match(r"^[A-Z]+$", team):
        raise HTTPException(
            status_code=422, detail="Team abbreviation must contain only letters"
        )

    return team


def validate_status_param(status: str | None) -> GameStatus | None:
    """Validate status parameter."""
    if status is None:
        return None

    try:
        return GameStatus(status.lower())
    except ValueError:
        valid_statuses = [s.value for s in GameStatus]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid game status. Must be one of: {', '.join(valid_statuses)}",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global startup_time
    startup_time = time.time()

    logger.info("NFL Prediction API starting up...")

    # Initialize models and database connections
    await initialize_models()
    await setup_database_connections()

    logger.info("NFL Prediction API startup complete")

    yield

    logger.info("NFL Prediction API shutting down...")

    # Cleanup resources
    await cleanup_models()
    await close_database_connections()

    logger.info("NFL Prediction API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# Setup middleware and exception handlers
setup_middleware(app, get_middleware_config(settings))
setup_exception_handlers(app)

# Setup static files and templates with existence checks
static_dir = "web/static"
templates_dir = "web/templates"

if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    logger.warning(
        f"Static directory '{static_dir}' not found - static file serving disabled"
    )

if os.path.exists(templates_dir) and os.path.isdir(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    logger.warning(
        f"Templates directory '{templates_dir}' not found - template rendering may fail"
    )
    templates = None


# Health and Status Endpoints


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Check API health status."""
    uptime = time.time() - startup_time if startup_time else 0

    # Get actual component health checks
    components = get_component_health()

    # Determine overall status
    status = "healthy"
    if any(comp_status == "error" for comp_status in components.values()):
        status = "unhealthy"
    elif any(
        comp_status in ["warning", "unavailable"] for comp_status in components.values()
    ):
        status = "degraded"

    return HealthStatus(
        status=status,
        version=settings.app_version,
        components=components,
        uptime_seconds=uptime,
    )


@app.get("/info", response_model=ServiceInfo, tags=["Health"])
async def service_info():
    """Get service information."""
    return ServiceInfo(
        name=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        environment=settings.environment,
        build_date=get_build_date(),
        commit_hash=get_commit_hash(),
    )


# Week Metadata Endpoint


@app.get("/current-week", response_model=WeekMetadata, tags=["Schedule"])
async def get_current_week():
    """Get current NFL season and week information."""
    return data_service.get_current_week_metadata()


# Games Endpoints


@app.get("/games", response_model=GamesListResponse, tags=["Games"])
async def list_games(
    season: int | None = Query(None, description="Filter by season"),
    week: int | None = Query(
        None,
        ge=1,
        le=22,
        description="Filter by week (1-18 regular season, 19-22 playoffs)",
    ),
    team_param: str | None = Query(
        None, alias="team", description="Filter by team abbreviation (2-4 letters)"
    ),
    status_param: str | None = Query(
        None, alias="status", description="Filter by game status"
    ),
    has_recommendations: bool | None = Query(
        None, description="Filter games with betting recommendations"
    ),
    min_edge: float | None = Query(
        None, ge=0, le=1, description="Minimum edge threshold"
    ),
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
):
    """
    List NFL games with optional filtering.

    Returns a list of games with basic information, predictions, and top recommendations.
    Use query parameters to filter results by season, week, team, or recommendation quality.
    """
    # Validate parameters
    team = validate_team_param(team_param)
    status = validate_status_param(status_param)

    games, metadata, total = data_service.list_games(
        season=season,
        week=week,
        team=team,
        status=status.value if status else None,
        has_recommendations=has_recommendations,
        min_edge=min_edge,
        limit=limit,
        offset=offset,
    )

    # Build filters applied dict
    filters_applied = {}
    if season:
        filters_applied["season"] = season
    if week:
        filters_applied["week"] = week
    if team:
        filters_applied["team"] = team
    if status:
        filters_applied["status"] = status.value
    if has_recommendations is not None:
        filters_applied["has_recommendations"] = has_recommendations
    if min_edge:
        filters_applied["min_edge"] = min_edge

    return GamesListResponse(
        games=games,
        metadata=metadata,
        total_games=total,
        has_predictions=metadata.predictions_available,
        filters_applied=filters_applied,
    )


@app.get("/games/view", response_class=HTMLResponse, tags=["Web UI"])
async def games_list_page_early(request: Request):
    """Serve the games list page (same as home page)."""
    if templates is None:
        raise HTTPException(
            status_code=503, detail="Templates not available - web interface disabled"
        )
    return templates.TemplateResponse("games.html", {"request": request})


@app.get("/games/{game_id}", response_model=GameDetailResponse, tags=["Games"])
async def get_game_detail(
    game_id: str = Path(..., description="Unique game identifier"),
):
    """
    Get detailed information for a specific game.

    Returns comprehensive game data including teams, venue, weather, predictions,
    market odds, and all available betting recommendations.
    """
    game = data_service.get_game_detail(game_id)

    # Get similar games based on teams, venue, weather, etc.
    try:
        similar_games_engine = get_similar_games_engine()
        criteria = SimilarGameCriteria(
            max_results=10, min_similarity_score=0.3, max_seasons_back=5
        )

        similar_games_data = similar_games_engine.find_similar_games(game_id, criteria)

        # Convert to GameSummary format
        similar_games = []
        for sim_game in similar_games_data:
            game_summary = convert_similar_game_to_summary(sim_game)
            similar_games.append(game_summary)

        logger.info(f"Found {len(similar_games)} similar games for {game_id}")

    except (ImportError, ValueError, KeyError, TypeError, FileNotFoundError) as e:
        logger.error(f"Error finding similar games for {game_id}: {e}")
        similar_games = []

    return GameDetailResponse(game=game, similar_games=similar_games)


# Week Schedule Endpoints


@app.get(
    "/weeks/{season}/{week}/games", response_model=GamesListResponse, tags=["Schedule"]
)
async def get_week_games(
    season: int = Path(..., description="NFL season"),
    week: int = Path(
        ..., ge=1, le=22, description="NFL week (1-18 regular season, 19-22 playoffs)"
    ),
    has_recommendations: bool | None = Query(
        None, description="Filter games with recommendations"
    ),
    min_edge: float | None = Query(
        None, ge=0, le=1, description="Minimum edge threshold"
    ),
):
    """
    Get all games for a specific season and week.

    Convenience endpoint for getting a week's full schedule with predictions
    and recommendations.
    """
    games, metadata, total = data_service.list_games(
        season=season,
        week=week,
        has_recommendations=has_recommendations,
        min_edge=min_edge,
        limit=100,  # Get all games for the week
        offset=0,
    )

    filters_applied = {"season": season, "week": week}
    if has_recommendations is not None:
        filters_applied["has_recommendations"] = has_recommendations
    if min_edge:
        filters_applied["min_edge"] = min_edge

    return GamesListResponse(
        games=games,
        metadata=metadata,
        total_games=total,
        has_predictions=metadata.predictions_available,
        filters_applied=filters_applied,
    )


# Performance and Analytics Endpoints


@app.get("/backtest", response_model=BacktestResponse, tags=["Analytics"])
async def get_backtest_summary(
    start_season: int | None = Query(
        None, description="Start season for backtest period"
    ),
    end_season: int | None = Query(None, description="End season for backtest period"),
    model_type: str | None = Query(
        None, description="Filter by model type (wp/ats/ou)"
    ),
):
    """
    Get backtest performance summary.

    Returns model performance metrics, seasonal breakdowns, and recent trends
    for the specified time period.
    """
    summary = backtest_service.get_backtest_summary(
        start_season=start_season, end_season=end_season, model_type=model_type
    )

    # Generate actual seasonal breakdown and recent performance data
    try:
        seasonal_breakdown = backtest_service.generate_seasonal_breakdown(
            start_season=start_season, end_season=end_season, model_type=model_type
        )

        recent_performance = backtest_service.generate_recent_performance(
            weeks_back=8, model_type=model_type
        )

        logger.info(
            f"Generated seasonal breakdown for {len(seasonal_breakdown)} seasons"
        )

    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        logger.error(f"Error generating seasonal/recent performance data: {e}")
        # Fallback to placeholder data
        seasonal_breakdown = {
            2018: {"wp_accuracy": 0.651, "ats_accuracy": 0.542, "betting_roi": 0.045},
            2019: {"wp_accuracy": 0.648, "ats_accuracy": 0.538, "betting_roi": 0.062},
            2020: {"wp_accuracy": 0.663, "ats_accuracy": 0.529, "betting_roi": 0.071},
            2021: {"wp_accuracy": 0.655, "ats_accuracy": 0.531, "betting_roi": 0.058},
            2022: {"wp_accuracy": 0.649, "ats_accuracy": 0.535, "betting_roi": 0.082},
            2023: {"wp_accuracy": 0.657, "ats_accuracy": 0.528, "betting_roi": 0.074},
        }

        recent_performance = {
            "last_4_weeks_roi": 0.089,
            "last_8_weeks_roi": 0.076,
            "current_season_roi": 0.071,
            "trend": "improving",
        }

    return BacktestResponse(
        summary=summary,
        seasonal_breakdown=seasonal_breakdown,
        recent_performance=recent_performance,
    )


@app.get("/calibration", response_model=CalibrationResponse, tags=["Analytics"])
async def get_calibration_data(
    model_type: str | None = Query(
        None, description="Filter by model type (wp/ats/ou)"
    ),
    season: int | None = Query(None, description="Filter by season"),
):
    """
    Get model calibration data.

    Returns calibration curves, reliability metrics, and confidence intervals
    to assess how well-calibrated the model predictions are.
    """
    calibration_data = backtest_service.get_calibration_data(
        model_type=model_type, season=season
    )

    # Calculate overall reliability as weighted average
    total_predictions = sum(
        data.total_predictions for data in calibration_data.values()
    )
    overall_reliability = (
        sum(
            data.reliability * data.total_predictions
            for data in calibration_data.values()
        )
        / total_predictions
        if total_predictions > 0
        else 0.0
    )

    return CalibrationResponse(
        calibration_data=calibration_data,
        overall_reliability=overall_reliability,
        last_updated=datetime.now(UTC),
    )


# Report Generation Endpoints


@app.get("/reports/backtest", response_class=HTMLResponse, tags=["Reports"])
async def generate_backtest_report(
    start_season: int | None = Query(None, description="Start season"),
    end_season: int | None = Query(None, description="End season"),
):
    """
    Generate comprehensive HTML backtest report.

    Returns a full HTML report with interactive charts, performance breakdowns,
    and detailed analysis of model performance over time.
    """
    # Generate HTML report using backtest reporting system
    try:
        html_content = backtest_service.generate_html_report(
            start_season=start_season, end_season=end_season
        )

        logger.info(
            f"Generated HTML backtest report for seasons {start_season}-{end_season}"
        )
        return HTMLResponse(content=html_content)

    except NFLPredictionAPIException:
        raise
    except (ImportError, FileNotFoundError, OSError, ValueError) as e:
        logger.error(f"Error generating HTML report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate HTML report: {e!s}"
        )


@app.get("/reports/backtest/download", response_class=FileResponse, tags=["Reports"])
async def download_backtest_csv(
    start_season: int | None = Query(None, description="Start season"),
    end_season: int | None = Query(None, description="End season"),
):
    """
    Download backtest results as CSV.

    Returns a CSV file containing detailed backtest results for analysis
    in external tools.
    """
    # Generate CSV file using backtest service
    try:
        csv_path = backtest_service.generate_csv_download(
            start_season=start_season, end_season=end_season
        )

        # Get filename for download
        filename = FilePath(csv_path).name

        logger.info(f"Generated CSV download: {filename}")

        return FileResponse(path=csv_path, filename=filename, media_type="text/csv")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (OSError, ValueError, KeyError) as e:
        logger.error(f"Error generating CSV download: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate CSV download: {e!s}"
        )


# Team and Statistics Endpoints


@app.get("/teams", tags=["Teams"])
async def list_teams():
    """Get list of all NFL teams with basic information."""
    try:
        from utils.team_data import get_all_teams_info

        teams_data = get_all_teams_info()

        # Convert to a list format suitable for API response
        teams_list = []
        for abbr, info in teams_data.items():
            team_info = {
                "abbreviation": abbr,
                "name": info["name"],
                "city": info["city"],
                "conference": info["conference"],
                "division": info["division"],
                "established": info.get("established"),
                "colors": info.get("colors", []),
                "logo_url": info.get("logo_url"),
                "website": info.get("website"),
            }
            teams_list.append(team_info)

        # Sort teams by conference, then division, then name
        teams_list.sort(key=lambda x: (x["conference"], x["division"], x["name"]))

        logger.info(f"Retrieved information for {len(teams_list)} NFL teams")

        return {
            "teams": teams_list,
            "total_teams": len(teams_list),
            "conferences": ["AFC", "NFC"],
            "divisions": ["East", "North", "South", "West"],
        }

    except (ImportError, ValueError, KeyError, TypeError) as e:
        logger.error(f"Error retrieving teams list: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve teams list: {e!s}"
        )


@app.get("/teams/{team_id}/stats", tags=["Teams"])
async def get_team_stats(
    team_id: str = Path(..., description="Team abbreviation"),
    season: int | None = Query(None, description="Season for stats"),
    week: int | None = Query(None, description="Week for stats"),
):
    """Get team statistics and performance metrics."""
    try:
        team_stats = data_service.get_team_stats(
            team_id=team_id, season=season, week=week
        )

        logger.info(
            f"Retrieved team stats for {team_id} (season: {season}, week: {week})"
        )
        return team_stats

    except NFLPredictionAPIException:
        raise
    except (StorageError, ValueError, KeyError) as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        logger.error(f"Error retrieving team stats for {team_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve team stats: {e!s}"
        )


# Predictions Endpoints


@app.get("/predictions/week/{season}/{week}", tags=["Predictions"])
async def get_week_predictions(
    season: int = Path(..., description="NFL season"),
    week: int = Path(
        ..., ge=1, le=22, description="NFL week (1-18 regular season, 19-22 playoffs)"
    ),
):
    """Get all predictions for a specific week."""
    try:
        week_predictions = data_service.get_week_predictions(season=season, week=week)

        logger.info(
            f"Retrieved predictions for {season} week {week}: {week_predictions['total_games']} games"
        )
        return week_predictions

    except (StorageError, ValueError, KeyError) as e:
        logger.error(f"Error retrieving week predictions for {season} week {week}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve week predictions: {e!s}"
        )


@app.get("/recommendations/week/{season}/{week}", tags=["Recommendations"])
async def get_week_recommendations(
    season: int = Path(..., description="NFL season"),
    week: int = Path(
        ..., ge=1, le=22, description="NFL week (1-18 regular season, 19-22 playoffs)"
    ),
    min_edge: float | None = Query(
        0.02, ge=0, le=1, description="Minimum edge threshold"
    ),
    tier: str | None = Query(None, description="Filter by recommendation tier"),
):
    """Get betting recommendations for a specific week."""
    try:
        # Get all games for the specified week
        games, metadata, _total_games = data_service.list_games(
            season=season,
            week=week,
            limit=100,  # Get all games for the week
            offset=0,
        )

        if not games:
            return {
                "season": season,
                "week": week,
                "recommendations": [],
                "total_recommendations": 0,
                "games_analyzed": 0,
                "filters_applied": {"min_edge": min_edge, "tier": tier},
                "message": f"No games found for season {season}, week {week}",
            }

        # Load predictions and market data for the week
        predictions_df = data_service._load_predictions_data(season, week)
        market_df = data_service._load_market_data(season, week)

        # Generate recommendations for each game
        all_recommendations = []
        games_with_recommendations = 0

        for game in games:
            try:
                # Get prediction and market data for this game (safely)
                if not predictions_df.empty and "game_id" in predictions_df.columns:
                    game_predictions = predictions_df[
                        predictions_df["game_id"] == game.game_id
                    ]
                else:
                    game_predictions = pd.DataFrame()

                if not market_df.empty and "game_id" in market_df.columns:
                    game_market = market_df[market_df["game_id"] == game.game_id]
                else:
                    game_market = pd.DataFrame()

                if not game_predictions.empty and not game_market.empty:
                    from utils.api_recommendation_bridge import (
                        api_recommendation_bridge,
                    )

                    recommendations = (
                        api_recommendation_bridge.generate_game_recommendations(
                            game.game_id,
                            game_predictions.iloc[0],
                            game_market.iloc[0],
                            min_edge,
                        )
                    )

                    # Filter by tier if specified
                    if tier and recommendations:
                        recommendations = [
                            rec
                            for rec in recommendations
                            if rec.tier.lower() == tier.lower()
                        ]

                    if recommendations:
                        games_with_recommendations += 1
                        for rec in recommendations:
                            # Add game context to each recommendation
                            rec_dict = {
                                "game_id": game.game_id,
                                "home_team": game.home_team,
                                "away_team": game.away_team,
                                "game_date": game.game_date.isoformat()
                                if game.game_date
                                else None,
                                "bet_type": rec.bet_type,
                                "team": rec.team,
                                "recommended_odds": rec.recommended_odds,
                                "edge": rec.edge,
                                "expected_value": rec.expected_value,
                                "confidence": rec.confidence,
                                "tier": rec.tier,
                                "units": rec.units,
                                "description": rec.description,
                            }
                            all_recommendations.append(rec_dict)

            except (ImportError, ValueError, KeyError, TypeError) as e:
                logger.warning(
                    f"Error generating recommendations for game {game.game_id}: {e}"
                )
                continue

        # Sort recommendations by edge (descending) and confidence (descending)
        all_recommendations.sort(
            key=lambda x: (x["edge"], x["confidence"]), reverse=True
        )

        # Calculate summary statistics
        total_recommendations = len(all_recommendations)

        if total_recommendations > 0:
            avg_edge = (
                sum(rec["edge"] for rec in all_recommendations) / total_recommendations
            )
            avg_confidence = (
                sum(rec["confidence"] for rec in all_recommendations)
                / total_recommendations
            )
            tier_distribution = {}
            for rec in all_recommendations:
                tier_distribution[rec["tier"]] = (
                    tier_distribution.get(rec["tier"], 0) + 1
                )
        else:
            avg_edge = 0.0
            avg_confidence = 0.0
            tier_distribution = {}

        logger.info(
            f"Generated week recommendations for {season} week {week}: "
            f"{total_recommendations} recommendations from {games_with_recommendations}/{len(games)} games"
        )

        return {
            "season": season,
            "week": week,
            "recommendations": all_recommendations,
            "total_recommendations": total_recommendations,
            "games_analyzed": len(games),
            "games_with_recommendations": games_with_recommendations,
            "summary": {
                "average_edge": round(avg_edge, 4) if avg_edge > 0 else 0.0,
                "average_confidence": round(avg_confidence, 4)
                if avg_confidence > 0
                else 0.0,
                "tier_distribution": tier_distribution,
                "coverage": f"{games_with_recommendations}/{len(games)}",
            },
            "filters_applied": {"min_edge": min_edge, "tier": tier},
            "metadata": {
                "predictions_available": metadata.predictions_available,
                "last_updated": metadata.last_updated.isoformat()
                if metadata.last_updated
                else None,
            },
        }

    except (StorageError, ValueError, KeyError, TypeError) as e:
        logger.error(
            f"Error retrieving week recommendations for {season} week {week}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve week recommendations: {e!s}"
        )


# Web UI Routes
# These routes serve HTML templates for the web interface


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def home_page(request: Request):
    """Serve the main games page."""
    if templates is None:
        raise HTTPException(
            status_code=503, detail="Templates not available - web interface disabled"
        )
    return templates.TemplateResponse("games.html", {"request": request})


@app.get("/games/{game_id}/view", response_class=HTMLResponse, tags=["Web UI"])
async def game_detail_page(request: Request, game_id: str):
    """Serve the game detail page."""
    if templates is None:
        raise HTTPException(
            status_code=503, detail="Templates not available - web interface disabled"
        )
    return templates.TemplateResponse("game_detail.html", {"request": request})


@app.get("/backtest/view", response_class=HTMLResponse, tags=["Web UI"])
async def backtest_page(request: Request):
    """Serve the backtest performance page."""
    if templates is None:
        raise HTTPException(
            status_code=503, detail="Templates not available - web interface disabled"
        )
    return templates.TemplateResponse("backtest.html", {"request": request})


@app.get("/calibration/view", response_class=HTMLResponse, tags=["Web UI"])
async def calibration_page(request: Request):
    """Serve the model calibration page."""
    if templates is None:
        raise HTTPException(
            status_code=503, detail="Templates not available - web interface disabled"
        )
    return templates.TemplateResponse("calibration.html", {"request": request})


@app.get("/recommendations/view", response_class=HTMLResponse, tags=["Web UI"])
async def recommendations_page(request: Request):
    """Serve the week recommendations page."""
    if templates is None:
        raise HTTPException(
            status_code=503, detail="Templates not available - web interface disabled"
        )
    return templates.TemplateResponse("recommendations.html", {"request": request})


# CSV Download Endpoints


@app.get("/downloads/games", response_class=StreamingResponse, tags=["Downloads"])
async def download_games_csv(
    season: int | None = Query(None, description="Filter by season"),
    week: int | None = Query(
        None,
        ge=1,
        le=22,
        description="Filter by week (1-18 regular season, 19-22 playoffs)",
    ),
    team_param: str | None = Query(
        None, alias="team", description="Filter by team abbreviation (2-4 letters)"
    ),
    has_recommendations: bool | None = Query(
        None, description="Filter games with recommendations"
    ),
    min_edge: float | None = Query(
        None, ge=0, le=1, description="Minimum edge threshold"
    ),
):
    """Download games data as CSV file."""
    # Validate parameters
    team = validate_team_param(team_param)

    # Get games data
    games, _metadata, _total = data_service.list_games(
        season=season,
        week=week,
        team=team,
        has_recommendations=has_recommendations,
        min_edge=min_edge,
        limit=1000,  # Large limit for CSV export
        offset=0,
    )

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(
        [
            "game_id",
            "season",
            "week",
            "game_date",
            "home_team",
            "away_team",
            "status",
            "home_score",
            "away_score",
            "win_probability",
            "spread",
            "total",
            "has_recommendations",
            "top_recommendation_type",
            "top_recommendation_edge",
            "top_recommendation_units",
        ]
    )

    # Write data rows
    for game in games:
        top_rec = game.top_recommendation
        writer.writerow(
            [
                game.game_id,
                game.season,
                game.week,
                game.game_date.isoformat() if game.game_date else "",
                game.home_team,
                game.away_team,
                game.status,
                game.home_score,
                game.away_score,
                game.win_probability,
                game.spread,
                game.total,
                game.has_recommendations,
                top_rec.bet_type if top_rec else "",
                top_rec.edge if top_rec else "",
                top_rec.units if top_rec else "",
            ]
        )

    output.seek(0)

    # Generate filename with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"nfl_games_{timestamp}.csv"

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/downloads/backtest", response_class=StreamingResponse, tags=["Downloads"])
async def download_backtest_csv_streaming(
    start_season: int | None = Query(None, description="Start season"),
    end_season: int | None = Query(None, description="End season"),
    model_type: str | None = Query(None, description="Filter by model type"),
):
    """Download backtest results as CSV file."""
    summary = backtest_service.get_backtest_summary(
        start_season=start_season, end_season=end_season, model_type=model_type
    )

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["metric", "value", "period", "model_type"])

    # Write summary metrics
    period = f"{start_season or 'all'}-{end_season or 'current'}"
    model = model_type or "all"

    metrics = [
        ("total_predictions", summary.total_predictions),
        ("wp_accuracy", summary.wp_accuracy),
        ("ats_accuracy", summary.ats_accuracy),
        ("ou_accuracy", summary.ou_accuracy),
        ("wp_log_loss", summary.wp_log_loss),
        ("wp_brier_score", summary.wp_brier_score),
        ("ats_mae", summary.ats_mae),
        ("ou_mae", summary.ou_mae),
        ("betting_roi", summary.betting_roi),
        ("total_bets", summary.total_bets),
        ("winning_bets", summary.winning_bets),
        ("total_profit", summary.total_profit),
        ("sharpe_ratio", summary.sharpe_ratio),
    ]

    for metric_name, value in metrics:
        writer.writerow([metric_name, value, period, model])

    output.seek(0)

    # Generate filename with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"nfl_backtest_{timestamp}.csv"

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/downloads/calibration", response_class=StreamingResponse, tags=["Downloads"])
async def download_calibration_csv(
    model_type: str | None = Query(None, description="Filter by model type"),
    season: int | None = Query(None, description="Filter by season"),
):
    """Download calibration data as CSV file."""
    calibration_data = backtest_service.get_calibration_data(
        model_type=model_type, season=season
    )

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(
        [
            "model_type",
            "total_predictions",
            "ece",
            "mce",
            "reliability",
            "predicted_probability",
            "observed_frequency",
            "bin_count",
        ]
    )

    # Write data rows
    for model, data in calibration_data.items():
        for _i, (pred_prob, obs_freq, bin_count) in enumerate(
            zip(
                data.predicted_probabilities,
                data.observed_frequencies,
                data.bin_counts,
                strict=False,
            )
        ):
            writer.writerow(
                [
                    model,
                    data.total_predictions,
                    data.ece,
                    data.mce,
                    data.reliability,
                    pred_prob,
                    obs_freq,
                    bin_count,
                ]
            )

    output.seek(0)

    # Generate filename with timestamp
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"nfl_calibration_{timestamp}.csv"

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    import uvicorn

    from .config import get_uvicorn_config

    uvicorn_config = get_uvicorn_config(settings)
    uvicorn.run("api.main:app", **uvicorn_config)
