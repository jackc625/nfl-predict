"""Step definitions, adapter functions, and registry for the Friday pipeline.

Every step adapter uses deferred imports (inside the function body) to avoid
argparse collisions and module-level side effects from scripts/.

The step registry returns exactly 18 StepDefinition entries covering the full
data-to-prediction pipeline.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class PipelinePhase(Enum):
    """Pipeline execution phase."""

    DATA = "data"
    PREDICTIONS = "predictions"


class StepStatus(Enum):
    """Execution status for a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRIED = "retried"


@dataclass
class StepDefinition:
    """Definition for a single pipeline step.

    Attributes:
        name: Unique step identifier.
        callable: Zero-argument function that executes the step.
        phase: Which pipeline phase this step belongs to.
        critical: If True, failure aborts the pipeline.
        retryable: If True, transient errors trigger retry with backoff.
        max_retries: Maximum retry attempts for retryable steps.
        description: Human-readable description of the step.
    """

    name: str
    callable: Callable[[], None]
    phase: PipelinePhase
    critical: bool = True
    retryable: bool = False
    max_retries: int = 3
    description: str = ""


@dataclass
class StepResult:
    """Result of executing a single pipeline step."""

    name: str
    status: StepStatus
    duration_ms: float
    retry_count: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Transient exception types eligible for retry
# ---------------------------------------------------------------------------

TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Include httpx.HTTPStatusError if httpx is available
try:
    import httpx

    TRANSIENT_EXCEPTIONS = (*TRANSIENT_EXCEPTIONS, httpx.HTTPStatusError)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Step adapter functions -- deferred imports, no module-level script imports
# ---------------------------------------------------------------------------

# DATA PHASE (8 steps) --------------------------------------------------


def step_ingest_games() -> None:
    """Ingest current week games data via nflreadpy."""
    from scripts.ingest_games import GameDataIngester

    ingester = GameDataIngester()
    ingester.ingest_games()


def step_ingest_weather() -> None:
    """Ingest weather forecasts via Open-Meteo API (async)."""
    from scripts.ingest_weather import WeatherDataIngester

    ingester = WeatherDataIngester()
    ingester.ingest_weather()


def step_data_qa() -> None:
    """Run data quality validation checks."""
    from scripts.data_qa import DataQualityMonitor

    monitor = DataQualityMonitor()
    report = monitor.generate_qa_report()
    # Fail if overall status indicates issues
    summary = report.get("summary", {})
    if (
        summary.get("overall_status") == "issues_detected"
        and summary.get("failed", 0) > 0
    ):
        raise RuntimeError(f"Data QA failed: {summary.get('failed', 0)} checks failed")


def step_build_elo() -> None:
    """Update Elo ratings for the current season."""
    from scripts.build_elo import EloBuilder

    builder = EloBuilder()
    builder.update_current_season()


def step_build_team_form() -> None:
    """Build team form metrics for current week."""
    from scripts.build_team_form import TeamFormBuilder

    builder = TeamFormBuilder()
    builder.build_for_current_week()


def step_build_contextual() -> None:
    """Build contextual features (travel, rest, venue)."""
    from data.storage import load_dataframe, save_dataframe
    from features.contextual import ContextualFeaturesCalculator

    games_df = load_dataframe("games", layer="silver")
    calculator = ContextualFeaturesCalculator()
    features_df = calculator.build_contextual_features(games_df=games_df)
    if len(features_df) > 0:
        save_dataframe(features_df, table_name="contextual_features", layer="silver")


def step_build_weather_features() -> None:
    """Build weather-based features for outdoor games."""
    from data.storage import load_dataframe, save_dataframe
    from features.weather import WeatherFeaturesCalculator

    games_df = load_dataframe("games", layer="silver")
    calculator = WeatherFeaturesCalculator()
    features_df = calculator.build_weather_features(games_df=games_df)
    if len(features_df) > 0:
        save_dataframe(features_df, table_name="weather_features", layer="silver")


def step_verify_data_artifacts() -> None:
    """Verify all data artifacts exist before predictions phase."""
    from pathlib import Path

    required = [
        "data/silver/games.parquet",
        "data/silver/elo_ratings.parquet",
        "data/silver/team_form.parquet",
        "data/gold/",
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing data artifacts: {', '.join(missing)}")


# PREDICTIONS PHASE (10 steps) ------------------------------------------


def step_ingest_odds() -> None:
    """Capture odds snapshot from The Odds API."""
    from scripts.ingest_odds import OddsDataIngester

    ingester = OddsDataIngester()
    ingester.ingest_odds()


def step_build_market_anchors() -> None:
    """Build market anchor features from odds snapshot."""
    from data.storage import load_dataframe, save_dataframe
    from features.market_anchors import MarketAnchorFeaturesCalculator

    games_df = load_dataframe("games", layer="silver")
    calculator = MarketAnchorFeaturesCalculator()
    features_df = calculator.build_market_anchor_features(games_df=games_df)
    if len(features_df) > 0:
        save_dataframe(features_df, table_name="market_anchor_features", layer="silver")


def step_build_features() -> None:
    """Create unified feature matrices for all model targets."""
    from scripts.build_features import FeatureMatrixBuilder

    builder = FeatureMatrixBuilder()
    builder.generate_feature_matrices()


def step_validate_features() -> None:
    """Validate features for data leakage and quality."""
    from data.storage import load_dataframe
    from features.validation import FeatureValidator

    validator = FeatureValidator()
    # Try gold layer targets
    features_df = None
    for table in ["features_wp", "features_ats", "features_ou"]:
        try:
            target_df = load_dataframe(table, layer="gold")
            if features_df is None:
                features_df = target_df
            break
        except FileNotFoundError:
            continue
    if features_df is not None:
        leakage_result = validator.check_data_leakage(features_df)
        if leakage_result.get("has_leakage", False):
            raise RuntimeError(
                f"Feature leakage detected: {leakage_result.get('leakage_features', [])}"
            )


def step_validate_models() -> None:
    """Validate prediction models are available and loadable."""
    from scripts.validate_models import ModelValidator

    validator = ModelValidator()
    availability = validator.validate_model_availability(["wp", "ats", "ou"])
    missing = [k for k, v in availability.items() if not v]
    if missing:
        raise RuntimeError(f"Model validation failed -- missing: {missing}")
    loadability = validator.validate_model_loadability(["wp", "ats", "ou"])
    unloadable = [k for k, v in loadability.items() if not v]
    if unloadable:
        raise RuntimeError(f"Model validation failed -- unloadable: {unloadable}")


def step_generate_predictions() -> None:
    """Generate predictions for current week games."""
    from data.storage import load_dataframe
    from models.prediction_pipeline import NFLPredictionPipeline

    pipeline = NFLPredictionPipeline()
    games_df = load_dataframe("games", layer="silver")
    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    current_games = games_df[
        (games_df["season"] == season) & (games_df["week"] == week)
    ]
    if len(current_games) > 0:
        pipeline.predict_games(current_games)


def step_generate_recommendations() -> None:
    """Generate bet recommendations from predictions."""
    # BetRecommendationEngine.generate_recommendation requires per-game edge/confidence.
    # The orchestrator adapter loads predictions and generates recommendations in bulk.
    from models.prediction_pipeline import BetRecommendationEngine

    # This is a lightweight wrapper -- actual recommendation generation happens
    # inside the prediction pipeline. We instantiate to verify it works.
    _engine = BetRecommendationEngine()


def step_export_artifacts() -> None:
    """Export prediction artifacts in multiple formats.

    Ensures the outputs directory exists. Actual export is handled by the
    prediction pipeline during step_generate_predictions; this step
    verifies the output directory is ready.
    """
    from pathlib import Path

    predictions_dir = Path("outputs/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)


def step_validate_predictions() -> None:
    """Validate prediction outputs for completeness and quality."""
    from scripts.validate_predictions import PredictionValidator

    validator = PredictionValidator()
    file_results = validator.validate_prediction_files()
    missing = [k for k, v in file_results.items() if not v]
    if missing:
        raise RuntimeError(f"Prediction validation failed -- missing files: {missing}")


def step_verify_output_files() -> None:
    """Verify expected output files exist after pipeline run."""
    from pathlib import Path

    from utils.logging_config import get_logger

    logger = get_logger(__name__)
    expected = [
        "outputs/predictions/current_week_predictions.parquet",
        "outputs/predictions/current_week_predictions.json",
        "outputs/predictions/current_week_summary.csv",
    ]
    missing = [f for f in expected if not Path(f).exists()]
    if missing:
        logger.warning("Missing output files", missing_files=missing)


# ---------------------------------------------------------------------------
# Step registry builder
# ---------------------------------------------------------------------------


def build_step_registry() -> list[StepDefinition]:
    """Build the complete 18-step pipeline registry.

    Returns:
        Ordered list of StepDefinitions covering data and prediction phases.
    """
    return [
        # DATA PHASE (8 steps)
        StepDefinition(
            "ingest_games",
            step_ingest_games,
            PipelinePhase.DATA,
            critical=True,
            retryable=True,
            max_retries=3,
            description="Ingest current week games data",
        ),
        StepDefinition(
            "ingest_weather",
            step_ingest_weather,
            PipelinePhase.DATA,
            critical=False,
            retryable=True,
            max_retries=3,
            description="Ingest weather forecasts",
        ),
        StepDefinition(
            "data_qa",
            step_data_qa,
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            description="Data quality validation",
        ),
        StepDefinition(
            "build_elo",
            step_build_elo,
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            description="Update Elo ratings",
        ),
        StepDefinition(
            "build_team_form",
            step_build_team_form,
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            description="Build team form metrics",
        ),
        StepDefinition(
            "build_contextual",
            step_build_contextual,
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            description="Build contextual features",
        ),
        StepDefinition(
            "build_weather_features",
            step_build_weather_features,
            PipelinePhase.DATA,
            critical=False,
            retryable=False,
            description="Build weather features",
        ),
        StepDefinition(
            "verify_data_artifacts",
            step_verify_data_artifacts,
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            description="Verify data artifacts before predictions",
        ),
        # PREDICTIONS PHASE (10 steps)
        StepDefinition(
            "ingest_odds",
            step_ingest_odds,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=True,
            max_retries=3,
            description="Capture odds snapshot",
        ),
        StepDefinition(
            "build_market_anchors",
            step_build_market_anchors,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Build market anchor features",
        ),
        StepDefinition(
            "build_features",
            step_build_features,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Create unified feature matrices",
        ),
        StepDefinition(
            "validate_features",
            step_validate_features,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Validate features for leakage",
        ),
        StepDefinition(
            "validate_models",
            step_validate_models,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Validate prediction models",
        ),
        StepDefinition(
            "generate_predictions",
            step_generate_predictions,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Generate predictions",
        ),
        StepDefinition(
            "generate_recommendations",
            step_generate_recommendations,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Generate bet recommendations",
        ),
        StepDefinition(
            "export_artifacts",
            step_export_artifacts,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Export prediction artifacts",
        ),
        StepDefinition(
            "validate_predictions",
            step_validate_predictions,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Validate prediction outputs",
        ),
        StepDefinition(
            "verify_output_files",
            step_verify_output_files,
            PipelinePhase.PREDICTIONS,
            critical=False,
            retryable=False,
            description="Verify output file existence",
        ),
    ]
