"""Step definitions, adapter functions, and registry for the Friday pipeline.

Every step adapter uses deferred imports (inside the function body) to avoid
argparse collisions and module-level side effects from scripts/.

The step registry returns exactly 18 StepDefinition entries covering the full
data-to-prediction pipeline.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
# Shared helpers
# ---------------------------------------------------------------------------


def _predictions_output_dir() -> Path:
    """Directory where current-week prediction artifacts are written.

    Factored into one place so the prediction-phase steps stay consistent and
    tests can redirect output without writing into the repo's outputs/ tree.
    """
    return Path("outputs/predictions")


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
        "data/gold/features_wp.parquet",
        "data/gold/features_ats.parquet",
        "data/gold/features_ou.parquet",
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
    from pipeline.model_validation import ModelValidator

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
    """Generate current-week predictions via the canonical generation path.

    Delegates to ``generate_current_week_predictions.generate_and_write``, which
    loads the model artifacts, computes edges, applies market blending (using the
    blend artifact when present), and writes ``predictions_<season>_week<week>.csv``
    plus the game-context CSV. Raises if the gold matrix lacks the current week
    so the orchestrator records a clean step failure.
    """
    from scripts.generate_current_week_predictions import generate_and_write
    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    generate_and_write(season=season, week=week, output_dir=_predictions_output_dir())


def step_generate_recommendations() -> None:
    """Derive bet recommendations from the generated predictions.

    A recommendation is any target whose edge cleared the medium/high confidence
    threshold during prediction generation. Writes
    ``recommendations_<season>_week<week>.json``.
    """
    import json

    import pandas as pd

    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    output_dir = _predictions_output_dir()
    pred_path = output_dir / f"predictions_{season}_week{week}.csv"
    if not pred_path.exists():
        raise RuntimeError(
            f"Cannot generate recommendations -- predictions missing: {pred_path}"
        )

    df = pd.read_csv(pred_path)
    target_cols = {
        "wp": ("wp_edge", "wp_confidence"),
        "ats": ("ats_edge", "ats_confidence"),
        "ou": ("ou_edge", "ou_confidence"),
    }
    recommendations: list[dict] = []
    for _, row in df.iterrows():
        for target, (edge_col, conf_col) in target_cols.items():
            if row.get(conf_col) in ("medium", "high"):
                edge = row.get(edge_col)
                recommendations.append(
                    {
                        "game_id": row.get("game_id"),
                        "target": target,
                        "edge": None if pd.isna(edge) else float(edge),
                        "confidence": row.get(conf_col),
                    }
                )

    rec_path = output_dir / f"recommendations_{season}_week{week}.json"
    rec_path.write_text(json.dumps(recommendations, indent=2))


def step_export_artifacts() -> None:
    """Export the current-week predictions to JSON alongside the CSV."""
    import pandas as pd

    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    output_dir = _predictions_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = output_dir / f"predictions_{season}_week{week}.csv"
    if not pred_csv.exists():
        raise RuntimeError(f"Cannot export -- predictions CSV missing: {pred_csv}")

    df = pd.read_csv(pred_csv)
    json_path = output_dir / f"predictions_{season}_week{week}.json"
    df.to_json(json_path, orient="records", indent=2)


def step_validate_predictions() -> None:
    """Validate the generated current-week prediction file."""
    import pandas as pd

    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    pred_path = _predictions_output_dir() / f"predictions_{season}_week{week}.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Prediction validation failed -- missing file: {pred_path}")

    df = pd.read_csv(pred_path)
    if df.empty:
        raise RuntimeError(f"Prediction validation failed -- no rows in {pred_path}")

    required = ["game_id", "wp_prob", "ats_prediction", "ou_prediction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Prediction validation failed -- missing columns: {missing}"
        )

    if not df["wp_prob"].between(0.0, 1.0).all():
        raise RuntimeError("Prediction validation failed -- wp_prob outside [0, 1]")


def step_verify_output_files() -> None:
    """Verify expected current-week output files exist after the run."""
    from utils.date_utils import get_current_nfl_week
    from utils.logging_config import get_logger

    logger = get_logger(__name__)
    season, week = get_current_nfl_week()
    output_dir = _predictions_output_dir()
    expected = [
        output_dir / f"predictions_{season}_week{week}.csv",
        output_dir / f"predictions_{season}_week{week}.json",
        output_dir / f"game_context_{season}_week{week}.csv",
    ]
    missing = [str(f) for f in expected if not f.exists()]
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
