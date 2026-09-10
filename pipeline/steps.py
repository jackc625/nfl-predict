"""Step definitions, adapter functions, and registry for the Friday pipeline.

Every step adapter uses deferred imports (inside the function body) to avoid
argparse collisions and module-level side effects from scripts/.

The step registry returns exactly 19 StepDefinition entries covering the full
data-to-prediction pipeline, ending with the NON-CRITICAL web-cache population
step Plan 31-18 added (SPEC R9, D31-29).
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


def _bet_list_output_dir() -> Path:
    """Directory where the DURABLE weekly bet-list artifacts are written.

    Separate from ``_predictions_output_dir`` on purpose. These two files are CACHE SOURCE
    artifacts -- the Plan 31-18 population step reads them INTO the temp cache build -- and they
    are also the durable home of forward recommendation history, which the predictions CSVs are
    not. Factored into one place so tests can redirect the write without touching the repo's
    outputs/ tree. Taken from the module that owns the artifact names rather than restated, so the
    writer here and the reader there cannot spell the directory differently.
    """
    from backtest.weekly_bet_list import DEFAULT_BET_LIST_DIR

    return DEFAULT_BET_LIST_DIR


def _web_cache_db_path() -> Path:
    """The live DuckDB web cache the FastAPI app serves from.

    The SAME default ``scripts/populate_cache.py`` uses, factored here so the scheduled step and
    the manual recovery command named in the ``/bets`` refusal text cannot rebuild two different
    files. Tests redirect it rather than writing the production cache.
    """
    return Path("data/web_cache.duckdb")


# Single source of truth for the data artifacts the DATA-phase integrity gate
# (step_verify_data_artifacts) requires before the PREDICTIONS phase may run.
# Factored into one place so a drift regression test can import the same list
# the gate checks (rather than re-hardcoding a second copy that could silently
# diverge from the real build-script output names).
_REQUIRED_ARTIFACTS = [
    "data/silver/games.parquet",
    "data/silver/elo_game_snapshots.parquet",
    "data/silver/team_form_features.parquet",
    "data/gold/features_wp.parquet",
    "data/gold/features_ats.parquet",
    "data/gold/features_ou.parquet",
]


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

    missing = [p for p in _REQUIRED_ARTIFACTS if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing data artifacts: {', '.join(missing)}")


# PREDICTIONS PHASE (10 steps) ------------------------------------------


def step_ingest_odds() -> None:
    """Capture odds snapshot from The Odds API."""
    from scripts.ingest_odds import OddsDataIngester

    ingester = OddsDataIngester()
    ingester.ingest_odds()


def step_build_market_anchors() -> None:
    """Build market anchor features from the odds snapshot into silver. OUTPUT IS UNREAD.

    REGISTERED NON-CRITICAL (WR-13). It was ``critical=True``, so a raise anywhere in
    ``build_market_anchor_features`` aborted the ENTIRE Friday run -- including the prediction and
    bet-list work that follows it. What it produces, silver ``market_anchor_features``, is read by
    no production code (``AUTOMATION.md``): the gold build calls
    ``MarketAnchorFeaturesCalculator.build_features`` directly at
    ``scripts/build_features.py``, not this table. Killing a run that has already ingested odds
    over a table nothing consumes is the wrong direction, so its failure now DEGRADES the run.

    The method it calls is ``@deprecated`` and fires a ``DeprecationWarning`` on every scheduled
    run. That warning is left in place deliberately: it is the honest signal that this step is on
    the deprecated path, and silencing it would hide the thing a future author needs to see.

    IF THIS STEP EVER GAINS A CONSUMER, revisit the criticality along with it -- a step whose
    output feeds gold should fail loudly. See WR-01 for the parse this path used to use.
    """
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


_GOLD_FEATURE_TABLES = ("features_wp", "features_ats", "features_ou")


def step_validate_features() -> None:
    """Hard-fail the run if any gold matrix carries a post-game feature column (CR-02).

    This is the registry's temporal-safety gate and it is registered ``critical=True``, so it
    must be able to FAIL. Three defects made it inert:

    * it read ``leakage_result["has_leakage"]``, a key ``FeatureValidator.check_data_leakage``
      has never returned (it returns ``passed`` / ``leakage_columns``), so ``.get(..., False)``
      was always ``False`` and the ``raise`` was unreachable;
    * the ``break`` sat inside the ``try``, so the loop stopped after the FIRST matrix that
      loaded -- ``features_ats`` and ``features_ou`` were never scanned at all; and
    * loading none of the three silently passed, reporting success for a check that did not run.

    All three gold matrices are now scanned, the verdict is read off ``passed``, and a run in
    which no matrix could be loaded raises rather than reporting a gate it never applied.

    The binding gate remains ``LeakageGate.validate_combined_matrix`` on the canonical gold build
    path (``scripts/build_features.py``); this is the defense-in-depth sibling that re-checks what
    actually landed on disk, which is why the two keyword lists are documented as kept in sync.
    """
    from data.storage import load_dataframe
    from features.validation import FeatureValidator
    from models.temporal import _DEFAULT_ID_COLS

    # The identifier and OUTCOME columns every gold matrix carries BY CONSTRUCTION -- the labels
    # a walk-forward fit trains against. They are excluded from the scan because a gold matrix is
    # supposed to contain them: ``features_ats`` carries ``home_margin`` as its ATS regression
    # label, and scanning it would hard-fail the Friday run on the target column itself. Taken
    # from ``models.temporal``, where the splitter already states the list, rather than copied
    # into a third place.
    label_and_id_columns = frozenset(_DEFAULT_ID_COLS)

    validator = FeatureValidator()
    checked: list[str] = []
    for table in _GOLD_FEATURE_TABLES:
        try:
            target_df = load_dataframe(table, layer="gold")
        except FileNotFoundError:
            continue
        checked.append(table)
        scanned = target_df[
            [c for c in target_df.columns if c not in label_and_id_columns]
        ]
        leakage_result = validator.check_data_leakage(scanned)
        if not leakage_result["passed"]:
            raise RuntimeError(
                f"Feature leakage detected in gold {table}: "
                f"{leakage_result['leakage_columns']}"
            )

    if not checked:
        raise RuntimeError(
            "The feature leakage gate did not run: none of "
            f"{list(_GOLD_FEATURE_TABLES)} could be loaded from the gold layer. A critical "
            "temporal-safety step must not report success for a check it never applied."
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
    """Select the week's +EV bet list through the single bet-decision source (PROD-02, SPEC R4).

    THE ONE weekly recommendation path (D31-31). It delegates to
    ``backtest.weekly_bet_list.generate_weekly_bet_list``, which routes every scheduled game times
    every registered target through ``backtest.bet_selector.BetSelector`` -- pricing, EV admission,
    Kelly sizing, suppression and grading all inside the LOCKED-2 decision engine -- and writes the
    durable bet-list artifact the Plan 31-18 cache population step reads.

    It REPLACES a legacy body that filtered on a confidence tier with no expected value, no sizing
    and no suppression, and wrote ``recommendations_<season>_week<week>.json`` -- a file no code
    ever read. That output is RETIRED (D31-32); nothing under the predictions output directory is
    written here any more.

    THE WRITE IS AN ARTIFACT, NEVER THE LIVE CACHE (REVIEW-CACHE). ``api.cache.populate_cache``
    builds a fresh temporary database and ends with ``db_path.unlink()`` then
    ``tmp_path.rename(db_path)``, so a live-cache write would be destroyed by the next population
    run. No connection to the configured cache path is opened anywhere in this step.

    THE TRACKER AGGREGATION MOVED INTO THE DELEGATE (Plan 31-22, T-31-114/117). It used to happen
    HERE, which meant the two halves of the durable pair were written from two different places:
    the library function wrote the bet list and this step wrote the companion tracker blocks. Any
    OTHER caller of the delegate therefore produced half the pair, and the realized-versus-expected
    tracker would stay permanently empty while ``/bets`` still looked correct. Both writes now
    travel together inside ``generate_weekly_bet_list``, so this scheduled step and the manual
    command ``scripts/generate_bet_list.py`` cannot produce different artifact SETS.

    The seam itself is UNCHANGED. ``api/cache.py`` still imports no ``backtest`` module and the
    sibling guard ``tests/api/test_import_guard_bets.py``'s allow-list is still not widened: the
    aggregation moved from one module permitted to import ``backtest`` to another, never into
    ``api/``. What this step is left with is exactly the thin delegate the rest of this docstring
    already described -- resolve the week, resolve the output directory, call the one selection
    facade.

    Raises:
        Whatever the delegate raises. Nothing is swallowed: a week that cannot be selected must
        record a clean step failure rather than publish a silently empty bet list.
    """
    from backtest.weekly_bet_list import generate_weekly_bet_list
    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    generate_weekly_bet_list(
        season=season, week=week, output_dir=_bet_list_output_dir()
    )


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


def step_populate_web_cache() -> None:
    """Rebuild the DuckDB web cache so the served bet list is this run's (SPEC R9, D31-29).

    THE BOUNDARY THIS STEP MOVED. Until Plan 31-18 the orchestrator did not rebuild the web cache
    at all, and both operator documents said so. It does now, LAST in the registry -- strictly
    after ``generate_recommendations`` (which writes the durable bet-list artifact), after
    ``export_artifacts``, and after the two validation steps, so the cache is never published from
    predictions that failed validation. ``tests/unit/test_step_registry_order.py`` pins that
    position by INDEX so a future insertion cannot silently move it above a dependency.

    REGISTERED NON-CRITICAL, deliberately. A cache failure must not fail a run whose prediction
    work succeeded: making it critical would discard good prediction output because a downstream
    convenience failed, which is the outcome SPEC R9 explicitly refuses. The orchestrator already
    sets ``status="degraded"`` on a non-critical failure and already routes a degraded completion
    to ``alert_degraded_completion``, so NO new alert code exists here or in ``pipeline/alert.py``.

    BE HONEST ABOUT WHAT THAT BUYS. ``pipeline/alert.py`` documents alerts as LOG-ONLY by default,
    and this project's record is that the email and messaging channels are inert across three
    independent breaks -- console and log are the working channel. So the alert is NOT the
    protection against a silently stale bet list. The protection is the ``/bets`` hard-block: the
    page refuses to serve a week whose bet-list populated-at marker predates that week's latest
    per-game line freeze, which a reader cannot miss. This step's contribution to that guard is
    that a FAILED run leaves the marker unadvanced, so the block fires deterministically.

    Raises:
        Whatever ``api.cache.populate_cache`` raises. Nothing is swallowed here -- the
        orchestrator's non-critical handling is what turns the raise into a degraded run, and
        swallowing it here would hide the failure from the run log as well as from the alert.
    """
    from api.cache import populate_cache
    from backtest.weekly_bet_list import read_bet_list_cache_sources

    silver_dir = Path("data/silver")
    # Read the DURABLE artifacts and the schedule HERE and hand over frames. ``api/cache.py`` may
    # import no ``backtest`` module (UIAP-01), so it cannot know the artifact names or derive a
    # per-game freeze; this module already imports ``backtest`` and is the permitted seam. The
    # frames are loaded into the population run's TEMPORARY database before its atomic swap, which
    # is what makes forward recommendation history survive a rebuild that replaces the whole file.
    sources = read_bet_list_cache_sources(_bet_list_output_dir(), silver_dir)

    populate_cache(
        db_path=_web_cache_db_path(),
        artifacts_dir=Path("artifacts"),
        outputs_dir=Path("outputs/backtest"),
        gold_dir=Path("data/gold"),
        silver_dir=silver_dir,
        bet_list_df=sources.bet_list,
        bet_tracker_df=sources.tracker,
        bet_schedule_df=sources.schedule,
    )


# ---------------------------------------------------------------------------
# Step registry builder
# ---------------------------------------------------------------------------


def build_step_registry() -> list[StepDefinition]:
    """Build the complete 19-step pipeline registry.

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
        # PREDICTIONS PHASE (11 steps)
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
            # NON-CRITICAL (WR-13): its silver output is read by no production code, so a failure
            # here must not abort the prediction and bet-list work that follows. See the step's
            # own docstring for the full reasoning and for when to revisit this.
            critical=False,
            retryable=False,
            description="Build market anchor features (silver output currently unread)",
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
            description="Select the +EV bet list through BetSelector",
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
        # LAST, and NON-CRITICAL (SPEC R9, D31-29). It must follow generate_recommendations and
        # export_artifacts so the blob it reads is the one this run wrote; placing it after the
        # two validation steps as well means a cache is never published from predictions that
        # failed validation. The non-critical flag routes a failure to the EXISTING
        # alert_degraded_completion path -- no new alert code -- and alerts are log-only by
        # default, so the real protection against a silently stale list is the /bets hard-block,
        # not this alert. See step_populate_web_cache's docstring.
        StepDefinition(
            "populate_web_cache",
            step_populate_web_cache,
            PipelinePhase.PREDICTIONS,
            critical=False,
            retryable=False,
            description="Rebuild the DuckDB web cache the site serves from",
        ),
    ]
