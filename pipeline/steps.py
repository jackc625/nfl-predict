"""Step definitions, adapter functions, and registry for the Friday pipeline.

Every step adapter uses deferred imports (inside the function body) to avoid
argparse collisions and module-level side effects from scripts/.

The step registry returns exactly 22 StepDefinition entries covering the full
data-to-prediction pipeline, ending with the NON-CRITICAL web-cache population
step Plan 31-18 added (SPEC R9, D31-29).
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

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


class RunStatus(Enum):
    """The TERMINAL outcome of a whole pipeline RUN (Plan 33.2-03, SPEC R3, D33.2-05).

    A RUN-LEVEL vocabulary, deliberately separate from :class:`StepStatus`, which is per-STEP.
    ``StepStatus.SKIPPED`` is NOT overloaded to mean "the run dropped games": one symbol with two
    subjects -- a step that did not run, and a run that ran and left some games out -- would make
    every reader of the log guess which one was meant.

    ``FINISHED_WITH_SKIPS`` is the live half of D33.2-05: the run COMPLETED, and at least one game
    was left out because an input post-dated its lock. It is neither a success (a game the owner
    expected has no prediction) nor a failure (every clean game was predicted, which is the
    outcome the rule exists to protect). It is its own alert and its own log line.

    THE VALUES ARE THE WIRE STRINGS. ``ExecutionLog.status`` is a bare ``str`` in
    ``logs/friday_pipeline.json``, and it carries ``RunStatus.<member>.value`` -- one vocabulary
    with two representations, never two vocabularies that agree today. The two non-terminal or
    legacy strings the log also carries (``"running"``, ``"degraded"``) are not terminal RUN
    verdicts introduced here and are documented beside ``ExecutionLog.status``.
    """

    SUCCESS = "success"
    FAILED = "failed"
    FINISHED_WITH_SKIPS = "finished_with_skips"


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


def _resolve_current_week() -> tuple[int, int]:
    """The ONE ``(season, week)`` resolution a step in this module may use.

    Factored into one place so the capture step and the ingest step cannot resolve
    the week TWICE (COLD-08, Plan 33-07 Task 3). Two resolutions can disagree across
    a midnight boundary, and a capture recorded against a different week than the one
    ingested is worse than no capture at all: it looks like evidence and is not.
    """
    from utils.date_utils import get_current_nfl_week

    return get_current_nfl_week()


def _decision_instant() -> datetime:
    """The instant a LIVE run decides the games it is about to predict or bet. Tz-aware UTC.

    ONE function so the prediction step and the bet-list step read the decision instant the same
    way (Plan 33.2-03), and so a test can pin it instead of the wall clock. It is the instant the
    passed-lock refusal judges -- ``pipeline.live_skip.refuse_passed_locks`` -- and the instant
    the bet list stamps as each row's ``decided_at_utc``, so the refusal and the stamp cannot
    disagree about when the decision happened. Plan 33.2-27 separates the capture instant from
    the compute instant for the daily path; until then the run clock is the only honest value.
    """
    return datetime.now(tz=UTC)


def _week_schedule(season: int, week: int) -> "pd.DataFrame":
    """The week's scheduled games (``game_id``, ``kickoff_et``) from silver ``games``. READ ONLY.

    The same relative store ``generate_current_week_predictions`` and the bet list read, so the
    passed-lock refusal judges exactly the games those two go on to score.

    Raises:
        FileNotFoundError: when the silver schedule is absent -- without it no game's lock can be
            established, and predicting a game whose lock cannot be checked is refused.
    """
    import pandas as pd

    games_path = Path("data/silver/games.parquet")
    if not games_path.exists():
        msg = (
            f"cannot check the week's locks -- schedule missing: {games_path.as_posix()}. A game "
            "whose lock cannot be established is not predicted."
        )
        raise FileNotFoundError(msg)
    games = pd.read_parquet(
        games_path, columns=["game_id", "season", "week", "kickoff_et"]
    )
    in_week = games.loc[(games["season"] == season) & (games["week"] == week)]
    return in_week[["game_id", "kickoff_et"]].reset_index(drop=True)


def _every_scheduled_game_excluded(
    season: int, week: int, excluded: frozenset[str]
) -> bool:
    """True when the week HAS scheduled games and every one of them was dropped this run.

    The discriminator SPEC R3 needs: an empty prediction file is the honest output of an
    all-skipped day, and a refusal-worthy defect on any other day. Read only when the register
    is non-empty, so a clean run's checks are exactly what they were.
    """
    if not excluded:
        return False
    scheduled = {str(game_id) for game_id in _week_schedule(season, week)["game_id"]}
    return bool(scheduled) and scheduled <= excluded


class StaleDataArtifactError(RuntimeError):
    """An artifact EXISTS but carries no row for the current ``(season, week)``.

    A SEPARATE CLASS FROM THE MISSING-ARTIFACT REFUSAL, deliberately (T-33-33). The two
    failures call for different actions -- run the ingest, versus find out why the ingest
    ran and produced nothing for this week -- and an operator who cannot tell them apart
    from the message is being sent to the wrong place. Bare ``Path.exists()`` could not
    tell them apart at all, which is the defect this class exists to close.
    """


# The closed vocabulary of PHASE BOUNDARIES an artifact's currency is checkable AT.
#
# WHY THE BOUNDARY IS PART OF THE DECLARATION (D33-30, Codex MEDIUM). Checking gold or
# prediction currency inside ``step_verify_data_artifacts`` -- which runs BEFORE the steps
# that produce them -- turns an ORDERING FACT into a stale-artifact refusal, and a gate
# that cries wolf on a correct run is worse than no gate. Each artifact therefore names
# the point in the run at which it can honestly be asked whether it is current.
ARTIFACT_BOUNDARY_DATA = "data"
ARTIFACT_BOUNDARY_GOLD = "gold"
ARTIFACT_BOUNDARY_PREDICTIONS = "predictions"

ARTIFACT_PHASE_BOUNDARIES: tuple[str, ...] = (
    ARTIFACT_BOUNDARY_DATA,
    ARTIFACT_BOUNDARY_GOLD,
    ARTIFACT_BOUNDARY_PREDICTIONS,
)

# The recovery command each boundary's refusal names. One per boundary rather than one
# generic line, because "rebuild something" is not an instruction.
_BOUNDARY_RECOVERY_COMMAND: dict[str, str] = {
    ARTIFACT_BOUNDARY_DATA: "uv run python scripts/friday_pipeline.py --data-only",
    ARTIFACT_BOUNDARY_GOLD: "uv run python -m scripts.build_features --all-seasons --save",
    ARTIFACT_BOUNDARY_PREDICTIONS: (
        "uv run python scripts/friday_pipeline.py --predictions-only"
    ),
}


def _parquet_rows_cover(
    path: str, season: int, week: int, *, season_column: str, week_column: str
) -> bool:
    """True when the parquet at *path* carries at least one row for ``(season, week)``.

    Only the two key columns are read, so the check costs a column projection rather than
    a full matrix load -- it runs on every artifact at every boundary of every run.

    AN UNREADABLE FILE ANSWERS FALSE, and that is the honest answer to the question being
    asked: a zero-byte stub or a corrupt parquet carries no row for this week, or for any
    other. It is reported through the STALE class, whose message says the artifact does
    not cover the week -- which is true of an unreadable file too.
    """
    import pandas as pd

    try:
        frame = pd.read_parquet(path, columns=[season_column, week_column])
    except Exception:  # noqa: BLE001 - any read failure means coverage is unprovable
        return False
    if frame.empty:
        return False
    return bool(((frame[season_column] == season) & (frame[week_column] == week)).any())


def _covers_season_week(path: str, season: int, week: int) -> bool:
    """The default coverage check: does this artifact carry a row for THIS week?

    IT ASKS NOTHING ELSE. No completeness requirement of any kind -- not a full regular
    season, not a minimum row count, not a contiguous week range. A week-2 run against a
    two-week-old season is the normal case the live cold start produces every September,
    and a completeness check would refuse it on its second Friday while reporting a stale
    artifact. The question is coverage OF THE CURRENT ``(season, week)``, full stop.
    """
    return _parquet_rows_cover(
        path, season, week, season_column="season", week_column="week"
    )


def _covers_target_season_week(path: str, season: int, week: int) -> bool:
    """The coverage check for ``team_form_features``, which has no ``week`` column.

    Its week is ``target_week`` -- the week the form is computed FOR -- and its season is
    ``target_season``. This is why the check is a PER-ARTIFACT callable rather than one
    hardcoded pair of column names: a single spelling would answer False for this artifact
    on every week of every season, and a gate that always refuses is as useless as one
    that never does.
    """
    return _parquet_rows_cover(
        path, season, week, season_column="target_season", week_column="target_week"
    )


# Single source of truth for the data artifacts the phase-boundary integrity gates
# require before the run may proceed past each boundary.
# Factored into one place so a drift regression test can import the same list
# the gates check (rather than re-hardcoding a second copy that could silently
# diverge from the real build-script output names).
#
# EACH ENTRY IS A ``(path, phase_boundary, coverage_check)`` TRIPLE (D33-30). The triple
# shape exists so the drift test can still import ONE object: a parallel list of coverage
# callables, or a second mapping of boundaries, would break exactly the single-source
# property this constant was created for -- two places to edit is how the original
# BLOCKER (a gate requiring two filenames no build script writes) shipped.
#
# The PREDICTIONS boundary carries no entry here and that is not an omission. The
# prediction artifact's name embeds the season and the week, so it has no static path to
# list; ``step_verify_prediction_currency`` resolves it per run and checks it there.
_REQUIRED_ARTIFACTS = [
    ("data/silver/games.parquet", ARTIFACT_BOUNDARY_DATA, _covers_season_week),
    (
        "data/silver/elo_game_snapshots.parquet",
        ARTIFACT_BOUNDARY_DATA,
        _covers_season_week,
    ),
    (
        "data/silver/team_form_features.parquet",
        ARTIFACT_BOUNDARY_DATA,
        _covers_target_season_week,
    ),
    ("data/gold/features_wp.parquet", ARTIFACT_BOUNDARY_GOLD, _covers_season_week),
    ("data/gold/features_ats.parquet", ARTIFACT_BOUNDARY_GOLD, _covers_season_week),
    ("data/gold/features_ou.parquet", ARTIFACT_BOUNDARY_GOLD, _covers_season_week),
]


def _verify_artifacts_at_boundary(boundary: str, season: int, week: int) -> None:
    """Check every artifact declared at *boundary*, reporting MISSING and STALE apart.

    Args:
        boundary: One of ``ARTIFACT_PHASE_BOUNDARIES``.
        season: The season the artifacts must cover.
        week: The week the artifacts must cover.

    Raises:
        RuntimeError: when an artifact is absent -- the original message, unchanged.
        StaleDataArtifactError: when an artifact exists and carries no row for the week.
    """
    scoped = [
        (path, check)
        for path, artifact_boundary, check in _REQUIRED_ARTIFACTS
        if artifact_boundary == boundary
    ]

    missing = [path for path, _check in scoped if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Missing data artifacts: {', '.join(missing)}")

    stale = [path for path, check in scoped if not check(path, season, week)]
    if stale:
        raise StaleDataArtifactError(
            f"Stale {boundary} artifact(s) for {season} week {week}: "
            f"{', '.join(stale)}. Each of these files EXISTS and carries no row for "
            f"{season} week {week}, so a run that proceeded would score the current "
            "week against inputs that do not contain it -- which is indistinguishable "
            "from a correct run right up until the prediction is published. This is a "
            "CURRENCY refusal, not a missing-file one: the artifact is there. Rebuild "
            f"it with `{_BOUNDARY_RECOVERY_COMMAND[boundary]}` and re-run."
        )


# ---------------------------------------------------------------------------
# Step adapter functions -- deferred imports, no module-level script imports
# ---------------------------------------------------------------------------

# DATA PHASE (9 steps) --------------------------------------------------


class LiveCaptureFailedError(RuntimeError):
    """``run_capture`` reported a non-zero exit code: the week has no live-zone record.

    RAISED RATHER THAN LOGGED (T-33-36). A capture that failed and said nothing leaves the
    run to proceed against LAST week's zone, and every downstream artifact then carries an
    upstream attribution that is quietly wrong. There is no honest degraded mode here: the
    thing the capture produces is the evidence of what upstream served, and a run without
    it is a run nobody can reconstruct.
    """


class LiveCaptureUsageError(RuntimeError):
    """The capture was ASKED for something it does not own -- an operator-shaped failure.

    A SEPARATE CLASS FROM :class:`LiveCaptureFailedError`, because the fixes differ. A
    usage error means nothing was fetched and nothing was written: the run pointed the live
    tool at a season the SEALED pin owns, or at a call it cannot serve. A capture failure
    means the attempt was legitimate and broke. Reporting both as one class would send an
    operator to the network logs for a zone-ownership mistake.
    """


def step_capture_live_season() -> None:
    """Capture what nflverse is serving RIGHT NOW, before anything consumes it (D32-04).

    THE FIRST STEP OF THE WHOLE REGISTRY, and the position is the point. This records the
    upstream bytes this run is about to build on; a capture taken after ``ingest_games``
    would attest to a slightly different upstream than the one the run actually used, which
    is precisely the confusion a capture exists to remove.

    IT DISCHARGES A NAMED PHASE-32 OBLIGATION. ``scripts/capture_live_season.py`` shipped
    as a standalone CLI whose own docstring says it is "NOT WIRED INTO THE PIPELINE,
    DELIBERATELY (D32-04)" and names Phase 33 as the phase that wires it. An unwired
    capture is a capture nobody runs.

    IT CALLS ``run_capture`` AND NEVER THE CLI ENTRY POINT (Codex HIGH, verified against
    live source). That entry point parses argv and, with no ``--week``, prints a usage error
    and returns ``EXIT_USAGE`` -- so a zero-argument step routed through it would return a
    usage error every Friday while looking like it ran. ``run_capture`` takes the season and
    the week as arguments, which is exactly what a step can supply. ``datasets`` and
    ``data_root`` are computed the way the CLI computes them, read off its own construction
    rather than invented, so the scheduled path and the hand-run path capture the same
    datasets into the same root.

    THE WEEK COMES FROM THE SHARED RESOLVER, not from a second resolution inside this step.
    Two resolutions can disagree across a midnight boundary, and a capture filed under a
    week nobody ingested is worse than no capture.

    Raises:
        LiveCaptureUsageError: the live tool was asked for a season it does not own, or for
            a call it cannot serve. Nothing was fetched and nothing was written.
        LiveCaptureFailedError: the capture was attempted and reported a non-zero code.
    """
    from data.upstream_pin import DATASET_COLUMNS, ZoneWriteRefused, default_data_root
    from scripts.capture_live_season import (
        EXIT_OK,
        EXIT_USAGE,
        LIVE_MANIFEST_DIR,
        run_capture,
    )

    season, week = _resolve_current_week()
    datasets = sorted(DATASET_COLUMNS)

    try:
        code = run_capture(
            datasets,
            season,
            week,
            data_root=Path(default_data_root()),
            manifest_dir=Path(LIVE_MANIFEST_DIR),
        )
    except ZoneWriteRefused as refusal:
        # The CLI turns this into EXIT_USAGE in its own boundary handler; a direct caller
        # has to do the same or the operator gets a bare traceback naming a zone rule.
        raise LiveCaptureUsageError(
            f"the live capture refused {season} week {week}: {refusal}. Nothing was "
            "fetched and nothing was written. This season is not the LIVE one -- the "
            "SEALED pin owns it, and `scripts/pin_upstream_snapshot.py` is the tool for "
            "that zone. Point the run at the season the live zone owns."
        ) from refusal

    if code == EXIT_OK:
        return

    if code == EXIT_USAGE:
        raise LiveCaptureUsageError(
            f"the live capture for {season} week {week} returned the USAGE code {code}: "
            "it was asked for something it cannot serve, so nothing was captured. This is "
            "an argument-shaped failure, not a fetch that broke -- check the season and "
            "week the run resolved before looking at the network."
        )

    raise LiveCaptureFailedError(
        f"the live capture for {season} week {week} FAILED with exit code {code}. The "
        "live zone therefore carries no record of what upstream served this run, and "
        "every DATA step after this one would build on an unattributable snapshot. Re-run "
        f"`uv run python -m scripts.capture_live_season --season {season} --week {week}` "
        "and read its output before restarting the pipeline."
    )


def step_ingest_games() -> None:
    """Ingest current week games data via nflreadpy, for the RESOLVED season.

    The season is resolved HERE, through the shared ``_resolve_current_week``, and passed
    in explicitly. Behaviour is unchanged -- ``ingest_games(seasons=None)`` resolved the
    same value internally -- but the pair this step operates on is now observable at the
    step boundary, which is what lets a test assert that the capture step recorded the SAME
    week this one ingested rather than merely that both call the same helper.
    """
    from scripts.ingest_games import GameDataIngester

    season, _week = _resolve_current_week()
    ingester = GameDataIngester()
    ingester.ingest_games(seasons=[season])


def step_ingest_weather() -> None:
    """Ingest the current week's weather via the Open-Meteo FORECAST API (async).

    R8 / Plan 33-09: this step used to reach the Open-Meteo ARCHIVE endpoint, which
    is a reanalysis product and cannot answer for a game that has not been played.
    `WeatherDataIngester.ingest_weather` now reads the FORECAST endpoint, so the
    call site is unchanged and its MEANING is not: a live Friday run now gets real
    forward values instead of nulls.

    ONE WEEK, the current one. A kickoff beyond the declared forecast horizon, or a
    payload missing any game in the week, raises by name and writes nothing --
    never an archive fallback and never an imputed value.
    """
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
    """Update Elo ratings for the current season AND PERSIST THEM (COLD-02, T-33-13).

    THE DEFECT THIS BODY REPLACES. Until Plan 33-03 this step called
    ``builder.update_current_season()``, discarded the return value, and wrote nothing.
    It reported success every Friday while ``data/silver/elo_game_snapshots.parquet``
    never gained a current-season row -- so the gold LEFT JOIN produced NaN Elo for
    every current-season game and the imputer filled those NaNs before the deployed WP
    model ever saw them. Nothing said so, because a step that persisted nothing is
    indistinguishable from one that worked.

    Wiring the OLD ``save_results`` in would have been worse, not better: it wrote the
    snapshot table with ``append_mode=False`` and would have replaced all 24 seasons of
    burn-in with one. That is why the write path was split first
    (``save_full_rebuild`` / ``save_live_append``) and only then wired in here.

    THE ASSERTION DISTINGUISHES TWO CASES THAT LOOK ALIKE FROM A DISTANCE. A season with
    ZERO completed games legitimately computes nothing and persists nothing -- the live
    2026 cold start is exactly that -- and must pass. A NON-ZERO computed count with
    nothing written is the refusal.

    Raises:
        EloSnapshotNotPersistedError: When snapshots were computed and not written.
    """
    from scripts.build_elo import EloBuilder, EloSnapshotNotPersistedError

    builder = EloBuilder()
    update = builder.update_current_season()
    builder.save_live_append(
        update.season,
        snapshots=update.snapshots,
        games_with_elo=update.games_with_elo,
        rating_history=update.rating_history,
    )

    if builder.pending_snapshot_rows:
        raise EloSnapshotNotPersistedError(
            rows=builder.pending_snapshot_rows, season=update.season
        )


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
    """Build weather features for the season TO DATE, refusing what cannot be accounted for.

    THE DEFECT THIS BODY REPLACES (Plan 33-18, owner ruling W1 of 2026-09-15). It used to
    hand the builder EVERY scheduled game in silver. The builder refuses any game with no
    weather record -- correctly, since the old answer was a 65 F dome default that reached
    6,485 of 6,499 gold rows -- so during a live season the step could never pass: the
    current season's opening weeks are never ingested and its future weeks are beyond any
    forecast. The live acceptance run's attempt 2 halted here on ``2026_W01_NE@SEA``.

    THE SCOPE: every completed season, plus the current season THROUGH THE CURRENT WEEK,
    with the week from the shared ``_resolve_current_week`` -- the same resolution the
    capture, ingest and currency gates use -- so this step cannot scope to a different week
    than the one the run ingested. Future weeks are never built and never demanded.

    WHAT IS STILL REFUSED: any game in scope with no weather record, EXCEPT an
    already-played week of the current season. Those are handed to the builder by name as
    ``unobserved_game_ids`` and become explicit no-observation rows (coverage 0.0, every
    measurement null). A completed season missing a record is still refused, and so is the
    week being predicted: a missing forecast for the week the run exists to price stops it.
    """
    from data.storage import load_dataframe, save_dataframe
    from features.weather import SILVER_WEATHER_TABLE, WeatherFeaturesCalculator

    season, week = _resolve_current_week()
    games_df = load_dataframe("games", layer="silver")
    in_scope = games_df.loc[
        (games_df["season"] < season)
        | ((games_df["season"] == season) & (games_df["week"] <= week))
    ]

    weather_df = load_dataframe(SILVER_WEATHER_TABLE, "silver", "parquet")
    played_this_season_unrecorded = (
        (in_scope["season"] == season)
        & (in_scope["week"] < week)
        & ~in_scope["game_id"].isin(weather_df["game_id"])
    )
    unobserved = frozenset(
        str(game_id)
        for game_id in in_scope.loc[played_this_season_unrecorded, "game_id"]
    )

    calculator = WeatherFeaturesCalculator()
    features_df = calculator.build_weather_features(
        games_df=in_scope,
        weather_df=weather_df,
        unobserved_game_ids=unobserved,
    )
    if len(features_df) > 0:
        save_dataframe(features_df, table_name="weather_features", layer="silver")


def step_verify_data_artifacts() -> None:
    """Verify the DATA-boundary artifacts exist AND cover the current week.

    IT CHECKS ONLY THE ``data`` BOUNDARY, and the restriction is the point (D33-30). Gold
    and the prediction artifacts are produced by PREDICTIONS-phase steps that run AFTER
    this gate, so at this instant the current week's gold cannot exist and asking whether
    it is current would report an ordering fact as a stale artifact -- refusing every
    correct full-mode run at the last DATA step. Their currency is checked by
    ``step_verify_gold_currency`` and ``step_verify_prediction_currency``, each registered
    immediately after its own producer.

    EXISTENCE WAS NEVER THE QUESTION THIS GATE MEANT TO ASK. Six bare ``Path.exists()``
    calls pass identically on last season's silver and on this week's, which is why the
    live cold start could reach the prediction phase with a silver layer that stops in
    2025. The two failure classes are reported apart: MISSING keeps its original message,
    STALE raises ``StaleDataArtifactError``.
    """
    season, week = _resolve_current_week()
    _verify_artifacts_at_boundary(ARTIFACT_BOUNDARY_DATA, season, week)


# PREDICTIONS PHASE (13 steps) ------------------------------------------


def step_ingest_odds() -> None:
    """Capture odds snapshot from The Odds API, each row stamped with its game's own lock.

    The step passes the slate's silver ``games`` slice as ``schedule`` -- the one input
    the ingest derives both each game's id and each game's lock from (Plan 33.2-02). It
    used to call ``ingest_odds()`` with no arguments and silently inherit a single
    preceding-Friday instant for the whole request. It computes NO lock map itself: that
    would be a second derivation of what ``utils.game_lock.lock_frame`` owns.
    """
    import scripts.ingest_odds as ingest_odds_module

    season, week = _resolve_current_week()
    schedule = ingest_odds_module.load_schedule_slice(season, week)
    ingester = ingest_odds_module.OddsDataIngester()
    ingester.ingest_odds(season=season, week=week, schedule=schedule)


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
    """Create unified feature matrices for all model targets.

    Passes the live-skip register (Plan 33.2-03, D33.2-05): the games this run has already
    dropped are removed from every source before the information-time gate, so a re-run after a
    skip checks the REMAINING games rather than re-refusing a game already recorded. The register
    is empty on a clean run, and the call is then the one it always was.
    """
    from pipeline import live_skip
    from scripts.build_features import FeatureMatrixBuilder

    builder = FeatureMatrixBuilder()
    builder.generate_feature_matrices(excluded_game_ids=live_skip.excluded_games())


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


def step_verify_gold_currency() -> None:
    """The gold matrices exist AND carry rows for the current week (R9, D33-30).

    WHERE R9'S SECOND REFUSAL NATURALLY LIVES. ``build_weekly_candidates`` already raises
    "gold matrix ... has no rows for {season} week {week}" at SELECTION time, and that
    error stays exactly where it is, as the backstop. But selection is four steps and one
    whole prediction later: by then the run has scored artifacts, written a predictions
    CSV and exported it, and the operator learns that the week could not be selected only
    at the end. A currency check at the GOLD/PREDICTION boundary is the SAME assertion at
    the first point in the run where gold actually exists -- one step earlier, and louder,
    because it stops the run instead of emptying its output.

    THE DISTINCTION IT PRESERVES IS THE WHOLE POINT. An empty bet list is either an honest
    no-edge week or a gold build that never produced the current week. Only a refusal can
    tell them apart, and a refusal that arrives before publication is the one that can
    still prevent the confusion.

    Registered AFTER ``build_features`` (which writes the matrices) and BEFORE the
    prediction steps that read them.
    """
    season, week = _resolve_current_week()
    _verify_artifacts_at_boundary(ARTIFACT_BOUNDARY_GOLD, season, week)


def step_verify_prediction_currency() -> None:
    """The current-week prediction file exists AND its ROWS are the current week.

    THE FILENAME IS NOT THE EVIDENCE. ``predictions_2026_week2.csv`` is a name a writer
    chose; the ``season`` / ``week`` columns inside it are what the models actually scored.
    A file whose name says week 2 and whose rows say week 1 passes every existence check
    in the registry and publishes last week's picks under this week's heading.

    Registered AFTER ``generate_predictions`` and BEFORE ``generate_recommendations``, so
    a stale prediction cannot become a bet row, an export or a served cache.

    ``_REQUIRED_ARTIFACTS`` carries no PREDICTIONS-boundary entry because the artifact's
    name embeds the season and the week and therefore has no static path to declare. The
    boundary subset is still checked first, so a future static prediction artifact is
    covered by declaring it and nothing else.
    """
    season, week = _resolve_current_week()
    _verify_artifacts_at_boundary(ARTIFACT_BOUNDARY_PREDICTIONS, season, week)

    import pandas as pd

    from pipeline import live_skip

    pred_path = _predictions_output_dir() / f"predictions_{season}_week{week}.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Missing data artifacts: {pred_path.as_posix()}")

    frame = pd.read_csv(pred_path)

    # A GAME THIS RUN DROPPED MUST HAVE NO PREDICTION ROW (D33.2-05). The writer drops excluded
    # games before scoring; this is the independent check on the file actually written, and it
    # stops the run before the row can become a bet, an export or a served page.
    excluded = live_skip.excluded_games()
    if excluded and "game_id" in frame.columns:
        leaked = sorted(excluded & {str(game_id) for game_id in frame["game_id"]})
        if leaked:
            raise RuntimeError(
                f"{pred_path.as_posix()} carries prediction rows for game(s) this run DROPPED "
                f"under the live-skip rule: {leaked}. A dropped game must have no prediction."
            )

    covered = (
        not frame.empty
        and {"season", "week"} <= set(frame.columns)
        and bool(((frame["season"] == season) & (frame["week"] == week)).any())
    )
    # An EMPTY file is current only on an all-skipped day (SPEC R3): every scheduled game was
    # dropped, so zero rows is the honest output. A week with no scheduled games is not that
    # day and still refuses below.
    all_skipped = frame.empty and _every_scheduled_game_excluded(season, week, excluded)
    if not covered and not all_skipped:
        raise StaleDataArtifactError(
            f"Stale {ARTIFACT_BOUNDARY_PREDICTIONS} artifact for {season} week {week}: "
            f"{pred_path.as_posix()}. The file EXISTS and carries no row for "
            f"{season} week {week}, so its name and its contents disagree about which "
            "week was scored. Publishing it would put a previous week's picks under "
            "this week's heading. Regenerate it with "
            f"`{_BOUNDARY_RECOVERY_COMMAND[ARTIFACT_BOUNDARY_PREDICTIONS]}`."
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

    THE LIVE-SKIP REGISTER IS READ HERE, NEVER WRITTEN (Plan 33.2-03, D33.2-05). First, every
    game whose lock is before this decision instant is refused by name
    (``live_skip.refuse_passed_locks``) -- the standing prohibition that no prediction exists for
    a game whose lock had passed, including after a missed day. That refusal is a per-game one,
    so the orchestrator's seam records and drops those games and re-runs this step. Then the
    register's games are handed to ``generate_and_write``, which drops them BEFORE scoring.
    """
    from pipeline import live_skip
    from scripts.generate_current_week_predictions import generate_and_write
    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    live_skip.refuse_passed_locks(
        _week_schedule(season, week),
        decided_at=_decision_instant(),
        excluded_game_ids=live_skip.excluded_games(),
    )
    generate_and_write(
        season=season,
        week=week,
        output_dir=_predictions_output_dir(),
        excluded_game_ids=live_skip.excluded_games(),
    )


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

    THE LIVE-SKIP REGISTER IS READ HERE, NEVER WRITTEN (Plan 33.2-03, D33.2-05). Games whose lock
    is before this decision instant are refused by name first, exactly as in
    ``step_generate_predictions``; then the register is passed as ``excluded_game_ids``, which
    ``backtest.weekly_bet_list`` drops from the week's schedule SPINE before scoring or pricing
    (Plan 33.2-02's thread), so a dropped game has no bet row of any kind -- not even a
    suppressed one. The decision instant is passed as ``now`` so the refusal here and the
    ``decided_at_utc`` stamp there are one instant.

    Raises:
        Whatever the delegate raises. Nothing is swallowed: a week that cannot be selected must
        record a clean step failure rather than publish a silently empty bet list.
    """
    from backtest.weekly_bet_list import generate_weekly_bet_list
    from pipeline import live_skip
    from utils.date_utils import get_current_nfl_week

    season, week = get_current_nfl_week()
    decided_at = _decision_instant()
    live_skip.refuse_passed_locks(
        _week_schedule(season, week),
        decided_at=decided_at,
        excluded_game_ids=live_skip.excluded_games(),
    )
    generate_weekly_bet_list(
        season=season,
        week=week,
        output_dir=_bet_list_output_dir(),
        now=decided_at,
        excluded_game_ids=live_skip.excluded_games(),
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

    from pipeline import live_skip

    df = pd.read_csv(pred_path)
    # Zero rows is valid ONLY on an all-skipped day (SPEC R3, D33.2-05): every scheduled game was
    # dropped under the live-skip rule. Any other empty file is still a failed prediction run.
    if df.empty and not _every_scheduled_game_excluded(
        season, week, live_skip.excluded_games()
    ):
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
    """Build the complete 22-step pipeline registry.

    Returns:
        Ordered list of StepDefinitions covering data and prediction phases.
    """
    return [
        # DATA PHASE (9 steps)
        #
        # FIRST, and the position is load-bearing (Plan 33-07, D32-04). The capture records
        # what nflverse served THIS RUN; anything that reads upstream before it has been
        # captured is unattributable. Its index is pinned by
        # tests/unit/test_step_registry_order.py so a future insertion cannot slide it below
        # the ingest it exists to attest to.
        StepDefinition(
            "capture_live_season",
            step_capture_live_season,
            PipelinePhase.DATA,
            # CRITICAL: a failed capture means the live zone has no row for this week, and
            # every downstream DATA step would then run against last week's zone with
            # nothing saying so (T-33-36).
            critical=True,
            # RETRYABLE on the same terms as the three ingest steps: it fetches over the
            # network. A retry is safe because the manifest is APPEND-ONLY -- a second
            # capture of the same week is a new sequence entry, never a rewrite.
            retryable=True,
            max_retries=3,
            description="Capture the live nflverse season into the append-only live zone",
        ),
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
            # CRITICAL since Plan 33-18 (owner ruling W1, 2026-09-15). As a non-critical
            # step its refusal DEGRADED the run, and build_features then rebuilt gold from a
            # weather-features table that did not cover the week being predicted -- a
            # silent no-weather gold. A refusal here must stop the run.
            critical=True,
            retryable=False,
            description="Build weather features for the season to date",
        ),
        StepDefinition(
            "verify_data_artifacts",
            step_verify_data_artifacts,
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            description="Verify data artifacts before predictions",
        ),
        # PREDICTIONS PHASE (13 steps)
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
        # The GOLD boundary (Plan 33-07, D33-30). Registered here and not one step
        # earlier: gold does not exist until ``build_features`` has run, so this is the
        # first instant at which "does gold carry the current week?" is a question with
        # an honest answer. See step_verify_gold_currency's docstring for why R9's second
        # refusal belongs here rather than only at selection time.
        StepDefinition(
            "verify_gold_currency",
            step_verify_gold_currency,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Verify the gold matrices carry the current week",
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
        # The PREDICTIONS boundary (Plan 33-07, D33-30). After the file is written and
        # BEFORE anything consumes it: the bet list, the export and the served cache all
        # read this week's predictions, so a stale one caught here reaches none of them.
        StepDefinition(
            "verify_prediction_currency",
            step_verify_prediction_currency,
            PipelinePhase.PREDICTIONS,
            critical=True,
            retryable=False,
            description="Verify the prediction file's rows are the current week",
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
