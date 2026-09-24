"""The steps of the DAILY lock-time run (Plan 33.2-27; D33.2-01, D33.2-18 as amended).

Every game locks at 18:00 ET on the ET calendar day before its kickoff (``utils.game_lock``), so
all of tomorrow's games share ONE lock: 18:00 ET today. The daily run predicts exactly those games
-- its SLATE -- and nothing else.

THE ORDER, AND WHY THE WHOLE RUN FINISHES BEFORE THE LOCK
---------------------------------------------------------
OWNER RULING 2026-09-23, Task 2c: "Finish before 6 PM (Recommended)". Collection, the gold build,
prediction and emission ALL complete at or before the slate's lock. So every emitted row's
``computed_at_utc`` is before its lock, the forward bet row's ``decided_at_utc`` is that same true
computation instant, and ``backtest.weekly_bet_list.assert_decided_at_before_freeze`` stands
exactly as written. A run that is too slow is refused per game (``live_skip.refuse_passed_locks``
at the computation instant), never back-dated.

0. (In ``scripts/daily_lock_pipeline.py``, before the slate is known) the live nflverse capture
   and the schedule ingest it feeds -- the schedule refresh.
1. COLLECTION -- the slate's weather forecasts, snaps, injuries and the slate's odds. :data:`COLLECTION_STEP_NAMES`; ``close_collection`` stamps ``captured_at_utc``
   and refuses the slate if collection itself ended after the lock.
2. BUILD -- Elo (with a flagged PROVISIONAL pre-game row for each slate game), team form,
   contextual, weather features, then the FULL-HISTORY gold build, saved.
3. PREDICT AND EMIT -- the slate's predictions, merged into the week's file with three stamps
   (``captured_at_utc``, ``information_cutoff_utc`` = the lock, ``computed_at_utc``), then the
   bet list, the exports and the web cache.

THE BUILD IS THE FULL HISTORY, EVERY NIGHT (OWNER RULING 2026-09-23)
--------------------------------------------------------------------
"Rebuild everything nightly (Recommended)", superseding D33.2-19's scoped build. Measured: a
one-season build diverges from the full build on 139 of 186 numeric columns for every 2025 game
(up to ~7 SD), because scaling, imputation and clipping are fitted on prior seasons. The full
build takes about 591.5 s, so the scheduler starts the run early enough to finish before the lock.

Why that does not bring back RESEARCH pitfall P10 (a nightly rebuild letting one defective
historical row stop the whole build and deny predictions to that night's clean games): the gate's
refusals are PER GAME, and the live-skip rule (``pipeline.live_skip``, D33.2-05) drops only the
games a refusal names -- a bad 2011 row skips that 2011 game, never tomorrow's slate.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.steps import (
    PipelinePhase,
    StepDefinition,
    persist_current_season_elo,
    step_build_contextual,
    step_build_features,
    step_build_team_form,
    step_data_qa,
    step_export_artifacts,
    step_ingest_injuries,
    step_ingest_snaps,
    step_populate_web_cache,
    step_validate_features,
    step_validate_models,
    step_validate_predictions,
    step_verify_data_artifacts,
    step_verify_gold_currency,
    step_verify_output_files,
    step_verify_prediction_currency,
)

if TYPE_CHECKING:
    import pandas as pd

#: Task 2c's ruling, echoed by the daily entry point's output contract.
DECISION_TIME_BRANCH = "finish-before-lock"

#: The three instants every emitted prediction row carries.
STAMP_COLUMNS: tuple[str, ...] = (
    "captured_at_utc",
    "information_cutoff_utc",
    "computed_at_utc",
)

#: The collection stage, in order. The snap and injury ingests sit here, BEFORE any build step,
#: so tomorrow's games are built on today's capture (D33.2-16); pinned by
#: ``tests/unit/test_step_registry_order.py``.
COLLECTION_STEP_NAMES: tuple[str, ...] = (
    "ingest_weather",
    "ingest_snaps",
    "ingest_injuries",
    "ingest_odds",
    "close_collection",
)


def slate_lock(run_date_et: date) -> datetime:
    """The lock every game kicking off on ``run_date_et + 1`` shares: 18:00 ET on *run_date_et*.

    Taken from the ONE rule by asking it for the lock of a kickoff on that day, so this module
    cannot state the lock hour a second time.
    """
    from utils.game_lock import ET, game_lock

    tomorrow_noon = datetime.combine(
        run_date_et + timedelta(days=1), time(12), tzinfo=ET
    )
    return game_lock(tomorrow_noon)


@dataclass
class DailySlate:
    """Tomorrow's games and their shared lock.

    Attributes:
        run_date_et: The ET calendar day of the run.
        lock: The slate's lock (tz-aware), 18:00 ET on ``run_date_et``.
        schedule: The slate's silver ``games`` rows.
        captured_at_utc: When collection finished; set by ``close_collection``.
    """

    run_date_et: date
    lock: datetime
    schedule: pd.DataFrame
    captured_at_utc: datetime | None = None

    @property
    def game_ids(self) -> frozenset[str]:
        """The slate's game ids."""
        return frozenset(str(game_id) for game_id in self.schedule["game_id"])

    @property
    def season(self) -> int:
        """The slate's season (one ET day names one slate)."""
        return int(self.schedule["season"].iloc[0])

    @property
    def week(self) -> int:
        """The slate's week (one ET day names one slate)."""
        return int(self.schedule["week"].iloc[0])


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def ingest_slate_weather(slate: DailySlate) -> None:
    """Capture the forecast for the SLATE's games only.

    Never the whole week: a game later in the week whose lock has already passed (Sunday night's
    game on a Sunday-afternoon run) must not have its pre-lock forecast replaced by a post-lock one.
    """
    from scripts.ingest_weather import WeatherDataIngester

    ingester = WeatherDataIngester()
    games = ingester._load_games_data(slate.season, slate.week)
    games = games.loc[games["game_id"].astype(str).isin(sorted(slate.game_ids))]
    ingester.ingest_week_forecast(
        games, ingester._load_venue_data(), as_of_utc=datetime.now(UTC)
    )


def ingest_slate_odds(slate: DailySlate) -> None:
    """Capture odds for the SLATE's games only, so the request window names nothing else."""
    import scripts.ingest_odds as ingest_odds_module

    week_schedule = ingest_odds_module.load_schedule_slice(slate.season, slate.week)
    schedule = week_schedule.loc[
        week_schedule["game_id"].astype(str).isin(sorted(slate.game_ids))
    ].reset_index(drop=True)
    ingest_odds_module.OddsDataIngester().ingest_odds(
        season=slate.season, week=slate.week, schedule=schedule
    )


def close_collection(slate: DailySlate) -> None:
    """Stamp ``captured_at_utc`` and refuse the slate if collection ended after its lock."""
    from pipeline import live_skip

    slate.captured_at_utc = datetime.now(UTC)
    live_skip.refuse_passed_locks(
        slate.schedule,
        decided_at=slate.captured_at_utc,
        excluded_game_ids=live_skip.excluded_games(),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_slate_weather_features(slate: DailySlate) -> None:
    """Build weather features for history, the played games, and every captured game.

    In scope: every earlier season; and, this season, every game that has kicked off, has a
    captured forecast, or is in the slate. A kicked-off game with no record becomes an explicit
    no-observation row (as ``step_build_weather_features`` does for past weeks); a SLATE game with
    no forecast is refused by the builder, because it is the game being priced. An unplayed game
    with no capture is out of scope: it is not in tonight's gold either (it has no Elo row).
    """
    import pandas as pd

    from data.storage import load_dataframe, save_dataframe
    from features.weather import SILVER_WEATHER_TABLE, WeatherFeaturesCalculator

    games_df = load_dataframe("games", layer="silver")
    weather_df = load_dataframe(SILVER_WEATHER_TABLE, "silver", "parquet")

    ids = games_df["game_id"].astype(str)
    recorded = ids.isin(set(weather_df["game_id"].astype(str)))
    kicked_off = pd.to_datetime(games_df["kickoff_et"], utc=True) <= datetime.now(UTC)
    this_season = games_df["season"] == slate.season
    in_slate = ids.isin(sorted(slate.game_ids))

    in_scope_mask = (games_df["season"] < slate.season) | (
        this_season & (kicked_off | recorded | in_slate)
    )
    unobserved = frozenset(ids[in_scope_mask & this_season & kicked_off & ~recorded])

    features_df = WeatherFeaturesCalculator().build_weather_features(
        games_df=games_df.loc[in_scope_mask],
        weather_df=weather_df,
        unobserved_game_ids=unobserved,
    )
    if len(features_df) > 0:
        save_dataframe(features_df, table_name="weather_features", layer="silver")


# ---------------------------------------------------------------------------
# Predict and emit
# ---------------------------------------------------------------------------


def _merge_into_week_file(frame: pd.DataFrame, path: Path, kind: str) -> None:
    """Replace this run's games in the week's CSV, keep every other game's row, write it."""
    import pandas as pd

    from data.write_sink import write_csv

    if path.exists():
        stored = pd.read_csv(path)
        if not stored.empty:
            stored = stored.loc[
                ~stored["game_id"].astype(str).isin(frame["game_id"].astype(str))
            ]
            frame = pd.concat([stored, frame], ignore_index=True)
    write_csv(frame, path, kind=kind)


def predict_slate(slate: DailySlate) -> None:
    """Score the slate, refuse any game computed after its lock, stamp and merge the rows.

    Every slate game ends with a prediction row or a recorded skip; one that has neither is a
    refusal naming it, never a silent gap.
    """
    import pandas as pd

    from pipeline import live_skip
    from pipeline.steps import _predictions_output_dir
    from scripts.generate_current_week_predictions import (
        PREDICTION_OUTPUT_COLUMNS,
        build_game_context,
        build_predictions,
    )

    if slate.captured_at_utc is None:
        msg = "the slate has no captured_at_utc: collection did not complete"
        raise RuntimeError(msg)

    excluded = live_skip.excluded_games()
    in_scope = slate.game_ids - excluded
    combined = build_predictions(
        slate.season,
        slate.week,
        excluded_game_ids=excluded,
        only_game_ids=in_scope,
    )

    # FINISH BEFORE THE LOCK (Task 2c). The computation instant is judged against each game's
    # own lock; a late game is refused and recorded, never back-dated.
    computed_at = datetime.now(UTC)
    live_skip.refuse_passed_locks(
        slate.schedule, decided_at=computed_at, excluded_game_ids=excluded
    )

    scored = set(combined["game_id"].astype(str)) if not combined.empty else set()
    unpredicted = sorted(in_scope - scored)
    if unpredicted:
        msg = (
            f"{len(unpredicted)} slate game(s) produced no prediction and were not skipped: "
            f"{unpredicted}. Every slate game must be predicted or skipped with a reason."
        )
        raise RuntimeError(msg)

    rows = combined.reindex(columns=PREDICTION_OUTPUT_COLUMNS)
    rows["captured_at_utc"] = slate.captured_at_utc.isoformat()
    rows["information_cutoff_utc"] = slate.lock.astimezone(UTC).isoformat()
    rows["computed_at_utc"] = computed_at.isoformat()

    output_dir = _predictions_output_dir()
    stem = f"{slate.season}_week{slate.week}"
    _merge_into_week_file(
        rows, output_dir / f"predictions_{stem}.csv", "daily_predictions_csv"
    )
    context = (
        build_game_context(sorted(scored), slate.season, slate.week)
        if scored
        else pd.DataFrame(columns=pd.Index(["game_id"]))
    )
    _merge_into_week_file(
        context, output_dir / f"game_context_{stem}.csv", "daily_game_context_csv"
    )


def recommend_slate(slate: DailySlate) -> None:
    """Select the slate's bets. ``decided_at_utc`` is the true computation instant (Task 2c)."""
    from backtest.weekly_bet_list import generate_weekly_bet_list
    from pipeline import live_skip
    from pipeline.steps import _bet_list_output_dir, _week_schedule

    decided_at = datetime.now(UTC)
    excluded = live_skip.excluded_games()
    live_skip.refuse_passed_locks(
        slate.schedule, decided_at=decided_at, excluded_game_ids=excluded
    )
    week_ids = {str(g) for g in _week_schedule(slate.season, slate.week)["game_id"]}
    outside_slate = frozenset(week_ids - slate.game_ids)
    generate_weekly_bet_list(
        season=slate.season,
        week=slate.week,
        output_dir=_bet_list_output_dir(),
        now=decided_at,
        excluded_game_ids=live_skip.excluded_games() | outside_slate,
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def build_daily_step_registry(slate: DailySlate) -> list[StepDefinition]:
    """The daily run's steps, collection first. Criticality mirrors the Friday registry."""
    collect, build = PipelinePhase.DATA, PipelinePhase.PREDICTIONS

    def step(name, fn, phase, *, critical=True, retryable=False, description=""):
        return StepDefinition(
            name,
            fn,
            phase,
            critical=critical,
            retryable=retryable,
            max_retries=3,
            description=description,
        )

    return [
        # -- COLLECTION (the nflverse capture already ran: it is the schedule refresh) ----
        step(
            "ingest_weather",
            lambda: ingest_slate_weather(slate),
            collect,
            critical=False,
            retryable=True,
            description="Capture the slate's weather forecasts",
        ),
        step(
            "ingest_snaps",
            step_ingest_snaps,
            collect,
            critical=False,
            retryable=True,
            description="Capture the season's snap counts",
        ),
        step(
            "ingest_injuries",
            step_ingest_injuries,
            collect,
            critical=False,
            retryable=True,
            description="Capture the season's injury reports",
        ),
        step(
            "ingest_odds",
            lambda: ingest_slate_odds(slate),
            collect,
            retryable=True,
            description="Capture the slate's odds",
        ),
        step(
            "close_collection",
            lambda: close_collection(slate),
            collect,
            description="Stamp captured_at_utc; refuse a capture after the lock",
        ),
        # -- BUILD -----------------------------------------------------------------------
        step("data_qa", step_data_qa, build, description="Data quality validation"),
        step(
            "build_elo",
            lambda: persist_current_season_elo(slate.game_ids),
            build,
            description="Update Elo, with provisional rows for the slate",
        ),
        step("build_team_form", step_build_team_form, build, description="Team form"),
        step(
            "build_contextual", step_build_contextual, build, description="Contextual"
        ),
        step(
            "build_weather_features",
            lambda: build_slate_weather_features(slate),
            build,
            description="Weather features through the slate",
        ),
        step(
            "verify_data_artifacts",
            step_verify_data_artifacts,
            build,
            description="Verify silver covers the week",
        ),
        step(
            "build_features",
            step_build_features,
            build,
            description="Full-history gold build, saved",
        ),
        step(
            "validate_features",
            step_validate_features,
            build,
            description="Leakage scan of gold",
        ),
        step(
            "verify_gold_currency",
            step_verify_gold_currency,
            build,
            description="Gold carries the week",
        ),
        step("validate_models", step_validate_models, build, description="Models load"),
        # -- PREDICT AND EMIT ------------------------------------------------------------
        step(
            "generate_predictions",
            lambda: predict_slate(slate),
            build,
            description="Predict the slate, stamped, before its lock",
        ),
        step(
            "verify_prediction_currency",
            step_verify_prediction_currency,
            build,
            description="The week's prediction file is current",
        ),
        step(
            "generate_recommendations",
            lambda: recommend_slate(slate),
            build,
            description="Select the slate's bets",
        ),
        step(
            "export_artifacts",
            step_export_artifacts,
            build,
            description="Export predictions JSON",
        ),
        step(
            "validate_predictions",
            step_validate_predictions,
            build,
            description="Validate the week's predictions",
        ),
        step(
            "verify_output_files",
            step_verify_output_files,
            build,
            critical=False,
            description="Output files exist",
        ),
        step(
            "populate_web_cache",
            step_populate_web_cache,
            build,
            critical=False,
            description="Rebuild the web cache",
        ),
    ]
