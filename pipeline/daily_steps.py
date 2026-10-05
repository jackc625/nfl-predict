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
   weather features, then the FULL-HISTORY gold build, saved.
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

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.steps import (
    PipelinePhase,
    StepDefinition,
    build_and_save_gold,
    persist_current_season_elo,
    step_build_team_form,
    step_data_qa,
    step_ingest_injuries,
    step_ingest_snaps,
    step_populate_web_cache,
    step_validate_features,
    step_validate_models,
    step_verify_data_artifacts,
    step_verify_output_files,
)
from utils.logging_config import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

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
        weather_failures: ``game_id -> reason`` for each slate game whose forecast failed to
            fetch; set by ``ingest_slate_weather``.
        weather_unknown: ``game_id -> reason`` for each slate game built with its weather
            unknown (no forecast on record); set by ``build_slate_weather_features``.
        odds_failure: Why the slate's odds capture failed, or None; set by
            ``capture_slate_odds``.
        odds_missing: ``game_id -> reason`` for each slate game with no admissible pre-lock
            line on record; set by ``close_collection``. Such a game is still predicted -- the
            models use no market input -- with its market side blank and no bet.
    """

    run_date_et: date
    lock: datetime
    schedule: pd.DataFrame
    captured_at_utc: datetime | None = None
    weather_failures: dict[str, str] = field(default_factory=dict)
    weather_unknown: dict[str, str] = field(default_factory=dict)
    odds_failure: str | None = None
    odds_missing: dict[str, str] = field(default_factory=dict)

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

    One game's failed forecast never costs the others theirs (step 27b): each failure is recorded
    in ``slate.weather_failures`` and that game is built with its weather unknown.

    Any OTHER error ends the capture with nothing written (the write is all-or-nothing and
    last), so every slate game is recorded as failed with that reason before it is re-raised:
    the night then follows the same policy (D33.2-05), not a halt at ``data_qa``. Each attempt
    starts from an empty record, so a retry that succeeds leaves no stale failure behind.
    """
    from scripts.ingest_weather import WeatherDataIngester

    slate.weather_failures.clear()
    try:
        ingester = WeatherDataIngester()
        games = ingester._load_games_data(slate.season, slate.week)
        games = games.loc[games["game_id"].astype(str).isin(sorted(slate.game_ids))]
        ingester.ingest_week_forecast(
            games,
            ingester._load_venue_data(),
            as_of_utc=datetime.now(UTC),
            failed_games=slate.weather_failures,
        )
    except Exception as exc:
        reason = f"the forecast capture failed ({type(exc).__name__}: {exc})"
        for game_id in sorted(slate.game_ids):
            slate.weather_failures.setdefault(game_id, reason)
        raise


#: What a no-write dry run captures odds from instead of the PAID Odds API (33.2 review C1
#: WR-02): the ingest's own mock board, built from the slate week's silver games. The dry run
#: prints this so its record says the odds step ran on a fixture, not on the market.
DRY_RUN_ODDS_SOURCE = (
    "fixture: the odds client's mock board for the slate's week "
    "(no paid Odds API request)"
)


def ingest_slate_odds(slate: DailySlate, *, fixture: bool = False) -> None:
    """Capture odds for the SLATE's games only, so the request window names nothing else.

    Args:
        slate: Tomorrow's games.
        fixture: Run the real ingest code on :data:`DRY_RUN_ODDS_SOURCE` instead of a paid
            request -- the no-write dry run's mode. A dry run must never spend Odds API credits.
    """
    import scripts.ingest_odds as ingest_odds_module

    week_schedule = ingest_odds_module.load_schedule_slice(slate.season, slate.week)
    schedule = week_schedule.loc[
        week_schedule["game_id"].astype(str).isin(sorted(slate.game_ids))
    ].reset_index(drop=True)
    ingester = ingest_odds_module.OddsDataIngester()
    if fixture:
        ingester.api_client.mock_mode = True
        ingester.api_client.mock_season_week = (slate.season, slate.week)
    ingester.ingest_odds(season=slate.season, week=slate.week, schedule=schedule)


def capture_slate_odds(slate: DailySlate) -> None:
    """The daily odds step: capture the slate's odds, remembering why a capture failed.

    NON-CRITICAL in the daily registry (33.2 review, batch 1a follow-up; OWNER REQUIREMENT:
    every slate game is predicted with the model's own numbers, whatever the market does). The
    models use no market input, so a failed or empty odds capture must not stop predictions.
    The failure is re-raised -- the orchestrator still retries a transient one and records the
    step as failed -- and ``close_collection`` names every game left without a line.
    """
    slate.odds_failure = None
    try:
        ingest_slate_odds(slate)
    except Exception as exc:
        slate.odds_failure = f"{type(exc).__name__}: {exc}"
        raise


def record_missing_odds(slate: DailySlate) -> None:
    """Name every slate game with no admissible pre-lock line, with the reason.

    Read through the SAME reader the predictions price with (``load_market_data``: the latest
    capture at or before each game's lock), so "missing" here means exactly "published with a
    blank market side and no bet". Never raises: a failure to read the odds names every game.
    """
    from scripts.generate_current_week_predictions import load_market_data

    reason = (
        f"the odds capture failed ({slate.odds_failure})"
        if slate.odds_failure
        else "no pre-lock line was captured for it"
    )
    try:
        market = load_market_data(sorted(slate.game_ids))
        priced = market.dropna(
            how="all", subset=[c for c in market.columns if c != "game_id"]
        )
        missing = slate.game_ids - set(priced["game_id"].astype(str))
    except Exception as exc:  # noqa: BLE001 - naming the games is the point; never fatal
        missing = slate.game_ids
        reason = f"the stored odds could not be read ({type(exc).__name__}: {exc})"
    for game_id in sorted(missing):
        slate.odds_missing[game_id] = reason
        logger.warning(
            "Slate game has no market line: predicted with its market side blank, no bet",
            game_id=game_id,
            reason=reason,
        )


def close_collection(slate: DailySlate) -> None:
    """Stamp ``captured_at_utc``, name the games with no line, and refuse a late collection."""
    from pipeline import live_skip

    slate.captured_at_utc = datetime.now(UTC)
    record_missing_odds(slate)
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
    no-observation row (as ``step_build_weather_features`` does for past weeks). So does a SLATE
    game with no forecast (step 27b, owner requirement: every slate game is predicted unless its
    information genuinely cannot be scored): its weather is unknown -- blank, as gold represents
    any unscorable cell -- and it is recorded by name with its reason in ``slate.weather_unknown``
    and the run log (the daily entry point prints a ``WEATHER_UNKNOWN`` line for each). An
    unplayed game with no capture outside the slate is out of scope: it is not in tonight's gold
    either (it has no Elo row).
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
    for game_id in sorted(set(ids[in_slate & ~recorded])):
        reason = slate.weather_failures.get(game_id, "no forecast was captured")
        slate.weather_unknown[game_id] = reason
        logger.warning(
            "Slate game built with weather unknown", game_id=game_id, reason=reason
        )
    unobserved |= frozenset(slate.weather_unknown)

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


def _merge_into_week_file(
    frame: pd.DataFrame,
    path: Path,
    kind: str,
    *,
    drop_game_ids: frozenset[str] = frozenset(),
) -> None:
    """Replace this run's games in the week's CSV, keep every other game's row, write it.

    *drop_game_ids* are removed from the stored file as well: a slate game this run DROPPED
    (D33.2-05) must have no prediction row, even one an earlier run of the same day wrote.
    """
    import pandas as pd

    from data.write_sink import write_csv

    if path.exists():
        stored = pd.read_csv(path)
        if not stored.empty:
            replaced = set(frame["game_id"].astype(str)) | set(drop_game_ids)
            stored = stored.loc[~stored["game_id"].astype(str).isin(replaced)]
            frame = pd.concat([stored, frame], ignore_index=True)
    write_csv(frame, path, kind=kind)


def _week_file(slate: DailySlate, prefix: str) -> Path:
    """The slate week's ``{prefix}_{season}_week{week}.csv`` under the predictions directory."""
    from pipeline.steps import _predictions_output_dir

    return _predictions_output_dir() / f"{prefix}_{slate.season}_week{slate.week}.csv"


def predict_slate(slate: DailySlate) -> None:
    """Score the slate, refuse any game computed after its lock, stamp and merge the rows.

    Every slate game ends with a prediction row or a recorded skip; one that has neither is a
    refusal naming it, never a silent gap.
    """
    import pandas as pd

    from pipeline import live_skip
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
    # AN ALL-SKIPPED SLATE SCORES NOTHING (33.2 review B WR-02): every slate game was dropped,
    # so there is nothing to hand the models -- and asking gold for the week would refuse
    # "no rows" when the dropped games were the only ones of the week built so far.
    combined = (
        build_predictions(
            slate.season,
            slate.week,
            excluded_game_ids=excluded,
            only_game_ids=in_scope,
        )
        if in_scope
        else pd.DataFrame(columns=pd.Index(PREDICTION_OUTPUT_COLUMNS))
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

    # FINISH BEFORE THE LOCK (Task 2c), judged on the REAL CLOCK IMMEDIATELY BEFORE THE WRITE
    # (33.2 review C2 WR-08 / C1 WR-05): no game's prediction is published after its own lock.
    # A late game is refused BY NAME (the live-skip records it and this step re-runs without
    # it), never back-dated; the stamp is this same instant.
    computed_at = datetime.now(UTC)
    live_skip.refuse_passed_locks(
        slate.schedule, decided_at=computed_at, excluded_game_ids=excluded
    )
    rows["captured_at_utc"] = slate.captured_at_utc.isoformat()
    rows["information_cutoff_utc"] = slate.lock.astimezone(UTC).isoformat()
    rows["computed_at_utc"] = computed_at.isoformat()

    dropped = frozenset(slate.game_ids & excluded)
    _merge_into_week_file(
        rows,
        _week_file(slate, "predictions"),
        "daily_predictions_csv",
        drop_game_ids=dropped,
    )
    context = (
        build_game_context(sorted(scored), slate.season, slate.week)
        if scored
        else pd.DataFrame(columns=pd.Index(["game_id"]))
    )
    _merge_into_week_file(
        context,
        _week_file(slate, "game_context"),
        "daily_game_context_csv",
        drop_game_ids=dropped,
    )


# ---------------------------------------------------------------------------
# The slate's own currency checks (33.2 review B WR-02)
# ---------------------------------------------------------------------------
#
# The Friday registry's checks ask about the whole WEEK, and the daily run excludes at SLATE
# grain. When every slate game was dropped (they share one lock, so a late collection drops all
# of them at once) and no earlier slate of the week has a row -- the Thursday game -- the week
# checks found no gold row, an empty prediction file and "no rows", and the run ended FAILED
# with a stale-artifact message sending the operator to a rebuild. The honest outcome is a
# recorded skip of the whole slate, FINISHED_WITH_SKIPS. These variants judge the slate.


def _slate_in_scope(slate: DailySlate) -> frozenset[str]:
    """The slate games this run has not dropped."""
    from pipeline import live_skip

    return slate.game_ids - live_skip.excluded_games()


def verify_slate_gold(slate: DailySlate) -> None:
    """Every slate game still in scope has a row in every gold matrix. All dropped: passes.

    Raises:
        RuntimeError: a gold matrix is missing.
        pipeline.steps.StaleDataArtifactError: a matrix lacks an in-scope slate game, named.
    """
    import pandas as pd

    from pipeline.steps import (
        _GOLD_FEATURE_TABLES,
        ARTIFACT_BOUNDARY_GOLD,
        StaleDataArtifactError,
        _verify_artifacts_at_boundary,
    )

    in_scope = _slate_in_scope(slate)
    if not in_scope:
        logger.warning("Every slate game was dropped; no gold row is required")
        return
    _verify_artifacts_at_boundary(ARTIFACT_BOUNDARY_GOLD, slate.season, slate.week)
    for table in _GOLD_FEATURE_TABLES:
        ids = pd.read_parquet(f"data/gold/{table}.parquet", columns=["game_id"])
        missing = sorted(in_scope - set(ids["game_id"].astype(str)))
        if missing:
            msg = (
                f"gold {table} carries no row for slate game(s) {missing}; they cannot be "
                "scored. Rebuild gold before predicting."
            )
            raise StaleDataArtifactError(msg)


def verify_slate_predictions(slate: DailySlate) -> None:
    """The week file holds a row made at THIS slate's lock for every in-scope game.

    And no row for a game this run dropped. All dropped: only the second holds.

    Raises:
        RuntimeError: the file is missing, or a dropped game has a row.
        pipeline.steps.StaleDataArtifactError: an in-scope slate game has no row for this lock.
    """
    import pandas as pd

    from pipeline import live_skip
    from pipeline.steps import StaleDataArtifactError

    path = _week_file(slate, "predictions")
    if not path.exists():
        raise RuntimeError(f"Missing data artifacts: {path.as_posix()}")
    frame = pd.read_csv(path)
    ids = set(frame["game_id"].astype(str)) if "game_id" in frame.columns else set()
    leaked = sorted(live_skip.excluded_games() & ids)
    if leaked:
        msg = (
            f"{path.as_posix()} carries prediction rows for game(s) this run DROPPED under "
            f"the live-skip rule: {leaked}. A dropped game must have no prediction."
        )
        raise RuntimeError(msg)
    in_scope = _slate_in_scope(slate)
    if not in_scope:
        return
    made_for_slate: set[str] = set()
    if not frame.empty and "information_cutoff_utc" in frame.columns:
        cutoff = pd.to_datetime(frame["information_cutoff_utc"], utc=True)
        made_for_slate = set(
            frame.loc[cutoff == pd.Timestamp(slate.lock), "game_id"].astype(str)
        )
    missing = sorted(in_scope - made_for_slate)
    if missing:
        msg = (
            f"{path.as_posix()} has no prediction made at this slate's lock for {missing}; "
            "publishing it would show an older row, or none, as tonight's."
        )
        raise StaleDataArtifactError(msg)


def export_slate_week(slate: DailySlate) -> None:
    """Export the slate week's prediction CSV to JSON beside it -- the SLATE's week."""
    import pandas as pd

    pred_csv = _week_file(slate, "predictions")
    if not pred_csv.exists():
        raise RuntimeError(f"Cannot export -- predictions CSV missing: {pred_csv}")
    pd.read_csv(pred_csv).to_json(
        pred_csv.with_suffix(".json"), orient="records", indent=2
    )


def validate_slate_predictions(slate: DailySlate) -> None:
    """The slate's in-scope rows carry every published prediction and a WP in [0, 1].

    Raises:
        RuntimeError: naming the defect.
    """
    import pandas as pd

    path = _week_file(slate, "predictions")
    if not path.exists():
        raise RuntimeError(f"Prediction validation failed -- missing file: {path}")
    frame = pd.read_csv(path)
    in_scope = _slate_in_scope(slate)
    if not in_scope:
        return
    required = ["game_id", "wp_prob", "ats_prediction", "ou_prediction"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Prediction validation failed -- missing columns: {missing}"
        )
    rows = frame.loc[frame["game_id"].astype(str).isin(in_scope)]
    if not rows["wp_prob"].between(0.0, 1.0).all():
        raise RuntimeError("Prediction validation failed -- wp_prob outside [0, 1]")


def recommend_slate(slate: DailySlate) -> None:
    """Select the slate's bets. ``decided_at_utc`` is the true computation instant (Task 2c).

    NEVER PUBLISHED AFTER THE LOCK (33.2 review C2 WR-08): the bet list is written only if the
    real clock is still at or before the slate's lock when the write happens
    (``publish_by``). A list that was ready too late writes nothing, and its games are refused
    by name through the same passed-lock refusal as a late decision.

    WHERE THE ROWS GO (Phase 34, Plan 34-15): the committed cutover switch
    (``forward_ledger.cutover.forward_rows_go_to_ledger``) decides. OFF -- until Plan 34-19 --
    the old writer ``generate_weekly_bet_list`` runs exactly as before. ON, the rows go to the
    forward ledger through ``forward_ledger.runner.record_forward_slate`` with the same decision
    instant, exclusions and deadline, and never to ``outputs/bet_list``. Both branches share the
    late-publish refusal below.
    """
    from backtest.weekly_bet_list import (
        PublishDeadlinePassedError,
        generate_weekly_bet_list,
    )
    from forward_ledger.cutover import forward_rows_go_to_ledger
    from pipeline import live_skip
    from pipeline.steps import _bet_list_output_dir, _week_schedule

    decided_at = datetime.now(UTC)
    excluded = live_skip.excluded_games()
    live_skip.refuse_passed_locks(
        slate.schedule, decided_at=decided_at, excluded_game_ids=excluded
    )
    week_ids = {str(g) for g in _week_schedule(slate.season, slate.week)["game_id"]}
    outside_slate = frozenset(week_ids - slate.game_ids)
    in_scope = sorted(slate.game_ids - excluded)
    try:
        if forward_rows_go_to_ledger():
            from forward_ledger import runner

            runner.record_forward_slate(
                slate,
                decided_at=decided_at,
                excluded_game_ids=live_skip.excluded_games() | outside_slate,
                publish_by=slate.lock if in_scope else None,
            )
        else:
            generate_weekly_bet_list(
                season=slate.season,
                week=slate.week,
                output_dir=_bet_list_output_dir(),
                now=decided_at,
                excluded_game_ids=live_skip.excluded_games() | outside_slate,
                publish_by=slate.lock if in_scope else None,
            )
    except PublishDeadlinePassedError as late:
        published_at = datetime.now(UTC).isoformat()
        raise live_skip.GamesLockPassedError(
            f"{late} Refusing the bets of {in_scope}.",
            {
                "source": live_skip.DECISION_INSTANT_SOURCE,
                "game_ids": in_scope,
                "information_times": [published_at] * len(in_scope),
                "locks": [slate.lock.isoformat()] * len(in_scope),
                "violation_type": "lock_passed",
            },
        ) from late


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
            lambda: capture_slate_odds(slate),
            collect,
            # NOT critical: an odds failure never stops predictions (see capture_slate_odds).
            critical=False,
            retryable=True,
            description="Capture the slate's odds (a failure leaves the market side blank)",
        ),
        step(
            "close_collection",
            lambda: close_collection(slate),
            collect,
            description="Stamp captured_at_utc; refuse a capture after the lock",
        ),
        # -- BUILD -----------------------------------------------------------------------
        # A slate game whose forecast failed WITH a recorded reason is built with its weather
        # unknown (step 27b, D33.2-05), so it is not expected here; a game missing its row
        # with NO recorded reason (an ingest that silently wrote nothing) still fails. A game
        # this run already DROPPED (close_collection refused it: collection ended after its
        # lock, and the weather ingest leaves a lock-passed game out) is not expected either,
        # so a late collection finishes with skips instead of failing here (review WR-10).
        step(
            "data_qa",
            lambda: step_data_qa(
                _slate_in_scope(slate) - frozenset(slate.weather_failures)
            ),
            build,
            description="Data quality validation (weather expected for the slate)",
        ),
        step(
            "build_elo",
            lambda: persist_current_season_elo(slate.game_ids),
            build,
            description="Update Elo, with provisional rows for the slate's week",
        ),
        step("build_team_form", step_build_team_form, build, description="Team form"),
        # NO build_contextual step (33.2 review B WR-14). The Friday registry's step runs the
        # DEPRECATED, lock-unfenced ``build_contextual_features`` to write silver
        # ``contextual_features``, which no production code reads: the gold build computes
        # contextual features itself, fenced per game (``ContextualFeaturesCalculator
        # .build_features``). As a critical step it could only abort the night's predictions.
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
            # The week's later games have Elo rows for ranking (review WR-08) but are not
            # built tonight: an unplayed game locking after the slate stays out of gold.
            lambda: build_and_save_gold(
                slate.game_ids, unplayed_through_lock=slate.lock
            ),
            build,
            description="Full-history gold build, saved; a history refusal keeps gold",
        ),
        step(
            "validate_features",
            step_validate_features,
            build,
            description="Leakage scan of gold",
        ),
        step(
            "verify_gold_currency",
            lambda: verify_slate_gold(slate),
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
            lambda: verify_slate_predictions(slate),
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
            lambda: export_slate_week(slate),
            build,
            description="Export predictions JSON",
        ),
        step(
            "validate_predictions",
            lambda: validate_slate_predictions(slate),
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
