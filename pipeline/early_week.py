"""The Tuesday early-week run: decide the coming NFL week, or skip it by name.

WHAT IT DOES. Called once from the daily run, right after the day's board capture. Each step below
either moves on, or -- for the first five -- ends the run with a named skip and no write:

1. which week is due today (``backtest.early_board.early_week_due``); none is ``not_due``;
2. the ledger already holds an early row for that week: ``already_decided``;
3. a game of a lower week has no score yet: ``earlier_week_unscored``;
4. no board was captured today: ``no_board``;
5. the models in force are not the v2 generation's: ``recipe_not_v2``;
6. silver is refreshed for the week: Elo with its provisional rows, team form, weather features;
7. a played game still resting on a provisional Elo row stops the run, because the gold it is about
   to build would train on a placeholder;
8. the full v2 gold is built through the week's latest lock and written under
   ``outputs/early_gold/`` -- never under ``data/``, and the board never enters silver;
9. a game whose lock has passed is left out by name; every other game is timed at the board;
10. each kept game's inputs are rebuilt as known at the board's capture;
11. the v2 models score them;
12. the week is decided (``backtest.early_decide``) and written to the ledger under the arm
    ``early``, with this record file's hash in the decision snapshot.

WHAT IS NOT A SKIP. A week the board matched no game of still goes through: every row is declined as
``missing_snapshot``. So does a game the early inputs left out, such as a neutral site: it has no
prediction and is declined as ``missing_prediction``. Any exception from any step is left to
propagate: the caller names the failure and the night's other steps go on.

ONE CLOCK READING. The instant is read once, after the gold build, and it is both the test for a
passed lock and the decision instant, as ``pipeline.daily_steps.recommend_slate`` reads its own
before it scores. A kept game's lock is therefore always after the decision instant by construction;
the ledger still re-checks the real clock against ``publish_by`` immediately before it writes.

THE REHEARSAL runs steps 8-12 with no silver refresh and no ledger write, against the latest board
of any day, and prints what it would have decided. ``python -m pipeline.early_week --rehearse-week N``.
Silver weather_features holds only the games whose slate night has run, so the rehearsal computes the
week's weather as step 6 would and hands it to the gold build in memory; nothing reaches silver. Elo
is not computed aside: a week is rehearsable once a slate run of that week has written its
provisional Elo rows (before that, its games have no Elo row and no prediction).

The stores are relative to the repository root, the daily run's working directory, and are
module-level names so a rehearsal can run from a worktree through temporary junctions.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import utils.game_lock as lock_rule
from api.cache import BET_STATUS_LIVE
from backtest.early_arm_2026 import (
    EARLY_FROM,
    EARLY_RECIPE,
    EARLY_TARGETS,
    KEY_NUMBER_TABLES_PATH,
    RECORD_PATH,
)
from backtest.early_board import (
    EarlyBoard,
    board_odds,
    early_week_due,
    latest_board,
    quotes_for,
    read_board_for_run,
)
from backtest.early_decide import EarlyDecision, decide_early_week
from backtest.study_1a import refuse_data_dir
from backtest.study_1b_inputs import EarlyInputs, build_early_inputs
from backtest.weekly_bet_list import DecisionBundle
from data.schemas import LIVE_ODDS_CAPTURE_KEY
from data.storage import load_dataframe
from data.upstream_pin import load_schedules
from features.elo_features import assert_no_provisional_training_rows
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.runner import RecordOutcome, record_early_week
from forward_ledger.schema import ARM_EARLY
from forward_ledger.store import LEDGER_DIR, read_entries
from models.artifacts import (
    ResolvedArtifacts,
    load_model_artifact,
    resolve_production_artifacts,
)
from models.key_numbers import load_key_number_tables
from models.v2_trainer import predict_v2
from pipeline import live_skip
from pipeline.daily_steps import (
    build_weather_features_through,
    weather_features_through,
)
from pipeline.steps import persist_current_season_elo, step_build_team_form
from refit.generations import in_force_recipe
from scripts.build_gold_v2 import TARGETS, FeatureMatrixBuilderV2Live
from scripts.capture_early_lines import EARLY_LINES_PATH
from scripts.ingest_odds import OddsDataIngester

__all__ = [
    "OUTCOME_DECIDED",
    "OUTCOME_NOT_DUE",
    "OUTCOME_SKIPPED",
    "EarlyWeekResult",
    "GoldBuilderWithWeather",
    "main",
    "rehearse_early_week",
    "run_early_week",
    "write_week_gold",
]

OUTCOME_NOT_DUE = "not_due"
OUTCOME_SKIPPED = "skipped"
OUTCOME_DECIDED = "decided"

REASON_ALREADY_DECIDED = "already_decided"
REASON_EARLIER_WEEK_UNSCORED = "earlier_week_unscored"
REASON_NO_BOARD = "no_board"
REASON_RECIPE_NOT_V2 = "recipe_not_v2"

# The stores, relative to the repository root (the daily run's working directory).
ARTIFACTS_DIR: Path = Path("artifacts")
EARLY_LINES_FILE: Path = EARLY_LINES_PATH
EARLY_GOLD_ROOT: Path = Path("outputs/early_gold")
REHEARSAL_GOLD_ROOT: Path = Path("outputs/early_gold_rehearsal")

# The schedule columns a decision snapshot keeps for each decided game.
_SCHEDULE_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "kickoff_et",
]


@dataclass(frozen=True)
class EarlyWeekResult:
    """What one run did.

    Attributes:
        outcome: ``not_due``, ``skipped`` or ``decided``.
        reason: Why the week was skipped; None otherwise.
        season, week: The week the run looked at; None when none was due.
        games: The games decided (a game past its lock is not one).
        bets: Decided rows that are live bets.
        declined: Decided rows that are not.
        appended_rows: The canonical rows this run wrote to the ledger.
    """

    outcome: str
    reason: str | None
    season: int | None
    week: int | None
    games: int
    bets: int
    declined: int
    appended_rows: tuple[dict[str, Any], ...]


# ---------------------------------------------------------------------------
# The heavy steps, each a module seam so a test can replace it
# ---------------------------------------------------------------------------


def _refresh_silver(season: int, week_ids: frozenset[str]) -> None:
    """Bring silver up to date for the coming week: Elo, team form, weather features."""
    persist_current_season_elo(week_ids)
    step_build_team_form()
    build_weather_features_through(season, week_ids, {})


class GoldBuilderWithWeather(FeatureMatrixBuilderV2Live):
    """The v2 gold builder, reading the weather features it is handed instead of silver's."""

    def __init__(self, weather_features: pd.DataFrame) -> None:
        super().__init__()
        self._weather_features = weather_features

    def _read_silver_weather_features(self) -> pd.DataFrame:
        return self._weather_features.copy()


def _build_gold(
    unplayed_through_lock: datetime,
    *,
    weather_features: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """The v2 gold matrices through the week's latest lock. Saves nothing.

    Args:
        unplayed_through_lock: The week's latest lock.
        weather_features: The weather to build with; None reads silver (the real run, whose
            refresh has just written it).
    """
    builder = (
        FeatureMatrixBuilderV2Live()
        if weather_features is None
        else GoldBuilderWithWeather(weather_features)
    )
    return builder.generate_feature_matrices(
        excluded_game_ids=live_skip.excluded_games(),
        unplayed_through_lock=unplayed_through_lock,
    )


def _rehearsal_weather(season: int, week_ids: frozenset[str]) -> pd.DataFrame:
    """The week's weather features as step 6 would write them, computed and not saved."""
    features, _unknown = weather_features_through(season, week_ids, {})
    return features


def _build_inputs(
    gold_dir: Path, games: pd.DataFrame, season: int, early_times: pd.Series
) -> EarlyInputs:
    """Each timed game's v2 inputs as known at its early capture."""
    return build_early_inputs(
        gold_dir, games, load_schedules([season - 1, season]), early_times
    )


def _score(
    rows: Mapping[str, pd.DataFrame], resolved: ResolvedArtifacts
) -> dict[str, dict[str, float]]:
    """The v2 models' number for every input row, by target and then game id."""
    predictions: dict[str, dict[str, float]] = {}
    for target in EARLY_TARGETS:
        model = load_model_artifact(
            target,
            version=resolved.model_id_for(target),
            artifacts_dir=ARTIFACTS_DIR,
        )["model"]
        frame = rows[target]
        predictions[target] = dict(
            zip(
                frame["game_id"].astype(str),
                map(float, predict_v2(target, model, frame)),
                strict=True,
            )
        )
    return predictions


def _now() -> datetime:
    return datetime.now(UTC)


REFRESH_SILVER: Callable[[int, frozenset[str]], None] = _refresh_silver
BUILD_GOLD: Callable[..., dict[str, pd.DataFrame]] = _build_gold
REHEARSAL_WEATHER: Callable[[int, frozenset[str]], pd.DataFrame] = _rehearsal_weather
BUILD_INPUTS: Callable[[Path, pd.DataFrame, int, pd.Series], EarlyInputs] = (
    _build_inputs
)
SCORE: Callable[
    [Mapping[str, pd.DataFrame], ResolvedArtifacts], dict[str, dict[str, float]]
] = _score
RECORD: Callable[..., RecordOutcome] = record_early_week
NOW: Callable[[], datetime] = _now


def odds_ingester() -> OddsDataIngester:
    """The ingester whose transform and validation read the board. Replaced in tests."""
    return OddsDataIngester()


# ---------------------------------------------------------------------------
# The checks before anything is built
# ---------------------------------------------------------------------------


def _week_already_decided(season: int, week: int) -> bool:
    """Whether the ledger holds any early row for the week, live or suppressed."""
    return any(
        entry.kind == ENTRY_KIND_ROW
        and entry.immutable.get("arm") == ARM_EARLY
        and int(entry.immutable["season"]) == season
        and int(entry.immutable["week"]) == week
        for entry in read_entries(LEDGER_DIR)
    )


def _has_unscored_game(lower_weeks: pd.DataFrame) -> bool:
    """Whether a game of a lower week lacks either side's score."""
    return bool(
        lower_weeks["home_score"].isna().any() or lower_weeks["away_score"].isna().any()
    )


# ---------------------------------------------------------------------------
# Building the week's gold and the decision (shared with the rehearsal)
# ---------------------------------------------------------------------------


def write_week_gold(
    matrices: Mapping[str, pd.DataFrame], week_games: pd.DataFrame, out_dir: Path
) -> Path:
    """Write the three gold matrices to *out_dir* and return its resolved path.

    Nothing knows the wind on Tuesday, so the week's unplayed games come out of the build with no
    wind. A fixed roof is a fact of the stadium, known on Tuesday as on Saturday, so an ``indoor``
    game of the week keeps a wind of 0.0, the value every indoor game carries in the training gold.

    Raises:
        ValueError: *out_dir* is under ``data/`` or the live lake.
    """
    target_dir = refuse_data_dir(
        out_dir, what="the early-week gold", default_dir=EARLY_GOLD_ROOT
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    indoor = week_games.loc[week_games["venue_roof"] == "indoor", "game_id"].astype(str)
    for target in TARGETS:
        matrix = matrices[target].copy()
        matrix.loc[matrix["game_id"].astype(str).isin(indoor), "wind_mph"] = 0.0
        matrix.to_parquet(target_dir / f"features_{target}.parquet", index=False)
    return target_dir


def _game_locks(games: pd.DataFrame) -> dict[str, datetime]:
    """Each game's lock, by game id, from the one rule."""
    return {
        str(game_id): lock.to_pydatetime()
        for game_id, lock in lock_rule.lock_frame(games).items()
    }


def _open_games(
    week_games: pd.DataFrame, locks: Mapping[str, datetime], clock: datetime
) -> pd.DataFrame:
    """The week's games whose lock is still ahead; each game past its lock is named and left out.

    A lock exactly at the clock counts as passed.
    """
    ids = week_games["game_id"].astype(str)
    passed = [game_id for game_id in ids if locks[game_id] <= clock]
    for game_id in passed:
        print(  # noqa: T201 -- the daily run's console contract names every game left out
            f"EARLY_LOCK_PASSED {game_id}"
        )
    return week_games.loc[~ids.isin(passed)]


def _board_rows(board: EarlyBoard, games: pd.DataFrame) -> pd.DataFrame:
    """The board's validated odds rows for *games*, with the key columns even when none matched.

    The reproduction key reads each row's key columns. A board matching no game comes back from
    ``board_odds`` with no columns at all, so it is handed on as empty rows of the right shape.
    """
    odds = board_odds(board, games, odds_ingester())
    if odds.empty:
        return pd.DataFrame(columns=pd.Index(LIVE_ODDS_CAPTURE_KEY))
    return odds


@dataclass(frozen=True)
class _WeekDecision:
    """Steps 8-12's results, up to the point where the run and the rehearsal part.

    Attributes:
        games: The games decided, in their silver columns.
        locks: Each decided game's lock.
        odds: The board's rows for those games.
        gold_dir: Where the week's gold was written.
        inputs: The games' early inputs.
        resolved: The artifact ids the games were scored with, read once.
        decision: The decided rows and how each was priced.
        decided_at: The one instant the decision was made at.
    """

    games: pd.DataFrame
    locks: Mapping[str, datetime]
    odds: pd.DataFrame
    gold_dir: Path
    inputs: EarlyInputs
    resolved: ResolvedArtifacts
    decision: EarlyDecision
    decided_at: datetime


def _decide_week(
    games: pd.DataFrame,
    week_games: pd.DataFrame,
    board: EarlyBoard,
    season: int,
    gold_out_dir: Path,
    weather_features: pd.DataFrame | None = None,
) -> _WeekDecision:
    """Build the week's gold, leave out passed locks, score the rest and decide. Writes no ledger.

    *weather_features* is the rehearsal's in-memory weather; None builds from silver.
    """
    week_locks = _game_locks(week_games)
    gold_dir = write_week_gold(
        BUILD_GOLD(max(week_locks.values()), weather_features=weather_features),
        week_games,
        gold_out_dir,
    )

    decided_at = NOW()
    kept = _open_games(week_games, week_locks, decided_at)
    locks = {str(game_id): week_locks[str(game_id)] for game_id in kept["game_id"]}
    early_times = pd.Series(
        board.captured_at, index=pd.Index(list(locks), name="game_id")
    )

    inputs = BUILD_INPUTS(gold_dir, games, season, early_times)
    resolved = resolve_production_artifacts(ARTIFACTS_DIR)
    predictions = SCORE(inputs.rows, resolved)

    odds = _board_rows(board, kept)
    decision = decide_early_week(
        kept,
        predictions=predictions,
        quotes={target: quotes_for(odds, target) for target in EARLY_TARGETS},
        tables=load_key_number_tables(),
        locks=locks,
        captured_at=board.captured_at,
        decided_at=decided_at,
    )
    return _WeekDecision(
        games=kept,
        locks=locks,
        odds=odds,
        gold_dir=gold_dir,
        inputs=inputs,
        resolved=resolved,
        decision=decision,
        decided_at=decided_at,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _report(result: EarlyWeekResult) -> EarlyWeekResult:
    """Print the run's one-line outcome (the daily run's console contract) and return it."""
    reason = f" reason={result.reason}" if result.reason else ""
    print(  # noqa: T201 -- the daily run's console contract names this step's outcome
        f"EARLY_BETS= {result.outcome}{reason} season={result.season} week={result.week} "
        f"games={result.games} bets={result.bets} declined={result.declined}"
    )
    return result


def _skipped(reason: str, season: int, week: int) -> EarlyWeekResult:
    return _report(EarlyWeekResult(OUTCOME_SKIPPED, reason, season, week, 0, 0, 0, ()))


def _week_rows(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return games.loc[(games["season"] == season) & (games["week"] == week)]


def _record_sha256(path: Path) -> str:
    """The sha256 of a text file as git holds it, with CRLF normalized to LF.

    core.autocrlf checks the record out with CRLF while git stores LF, so a hash of the bytes on
    disk would make a later comparison with git show an edit nobody made.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_early_week(run_date_et: date, *, board_captured: bool) -> EarlyWeekResult:
    """Decide the NFL week due on *run_date_et*, or skip it by name (see the module docstring).

    Args:
        run_date_et: The ET calendar date of the run.
        board_captured: Whether today's board capture succeeded.

    Returns:
        What the run did. A skip writes nothing.

    Raises:
        Whatever any step refuses with, by name; the caller reports the failure.
    """
    games = load_dataframe("games", layer="silver")
    due = early_week_due(games, run_date_et)
    if due is None:
        return _report(EarlyWeekResult(OUTCOME_NOT_DUE, None, None, None, 0, 0, 0, ()))
    season, week = due
    week_games = _week_rows(games, season, week)
    lower_weeks = games.loc[(games["season"] == season) & (games["week"] < week)]

    if _week_already_decided(season, week):
        return _skipped(REASON_ALREADY_DECIDED, season, week)
    if _has_unscored_game(lower_weeks):
        return _skipped(REASON_EARLIER_WEEK_UNSCORED, season, week)
    board = (
        read_board_for_run(EARLY_LINES_FILE, run_date_et) if board_captured else None
    )
    if board is None:
        return _skipped(REASON_NO_BOARD, season, week)
    if in_force_recipe(ARTIFACTS_DIR, LEDGER_DIR) != EARLY_RECIPE:
        return _skipped(REASON_RECIPE_NOT_V2, season, week)

    REFRESH_SILVER(season, frozenset(week_games["game_id"].astype(str)))
    # After the refresh, which is what replaces a played game's provisional row.
    assert_no_provisional_training_rows(
        lower_weeks.loc[:, ["game_id"]], "early_week:played_weeks"
    )

    week_decision = _decide_week(
        games, week_games, board, season, EARLY_GOLD_ROOT / f"{season}_w{week}"
    )
    decision = week_decision.decision
    bundle = DecisionBundle(
        frame=decision.frame,
        candidates=decision.candidates,
        schedule=week_decision.games.loc[:, _SCHEDULE_COLUMNS],
        gold_inputs=week_decision.inputs.rows,
        fits={},
        resolved=week_decision.resolved,
        chain_fit_path=Path(KEY_NUMBER_TABLES_PATH),
        run_instant=week_decision.decided_at,
    )
    outcome = RECORD(
        bundle,
        odds=week_decision.odds,
        gold_dir=week_decision.gold_dir,
        decided_at=week_decision.decided_at,
        publish_by=min(week_decision.locks.values()),
        meta_extra={"early_arm_record_sha256": _record_sha256(Path(RECORD_PATH))},
        ledger_dir=LEDGER_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        repo_dir=Path(),
    )
    bets = int((decision.frame["status"] == BET_STATUS_LIVE).sum())
    return _report(
        EarlyWeekResult(
            OUTCOME_DECIDED,
            None,
            season,
            week,
            games=len(week_decision.games),
            bets=bets,
            declined=len(decision.frame) - bets,
            appended_rows=tuple(outcome.appended_rows),
        )
    )


# ---------------------------------------------------------------------------
# The rehearsal
# ---------------------------------------------------------------------------


def _shown(value: Any, decimals: int = 2) -> str:
    """A rehearsal cell: text as it stands, a number rounded, nothing as a dash."""
    if pd.isna(value):
        return "-"
    return value if isinstance(value, str) else f"{float(value):.{decimals}f}"


def _print_rehearsal(decision: EarlyDecision) -> None:
    """One line per game and target: how it was priced and what became of it."""
    statuses = {
        (str(row["game_id"]), str(row["target"])): (
            BET_STATUS_LIVE
            if row["status"] == BET_STATUS_LIVE
            else str(row["rejection_reason"])
        )
        for row in decision.frame.to_dict("records")
    }
    for row in decision.candidates.to_dict("records"):
        key = (str(row["game_id"]), str(row["target"]))
        print(  # noqa: T201 -- a rehearsal's whole output is this table
            f"{key[0]} {key[1]} consensus={_shown(row['consensus'])} "
            f"model={_shown(row['model_value'])} side={_shown(row['bet_side'])} "
            f"book={_shown(row['sportsbook'])} price={_shown(row['price'], decimals=0)} "
            f"ev={_shown(row['ev'], decimals=4)} status={statuses[key]}"
        )


def rehearse_early_week(week: int) -> None:
    """Build, score and decide *week* of the arm's season, and print it. No ledger write.

    Steps 8-12 of the run, against the latest board of any day: no silver refresh, and the gold
    goes to its own directory. The week's weather is computed as step 6 would write it and handed
    to the gold build in memory (see the module docstring).

    Raises:
        RuntimeError: no board has ever been recorded.
    """
    season = EARLY_FROM[0]
    board = latest_board(EARLY_LINES_FILE)
    if board is None:
        msg = f"no board is recorded in {EARLY_LINES_FILE.as_posix()}; nothing to rehearse"
        raise RuntimeError(msg)
    games = load_dataframe("games", layer="silver")
    week_games = _week_rows(games, season, week)
    weather = REHEARSAL_WEATHER(season, frozenset(week_games["game_id"].astype(str)))
    week_decision = _decide_week(
        games, week_games, board, season, REHEARSAL_GOLD_ROOT, weather_features=weather
    )
    _print_rehearsal(week_decision.decision)


def main(argv: list[str] | None = None) -> int:
    """``python -m pipeline.early_week --rehearse-week N``."""
    parser = argparse.ArgumentParser(
        description="Rehearse one early week: no silver refresh and no ledger write."
    )
    parser.add_argument(
        "--rehearse-week",
        type=int,
        required=True,
        metavar="N",
        help=f"the week of the {EARLY_FROM[0]} season to rehearse",
    )
    args = parser.parse_args(argv)
    rehearse_early_week(args.rehearse_week)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
