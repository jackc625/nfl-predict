"""The blend's tuning corpus: lines we OWNED before each game's lock, and nothing else.

WHAT CHANGED, AND WHY (Plan 33.2-24, D33.2-03 / D33.2-10)
---------------------------------------------------------
This module used to fetch nflverse schedules live and tune the blend on their 2010-2017
CLOSING lines. A closing line did not exist at a game's lock, so a blend tuned on it learned
how far to trust the market from a number nobody could have had when the prediction was
made. That corpus is retired, and so is the live fetch: nothing here reaches nflverse.

The corpus is now the owned ``odds_timeline`` -- the Phase-29 purchase, spreads and totals
as consensus medians over 2020-2024, each row carrying the real time its snapshot was taken.
For every scheduled game the loader takes the LATEST snapshot timed AT OR BEFORE that game's
own lock (``utils.game_lock``, D33.2-01).

THE THREE PROPERTIES THE LOADER CARRIES
---------------------------------------
1. AT-LOCK IS ADMISSIBLE. The comparison is ``snapshot_ts <= lock`` -- the same ``<=`` as
   ``utils.game_lock.is_admissible``. One second after the lock is not admissible.
2. LATEST-AT-OR-BEFORE WINS, DETERMINISTICALLY, EXACTLY ONE ROW PER GAME. Rows are ordered by
   ``(game_id, snapshot_ts, created_at)`` and the LAST admissible row per game is taken; an
   identical ``snapshot_ts`` is therefore broken by the later ``created_at``. The writer
   already dedupes ``odds_timeline`` on ``(game_id, snapshot_ts)``
   (``data.storage.upsert_silver_composite``), but that is an invariant owned by a different
   module, so this one does not lean on it: :func:`assert_one_row_per_game` refuses a frame
   that still carries two rows for a game, because a second row would silently double that
   game's weight in the fit.
3. NO FILL, EVER. A game with no pre-lock snapshot is EXCLUDED, counted and RETURNED. There
   is no fallback to the stored closing-line table, no fallback to a later snapshot and no
   neutral or league-average fill. A filled row is a row the tuner treats as evidence about
   a line that did not exist at the lock -- which is precisely the leak this corpus replaces.

The excluded games travel WITH the tuning frame (:class:`PrelockTuningCorpus`), because an
exclusion that is only logged is an exclusion nobody can audit; the readout publishes them.

THE SIGN, STATED ONCE
---------------------
``odds_timeline`` stores the spread on the OPPOSITE sign from the serving convention
(D33.2-23). The frame this module returns carries ``market_spread`` on the HOME-MARGIN
scale -- POSITIVE when the home team is favoured, the scale the ATS model predicts on and
the scale ``models.blending.home_fav_margin_from_prelock_spread`` documents -- converted
through ``models.market_probability.timeline_spread_to_home_fav_margin``, the one function
that knows the timeline's convention.

A BETTING LINE IS STILL NOT A MODEL INPUT. This corpus is the MARKET half of the blend,
applied after the models have predicted (D33.2-03). Nothing here reaches a feature matrix.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from models.blending import (
    BLEND_TUNING_COLUMNS,
    MarketBlender,
    MarketProbabilityUnavailable,
)
from models.market_probability import (
    OWNED_LINE_SEASONS,
    load_market_probability_artifact,
    oof_market_probability,
    timeline_spread_to_home_fav_margin,
)
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The owned line history's silver table. Its seasons are a COVERAGE FACT about the
#: purchase (``models.market_probability.OWNED_LINE_SEASONS``), not a window rule.
OWNED_TIMELINE_TABLE: str = "odds_timeline"

#: Why a scheduled game is absent from the tuning frame. It is the ONE reason this loader
#: can give: the game has no line timed at or before its own lock.
EXCLUSION_NO_PRELOCK_LINE: str = "no_prelock_line"

#: Why a game WITH a pre-lock line is absent from the WP fit ALONE: its season has no
#: prior-fold converter slope, so its market probability cannot be computed out of fold. It
#: is never converted with the serving slope instead (see
#: :func:`attach_oof_market_probability`). The ATS and O/U fits keep it: they blend the line
#: itself and never touch the converter.
EXCLUSION_NO_PRIOR_FOLD_CONVERTER: str = "no_prior_fold_converter"

#: Every exclusion class, in the order a readout publishes them.
EXCLUSION_REASONS: tuple[str, ...] = (
    EXCLUSION_NO_PRELOCK_LINE,
    EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
)

#: The columns a target's walk-forward predictions arrive in. ``actual`` is each trainer's
#: OWN target column (``home_win`` / ``home_margin`` / ``total_points``), so the blend is
#: scored in the model's own metric on the model's own labels.
_PREDICTION_COLUMNS: tuple[str, ...] = ("game_id", "season", "prediction", "actual")

#: The two ways a game ends up with no pre-lock line, recorded per row so the readout's
#: breakdown is read off the data rather than re-derived: the timeline has no row for the
#: game at all, or every row it has is timed AFTER the game's lock.
DETAIL_ABSENT_FROM_TIMELINE: str = "absent_from_timeline"
DETAIL_POST_LOCK_ONLY: str = "post_lock_snapshots_only"

#: The columns of the tuning frame, one row per game.
TUNING_FRAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "snapshot_ts",
    "lock",
    "created_at",
    "market_spread",
    "market_total",
)

#: The columns of the excluded set.
EXCLUDED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "reason",
    "detail",
)

#: The columns a timeline row must carry to be read at all.
_TIMELINE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "snapshot_ts",
    "spread",
    "total",
    "created_at",
)

#: The columns a schedule must carry to have a lock.
_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "kickoff_et",
)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TuningCorpusError(Exception):
    """The tuning corpus cannot be built honestly, so it is not built at all.

    Inherits ``Exception`` rather than ``ValueError`` / ``KeyError``: several call sites in
    this repository catch those and degrade quietly, and a corpus refusal degraded into
    "tune on whatever loaded" would be the silent failure this module exists to prevent.
    """


class DuplicateTuningRowError(TuningCorpusError):
    """A game carries more than one row after selection, so it would be over-weighted."""


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrelockTuningCorpus:
    """The owned pre-lock lines, with the games that have none returned beside them.

    Attributes:
        frame: One row per game with a line at or before its lock
            (:data:`TUNING_FRAME_COLUMNS`). ``market_spread`` is on the home-margin scale.
        excluded: One row per scheduled game with NO pre-lock line
            (:data:`EXCLUDED_COLUMNS`), each carrying its reason and detail.
        unjoined_timeline_ids: Timeline game ids that name no scheduled game. They are not
            exclusions -- no scheduled game is missing because of them -- but they are
            reported so the readout can say what they are rather than drop them silently.
    """

    frame: pd.DataFrame
    excluded: pd.DataFrame
    unjoined_timeline_ids: tuple[str, ...]


def assert_one_row_per_game(frame: pd.DataFrame) -> None:
    """Refuse *frame* if any ``game_id`` appears more than once.

    Called on the selected frame before it is returned, so a duplicate can never reach the
    tuner. The writer's own dedupe is not relied on: it is another module's invariant.

    Raises:
        DuplicateTuningRowError: naming every repeated game.
    """
    repeated = sorted(
        {
            str(game_id)
            for game_id in frame.loc[frame["game_id"].duplicated(), "game_id"]
        }
    )
    if repeated:
        msg = (
            f"{len(repeated)} game(s) carry more than one tuning row, e.g. {repeated[:10]}. "
            "A second row for a game doubles that game's weight in the blend fit, silently; "
            "refusing rather than letting it reach the tuner."
        )
        raise DuplicateTuningRowError(msg)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"the {what} frame is missing {missing}; it must carry {list(columns)}"
        raise TuningCorpusError(msg)


def _require_aware(series: pd.Series, name: str) -> pd.Series:
    """*series* as tz-aware UTC instants. A NAIVE instant is refused, never relabelled.

    Relabelling a naive value as UTC is the convert-never-relabel defect D33.2-01 forbids:
    a line whose capture time is ambiguous cannot be shown to be pre-lock.
    """
    if series.empty:
        # No instant exists to be relabelled; an empty column only needs the dtype.
        return pd.to_datetime(series, utc=True)
    parsed = pd.to_datetime(series)
    if getattr(parsed.dt, "tz", None) is None:
        msg = (
            f"{name} is timezone-naive; a naive capture time cannot be compared with a "
            "game's lock without guessing its zone, and guessing is what D33.2-01 forbids. "
            "Refusing rather than relabelling it as UTC."
        )
        raise TuningCorpusError(msg)
    return parsed.dt.tz_convert("UTC")


def select_prelock_lines(
    timeline: pd.DataFrame,
    schedule: pd.DataFrame,
) -> PrelockTuningCorpus:
    """Select, for each scheduled game, the latest owned line at or before its lock.

    Pure: reads nothing from disk, so every property is testable on a hand-built frame.

    Args:
        timeline: ``odds_timeline``-shaped rows. ``spread`` is on the TIMELINE's own
            (inverted) sign; ``snapshot_ts`` and ``created_at`` must be tz-aware.
        schedule: The scheduled games the corpus is drawn for (silver ``games`` shape),
            carrying ``game_id``, ``season``, ``week``, ``game_type`` and ``kickoff_et``.

    Returns:
        The :class:`PrelockTuningCorpus`.

    Raises:
        TuningCorpusError: when a required column is absent or a capture time is naive.
        DuplicateTuningRowError: when a game survives selection with two rows.
        utils.game_lock.MissingKickoffError: when a scheduled game has no kickoff.
    """
    _require(timeline, _TIMELINE_COLUMNS, "timeline")
    _require(schedule, _SCHEDULE_COLUMNS, "schedule")

    schedule = schedule.loc[:, list(_SCHEDULE_COLUMNS)].copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    # THE lock, one per game, from THE rule -- never a second derivation here. Imported
    # HERE rather than at module level: a module-level binding of the rule's function
    # hides this call from the one-lock-rule identity delegate
    # (tests/unit/test_one_lock_rule_source_scan.py, control 5).
    from utils.game_lock import lock_frame

    locks = lock_frame(schedule)
    schedule["lock"] = pd.to_datetime(schedule["game_id"].map(locks), utc=True)

    lines = timeline.loc[:, list(_TIMELINE_COLUMNS)].copy()
    lines["game_id"] = lines["game_id"].astype(str)
    lines["snapshot_ts"] = _require_aware(lines["snapshot_ts"], "snapshot_ts")
    lines["created_at"] = _require_aware(lines["created_at"], "created_at")
    # A row with no spread or no total is not a line; it cannot stand in for one.
    lines = lines.dropna(subset=["snapshot_ts", "spread", "total"])

    scheduled_ids = set(schedule["game_id"])
    unjoined = tuple(sorted(set(lines["game_id"]) - scheduled_ids))

    joined = lines.merge(schedule, on="game_id", how="inner")
    # AT-LOCK IS ADMISSIBLE: `<=`, the same operator as utils.game_lock.is_admissible.
    admissible = joined[joined["snapshot_ts"] <= joined["lock"]]

    # LATEST-AT-OR-BEFORE WINS: sorted by (game_id, snapshot_ts, created_at), last row per
    # game. A stable sort keeps any residual tie in input order, so the choice is a
    # function of the data rather than of the sort algorithm.
    ordered = admissible.sort_values(
        ["game_id", "snapshot_ts", "created_at"], kind="mergesort"
    )
    latest = ordered.groupby("game_id", sort=True).tail(1)

    frame = (
        latest.assign(
            # THE SIGN: through the one named flip, onto the home-margin scale.
            market_spread=timeline_spread_to_home_fav_margin(latest["spread"]),
            market_total=latest["total"].astype(float),
        )
        .astype({"season": int, "week": int, "game_type": str})
        .loc[:, list(TUNING_FRAME_COLUMNS)]
        .sort_values("game_id", ignore_index=True)
    )
    assert_one_row_per_game(frame)

    # NO FILL, EVER. Every scheduled game without a selected row is EXCLUDED and returned.
    # There is no fallback of any kind: not the closing-line table, not a later snapshot,
    # not a neutral value.
    tuned_ids = set(frame["game_id"])
    seen_in_timeline = set(joined["game_id"])
    missing = schedule[~schedule["game_id"].isin(tuned_ids)].copy()
    missing["reason"] = EXCLUSION_NO_PRELOCK_LINE
    missing["detail"] = [
        DETAIL_POST_LOCK_ONLY
        if game_id in seen_in_timeline
        else DETAIL_ABSENT_FROM_TIMELINE
        for game_id in missing["game_id"]
    ]
    excluded = (
        missing.loc[:, list(EXCLUDED_COLUMNS)]
        .astype({"season": int, "week": int, "game_type": str})
        .sort_values("game_id", ignore_index=True)
    )

    return PrelockTuningCorpus(
        frame=frame, excluded=excluded, unjoined_timeline_ids=unjoined
    )


def load_tuning_period_data(
    silver_dir: Path | str = Path("data/silver"),
) -> PrelockTuningCorpus:
    """Load the owned pre-lock tuning corpus from silver.

    Reads silver ``odds_timeline`` and the silver ``games`` schedule for the owned seasons
    (``models.market_probability.OWNED_LINE_SEASONS``) and hands both to
    :func:`select_prelock_lines`. Every scheduled game in those seasons is either in the
    returned frame or in its excluded set -- none is silently dropped.

    Args:
        silver_dir: The silver layer root.

    Returns:
        The :class:`PrelockTuningCorpus`.
    """
    root = Path(silver_dir)
    timeline = pd.read_parquet(root / f"{OWNED_TIMELINE_TABLE}.parquet")
    games = pd.read_parquet(root / "games.parquet")
    schedule = games[games["season"].isin(OWNED_LINE_SEASONS)]

    corpus = select_prelock_lines(timeline, schedule)
    logger.info(
        "Owned pre-lock tuning corpus loaded",
        n_games=len(corpus.frame),
        n_excluded=len(corpus.excluded),
        n_unjoined_timeline_ids=len(corpus.unjoined_timeline_ids),
        seasons=list(OWNED_LINE_SEASONS),
    )
    return corpus


# ---------------------------------------------------------------------------
# The tuning frames: corpus x walk-forward predictions x out-of-fold market
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlendTuningFrames:
    """One tuning frame per target, plus the rows the WP fit alone had to drop.

    Attributes:
        frames: ``{wp, ats, ou}`` -> one row per game, carrying ``game_id``, ``season``,
            ``week`` and that target's ``models.blending.BLEND_TUNING_COLUMNS``.
        excluded: The WP-only :data:`EXCLUSION_NO_PRIOR_FOLD_CONVERTER` rows
            (:data:`EXCLUDED_COLUMNS`).
    """

    frames: dict[str, pd.DataFrame]
    excluded: pd.DataFrame


def attach_oof_market_probability(
    frame: pd.DataFrame,
    walk_forward_slopes: dict[str, float] | dict[int, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The WP market side of every HISTORICAL row, out of fold, and the rows that have none.

    WHY OUT OF FOLD. The converter artifact carries two outputs because history and serving
    need different things (Plan 33.2-21): ``slope_beta``, fitted over EVERY owned season,
    which serving binds; and ``walk_forward_slopes``, per season ``S`` a slope fitted on
    seasons strictly before ``S``. A historical row converted with ``slope_beta`` would be
    priced by a slope fitted partly on that row's own outcome -- in-sample leakage that
    flatters the market side and biases the fitted weight, invisibly, because the number
    still looks like a probability. So every row here goes through ``oof_market_probability``
    with its own season's prior-only slope, and ``slope_beta`` converts nothing.

    A row whose season has no prior fold (the first owned season -- 2020 on today's corpus,
    for which ``oof_market_probability`` refuses by design) is returned in the excluded set
    under :data:`EXCLUSION_NO_PRIOR_FOLD_CONVERTER`. It is never filled with ``slope_beta``,
    a neighbouring season's slope or anything else.

    Args:
        frame: Corpus rows carrying ``season`` and ``market_spread`` (home-margin scale).
        walk_forward_slopes: The converter artifact's ``walk_forward_slopes``.

    Returns:
        ``(rows, excluded)``: the convertible rows with a ``market_prob_oof`` column, and the
        excluded rows (:data:`EXCLUDED_COLUMNS`).
    """
    covered = {int(season) for season in walk_forward_slopes}
    convertible = frame["season"].astype(int).isin(covered)

    rows = frame.loc[convertible].copy()
    rows["market_prob_oof"] = oof_market_probability(
        rows.assign(home_fav_margin=rows["market_spread"]), walk_forward_slopes
    )

    dropped = frame.loc[~convertible].copy()
    dropped["reason"] = EXCLUSION_NO_PRIOR_FOLD_CONVERTER
    dropped["detail"] = [
        f"season {int(season)} has no prior-fold converter slope"
        for season in dropped["season"]
    ]
    for column in ("week", "game_type"):
        if column not in dropped.columns:
            dropped[column] = pd.NA
    excluded = dropped.loc[:, list(EXCLUDED_COLUMNS)].reset_index(drop=True)
    return rows.reset_index(drop=True), excluded


def build_tuning_frames(
    corpus_frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    walk_forward_slopes: dict[str, float] | dict[int, float],
) -> BlendTuningFrames:
    """Join the owned pre-lock corpus to each model's walk-forward predictions.

    Every corpus game must carry a prediction for every target: a game silently dropped here
    would be an exclusion nobody counted. The market side per target:

    * WP -- ``market_prob_oof``, the out-of-fold converter probability
      (:func:`attach_oof_market_probability`); first-owned-season rows leave the WP fit ONLY;
    * ATS -- ``market_spread``, the pre-lock line itself on the home-margin scale;
    * O/U -- ``market_total``, the pre-lock total itself.

    ATS and O/U never pass through the converter, so they keep the FULL corpus -- the
    per-target difference in row counts is a decision, not a defect.

    Args:
        corpus_frame: :attr:`PrelockTuningCorpus.frame`.
        predictions: ``{wp, ats, ou}`` -> ``game_id``, ``season``, ``prediction``,
            ``actual``.
        walk_forward_slopes: The converter artifact's ``walk_forward_slopes``.

    Returns:
        The :class:`BlendTuningFrames`.

    Raises:
        TuningCorpusError: when a target's predictions miss a corpus game, repeat one, or
            disagree with the corpus about a game's season.
    """
    frames: dict[str, pd.DataFrame] = {}
    excluded = pd.DataFrame(columns=list(EXCLUDED_COLUMNS))

    for target, (model_col, market_col, outcome_col) in BLEND_TUNING_COLUMNS.items():
        if target not in predictions:
            msg = f"no walk-forward predictions were supplied for {target!r}"
            raise TuningCorpusError(msg)
        preds = predictions[target]
        _require(preds, _PREDICTION_COLUMNS, f"{target} predictions")
        preds = preds.loc[:, list(_PREDICTION_COLUMNS)].copy()
        preds["game_id"] = preds["game_id"].astype(str)
        assert_one_row_per_game(preds)

        missing = sorted(set(corpus_frame["game_id"]) - set(preds["game_id"]))
        if missing:
            msg = (
                f"{len(missing)} owned pre-lock game(s) have no {target} walk-forward "
                f"prediction, e.g. {missing[:10]}. Refusing rather than dropping them: a "
                "game silently lost here is an exclusion nobody counted."
            )
            raise TuningCorpusError(msg)

        joined = corpus_frame.merge(
            preds.rename(columns={"season": "prediction_season"}),
            on="game_id",
            how="inner",
        )
        disagree = joined.loc[
            joined["season"].astype(int) != joined["prediction_season"].astype(int),
            "game_id",
        ]
        if not disagree.empty:
            msg = (
                f"{target} predictions disagree with the corpus about the season of "
                f"{sorted(disagree)[:10]}"
            )
            raise TuningCorpusError(msg)

        if target == "wp":
            joined, excluded = attach_oof_market_probability(
                joined, walk_forward_slopes
            )

        frames[target] = (
            joined.rename(columns={"prediction": model_col, "actual": outcome_col})
            .loc[:, ["game_id", "season", "week", model_col, market_col, outcome_col]]
            .sort_values("game_id", ignore_index=True)
        )

    return BlendTuningFrames(frames=frames, excluded=excluded)


# ---------------------------------------------------------------------------
# Blending HISTORY: the one path every historical consumer takes
# ---------------------------------------------------------------------------

#: The market line each target's historical blend needs a game to carry.
_HISTORICAL_LINE_COLUMN: dict[str, str] = {"ats": "spread", "ou": "total"}


def blend_historical_predictions(
    blender: MarketBlender,
    predictions_df: pd.DataFrame,
    market_df: pd.DataFrame,
    target: str,
    *,
    artifacts_dir: Path | str = Path("artifacts"),
    silver_dir: Path | str = Path("data/silver"),
) -> pd.DataFrame:
    """Blend HISTORICAL predictions for *target*, honestly, over the games that can be blended.

    A33.2-review WR-04/WR-05. The diagnosis blended cut, the gate's blend re-score and the
    backtest's ``--blend`` all blend COMPLETED seasons, and all three used to route WP
    through the serving blend -- converting a CLOSING spread with the bound serving slope,
    which was fitted partly on those games' own outcomes (and, for the backtest, with no
    converter bound at all, so ``run_backtest(blend=True)`` raised on WP).

    * WP: the market side is each game's OWNED pre-lock spread (:func:`load_tuning_period_data`)
      converted with its own season's PRIOR-ONLY slope from the blend's BOUND converter
      artifact (``MarketBlender.blend_historical_wp_predictions``). *market_df* is not read.
    * ATS / O/U: the blend reads *market_df*'s line. Only games carrying that line are
      blended; the rest are LEFT OUT and counted, never kept unblended in a blended column.

    Args:
        blender: The loaded blend. For WP it must carry a converter binding.
        predictions_df: One row per game in the backtest contract.
        market_df: One row per game (ATS/O/U line source).
        target: ``"wp"``, ``"ats"`` or ``"ou"``.
        artifacts_dir: The root holding the blend's bound converter artifact.
        silver_dir: The silver root holding ``odds_timeline`` and ``games``.

    Returns:
        The blended rows.

    Raises:
        MarketProbabilityUnavailable: for WP, when the blender has no bound converter.
        ValueError: for an unknown target.
    """
    if target == "wp":
        if blender.market_probability_artifact_id is None:
            msg = (
                "the historical WP blend needs the blend's BOUND converter artifact for its "
                "prior-only walk_forward_slopes, and this blender has none bound. Load the "
                "blend with MarketBlender.from_artifacts rather than building one from a "
                "bare BlendConfig."
            )
            raise MarketProbabilityUnavailable(msg)
        converter = load_market_probability_artifact(
            blender.market_probability_artifact_id, artifacts_dir
        )
        corpus = load_tuning_period_data(silver_dir)
        return blender.blend_historical_wp_predictions(
            predictions_df, corpus.frame, converter["walk_forward_slopes"]
        )

    if target not in _HISTORICAL_LINE_COLUMN:
        msg = f"Unknown target: {target}. Must be 'wp', 'ats', or 'ou'."
        raise ValueError(msg)

    line_column = _HISTORICAL_LINE_COLUMN[target]
    with_line = set(market_df.loc[market_df[line_column].notna(), "game_id"])
    has_line = predictions_df["game_id"].isin(with_line)
    if not bool(has_line.all()):
        logger.info(
            "Historical blend leaves out games with no market line",
            target=target,
            n_excluded=int((~has_line).sum()),
            n_total=len(predictions_df),
        )
    return blender.blend_predictions(predictions_df.loc[has_line], market_df, target)
