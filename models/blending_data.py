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

from models.market_probability import (
    OWNED_LINE_SEASONS,
    timeline_spread_to_home_fav_margin,
)
from utils import get_logger
from utils.game_lock import lock_frame

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
    # THE lock, one per game, from THE rule -- never a second derivation here.
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
# Retired with the week-varying blend by Plan 33.2-24 Task 2, together with their readers
# in models/blending.py and backtest/tune.py. They are the 2010-2017 closing-line window and
# the synthetic-prediction noise profile the dynamic sigmoid was fitted on.
# ---------------------------------------------------------------------------

TUNING_SEASONS: list[int] = list(range(2010, 2018))
"""The retired closing-line tuning window (2010-2017). Removed in Task 2."""

MIN_NOISE_SAMPLE_COUNT = 30
"""Minimum games per week for the retired noise profile. Removed in Task 2."""


def extract_noise_profile(
    baselines_dir: Path | None = None,
    min_sample_count: int = MIN_NOISE_SAMPLE_COUNT,
) -> dict[str, pd.DataFrame]:
    """Per-week model-vs-closing-market error statistics. Removed in Task 2.

    Used only by the retired dynamic blend tuning to synthesize predictions for the
    2010-2017 closing-line window.
    """
    if baselines_dir is None:
        baselines_dir = Path("data/baselines/v2.0")

    if not baselines_dir.exists():
        msg = (
            f"Baselines directory not found: {baselines_dir}. "
            "Run backtest first to generate baseline predictions."
        )
        raise FileNotFoundError(msg)

    profiles: dict[str, pd.DataFrame] = {}
    target_configs = {
        "wp": ("predictions_wp.parquet", "model_prob", "fair_closing_prob"),
        "ats": ("predictions_ats.parquet", "model_spread", "spread"),
        "ou": ("predictions_ou.parquet", "model_total", "total"),
    }

    for target, (filename, model_col, market_col) in target_configs.items():
        parquet_path = baselines_dir / filename
        if not parquet_path.exists():
            msg = (
                f"{filename} not found in {baselines_dir}. "
                "Run backtest first to generate baseline predictions."
            )
            raise FileNotFoundError(msg)

        df = pd.read_parquet(parquet_path)
        df["week"] = df["game_id"].str.extract(r"_W(\d+)_")[0].astype(int)
        df["season"] = df["game_id"].str.split("_").str[0].astype(int)
        df["error"] = df[model_col] - df[market_col]
        max_week = df["season"].apply(lambda s: 17 if s <= 2020 else 18)
        df = df[df["week"] <= max_week]

        season_mean = float(df["error"].mean())
        season_std = float(df["error"].std())
        stats = df.groupby("week")["error"].agg(["mean", "std", "count"]).reset_index()
        stats["count"] = stats["count"].astype(int)
        for idx in stats.index:
            count = stats.at[idx, "count"]
            if count < min_sample_count:
                blend_weight = count / min_sample_count
                stats.at[idx, "mean"] = (
                    blend_weight * stats.at[idx, "mean"]
                    + (1 - blend_weight) * season_mean
                )
                stats.at[idx, "std"] = (
                    blend_weight * stats.at[idx, "std"]
                    + (1 - blend_weight) * season_std
                )
        profiles[target] = stats

    return profiles
