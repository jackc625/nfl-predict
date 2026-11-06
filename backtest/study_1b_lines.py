"""Study 1b's lines: per game, the early-week line, the day-before line and the close.

THE EARLY LINE. A game's early line is its EARLIEST ``odds_timeline`` capture that is both

* at or after the end (kickoff + ``features.provenance.DECLARED_GAME_DURATION``) of every
  game of the same season with a LOWER week number that has a kickoff, scored or not -- the
  floor. Week 1 has no floor. The floor per (season, week) is the running maximum of the
  lower weeks' ends, so a game moved into midweek (the 2020 Tuesday and Wednesday games)
  holds back every later week's early line until it has ended; and
* within ``EARLY_WINDOW`` (7 days) before kickoff: ``kickoff - EARLY_WINDOW <= snapshot_ts <
  kickoff``. This keeps the lookahead boards (games listed two weeks out, week 1, the gap
  before the Super Bowl) out.

A capture carrying neither a spread nor a total is no line and is skipped. A capture missing
only one of them (in 2020 some Tuesday boards carry a spread but no total yet) is still the
early line: the missing number stays NaN, and only that target's move test drops the game.
The capture's own ``snapshot_ts`` is the information time the later tasks fence the model's
inputs to; ``created_at`` is an ingest time and is never read. No weekday arithmetic: the
floor and the window are instants compared in UTC.

THE DAY-BEFORE LINE is the same source's pre-lock capture (``models.blending_data``'s corpus:
the latest capture at or before the game's lock carrying both numbers). THE CLOSE is the
nflverse closing line (``backtest.study_1a.close_lines``).

ONE ROW PER GAME, OR ONE REASON. Every game lands in exactly one of ``frame`` and
``excluded``. An excluded game carries the FIRST failing reason in this order: no early
capture; no day-before line; an early capture not strictly before the day-before line (the
two would be one board, or the day-before line would predate the earlier weeks' end -- the
game is never paired with the previous week's board); no close. A day-before or closing row
carrying neither number counts as absent.

THREE SIGN CONVENTIONS, ONE SCALE. Every spread here is HOME-POSITIVE (positive when the
home team is favoured):

* ``early_spread``: the timeline stores the INVERTED sign, flipped exactly once, through
  ``models.market_probability.timeline_spread_to_home_fav_margin``;
* ``day_before_spread``: the corpus ``market_spread``, already home-positive, never flipped;
* ``close_spread``: nflverse ``spread_line``, already home-positive, never flipped.

Reads nothing at import and writes nothing. ASCII only, no emoji.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from backtest.study_1a import close_lines
from data.upstream_pin import load_schedules
from features.provenance import DECLARED_GAME_DURATION
from forward_ledger.settle import load_silver_games
from models.blending_data import OWNED_TIMELINE_TABLE, load_tuning_period_data
from models.market_probability import (
    OWNED_LINE_SEASONS,
    timeline_spread_to_home_fav_margin,
)

__all__ = [
    "EARLY_WINDOW",
    "EXCLUDED_COLUMNS",
    "EXCLUSION_REASONS",
    "LINE_COLUMNS",
    "NOT_BEFORE_DAY_BEFORE",
    "NO_CLOSE",
    "NO_DAY_BEFORE",
    "NO_EARLY_CAPTURE",
    "StudyLines",
    "early_captures",
    "load_study_lines",
    "study_lines",
]

#: No capture taken more than this long before kickoff is an early line.
EARLY_WINDOW: pd.Timedelta = cast(pd.Timedelta, pd.Timedelta(days=7))

NO_EARLY_CAPTURE = "no_early_capture"
NO_DAY_BEFORE = "no_day_before_line"
NOT_BEFORE_DAY_BEFORE = "early_capture_not_before_day_before_line"
NO_CLOSE = "no_close"

#: Every exclusion reason, in the order a game is tested against them.
EXCLUSION_REASONS: tuple[str, ...] = (
    NO_EARLY_CAPTURE,
    NO_DAY_BEFORE,
    NOT_BEFORE_DAY_BEFORE,
    NO_CLOSE,
)

#: One row per kept game. Every spread is home-positive; every ``*_ts`` is UTC.
LINE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "early_ts",
    "early_spread",
    "early_total",
    "day_before_ts",
    "day_before_spread",
    "day_before_total",
    "close_spread",
    "close_total",
)

EXCLUDED_COLUMNS: tuple[str, ...] = ("game_id", "season", "week", "reason")

_GAME_COLUMNS: tuple[str, ...] = ("game_id", "season", "week", "kickoff_et")
_TIMELINE_COLUMNS: tuple[str, ...] = ("game_id", "snapshot_ts", "spread", "total")
_DAY_BEFORE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "snapshot_ts",
    "market_spread",
    "market_total",
)
_CLOSE_COLUMNS: tuple[str, ...] = ("game_id", "close_spread", "close_total")


@dataclass(frozen=True)
class StudyLines:
    """The study's lines and the games that have none.

    Attributes:
        frame: One row per kept game (:data:`LINE_COLUMNS`), sorted by ``game_id``.
        excluded: One row per left-out game (:data:`EXCLUDED_COLUMNS`), carrying its first
            failing reason (:data:`EXCLUSION_REASONS`), sorted by ``game_id``.
    """

    frame: pd.DataFrame
    excluded: pd.DataFrame


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _columns(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> pd.DataFrame:
    """*frame* narrowed to *columns* (all required) with a text ``game_id``."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        msg = f"the {what} frame is missing {missing}; it must carry {list(columns)}"
        raise ValueError(msg)
    out = frame.loc[:, list(columns)].copy()
    out["game_id"] = out["game_id"].astype(str)
    return out


def _one_row_per_game(frame: pd.DataFrame, what: str) -> None:
    repeated = sorted(set(frame.loc[frame["game_id"].duplicated(), "game_id"]))
    if repeated:
        msg = f"the {what} frame carries more than one row for {repeated[:10]}"
        raise ValueError(msg)


def _utc(frame: pd.DataFrame, column: str, what: str) -> pd.Series:
    """*frame*'s *column* as UTC instants. A naive value is refused, never relabelled."""
    series = cast(pd.Series, frame[column])
    if series.empty:
        # No instant exists to be relabelled; an empty column only needs the dtype.
        return pd.to_datetime(series, utc=True)
    parsed = pd.to_datetime(series)
    if getattr(parsed.dt, "tz", None) is None:
        msg = (
            f"the {what} {column} is timezone-naive; its instant cannot be compared with a "
            "kickoff without guessing its zone. Refusing rather than relabelling it as UTC."
        )
        raise ValueError(msg)
    return parsed.dt.tz_convert("UTC")


def _schedule(games: pd.DataFrame) -> pd.DataFrame:
    """``game_id, season, week, kickoff`` (UTC), one row per game."""
    schedule = _columns(games, _GAME_COLUMNS, "games")
    _one_row_per_game(schedule, "games")
    schedule["season"] = schedule["season"].astype(int)
    schedule["week"] = schedule["week"].astype(int)
    schedule["kickoff"] = _utc(schedule, "kickoff_et", "games")
    return schedule.drop(columns="kickoff_et")


# ---------------------------------------------------------------------------
# The early line
# ---------------------------------------------------------------------------


def _week_floors(schedule: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week): the latest end of every lower week's games; NaT for the first.

    Only games with a kickoff count. The running maximum carries a game played late (after
    the next week's games) forward to every later week, not only the next.
    """
    played = schedule.dropna(subset=["kickoff"])
    week_end = (
        (played["kickoff"] + DECLARED_GAME_DURATION)
        .groupby([played["season"], played["week"]])
        .max()
        .sort_index()
    )
    running = week_end.groupby(level="season").cummax()
    floor = running.groupby(level="season").shift(1)
    return floor.rename("floor").reset_index()


def early_captures(timeline: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each game's earliest eligible capture: ``game_id, early_ts, early_spread, early_total``.

    Eligible: at or after the end of every game of the season's lower weeks, and
    ``kickoff - EARLY_WINDOW <= snapshot_ts < kickoff``; a capture carrying neither a spread
    nor a total is skipped. A game with no eligible capture is absent. ``early_ts`` is UTC
    and ``early_spread`` home-positive (the one flip).

    Raises:
        ValueError: a required column is missing, a game repeats in *games*, an instant is
            naive, or two eligible captures of one game share a timestamp.
    """
    schedule = _schedule(games)
    schedule = schedule.merge(
        _week_floors(schedule),
        on=["season", "week"],
        how="left",
        validate="many_to_one",
    )

    captures = _columns(timeline, _TIMELINE_COLUMNS, "timeline")
    captures["snapshot_ts"] = _utc(captures, "snapshot_ts", "timeline")
    # A capture carrying neither number is no line; one missing a single number still is.
    captures = captures.dropna(subset=["snapshot_ts"]).dropna(
        subset=["spread", "total"], how="all"
    )

    joined = captures.merge(
        schedule[["game_id", "kickoff", "floor"]], on="game_id", how="inner"
    )
    after_floor = joined["floor"].isna() | (joined["snapshot_ts"] >= joined["floor"])
    in_window = (joined["snapshot_ts"] >= joined["kickoff"] - EARLY_WINDOW) & (
        joined["snapshot_ts"] < joined["kickoff"]
    )
    eligible = joined.loc[after_floor & in_window]

    tied = eligible.duplicated(["game_id", "snapshot_ts"])
    if bool(tied.any()):
        games_tied = sorted(set(eligible.loc[tied, "game_id"]))
        msg = (
            f"two eligible captures share a timestamp for {games_tied[:10]}; refusing "
            "rather than picking one of them"
        )
        raise ValueError(msg)

    earliest = (
        eligible.sort_values(["game_id", "snapshot_ts"], kind="mergesort")
        .groupby("game_id", sort=True)
        .head(1)
        .reset_index(drop=True)
    )
    return pd.DataFrame(
        {
            "game_id": earliest["game_id"],
            "early_ts": earliest["snapshot_ts"],
            # THE ONE FLIP of the timeline's inverted spread onto the home-positive scale.
            "early_spread": timeline_spread_to_home_fav_margin(earliest["spread"]),
            "early_total": earliest["total"].astype(float),
        }
    )


# ---------------------------------------------------------------------------
# One row per game
# ---------------------------------------------------------------------------


def study_lines(
    timeline: pd.DataFrame,
    games: pd.DataFrame,
    day_before: pd.DataFrame,
    close: pd.DataFrame,
) -> StudyLines:
    """The early, day-before and closing lines of every game in *games*, or its reason.

    Pure: reads and writes nothing.

    Args:
        timeline: ``odds_timeline`` rows (``game_id, snapshot_ts, spread, total``; the
            spread on the timeline's inverted sign).
        games: The games (``game_id, season, week, kickoff_et``), one row each.
        day_before: The corpus rows (``game_id, snapshot_ts, market_spread,
            market_total``); ``market_spread`` is already home-positive.
        close: ``game_id, close_spread, close_total`` (nflverse, home-positive).

    Raises:
        ValueError: a required column is missing, a game repeats in any input frame, or an
            instant is naive.
    """
    schedule = _schedule(games)
    early = early_captures(timeline, games)

    day = _columns(day_before, _DAY_BEFORE_COLUMNS, "day-before")
    _one_row_per_game(day, "day-before")
    day["snapshot_ts"] = _utc(day, "snapshot_ts", "day-before")
    day = day.rename(
        columns={
            "snapshot_ts": "day_before_ts",
            "market_spread": "day_before_spread",
            "market_total": "day_before_total",
        }
    )

    closing = _columns(close, _CLOSE_COLUMNS, "close")
    _one_row_per_game(closing, "close")

    merged = (
        schedule[["game_id", "season", "week"]]
        .merge(early, on="game_id", how="left", validate="one_to_one")
        .merge(day, on="game_id", how="left", validate="one_to_one")
        .merge(closing, on="game_id", how="left", validate="one_to_one")
    )

    no_day_before = merged["day_before_ts"].isna() | (
        merged["day_before_spread"].isna() & merged["day_before_total"].isna()
    )
    failures = [
        merged["early_ts"].isna(),
        no_day_before,
        ~(merged["early_ts"] < merged["day_before_ts"]),
        merged["close_spread"].isna() & merged["close_total"].isna(),
    ]
    # np.select takes the FIRST true condition, so each game carries its first failure.
    reason = pd.Series(
        np.select(failures, list(EXCLUSION_REASONS), default=""), index=merged.index
    )

    kept = reason == ""
    frame = merged.loc[kept, list(LINE_COLUMNS)].sort_values(
        "game_id", ignore_index=True
    )
    excluded = (
        merged.loc[~kept, ["game_id", "season", "week"]]
        .assign(reason=reason[~kept])
        .loc[:, list(EXCLUDED_COLUMNS)]
        .sort_values("game_id", ignore_index=True)
    )
    return StudyLines(frame=frame, excluded=excluded)


def load_study_lines(
    silver_dir: Path, seasons: tuple[int, ...] = OWNED_LINE_SEASONS
) -> StudyLines:
    """The study's lines for *seasons*, read from silver and the pinned nflverse schedules.

    Reads silver ``odds_timeline`` and ``games`` (the *seasons* only), the day-before
    corpus (``models.blending_data.load_tuning_period_data``) and the close from
    ``data.upstream_pin.load_schedules``. Writes nothing.
    """
    root = Path(silver_dir)
    timeline = pd.read_parquet(root / f"{OWNED_TIMELINE_TABLE}.parquet")
    games = load_silver_games(root)
    games = games.loc[games["season"].isin(seasons)]
    corpus = load_tuning_period_data(root)
    close = close_lines(load_schedules(list(seasons)))
    return study_lines(timeline, games, corpus.frame, close)
