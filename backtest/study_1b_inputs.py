"""Study 1b's inputs: each studied game's v2 input row AS KNOWN AT ITS EARLY LINE CAPTURE.

The v2 gold (``outputs/gold_v2/``) fences every game at its day-before lock. Study 1b scores
a game at its early capture instead (``early_ts``, usually Tuesday 11:55 ET), so each row
here is a gold row with the inputs that can change between the two instants rebuilt:

* QB (``home_/away_qb_adjustment``): ``features.qb_early.QBTrackerEarly``, which never reads
  an untimed depth chart, run with the EARLY lock frame (without ``lock_frame=`` it would
  silently use the day-before locks);
* the market rating (``home_/away_mkt_rating``, ``mkt_rating_diff``, ``mkt_total_sum``,
  ``mkt_rating_coverage``) and the league scoring level (``league_points_level``), each read
  through its ``locks=`` keyword at the same early lock frame;
* wind (``wind_mph``), which nothing knows on Tuesday: it is left blank and the recipe's
  median imputer fills it.

``scripts.build_gold_v2.add_v2_derived_columns`` then recomputes ``qb_value_diff`` and the
other derived inputs from the per-side columns. EVERY OTHER COLUMN IS THE GOLD'S, unchanged.
Elo, success rate, opponent-adjusted EPA, pace and the home-field advantage rest only on
finished games, so they are the same at the early capture as at the lock -- PROVIDED both
teams' previous games had ended by then. ``settled_by_early_time`` checks exactly that per
game; a game failing it is left out and counted (``NOT_SETTLED``), never scored with
day-before inputs. A game with no row in a gold matrix is left out too (``NO_GOLD_ROW``).

THE EARLY LOCK FRAME is ``utils.game_lock.lock_frame`` over the family scope with each
early-timed game's lock replaced by its early time. An early time after the game's own lock
is refused: it would read more than the day-before gold, not less. After the families run,
any early-timed game whose rating or level ``information_time`` is after its early time is
refused with ``RuntimeError``.

Reads only the gold matrices it is pointed at (and, through the QB tracker, the pinned
play-by-play); writes nothing. No weekday arithmetic. ASCII only, no emoji.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from features.league_level import league_points_level
from features.market_rating import market_ratings
from features.provenance import DECLARED_GAME_DURATION
from features.qb_early import QBTrackerEarly
from models.v2_recipe import V2_TARGETS
from scripts.build_gold_v2 import add_v2_derived_columns

__all__ = [
    "EARLY_REPLACED_COLUMNS",
    "NOT_SETTLED",
    "NO_GOLD_ROW",
    "EarlyInputs",
    "build_early_inputs",
    "early_lock_frame",
    "replace_early_columns",
    "settled_by_early_time",
]

#: The gold columns rebuilt as of the early capture (``wind_mph`` is blanked). The derived
#: inputs (``qb_value_diff`` and the rest) are recomputed from them afterwards.
EARLY_REPLACED_COLUMNS: tuple[str, ...] = (
    "home_qb_adjustment",
    "away_qb_adjustment",
    "home_mkt_rating",
    "away_mkt_rating",
    "mkt_rating_diff",
    "mkt_total_sum",
    "mkt_rating_coverage",
    "league_points_level",
    "wind_mph",
)

NOT_SETTLED = "inputs_not_settled_at_early_capture"
NO_GOLD_ROW = "no_gold_row"

_MARKET_COLUMNS: tuple[str, ...] = (
    "home_mkt_rating",
    "away_mkt_rating",
    "mkt_rating_diff",
    "mkt_total_sum",
    "mkt_rating_coverage",
)


@dataclass(frozen=True)
class EarlyInputs:
    """The early-capture input rows and the games left out.

    Attributes:
        rows: One frame per target (``V2_TARGETS``): the gold rows of the kept games with
            the early columns replaced. Never empty, one row per game.
        excluded: ``game_id, reason`` for every early-timed game left out, sorted by
            ``game_id``; the reason is ``NOT_SETTLED`` or ``NO_GOLD_ROW``, the first that
            applies.
    """

    rows: dict[str, pd.DataFrame]
    excluded: pd.DataFrame


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _first_ten(ids: pd.Index) -> list[str]:
    return [str(game_id) for game_id in ids[:10]]


def _aware_utc(values: pd.Series, what: str) -> pd.Series:
    """*values* as UTC instants (nulls stay NaT). A naive instant is refused, not relabelled."""
    naive = pd.Index(
        [
            key
            for key, value in values.dropna().items()
            if pd.Timestamp(value).tzinfo is None
        ]
    )
    if len(naive):
        msg = (
            f"{len(naive)} {what} value(s) carry no timezone, first ten: "
            f"{_first_ten(naive)}; refusing rather than guessing their zone"
        )
        raise ValueError(msg)
    return pd.to_datetime(values, utc=True)


def _early_instants(early_times: pd.Series) -> pd.Series:
    """*early_times* as ET instants on a text ``game_id`` index, one non-null value each."""
    index = pd.Index(early_times.index.astype(str), name="game_id")
    repeated = pd.Index(index[index.duplicated()]).unique()
    if len(repeated):
        msg = f"early times name {len(repeated)} game(s) more than once: {_first_ten(repeated)}"
        raise ValueError(msg)
    empty = pd.Index(index[early_times.isna().to_numpy()])
    if len(empty):
        msg = (
            f"early times hold no instant for {len(empty)} game(s): {_first_ten(empty)}"
        )
        raise ValueError(msg)
    values = pd.Series(early_times.to_numpy(), index=index)
    early = _aware_utc(values, "early time").dt.tz_convert(lock_rule.ET)
    return early.rename("early_time")


def _teams(games: pd.DataFrame) -> pd.DataFrame:
    """``home_team, away_team`` indexed by text ``game_id``; a repeated id is refused."""
    missing = [
        c for c in ("game_id", "home_team", "away_team") if c not in games.columns
    ]
    if missing:
        msg = f"the games frame is missing {missing}"
        raise ValueError(msg)
    teams = games.loc[:, ["home_team", "away_team"]].set_axis(
        pd.Index(games["game_id"].astype(str), name="game_id")
    )
    repeated = pd.Index(teams.index[teams.index.duplicated()]).unique()
    if len(repeated):
        msg = f"the games frame repeats {len(repeated)} game id(s): {_first_ten(repeated)}"
        raise ValueError(msg)
    return teams


def _schedule(games: pd.DataFrame) -> pd.DataFrame:
    """``home_team, away_team, kickoff`` (UTC, NaT where unknown) indexed by ``game_id``."""
    if "kickoff_et" not in games.columns:
        msg = "the games frame is missing ['kickoff_et']"
        raise ValueError(msg)
    schedule = _teams(games)
    kickoff = pd.Series(games["kickoff_et"].to_numpy(), index=schedule.index)
    return schedule.assign(kickoff=_aware_utc(kickoff, "kickoff_et"))


def _by_game(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    """*frame* indexed by text ``game_id``; a repeated id is refused."""
    indexed = frame.set_axis(pd.Index(frame["game_id"].astype(str), name="game_id"))
    repeated = pd.Index(indexed.index[indexed.index.duplicated()]).unique()
    if len(repeated):
        msg = f"the {what} frame repeats {len(repeated)} game id(s): {_first_ten(repeated)}"
        raise ValueError(msg)
    return indexed


# ---------------------------------------------------------------------------
# The early lock frame and the settled check
# ---------------------------------------------------------------------------


def early_lock_frame(games: pd.DataFrame, early_times: pd.Series) -> pd.Series:
    """``lock_rule.lock_frame(games)`` with each early-timed game's lock set to its early time.

    The early times are converted to ``lock_rule.ET``; every other game keeps its
    day-before lock, so the series names every game of *games* exactly once, tz-aware and
    non-null, as ``features.supplied_locks`` requires.

    Raises:
        ValueError: an early time names a game not in *games*, falls after that game's own
            lock (at the lock is allowed), repeats a game, is null or is naive.
        utils.game_lock.MissingKickoffError: a game of *games* has no kickoff.
    """
    early = _early_instants(early_times)
    locks = lock_rule.lock_frame(games)
    unknown = early.index.difference(locks.index)
    if len(unknown):
        msg = (
            f"early times name {len(unknown)} game(s) not in the games frame: "
            f"{_first_ten(unknown)}"
        )
        raise ValueError(msg)
    late = pd.Index(early.index[(early > locks.reindex(early.index)).to_numpy()])
    if len(late):
        msg = (
            f"{len(late)} early time(s) fall after the game's own day-before lock, so they "
            f"would read more than the gold does: {_first_ten(late)}"
        )
        raise ValueError(msg)
    out = locks.copy()
    out.loc[early.index] = early
    return out


def settled_by_early_time(games: pd.DataFrame, early_times: pd.Series) -> pd.Series:
    """Per early-timed game: had both teams' previous games ended by its early time?

    A team's previous game is the latest one it kicked off before this game (any season);
    it ended at kickoff + ``DECLARED_GAME_DURATION``, and an end AT the early time counts as
    ended. A team with no earlier game counts as settled. Games without a kickoff are never
    anyone's previous game.

    Returns:
        A bool series named ``settled``, indexed by the early-timed ``game_id``s in the
        order given.

    Raises:
        ValueError: an early-timed game is not in *games* or has no kickoff, *games*
            repeats an id, or an early time is unusable (see ``early_lock_frame``).
    """
    early = _early_instants(early_times).dt.tz_convert("UTC")
    schedule = _schedule(games)
    unknown = early.index.difference(schedule.index)
    if len(unknown):
        msg = f"{len(unknown)} early-timed game(s) are not in the games frame: {_first_ten(unknown)}"
        raise ValueError(msg)
    undated = pd.Index(
        early.index[schedule["kickoff"].reindex(early.index).isna().to_numpy()]
    )
    if len(undated):
        msg = (
            f"{len(undated)} early-timed game(s) have no kickoff: {_first_ten(undated)}"
        )
        raise ValueError(msg)

    kickoffs = schedule["kickoff"]
    sides = pd.concat(
        [
            pd.DataFrame({"team": schedule["home_team"], "kickoff": kickoffs}),
            pd.DataFrame({"team": schedule["away_team"], "kickoff": kickoffs}),
        ]
    ).reset_index()
    sides = sides.dropna(subset=["kickoff"]).sort_values(
        ["team", "kickoff"], kind="mergesort"
    )
    sides["previous_end"] = (
        sides.groupby("team")["kickoff"].shift(1) + DECLARED_GAME_DURATION
    )
    # The later of the two teams' previous ends; NaT when neither team played before.
    latest_end = sides.groupby("game_id")["previous_end"].max().reindex(early.index)
    settled = latest_end.isna() | (latest_end <= early)
    return settled.astype(bool).rename("settled")


# ---------------------------------------------------------------------------
# The replaced columns
# ---------------------------------------------------------------------------


def _qb_by_team(qb: pd.DataFrame) -> pd.Series:
    """``qb_adjustment`` keyed by ``(game_id, team)``; a repeated key is refused."""
    keys = pd.MultiIndex.from_arrays(
        [qb["game_id"].astype(str).to_numpy(), qb["team"].to_numpy()]
    )
    repeated = pd.Index(keys[keys.duplicated()]).unique()
    if len(repeated):
        msg = f"the QB frame repeats {len(repeated)} (game, team) key(s): {list(repeated[:10])}"
        raise ValueError(msg)
    return pd.Series(qb["qb_adjustment"].to_numpy(dtype=float), index=keys)


def _side_qb(
    qb: pd.Series, game_ids: pd.Series, teams: np.ndarray, side: str
) -> np.ndarray:
    """The ``qb_adjustment`` of each game's *side* team; a missing row is refused."""
    keys = pd.MultiIndex.from_arrays([game_ids.to_numpy(), teams])
    absent = pd.Index(game_ids[~keys.isin(qb.index)]).unique()
    if len(absent):
        msg = (
            f"the QB frame has no {side}-team row for {len(absent)} game(s): "
            f"{_first_ten(absent)}"
        )
        raise ValueError(msg)
    return qb.reindex(keys).to_numpy(dtype=float)


def replace_early_columns(
    gold: pd.DataFrame,
    games: pd.DataFrame,
    qb: pd.DataFrame,
    ratings: pd.DataFrame,
    level: pd.DataFrame,
) -> pd.DataFrame:
    """The gold rows of the games in *ratings*, with the early columns replaced.

    ``home_/away_qb_adjustment`` come from *qb* (``game_id, team, qb_adjustment``, the team
    matched to the game's home and away team in *games*), the five market columns from
    *ratings*, ``league_points_level`` from *level*, ``wind_mph`` is NaN, and then
    ``add_v2_derived_columns`` recomputes the derived inputs. Every other column, the column
    order and the gold's row order are unchanged; the index is reset.

    Raises:
        ValueError: a game lacks a QB row for either team, a level row, a gold row or a
            *games* row; *gold* lacks a replaced column; or any frame repeats a key.
    """
    missing = [c for c in ("game_id", *EARLY_REPLACED_COLUMNS) if c not in gold.columns]
    if missing:
        msg = f"the gold matrix is missing {missing}"
        raise ValueError(msg)
    rated = _by_game(ratings, "ratings")
    wanted = rated.index
    gold_ids = _by_game(gold, "gold").index
    no_gold = wanted.difference(gold_ids)
    if len(no_gold):
        msg = f"{len(no_gold)} rated game(s) have no gold row: {_first_ten(no_gold)}"
        raise ValueError(msg)

    out = gold.loc[gold_ids.isin(wanted)].reset_index(drop=True)
    game_ids = out["game_id"].astype(str)
    teams = _teams(games)
    unscheduled = pd.Index(game_ids).difference(teams.index)
    if len(unscheduled):
        msg = f"{len(unscheduled)} game(s) are not in the games frame: {_first_ten(unscheduled)}"
        raise ValueError(msg)
    teams = teams.reindex(game_ids)

    qb_values = _qb_by_team(qb)
    for side in ("home", "away"):
        out[f"{side}_qb_adjustment"] = _side_qb(
            qb_values, game_ids, teams[f"{side}_team"].to_numpy(), side
        )

    market = rated.reindex(game_ids)
    for column in _MARKET_COLUMNS:
        out[column] = market[column].to_numpy(dtype=float)

    levels = _by_game(level, "level")
    no_level = pd.Index(game_ids).difference(levels.index)
    if len(no_level):
        msg = f"the level frame has no row for {len(no_level)} game(s): {_first_ten(no_level)}"
        raise ValueError(msg)
    out["league_points_level"] = (
        levels["league_points_level"].reindex(game_ids).to_numpy(dtype=float)
    )
    # Nothing knows the wind at the early capture; the recipe's median imputer fills it.
    out["wind_mph"] = np.nan
    return add_v2_derived_columns(out)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def _family_scope(games: pd.DataFrame, early_ids: pd.Index) -> pd.DataFrame:
    """The games of the seasons (first early season - 1) through (last early season)."""
    ids = games["game_id"].astype(str)
    unknown = early_ids.difference(pd.Index(ids))
    if len(unknown):
        msg = f"{len(unknown)} early-timed game(s) are not in the games frame: {_first_ten(unknown)}"
        raise ValueError(msg)
    seasons = games["season"].astype(int)
    timed = seasons[ids.isin(early_ids.tolist()).to_numpy()]
    in_scope = seasons.between(int(timed.min()) - 1, int(timed.max()))
    return games.loc[in_scope].reset_index(drop=True)


def _early_qb(
    scope: pd.DataFrame, early_ids: pd.Index, locks: pd.Series
) -> pd.DataFrame:
    """QB rows of every (season, week) holding an early-timed game, at the early locks.

    One tracker and one *scope* object for every week, so the pinned play-by-play is loaded
    once per season and each week's context is reused. ``as_of_datetime`` is no fence.
    """
    timed = scope.loc[scope["game_id"].astype(str).isin(early_ids.tolist())]
    weeks = sorted(
        {(int(s), int(w)) for s, w in zip(timed["season"], timed["week"], strict=True)}
    )
    tracker = QBTrackerEarly()
    as_of = datetime.now(UTC)
    frames = [
        tracker.build_features(
            scope, as_of, target_season=season, target_week=week, lock_frame=locks
        )
        for season, week in weeks
    ]
    return pd.concat(frames, ignore_index=True)


def _refuse_late_information(name: str, family: pd.DataFrame, early: pd.Series) -> None:
    """Refuse when an early-timed game's family row is missing or read past its early time."""
    rows = _by_game(family, name)
    absent = early.index.difference(rows.index)
    if len(absent):
        msg = f"family {name!r} has no row for {len(absent)} early-timed game(s): {_first_ten(absent)}"
        raise ValueError(msg)
    times = pd.Series(rows["information_time"]).reindex(early.index)
    read = _aware_utc(times, f"{name} information_time")
    late = pd.Index(early.index[(read > early.dt.tz_convert("UTC")).to_numpy()])
    if len(late):
        msg = (
            f"family {name!r} read information timed after the early capture for "
            f"{len(late)} game(s): {_first_ten(late)}"
        )
        raise RuntimeError(msg)


def _exclusion_reasons(
    games: pd.DataFrame, early: pd.Series, golds: dict[str, pd.DataFrame]
) -> pd.Series:
    """Per early-timed game, its first failing reason, or "" when it is kept."""
    settled = settled_by_early_time(games, early).to_numpy()
    in_every_gold = np.ones(len(early), dtype=bool)
    for gold in golds.values():
        in_every_gold &= early.index.isin(gold["game_id"].astype(str))
    reason = np.select(
        [~settled, ~in_every_gold], [NOT_SETTLED, NO_GOLD_ROW], default=""
    )
    return pd.Series(reason, index=early.index, name="reason")


def build_early_inputs(
    gold_dir: Path, games: pd.DataFrame, nflverse: pd.DataFrame, early_times: pd.Series
) -> EarlyInputs:
    """The v2 input rows of every early-timed game, as known at its early capture.

    Args:
        gold_dir: The v2 gold directory holding ``features_{target}.parquet``.
        games: All silver games. The families run on the seasons from the first early
            season minus one through the last, every one of which must have a kickoff.
        nflverse: The pinned nflverse schedules the market rating reads its lines from.
        early_times: ``game_id``-indexed tz-aware early capture instants.

    Raises:
        RuntimeError: an early-timed game's market-rating or league-level
            ``information_time`` is after its early time.
        ValueError: an early time is unusable (see ``early_lock_frame``), a family has no
            row for an early-timed game, or no early-timed game is kept.
    """
    early = _early_instants(early_times)
    if early.empty:
        msg = "no early-timed game was supplied"
        raise ValueError(msg)
    scope = _family_scope(games, early.index)
    locks = early_lock_frame(scope, early)

    qb = _early_qb(scope, early.index, locks)
    ratings = market_ratings(scope, nflverse, locks=locks)
    level = league_points_level(scope, locks=locks)
    _refuse_late_information("market_rating", ratings, early)
    _refuse_late_information("league_level", level, early)

    golds = {
        target: pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
        for target in V2_TARGETS
    }
    reasons = _exclusion_reasons(games, early, golds)
    kept = pd.Index(reasons.index[(reasons == "").to_numpy()]).tolist()
    if not kept:
        counts = reasons.value_counts().to_dict()
        msg = f"no early-timed game is kept; every one was left out: {counts}"
        raise ValueError(msg)

    kept_ratings = ratings.loc[ratings["game_id"].astype(str).isin(kept)]
    kept_level = level.loc[level["game_id"].astype(str).isin(kept)]
    rows = {
        target: replace_early_columns(
            golds[target], scope, qb, kept_ratings, kept_level
        )
        for target in V2_TARGETS
    }
    left_out = reasons.loc[reasons != ""]
    excluded = pd.DataFrame(
        {"game_id": left_out.index.to_numpy(), "reason": left_out.to_numpy()}
    ).sort_values("game_id", ignore_index=True)
    return EarlyInputs(rows=rows, excluded=excluded)
