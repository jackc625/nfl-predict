"""Point-in-time statistics for the gold imputer (p332_ extra step 7b, owner ruling 2026-09-22).

WHY THIS MODULE EXISTS. ``scripts.build_features.FeatureMatrixBuilder`` fills a missing feature
value before normalization. It used to fill a gap with that team's mean over the WHOLE season --
weeks after the gap included -- then with the whole-season league mean, and to fall back to a
median of the gap's own season when no earlier season could be fitted. Under D33.2-01 a game's
inputs are what was known at its lock (18:00 ET on the day before kickoff), so every one of those
statistics could read post-lock information.

THE TIMING IS THE BUILDERS' OWN. A game's RESULT exists at its END -- kickoff plus
``features.provenance.DECLARED_GAME_DURATION`` -- and a value from that game is admissible for
another game when its end is at or before that game's lock (``utils.game_lock``: at-lock counts,
one second after does not). That is the rule every lock-keyed rolling window already uses
(``features.team_form.team_game_schedule``); this module restates none of it.

A GAME THAT CANNOT BE TIMED IS NEVER ADMITTED, and a gap in a row that cannot be timed takes no
within-season statistic at all: its sentinel end is later than every lock and its sentinel lock
earlier than every end.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import utils.game_lock as lock_rule
from features.provenance import DECLARED_GAME_DURATION
from features.qb_tracking import to_aware_utc

#: The timing frame's columns: when each game's RESULT existed, and its lock (UTC nanoseconds).
TIMING_COLUMNS: tuple[str, ...] = ("end_ns", "lock_ns")

#: An untimed row's END: later than every lock, so its value is never admitted.
UNTIMED_END_NS: int = int(np.iinfo(np.int64).max)

#: An untimed row's LOCK: earlier than every end, so it admits nothing.
UNTIMED_LOCK_NS: int = int(np.iinfo(np.int64).min)


def _utc_nanoseconds(values: pd.Series) -> np.ndarray:
    """Tz-aware instants as int64 UTC nanoseconds, whatever unit they were stored in."""
    return values.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def imputation_game_timing(games_df: pd.DataFrame) -> pd.DataFrame | None:
    """``game_id``-indexed ``end_ns`` / ``lock_ns`` for every game that carries a kickoff.

    Args:
        games_df: Silver ``games`` shape: ``game_id`` and a tz-aware ``kickoff_et``.

    Returns:
        A frame with exactly ``TIMING_COLUMNS``, or ``None`` when the frame has no
        ``game_id`` / ``kickoff_et`` column at all (nothing in it can be timed). A game with no
        kickoff is left out, so it is untimed; a naive kickoff is refused by name, never
        relabelled (``to_aware_utc`` and ``utils.game_lock``).
    """
    if "game_id" not in games_df.columns or "kickoff_et" not in games_df.columns:
        return None
    games = games_df.loc[games_df["kickoff_et"].notna()]
    index = pd.Index(games["game_id"].astype(str), name="game_id")
    if len(games) == 0:
        return pd.DataFrame(
            {column: pd.Series(dtype="int64") for column in TIMING_COLUMNS}, index=index
        )
    ends = (
        to_aware_utc(pd.Series(games["kickoff_et"]), column="kickoff_et")
        + DECLARED_GAME_DURATION
    )
    locks = lock_rule.lock_frame(games)
    return pd.DataFrame(
        {
            "end_ns": _utc_nanoseconds(ends.set_axis(index)),
            "lock_ns": _utc_nanoseconds(locks.set_axis(index)),
        },
        index=index,
    )


def row_timing(
    frame: pd.DataFrame, timing: pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(end_ns, lock_ns)`` for *frame*, positionally aligned with its rows.

    A row whose game the timing does not cover -- or every row, when there is no timing or no
    ``game_id`` column -- gets the untimed sentinels. The lookup goes through a nullable integer
    so no nanosecond value is ever rounded through a float.
    """
    rows = len(frame)
    if timing is None or "game_id" not in frame.columns:
        return (
            np.full(rows, UNTIMED_END_NS, dtype="int64"),
            np.full(rows, UNTIMED_LOCK_NS, dtype="int64"),
        )
    ids = frame["game_id"].astype(str).to_numpy()
    ends = timing["end_ns"].astype("Int64").reindex(ids)
    locks = timing["lock_ns"].astype("Int64").reindex(ids)
    return (
        ends.fillna(UNTIMED_END_NS).astype("int64").to_numpy(),
        locks.fillna(UNTIMED_LOCK_NS).astype("int64").to_numpy(),
    )


def admitted_mean(values: pd.Series, ends_ns: np.ndarray, lock_ns: int) -> float:
    """Mean of the non-null *values* whose game had ENDED at or before *lock_ns*.

    Args:
        values: The candidate values (a Series, so the mean is pandas' own).
        ends_ns: Each candidate's end, positionally aligned with *values*.
        lock_ns: The lock of the game being filled.

    Returns:
        The mean, or NaN when no admitted candidate carries a value (or the gap is untimed).
    """
    if lock_ns == UNTIMED_LOCK_NS:
        return float("nan")
    admitted = values[ends_ns <= lock_ns]
    if not admitted.notna().any():
        return float("nan")
    return float(admitted.mean())
