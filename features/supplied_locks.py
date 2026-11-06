"""The checks a supplied ``locks`` series passes before a feature reads it.

``features.market_rating`` and ``features.league_level`` accept a ``game_id``-indexed
series of tz-aware instants in place of the day-before lock rule. The rule's own
``utils.game_lock.lock_frame`` refuses repeated ids and null kickoffs; a supplied series
skips that frame, so the same refusals are made here, once, for both modules. Each one
closes a silent leak: a repeated id can hand a game a neighbour's later lock, and a null
instant sorts after every game end, so the game reads its whole season.

ASCII only, no emoji.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _first_ten(ids: pd.Index) -> list[str]:
    return [str(game_id) for game_id in ids[:10]]


def check_supplied_locks(locks: pd.Series, game_ids: Iterable[object]) -> None:
    """Raise ``ValueError`` unless ``locks`` gives every one of ``game_ids`` one instant.

    Refused, each message naming the count and the first ten ids: an id of ``game_ids``
    absent from ``locks.index``, an id named more than once in ``locks.index``, and a null
    instant anywhere in ``locks``.
    """
    wanted = pd.Index(list(game_ids))
    named = pd.Index(locks.index)
    missing = pd.Index(wanted[~wanted.isin(named)])
    if len(missing):
        raise ValueError(
            f"locks must name every game; {len(missing)} missing, "
            f"first ten: {_first_ten(missing)}"
        )
    repeated = pd.Index(named[named.duplicated()]).unique()
    if len(repeated):
        raise ValueError(
            f"locks name {len(repeated)} game id(s) more than once, "
            f"first ten: {_first_ten(repeated)}"
        )
    empty = pd.Index(named[locks.isna().to_numpy()])
    if len(empty):
        raise ValueError(
            f"locks hold no instant for {len(empty)} game(s), "
            f"first ten: {_first_ten(empty)}"
        )
