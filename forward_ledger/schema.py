"""The forward ledger's row key and its view of the bet-list schema (LDGR-02, LDGR-04).

ONE SCHEMA, NOT TWO
-------------------
``api.cache`` is the single source of the bet-list schema: its four column classes, their order and
the closed vocabularies. This module RE-EXPORTS them for the ledger code rather than restating
them, and ``api/`` never imports this package (D-18) -- the dependency runs one way.

THE ROW KEY
-----------
``LEDGER_ROW_KEY = (game_id, season, week, target, arm)`` identifies a forward ledger row, defined
here ONCE. ``arm`` is in the key so a ``live`` and a ``shadow`` row for the same game, week and
target are two rows, never one silently overwriting the other (34-RESEARCH Pitfall 7). Every key
derivation over forward rows -- the store, first-pick-stands, graded weeks, corrections, the cache
join -- uses this tuple. ``backtest.weekly_bet_list.BET_LIST_ROW_KEY`` stays the four-tuple for the
regenerable REPLAY store in ``outputs/bet_list/``, whose rows have no arm: a different store.

CHECKED AT IMPORT
-----------------
Two facts are asserted the moment this module loads, so a drift cannot ship silently:

  * the cache's immutable column list IS the chain's frozen ``IMMUTABLE_COLUMNS_V1``, in order --
    a column added to one and not the other would make the chain hash a different set of facts
    from the ones the ledger stores as immutable;
  * the immutable, grading, fill and closing classes are pairwise disjoint and their concatenation
    IS ``BET_LIST_COLUMNS`` (LDGR-04): no column belongs to two write paths.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from api.cache import (
    ARM_LIVE,
    ARM_SHADOW,
    ARMS,
    BET_LIST_CLOSING_COLUMNS,
    BET_LIST_COLUMNS,
    BET_LIST_FILL_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    CLOSING_NULL_REASONS,
    REGIME_LABEL_BOOTSTRAP,
    VERDICT_SCOPE_PRE_VERDICT,
    VERDICT_SCOPE_VERDICT,
    VERDICT_SCOPES,
)
from forward_ledger.canonical import IMMUTABLE_COLUMNS_V1

__all__ = [
    "ARMS",
    "ARM_LIVE",
    "ARM_SHADOW",
    "BET_LIST_CLOSING_COLUMNS",
    "BET_LIST_COLUMNS",
    "BET_LIST_FILL_COLUMNS",
    "BET_LIST_GRADING_COLUMNS",
    "BET_LIST_IMMUTABLE_COLUMNS",
    "CLOSING_NULL_REASONS",
    "LEDGER_ROW_KEY",
    "REGIME_LABEL_BOOTSTRAP",
    "VERDICT_SCOPES",
    "VERDICT_SCOPE_PRE_VERDICT",
    "VERDICT_SCOPE_VERDICT",
]

LEDGER_ROW_KEY: tuple[str, ...] = ("game_id", "season", "week", "target", "arm")


if tuple(BET_LIST_IMMUTABLE_COLUMNS) != IMMUTABLE_COLUMNS_V1:  # pragma: no cover
    msg = (
        "api.cache.BET_LIST_IMMUTABLE_COLUMNS no longer equals "
        "forward_ledger.canonical.IMMUTABLE_COLUMNS_V1; a schema change to the immutable half "
        "needs a new canonical version, never an edit to the frozen v1 list"
    )
    raise RuntimeError(msg)

_CLASS_CONCATENATION: list[str] = [
    *BET_LIST_IMMUTABLE_COLUMNS,
    *BET_LIST_GRADING_COLUMNS,
    *BET_LIST_FILL_COLUMNS,
    *BET_LIST_CLOSING_COLUMNS,
]
if (
    len(set(_CLASS_CONCATENATION)) != len(_CLASS_CONCATENATION)
    or _CLASS_CONCATENATION != BET_LIST_COLUMNS
):  # pragma: no cover
    msg = (
        "the immutable / grading / fill / closing column classes are not pairwise disjoint, or "
        "their concatenation is not api.cache.BET_LIST_COLUMNS"
    )
    raise RuntimeError(msg)
