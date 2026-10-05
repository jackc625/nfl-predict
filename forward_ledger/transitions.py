"""The one-way transitions of a ledger row's three mutable halves (Phase 34, LDGR-04, LDGR-07).

A ledger row has one immutable half (covered by the hash chain) and three mutable halves, each
written ONLY through its own path and only one way:

  * GRADING -- out of ``pending`` into a terminal status, through the existing
    ``api.cache.assert_grading_transition`` (the one-way grader stays one-way);
  * FILL -- what was actually bet (D-17): each column moves from NULL to a value ONCE. A second
    write touching a column that already holds a value is refused, even with the same value -- a
    real fill is recorded once, never restated;
  * CLOSING -- the near-kickoff closing line (LDGR-07): set ONCE, either as values with no NULL
    reason or as every value NULL with a reason from ``CLOSING_NULL_REASONS`` (NULL is never
    silent). Re-applying the identical closing half is a no-op, so a re-run of finalization is
    safe; anything else is refused, so a set value is never overwritten and a recorded reason never
    later gains values.

:func:`assert_update_confined` keeps each update inside its own column set: the grading path
cannot write a fill or closing column, the fill path cannot write an immutable or grading column.
No transition here touches the immutable half or a chain hash; ``forward_ledger.store`` applies
them inside its one write path.

The named refusals inherit bare ``Exception`` (the ``data.graded_weeks`` rule): a refused write
must never be swallowed by a broad ``ValueError`` handler into "nothing changed".

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from forward_ledger.schema import (
    BET_LIST_CLOSING_COLUMNS,
    BET_LIST_FILL_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    CLOSING_NULL_REASONS,
)

__all__ = [
    "CLOSING_SET",
    "FILL_SET",
    "GRADING_SET",
    "UPDATE_KIND_CLOSING",
    "UPDATE_KIND_FILL",
    "UPDATE_KIND_GRADING",
    "ClosingAlreadySetError",
    "ColumnSetViolationError",
    "FillAlreadyRecordedError",
    "assert_closing_transition",
    "assert_fill_transition",
    "assert_update_confined",
]

GRADING_SET: frozenset[str] = frozenset(BET_LIST_GRADING_COLUMNS)
FILL_SET: frozenset[str] = frozenset(BET_LIST_FILL_COLUMNS)
CLOSING_SET: frozenset[str] = frozenset(BET_LIST_CLOSING_COLUMNS)

UPDATE_KIND_GRADING = "grading"
UPDATE_KIND_FILL = "fill"
UPDATE_KIND_CLOSING = "closing"

_SET_BY_KIND: dict[str, frozenset[str]] = {
    UPDATE_KIND_GRADING: GRADING_SET,
    UPDATE_KIND_FILL: FILL_SET,
    UPDATE_KIND_CLOSING: CLOSING_SET,
}

_CLOSING_REASON_COLUMN = "closing_null_reason"
_CLOSING_VALUE_COLUMNS: tuple[str, ...] = tuple(
    name for name in BET_LIST_CLOSING_COLUMNS if name != _CLOSING_REASON_COLUMN
)


class ColumnSetViolationError(Exception):
    """An update names a column outside its own mutability class (LDGR-04)."""


class FillAlreadyRecordedError(Exception):
    """A fill update touches a fill column that already holds a value (NULL -> value once)."""


class ClosingAlreadySetError(Exception):
    """The closing half refuses this transition: it is already set, or the write is malformed."""


def assert_update_confined(update_kind: str, columns: Iterable[str]) -> None:
    """Refuse an update of *update_kind* that names any column outside its own set.

    Raises:
        ColumnSetViolationError: a column belongs to another class (named in the message).
        ValueError: *update_kind* is not grading, fill or closing.
    """
    if update_kind not in _SET_BY_KIND:
        msg = f"update kind {update_kind!r} is outside {tuple(_SET_BY_KIND)}"
        raise ValueError(msg)
    allowed = _SET_BY_KIND[update_kind]
    extra = sorted(set(columns) - allowed)
    if extra:
        msg = (
            f"a {update_kind} update may write only {sorted(allowed)}; it names {extra}, which "
            "belong to another write path"
        )
        raise ColumnSetViolationError(msg)


def assert_fill_transition(current: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    """Every fill column *new* names must still be NULL in *current*.

    Raises:
        ColumnSetViolationError: *new* names a non-fill column.
        FillAlreadyRecordedError: *new* touches a fill column that already holds a value.
    """
    assert_update_confined(UPDATE_KIND_FILL, new)
    recorded = sorted(name for name in new if current.get(name) is not None)
    if recorded:
        msg = (
            f"fill column(s) {recorded} already hold a recorded value; a real fill moves from "
            "NULL to a value once and is never restated"
        )
        raise FillAlreadyRecordedError(msg)


def assert_closing_transition(
    current: Mapping[str, Any], new: Mapping[str, Any]
) -> None:
    """The closing half is set once, as values or as a recorded NULL reason, never overwritten.

    An identical re-application (every named column already equal) is a no-op and passes.

    Raises:
        ColumnSetViolationError: *new* names a non-closing column.
        ClosingAlreadySetError: the half is already set to something else, or the write is
            neither "values with no reason" nor "every value NULL with a known reason".
    """
    assert_update_confined(UPDATE_KIND_CLOSING, new)
    before = {name: current.get(name) for name in BET_LIST_CLOSING_COLUMNS}
    after = {**before, **new}
    if after == before:
        return
    if any(value is not None for value in before.values()):
        msg = (
            f"the closing half is already set ({before}); a set closing value is never "
            "overwritten and a recorded NULL reason never later gains values"
        )
        raise ClosingAlreadySetError(msg)

    reason = after[_CLOSING_REASON_COLUMN]
    has_values = any(after[name] is not None for name in _CLOSING_VALUE_COLUMNS)
    if reason is None and has_values:
        return
    if reason in CLOSING_NULL_REASONS and not has_values:
        return
    msg = (
        f"a closing write must be values with no {_CLOSING_REASON_COLUMN}, or every value NULL "
        f"with a reason from {CLOSING_NULL_REASONS}; got {after}"
    )
    raise ClosingAlreadySetError(msg)
