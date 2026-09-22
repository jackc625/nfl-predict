"""A per-row lock for tests of ``expanding_normalize`` that are NOT about lock ordering.

p332_ extra step 8c (owner ruling 2026-09-22) made the lock the window every expanding
statistic is taken over, and ``expanding_normalize`` now refuses a caller that cannot
supply one -- week order is the leak, not a safe fallback.

Most tests of that function predate the lock and assert something else entirely: the
prior-season bootstrap, the neutral-0.0 fallback, ``preserve_missing_cols``,
``preserve_level_cols``, the season reset, the output shape. Their intent is unchanged by
step 8c, so they are given a lock that REPRODUCES the window they already assumed: one
distinct, strictly increasing instant per row, in the same order ``expanding_normalize``
itself sorts the frame into. Every such test then asserts exactly what it asserted before.

Do NOT use this helper to test the ordering itself. One lock per row means no two games
ever share a statistic, which is the opposite of the rule step 8c introduced
(``tests/unit/test_normalization_lock_order.py`` builds real tie groups for that).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: An arbitrary epoch, far from any sentinel, in the UTC nanoseconds production uses.
_BASE_NS: int = 1_600_000_000_000_000_000

#: One day between consecutive rows -- wide enough that no two are ever equal.
_STEP_NS: int = 86_400_000_000_000


def row_order_locks(
    frame: pd.DataFrame,
    sort_cols: Sequence[str] | None = None,
) -> pd.Series:
    """One distinct lock per row of *frame*, increasing in ``sort_cols`` order.

    Args:
        frame: The frame about to be normalized.
        sort_cols: The columns ``expanding_normalize`` will sort by. Defaults to
            ``["season", "week"]``, that function's own default.

    Returns:
        A ``Series`` of int64 nanosecond locks, indexed like *frame*.
    """
    columns = list(sort_cols) if sort_cols is not None else ["season", "week"]
    present = [column for column in columns if column in frame.columns]
    ordered_index = frame.sort_values(present).index if present else frame.index
    locks = {
        label: _BASE_NS + position * _STEP_NS
        for position, label in enumerate(ordered_index)
    }
    return pd.Series(
        [locks[label] for label in frame.index],
        index=frame.index,
        name="lock_ns",
        dtype="int64",
    )
