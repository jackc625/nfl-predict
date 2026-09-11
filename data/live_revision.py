"""RED-phase skeleton for the live-capture revision ruling (Plan 32-07, Task 1).

The API surface only, so that ``tests/unit/test_revision_severity.py`` fails on the
ASSERTIONS it makes about the ruling rather than on a collection error. A nonzero exit
from a module that cannot be imported proves nothing about the behaviour under test.
The ruling itself lands in the GREEN commit.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "LIVE_REVISION_SCHEMA_VERSION",
    "WeekDiff",
    "as_record",
    "compare_week_digests",
]

LIVE_REVISION_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class WeekDiff:
    """What moved between two captures, week by week and column by column."""

    weeks_changed: tuple[str, ...] = ()
    weeks_added: tuple[str, ...] = ()
    weeks_removed: tuple[str, ...] = ()
    columns_changed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rows_changed: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def is_revision(self) -> bool:
        """Whether this difference is a REVISION rather than ordinary in-season growth."""
        raise NotImplementedError


def compare_week_digests(prior: dict, current: dict) -> WeekDiff:
    """Diff two ``week_digests`` maps. Not yet implemented (RED phase)."""
    return WeekDiff()


def as_record(diff: WeekDiff) -> dict:
    """Render *diff* in its JSON-serialisable form. Not yet implemented (RED phase)."""
    raise NotImplementedError
