"""The one switch that sends forward rows to the ledger (Phase 34, D-14 / D-15).

``FORWARD_ROWS_GO_TO_LEDGER`` is an explicit committed constant, OFF until go-live. It is flipped
ONLY by Plan 34-19, in a commit of its own, in the same window as the migration of the old
writer's forward rows. Before that commit the daily run writes forward rows exactly as it does
today, so ledger wiring can land and be tested days before go-live without moving a single
forward row early.

WHY A CONSTANT AND NOT A PROBE: the 17:00 ET daily task runs this working tree. A "does the ledger
file exist?" probe would silently fall back to the old writer if ``ledger/`` were deleted; a
constant cannot.

Every caller reads it through :func:`forward_rows_go_to_ledger`, the ONE read point (tests
monkeypatch the constant or the accessor).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

__all__ = ["FORWARD_ROWS_GO_TO_LEDGER", "forward_rows_go_to_ledger"]

FORWARD_ROWS_GO_TO_LEDGER: bool = False


def forward_rows_go_to_ledger() -> bool:
    """Whether forward rows are written to the ledger. Reads the constant at call time."""
    return FORWARD_ROWS_GO_TO_LEDGER
