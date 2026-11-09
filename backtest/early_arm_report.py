"""The early-week arm's record: the figures the /bets page shows beside the arm's own bets.

Worked out at cache build and stored as one value, so the page computes nothing (UIAP-01). The
function is pure: a frame and the in-force corrections in, a plain dictionary out.

WHAT IT COUNTS. Only the placed bets of the arm ``early`` (``backtest/early_arm_2026.py``). The
retired day-before record in ``backtest/bet_tracker.py`` drops those same rows, so a bet is in one
record or the other and never both. A correction changes this record exactly as it changes that
one, through the tracker's own overlay.

CLOSING-LINE VALUE IS ``forward_clv``. It is written once the game is played, and it is the bet's
line against the same book's close, in points, positive when the bet beat the close. The older
``clv`` column is a decision-time figure and is always empty on an early row. ``closing_share`` is
the share of the SETTLED bets whose close was read: a game still to be played has no close to miss,
so counting it would make every week in progress look like a capture failure.

NOTHING BEHIND A FIGURE IS NONE. A return with no stake, a mean of no readings and a share of no
settled bets are all None, never 0.0, which would read as a measured break-even. A count of zero
is a real count and stays 0.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from api.cache import (
    ARM_EARLY,
    BET_STATUS_LIVE,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
)
from backtest.bet_tracker import _flat_return_units, _overlay_in_force_corrections

__all__ = ["early_arm_summary"]

_SETTLED_STATUSES: tuple[str, ...] = (
    GRADING_STATUS_WIN,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
)


def _early_bets(bet_list: pd.DataFrame) -> pd.DataFrame:
    """A copy of the early arm's placed bets. A frame with no ``arm`` column has none."""
    if "arm" not in bet_list.columns:
        return pd.DataFrame()
    placed = (bet_list["arm"] == ARM_EARLY) & (bet_list["status"] == BET_STATUS_LIVE)
    return cast(pd.DataFrame, bet_list.loc[placed].copy())


def early_arm_summary(
    bet_list: pd.DataFrame, corrections: pd.DataFrame | None
) -> dict[str, Any]:
    """The early arm's record over every week in *bet_list*.

    Args:
        bet_list: Bet-list rows of any arm, in ``BET_LIST_COLUMNS``.
        corrections: The in-force corrections (``BET_LIST_CORRECTIONS_COLUMNS``), or None.

    Returns:
        ``bets``, ``graded``, ``wins``, ``losses``, ``pushes``, ``closing_bets`` (counts) and
        ``flat_return``, ``mean_clv``, ``closing_share`` (a float, or None with nothing behind it).
        Plain Python numbers only, because the value is stored as JSON.

    Raises:
        KeyError: A correction cannot be matched to its row (see ``prepare_tracker_rows``).
        ValueError: *corrections* holds two rows for one ledger key.
    """
    early = _early_bets(bet_list)
    if early.empty:
        rows, graded = early, early
        closing = pd.Series(dtype="float64")
    else:
        rows = _overlay_in_force_corrections(early, corrections)
        graded = cast(
            pd.DataFrame, rows[rows["grading_status"].isin(_SETTLED_STATUSES)]
        )
        closing = cast(
            pd.Series, pd.to_numeric(rows["forward_clv"], errors="coerce")
        ).dropna()

    def count(status: str) -> int:
        return 0 if graded.empty else int((graded["grading_status"] == status).sum())

    return {
        "bets": len(rows),
        "graded": len(graded),
        "wins": count(GRADING_STATUS_WIN),
        "losses": count(GRADING_STATUS_LOSS),
        "pushes": count(GRADING_STATUS_PUSH),
        "flat_return": None if graded.empty else _flat_return_units(graded),
        "closing_bets": len(closing),
        "mean_clv": None if closing.empty else float(closing.mean()),
        "closing_share": None if graded.empty else len(closing) / len(graded),
    }
