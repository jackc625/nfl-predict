"""The key-number converter: win, push and lose chances from an expected closing line.

WHY IT EXISTS. NFL finals do not fall smoothly. A game ends on a margin of 3 or 7 far more often
than a normal curve says, and on a tie far less often, so a half point that crosses 3 is worth more
than a half point out at 12. This module carries that shape as a table of factors, one per whole
number, fitted once on past seasons by ``scripts/fit_key_numbers.py`` and committed.

WHAT A FACTOR IS. How often past games finished on a whole number, divided by how often a normal
curve around each game's closing line said they would. Both counts get 10 added before dividing,
so a number that was rarely reached keeps a factor close to 1 rather than a noisy one.

THE MARKET LINE IS TREATED AS FAIR. The factors would otherwise tilt a line toward whichever side
holds the bigger key number. So the side above ``market_line`` and the side below it are each
scaled until they carry equal mass when the game is centered on that line. Moving the center away
from the line then moves the chances away from even, and only that.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from backtest.early_arm_2026 import KEY_NUMBER_TABLES_PATH

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "KeyNumberTable",
    "OutcomeChance",
    "fit_key_number_table",
    "load_key_number_tables",
    "outcome_chance",
]

# Added to both the observed and the expected count of every whole number before dividing.
_SMOOTHING_GAMES: float = 10.0

# The targets the committed file holds, by the key each table is stored under.
_TABLE_KEYS: tuple[str, ...] = ("ats", "ou")


@dataclass(frozen=True)
class KeyNumberTable:
    """One target's fitted shape.

    Attributes:
        sd: The sample standard deviation of finals around the closing line.
        low: The whole number that ``factors[0]`` belongs to.
        factors: One factor per consecutive whole number, starting at ``low``.
    """

    sd: float
    low: int
    factors: tuple[float, ...]


@dataclass(frozen=True)
class OutcomeChance:
    """The chance a game finishes above, exactly on, or below a threshold. Sums to 1."""

    above: float
    at: float
    below: float


def _normal_mass(
    center: float | NDArray[np.float64], sd: float, outcomes: NDArray[np.int64]
) -> NDArray[np.float64]:
    """The normal curve's mass on each whole number in ``outcomes``."""
    upper = norm.cdf((outcomes + 0.5 - center) / sd)
    lower = norm.cdf((outcomes - 0.5 - center) / sd)
    return upper - lower


def fit_key_number_table(
    actual: ArrayLike,
    closing: ArrayLike,
    *,
    low: int,
    high: int,
    mirror: bool,
) -> KeyNumberTable:
    """Fit one target's table from past finals and their closing lines.

    Args:
        actual: Each game's final margin or total.
        closing: Each game's closing line, on the same scale as ``actual``.
        low: The lowest whole number the table covers.
        high: The highest whole number the table covers.
        mirror: Pool each number with its negative, for a target whose two directions are
            the same thing (a margin of +3 and -3). A pooled number pairs with its reverse,
            so 0 is counted from both directions, in both counts, like every other number.

    Returns:
        The fitted table.

    Raises:
        ValueError: When the inputs differ in length, or when ``mirror`` is asked for on a
            range not centered on zero, where each number's reverse is not its negative.
    """
    actual_values = np.asarray(actual, dtype=float)
    closing_values = np.asarray(closing, dtype=float)
    if actual_values.shape != closing_values.shape:
        msg = (
            f"actual {actual_values.shape} and closing {closing_values.shape} must be "
            "the same length"
        )
        raise ValueError(msg)
    if mirror and low != -high:
        msg = f"a mirror fit needs a range centered on zero, not [{low}, {high}]"
        raise ValueError(msg)

    sd = float(np.std(actual_values - closing_values, ddof=1))
    outcomes = np.arange(low, high + 1)
    expected = _normal_mass(closing_values[:, np.newaxis], sd, outcomes).sum(axis=0)
    finals = np.rint(actual_values)
    observed = np.array([np.count_nonzero(finals == k) for k in outcomes], dtype=float)

    if mirror:
        observed = observed + observed[::-1]
        expected = expected + expected[::-1]

    factors = (observed + _SMOOTHING_GAMES) / (expected + _SMOOTHING_GAMES)
    return KeyNumberTable(sd=sd, low=low, factors=tuple(factors.tolist()))


def outcome_chance(
    table: KeyNumberTable,
    *,
    market_line: float,
    center: float,
    threshold: float,
) -> OutcomeChance:
    """The chance a game centered on ``center`` finishes above, on or below ``threshold``.

    Args:
        table: The target's fitted table.
        market_line: The line treated as fair, which sets how the two sides are scaled.
        center: Where the game is expected to be centered.
        threshold: The number the result is compared with, usually a quoted line.

    Returns:
        The three chances.

    Raises:
        ValueError: When ``market_line`` is not strictly inside the table. One side would
            then hold no numbers at all, and the chances would come back as NaN.
    """
    factors = np.asarray(table.factors, dtype=float)
    outcomes = np.arange(table.low, table.low + factors.size)
    if not outcomes[0] < market_line < outcomes[-1]:
        msg = (
            f"market_line {market_line} is not strictly inside the table's range "
            f"[{outcomes[0]}, {outcomes[-1]}]"
        )
        raise ValueError(msg)

    at_line = _normal_mass(market_line, table.sd, outcomes) * factors
    is_above = outcomes > market_line
    is_below = outcomes < market_line
    mass_above = at_line[is_above].sum()
    mass_below = at_line[is_below].sum()
    # The midpoint leaves the two sides' combined mass unchanged, so a whole-number line
    # keeps its share of pushes.
    mass_per_side = (mass_above + mass_below) / 2.0
    scale = np.ones(factors.size)
    scale[is_above] = mass_per_side / mass_above
    scale[is_below] = mass_per_side / mass_below

    mass = _normal_mass(center, table.sd, outcomes) * factors * scale
    mass /= mass.sum()
    return OutcomeChance(
        above=float(mass[outcomes > threshold].sum()),
        at=float(mass[outcomes == threshold].sum()),
        below=float(mass[outcomes < threshold].sum()),
    )


def load_key_number_tables(
    path: Path | str = KEY_NUMBER_TABLES_PATH,
) -> dict[str, KeyNumberTable]:
    """Read the committed tables, keyed ``ats`` and ``ou``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key: KeyNumberTable(
            sd=float(payload[key]["sd"]),
            low=int(payload[key]["low"]),
            factors=tuple(float(factor) for factor in payload[key]["factors"]),
        )
        for key in _TABLE_KEYS
    }
