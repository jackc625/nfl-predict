"""The early-week arm's record: every number the arm runs on, from 2026 week 11.

WHAT THE ARM IS. On the Tuesday before each NFL week from 2026 week 11 (the first is Tue
2026-11-17), the daily run reads the recorded early-week board, scores every regular-season,
non-neutral game of the coming week with the installed v2 models, and prices one spread bet and
one over/under bet against the best quote at a regulated sportsbook. Each decision goes into the
forward ledger under the arm ``early``. It is paper only: no money is placed. Moneyline is out,
because it did not pass Study 1b. Every early row is ``pre_verdict``, so it can never count toward
the R6 money verdict.

WHAT THIS RECORD ANSWERS. Section 7 of ``MIDSEASON-SWITCH-RECORD.md`` says that record adds no
``fill-v2`` pricing convention and no ``early`` arm, and that each of those "is for a later record,
which will name what it changes". This is that later record, for those two things only. It adds
the arm ``early`` and the convention ``fill-v2``, and it has the ledger accept rows stamped with
the v2 recipe. Nothing else in that record changes: the day-before system stays retired from week
10, ``fill-v1`` stands for every row already stamped with it, the ``market`` arm is still not
built, and this record adds no new test, family or binding readout for 2026.

THE OWNER RULING, 2026-11-09 (OPTION B). The design's rule, the same-source slope plus 0.5 points
of slippage, placed 0 bets on 2,625 replayed game-bets, so the owner chose to loosen it. Early bets
are priced with Study 1b's CLOSE slopes (O/U 0.13624, ATS 0.03688) at the real best regulated-book
quote, with no slippage. On the 2020-2024 replay at a flat -110, that rule placed 62 O/U bets on
1,310 games (12.4 a season) with a flat return of +21.7%, and the line moved our way by 0.89 points.
It placed 0 ATS bets.

THOSE FIGURES ARE NOT EVIDENCE OF PROFIT. They are IN-SAMPLE: the slopes were measured on the same
2020-2024 games the replay then bet. And they MAY INCLUDE A SOURCE EFFECT: the close comes from a
different data source than the early line, and a consistent difference between two sources can
look like a line moving toward the model. The same-source O/U slope is only 0.0475. The owner
accepted this knowingly: paper only, and live same-book closing lines test it.

WHAT ``fill-v2`` MEANS. A row stamped ``fill-v2`` was priced at a real quote: the line and the
American price that one named regulated book (``REGULATED_BOOKS``) was offering when the board was
read, with no slippage added. It is registered HERE because ``backtest/fill_conventions.py`` is
frozen and is never edited; ``forward_ledger.declarations.resolve_fill_convention`` accepts that
file's conventions plus this one.

A QUOTE FAR FROM THE OTHERS IS NOT A PRICE. A quote more than ``MAX_QUOTE_DISTANCE_POINTS`` from
the median line of its game's usable quotes is dropped before the consensus is taken, so it can
neither move the consensus nor be the best price. This is a data-integrity rule, added 2026-11-09
by the controller under the plan's review focus 1 (a stale or outlying quote becoming the best
price). On the four recorded boards of 2026-11-05 to 2026-11-08, regulated books only, 99% of
quotes sat within 1.0 point of their game's median. The top of the tail was FanDuel's NE@DET at
-1.5 against about +1.5 on three straight days, which would have been priced as a large fake edge
into an append-only ledger.

THE CHOSEN BOOK IS ``eligibility_label``. Each early row stores the book it was priced at in that
ledger column, as ``book:<key>`` (``BOOK_LABEL_PREFIX``), and so does its decision snapshot. On
the other arms the column never names a book (it reads ``no_subpopulation`` or is NULL); it is
covered by the hash chain and already reaches the cache, while a new column would have needed a
new chain version.

Stated plainly because it is easy to forget later in the season: EDITING THIS FILE AFTER WEEK 11'S
FIRST DECISION DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A value that is wrong here is wrong
for every row decided under it, and the only honest repair is a NEW, visibly later record that
names this file's commit and says what it INVALIDATES. Never an edit in place, never a quiet
supersession.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH OR COMMIT. A file that must contain its own
whole-file hash has no fixed point. Every decision snapshot stores this file's sha256 instead, so a
later edit is visible. By owner ruling (no new witness or frozen test in 2026) no state module
witnesses it.

Constants only: standard-library imports only, NO project imports, NO I/O and NO logic. A value
that restates a fact other code also holds (the v2 recipe name in ``models/v2_recipe.py``, the
page's copy of the start week in ``api/cache_schema.py``) is COPIED here as a literal, never
imported, so this record cannot change meaning when the code around it moves;
``tests/unit/test_early_arm_record.py`` proves the copies agree.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "BOOK_LABEL_PREFIX",
    "EARLY_FROM",
    "EARLY_RECIPE",
    "EARLY_TARGETS",
    "EV_FLOOR",
    "EXPECTED_MOVE_SLOPE",
    "FILL_V2_ID",
    "KELLY_FRACTION",
    "KEY_NUMBER_FIT_SEASONS",
    "KEY_NUMBER_TABLES_PATH",
    "MAX_QUOTE_DISTANCE_POINTS",
    "MIN_BOOKS",
    "PER_GAME_CAP",
    "PER_WEEK_CAP",
    "QUOTE_MAX_AGE_HOURS",
    "RECORD_PATH",
    "REGULATED_BOOKS",
    "SLIPPAGE_POINTS",
]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# This file, from the repository root. Its sha256 goes into every decision snapshot.
RECORD_PATH: str = "backtest/early_arm_2026.py"

# The recipe every early row is stamped with: models/v2_recipe.V2_RECIPE_NAME, the recipe the owner
# install put in force (backtest/midseason_switch_2026.INSTALLED_RECIPE).
EARLY_RECIPE: str = "recipe-2026-v2"

# The pricing and sizing convention every early row is stamped with, as this record defines it.
FILL_V2_ID: str = "fill-v2"


# ---------------------------------------------------------------------------
# What the arm decides
# ---------------------------------------------------------------------------

# (season, week): the first NFL week the arm decides. Keyed on the NFL week, never the clock.
EARLY_FROM: tuple[int, int] = (2026, 11)

# The bet types the arm decides. Moneyline ("wp") did not pass Study 1b, so it has no early arm.
EARLY_TARGETS: tuple[str, ...] = ("ats", "ou")


# ---------------------------------------------------------------------------
# Which quotes may be the price
# ---------------------------------------------------------------------------

# The only books a price may be taken from, by their board key.
REGULATED_BOOKS: tuple[str, ...] = ("betmgm", "betrivers", "draftkings", "fanduel")

# A game's market with fewer usable quotes than this is no market at all.
MIN_BOOKS: int = 2

# A quote older than this when the board is read cannot be the best price: best-price selection
# actively picks outliers, and stale quotes are the likeliest outliers.
QUOTE_MAX_AGE_HOURS: int = 6

# A quote further than this from the median line of its game's usable quotes is dropped before the
# consensus is taken. A quote exactly this far away stays.
MAX_QUOTE_DISTANCE_POINTS: float = 1.5


# ---------------------------------------------------------------------------
# The price (owner ruling 2026-11-09, option B)
# ---------------------------------------------------------------------------

# Study 1b's slopes against the close: the share of the gap between the model and the early line
# that the close moved. The expected close is consensus + slope * (model - consensus).
EXPECTED_MOVE_SLOPE: Mapping[str, float] = MappingProxyType(
    {"ats": 0.03688, "ou": 0.13624}
)

# Points taken against the bettor. None: the price is a real quote at a named book.
SLIPPAGE_POINTS: float = 0.0

# A priced side is a bet only when its expected value per unit staked is strictly above this.
EV_FLOOR: float = 0.0


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

# Quarter-Kelly, with no other multiplier.
KELLY_FRACTION: float = 0.25

# Stake caps as fractions of the bankroll: per game first, then per NFL week, each scaling the
# stakes it covers proportionally.
PER_GAME_CAP: float = 0.05
PER_WEEK_CAP: float = 0.10


# ---------------------------------------------------------------------------
# The book's name on the row
# ---------------------------------------------------------------------------

# A row's eligibility_label is this prefix plus the chosen book's key, for example "book:fanduel".
BOOK_LABEL_PREFIX: str = "book:"


# ---------------------------------------------------------------------------
# The key-number converter
# ---------------------------------------------------------------------------

# The seasons (first, last, both included) the key-number tables are fitted on, once, and the
# committed file that holds the fit.
KEY_NUMBER_FIT_SEASONS: tuple[int, int] = (2015, 2025)
KEY_NUMBER_TABLES_PATH: str = "config/key_number_tables.json"
