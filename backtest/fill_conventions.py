"""The pre-registered fill convention ``fill-v1`` (Phase 34, LDGR-11).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/ev_chain_constants.py``. This module IS the
pre-registration of the pricing and sizing rule every 2026 forward ledger row is stamped with, not
a description of one. It records the rule EXACTLY AS IT RUNS TODAY (owner ruling, SPEC interview
round 2: "fill-v1 = today's rule unchanged"), stated completely: the SPEC's "consensus then
draftkings" wording omits the live fallback, so the full book ordering is recorded below
(34-RESEARCH.md Open Question 1 and Pitfall 10).

Stated plainly because it is easy to forget later in the season: EDITING THIS FILE AFTER WEEK W'S
FIRST LOCK DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A change to the pricing or sizing rule
is a NEW convention id with its own record, registered beside this one; it is never an edit to
``fill-v1``. A value that is wrong here is wrong for every row stamped with it, and the only
honest response is to say so in the readout.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH. A file that must contain its own whole-file hash
has no fixed point. The witness lives OUTSIDE it: ``tests/phase34_state.py`` records this file's
commit and normalized sha256 in a LATER commit under its APPEND PROTOCOL, and the Phase-34 ancestry
test recomputes and compares (Plan 34-22).

Constants only: NO project imports, NO I/O and NO logic, so the record cannot change meaning when
the code around it moves. ``tests/unit/test_fill_conventions.py`` pins every value below to the
live constant the selector actually uses, so the live rule cannot drift from this record without a
failing test.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

__all__ = [
    "BOOK_ORDERING",
    "BOOK_PREFERENCE",
    "FILL_CONVENTION_ID",
    "KELLY_CONFIDENCE_THRESHOLD",
    "KELLY_FRACTION",
    "MAX_BET_FRACTION",
    "MONEYLINE_PRICE_RULE",
    "NOTIONAL_BANKROLL",
    "REGISTERED_FILL_CONVENTIONS",
    "SPREAD_TOTAL_SLIPPAGE_POINTS",
    "UNIT_FRACTION_OF_BANKROLL",
]


# The id every ledger row stamps in its fill-convention column. A row without it is refused at
# write (LDGR-11).
FILL_CONVENTION_ID: str = "fill-v1"


# ---------------------------------------------------------------------------
# Which book's line and price a row is struck at
# ---------------------------------------------------------------------------

# The named book preference, in order. Live equivalent:
# backtest.ou_divergence.SPORTSBOOK_PREFERENCE.
BOOK_PREFERENCE: tuple[str, ...] = ("consensus", "draftkings")

# The FULL ordering, most significant step first, exactly as
# backtest.ou_divergence.order_by_book_preference and dedupe_odds_by_book_preference run it.
# Live 2026 captures never carry ``consensus``, so in practice the rule reads "DraftKings, else the
# alphabetically first live book at the latest capture instant".
BOOK_ORDERING: tuple[str, ...] = (
    "Decision reading: only captures admissible at the game's lock are eligible (recorded capture "
    "instant at or before the lock); a game with no admissible capture is never priced from a "
    "later one.",
    "Then the latest recorded capture instant (created_at) first; a capture with no recorded "
    "instant ranks last.",
    "Then the BOOK_PREFERENCE rank: consensus, then draftkings; any other book ranks after both "
    "and is never dropped.",
    "Then the book name, ascending.",
    "Then the later recorded capture instant.",
    "Closing reading: the same steps, applied to the captures inside the closing window instead "
    "of the captures admissible at the lock.",
)


# ---------------------------------------------------------------------------
# The price a row is struck at
# ---------------------------------------------------------------------------

# Half a point against the bettor on every spread and total bet. Live equivalent:
# backtest.simulation.SLIPPAGE_POINTS.
SPREAD_TOTAL_SLIPPAGE_POINTS: float = 0.5

MONEYLINE_PRICE_RULE: str = (
    "Moneyline (win) bets are struck at the selected book's posted American price for the side "
    "bet, with no slippage and no price adjustment."
)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

# Quarter-Kelly. Live equivalent: BetSelector's KellyCalculator(default_kelly_fraction=...).
KELLY_FRACTION: float = 0.25

# The per-bet stake cap as a fraction of the bankroll. Live equivalent:
# BetSelector's KellyCalculator(max_bet_pct=...).
MAX_BET_FRACTION: float = 0.05

# One unit is this fraction of the notional bankroll. Live equivalents:
# backtest.weekly_bet_list.UNIT_FRACTION_OF_BANKROLL and BetSelector's
# KellyCalculator(base_unit_size=bankroll * ...).
UNIT_FRACTION_OF_BANKROLL: float = 0.01

# The notional bankroll stakes are sized against. Live equivalent:
# backtest.weekly_bet_list.DEFAULT_BANKROLL.
NOTIONAL_BANKROLL: float = 10_000.0

# Kelly sizing applies no edge threshold of its own: the EV chain owns admission. Live equivalent:
# BetSelector's KellyCalculator(confidence_threshold=...).
KELLY_CONFIDENCE_THRESHOLD: float = 0.0


# Every fill convention id a ledger row may carry. A later convention is APPENDED here in its own
# record; this tuple never loses an entry.
REGISTERED_FILL_CONVENTIONS: tuple[str, ...] = (FILL_CONVENTION_ID,)
