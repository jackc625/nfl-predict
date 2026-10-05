"""The pre-registered scope of the 2026 forward verdict (Phase 34, Plan 34-22; LDGR-10, D-06, D-15).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/ev_chain_constants.py`` and
``backtest/fill_conventions.py``. This module IS the pre-registration of which forward bets the
2026 profitability verdict counts and how each one's outcome is read, not a description of one.
It is committed ALONE, before week ``VERDICT_START_WEEK``'s first lock (18:00 ET on the day before
that week's first game, ``utils.game_lock``), and pushed to GitHub's ``master`` before that week's
first decision run.

Stated plainly because it is easy to forget later in the season: EDITING THIS FILE AFTER WEEK
``VERDICT_START_WEEK``'S FIRST LOCK DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. Changing the
counted weeks, the counted arm or the outcome rule once counted bets exist is exactly the
after-the-fact rule choice this record exists to forbid. A value that is wrong here is wrong for
the whole 2026 verdict, and the only honest response is to say so in the readout.

HOW THE START WEEK WAS CHOSEN (D-15). It was COMPUTED, not picked:
``forward_ledger.verdict.compute_start_week`` over the live ledger and the recorded 2026 schedule
returns the first week after every week that already holds a ledger row (weeks 3 and 4 hold the
87 migrated old-writer rows, all ``pre_verdict`` with NULL stamps) whose first decision run is at
least ``DECLARATION_SAFETY_MARGIN`` in the future. Every row of an earlier week stays
``pre_verdict`` and is outside every verdict figure.

WHAT IS DELIBERATELY NOT PINNED HERE. No recipe set: a counted row's recipe only has to resolve in
the committed ``backtest.recipe_registry``, because Phase 37 registers new recipes mid-season by
their own pre-registration (RFIT-01). The pricing and sizing rule is named by id
(``FILL_CONVENTION_ID``); its content is ``backtest/fill_conventions.py``.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH. A file that must contain its own whole-file hash
has no fixed point. The witness lives OUTSIDE it: ``tests/phase34_state.py`` records this file's
commit and newline-normalized sha256 in a LATER commit under its APPEND PROTOCOL, and
``tests/unit/test_phase34_preregistration_ancestry.py`` recomputes and compares.

Constants only: NO project imports, NO I/O and NO logic. ``forward_ledger.declarations.
load_verdict_scope`` reads exactly the nine constants below and refuses the module by name if one
is missing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

__all__ = [
    "BOOTSTRAP_REGIME_WEEKS",
    "COUNTED_ARM",
    "FILL_CONVENTION_ID",
    "INCLUDES_NEUTRAL_SITE_GAMES",
    "INCLUDES_PLAYOFF_WEEKS",
    "OUTCOME_RULE",
    "VERDICT_END_WEEK",
    "VERDICT_SEASON",
    "VERDICT_START_WEEK",
]


# ---------------------------------------------------------------------------
# Which weeks count
# ---------------------------------------------------------------------------

# The season the verdict judges.
VERDICT_SEASON: int = 2026

# W: the first counted week, computed at go-live from the ledger's own contents (D-15).
VERDICT_START_WEEK: int = 5

# The last counted week, INCLUSIVE: week 22 is the Super Bowl. There is no week 23.
VERDICT_END_WEEK: int = 22

# The playoff weeks (19 wild card, 20 divisional, 21 conference, 22 Super Bowl) COUNT (D40-08).
INCLUDES_PLAYOFF_WEEKS: bool = True

# Neutral-site and international games COUNT, like any other game of a counted week (D40-08).
INCLUDES_NEUTRAL_SITE_GAMES: bool = True

# Weeks 2-4 were decided under the cold-start bootstrap regime and carry
# ``regime_label = bootstrap_regime`` (D40-08). All three are before VERDICT_START_WEEK, so none of
# them is counted; the tuple is recorded so the regime boundary is part of the registered record.
BOOTSTRAP_REGIME_WEEKS: tuple[int, ...] = (2, 3, 4)


# ---------------------------------------------------------------------------
# Which bets count, and how each one's result is read
# ---------------------------------------------------------------------------

# The verdict is the LIVE arm's: the bets the system actually recommended. A shadow arm (Phase 37)
# is a comparison and is never pooled into the verdict; deciding that after shadow rows exist
# would itself be an after-the-fact rule.
COUNTED_ARM: str = "live"

# D-06, in plain words: each counted bet is graded on the outcome IN FORCE -- if the official score
# is corrected after the bet was first graded, the verdict counts the CORRECTED result (what a real
# sportsbook would have paid on the true official score), taken from the latest correction entry
# for that bet; a bet with no correction counts its own grade. The original grade is never
# overwritten: it stays visible beside the corrected one, and the bet is marked as corrected.
OUTCOME_RULE: str = "in_force_correction"

# The pricing and sizing rule every counted row is stamped with (LDGR-11): the pre-registered
# ``fill-v1`` in ``backtest/fill_conventions.py``.
FILL_CONVENTION_ID: str = "fill-v1"
