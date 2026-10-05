"""fill-v1 equals the rule that actually runs (Phase 34, LDGR-11).

``backtest/fill_conventions.py`` is a pre-registration: it records the pricing and sizing rule
every 2026 ledger row is stamped with. A record is only worth something while it equals the live
rule, so each recorded value is compared here with the LIVE constant the selector uses -- never with
a literal restated in this file. Changing any live constant without registering a new convention id
therefore fails this module (SPEC LDGR-11 acceptance; edge "ordering": the recorded book preference
equals the live ORDERED tuple).

The Kelly arguments are literals inside ``BetSelector.__init__``, not module constants, so they are
read back from the selector ``select_weekly_bets`` actually constructs.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

from backtest import fill_conventions as record
from backtest import weekly_bet_list
from backtest.ou_divergence import (
    SPORTSBOOK_PREFERENCE,
    dedupe_odds_by_book_preference,
    order_by_book_preference,
)
from backtest.simulation import SLIPPAGE_POINTS
from tests.fixtures.decision_frame import (
    fits_with_floor,
    mini_strategies,
    week_candidates,
    week_schedule,
)

_RECORD_PATH = Path(record.__file__)


def test_the_record_is_fill_v1_and_registered() -> None:
    assert record.FILL_CONVENTION_ID == "fill-v1"
    assert record.REGISTERED_FILL_CONVENTIONS == (record.FILL_CONVENTION_ID,)


def test_book_preference_equals_live_tuple() -> None:
    """Ordered equality: swapping the two books is a different rule."""
    assert record.BOOK_PREFERENCE == SPORTSBOOK_PREFERENCE


def test_slippage_equals_live() -> None:
    assert record.SPREAD_TOTAL_SLIPPAGE_POINTS == SLIPPAGE_POINTS == 0.5


def test_kelly_arguments_equal_live_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    """The KellyCalculator the weekly path really builds carries exactly the recorded values."""
    built: list[weekly_bet_list.BetSelector] = []

    class _RecordingSelector(weekly_bet_list.BetSelector):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            built.append(self)

    monkeypatch.setattr(weekly_bet_list, "BetSelector", _RecordingSelector)
    weekly_bet_list.select_weekly_bets(
        week_candidates(),
        week_schedule(),
        fits_with_floor(0.0),
        strategies=mini_strategies(),
    )

    assert len(built) == 1, (
        "select_weekly_bets no longer builds exactly one BetSelector"
    )
    kelly = built[0]._kelly
    assert kelly.default_kelly_fraction == record.KELLY_FRACTION
    assert kelly.max_bet_pct == record.MAX_BET_FRACTION
    assert kelly.base_unit_size == (
        record.NOTIONAL_BANKROLL * record.UNIT_FRACTION_OF_BANKROLL
    )
    assert kelly.starting_bankroll == record.NOTIONAL_BANKROLL
    assert kelly.confidence_threshold == record.KELLY_CONFIDENCE_THRESHOLD


def test_bankroll_and_unit_equal_live() -> None:
    assert record.NOTIONAL_BANKROLL == weekly_bet_list.DEFAULT_BANKROLL
    assert record.UNIT_FRACTION_OF_BANKROLL == weekly_bet_list.UNIT_FRACTION_OF_BANKROLL


_GAME = "2026_W06_DAL@PHI"
_LOCK = pd.Timestamp("2026-10-10T22:00:00Z")


def _capture(book: str, created_at: str | None, label: str) -> dict[str, object]:
    return {
        "game_id": _GAME,
        "sportsbook": book,
        "created_at": pd.Timestamp(created_at) if created_at else pd.NaT,
        "label": label,
    }


def test_recorded_ordering_matches_helper() -> None:
    """Each recorded step decides the pair it names, in the order the record states them."""
    later, earlier = "2026-10-10T21:00:00Z", "2026-10-10T20:00:00Z"
    stored = pd.DataFrame(
        [
            _capture("bovada", earlier, "e"),
            _capture("consensus", None, "f"),
            _capture("betmgm", earlier, "d"),
            _capture("draftkings", earlier, "c"),
            _capture("consensus", earlier, "b"),
            _capture("fanduel", later, "a"),
        ]
    )

    ordered = list(order_by_book_preference(stored)["label"])

    # Step: latest recorded capture instant first -- "a" (a non-preferred book) leads, and "f"
    # (no recorded instant) ranks last.
    # Step: BOOK_PREFERENCE rank -- consensus "b", then draftkings "c", then every other book.
    # Step: book name ascending -- betmgm "d" before bovada "e".
    assert ordered == ["a", "b", "c", "d", "e", "f"]

    steps = record.BOOK_ORDERING
    assert "admissible" in steps[0] and "lock" in steps[0]
    assert "latest recorded capture instant" in steps[1]
    assert "BOOK_PREFERENCE" in steps[2]
    assert "book name, ascending" in steps[3]
    assert "later recorded capture instant" in steps[4]
    assert steps[5].startswith("Closing reading: the same steps")

    # Step 0, decision reading: admissibility at the lock comes before every other step, so a
    # post-lock capture never outranks an admissible one.
    post_lock = pd.DataFrame(
        [
            _capture("draftkings", "2026-10-10T21:00:00Z", "admissible"),
            _capture("consensus", "2026-10-10T23:00:00Z", "after-lock"),
        ]
    )
    chosen = dedupe_odds_by_book_preference(post_lock, locks={_GAME: _LOCK})
    assert list(chosen["label"]) == ["admissible"]


def test_record_has_no_project_imports() -> None:
    """A record that imports project code could change meaning when that code moves."""
    tree = ast.parse(_RECORD_PATH.read_text(encoding="ascii"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in the fill-v1 record"
            imported.add((node.module or "").split(".")[0])

    allowed = {"__future__", *sys.stdlib_module_names}
    assert imported <= allowed, f"non-stdlib imports: {sorted(imported - allowed)}"
