"""ONE strict parse helper serves all three freeze comparisons, proven BY IDENTITY.

Phase 33, Plan 33-05 Task 1 (COLD-03, D33-27, T-33-23).

WHY IDENTITY AND NOT A SOURCE GREP
----------------------------------
A source scan can only show that a NAME appears in three files. It cannot show that the three
names resolve to the same object -- a second module defining its own
``require_aware_snapshot_ts`` would satisfy the grep and defeat the property entirely. So the
primary assertion here replaces ``scripts.ingest_historical_odds.require_aware_snapshot_ts``
with a COUNTING STUB and drives all three call sites: if any of them reached a different
object, its call would not land on the shared counter.

The source scan is kept as a SECOND, STRUCTURAL control with its own non-vacuity and
planted-violation companions -- it answers a different question (is the helper named in the
function's own body, so a future reader can see the single source without running anything)
and it is the check that fails if somebody re-introduces the silent UTC assumption.

THE THREE CALL SITES
--------------------
1. ``_is_frozen`` -- the upsert-time fence protecting an already-stored row.
2. ``select_games_for_freeze_instant`` -- the EMISSION-time fence this plan adds (R6).
3. ``assert_decided_at_before_freeze`` -- the WRITE-time assertion that a row's own
   observation time is at or before its own game freeze (R7).

They sit on OPPOSITE SIDES of the same boundary and both go through one parse, which is the
only way the two fences can be reasoned about together at all.

Run this module:  uv run pytest tests/unit/test_freeze_parse_single_source.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

import scripts.ingest_historical_odds as odds_module
from api.cache import PROVENANCE_FORWARD
from backtest import weekly_bet_list
from backtest.weekly_bet_list import (
    DECIDED_AT_COLUMN,
    assert_decided_at_before_freeze,
    select_games_for_freeze_instant,
)

_STRICT_PARSER_NAME = "require_aware_snapshot_ts"

# Week 3's Thursday game and the instant it shares with week 2's Sunday slate.
_THURSDAY_GAMEDAY = "2026-09-24"
_FREEZE_TEXT = "2026-09-18T18:00:00-04:00"


class _CountingParser:
    """A stub that DELEGATES and counts, so the real parse behaviour is unchanged.

    Counting a stub that replaced the behaviour would prove the call site reaches the stub and
    nothing about whether the parse it performs is the shared one. Delegating keeps every
    assertion downstream of it true.
    """

    def __init__(self, real: Callable[[Any], datetime]) -> None:
        self._real = real
        self.calls: list[Any] = []

    def __call__(self, value: Any) -> datetime:
        self.calls.append(value)
        return self._real(value)


@pytest.fixture
def counting_parser(monkeypatch: pytest.MonkeyPatch) -> _CountingParser:
    """Replace the ONE strict parser with a counting delegate, module-wide."""
    stub = _CountingParser(odds_module.require_aware_snapshot_ts)
    monkeypatch.setattr(odds_module, _STRICT_PARSER_NAME, stub)
    return stub


def _forward_row(*, decided_at: str, freeze_ts: str = _FREEZE_TEXT) -> dict[str, Any]:
    return {
        "game_id": "2026_03_ATL_GB",
        "provenance": PROVENANCE_FORWARD,
        "freeze_ts": freeze_ts,
        DECIDED_AT_COLUMN: decided_at,
    }


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2026_03_ATL_GB",
                "season": 2026,
                "week": 3,
                "gameday": _THURSDAY_GAMEDAY,
            }
        ]
    )


# ---------------------------------------------------------------------------
# The primary assertion: one object, three call sites.
# ---------------------------------------------------------------------------


class TestAllThreeCallSitesReachTheSameHelperObject:
    """The single-source claim, asserted by a shared counter rather than by three greps."""

    def test_the_upsert_fence_reaches_the_shared_parser(
        self, counting_parser: _CountingParser
    ) -> None:
        row = pd.Series({"provenance": PROVENANCE_FORWARD, "freeze_ts": _FREEZE_TEXT})
        weekly_bet_list._is_frozen(row, datetime(2026, 9, 19, tzinfo=UTC))
        assert len(counting_parser.calls) >= 1

    def test_the_selection_fence_reaches_the_shared_parser(
        self, counting_parser: _CountingParser
    ) -> None:
        instant = odds_module.get_synthetic_snapshot_ts(_THURSDAY_GAMEDAY)
        select_games_for_freeze_instant(
            _schedule(), instant, now=instant - timedelta(seconds=1)
        )
        assert len(counting_parser.calls) >= 1

    def test_the_write_time_assertion_reaches_the_shared_parser(
        self, counting_parser: _CountingParser
    ) -> None:
        assert_decided_at_before_freeze(
            _forward_row(decided_at="2026-09-18T17:59:59-04:00")
        )
        assert len(counting_parser.calls) >= 1

    def test_all_three_increment_ONE_counter_in_a_single_run(
        self, counting_parser: _CountingParser
    ) -> None:
        """The assertion that makes the other three more than a formality.

        If any call site imported its own copy, or re-derived the parse locally, its
        contribution to this single counter would be zero and the deltas below would not all
        be positive.
        """
        instant = odds_module.get_synthetic_snapshot_ts(_THURSDAY_GAMEDAY)

        before_fence = len(counting_parser.calls)
        weekly_bet_list._is_frozen(
            pd.Series({"provenance": PROVENANCE_FORWARD, "freeze_ts": _FREEZE_TEXT}),
            datetime(2026, 9, 19, tzinfo=UTC),
        )
        after_fence = len(counting_parser.calls)

        select_games_for_freeze_instant(
            _schedule(), instant, now=instant - timedelta(seconds=1)
        )
        after_selection = len(counting_parser.calls)

        assert_decided_at_before_freeze(
            _forward_row(decided_at="2026-09-18T17:59:59-04:00")
        )
        after_write = len(counting_parser.calls)

        assert after_fence > before_fence, "_is_frozen did not reach the shared parser"
        assert after_selection > after_fence, (
            "the selection fence did not reach the shared parser"
        )
        assert after_write > after_selection, (
            "the write-time assertion did not reach the shared parser"
        )


# ---------------------------------------------------------------------------
# The structural control: the helper is NAMED in each function's own body.
# ---------------------------------------------------------------------------


def names_the_strict_parser(function: Callable[..., Any]) -> bool:
    """True when *function*'s own source names the strict parse helper.

    The importable form the planted-violation control drives, so the control exercises this
    code path rather than a copy of its logic.
    """
    return _STRICT_PARSER_NAME in inspect.getsource(function)


_SCANNED_FUNCTIONS: tuple[Callable[..., Any], ...] = (
    weekly_bet_list._is_frozen,
    select_games_for_freeze_instant,
    assert_decided_at_before_freeze,
)


class TestTheStructuralControl:
    """Four controls: non-vacuity, the assertion, a planted violation, no false positive."""

    def test_the_scan_resolves_three_functions(self) -> None:
        """NON-VACUITY: an empty scanned set would make every assertion below pass."""
        assert len(_SCANNED_FUNCTIONS) == 3
        for function in _SCANNED_FUNCTIONS:
            assert inspect.getsource(function).strip(), function.__name__

    @pytest.mark.parametrize(
        "function", _SCANNED_FUNCTIONS, ids=lambda f: str(f.__name__)
    )
    def test_every_freeze_comparison_names_the_strict_parser(
        self, function: Callable[..., Any]
    ) -> None:
        assert names_the_strict_parser(function), (
            f"{function.__name__} does not name {_STRICT_PARSER_NAME}; a freeze comparison "
            "reached by a second parse path is a second answer wearing one name"
        )

    def test_a_planted_violation_is_flagged(self) -> None:
        """PLANTED VIOLATION: the shape this plan REMOVED must still be detectable."""

        def _plants_the_silent_utc_assumption(freeze_dt: datetime) -> datetime:
            if freeze_dt.tzinfo is None:
                freeze_dt = freeze_dt.replace(tzinfo=UTC)
            return freeze_dt

        assert not names_the_strict_parser(_plants_the_silent_utc_assumption)

    def test_no_false_positive_on_a_function_that_does_name_it(self) -> None:
        """NO FALSE POSITIVE: the scan does not simply return False for everything."""

        def _honest(value: Any) -> datetime:
            return odds_module.require_aware_snapshot_ts(value)

        assert names_the_strict_parser(_honest)


class TestTheUpsertFenceOperatorIsUnchanged:
    """The ``>=`` at ``_is_frozen``'s last line is the operator R6 matches. It stays."""

    def test_the_silent_utc_assumption_is_gone(self) -> None:
        source = inspect.getsource(weekly_bet_list._is_frozen)
        assert "replace(tzinfo=" not in source, (
            "_is_frozen still assumes UTC for a naive freeze instead of raising"
        )

    def test_the_greater_or_equal_operator_survives(self) -> None:
        """A reviewer believed this plan corrected ``>=`` to ``>``. It does NOT, by ruling."""
        source = inspect.getsource(weekly_bet_list._is_frozen)
        assert "now >= freeze_dt" in source
        assert "now > freeze_dt" not in source

    def test_the_nan_and_coerce_guards_are_untouched(self) -> None:
        """A null or unparseable stored freeze still reads NOT FROZEN, as it always did."""
        assert (
            weekly_bet_list._is_frozen(
                pd.Series({"provenance": PROVENANCE_FORWARD, "freeze_ts": None}),
                datetime(2026, 9, 19, tzinfo=UTC),
            )
            is False
        )
        assert (
            weekly_bet_list._is_frozen(
                pd.Series(
                    {"provenance": PROVENANCE_FORWARD, "freeze_ts": "not a timestamp"}
                ),
                datetime(2026, 9, 19, tzinfo=UTC),
            )
            is False
        )
