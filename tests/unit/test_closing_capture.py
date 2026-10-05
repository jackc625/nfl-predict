"""The closing reading, forward closing-line value and one-time finalization (Phase 34, LDGR-07).

Every case here is a pure function over small synthetic frames: no network, no store, no clock.
The window is ``kickoff - 60:00 <= created_at <= kickoff`` against the game's FINAL recorded
kickoff, ties on the capture instant resolve by the ``fill-v1`` book ordering, a row's closing half
is set once, and the decision-time capture can never become a closing reading.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.ou_ev_chain import american_to_implied
from backtest.selector_strategies import ATSStrategy, OUStrategy, WPStrategy
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.closing import (
    CLOSING_WINDOW,
    closing_values_for,
    finalize_closing,
    select_closing_capture,
)
from forward_ledger.schema import BET_LIST_CLOSING_COLUMNS, LEDGER_ROW_KEY
from forward_ledger.store import LedgerEntry
from forward_ledger.transitions import assert_closing_transition

REPO_ROOT = Path(__file__).resolve().parents[2]

GAME = "2026_W06_HOU@JAX"
# 13:00 ET on Sunday 2026-10-18 (EDT, UTC-4).
KICKOFF = datetime(2026, 10, 18, 17, 0, tzinfo=UTC)
# The game's day-before lock, 18:00 ET Saturday -- the decision capture's instant.
LOCK = datetime(2026, 10, 17, 22, 0, tzinfo=UTC)


def _strategies() -> dict[str, Any]:
    return {
        "wp": WPStrategy(),
        "ats": ATSStrategy(frozen_sd=13.0, season_bias_by_season={}),
        "ou": OUStrategy(frozen_sd=13.0, season_bias_by_season={}),
    }


def _capture(
    created_at: datetime, sportsbook: str = "draftkings", **overrides: Any
) -> dict[str, Any]:
    """One silver odds row for GAME, every market present unless overridden."""
    row: dict[str, Any] = {
        "game_id": GAME,
        "snapshot_ts": LOCK.isoformat(),
        "sportsbook": sportsbook,
        "ml_home": -150.0,
        "ml_away": 130.0,
        "spread": 3.0,
        "spread_ju_home": -110.0,
        "spread_ju_away": -110.0,
        "total": 44.5,
        "total_over_ju": -110.0,
        "total_under_ju": -110.0,
        "is_live": False,
        "created_at": created_at,
    }
    row.update(overrides)
    return row


def _odds(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _games(kickoff: datetime = KICKOFF, *, scored: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": GAME,
                "kickoff_et": kickoff,
                "home_score": 24.0 if scored else float("nan"),
                "away_score": 17.0 if scored else float("nan"),
            }
        ]
    )


def _immutable(target: str, bet_side: str | None, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": GAME,
        "season": 2026,
        "week": 6,
        "target": target,
        "arm": "live",
        "bet_side": bet_side,
        "line": {"wp": None, "ats": 3.0, "ou": 44.5}[target],
        "slipped_line": {"wp": None, "ats": 3.5, "ou": 45.0}[target],
        "selected_odds": -150.0 if target == "wp" else -110.0,
    }
    row.update(overrides)
    return row


def _entry(
    immutable: dict[str, Any], closing: dict[str, Any] | None = None
) -> LedgerEntry:
    return LedgerEntry(
        seq=0,
        kind=ENTRY_KIND_ROW,
        canon_v=1,
        immutable=immutable,
        chain_hash="0" * 64,
        grading={"grading_status": "pending"},
        fill={},
        closing={**dict.fromkeys(BET_LIST_CLOSING_COLUMNS), **(closing or {})},
    )


def _key(immutable: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(immutable[name] for name in LEDGER_ROW_KEY)


# ---------------------------------------------------------------------------
# The window, the ordering and the market filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "qualifies"),
    [
        (timedelta(0), True),
        (-CLOSING_WINDOW, True),
        (-(CLOSING_WINDOW + timedelta(seconds=1)), False),
        (timedelta(seconds=1), False),
    ],
    ids=["at_kickoff", "exactly_60_00_before", "60_01_before", "one_second_after"],
)
def test_window_edges(offset: timedelta, qualifies: bool) -> None:
    odds = _odds(_capture(KICKOFF + offset))
    row, reason = select_closing_capture(GAME, KICKOFF, odds, "ats", "home_cover")
    assert reason is None
    if qualifies:
        assert row is not None
        assert pd.Timestamp(row["created_at"]) == KICKOFF + offset
    else:
        assert row is None


def test_latest_in_window_wins() -> None:
    odds = _odds(
        _capture(KICKOFF - timedelta(minutes=50), spread=2.5),
        _capture(KICKOFF - timedelta(minutes=5), spread=4.0),
        _capture(KICKOFF - timedelta(minutes=30), spread=3.5),
    )
    row, reason = select_closing_capture(GAME, KICKOFF, odds, "ats", "home_cover")
    assert reason is None
    assert row is not None
    assert pd.Timestamp(row["created_at"]) == KICKOFF - timedelta(minutes=5)
    assert row["spread"] == 4.0


def test_time_tie_uses_fill_v1_order() -> None:
    instant = KICKOFF - timedelta(minutes=10)
    first, _ = select_closing_capture(
        GAME,
        KICKOFF,
        _odds(_capture(instant, "betmgm"), _capture(instant, "draftkings")),
        "ou",
        "over",
    )
    assert first is not None
    assert first["sportsbook"] == "draftkings"

    second, _ = select_closing_capture(
        GAME,
        KICKOFF,
        _odds(_capture(instant, "bovada"), _capture(instant, "betmgm")),
        "ou",
        "over",
    )
    assert second is not None
    assert second["sportsbook"] == "betmgm"


def test_market_filter() -> None:
    early = KICKOFF - timedelta(minutes=40)
    late = KICKOFF - timedelta(minutes=5)
    odds = _odds(
        _capture(early, total=46.0),
        _capture(late, total=float("nan")),
    )
    row, reason = select_closing_capture(GAME, KICKOFF, odds, "ou", "over")
    assert reason is None
    assert row is not None
    assert pd.Timestamp(row["created_at"]) == early
    assert row["total"] == 46.0

    none_carry = _odds(
        _capture(early, total=float("nan")), _capture(late, total=float("nan"))
    )
    row, reason = select_closing_capture(GAME, KICKOFF, none_carry, "ou", "over")
    assert row is None
    assert reason == "no_closing_market"


def test_wp_partial_h2h_board_uses_bet_side_price() -> None:
    strategies = _strategies()
    early = KICKOFF - timedelta(minutes=45)
    late = KICKOFF - timedelta(minutes=5)

    home_board = _odds(_capture(late, ml_home=-170.0, ml_away=float("nan")))
    row, reason = select_closing_capture(GAME, KICKOFF, home_board, "wp", "home")
    assert reason is None
    assert row is not None
    values = closing_values_for(_immutable("wp", "home"), row, strategies)
    assert values["closing_odds"] == -170.0
    assert values["closing_line"] is None

    away_board = _odds(
        _capture(early, ml_home=-160.0, ml_away=140.0),
        _capture(late, ml_home=-170.0, ml_away=float("nan")),
    )
    row, reason = select_closing_capture(GAME, KICKOFF, away_board, "wp", "away")
    assert reason is None
    assert row is not None
    assert pd.Timestamp(row["created_at"]) == early
    values = closing_values_for(_immutable("wp", "away"), row, strategies)
    assert values["closing_odds"] == 140.0

    no_side = _odds(_capture(late, ml_home=-170.0, ml_away=float("nan")))
    row, reason = select_closing_capture(GAME, KICKOFF, no_side, "wp", "away")
    assert row is None
    assert reason == "no_closing_market"


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def test_decision_capture_never_used() -> None:
    immutable = _immutable("ats", "home_cover")
    odds = _odds(_capture(LOCK))  # the decision capture, a day before kickoff
    updates = finalize_closing([_entry(immutable)], _games(), odds, [], _strategies())
    update = updates[_key(immutable)]
    assert update["closing_null_reason"] == "capture_missed"
    assert all(update[name] is None for name in BET_LIST_CLOSING_COLUMNS[:-1])
    assert_closing_transition(dict.fromkeys(BET_LIST_CLOSING_COLUMNS), update)


def test_reason_from_skip_record() -> None:
    immutable = _immutable("ou", "under")
    entries = [_entry(immutable)]
    events = [
        {
            "event": "closing_capture",
            "outcome": "skipped",
            "reason": "credit_header_unreadable",
            "game_ids": ["2026_W06_OTHER@GAME"],
        },
        {
            "event": "closing_capture",
            "outcome": "skipped",
            "reason": "credit_reserve",
            "game_ids": [GAME, "2026_W06_OTHER@GAME"],
        },
    ]
    updates = finalize_closing(
        entries, _games(), _odds(_capture(LOCK)), events, _strategies()
    )
    assert updates[_key(immutable)]["closing_null_reason"] == "credit_reserve"

    updates = finalize_closing(
        entries, _games(), _odds(_capture(LOCK)), [], _strategies()
    )
    assert updates[_key(immutable)]["closing_null_reason"] == "capture_missed"


def test_finalize_waits_for_result() -> None:
    immutable = _immutable("ats", "home_cover")
    odds = _odds(_capture(KICKOFF - timedelta(minutes=5)))
    updates = finalize_closing(
        [_entry(immutable)], _games(scored=False), odds, [], _strategies()
    )
    assert updates == {}


def test_finalize_uses_final_recorded_kickoff() -> None:
    immutable = _immutable("ats", "home_cover")
    moved = datetime(2026, 10, 18, 20, 25, tzinfo=UTC)  # flexed from 13:00 to 16:25 ET
    near_old = _capture(KICKOFF - timedelta(minutes=10), spread=2.5)
    near_new = _capture(moved - timedelta(minutes=10), spread=4.0)

    updates = finalize_closing(
        [_entry(immutable)], _games(moved), _odds(near_old, near_new), [], _strategies()
    )
    update = updates[_key(immutable)]
    assert update["closing_line"] == 4.0
    assert update["closing_null_reason"] is None
    assert_closing_transition(dict.fromkeys(BET_LIST_CLOSING_COLUMNS), update)

    updates = finalize_closing(
        [_entry(immutable)], _games(moved), _odds(near_old), [], _strategies()
    )
    assert updates[_key(immutable)]["closing_null_reason"] == "capture_missed"


def test_finalize_is_once() -> None:
    odds = _odds(_capture(KICKOFF - timedelta(minutes=5), spread=4.0))
    first = _immutable("ats", "home_cover")
    updates = finalize_closing([_entry(first)], _games(), odds, [], _strategies())
    applied = updates[_key(first)]

    finalized = _entry(first, applied)
    reasoned = _entry(
        _immutable("ou", "over"), {"closing_null_reason": "capture_missed"}
    )
    assert (
        finalize_closing([finalized, reasoned], _games(), odds, [], _strategies()) == {}
    )


# ---------------------------------------------------------------------------
# Forward closing-line value (D-09)
# ---------------------------------------------------------------------------


def test_forward_clv_signs() -> None:
    strategies = _strategies()
    close = _capture(
        KICKOFF - timedelta(minutes=5), spread=4.0, total=46.0, ml_home=-170.0
    )

    home = closing_values_for(_immutable("ats", "home_cover"), close, strategies)
    away = closing_values_for(_immutable("ats", "away_cover"), close, strategies)
    assert home["forward_clv"] == pytest.approx(1.0)
    assert away["forward_clv"] == pytest.approx(-1.0)
    assert home["closing_line"] == 4.0
    assert home["closing_odds"] == -110.0

    over = closing_values_for(_immutable("ou", "over"), close, strategies)
    under = closing_values_for(_immutable("ou", "under"), close, strategies)
    assert over["forward_clv"] == pytest.approx(1.5)
    assert under["forward_clv"] == pytest.approx(-1.5)

    wp = closing_values_for(_immutable("wp", "home"), close, strategies)
    expected = american_to_implied(-170) - american_to_implied(-150)
    assert expected > 0
    assert wp["forward_clv"] == pytest.approx(expected)
    assert wp["closing_sportsbook"] == "draftkings"
    assert wp["closing_captured_at"] == (KICKOFF - timedelta(minutes=5)).isoformat()
    assert wp["closing_null_reason"] is None


def test_no_slippage_used() -> None:
    close = _capture(KICKOFF - timedelta(minutes=5), spread=4.0)
    row = _immutable("ats", "home_cover", line=3.0, slipped_line=3.5)
    values = closing_values_for(row, close, _strategies())
    assert values["forward_clv"] == pytest.approx(1.0)


def test_no_bet_side_reason() -> None:
    immutable = _immutable("wp", None)
    odds = _odds(_capture(KICKOFF - timedelta(minutes=5)))
    updates = finalize_closing([_entry(immutable)], _games(), odds, [], _strategies())
    update = updates[_key(immutable)]
    assert update["closing_null_reason"] == "no_bet_side"
    assert all(update[name] is None for name in BET_LIST_CLOSING_COLUMNS[:-1])


# ---------------------------------------------------------------------------
# Report-only: the selection path never reads a closing column
# ---------------------------------------------------------------------------

_CLOSING_NAMES = frozenset(
    {
        "forward_clv",
        "closing_line",
        "closing_odds",
        "closing_captured_at",
        "closing_null_reason",
    }
)
_SELECTION_MODULES = (
    "backtest/bet_selector.py",
    "backtest/selector_strategies.py",
    "backtest/weekly_bet_list.py",
)
# The cutover switch is one boolean (Plan 34-15 routes forward rows on it); it reads no closing.
_SELECTION_ALLOWED_LEDGER_IMPORTS = frozenset({"forward_ledger.cutover"})


def _closing_references(source: str) -> tuple[set[str], set[str]]:
    """(closing names referenced, forward_ledger imports) found in *source*."""
    names: set[str] = set()
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in _CLOSING_NAMES:
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _CLOSING_NAMES:
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg in _CLOSING_NAMES:
            names.add(node.arg)
        elif isinstance(node, ast.Constant) and node.value in _CLOSING_NAMES:
            names.add(str(node.value))
        elif isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("forward_ledger")
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "forward_ledger"
        ):
            imports.add(str(node.module))
    return names, imports


def test_selection_path_never_reads_closing() -> None:
    control_names, control_imports = _closing_references(
        "from forward_ledger.closing import x\nrow['closing_line']\nrow.forward_clv\n"
    )
    assert control_names == {"closing_line", "forward_clv"}
    assert control_imports == {"forward_ledger.closing"}

    for relative in _SELECTION_MODULES:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        names, imports = _closing_references(source)
        assert names == set(), (
            f"{relative} references closing column(s) {sorted(names)}"
        )
        unexpected = imports - _SELECTION_ALLOWED_LEDGER_IMPORTS
        assert unexpected == set(), f"{relative} imports {sorted(unexpected)}"
