"""No shadow row surfaces as a bet or enters the verdict (Phase 37, Plan 37-02; SPEC R13, must-NOT P3).

The shadow arm is a COMPARISON arm: it scores the frozen row-19 models beside the live arm and
writes its rows to the same ledger with ``arm = "shadow"``, but none of those rows is ever a bet.
Each test below plants shadow rows into one owner-facing read path and proves the path is blind
to them:

* the Saturday email's inputs (``notify.digest.collect_digest_inputs``): its bet list and its
  per-game states come out identical with and without the planted rows, and a game carrying only
  shadow rows is never read as evaluated;
* the bets push (``notify.messages.bets_message``): it names only live-arm rows, and a run that
  appended only shadow rows sends no push;
* the 2026 verdict (``forward_ledger.verdict.verdict_rows``): the counted frame is identical with
  and without a shadow twin for every row, compared with ``assert_frame_equal``.

The /bets page, its coverage list and the export are proven in ``tests/api/test_bets_page.py`` and
``tests/api/test_export.py``. Every store here lives under ``tmp_path`` (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas.testing as pdt

from api.services import DataService, clear_cache
from forward_ledger.store import append_rows, apply_updates, read_entries
from forward_ledger.verdict import verdict_rows
from notify import digest
from notify.digest import collect_digest_inputs
from notify.messages import bets_message
from tests.unit.test_forward_ledger_store import graded, key_of
from tests.unit.test_notify_digest import (
    MONDAY_GAME,
    WEEK6,
    build_cache,
    saturday_report,
    schedule_frame,
    week6_bets,
)
from tests.unit.test_notify_messages import live_bet
from tests.unit.test_verdict_scope import SCOPE, verdict_row

ARM_SHADOW = "shadow"


def _shadow_twin(row: dict[str, Any]) -> dict[str, Any]:
    """*row*'s shadow-arm twin, with a different EV so a leak would be visible."""
    twin = {**row, "arm": ARM_SHADOW}
    if twin["per_bet_ev"] is not None:
        twin["per_bet_ev"] = 0.09
    return twin


# ---------------------------------------------------------------------------
# The Saturday email's inputs
# ---------------------------------------------------------------------------


def _shadow_only_rows(game_id: str) -> list[dict[str, Any]]:
    """A game the shadow arm evaluated and the live arm did not: a slip and a declined row."""
    return [
        live_bet(game_id, arm=ARM_SHADOW, target="ou", per_bet_ev=0.09),
        live_bet(
            game_id,
            arm=ARM_SHADOW,
            status="suppressed",
            rejection_reason="ev_below_floor",
            per_bet_ev=None,
            stake_units=None,
            ev_tier=None,
        ),
    ]


def _digest_inputs(cache: Path) -> digest.DigestInputs:
    """``collect_digest_inputs`` for the Saturday run over the cache at *cache*."""
    report = saturday_report()
    clear_cache()
    conn = duckdb.connect(str(cache), read_only=True)
    try:
        return collect_digest_inputs(
            report,
            schedule=schedule_frame(WEEK6),
            service=DataService(conn),
            notifications=[],
            failed_titles=(),
            now=report.start + timedelta(minutes=20),
        )
    finally:
        conn.close()


def test_digest_bets_and_games_ignore_shadow_rows(tmp_path: Path) -> None:
    stamped_at = saturday_report().start + timedelta(minutes=12)
    live_rows = week6_bets()
    planted = [
        *live_rows,
        *(_shadow_twin(row) for row in live_rows),
        *_shadow_only_rows(MONDAY_GAME),
    ]
    plain_cache = build_cache(
        tmp_path / "plain.duckdb", bets=live_rows, stamped_at=stamped_at
    )
    planted_cache = build_cache(
        tmp_path / "planted.duckdb", bets=planted, stamped_at=stamped_at
    )

    plain = _digest_inputs(plain_cache)
    with_shadow = _digest_inputs(planted_cache)

    assert with_shadow.cache_current is True
    assert len(with_shadow.bets) == 3, "the week's three live-arm live bets"
    assert {row["arm"] for row in with_shadow.bets} == {"live"}
    assert with_shadow.bets == plain.bets
    assert with_shadow.games == plain.games
    monday = next(game for game in with_shadow.games if game["game_id"] == MONDAY_GAME)
    assert monday["state"] not in (digest.STATE_BET, digest.STATE_NO_BET), (
        "a game carrying only shadow rows was read as evaluated"
    )


# ---------------------------------------------------------------------------
# The bets push
# ---------------------------------------------------------------------------


def _live_rows_for_push() -> list[dict[str, Any]]:
    return [
        live_bet("2026_06_CIN@CLE", target="ats", line=-5.5, per_bet_ev=0.062),
        live_bet("2026_06_DAL@PHI", target="ou", line=47.5, per_bet_ev=0.041),
    ]


def test_bets_push_lists_live_arm_only() -> None:
    live_rows = _live_rows_for_push()
    shadow_rows = [
        *(_shadow_twin(row) for row in live_rows),
        live_bet("2026_06_KC@BUF", arm=ARM_SHADOW, target="wp", per_bet_ev=0.09),
    ]

    plain = bets_message(saturday_report(appended_rows=live_rows))
    mixed = bets_message(saturday_report(appended_rows=[*shadow_rows, *live_rows]))
    shadow_only = bets_message(saturday_report(appended_rows=shadow_rows))

    assert plain is not None and mixed is not None
    assert len(mixed.item_lines) == 2
    assert mixed.title == plain.title
    assert mixed.item_lines == plain.item_lines
    assert not any("KC @ BUF" in line for line in mixed.item_lines)
    assert shadow_only is None


# ---------------------------------------------------------------------------
# The 2026 verdict (must-NOT P3)
# ---------------------------------------------------------------------------


def _verdict_week_rows() -> list[dict[str, Any]]:
    """Live rows inside the declared verdict weeks, the Super Bowl included."""
    return [
        verdict_row("KC@BUF", 6),
        verdict_row("DAL@PHI", 7),
        verdict_row("SF@SEA", 22),
    ]


def test_verdict_unchanged_by_planted_shadow_rows(tmp_path: Path) -> None:
    live_rows = _verdict_week_rows()
    twins = [_shadow_twin(row) for row in live_rows]
    win = graded("win", 0.9090909090909091, 1.1363636363636365)
    loss = graded("loss", -1.0, -1.25)

    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    append_rows(plain_dir, live_rows)
    apply_updates(plain_dir, grading_updates={key_of(row): win for row in live_rows})

    # Every twin is graded the OTHER way, so a shadow grade that leaked would change the frame.
    planted_dir = tmp_path / "planted"
    planted_dir.mkdir()
    append_rows(planted_dir, [*live_rows, *twins])
    apply_updates(
        planted_dir,
        grading_updates={
            **{key_of(row): win for row in live_rows},
            **{key_of(row): loss for row in twins},
        },
    )

    plain = verdict_rows(read_entries(plain_dir), SCOPE)
    with_shadow = verdict_rows(read_entries(planted_dir), SCOPE)

    assert len(read_entries(planted_dir)) == 2 * len(live_rows)
    assert len(plain) == len(live_rows), "the plain verdict counts every live row"
    pdt.assert_frame_equal(with_shadow, plain)
