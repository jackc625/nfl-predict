"""The closing reading, forward closing-line value, and one-time finalization (Phase 34, LDGR-07).

Pure functions over frames and ledger entries. Nothing here reads a file, touches the network or
reads the clock: the near-kickoff capture is ``scripts/capture_closing_lines.py`` (Plan 34-14) and
the settle pass that applies these updates through ``forward_ledger.store.commit_changes`` is Plan
34-15.

THE WINDOW
----------
A row's closing reading is the LATEST capture of its game with
``kickoff - CLOSING_WINDOW <= created_at <= kickoff``. Both edges are inclusive: a capture exactly
at kickoff and one exactly 60:00 before it qualify; 60:01 before and any instant after kickoff do
not. The comparison is between tz-aware instants (``timedelta``, never rounded minutes), and
``created_at`` is parsed strictly by ``backtest.ou_divergence.odds_information_time`` -- the same
reader every decision path uses. ``snapshot_ts`` is a LABEL (the game's lock) and is never read.

The kickoff is the game's FINAL recorded kickoff: :func:`finalize_closing` runs only once silver
``games`` records the result, against the ``kickoff_et`` stored then. A flexed game is judged
against the slot it was actually played in.

NEVER THE DECISION LINE (the SPEC must-not)
-------------------------------------------
The decision capture is taken at 17:00 ET on the day before kickoff, 15 to 29.5 hours before the
game (``utils.game_lock``), so it can never sit inside a 60-minute window. A game with no in-window
capture finalizes with every closing value NULL and a recorded reason -- never a value copied from
the decision row.

THE BOOK
--------
Among in-window captures that carry the row's market, the first by
``backtest.ou_divergence.order_by_book_preference`` wins: latest capture instant, then the
``fill-v1`` preference, then book name. The decision used the same ordering, so decision-versus-
close compares the same rule's choice at two times.

FORWARD CLOSING-LINE VALUE (D-09)
---------------------------------
Freeze-vs-close, with NO slippage, so it measures market movement rather than fill quality.
Positive always means the bet beat the close:

  * ATS (``line`` is the home margin the home team must EXCEED, positive when home is favored):
    ``home_cover`` = closing line - decision line; ``away_cover`` the reverse.
  * O/U: ``over`` = closing total - decision total; ``under`` the reverse.
  * WP: ``american_to_implied(closing price of the bet side) - american_to_implied(decision
    price)`` -- the codebase's implied-probability convention (``backtest.ou_ev_chain``).

The decision values are the row's ``line`` (or ``selected_odds`` for WP), never ``slipped_line``.

REPORT-ONLY. These columns feed no decision; ``tests/unit/test_closing_capture.py`` asserts by AST
that the selection modules never read them.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from backtest.ats_ev_chain import ATS_SIDES
from backtest.ou_divergence import odds_information_time, order_by_book_preference
from backtest.ou_ev_chain import american_to_implied
from backtest.selector_strategies import OU_SIDES, is_absent
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.schema import (
    BET_LIST_CLOSING_COLUMNS,
    CLOSING_NULL_REASONS,
    LEDGER_ROW_KEY,
)
from forward_ledger.store import LedgerEntry
from utils.date_utils import kickoff_wall_clock_et

__all__ = [
    "CLOSING_SKIP_REASONS",
    "CLOSING_WINDOW",
    "REASON_CAPTURE_MISSED",
    "REASON_NO_BET_SIDE",
    "REASON_NO_CLOSING_MARKET",
    "closing_values_for",
    "finalize_closing",
    "market_fields_for",
    "select_closing_capture",
]

CLOSING_WINDOW = timedelta(minutes=60)

# The NULL reasons this module assigns itself. The other three come from the closing task's own
# run-log records (CLOSING_SKIP_REASONS). All six are api.cache.CLOSING_NULL_REASONS.
REASON_CAPTURE_MISSED = "capture_missed"
REASON_NO_CLOSING_MARKET = "no_closing_market"
REASON_NO_BET_SIDE = "no_bet_side"
CLOSING_SKIP_REASONS: tuple[str, ...] = (
    "credit_reserve",
    "credit_header_unreadable",
    "capture_failed",
)
_OWN_REASONS = (REASON_CAPTURE_MISSED, REASON_NO_CLOSING_MARKET, REASON_NO_BET_SIDE)
if set(_OWN_REASONS) | set(CLOSING_SKIP_REASONS) != set(
    CLOSING_NULL_REASONS
):  # pragma: no cover
    msg = "forward_ledger.closing's reasons no longer equal api.cache.CLOSING_NULL_REASONS"
    raise RuntimeError(msg)

_REASON_COLUMN = "closing_null_reason"

# The silver odds column each LINE target's market is read from -- the same column the decision
# frame renames into ``closing_spread`` / ``closing_total`` (``weekly_bet_list._ODDS_SOURCE_COLUMN``),
# so the closing line is on the row's own ``line`` scale. WP has no line: a moneyline is a price.
_LINE_MARKET_FIELD: dict[str, str] = {"ats": "spread", "ou": "total"}

# +1 where closing minus decision is the bet's gain, -1 where it is the bet's loss (D-09). Keyed by
# the LOCKED side vocabularies, so an unknown side is refused rather than given a sign.
_LINE_CLV_SIGN: dict[str, dict[str, int]] = {
    "ats": {ATS_SIDES[0]: 1, ATS_SIDES[1]: -1},
    "ou": {OU_SIDES[0]: 1, OU_SIDES[1]: -1},
}


def market_fields_for(target: str, bet_side: str | None) -> tuple[str, ...]:
    """The silver odds columns a capture must carry to be a closing reading for this row.

    WP reads only the BET SIDE's own moneyline, by the LOCKED ``BettingSimulator._get_wp_odds``
    convention (``ml_home`` when the side is ``home``, else ``ml_away``): a board missing only the
    other side's price still qualifies, and one missing the bet side's price never does -- the WP
    strategy's ``bet_odds`` refuses a missing side price, so it must never be handed one. ATS needs
    ``spread`` and O/U ``total``.

    Raises:
        ValueError: *target* is not ``wp``, ``ats`` or ``ou``.
    """
    if target == "wp":
        return ("ml_home",) if bet_side == "home" else ("ml_away",)
    if target in _LINE_MARKET_FIELD:
        return (_LINE_MARKET_FIELD[target],)
    msg = f"unknown target {target!r}; a closing reading is defined for wp, ats and ou"
    raise ValueError(msg)


def _aware_utc(value: Any) -> datetime:
    """*value* as a tz-aware UTC instant through the project's ONE strict parser (naive refused).

    Lazy import: ``scripts.ingest_historical_odds`` cycles back through ``backtest.bet_selector``.
    """
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    return require_aware_snapshot_ts(value)


def select_closing_capture(
    game_id: str,
    kickoff_utc: Any,
    odds_rows: pd.DataFrame,
    target: str,
    bet_side: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """The game's closing reading for one row, or why there is none.

    Args:
        game_id: The game.
        kickoff_utc: The game's final recorded kickoff; must be timezone-aware.
        odds_rows: Silver odds rows (any games; filtered here), ``created_at`` the capture instant.
        target: ``wp``, ``ats`` or ``ou``.
        bet_side: The row's bet side (selects the WP moneyline).

    Returns:
        ``(capture_row, None)`` for the chosen capture; ``(None, "no_closing_market")`` when
        in-window captures exist but none carries the row's market; ``(None, None)`` when no
        capture lies in the window at all (the caller names that reason).

    Raises:
        scripts.ingest_historical_odds.NaiveTimestampError: a naive kickoff or ``created_at``.
    """
    kickoff = _aware_utc(kickoff_utc)
    rows = odds_rows.loc[odds_rows["game_id"].astype(str) == str(game_id)]
    if rows.empty:
        return None, None

    captured = odds_information_time(rows)
    in_window = (
        captured.notna()
        & (captured >= kickoff - CLOSING_WINDOW)
        & (captured <= kickoff)
    )
    window = rows.loc[in_window]
    if window.empty:
        return None, None

    fields = list(market_fields_for(target, bet_side))
    carrying = window.loc[window.reindex(columns=fields).notna().all(axis=1)]
    if carrying.empty:
        return None, REASON_NO_CLOSING_MARKET
    return order_by_book_preference(carrying).iloc[0].to_dict(), None


def _forward_clv(
    immutable: Mapping[str, Any],
    closing_line: float | None,
    closing_odds: float | None,
) -> float | None:
    """Freeze-vs-close for the bet side, positive when the bet beat the close (D-09)."""
    target = immutable["target"]
    bet_side = immutable["bet_side"]
    if target == "wp":
        decision_odds = immutable.get("selected_odds")
        if closing_odds is None or decision_odds is None or is_absent(decision_odds):
            return None
        return american_to_implied(closing_odds) - american_to_implied(
            float(decision_odds)
        )

    signs = _LINE_CLV_SIGN[target]
    if bet_side not in signs:
        msg = f"unknown {target} bet side {bet_side!r}; the LOCKED sides are {list(signs)}"
        raise ValueError(msg)
    decision_line = immutable.get("line")
    if closing_line is None or decision_line is None or is_absent(decision_line):
        return None
    return signs[bet_side] * (closing_line - float(decision_line))


def closing_values_for(
    row_entry: LedgerEntry | Mapping[str, Any],
    capture_row: Mapping[str, Any],
    strategies: Mapping[str, Any],
) -> dict[str, Any]:
    """The six closing columns for a row whose closing reading is *capture_row*.

    Args:
        row_entry: The ledger row (or its immutable half).
        capture_row: The chosen silver odds row (from :func:`select_closing_capture`).
        strategies: ``target -> strategy`` (the live registry); the price is the target
            strategy's own ``bet_odds`` on the closing row, so the closing price is read exactly as
            the decision price was. A line target's book with no side price gives a NULL
            ``closing_odds`` -- never a default.

    Returns:
        ``closing_line`` (``spread`` / ``total`` on the row's ``line`` scale; NULL for WP),
        ``closing_odds``, ``closing_sportsbook``, ``closing_captured_at`` (ISO with offset),
        ``forward_clv`` and ``closing_null_reason`` (NULL).
    """
    immutable = row_entry.immutable if isinstance(row_entry, LedgerEntry) else row_entry
    target = immutable["target"]
    capture = dict(capture_row)

    price = strategies[target].bet_odds(capture, immutable["bet_side"])
    closing_odds = None if price is None else float(price)
    line_field = _LINE_MARKET_FIELD.get(target)
    closing_line = None if line_field is None else float(capture[line_field])

    return {
        "closing_line": closing_line,
        "closing_odds": closing_odds,
        "closing_sportsbook": str(capture["sportsbook"]),
        "closing_captured_at": _aware_utc(capture["created_at"]).isoformat(),
        "forward_clv": _forward_clv(immutable, closing_line, closing_odds),
        _REASON_COLUMN: None,
    }


def _null_closing(reason: str) -> dict[str, Any]:
    """Every closing value NULL, with the recorded *reason* (NULL is never silent)."""
    return {**dict.fromkeys(BET_LIST_CLOSING_COLUMNS), _REASON_COLUMN: reason}


def _recorded_kickoffs(games: pd.DataFrame) -> dict[str, datetime]:
    """``game_id -> final recorded kickoff (UTC)`` for every game with both scores recorded.

    The ET wall clock goes through ``utils.date_utils.kickoff_wall_clock_et``, THE accessor for
    ``kickoff_et``: an aware value is converted, never relabelled. A game with a result but no
    kickoff cannot be judged against a window and is left unfinalized, never guessed.
    """
    recorded = games.loc[
        games["home_score"].notna()
        & games["away_score"].notna()
        & games["kickoff_et"].notna()
    ]
    return {
        str(game_id): kickoff_wall_clock_et(kickoff).astimezone(UTC)
        for game_id, kickoff in zip(
            recorded["game_id"], recorded["kickoff_et"], strict=True
        )
    }


def _skip_reason(game_id: str, capture_events: Sequence[Mapping[str, Any]]) -> str:
    """The latest closing-task skip or failure reason covering *game_id*, else ``capture_missed``.

    *capture_events* are the run log's ``closing_capture`` records, oldest first.
    """
    reason = REASON_CAPTURE_MISSED
    for event in capture_events:
        if event.get("reason") in CLOSING_SKIP_REASONS and game_id in (
            event.get("game_ids") or ()
        ):
            reason = str(event["reason"])
    return reason


def finalize_closing(
    entries: Sequence[LedgerEntry],
    games: pd.DataFrame,
    odds: pd.DataFrame,
    capture_events: Sequence[Mapping[str, Any]],
    strategies: Mapping[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """The closing-half updates owed now, keyed by ``LEDGER_ROW_KEY``. Returns updates only.

    A row is finalized once, and only when its closing half is entirely NULL (no values and no
    reason) and its game has both scores recorded in *games* -- by then the kickoff cannot move,
    so the window is judged against the ``kickoff_et`` stored now. A row already holding values or
    a reason gets no update, so a re-run changes nothing. Per row:

      * no ``bet_side`` -> NULL with ``no_bet_side``;
      * an in-window capture carrying the row's market -> the six values (:func:`closing_values_for`);
      * in-window captures, none carrying the market -> NULL with ``no_closing_market``;
      * no in-window capture -> NULL with the reason of the latest closing-task skip or failure
        event naming the game (``credit_reserve``, ``credit_header_unreadable``,
        ``capture_failed``), else ``capture_missed``.

    Plan 34-15 applies the result through ``commit_changes(closing_updates=...)``, whose one-way
    closing transition refuses any overwrite.

    Args:
        entries: The ledger entries (correction entries are skipped).
        games: Silver games: ``game_id``, ``kickoff_et``, ``home_score``, ``away_score``.
        odds: Silver odds rows (``data/silver/odds_snapshot.parquet`` shape).
        capture_events: The run log's ``closing_capture`` records, oldest first.
        strategies: ``target -> strategy`` (the live registry).
    """
    kickoffs = _recorded_kickoffs(games)
    updates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        if entry.kind != ENTRY_KIND_ROW:
            continue
        if any(value is not None for value in (entry.closing or {}).values()):
            continue
        immutable = entry.immutable
        game_id = str(immutable["game_id"])
        kickoff = kickoffs.get(game_id)
        if kickoff is None:
            continue

        key = tuple(immutable[name] for name in LEDGER_ROW_KEY)
        bet_side = immutable.get("bet_side")
        if is_absent(bet_side):
            updates[key] = _null_closing(REASON_NO_BET_SIDE)
            continue

        capture, reason = select_closing_capture(
            game_id, kickoff, odds, immutable["target"], bet_side
        )
        if capture is not None:
            updates[key] = closing_values_for(immutable, capture, strategies)
        elif reason is not None:
            updates[key] = _null_closing(reason)
        else:
            updates[key] = _null_closing(_skip_reason(game_id, capture_events))
    return updates
