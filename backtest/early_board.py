"""The early-week calendar and board: which NFL week is due, and what the books offered for it.

WHICH WEEK IS DUE. A week from ``EARLY_FROM`` on is decided on one day only, the ET calendar day
after the last game of the weeks below it. That is a Tuesday in an ordinary week, and it moves by
itself when the schedule does, because it is read from the kickoffs and never from a weekday. A
day the run missed is not made up later: the next day no week is due.

READING THE BOARD. ``scripts/capture_early_lines.py`` appends one JSON line per capture to
``early_lines.jsonl``. The run reads the last line captured on its own ET date, so it can never be
priced from another day's board.

FROM BOARD TO QUOTES. The board goes through the same transform and validation as a live decision
capture, so it is matched to schedule games, sided by team name and refused past each game's lock
by the code the ledger already trusts. The result stays in memory: an early board never enters
silver ``odds_snapshot``. ``spread`` leaves that transform positive when the home team is favoured,
which is the scale ``Quote.line`` is on, so it is copied unchanged.

NO INVENTED PRICE. The shared transform and ``OddsSchema`` both default a price they do not have to
-110. An early row claims a real quote, so ``board_odds`` closes both routes without touching the
live ingest: it works on a copy of the board with every spreads or totals market missing a price
removed, and it keeps every quote column present so the schema's default never applies.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

import utils.game_lock as lock_rule
from backtest.early_arm_2026 import EARLY_FROM
from backtest.early_pricing import Quote

if TYPE_CHECKING:  # pragma: no cover - typing only; the run hands in its own ingester
    from scripts.ingest_odds import OddsDataIngester

__all__ = [
    "EarlyBoard",
    "board_odds",
    "early_week_due",
    "latest_board",
    "quotes_for",
    "read_board_for_run",
]

# Each early target's (line, up price, down price) columns in a validated odds frame.
_QUOTE_COLUMNS: Mapping[str, tuple[str, str, str]] = {
    "ats": ("spread", "spread_ju_home", "spread_ju_away"),
    "ou": ("total", "total_over_ju", "total_under_ju"),
}

# The board markets those columns are read from.
_QUOTED_MARKETS: frozenset[str] = frozenset({"spreads", "totals"})


@dataclass(frozen=True)
class EarlyBoard:
    """One recorded capture of the odds board.

    Attributes:
        captured_at: The instant the board was read, tz-aware.
        events: The board exactly as the Odds API returned it, one entry per game.
    """

    captured_at: datetime
    events: list[dict[str, Any]]


def _require_aware(value: Any) -> datetime:
    """*value* as a plain UTC-aware datetime; a value with no timezone is refused.

    Lazy import, deliberately: ``scripts.ingest_historical_odds`` cycles through
    ``backtest.bet_selector`` and must not be reached while this module is being imported.
    """
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    return require_aware_snapshot_ts(value)


def early_week_due(games: pd.DataFrame, run_date_et: date) -> tuple[int, int] | None:
    """The NFL week that is decided on *run_date_et*, if any.

    Args:
        games: Silver ``games`` rows: ``season``, ``week``, ``game_type`` and ``kickoff_et``.
        run_date_et: The ET calendar date of the run.

    Returns:
        ``(season, week)`` for the week of ``EARLY_FROM``'s season, at or after its first week,
        with every game a regular-season game, whose decision day is *run_date_et*. None when
        no week qualifies.

    Raises:
        lock_rule.MissingKickoffError: When a lower week has a game with no kickoff.
        ValueError: When an eligible week has no week below it in *games*, so the frame is not
            the whole season and cannot say when that week is due.
    """
    season, first_week = EARLY_FROM
    season_games = games.loc[games["season"] == season]
    arm_weeks = sorted(int(w) for w in season_games["week"].unique() if w >= first_week)
    for week in arm_weeks:
        week_types = season_games.loc[season_games["week"] == week, "game_type"]
        if not week_types.eq("REG").all():
            continue
        lower_kickoffs = season_games.loc[season_games["week"] < week, "kickoff_et"]
        if lock_rule.early_decision_date(lower_kickoffs) == run_date_et:
            return season, week
    return None


def _read_boards(path: Path) -> list[EarlyBoard]:
    """Every recorded board, in the order captured. No file yet means no boards."""
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        EarlyBoard(
            captured_at=_require_aware(record["captured_at_utc"]),
            events=record["board"],
        )
        for record in records
    ]


def read_board_for_run(path: Path, run_date_et: date) -> EarlyBoard | None:
    """The last board captured on *run_date_et*, an ET calendar date, or None when there is none.

    Args:
        path: The ``early_lines.jsonl`` file.
        run_date_et: The ET calendar date of the run.
    """
    todays_boards = [
        board
        for board in _read_boards(path)
        if board.captured_at.astimezone(lock_rule.ET).date() == run_date_et
    ]
    return todays_boards[-1] if todays_boards else None


def latest_board(path: Path) -> EarlyBoard | None:
    """The last board recorded on any day, or None. For a rehearsal only, never a decision."""
    boards = _read_boards(path)
    return boards[-1] if boards else None


def _every_outcome_priced(market: Mapping[str, Any]) -> bool:
    """Whether every outcome of *market* has a price; one may lack the key or hold None."""
    return all(
        outcome.get("price") is not None for outcome in market.get("outcomes", [])
    )


def _without_unpriced_markets(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of *events* holding no quoted market with a price missing. *events* is unchanged.

    The shared transform writes -110 for an outcome with no price, so the market is removed
    first. The book then simply has no such market, instead of an invented price.
    """
    return [
        {
            **event,
            "bookmakers": [
                {
                    **book,
                    "markets": [
                        market
                        for market in book.get("markets", [])
                        if market.get("key") not in _QUOTED_MARKETS
                        or _every_outcome_priced(market)
                    ],
                }
                for book in event.get("bookmakers", [])
            ],
        }
        for event in events
    ]


def board_odds(
    board: EarlyBoard, week_games: pd.DataFrame, ingester: OddsDataIngester
) -> pd.DataFrame:
    """The board's validated odds rows for one week's games. Never written anywhere.

    A game whose lock had passed when the board was read is dropped by the transform, and is
    named in ``ingester.last_match_report.post_lock_games``.

    Args:
        board: The board.
        week_games: The silver ``games`` rows of the week being decided.
        ingester: Supplies the transform and the validation.

    Returns:
        One row per bookmaker per matched game; an empty frame with no columns when none matched.
    """
    locks = {
        str(game_id): lock.to_pydatetime()
        for game_id, lock in lock_rule.lock_frame(week_games).items()
    }
    transformed = ingester.transform_odds_data(
        _without_unpriced_markets(board.events),
        schedule=week_games,
        locks=locks,
        captured_at=board.captured_at,
        capture_kind="decision",
    )
    # OddsSchema gives a juice column that no row has its -110 default. Present but blank, it
    # stays blank, whatever the rest of the board holds.
    every_quote_column = transformed.columns.union(
        [column for columns in _QUOTE_COLUMNS.values() for column in columns],
        sort=False,
    )
    return ingester.validate_odds_data(transformed.reindex(columns=every_quote_column))


def _line_of(value: Any) -> float:
    """A row's line as a float. A book without the market has none: NaN, which is never priced."""
    return float("nan") if pd.isna(value) else float(value)


def _price_of(value: Any) -> int | None:
    """A row's American price as an int, or None. These columns can be float in a DataFrame."""
    return None if pd.isna(value) else int(value)


def _freshness_of(market_update: Any, book_update: Any) -> datetime | None:
    """The bookmaker's own time for a row: the latest per-market stamp, else the book's stamp."""
    stamp = book_update if pd.isna(market_update) else market_update
    return None if pd.isna(stamp) else _require_aware(stamp)


def _is_fresher(candidate: Quote, held: Quote) -> bool:
    """Whether *candidate* is strictly newer than *held*. A quote with no time never is."""
    if candidate.last_update is None:
        return False
    return held.last_update is None or candidate.last_update > held.last_update


def quotes_for(odds: pd.DataFrame, target: str) -> dict[str, list[Quote]]:
    """Each game's quotes for one target: exactly one per book, because the pricing counts books.

    When a book appears twice for a game, the row with the latest bookmaker time is kept; between
    rows with the same time, the first.

    Args:
        odds: The frame from :func:`board_odds`.
        target: ``ats`` or ``ou``.

    Returns:
        Game id to quotes. A game with no row on the board is absent.

    Raises:
        KeyError: When ``target`` has no early arm.
    """
    line_column, up_column, down_column = _QUOTE_COLUMNS[target]
    if odds.empty:
        return {}

    freshest: dict[tuple[str, str], Quote] = {}
    for row in odds.to_dict("records"):
        quote = Quote(
            sportsbook=str(row["sportsbook"]),
            line=_line_of(row[line_column]),
            price_up=_price_of(row[up_column]),
            price_down=_price_of(row[down_column]),
            last_update=_freshness_of(row["market_last_update"], row["last_update"]),
        )
        key = (str(row["game_id"]), quote.sportsbook)
        if key not in freshest or _is_fresher(quote, freshest[key]):
            freshest[key] = quote

    quotes_by_game: defaultdict[str, list[Quote]] = defaultdict(list)
    for (game_id, _book), quote in freshest.items():
        quotes_by_game[game_id].append(quote)
    return dict(quotes_by_game)
