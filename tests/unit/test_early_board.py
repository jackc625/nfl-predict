"""Which NFL week is due on a given day, and what a recorded early-week board offered.

``backtest/early_board.py`` decides the Tuesday a week is decided on and turns the recorded board
into per-book quotes. These tests pin the calendar against the real shape of the 2026 schedule,
read boards from files written under ``tmp_path``, and send every board through the real
``OddsDataIngester`` transform, so the sign of a stored spread is the one the ingest really stores.

Nothing is written under ``data/``, and no production store is read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import copy
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from backtest.early_board import (
    EarlyBoard,
    board_odds,
    early_week_due,
    latest_board,
    quotes_for,
    read_board_for_run,
)
from backtest.early_pricing import Quote, usable_quotes
from scripts.ingest_odds import OddsDataIngester


def _game(
    week: int,
    away: str,
    home: str,
    kickoff: str,
    *,
    game_type: str = "REG",
    season: int = 2026,
) -> dict[str, Any]:
    """One silver ``games`` row, carrying the columns the calendar and the transform read."""
    return {
        "game_id": f"{season}_W{week:02d}_{away}@{home}",
        "season": season,
        "week": week,
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
        "kickoff_et": pd.Timestamp(kickoff),
    }


# Each week ends on a Monday night game, as the real 2026 weeks 9 to 11 do.
SCHEDULE = pd.DataFrame(
    [
        _game(9, "BUF", "MIA", "2026-11-10T01:15Z"),  # Mon 11-09, 20:15 ET
        _game(10, "GB", "CHI", "2026-11-15T18:00Z"),
        _game(10, "DAL", "PHI", "2026-11-17T01:15Z"),  # Mon 11-16, 20:15 ET
        _game(11, "NE", "DET", "2026-11-22T18:00Z"),
        _game(11, "SF", "SEA", "2026-11-24T01:15Z"),  # Mon 11-23, 20:15 ET
        _game(12, "KC", "DEN", "2026-11-29T18:00Z"),
    ]
)

# A week below the arm, so every week of the arm has a week below it, then the season's last.
LAST_REGULAR_WEEKS = [
    _game(10, "DAL", "PHI", "2026-11-17T01:15Z"),
    _game(18, "BUF", "MIA", "2027-01-10T18:00Z"),
]

GAME = "2026_W11_NE@DET"
GAMES = pd.DataFrame([_game(11, "NE", "DET", "2026-11-22T18:00Z")])
READ_AT = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)
MARKET_STAMP = "2026-11-17T21:50:00Z"
BOOK_STAMP = "2026-11-17T21:40:00Z"

INGESTER = OddsDataIngester.__new__(OddsDataIngester)
INGESTER.sportsbook_priority = []


def _spreads(home_point: float, home_price: int, away_price: int, stamp: Any) -> dict:
    """A spreads market, with its outcomes named as the Odds API names them."""
    return {
        "key": "spreads",
        "last_update": stamp,
        "outcomes": [
            {"name": "Detroit Lions", "price": home_price, "point": home_point},
            {
                "name": "New England Patriots",
                "price": away_price,
                "point": -home_point,
            },
        ],
    }


def _totals(line: float, over_price: int, under_price: int, stamp: Any) -> dict:
    """A totals market."""
    return {
        "key": "totals",
        "last_update": stamp,
        "outcomes": [
            {"name": "Over", "price": over_price, "point": line},
            {"name": "Under", "price": under_price, "point": line},
        ],
    }


def _book(
    key: str = "fanduel",
    *,
    home_point: float = -3.5,
    market_stamp: Any = MARKET_STAMP,
    book_stamp: Any = BOOK_STAMP,
) -> dict[str, Any]:
    """One bookmaker's entry: Detroit at home at ``home_point``, -105 home and -115 away."""
    return {
        "key": key,
        "last_update": book_stamp,
        "markets": [
            _spreads(home_point, -105, -115, market_stamp),
            _totals(47.5, -108, -112, market_stamp),
        ],
    }


def _board(*bookmakers: dict[str, Any], captured_at: datetime = READ_AT) -> EarlyBoard:
    """A board holding one event, New England at Detroit."""
    return EarlyBoard(
        captured_at=captured_at,
        events=[
            {
                "id": "abc123",
                "commence_time": "2026-11-22T18:00:00Z",
                "home_team": "Detroit Lions",
                "away_team": "New England Patriots",
                "bookmakers": list(bookmakers),
            }
        ],
    )


def _only_quote(board: EarlyBoard, target: str = "ats") -> Quote:
    """The single quote the board holds for the game."""
    quotes = quotes_for(board_odds(board, GAMES, INGESTER), target)[GAME]
    assert len(quotes) == 1
    return quotes[0]


def _record(captured_at_utc: str, marker: str) -> dict[str, Any]:
    """One line of ``early_lines.jsonl``, in the shape the capture script writes."""
    return {
        "captured_at_utc": captured_at_utc,
        "remaining_credits_before": 400,
        "events": 1,
        "board": [{"id": marker}],
    }


def _write_records(path: Path, *records: dict[str, Any]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The decision day
# ---------------------------------------------------------------------------


def test_the_day_after_a_monday_night_game() -> None:
    assert lock_rule.early_decision_date([pd.Timestamp("2026-11-17T01:15Z")]) == date(
        2026, 11, 17
    )


def test_the_day_after_a_sunday_night_game() -> None:
    assert lock_rule.early_decision_date([pd.Timestamp("2026-11-16T01:20Z")]) == date(
        2026, 11, 16
    )


def test_the_last_kickoff_decides_wherever_it_sits_in_the_list() -> None:
    kickoffs = [
        pd.Timestamp("2026-11-15T18:00Z"),
        pd.Timestamp("2026-11-17T01:15Z"),
        pd.Timestamp("2026-11-13T01:15Z"),
    ]
    assert lock_rule.early_decision_date(kickoffs) == date(2026, 11, 17)


def test_a_series_of_kickoffs_is_accepted() -> None:
    assert lock_rule.early_decision_date(GAMES["kickoff_et"]) == date(2026, 11, 23)


def test_a_naive_kickoff_is_refused() -> None:
    with pytest.raises(ValueError, match="NO timezone"):
        lock_rule.early_decision_date([datetime(2026, 11, 16, 20, 15)])


def test_one_naive_kickoff_among_aware_ones_is_refused() -> None:
    with pytest.raises(ValueError, match="NO timezone"):
        lock_rule.early_decision_date(
            [pd.Timestamp("2026-11-15T18:00Z"), pd.Timestamp("2026-11-16 20:15")]
        )


@pytest.mark.parametrize(
    "missing", [None, pd.NaT, float("nan")], ids=["none", "nat", "nan"]
)
def test_a_null_kickoff_is_refused_by_name(missing: object) -> None:
    with pytest.raises(lock_rule.MissingKickoffError):
        lock_rule.early_decision_date([pd.Timestamp("2026-11-15T18:00Z"), missing])


def test_no_kickoffs_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no kickoffs"):
        lock_rule.early_decision_date([])


# ---------------------------------------------------------------------------
# Which week is due
# ---------------------------------------------------------------------------


def test_week_11_is_due_on_tuesday_11_17() -> None:
    assert early_week_due(SCHEDULE, date(2026, 11, 17)) == (2026, 11)


def test_the_week_after_the_first_is_due_on_its_own_tuesday() -> None:
    assert early_week_due(SCHEDULE, date(2026, 11, 24)) == (2026, 12)


def test_a_missed_tuesday_is_never_back_filled() -> None:
    assert early_week_due(SCHEDULE, date(2026, 11, 18)) is None


def test_week_10_is_before_the_arm() -> None:
    # Week 10's own decision day: the last game below it, week 9's, was Monday 11-09.
    assert early_week_due(SCHEDULE, date(2026, 11, 10)) is None


def test_a_postseason_week_is_never_due() -> None:
    def schedule(wild_card_type: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                *LAST_REGULAR_WEEKS,
                _game(19, "BUF", "KC", "2027-01-16T21:30Z", game_type=wild_card_type),
            ]
        )

    decision_day = date(2027, 1, 11)
    assert early_week_due(schedule("WC"), decision_day) is None
    # The control: nothing but the game type keeps that week from being due.
    assert early_week_due(schedule("REG"), decision_day) == (2026, 19)


def test_a_week_holding_one_non_regular_game_is_not_due() -> None:
    schedule = pd.DataFrame(
        [
            *LAST_REGULAR_WEEKS,
            _game(19, "BUF", "KC", "2027-01-16T21:30Z"),
            _game(19, "GB", "DAL", "2027-01-17T18:00Z", game_type="WC"),
        ]
    )
    assert early_week_due(schedule, date(2027, 1, 11)) is None


def test_a_schedule_with_nothing_below_the_arm_is_refused() -> None:
    # A frame cut down to the arm's weeks cannot say when they are due; "not due" would be a guess.
    only_week_11 = pd.DataFrame([_game(11, "NE", "DET", "2026-11-22T18:00Z")])
    with pytest.raises(ValueError, match="no kickoffs"):
        early_week_due(only_week_11, date(2026, 11, 17))


def test_a_lower_week_with_no_kickoff_is_refused() -> None:
    blank = SCHEDULE.copy()
    blank.loc[blank["week"] == 10, "kickoff_et"] = pd.NaT
    with pytest.raises(lock_rule.MissingKickoffError):
        early_week_due(blank, date(2026, 11, 17))


def test_only_the_arms_season_is_ever_due() -> None:
    next_year = pd.DataFrame(
        [
            _game(10, "DAL", "PHI", "2026-11-17T01:15Z", season=2027),
            _game(11, "NE", "DET", "2026-11-22T18:00Z", season=2027),
        ]
    )
    assert early_week_due(next_year, date(2026, 11, 17)) is None


def test_an_earlier_season_does_not_count_as_a_lower_week() -> None:
    # A 2025 game dated later than the 2026 week 10 games would move the day if it were counted.
    with_old_season = pd.concat(
        [
            SCHEDULE,
            pd.DataFrame([_game(10, "NYJ", "NE", "2026-11-19T18:00Z", season=2025)]),
        ],
        ignore_index=True,
    )
    assert early_week_due(with_old_season, date(2026, 11, 17)) == (2026, 11)


def test_a_week_is_returned_as_plain_integers() -> None:
    due = early_week_due(SCHEDULE, date(2026, 11, 17))
    assert due is not None
    assert [type(part) for part in due] == [int, int]


# ---------------------------------------------------------------------------
# Reading the recorded boards
# ---------------------------------------------------------------------------


def test_only_todays_board_is_read(tmp_path: Path) -> None:
    path = tmp_path / "early_lines.jsonl"
    _write_records(
        path,
        _record("2026-11-16T22:00:00.5+00:00", "monday"),
        # 02:00 UTC on the 18th is 21:00 ET on the 17th: the ET date is what counts.
        _record("2026-11-18T02:00:00.5+00:00", "tuesday_night"),
        _record("2026-11-18T22:00:00.5+00:00", "wednesday"),
    )
    board = read_board_for_run(path, date(2026, 11, 17))
    assert board is not None
    assert board.events == [{"id": "tuesday_night"}]
    assert board.captured_at == datetime(2026, 11, 18, 2, 0, 0, 500000, tzinfo=UTC)
    assert read_board_for_run(path, date(2026, 11, 19)) is None


def test_the_last_of_todays_boards_is_used(tmp_path: Path) -> None:
    path = tmp_path / "early_lines.jsonl"
    _write_records(
        path,
        _record("2026-11-17T14:00:00+00:00", "morning"),
        _record("2026-11-17T22:00:00+00:00", "evening"),
    )
    board = read_board_for_run(path, date(2026, 11, 17))
    assert board is not None
    assert board.events == [{"id": "evening"}]


def test_no_file_yet_means_no_board(tmp_path: Path) -> None:
    missing = tmp_path / "early_lines.jsonl"
    assert read_board_for_run(missing, date(2026, 11, 17)) is None
    assert latest_board(missing) is None


def test_the_latest_board_is_the_last_record_whatever_its_day(tmp_path: Path) -> None:
    path = tmp_path / "early_lines.jsonl"
    _write_records(
        path,
        _record("2026-11-16T22:00:00+00:00", "older"),
        _record("2026-11-17T22:00:00+00:00", "newest"),
    )
    board = latest_board(path)
    assert board is not None
    assert board.events == [{"id": "newest"}]


def test_a_capture_time_with_no_zone_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "early_lines.jsonl"
    _write_records(path, _record("2026-11-17T22:00:00", "naive"))
    with pytest.raises(ValueError, match="NO timezone"):
        read_board_for_run(path, date(2026, 11, 17))


# ---------------------------------------------------------------------------
# From the board to quotes, through the real transform
# ---------------------------------------------------------------------------


def test_a_home_favourite_is_stored_positive() -> None:
    # Detroit -3.5 at home, -105 home price, -115 away price.
    quote = quotes_for(board_odds(_board(_book()), GAMES, INGESTER), "ats")[GAME][0]
    assert quote.line == 3.5 and quote.price_up == -105 and quote.price_down == -115
    assert quote.sportsbook == "fanduel"


def test_an_away_favourite_is_stored_negative() -> None:
    assert _only_quote(_board(_book(home_point=3.5))).line == -3.5


def test_a_total_reads_the_over_as_up_and_the_under_as_down() -> None:
    quote = _only_quote(_board(_book()), "ou")
    assert (quote.line, quote.price_up, quote.price_down) == (47.5, -108, -112)


def test_a_target_with_no_early_arm_is_refused() -> None:
    with pytest.raises(KeyError):
        quotes_for(board_odds(_board(_book()), GAMES, INGESTER), "wp")


def test_every_book_gets_its_own_quote() -> None:
    odds = board_odds(_board(_book("fanduel"), _book("betmgm")), GAMES, INGESTER)
    assert sorted(quote.sportsbook for quote in quotes_for(odds, "ats")[GAME]) == [
        "betmgm",
        "fanduel",
    ]


def test_a_book_without_the_market_has_no_prices_and_no_line() -> None:
    no_spreads = _book("betmgm")
    no_spreads["markets"] = [no_spreads["markets"][1]]
    odds = board_odds(_board(_book("fanduel"), no_spreads), GAMES, INGESTER)
    # The missing book's blanks are what turn these columns to float, which Quote must not carry.
    assert odds["spread_ju_home"].dtype == float
    quotes = {quote.sportsbook: quote for quote in quotes_for(odds, "ats")[GAME]}
    assert quotes["betmgm"].price_up is None and quotes["betmgm"].price_down is None
    assert math.isnan(quotes["betmgm"].line)
    assert type(quotes["fanduel"].price_up) is int
    assert type(quotes["fanduel"].price_down) is int
    assert (quotes["fanduel"].price_up, quotes["fanduel"].price_down) == (-105, -115)


def _with_unpriced_outcome(
    book: dict[str, Any], market_key: str, missing: str
) -> dict[str, Any]:
    """*book* with the first outcome of one market lacking its price, as ``key`` or ``none``."""
    market = next(found for found in book["markets"] if found["key"] == market_key)
    if missing == "key":
        del market["outcomes"][0]["price"]
    else:
        market["outcomes"][0]["price"] = None
    return book


def _book_quote(odds: pd.DataFrame, target: str, sportsbook: str) -> Quote:
    """One book's quote for the game."""
    return next(
        quote
        for quote in quotes_for(odds, target)[GAME]
        if quote.sportsbook == sportsbook
    )


_HOW_A_PRICE_GOES_MISSING = pytest.mark.parametrize("missing", ["key", "none"])
# Alone, no row on the board has the market's columns, which is when OddsSchema's -110 applies.
_WHO_ELSE_IS_ON_THE_BOARD = pytest.mark.parametrize(
    "alone", [True, False], ids=["alone", "beside_another_book"]
)


@_HOW_A_PRICE_GOES_MISSING
@_WHO_ELSE_IS_ON_THE_BOARD
def test_a_spread_missing_a_price_is_no_market_and_the_total_is_untouched(
    missing: str, alone: bool
) -> None:
    broken = _with_unpriced_outcome(_book("betmgm"), "spreads", missing)
    books = [broken] if alone else [_book("fanduel"), broken]
    odds = board_odds(_board(*books), GAMES, INGESTER)

    spread = _book_quote(odds, "ats", "betmgm")
    assert spread.price_up is None and spread.price_down is None
    assert math.isnan(spread.line)
    assert usable_quotes([spread], READ_AT) == []

    total = _book_quote(odds, "ou", "betmgm")
    assert (total.line, total.price_up, total.price_down) == (47.5, -108, -112)


@_HOW_A_PRICE_GOES_MISSING
@_WHO_ELSE_IS_ON_THE_BOARD
def test_a_total_missing_a_price_is_no_market_and_the_spread_is_untouched(
    missing: str, alone: bool
) -> None:
    broken = _with_unpriced_outcome(_book("betmgm"), "totals", missing)
    books = [broken] if alone else [_book("fanduel"), broken]
    odds = board_odds(_board(*books), GAMES, INGESTER)

    total = _book_quote(odds, "ou", "betmgm")
    assert total.price_up is None and total.price_down is None
    assert math.isnan(total.line)
    assert usable_quotes([total], READ_AT) == []

    spread = _book_quote(odds, "ats", "betmgm")
    assert (spread.line, spread.price_up, spread.price_down) == (3.5, -105, -115)


def test_another_books_market_survives_a_neighbours_missing_price() -> None:
    broken = _with_unpriced_outcome(_book("betmgm"), "spreads", "key")
    odds = board_odds(_board(_book("fanduel"), broken), GAMES, INGESTER)
    spread = _book_quote(odds, "ats", "fanduel")
    assert (spread.line, spread.price_up, spread.price_down) == (3.5, -105, -115)


@_HOW_A_PRICE_GOES_MISSING
def test_the_board_itself_is_left_as_it_was_read(missing: str) -> None:
    board = _board(
        _book("fanduel"), _with_unpriced_outcome(_book("betmgm"), "spreads", missing)
    )
    as_read = copy.deepcopy(board.events)
    board_odds(board, GAMES, INGESTER)
    assert board.events == as_read


def test_a_side_no_book_offers_stays_blank_and_never_takes_the_schema_default() -> None:
    over_only = _book()
    totals = over_only["markets"][1]
    totals["outcomes"] = [totals["outcomes"][0]]
    quote = _only_quote(_board(over_only), "ou")
    assert (quote.line, quote.price_up, quote.price_down) == (47.5, -108, None)
    assert usable_quotes([quote], READ_AT) == []


def test_a_book_seen_twice_keeps_only_its_freshest_row() -> None:
    older = _book(home_point=-3.5, market_stamp="2026-11-17T20:00:00Z")
    newer = _book(home_point=-4.5, market_stamp="2026-11-17T21:00:00Z")
    for books in ((older, newer), (newer, older)):
        assert _only_quote(_board(*books)).line == 4.5


def test_a_row_with_a_time_beats_a_row_without_one() -> None:
    timed = _book(home_point=-4.5)
    untimed = _book(home_point=-3.5, market_stamp=None, book_stamp=None)
    for books in ((timed, untimed), (untimed, timed)):
        assert _only_quote(_board(*books)).line == 4.5


def test_two_rows_with_the_same_time_keep_the_first() -> None:
    first = _book(home_point=-4.5)
    second = _book(home_point=-3.5)
    assert _only_quote(_board(first, second)).line == 4.5


def test_a_book_on_two_games_keeps_both() -> None:
    two_games = pd.DataFrame(
        [
            _game(11, "NE", "DET", "2026-11-22T18:00Z"),
            _game(11, "SF", "SEA", "2026-11-24T01:15Z"),
        ]
    )
    board = _board(_book())
    board.events.append(
        {
            "id": "def456",
            "commence_time": "2026-11-24T01:15:00Z",
            "home_team": "Seattle Seahawks",
            "away_team": "San Francisco 49ers",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "last_update": BOOK_STAMP,
                    "markets": [_totals(44.5, -110, -110, MARKET_STAMP)],
                }
            ],
        }
    )
    quotes = quotes_for(board_odds(board, two_games, INGESTER), "ou")
    assert {game: len(found) for game, found in quotes.items()} == {
        GAME: 1,
        "2026_W11_SF@SEA": 1,
    }


def test_the_time_is_the_markets_own_stamp_when_there_is_one() -> None:
    assert _only_quote(_board(_book())).last_update == datetime(
        2026, 11, 17, 21, 50, tzinfo=UTC
    )


def test_the_time_falls_back_to_the_books_stamp() -> None:
    assert _only_quote(_board(_book(market_stamp=None))).last_update == datetime(
        2026, 11, 17, 21, 40, tzinfo=UTC
    )


def test_the_time_is_none_only_when_both_stamps_are_null() -> None:
    assert (
        _only_quote(_board(_book(market_stamp=None, book_stamp=None))).last_update
        is None
    )


def test_a_blank_stamp_beside_a_real_one_arrives_as_nat_and_becomes_none() -> None:
    blank = _book("betmgm", market_stamp=None, book_stamp=None)
    odds = board_odds(_board(_book("fanduel"), blank), GAMES, INGESTER)
    assert isinstance(odds["market_last_update"].dtype, pd.DatetimeTZDtype)
    quotes = {quote.sportsbook: quote for quote in quotes_for(odds, "ats")[GAME]}
    assert quotes["betmgm"].last_update is None


def test_the_time_is_a_plain_aware_datetime() -> None:
    stamp = _only_quote(_board(_book())).last_update
    assert isinstance(stamp, datetime) and not isinstance(stamp, pd.Timestamp)
    assert stamp.tzinfo is not None


def test_a_time_with_no_zone_is_refused() -> None:
    odds = pd.DataFrame(
        [
            {
                "game_id": GAME,
                "sportsbook": "fanduel",
                "spread": 3.5,
                "spread_ju_home": -105,
                "spread_ju_away": -115,
                "market_last_update": pd.Timestamp("2026-11-17 21:50"),
                "last_update": pd.NaT,
            }
        ]
    )
    with pytest.raises(ValueError, match="NO timezone"):
        quotes_for(odds, "ats")


# ---------------------------------------------------------------------------
# What the board may not contribute
# ---------------------------------------------------------------------------


def test_a_game_past_its_lock_is_dropped() -> None:
    # Detroit's lock is 18:00 ET on Saturday 11-21, which is 23:00 UTC.
    past_lock = datetime(2026, 11, 21, 23, 0, 1, tzinfo=UTC)
    odds = board_odds(_board(_book(), captured_at=past_lock), GAMES, INGESTER)
    assert odds.empty
    assert quotes_for(odds, "ats") == {}
    assert INGESTER.last_match_report.post_lock_games == (GAME,)


def test_a_board_read_exactly_at_the_lock_is_kept() -> None:
    at_lock = datetime(2026, 11, 21, 23, 0, 0, tzinfo=UTC)
    assert _only_quote(_board(_book(), captured_at=at_lock)).line == 3.5


def test_a_game_outside_the_week_is_left_out() -> None:
    other_week = pd.DataFrame([_game(11, "SF", "SEA", "2026-11-24T01:15Z")])
    odds = board_odds(_board(_book()), other_week, INGESTER)
    assert quotes_for(odds, "ats") == {}
