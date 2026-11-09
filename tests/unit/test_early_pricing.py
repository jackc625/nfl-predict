"""The early-week pricing module takes a real quote, values it, and sizes and caps the stake.

``backtest/early_pricing.py`` is pure, so these tests hand it quotes built right here. They pin
which quotes may be the price at all, that the best one is chosen by expected value, that win and
lose are the right way round for each side, and that Kelly and the two caps give the recorded
numbers.

No file is written. The one committed file read is the fitted key-number table.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest import early_pricing
from backtest.early_arm_2026 import (
    EXPECTED_MOVE_SLOPE,
    KEY_NUMBER_TABLES_PATH,
    MAX_QUOTE_DISTANCE_POINTS,
    QUOTE_MAX_AGE_HOURS,
    REGULATED_BOOKS,
)
from backtest.early_pricing import (
    PricedSide,
    Quote,
    apply_caps,
    assert_no_contradiction,
    book_label,
    price_target,
    quarter_kelly_fraction,
    usable_quotes,
)
from models.key_numbers import load_key_number_tables, outcome_chance

REPO_ROOT = Path(__file__).resolve().parents[2]

NOW = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)
TABLES = load_key_number_tables(REPO_ROOT / KEY_NUMBER_TABLES_PATH)


def q(
    book: str,
    line: float = 44.5,
    up: Any = -110,
    down: Any = -110,
    age_h: float = 1,
) -> Quote:
    """A quote from ``book``, last updated ``age_h`` hours before the board was read."""
    return Quote(book, line, up, down, NOW - timedelta(hours=age_h))


FOUR_BOOKS = [q(book) for book in REGULATED_BOOKS]
SPREAD_BOOKS_AT_3_5 = [q(book, line=3.5) for book in REGULATED_BOOKS]


def priced_side(target: str, model_value: float, quotes: Sequence[Quote]) -> PricedSide:
    """Price a target the test expects to reach a side, and say so plainly if it does not."""
    result = price_target(target, model_value, quotes, TABLES[target], captured_at=NOW)
    assert isinstance(result, PricedSide), f"expected a priced side, got {result!r}"
    return result


def bet(game_id: str, target: str, side: str) -> dict[str, str]:
    """A selected bet record, carrying the fields the contradiction check reads."""
    return {"game_id": game_id, "target": target, "bet_side": side}


# ---------------------------------------------------------------------------
# Which quotes may be the price
# ---------------------------------------------------------------------------


def test_an_offshore_book_is_never_used() -> None:
    priced = priced_side(
        "ou", 53.0, [q("draftkings"), q("fanduel"), q("bovada", up=120)]
    )
    assert priced.sportsbook != "bovada"


def test_a_stale_quote_is_ignored() -> None:
    priced = priced_side(
        "ou", 53.0, [q("draftkings"), q("fanduel"), q("betmgm", up=120, age_h=7)]
    )
    assert priced.sportsbook != "betmgm"


def test_a_quote_exactly_at_the_age_limit_is_usable() -> None:
    quote = q("fanduel", age_h=QUOTE_MAX_AGE_HOURS)
    assert usable_quotes([quote], NOW) == [quote]


def test_a_quote_one_second_past_the_age_limit_is_not_usable() -> None:
    quote = q("fanduel", age_h=QUOTE_MAX_AGE_HOURS + 1 / 3600)
    assert usable_quotes([quote], NOW) == []


def test_a_quote_stamped_after_the_read_is_usable() -> None:
    """A bookmaker's clock can run ahead of ours; that is not staleness."""
    quote = q("fanduel", age_h=-1)
    assert usable_quotes([quote], NOW) == [quote]


@pytest.mark.parametrize("missing_time", [None, pd.NaT])
def test_a_quote_with_no_bookmaker_time_is_not_usable(missing_time: Any) -> None:
    assert usable_quotes([Quote("fanduel", 44.5, -110, -110, missing_time)], NOW) == []


@pytest.mark.parametrize(
    ("up", "down"),
    [
        (None, -110),
        (-110, None),
        (float("nan"), -110),
        (-110, float("nan")),
        (pd.NA, -110),
        (-110, pd.NA),
    ],
)
def test_a_book_missing_either_price_is_not_usable(up: Any, down: Any) -> None:
    assert usable_quotes([q("fanduel", up=up, down=down)], NOW) == []


def test_the_usable_quotes_keep_their_order() -> None:
    quotes = [q("fanduel"), q("bovada"), q("betmgm")]
    assert [quote.sportsbook for quote in usable_quotes(quotes, NOW)] == [
        "fanduel",
        "betmgm",
    ]


# ---------------------------------------------------------------------------
# When a game cannot be priced
# ---------------------------------------------------------------------------


def test_one_book_is_no_market() -> None:
    assert (
        price_target("ou", 53.0, [q("draftkings")], TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


def test_no_disagreement_is_no_side() -> None:
    assert (
        price_target(
            "ou", 44.5, [q("draftkings"), q("fanduel")], TABLES["ou"], captured_at=NOW
        )
        == "no_bet_side"
    )


def test_a_quote_with_no_time_is_ignored_when_enough_books_remain() -> None:
    no_time = Quote("betmgm", 44.5, 120, -110, None)
    priced = priced_side("ou", 53.0, [q("draftkings"), q("fanduel"), no_time])
    assert priced.sportsbook != "betmgm"


def test_a_quote_with_no_time_does_not_count_toward_the_minimum() -> None:
    no_time = Quote("fanduel", 44.5, -110, -110, None)
    assert (
        price_target(
            "ou", 53.0, [q("draftkings"), no_time], TABLES["ou"], captured_at=NOW
        )
        == "missing_snapshot"
    )


def test_a_zero_total_is_ignored_when_enough_books_remain() -> None:
    """The bad book neither gets picked nor pulls the median down (44.5 if it counted)."""
    quotes = [
        q("draftkings", line=44.5),
        q("fanduel", line=45.5),
        q("betmgm", line=0.0),
    ]
    priced = priced_side("ou", 53.0, quotes)
    assert priced.sportsbook != "betmgm"
    assert priced.consensus == 45.0


def test_a_zero_total_does_not_count_toward_the_minimum() -> None:
    quotes = [q("draftkings"), q("fanduel", line=0.0)]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


@pytest.mark.parametrize("bad_line", [120.0, 130.5, -3.0, float("nan")])
def test_a_line_the_table_cannot_price_drops_only_that_book(bad_line: float) -> None:
    quotes = [q("draftkings"), q("fanduel"), q("betmgm", line=bad_line, up=120)]
    assert priced_side("ou", 53.0, quotes).sportsbook != "betmgm"


def test_a_target_with_no_early_arm_is_refused() -> None:
    with pytest.raises(KeyError, match="wp"):
        price_target("wp", 0.6, FOUR_BOOKS, TABLES["ou"], captured_at=NOW)


@pytest.mark.parametrize("model_value", [float("nan"), float("inf")])
def test_a_model_value_that_is_not_a_number_is_refused(model_value: float) -> None:
    with pytest.raises(ValueError, match="model_value"):
        price_target("ou", model_value, FOUR_BOOKS, TABLES["ou"], captured_at=NOW)


# ---------------------------------------------------------------------------
# A quote far from the others
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("far_line", "model_value"),
    [(38.5, 53.0), (50.5, 36.0)],
)
def test_a_far_book_is_never_chosen_and_the_consensus_comes_from_the_others(
    far_line: float, model_value: float
) -> None:
    """The far book holds the best number for the side, and the four-book median is not 44.5."""
    quotes = [
        q("draftkings", line=44.0),
        q("fanduel", line=44.5),
        q("betmgm", line=45.0),
        q("betrivers", line=far_line),
    ]
    priced = priced_side("ou", model_value, quotes)
    assert priced.sportsbook != "betrivers"
    assert priced.consensus == 44.5


@pytest.mark.parametrize(
    ("line", "model_value"),
    [
        (44.5 + MAX_QUOTE_DISTANCE_POINTS, 36.0),
        (44.5 - MAX_QUOTE_DISTANCE_POINTS, 53.0),
    ],
)
def test_a_quote_exactly_at_the_distance_limit_stays(
    line: float, model_value: float
) -> None:
    """The book at the limit holds the best number for the side, so being chosen proves it stayed."""
    quotes = [q("draftkings"), q("fanduel"), q("betmgm", line=line)]
    assert priced_side("ou", model_value, quotes).sportsbook == "betmgm"


@pytest.mark.parametrize(
    ("line", "model_value"),
    [
        (44.5 + MAX_QUOTE_DISTANCE_POINTS + 0.25, 36.0),
        (44.5 - MAX_QUOTE_DISTANCE_POINTS - 0.25, 53.0),
    ],
)
def test_a_quote_just_past_the_distance_limit_is_dropped(
    line: float, model_value: float
) -> None:
    quotes = [q("draftkings"), q("fanduel"), q("betmgm", line=line)]
    priced = priced_side("ou", model_value, quotes)
    assert priced.sportsbook != "betmgm"
    assert priced.consensus == 44.5


def test_two_books_four_points_apart_are_no_market() -> None:
    """The median sits 2.0 from each, so both drop."""
    quotes = [q("draftkings", line=44.5), q("fanduel", line=48.5)]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


def test_a_lone_survivor_of_the_distance_rule_is_no_market() -> None:
    quotes = [
        q("draftkings", line=40.5),
        q("fanduel", line=44.5),
        q("betmgm", line=48.5),
    ]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


@pytest.mark.parametrize(
    "quotes",
    [[], [q("bovada"), q("betonlineag")]],
    ids=["no_quotes", "no_usable_quotes"],
)
def test_nothing_to_take_a_median_of_is_no_market(quotes: list[Quote]) -> None:
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


@pytest.mark.parametrize(
    ("target", "model_value", "good_line", "bad_line"),
    [("ats", 6.0, 3.5, 0.0), ("ou", 46.5, 44.5, 30.5)],
    ids=["spread_with_a_zero_line", "total_far_below_the_rest"],
)
def test_the_measured_outliers_no_longer_make_an_edge(
    target: str, model_value: float, good_line: float, bad_line: float
) -> None:
    """These two boards priced at EV +0.2025 and +0.6546 at the bad book before the rule."""
    quotes = [
        q("betmgm", line=good_line),
        q("betrivers", line=good_line),
        q("draftkings", line=good_line),
        q("fanduel", line=bad_line),
    ]
    priced = priced_side(target, model_value, quotes)
    assert priced.sportsbook != "fanduel"
    assert priced.line == good_line
    assert priced.ev < 0


def test_a_recorded_wrong_way_spread_is_never_the_price() -> None:
    """FanDuel's NE@DET at -1.5 against about +1.5, on the boards of 2026-11-05 to 2026-11-08."""
    quotes = [
        q("betmgm", line=1.5),
        q("betrivers", line=1.5),
        q("draftkings", line=1.5),
        q("fanduel", line=-1.5),
    ]
    priced = priced_side("ats", 4.0, quotes)
    assert priced.sportsbook != "fanduel"
    assert priced.consensus == 1.5


def test_two_books_too_far_apart_are_no_market() -> None:
    """DraftKings +1.5 against FanDuel -1.5: with two quotes, nothing says which is wrong.

    The median of two quotes is their midpoint, so each is 1.5 from it and the distance rule
    alone keeps both.
    """
    quotes = [q("draftkings", line=1.5), q("fanduel", line=-1.5)]
    assert (
        price_target("ats", 4.0, quotes, TABLES["ats"], captured_at=NOW)
        == "missing_snapshot"
    )


def test_two_books_just_past_the_limit_are_no_market() -> None:
    quotes = [
        q("draftkings", line=44.0),
        q("fanduel", line=44.0 + MAX_QUOTE_DISTANCE_POINTS + 0.25),
    ]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


@pytest.mark.parametrize(
    "gap", [MAX_QUOTE_DISTANCE_POINTS, 1.0], ids=["exactly_at_the_limit", "one_point"]
)
def test_two_books_close_enough_still_price(gap: float) -> None:
    quotes = [q("draftkings", line=44.0), q("fanduel", line=44.0 + gap)]
    assert priced_side("ou", 53.0, quotes).consensus == 44.0 + gap / 2


def test_two_books_left_by_the_distance_drop_are_held_to_the_same_limit() -> None:
    """The median is 45.25: 40.0 and 50.0 drop, and 44.0 and 46.5 are left 2.5 apart."""
    quotes = [
        q("betmgm", line=40.0),
        q("draftkings", line=44.0),
        q("fanduel", line=46.5),
        q("betrivers", line=50.0),
    ]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


def test_two_books_left_by_the_range_check_are_held_to_the_same_limit() -> None:
    """0.0 goes to the range check, which leaves 44.5 and 46.5, 2.0 apart."""
    quotes = [
        q("betmgm", line=0.0),
        q("draftkings", line=44.5),
        q("fanduel", line=46.5),
    ]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


def test_three_books_wider_than_the_limit_still_price() -> None:
    """The two-quote rule is for exactly two: with three, the median can say which is wrong.

    The widest pair is listed first, so a rule looking at the first two would refuse this board.
    """
    quotes = [
        q("draftkings", line=44.0),
        q("betmgm", line=47.0),
        q("fanduel", line=45.5),
    ]
    assert priced_side("ou", 53.0, quotes).consensus == 45.5


# ---------------------------------------------------------------------------
# The table's range comes before the distance rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_line", [0.0, 120.0, 130.5, -3.0])
def test_every_book_on_a_line_the_table_cannot_price_is_no_market(
    bad_line: float,
) -> None:
    """The books agree, so the distance rule keeps them; only the range check stops a raise."""
    quotes = [q("draftkings", line=bad_line), q("fanduel", line=bad_line)]
    assert (
        price_target("ou", 53.0, quotes, TABLES["ou"], captured_at=NOW)
        == "missing_snapshot"
    )


def test_a_line_the_table_cannot_price_never_counts_in_the_first_median() -> None:
    """Counted, the first median is 44.25, 46.0 is 1.75 from it and drops, and 44.25 is the consensus.

    Left out, the median is 44.5 and 46.0 is exactly at the limit, so it stays and is the best
    number for the under.
    """
    quotes = [
        q("betmgm", line=0.0),
        q("draftkings", line=44.0),
        q("fanduel", line=44.5),
        q("betrivers", line=46.0),
    ]
    priced = priced_side("ou", 36.0, quotes)
    assert priced.consensus == 44.5
    assert priced.sportsbook == "betrivers"


# ---------------------------------------------------------------------------
# The side, the expected close and the chances
# ---------------------------------------------------------------------------


def test_a_two_point_gap_does_not_clear_the_cut() -> None:
    assert priced_side("ou", 46.5, FOUR_BOOKS).ev < 0


def test_an_eight_point_gap_clears_the_cut() -> None:
    priced = priced_side("ou", 52.5, FOUR_BOOKS)
    assert priced.side == "over" and priced.ev > 0


def test_a_model_below_the_consensus_takes_the_under() -> None:
    assert priced_side("ou", 36.5, FOUR_BOOKS).side == "under"


def test_a_spread_side_uses_the_home_positive_scale() -> None:
    priced = priced_side("ats", 6.0, SPREAD_BOOKS_AT_3_5)
    assert priced.side == "home_cover" and priced.line == 3.5


def test_a_model_below_the_spread_takes_the_away_side() -> None:
    priced = priced_side("ats", 1.0, SPREAD_BOOKS_AT_3_5)
    assert priced.side == "away_cover" and priced.line == 3.5


def test_the_consensus_is_the_median_and_sets_the_expected_close() -> None:
    """RE-POINTED 2026-11-09 for backtest/early_arm_2026.py: the top book moves from 47.0 to 45.5.

    47.0 is now further than MAX_QUOTE_DISTANCE_POINTS from the median and would be dropped.
    """
    quotes = [
        q("draftkings", line=44.0),
        q("fanduel", line=44.5),
        q("betmgm", line=45.5),
    ]
    priced = priced_side("ou", 53.0, quotes)
    assert priced.consensus == 44.5
    assert priced.expected_close == pytest.approx(
        44.5 + EXPECTED_MOVE_SLOPE["ou"] * (53.0 - 44.5)
    )


def test_the_median_of_an_even_number_of_books_is_their_midpoint() -> None:
    priced = priced_side("ou", 53.0, [q("draftkings", 44.5), q("fanduel", 45.5)])
    assert priced.consensus == 45.0


def test_going_up_wins_above_and_loses_below() -> None:
    quotes = [q("draftkings", line=45.0), q("fanduel", line=45.0)]
    priced = priced_side("ou", 53.0, quotes)

    chance = outcome_chance(
        TABLES["ou"],
        market_line=45.0,
        center=priced.expected_close,
        threshold=45.0,
    )
    assert chance.at > 0, "a whole-number line should carry a push"
    assert priced.p_win == pytest.approx(chance.above)
    assert priced.p_push == pytest.approx(chance.at)
    assert priced.p_lose == pytest.approx(chance.below)


def test_going_down_wins_below_and_loses_above() -> None:
    quotes = [q("draftkings", line=45.0), q("fanduel", line=45.0)]
    priced = priced_side("ou", 37.0, quotes)

    chance = outcome_chance(
        TABLES["ou"],
        market_line=45.0,
        center=priced.expected_close,
        threshold=45.0,
    )
    assert priced.side == "under"
    assert priced.p_win == pytest.approx(chance.below)
    assert priced.p_push == pytest.approx(chance.at)
    assert priced.p_lose == pytest.approx(chance.above)


def test_an_away_side_on_an_away_favourite_wins_below_the_negative_line() -> None:
    quotes = [q("draftkings", line=-3.5), q("fanduel", line=-3.5)]
    priced = priced_side("ats", -6.0, quotes)

    chance = outcome_chance(
        TABLES["ats"],
        market_line=-3.5,
        center=priced.expected_close,
        threshold=-3.5,
    )
    assert priced.side == "away_cover" and priced.line == -3.5
    assert priced.p_win == pytest.approx(chance.below)
    assert priced.p_lose == pytest.approx(chance.above)


def test_a_book_off_the_consensus_is_compared_with_its_own_line() -> None:
    """The consensus is what counts as fair; the bet is still graded against the book's line."""
    quotes = [
        q("draftkings", line=44.5),
        q("fanduel", line=45.5),
        q("betmgm", line=45.5),
    ]
    priced = priced_side("ou", 53.0, quotes)

    chance = outcome_chance(
        TABLES["ou"],
        market_line=45.5,
        center=priced.expected_close,
        threshold=44.5,
    )
    assert priced.sportsbook == "draftkings" and priced.line == 44.5
    assert priced.p_win == pytest.approx(chance.above)


def test_the_expected_value_is_the_payout_weighed_against_the_loss() -> None:
    priced = priced_side("ou", 53.0, [q("draftkings", up=-105), q("fanduel", up=-105)])
    assert priced.ev == pytest.approx(priced.p_win * (100 / 105) - priced.p_lose)


@pytest.mark.parametrize(
    ("model_value", "expected_threshold", "wins_above"),
    [(53.0, 45.5, True), (37.0, 44.5, False)],
)
def test_slippage_moves_the_threshold_against_the_bettor(
    monkeypatch: pytest.MonkeyPatch,
    model_value: float,
    expected_threshold: float,
    wins_above: bool,
) -> None:
    monkeypatch.setattr(early_pricing, "SLIPPAGE_POINTS", 0.5)
    quotes = [q("draftkings", line=45.0), q("fanduel", line=45.0)]

    priced = priced_side("ou", model_value, quotes)

    chance = outcome_chance(
        TABLES["ou"],
        market_line=45.0,
        center=priced.expected_close,
        threshold=expected_threshold,
    )
    assert priced.line == 45.0, "slippage changes the chances, never the quoted line"
    assert priced.p_win == pytest.approx(chance.above if wins_above else chance.below)


# ---------------------------------------------------------------------------
# The best quote
# ---------------------------------------------------------------------------


def test_the_better_price_wins() -> None:
    priced = priced_side("ou", 53.0, [q("draftkings"), q("fanduel", up=-105)])
    assert priced.sportsbook == "fanduel" and priced.price == -105


def test_the_price_is_read_from_the_side_being_bet() -> None:
    priced = priced_side("ou", 36.5, [q("draftkings"), q("fanduel", up=120, down=-105)])
    assert priced.side == "under"
    assert priced.sportsbook == "fanduel" and priced.price == -105


def test_a_lower_total_is_the_better_over() -> None:
    priced = priced_side("ou", 53.0, [q("draftkings", 45.5), q("fanduel", 44.5)])
    assert priced.sportsbook == "fanduel" and priced.line == 44.5


def test_a_higher_total_is_the_better_under() -> None:
    priced = priced_side("ou", 37.0, [q("draftkings", 45.5), q("fanduel", 44.5)])
    assert priced.sportsbook == "draftkings" and priced.line == 45.5


def test_a_tie_goes_to_the_lower_book_name() -> None:
    assert priced_side("ou", 53.0, [q("fanduel"), q("draftkings")]).sportsbook == (
        "draftkings"
    )


# ---------------------------------------------------------------------------
# Sizing and caps
# ---------------------------------------------------------------------------


def test_quarter_kelly_value() -> None:
    assert quarter_kelly_fraction(0.55, 0.45, -110) == pytest.approx(0.01375)
    assert quarter_kelly_fraction(0.50, 0.50, -110) == 0.0


def test_kelly_measures_the_edge_only_where_money_moves() -> None:
    """With a chance of a push, p + q is below 1 and the edge is divided by that smaller sum."""
    assert quarter_kelly_fraction(0.50, 0.40, 100) == pytest.approx(
        0.25 * ((1.0 * 0.50 - 0.40) / (1.0 * (0.50 + 0.40)))
    )


def test_the_game_cap_scales_a_game() -> None:
    assert apply_caps([("g", 0.04), ("g", 0.04)]) == pytest.approx([0.025, 0.025])


def test_the_week_cap_scales_the_week() -> None:
    assert apply_caps([(f"g{i}", 0.04) for i in range(4)]) == pytest.approx([0.025] * 4)


def test_stakes_under_both_caps_are_untouched() -> None:
    assert apply_caps([("a", 0.01), ("b", 0.02)]) == pytest.approx([0.01, 0.02])


def test_the_game_cap_keeps_a_games_stakes_in_proportion() -> None:
    assert apply_caps([("g", 0.06), ("g", 0.02)]) == pytest.approx([0.0375, 0.0125])


def test_the_game_cap_runs_before_the_week_cap() -> None:
    """Week-first would give [0.05, 0.0333]: 0.15 scaled to 0.10, then game a capped."""
    assert apply_caps([("a", 0.10), ("b", 0.05)]) == pytest.approx([0.05, 0.05])


def test_no_stakes_and_zero_stakes_are_handled() -> None:
    assert apply_caps([]) == []
    assert apply_caps([("a", 0.0), ("b", 0.0)]) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# No contradiction, and the book's name on the row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spread_side", ["home_cover", "away_cover"])
@pytest.mark.parametrize("total_side", ["over", "under"])
def test_a_spread_and_a_total_never_contradict(
    spread_side: str, total_side: str
) -> None:
    assert_no_contradiction([bet("g", "ats", spread_side), bet("g", "ou", total_side)])


@pytest.mark.parametrize(
    ("target", "first", "second"),
    [("ou", "over", "under"), ("ats", "home_cover", "away_cover")],
)
def test_both_sides_of_one_target_are_refused(
    target: str, first: str, second: str
) -> None:
    with pytest.raises(ValueError, match="g1"):
        assert_no_contradiction([bet("g1", target, first), bet("g1", target, second)])


@pytest.mark.parametrize(
    ("win_side", "spread_side"), [("home", "away_cover"), ("away", "home_cover")]
)
def test_a_win_bet_and_a_spread_bet_on_opposite_teams_are_refused(
    win_side: str, spread_side: str
) -> None:
    with pytest.raises(ValueError, match="g1"):
        assert_no_contradiction(
            [bet("g1", "wp", win_side), bet("g1", "ats", spread_side)]
        )


def test_a_win_bet_and_a_spread_bet_on_the_same_team_agree() -> None:
    assert_no_contradiction([bet("g", "wp", "home"), bet("g", "ats", "home_cover")])


def test_opposite_sides_on_different_games_agree() -> None:
    assert_no_contradiction([bet("g1", "ou", "over"), bet("g2", "ou", "under")])


def test_no_bets_agree() -> None:
    assert_no_contradiction([])


def test_the_book_label_is_what_the_ledger_stores() -> None:
    assert book_label("fanduel") == "book:fanduel"
