"""One NFL week's early-arm rows: every game and target, a bet or a refusal that says why.

``backtest/early_decide.py`` is pure, so these tests hand it a week built right here and read the
rows it returns. They pin what a bet row carries, that a declined row keeps what it was priced at,
that a row never priced names no side and no book, and that the stakes are capped across the whole
week.

No file is written. The one committed file read is the fitted key-number table.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from api.cache import BET_LIST_COLUMNS
from backtest import early_decide
from backtest.early_arm_2026 import KEY_NUMBER_TABLES_PATH
from backtest.early_decide import CANDIDATE_COLUMNS, EarlyDecision, decide_early_week
from backtest.early_pricing import (
    PricedSide,
    Quote,
    price_target,
    quarter_kelly_fraction,
)
from backtest.weekly_bet_list import (
    DecidedAfterFreezeError,
    assert_decided_at_before_freeze,
)
from forward_ledger.canonical import frame_digest
from forward_ledger.snapshots import SNAPSHOT_PART_KEYS
from models.key_numbers import load_key_number_tables

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES = load_key_number_tables(REPO_ROOT / KEY_NUMBER_TABLES_PATH)

CAPTURED_AT = datetime(2026, 11, 17, 21, 40, tzinfo=UTC)
DECIDED_AT = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)
LOCK = datetime(2026, 11, 21, 23, 0, tzinfo=UTC)

# The spread's expected move is small, so only a very large gap clears the floor at -105.
SPREAD_LINE = 3.5
SPREAD_MODEL_CLEARING_THE_FLOOR = 21.5
TOTAL_LINE = 44.5
TOTAL_MODEL_BELOW_THE_FLOOR = 46.5
TOTAL_MODEL_CLEARING_THE_FLOOR = 53.0

NEVER_PRICED_COLUMNS = (
    "bet_side",
    "line",
    "slipped_line",
    "calibrated_p_side",
    "per_bet_ev",
    "eligibility_label",
    "selected_odds",
    "flat_stake",
    "ev_tier",
)


def games(*game_ids: str) -> pd.DataFrame:
    """A week's games, as many as named."""
    return pd.DataFrame(
        {
            "game_id": list(game_ids),
            "season": [2026] * len(game_ids),
            "week": [11] * len(game_ids),
        }
    )


def quote(book: str, line: float, up: int = -110, down: int = -110) -> Quote:
    """A fresh quote from ``book``."""
    return Quote(book, line, up, down, CAPTURED_AT - timedelta(hours=1))


def board(line: float) -> list[Quote]:
    """Three books on one line, FanDuel holding the best price for the over or the home side."""
    return [
        quote("draftkings", line),
        quote("fanduel", line, up=-105),
        quote("betmgm", line),
    ]


TWO_GAMES = games("g1", "g2")
BOARDS: dict[str, dict[str, list[Quote]]] = {
    "ats": {game_id: board(SPREAD_LINE) for game_id in ("g1", "g2", "g3")},
    "ou": {game_id: board(TOTAL_LINE) for game_id in ("g1", "g2", "g3")},
}
ARGS: dict[str, Any] = {
    # g2 has no prediction at all. On g1 the spread is a bet and the total is declined.
    "predictions": {
        "ats": {"g1": SPREAD_MODEL_CLEARING_THE_FLOOR},
        "ou": {"g1": TOTAL_MODEL_BELOW_THE_FLOOR},
    },
    "quotes": BOARDS,
    "tables": TABLES,
    "locks": {"g1": LOCK, "g2": LOCK, "g3": LOCK},
    "captured_at": CAPTURED_AT,
    "decided_at": DECIDED_AT,
}
# Both targets of every game clear the floor.
ALL_BETS: dict[str, dict[str, float]] = {
    "ats": dict.fromkeys(("g1", "g2", "g3"), SPREAD_MODEL_CLEARING_THE_FLOOR),
    "ou": dict.fromkeys(("g1", "g2", "g3"), TOTAL_MODEL_CLEARING_THE_FLOOR),
}


def decide(week: pd.DataFrame = TWO_GAMES, **overrides: Any) -> EarlyDecision:
    """The decision for ``week``, with any of the shared arguments replaced."""
    return decide_early_week(week, **{**ARGS, **overrides})


@pytest.fixture
def d() -> EarlyDecision:
    """The two-game week: g1's spread is a bet, g1's total is declined, g2 has no prediction."""
    return decide()


def row_of(decision: EarlyDecision, game_id: str, target: str) -> Any:
    """The one frame row for a game and target."""
    rows = decision.frame[
        (decision.frame["game_id"] == game_id) & (decision.frame["target"] == target)
    ]
    assert len(rows) == 1, f"expected one row for {game_id}/{target}, got {len(rows)}"
    return rows.iloc[0]


def candidate_of(decision: EarlyDecision, game_id: str, target: str) -> Any:
    """The one candidates row for a game and target."""
    rows = decision.candidates[
        (decision.candidates["game_id"] == game_id)
        & (decision.candidates["target"] == target)
    ]
    assert len(rows) == 1, f"expected one row for {game_id}/{target}, got {len(rows)}"
    return rows.iloc[0]


def bet_row(decision: EarlyDecision) -> Any:
    """The week's one bet."""
    return row_of(decision, "g1", "ats")


def declined_row(decision: EarlyDecision) -> Any:
    """The week's one priced row that fell below the floor."""
    return row_of(decision, "g1", "ou")


def stamped_row(decision: EarlyDecision) -> Any:
    """A row as the ledger's write-time check reads it."""
    return decision.frame.iloc[0]


def game_rows(decision: EarlyDecision, game_id: str) -> pd.DataFrame:
    """Every row of one game."""
    return cast("pd.DataFrame", decision.frame[decision.frame["game_id"] == game_id])


def reasons(decision: EarlyDecision, game_id: str) -> set[str]:
    """Every rejection reason a game's rows carry."""
    return set(game_rows(decision, game_id)["rejection_reason"])


def all_empty(frame: pd.DataFrame, column: str) -> bool:
    """Whether every cell of a column is empty."""
    return bool(frame[column].isna().all())


def priced_as(target: str, model_value: float, quotes: list[Quote]) -> PricedSide:
    """What the pricing module itself makes of this game."""
    priced = price_target(
        target, model_value, quotes, TABLES[target], captured_at=CAPTURED_AT
    )
    assert isinstance(priced, PricedSide), f"expected a side, got {priced!r}"
    return priced


def game_and_target_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """The frame's (game_id, target) pairs, sorted."""
    return sorted(zip(frame["game_id"], frame["target"], strict=True))


# ---------------------------------------------------------------------------
# Every game, every target
# ---------------------------------------------------------------------------


def test_every_game_gets_one_row_per_early_target(d: EarlyDecision) -> None:
    assert len(d.frame) == 4 and set(d.frame["target"]) == {"ats", "ou"}
    assert game_and_target_pairs(d.frame) == [
        ("g1", "ats"),
        ("g1", "ou"),
        ("g2", "ats"),
        ("g2", "ou"),
    ]


def test_no_moneyline_row_is_ever_written(d: EarlyDecision) -> None:
    assert "wp" not in set(d.frame["target"])


def test_the_frame_has_the_ledgers_columns(d: EarlyDecision) -> None:
    assert list(d.frame.columns) == list(BET_LIST_COLUMNS)


def test_a_week_with_no_games_has_no_rows() -> None:
    decision = decide(TWO_GAMES.iloc[0:0])
    assert decision.frame.empty and list(decision.frame.columns) == BET_LIST_COLUMNS
    assert decision.candidates.empty
    assert list(decision.candidates.columns) == list(CANDIDATE_COLUMNS)


# ---------------------------------------------------------------------------
# A bet
# ---------------------------------------------------------------------------


def test_a_bet_row_carries_book_price_and_stake(d: EarlyDecision) -> None:
    priced = priced_as("ats", SPREAD_MODEL_CLEARING_THE_FLOOR, board(SPREAD_LINE))
    expected_fraction = quarter_kelly_fraction(
        priced.p_win, priced.p_lose, priced.price
    )
    assert expected_fraction > 0

    row = bet_row(d)
    assert row.status == "live" and row.eligibility_label == "book:fanduel"
    assert row.selected_odds == -105 and row.slipped_line == row.line
    assert row.flat_stake == 1.0 and row.ev_tier in ("low", "medium", "high")
    assert row.stake_units == pytest.approx(expected_fraction * 100)
    assert pd.isna(row.rejection_reason)


def test_a_bet_row_is_the_priced_side(d: EarlyDecision) -> None:
    priced = priced_as("ats", SPREAD_MODEL_CLEARING_THE_FLOOR, board(SPREAD_LINE))

    row = bet_row(d)
    assert row.bet_side == "home_cover" and row.line == SPREAD_LINE
    assert row.calibrated_p_side == pytest.approx(priced.p_win)
    assert row.per_bet_ev == pytest.approx(priced.ev)
    assert pd.isna(row.clv)


def test_each_target_reads_its_own_model_and_line(d: EarlyDecision) -> None:
    spread = row_of(d, "g1", "ats")
    total = row_of(d, "g1", "ou")
    assert spread.model_value == SPREAD_MODEL_CLEARING_THE_FLOOR
    assert spread.market_value == SPREAD_LINE and spread.line == SPREAD_LINE
    assert total.model_value == TOTAL_MODEL_BELOW_THE_FLOOR
    assert total.market_value == TOTAL_LINE and total.line == TOTAL_LINE


def test_the_line_is_the_chosen_books_and_not_the_consensus() -> None:
    quotes = {
        "ats": {},
        "ou": {
            "g1": [
                quote("draftkings", 44.5),
                quote("fanduel", 45.5),
                quote("betmgm", 45.5),
            ]
        },
    }
    predictions = {"ats": {}, "ou": {"g1": TOTAL_MODEL_CLEARING_THE_FLOOR}}
    decision = decide(games("g1"), quotes=quotes, predictions=predictions)

    row = row_of(decision, "g1", "ou")
    assert row.line == 44.5 and row.slipped_line == 44.5
    assert row.eligibility_label == "book:draftkings"
    assert candidate_of(decision, "g1", "ou").consensus == 45.5


# ---------------------------------------------------------------------------
# Declined, and never priced
# ---------------------------------------------------------------------------


def test_a_declined_row_keeps_its_side_line_and_book(d: EarlyDecision) -> None:
    row = declined_row(d)
    assert row.status == "suppressed" and row.rejection_reason == "ev_below_floor"
    assert row.bet_side in ("over", "under") and row.line == 44.5
    assert row.eligibility_label.startswith("book:")
    assert pd.isna(row.flat_stake)


def test_a_declined_row_keeps_what_it_was_priced_at(d: EarlyDecision) -> None:
    priced = priced_as("ou", TOTAL_MODEL_BELOW_THE_FLOOR, board(TOTAL_LINE))

    row = declined_row(d)
    assert row.eligibility_label == f"book:{priced.sportsbook}"
    assert row.selected_odds == priced.price and row.slipped_line == row.line
    assert row.calibrated_p_side == pytest.approx(priced.p_win)
    assert row.per_bet_ev == pytest.approx(priced.ev)
    assert row.stake_units == 0.0 and pd.isna(row.ev_tier)


def test_an_expected_value_exactly_on_the_floor_is_declined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor is 0.0 and a bet must be strictly above it."""
    priced = priced_as("ou", TOTAL_MODEL_BELOW_THE_FLOOR, board(TOTAL_LINE))
    on_the_floor = replace(priced, ev=0.0)
    monkeypatch.setattr(early_decide, "price_target", lambda *_, **__: on_the_floor)

    assert reasons(decide(games("g1")), "g1") == {"ev_below_floor"}


def test_a_game_with_no_prediction_is_declined_by_name(d: EarlyDecision) -> None:
    assert reasons(d, "g2") == {"missing_prediction"}
    rows = game_rows(d, "g2")
    assert all_empty(rows, "bet_side") and all_empty(rows, "eligibility_label")


NEVER_PRICED = [
    pytest.param(
        {"ats": {"g1": 21.5}, "ou": {"g1": 53.0}},
        {"ats": {}, "ou": {}},
        "missing_snapshot",
        id="no_quotes_for_the_game",
    ),
    pytest.param(
        {"ats": {"g1": 21.5}, "ou": {"g1": 53.0}},
        {
            "ats": {"g1": [quote("draftkings", SPREAD_LINE)]},
            "ou": {"g1": [quote("draftkings", TOTAL_LINE)]},
        },
        "missing_snapshot",
        id="one_book",
    ),
    pytest.param(
        {"ats": {"g1": SPREAD_LINE}, "ou": {"g1": TOTAL_LINE}},
        BOARDS,
        "no_bet_side",
        id="model_on_the_consensus",
    ),
    pytest.param(
        {"ats": {"g1": float("nan")}, "ou": {"g1": float("nan")}},
        BOARDS,
        "missing_prediction",
        id="nan_prediction",
    ),
    pytest.param(
        {"ats": {"g1": float("inf")}, "ou": {"g1": float("-inf")}},
        BOARDS,
        "missing_prediction",
        id="infinite_prediction",
    ),
    pytest.param(
        {"ats": {}, "ou": {}},
        BOARDS,
        "missing_prediction",
        id="no_prediction_for_the_game",
    ),
]


@pytest.mark.parametrize(("predictions", "quotes", "reason"), NEVER_PRICED)
def test_a_row_never_priced_names_no_side_and_no_book(
    predictions: dict[str, dict[str, float]],
    quotes: dict[str, dict[str, list[Quote]]],
    reason: str,
) -> None:
    """The closing step logs an error for a row with a side and no book, so neither may appear."""
    decision = decide(games("g1"), predictions=predictions, quotes=quotes)

    assert set(decision.frame["status"]) == {"suppressed"}
    assert reasons(decision, "g1") == {reason}
    for column in NEVER_PRICED_COLUMNS:
        assert all_empty(decision.frame, column), f"{column} should be empty"


def test_a_game_with_no_market_still_records_what_the_model_said() -> None:
    decision = decide(
        games("g1"),
        predictions={"ats": {"g1": 21.5}, "ou": {"g1": 53.0}},
        quotes={"ats": {}, "ou": {}},
    )
    assert row_of(decision, "g1", "ats").model_value == 21.5
    assert row_of(decision, "g1", "ou").model_value == 53.0


def test_a_zero_capped_stake_is_declined_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(early_decide, "quarter_kelly_fraction", lambda *_: 0.0)

    row = bet_row(decide())
    assert row.status == "suppressed" and row.rejection_reason == "zero_kelly_stake"
    assert row.bet_side == "home_cover" and row.eligibility_label == "book:fanduel"
    assert pd.isna(row.flat_stake)


# ---------------------------------------------------------------------------
# Sizing across the week
# ---------------------------------------------------------------------------


def test_the_week_cap_covers_both_targets_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each target alone totals 0.075, inside the cap. Pooled, 0.15 is scaled down to 0.10."""
    monkeypatch.setattr(early_decide, "quarter_kelly_fraction", lambda *_: 0.025)

    decision = decide(games("g1", "g2", "g3"), predictions=ALL_BETS)

    assert set(decision.frame["status"]) == {"live"}
    assert list(decision.frame["stake_units"]) == pytest.approx([10.0 / 6] * 6)


def test_the_game_cap_covers_both_targets_of_a_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(early_decide, "quarter_kelly_fraction", lambda *_: 0.04)

    decision = decide(games("g1"), predictions=ALL_BETS)

    assert list(decision.frame["stake_units"]) == pytest.approx([2.5, 2.5])


def test_the_selected_records_are_checked_for_contradiction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[dict[str, Any]]] = []
    monkeypatch.setattr(early_decide, "assert_no_contradiction", seen.append)

    decide()

    assert [(r["game_id"], r["target"], r["bet_side"]) for r in seen[0]] == [
        ("g1", "ats", "home_cover")
    ]


# ---------------------------------------------------------------------------
# The two instants
# ---------------------------------------------------------------------------


def test_the_row_is_decided_before_its_lock(d: EarlyDecision) -> None:
    assert_decided_at_before_freeze(stamped_row(d))


def test_every_row_carries_the_read_and_its_own_lock(d: EarlyDecision) -> None:
    assert set(d.frame["snapshot_ts"]) == {CAPTURED_AT.isoformat()}
    assert set(d.frame["freeze_ts"]) == {LOCK.isoformat()}
    assert set(d.frame["decided_at_utc"]) == {DECIDED_AT.isoformat()}


def test_each_game_is_judged_against_its_own_lock() -> None:
    earlier = LOCK - timedelta(days=2)
    decision = decide(locks={"g1": earlier, "g2": LOCK})

    assert row_of(decision, "g1", "ats").freeze_ts == earlier.isoformat()
    assert row_of(decision, "g2", "ats").freeze_ts == LOCK.isoformat()


def test_a_row_decided_after_its_lock_is_refused_by_the_write_time_check() -> None:
    decision = decide(decided_at=LOCK + timedelta(seconds=1))

    with pytest.raises(DecidedAfterFreezeError):
        assert_decided_at_before_freeze(stamped_row(decision))


# ---------------------------------------------------------------------------
# The candidates table
# ---------------------------------------------------------------------------


def test_candidates_has_one_row_per_game_and_target(d: EarlyDecision) -> None:
    assert list(d.candidates.columns) == list(CANDIDATE_COLUMNS)
    assert game_and_target_pairs(d.candidates) == [
        ("g1", "ats"),
        ("g1", "ou"),
        ("g2", "ats"),
        ("g2", "ou"),
    ]


def test_a_priced_candidate_shows_how_it_was_priced(d: EarlyDecision) -> None:
    priced = priced_as("ats", SPREAD_MODEL_CLEARING_THE_FLOOR, board(SPREAD_LINE))

    row = candidate_of(d, "g1", "ats")
    assert row.model_value == SPREAD_MODEL_CLEARING_THE_FLOOR
    assert row.bet_side == priced.side and row.sportsbook == priced.sportsbook
    assert row.line == priced.line and row.price == priced.price
    assert row.consensus == priced.consensus
    assert row.expected_close == pytest.approx(priced.expected_close)
    assert row.p_win == pytest.approx(priced.p_win)
    assert row.p_push == pytest.approx(priced.p_push)
    assert row.p_lose == pytest.approx(priced.p_lose)
    assert row.ev == pytest.approx(priced.ev)


def test_an_unpriced_candidate_holds_nothing_it_does_not_know(d: EarlyDecision) -> None:
    row = candidate_of(d, "g2", "ou")
    for column in CANDIDATE_COLUMNS:
        if column not in ("game_id", "target"):
            assert pd.isna(row[column]), f"{column} should be empty"


@pytest.mark.parametrize(
    "predictions",
    [
        pytest.param(ARGS["predictions"], id="a_mixed_week"),
        pytest.param({"ats": {}, "ou": {}}, id="nothing_priced"),
        pytest.param(ALL_BETS, id="everything_priced"),
    ],
)
def test_the_candidates_can_be_stored_in_the_decision_snapshot(
    predictions: dict[str, dict[str, float]],
) -> None:
    """The snapshot refuses a numeric cell sitting in an object column."""
    decision = decide(predictions=predictions)
    digest = frame_digest(decision.candidates, SNAPSHOT_PART_KEYS["candidates"])
    assert len(digest) == 64
