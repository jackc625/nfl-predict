"""WR-08 / WR-15: the book a bet is priced at is chosen BY NAME, not by parquet row order.

WHAT WAS WRONG
--------------
Both pricing paths did::

    odds[odds["game_id"].isin(game_ids)].drop_duplicates(subset=["game_id"], keep="first")

``keep="first"`` selects whichever row happens to appear first in the parquet file. The OUM-06
allowlist admits TWO sportsbooks (``consensus``, ``draftkings``) and neither call site carried a
preference between them. The selected row supplies ``spread``, ``total``, ``ml_home``, ``ml_away``
and all four juice columns -- the price the per-bet EV, the Kelly stake and the published
``selected_odds`` are all struck at. Deterministic for a fixed file, but not STABLE: appending a
``draftkings`` row, or any rewrite that reorders the parquet, silently changes which book's price a
published bet was struck at, with nothing on the record to attribute the change to.

WR-15 is the same defect with a second symptom. ``profitability_2025._load_candidate_frames``
joined the WHOLE silver odds table with no dedupe at all -- and ``_load_closing_odds`` normalizes
``game_id`` (LAR -> LA) on the way, which can itself create two rows sharing one key. A left join
against a duplicated key FANS OUT the candidate frame: one game becomes two priced bets and double
weight in both the ROI numerator and the block bootstrap. ``BetSelector._build_universe`` refuses a
duplicate ``(game_id, target)`` pair by name, but the verdict path passes no ``scheduled_games``,
so nothing would have caught it.

Live silver holds 2,140 rows / 2,140 distinct ``game_id`` / one sportsbook, so nothing published is
affected. The tests below therefore construct the multi-book state deliberately.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.ou_divergence import (
    _ALLOWED_SPORTSBOOKS,
    SPORTSBOOK_PREFERENCE,
    dedupe_odds_by_book_preference,
)

_GAME = "2025_W01_DAL@PHI"


def _row(sportsbook: str, spread: float) -> dict[str, object]:
    return {
        "game_id": _GAME,
        "sportsbook": sportsbook,
        "spread": spread,
        "total": 47.5,
        "ml_home": -425,
        "ml_away": 330,
    }


def test_the_preference_and_the_provenance_allowlist_are_the_same_set() -> None:
    """A book added to the allowlist must not fall silently off the end of the preference."""
    assert set(SPORTSBOOK_PREFERENCE) == set(_ALLOWED_SPORTSBOOKS)


@pytest.mark.parametrize(
    "stored_order", [("consensus", "draftkings"), ("draftkings", "consensus")]
)
def test_the_same_book_wins_regardless_of_stored_row_order(
    stored_order: tuple[str, str],
) -> None:
    """THE finding: the answer must not depend on which row the parquet happens to hold first."""
    prices = {"consensus": -3.0, "draftkings": -7.5}
    frame = pd.DataFrame([_row(book, prices[book]) for book in stored_order])

    deduped = dedupe_odds_by_book_preference(frame)

    assert len(deduped) == 1
    assert deduped.iloc[0]["sportsbook"] == SPORTSBOOK_PREFERENCE[0]
    assert deduped.iloc[0]["spread"] == pytest.approx(prices[SPORTSBOOK_PREFERENCE[0]])


def test_the_old_expression_really_did_depend_on_row_order() -> None:
    """Without this, the test above guards a hazard nobody has shown exists."""
    prices = {"consensus": -3.0, "draftkings": -7.5}
    forward = pd.DataFrame([_row(b, prices[b]) for b in ("consensus", "draftkings")])
    reversed_ = pd.DataFrame([_row(b, prices[b]) for b in ("draftkings", "consensus")])

    old_forward = forward.drop_duplicates(subset=["game_id"], keep="first")
    old_reversed = reversed_.drop_duplicates(subset=["game_id"], keep="first")

    assert old_forward.iloc[0]["sportsbook"] != old_reversed.iloc[0]["sportsbook"], (
        "drop_duplicates(keep='first') no longer follows row order, so this module is pinning "
        "the wrong mechanism"
    )


def test_an_unrecognised_book_ranks_last_but_is_not_dropped() -> None:
    """Refusing an unrecognised label is the provenance guard's job, by name, in its own place.

    Dropping it here would turn a provenance failure into a game that silently has no price.
    """
    frame = pd.DataFrame([_row("mystery_book", -1.0), _row("consensus", -3.0)])

    deduped = dedupe_odds_by_book_preference(frame)

    assert deduped.iloc[0]["sportsbook"] == "consensus"

    only_unknown = dedupe_odds_by_book_preference(
        pd.DataFrame([_row("mystery_book", -1.0)])
    )
    assert len(only_unknown) == 1
    assert only_unknown.iloc[0]["sportsbook"] == "mystery_book"


def test_it_returns_exactly_one_row_per_game() -> None:
    """The fan-out WR-15 names is prevented by construction, not by an assertion downstream."""
    frame = pd.DataFrame(
        [
            _row("consensus", -3.0),
            _row("draftkings", -7.5),
            {**_row("consensus", 2.5), "game_id": "2025_W01_AAA@BBB"},
        ]
    )

    deduped = dedupe_odds_by_book_preference(frame)

    assert deduped["game_id"].is_unique
    assert len(deduped) == 2


def test_a_frame_with_no_sportsbook_column_is_handled_not_crashed() -> None:
    """Some callers project the columns before deduping; there is no preference to apply."""
    frame = pd.DataFrame(
        [
            {"game_id": _GAME, "spread": -3.0},
            {"game_id": _GAME, "spread": -7.5},
        ]
    )

    deduped = dedupe_odds_by_book_preference(frame)

    assert len(deduped) == 1
    assert deduped.iloc[0]["spread"] == pytest.approx(-3.0)


def test_an_empty_frame_round_trips() -> None:
    frame = pd.DataFrame(columns=pd.Index(["game_id", "sportsbook", "spread"]))

    assert dedupe_odds_by_book_preference(frame).empty


def test_the_verdict_runner_refuses_a_fan_out_rather_than_double_counting() -> None:
    """WR-15's assertion, exercised directly on the guard's own message.

    The join guard in ``_load_candidate_frames`` cannot be reached without a full scoring run, so
    what is pinned here is that the guard EXISTS in that function and names the failure -- a
    source-level check in the same shape ``tests/unit/test_simulation_ou_routing.py`` already uses
    for the LOCKED-2 routing.
    """
    import inspect

    from backtest.profitability_2025 import _load_candidate_frames

    source = inspect.getsource(_load_candidate_frames)

    assert "dedupe_odds_by_book_preference" in source, (
        "the verdict runner joins the raw odds table again; a duplicated game_id fans the "
        "candidate frame out into two priced bets"
    )
    assert "fanned out" in source, (
        "the post-join length assertion is gone, so a fan-out introduced another way would pass "
        "silently"
    )


# ---------------------------------------------------------------------------
# 33.2 review A WR-01 / C1 WR-01: the latest line known at the lock wins; the book breaks ties
# ---------------------------------------------------------------------------

_LIVE_GAME = "2026_W04_DAL@PHI"
_LOCK = pd.Timestamp("2026-09-26T22:00:00Z")  # Saturday 18:00 ET before a Sunday game


def _capture(book: str, created_at: str, spread: float) -> dict[str, object]:
    """One accumulated live capture: snapshot_ts is the lock LABEL, created_at the capture."""
    return {
        "game_id": _LIVE_GAME,
        "sportsbook": book,
        "snapshot_ts": _LOCK,
        "created_at": pd.Timestamp(created_at),
        "spread": spread,
    }


def test_the_latest_pre_lock_capture_wins_over_an_older_one_of_the_same_book() -> None:
    """The store appends, so the OLDEST capture sat first in the file and used to win."""
    frame = pd.DataFrame(
        [
            _capture("draftkings", "2026-09-23T15:00:00Z", -2.5),  # opening line
            _capture("draftkings", "2026-09-26T21:00:00Z", -4.0),  # lock day, pre-lock
            _capture("draftkings", "2026-09-26T23:00:00Z", -9.0),  # after the lock
        ]
    )
    chosen = dedupe_odds_by_book_preference(frame, locks={_LIVE_GAME: _LOCK})
    assert len(chosen) == 1
    assert chosen.iloc[0]["spread"] == pytest.approx(-4.0)


def test_a_capture_exactly_at_the_lock_is_admissible_and_one_second_later_is_not() -> (
    None
):
    at_lock = pd.DataFrame([_capture("fanduel", "2026-09-26T22:00:00Z", -3.0)])
    late = pd.DataFrame([_capture("fanduel", "2026-09-26T22:00:01Z", -3.0)])
    assert len(dedupe_odds_by_book_preference(at_lock, locks={_LIVE_GAME: _LOCK})) == 1
    assert dedupe_odds_by_book_preference(late, locks={_LIVE_GAME: _LOCK}).empty


def test_the_book_only_breaks_a_tie_in_capture_time() -> None:
    same_instant = "2026-09-26T21:00:00Z"
    frame = pd.DataFrame(
        [
            _capture("fanduel", "2026-09-25T21:00:00Z", -1.0),  # older, any book
            _capture("mybookieag", same_instant, -5.0),
            _capture("betmgm", same_instant, -6.0),
            _capture("draftkings", same_instant, -7.0),
        ]
    )
    chosen = dedupe_odds_by_book_preference(frame, locks={_LIVE_GAME: _LOCK})
    assert chosen.iloc[0]["sportsbook"] == "draftkings"

    no_preferred = frame[frame["sportsbook"] != "draftkings"].iloc[::-1]
    chosen = dedupe_odds_by_book_preference(no_preferred, locks={_LIVE_GAME: _LOCK})
    assert chosen.iloc[0]["sportsbook"] == "betmgm", "by name, never by file order"


def test_a_historical_row_is_judged_by_its_capture_time_never_its_label() -> None:
    """Owner ruling 2026-09-22 (Option B): a pre-lock LABEL with a post-lock capture is refused.

    A consensus row's ``snapshot_ts`` is a manufactured label; only ``created_at`` (here the
    backfill, after the lock) is an information time, so the row is admissible at no lock.
    """
    frame = pd.DataFrame(
        [
            {
                "game_id": _LIVE_GAME,
                "sportsbook": "consensus",
                "snapshot_ts": "2026-09-25T22:00:00+00:00",
                "created_at": pd.Timestamp("2027-02-20T00:00:00Z"),
                "spread": -3.5,
            }
        ]
    )
    assert dedupe_odds_by_book_preference(frame, locks={_LIVE_GAME: _LOCK}).empty


def test_the_serving_market_line_reads_the_latest_pre_lock_capture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.generate_current_week_predictions import load_market_data

    silver = tmp_path / "data" / "silver"
    silver.mkdir(parents=True)
    pd.DataFrame(
        {"game_id": [_LIVE_GAME], "kickoff_et": [pd.Timestamp("2026-09-27T17:00:00Z")]}
    ).to_parquet(silver / "games.parquet", index=False)
    pd.DataFrame(
        [
            _capture("draftkings", "2026-09-23T15:00:00Z", -2.5),
            _capture("draftkings", "2026-09-26T21:00:00Z", -4.0),
            _capture("draftkings", "2026-09-26T23:00:00Z", -9.0),
        ]
    ).to_parquet(silver / "odds_snapshot.parquet", index=False)
    monkeypatch.chdir(tmp_path)

    market = load_market_data([_LIVE_GAME])
    assert market.set_index("game_id").loc[_LIVE_GAME, "spread"] == pytest.approx(-4.0)
