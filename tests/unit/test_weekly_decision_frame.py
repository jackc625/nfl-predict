"""A week's decisions, readable WITHOUT persisting a bet row.

Phase 33, Plan 33-07 Task 1 (COLD-08, R9; Codex HIGH, raised on both 33-07 and 33-18).

THE SEAM THAT DID NOT EXIST
---------------------------
``build_weekly_candidates`` returns feature/odds candidates and carries NO ``status`` and NO
``rejection_reason``; those two columns are created by ``records_to_bet_list_frame`` and the
only caller that reaches it, ``generate_weekly_bet_list``, WRITES the result to the durable
artifact pair. So there was no way to ask "what did this week decide?" without also
recording that the week had been decided.

Two things need exactly that. R9's no-edge-week criterion has to show that an honest quiet
week still produces rows with reasons, and Plan 33-18's acceptance run has to read a week's
decisions without publishing them. ``build_weekly_decision_frame`` is that seam.

WHY IT DELEGATES INSTEAD OF REIMPLEMENTING
--------------------------------------------
It calls the EXISTING ``records_to_bet_list_frame`` for the stamping, and
``generate_weekly_bet_list`` was refactored to call IT and then persist. One decision path
with a persistence step bolted on, rather than two paths that agree today. This repository
has been bitten by duplicated logic three times, and the equivalence test below is what
makes the single-path claim checkable rather than stated.

THE WRITES-NOTHING CLAIM IS MEASURED, NOT ASSERTED IN PROSE
-------------------------------------------------------------
``test_it_writes_nothing_at_all`` brackets the call with a CONTENT digest of a sandbox
``outputs/`` and ``data/`` tree -- through ``content_digest_tree``, which refuses to return
a stat signature (D33-32) -- and the module's default artifact directory is redirected into
that sandbox first, so a stray default-path write would land inside the bracket rather than
outside it.

Run this module:  uv run pytest tests/unit/test_weekly_decision_frame.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.cache import BET_LIST_COLUMNS, RUN_MODE_FORWARD
from backtest import weekly_bet_list
from backtest.weekly_bet_list import (
    CANONICAL_TARGETS,
    records_to_bet_list_frame,
    select_weekly_bets,
)
from tests.data_boundary import content_digest_tree
from tests.fixtures.decision_frame import (
    N_GAMES,
    SEASON,
    WEEK,
    builder_stub_frames,
    chain_fit_record,
    fits_with_floor,
    mini_strategies,
)

# One instant for every case, so a frame comparison cannot fail on two clock reads.
_RUN_INSTANT = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)

# An EV floor of 0.0 admits what the fixture can price; the no-edge case lives in
# tests/unit/test_empty_week_refusals.py and turns this dial the other way.
_ADMITTING_FLOOR = 0.0


def _seam() -> Any:
    """The pure seam, or an assertion naming what is missing."""
    seam = getattr(weekly_bet_list, "build_weekly_decision_frame", None)
    assert seam is not None, (
        "backtest.weekly_bet_list defines no build_weekly_decision_frame; a week's "
        "decision statuses cannot be read without persisting a bet row"
    )
    return seam


@pytest.fixture
def stubbed_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stand ``build_weekly_candidates`` in for the real one and return its frames.

    The real builder scores three deployed artifacts against three gold matrices and joins
    the silver odds snapshot -- none of which exists for 2026 yet, which is the cold start
    this whole phase is about. The seam under test is a composition over that builder, so a
    stub is the correct instrument and it claims nothing about the builder's own
    correctness: ``tests/unit/test_empty_week_refusals.py`` drives the REAL builder against
    a real (tmp) gold tree for exactly that reason.
    """
    candidates, schedule = builder_stub_frames()
    monkeypatch.setattr(
        weekly_bet_list,
        "build_weekly_candidates",
        lambda season, week, **_kwargs: (candidates.copy(), schedule.copy()),
    )
    return candidates, schedule


def _decide(**overrides: Any) -> pd.DataFrame:
    return _seam()(
        SEASON,
        WEEK,
        fits=fits_with_floor(_ADMITTING_FLOOR),
        strategies=mini_strategies(),
        now=_RUN_INSTANT,
        **overrides,
    )


# ---------------------------------------------------------------------------
# It produces the two decision columns
# ---------------------------------------------------------------------------


def test_every_candidate_carries_a_status_and_every_suppressed_one_a_reason(
    stubbed_builder: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """``status`` is populated on every row; ``rejection_reason`` on every non-live one.

    A LIVE row's ``rejection_reason`` is None BY DESIGN and that is not an omission: the
    row was not rejected, and inventing a reason for it would make the column unreadable
    as the record of why a candidate did not become a bet.
    """
    frame = _decide()

    assert not frame.empty
    assert frame["status"].notna().all(), "a candidate row carries no status"
    assert set(frame["status"]) <= {"live", "suppressed"}, sorted(set(frame["status"]))

    suppressed = frame[frame["status"] == "suppressed"]
    assert suppressed["rejection_reason"].notna().all(), (
        "a suppressed row carries no rejection_reason"
    )
    live = frame[frame["status"] == "live"]
    assert live["rejection_reason"].isna().all(), (
        "a LIVE row carries a rejection reason; it was not rejected"
    )


def test_the_frame_carries_the_locked_column_set(
    stubbed_builder: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The seam returns the same shape the persisting path writes, not a private one."""
    frame = _decide()
    assert list(frame.columns) == list(BET_LIST_COLUMNS)


# ---------------------------------------------------------------------------
# It writes nothing
# ---------------------------------------------------------------------------


def test_it_writes_nothing_at_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stubbed_builder: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A content-digest bracket over a sandbox ``outputs/`` and ``data/`` tree.

    The module's DEFAULT bet-list directory is redirected into the sandbox before the call,
    so a write that used the default rather than an argument would land INSIDE the bracket
    and be caught, instead of escaping to the repository's real ``outputs/bet_list/``.
    """
    bet_list_dir = tmp_path / "outputs" / "bet_list"
    bet_list_dir.mkdir(parents=True)
    monkeypatch.setattr(weekly_bet_list, "DEFAULT_BET_LIST_DIR", bet_list_dir)

    # Sentinel content, so the digest is a comparison of real bytes rather than of two
    # empty mappings -- an empty tree is trivially unchanged and would prove nothing.
    sentinel = pd.DataFrame({"sentinel": [1, 2, 3]})
    sentinel.to_parquet(bet_list_dir / "bet_list.parquet", index=False)
    (bet_list_dir / "bet_tracker.json").write_text("[]", encoding="utf-8")
    gold_dir = tmp_path / "data" / "gold"
    gold_dir.mkdir(parents=True)
    sentinel.to_parquet(gold_dir / "features_wp.parquet", index=False)

    before = content_digest_tree(tmp_path)
    assert before, "the bracket digested nothing; it would pass on any behaviour"

    _decide()

    after = content_digest_tree(tmp_path)
    assert after == before, (
        "the decision seam changed the sandbox tree. It must produce statuses WITHOUT "
        f"persisting anything.\nbefore: {sorted(before)}\nafter:  {sorted(after)}"
    )


# ---------------------------------------------------------------------------
# It is the SAME decision path the persisting one uses
# ---------------------------------------------------------------------------


def test_its_statuses_match_records_to_bet_list_frame_for_the_same_inputs(
    stubbed_builder: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The seam's output IS the existing stamper's output, byte for byte.

    Computed independently here -- select, then stamp -- and compared to the seam's whole
    frame. If the seam had reimplemented the stamping, the two could agree today and drift
    on the first change to either.
    """
    candidates, schedule = stubbed_builder
    fits = fits_with_floor(_ADMITTING_FLOOR)
    strategies = mini_strategies()

    result = select_weekly_bets(
        candidates.copy(), schedule.copy(), fits, strategies=strategies
    )
    expected = records_to_bet_list_frame(
        result, fits, run_mode=RUN_MODE_FORWARD, decided_at=_RUN_INSTANT
    )

    actual = _decide()

    pd.testing.assert_frame_equal(actual, expected)


def test_the_live_and_suppressed_sets_partition_the_candidate_universe_exactly(
    stubbed_builder: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Every scheduled game times every registered target, once each. No drops, no doubles.

    Both factors are read from the objects under test rather than restated, so a plan that
    registers a fourth target moves the expectation with it.
    """
    frame = _decide()
    _candidates, schedule = stubbed_builder

    expected_rows = len(schedule) * len(CANONICAL_TARGETS)
    assert len(frame) == expected_rows, (
        f"{len(frame)} rows for {len(schedule)} scheduled games x "
        f"{len(CANONICAL_TARGETS)} targets; a candidate was dropped or doubled"
    )
    assert len(schedule) == N_GAMES

    keys = list(zip(frame["game_id"], frame["target"], strict=True))
    assert len(set(keys)) == len(keys), "a (game_id, target) pair appears twice"

    live = frame[frame["status"] == "live"]
    suppressed = frame[frame["status"] == "suppressed"]
    assert len(live) + len(suppressed) == len(frame), (
        "a row carries a status outside the live/suppressed pair, so the two sets do not "
        "partition the universe"
    )
    live_keys = set(zip(live["game_id"], live["target"], strict=True))
    suppressed_keys = set(zip(suppressed["game_id"], suppressed["target"], strict=True))
    assert not (live_keys & suppressed_keys), sorted(live_keys & suppressed_keys)
    assert live_keys | suppressed_keys == set(keys)


# ---------------------------------------------------------------------------
# The persisting path delegates to it
# ---------------------------------------------------------------------------


def test_the_weekly_generator_delegates_to_the_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``generate_weekly_bet_list`` calls the seam ONCE and persists what it returns.

    Asserted by a recording stub rather than by reading the source: a source scan cannot
    tell a call from a mention, and the property that matters is that exactly one decision
    is made per run.
    """
    assert _seam() is not None

    fit_path = tmp_path / "verdict.json"
    fit_path.write_text(
        json.dumps(chain_fit_record(_ADMITTING_FLOOR)), encoding="utf-8"
    )

    calls: list[tuple[int, int]] = []

    def recorder(season: int, week: int, **_kwargs: Any) -> pd.DataFrame:
        calls.append((season, week))
        return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))

    monkeypatch.setattr(weekly_bet_list, "build_weekly_decision_frame", recorder)

    output_dir = tmp_path / "bet_list"
    weekly_bet_list.generate_weekly_bet_list(
        SEASON,
        WEEK,
        output_dir=output_dir,
        chain_fit_path=fit_path,
        now=_RUN_INSTANT,
    )

    assert calls == [(SEASON, WEEK)], calls
    assert (output_dir / "bet_list.parquet").exists(), (
        "the generator delegated but persisted nothing; the seam is pure and the "
        "persistence has to live in the caller"
    )
