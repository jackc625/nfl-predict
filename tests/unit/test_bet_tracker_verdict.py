"""The tracker's verdict split, corrected outcomes and the result strip's rows (Plan 34-16 Task 1).

Phase 34 (LDGR-08, LDGR-10, D-06, D-16; research Pitfall 12; review finding 4). Three facts the
``/bets`` forward record must carry honestly, all decided at CACHE BUILD so the request path stays
computation-free (UIAP-01):

1. **A pre-verdict row never counts.** Migrated rows keep their STORED ``validation_type``
   (``forward_realized``, immutable, chained). The split comes from the row's ``verdict_scope`` and
   is applied to a COPY for grouping only, so the stored label is never rewritten.
2. **Shadow rows are not the live record.** A ``shadow``-arm forward row is absent from every
   forward block.
3. **The in-force outcome counts (D-06).** A row whose score was corrected after grading counts as
   its latest correction; the strip rows carry the same corrected grade and say so.

The strip rows and the tiles come from ONE prepared frame (``prepare_tracker_rows``), so the strip
cannot put a pre-verdict mark under the verdict block (review finding 4).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from api.cache import (
    BET_GRADED_OUTCOMES_COLUMNS,
    BET_LIST_CORRECTIONS_COLUMNS,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
    VALIDATION_TYPE_CLEAN_HOLDOUT,
    VALIDATION_TYPE_FORWARD_REALIZED,
    VALIDATION_TYPE_PRE_VERDICT,
)
from backtest.bet_tracker import (
    TRACKER_BLOCK_ORDER,
    TrackerBlock,
    aggregate_all_blocks,
    graded_outcome_rows,
)

_PRE = (PROVENANCE_FORWARD, VALIDATION_TYPE_PRE_VERDICT)
_VERDICT = (PROVENANCE_FORWARD, VALIDATION_TYPE_FORWARD_REALIZED)
_CLEAN = (PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CLEAN_HOLDOUT)

_WIN_PAYOUT = 100.0 / 110.0


def _row(
    game_id: str,
    grading_status: str,
    *,
    verdict_scope: str | None = "verdict",
    arm: str | None = "live",
    status: str = "live",
    provenance: str = PROVENANCE_FORWARD,
    validation_type: str = VALIDATION_TYPE_FORWARD_REALIZED,
    season: int = 2026,
    week: int = 6,
    target: str = "ats",
) -> dict[str, Any]:
    """One bet-list row with the columns the tracker and the strip read."""
    outcome = {
        GRADING_STATUS_WIN: True,
        GRADING_STATUS_LOSS: False,
        GRADING_STATUS_PUSH: None,
        GRADING_STATUS_PENDING: None,
    }[grading_status]
    payout = {
        GRADING_STATUS_WIN: _WIN_PAYOUT,
        GRADING_STATUS_LOSS: -1.0,
        GRADING_STATUS_PUSH: 0.0,
        GRADING_STATUS_PENDING: None,
    }[grading_status]
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "target": target,
        "arm": arm,
        "status": status,
        "provenance": provenance,
        "validation_type": validation_type,
        "verdict_scope": verdict_scope,
        "grading_status": grading_status,
        "outcome": outcome,
        "flat_stake": 1.0,
        "payout_flat": payout,
        "realized_units": payout,
    }


def _correction(
    game_id: str,
    *,
    original: str,
    corrected: str,
    corrected_payout: float,
    season: int = 2026,
    week: int = 6,
    target: str = "ats",
) -> dict[str, Any]:
    """One in-force correction row in ``BET_LIST_CORRECTIONS_COLUMNS``."""
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "target": target,
        "arm": "live",
        "original_grading_status": original,
        "corrected_grading_status": corrected,
        "corrected_outcome": {"win": True, "loss": False}.get(corrected),
        "corrected_payout_flat": corrected_payout,
        "corrected_realized_units": corrected_payout,
        "realized_value": -3.0,
        "detected_at_utc": "2026-10-20T12:00:00+00:00",
        "corrected_at_utc": "2026-10-20T21:00:00+00:00",
    }


def _blocks_by_pair(frame: pd.DataFrame, **kwargs: Any) -> dict[tuple[str, str], Any]:
    return {
        (block.provenance, block.validation_type): block
        for block in aggregate_all_blocks(frame, **kwargs)
    }


def test_pre_verdict_rows_form_their_own_block() -> None:
    """``pre_verdict`` rows aggregate into (forward, pre_verdict); ``verdict`` rows do not."""
    frame = pd.DataFrame(
        [
            _row("A", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=3),
            _row("B", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=4),
            _row("C", GRADING_STATUS_LOSS, verdict_scope="verdict"),
        ]
    )
    blocks = _blocks_by_pair(frame)

    assert set(blocks) == {_PRE, _VERDICT}
    assert (blocks[_PRE].wins, blocks[_PRE].losses, blocks[_PRE].bets_graded) == (
        2,
        0,
        2,
    )
    assert (blocks[_VERDICT].wins, blocks[_VERDICT].losses) == (0, 1)
    assert blocks[_VERDICT].bets_graded == 1
    # The weakest-evidence-first order puts the pre-verdict block before the verdict block.
    assert TRACKER_BLOCK_ORDER.index(_PRE) == TRACKER_BLOCK_ORDER.index(_VERDICT) - 1


def test_pre_verdict_absent_from_verdict_totals() -> None:
    """Adding pre-verdict wins moves only the pre-verdict block."""
    verdict_only = pd.DataFrame(
        [
            _row("C", GRADING_STATUS_LOSS),
            _row("D", GRADING_STATUS_WIN),
        ]
    )
    with_pre_verdict = pd.DataFrame(
        [
            *verdict_only.to_dict("records"),
            _row("A", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=3),
            _row("B", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=3),
            _row("E", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=4),
        ]
    )

    before = _blocks_by_pair(verdict_only)[_VERDICT]
    after = _blocks_by_pair(with_pre_verdict)

    assert after[_VERDICT] == before
    assert after[_PRE].wins == 3


def test_shadow_rows_excluded_from_forward_blocks() -> None:
    """A shadow-arm forward row is in no forward block, graded or not."""
    frame = pd.DataFrame(
        [
            _row("A", GRADING_STATUS_LOSS),
            _row("A", GRADING_STATUS_WIN, arm="shadow"),
            _row("B", GRADING_STATUS_WIN, arm="shadow", verdict_scope="pre_verdict"),
        ]
    )
    blocks = _blocks_by_pair(frame)

    assert set(blocks) == {_VERDICT}
    assert (blocks[_VERDICT].wins, blocks[_VERDICT].losses) == (0, 1)


def test_corrected_outcome_counted() -> None:
    """A win with an in-force correction to loss counts as a loss at the corrected payout."""
    frame = pd.DataFrame(
        [
            _row("A", GRADING_STATUS_WIN),
            _row("B", GRADING_STATUS_WIN),
        ]
    )
    corrections = pd.DataFrame(
        [_correction("A", original="win", corrected="loss", corrected_payout=-1.0)],
        columns=pd.Index(BET_LIST_CORRECTIONS_COLUMNS),
    )

    block = _blocks_by_pair(frame, corrections=corrections)[_VERDICT]

    assert isinstance(block, TrackerBlock)
    assert (block.wins, block.losses, block.bets_graded) == (1, 1, 2)
    assert block.flat_return_units == (_WIN_PAYOUT - 1.0) / 2.0
    # Without the correction the same rows read as two wins: the overlay is what moved it.
    uncorrected = _blocks_by_pair(frame)[_VERDICT]
    assert (uncorrected.wins, uncorrected.losses) == (2, 0)


def test_stored_validation_type_untouched() -> None:
    """The caller's frame keeps its stored ``validation_type`` and grading after aggregation."""
    frame = pd.DataFrame(
        [
            _row("A", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=3),
            _row("B", GRADING_STATUS_WIN),
            _row("C", GRADING_STATUS_WIN, arm="shadow"),
        ]
    )
    snapshot = frame.copy(deep=True)
    corrections = pd.DataFrame(
        [_correction("B", original="win", corrected="loss", corrected_payout=-1.0)],
        columns=pd.Index(BET_LIST_CORRECTIONS_COLUMNS),
    )

    aggregate_all_blocks(frame, corrections=corrections)
    graded_outcome_rows(frame, corrections=corrections)

    pd.testing.assert_frame_equal(frame, snapshot)
    assert list(frame["validation_type"]) == [VALIDATION_TYPE_FORWARD_REALIZED] * 3


def test_old_rows_without_verdict_scope_unchanged() -> None:
    """Rows with NULL ``verdict_scope`` and ``arm`` aggregate exactly as a frame without them."""
    rows = [
        _row("A", GRADING_STATUS_WIN, verdict_scope=None, arm=None),
        _row("B", GRADING_STATUS_LOSS, verdict_scope=None, arm=None),
        _row("C", GRADING_STATUS_PUSH, verdict_scope=None, arm=None),
        _row(
            "R",
            GRADING_STATUS_WIN,
            verdict_scope=None,
            arm=None,
            provenance=PROVENANCE_BACKTEST_REPLAY,
            validation_type=VALIDATION_TYPE_CLEAN_HOLDOUT,
            season=2025,
        ),
    ]
    with_null_columns = pd.DataFrame(rows)
    without_columns = with_null_columns.drop(columns=["verdict_scope", "arm"])

    blocks = aggregate_all_blocks(with_null_columns)

    assert blocks == aggregate_all_blocks(without_columns)
    by_pair = {(b.provenance, b.validation_type): b for b in blocks}
    assert set(by_pair) == {_CLEAN, _VERDICT}
    assert (by_pair[_VERDICT].wins, by_pair[_VERDICT].losses) == (1, 1)
    assert by_pair[_VERDICT].pushes == 1


def test_graded_outcome_rows_match_the_tracker() -> None:
    """The strip rows are the tracker's rows: same classes, same in-force grades, same counts."""
    frame = pd.DataFrame(
        [
            _row("PRE", GRADING_STATUS_WIN, verdict_scope="pre_verdict", week=4),
            _row("LOSS", GRADING_STATUS_LOSS),
            _row("FIXED", GRADING_STATUS_WIN),
            _row("PEND", GRADING_STATUS_PENDING),
            _row("SUPP", GRADING_STATUS_PENDING, status="suppressed"),
            _row("SHADOW", GRADING_STATUS_WIN, arm="shadow"),
        ]
    )
    corrections = pd.DataFrame(
        [_correction("FIXED", original="win", corrected="loss", corrected_payout=-1.0)],
        columns=pd.Index(BET_LIST_CORRECTIONS_COLUMNS),
    )

    rows = graded_outcome_rows(frame, corrections=corrections)

    assert list(rows.columns) == list(BET_GRADED_OUTCOMES_COLUMNS)
    # Ordered by (provenance, validation_type, season, week, game_id, ...) -- the same ORDER BY the
    # strip's getter has always used, so the class labels sort as text.
    assert list(zip(rows["game_id"], rows["validation_type"], strict=True)) == [
        ("FIXED", VALIDATION_TYPE_FORWARD_REALIZED),
        ("LOSS", VALIDATION_TYPE_FORWARD_REALIZED),
        ("PRE", VALIDATION_TYPE_PRE_VERDICT),
    ]
    by_game = rows.set_index("game_id")
    assert by_game.loc["FIXED", "grading_status"] == GRADING_STATUS_LOSS
    assert bool(by_game.loc["FIXED", "corrected"]) is True
    assert bool(by_game.loc["LOSS", "corrected"]) is False
    assert bool(by_game.loc["PRE", "corrected"]) is False
    assert set(rows["arm"]) == {"live"}

    # For every block the tracker publishes, its counts are the counts of that class's strip rows.
    for block in aggregate_all_blocks(frame, corrections=corrections):
        in_class = rows[
            (rows["provenance"] == block.provenance)
            & (rows["validation_type"] == block.validation_type)
        ]
        statuses = list(in_class["grading_status"])
        assert block.bets_graded == len(statuses)
        assert block.wins == statuses.count(GRADING_STATUS_WIN)
        assert block.losses == statuses.count(GRADING_STATUS_LOSS)
        assert block.pushes == statuses.count(GRADING_STATUS_PUSH)
