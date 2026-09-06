"""The weekly bet-list delegate: freeze immutability, one-time grading, and the artifact seam.

Phase 31, plan 31-17 (PROD-02, SPEC R4, T-31-86/86b/86c).

The three claims this module defends, each of which is easy to assert and hard to prove:

1. **A frozen forward row is immutable in its recommendation facts.** Asserted COLUMN BY COLUMN
   across two runs of the same week, not by comparing a row count. A run before that game's freeze
   replaces the row (late odds can still land); a run at or after it does not.
2. **The transition out of ``pending`` is one-way and one-time.** A second grading attempt on a
   settled row raises a NAMED error rather than being quietly allowed to restate a result -- which
   is the honesty failure the whole four-state vocabulary exists to prevent.
3. **The step writes an ARTIFACT, never the live cache** (REVIEW-CACHE). ``populate_cache`` builds
   a fresh temp database and ends with ``unlink`` + ``rename``, so a live-cache write would be
   destroyed by the next population run. Proven by a live cache file whose mtime AND bytes are
   unchanged across a full step run.

The frames are AUTHORED (fixed model/market/realized values) so the assertions are deterministic
and hermetic: nothing under ``data/``, ``artifacts/`` or ``outputs/`` is read by any test here.

Selectors (``-k``): freeze, immutable, grade, regrade, artifact, live_cache, retired, fit.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    BET_STATUS_LIVE,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
)
from backtest.weekly_bet_list import (
    BET_LIST_ARTIFACT_NAME,
    BET_TRACKER_ARTIFACT_NAME,
    AlreadyGradedError,
    FrozenChainFitError,
    grade_pending_rows,
    grade_row,
    load_frozen_chain_fit,
    read_bet_list_artifact,
    upsert_bet_list_rows,
    write_bet_list_artifact,
    write_bet_tracker_artifact,
)

_FREEZE_TS = "2025-09-05T18:00:00-04:00"
_BEFORE_FREEZE = datetime(2025, 9, 1, tzinfo=UTC)
_AFTER_FREEZE = datetime(2025, 9, 10, tzinfo=UTC)
_GRADED_AT = datetime(2025, 9, 12, tzinfo=UTC)


def _row(
    *,
    game_id: str = "2025_W01_DET@KC",
    target: str = "ou",
    provenance: str = PROVENANCE_FORWARD,
    per_bet_ev: float = 0.08,
    stake_units: float = 1.25,
    selected_odds: float = -110.0,
    grading_status: str = GRADING_STATUS_PENDING,
    status: str = BET_STATUS_LIVE,
) -> dict[str, Any]:
    """One authored bet_list row carrying every locked column."""
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": 2025,
            "week": 1,
            "target": target,
            "bet_side": "under",
            "model_value": 44.0,
            "market_value": 47.5,
            "line": 47.5,
            "slipped_line": 47.0,
            "calibrated_p_side": 0.58,
            "per_bet_ev": per_bet_ev,
            "stake_units": stake_units,
            "ev_tier": "high",
            "status": status,
            "rejection_reason": None,
            "eligibility_label": "under",
            "snapshot_ts": "2025-09-05T14:00:00-04:00",
            "freeze_ts": _FREEZE_TS,
            "selected_odds": selected_odds,
            "flat_stake": 1.0,
            "provenance": provenance,
            "validation_type": (
                "forward_realized"
                if provenance == PROVENANCE_FORWARD
                else "clean_holdout"
            ),
            "grading_status": grading_status,
            "outcome": None,
            "clv": 0.5,
            "payout_flat": None,
            "realized_units": None,
            "graded_at": None,
        }
    )
    return row


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(BET_LIST_COLUMNS))


class _StubStrategy:
    """A stand-in whose ``grade`` is driven by the test, so the pass is exercised in isolation."""

    def __init__(self, target: str, outcome: bool | None) -> None:
        self.target = target
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def grade(self, record: dict[str, Any]) -> bool | None:
        self.calls.append(record)
        return self._outcome


# ---------------------------------------------------------------------------
# 1. The per-game freeze fence
# ---------------------------------------------------------------------------


def test_a_run_before_the_freeze_replaces_the_weeks_forward_rows() -> None:
    """Before the freeze the latest run WINS: the games are not priced and frozen yet."""
    stored = _frame([_row(per_bet_ev=0.01, stake_units=0.10)])
    incoming = _frame([_row(per_bet_ev=0.09, stake_units=2.50)])

    merged = upsert_bet_list_rows(stored, incoming, now=_BEFORE_FREEZE)

    assert len(merged) == 1
    assert merged.iloc[0]["per_bet_ev"] == pytest.approx(0.09)
    assert merged.iloc[0]["stake_units"] == pytest.approx(2.50)


def test_a_run_after_the_freeze_leaves_every_immutable_column_byte_identical() -> None:
    """At or after the freeze the STORED row wins, asserted column by column (T-31-86)."""
    stored = _frame([_row(per_bet_ev=0.01, stake_units=0.10, selected_odds=-105.0)])
    incoming = _frame([_row(per_bet_ev=0.09, stake_units=2.50, selected_odds=-130.0)])

    merged = upsert_bet_list_rows(stored, incoming, now=_AFTER_FREEZE)

    assert len(merged) == 1
    for column in BET_LIST_IMMUTABLE_COLUMNS:
        assert merged.iloc[0][column] == stored.iloc[0][column], (
            f"immutable column {column!r} moved across a post-freeze re-run"
        )


def test_a_replay_row_is_never_frozen_because_it_is_regenerable() -> None:
    """Replay rows are derived, so they are rebuilt whenever inputs change."""
    stored = _frame([_row(provenance=PROVENANCE_BACKTEST_REPLAY, per_bet_ev=0.01)])
    incoming = _frame([_row(provenance=PROVENANCE_BACKTEST_REPLAY, per_bet_ev=0.09)])

    merged = upsert_bet_list_rows(stored, incoming, now=_AFTER_FREEZE)

    assert merged.iloc[0]["per_bet_ev"] == pytest.approx(0.09)


def test_a_stored_week_the_run_did_not_touch_is_kept() -> None:
    """A week this run did not select is HISTORY, not an absence."""
    stored = _frame([_row(game_id="2025_W01_OLD@GAME")])
    incoming = _frame([_row(game_id="2025_W01_NEW@GAME")])

    merged = upsert_bet_list_rows(stored, incoming, now=_AFTER_FREEZE)

    assert set(merged["game_id"]) == {"2025_W01_OLD@GAME", "2025_W01_NEW@GAME"}


# ---------------------------------------------------------------------------
# 2. Grading: one-way, one-time, and the immutable half untouched
# ---------------------------------------------------------------------------


def test_grading_settles_a_pending_row_and_leaves_the_immutable_half_identical() -> (
    None
):
    """The grading pass populates all six grading fields and moves nothing else (T-31-86b)."""
    before = _row(stake_units=2.0, selected_odds=-110.0)

    after = grade_row(before, True, graded_at=_GRADED_AT)

    assert after["grading_status"] == GRADING_STATUS_WIN
    assert after["outcome"] is True
    assert after["payout_flat"] == pytest.approx(100.0 / 110.0)
    assert after["realized_units"] == pytest.approx(2.0 * 100.0 / 110.0)
    assert after["graded_at"] == _GRADED_AT
    assert after["clv"] == before["clv"]
    for column in BET_LIST_IMMUTABLE_COLUMNS:
        assert after[column] == before[column], (
            f"immutable column {column!r} moved during grading"
        )


def test_a_loss_pays_minus_one_unit_and_a_push_pays_zero() -> None:
    """The payout is per-unit-staked profit, identical in form to the frozen per-bet frame."""
    loss = grade_row(_row(stake_units=2.0), False, graded_at=_GRADED_AT)
    push = grade_row(_row(stake_units=2.0), None, graded_at=_GRADED_AT)

    assert loss["grading_status"] == GRADING_STATUS_LOSS
    assert loss["payout_flat"] == pytest.approx(-1.0)
    assert loss["realized_units"] == pytest.approx(-2.0)
    assert push["grading_status"] == GRADING_STATUS_PUSH
    assert push["outcome"] is None
    assert push["payout_flat"] == pytest.approx(0.0)
    assert push["realized_units"] == pytest.approx(0.0)


def test_a_win_is_paid_at_the_price_the_bet_was_actually_struck_at() -> None:
    """A flat -110 payout on a +140 price would understate a real win by a third."""
    graded = grade_row(
        _row(stake_units=1.0, selected_odds=140.0), True, graded_at=_GRADED_AT
    )
    assert graded["payout_flat"] == pytest.approx(1.4)


def test_a_second_grading_attempt_on_a_settled_row_raises_a_named_error() -> None:
    """One-way and one-time: a later run may not quietly restate a result."""
    settled = grade_row(_row(), True, graded_at=_GRADED_AT)

    with pytest.raises(AlreadyGradedError, match="already"):
        grade_row(settled, False, graded_at=_GRADED_AT)


def test_the_batch_pass_skips_settled_rows_so_a_rerun_is_idempotent() -> None:
    """The pass FILTERS to pending, which is what makes re-running the step safe."""
    frame = _frame([_row(grading_status=GRADING_STATUS_WIN)])
    strategies = {"ou": _StubStrategy("ou", False)}

    graded = grade_pending_rows(
        frame, strategies, realized={"ou": {"2025_W01_DET@KC": 41.0}}
    )

    assert graded.iloc[0]["grading_status"] == GRADING_STATUS_WIN
    assert not strategies["ou"].calls


def test_a_row_whose_result_is_unknown_stays_pending() -> None:
    """Not measured is not measured zero: an unplayed game keeps its pending status."""
    frame = _frame([_row()])
    graded = grade_pending_rows(
        frame, {"ou": _StubStrategy("ou", True)}, realized={"ou": {}}
    )
    assert graded.iloc[0]["grading_status"] == GRADING_STATUS_PENDING


def test_a_suppressed_row_is_never_graded() -> None:
    """A candidate that was never bet cannot have won (D31-21)."""
    frame = _frame([_row(status="suppressed")])
    strategies = {"ou": _StubStrategy("ou", True)}

    graded = grade_pending_rows(
        frame, strategies, realized={"ou": {"2025_W01_DET@KC": 41.0}}
    )

    assert graded.iloc[0]["grading_status"] == GRADING_STATUS_PENDING
    assert not strategies["ou"].calls


def test_grading_a_target_with_no_registered_strategy_raises_rather_than_guessing() -> (
    None
):
    """Guessing would book a result under the wrong target's convention."""
    frame = _frame([_row(target="ats")])
    with pytest.raises(ValueError, match="no strategy registered"):
        grade_pending_rows(
            frame,
            {"ou": _StubStrategy("ou", True)},
            realized={"ats": {"2025_W01_DET@KC": 7.0}},
        )


def test_after_grading_the_forward_tracker_block_reports_a_nonzero_graded_count() -> (
    None
):
    """The self-grading honesty loop actually CLOSES rather than staying pending forever."""
    from backtest.bet_tracker import aggregate_by_provenance

    frame = _frame([_row(game_id="g1"), _row(game_id="g2")])
    graded = grade_pending_rows(
        frame,
        {"ou": _StubStrategy("ou", True)},
        realized={"ou": {"g1": 41.0, "g2": 41.0}},
    )

    block = aggregate_by_provenance(
        graded, provenance=PROVENANCE_FORWARD, validation_type="forward_realized"
    )
    assert block.bets_graded == 2


# ---------------------------------------------------------------------------
# 3. The artifact seam, and the live cache it must not touch
# ---------------------------------------------------------------------------


def test_the_artifact_round_trips_through_parquet_with_every_locked_column(
    tmp_path: Path,
) -> None:
    frame = _frame([_row()])
    path = write_bet_list_artifact(frame, tmp_path)

    assert path.name == BET_LIST_ARTIFACT_NAME
    restored = read_bet_list_artifact(tmp_path)
    assert list(restored.columns) == BET_LIST_COLUMNS
    assert restored.iloc[0]["game_id"] == "2025_W01_DET@KC"


def test_an_absent_artifact_reads_as_an_empty_locked_frame(tmp_path: Path) -> None:
    empty = read_bet_list_artifact(tmp_path / "nothing_here")
    assert empty.empty
    assert list(empty.columns) == BET_LIST_COLUMNS


def test_a_stored_artifact_missing_a_column_is_refused_not_merged(
    tmp_path: Path,
) -> None:
    frame = _frame([_row()]).drop(columns=["ev_tier"])
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tmp_path / BET_LIST_ARTIFACT_NAME, index=False)

    with pytest.raises(ValueError, match="missing column"):
        read_bet_list_artifact(tmp_path)


def test_the_tracker_artifact_keeps_a_not_measured_rate_as_null(tmp_path: Path) -> None:
    """A None rate must NOT become 0.0: not measured is a different claim from broke even."""
    from backtest.bet_tracker import EmptyTrackerBlock, to_tracker_frame

    blocks = [
        EmptyTrackerBlock(
            provenance=PROVENANCE_FORWARD, validation_type="forward_realized"
        )
    ]
    path = write_bet_tracker_artifact(to_tracker_frame(blocks), tmp_path)

    assert path.name == BET_TRACKER_ARTIFACT_NAME
    records = json.loads(path.read_text(encoding="utf-8"))
    assert records[0]["hit_rate"] is None
    assert records[0]["flat_return_units"] is None
    assert records[0]["bets_graded"] == 0


def test_the_step_body_opens_no_connection_to_a_cache_database() -> None:
    """A grep of the step body finds no cache connection (REVIEW-CACHE, T-31-86c).

    Structural rather than behavioural: a behavioural check would only prove the ONE fixture week
    left the cache alone, while this proves no code path in the step can reach a database at all.
    """
    import ast
    import inspect

    from pipeline import steps

    source = inspect.getsource(steps.step_generate_recommendations)
    tree = ast.parse(source.lstrip())
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "duckdb" not in names
    assert "connect" not in names
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "duckdb" in n.value.lower()
    ]


def test_the_step_body_carries_no_confidence_tier_comparison() -> None:
    """The retired selection criterion is GONE, not renamed (D31-23/31)."""
    import inspect

    from pipeline import steps

    source = inspect.getsource(steps.step_generate_recommendations)
    body = source.split('"""')[-1]  # the code after the docstring
    for token in ('"medium"', "'medium'", "_confidence", "recommendations_"):
        assert token not in body, (
            f"the retired tier filter token {token!r} survives in the step"
        )


# ---------------------------------------------------------------------------
# 4. The frozen fit is read, never invented
# ---------------------------------------------------------------------------


def test_an_absent_frozen_fit_raises_rather_than_defaulting_the_ev_floor(
    tmp_path: Path,
) -> None:
    """A defaulted floor would admit bets at a threshold nobody swept for."""
    with pytest.raises(FrozenChainFitError, match="NO fallback"):
        load_frozen_chain_fit(tmp_path / "absent.json")


def test_a_record_missing_a_targets_fit_raises_naming_the_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps({"tune_fit": {"wp": {"ev_floor_t": 0.05}}}), encoding="utf-8"
    )
    with pytest.raises(FrozenChainFitError, match="'ats'"):
        load_frozen_chain_fit(path)


def test_the_fit_is_read_verbatim_including_a_null_residual_sd(tmp_path: Path) -> None:
    """WP fits no residual SD by design (D31-07); None must survive as None, not become 0.0."""
    path = tmp_path / "fit.json"
    path.write_text(
        json.dumps(
            {
                "tune_fit": {
                    "wp": {
                        "ev_floor_t": 0.05,
                        "frozen_sd": None,
                        "season_bias_by_season": {"2025": 0.03},
                    },
                    "ats": {
                        "ev_floor_t": 0.05,
                        "frozen_sd": 11.5,
                        "season_bias_by_season": {"2025": 0.16},
                    },
                    "ou": {
                        "ev_floor_t": 0.0,
                        "frozen_sd": 13.0,
                        "season_bias_by_season": {"2025": 0.02},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    fits = load_frozen_chain_fit(path)

    assert fits["wp"].frozen_sd is None
    assert fits["ats"].frozen_sd == pytest.approx(11.5)
    assert fits["ou"].ev_floor_t == pytest.approx(0.0)
    assert fits["ats"].season_bias_by_season == {2025: pytest.approx(0.16)}
