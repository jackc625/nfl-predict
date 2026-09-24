"""The owner's ruling on the old-rule 2026 rows is EXECUTED, whichever it was (Plan 33.2-26 Task 4).

Every branch runs against a ``tmp_path`` tree carrying a synthetic 2026 artifact set beside a 2025
control, never the real ``outputs/``. Pinned here, because only one branch runs for real:

* each of the three rulings has exactly its declared effect;
* an ABSENT ruling and an UNRECOGNISED ruling are two different named refusals, and neither
  writes anything -- a missing owner answer and a mistyped one are different failures;
* a 2025 artifact in the same directory is never touched (scope is by season);
* ``keep-and-supersede`` destroys nothing: every source is readable at its new path;
* the dry-run file list IS the ``--apply`` file list;
* non-vacuity: the synthetic 2026 set is non-empty, so "nothing touched" cannot pass for the
  wrong reason.

Run this module:  uv run python -m pytest tests/unit/test_old_rule_prediction_disposition.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.cache import BET_LIST_COLUMNS, GRADING_STATUS_PENDING
from backtest.weekly_bet_list import (
    read_bet_list_artifact,
    write_bet_list_artifact,
    write_bet_tracker_artifact,
)
from scripts import dispose_old_rule_predictions as dispose

PREDICTIONS_2026 = ("predictions_2026_week1.csv", "predictions_2026_week2.csv")
PREDICTIONS_2025 = "predictions_2025_week1.csv"


def _bet_row(season: int, game_id: str) -> dict[str, Any]:
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": season,
            "week": 1,
            "target": "ou",
            "status": "suppressed",
            "rejection_reason": "ev_below_floor",
            "provenance": "backtest_replay",
            "validation_type": "contaminated",
            "grading_status": GRADING_STATUS_PENDING,
        }
    )
    return row


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    """A predictions dir and a bet-list dir holding a 2026 set beside a 2025 control."""
    from backtest.bet_tracker import aggregate_all_blocks, to_tracker_frame

    predictions = tmp_path / "outputs" / "predictions"
    bet_list = tmp_path / "outputs" / "bet_list"
    predictions.mkdir(parents=True)
    for name in (*PREDICTIONS_2026, PREDICTIONS_2025):
        (predictions / name).write_text(f"game_id\n{name}\n", encoding="utf-8")
    frame = pd.DataFrame(
        [
            _bet_row(2025, "2025_W01_A@B"),
            _bet_row(2026, "2026_W01_C@D"),
            _bet_row(2026, "2026_W02_E@F"),
        ],
        columns=pd.Index(BET_LIST_COLUMNS),
    )
    write_bet_list_artifact(frame, bet_list)
    write_bet_tracker_artifact(
        to_tracker_frame(aggregate_all_blocks(frame)), output_dir=bet_list
    )
    return predictions, bet_list


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(tmp_path: Path, ruling: str) -> tuple[Path, Path, tuple[Path, ...], Any]:
    predictions, bet_list = _tree(tmp_path)
    plan = dispose.plan_disposition(ruling, predictions, bet_list)
    touched = dispose.apply_disposition(plan, decided_on="2026-09-23")
    return predictions, bet_list, touched, plan


# ---------------------------------------------------------------------------
# 0. Non-vacuity
# ---------------------------------------------------------------------------


def test_the_synthetic_2026_set_is_non_empty(tmp_path: Path) -> None:
    predictions, bet_list = _tree(tmp_path)
    plan = dispose.plan_disposition(dispose.RULING_LABEL, predictions, bet_list)
    assert len(plan.prediction_files) == len(PREDICTIONS_2026)
    assert plan.bet_rows == 2


# ---------------------------------------------------------------------------
# 1. Each branch does exactly what it declares
# ---------------------------------------------------------------------------


def test_delete_and_repredict_removes_only_the_2026_artifacts(tmp_path: Path) -> None:
    predictions, bet_list, touched, _plan = _run(tmp_path, dispose.RULING_DELETE)

    assert sorted(p.name for p in predictions.glob("*.csv")) == [PREDICTIONS_2025]
    assert not (predictions / dispose.SUPERSEDED_DIR_NAME).exists()
    remaining = read_bet_list_artifact(bet_list)
    assert list(remaining["season"]) == [2025]
    assert touched, "the delete branch touched nothing on a non-empty 2026 set"


def test_keep_and_supersede_moves_and_destroys_nothing(tmp_path: Path) -> None:
    predictions, bet_list, _touched, _plan = _run(tmp_path, dispose.RULING_SUPERSEDE)

    moved = predictions / dispose.SUPERSEDED_DIR_NAME
    for name in PREDICTIONS_2026:
        assert not (predictions / name).exists()
        assert (moved / name).read_text(encoding="utf-8") == f"game_id\n{name}\n"
    assert (predictions / PREDICTIONS_2025).exists()

    old_rows = pd.read_parquet(
        bet_list
        / dispose.SUPERSEDED_DIR_NAME
        / dispose.SUPERSEDED_BET_ROWS_NAME.format(season=2026)
    )
    assert sorted(old_rows["game_id"]) == ["2026_W01_C@D", "2026_W02_E@F"]
    assert list(read_bet_list_artifact(bet_list)["season"]) == [2025]

    marker = json.loads((moved / dispose.MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["ruling"] == dispose.RULING_SUPERSEDE
    assert marker["ruled_on"] == "2026-09-23"
    assert set(marker["corrective_commits"]) == {
        "cold_start_11761c7",
        "ev_chain_ee20773",
    }
    assert all(len(sha) == 40 for sha in marker["corrective_commits"].values())
    assert marker["reason"]


def test_keep_and_label_writes_nothing(tmp_path: Path) -> None:
    predictions, bet_list = _tree(tmp_path)
    before = _snapshot(tmp_path)
    plan = dispose.plan_disposition(dispose.RULING_LABEL, predictions, bet_list)
    assert dispose.apply_disposition(plan, decided_on="2026-09-23") == ()
    assert _snapshot(tmp_path) == before


# ---------------------------------------------------------------------------
# 2. Two distinct named refusals, neither of which writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("absent", [None, "", "   "])
def test_an_absent_ruling_refuses_by_name_and_writes_nothing(
    tmp_path: Path, absent: str | None
) -> None:
    predictions, bet_list = _tree(tmp_path)
    before = _snapshot(tmp_path)
    with pytest.raises(dispose.AbsentRulingError):
        dispose.plan_disposition(absent, predictions, bet_list)
    assert _snapshot(tmp_path) == before


def test_an_unrecognised_ruling_refuses_by_name_and_writes_nothing(
    tmp_path: Path,
) -> None:
    predictions, bet_list = _tree(tmp_path)
    before = _snapshot(tmp_path)
    with pytest.raises(dispose.UnrecognisedRulingError, match="delete-and-predict"):
        dispose.plan_disposition("delete-and-predict", predictions, bet_list)
    assert _snapshot(tmp_path) == before


def test_the_two_refusals_are_different_failures() -> None:
    assert not issubclass(dispose.AbsentRulingError, dispose.UnrecognisedRulingError)
    assert not issubclass(dispose.UnrecognisedRulingError, dispose.AbsentRulingError)
    for refusal in (dispose.AbsentRulingError, dispose.UnrecognisedRulingError):
        assert not issubclass(
            refusal, (ValueError, KeyError, RuntimeError, LookupError)
        )


# ---------------------------------------------------------------------------
# 3. Season scoping, and the dry run IS the plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ruling", dispose.RULINGS)
def test_no_branch_touches_a_2025_artifact(tmp_path: Path, ruling: str) -> None:
    predictions, bet_list, touched, _plan = _run(tmp_path, ruling)
    assert (
        (predictions / PREDICTIONS_2025)
        .read_text(encoding="utf-8")
        .endswith(f"{PREDICTIONS_2025}\n")
    )
    assert "2025_W01_A@B" in set(read_bet_list_artifact(bet_list)["game_id"])
    assert not any("2025" in path.name for path in touched)


@pytest.mark.parametrize("ruling", dispose.RULINGS)
def test_the_dry_run_list_equals_the_apply_list(tmp_path: Path, ruling: str) -> None:
    predictions, bet_list = _tree(tmp_path)
    dry = dispose.plan_disposition(ruling, predictions, bet_list).touched
    applied = dispose.apply_disposition(
        dispose.plan_disposition(ruling, predictions, bet_list), decided_on="2026-09-23"
    )
    assert applied == dry


def test_an_empty_season_is_a_no_op_for_every_branch(tmp_path: Path) -> None:
    """The measured production case: no 2026 artifact exists, so no branch writes."""
    predictions = tmp_path / "outputs" / "predictions"
    predictions.mkdir(parents=True)
    (predictions / PREDICTIONS_2025).write_text("game_id\nx\n", encoding="utf-8")
    bet_list = tmp_path / "outputs" / "bet_list"
    before = _snapshot(tmp_path)
    for ruling in dispose.RULINGS:
        plan = dispose.plan_disposition(ruling, predictions, bet_list)
        assert plan.touched == ()
        assert dispose.apply_disposition(plan, decided_on="2026-09-23") == ()
    assert _snapshot(tmp_path) == before


def test_the_files_are_resolved_through_the_pipeline_helpers() -> None:
    """One module decides where these artifacts live; this program does not re-type it."""
    import inspect

    source = inspect.getsource(dispose.main)
    assert "_predictions_output_dir()" in source
    assert "_bet_list_output_dir()" in source
