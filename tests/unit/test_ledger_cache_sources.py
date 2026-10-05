"""The cutover-aware web-cache sources (Phase 34, Plan 34-16 Task 2; LDGR-01, LDGR-08, D-06).

``forward_ledger.cache_sources.read_bet_cache_sources`` is the ONE reader both cache builders use
(the scheduled ``step_populate_web_cache`` and the manual ``scripts/populate_cache.py``):

* switch OFF -- exactly today's sources (``read_bet_list_cache_sources``), no corrections, an
  undeclared verdict context;
* switch ON -- forward rows from the chain-verified ledger UNIONED with the 2025 replay rows in
  ``outputs/bet_list``; a forward row still found in ``outputs/bet_list`` is refused by name (one
  forward store, research Pitfall 2), and an absent or broken ledger is refused, never served as
  "nothing recommended".

In both states the result strip's rows come from ``graded_outcome_rows`` over the very frame the
sources return (review finding 4).

Synthetic ledgers and artifacts only; everything lives under ``tmp_path`` (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from forward_ledger.cache_sources import (
    BetCacheSources,
    ForwardRowsOutsideLedgerError,
    LedgerAbsentAfterCutoverError,
    read_bet_cache_sources,
)

from api.cache import BET_LIST_COLUMNS, BET_LIST_CORRECTIONS_COLUMNS
from backtest.bet_tracker import (
    aggregate_all_blocks,
    graded_outcome_rows,
    to_tracker_frame,
)
from backtest.weekly_bet_list import (
    read_bet_list_artifact,
    read_bet_list_cache_sources,
    write_bet_list_pair,
)
from forward_ledger import cache_sources, cutover
from forward_ledger.declarations import VerdictScope, VerdictScopeUndeclaredError
from forward_ledger.store import (
    LedgerChainBrokenError,
    append_rows,
    apply_updates,
    commit_changes,
    ledger_path,
)
from tests.unit.test_forward_ledger_store import graded, key_of, make_row

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCOPE = VerdictScope(
    season=2026,
    start_week=6,
    end_week=22,
    includes_playoff_weeks=True,
    includes_neutral_site_games=True,
    counted_arm="live",
    outcome_rule="in_force_corrected_outcome",
    fill_convention_id="fill-v1",
    bootstrap_regime_weeks=(2, 3, 4),
)

_WIN = graded("win", 0.9090909090909091, 1.1363636363636365)
_LOSS = graded("loss", -1.0, -1.25)


def _stored_row(
    game_id: str,
    *,
    season: int,
    provenance: str,
    validation_type: str,
    grading_status: str,
) -> dict[str, Any]:
    """One row of the durable ``outputs/bet_list`` artifact, all 51 columns."""
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": season,
            "week": 3,
            "target": "ou",
            "bet_side": "under",
            "status": "live",
            "provenance": provenance,
            "validation_type": validation_type,
            "flat_stake": 1.0,
            "selected_odds": -110.0,
            "grading_status": grading_status,
            "outcome": {"win": True, "loss": False}.get(grading_status),
            "payout_flat": {"win": 0.909, "loss": -1.0}.get(grading_status),
        }
    )
    return row


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=pd.Index(BET_LIST_COLUMNS))
    frame["season"] = frame["season"].astype("int64")
    frame["week"] = frame["week"].astype("int64")
    write_bet_list_pair(frame, output_dir)


def _replay_rows() -> list[dict[str, Any]]:
    return [
        _stored_row(
            "2025_03_KC_BUF",
            season=2025,
            provenance="backtest_replay",
            validation_type="clean_holdout",
            grading_status="win",
        ),
        _stored_row(
            "2025_03_DAL_NYG",
            season=2025,
            provenance="backtest_replay",
            validation_type="clean_holdout",
            grading_status="loss",
        ),
    ]


def _forward_output_row() -> dict[str, Any]:
    return _stored_row(
        "2026_03_SF_LA",
        season=2026,
        provenance="forward",
        validation_type="forward_realized",
        grading_status="win",
    )


def _ledger_rows() -> list[dict[str, Any]]:
    return [
        make_row(game_id="2026_04_KC_BUF", week=4, verdict_scope="pre_verdict"),
        make_row(game_id="2026_06_DAL_PHI"),
        make_row(game_id="2026_06_NYJ_MIA"),
    ]


def _build_ledger(ledger_dir: Path) -> list[dict[str, Any]]:
    rows = _ledger_rows()
    append_rows(ledger_dir, rows)
    apply_updates(
        ledger_dir,
        grading_updates={
            key_of(rows[0]): _WIN,
            key_of(rows[1]): _WIN,
            key_of(rows[2]): _LOSS,
        },
    )
    return rows


def _correct_to_loss(ledger_dir: Path, row: dict[str, Any]) -> None:
    commit_changes(
        ledger_dir,
        new_corrections=[
            {
                "game_id": row["game_id"],
                "season": row["season"],
                "week": row["week"],
                "target": row["target"],
                "arm": row["arm"],
                "original_grading_status": "win",
                "original_payout_flat": _WIN["payout_flat"],
                "prior_grading_status": "win",
                "prior_payout_flat": _WIN["payout_flat"],
                "prior_realized_units": _WIN["realized_units"],
                "prior_correction_seq": None,
                "corrected_grading_status": "loss",
                "corrected_outcome": False,
                "corrected_payout_flat": -1.0,
                "corrected_realized_units": -1.25,
                "realized_value": -10.0,
                "source_dataset": "schedules",
                "source_season": 2026,
                "source_week": 7,
                "source_sequence": 2,
                "detected_at_utc": "2026-10-21T21:00:00+00:00",
                "corrected_at_utc": "2026-10-21T21:00:05+00:00",
            }
        ],
    )


@pytest.fixture
def switch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", True)


@pytest.fixture
def undeclared(monkeypatch: pytest.MonkeyPatch) -> None:
    def _undeclared() -> VerdictScope:
        raise VerdictScopeUndeclaredError("no declaration in this test")

    monkeypatch.setattr(cache_sources, "load_verdict_scope", _undeclared)


def _dirs(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_dir": tmp_path / "outputs",
        "ledger_dir": tmp_path / "ledger",
        "silver_dir": tmp_path / "silver",
    }


def test_switch_off_sources_equal_today(tmp_path: Path) -> None:
    """Switch off: today's three frames, no corrections, an undeclared verdict context."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], [*_replay_rows(), _forward_output_row()])
    # A ledger present on disk is not read while the switch is off.
    _build_ledger(dirs["ledger_dir"])

    sources = read_bet_cache_sources(**dirs)
    today = read_bet_list_cache_sources(dirs["output_dir"], dirs["silver_dir"])

    assert isinstance(sources, BetCacheSources)
    pd.testing.assert_frame_equal(sources.bet_list, today.bet_list)
    pd.testing.assert_frame_equal(sources.tracker, today.tracker)
    pd.testing.assert_frame_equal(sources.schedule, today.schedule)
    assert sources.corrections.empty
    assert list(sources.corrections.columns) == list(BET_LIST_CORRECTIONS_COLUMNS)
    assert sources.verdict_context == {
        "declared": False,
        "season": None,
        "start_week": None,
    }


def test_switch_on_unions_ledger_and_replay(
    tmp_path: Path, switch_on: None, undeclared: None
) -> None:
    """2 replay rows in outputs + 3 ledger rows -> 5 rows; the tracker is recomputed over them."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], _replay_rows())
    _build_ledger(dirs["ledger_dir"])

    sources = read_bet_cache_sources(**dirs)

    assert len(sources.bet_list) == 5
    assert list(sources.bet_list.columns) == list(BET_LIST_COLUMNS)
    assert (
        sorted(sources.bet_list["provenance"])
        == ["backtest_replay"] * 2 + ["forward"] * 3
    )
    expected_tracker = to_tracker_frame(
        aggregate_all_blocks(sources.bet_list, corrections=sources.corrections)
    )
    pd.testing.assert_frame_equal(sources.tracker, expected_tracker)
    by_pair = {
        (row.provenance, row.validation_type): row
        for row in sources.tracker.itertuples(index=False)
    }
    assert by_pair[("forward", "pre_verdict")].wins == 1
    assert (
        by_pair[("forward", "forward_realized")].wins,
        by_pair[("forward", "forward_realized")].losses,
    ) == (1, 1)


def test_forward_rows_in_outputs_refused(
    tmp_path: Path, switch_on: None, undeclared: None
) -> None:
    """Switch on and a forward row still in outputs: refused by name, naming the count."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], [*_replay_rows(), _forward_output_row()])
    _build_ledger(dirs["ledger_dir"])

    with pytest.raises(ForwardRowsOutsideLedgerError, match=r"\b1 forward row"):
        read_bet_cache_sources(**dirs)


def test_broken_ledger_refused_not_degraded(
    tmp_path: Path, switch_on: None, undeclared: None
) -> None:
    """A tampered ledger raises; it is never served as an empty record."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], _replay_rows())
    _build_ledger(dirs["ledger_dir"])
    path = ledger_path(dirs["ledger_dir"])
    text = path.read_bytes().decode("utf-8")
    assert '"home"' in text
    path.write_bytes(text.replace('"home"', '"away"', 1).encode("utf-8"))

    with pytest.raises(LedgerChainBrokenError):
        read_bet_cache_sources(**dirs)


def test_absent_ledger_with_switch_on_refused(
    tmp_path: Path, switch_on: None, undeclared: None
) -> None:
    """Switch on with no ledger file is a cutover without migration: refused by name."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], _replay_rows())

    with pytest.raises(LedgerAbsentAfterCutoverError, match=r"forward_2026\.jsonl"):
        read_bet_cache_sources(**dirs)


def test_verdict_context_from_declaration(
    tmp_path: Path, switch_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed declaration with W=6 is stamped as declared; none is undeclared."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], _replay_rows())
    _build_ledger(dirs["ledger_dir"])

    monkeypatch.setattr(cache_sources, "load_verdict_scope", lambda: _SCOPE)
    assert read_bet_cache_sources(**dirs).verdict_context == {
        "declared": True,
        "season": 2026,
        "start_week": 6,
    }

    def _undeclared() -> VerdictScope:
        raise VerdictScopeUndeclaredError("no declaration in this test")

    monkeypatch.setattr(cache_sources, "load_verdict_scope", _undeclared)
    assert read_bet_cache_sources(**dirs).verdict_context["declared"] is False


def test_corrections_frame_from_in_force(
    tmp_path: Path, switch_on: None, undeclared: None
) -> None:
    """One correction entry -> one corrections row with the corrected and original statuses."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], _replay_rows())
    rows = _build_ledger(dirs["ledger_dir"])
    _correct_to_loss(dirs["ledger_dir"], rows[1])

    sources = read_bet_cache_sources(**dirs)

    assert list(sources.corrections.columns) == list(BET_LIST_CORRECTIONS_COLUMNS)
    assert len(sources.corrections) == 1
    correction = sources.corrections.iloc[0]
    assert correction["game_id"] == rows[1]["game_id"]
    assert correction["original_grading_status"] == "win"
    assert correction["corrected_grading_status"] == "loss"
    assert correction["corrected_payout_flat"] == -1.0
    # The bet list keeps the original grade; the tracker counts the corrected one.
    stored = sources.bet_list.set_index("game_id").loc[rows[1]["game_id"]]
    assert stored["grading_status"] == "win"
    verdict_block = sources.tracker[
        sources.tracker["validation_type"] == "forward_realized"
    ].iloc[0]
    assert (verdict_block["wins"], verdict_block["losses"]) == (0, 2)


def test_graded_outcomes_from_the_same_union(
    tmp_path: Path, switch_on: None, undeclared: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strip rows are ``graded_outcome_rows`` over the frame returned, in both states."""
    dirs = _dirs(tmp_path)
    _write_outputs(dirs["output_dir"], _replay_rows())
    rows = _build_ledger(dirs["ledger_dir"])
    _correct_to_loss(dirs["ledger_dir"], rows[1])

    on = read_bet_cache_sources(**dirs)
    pd.testing.assert_frame_equal(
        on.graded_outcomes,
        graded_outcome_rows(on.bet_list, corrections=on.corrections),
    )
    fixed = on.graded_outcomes.set_index("game_id").loc[rows[1]["game_id"]]
    assert (fixed["grading_status"], bool(fixed["corrected"])) == ("loss", True)

    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", False)
    off_dirs = _dirs(tmp_path / "off")
    _write_outputs(off_dirs["output_dir"], [*_replay_rows(), _forward_output_row()])
    off = read_bet_cache_sources(**off_dirs)
    pd.testing.assert_frame_equal(
        off.graded_outcomes,
        graded_outcome_rows(read_bet_list_artifact(off_dirs["output_dir"])),
    )
    assert len(off.graded_outcomes) == 3
    assert not off.graded_outcomes["corrected"].any()


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_step_and_cli_use_the_same_reader() -> None:
    """The scheduled step and the manual recovery command call the one reader."""
    for relative in ("pipeline/steps.py", "scripts/populate_cache.py"):
        called = _calls(REPO_ROOT / relative)
        assert "read_bet_cache_sources" in called, relative
        assert "read_bet_list_cache_sources" not in called, relative
