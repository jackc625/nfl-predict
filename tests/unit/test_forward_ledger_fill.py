"""The three one-way mutable halves -- grading, fill, closing (Phase 34, Plan 34-04 Task 3; LDGR-04).

Each mutable half of a ledger row is written only through its own path and only one way: grading
out of ``pending`` through ``api.cache.assert_grading_transition``, each real-fill column from NULL
to a value once, and the closing half set once (values, or a recorded NULL reason) and never
overwritten. None of them can touch the immutable half or a chain hash (LDGR-05).

Every test writes under ``tmp_path`` only (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from forward_ledger.canonical import canonical_entry_bytes
from forward_ledger.schema import BET_LIST_FILL_COLUMNS
from forward_ledger.store import (
    ColumnSetViolationError,
    append_rows,
    apply_updates,
    entries_to_frame,
    ledger_path,
    read_entries,
)
from forward_ledger.transitions import (
    CLOSING_SET,
    FILL_SET,
    GRADING_SET,
    ClosingAlreadySetError,
    FillAlreadyRecordedError,
)
from tests.unit.test_forward_ledger_store import graded, key_of, week_rows

_WIN = graded("win", 0.9090909090909091, 1.1363636363636365)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closing_values(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "closing_line": -3.5,
        "closing_odds": -112.0,
        "closing_sportsbook": "draftkings",
        "closing_captured_at": "2026-10-18T16:58:02+00:00",
        "forward_clv": -0.5,
        "closing_null_reason": None,
    }
    values.update(overrides)
    return values


def _ledger(tmp_path: Path) -> list[dict[str, Any]]:
    rows = week_rows(6, ("KC_BUF", "DAL_PHI"))
    append_rows(tmp_path, rows)
    return rows


def test_paper_rows_have_null_fill(tmp_path: Path) -> None:
    _ledger(tmp_path)
    for entry in read_entries(tmp_path):
        assert entry.fill == dict.fromkeys(BET_LIST_FILL_COLUMNS)
    frame = entries_to_frame(read_entries(tmp_path))
    assert frame[BET_LIST_FILL_COLUMNS].isna().all().all()
    assert set(BET_LIST_FILL_COLUMNS) == FILL_SET
    assert GRADING_SET.isdisjoint(FILL_SET) and FILL_SET.isdisjoint(CLOSING_SET)


def test_fill_null_to_value_once(tmp_path: Path) -> None:
    rows = _ledger(tmp_path)
    key = key_of(rows[0])

    first = apply_updates(
        tmp_path,
        fill_updates={key: {"fill_sportsbook": "draftkings", "fill_line": -3.0}},
    )
    assert first.wrote is True
    assert first.fill_changed == 1
    # A still-NULL column of the same row may be recorded later, once.
    apply_updates(tmp_path, fill_updates={key: {"fill_odds": -110.0}})
    fill = read_entries(tmp_path)[0].fill
    assert fill is not None
    assert (fill["fill_sportsbook"], fill["fill_line"], fill["fill_odds"]) == (
        "draftkings",
        -3.0,
        -110.0,
    )
    digest = _sha(ledger_path(tmp_path))

    with pytest.raises(FillAlreadyRecordedError, match="fill_line"):
        apply_updates(tmp_path, fill_updates={key: {"fill_line": -2.5}})
    with pytest.raises(FillAlreadyRecordedError, match="fill_odds"):
        apply_updates(tmp_path, fill_updates={key: {"fill_odds": -110.0}})
    assert _sha(ledger_path(tmp_path)) == digest


def test_fill_leaves_immutable_and_grading_bytes(tmp_path: Path) -> None:
    rows = _ledger(tmp_path)
    key = key_of(rows[0])
    apply_updates(tmp_path, grading_updates={key: _WIN})
    before = read_entries(tmp_path)[0]

    apply_updates(
        tmp_path,
        fill_updates={
            key: {
                "fill_sportsbook": "draftkings",
                "fill_line": -3.0,
                "fill_odds": -110.0,
                "fill_stake_dollars": 125.0,
                "fill_at_utc": "2026-10-14T22:30:00+00:00",
            }
        },
    )
    after = read_entries(tmp_path)[0]

    assert canonical_entry_bytes(after.kind, after.immutable) == canonical_entry_bytes(
        before.kind, before.immutable
    )
    assert after.chain_hash == before.chain_hash
    assert json.dumps(after.grading) == json.dumps(before.grading)
    assert after.fill is not None
    assert after.fill["fill_stake_dollars"] == 125.0


def test_grading_path_cannot_write_fill(tmp_path: Path) -> None:
    rows = _ledger(tmp_path)
    digest = _sha(ledger_path(tmp_path))
    with pytest.raises(ColumnSetViolationError, match="fill_line"):
        apply_updates(
            tmp_path, grading_updates={key_of(rows[0]): {**_WIN, "fill_line": -3.0}}
        )
    assert _sha(ledger_path(tmp_path)) == digest


@pytest.mark.parametrize("column", ["bet_side", "grading_status"])
def test_fill_path_cannot_write_immutable_or_grading(
    tmp_path: Path, column: str
) -> None:
    rows = _ledger(tmp_path)
    digest = _sha(ledger_path(tmp_path))
    with pytest.raises(ColumnSetViolationError, match=column):
        apply_updates(tmp_path, fill_updates={key_of(rows[0]): {column: "away"}})
    assert _sha(ledger_path(tmp_path)) == digest


def test_grading_reuses_one_way_rule(tmp_path: Path) -> None:
    rows = _ledger(tmp_path)
    key = key_of(rows[0])
    assert apply_updates(tmp_path, grading_updates={key: _WIN}).grading_changed == 1
    digest = _sha(ledger_path(tmp_path))

    with pytest.raises(ValueError, match="assert_grading_transition"):
        apply_updates(tmp_path, grading_updates={key: graded("loss", -1.0, -1.25)})
    assert _sha(ledger_path(tmp_path)) == digest


def test_closing_set_once_idempotent(tmp_path: Path) -> None:
    rows = _ledger(tmp_path)
    priced, missed = key_of(rows[0]), key_of(rows[1])

    first = apply_updates(tmp_path, closing_updates={priced: _closing_values()})
    assert first.wrote is True
    assert first.closing_changed == 1
    digest = _sha(ledger_path(tmp_path))

    again = apply_updates(tmp_path, closing_updates={priced: _closing_values()})
    assert again.wrote is False
    assert again.closing_changed == 0
    assert _sha(ledger_path(tmp_path)) == digest

    with pytest.raises(ClosingAlreadySetError):
        apply_updates(
            tmp_path, closing_updates={priced: _closing_values(closing_line=-4.0)}
        )

    reason = dict.fromkeys(CLOSING_SET) | {"closing_null_reason": "capture_missed"}
    assert apply_updates(tmp_path, closing_updates={missed: reason}).wrote is True
    with pytest.raises(ClosingAlreadySetError):
        apply_updates(tmp_path, closing_updates={missed: _closing_values()})
    with pytest.raises(ClosingAlreadySetError):
        apply_updates(
            tmp_path,
            closing_updates={
                priced: _closing_values(closing_null_reason="not_a_reason")
            },
        )
    assert read_entries(tmp_path)[1].closing == reason


def test_no_hash_changes_after_mutable_writes(tmp_path: Path) -> None:
    rows = _ledger(tmp_path)
    hashes = [entry.chain_hash for entry in read_entries(tmp_path)]
    key = key_of(rows[0])

    apply_updates(tmp_path, grading_updates={key: _WIN})
    apply_updates(tmp_path, fill_updates={key: {"fill_sportsbook": "draftkings"}})
    apply_updates(tmp_path, closing_updates={key: _closing_values()})

    entries = read_entries(tmp_path)
    assert [entry.chain_hash for entry in entries] == hashes
    assert entries[0].grading is not None and entries[0].fill is not None
    assert entries[0].grading["grading_status"] == "win"
    assert entries[0].fill["fill_sportsbook"] == "draftkings"
    assert entries[0].closing == _closing_values()
