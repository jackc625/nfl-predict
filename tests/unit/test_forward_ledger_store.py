"""The ledger's guarded write path (Phase 34, Plan 34-04 Task 2; LDGR-01, LDGR-02, LDGR-03, D-13).

``forward_ledger.store.commit_changes`` is the ONE write path: it takes the OS writer lock, refuses
a broken chain, validates every incoming row, applies first pick stands, proves before the replace
that no prior entry's canonical bytes or chain hash moved, and writes atomically. These tests drive
each of those properties through the public functions, against a ledger under ``tmp_path`` only --
nothing here touches the repository's ``ledger/``, ``data/`` or ``artifacts/`` (COLD-05).

``make_row`` / ``graded`` are shared with ``test_forward_ledger_fill.py``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from forward_ledger.run_log import LEDGER_EVENTS, record_event
from utils.file_lock import FileLockHeldError, exclusive_file_lock

from api.cache import BET_LIST_COLUMNS
from backtest.weekly_bet_list import DecidedAfterFreezeError, PublishDeadlinePassedError
from data.write_sink import RecordingSink, active_sink
from forward_ledger import store
from forward_ledger.canonical import canonical_entry_bytes
from forward_ledger.declarations import UnknownFillConventionError, UnknownRecipeError
from forward_ledger.schema import LEDGER_ROW_KEY
from forward_ledger.store import (
    LEDGER_DIR,
    LEDGER_STAMP_COLUMNS,
    LEDGER_WRITER_LOCK_NAME,
    FirstPickStandsError,
    InvalidArmError,
    LedgerChainBrokenError,
    LedgerEntryRefusedError,
    LedgerWriterBusyError,
    MissingStampError,
    append_rows,
    apply_updates,
    classify_incoming,
    commit_changes,
    entries_to_frame,
    ledger_path,
    read_entries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixture builders (shared with test_forward_ledger_fill.py)
# ---------------------------------------------------------------------------


def make_row(**overrides: Any) -> dict[str, Any]:
    """One ledger-written forward row's immutable half: all 34 v1 columns, every stamp set."""
    row: dict[str, Any] = {
        "game_id": "2026_06_KC_BUF",
        "season": 2026,
        "week": 6,
        "target": "ats",
        "bet_side": "home",
        "model_value": -3.2,
        "market_value": -2.5,
        "line": -2.5,
        "slipped_line": -3.0,
        "calibrated_p_side": 0.5431,
        "per_bet_ev": 0.0377,
        "stake_units": 1.25,
        "ev_tier": "medium",
        "status": "live",
        "rejection_reason": None,
        "eligibility_label": "no_subpopulation",
        "snapshot_ts": "2026-10-14T17:00:48.334399-04:00",
        "freeze_ts": "2026-10-14T18:00:00-04:00",
        "selected_odds": -110.0,
        "flat_stake": 1.0,
        "provenance": "forward",
        "validation_type": "forward_realized",
        "decided_at_utc": "2026-10-14T21:18:10.874141+00:00",
        "arm": "live",
        "model_artifact_id": "ats_20261004_050228",
        "blend_id": "blend_20261004_050521",
        "recipe_id": "recipe-2026-row19-v1",
        "fill_convention_id": "fill-v1",
        "upstream_capture_key": "2026|6|3",
        "gold_generation_key": "gold-key-20261014",
        "odds_snapshot_digest": "a" * 64,
        "decision_snapshot_digest": "b" * 64,
        "verdict_scope": "verdict",
        "regime_label": None,
    }
    row.update(overrides)
    return row


def week_rows(week: int, games: tuple[str, ...]) -> list[dict[str, Any]]:
    """One row per game for *week*, decided the day before a Thursday lock of that week."""
    return [
        make_row(
            game_id=f"2026_{week:02d}_{game}",
            week=week,
            freeze_ts="2026-10-14T18:00:00-04:00",
        )
        for game in games
    ]


def key_of(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in LEDGER_ROW_KEY)


def graded(status: str, payout: float, units: float) -> dict[str, Any]:
    """A grading update moving a row to a terminal status."""
    return {
        "grading_status": status,
        "outcome": {"win": True, "loss": False, "push": None}[status],
        "payout_flat": payout,
        "realized_units": units,
        "graded_at": datetime(2026, 10, 19, 21, 0, tzinfo=UTC),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(ledger_dir: Path) -> list[bytes]:
    return ledger_path(ledger_dir).read_bytes().split(b"\n")[:-1]


# ---------------------------------------------------------------------------
# Append-only, first pick stands
# ---------------------------------------------------------------------------


def test_append_next_week_and_grade_keeps_prior_bytes(tmp_path: Path) -> None:
    week_n = week_rows(6, ("KC_BUF", "DAL_PHI", "SF_SEA"))
    append_rows(tmp_path, week_n)
    apply_updates(
        tmp_path,
        grading_updates={
            key_of(week_n[0]): graded("win", 0.9090909090909091, 1.1363636363636365),
            key_of(week_n[1]): graded("loss", -1.0, -1.25),
        },
    )
    before = read_entries(tmp_path)
    before_lines = _lines(tmp_path)
    settled = [0, 1]

    result = commit_changes(
        tmp_path,
        new_rows=week_rows(7, ("NYJ_MIA", "ARI_ATL")),
        grading_updates={key_of(week_n[2]): graded("push", 0.0, 0.0)},
    )

    after = read_entries(tmp_path)
    assert result.wrote is True
    assert result.appended_rows == 2
    assert result.grading_changed == 1
    assert len(after) == 5
    for old, new in zip(before, after, strict=False):
        assert canonical_entry_bytes(old.kind, old.immutable) == canonical_entry_bytes(
            new.kind, new.immutable
        )
        assert old.chain_hash == new.chain_hash
    after_lines = _lines(tmp_path)
    for index in settled:
        assert after_lines[index] == before_lines[index]
    assert after[2].grading is not None
    assert after[2].grading["grading_status"] == "push"
    assert after[3].immutable["week"] == 7
    assert store.verify_chain(after).ok


def test_identical_repeat_is_a_noop(tmp_path: Path) -> None:
    rows = week_rows(6, ("KC_BUF", "DAL_PHI"))
    append_rows(tmp_path, rows)
    digest = _sha(ledger_path(tmp_path))

    # The same decision observed by a later run: new observation time, new nightly gold key.
    repeat = [
        {
            **row,
            "decided_at_utc": "2026-10-14T21:40:00+00:00",
            "gold_generation_key": "gold-key-20261014-rerun",
            "snapshot_ts": "2026-10-14T17:35:00-04:00",
        }
        for row in rows
    ]
    result = append_rows(tmp_path, repeat)

    assert result.wrote is False
    assert result.appended_rows == 0
    assert _sha(ledger_path(tmp_path)) == digest


def test_different_repeat_is_refused(tmp_path: Path) -> None:
    row = make_row()
    append_rows(tmp_path, [row])
    digest = _sha(ledger_path(tmp_path))

    with pytest.raises(FirstPickStandsError, match="bet_side") as caught:
        append_rows(tmp_path, [{**row, "bet_side": "away"}])

    assert caught.value.key == key_of(row)
    assert caught.value.fields == ("bet_side",)
    assert _sha(ledger_path(tmp_path)) == digest
    assert read_entries(tmp_path)[0].immutable["bet_side"] == "home"


def test_classify_incoming_partitions(tmp_path: Path) -> None:
    rows = week_rows(6, ("KC_BUF", "DAL_PHI"))
    append_rows(tmp_path, rows)
    digest = _sha(ledger_path(tmp_path))
    entries = read_entries(tmp_path)

    incoming = [
        {
            **rows[0],
            "decided_at_utc": "2026-10-14T21:59:00+00:00",
        },  # identical decision
        {**rows[1], "line": -1.5},  # a different decision for a stored key
        make_row(game_id="2026_06_SF_SEA"),  # a new key
    ]
    classification = classify_incoming(entries, incoming)

    assert [key_of(v) for v in classification.identical] == [key_of(rows[0])]
    assert [conflict[0] for conflict in classification.conflicting] == [key_of(rows[1])]
    assert classification.conflicting[0][1] == ("line",)
    assert [v["game_id"] for v in classification.new] == ["2026_06_SF_SEA"]
    assert _sha(ledger_path(tmp_path)) == digest


# ---------------------------------------------------------------------------
# Atomicity, the single writer, and the empty commit
# ---------------------------------------------------------------------------


def test_interrupted_write_leaves_prior_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    append_rows(tmp_path, [make_row()])
    digest = _sha(ledger_path(tmp_path))

    def _half_then_interrupt(tmp: Path, payload: bytes) -> None:
        tmp.write_bytes(payload[: len(payload) // 2])
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "_write_payload", _half_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        append_rows(tmp_path, [make_row(game_id="2026_06_DAL_PHI")])

    assert _sha(ledger_path(tmp_path)) == digest
    assert list(tmp_path.glob("*.tmp")) == []
    # The lock was released on the way out: the next writer proceeds.
    monkeypatch.undo()
    assert append_rows(tmp_path, [make_row(game_id="2026_06_DAL_PHI")]).wrote is True


def test_second_writer_is_refused(tmp_path: Path) -> None:
    append_rows(tmp_path, [make_row()])
    digest = _sha(ledger_path(tmp_path))

    with (
        exclusive_file_lock(tmp_path / LEDGER_WRITER_LOCK_NAME),
        pytest.raises(LedgerWriterBusyError, match="writer"),
    ):
        append_rows(tmp_path, [make_row(game_id="2026_06_DAL_PHI")])

    assert _sha(ledger_path(tmp_path)) == digest


def test_lock_released_when_holder_dies(tmp_path: Path) -> None:
    lock_path = tmp_path / LEDGER_WRITER_LOCK_NAME
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys, time\n"
                "from utils.file_lock import exclusive_file_lock\n"
                "lock = exclusive_file_lock(sys.argv[1])\n"
                "lock.__enter__()\n"
                "print('LOCKED', flush=True)\n"
                "time.sleep(120)\n"
            ),
            str(lock_path),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "LOCKED"
        # While the holder lives, the lock is really held across processes.
        with pytest.raises(FileLockHeldError), exclusive_file_lock(lock_path):
            pass
    finally:
        holder.kill()
        holder.wait(timeout=30)

    # The OS released it when the holder died: no stale lock, no recovery logic.
    with exclusive_file_lock(lock_path, wait_seconds=10.0):
        pass


def test_zero_change_commit_writes_nothing(tmp_path: Path) -> None:
    absent = tmp_path / "absent"
    empty = commit_changes(absent)
    assert empty.wrote is False
    assert not ledger_path(absent).exists()

    append_rows(tmp_path, [make_row()])
    path = ledger_path(tmp_path)
    digest = _sha(path)
    mtime = path.stat().st_mtime_ns

    result = commit_changes(tmp_path)

    assert result.wrote is False
    assert result.entry_count == 1
    assert _sha(path) == digest
    assert path.stat().st_mtime_ns == mtime


# ---------------------------------------------------------------------------
# Row rules: arm, stamps, freeze, chain, deadline
# ---------------------------------------------------------------------------


def test_live_and_shadow_both_persist(tmp_path: Path) -> None:
    live = make_row(arm="live")
    shadow = make_row(arm="shadow")
    append_rows(tmp_path, [live, shadow])

    entries = read_entries(tmp_path)
    assert len(entries) == 2
    by_key = {tuple(e.immutable[n] for n in LEDGER_ROW_KEY): e for e in entries}
    assert by_key[key_of(live)].immutable["arm"] == "live"
    assert by_key[key_of(shadow)].immutable["arm"] == "shadow"


@pytest.mark.parametrize("arm", [None, "other"])
def test_null_and_unknown_arm_refused(tmp_path: Path, arm: str | None) -> None:
    with pytest.raises(InvalidArmError, match="arm"):
        append_rows(tmp_path, [make_row(arm=arm)])
    assert not ledger_path(tmp_path).exists()


@pytest.mark.parametrize("column", LEDGER_STAMP_COLUMNS)
def test_missing_stamp_refused(tmp_path: Path, column: str) -> None:
    assert len(LEDGER_STAMP_COLUMNS) == 8
    with pytest.raises(MissingStampError, match=column):
        append_rows(tmp_path, [make_row(**{column: None})])
    assert not ledger_path(tmp_path).exists()


def test_unknown_recipe_refused(tmp_path: Path) -> None:
    with pytest.raises(UnknownRecipeError, match="recipe-unknown"):
        append_rows(tmp_path, [make_row(recipe_id="recipe-unknown")])
    assert not ledger_path(tmp_path).exists()


def test_missing_fill_id_refused(tmp_path: Path) -> None:
    with pytest.raises(MissingStampError, match="fill_convention_id"):
        append_rows(tmp_path, [make_row(fill_convention_id=None)])
    with pytest.raises(UnknownFillConventionError, match="fill-v2"):
        append_rows(tmp_path, [make_row(fill_convention_id="fill-v2")])
    assert not ledger_path(tmp_path).exists()


def test_decided_after_freeze_refused(tmp_path: Path) -> None:
    late = make_row(decided_at_utc="2026-10-14T22:00:01+00:00")  # 18:00:01 EDT
    with pytest.raises(DecidedAfterFreezeError):
        append_rows(tmp_path, [late])
    assert not ledger_path(tmp_path).exists()


def test_broken_chain_blocks_writing(tmp_path: Path) -> None:
    append_rows(tmp_path, [make_row()])
    path = ledger_path(tmp_path)
    original = path.read_bytes()
    assert original.count(b'"model_value":-3.2,') == 1
    path.write_bytes(original.replace(b'"model_value":-3.2,', b'"model_value":-3.3,'))
    tampered = path.read_bytes()

    with pytest.raises(LedgerChainBrokenError, match="seq 0"):
        append_rows(tmp_path, [make_row(game_id="2026_06_DAL_PHI")])
    assert path.read_bytes() == tampered


def test_publish_deadline(tmp_path: Path) -> None:
    deadline = datetime(2026, 10, 14, 22, 0, tzinfo=UTC)
    with pytest.raises(PublishDeadlinePassedError):
        append_rows(
            tmp_path,
            [make_row()],
            publish_by=deadline,
            clock=lambda: datetime(2026, 10, 14, 22, 0, 1, tzinfo=UTC),
        )
    assert not ledger_path(tmp_path).exists()

    on_time = append_rows(
        tmp_path, [make_row()], publish_by=deadline, clock=lambda: deadline
    )
    assert on_time.wrote is True


def test_correction_requires_a_terminal_graded_row(tmp_path: Path) -> None:
    row = make_row()
    append_rows(tmp_path, [row])
    correction = {
        "game_id": row["game_id"],
        "season": 2026,
        "week": 6,
        "target": "ats",
        "arm": "live",
        "original_grading_status": "win",
        "original_payout_flat": 0.9090909090909091,
        "prior_grading_status": "win",
        "prior_payout_flat": 0.9090909090909091,
        "prior_realized_units": 1.1363636363636365,
        "prior_correction_seq": None,
        "corrected_grading_status": "loss",
        "corrected_outcome": False,
        "corrected_payout_flat": -1.0,
        "corrected_realized_units": -1.25,
        "realized_value": -4.0,
        "source_dataset": "schedules",
        "source_season": 2026,
        "source_week": 6,
        "source_sequence": 4,
        "detected_at_utc": "2026-10-21T21:00:00+00:00",
        "corrected_at_utc": "2026-10-21T21:00:05+00:00",
    }
    with pytest.raises(LedgerEntryRefusedError, match="terminal"):
        commit_changes(tmp_path, new_corrections=[correction])

    apply_updates(
        tmp_path,
        grading_updates={
            key_of(row): graded("win", 0.9090909090909091, 1.1363636363636365)
        },
    )
    result = commit_changes(tmp_path, new_corrections=[correction])

    entries = read_entries(tmp_path)
    assert result.appended_corrections == 1
    assert [e.kind for e in entries] == ["row", "correction"]
    assert entries[0].grading is not None
    assert entries[0].grading["grading_status"] == "win"
    assert store.verify_chain(entries).ok


# ---------------------------------------------------------------------------
# Location, frame, sink, run log, and the one replace caller
# ---------------------------------------------------------------------------


def test_store_not_under_outputs() -> None:
    assert Path("ledger") == LEDGER_DIR
    assert "outputs" not in ledger_path(LEDGER_DIR).parts


def test_entries_to_frame_matches_bet_list_dtypes(tmp_path: Path) -> None:
    rows = week_rows(6, ("KC_BUF", "DAL_PHI"))
    append_rows(tmp_path, rows)
    apply_updates(
        tmp_path,
        grading_updates={
            key_of(rows[0]): graded("win", 0.9090909090909091, 1.1363636363636365)
        },
    )

    frame = entries_to_frame(read_entries(tmp_path))

    assert list(frame.columns) == BET_LIST_COLUMNS
    assert len(frame.columns) == 51
    assert list(frame["game_id"]) == ["2026_06_KC_BUF", "2026_06_DAL_PHI"]
    assert isinstance(frame["graded_at"].dtype, pd.DatetimeTZDtype)
    assert str(frame["graded_at"].dtype.tz) == "UTC"
    assert frame["graded_at"].iloc[0] == pd.Timestamp("2026-10-19T21:00:00+00:00")
    assert pd.isna(frame["graded_at"].iloc[1])
    assert frame["outcome"].dtype == object
    assert frame["season"].dtype == "int64"
    assert frame["model_value"].dtype == "float64"
    assert frame["fill_line"].isna().all()


def test_recording_sink_writes_nothing(tmp_path: Path) -> None:
    with active_sink(RecordingSink()) as sink:
        result = append_rows(tmp_path, [make_row()])

    assert result.wrote is False
    assert not ledger_path(tmp_path).exists()
    assert [w.kind for w in sink.intended_writes] == ["ledger_commit"]
    assert sink.intended_writes[0].target == str(ledger_path(tmp_path))


def test_run_log_records_and_refuses(tmp_path: Path) -> None:
    log = tmp_path / "ledger_runs.jsonl"
    assert "first_pick_refusal" in LEDGER_EVENTS
    assert record_event("append", path=log, appended_rows=2) is True

    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["event"] == "append"
    assert record["appended_rows"] == 2
    assert datetime.fromisoformat(record["at_utc"]).tzinfo is not None
    with pytest.raises(ValueError, match="not_an_event"):
        record_event("not_an_event", path=log)


def test_only_the_store_calls_replace_atomically() -> None:
    package = REPO_ROOT / "forward_ledger"
    callers = set()
    for source in sorted(package.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else getattr(func, "attr", None)
                )
                if name == "_replace_atomically":
                    callers.add(source.name)
    assert callers == {"store.py"}
    assert "def commit_changes(" in (package / "store.py").read_text(encoding="utf-8")
