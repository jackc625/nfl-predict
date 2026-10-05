"""The forward ledger's spine, proven end to end (Phase 34, Plan 34-01 Task 1; LDGR-05/06).

ONE immutable row travels through every layer the record-keeping core has: schema-typed canonical
bytes, the per-row hash chain from the published genesis constant, the one-file JSON-lines store
written atomically, and the verification CLI run as a REAL subprocess. If canonicalization or
chaining is wrong here, every later plan inherits the defect, which is why the tracer is proven
before anything is built on it.

Every test writes under ``tmp_path``. Nothing here reads or writes the repository's own
``ledger/`` directory (COLD-05: tests never touch a production store).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from forward_ledger.canonical import (
    CORRECTION_COLUMN_TYPES_V1,
    CORRECTION_COLUMNS_V1,
    ENTRY_KIND_CORRECTION,
    ENTRY_KIND_ROW,
    GENESIS_HASH,
    GENESIS_SEED,
    IMMUTABLE_COLUMN_TYPES_V1,
    IMMUTABLE_COLUMNS_V1,
    CanonicalValueError,
    canonical_entry_bytes,
    chain_hash,
)
from forward_ledger.store import (
    LedgerFormatError,
    build_entry,
    ledger_head,
    ledger_path,
    read_entries,
    verify_chain,
    write_entries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SOURCE = REPO_ROOT / "forward_ledger" / "canonical.py"


def _row(**overrides: Any) -> dict[str, Any]:
    """One realistic forward row's immutable half: all 34 ``IMMUTABLE_COLUMNS_V1`` names."""
    row: dict[str, Any] = {
        "game_id": "2026_W06_KC@BUF",
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
        "snapshot_ts": "2026-10-10T17:00:48.334399-04:00",
        "freeze_ts": "2026-10-10T18:00:00-04:00",
        "selected_odds": -110.0,
        "flat_stake": 1.0,
        "provenance": "forward",
        "validation_type": "forward_realized",
        "decided_at_utc": "2026-10-10T21:18:10.874141+00:00",
        "arm": "live",
        "model_artifact_id": "ats_20261004_120000",
        "blend_id": "blend_dynamic_20260606_020635",
        "recipe_id": "recipe-v1",
        "fill_convention_id": "fill-v1",
        "upstream_capture_key": "2026|6|3",
        "gold_generation_key": "gold_20261010_210000",
        "odds_snapshot_digest": "a" * 64,
        "decision_snapshot_digest": "b" * 64,
        "verdict_scope": "verdict",
        "regime_label": None,
    }
    row.update(overrides)
    return row


def _correction(**overrides: Any) -> dict[str, Any]:
    """One correction entry's payload: all 22 ``CORRECTION_COLUMNS_V1`` names."""
    entry: dict[str, Any] = {
        "game_id": "2026_W06_KC@BUF",
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
        "detected_at_utc": "2026-10-14T21:00:00+00:00",
        "corrected_at_utc": "2026-10-14T21:00:05+00:00",
    }
    entry.update(overrides)
    return entry


def _settled_grading() -> dict[str, Any]:
    return {
        "grading_status": "win",
        "outcome": True,
        "clv": 0.0123,
        "payout_flat": 0.9090909090909091,
        "realized_units": 1.1363636363636365,
        "graded_at": "2026-10-12T21:14:40.617525+00:00",
    }


def _two_row_ledger(ledger_dir: Path) -> list:
    """A settled seq-0 row and a pending seq-1 row, written to *ledger_dir*."""
    first = build_entry(
        GENESIS_HASH, 0, ENTRY_KIND_ROW, _row(), grading=_settled_grading()
    )
    second = build_entry(
        first.chain_hash,
        1,
        ENTRY_KIND_ROW,
        _row(game_id="2026_W06_DAL@PHI", model_value=4.75, line=3.5, slipped_line=4.0),
    )
    entries = [first, second]
    write_entries(ledger_dir, entries)
    return entries


def _run_cli(ledger_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_ledger",
            "--ledger-dir",
            str(ledger_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_genesis_constant_is_the_published_seed_hash() -> None:
    assert hashlib.sha256(GENESIS_SEED).hexdigest() == GENESIS_HASH
    source = CANONICAL_SOURCE.read_text(encoding="ascii")
    match = re.search(r'^GENESIS_HASH = "([0-9a-f]{64})"$', source, flags=re.MULTILINE)
    assert match is not None, (
        "GENESIS_HASH must be a 64-char lowercase hex LITERAL in source"
    )
    assert match.group(1) == GENESIS_HASH


def test_immutable_columns_v1_is_the_frozen_34_name_list() -> None:
    assert len(IMMUTABLE_COLUMNS_V1) == 34
    assert len(set(IMMUTABLE_COLUMNS_V1)) == 34
    assert IMMUTABLE_COLUMNS_V1[22] == "decided_at_utc"
    assert IMMUTABLE_COLUMNS_V1[23] == "arm"
    assert IMMUTABLE_COLUMNS_V1[-1] == "regime_label"
    assert tuple(IMMUTABLE_COLUMN_TYPES_V1) == IMMUTABLE_COLUMNS_V1
    assert set(IMMUTABLE_COLUMN_TYPES_V1.values()) <= {
        "VARCHAR",
        "INTEGER",
        "DOUBLE",
        "BOOLEAN",
    }


def test_correction_columns_name_the_prior_in_force_state() -> None:
    assert len(CORRECTION_COLUMNS_V1) == 22
    assert CORRECTION_COLUMNS_V1[7:11] == (
        "prior_grading_status",
        "prior_payout_flat",
        "prior_realized_units",
        "prior_correction_seq",
    )
    # Between the row's own grade and the corrected values: original_* < prior_* < corrected_*.
    assert CORRECTION_COLUMNS_V1.index("original_payout_flat") < 7
    assert CORRECTION_COLUMNS_V1.index("corrected_grading_status") == 11
    assert CORRECTION_COLUMN_TYPES_V1["prior_correction_seq"] == "INTEGER"
    assert tuple(CORRECTION_COLUMN_TYPES_V1) == CORRECTION_COLUMNS_V1

    row = build_entry(
        GENESIS_HASH, 0, ENTRY_KIND_ROW, _row(), grading=_settled_grading()
    )
    correction = build_entry(row.chain_hash, 1, ENTRY_KIND_CORRECTION, _correction())
    assert correction.grading is None
    assert correction.chain_hash == chain_hash(
        row.chain_hash, canonical_entry_bytes(ENTRY_KIND_CORRECTION, _correction())
    )
    verdict = verify_chain([row, correction])
    assert verdict.ok
    assert verdict.head_hash == correction.chain_hash


# ---------------------------------------------------------------------------
# Chain and store
# ---------------------------------------------------------------------------


def test_empty_ledger_verifies(tmp_path: Path) -> None:
    entries = read_entries(tmp_path)
    assert entries == []
    verdict = verify_chain(entries)
    assert verdict.ok
    assert (verdict.head_hash, verdict.entry_count) == (GENESIS_HASH, 0)
    assert ledger_head(entries) == (GENESIS_HASH, 0)


def test_one_row_ledger_verifies_and_chains_from_genesis(tmp_path: Path) -> None:
    entry = build_entry(GENESIS_HASH, 0, ENTRY_KIND_ROW, _row())
    assert entry.chain_hash == chain_hash(
        GENESIS_HASH, canonical_entry_bytes(ENTRY_KIND_ROW, _row())
    )
    write_entries(tmp_path, [entry])

    read_back = read_entries(tmp_path)
    assert read_back == [entry]
    verdict = verify_chain(read_back)
    assert verdict.ok
    assert ledger_head(read_back) == (entry.chain_hash, 1)


def test_noop_reencode_changes_no_hash(tmp_path: Path) -> None:
    entries = _two_row_ledger(tmp_path)
    path = ledger_path(tmp_path)
    first_bytes = path.read_bytes()
    assert b"\r" not in first_bytes, (
        "the store is written as bytes; Windows must never add CR"
    )
    assert first_bytes.count(b"\n") == len(entries)

    write_entries(tmp_path, read_entries(tmp_path))
    second_bytes = path.read_bytes()
    reread = read_entries(tmp_path)

    assert second_bytes == first_bytes
    assert [e.chain_hash for e in reread] == [e.chain_hash for e in entries]
    assert verify_chain(reread).ok


def test_reader_keeps_file_order(tmp_path: Path) -> None:
    game_ids = ["2026_W06_SF@SEA", "2026_W06_ARI@ATL", "2026_W06_NYJ@MIA"]
    entries = []
    previous = GENESIS_HASH
    for seq, game_id in enumerate(game_ids):
        entry = build_entry(previous, seq, ENTRY_KIND_ROW, _row(game_id=game_id))
        entries.append(entry)
        previous = entry.chain_hash
    write_entries(tmp_path, entries)

    read_back = read_entries(tmp_path)
    assert [e.immutable["game_id"] for e in read_back] == game_ids
    assert [e.seq for e in read_back] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", ["rejection_reason", "stake_units", "week"])
def test_one_null_form(column: str) -> None:
    null_forms = [None, float("nan"), pd.NA, pd.NaT, np.nan]
    encodings = {
        canonical_entry_bytes(ENTRY_KIND_ROW, _row(**{column: value}))
        for value in null_forms
    }
    assert len(encodings) == 1
    assert f'["{column}",null]'.encode("ascii") in encodings.pop()


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), np.inf])
def test_infinite_double_is_refused_by_name(value: float) -> None:
    with pytest.raises(CanonicalValueError, match="model_value"):
        canonical_entry_bytes(ENTRY_KIND_ROW, _row(model_value=value))


def test_float_round_trip_and_negative_zero(tmp_path: Path) -> None:
    values = [0.1 + 0.2, 1e-300, -0.0, 0.0]
    entries = []
    previous = GENESIS_HASH
    for seq, value in enumerate(values):
        entry = build_entry(
            previous, seq, ENTRY_KIND_ROW, _row(game_id=f"g{seq}", model_value=value)
        )
        entries.append(entry)
        previous = entry.chain_hash
    write_entries(tmp_path, entries)

    read_back = [e.immutable["model_value"] for e in read_entries(tmp_path)]
    for written, read in zip(values, read_back, strict=True):
        assert struct.pack("<d", written) == struct.pack("<d", read)
    assert math.copysign(1.0, read_back[2]) == -1.0

    negative = canonical_entry_bytes(ENTRY_KIND_ROW, _row(model_value=-0.0))
    positive = canonical_entry_bytes(ENTRY_KIND_ROW, _row(model_value=0.0))
    assert negative != positive
    assert negative == canonical_entry_bytes(ENTRY_KIND_ROW, _row(model_value=-0.0))


def test_numpy_scalars_encode_like_builtins() -> None:
    builtin = canonical_entry_bytes(ENTRY_KIND_ROW, _row())
    numpy_row = _row(
        season=np.int64(2026), week=np.int32(6), model_value=np.float64(-3.2)
    )
    assert canonical_entry_bytes(ENTRY_KIND_ROW, numpy_row) == builtin


def test_integer_column_refuses_non_integral_float() -> None:
    assert canonical_entry_bytes(
        ENTRY_KIND_ROW, _row(week=6.0)
    ) == canonical_entry_bytes(ENTRY_KIND_ROW, _row())
    with pytest.raises(CanonicalValueError, match="week"):
        canonical_entry_bytes(ENTRY_KIND_ROW, _row(week=6.5))
    with pytest.raises(CanonicalValueError, match="season"):
        canonical_entry_bytes(ENTRY_KIND_ROW, _row(season=True))


def test_boolean_column_refuses_non_bool() -> None:
    assert canonical_entry_bytes(
        ENTRY_KIND_CORRECTION, _correction(corrected_outcome=np.bool_(False))
    ) == canonical_entry_bytes(ENTRY_KIND_CORRECTION, _correction())
    for bad in (0, 1.0, "false"):
        with pytest.raises(CanonicalValueError, match="corrected_outcome"):
            canonical_entry_bytes(
                ENTRY_KIND_CORRECTION, _correction(corrected_outcome=bad)
            )


def test_varchar_column_is_stored_exactly_and_refuses_non_strings() -> None:
    stamp = "2026-10-10T21:18:10.874141+00:00"
    assert stamp.encode("ascii") in canonical_entry_bytes(
        ENTRY_KIND_ROW, _row(decided_at_utc=stamp)
    )
    with pytest.raises(CanonicalValueError, match="decided_at_utc"):
        canonical_entry_bytes(ENTRY_KIND_ROW, _row(decided_at_utc=pd.Timestamp(stamp)))


def test_payload_is_exactly_the_v1_column_list() -> None:
    missing = _row()
    del missing["arm"]
    with pytest.raises(CanonicalValueError, match="arm"):
        canonical_entry_bytes(ENTRY_KIND_ROW, missing)
    with pytest.raises(CanonicalValueError, match="surprise"):
        canonical_entry_bytes(ENTRY_KIND_ROW, _row(surprise=1))


# ---------------------------------------------------------------------------
# The verify CLI, as a real subprocess
# ---------------------------------------------------------------------------


def test_verify_cli_clean_exit_zero(tmp_path: Path) -> None:
    entries = _two_row_ledger(tmp_path)
    completed = _run_cli(tmp_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CHAIN_OK= True" in completed.stdout
    assert "LEDGER_ENTRIES= 2" in completed.stdout
    assert f"HEAD_HASH= {entries[-1].chain_hash}" in completed.stdout


def test_verify_cli_names_first_broken_row(tmp_path: Path) -> None:
    _two_row_ledger(tmp_path)
    path = ledger_path(tmp_path)
    original = path.read_bytes()
    # One character inside an immutable value of the SETTLED seq-0 row: -3.2 -> -3.3.
    assert original.count(b'"model_value":-3.2,') == 1
    path.write_bytes(original.replace(b'"model_value":-3.2,', b'"model_value":-3.3,'))

    completed = _run_cli(tmp_path)

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "CHAIN_OK= False" in completed.stdout
    assert "FIRST_BROKEN_SEQ= 0" in completed.stdout
    assert "FIRST_BROKEN_KEY= 2026_W06_KC@BUF|2026|6|ats|live" in completed.stdout


def test_malformed_line_is_refused_by_name(tmp_path: Path) -> None:
    _two_row_ledger(tmp_path)
    path = ledger_path(tmp_path)
    lines = path.read_bytes().split(b"\n")
    # Line 2 cut mid-object, as a crash during a non-atomic write would leave it.
    lines[1] = lines[1][: len(lines[1]) // 2]
    path.write_bytes(b"\n".join(lines))

    with pytest.raises(LedgerFormatError, match="line 2"):
        read_entries(tmp_path)

    completed = _run_cli(tmp_path)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "LEDGER_UNREADABLE=" in completed.stdout
    assert "line 2" in completed.stdout
