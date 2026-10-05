"""The forward ledger's one-file store: canonical JSON lines, read in order, verified by chain.

WHY ONE FILE (34-RESEARCH F1)
-----------------------------
Everything that must change together -- each row's immutable half and its chain hash, its grading,
fill and closing halves, and every correction entry -- lives in ONE file, ``forward_2026.jsonl``,
replaced whole through ``backtest.weekly_bet_list._replace_atomically`` (the one reviewed atomic
writer, WR-05). A second file for the chain would reintroduce the two-file non-atomicity the
bet-list/tracker pair already admits to. An interrupted write therefore leaves either the whole old
file or the whole new one.

WHY JSON LINES AND NOT PARQUET (34-RESEARCH F2)
-----------------------------------------------
JSON's float form is ``float.__repr__``, which round-trips to the identical double; there is no
dtype inference (parquet via pandas upcasts an int column holding a NULL to float64 and re-types
timestamps by writer version); and the private backup's history becomes reviewable text diffs.
The chain hashes canonical CONTENT (:mod:`forward_ledger.canonical`), never these file bytes.

THE LINE FORMAT
---------------
One entry per line, keys in the order ``seq, kind, canon_v, immutable, chain_hash, grading, fill,
closing``, compact separators, ASCII only, ``allow_nan=False``, terminated by a bare LF. The file is
written as BYTES so Windows never inserts a CR. ``grading`` / ``fill`` / ``closing`` are objects on
a row entry and ``null`` on a correction entry.

ORDER IS APPEND ORDER
---------------------
The chain links each entry to the one before it IN THE FILE, so the reader returns entries in file
order and never sorts them; ``seq`` must equal the entry's position.

SCOPE OF THIS MODULE TODAY
--------------------------
The reader, the atomic writer, the entry builder and the chain verifier. The guarded write
protocol -- single-writer lock, first pick stands, the write-time proof that no prior entry's
canonical bytes moved -- is Plan 34-04's expansion of THIS module, not a second store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtest.weekly_bet_list import _replace_atomically
from forward_ledger.canonical import (
    CANON_VERSION,
    ENTRY_KIND_CORRECTION,
    ENTRY_KIND_ROW,
    GENESIS_HASH,
    CanonicalValueError,
    canonical_entry_bytes,
    canonical_values,
    chain_hash,
)

__all__ = [
    "LEDGER_DIR",
    "LEDGER_FILENAME",
    "LEDGER_SEASON",
    "ChainVerdict",
    "LedgerEntry",
    "LedgerFormatError",
    "build_entry",
    "ledger_head",
    "ledger_path",
    "read_entries",
    "verify_chain",
    "write_entries",
]

# The data directory, top-level and gitignored by the public repo (D-10): it fits no data-lake
# layer, because silver is latest-wins and gold is replaced by rebuilds -- both the opposite of
# append-only. Relative, like every other store root in this repository: it resolves against the
# working directory the scheduled task and the CLIs run from.
LEDGER_DIR: Path = Path("ledger")
LEDGER_FILENAME: str = "forward_2026.jsonl"
LEDGER_SEASON: int = 2026

# The five columns that identify a forward row (LDGR-02). Named here so a broken entry can be
# reported by its key.
_ENTRY_KEY_COLUMNS: tuple[str, ...] = ("game_id", "season", "week", "target", "arm")

# The exact keys of one stored line, in their written order.
_LINE_KEYS: tuple[str, ...] = (
    "seq",
    "kind",
    "canon_v",
    "immutable",
    "chain_hash",
    "grading",
    "fill",
    "closing",
)


class LedgerFormatError(Exception):
    """The stored ledger cannot be parsed. NOT a claim that it is empty.

    Inherits ``Exception`` and deliberately NOT ``ValueError`` / ``RuntimeError``, for the reason
    ``data.graded_weeks.GradedWeeksUnavailable`` states: broad handlers elsewhere in the tree
    degrade those to an empty result, and an unreadable ledger read as "nothing recorded yet"
    would publish a record that claims no bets were made.
    """


@dataclass(frozen=True)
class LedgerEntry:
    """One stored ledger line.

    ``immutable`` holds the canonical (coerced) v1 values in column order. ``grading``, ``fill``
    and ``closing`` are dicts on a row entry and ``None`` on a correction entry; none of the three
    is covered by the chain (LDGR-05 as locked; D-19 protects settled outcomes by re-grading).
    """

    seq: int
    kind: str
    canon_v: int
    immutable: dict[str, Any]
    chain_hash: str
    grading: dict[str, Any] | None
    fill: dict[str, Any] | None
    closing: dict[str, Any] | None


@dataclass(frozen=True)
class ChainVerdict:
    """The result of recomputing every chain hash from ``GENESIS_HASH``.

    ``head_hash`` is the hash through the last entry that verified -- the whole ledger's head when
    ``ok``, else the head of the intact prefix before ``first_broken_seq``.
    """

    ok: bool
    entry_count: int
    head_hash: str
    first_broken_seq: int | None
    first_broken_key: tuple[Any, ...] | None
    reason: str | None


def ledger_path(ledger_dir: Path | str = LEDGER_DIR) -> Path:
    """The one store file inside *ledger_dir*."""
    return Path(ledger_dir) / LEDGER_FILENAME


def build_entry(
    prev_hash: str,
    seq: int,
    kind: str,
    immutable: Mapping[str, Any],
    *,
    grading: Mapping[str, Any] | None = None,
    fill: Mapping[str, Any] | None = None,
    closing: Mapping[str, Any] | None = None,
) -> LedgerEntry:
    """Build the entry that chains *immutable* onto *prev_hash* at position *seq*.

    A row entry carries its three mutable halves as dicts (empty when not given); a correction
    entry carries none, and passing one is refused rather than dropped.

    Raises:
        CanonicalValueError: when *immutable* is not exactly the kind's v1 columns, a value cannot
            be canonicalized, or a correction is given a mutable half.
    """
    if kind == ENTRY_KIND_CORRECTION and any(
        half is not None for half in (grading, fill, closing)
    ):
        msg = "a correction entry carries no grading, fill or closing half"
        raise CanonicalValueError(msg)

    values = canonical_values(kind, immutable)
    is_row = kind == ENTRY_KIND_ROW
    return LedgerEntry(
        seq=seq,
        kind=kind,
        canon_v=CANON_VERSION,
        immutable=values,
        chain_hash=chain_hash(prev_hash, canonical_entry_bytes(kind, values)),
        grading=dict(grading or {}) if is_row else None,
        fill=dict(fill or {}) if is_row else None,
        closing=dict(closing or {}) if is_row else None,
    )


def _entry_line(entry: LedgerEntry) -> bytes:
    record = {
        "seq": entry.seq,
        "kind": entry.kind,
        "canon_v": entry.canon_v,
        "immutable": entry.immutable,
        "chain_hash": entry.chain_hash,
        "grading": entry.grading,
        "fill": entry.fill,
        "closing": entry.closing,
    }
    text = json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return text.encode("ascii") + b"\n"


def _optional_object(record: dict[str, Any], key: str, line_number: int) -> dict | None:
    value = record[key]
    if value is not None and not isinstance(value, dict):
        msg = f"ledger line {line_number}: {key!r} must be an object or null"
        raise LedgerFormatError(msg)
    return value


def _parse_line(raw: bytes, line_number: int) -> LedgerEntry:
    try:
        record = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"ledger line {line_number} is not one ASCII JSON object: {error}"
        raise LedgerFormatError(msg) from error

    if not isinstance(record, dict) or tuple(record) != _LINE_KEYS:
        msg = f"ledger line {line_number} must carry exactly the keys {_LINE_KEYS} in order"
        raise LedgerFormatError(msg)

    for key in ("seq", "canon_v"):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            msg = f"ledger line {line_number}: {key!r} must be an integer"
            raise LedgerFormatError(msg)
    for key in ("kind", "chain_hash"):
        if not isinstance(record[key], str):
            msg = f"ledger line {line_number}: {key!r} must be a string"
            raise LedgerFormatError(msg)
    if not isinstance(record["immutable"], dict):
        msg = f"ledger line {line_number}: 'immutable' must be an object"
        raise LedgerFormatError(msg)

    return LedgerEntry(
        seq=record["seq"],
        kind=record["kind"],
        canon_v=record["canon_v"],
        immutable=record["immutable"],
        chain_hash=record["chain_hash"],
        grading=_optional_object(record, "grading", line_number),
        fill=_optional_object(record, "fill", line_number),
        closing=_optional_object(record, "closing", line_number),
    )


def read_entries(ledger_dir: Path | str = LEDGER_DIR) -> list[LedgerEntry]:
    """Every stored entry, in FILE order -- never sorted. An absent store is ``[]``.

    The file is read in one call and closed before parsing, so a concurrent ``os.replace`` by the
    writer is never blocked by an open handle held here (34-RESEARCH Pitfall 6).

    Raises:
        LedgerFormatError: the store exists but cannot be read, or a line is malformed (named by
            its 1-based line number). Absent is not corrupt; unreadable is never read as empty.
    """
    path = ledger_path(ledger_dir)
    if not path.exists():
        return []
    try:
        data = path.read_bytes()
    except OSError as error:
        msg = f"cannot read the ledger store {path.as_posix()}: {error}"
        raise LedgerFormatError(msg) from error
    if not data:
        return []

    lines = data.split(b"\n")
    if lines[-1] != b"":
        msg = f"ledger line {len(lines)} is not LF-terminated; the store was not written whole"
        raise LedgerFormatError(msg)
    return [_parse_line(raw, number) for number, raw in enumerate(lines[:-1], start=1)]


def write_entries(ledger_dir: Path | str, entries: Sequence[LedgerEntry]) -> Path:
    """Replace the store with *entries*, atomically, and return its path.

    Raises:
        ValueError: a mutable half carries a value JSON cannot encode exactly (NaN, infinity).
    """
    directory = Path(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = ledger_path(directory)
    payload = b"".join(_entry_line(entry) for entry in entries)
    _replace_atomically(lambda tmp: tmp.write_bytes(payload), path)
    return path


def _entry_key(entry: LedgerEntry) -> tuple[Any, ...]:
    return tuple(entry.immutable.get(name) for name in _ENTRY_KEY_COLUMNS)


def verify_chain(entries: Sequence[LedgerEntry]) -> ChainVerdict:
    """Recompute every chain hash from ``GENESIS_HASH``; stop at the first entry that fails.

    An entry fails when its ``seq`` is not its position, its ``canon_v`` is not 1 (an unknown
    version is refused by name, never mis-hashed), its payload cannot be canonicalized, or its
    stored hash differs from the recomputed one.
    """
    previous = GENESIS_HASH
    for position, entry in enumerate(entries):
        reason: str | None = None
        if entry.seq != position:
            reason = f"stored seq {entry.seq} at position {position}"
        elif entry.canon_v != CANON_VERSION:
            reason = f"canon_v {entry.canon_v} is not {CANON_VERSION}"
        else:
            try:
                expected = chain_hash(
                    previous, canonical_entry_bytes(entry.kind, entry.immutable)
                )
            except CanonicalValueError as error:
                reason = f"payload cannot be canonicalized: {error}"
            else:
                if expected != entry.chain_hash:
                    reason = "stored chain_hash differs from the recomputed hash"
        if reason is not None:
            return ChainVerdict(
                ok=False,
                entry_count=len(entries),
                head_hash=previous,
                first_broken_seq=position,
                first_broken_key=_entry_key(entry),
                reason=reason,
            )
        previous = entry.chain_hash

    return ChainVerdict(
        ok=True,
        entry_count=len(entries),
        head_hash=previous,
        first_broken_seq=None,
        first_broken_key=None,
        reason=None,
    )


def ledger_head(entries: Sequence[LedgerEntry]) -> tuple[str, int]:
    """``(head hash, entry count)``; ``(GENESIS_HASH, 0)`` for an empty ledger."""
    if not entries:
        return GENESIS_HASH, 0
    return entries[-1].chain_hash, len(entries)
