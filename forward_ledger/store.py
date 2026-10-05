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

THE ONE WRITE PATH (D-13, Plan 34-04)
-------------------------------------
:func:`commit_changes` is the guarded writer every caller goes through (``append_rows`` and
``apply_updates`` are thin wrappers). In order, under one lock held for the whole operation:

  1. take the single-writer OS lock on ``ledger/.writer.lock`` FAIL-FAST
     (:class:`LedgerWriterBusyError`) -- two writers are refused, never interleaved;
  2. read the store and recompute every chain hash -- never build on a broken chain
     (:class:`LedgerChainBrokenError`);
  3. validate every incoming row: exact v1 columns, season 2026, ``arm`` in ``ARMS``, forward
     provenance, every stamp non-null, recipe and fill id resolvable, a known verdict scope, a
     regime label only on the bootstrap weeks, and the existing decided-before-freeze assertion;
  4. FIRST PICK STANDS (keyed on ``LEDGER_ROW_KEY``): an identical repeat is skipped, a different
     one raises :class:`FirstPickStandsError` before anything is written;
  5. apply the mutable-half updates, each confined to its own column set and one-way;
  6. PROVE that every prior entry's canonical bytes and chain hash are unchanged and that the new
     tail chains from the old head -- append-only is checked at write time, not only by verify;
  7. write nothing when nothing changed; otherwise re-check ``publish_by`` against the clock, ask
     the write sink, and replace the file atomically (bounded retry for a reader holding it open).

The migration of the old writer's NULL-stamp rows is NOT this path: it is a separate entry point
(Plan 34-12), because only it may write a row without stamps.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import (
    GRADING_STATUS_PENDING,
    GRADING_STATUSES,
    PROVENANCE_FORWARD,
    assert_grading_transition,
)
from backtest.weekly_bet_list import (
    AlreadyGradedError,
    PublishDeadlinePassedError,
    _replace_atomically,
    assert_decided_at_before_freeze,
)
from data.write_sink import current_sink
from forward_ledger.canonical import (
    CANON_VERSION,
    ENTRY_KIND_CORRECTION,
    ENTRY_KIND_ROW,
    GENESIS_HASH,
    IMMUTABLE_COLUMN_TYPES_V1,
    IMMUTABLE_COLUMNS_V1,
    CanonicalValueError,
    canonical_entry_bytes,
    canonical_values,
    chain_hash,
    coerce_canonical_value,
)
from forward_ledger.declarations import (
    BOOTSTRAP_REGIME_WEEKS,
    resolve_fill_convention,
    resolve_recipe,
)
from forward_ledger.schema import (
    ARMS,
    BET_LIST_CLOSING_COLUMNS,
    BET_LIST_COLUMNS,
    BET_LIST_FILL_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    LEDGER_ROW_KEY,
    REGIME_LABEL_BOOTSTRAP,
    VERDICT_SCOPES,
)
from utils.file_lock import FileLockHeldError, exclusive_file_lock

__all__ = [
    "FIRST_PICK_IDENTITY_COLUMNS",
    "LEDGER_DIR",
    "LEDGER_FILENAME",
    "LEDGER_SEASON",
    "LEDGER_STAMP_COLUMNS",
    "LEDGER_WRITER_LOCK_NAME",
    "RUN_OBSERVATION_COLUMNS",
    "ChainVerdict",
    "Classification",
    "ColumnSetViolationError",
    "CommitResult",
    "FirstPickStandsError",
    "InvalidArmError",
    "LedgerChainBrokenError",
    "LedgerEntry",
    "LedgerEntryRefusedError",
    "LedgerFormatError",
    "LedgerWriterBusyError",
    "MissingStampError",
    "append_rows",
    "apply_updates",
    "build_entry",
    "classify_incoming",
    "commit_changes",
    "entries_to_frame",
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

# The single-writer lock file inside the ledger directory (excluded from the backup repo).
LEDGER_WRITER_LOCK_NAME: str = ".writer.lock"

# The stamp and reproduction columns that must be NON-NULL on every row this path writes (LDGR-03,
# LDGR-09, LDGR-11). ``arm`` and ``verdict_scope`` are checked against their own vocabularies, and
# ``regime_label`` is legitimately NULL outside weeks 2-4.
LEDGER_STAMP_COLUMNS: tuple[str, ...] = (
    "model_artifact_id",
    "blend_id",
    "recipe_id",
    "fill_convention_id",
    "upstream_capture_key",
    "gold_generation_key",
    "odds_snapshot_digest",
    "decision_snapshot_digest",
)

# WHEN and FROM WHICH BYTES a run observed a decision, not WHAT it decided. A repeat run always has
# a new ``decided_at_utc`` and, because gold is rebuilt nightly, usually a new gold key; comparing
# them would refuse every honest repeat. First pick stands compares every OTHER immutable column,
# exactly (scoring is bit-deterministic on this machine, 34-RESEARCH H).
RUN_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "decided_at_utc",
    "snapshot_ts",
    "upstream_capture_key",
    "gold_generation_key",
    "odds_snapshot_digest",
    "decision_snapshot_digest",
)
FIRST_PICK_IDENTITY_COLUMNS: tuple[str, ...] = tuple(
    name for name in IMMUTABLE_COLUMNS_V1 if name not in RUN_OBSERVATION_COLUMNS
)

# The DECLARED types of the three mutable halves, copied from the ``bet_list`` DDL in ``api.cache``.
# One map serves both the value normalization on write and the frame dtypes on read.
_MUTABLE_COLUMN_TYPES: dict[str, str] = {
    "grading_status": "VARCHAR",
    "outcome": "BOOLEAN",
    "clv": "DOUBLE",
    "payout_flat": "DOUBLE",
    "realized_units": "DOUBLE",
    "graded_at": "TIMESTAMP",
    "fill_sportsbook": "VARCHAR",
    "fill_line": "DOUBLE",
    "fill_odds": "DOUBLE",
    "fill_stake_dollars": "DOUBLE",
    "fill_at_utc": "VARCHAR",
    "closing_line": "DOUBLE",
    "closing_odds": "DOUBLE",
    "closing_sportsbook": "VARCHAR",
    "closing_captured_at": "VARCHAR",
    "forward_clv": "DOUBLE",
    "closing_null_reason": "VARCHAR",
}
if list(_MUTABLE_COLUMN_TYPES) != [
    *BET_LIST_GRADING_COLUMNS,
    *BET_LIST_FILL_COLUMNS,
    *BET_LIST_CLOSING_COLUMNS,
]:  # pragma: no cover
    msg = "the ledger's mutable column types no longer match api.cache's three mutable classes"
    raise RuntimeError(msg)

_TERMINAL_GRADING_STATUSES: tuple[str, ...] = tuple(
    status for status in GRADING_STATUSES if status != GRADING_STATUS_PENDING
)

# A reader (the cache build) holding the store open makes ``os.replace`` fail on Windows; the lock
# excludes writers, not readers, so the replace is retried briefly (34-RESEARCH Pitfall 6).
_REPLACE_ATTEMPTS = 3
_REPLACE_RETRY_SECONDS = 0.5

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
    _replace_with_retry(path, _serialize(entries))
    return path


def _serialize(entries: Sequence[LedgerEntry]) -> bytes:
    return b"".join(_entry_line(entry) for entry in entries)


def _write_payload(tmp: Path, payload: bytes) -> None:
    """Write the whole store's bytes to the sibling temp file (bytes, so no CR is ever added)."""
    tmp.write_bytes(payload)


def _replace_with_retry(path: Path, payload: bytes) -> None:
    """Replace *path* atomically with *payload*, retrying only a reader-held ``PermissionError``.

    Any other failure -- including an interrupt -- propagates at once, and ``_replace_atomically``
    has already removed its temp file, so the store keeps its previous bytes.
    """
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            _replace_atomically(lambda tmp: _write_payload(tmp, payload), path)
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_RETRY_SECONDS)
        else:
            return


def _entry_key(entry: LedgerEntry) -> tuple[Any, ...]:
    """The entry's ``LEDGER_ROW_KEY`` values, so a broken entry is reported by its key."""
    return tuple(entry.immutable.get(name) for name in LEDGER_ROW_KEY)


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


# ---------------------------------------------------------------------------
# The guarded write path (D-13): named refusals
# ---------------------------------------------------------------------------
# Every refusal inherits bare ``Exception`` (the ``LedgerFormatError`` rule): broad ``ValueError`` /
# ``RuntimeError`` handlers elsewhere degrade to empty results, and a refused write must never read
# as "nothing to record".


class LedgerWriterBusyError(Exception):
    """Another writer holds the ledger's single-writer lock; this write was refused, not queued."""


class LedgerChainBrokenError(Exception):
    """The stored chain does not verify, or the write-time append-only proof failed."""


class FirstPickStandsError(Exception):
    """A repeat run reached a DIFFERENT decision for a stored key (D-13, first pick stands).

    Attributes:
        key: The ``LEDGER_ROW_KEY`` values of the first conflicting row.
        fields: The identity columns whose values differ from the stored row.
        conflicts: Every conflicting ``(key, fields)`` pair in the refused commit.
    """

    def __init__(
        self,
        key: tuple[Any, ...],
        fields: tuple[str, ...],
        conflicts: tuple[tuple[tuple[Any, ...], tuple[str, ...]], ...],
    ) -> None:
        self.key = key
        self.fields = fields
        self.conflicts = conflicts
        super().__init__(
            f"first pick stands: ledger row {key} already holds a different decision "
            f"(differing field(s) {list(fields)}; {len(conflicts)} conflicting row(s) in this "
            "commit). The stored row is unchanged and nothing was written."
        )


class InvalidArmError(Exception):
    """A row or correction whose ``arm`` is NULL or outside ``ARMS`` (LDGR-02)."""


class MissingStampError(Exception):
    """A row this path writes carries a NULL stamp (LDGR-03, LDGR-09, LDGR-11)."""


class ColumnSetViolationError(Exception):
    """An update names a column outside its own mutability class (LDGR-04)."""


class LedgerEntryRefusedError(Exception):
    """A row or correction breaks a ledger rule not covered by a more specific refusal."""


@dataclass(frozen=True)
class Classification:
    """Incoming rows partitioned against the store by ``LEDGER_ROW_KEY`` (nothing is written).

    ``new`` and ``identical`` hold canonical values; ``conflicting`` holds ``(key, differing
    identity columns)`` pairs.
    """

    new: tuple[dict[str, Any], ...]
    identical: tuple[dict[str, Any], ...]
    conflicting: tuple[tuple[tuple[Any, ...], tuple[str, ...]], ...]


@dataclass(frozen=True)
class CommitResult:
    """What one :func:`commit_changes` call did. ``wrote`` is False when nothing was persisted."""

    appended_rows: int
    appended_corrections: int
    grading_changed: int
    wrote: bool
    head_hash: str
    entry_count: int


# ---------------------------------------------------------------------------
# Row validation and first pick stands
# ---------------------------------------------------------------------------


def _row_key(values: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(values[name] for name in LEDGER_ROW_KEY)


def _validate_new_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical values of *row*, or a named refusal of the first rule it breaks."""
    values = canonical_values(ENTRY_KIND_ROW, row)
    key = _row_key(values)
    if values["arm"] not in ARMS:
        msg = f"ledger row {key}: arm {values['arm']!r} is outside the closed vocabulary {ARMS}"
        raise InvalidArmError(msg)
    if values["season"] != LEDGER_SEASON:
        msg = f"ledger row {key}: season {values['season']} is not the ledger's {LEDGER_SEASON}"
        raise LedgerEntryRefusedError(msg)
    if values["provenance"] != PROVENANCE_FORWARD:
        msg = (
            f"ledger row {key}: provenance {values['provenance']!r} is not "
            f"{PROVENANCE_FORWARD!r}; replay rows stay in outputs/bet_list"
        )
        raise LedgerEntryRefusedError(msg)
    missing = [name for name in LEDGER_STAMP_COLUMNS if values[name] is None]
    if missing:
        msg = (
            f"ledger row {key} carries no stamp for {missing}; a row this path writes is "
            "attributed at the moment it is decided, and only the migration may write NULL stamps"
        )
        raise MissingStampError(msg)
    resolve_recipe(values["recipe_id"])
    resolve_fill_convention(values["fill_convention_id"])
    if values["verdict_scope"] not in VERDICT_SCOPES:
        msg = (
            f"ledger row {key}: verdict_scope {values['verdict_scope']!r} is outside "
            f"{VERDICT_SCOPES}"
        )
        raise LedgerEntryRefusedError(msg)
    regime = values["regime_label"]
    if regime is not None and (
        regime != REGIME_LABEL_BOOTSTRAP or values["week"] not in BOOTSTRAP_REGIME_WEEKS
    ):
        msg = (
            f"ledger row {key}: regime_label {regime!r} is allowed only as "
            f"{REGIME_LABEL_BOOTSTRAP!r} on weeks {BOOTSTRAP_REGIME_WEEKS}"
        )
        raise LedgerEntryRefusedError(msg)
    assert_decided_at_before_freeze(values)
    return values


def _exact(value: Any) -> str:
    """A canonical value's exact text: ``-0.0`` and ``0.0`` differ, as they do in the chain."""
    return json.dumps(value, allow_nan=False)


def _first_pick_differences(
    stored: Mapping[str, Any], incoming: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(
        name
        for name in FIRST_PICK_IDENTITY_COLUMNS
        if _exact(stored[name]) != _exact(incoming[name])
    )


def classify_incoming(
    entries: Sequence[LedgerEntry], rows: Sequence[Mapping[str, Any]]
) -> Classification:
    """Partition *rows* into new, identical and conflicting against *entries*. Writes nothing.

    Keyed on ``LEDGER_ROW_KEY`` (34-RESEARCH Pitfall 7), so a live and a shadow row of the same
    game, week and target are two keys. A later row in *rows* is compared against an earlier one
    with the same key exactly as against a stored row.
    """
    known: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        if entry.kind == ENTRY_KIND_ROW:
            values = canonical_values(ENTRY_KIND_ROW, entry.immutable)
            known[_row_key(values)] = values

    new: list[dict[str, Any]] = []
    identical: list[dict[str, Any]] = []
    conflicting: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
    for row in rows:
        values = canonical_values(ENTRY_KIND_ROW, row)
        key = _row_key(values)
        if key not in known:
            new.append(values)
            known[key] = values
            continue
        differences = _first_pick_differences(known[key], values)
        if differences:
            conflicting.append((key, differences))
        else:
            identical.append(values)
    return Classification(tuple(new), tuple(identical), tuple(conflicting))


# ---------------------------------------------------------------------------
# The mutable halves
# ---------------------------------------------------------------------------


def _mutable_value(column: str, value: Any) -> Any:
    """*value* in the JSON form the store keeps for mutable *column*.

    Every null form becomes ``None``; numbers and booleans become builtins by the declared type
    (infinity and wrong types refused by name); ``graded_at`` keeps an offset-carrying string as
    given and renders a tz-aware datetime in UTC. A naive datetime is refused.
    """
    declared = _MUTABLE_COLUMN_TYPES[column]
    if declared == "TIMESTAMP":
        if isinstance(value, datetime) and value is not pd.NaT:
            if value.tzinfo is None:
                msg = f"column {column!r} refuses the naive datetime {value!r}"
                raise CanonicalValueError(msg)
            return value.astimezone(UTC).isoformat()
        return coerce_canonical_value(column, "VARCHAR", value)
    return coerce_canonical_value(column, declared, value)


def _confine(update_kind: str, columns: Sequence[str], allowed: Sequence[str]) -> None:
    extra = sorted(set(columns) - set(allowed))
    if extra:
        msg = (
            f"a {update_kind} update may write only {list(allowed)}; it names {extra}, which "
            "belong to another write path"
        )
        raise ColumnSetViolationError(msg)


def _apply_grading(entry: LedgerEntry, update: Mapping[str, Any]) -> LedgerEntry:
    """*entry* with its grading half moved one way by *update*; the same object when unchanged.

    A pending row may take any grading values; the status transition is checked by
    ``api.cache.assert_grading_transition``. A settled row accepts only an identical
    re-application -- a later run cannot restate a result (``AlreadyGradedError``).
    """
    _confine("grading", list(update), BET_LIST_GRADING_COLUMNS)
    current = {
        name: (entry.grading or {}).get(name) for name in BET_LIST_GRADING_COLUMNS
    }
    current_status = current["grading_status"] or GRADING_STATUS_PENDING
    normalized = {name: _mutable_value(name, value) for name, value in update.items()}
    assert_grading_transition(
        current_status, normalized.get("grading_status", current_status)
    )
    merged = {**current, **normalized}
    if merged == current:
        return entry
    if current_status != GRADING_STATUS_PENDING:
        key = _row_key(entry.immutable)
        msg = (
            f"ledger row {key} is already settled as {current_status!r}; a later run may not "
            "restate its grading values"
        )
        raise AlreadyGradedError(msg)
    return dataclasses.replace(entry, grading=merged)


def _new_row_halves() -> dict[str, dict[str, Any]]:
    """A freshly appended row: pending, and every fill and closing column NULL (a paper row)."""
    return {
        "grading": {
            **dict.fromkeys(BET_LIST_GRADING_COLUMNS),
            "grading_status": GRADING_STATUS_PENDING,
        },
        "fill": dict.fromkeys(BET_LIST_FILL_COLUMNS),
        "closing": dict.fromkeys(BET_LIST_CLOSING_COLUMNS),
    }


def _validate_correction(
    correction: Mapping[str, Any],
    working: Sequence[LedgerEntry],
    row_index: Mapping[tuple[Any, ...], int],
) -> dict[str, Any]:
    """The canonical values of a correction entry whose row exists and is terminal-graded."""
    values = canonical_values(ENTRY_KIND_CORRECTION, correction)
    key = _row_key(values)
    if values["arm"] not in ARMS:
        msg = f"correction for {key}: arm {values['arm']!r} is outside {ARMS}"
        raise InvalidArmError(msg)
    position = row_index.get(key)
    if position is None:
        msg = f"correction names {key}, which is not a ledger row"
        raise LedgerEntryRefusedError(msg)
    status = (working[position].grading or {}).get("grading_status")
    if status not in _TERMINAL_GRADING_STATUSES:
        msg = (
            f"correction for {key} refused: the row's grading_status is {status!r}, not a "
            f"terminal status {_TERMINAL_GRADING_STATUSES}; only a settled result is corrected"
        )
        raise LedgerEntryRefusedError(msg)
    return values


# ---------------------------------------------------------------------------
# The write-time append-only proof
# ---------------------------------------------------------------------------


def _prove_append_only(old: Sequence[LedgerEntry], new: Sequence[LedgerEntry]) -> None:
    """Every prior entry's canonical bytes and chain hash are unchanged; the tail chains on.

    Raises:
        LedgerChainBrokenError: the proof failed, so nothing may be written.
    """
    if len(new) < len(old):
        msg = (
            f"append-only proof failed: {len(old)} stored entries, {len(new)} to write"
        )
        raise LedgerChainBrokenError(msg)
    for before, after in zip(old, new, strict=False):
        if (
            before.seq != after.seq
            or before.kind != after.kind
            or before.chain_hash != after.chain_hash
            or canonical_entry_bytes(before.kind, before.immutable)
            != canonical_entry_bytes(after.kind, after.immutable)
        ):
            msg = (
                f"append-only proof failed at seq {before.seq}: a stored entry's immutable "
                "bytes or chain hash would change"
            )
            raise LedgerChainBrokenError(msg)
    if len(new) > len(old):
        head, _ = ledger_head(old)
        first = new[len(old)]
        expected = chain_hash(head, canonical_entry_bytes(first.kind, first.immutable))
        if first.chain_hash != expected:
            msg = f"append-only proof failed: seq {first.seq} does not chain from the old head"
            raise LedgerChainBrokenError(msg)


# ---------------------------------------------------------------------------
# commit_changes: the ONE write path
# ---------------------------------------------------------------------------


def commit_changes(
    ledger_dir: Path | str,
    *,
    new_rows: Sequence[Mapping[str, Any]] = (),
    new_corrections: Sequence[Mapping[str, Any]] = (),
    grading_updates: Mapping[tuple[Any, ...], Mapping[str, Any]] | None = None,
    publish_by: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CommitResult:
    """Append rows and corrections and apply mutable-half updates, under every D-13 guard.

    Args:
        ledger_dir: The ledger directory (``LEDGER_DIR`` in production; ``tmp_path`` in tests).
        new_rows: Immutable halves (exactly the 34 v1 columns) of newly decided rows.
        new_corrections: Correction entries (exactly the 22 v1 correction columns).
        grading_updates: ``LEDGER_ROW_KEY`` values -> grading columns to set.
        publish_by: When given, nothing is written once ``clock()`` is past it.
        clock: The instant source for *publish_by*; ``datetime.now(UTC)`` by default.

    Returns:
        What changed, whether it was written, and the resulting head.

    Raises:
        LedgerWriterBusyError, LedgerChainBrokenError, FirstPickStandsError, InvalidArmError,
        MissingStampError, LedgerEntryRefusedError, ColumnSetViolationError,
        backtest.weekly_bet_list.PublishDeadlinePassedError, and the refusals of the checks this
        path reuses (``UnknownRecipeError``, ``UnknownFillConventionError``,
        ``DecidedAfterFreezeError``, ``CanonicalValueError``, ``AlreadyGradedError``, the
        grading transition's ``ValueError``). Every refusal happens before anything is written.

    A refused first pick is reported by :class:`FirstPickStandsError` carrying the key and fields;
    recording it in the run log is the caller's job.
    """
    directory = Path(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / LEDGER_WRITER_LOCK_NAME
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(exclusive_file_lock(lock_path))
        except FileLockHeldError as error:
            msg = (
                f"another writer holds the ledger at {directory.as_posix()}; this write was "
                "refused, never interleaved"
            )
            raise LedgerWriterBusyError(msg) from error
        return _commit_under_lock(
            directory,
            new_rows=new_rows,
            new_corrections=new_corrections,
            grading_updates=grading_updates or {},
            publish_by=publish_by,
            clock=clock,
        )


def _commit_under_lock(
    directory: Path,
    *,
    new_rows: Sequence[Mapping[str, Any]],
    new_corrections: Sequence[Mapping[str, Any]],
    grading_updates: Mapping[tuple[Any, ...], Mapping[str, Any]],
    publish_by: datetime | None,
    clock: Callable[[], datetime] | None,
) -> CommitResult:
    old = read_entries(directory)
    verdict = verify_chain(old)
    if not verdict.ok:
        msg = (
            f"the ledger's chain is broken at seq {verdict.first_broken_seq} (key "
            f"{verdict.first_broken_key}): {verdict.reason}. Nothing is written onto a broken "
            "chain."
        )
        raise LedgerChainBrokenError(msg)

    validated = [_validate_new_row(row) for row in new_rows]
    classification = classify_incoming(old, validated)
    if classification.conflicting:
        key, fields = classification.conflicting[0]
        raise FirstPickStandsError(key, fields, classification.conflicting)

    working = list(old)
    for values in classification.new:
        head, count = ledger_head(working)
        working.append(
            build_entry(head, count, ENTRY_KIND_ROW, values, **_new_row_halves())
        )

    row_index = {
        _row_key(entry.immutable): position
        for position, entry in enumerate(working)
        if entry.kind == ENTRY_KIND_ROW
    }
    grading_changed = 0
    for key, update in grading_updates.items():
        position = row_index.get(tuple(key))
        if position is None:
            msg = f"grading update names {tuple(key)}, which is not a ledger row"
            raise LedgerEntryRefusedError(msg)
        updated = _apply_grading(working[position], update)
        if updated is not working[position]:
            working[position] = updated
            grading_changed += 1

    appended_corrections = 0
    for correction in new_corrections:
        values = _validate_correction(correction, working, row_index)
        head, count = ledger_head(working)
        working.append(build_entry(head, count, ENTRY_KIND_CORRECTION, values))
        appended_corrections += 1

    _prove_append_only(old, working)
    head_hash, entry_count = ledger_head(working)
    result = CommitResult(
        appended_rows=len(classification.new),
        appended_corrections=appended_corrections,
        grading_changed=grading_changed,
        wrote=False,
        head_hash=head_hash,
        entry_count=entry_count,
    )
    if working == old:
        return result

    if publish_by is not None:
        from scripts.ingest_historical_odds import require_aware_snapshot_ts

        deadline = require_aware_snapshot_ts(publish_by)
        write_instant = clock() if clock is not None else datetime.now(tz=UTC)
        if write_instant > deadline:
            msg = (
                f"the ledger commit was ready at {write_instant.isoformat()}, after its publish "
                f"deadline {deadline.isoformat()}; nothing was written."
            )
            raise PublishDeadlinePassedError(msg)

    path = ledger_path(directory)
    if not current_sink().authorize(str(path), "ledger_commit", len(working)):
        return result
    _replace_with_retry(path, _serialize(working))
    return dataclasses.replace(result, wrote=True)


def append_rows(
    ledger_dir: Path | str, rows: Sequence[Mapping[str, Any]], **kwargs: Any
) -> CommitResult:
    """:func:`commit_changes` with *rows* as the new rows."""
    return commit_changes(ledger_dir, new_rows=rows, **kwargs)


def apply_updates(ledger_dir: Path | str, **kwargs: Any) -> CommitResult:
    """:func:`commit_changes` with no new rows: mutable-half updates and corrections only."""
    return commit_changes(ledger_dir, **kwargs)


# ---------------------------------------------------------------------------
# Reading the ledger as a bet-list frame
# ---------------------------------------------------------------------------

_FRAME_COLUMN_TYPES: dict[str, str] = {
    **IMMUTABLE_COLUMN_TYPES_V1,
    **_MUTABLE_COLUMN_TYPES,
}


def entries_to_frame(entries: Sequence[LedgerEntry]) -> pd.DataFrame:
    """The row entries as a bet-list frame: ``BET_LIST_COLUMNS`` order, file order, never sorted.

    Dtypes follow what ``backtest.weekly_bet_list.read_bet_list_artifact`` produces: INTEGER ->
    ``int64``, DOUBLE -> ``float64`` (NULL as NaN), ``graded_at`` tz-aware UTC, ``outcome`` and
    every VARCHAR ``object``. Correction entries are not rows and are skipped. No row -> the
    same empty frame the bet-list reader returns.
    """
    records = [
        {
            **entry.immutable,
            **(entry.grading or {}),
            **(entry.fill or {}),
            **(entry.closing or {}),
        }
        for entry in entries
        if entry.kind == ENTRY_KIND_ROW
    ]
    if not records:
        return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))
    frame = pd.DataFrame.from_records(records, columns=BET_LIST_COLUMNS)
    for column, declared in _FRAME_COLUMN_TYPES.items():
        if declared == "INTEGER":
            frame[column] = frame[column].astype("int64")
        elif declared == "DOUBLE":
            frame[column] = frame[column].astype("float64")
        elif declared == "TIMESTAMP":
            frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")
        else:
            frame[column] = frame[column].astype(object)
    return frame
