"""Move the old writer's 2026 forward rows into the ledger, as stored (Phase 34, D-14, LDGR-01/03).

WHAT MOVES, AND HOW
-------------------
Every 2026 ``forward`` row in ``outputs/bet_list/bet_list.parquet`` -- weeks 3-4 and any later
week the old writer wrote before go-live -- becomes one ledger ROW entry, in STORED order, chained
from the published ``GENESIS_HASH`` as the ledger's first entries:

  * the 23 immutable values the old writer stored, UNCHANGED -- ``decided_at_utc`` and
    ``validation_type`` included (34-RESEARCH Pitfall 12: the pre-verdict split comes from the new
    ``verdict_scope`` column, never from rewriting a stored value);
  * ``arm = 'live'``, ``verdict_scope = 'pre_verdict'``, and ``regime_label = 'bootstrap_regime'``
    on weeks 2-4 only (D40-08);
  * EVERY artifact, blend, recipe, fill and reproduction stamp NULL, forever. Those facts were not
    recorded when the rows were decided, and the pre-registration forbids adding them afterwards
    (SPEC must-not; LDGR-03). A payload that carries one is refused by name, and so is a stored
    row that already holds any Phase-34 value: it is not an old-writer row;
  * the grading half as stored (``graded_at`` rendered as an ISO UTC string, the form the store
    keeps), and the fill and closing halves NULL.

This is the ONE entry point allowed to write NULL-stamp rows. ``forward_ledger.store
.commit_changes`` refuses every such row by design, so the migration writes through the store's
lower-level ``build_entry`` / ``write_entries`` under the same single-writer lock -- and only into
an EMPTY ledger: a second migration is refused (:class:`AlreadyMigratedError`).

TWO PHASES, IN THIS ORDER (34-RESEARCH Pitfall 9)
------------------------------------------------
``--apply`` writes the ledger, re-reads it and verifies the chain and every entry; ONLY THEN does it
rewrite the outputs pair (``write_bet_list_pair``) with the 2025 ``backtest_replay`` rows alone, so
forward rows never live in two stores once the command finishes (Pitfall 2). If the rewrite is
interrupted, the ledger holds the rows and the outputs still do too; ``--finish`` completes the
rewrite, but only after proving the ledger's first entries equal the stored forward rows field by
field (:class:`MigrationMismatchError` names the first difference). The default is a dry run that
reads both stores and writes nothing.

The migration is EXECUTED once, by hand, in the go-live window (Plan 34-19). Tests run it against
fixture copies under ``tmp_path`` only.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import (
    GRADING_STATUSES,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
)
from backtest.weekly_bet_list import (
    DECIDED_AT_COLUMN,
    DEFAULT_BET_LIST_DIR,
    PHASE34_ADDED_COLUMNS,
    assert_decided_at_before_freeze,
    read_bet_list_artifact,
    write_bet_list_pair,
)
from forward_ledger.canonical import (
    ENTRY_KIND_ROW,
    GENESIS_HASH,
    _is_null,
    canonical_values,
)
from forward_ledger.declarations import BOOTSTRAP_REGIME_WEEKS
from forward_ledger.run_log import record_event
from forward_ledger.schema import (
    ARM_LIVE,
    BET_LIST_CLOSING_COLUMNS,
    BET_LIST_FILL_COLUMNS,
    BET_LIST_GRADING_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    LEDGER_ROW_KEY,
    REGIME_LABEL_BOOTSTRAP,
    VERDICT_SCOPE_PRE_VERDICT,
)
from forward_ledger.store import (
    LEDGER_DIR,
    LEDGER_SEASON,
    LEDGER_STAMP_COLUMNS,
    LEDGER_WRITER_LOCK_NAME,
    LedgerChainBrokenError,
    LedgerEntry,
    LedgerEntryRefusedError,
    LedgerWriterBusyError,
    _mutable_value,
    build_entry,
    ledger_head,
    read_entries,
    verify_chain,
    write_entries,
)
from utils.file_lock import FileLockHeldError, exclusive_file_lock

__all__ = [
    "MODE_APPLY",
    "MODE_DRY_RUN",
    "MODE_FINISH",
    "AlreadyMigratedError",
    "MigratedRow",
    "MigrationMismatchError",
    "MigrationReport",
    "MigrationStampError",
    "build_migrated_entries",
    "commit_migrated_rows",
    "migrate_forward_rows",
]

MODE_DRY_RUN = "dry_run"
MODE_APPLY = "apply"
MODE_FINISH = "finish"

# The immutable values the old writer stored: every immutable column up to ``decided_at_utc``.
STORED_IMMUTABLE_COLUMNS: tuple[str, ...] = tuple(
    BET_LIST_IMMUTABLE_COLUMNS[
        : BET_LIST_IMMUTABLE_COLUMNS.index(DECIDED_AT_COLUMN) + 1
    ]
)

LogCallable = Callable[..., Any]


# Every refusal inherits bare ``Exception`` (the ``forward_ledger.store`` rule): broad ``ValueError``
# handlers elsewhere degrade to empty results, and a refused migration must never read as "nothing
# to migrate".


class AlreadyMigratedError(Exception):
    """The ledger already holds entries; the migration runs once, into an empty ledger."""


class MigrationStampError(Exception):
    """A payload carries a stamp, label or half the old writer never recorded (D-14)."""


class MigrationMismatchError(Exception):
    """The two stores disagree: the ledger does not hold the stored forward rows, or a row fits
    neither store."""


@dataclass(frozen=True)
class MigratedRow:
    """One old-writer row as the ledger will hold it: its four halves."""

    immutable: dict[str, Any]
    grading: dict[str, Any]
    fill: dict[str, Any] = field(
        default_factory=lambda: dict.fromkeys(BET_LIST_FILL_COLUMNS)
    )
    closing: dict[str, Any] = field(
        default_factory=lambda: dict.fromkeys(BET_LIST_CLOSING_COLUMNS)
    )


@dataclass(frozen=True)
class MigrationReport:
    """What one invocation found and did.

    ``head_hash`` / ``entry_count`` are the ledger's head after the command -- for a dry run, the
    head ``--apply`` would write into an empty ledger.
    """

    mode: str
    migrate_rows: int
    weeks: tuple[int, ...]
    status_counts: dict[str, int]
    head_hash: str
    entry_count: int
    ledger_entries_before: int
    outputs_replay_rows: int
    ledger_written: bool
    outputs_rewritten: bool


# ---------------------------------------------------------------------------
# Reading the old writer's store
# ---------------------------------------------------------------------------


def _split_stores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(2026 forward rows, replay rows)`` in stored order; any other row is refused by name."""
    forward = (frame["provenance"] == PROVENANCE_FORWARD) & (
        frame["season"] == LEDGER_SEASON
    )
    replay = frame["provenance"] == PROVENANCE_BACKTEST_REPLAY
    stray = frame[~(forward | replay)]
    if not stray.empty:
        first = stray.iloc[0]
        msg = (
            f"the bet list holds {len(stray)} row(s) that are neither 2026 forward nor "
            f"backtest_replay (first: {first['game_id']!r} season {first['season']} provenance "
            f"{first['provenance']!r}); the migration places every row in exactly one store "
            "and refuses rather than guessing"
        )
        raise MigrationMismatchError(msg)
    return (
        frame.loc[forward].reset_index(drop=True),
        frame.loc[replay].reset_index(drop=True),
    )


def _regime_label(week: int) -> str | None:
    return REGIME_LABEL_BOOTSTRAP if week in BOOTSTRAP_REGIME_WEEKS else None


def _migrated_row(stored: Mapping[str, Any]) -> MigratedRow:
    """One stored forward row as a payload: stored values as-is, stamps NULL, pre-verdict."""
    key = (stored["game_id"], stored["season"], stored["week"], stored["target"])
    recorded = [
        name for name in PHASE34_ADDED_COLUMNS if not _is_null(stored.get(name))
    ]
    if recorded:
        msg = (
            f"stored row {key} already carries Phase-34 value(s) for {recorded}; the migration "
            "moves the old writer's rows exactly as stored and never carries or adds a stamp"
        )
        raise MigrationStampError(msg)

    immutable = {
        **{name: stored[name] for name in STORED_IMMUTABLE_COLUMNS},
        "arm": ARM_LIVE,
        **dict.fromkeys(LEDGER_STAMP_COLUMNS),
        "verdict_scope": VERDICT_SCOPE_PRE_VERDICT,
        "regime_label": _regime_label(int(stored["week"])),
    }
    grading = {
        name: _mutable_value(name, stored[name]) for name in BET_LIST_GRADING_COLUMNS
    }
    return MigratedRow(
        immutable=canonical_values(ENTRY_KIND_ROW, immutable), grading=grading
    )


def build_migrated_entries(frame: pd.DataFrame) -> list[MigratedRow]:
    """The payloads for every 2026 forward row of *frame*, in stored order.

    Args:
        frame: The bet list as ``backtest.weekly_bet_list.read_bet_list_artifact`` returns it.

    Raises:
        MigrationStampError: a stored forward row already carries a Phase-34 value.
        MigrationMismatchError: a row is neither 2026 forward nor ``backtest_replay``.
        CanonicalValueError: a stored value cannot be canonicalized.
    """
    forward, _ = _split_stores(frame)
    return [_migrated_row(record) for record in forward.to_dict("records")]


# ---------------------------------------------------------------------------
# Writing the ledger
# ---------------------------------------------------------------------------


def _check_payload(row: MigratedRow) -> dict[str, Any]:
    """The canonical immutable values of *row*, or a named refusal of what it must not carry."""
    values = canonical_values(ENTRY_KIND_ROW, row.immutable)
    key = tuple(values[name] for name in LEDGER_ROW_KEY)
    if values["season"] != LEDGER_SEASON or values["provenance"] != PROVENANCE_FORWARD:
        msg = (
            f"migrated row {key}: only {LEDGER_SEASON} {PROVENANCE_FORWARD!r} rows move into the "
            f"ledger (season {values['season']}, provenance {values['provenance']!r})"
        )
        raise LedgerEntryRefusedError(msg)

    stamped = [name for name in LEDGER_STAMP_COLUMNS if values[name] is not None]
    if stamped:
        msg = (
            f"migrated row {key} carries stamp(s) {stamped}; a pre-verdict row keeps NULL "
            "stamps forever and is never back-stamped (D-14, LDGR-03)"
        )
        raise MigrationStampError(msg)
    expected = {
        "arm": ARM_LIVE,
        "verdict_scope": VERDICT_SCOPE_PRE_VERDICT,
        "regime_label": _regime_label(values["week"]),
    }
    wrong = sorted(name for name, value in expected.items() if values[name] != value)
    if wrong:
        msg = f"migrated row {key} carries {wrong} other than {expected}"
        raise MigrationStampError(msg)

    filled = sorted(
        name
        for half in (row.fill, row.closing)
        for name, value in half.items()
        if value is not None
    )
    if (
        set(row.fill) != set(BET_LIST_FILL_COLUMNS)
        or set(row.closing) != set(BET_LIST_CLOSING_COLUMNS)
        or filled
    ):
        msg = f"migrated row {key}: the fill and closing halves must be all NULL ({filled})"
        raise MigrationStampError(msg)
    if (
        set(row.grading) != set(BET_LIST_GRADING_COLUMNS)
        or row.grading["grading_status"] not in GRADING_STATUSES
    ):
        msg = f"migrated row {key}: the grading half is not a stored grading half"
        raise MigrationStampError(msg)

    assert_decided_at_before_freeze(values)
    return values


def _chain(payloads: Sequence[MigratedRow]) -> list[LedgerEntry]:
    """The entries *payloads* become, chained from ``GENESIS_HASH`` in order."""
    entries: list[LedgerEntry] = []
    previous = GENESIS_HASH
    for seq, row in enumerate(payloads):
        entry = build_entry(
            previous,
            seq,
            ENTRY_KIND_ROW,
            _check_payload(row),
            grading=row.grading,
            fill=row.fill,
            closing=row.closing,
        )
        entries.append(entry)
        previous = entry.chain_hash
    return entries


def commit_migrated_rows(
    ledger_dir: Path | str, payloads: Sequence[MigratedRow]
) -> tuple[str, int]:
    """Write *payloads* as the ledger's first entries and return the verified ``(head, count)``.

    Under the single-writer lock: refuse a non-empty ledger, refuse any payload carrying a stamp
    or a label other than the pre-verdict ones, re-check every row's decided-before-freeze
    assertion, chain from ``GENESIS_HASH``, write, then re-read and verify the chain and every
    entry.

    Raises:
        LedgerWriterBusyError: another writer holds the ledger.
        AlreadyMigratedError: the ledger already holds entries.
        MigrationStampError: a payload carries a stamp, a wrong label, or a non-NULL fill/closing.
        LedgerEntryRefusedError: a payload is not a 2026 forward row.
        LedgerChainBrokenError: the written ledger does not read back as written.
        DecidedAfterFreezeError, MissingDecidedAtError, CanonicalValueError: as the checks raise.
    """
    directory = Path(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_file_lock(directory / LEDGER_WRITER_LOCK_NAME):
            existing = read_entries(directory)
            if existing:
                msg = (
                    f"the ledger at {directory.as_posix()} already holds {len(existing)} "
                    "entries; the migration runs once, into an empty ledger"
                )
                raise AlreadyMigratedError(msg)
            entries = _chain(payloads)
            if not entries:
                return ledger_head(entries)
            write_entries(directory, entries)
            written = read_entries(directory)
            if not verify_chain(written).ok or written != entries:
                msg = (
                    f"the migrated ledger at {directory.as_posix()} does not read back as "
                    "written; the outputs are left untouched"
                )
                raise LedgerChainBrokenError(msg)
            return ledger_head(written)
    except FileLockHeldError as error:
        msg = f"another writer holds the ledger at {directory.as_posix()}; the migration waits"
        raise LedgerWriterBusyError(msg) from error


# ---------------------------------------------------------------------------
# --finish: prove, then rewrite
# ---------------------------------------------------------------------------


def _prove_ledger_holds(
    entries: Sequence[LedgerEntry], payloads: Sequence[MigratedRow]
) -> None:
    """The ledger's first entries equal *payloads* field by field, or a named mismatch."""
    verdict = verify_chain(entries)
    if not verdict.ok:
        msg = (
            f"the ledger chain is broken at seq {verdict.first_broken_seq} "
            f"({verdict.reason}); --finish proves nothing against a broken chain"
        )
        raise LedgerChainBrokenError(msg)
    if not entries:
        msg = "the ledger is empty: nothing was migrated, so there is nothing to finish (use --apply)"
        raise MigrationMismatchError(msg)
    if len(entries) < len(payloads):
        msg = (
            f"the ledger holds {len(entries)} entries but the outputs hold {len(payloads)} "
            "forward rows; the ledger does not hold the stored rows"
        )
        raise MigrationMismatchError(msg)

    for entry, row in zip(entries, payloads, strict=False):
        expected = canonical_values(ENTRY_KIND_ROW, row.immutable)
        halves = (
            ("immutable", entry.immutable, expected),
            ("grading", entry.grading or {}, row.grading),
            ("fill", entry.fill or {}, row.fill),
            ("closing", entry.closing or {}, row.closing),
        )
        for half, held, stored in halves:
            differing = sorted(
                name
                for name in set(held) | set(stored)
                if held.get(name) != stored.get(name)
            )
            if entry.kind != ENTRY_KIND_ROW or differing:
                key = "|".join(str(expected[name]) for name in LEDGER_ROW_KEY)
                msg = (
                    f"ledger seq {entry.seq} does not equal the stored forward row {key}: "
                    f"{half} field(s) {differing} differ. The outputs are left untouched."
                )
                raise MigrationMismatchError(msg)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _status_counts(forward: pd.DataFrame) -> dict[str, int]:
    return dict(sorted(Counter(str(s) for s in forward["status"]).items()))


def migrate_forward_rows(
    output_dir: Path | str = DEFAULT_BET_LIST_DIR,
    ledger_dir: Path | str = LEDGER_DIR,
    *,
    apply: bool = False,
    finish: bool = False,
    log: LogCallable = record_event,
) -> MigrationReport:
    """Dry run (default), ``apply`` the migration, or ``finish`` an interrupted outputs rewrite.

    Args:
        output_dir: The old writer's store (``outputs/bet_list`` in production).
        ledger_dir: The ledger directory (``ledger`` in production).
        apply: Write the ledger, verify it, then rewrite the outputs with the replay rows only.
        finish: Prove the ledger holds the stored forward rows, then rewrite the outputs.
        log: ``(event, **fields)``; ``forward_ledger.run_log.record_event`` in production.

    Raises:
        ValueError: *apply* and *finish* are both set.
        AlreadyMigratedError, MigrationStampError, MigrationMismatchError, and the refusals of
        :func:`commit_migrated_rows` -- each before the outputs are touched.
    """
    if apply and finish:
        msg = "--apply and --finish are separate steps; pass one"
        raise ValueError(msg)
    mode = MODE_APPLY if apply else MODE_FINISH if finish else MODE_DRY_RUN

    frame = read_bet_list_artifact(Path(output_dir))
    forward, replay = _split_stores(frame)
    payloads = build_migrated_entries(frame)
    entries = read_entries(ledger_dir)
    weeks = tuple(sorted({int(week) for week in forward["week"]}))
    status_counts = _status_counts(forward)

    def report(
        head: tuple[str, int], *, ledger_written: bool, outputs_rewritten: bool
    ) -> MigrationReport:
        return MigrationReport(
            mode=mode,
            migrate_rows=len(payloads),
            weeks=weeks,
            status_counts=status_counts,
            head_hash=head[0],
            entry_count=head[1],
            ledger_entries_before=len(entries),
            outputs_replay_rows=len(replay),
            ledger_written=ledger_written,
            outputs_rewritten=outputs_rewritten,
        )

    if mode == MODE_DRY_RUN:
        return report(
            ledger_head(_chain(payloads)), ledger_written=False, outputs_rewritten=False
        )

    if mode == MODE_APPLY:
        if entries:
            msg = (
                f"the ledger already holds {len(entries)} entries; the migration runs once "
                "(use --finish to complete an interrupted outputs rewrite)"
            )
            raise AlreadyMigratedError(msg)
        head = commit_migrated_rows(ledger_dir, payloads)
        ledger_written = head[1] > 0
    else:
        _prove_ledger_holds(entries, payloads)
        head = ledger_head(entries)
        ledger_written = False

    # Only after the ledger write is verified (apply) or proven (finish): the replay rows alone.
    outputs_rewritten = bool(payloads)
    if outputs_rewritten:
        write_bet_list_pair(replay, Path(output_dir))
        log(
            "migration",
            mode=mode,
            rows=len(payloads),
            weeks=list(weeks),
            status_counts=status_counts,
            head_hash=head[0],
            entry_count=head[1],
            outputs_replay_rows=len(replay),
        )
    return report(
        head, ledger_written=ledger_written, outputs_rewritten=outputs_rewritten
    )
