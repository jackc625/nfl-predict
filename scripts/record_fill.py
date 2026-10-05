#!/usr/bin/env python3
"""Record a REAL bet's fill on one ledger row (Phase 34, D-17, LDGR-04). Owner-run, never a page.

Paper rows carry NULL fill columns. When the owner actually places a bet, this command records
what was bet -- book, line, price, stake and the fill instant -- on the row named by its full
ledger key ``(game_id, season, week, target, arm)``. Each fill column moves from NULL to a value
ONCE: a second write touching a column that already holds a value is refused by name (even with
the same value) and the store is left byte-identical. The write goes through the ledger's one
guarded write path (``forward_ledger.store.commit_changes``), which confines it to the fill
columns, so the row's immutable half, chain hash and grading are untouched.

A successful fill changes the ledger directory, so it triggers the one sync (D-01): the private
backup is committed and pushed (a fill moves no chain head, so no anchor commit is made). A failed
push is printed and recorded, never fails the command; ``scripts/sync_ledger.py`` retries it.

``--filled-at`` must carry a UTC offset (``2026-10-18T12:01:00-04:00``); a naive instant is refused,
never assumed. It is stored in UTC. At least one fill value is required.

Output: ``FILL_RECORDED=`` (key and columns) then the sync's outcome lines, or ``FILL_REFUSED=``.

Exit codes: 0 recorded; 1 refused by the ledger (already recorded, unknown key, busy, broken
chain); 2 bad input (naive or unparseable ``--filled-at``, no fill value).

Usage:
    uv run python -m scripts.record_fill --game-id 2026_07_KC_BUF --season 2026 --week 7 \\
        --target ats --sportsbook draftkings --line -3.5 --odds -110 --stake-dollars 50 \\
        --filled-at 2026-10-18T12:01:00-04:00

No module under ``api/`` may import this CLI (D-18, UIAP-01): a fill never reaches a page.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from forward_ledger.canonical import CanonicalValueError
from forward_ledger.run_log import record_event
from forward_ledger.schema import ARM_LIVE, ARMS
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerChainBrokenError,
    LedgerEntryRefusedError,
    LedgerFormatError,
    LedgerWriterBusyError,
    commit_changes,
)
from forward_ledger.sync import publish_ledger_state
from forward_ledger.transitions import ColumnSetViolationError, FillAlreadyRecordedError
from scripts.ingest_historical_odds import require_aware_snapshot_ts
from scripts.sync_ledger import outcome_lines

# Seams: tests replace these so nothing is pushed and nothing is appended to logs/.
PUBLISH = publish_ledger_state
LOG = record_event

# The repository whose ledger-anchor branch the sync reads (the working directory, as the daily
# task and every owner CLI run from the repository root).
REPO_DIR = Path()

# Every refusal the ledger names for a fill write.
_REFUSALS: tuple[type[BaseException], ...] = (
    FillAlreadyRecordedError,
    LedgerEntryRefusedError,
    LedgerWriterBusyError,
    LedgerChainBrokenError,
    LedgerFormatError,
    ColumnSetViolationError,
    CanonicalValueError,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description="Record a real bet's fill on one ledger row (each column once)"
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory (default: {LEDGER_DIR.as_posix()})",
    )
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--target", required=True, choices=("wp", "ats", "ou"))
    parser.add_argument("--arm", default=ARM_LIVE, choices=ARMS)
    parser.add_argument("--sportsbook", help="The book the bet was placed at")
    parser.add_argument("--line", type=float, help="The line taken (spread or total)")
    parser.add_argument("--odds", type=float, help="The American price taken")
    parser.add_argument("--stake-dollars", type=float, help="The stake, in dollars")
    parser.add_argument(
        "--filled-at", help="When the bet was placed, ISO 8601 WITH a UTC offset"
    )
    return parser


def _fill_values(args: argparse.Namespace) -> dict[str, Any]:
    """The fill columns the owner named; ``--filled-at`` rendered in UTC.

    Raises:
        ValueError: the instant is naive or unparseable, or no fill value was given.
    """
    values: dict[str, Any] = {
        "fill_sportsbook": args.sportsbook,
        "fill_line": args.line,
        "fill_odds": args.odds,
        "fill_stake_dollars": args.stake_dollars,
    }
    if args.filled_at is not None:
        values["fill_at_utc"] = require_aware_snapshot_ts(args.filled_at).isoformat()
    given = {name: value for name, value in values.items() if value is not None}
    if not given:
        msg = (
            "no fill value given; name at least one of --sportsbook, --line, --odds, "
            "--stake-dollars, --filled-at"
        )
        raise ValueError(msg)
    return given


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    # The full ledger key, in LEDGER_ROW_KEY order: (game_id, season, week, target, arm).
    key = (args.game_id, args.season, args.week, args.target, args.arm)
    try:
        fill = _fill_values(args)
    except ValueError as error:
        print(f"FILL_REFUSED= {error}")
        return 2

    try:
        result = commit_changes(args.ledger_dir, fill_updates={key: fill})
    except _REFUSALS as refusal:
        print(f"FILL_REFUSED= {type(refusal).__name__}: {refusal}")
        return 1

    print(f"FILL_RECORDED= {'|'.join(str(part) for part in key)} {sorted(fill)}")
    LOG(
        "fill_recorded",
        key=list(key),
        columns=sorted(fill),
        head_hash=result.head_hash,
        entry_count=result.entry_count,
    )
    # D-01: a real-fill write changed the ledger directory, so the backup is pushed now.
    outcome = PUBLISH(ledger_dir=args.ledger_dir, repo_dir=REPO_DIR, log=LOG)
    for line in outcome_lines(outcome):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
