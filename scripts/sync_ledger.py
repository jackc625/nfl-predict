#!/usr/bin/env python3
"""Run the ledger's one sync by hand: anchor the head, back up the directory (Phase 34, D-01).

The same ``forward_ledger.sync.publish_ledger_state`` the daily run calls after a ledger write: it
verifies the chain, makes ONE anchor commit on ``ledger-anchor`` when the head moved, pushes the
anchor when it is ahead of the last successful push, commits the ledger directory when anything in
it changed and pushes the private backup when it is ahead. Use it after the go-live migration, after
a failed push, or whenever the owner wants the remote current.

A failed push is an OUTCOME, not a failure: it is printed and recorded in the run log, and the
next sync carries the current head (LDGR-05). The command fails only when the ledger cannot be
read or its chain is broken -- nothing is published from a broken chain.

Output: ``ANCHOR_COMMITTED=``, ``ANCHOR_PUSHED=``, ``ANCHOR_ERROR=``, ``BACKUP_COMMITTED=``,
``BACKUP_PUSHED=``, ``BACKUP_ERROR=``, then ``SYNC_REFUSED=`` when refused, and ``SYNC_RESULT=``.

Exit codes: 0 synced (push failures allowed); 1 refused (unreadable ledger or broken chain).

Usage:
    uv run python -m scripts.sync_ledger
    uv run python -m scripts.sync_ledger --ledger-dir path/to/ledger --repo-dir path/to/repo

No module under ``api/`` may import this CLI (D-18, UIAP-01).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forward_ledger.run_log import record_event
from forward_ledger.store import LEDGER_DIR
from forward_ledger.sync import SyncOutcome, publish_ledger_state
from forward_ledger.transport import run_git

# Seams: tests replace these so no git push runs and nothing is appended to logs/.
RUNNER = run_git
LOG = record_event


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description="Anchor the ledger head and back up the ledger directory, by hand"
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory and backup repository (default: {LEDGER_DIR.as_posix()})",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(),
        help="The repository carrying the ledger-anchor branch (default: .)",
    )
    return parser


def outcome_lines(outcome: SyncOutcome) -> list[str]:
    """One ``FIELD= value`` line per fact of *outcome*, ending with ``SYNC_RESULT=``."""
    lines = [
        f"ANCHOR_COMMITTED= {outcome.anchor_committed}",
        f"ANCHOR_PUSHED= {outcome.anchor_pushed}",
        f"ANCHOR_ERROR= {outcome.anchor_error or 'none'}",
        f"BACKUP_COMMITTED= {outcome.backup_committed}",
        f"BACKUP_PUSHED= {outcome.backup_pushed}",
        f"BACKUP_ERROR= {outcome.backup_error or 'none'}",
    ]
    if outcome.refused_reason is not None:
        lines.append(f"SYNC_REFUSED= {outcome.refused_reason}")
    lines.append(
        f"SYNC_RESULT= {'REFUSED' if outcome.refused_reason is not None else 'DONE'}"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    outcome = publish_ledger_state(
        ledger_dir=args.ledger_dir,
        repo_dir=args.repo_dir,
        runner=RUNNER,
        log=LOG,
    )
    for line in outcome_lines(outcome):
        print(line)
    return 1 if outcome.refused_reason is not None else 0


if __name__ == "__main__":
    sys.exit(main())
