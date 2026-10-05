#!/usr/bin/env python3
"""Verify the 2026 forward ledger's hash chain (Phase 34, LDGR-06). A CLI, never a route.

Recomputes every chain hash from the published ``GENESIS_HASH`` and reports, one ``FIELD= value``
line each, how many entries the store holds, its head, and whether the chain is intact. On a broken
chain it names the FIRST entry that fails by its seq and its ``game_id|season|week|target|arm`` key.

Plan 34-10 adds the local and remote anchor comparison, the backup lag, the verdict-week stamp
check and the D-19 re-grade to THIS CLI. No module under ``api/`` may import it or the
``forward_ledger`` package (D-18, UIAP-01).

Exit codes: 0 intact; 1 the chain is broken; 2 the store exists but cannot be read.

Usage:
    uv run python -m scripts.verify_ledger
    uv run python -m scripts.verify_ledger --ledger-dir path/to/ledger

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forward_ledger.store import (
    LEDGER_DIR,
    LedgerFormatError,
    read_entries,
    verify_chain,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description="Verify the 2026 forward ledger's hash chain from its genesis constant"
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory (default: {LEDGER_DIR.as_posix()})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        entries = read_entries(args.ledger_dir)
    except LedgerFormatError as error:
        print(f"LEDGER_UNREADABLE= {error}")
        return 2

    verdict = verify_chain(entries)
    print(f"LEDGER_ENTRIES= {verdict.entry_count}")
    print(f"HEAD_HASH= {verdict.head_hash}")
    print(f"CHAIN_OK= {verdict.ok}")
    if verdict.ok:
        return 0

    key = verdict.first_broken_key or ()
    print(f"FIRST_BROKEN_SEQ= {verdict.first_broken_seq}")
    print(f"FIRST_BROKEN_KEY= {'|'.join(str(part) for part in key)}")
    print(f"CHAIN_BROKEN_REASON= {verdict.reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
