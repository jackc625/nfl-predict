#!/usr/bin/env python3
"""Replay the 2026 forward ledger from what it stored (Phase 34, LDGR-09, D-02). A CLI, never a route.

Each ledger row is re-derived from its own decision-input snapshot, the ledger's copies of the
artifacts it was scored by and the ledger's copy of the chain-fit record -- never from
``artifacts/``, ``outputs/`` or a rebuilt gold (``forward_ledger.replay`` holds the rules) -- and
compared field by field: categoricals exactly, numerics within 1e-9.

Output is one ``REPLAY= <game_id|season|week|target|arm> <status> [reason]`` line per row, in
ledger order, then ``REPLAY_PASS=``, ``REPLAY_NOT_REPLAYABLE=`` and ``REPLAY_FAIL=``. A pre-verdict
row with no snapshot is ``not_replayable`` -- reported, never counted as a pass; a verdict row with
no snapshot is a ``fail``. The chain is verified first: replay never runs on an unverified ledger.

No module under ``api/`` may import this CLI or the ``forward_ledger`` package (D-18, UIAP-01);
nothing on the daily run imports it either.

Exit codes: 0 no row failed; 1 any row failed; 2 the store cannot be read or its chain is broken.

Usage:
    uv run python -m scripts.replay_ledger
    uv run python -m scripts.replay_ledger --week 6
    uv run python -m scripts.replay_ledger --week 6 --game-id 2026_06_KC_BUF
    uv run python -m scripts.replay_ledger --ledger-dir path/to/restored/ledger

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forward_ledger.replay import (
    REPLAY_FAIL,
    REPLAY_NOT_REPLAYABLE,
    REPLAY_PASS,
    ReplayResult,
    replay_ledger,
)
from forward_ledger.store import LEDGER_DIR, LedgerChainBrokenError, LedgerFormatError


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay the 2026 forward ledger: re-derive every row from its stored snapshot and "
            "the ledger's own artifact copies, and compare it field by field"
        )
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory (default: {LEDGER_DIR.as_posix()})",
    )
    parser.add_argument("--season", type=int, help="Replay only this season's rows")
    parser.add_argument("--week", type=int, help="Replay only this week's rows")
    parser.add_argument(
        "--game-id",
        action="append",
        dest="game_ids",
        help="Replay only this game's rows (repeatable)",
    )
    return parser


def result_line(result: ReplayResult) -> str:
    """One ``REPLAY=`` line: the row key, its status and, unless it passed, the reason."""
    key = "|".join(str(part) for part in result.key)
    line = f"REPLAY= {key} {result.status}"
    return line if result.reason is None else f"{line} {result.reason}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        results = replay_ledger(
            args.ledger_dir,
            season=args.season,
            week=args.week,
            game_ids=args.game_ids,
        )
    except LedgerFormatError as error:
        print(f"LEDGER_UNREADABLE= {error}")
        return 2
    except LedgerChainBrokenError as error:
        print(f"LEDGER_CHAIN_BROKEN= {error}")
        return 2

    for result in results:
        print(result_line(result))
    counts = {
        status: sum(result.status == status for result in results)
        for status in (REPLAY_PASS, REPLAY_NOT_REPLAYABLE, REPLAY_FAIL)
    }
    print(f"REPLAY_PASS= {counts[REPLAY_PASS]}")
    print(f"REPLAY_NOT_REPLAYABLE= {counts[REPLAY_NOT_REPLAYABLE]}")
    print(f"REPLAY_FAIL= {counts[REPLAY_FAIL]}")
    return 1 if counts[REPLAY_FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
