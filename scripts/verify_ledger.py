#!/usr/bin/env python3
"""Verify the 2026 forward ledger (Phase 34, LDGR-05, LDGR-06). A CLI, never a route.

What it checks (``forward_ledger.verify`` holds the rules):

* the hash chain, recomputed from the published ``GENESIS_HASH`` -- a broken chain names the FIRST
  entry that fails by its seq and its ``game_id|season|week|target|arm`` key;
* the LOCAL ``ledger-anchor`` commit (authoritative): a missing anchor while rows exist, a tail
  shorter than the anchored row count, or a head that is not the ledger's hash at that count fails;
  rows newer than the anchor are a warning;
* the GitHub anchor, read anonymously over HTTPS: unreachable or behind is a warning, a remote
  that DISAGREES with the ledger fails;
* the private backup's lag (commits not yet pushed, files not yet committed), a warning;
* the committed verdict-scope declaration (LDGR-10, LDGR-11): every live row of the declared
  season must carry the label its week is given, and every row in weeks W..end must carry every
  stamp, a registered recipe and the declared fill convention. No declaration yet is a warning;
* every correction entry must name an existing ledger row;
* D-19: every settled row (win/loss/push) is re-graded from the silver scores through the one
  grader and compared with its in-force outcome (its latest correction entry, else its own
  grade). A different status, payout or realized units, a settled row with no recorded score, or
  a missing silver store or chain-fit record while settled rows exist, fails by name. On a
  machine restored from the backup alone, pass the ledger's own chain-fit copy under
  ``ledger/recipes/`` as ``--chain-fit-path``.

Output is one ``FIELD= value`` line each, then ``LOCAL_RESULT=`` (every local check),
``EXTERNAL_RESULT=`` (the GitHub anchor: VERIFIED, BEHIND, UNREACHABLE, SKIPPED or DISAGREES) and
last ``VERIFY_RESULT=``. No module under ``api/`` may import this CLI or the ``forward_ledger``
package (D-18, UIAP-01); nothing on the daily run imports it either.

Exit codes: 0 PASS (warnings allowed); 1 any failure; 2 the store exists but cannot be read.

Usage:
    uv run python -m scripts.verify_ledger
    uv run python -m scripts.verify_ledger --skip-remote
    uv run python -m scripts.verify_ledger --ledger-dir path/to/ledger --repo-dir path/to/repo
    uv run python -m scripts.verify_ledger --silver-dir data/silver --chain-fit-path path/to/fit.json

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest.weekly_bet_list import DEFAULT_CHAIN_FIT_PATH
from forward_ledger.declarations import VERDICT_SCOPE_MODULE
from forward_ledger.remote_config import ANCHOR_REMOTE_HTTPS_URL
from forward_ledger.settle import DEFAULT_SILVER_DIR
from forward_ledger.store import LEDGER_DIR, LedgerFormatError
from forward_ledger.verify import (
    REMOTE_BEHIND,
    REMOTE_VERIFIED,
    VerifyReport,
    verify_ledger,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify the 2026 forward ledger: its hash chain, its local and GitHub anchors "
            "and its backup lag"
        )
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory (default: {LEDGER_DIR.as_posix()})",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(),
        help="The repository carrying the ledger-anchor branch (default: .)",
    )
    parser.add_argument(
        "--remote-url",
        default=ANCHOR_REMOTE_HTTPS_URL,
        help=f"Where the remote anchor is read from (default: {ANCHOR_REMOTE_HTTPS_URL})",
    )
    parser.add_argument(
        "--skip-remote",
        action="store_true",
        help="Do not read the remote anchor (reported as EXTERNAL_RESULT= SKIPPED)",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=DEFAULT_SILVER_DIR,
        help=(
            "The silver store whose games.parquet the settled rows are re-graded from "
            f"(default: {DEFAULT_SILVER_DIR.as_posix()})"
        ),
    )
    parser.add_argument(
        "--chain-fit-path",
        type=Path,
        default=DEFAULT_CHAIN_FIT_PATH,
        help=(
            "The chain-fit record the grading strategies are built from (default: "
            f"{DEFAULT_CHAIN_FIT_PATH.as_posix()}; on a machine restored from the backup "
            "alone, the ledger's own copy under ledger/recipes/)"
        ),
    )
    return parser


def _chain_lines(report: VerifyReport) -> list[str]:
    lines = [
        f"LEDGER_ENTRIES= {report.entries}",
        f"HEAD_HASH= {report.head_hash}",
        f"CHAIN_OK= {report.chain.ok}",
    ]
    if not report.chain.ok:
        key = report.chain.first_broken_key or ()
        lines += [
            f"FIRST_BROKEN_SEQ= {report.chain.first_broken_seq}",
            f"FIRST_BROKEN_KEY= {'|'.join(str(part) for part in key)}",
            f"CHAIN_BROKEN_REASON= {report.chain.reason}",
        ]
    return lines


def _anchor_lines(report: VerifyReport) -> list[str]:
    anchor = report.local_anchor
    rows = "none" if anchor.rows is None else anchor.rows
    lines = [f"LOCAL_ANCHOR_ROWS= {rows}", f"LOCAL_ANCHOR_OK= {anchor.ok}"]
    if anchor.reason is not None:
        lines.append(f"LOCAL_ANCHOR_REASON= {anchor.reason}")
    lines.append(f"UNANCHORED_ENTRIES= {report.unanchored}")

    remote = report.remote
    if remote.state in (REMOTE_VERIFIED, REMOTE_BEHIND):
        lines.append(f"REMOTE_ANCHOR_BEHIND_BY= {remote.behind_by}")
    else:
        lines.append(f"REMOTE_ANCHOR= {remote.state}")
    return lines


def _backup_lines(report: VerifyReport) -> list[str]:
    backup = report.backup
    if backup.behind_by is None or backup.uncommitted is None:
        return [f"BACKUP= {backup.error}"]
    return [
        f"BACKUP_BEHIND_BY= {backup.behind_by}",
        f"BACKUP_UNCOMMITTED= {backup.uncommitted}",
    ]


def _flag(value: bool | None) -> str:
    return "n/a" if value is None else str(value)


def _verdict_lines(report: VerifyReport) -> list[str]:
    verdict = report.verdict
    start = "none" if verdict.start_week is None else verdict.start_week
    return [
        f"VERDICT_DECLARED= {verdict.declared}",
        f"VERDICT_START_WEEK= {start}",
        f"VERDICT_STAMPS_OK= {_flag(verdict.stamps_ok)}",
        f"VERDICT_LABELS_OK= {_flag(verdict.labels_ok)}",
        f"CORRECTIONS_OK= {verdict.corrections_ok}",
    ]


def _regrade_lines(report: VerifyReport) -> list[str]:
    regrade = report.regrade
    status = "unavailable" if regrade.unavailable is not None else str(regrade.ok)
    lines = [f"REGRADE_CHECKED= {regrade.checked}", f"REGRADE_OK= {status}"]
    lines += [
        f"REGRADE_MISMATCH= {mismatch.seq} {'|'.join(str(part) for part in mismatch.key)} "
        f"{mismatch.field} stored={mismatch.stored} regraded={mismatch.regraded} "
        f"{mismatch.reason}"
        for mismatch in regrade.mismatches
    ]
    return lines


def report_lines(report: VerifyReport) -> list[str]:
    """Every output line for *report*, ``VERIFY_RESULT=`` last."""
    lines = [
        *_chain_lines(report),
        *_anchor_lines(report),
        *_backup_lines(report),
        *_verdict_lines(report),
        *_regrade_lines(report),
    ]
    lines += [f"WARNING= {warning}" for warning in report.warnings]
    lines += [f"FAILURE= {failure}" for failure in report.failures]
    lines += [
        f"LOCAL_RESULT= {'PASS' if report.local_ok else 'FAIL'}",
        f"EXTERNAL_RESULT= {report.remote.state.upper()}",
        f"VERIFY_RESULT= {'PASS' if report.ok else 'FAIL'}",
    ]
    return lines


def main(
    argv: list[str] | None = None, *, verdict_scope_module: str = VERDICT_SCOPE_MODULE
) -> int:
    """CLI entry point.

    *verdict_scope_module* is not a command-line flag: the owner always verifies against the
    committed declaration. Tests pass a fixture module name.
    """
    args = build_parser().parse_args(argv)
    try:
        report = verify_ledger(
            args.ledger_dir,
            args.repo_dir,
            remote_url=args.remote_url,
            check_remote=not args.skip_remote,
            module_name=verdict_scope_module,
            silver_dir=args.silver_dir,
            chain_fit_path=args.chain_fit_path,
        )
    except LedgerFormatError as error:
        print(f"LEDGER_UNREADABLE= {error}")
        return 2

    for line in report_lines(report):
        print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
