#!/usr/bin/env python3
"""Mark one observed sealed-zone upstream change as seen (Phase 35, D-04). Owner-run, never a page.

The weekly probe compares every sealed (dataset, season) pair with the baseline the sealed lock
records. When upstream moves, the probe line carries an UNACKNOWLEDGED finding and the Saturday
email lists it until the owner rules on it. This command records that ruling: it copies the
observed values (``observed_updated_at``, ``observed_size``, ``observed_sha256``) from the LATEST
probe line's finding for the pair -- so the owner never types a digest -- and writes them, with
who ruled (``--ruled-by``, default ``owner``), when (the current UTC instant) and a REQUIRED
reason, onto that pair's entry in ``config/upstream_pin.sealed.lock`` through
``data.sealed_probe.acknowledge_divergence``. The probe then reads the change as acknowledged for
those same observed values; a later, different move is a new finding again.

THIS IS NOT A RE-FREEZE (D32-10). ``config/upstream_pin.json`` is never opened: the pin keeps the
bytes this project measured on, and the acknowledgement is the record that upstream moved and the
owner chose not to follow. A second acknowledgement on the same pair moves the first into
``supersedes``, so the history of rulings stays readable.

Sealed zone only (D-21): live-season (2026) verdicts have no acknowledgement path.

The lock is rewritten ATOMICALLY: it is serialised in full first, written to a temp file in the
lock's own directory, flushed and fsynced, then moved over the lock with ``os.replace``. An
interrupted acknowledgement therefore leaves the previous complete lock, never a truncated
binding record (the ``data.upstream_live.write_live_manifest`` pattern, WR-04).

Output: ``ACK_RECORDED= <dataset> <season>`` and a reminder to commit the lock, or
``ACK_REFUSED= <why>``.

Exit codes: 0 recorded; 1 refused (no finding for the pair in the latest probe line, no readable
lock or probe log, a pair the lock does not track, the lock could not be written); 2 bad input (a
missing or blank reason, a missing flag).

Usage:
    uv run python -m scripts.acknowledge_upstream_change --dataset schedules --season 2025 \\
        --reason "re-upload checked by hand: the pinned bytes still match"

No module under ``api/`` may import this CLI (UIAP-01): an owner ruling never reaches a page.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.sealed_probe import acknowledge_divergence
from data.sealed_probe_log import (
    SEALED_PROBE_LOG_PATH,
    SealedProbeLogCorrupt,
    latest_entry,
)
from data.upstream_pin import SEALED_LOCK_PATH, UpstreamPinError, load_sealed_lock
from data.write_sink import current_sink

# The write-sink kind for this one write (data.write_sink: a dry run records it, writes nothing).
SINK_KIND = "sealed_lock_acknowledgement"


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Mark one observed sealed-zone upstream change as seen (records an owner "
            "ruling on the sealed lock; never re-freezes the pin)"
        )
    )
    parser.add_argument("--dataset", required=True, help="The dataset, e.g. schedules")
    parser.add_argument("--season", type=int, required=True, help="The sealed season")
    parser.add_argument(
        "--reason", required=True, help="Why the change is accepted (required)"
    )
    parser.add_argument(
        "--ruled-by", default="owner", help="Who is ruling (default: owner)"
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=SEALED_LOCK_PATH,
        help=f"The sealed lock (default: {SEALED_LOCK_PATH.as_posix()})",
    )
    parser.add_argument(
        "--probe-log",
        type=Path,
        default=SEALED_PROBE_LOG_PATH,
        help=f"The sealed probe log (default: {SEALED_PROBE_LOG_PATH.as_posix()})",
    )
    return parser


def _latest_finding(
    probe_log: Path, dataset: str, season: int
) -> dict[str, Any] | None:
    """The finding for (*dataset*, *season*) in the latest probe line, or None.

    Raises:
        SealedProbeLogCorrupt: the probe log cannot be read.
    """
    latest = latest_entry(probe_log)
    findings = (latest or {}).get("findings") or []
    for finding in findings:
        if (
            isinstance(finding, dict)
            and finding.get("dataset") == dataset
            and str(finding.get("season")) == str(season)
        ):
            return finding
    return None


def _write_sealed_lock(lock: dict, lock_path: Path) -> bool:
    """Write *lock* to *lock_path* atomically, through the write sink. True when written.

    The serializer is ``scripts/pin_upstream_snapshot.py``'s (``json.dumps(indent=2)`` plus a
    newline), built in full before any file is opened. The write is
    ``data.upstream_live.write_live_manifest``'s: a temp file in the same directory, flushed and
    fsynced, then ``os.replace``; the temp file is removed on any exception.

    Raises:
        OSError: the lock could not be written (the previous lock is left as it was).
    """
    if not current_sink().authorize(str(lock_path), SINK_KIND, 1):
        return False
    payload = json.dumps(lock, indent=2) + "\n"

    handle, temporary = tempfile.mkstemp(
        dir=str(lock_path.parent), prefix=f".{lock_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # os.replace and not Path.replace: the same deliberate choice
        # data/upstream_live.py and pipeline/execution_log.py record.
        os.replace(temporary, lock_path)  # noqa: PTH105
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if not args.reason.strip():
        print("ACK_REFUSED= a reason is required")
        return 2
    pair = f"{args.dataset} {args.season}"

    try:
        finding = _latest_finding(args.probe_log, args.dataset, args.season)
        lock = load_sealed_lock(args.lock_path)
    except (SealedProbeLogCorrupt, UpstreamPinError) as error:
        print(f"ACK_REFUSED= {type(error).__name__}: {error}")
        return 1
    if finding is None:
        print(f"ACK_REFUSED= {pair} has no finding in the latest probe line")
        return 1
    if lock is None:
        print(f"ACK_REFUSED= no sealed lock at {args.lock_path.as_posix()}")
        return 1

    try:
        updated = acknowledge_divergence(
            lock,
            dataset=args.dataset,
            season=args.season,
            observed_updated_at=finding.get("observed_updated_at"),
            observed_size=finding.get("observed_size"),
            observed_sha256=finding.get("observed_sha256"),
            ruled_by=args.ruled_by,
            ruled_at_utc=datetime.now(UTC).isoformat(),
            reason=args.reason,
        )
    except ValueError as error:
        print(f"ACK_REFUSED= {error}")
        return 1

    try:
        written = _write_sealed_lock(updated, args.lock_path)
    except OSError as error:
        print(
            f"ACK_REFUSED= the sealed lock could not be written ({type(error).__name__})"
        )
        return 1
    if not written:
        print("ACK_REFUSED= the write sink withheld the write; nothing was written")
        return 1

    print(f"ACK_RECORDED= {pair}")
    print(
        f"ACK_NEXT= commit {args.lock_path.as_posix()}: the acknowledgement is a "
        "committed owner ruling"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
