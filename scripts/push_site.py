#!/usr/bin/env python3
"""Push the website now, without re-running the day (Phase 36, D-05, SPEC R11).

WHAT IT DOES
------------
Re-pushes the current web cache (``data/web_cache.duckdb``) to the hosted site through the daily
push's OWN routine, ``pipeline.site_push.push_site``: the same setup switch, the same refusals (an
unfinished rebuild or missing cache, a stylesheet out of step, a data-format mismatch that one
rebuild cannot fix), the same fresh publish id, the same read-back proof and the same push-log
record, with source ``manual``. Use it after a failed daily push, or after renewing the token.

It refuses at once, without waiting, while a cache rebuild or another push holds the publish lock,
so do not run it between 16:30 and 18:15 ET, when the 5 PM run's rebuild and push hold that lock.
Until ``config/site_push_enabled.json`` exists it does nothing but record "not pushed".

OUTPUT CONTRACT
---------------
The routine's own ``SITE_PUSH=`` line, then one ``PUSH_SITE= <outcome> <publish id | reason>``
line. A failure is named by a reason or an exception CLASS only; a secret is never printed.

EXIT CODES
----------
0 the push succeeded and read back; 1 anything else, including a refusal because a cache rebuild
or another push holds the lock.

Usage:
    uv run python -m scripts.push_site

No module under ``api/`` may import this CLI (UIAP-01): a page request never deploys.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys

from pipeline import site_push

_HELP = """\
Re-push the current web cache to the website through the daily push's own
routine (same switch, refusals, read-back and push-log record), without
re-running the day.

Exits 0 only when the push succeeded and read back; 1 otherwise, including a
refusal (lock_held) because a cache rebuild or another push holds the lock.
Do not run it between 16:30 and 18:15 ET, while the 5 PM run's rebuild and
push hold that lock. Until config/site_push_enabled.json exists it only
records "not pushed".
"""


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (no options beyond ``--help``: it always pushes the current cache)."""
    return argparse.ArgumentParser(
        prog="uv run python -m scripts.push_site",
        description=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main(argv: list[str] | None = None) -> int:
    """Push once through the daily push's routine; 0 only for a succeeded push, else 1."""
    build_parser().parse_args(argv)
    try:
        outcome = site_push.push_site(
            source=site_push.SOURCE_MANUAL,
            lock_wait_seconds=site_push.MANUAL_LOCK_WAIT_SECONDS,
        )
    except Exception as error:  # noqa: BLE001 - the class name only: a message can carry a value
        # A raise the routine could not record itself (the switch check or the record write).
        print(f"PUSH_SITE= failed {type(error).__name__}")
        return 1
    succeeded = outcome.outcome == site_push.OUTCOME_SUCCEEDED
    said = outcome.publish_id if succeeded else outcome.reason
    print(f"PUSH_SITE= {outcome.outcome} {said or 'unknown'}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
