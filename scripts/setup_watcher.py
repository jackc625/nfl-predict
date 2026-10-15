#!/usr/bin/env python3
"""Configure the outside watcher and prove it by reading it back (Phase 35, D-02, D-22, SPEC R7).

The watcher is a healthchecks.io check the daily run checks in with. If no check-in arrives, the
check goes late at 17:00 ET and down 90 minutes later, at 18:30 ET, and healthchecks.io alerts the
owner through two integrations -- from outside this laptop, so a dead machine still raises an alarm.

This command makes the check's settings exactly:

* schedule ``0 17 * * *`` in tz ``America/New_York`` (the zone, not an offset, so the schedule
  follows daylight saving: 17:00 ET stays 17:00 ET across 2026-11-01);
* grace 5400 seconds (late at 17:00 ET, down at 18:30 ET);
* ``manual_resume`` false (a paused check resumes on its next check-in, D-01);
* both integrations attached: ``NFL Predict ntfy`` and ``NFL Predict email``, matched by the NAME
  the owner gave them, so the integration kind strings never matter.

Then it reads the check back and prints one ``<field>= <want> | <got> | MATCH|MISMATCH`` line per
field (channels compared as a set and printed by integration NAME) and
``WATCHER_READBACK_MATCH= True|False`` -- the same read-back shape as
``deployment/setup_scheduling.py``. That read-back is the evidence for SPEC R7.

UPDATE-ONLY (D-22). The owner creates the check and the two integrations ONCE in the healthchecks.io
web UI (the Management API cannot create integrations, D-02) and puts the check's uuid ping URL in
``.env`` as ``HEALTHCHECKS_PING_URL``, with the project's READ-WRITE key as
``HEALTHCHECKS_API_KEY``. The check uuid is derived from the ping URL. This command never creates a
check: an unknown uuid is refused by name.

It prints no secret: no ping URL, no uuid, no API key, no integration id and never a response body
or exception message (a read-write response carries the uuid and the ping, update and pause URLs).

Output: the read-back lines, or ``WATCHER_SETUP_REFUSED= <why>``, then ``WATCHER_READBACK_MATCH=``.

Exit codes: 0 every field matches; 1 refused (an integration or the check is missing, the key is
refused, the ping URL is not the uuid form) or any field differs; 2 a watcher secret is not set.

Usage:
    uv run python -m scripts.setup_watcher                 # update, then read back
    uv run python -m scripts.setup_watcher --verify-only   # read back only

No module under ``api/`` may import this CLI (UIAP-01).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import httpx

from notify.channels import (
    WatcherApiError,
    WatcherChannel,
    WatcherCheck,
    check_uuid,
    hc_get_check,
    hc_list_channels,
    hc_update_check,
)
from notify.secrets import NotifySecrets, load_secrets

WATCHER_SCHEDULE = "0 17 * * *"
WATCHER_TZ = "America/New_York"
# Late at 17:00 ET (the cron above), down 90 minutes later at 18:30 ET. The tz is the zone, so
# both times follow daylight saving.
WATCHER_GRACE_SECONDS = 5400
WATCHER_NTFY_CHANNEL_NAME = "NFL Predict ntfy"
WATCHER_EMAIL_CHANNEL_NAME = "NFL Predict email"

#: The fields the read-back compares, in the order it prints them.
READBACK_FIELDS: tuple[str, ...] = (
    "schedule",
    "tz",
    "grace",
    "manual_resume",
    "channels",
)

# Seam: tests inject dummy secrets so the owner's .env is never read.
LOAD_SECRETS = load_secrets

_MISSING_SECRETS = (
    "HEALTHCHECKS_PING_URL and HEALTHCHECKS_API_KEY must both be set in .env"
)
_NOT_UUID_FORM = (
    "HEALTHCHECKS_PING_URL is not of the form https://hc-ping.com/<uuid>; copy the uuid "
    "ping URL from the check's page in the healthchecks.io web UI"
)


class _Refused(Exception):
    """A named refusal; its message is printed after ``WATCHER_SETUP_REFUSED=``."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Update the healthchecks.io check (schedule, tz, grace, manual_resume, "
            "channels) and read it back field by field. Never creates a check."
        )
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read the check back without updating it",
    )
    return parser


def _emit(line: str) -> None:
    """One output line on stdout, where the owner reads it."""
    sys.stdout.write(line + "\n")


def _api_refusal(error: WatcherApiError) -> str:
    """The named refusal for a Management API status (never the response body)."""
    if error.status == 404:
        return (
            "the check named by HEALTHCHECKS_PING_URL was not found (HTTP 404); create "
            "the check in the healthchecks.io web UI first and put its uuid ping URL in "
            ".env -- this command never creates a check"
        )
    if error.status in (401, 403):
        return (
            f"healthchecks.io refused HEALTHCHECKS_API_KEY (HTTP {error.status}); it must "
            "be the project's read-write API key (a read-only key cannot update or read "
            "back the check)"
        )
    return f"the healthchecks.io Management API answered HTTP {error.status}"


def _pick_channels(
    channels: Sequence[WatcherChannel],
) -> tuple[WatcherChannel, WatcherChannel]:
    """The ntfy and email integrations, by the names the owner gave them.

    Raises:
        _Refused: an integration is missing or its name is not unique.
    """
    picked: list[WatcherChannel] = []
    problems: list[str] = []
    for name in (WATCHER_NTFY_CHANNEL_NAME, WATCHER_EMAIL_CHANNEL_NAME):
        named = [channel for channel in channels if channel.name == name]
        if not named:
            problems.append(f"no integration named '{name}'")
        elif len(named) > 1:
            problems.append(f"{len(named)} integrations are named '{name}'")
        else:
            picked.append(named[0])
    if problems:
        msg = (
            f"{'; '.join(problems)}. Create (or rename) the integrations in the "
            f"healthchecks.io web UI so exactly one is named '{WATCHER_NTFY_CHANNEL_NAME}' "
            f"and one '{WATCHER_EMAIL_CHANNEL_NAME}', then re-run"
        )
        raise _Refused(msg)
    return picked[0], picked[1]


def _channel_names(ids: frozenset[str], channels: Sequence[WatcherChannel]) -> str:
    """*ids* printed as the owner's integration names (an id itself is never printed)."""
    names_by_id = {channel.id: channel.name for channel in channels}
    names = sorted(
        names_by_id.get(channel_id, "an unlisted integration") for channel_id in ids
    )
    return ", ".join(names) if names else "none"


def compare_check_fields(
    check: WatcherCheck,
    wanted: tuple[WatcherChannel, WatcherChannel],
    channels: Sequence[WatcherChannel],
) -> list[tuple[str, str, str, bool]]:
    """``(field, want, got, match)`` for each of :data:`READBACK_FIELDS`."""
    wanted_ids = frozenset(channel.id for channel in wanted)
    return [
        (
            "schedule",
            WATCHER_SCHEDULE,
            str(check.schedule),
            check.schedule == WATCHER_SCHEDULE,
        ),
        ("tz", WATCHER_TZ, str(check.tz), check.tz == WATCHER_TZ),
        (
            "grace",
            str(WATCHER_GRACE_SECONDS),
            str(check.grace),
            check.grace == WATCHER_GRACE_SECONDS,
        ),
        (
            "manual_resume",
            "False",
            str(check.manual_resume),
            check.manual_resume is False,
        ),
        (
            "channels",
            _channel_names(wanted_ids, channels),
            _channel_names(check.channel_ids, channels),
            check.channel_ids == wanted_ids,
        ),
    ]


def _configure_and_read_back(
    secrets: NotifySecrets, verify_only: bool
) -> list[tuple[str, str, str, bool]]:
    """List the integrations, update the check unless *verify_only*, and read it back.

    Raises:
        _Refused: any named refusal.
    """
    try:
        channels = hc_list_channels(secrets)
        wanted = _pick_channels(channels)
        if not verify_only:
            hc_update_check(
                secrets,
                {
                    "schedule": WATCHER_SCHEDULE,
                    "tz": WATCHER_TZ,
                    "grace": WATCHER_GRACE_SECONDS,
                    "manual_resume": False,
                    "channels": ",".join(channel.id for channel in wanted),
                },
            )
        check = hc_get_check(secrets)
    except WatcherApiError as error:
        raise _Refused(_api_refusal(error)) from None
    except httpx.HTTPError as error:
        msg = f"healthchecks.io could not be reached ({type(error).__name__})"
        raise _Refused(msg) from None
    return compare_check_fields(check, wanted, channels)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    secrets = LOAD_SECRETS()
    if secrets.healthchecks_ping_url is None or secrets.healthchecks_api_key is None:
        _emit(f"WATCHER_SETUP_REFUSED= {_MISSING_SECRETS}")
        _emit("WATCHER_READBACK_MATCH= False")
        return 2
    try:
        check_uuid(secrets.healthchecks_ping_url.get_secret_value())
    except ValueError:
        _emit(f"WATCHER_SETUP_REFUSED= {_NOT_UUID_FORM}")
        _emit("WATCHER_READBACK_MATCH= False")
        return 1

    try:
        rows = _configure_and_read_back(secrets, args.verify_only)
    except _Refused as refusal:
        _emit(f"WATCHER_SETUP_REFUSED= {refusal}")
        _emit("WATCHER_READBACK_MATCH= False")
        return 1

    for name, want, got, match in rows:
        _emit(f"{name}= {want} | {got} | {'MATCH' if match else 'MISMATCH'}")
    matched = all(match for *_row, match in rows)
    _emit(f"WATCHER_READBACK_MATCH= {matched}")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
