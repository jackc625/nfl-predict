#!/usr/bin/env python3
"""Prove every channel fires: one TEST failed push, one TEST bets push, one TEST digest email (SPEC R9).

WHAT IT PROVES
--------------
The owner runs this once, by hand, to see that the phone and the inbox are really reached. It sends
three messages through the real sender -- the same router, builders, provider calls and notification
log the daily run uses -- each labelled ``TEST - `` so it can never be mistaken for a real one:

1. a ``test`` push titled ``TEST - FAILED: notification test`` at top priority (5), whose body
   says it is a test and that no action is needed;
2. a ``test`` push titled ``TEST - <n> new bets: notification test`` at normal priority (3, as the
   real bets push), listing the live bets of the web cache's current slate through the real bets
   template, or one line saying the cache holds none;
3. a ``test`` email: the real weekly digest built from the current web cache, its subject and
   heading prefixed ``TEST - ``.

Each message's provider receipt is printed on its own line, and each leaves exactly one
notification-log record with source ``test_command``. The three records tell apart by priority:
5, 3 and None (Plan 35-18 ties each printed receipt to its record that way).

IT NEVER CHECKS IN
------------------
This command imports no check-in function and never contacts the outside watcher: a TEST check-in
would make a laptop that missed its daily run look alive. Only the daily run's end-of-run step
checks in (SPEC must-NOT).

Output: ``NOTIFY_TEST_FAILED=``, ``NOTIFY_TEST_BETS=`` and ``NOTIFY_TEST_DIGEST=``, each
``delivered <receipt id>`` or ``failed <failure kind>``. A secret is never printed.

Exit codes: 0 all three delivered; 1 any one not delivered (or the secrets could not be read).

Usage:
    uv run python -m scripts.notify_test

No module under ``api/`` may import this CLI (UIAP-01): sending never reaches a page.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from api.cache import ARM_LIVE
from notify import digest
from notify import secrets as notify_secrets
from notify.digest import DigestInputs, collect_digest_inputs, render_digest
from notify.events import (
    PRIORITY_NORMAL,
    PRIORITY_TOP,
    EmailMessage,
    PushMessage,
    SendContext,
    send_email,
    send_push,
)
from notify.log import read_notifications
from notify.messages import BETS_PUSH_TEMPLATE
from notify.rendering import template_environment
from notify.report import STATUS_SUCCESS, RunReport
from notify.transport import OUTCOME_FAILED, Delivery
from pipeline.daily_steps import slate_lock
from utils.game_lock import ET

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from api.services import DataService

# Seams, read at call time: tests hand in dummy secrets and a tmp web cache, so the owner's .env and
# the real cache are never read under pytest.
LOAD_SECRETS = notify_secrets.load_secrets
CACHE_PATH = digest.WEB_CACHE_PATH

TEST_PREFIX = "TEST - "
SOURCE = "test_command"
EVENT_CLASS = "test"

_NO_ACTION = "No action is needed."


@dataclasses.dataclass(frozen=True)
class _CacheView:
    """What the TEST bets push and the TEST digest read from the web cache, read once.

    Attributes:
        slate: The cache's stamped current ``(season, week)``, or None.
        bets: The slate's live-arm live bets in ``/bets`` order; None when no cache could be read.
        digest_inputs: The digest's selected inputs.
    """

    slate: tuple[int, int] | None
    bets: list[dict[str, Any]] | None
    digest_inputs: DigestInputs


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (no options: the command always sends all three TEST messages)."""
    return argparse.ArgumentParser(
        description=(
            "Send one TEST failed push, one TEST bets push and one TEST digest email through the "
            "real sender, print each receipt id, and never check in with the watcher"
        )
    )


def failed_test_push() -> PushMessage:
    """The TEST failed push: top priority, saying it is a test that needs no action."""
    return PushMessage(
        event_class=EVENT_CLASS,
        title=f"{TEST_PREFIX}FAILED: notification test",
        header_lines=(
            "What failed: nothing. This is a test of the failed notification.",
            _NO_ACTION,
        ),
        item_lines=(),
        priority=PRIORITY_TOP,
    )


def bets_test_push(
    slate: tuple[int, int] | None, bets: Sequence[dict[str, Any]] | None
) -> PushMessage:
    """The TEST bets push: normal priority, the cache's current-slate live bets or a line saying none.

    *bets* is None when no web cache could be read: the push then says the bets are unknown, never
    zero.
    """
    header = ["This is a test of the bets notification.", _NO_ACTION]
    if bets is None:
        title = f"{TEST_PREFIX}bets unknown: notification test"
        header.append("The web cache could not be read, so no bet is listed.")
        items: tuple[str, ...] = ()
    elif not bets:
        title = f"{TEST_PREFIX}0 new bets: notification test"
        header.append("The web cache holds no live bet for its current slate.")
        items = ()
    else:
        noun = "new bet" if len(bets) == 1 else "new bets"
        title = f"{TEST_PREFIX}{len(bets)} {noun}: notification test"
        if slate is not None:
            header.append(f"Live bets in the web cache for {slate[0]} week {slate[1]}:")
        rendered = (
            template_environment().get_template(BETS_PUSH_TEMPLATE).render(bets=bets)
        )
        items = tuple(line.strip() for line in rendered.splitlines() if line.strip())
    return PushMessage(
        event_class=EVENT_CLASS,
        title=title,
        header_lines=tuple(header),
        item_lines=items,
        priority=PRIORITY_NORMAL,
    )


def _schedule_or_none() -> pd.DataFrame | None:
    """The recorded schedule through the digest's own seam, or None when it cannot be read."""
    try:
        return digest.LOAD_SCHEDULE()
    except Exception as error:  # noqa: BLE001 - the email still goes out, saying so
        print(f"NOTIFY_TEST_SCHEDULE= unreadable {type(error).__name__}")
        return None


def _population_instant(
    service: DataService | None, slate: tuple[int, int] | None
) -> datetime | None:
    """When the cache's current slate was populated, as an aware instant, or None."""
    if service is None or slate is None:
        return None
    stored = service.get_bet_list_populated_at(*slate)
    if not isinstance(stored, str) or not stored.strip():
        return None
    try:
        moment = datetime.fromisoformat(stored.strip())
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _read_cache(report: RunReport) -> _CacheView:
    """Read the TEST bets and the TEST digest's inputs from the web cache, in one read-only open.

    The digest judges the cache "current" against the run's start. This command did not rebuild
    the cache, so it is judged against the cache's own population of its current slate: a cache
    that holds that slate shows its bets as the real email would, and an absent or unstamped one
    shows the stale notice. The email's idempotency key still carries this command's own start.
    """
    schedule = _schedule_or_none()
    with digest.open_cache_service(CACHE_PATH) as service:
        slate = service.get_current_slate() if service is not None else None
        bets: list[dict[str, Any]] | None = None
        if service is not None:
            bets = service.get_bet_list(*slate, arm=ARM_LIVE) if slate else []
        if slate is not None:
            report.slate_season, report.slate_week = slate
        population = _population_instant(service, slate)
        judged = dataclasses.replace(report, start=population or report.start)
        inputs = collect_digest_inputs(
            judged,
            schedule=schedule,
            service=service,
            notifications=read_notifications(),
            failed_titles=(),
            now=report.start,
            label_prefix=TEST_PREFIX,
        )
    return _CacheView(
        slate=slate,
        bets=bets,
        digest_inputs=dataclasses.replace(inputs, run_start=report.start),
    )


def digest_test_email(inputs: DigestInputs) -> EmailMessage:
    """The real digest rendered from *inputs*, sent under the ``test`` event class."""
    return dataclasses.replace(render_digest(inputs), event_class=EVENT_CLASS)


def _raised(error: Exception) -> Delivery:
    """A send that raised: failed, no attempt, the class name only."""
    return Delivery(
        outcome=OUTCOME_FAILED,
        attempts=0,
        receipt_id=None,
        failure_kind=f"raised_{type(error).__name__}",
    )


def _report_line(name: str, send: Callable[[], Delivery]) -> bool:
    """Run *send*, print ``NOTIFY_TEST_<NAME>=`` with its receipt or failure kind; True if delivered."""
    try:
        delivery = send()
    except Exception as error:  # noqa: BLE001 - the other TEST messages still go out
        delivery = _raised(error)
    if delivery.delivered:
        note = (
            "" if delivery.logged else " (receipt not written to the notification log)"
        )
        print(f"NOTIFY_TEST_{name}= delivered {delivery.receipt_id}{note}")
        return True
    print(f"NOTIFY_TEST_{name}= failed {delivery.failure_kind}")
    return False


def main(argv: list[str] | None = None) -> int:
    """Send the three TEST messages; 0 when all three were delivered, else 1."""
    build_parser().parse_args(argv)
    started = datetime.now(UTC)
    run_date_et: date = started.astimezone(ET).date()
    try:
        secrets = LOAD_SECRETS()
    except Exception as error:  # noqa: BLE001 - the class name only: a message can carry a value
        print(
            f"NOTIFY_TEST= refused: the notification secrets could not be read "
            f"({type(error).__name__})"
        )
        return 1
    context = SendContext(SOURCE, run_date_et, secrets)
    report = RunReport(
        run_date_et=run_date_et,
        start=started,
        lock=slate_lock(run_date_et),
        status=STATUS_SUCCESS,
    )

    delivered = [_report_line("FAILED", lambda: send_push(failed_test_push(), context))]
    view: _CacheView | None = None
    build_error: str | None = None
    try:
        view = _read_cache(report)
    except Exception as error:  # noqa: BLE001 - the bets push still proves the phone channel
        build_error = f"build_{type(error).__name__}"
    delivered.append(
        _report_line(
            "BETS",
            lambda: send_push(
                bets_test_push(
                    view.slate if view else None, view.bets if view else None
                ),
                context,
            ),
        )
    )
    if view is None:
        print(f"NOTIFY_TEST_DIGEST= failed {build_error}")
        delivered.append(False)
    else:
        inputs = view.digest_inputs
        delivered.append(
            _report_line(
                "DIGEST", lambda: send_email(digest_test_email(inputs), context)
            )
        )
    return 0 if all(delivered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
