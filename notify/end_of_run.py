"""What the daily run sends when it ends: the failed pushes, then the bets push (D-13).

WHEN THIS RUNS
--------------
The daily run calls :func:`notify_end_of_run` once, at the very end -- after the bet-list and ledger
writes and after the ledger sync -- on both the returning and the raising path, and on the
lock-passed early return. It never runs on a dry run, a refused ``--date`` or a Ctrl+C
(``scripts/daily_lock_pipeline.py`` owns those rules). Sending therefore cannot change a decision:
every decision is already written when the first message leaves.

ORDER (D-13)
------------
1. Every failed push the run owes (:func:`notify.messages.failed_messages`), top priority first.
2. The one bets push for the live bets the run newly appended, or none.
3. The weekly digest email, on its digest run only (:func:`notify.digest.send_digest_if_due`). An
   email that is not delivered sends one normal-priority ``FAILED: weekly email`` push naming the
   failure kind (SPEC R8), and both count toward :attr:`EndOfRunOutcome.any_undelivered`.

Plan 35-13 adds the watcher check-in after these three.

NEVER RAISES ON A SEND
----------------------
Each step is wrapped so a failing step prints its exception CLASS name, counts as undelivered and
lets the next step run. Only ``Exception`` is caught: a Ctrl+C, and the test suite's network guard
(``pytest.fail``), always propagate. Nothing here suppresses or rate-limits a message.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from notify import secrets as notify_secrets
from notify.digest import send_digest_if_due
from notify.events import PushMessage, SendContext, send_push
from notify.messages import bets_message, email_failed_message, failed_messages
from notify.report import RunReport
from notify.secrets import NotifySecrets
from notify.transport import OUTCOME_FAILED, Delivery

__all__ = ["LOAD_SECRETS", "EndOfRunOutcome", "notify_end_of_run"]

# Seam, read at call time: tests hand in dummy secrets so the owner's .env is never read.
LOAD_SECRETS = notify_secrets.load_secrets


@dataclass
class EndOfRunOutcome:
    """Every message the end of one run tried to send, in send order.

    Attributes:
        deliveries: ``(step, Delivery)`` pairs; ``step`` is ``failed``, ``bets`` or ``digest``. A
            step that raised is one failed delivery with no attempt, its failure kind
            ``raised_<Class>``.
    """

    deliveries: list[tuple[str, Delivery]] = field(default_factory=list)

    @property
    def any_undelivered(self) -> bool:
        """True when any message did not reach the owner."""
        return any(not delivery.delivered for _step, delivery in self.deliveries)

    @property
    def any_unlogged(self) -> bool:
        """True when a message reached the owner but its receipt is not in the log (Plan 35-01)."""
        return any(
            delivery.delivered and not delivery.logged
            for _step, delivery in self.deliveries
        )


def _raised(error: Exception) -> Delivery:
    """The delivery recorded for a step that raised: failed, no attempt, the class name only."""
    return Delivery(
        outcome=OUTCOME_FAILED,
        attempts=0,
        receipt_id=None,
        failure_kind=f"raised_{type(error).__name__}",
    )


def _send_one(
    step: str, message: PushMessage, context: SendContext, outcome: EndOfRunOutcome
) -> None:
    """Send *message*, record its delivery under *step* and print its console line."""
    delivery = send_push(message, context)
    outcome.deliveries.append((step, delivery))
    state = (
        f"delivered {delivery.receipt_id}"
        if delivery.delivered
        else f"failed {delivery.failure_kind}"
    )
    print(f"NOTIFY_{step.upper()}= {state} {message.title}")  # noqa: T201


def notify_end_of_run(
    report: RunReport,
    *,
    secrets: NotifySecrets | None = None,
    now: datetime | None = None,
) -> EndOfRunOutcome:
    """Send the run's failed pushes, its bets push, then the digest. Never raises an ``Exception``.

    Args:
        report: What the run did.
        secrets: The notification secrets (default: :data:`LOAD_SECRETS`, read at call time).
        now: The instant the next-action line is judged at (default: now, UTC).

    Returns:
        Every delivery, in send order.
    """
    outcome = EndOfRunOutcome()
    try:
        loaded = secrets if secrets is not None else LOAD_SECRETS()
    except Exception as error:  # noqa: BLE001 - an unreadable .env must not fail the run
        print(f"NOTIFY_FAILED= failed {type(error).__name__}")  # noqa: T201
        outcome.deliveries.append(("failed", _raised(error)))
        return outcome
    instant = now if now is not None else datetime.now(UTC)
    context = SendContext("daily_run", report.run_date_et, loaded)
    failed_titles: list[str] = []

    try:
        for message in failed_messages(
            report, now=instant, secret_values=loaded.secret_values()
        ):
            failed_titles.append(message.title)
            _send_one("failed", message, context, outcome)
    except Exception as error:  # noqa: BLE001 - the bets push still goes out
        print(f"NOTIFY_FAILED= failed {type(error).__name__}")  # noqa: T201
        outcome.deliveries.append(("failed", _raised(error)))

    try:
        bets = bets_message(report)
        if bets is None:
            print("NOTIFY_BETS= none")  # noqa: T201
        else:
            _send_one("bets", bets, context, outcome)
    except Exception as error:  # noqa: BLE001 - a later step still runs
        print(f"NOTIFY_BETS= failed {type(error).__name__}")  # noqa: T201
        outcome.deliveries.append(("bets", _raised(error)))

    try:
        _send_digest(report, context, failed_titles, instant, outcome)
    except Exception as error:  # noqa: BLE001 - a later step still runs
        print(f"NOTIFY_DIGEST= failed {type(error).__name__}")  # noqa: T201
        outcome.deliveries.append(("digest", _raised(error)))

    return outcome


def _send_digest(
    report: RunReport,
    context: SendContext,
    failed_titles: list[str],
    now: datetime,
    outcome: EndOfRunOutcome,
) -> None:
    """Step 3: the digest on its digest run; an undelivered one also sends a failed push (R8)."""
    delivery = send_digest_if_due(
        report, context=context, failed_titles=tuple(failed_titles), now=now
    )
    if delivery is None:
        print("NOTIFY_DIGEST= not a digest day")  # noqa: T201
        return
    outcome.deliveries.append(("digest", delivery))
    if delivery.delivered:
        print(f"NOTIFY_DIGEST= delivered {delivery.receipt_id}")  # noqa: T201
        return
    failure_kind = delivery.failure_kind or "unknown"
    print(f"NOTIFY_DIGEST= failed {failure_kind}")  # noqa: T201
    _send_one(
        "failed",
        email_failed_message(report.run_date_et, failure_kind),
        context,
        outcome,
    )
