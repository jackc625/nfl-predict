"""What the daily run sends when it ends: failed pushes, bets push, digest, then the check-in (D-13).

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
4. The watcher check-in, LAST, after every message (:func:`_watcher_step`, Plan 35-13): one ping
   whose body is :func:`checkin_summary` (D-15). It goes to ``<ping_url>/fail`` when any message
   stayed undelivered or a delivered message's receipt never reached the notification log, so
   the watcher alerts through its own channels (SPEC R8). This is the ONE production call of
   :func:`notify.events.send_checkin`.

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

from api.cache import ARM_LIVE, BET_STATUS_LIVE
from notify import secrets as notify_secrets
from notify.digest import send_digest_if_due
from notify.events import PushMessage, SendContext, send_checkin, send_push
from notify.messages import bets_message, email_failed_message, failed_messages
from notify.report import RunReport
from notify.secrets import NotifySecrets
from notify.transport import OUTCOME_FAILED, Delivery

__all__ = ["LOAD_SECRETS", "EndOfRunOutcome", "checkin_summary", "notify_end_of_run"]

# The season-window values that check in (Plan 35-13 Task 2 adds the "out" pause); "unknown"
# checks in because a resumed watcher can only cause a false alarm, never silence (D-17).
_CHECK_IN_WINDOWS = frozenset({"in", "unknown"})

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
    """Send the failed pushes, the bets push, the digest, then check in. Never raises ``Exception``.

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

    try:
        _watcher_step(report, outcome, context)
    except Exception as error:  # noqa: BLE001 - the missing check-in is itself the alert
        print(f"WATCHER_CHECKIN= failed {type(error).__name__}")  # noqa: T201

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


def _step_outcome(outcome: EndOfRunOutcome, step: str) -> str:
    """The outcome of *step*'s last delivery (``delivered`` / ``failed``), or ``none``."""
    found = [delivery.outcome for name, delivery in outcome.deliveries if name == step]
    return found[-1] if found else "none"


def checkin_summary(report: RunReport, outcome: EndOfRunOutcome) -> str:
    """The D-15 ping body: one ASCII line of stored facts, never a secret or an error text."""
    games = (
        "unknown"
        if report.tomorrow_game_ids is None
        else str(len(report.tomorrow_game_ids))
    )
    live_bets = sum(
        1
        for row in report.appended_rows
        if row.get("arm") == ARM_LIVE and row.get("status") == BET_STATUS_LIVE
    )
    failed = [delivery for step, delivery in outcome.deliveries if step == "failed"]
    delivered_failed = sum(1 for delivery in failed if delivery.delivered)
    unlogged = sum(
        1
        for _step, delivery in outcome.deliveries
        if delivery.delivered and not delivery.logged
    )
    return (
        f"NFL Predict daily run {report.run_date_et.isoformat()}: "
        f"outcome={report.status or 'unknown'}; games tomorrow={games}; "
        f"bets appended={live_bets}; failed pushes={delivered_failed}/{len(failed)}; "
        f"bets push={_step_outcome(outcome, 'bets')}; "
        f"digest={_step_outcome(outcome, 'digest')}; unlogged receipts={unlogged}"
    )


def _watcher_step(
    report: RunReport, outcome: EndOfRunOutcome, context: SendContext
) -> None:
    """Step 4, LAST: check in with the outside watcher (D-13, D-15; SPEC R7, R8).

    The check-in is the failure signal (``/fail``) when any message stayed undelivered -- a step
    that raised counts -- or was delivered without its receipt reaching the log (T-35-47).
    """
    if report.season_window not in _CHECK_IN_WINDOWS:
        return
    fail = outcome.any_undelivered or outcome.any_unlogged
    delivery = send_checkin(
        body=checkin_summary(report, outcome), fail=fail, context=context
    )
    if not delivery.delivered:
        state = f"failed {delivery.failure_kind}"
    elif fail:
        state = "delivered as failure"
    else:
        state = "delivered"
    print(f"WATCHER_CHECKIN= {state}")  # noqa: T201
