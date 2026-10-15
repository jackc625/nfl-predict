"""The closed event-class vocabulary and the push router (D-11; SPEC must-NOTs).

WHY CLOSED
----------
NOTIF-01 allows exactly two kinds of phone push -- a FAILED run and new BETS -- plus the TEST message
the owner sends on purpose. :func:`send_push` refuses every other class with a ValueError BEFORE any
request is made, so a new caller cannot quietly invent a third kind of push. The email and watcher
classes are declared here too, so the whole vocabulary lives in one place, and each has its own
router below (:func:`send_email`, the watcher sends).

NO SUPPRESSION
--------------
There is no cooldown, dedupe window or rate limit anywhere in this path. Two identical failures sent
back to back are two deliveries and two log records: a suppressed failure is a silent one.

ONE RECORD PER MESSAGE
----------------------
Every send goes through :func:`_send_and_record`, which delivers and then appends exactly one
notification-log record. A message the provider accepted whose record could NOT be written (the lock
wait timed out, any other append error, or the sink declined the write) is still returned as
delivered -- it did reach the owner -- but with ``logged=False``, a visibly different state from a
receipt that is on disk.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from forward_ledger.run_log import current_run_id
from notify import channels
from notify.log import NOTIFICATION_SOURCES, append_notification
from notify.secrets import NotifySecrets
from notify.transport import OUTCOME_FAILED, Delivery

__all__ = [
    "CHECKIN_BODY_LIMIT_BYTES",
    "EMAIL_EVENT_CLASSES",
    "OUTCOME_WITHHELD",
    "PRIORITY_NORMAL",
    "PRIORITY_TOP",
    "PUSH_BODY_LIMIT_BYTES",
    "PUSH_EVENT_CLASSES",
    "PUSH_TITLE_LIMIT_CHARS",
    "WATCHER_EVENT_CLASSES",
    "EmailMessage",
    "PushMessage",
    "SendContext",
    "ascii_text",
    "fit_lines",
    "record_unbuilt_email",
    "record_withheld_checkin",
    "send_checkin",
    "send_email",
    "send_pause",
    "send_push",
]

PUSH_EVENT_CLASSES: tuple[str, ...] = ("failed", "bets", "test")
EMAIL_EVENT_CLASSES: tuple[str, ...] = ("digest", "test")
# Log-only: the watcher check-in and the off-season pause never go to ntfy.
WATCHER_EVENT_CLASSES: tuple[str, ...] = ("checkin", "pause")
# A check-in deliberately withheld on the D-16 test day (``notify.log.NOTIFICATION_OUTCOMES``).
OUTCOME_WITHHELD = "withheld"

# ntfy priorities: 5 is max (urgent), 3 is the default.
PRIORITY_TOP = 5
PRIORITY_NORMAL = 3

# ntfy turns a body over 4,096 bytes into a hosted attachment; the SPEC caps it at 4,000.
PUSH_BODY_LIMIT_BYTES = 4000
# ntfy refuses a title over 1 KB; a lock-screen title needs far less.
PUSH_TITLE_LIMIT_CHARS = 200
# The check-in body is a short run summary (D-15); healthchecks.io stores far more than this.
CHECKIN_BODY_LIMIT_BYTES = 1000


@dataclass(frozen=True)
class PushMessage:
    """One phone push, before it is sanitized and sized.

    Attributes:
        event_class: One of :data:`PUSH_EVENT_CLASSES`.
        title: The lock-screen line (D-06: what happened + the count).
        header_lines: Body lines that are always kept.
        item_lines: Body lines (one per game or bet) that may be cut to fit the byte limit.
        priority: :data:`PRIORITY_TOP` or :data:`PRIORITY_NORMAL`.
    """

    event_class: str
    title: str
    header_lines: tuple[str, ...]
    item_lines: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class EmailMessage:
    """One email (the Saturday digest or the TEST email), sent by :func:`send_email`.

    Attributes:
        event_class: One of :data:`EMAIL_EVENT_CLASSES`.
        subject: The subject line (D-07).
        html: The HTML part.
        text: The plain-text alternative.
        idempotency_key: Resend's per-message key, so a retry after a timeout cannot double-send.
    """

    event_class: str
    subject: str
    html: str
    text: str
    idempotency_key: str


@dataclass(frozen=True)
class SendContext:
    """Who is sending, for which run date, with which secrets.

    Attributes:
        source: One of ``notify.log.NOTIFICATION_SOURCES``.
        run_date_et: The run's ET date, or None (a TEST send outside any run).
        secrets: The notification secrets.

    Raises:
        ValueError: *source* is outside ``NOTIFICATION_SOURCES``.
    """

    source: str
    run_date_et: date | None
    secrets: NotifySecrets

    def __post_init__(self) -> None:
        if self.source not in NOTIFICATION_SOURCES:
            msg = f"send source {self.source!r} is outside {NOTIFICATION_SOURCES}"
            raise ValueError(msg)


def ascii_text(text: str) -> str:
    """*text* as printable ASCII: non-ASCII becomes ``?``; control characters except newline drop."""
    kept: list[str] = []
    for character in text:
        code = ord(character)
        if character == "\n":
            kept.append(character)
        elif code > 127:
            kept.append("?")
        elif code < 32 or code == 127:
            continue
        else:
            kept.append(character)
    return "".join(kept)


def fit_lines(
    header_lines: Sequence[str],
    item_lines: Sequence[str],
    *,
    limit: int = PUSH_BODY_LIMIT_BYTES,
) -> str:
    """Join the lines as ASCII text of at most *limit* bytes.

    When everything does not fit, the header is kept and as many item lines as fit, followed by
    ``... and N more`` (N = item lines left out). A header alone over the limit is cut at it.
    """
    header = [ascii_text(line) for line in header_lines]
    items = [ascii_text(line) for line in item_lines]
    # Every character is ASCII from here on, so a character count IS the UTF-8 byte count.
    full = "\n".join([*header, *items])
    if len(full) <= limit:
        return full
    for keep in range(len(items) - 1, -1, -1):
        candidate = "\n".join(
            [*header, *items[:keep], f"... and {len(items) - keep} more"]
        )
        if len(candidate) <= limit:
            return candidate
    return "\n".join(header)[:limit]


def _send_and_record(
    send: Callable[[], Delivery],
    *,
    event_class: str,
    priority: int | None,
    context: SendContext,
) -> Delivery:
    """Deliver once through *send*, then append exactly one notification-log record for it.

    A record that is not durably written leaves the outcome, attempts and receipt unchanged and
    returns the delivery with ``logged=False``. The append error is printed by class name only.
    """
    delivery = send()
    record = {
        "run_id": current_run_id(),
        "run_date_et": (
            context.run_date_et.isoformat() if context.run_date_et is not None else None
        ),
        "source": context.source,
        "event_class": event_class,
        "priority": priority,
        "attempts": delivery.attempts,
        "outcome": delivery.outcome,
        "receipt_id": delivery.receipt_id,
        "failure_kind": delivery.failure_kind,
        "sent_at": datetime.now(tz=UTC).isoformat(),
    }
    try:
        written = append_notification(record)
    except Exception as error:  # noqa: BLE001 - a log failure must never undo a delivery
        print(f"NOTIFICATION_LOG= failed {type(error).__name__}")  # noqa: T201
        written = False
    if not written:
        return dataclasses.replace(delivery, logged=False)
    return delivery


def send_push(message: PushMessage, context: SendContext) -> Delivery:
    """Send one phone push and record it once.

    Raises:
        ValueError: the event class is outside :data:`PUSH_EVENT_CLASSES` (before any request).
    """
    if message.event_class not in PUSH_EVENT_CLASSES:
        msg = (
            f"push event class {message.event_class!r} is outside the closed vocabulary "
            f"{PUSH_EVENT_CLASSES}"
        )
        raise ValueError(msg)
    title = ascii_text(message.title)[:PUSH_TITLE_LIMIT_CHARS]
    body = fit_lines(message.header_lines, message.item_lines)
    return _send_and_record(
        lambda: channels.ntfy_publish(
            context.secrets, title=title, message=body, priority=message.priority
        ),
        event_class=message.event_class,
        priority=message.priority,
        context=context,
    )


def send_email(message: EmailMessage, context: SendContext) -> Delivery:
    """Send one email through Resend and record it once (priority None; never the recipient).

    Raises:
        ValueError: the event class is outside :data:`EMAIL_EVENT_CLASSES` (before any request).
    """
    if message.event_class not in EMAIL_EVENT_CLASSES:
        msg = (
            f"email event class {message.event_class!r} is outside the closed vocabulary "
            f"{EMAIL_EVENT_CLASSES}"
        )
        raise ValueError(msg)
    subject = ascii_text(message.subject)
    return _send_and_record(
        lambda: channels.resend_send(
            context.secrets,
            subject=subject,
            html=message.html,
            text=message.text,
            idempotency_key=message.idempotency_key,
        ),
        event_class=message.event_class,
        priority=None,
        context=context,
    )


def send_checkin(*, body: str, fail: bool, context: SendContext) -> Delivery:
    """Check in with the outside watcher (``/fail`` when *fail*) and record it once.

    *body* is the ASCII run summary, cut to :data:`CHECKIN_BODY_LIMIT_BYTES`. Plan 35-13 owns when
    this runs; it has exactly one production call site.
    """
    text = ascii_text(body)[:CHECKIN_BODY_LIMIT_BYTES]
    return _send_and_record(
        lambda: channels.hc_ping(context.secrets, body=text, fail=fail),
        event_class="checkin",
        priority=None,
        context=context,
    )


def record_withheld_checkin(context: SendContext) -> None:
    """Record the one deliberately withheld check-in (the D-16 test day): no request is made.

    The record (outcome ``withheld``, attempts 0) keeps the log the one place to read what
    reached the watcher.
    """
    _send_and_record(
        lambda: Delivery(
            outcome=OUTCOME_WITHHELD, attempts=0, receipt_id=None, failure_kind=None
        ),
        event_class="checkin",
        priority=None,
        context=context,
    )


def record_unbuilt_email(failure_kind: str, context: SendContext) -> Delivery:
    """Record a weekly email that could not be BUILT: no request is made (SPEC R4, R8).

    The email twin of :func:`record_withheld_checkin`: one ``digest`` record, outcome ``failed``,
    attempts 0, no receipt, the given *failure_kind* (``build_<ExceptionClass>``). The returned
    failed delivery is what the caller escalates, so a due digest never disappears silently.
    """
    return _send_and_record(
        lambda: Delivery(
            outcome=OUTCOME_FAILED,
            attempts=0,
            receipt_id=None,
            failure_kind=failure_kind,
        ),
        event_class="digest",
        priority=None,
        context=context,
    )


def send_pause(context: SendContext) -> Delivery:
    """Pause the watcher's check (out of season, D-01) and record it once."""
    return _send_and_record(
        lambda: channels.hc_pause(context.secrets),
        event_class="pause",
        priority=None,
        context=context,
    )
