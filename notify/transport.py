"""The one delivery loop every provider call goes through: bounded, classified retries (D-10, D-12).

WHY A RECEIPT AND NOT A STATUS CODE
-----------------------------------
A message counts as delivered only when the channel's own check finds the provider's receipt (an ntfy
or Resend message id; healthchecks.io's ``OK`` body). A 2xx answer without one is a failure: the
provider did not say it accepted the message.

RETRIES (D-12)
--------------
At most :data:`MAX_ATTEMPTS` attempts per message, in one client. Only TRANSIENT failures are
retried -- a transport error (network, timeout), HTTP 429 or any 5xx -- after about 10 s and then
about 30 s. A PERMANENT failure (any other 4xx, a missing secret, a success with no receipt) fails at
once with ``attempts=1``. The worst case, three 15 s timeouts plus 40 s of waiting, stays well inside
the watcher's 18:30 ET grace.

NO SECRET THROUGH THE HTTP STACK (D-10)
---------------------------------------
* A failure is recorded as the exception CLASS name (``transport_ReadTimeout``) or ``http_<status>``
  -- never ``str(error)``, which can carry the request URL, and the healthchecks ping URL embeds the
  check uuid.
* httpx logs every request line at INFO, URL included, so the httpx and httpcore loggers are held at
  WARNING from the moment this module is imported (the daily run never calls ``setup_logging``).
* A response body is never read, logged or returned beyond what the channel's check parses: ntfy's
  echoes the topic.

SEAMS
-----
:data:`CLIENT_FACTORY` and :data:`SLEEP` are module-level names read at call time, so tests inject an
``httpx.MockTransport`` client and a recording sleeper; nothing in a test leaves the machine or sleeps.
Only Exception-family httpx types are caught, so the test suite's network guard (``pytest.fail``, a
BaseException) always propagates.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx

__all__ = [
    "CLIENT_FACTORY",
    "MAX_ATTEMPTS",
    "OUTCOME_DELIVERED",
    "OUTCOME_FAILED",
    "REQUEST_TIMEOUT_SECONDS",
    "RETRY_DELAYS_SECONDS",
    "SLEEP",
    "USER_AGENT",
    "Delivery",
    "Receipt",
    "deliver",
    "missing_secret_delivery",
    "scrub_text",
]

REQUEST_TIMEOUT_SECONDS = 15.0
# The wait BEFORE attempt 2 and attempt 3; there is never a wait after the last attempt.
RETRY_DELAYS_SECONDS: tuple[float, ...] = (10.0, 30.0)
MAX_ATTEMPTS = 3
USER_AGENT = "nfl-predict-notify/1"

OUTCOME_DELIVERED = "delivered"
OUTCOME_FAILED = "failed"

_REDACTED = "<redacted>"

# httpx logs each request line -- the ping URL with its uuid included -- at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass(frozen=True)
class Delivery:
    """The result of one message: what reached the owner, and whether it is on record.

    Attributes:
        outcome: ``delivered`` or ``failed``.
        attempts: Attempts made (a missing secret counts as one, with no request).
        receipt_id: The provider's message id when delivered, else None.
        failure_kind: ``missing_secret``, ``no_receipt``, ``invalid_url``, ``http_<status>`` or
            ``transport_<ExceptionClass>`` when failed, else None.
        logged: False when the message's notification-log record was NOT durably written. A
            delivered message stays delivered (it did reach the owner), but its receipt is not
            on disk, which is a distinct state from a fully recorded delivery.
    """

    outcome: str
    attempts: int
    receipt_id: str | None
    failure_kind: str | None
    logged: bool = True

    @property
    def delivered(self) -> bool:
        """True when the provider returned a receipt."""
        return self.outcome == OUTCOME_DELIVERED


@dataclass(frozen=True)
class Receipt:
    """What one attempt's response check found.

    Attributes:
        ok: True when the provider's receipt was found.
        receipt_id: The receipt (message id) when ok.
        failure_kind: Why it was not ok.
        transient: True when another attempt may succeed.
    """

    ok: bool
    receipt_id: str | None = None
    failure_kind: str | None = None
    transient: bool = False


def _default_client() -> httpx.Client:
    """The production client: a 15 s timeout, our User-Agent, TLS verification left on."""
    return httpx.Client(
        timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
    )


# Seams: tests replace these so no request leaves the machine and no retry sleeps.
CLIENT_FACTORY: Callable[[], httpx.Client] = _default_client
SLEEP: Callable[[float], None] = time.sleep


def _is_transient_status(status: int, transient_status: frozenset[int]) -> bool:
    return status == 429 or 500 <= status <= 599 or status in transient_status


def _attempt(
    client: httpx.Client,
    send_once: Callable[[httpx.Client], httpx.Response],
    check: Callable[[httpx.Response], Receipt],
    transient_status: frozenset[int],
) -> Receipt:
    """One attempt, classified. The class name only: an exception message can carry the URL."""
    try:
        response = send_once(client)
    except httpx.TransportError as error:
        return Receipt(
            ok=False, failure_kind=f"transport_{type(error).__name__}", transient=True
        )
    except httpx.InvalidURL:
        return Receipt(ok=False, failure_kind="invalid_url", transient=False)
    status = response.status_code
    if _is_transient_status(status, transient_status):
        return Receipt(ok=False, failure_kind=f"http_{status}", transient=True)
    if not response.is_success:
        return Receipt(ok=False, failure_kind=f"http_{status}", transient=False)
    return check(response)


def deliver(
    send_once: Callable[[httpx.Client], httpx.Response],
    check: Callable[[httpx.Response], Receipt],
    *,
    transient_status: frozenset[int] = frozenset(),
) -> Delivery:
    """Send one message with up to :data:`MAX_ATTEMPTS` classified attempts in one client.

    Args:
        send_once: Makes ONE request with the client it is given and returns the response.
        check: Reads a 2xx response and returns the provider's receipt (or why there is none).
        transient_status: Extra statuses this channel retries (beyond 429 and 5xx).

    Returns:
        ``delivered`` with the receipt id, or ``failed`` with the last attempt's failure kind.
    """
    receipt = Receipt(ok=False, failure_kind="not_attempted")
    attempts = 0
    with CLIENT_FACTORY() as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            receipt = _attempt(client, send_once, check, transient_status)
            if receipt.ok:
                return Delivery(
                    outcome=OUTCOME_DELIVERED,
                    attempts=attempt,
                    receipt_id=receipt.receipt_id,
                    failure_kind=None,
                )
            if not receipt.transient or attempt == MAX_ATTEMPTS:
                break
            SLEEP(RETRY_DELAYS_SECONDS[attempt - 1])
    return Delivery(
        outcome=OUTCOME_FAILED,
        attempts=attempts,
        receipt_id=None,
        failure_kind=receipt.failure_kind,
    )


def missing_secret_delivery() -> Delivery:
    """A permanent failure with no request made: the channel's secret is not configured."""
    return Delivery(
        outcome=OUTCOME_FAILED,
        attempts=1,
        receipt_id=None,
        failure_kind="missing_secret",
    )


def scrub_text(text: str, values: Iterable[str]) -> str:
    """*text* with every value in *values* replaced by ``<redacted>``.

    The LONGEST value is replaced first, so a secret that contains another (the ping URL contains
    its uuid) is never left partly visible.
    """
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        text = text.replace(value, _REDACTED)
    return text
