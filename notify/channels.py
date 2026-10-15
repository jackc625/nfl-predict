"""One function per provider call; each goes through ``notify.transport.deliver`` (D-10).

ntfy (https://docs.ntfy.sh/publish/): the message is published as JSON to the server ROOT, so the
topic travels in the body and never in a URL. ``priority`` is an integer (5 = max, 3 = default).
There is NO ``tags`` key: a tag matching an emoji short code renders as an emoji. The response is the
message JSON -- its ``id`` is the receipt, and it ALSO echoes the topic, so only ``id`` is parsed and
the body is never logged or returned.

Resend (https://resend.com/docs/api-reference/emails/send-email): a raw httpx POST, no SDK (D-10).
Resend answers 403 to a request without a User-Agent, so one is always sent explicitly. Every email
carries an ``Idempotency-Key`` so a retry after a timeout cannot double-send; a retry that collides
with the still-running first request is answered 409, which is therefore transient (Pitfall 13).
Only the response's ``id`` is parsed.

healthchecks.io (https://healthchecks.io/docs/http_api/, /docs/api/):

* A ping answers HTTP 200 even when it did nothing: ``OK (not found)`` for an unknown uuid and
  ``OK (rate limited)`` for a throttled ping (Pitfall 2). A ping is delivered only on 200 with the
  body exactly ``OK``; it has no message id, so the ``OK`` itself is the receipt.
* The pause and the management calls use the project's READ-WRITE key in ``X-Api-Key``. The check
  uuid is derived from the ping URL, which must be the uuid form (D-01).
* A read-write management response carries the uuid and the ping, update, pause and resume URLs
  (Pitfall 3). Each response is parsed HERE into :class:`WatcherChannel` / :class:`WatcherCheck`,
  which hold only the read-back fields, and the raw JSON never leaves this module.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from notify import transport
from notify.secrets import (
    HEALTHCHECKS_API_URL,
    NTFY_SERVER_URL,
    RESEND_EMAILS_URL,
    RESEND_SENDER,
    NotifySecrets,
)
from notify.transport import (
    OUTCOME_FAILED,
    USER_AGENT,
    Delivery,
    Receipt,
    deliver,
    missing_secret_delivery,
)

__all__ = [
    "WatcherApiError",
    "WatcherChannel",
    "WatcherCheck",
    "check_uuid",
    "hc_get_check",
    "hc_list_channels",
    "hc_pause",
    "hc_ping",
    "hc_update_check",
    "ntfy_publish",
    "resend_send",
]

_NO_RECEIPT = Receipt(ok=False, failure_kind="no_receipt", transient=False)
# A retry colliding with an in-flight request under the same Idempotency-Key.
_RESEND_TRANSIENT_STATUS = frozenset({409})

_PING_URL_SHAPE = re.compile(
    r"^https://hc-ping\.com/"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_PING_OK = "OK"
_PING_RATE_LIMITED = "OK (rate limited)"
_PING_NOT_FOUND = "OK (not found)"


class WatcherApiError(Exception):
    """The healthchecks.io Management API answered non-2xx. Carries the HTTP status only.

    Attributes:
        status: The HTTP status code.
    """

    def __init__(self, status: int) -> None:
        super().__init__(f"the healthchecks.io Management API answered HTTP {status}")
        self.status = status


@dataclass(frozen=True)
class WatcherChannel:
    """One healthchecks.io integration, as the setup command needs it (no URL, no secret).

    Attributes:
        id: The integration's id (what a check's ``channels`` lists).
        name: The owner-given name the setup command matches on.
        kind: The integration kind (``ntfy``, ``email``, ...).
    """

    id: str
    name: str
    kind: str


@dataclass(frozen=True)
class WatcherCheck:
    """The read-back fields of the check -- and only those (no uuid, no URL).

    Attributes:
        schedule: The cron expression, or None when absent.
        tz: The cron time zone, or None when absent.
        grace: The grace period in seconds, or None when absent.
        manual_resume: Whether a paused check stays paused on a ping, or None when absent.
        channel_ids: The attached integration ids (blanks dropped).
    """

    schedule: str | None
    tz: str | None
    grace: int | None
    manual_resume: bool | None
    channel_ids: frozenset[str]


def _message_id_receipt(response: httpx.Response) -> Receipt:
    """The message id from a 2xx JSON response; anything else is a permanent ``no_receipt``.

    ntfy and Resend both answer with a JSON object whose ``id`` is the message id.
    """
    try:
        body = response.json()
    except ValueError:
        return _NO_RECEIPT
    message_id = body.get("id") if isinstance(body, dict) else None
    if isinstance(message_id, str) and message_id.strip():
        return Receipt(ok=True, receipt_id=message_id)
    return _NO_RECEIPT


def _ping_receipt(response: httpx.Response) -> Receipt:
    """Delivered only on HTTP 200 with the body ``OK`` (stripped); receipt id None."""
    if response.status_code != 200:
        return Receipt(ok=False, failure_kind=f"http_{response.status_code}")
    answer = response.text.strip()
    if answer == _PING_OK:
        return Receipt(ok=True)
    if answer == _PING_RATE_LIMITED:
        return Receipt(ok=False, failure_kind="rate_limited", transient=True)
    if answer == _PING_NOT_FOUND:
        return Receipt(ok=False, failure_kind="not_found")
    return Receipt(ok=False, failure_kind="unexpected_body")


def _ok_on_200(response: httpx.Response) -> Receipt:
    """A management action is done on HTTP 200; its body (the check, URLs included) is not read."""
    if response.status_code == 200:
        return Receipt(ok=True)
    return Receipt(ok=False, failure_kind=f"http_{response.status_code}")


def ntfy_publish(
    secrets: NotifySecrets, *, title: str, message: str, priority: int
) -> Delivery:
    """Publish one push to the owner's ntfy topic.

    An unset topic is a recorded failure (``missing_secret``, no request made), never a skip.
    """
    if secrets.ntfy_topic is None:
        return missing_secret_delivery()
    payload = {
        "topic": secrets.ntfy_topic.get_secret_value(),
        "title": title,
        "message": message,
        "priority": int(priority),
    }

    def send_once(client: httpx.Client) -> httpx.Response:
        return client.post(NTFY_SERVER_URL, json=payload)

    return deliver(send_once, _message_id_receipt)


def resend_send(
    secrets: NotifySecrets,
    *,
    subject: str,
    html: str,
    text: str,
    idempotency_key: str,
) -> Delivery:
    """Send one email to the digest recipient through Resend.

    An unset key or recipient is a recorded failure (``missing_secret``, no request made).
    """
    if secrets.resend_api_key is None or secrets.digest_to_address is None:
        return missing_secret_delivery()
    headers = {
        "Authorization": f"Bearer {secrets.resend_api_key.get_secret_value()}",
        "User-Agent": USER_AGENT,
        "Idempotency-Key": idempotency_key,
    }
    payload = {
        "from": RESEND_SENDER,
        "to": [secrets.digest_to_address.get_secret_value()],
        "subject": subject,
        "html": html,
        "text": text,
    }

    def send_once(client: httpx.Client) -> httpx.Response:
        return client.post(RESEND_EMAILS_URL, headers=headers, json=payload)

    return deliver(
        send_once, _message_id_receipt, transient_status=_RESEND_TRANSIENT_STATUS
    )


def check_uuid(ping_url: str) -> str:
    """The check uuid from a ``https://hc-ping.com/<uuid>`` ping URL (D-01).

    Raises:
        ValueError: the URL is not that shape (a ping-key/slug URL, a trailing slash, another
            host). The message names the expected shape and never echoes the URL.
    """
    match = _PING_URL_SHAPE.fullmatch(ping_url.strip())
    if match is None:
        msg = (
            "the healthchecks.io ping URL is not of the form https://hc-ping.com/<uuid> "
            "(a canonical 36-character uuid); copy the uuid ping URL from the check's page"
        )
        raise ValueError(msg)
    return match.group(1)


def hc_ping(secrets: NotifySecrets, *, body: str, fail: bool) -> Delivery:
    """Check in with the watcher: POST the ping URL (``/fail`` appended when *fail*).

    *body* is sent as ASCII (anything else becomes ``?``). An unset ping URL is a recorded failure.
    """
    if secrets.healthchecks_ping_url is None:
        return missing_secret_delivery()
    url = secrets.healthchecks_ping_url.get_secret_value() + ("/fail" if fail else "")
    content = body.encode("ascii", "replace")

    def send_once(client: httpx.Client) -> httpx.Response:
        return client.post(url, content=content)

    return deliver(send_once, _ping_receipt)


def hc_pause(secrets: NotifySecrets) -> Delivery:
    """Pause the check (out of season), with the read-write key. Idempotent on the provider side.

    An unset ping URL or key is ``missing_secret``; a ping URL that is not the uuid form is a
    recorded ``invalid_url`` failure, so a pause never raises out of the daily run.
    """
    if secrets.healthchecks_ping_url is None or secrets.healthchecks_api_key is None:
        return missing_secret_delivery()
    try:
        uuid = check_uuid(secrets.healthchecks_ping_url.get_secret_value())
    except ValueError:
        return Delivery(
            outcome=OUTCOME_FAILED,
            attempts=1,
            receipt_id=None,
            failure_kind="invalid_url",
        )
    url = f"{HEALTHCHECKS_API_URL}/checks/{uuid}/pause"
    headers = {"X-Api-Key": secrets.healthchecks_api_key.get_secret_value()}

    def send_once(client: httpx.Client) -> httpx.Response:
        return client.post(url, headers=headers)

    return deliver(send_once, _ok_on_200)


def _management_target(secrets: NotifySecrets) -> tuple[str, dict[str, str]]:
    """The check's API URL and the auth header.

    Raises:
        ValueError: the ping URL or the API key is unset (named by setting), or the ping URL is
            not the uuid form.
    """
    if secrets.healthchecks_ping_url is None:
        msg = "HEALTHCHECKS_PING_URL is not set"
        raise ValueError(msg)
    if secrets.healthchecks_api_key is None:
        msg = "HEALTHCHECKS_API_KEY is not set"
        raise ValueError(msg)
    uuid = check_uuid(secrets.healthchecks_ping_url.get_secret_value())
    headers = {"X-Api-Key": secrets.healthchecks_api_key.get_secret_value()}
    return f"{HEALTHCHECKS_API_URL}/checks/{uuid}", headers


def _management_json(
    method: str, url: str, headers: Mapping[str, str], payload: Any = None
) -> Any:
    """One management request (a single attempt). The parsed JSON never leaves this module.

    Raises:
        WatcherApiError: the answer was not 2xx (status only).
    """
    with transport.CLIENT_FACTORY() as client:
        response = client.request(method, url, headers=dict(headers), json=payload)
    if not response.is_success:
        raise WatcherApiError(response.status_code)
    try:
        return response.json()
    except ValueError:
        return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _parse_check(body: Any) -> WatcherCheck:
    """The read-back fields of a check object; everything else (uuid, URLs) is dropped here."""
    fields = body if isinstance(body, dict) else {}
    grace = fields.get("grace")
    manual_resume = fields.get("manual_resume")
    channels = fields.get("channels")
    channel_ids = (
        frozenset(part.strip() for part in channels.split(",") if part.strip())
        if isinstance(channels, str)
        else frozenset()
    )
    return WatcherCheck(
        schedule=_optional_str(fields.get("schedule")),
        tz=_optional_str(fields.get("tz")),
        grace=grace if isinstance(grace, int) and not isinstance(grace, bool) else None,
        manual_resume=manual_resume if isinstance(manual_resume, bool) else None,
        channel_ids=channel_ids,
    )


def hc_list_channels(secrets: NotifySecrets) -> tuple[WatcherChannel, ...]:
    """The project's integrations (``GET /channels/``), as id/name/kind only.

    Raises:
        ValueError: a watcher secret is unset or the ping URL is not the uuid form.
        WatcherApiError: the API answered non-2xx.
    """
    _, headers = _management_target(secrets)
    body = _management_json("GET", f"{HEALTHCHECKS_API_URL}/channels/", headers)
    entries = body.get("channels") if isinstance(body, dict) else None
    return tuple(
        WatcherChannel(
            id=str(entry.get("id", "")),
            name=str(entry.get("name", "")),
            kind=str(entry.get("kind", "")),
        )
        for entry in (entries if isinstance(entries, list) else [])
        if isinstance(entry, dict)
    )


def hc_get_check(secrets: NotifySecrets) -> WatcherCheck:
    """Read the check back (``GET /checks/<uuid>``).

    Raises:
        ValueError: a watcher secret is unset or the ping URL is not the uuid form.
        WatcherApiError: the API answered non-2xx (404: no check with that uuid).
    """
    url, headers = _management_target(secrets)
    return _parse_check(_management_json("GET", url, headers))


def hc_update_check(secrets: NotifySecrets, payload: Mapping[str, Any]) -> WatcherCheck:
    """Update the check (``POST /checks/<uuid>`` with *payload*); omitted fields stay unchanged.

    Raises:
        ValueError: a watcher secret is unset or the ping URL is not the uuid form.
        WatcherApiError: the API answered non-2xx (404: no check with that uuid).
    """
    url, headers = _management_target(secrets)
    return _parse_check(_management_json("POST", url, headers, dict(payload)))
