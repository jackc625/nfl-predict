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

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import httpx

from notify.secrets import (
    NTFY_SERVER_URL,
    RESEND_EMAILS_URL,
    RESEND_SENDER,
    NotifySecrets,
)
from notify.transport import (
    USER_AGENT,
    Delivery,
    Receipt,
    deliver,
    missing_secret_delivery,
)

__all__ = ["ntfy_publish", "resend_send"]

_NO_RECEIPT = Receipt(ok=False, failure_kind="no_receipt", transient=False)
# A retry colliding with an in-flight request under the same Idempotency-Key.
_RESEND_TRANSIENT_STATUS = frozenset({409})


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
