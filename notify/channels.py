"""One function per provider call; each goes through ``notify.transport.deliver`` (D-10).

ntfy only in this plan; Plan 35-05 adds the Resend email and the healthchecks.io check-in, pause and
management calls here.

ntfy (https://docs.ntfy.sh/publish/): the message is published as JSON to the server ROOT, so the
topic travels in the body and never in a URL. ``priority`` is an integer (5 = max, 3 = default).
There is NO ``tags`` key: a tag matching an emoji short code renders as an emoji. The response is the
message JSON -- its ``id`` is the receipt, and it ALSO echoes the topic, so only ``id`` is parsed and
the body is never logged or returned.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import httpx

from notify.secrets import NTFY_SERVER_URL, NotifySecrets
from notify.transport import Delivery, Receipt, deliver, missing_secret_delivery

__all__ = ["ntfy_publish"]

_NO_RECEIPT = Receipt(ok=False, failure_kind="no_receipt", transient=False)


def _ntfy_receipt(response: httpx.Response) -> Receipt:
    """The message id from a 2xx ntfy response; anything else is a permanent ``no_receipt``."""
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

    return deliver(send_once, _ntfy_receipt)
