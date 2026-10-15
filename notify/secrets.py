"""What the sender reads from the settings, and the provider constants that are NOT secrets (D-09).

WHY ONE VALUE
-------------
The five secrets (ntfy topic, Resend key, digest recipient, healthchecks.io ping URL, healthchecks.io
read-write API key) travel together in one frozen :class:`NotifySecrets`, each a pydantic
``SecretStr`` so a repr, a log line or a traceback never prints it. They are read through
``conf.settings`` -- the same ``.env`` path that already serves ``ODDS_API_KEY`` under the scheduled
task's S4U logon -- and nowhere else.

The ntfy server and the Resend sender are code constants: neither is a secret, and keeping them out
of ``.env`` means a mistyped line cannot redirect a message.

NOTIFY_HOSTS is the one list of provider hosts. ``tests/conftest.py`` refuses every one of them (and
their subdomains) for the whole suite, so a test can never reach a real provider.

This module never prints or logs.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import SecretStr

__all__ = [
    "HEALTHCHECKS_API_URL",
    "NOTIFY_HOSTS",
    "NTFY_SERVER_URL",
    "RESEND_EMAILS_URL",
    "RESEND_SENDER",
    "NotifySecrets",
    "load_secrets",
]

# JSON publishing goes to the server ROOT, so the topic travels in the body and never in a URL.
NTFY_SERVER_URL = "https://ntfy.sh/"
RESEND_EMAILS_URL = "https://api.resend.com/emails"
# The free tier sends only from this address, and only to the account owner's own inbox.
RESEND_SENDER = "NFL Predict <onboarding@resend.dev>"
HEALTHCHECKS_API_URL = "https://healthchecks.io/api/v3"

NOTIFY_HOSTS: frozenset[str] = frozenset(
    {"ntfy.sh", "api.resend.com", "hc-ping.com", "healthchecks.io"}
)


@dataclass(frozen=True)
class NotifySecrets:
    """Everything the sender needs from ``.env``; a missing secret is ``None``.

    Attributes:
        ntfy_topic: The ntfy topic (a long random string: anyone who knows it can read and post).
        resend_api_key: The Resend API key.
        digest_to_address: The digest recipient. Treated as a secret: never in a log record.
        healthchecks_ping_url: The check's ping URL; it embeds the check uuid.
        healthchecks_api_key: The healthchecks.io project's READ-WRITE API key (pause, read-back).
        withhold_checkin_on: The one D-16 watcher test day, an ET date. Not a secret.
        other_secret_values: The Odds API key and SECRET_KEY, carried ONLY so error text can be
            scrubbed of them. Plain strings, so they are kept out of the repr.
    """

    ntfy_topic: SecretStr | None = None
    resend_api_key: SecretStr | None = None
    digest_to_address: SecretStr | None = None
    healthchecks_ping_url: SecretStr | None = None
    healthchecks_api_key: SecretStr | None = None
    withhold_checkin_on: date | None = None
    other_secret_values: tuple[str, ...] = field(default=(), repr=False)

    def secret_values(self) -> tuple[str, ...]:
        """Every non-empty secret string, for SCRUBBING ONLY -- never print or record the result.

        Includes the uuid tail of the ping URL on its own (an error can name the uuid without the
        URL around it) and :attr:`other_secret_values`.
        """
        values: list[str] = []
        for secret in (
            self.ntfy_topic,
            self.resend_api_key,
            self.digest_to_address,
            self.healthchecks_ping_url,
            self.healthchecks_api_key,
        ):
            if secret is not None:
                values.append(secret.get_secret_value())
        if self.healthchecks_ping_url is not None:
            ping_url = self.healthchecks_ping_url.get_secret_value()
            values.append(ping_url.rstrip("/").rsplit("/", 1)[-1])
        values.extend(self.other_secret_values)
        return tuple(dict.fromkeys(value for value in values if value.strip()))


def _secret(value: Any) -> SecretStr | None:
    """*value* (a SecretStr, a str or None) as a stripped SecretStr; empty or blank reads as None."""
    if value is None:
        return None
    text = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
    text = text.strip()
    return SecretStr(text) if text else None


def _plain(value: Any) -> str | None:
    """*value* (a SecretStr, a str or None) as a stripped str; empty or blank reads as None."""
    secret = _secret(value)
    return secret.get_secret_value() if secret is not None else None


def _withhold_date(value: Any) -> date | None:
    """The D-16 test day: a date, an ISO ``YYYY-MM-DD`` string, or None."""
    if value is None or isinstance(value, date):
        return value
    text = str(value).strip()
    return date.fromisoformat(text) if text else None


def load_secrets(settings: Any | None = None) -> NotifySecrets:
    """Read the notification secrets from *settings* (default: ``conf.settings.get_settings()``).

    Attributes the settings object does not carry read as None, so the sender degrades to recorded
    ``missing_secret`` failures rather than an AttributeError. Never prints or logs.
    """
    if settings is None:
        from conf.settings import get_settings

        settings = get_settings()
    other = (
        _plain(getattr(settings, "odds_api_key", None)),
        _plain(getattr(settings, "secret_key", None)),
    )
    return NotifySecrets(
        ntfy_topic=_secret(getattr(settings, "ntfy_topic", None)),
        resend_api_key=_secret(getattr(settings, "resend_api_key", None)),
        digest_to_address=_secret(getattr(settings, "digest_to_address", None)),
        healthchecks_ping_url=_secret(getattr(settings, "healthchecks_ping_url", None)),
        healthchecks_api_key=_secret(getattr(settings, "healthchecks_api_key", None)),
        withhold_checkin_on=_withhold_date(
            getattr(settings, "watcher_test_withhold_date", None)
        ),
        other_secret_values=tuple(value for value in other if value is not None),
    )
