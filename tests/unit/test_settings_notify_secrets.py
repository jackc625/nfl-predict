"""The five notification secrets reach ``Settings`` as ``SecretStr`` and never print (Phase 35 D-01, D-09).

Every test builds ``Settings(_env_file=<a file under tmp_path>)``, so the owner's real ``.env`` is
never read. The values below are dummies; none is a real secret.

Pitfall 1 (RESEARCH): ``Settings`` forbids unknown ``.env`` keys, which is typo protection worth
keeping, but by default the validation error echoes the offending line's VALUE -- so a mistyped
secret name would print the secret into the task log. ``hide_input_in_errors = True`` keeps the
field name in the error and drops the value.

Stage 2 of the legacy-flag removal (Plan 35-17, D-05): ``enable_email_reports`` /
``enable_slack_notifications`` are no longer declared, so a ``.env`` line naming either is refused
by name with its value hidden.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from conf.settings import Settings

# env-var name -> (Settings field, dummy value). The dummies are distinctive so a leak is findable.
SECRET_FIELDS: dict[str, tuple[str, str]] = {
    "NTFY_TOPIC": ("ntfy_topic", "dummytopic0123456789abcdefghijkl"),
    "RESEND_API_KEY": ("resend_api_key", "re_dummy_resend_key_0123456789"),
    "DIGEST_TO_ADDRESS": ("digest_to_address", "dummy-recipient@example.invalid"),
    "HEALTHCHECKS_PING_URL": (
        "healthchecks_ping_url",
        "https://hc-ping.com/00000000-1111-2222-3333-444444444444",
    ),
    "HEALTHCHECKS_API_KEY": ("healthchecks_api_key", "hcw_dummy_read_write_key_0123"),
}
TEST_DAY_ENV = "WATCHER_TEST_WITHHOLD_DATE"
LEGACY_FLAG_ENVS = ("ENABLE_EMAIL_REPORTS", "ENABLE_SLACK_NOTIFICATIONS")


@pytest.fixture(autouse=True)
def _no_inherited_notification_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process environment beats the env file in pydantic-settings; clear every name under test."""
    for name in (*SECRET_FIELDS, TEST_DAY_ENV, *LEGACY_FLAG_ENVS, "NTFY_TOPICS"):
        monkeypatch.delenv(name, raising=False)


def _env_file(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / ".env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_five_secret_env_file_loads_as_secret_str(tmp_path: Path) -> None:
    lines = [f"{env}={dummy}" for env, (_, dummy) in SECRET_FIELDS.items()]
    settings = Settings(_env_file=_env_file(tmp_path, lines))

    for _, (field_name, dummy) in SECRET_FIELDS.items():
        value = getattr(settings, field_name)
        assert isinstance(value, SecretStr), field_name
        assert value.get_secret_value() == dummy, field_name

    shown = repr(settings)
    for _, (field_name, dummy) in SECRET_FIELDS.items():
        assert dummy not in shown, f"{field_name} value appears in repr(Settings)"


def test_an_unknown_key_names_the_field_but_hides_the_value(tmp_path: Path) -> None:
    leaked = "dummy-secret-value-xyz"
    env_file = _env_file(tmp_path, [f"NTFY_TOPICS={leaked}"])

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=env_file)

    message = str(caught.value)
    assert "ntfy_topics" in message
    assert leaked not in message


def test_the_test_day_line_parses_as_a_date(tmp_path: Path) -> None:
    settings = Settings(_env_file=_env_file(tmp_path, [f"{TEST_DAY_ENV}=2026-10-16"]))

    assert settings.watcher_test_withhold_date == date(2026, 10, 16)


@pytest.mark.parametrize("value", ["", "   ", "YYYY-MM-DD"])
def test_a_blank_or_placeholder_test_day_reads_as_unset(
    tmp_path: Path, value: str
) -> None:
    """35-REVIEW IN-05: a blank or unedited test-day line never stops the settings load."""
    settings = Settings(_env_file=_env_file(tmp_path, [f"{TEST_DAY_ENV}={value}"]))

    assert settings.watcher_test_withhold_date is None


def test_a_legacy_key_is_now_refused_with_its_value_hidden(tmp_path: Path) -> None:
    leaked = "dummy-legacy-value-xyz"
    env_file = _env_file(tmp_path, [f"ENABLE_EMAIL_REPORTS={leaked}"])

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=env_file)

    # extra_forbidden, not bool_parsing: the dummy is not a bool either, so only the error type
    # proves the field is gone rather than merely given a bad value.
    assert [error["type"] for error in caught.value.errors()] == ["extra_forbidden"]
    message = str(caught.value)
    assert "enable_email_reports" in message
    assert leaked not in message


def test_absent_secrets_are_none(tmp_path: Path) -> None:
    settings = Settings(_env_file=_env_file(tmp_path, []))

    for _, (field_name, _) in SECRET_FIELDS.items():
        assert getattr(settings, field_name) is None, field_name
    assert settings.watcher_test_withhold_date is None
