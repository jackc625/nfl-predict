"""The push builders' wording rules (Phase 35 SPEC R1/R2/R8 edges, D-06, D-10).

* The next-action line flips exactly at the slate's lock, and shows the lock instant it was given
  (formatted in ET), never a clock time written into ``notify/messages.py``.
* Unknown games read ``unknown``.
* The one-line error is one ASCII line of at most 160 characters with every secret redacted and
  every URL cut to its scheme and host.
* No title carries the app name; the weekly-email failure push names its date and failure kind.

Reports are built directly; secrets are dummy values from ``tests/unit/test_notify_transport.py``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from pydantic import SecretStr

from notify.events import PRIORITY_NORMAL, PRIORITY_TOP
from notify.messages import (
    RERUN_COMMAND,
    closing_failure_message,
    email_failed_message,
    failed_messages,
    game_labels,
    next_action_line,
    one_line_error,
)
from notify.report import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_FINISHED_WITH_SKIPS,
    STATUS_RUN_FAILED,
    STATUS_SUCCESS,
    RunReport,
)
from pipeline.daily_steps import slate_lock
from tests.unit.test_notify_transport import (
    DUMMY_PING_URL,
    DUMMY_PING_UUID,
    DUMMY_TOPIC,
    RUN_DATE,
    dummy_secrets,
)
from utils.game_lock import ET

DUMMY_RESEND_KEY = "re_dummyresendkey0123456789"
DUMMY_ODDS_KEY = "dummyoddsapikey0123456789abcdef"


def _report(**fields: Any) -> RunReport:
    lock = slate_lock(RUN_DATE)
    values: dict[str, Any] = {
        "run_date_et": RUN_DATE,
        "start": lock - timedelta(hours=1),
        "lock": lock,
    }
    values.update(fields)
    return RunReport(**values)


def test_status_constants_match_the_orchestrator() -> None:
    from pipeline.steps import RunStatus

    assert RunStatus.SUCCESS.value == STATUS_SUCCESS
    assert RunStatus.FAILED.value == STATUS_FAILED
    assert RunStatus.FINISHED_WITH_SKIPS.value == STATUS_FINISHED_WITH_SKIPS
    assert STATUS_DEGRADED == "degraded"


def test_the_next_action_flips_exactly_at_the_lock() -> None:
    lock = slate_lock(RUN_DATE)

    before = next_action_line(lock, lock - timedelta(seconds=1))
    at = next_action_line(lock, lock)
    after = next_action_line(lock, lock + timedelta(seconds=1))

    assert before.startswith("Next: re-run before ")
    assert before.endswith(RERUN_COMMAND)
    assert RERUN_COMMAND == "uv run python -m scripts.daily_lock_pipeline"
    for line in (at, after):
        assert "can no longer be predicted" in line
        assert RERUN_COMMAND not in line


def test_the_next_action_shows_the_lock_it_was_given() -> None:
    half_past_three = datetime.combine(RUN_DATE, time(15, 30), tzinfo=ET)

    for now in (half_past_three - timedelta(minutes=1), half_past_three):
        line = next_action_line(half_past_three, now)
        assert "3:30 PM ET" in line
        assert "6:00 PM" not in line

    # After the 2026-11-01 daylight-saving change the wall clock still follows the zone.
    after_dst = slate_lock(date(2026, 11, 7))
    for now in (after_dst - timedelta(minutes=1), after_dst):
        assert "6:00 PM ET" in next_action_line(after_dst, now)


def test_unreadable_games_read_unknown() -> None:
    report = _report(
        status=STATUS_RUN_FAILED, tomorrow_game_ids=None, unpredicted_game_ids=None
    )
    report.record_exception(RuntimeError("schedule refresh exploded"))

    (message,) = failed_messages(report, now=report.start)

    assert message.priority == PRIORITY_TOP
    assert "Games: unknown" in message.header_lines
    assert game_labels(None) == "unknown"
    assert game_labels(()) == "none"


def test_the_one_line_error_is_one_ascii_line_without_secrets_or_url_paths() -> None:
    secrets = dummy_secrets(
        resend_api_key=SecretStr(DUMMY_RESEND_KEY),
        other_secret_values=(DUMMY_ODDS_KEY,),
    )
    text = (
        f"ping {DUMMY_PING_URL} failed for topic {DUMMY_TOPIC} with key {DUMMY_RESEND_KEY} "
        f"at https://example.org/v4/odds?apiKey={DUMMY_ODDS_KEY}&regions=us café "
        + "x" * 200
        + "\nsecond line names the uuid "
        + DUMMY_PING_UUID
    )

    line = one_line_error("ConnectError", text, secrets.secret_values())

    assert "\n" not in line
    assert line.isascii()
    assert len(line) <= 160
    assert line.startswith("ConnectError: ping <redacted> failed for topic <redacted>")
    assert "https://example.org " in line
    assert "/v4/odds" not in line
    assert "apiKey" not in line
    assert "second line" not in line
    for secret in (
        DUMMY_PING_URL,
        DUMMY_PING_UUID,
        DUMMY_TOPIC,
        DUMMY_RESEND_KEY,
        DUMMY_ODDS_KEY,
    ):
        assert secret not in line


def test_titles_carry_no_app_name() -> None:
    report = _report(status=STATUS_DEGRADED, failed_steps=("populate_web_cache",))
    titles = [message.title for message in failed_messages(report, now=report.start)]
    titles.append(email_failed_message(RUN_DATE, "http_500").title)
    titles.append(closing_failure_message(("2026_06_CIN@CLE",), "ReadTimeout").title)

    assert titles
    for title in titles:
        assert "NFL Predict" not in title
        assert title.startswith("FAILED: ")


def test_the_weekly_email_failure_names_its_kind() -> None:
    message = email_failed_message(date(2026, 10, 17), "build_UndefinedError")

    assert message.event_class == "failed"
    assert message.priority == PRIORITY_NORMAL
    assert message.title == "FAILED: weekly email"
    body = "\n".join([*message.header_lines, *message.item_lines])
    assert "2026-10-17" in body
    assert "build_UndefinedError" in body
    assert "not re-sent automatically" in body
