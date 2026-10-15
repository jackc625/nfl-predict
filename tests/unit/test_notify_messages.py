"""The push builders' wording rules (Phase 35 SPEC R1/R2/R8 edges, D-06, D-10).

* The next-action line flips exactly at the slate's lock, and shows the lock instant it was given
  (formatted in ET), never a clock time written into ``notify/messages.py``.
* Unknown games read ``unknown``.
* The one-line error is one ASCII line of at most 160 characters with every secret redacted and
  every URL cut to its scheme and host.
* No title carries the app name; the weekly-email failure push names its date and failure kind.
* The bets push lines use /bets' own labels and pick macros, and a 50-bet body is capped at
  4,000 bytes ending ``... and N more`` when the router sends it.

Reports are built directly; secrets are dummy values from ``tests/unit/test_notify_transport.py``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from typing import Any

import pytest
from pydantic import SecretStr

from api.cache import ARM_LIVE, BET_STATUS_LIVE
from notify.events import (
    PRIORITY_NORMAL,
    PRIORITY_TOP,
    PUSH_BODY_LIMIT_BYTES,
    send_push,
)
from notify.messages import (
    RERUN_COMMAND,
    bets_message,
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
from tests.unit.test_forward_ledger_store import make_row
from tests.unit.test_notify_transport import (
    DUMMY_PING_URL,
    DUMMY_PING_UUID,
    DUMMY_TOPIC,
    RUN_DATE,
    daily_context,
    dummy_secrets,
    install_sleep_recorder,
    install_transport,
    ntfy_ok,
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


# ---------------------------------------------------------------------------
# The bets push lines (SPEC R1 encoding, R3, D-14)
# ---------------------------------------------------------------------------


def live_bet(game_id: str, **fields: Any) -> dict[str, Any]:
    """One appended forward row of the live arm with ``status`` live, as the ledger stores it."""
    values: dict[str, Any] = {
        "game_id": game_id,
        "season": 2026,
        "week": 6,
        "arm": ARM_LIVE,
        "status": BET_STATUS_LIVE,
    }
    values.update(fields)
    return make_row(**values)


def test_each_line_uses_the_bets_labels() -> None:
    rows = [
        live_bet(
            "2026_06_KC@DEN",
            target="wp",
            bet_side="away",
            line=None,
            per_bet_ev=0.031,
            stake_units=0.5,
            ev_tier=None,
        ),
        live_bet(
            "2026_06_CIN@CLE",
            target="ats",
            bet_side="home_cover",
            line=-5.5,
            per_bet_ev=0.062,
            stake_units=1.25,
            ev_tier="high",
        ),
        live_bet(
            "2026_06_DAL@PHI",
            target="ou",
            bet_side="under",
            line=47.5,
            per_bet_ev=0.041,
            stake_units=2.0,
            ev_tier="medium",
        ),
    ]

    message = bets_message(_report(slate_season=2026, slate_week=6, appended_rows=rows))

    assert message is not None
    assert message.item_lines == (
        "CIN @ CLE | Spread | CLE +5.5 | 1.25 units | EV band high",
        "DAL @ PHI | Totals | Under 47.5 | 2.00 units | EV band medium",
        "KC @ DEN | Winner | KC | 0.50 units | EV band low",
    )


def test_a_fifty_bet_body_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, ntfy_ok())
    # A real line is about 60 bytes, so 50 of them fit in 4,000; long synthetic team codes make
    # each line about 120 bytes, so the cap has to cut.
    away, home = "A" * 40, "H" * 40
    rows = [
        live_bet(f"2026_06_{away}{index:02d}@{home}", per_bet_ev=0.05 - index / 10000)
        for index in range(50)
    ]
    message = bets_message(_report(slate_season=2026, slate_week=6, appended_rows=rows))
    assert message is not None
    assert len(message.item_lines) == 50

    send_push(message, daily_context())

    (request,) = requests
    body = json.loads(request.content)["message"]
    assert body.isascii()
    assert len(body.encode("utf-8")) <= PUSH_BODY_LIMIT_BYTES
    lines = body.splitlines()
    kept = [line for line in lines if line.startswith(away)]
    assert 0 < len(kept) < 50
    assert lines[-1] == f"... and {50 - len(kept)} more"
