"""The outside watcher hears from every in-season daily run (Phase 35 NOTIF-04, SPEC R7, D-13, D-15).

* Every in-season run checks in ONCE, LAST -- after every message -- with a short ASCII run summary
  as the ping body, never a secret (D-15).
* A run whose messages were all delivered checks in normally, a delivered failed push included; a
  message that stayed undelivered, or a delivered one whose receipt never reached the notification
  log, sends the check-in to ``<ping_url>/fail`` (SPEC R8; T-35-47).
* The season window opens 7 days before the season's first lock day and runs through the day the
  Super Bowl result is recorded, read at RUN START; the day after the Super Bowl is always in,
  whoever wrote the score (review 35-13). Outside it the run pauses the check and never pings
  (D-01); a failed pause is recorded, never escalated. An unknown window checks in (D-17).
* The one watcher test day withholds the check-in only when every D-16 condition holds.
* ``send_checkin`` has exactly one production call site, and the closing-line task and the manual
  Friday tool make no check-in (SPEC must-NOT).

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
that answers per host and records every request in send order; every retry delay goes through a
recording ``SLEEP``. Secrets are dummy values; no real provider is called. The suite-wide
``tests/conftest.py`` fixtures keep the digest step on "not a digest day".

Run this module:  uv run python -m pytest tests/unit/test_notify_watcher.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import subprocess
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

import scripts.daily_lock_pipeline as daily
from notify import log as notification_log
from notify import transport
from notify.end_of_run import notify_end_of_run
from notify.log import read_notifications
from notify.report import (
    STATUS_LOCK_PASSED,
    STATUS_NO_GAMES,
    STATUS_OFFSEASON,
    STATUS_RUN_FAILED,
    STATUS_SUCCESS,
    RunReport,
)
from notify.season_window import (
    SEASON_WINDOW_IN,
    SEASON_WINDOW_OUT,
    SEASON_WINDOW_UNKNOWN,
    current_season_window,
    season_window_state,
)
from notify.secrets import NotifySecrets
from pipeline.daily_steps import slate_lock
from tests.unit.test_daily_lock_run import RUN_DATE as DAILY_RUN_DATE
from tests.unit.test_daily_lock_run import (
    _arrange_lock_passed,
    _no_games_env,
    _start,
)
from tests.unit.test_notify_channels import (
    DUMMY_HC_API_KEY,
    email_secrets,
    watcher_secrets,
)
from tests.unit.test_notify_messages import live_bet
from tests.unit.test_notify_transport import (
    DUMMY_PING_URL,
    DUMMY_PING_UUID,
    RUN_DATE,
    install_sleep_recorder,
    ntfy_ok,
)
from utils.file_lock import FileLockHeldError, exclusive_file_lock
from utils.game_lock import ET

REPO_ROOT = Path(__file__).resolve().parents[2]

TOMORROW = ("2026_06_CIN@CLE", "2026_06_DAL@PHI")

Reply = Callable[[], httpx.Response] | Exception


def ping_ok() -> Callable[[], httpx.Response]:
    """healthchecks.io's answer to a ping that did something: HTTP 200 with the body ``OK``."""
    return lambda: httpx.Response(200, text="OK")


def install_hosts(
    monkeypatch: pytest.MonkeyPatch, by_host: dict[str, tuple[Reply, ...]]
) -> list[httpx.Request]:
    """Answer each provider host from its own reply list; return EVERY request in send order.

    The last reply of each list repeats once it runs out. A request to a host with no list fails
    the test.
    """
    seen: list[httpx.Request] = []
    counts: dict[str, int] = dict.fromkeys(by_host, 0)

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen.append(request)
        if host not in by_host:
            pytest.fail(f"unexpected request to {host}", pytrace=False)
        counts[host] += 1
        options = by_host[host]
        reply = options[min(counts[host], len(options)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply()

    monkeypatch.setattr(
        transport,
        "CLIENT_FACTORY",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return seen


def hosts(requests: list[httpx.Request]) -> list[str]:
    """The host of each request, in send order."""
    return [request.url.host for request in requests]


def in_season_report(**fields: Any) -> RunReport:
    """An in-season run for RUN_DATE that started an hour before its slate's lock."""
    lock = slate_lock(RUN_DATE)
    values: dict[str, Any] = {
        "run_date_et": RUN_DATE,
        "start": lock - timedelta(hours=1),
        "lock": lock,
        "status": STATUS_SUCCESS,
        "tomorrow_game_ids": TOMORROW,
        "unpredicted_game_ids": (),
        "slate_season": 2026,
        "slate_week": 6,
        "season_window": "in",
    }
    values.update(fields)
    return RunReport(**values)


def every_secret() -> NotifySecrets:
    """Dummy values for all five notification secrets."""
    return email_secrets(healthchecks_api_key=watcher_secrets().healthchecks_api_key)


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No retry delay in this module actually sleeps."""
    return install_sleep_recorder(monkeypatch)


# ---------------------------------------------------------------------------
# Task 1 (tracer): an in-season run checks in once, last, with an ASCII run summary
# ---------------------------------------------------------------------------


def test_an_in_season_success_checks_in_last_with_a_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(monkeypatch, {"hc-ping.com": (ping_ok(),)})
    report = in_season_report()

    notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    (ping,) = requests
    assert requests[-1] is ping
    assert ping.method == "POST"
    assert str(ping.url) == DUMMY_PING_URL
    body = ping.content.decode("ascii")
    assert "outcome=success" in body
    assert f"NFL Predict daily run {RUN_DATE.isoformat()}" in body
    assert "games tomorrow=2" in body
    (record,) = read_notifications()
    assert record["event_class"] == "checkin"
    assert record["outcome"] == "delivered"
    assert "WATCHER_CHECKIN= delivered" in capsys.readouterr().out


def test_a_failed_run_whose_push_was_delivered_checks_in_normally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    report = in_season_report(
        status=STATUS_RUN_FAILED,
        exception_class="RuntimeError",
        unpredicted_game_ids=TOMORROW,
    )

    outcome = notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    assert hosts(requests) == ["ntfy.sh", "hc-ping.com"]
    assert json.loads(requests[0].content)["title"].startswith("FAILED:")
    assert str(requests[-1].url) == DUMMY_PING_URL
    assert outcome.any_undelivered is False
    body = requests[-1].content.decode("ascii")
    assert "outcome=run_failed" in body
    assert "failed pushes=1/1" in body
    assert [r["event_class"] for r in read_notifications()] == ["failed", "checkin"]
    assert "WATCHER_CHECKIN= delivered\n" in capsys.readouterr().out


def test_the_summary_carries_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    secrets = every_secret()
    report = in_season_report(
        appended_rows=[live_bet(TOMORROW[0], per_bet_ev=0.05)],
    )

    notify_end_of_run(report, secrets=secrets, now=report.start)

    body = requests[-1].content
    assert len(body) <= 1000
    text = body.decode("ascii")
    assert "bets appended=1" in text
    assert "bets push=delivered" in text
    assert "digest=none" in text
    for secret in secrets.secret_values():
        assert secret not in text


def test_an_unlogged_receipt_fails_the_check_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    locks_taken: list[int] = []

    def _bets_record_lock_never_frees(*args: Any, **kwargs: Any) -> Any:
        locks_taken.append(1)
        if len(locks_taken) == 1:  # the bets record; the check-in record is written
            msg = "the lock is held by another writer (waited 60s)."
            raise FileLockHeldError(msg)
        return exclusive_file_lock(*args, **kwargs)

    monkeypatch.setattr(
        notification_log, "exclusive_file_lock", _bets_record_lock_never_frees
    )
    report = in_season_report(appended_rows=[live_bet(TOMORROW[0], per_bet_ev=0.05)])

    outcome = notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    assert (outcome.any_undelivered, outcome.any_unlogged) == (False, True)
    assert hosts(requests) == ["ntfy.sh", "hc-ping.com"]
    assert str(requests[-1].url) == f"{DUMMY_PING_URL}/fail"
    assert "unlogged receipts=1" in requests[-1].content.decode("ascii")
    assert "WATCHER_CHECKIN= delivered as failure" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Task 2: the season window, the off-season pause, the test-day withhold, the one call site
# ---------------------------------------------------------------------------

PAUSE_URL = f"https://healthchecks.io/api/v3/checks/{DUMMY_PING_UUID}/pause"


def season_2026(*, super_bowl_scored: bool) -> pd.DataFrame:
    """The 2026 season as silver ``games`` holds it: opener 2026-09-10 (its lock is 2026-09-09)
    and the Super Bowl on 2027-02-14 ET, with or without its scores."""
    score = 24 if super_bowl_scored else None
    rows = [
        ("2026_01_DAL@PHI", "2026-09-10 20:20", "REG", 20, 17),
        ("2026_10_KC@BUF", "2026-11-15 13:00", "REG", 27, 24),
        ("2026_22_SF@KC", "2027-02-14 18:30", "SB", score, score),
    ]
    return pd.DataFrame(
        {
            "game_id": [row[0] for row in rows],
            "season": 2026,
            "kickoff_et": [pd.Timestamp(row[1]).tz_localize(ET) for row in rows],
            "game_type": [row[2] for row in rows],
            "home_score": [row[3] for row in rows],
            "away_score": [row[4] for row in rows],
        }
    )


def _unreadable_kickoff() -> pd.DataFrame:
    frame = season_2026(super_bowl_scored=False)
    frame["kickoff_et"] = frame["kickoff_et"].astype(object)
    frame.loc[0, "kickoff_et"] = "not a kickoff"
    return frame


@pytest.mark.parametrize(
    ("frame", "run_date", "expected"),
    [
        pytest.param(
            lambda: season_2026(super_bowl_scored=False),
            date(2026, 9, 1),
            SEASON_WINDOW_OUT,
            id="eight-days-before-the-first-lock",
        ),
        pytest.param(
            lambda: season_2026(super_bowl_scored=False),
            date(2026, 9, 2),
            SEASON_WINDOW_IN,
            id="seven-days-before-the-first-lock",
        ),
        pytest.param(
            lambda: season_2026(super_bowl_scored=False),
            date(2027, 2, 20),
            SEASON_WINDOW_IN,
            id="super-bowl-with-no-result",
        ),
        pytest.param(
            lambda: season_2026(super_bowl_scored=True),
            date(2027, 2, 16),
            SEASON_WINDOW_OUT,
            id="result-recorded-two-days-after-the-game",
        ),
        pytest.param(
            lambda: season_2026(super_bowl_scored=True),
            date(2027, 7, 15),
            SEASON_WINDOW_OUT,
            id="next-season-not-recorded-in-july",
        ),
        pytest.param(
            pd.DataFrame, date(2026, 10, 15), SEASON_WINDOW_UNKNOWN, id="empty-frame"
        ),
        pytest.param(
            lambda: season_2026(super_bowl_scored=False).drop(columns="home_score"),
            date(2026, 10, 15),
            SEASON_WINDOW_UNKNOWN,
            id="missing-column",
        ),
        pytest.param(
            _unreadable_kickoff,
            date(2026, 10, 15),
            SEASON_WINDOW_UNKNOWN,
            id="unreadable-kickoff",
        ),
    ],
)
def test_window_edges(
    frame: Callable[[], pd.DataFrame], run_date: date, expected: str
) -> None:
    assert season_window_state(frame(), run_date) == expected


def _silver_games(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    """Silver ``games`` reads *frame* (the window's only data source)."""
    monkeypatch.setattr(
        "data.storage.load_dataframe",
        lambda table, **_k: frame.copy() if table == "games" else pd.DataFrame(),
    )


def test_the_result_day_is_still_in(monkeypatch: pytest.MonkeyPatch) -> None:
    # At the start of the 17:00 ET run the day after the game, silver has no score yet.
    _silver_games(monkeypatch, season_2026(super_bowl_scored=False))
    assert current_season_window(date(2027, 2, 15)) == SEASON_WINDOW_IN

    # That run ingested the result; the next day's run pauses.
    _silver_games(monkeypatch, season_2026(super_bowl_scored=True))
    assert current_season_window(date(2027, 2, 16)) == SEASON_WINDOW_OUT


def test_a_score_written_early_on_the_day_after_the_super_bowl_keeps_that_day_in() -> (
    None
):
    scored = season_2026(super_bowl_scored=True)
    unscored = season_2026(super_bowl_scored=False)

    # A hand-run ingest recorded the score before the 17:00 ET run: the day stays in.
    assert season_window_state(scored, date(2027, 2, 15)) == SEASON_WINDOW_IN
    assert season_window_state(scored, date(2027, 2, 16)) == SEASON_WINDOW_OUT
    # With no score recorded, the result has not been recorded: still in.
    assert season_window_state(unscored, date(2027, 2, 16)) == SEASON_WINDOW_IN


def test_checks_in_on_success_no_games_failed_and_lock_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    reports = [
        in_season_report(),
        in_season_report(status=STATUS_NO_GAMES, tomorrow_game_ids=()),
        in_season_report(
            status=STATUS_RUN_FAILED,
            exception_class="RuntimeError",
            unpredicted_game_ids=TOMORROW,
        ),
        in_season_report(status=STATUS_LOCK_PASSED, unpredicted_game_ids=TOMORROW),
    ]

    for report in reports:
        before = len(requests)
        notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)
        this_run = requests[before:]
        assert hosts(this_run).count("hc-ping.com") == 1
        assert this_run[-1].url.host == "hc-ping.com"

    assert hosts(requests).count("hc-ping.com") == 4
    assert "healthchecks.io" not in hosts(requests)


def test_out_of_season_pauses_and_never_pings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(
        monkeypatch, {"healthchecks.io": (lambda: httpx.Response(200, json={}),)}
    )
    report = in_season_report(
        status=STATUS_OFFSEASON, tomorrow_game_ids=(), season_window=SEASON_WINDOW_OUT
    )

    notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    (pause,) = requests
    assert pause.method == "POST"
    assert str(pause.url) == PAUSE_URL
    assert pause.headers["X-Api-Key"] == DUMMY_HC_API_KEY
    assert [(r["event_class"], r["outcome"]) for r in read_notifications()] == [
        ("pause", "delivered")
    ]
    assert "WATCHER_CHECKIN= paused (out of season)" in capsys.readouterr().out


def test_a_failed_pause_is_recorded_not_escalated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only healthchecks.io answers: an ntfy push or a ping would fail the test.
    requests = install_hosts(
        monkeypatch, {"healthchecks.io": (lambda: httpx.Response(500),)}
    )
    report = in_season_report(
        status=STATUS_OFFSEASON, tomorrow_game_ids=(), season_window=SEASON_WINDOW_OUT
    )

    notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    assert hosts(requests) == ["healthchecks.io"] * 3
    (record,) = read_notifications()
    assert (record["event_class"], record["outcome"]) == ("pause", "failed")
    assert record["attempts"] == 3
    assert "WATCHER_CHECKIN= pause failed http_500" in capsys.readouterr().out


def test_an_unknown_window_checks_in(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unreadable(*_a: Any, **_k: Any) -> pd.DataFrame:
        raise OSError("the silver games table cannot be opened")

    monkeypatch.setattr("data.storage.load_dataframe", _unreadable)
    window = current_season_window(RUN_DATE)
    assert window == SEASON_WINDOW_UNKNOWN
    requests = install_hosts(monkeypatch, {"hc-ping.com": (ping_ok(),)})
    report = in_season_report(season_window=window)

    notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    (ping,) = requests
    assert str(ping.url) == DUMMY_PING_URL


@pytest.mark.parametrize(
    ("withhold_on", "window", "tomorrow", "reason"),
    [
        pytest.param(RUN_DATE, SEASON_WINDOW_IN, (), None, id="every-condition-holds"),
        pytest.param(
            RUN_DATE,
            SEASON_WINDOW_IN,
            TOMORROW,
            "games tomorrow=2",
            id="games-tomorrow",
        ),
        pytest.param(
            RUN_DATE,
            SEASON_WINDOW_UNKNOWN,
            (),
            "season window unknown",
            id="window-unknown",
        ),
        pytest.param(
            RUN_DATE + timedelta(days=1), SEASON_WINDOW_IN, (), "", id="another-date"
        ),
    ],
)
def test_the_test_day_withholds_only_when_every_condition_holds(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    withhold_on: date,
    window: str,
    tomorrow: tuple[str, ...],
    reason: str | None,
) -> None:
    replies: dict[str, tuple[Reply, ...]] = (
        {} if reason is None else {"hc-ping.com": (ping_ok(),)}
    )
    requests = install_hosts(monkeypatch, replies)
    report = in_season_report(
        status=STATUS_NO_GAMES if tomorrow == () else STATUS_SUCCESS,
        tomorrow_game_ids=tomorrow,
        season_window=window,
    )

    notify_end_of_run(
        report,
        secrets=watcher_secrets(withhold_checkin_on=withhold_on),
        now=report.start,
    )

    out = capsys.readouterr().out
    (record,) = read_notifications()
    assert record["event_class"] == "checkin"
    if reason is None:
        assert requests == []
        assert (record["outcome"], record["attempts"]) == ("withheld", 0)
        assert "WATCHER_CHECKIN= withheld (test day)" in out
        return
    (ping,) = requests
    assert str(ping.url) == DUMMY_PING_URL
    assert record["outcome"] == "delivered"
    if reason:
        assert f"WATCHER_TEST_DAY= not withheld: {reason}" in out
    else:
        assert "WATCHER_TEST_DAY=" not in out


def test_run_daily_sets_the_window_at_run_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from forward_ledger import closing_schedule

    monkeypatch.setattr(
        closing_schedule,
        "register_closing_triggers",
        lambda *_a, **_k: closing_schedule.TriggerRegistration(
            "skipped", "test_stand_in", (), False
        ),
    )
    order = _no_games_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        daily, "_refresh_schedule", lambda *_a, **_k: order.append("refresh") or 2026
    )
    asked: list[date] = []

    def _window(run_date_et: date) -> str:
        asked.append(run_date_et)
        order.append("window")
        return SEASON_WINDOW_OUT

    monkeypatch.setattr(daily, "SEASON_WINDOW", _window)
    seen: list[RunReport] = []
    monkeypatch.setattr(daily, "NOTIFY_END_OF_RUN", seen.append)

    assert daily.run_daily(DAILY_RUN_DATE, start=_start(), dry_run=False) == 0
    assert order[:2] == ["window", "refresh"], "the window was not read at run start"
    at_the_lock: datetime = _arrange_lock_passed(monkeypatch, tmp_path)
    assert daily.run_daily(DAILY_RUN_DATE, start=at_the_lock, dry_run=False) == 0

    assert [report.status for report in seen] == [STATUS_NO_GAMES, STATUS_LOCK_PASSED]
    assert [report.season_window for report in seen] == [SEASON_WINDOW_OUT] * 2
    assert asked == [DAILY_RUN_DATE] * 2

    assert daily.run_daily(DAILY_RUN_DATE, start=_start(), dry_run=True) == 0
    assert len(asked) == 2, "a dry run read the season window"


CHECKIN_FUNCTIONS = frozenset({"send_checkin", "record_withheld_checkin", "hc_ping"})
NEVER_CHECK_IN = (
    "scripts/capture_closing_lines.py",
    "notify/closing_task.py",
    "scripts/friday_pipeline.py",
)


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _tracked_production_sources() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.split()
    return [path for path in listed if not path.startswith("tests/")]


def test_exactly_one_checkin_call_site() -> None:
    sources = _tracked_production_sources()
    assert "notify/end_of_run.py" in sources
    call_sites: list[str] = []
    for path in sources:
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        call_sites.extend(
            path
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _called_name(node) == "send_checkin"
        )
    assert call_sites == ["notify/end_of_run.py"]

    for path in NEVER_CHECK_IN:
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        imported = {
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        }
        called = {
            _called_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        assert not (imported | called) & CHECKIN_FUNCTIONS, path
