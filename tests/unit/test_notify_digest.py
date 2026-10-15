"""The Saturday digest email's send path (Phase 35 NOTIF-02, NOTIF-03, NOTIF-05, SPEC R4/R8, D-07, D-19).

* The digest run is the run whose tomorrow is its week's Sunday; a week with no Sunday game falls
  back to its last slate day (SPEC R4 boundary).
* A clean Saturday sends ONE Resend request, read from the web cache through ``DataService``, with
  the D-07 subject and an Idempotency-Key, and its receipt lands in the notification log.
* A day that is not a digest day sends no email.
* An email still undelivered after its attempts sends one normal-priority ``FAILED: weekly email``
  push naming the failure kind (SPEC R8).
* A due digest never vanishes: a cache file that cannot be opened still emails as not current
  (``bets unknown``, never ``0 bets``); a failure while building -- the render, the collect, or a
  malformed notification log -- records one failed digest (attempts 0, ``build_<Class>``) and sends
  the ``FAILED: weekly email`` push; and every run sends under its own Idempotency-Key.
* Every other R4 state (Plan 35-12): a failed Saturday names the failure and marks Sunday's games
  ``not predicted``; a zero-bet week says ``0 bets this week``; an absent cache shows the stale
  notice and never ``0 bets``; Monday reads ``picked Sunday evening``; a re-send after a delivered
  digest is ``(update)``; a lock-passed Saturday (D-18) and an unreadable-schedule Saturday (D-24)
  still email; a playoff week keeps its number (D-25); no CLV in the subject or header; the
  not-advice line is last; a shadow-arm row never reaches the email.

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
that answers per host, and every retry delay through a recording ``SLEEP``. The schedule is a
prepared ``tmp`` frame handed in through ``notify.digest.LOAD_SCHEDULE`` and the cache is a
``tmp_path`` DuckDB file at ``notify.digest.WEB_CACHE_PATH``: nothing under ``data/`` is read and no
real provider is called. Secrets are dummy values.

Run this module:  uv run python -m pytest tests/unit/test_notify_digest.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import httpx
import jinja2
import pandas as pd
import pytest

from api.cache import (
    BET_LIST_COLUMNS,
    BET_TRACKER_BLOCK_COLUMNS,
    BET_WEEK_RESULTS_COLUMNS,
    CACHE_SCHEMA,
    CURRENT_SLATE_KEY,
    GRADING_STATUS_PENDING,
    bet_list_populated_at_key,
    format_current_slate,
    materialize_bet_list,
    materialize_bet_tracker_blocks,
    materialize_bet_week_results,
)
from api.presentation import NOT_ADVICE_KEY, NOT_ADVICE_LINE
from api.services import DataService
from notify import digest, digest_sections, transport
from notify import log as notification_log
from notify.end_of_run import notify_end_of_run
from notify.events import PRIORITY_NORMAL, SendContext
from notify.log import read_notifications
from notify.report import (
    STATUS_LOCK_PASSED,
    STATUS_RUN_FAILED,
    STATUS_SUCCESS,
    RunReport,
)
from pipeline.daily_steps import slate_lock
from tests.unit.test_notify_channels import DUMMY_EMAIL_ID, email_secrets, resend_ok
from tests.unit.test_notify_messages import live_bet
from tests.unit.test_notify_transport import (
    DUMMY_RECEIPT_ID,
    RUN_DATE,
    daily_context,
    install_sleep_recorder,
    ntfy_ok,
)
from utils.current_slate import ScheduleUnavailableError, prepare_schedule
from utils.game_lock import ET

# RUN_DATE (2026-10-17) is a Saturday: its slate is week 6's Sunday games.
WEDNESDAY = date(2026, 10, 14)
THURSDAY_GAME = "2026_06_NYJ@NE"
SUNDAY_GAMES = ("2026_06_CIN@CLE", "2026_06_DAL@PHI", "2026_06_KC@BUF")
MONDAY_GAME = "2026_06_SF@SEA"
WEEK6: tuple[tuple[str, int, datetime], ...] = (
    (THURSDAY_GAME, 6, datetime(2026, 10, 15, 20, 15)),
    (SUNDAY_GAMES[0], 6, datetime(2026, 10, 18, 13, 0)),
    (SUNDAY_GAMES[1], 6, datetime(2026, 10, 18, 16, 25)),
    (SUNDAY_GAMES[2], 6, datetime(2026, 10, 18, 20, 20)),
    (MONDAY_GAME, 6, datetime(2026, 10, 19, 20, 15)),
)

Reply = Callable[[], httpx.Response] | Exception


def schedule_frame(games: tuple[tuple[str, int, datetime], ...]) -> pd.DataFrame:
    """The recorded schedule as ``utils.current_slate.load_recorded_schedule`` returns it."""
    return prepare_schedule(
        pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "season": 2026,
                    "week": week,
                    "kickoff_et": pd.Timestamp(kickoff).tz_localize(ET),
                    "game_type": "REG",
                }
                for game_id, week, kickoff in games
            ]
        )
    )


def _cache_row(row: dict[str, Any]) -> dict[str, Any]:
    """A ledger row as the cache's ``bet_list`` stores it, still ungraded."""
    return {
        **dict.fromkeys(BET_LIST_COLUMNS),
        **{name: row[name] for name in BET_LIST_COLUMNS if name in row},
        "grading_status": GRADING_STATUS_PENDING,
    }


def week6_bets() -> list[dict[str, Any]]:
    """Three live bets and one suppressed candidate for week 6, handed over out of /bets order."""
    return [
        live_bet(THURSDAY_GAME, target="ats", bet_side="home_cover", line=-3.5),
        live_bet(
            SUNDAY_GAMES[0], target="ou", bet_side="under", line=47.5, per_bet_ev=0.05
        ),
        live_bet(
            SUNDAY_GAMES[1],
            target="wp",
            bet_side="away",
            line=None,
            per_bet_ev=0.061,
            ev_tier="high",
        ),
        live_bet(
            SUNDAY_GAMES[2],
            status="suppressed",
            rejection_reason="ev_below_floor",
            per_bet_ev=None,
            stake_units=None,
            ev_tier=None,
        ),
    ]


def last_week_row() -> dict[str, Any]:
    """Week 5's stored actual vs expected row: the SPEC R5 worked example."""
    values = [2026, 5, "forward_realized", 2, 1, 1, 4, 0, 0.32725, 0.05]
    return dict(zip(BET_WEEK_RESULTS_COLUMNS, values, strict=True))


def tracker_rows() -> list[dict[str, Any]]:
    """Season-to-date blocks: the two forward classes plus a replay block the email leaves out."""
    rows = [
        ("forward", "pre_verdict", 3, 1, 2, 0, 1 / 3, -1.09),
        ("backtest_replay", "contaminated", 40, 22, 18, 0, 0.55, 2.0),
        ("forward", "forward_realized", 4, 2, 1, 1, 2 / 3, 1.309),
    ]
    return [dict(zip(BET_TRACKER_BLOCK_COLUMNS, row, strict=True)) for row in rows]


def build_cache(
    path: Path, *, bets: list[dict[str, Any]], stamped_at: datetime | None
) -> Path:
    """A web cache at *path* holding *bets* for week 6, week 5's result row and the tracker blocks.

    *stamped_at* is week 6's bet-list populated-at marker (None: never stamped).
    """
    conn = duckdb.connect(str(path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            if statement.strip():
                conn.execute(statement.strip())
        if bets:
            materialize_bet_list(conn, pd.DataFrame([_cache_row(row) for row in bets]))
        materialize_bet_week_results(conn, pd.DataFrame([last_week_row()]))
        materialize_bet_tracker_blocks(conn, pd.DataFrame(tracker_rows()))
        if stamped_at is not None:
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
                [
                    bet_list_populated_at_key(2026, 6),
                    stamped_at.isoformat(),
                    stamped_at,
                ],
            )
    finally:
        conn.close()
    return path


def install_router(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resend: tuple[Reply, ...] = (resend_ok(),),
    ntfy: tuple[Reply, ...] = (ntfy_ok(),),
) -> dict[str, list[httpx.Request]]:
    """Answer Resend, ntfy and watcher-ping requests from their own reply lists; return them per host.

    The last reply of each list repeats once the list runs out. The watcher host answers every
    check-in with healthchecks.io's ``OK``, so ``notify_end_of_run``'s last step really runs
    (35-REVIEW WR-02: an unanswered host used to raise inside the handler and be swallowed).
    """
    replies = {
        "api.resend.com": resend,
        "ntfy.sh": ntfy,
        "hc-ping.com": (lambda: httpx.Response(200, text="OK"),),
    }
    requests: dict[str, list[httpx.Request]] = {host: [] for host in replies}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen = requests[host]
        seen.append(request)
        options = replies[host]
        reply = options[min(len(seen), len(options)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply()

    monkeypatch.setattr(
        transport,
        "CLIENT_FACTORY",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return requests


def saturday_report(**fields: Any) -> RunReport:
    """A clean Saturday run that started at 17:00 ET, an hour before its slate's lock."""
    lock = slate_lock(RUN_DATE)
    values: dict[str, Any] = {
        "run_date_et": RUN_DATE,
        "start": lock - timedelta(hours=1),
        "lock": lock,
        "status": STATUS_SUCCESS,
        "tomorrow_game_ids": SUNDAY_GAMES,
        "unpredicted_game_ids": (),
        "slate_season": 2026,
        "slate_week": 6,
    }
    values.update(fields)
    return RunReport(**values)


@pytest.fixture
def cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Week 6's schedule behind ``LOAD_SCHEDULE``; ``WEB_CACHE_PATH`` at a tmp file; no sleeps."""
    install_sleep_recorder(monkeypatch)
    schedule = schedule_frame(WEEK6)
    monkeypatch.setattr(digest, "LOAD_SCHEDULE", lambda: schedule)
    path = tmp_path / "web_cache.duckdb"
    monkeypatch.setattr(digest, "WEB_CACHE_PATH", path)
    return path


def _sent_email(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Task 1 (tracer): the digest week, a clean Saturday's email, a non-digest day, an undelivered email
# ---------------------------------------------------------------------------


def test_digest_week_for_run_picks_the_saturday_run_for_a_sunday_week() -> None:
    schedule = schedule_frame(WEEK6)

    assert digest.digest_week_for_run(schedule, RUN_DATE) == (2026, 6)
    for not_a_digest_run in (
        WEDNESDAY,  # tomorrow is Thursday's game
        date(2026, 10, 15),  # Thursday's run: nothing on Friday
        date(2026, 10, 18),  # Sunday's run: tomorrow is Monday's game
        date(2026, 10, 19),  # Monday's run: nothing on Tuesday
    ):
        assert digest.digest_week_for_run(schedule, not_a_digest_run) is None


def test_a_week_with_no_sunday_game_falls_back_to_its_last_slate_run() -> None:
    schedule = schedule_frame(
        (
            ("2026_16_LA@SF", 16, datetime(2026, 12, 17, 20, 15)),  # Thursday
            ("2026_16_DEN@KC", 16, datetime(2026, 12, 19, 16, 30)),  # Saturday
            ("2026_16_GB@CHI", 16, datetime(2026, 12, 19, 20, 15)),  # Saturday
        )
    )

    assert digest.digest_week_for_run(schedule, date(2026, 12, 18)) == (2026, 16)
    assert digest.digest_week_for_run(schedule, date(2026, 12, 16)) is None
    assert digest.digest_week_for_run(schedule, date(2026, 12, 19)) is None


def test_a_clean_saturday_sends_one_email_with_a_receipt(
    monkeypatch: pytest.MonkeyPatch,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(monkeypatch)

    outcome = notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    assert requests["ntfy.sh"] == []
    (request,) = requests["api.resend.com"]
    assert request.headers["Idempotency-Key"].startswith(
        f"digest/{RUN_DATE.isoformat()}/"
    )
    sent = _sent_email(request)
    assert sent["subject"] == "NFL Predict - 2026 week 6 - 3 bets - run clean"
    html, text = sent["html"], sent["text"]
    assert html.isascii() and text.isascii()
    lowered = html.lower()
    assert "<img" not in lowered
    assert "<a " not in lowered and "<a>" not in lowered
    assert "://" not in html
    # Read from the cache, in /bets order (per-bet EV descending), live arm only.
    assert html.index("DAL @ PHI") < html.index("CIN @ CLE") < html.index("NYJ @ NE")
    for part in (html, text):
        assert "picked Sunday evening" in part  # Monday night's game, lock still ahead
        assert "no bet" in part  # KC @ BUF: evaluated, nothing cleared the floor
        assert "2-1-1 (W-L-P)" in part
        assert NOT_ADVICE_LINE in part
    assert text.rstrip().endswith(NOT_ADVICE_LINE)

    record, checkin = read_notifications()
    assert record["event_class"] == "digest"
    assert record["outcome"] == "delivered"
    assert record["receipt_id"] == DUMMY_EMAIL_ID
    assert [step for step, _delivery in outcome.deliveries] == ["digest"]
    assert outcome.any_undelivered is False
    # The watcher check-in ran last and checked in clean.
    (ping,) = requests["hc-ping.com"]
    assert not ping.url.path.endswith("/fail")
    assert (checkin["event_class"], checkin["outcome"]) == ("checkin", "delivered")
    out = capsys.readouterr().out
    assert f"NOTIFY_DIGEST= delivered {DUMMY_EMAIL_ID}" in out
    assert "WATCHER_CHECKIN= delivered" in out


def test_a_non_digest_day_sends_no_email(
    monkeypatch: pytest.MonkeyPatch,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock = slate_lock(WEDNESDAY)
    report = saturday_report(
        run_date_et=WEDNESDAY,
        start=lock - timedelta(hours=1),
        lock=lock,
        tomorrow_game_ids=(THURSDAY_GAME,),
    )
    build_cache(cache_path, bets=week6_bets(), stamped_at=report.start)
    requests = install_router(monkeypatch)

    outcome = notify_end_of_run(report, secrets=email_secrets(), now=report.start)

    assert requests["api.resend.com"] == [] and requests["ntfy.sh"] == []
    # Only the watcher check-in left the run.
    assert len(requests["hc-ping.com"]) == 1
    assert [(r["event_class"], r["outcome"]) for r in read_notifications()] == [
        ("checkin", "delivered")
    ]
    assert outcome.deliveries == []
    out = capsys.readouterr().out
    assert "NOTIFY_DIGEST= not a digest day" in out
    assert "WATCHER_CHECKIN= delivered" in out


def test_an_undelivered_email_sends_one_normal_failed_push(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(
        monkeypatch,
        resend=(lambda: httpx.Response(500),),
        ntfy=(ntfy_ok(DUMMY_RECEIPT_ID),),
    )

    outcome = notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    assert len(requests["api.resend.com"]) == 3
    (push,) = requests["ntfy.sh"]
    body = json.loads(push.content)
    assert body["priority"] == PRIORITY_NORMAL == 3
    assert body["title"] == "FAILED: weekly email"
    assert "http_500" in body["message"]
    assert [step for step, _delivery in outcome.deliveries] == ["digest", "failed"]
    assert outcome.any_undelivered is True
    records = read_notifications()
    assert [(r["event_class"], r["outcome"]) for r in records] == [
        ("digest", "failed"),
        ("failed", "delivered"),
        ("checkin", "delivered"),
    ]
    assert records[0]["failure_kind"] == "http_500"
    # An undelivered email makes the check-in the watcher's failure signal.
    (ping,) = requests["hc-ping.com"]
    assert ping.url.path.endswith("/fail")


# ---------------------------------------------------------------------------
# Task 2: a due digest never vanishes, and every run sends under its own key
# ---------------------------------------------------------------------------


def _assert_one_build_failure(
    requests: dict[str, list[httpx.Request]], records: list[dict[str, Any]], kind: str
) -> None:
    """No email left; one failed digest record (no attempt), one FAILED: weekly email push, then
    the watcher check-in sent as the failure signal."""
    assert requests["api.resend.com"] == []
    (push,) = requests["ntfy.sh"]
    body = json.loads(push.content)
    assert body["priority"] == PRIORITY_NORMAL
    assert body["title"] == "FAILED: weekly email"
    assert kind in body["message"]
    assert [(r["event_class"], r["outcome"]) for r in records] == [
        ("digest", "failed"),
        ("failed", "delivered"),
        ("checkin", "delivered"),
    ]
    (ping,) = requests["hc-ping.com"]
    assert ping.url.path.endswith("/fail")
    assert records[0]["attempts"] == 0
    assert records[0]["receipt_id"] is None
    assert records[0]["failure_kind"] == kind


def test_an_unopenable_cache_still_sends_the_email_as_not_current(
    monkeypatch: pytest.MonkeyPatch,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_path.write_bytes(b"this is not a duckdb database file" * 64)
    report = saturday_report()
    requests = install_router(monkeypatch)

    notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    (request,) = requests["api.resend.com"]
    sent = _sent_email(request)
    assert "bets unknown" in sent["subject"]
    for part in (sent["html"], sent["text"]):
        assert digest.STALE_NOTICE in part
        assert "0 bets" not in part
    assert "0 bets" not in sent["subject"]
    out = capsys.readouterr().out
    (cache_line,) = [
        line for line in out.splitlines() if line.startswith("NOTIFY_DIGEST_CACHE=")
    ]
    assert cache_line.startswith("NOTIFY_DIGEST_CACHE= unreadable ")
    assert cache_line.split()[-1].isidentifier()
    assert f"NOTIFY_DIGEST= delivered {DUMMY_EMAIL_ID}" in out


def test_a_render_failure_records_a_failed_digest_and_pushes(
    monkeypatch: pytest.MonkeyPatch,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(monkeypatch)

    def _typo(_inputs: digest.DigestInputs) -> Any:
        msg = "'inputs' has no attribute 'betz'"
        raise jinja2.UndefinedError(msg)

    monkeypatch.setattr(digest, "render_digest", _typo)

    outcome = notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    _assert_one_build_failure(requests, read_notifications(), "build_UndefinedError")
    assert "NOTIFY_DIGEST= failed build_UndefinedError" in capsys.readouterr().out
    assert outcome.any_undelivered is True


def test_a_collect_failure_is_handled_the_same_way(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(monkeypatch)

    def _broken_getter(*_args: Any, **_kwargs: Any) -> Any:
        msg = "the bet list could not be read"
        raise RuntimeError(msg)

    monkeypatch.setattr(DataService, "get_bet_list", _broken_getter)

    outcome = notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    _assert_one_build_failure(requests, read_notifications(), "build_RuntimeError")
    assert outcome.any_undelivered is True
    # The read-only handle was closed on the failure path: the cache can be rebuilt in place.
    cache_path.unlink()
    build_cache(cache_path, bets=[], stamped_at=None)


def test_two_runs_on_one_day_use_two_idempotency_keys(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    lock = slate_lock(RUN_DATE)
    first, second = (
        saturday_report(
            status=STATUS_LOCK_PASSED,
            start=lock + timedelta(minutes=minutes),
            unpredicted_game_ids=SUNDAY_GAMES,
            run_id=None,
        )
        for minutes in (5, 40)
    )
    build_cache(cache_path, bets=week6_bets(), stamped_at=None)
    context = daily_context(email_secrets())
    titles = ("FAILED: 3 games not predicted",)
    requests = install_router(monkeypatch)

    for report in (first, second):
        delivery = digest.send_digest_if_due(
            report, context=context, failed_titles=titles, now=report.start
        )
        assert delivery is not None and delivery.delivered

    keys = [r.headers["Idempotency-Key"] for r in requests["api.resend.com"]]
    assert len(keys) == 2
    assert keys[0] != keys[1]

    retried = install_router(
        monkeypatch, resend=(lambda: httpx.Response(500), resend_ok())
    )
    delivery = digest.send_digest_if_due(
        first, context=context, failed_titles=titles, now=first.start
    )

    assert delivery is not None and delivery.attempts == 2
    retry_keys = [r.headers["Idempotency-Key"] for r in retried["api.resend.com"]]
    assert retry_keys == [keys[0], keys[0]]


def test_a_malformed_notification_log_records_a_failed_digest_and_pushes(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    log_path = notification_log.NOTIFICATION_LOG
    log_path.write_text("this line is not JSON\n", encoding="utf-8")
    requests = install_router(monkeypatch)

    outcome = notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    malformed, *appended = log_path.read_text(encoding="utf-8").splitlines()
    assert malformed == "this line is not JSON"
    _assert_one_build_failure(
        requests, [json.loads(line) for line in appended], "build_ValueError"
    )
    assert outcome.any_undelivered is True


# ---------------------------------------------------------------------------
# Plan 35-12 Task 1: every other R4 state, D-18, D-24, D-25 and the prohibitions
# ---------------------------------------------------------------------------

TOP_FAILED_TITLE = "FAILED: 3 games not predicted"
SHADOW_GAME = "2026_06_KC@BUF"
SCHEDULE_UNREADABLE_TEXT = "The recorded schedule could not be read"


def stamp_current_slate(path: Path, season: int, week: int) -> None:
    """Stamp the cache's in-season slate the way ``api.cache.stamp_current_slate`` stores it."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [
                CURRENT_SLATE_KEY,
                format_current_slate(season, week),
                datetime(2026, 10, 17),
            ],
        )
    finally:
        conn.close()


def game_states(text: str) -> dict[str, str]:
    """The text part's ``GAMES THIS WEEK`` lines as ``{matchup: state}``."""
    section = text.split("GAMES THIS WEEK\n", 1)[1].split("\n\n", 1)[0]
    states: dict[str, str] = {}
    for line in section.splitlines():
        label, _sep, state = line.partition(": ")
        states[label] = state
    return states


def status_header(html: str) -> str:
    """The HTML part's status header block."""
    return html.split("<div data-digest-status", 1)[1].split("</div>", 1)[0]


def text_status_header(text: str) -> str:
    """The text part's status header: everything before the bets section."""
    return text.split("LIVE BETS", 1)[0]


def matchup(game_id: str) -> str:
    """``AWAY @ HOME`` for a game id, as the email labels a game."""
    return game_id.rsplit("_", 1)[-1].replace("@", " @ ")


def send_one(
    report: RunReport,
    *,
    titles: tuple[str, ...] = (),
    context: SendContext | None = None,
) -> Any:
    """Run the digest step once for *report*; return its delivery."""
    return digest.send_digest_if_due(
        report,
        context=context if context is not None else daily_context(email_secrets()),
        failed_titles=titles,
        now=report.start + timedelta(minutes=20),
    )


def test_a_failed_saturday_names_the_failure_and_marks_sunday_not_predicted(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    """Tracer: a run_failed Saturday -> the digest step -> cache -> templates -> Resend."""
    report = saturday_report(
        status=STATUS_RUN_FAILED,
        unpredicted_game_ids=SUNDAY_GAMES,
        exception_class="RuntimeError",
    )
    # Stale: the cache was last populated two days before this run started.
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start - timedelta(days=2)
    )
    requests = install_router(monkeypatch)

    outcome = notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=20)
    )

    (failed_push,) = requests["ntfy.sh"]
    assert json.loads(failed_push.content)["title"] == TOP_FAILED_TITLE
    (request,) = requests["api.resend.com"]
    sent = _sent_email(request)
    assert sent["subject"] == "NFL Predict - 2026 week 6 - RUN FAILED - bets unknown"
    html, text = sent["html"], sent["text"]
    assert "<li data-digest-failed>3 games not predicted</li>" in status_header(html)
    assert "RUN FAILED:\n- 3 games not predicted" in text_status_header(text)
    states = game_states(text)
    assert [states[matchup(game_id)] for game_id in SUNDAY_GAMES] == [
        "not predicted"
    ] * len(SUNDAY_GAMES)
    assert html.count("data-digest-game-state>not predicted<") == len(SUNDAY_GAMES)
    assert digest.STALE_NOTICE in html and digest.STALE_NOTICE in text
    assert [step for step, _delivery in outcome.deliveries] == ["failed", "digest"]
    # Every message was delivered, so the check-in is a plain one, and it was recorded.
    (ping,) = requests["hc-ping.com"]
    assert not ping.url.path.endswith("/fail")
    assert read_notifications()[-1]["event_class"] == "checkin"
    assert read_notifications()[-1]["outcome"] == "delivered"


def test_a_zero_bet_week_says_so(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    suppressed_only = [row for row in week6_bets() if row["status"] == "suppressed"]
    build_cache(
        cache_path,
        bets=suppressed_only,
        stamped_at=report.start + timedelta(minutes=12),
    )
    requests = install_router(monkeypatch)

    send_one(report)

    sent = _sent_email(requests["api.resend.com"][0])
    assert sent["subject"] == "NFL Predict - 2026 week 6 - 0 bets - run clean"
    for part in (sent["html"], sent["text"]):
        assert digest.ZERO_BETS_LINE in part
        assert digest.STALE_NOTICE not in part


def test_an_absent_cache_shows_the_stale_notice_and_never_zero_bets(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    assert not cache_path.exists()
    requests = install_router(monkeypatch)

    send_one(saturday_report())

    sent = _sent_email(requests["api.resend.com"][0])
    assert "bets unknown" in sent["subject"]
    for part in (sent["subject"], sent["html"], sent["text"]):
        assert "0 bets" not in part
        assert digest.ZERO_BETS_LINE not in part
    for part in (sent["html"], sent["text"]):
        assert digest.STALE_NOTICE in part


def test_unreadable_cache_tables_read_unknown_never_zero(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    """35-REVIEW WR-03: a current cache whose tables cannot be read is unknown, never empty."""
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    conn = duckdb.connect(str(cache_path))
    try:
        for table in ("bet_list", "bet_week_results", "bet_list_corrections"):
            conn.execute(f"DROP TABLE {table}")
    finally:
        conn.close()
    requests = install_router(monkeypatch)

    send_one(report)

    sent = _sent_email(requests["api.resend.com"][0])
    assert sent["subject"] == "NFL Predict - 2026 week 6 - bets unknown - run clean"
    for part in (sent["html"], sent["text"]):
        assert "0 bets" not in part
        assert digest.STALE_NOTICE in part
        assert "nothing graded" not in part
        assert "not available" in part
        assert digest_sections.UNREADABLE_CORRECTIONS in part


def test_monday_reads_picked_sunday_evening(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(monkeypatch)

    send_one(report)

    sent = _sent_email(requests["api.resend.com"][0])
    assert game_states(sent["text"])[matchup(MONDAY_GAME)] == "picked Sunday evening"
    assert re.search(
        r"SF @ SEA</td>\s*<td[^>]*data-digest-game-state>picked Sunday evening<",
        sent["html"],
    )


def test_a_rerun_is_marked_update(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    context = daily_context(email_secrets())

    # A FAILED earlier digest is not a delivered one: the next send is not an update.
    install_router(monkeypatch, resend=(lambda: httpx.Response(500),))
    assert not send_one(report, context=context).delivered
    first = install_router(monkeypatch)
    assert send_one(report, context=context).delivered
    second = install_router(monkeypatch)
    assert send_one(report, context=context).delivered

    subjects = [
        _sent_email(requests["api.resend.com"][0])["subject"]
        for requests in (first, second)
    ]
    assert subjects == [
        "NFL Predict - 2026 week 6 - 3 bets - run clean",
        "NFL Predict - 2026 week 6 - 3 bets - run clean (update)",
    ]


def test_a_lock_passed_saturday_with_games_still_emails(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    """D-18: a lock-passed Saturday run with games tomorrow is a failed Saturday run."""
    lock = slate_lock(RUN_DATE)
    report = saturday_report(
        status=STATUS_LOCK_PASSED,
        start=lock + timedelta(minutes=5),
        unpredicted_game_ids=SUNDAY_GAMES,
        run_id=None,
    )
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start - timedelta(days=1)
    )
    requests = install_router(monkeypatch)

    notify_end_of_run(
        report, secrets=email_secrets(), now=report.start + timedelta(minutes=1)
    )

    (request,) = requests["api.resend.com"]
    sent = _sent_email(request)
    assert sent["subject"] == "NFL Predict - 2026 week 6 - RUN FAILED - bets unknown"
    assert "3 games not predicted" in status_header(sent["html"])
    states = game_states(sent["text"])
    assert [states[matchup(game_id)] for game_id in SUNDAY_GAMES] == [
        "not predicted"
    ] * len(SUNDAY_GAMES)


@pytest.mark.parametrize(
    "error",
    [ScheduleUnavailableError("schedule file missing"), OSError("disk read failed")],
    ids=["schedule_unavailable", "os_error"],
)
def test_an_unreadable_schedule_on_saturday_still_emails(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path, error: Exception
) -> None:
    """D-24: a Saturday that cannot read the recorded schedule still sends the weekly email."""

    def _unreadable() -> pd.DataFrame:
        raise error

    monkeypatch.setattr(digest, "LOAD_SCHEDULE", _unreadable)
    report = saturday_report(slate_season=None, slate_week=None)
    # The cache opens, is freshly populated and carries a stamped slate: the week comes from it,
    # and the bets still read unknown, because no schedule confirms the list is this week's.
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    stamp_current_slate(cache_path, 2026, 6)
    requests = install_router(monkeypatch)

    delivery = send_one(report)

    assert delivery is not None and delivery.delivered
    (request,) = requests["api.resend.com"]
    sent = _sent_email(request)
    assert sent["subject"] == "NFL Predict - 2026 week 6 - RUN FAILED - bets unknown"
    assert "recorded schedule could not be read" in status_header(sent["html"])
    for part in (sent["html"], sent["text"]):
        assert SCHEDULE_UNREADABLE_TEXT in part
        assert "0 bets" not in part

    # No cache at all, and the run selected no slate: the week is unknown. The first email of
    # this run date was delivered above, so this one is the same day's update.
    cache_path.unlink()
    requests = install_router(monkeypatch)
    send_one(report)
    sent = _sent_email(requests["api.resend.com"][0])
    assert sent["subject"] == (
        "NFL Predict - week unknown - RUN FAILED - bets unknown (update)"
    )

    # On any other weekday no email is attempted.
    lock = slate_lock(WEDNESDAY)
    wednesday = saturday_report(
        run_date_et=WEDNESDAY, start=lock - timedelta(hours=1), lock=lock
    )
    requests = install_router(monkeypatch)
    assert send_one(wednesday) is None
    assert requests == {"api.resend.com": [], "ntfy.sh": [], "hc-ping.com": []}


def test_a_playoff_week_keeps_its_number(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    """D-25: a playoff week reads ``2026 week 19``, never a round name."""
    saturday = date(2027, 1, 16)
    playoff = schedule_frame(
        (
            ("2026_19_HOU@KC", 19, datetime(2027, 1, 16, 16, 30)),
            ("2026_19_GB@DET", 19, datetime(2027, 1, 17, 15, 0)),
            ("2026_19_LA@PHI", 19, datetime(2027, 1, 17, 18, 30)),
        )
    )
    monkeypatch.setattr(digest, "LOAD_SCHEDULE", lambda: playoff)
    lock = slate_lock(saturday)
    report = RunReport(
        run_date_et=saturday,
        start=lock - timedelta(hours=1),
        lock=lock,
        status=STATUS_SUCCESS,
        unpredicted_game_ids=(),
        slate_season=2026,
        slate_week=19,
    )
    requests = install_router(monkeypatch)

    send_one(report, context=SendContext("daily_run", saturday, email_secrets()))

    sent = _sent_email(requests["api.resend.com"][0])
    assert sent["subject"].startswith("NFL Predict - 2026 week 19 - ")
    assert ">NFL Predict - 2026 week 19</h1>" in sent["html"]
    assert sent["text"].splitlines()[0] == "NFL Predict - 2026 week 19"


def test_no_clv_in_subject_or_header(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report(
        status=STATUS_RUN_FAILED, unpredicted_game_ids=SUNDAY_GAMES
    )
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(monkeypatch)

    send_one(report, titles=(TOP_FAILED_TITLE, "FAILED: closing line capture"))
    send_one(saturday_report())

    assert len(requests["api.resend.com"]) == 2
    for request in requests["api.resend.com"]:
        sent = _sent_email(request)
        for header in (
            sent["subject"],
            status_header(sent["html"]),
            text_status_header(sent["text"]),
        ):
            assert "CLV" not in header
            assert "closing line value" not in header.lower()


def test_the_not_advice_line_is_last_and_the_email_is_plain(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_router(monkeypatch)

    send_one(report)

    sent = _sent_email(requests["api.resend.com"][0])
    html, text = sent["html"], sent["text"]
    expected_last = f"{NOT_ADVICE_KEY}: {NOT_ADVICE_LINE}"
    assert text.rstrip().splitlines()[-1] == expected_last
    html_lines = [
        line.strip()
        for line in re.sub(r"<[^>]+>", "", html).splitlines()
        if line.strip()
    ]
    assert html_lines[-1] == expected_last
    assert html.isascii() and text.isascii()
    lowered = html.lower()
    assert "<img" not in lowered
    assert re.search(r"<a[\s>]", lowered) is None
    assert re.search(r"[a-z][a-z0-9+.-]*://", lowered) is None


def test_a_shadow_row_never_reaches_the_email(
    monkeypatch: pytest.MonkeyPatch, cache_path: Path
) -> None:
    report = saturday_report()
    shadow = live_bet(
        SHADOW_GAME,
        arm="shadow",
        target="wp",
        bet_side="home",
        line=None,
        per_bet_ev=0.09,
        ev_tier="high",
    )
    build_cache(
        cache_path,
        bets=[*week6_bets(), shadow],
        stamped_at=report.start + timedelta(minutes=12),
    )
    requests = install_router(monkeypatch)

    send_one(report)

    sent = _sent_email(requests["api.resend.com"][0])
    assert sent["subject"] == "NFL Predict - 2026 week 6 - 3 bets - run clean"
    bets_table = sent["html"].split("data-digest-bets>", 1)[1].split("</table>", 1)[0]
    assert matchup(SHADOW_GAME) not in bets_table
    assert bets_table.count("<tr data-digest-row>") == 3
    assert "+9.00%" not in sent["html"] and "+9.00%" not in sent["text"]
