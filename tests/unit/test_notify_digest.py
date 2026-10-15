"""The Saturday digest email's send path (Phase 35 NOTIF-02, NOTIF-03, NOTIF-05, SPEC R4/R8, D-07, D-19).

* The digest run is the run whose tomorrow is its week's Sunday; a week with no Sunday game falls
  back to its last slate day (SPEC R4 boundary).
* A clean Saturday sends ONE Resend request, read from the web cache through ``DataService``, with
  the D-07 subject and an Idempotency-Key, and its receipt lands in the notification log.
* A day that is not a digest day sends no email.
* An email still undelivered after its attempts sends one normal-priority ``FAILED: weekly email``
  push naming the failure kind (SPEC R8).

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
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import httpx
import pandas as pd
import pytest

from api.cache import (
    BET_LIST_COLUMNS,
    BET_TRACKER_BLOCK_COLUMNS,
    BET_WEEK_RESULTS_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    bet_list_populated_at_key,
    materialize_bet_list,
    materialize_bet_tracker_blocks,
    materialize_bet_week_results,
)
from api.presentation import NOT_ADVICE_LINE
from notify import digest, transport
from notify.end_of_run import notify_end_of_run
from notify.events import PRIORITY_NORMAL
from notify.log import read_notifications
from notify.report import STATUS_SUCCESS, RunReport
from pipeline.daily_steps import slate_lock
from tests.unit.test_notify_channels import DUMMY_EMAIL_ID, email_secrets, resend_ok
from tests.unit.test_notify_messages import live_bet
from tests.unit.test_notify_transport import (
    DUMMY_RECEIPT_ID,
    RUN_DATE,
    install_sleep_recorder,
    ntfy_ok,
)
from utils.current_slate import prepare_schedule
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
    """Answer Resend and ntfy requests from their own reply lists; return the requests per host.

    The last reply of each list repeats once the list runs out.
    """
    replies = {"api.resend.com": resend, "ntfy.sh": ntfy}
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

    (record,) = read_notifications()
    assert record["event_class"] == "digest"
    assert record["outcome"] == "delivered"
    assert record["receipt_id"] == DUMMY_EMAIL_ID
    assert [step for step, _delivery in outcome.deliveries] == ["digest"]
    assert outcome.any_undelivered is False
    assert f"NOTIFY_DIGEST= delivered {DUMMY_EMAIL_ID}" in capsys.readouterr().out


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

    assert requests == {"api.resend.com": [], "ntfy.sh": []}
    assert read_notifications() == []
    assert outcome.deliveries == []
    assert "NOTIFY_DIGEST= not a digest day" in capsys.readouterr().out


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
    ]
    assert records[0]["failure_kind"] == "http_500"
