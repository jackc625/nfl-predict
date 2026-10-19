"""No notification secret leaks anywhere the SPEC names (Phase 35 must-NOT; T-35-34, T-35-45).

* A full notifying run -- a failing Saturday with appended bets, a digest-day schedule and a fixture
  cache -- and the TEST command, with providers that answer and with providers whose connection
  errors embed the secrets, put no secret in the notification log, stdout, stderr, the logging
  records or any request field other than the one that must carry it.
* A failing ``run_daily`` whose closing-trigger regeneration, settle pass, orchestrator and ledger
  sync each raise an error embedding every secret writes none of them to the daily-run records,
  the ledger run log, the notification log, stdout, stderr, the daily run's or the ledger runner's
  logger, or any request field other than its one legitimate carrier. This traverses
  ``run_daily`` -> ``_record_run_failure`` and every side step's failure path, which the direct
  ``notify_end_of_run`` test never reaches (review HIGH, 35-08 + 35-15).
* The values configured in ``.env`` (ntfy topic, Resend key, ping URL and its uuid, healthchecks
  API key, and the two website-push secrets of Phase 36, ``VERCEL_TOKEN`` and
  ``VERCEL_BYPASS_SECRET``) are in no committed file and no file under ``logs/`` (which includes
  the push log ``logs/site_pushes.jsonl``); the recipient address is in neither the notification
  log nor the daily-run records. Each value is searched as UTF-8 AND UTF-16-LE bytes, because
  ``deployment/*.xml`` is UTF-16. A hit names the file, the line and the ENV NAME -- never the
  value. The recipient is not scanned in committed files: it is the owner's own address,
  committed on purpose as the ledger's git identity.

The two hermetic tests use dummy values only. Every request goes through
``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport`` (any other host fails the
test), every retry delay through a recording ``SLEEP``, and the ledger run log and the notification
log stay under the ``tests/conftest.py`` quarantine. The configured-value test reads the real
``.env`` in memory only, compares bytes, and never prints, asserts on or writes a value.

Run this module:  uv run python -m pytest tests/unit/test_notify_secrets.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from pydantic import SecretStr

import scripts.daily_lock_pipeline as daily
from conf.settings import Settings
from notify import digest, end_of_run
from notify import log as notification_log
from notify.end_of_run import notify_end_of_run
from notify.report import STATUS_RUN_FAILED
from notify.secrets import NotifySecrets, load_secrets
from pipeline.daily_steps import slate_lock
from scripts import notify_test
from tests.unit.test_notify_channels import resend_ok
from tests.unit.test_notify_digest import (
    WEEK6,
    build_cache,
    saturday_report,
    schedule_frame,
    stamp_current_slate,
    week6_bets,
)
from tests.unit.test_notify_transport import install_sleep_recorder, ntfy_ok
from tests.unit.test_notify_watcher import REPO_ROOT, install_hosts, ping_ok

# Dummy values only, each distinctive so a leak is findable. The topic has the SPEC's 32 characters.
DUMMY_TOPIC = "leaktopicZ0123456789abcdefghijkl"
DUMMY_RESEND_KEY = "re_leakKeyZ0123456789abcdefghij"
DUMMY_RECIPIENT = "leak.owner@example.invalid"
DUMMY_PING_UUID = "9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b"
DUMMY_PING_URL = f"https://hc-ping.com/{DUMMY_PING_UUID}"
DUMMY_HC_API_KEY = "hcw_leakReadWriteKeyZ0123456789"
DUMMY_ODDS_KEY = "oddsLeakKeyZ0123456789abcdef"
DUMMY_SECRET_KEY = "secretLeakKeyZ0123456789abcdefghijkl"

DUMMIES: dict[str, str] = {
    "NTFY_TOPIC": DUMMY_TOPIC,
    "RESEND_API_KEY": DUMMY_RESEND_KEY,
    "DIGEST_TO_ADDRESS": DUMMY_RECIPIENT,
    "HEALTHCHECKS_PING_URL": DUMMY_PING_URL,
    "HEALTHCHECKS_PING_URL uuid": DUMMY_PING_UUID,
    "HEALTHCHECKS_API_KEY": DUMMY_HC_API_KEY,
    "ODDS_API_KEY": DUMMY_ODDS_KEY,
    "SECRET_KEY": DUMMY_SECRET_KEY,
}

# A Wednesday: no digest is due, so this run's sends are the failed pushes and the check-in.
RUN_DATE = date(2026, 9, 30)
TOMORROW_GAME = "2026_W04_SEA@ARI"

# The configured-value scan (T-35-34): env name -> Settings field. The recipient is scanned in the
# logs only (see the module docstring).
COMMITTED_SECRET_FIELDS: dict[str, str] = {
    "NTFY_TOPIC": "ntfy_topic",
    "RESEND_API_KEY": "resend_api_key",
    "HEALTHCHECKS_PING_URL": "healthchecks_ping_url",
    "HEALTHCHECKS_API_KEY": "healthchecks_api_key",
}
# The website-push secrets (Phase 36 D-01, D-15). Not notification secrets, so they are read
# straight from Settings; scanned in committed files and logs like the set above.
VERCEL_SECRET_FIELDS: dict[str, str] = {
    "VERCEL_TOKEN": "vercel_token",
    "VERCEL_BYPASS_SECRET": "vercel_bypass_secret",
}
# A value this short could be a placeholder matching everywhere; none of the real five is.
MIN_SCANNED_LENGTH = 12
RECIPIENT_LOGS = ("logs/notifications.jsonl", "logs/daily_lock_runs.jsonl")


def dummy_secrets() -> NotifySecrets:
    """Every notification secret plus the two scrub-only values, all dummies."""
    return NotifySecrets(
        ntfy_topic=SecretStr(DUMMY_TOPIC),
        resend_api_key=SecretStr(DUMMY_RESEND_KEY),
        digest_to_address=SecretStr(DUMMY_RECIPIENT),
        healthchecks_ping_url=SecretStr(DUMMY_PING_URL),
        healthchecks_api_key=SecretStr(DUMMY_HC_API_KEY),
        other_secret_values=(DUMMY_ODDS_KEY, DUMMY_SECRET_KEY),
    )


def leaky_message(step: str) -> str:
    """An error message carrying every dummy value, bare and inside an ``https://`` URL."""
    bare = " ".join(DUMMIES.values())
    urls = " ".join(
        f"https://leak.example/{step}?v={value}" for value in DUMMIES.values()
    )
    return f"{step} failed: {bare} {urls}"


def leaks(texts: Mapping[str, str]) -> list[str]:
    """``<where>: <env name>`` for every dummy value found in any of *texts*."""
    return [
        f"{where}: {name}"
        for where, text in texts.items()
        for name, value in DUMMIES.items()
        if value in text
    ]


def request_texts(requests: Iterable[httpx.Request]) -> dict[str, str]:
    """Each request's URL, headers and body, with its ONE legitimate secret carrier removed.

    The ntfy JSON ``topic``; Resend's ``Authorization`` header and ``to`` list; the ping
    request's URL; the management API's ``X-Api-Key`` header.
    """
    texts: dict[str, str] = {}
    for number, request in enumerate(requests, start=1):
        host = request.url.host
        url = str(request.url)
        headers = {name.lower(): value for name, value in request.headers.items()}
        body = request.content.decode("utf-8", "replace")
        if host == "ntfy.sh":
            payload = json.loads(body)
            payload.pop("topic")
            body = json.dumps(payload)
        elif host == "api.resend.com":
            headers.pop("authorization")
            payload = json.loads(body)
            payload.pop("to")
            body = json.dumps(payload)
        elif host == "hc-ping.com":
            url = request.url.scheme + "://" + host
        elif host == "healthchecks.io":
            headers.pop("x-api-key")
        texts[f"request {number} ({host})"] = " ".join((url, json.dumps(headers), body))
    return texts


def file_text(path: Path) -> str:
    """*path*'s text, or empty when the file was never written."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


class LoggerRecorder:
    """Stands in for a module's structlog logger, which caches on first use: records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Callable[..., None]:
        return lambda event, *args, **kwargs: self.calls.append(
            (level, event, {"args": args, **kwargs})
        )


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No retry delay in this module actually sleeps."""
    return install_sleep_recorder(monkeypatch)


# ---------------------------------------------------------------------------
# A full notifying run, and the TEST command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("providers", ["answer", "unreachable"])
def test_a_full_notifying_run_leaks_no_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    providers: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    schedule = schedule_frame(WEEK6)
    monkeypatch.setattr(digest, "LOAD_SCHEDULE", lambda: schedule)
    cache = tmp_path / "web_cache.duckdb"
    monkeypatch.setattr(digest, "WEB_CACHE_PATH", cache)
    report = saturday_report(
        status=STATUS_RUN_FAILED,
        exception_class="RuntimeError",
        exception_text=leaky_message("orchestrator"),
        unpredicted_game_ids=None,
        appended_rows=[row for row in week6_bets() if row["status"] == "live"],
        season_window="in",
    )
    build_cache(
        cache, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    stamp_current_slate(cache, 2026, 6)
    if providers == "answer":
        replies: dict[str, tuple[Any, ...]] = {
            "ntfy.sh": (ntfy_ok(),),
            "api.resend.com": (resend_ok(),),
            "hc-ping.com": (ping_ok(),),
        }
    else:
        down = httpx.ConnectError(
            f"cannot reach {DUMMY_PING_URL} for {DUMMY_TOPIC} with {DUMMY_RESEND_KEY}"
        )
        replies = dict.fromkeys(("ntfy.sh", "api.resend.com", "hc-ping.com"), (down,))
    requests = install_hosts(monkeypatch, replies)
    secrets = dummy_secrets()

    notify_end_of_run(report, secrets=secrets, now=report.start + timedelta(minutes=20))
    monkeypatch.setattr(notify_test, "LOAD_SECRETS", lambda: secrets)
    monkeypatch.setattr(notify_test, "CACHE_PATH", cache)
    notify_test.main([])

    records = notification_log.read_notifications()
    sources = {record["source"] for record in records}
    assert sources == {"daily_run", "test_command"}, "both senders must have recorded"
    expected_hosts = {"ntfy.sh", "api.resend.com", "hc-ping.com"}
    assert {request.url.host for request in requests} == expected_hosts
    if providers == "unreachable":
        assert {record["failure_kind"] for record in records} == {
            "transport_ConnectError"
        }

    captured = capfd.readouterr()
    logged = [f"{record.getMessage()} {record.args!r}" for record in caplog.records]
    texts = {
        "notification log": file_text(notification_log.NOTIFICATION_LOG),
        "stdout": captured.out,
        "stderr": captured.err,
        "logging records": "\n".join([caplog.text, *logged]),
        **request_texts(requests),
    }
    assert leaks(texts) == []


# ---------------------------------------------------------------------------
# A failing run_daily: every side step's failure path, then the orchestrator's
# ---------------------------------------------------------------------------


def _wednesday_schedule() -> pd.DataFrame:
    """Silver ``games`` with one game tomorrow (Thursday night) and one on Sunday."""
    rows = [
        (TOMORROW_GAME, 4, "2026-10-02 00:15", "ARI", "SEA"),  # 20:15 ET Thursday
        ("2026_W04_NYG@DAL", 4, "2026-10-04 17:00", "DAL", "NYG"),
    ]
    return pd.DataFrame(
        {
            "game_id": [row[0] for row in rows],
            "season": 2026,
            "week": [row[1] for row in rows],
            "kickoff_et": [pd.Timestamp(row[2], tz="UTC") for row in rows],
            "home_team": [row[3] for row in rows],
            "away_team": [row[4] for row in rows],
        }
    )


def _raises(step: str) -> Callable[..., Any]:
    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(leaky_message(step))

    return _raise


def test_a_failing_run_daily_leaks_no_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import pipeline.steps as steps_mod
    from forward_ledger import closing_schedule, cutover, run_log, runner
    from pipeline import orchestrator

    # The real end-of-run notifier, fed the dummy secrets; an in-season day.
    monkeypatch.setattr(daily, "NOTIFY_END_OF_RUN", daily._notify_end_of_run)
    monkeypatch.setattr(end_of_run, "LOAD_SECRETS", dummy_secrets)
    monkeypatch.setattr(daily, "SEASON_WINDOW", lambda _day: "in")
    records = tmp_path / "daily_lock_runs.jsonl"
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", records)
    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", True)

    # A slate of one game tomorrow, selected from fixture silver games.
    monkeypatch.setattr(daily, "_refresh_schedule", lambda *_a, **_k: 2026)
    monkeypatch.setattr(
        "data.storage.load_dataframe", lambda *_a, **_k: _wednesday_schedule()
    )
    monkeypatch.setattr(daily, "_require_slate_is_current_week", lambda *_a: None)
    monkeypatch.setattr(daily, "build_daily_step_registry", lambda _slate: [])
    monkeypatch.setattr(steps_mod, "_predictions_output_dir", lambda: tmp_path)

    # Four failure paths, each raising an error that carries every secret.
    monkeypatch.setattr(
        closing_schedule, "register_closing_triggers", _raises("closing_triggers")
    )
    monkeypatch.setattr(runner, "settle_ledger", _raises("settle"))
    monkeypatch.setattr(runner, "sync_after_run", _raises("sync"))

    class _RaisingOrchestrator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self) -> Any:
            raise RuntimeError(leaky_message("orchestrator"))

    monkeypatch.setattr(orchestrator, "FridayPipeline", _RaisingOrchestrator)

    daily_logger, runner_logger = LoggerRecorder(), LoggerRecorder()
    monkeypatch.setattr(daily, "logger", daily_logger)
    monkeypatch.setattr(runner, "logger", runner_logger)
    requests = install_hosts(
        monkeypatch,
        {
            "ntfy.sh": (ntfy_ok(),),
            "api.resend.com": (resend_ok(),),
            "hc-ping.com": (ping_ok(),),
            "healthchecks.io": (lambda: httpx.Response(200, json={}),),
        },
    )

    start = slate_lock(RUN_DATE).astimezone(UTC) - timedelta(hours=1)
    with pytest.raises(RuntimeError) as raised:
        daily.run_daily(RUN_DATE, start=start, dry_run=False)
    assert str(raised.value).startswith("orchestrator failed:")

    # Positive controls: the scan below looked at real output.
    (record,) = [
        json.loads(line) for line in file_text(records).splitlines() if line.strip()
    ]
    assert record["outcome"] == "run_failed"
    assert record["error"] == "RuntimeError"
    pushed = [
        json.loads(request.content)["title"]
        for request in requests
        if request.url.host == "ntfy.sh"
    ]
    assert any(title.startswith("FAILED") for title in pushed)
    assert "hc-ping.com" in {request.url.host for request in requests}
    assert file_text(run_log.LEDGER_RUN_LOG), "the side steps recorded nothing"

    captured = capfd.readouterr()
    texts = {
        "daily-run records": file_text(records),
        "ledger run log": file_text(run_log.LEDGER_RUN_LOG),
        "notification log": file_text(notification_log.NOTIFICATION_LOG),
        "stdout": captured.out,
        "stderr": captured.err,
        "daily run logger": repr(daily_logger.calls),
        "ledger runner logger": repr(runner_logger.calls),
        **request_texts(requests),
    }
    assert leaks(texts) == []


# ---------------------------------------------------------------------------
# The configured values: in no committed file and no log
# ---------------------------------------------------------------------------


def _configured(value: SecretStr | None) -> SecretStr | None:
    """*value* when it is long enough to scan for, else None. Never unwrapped outside a scan."""
    if value is None or len(value.get_secret_value().strip()) < MIN_SCANNED_LENGTH:
        return None
    return SecretStr(value.get_secret_value().strip())


def _configured_values() -> tuple[dict[str, SecretStr], dict[str, SecretStr]]:
    """The configured secrets as ``{env name: SecretStr}``: the committed-file set, the logs-only set.

    Read in memory from the real ``.env`` through ``Settings`` and ``load_secrets``; kept wrapped
    so no repr, assertion or traceback can show one.
    """
    settings = Settings(_env_file=REPO_ROOT / ".env")
    secrets = load_secrets(settings)
    committed: dict[str, SecretStr] = {}
    for env_name, field_name in COMMITTED_SECRET_FIELDS.items():
        value = _configured(getattr(secrets, field_name))
        if value is not None:
            committed[env_name] = value
    for env_name, field_name in VERCEL_SECRET_FIELDS.items():
        value = _configured(getattr(settings, field_name))
        if value is not None:
            committed[env_name] = value
    ping_url = committed.get("HEALTHCHECKS_PING_URL")
    if ping_url is not None:
        uuid = _configured(
            SecretStr(ping_url.get_secret_value().rstrip("/").rsplit("/", 1)[-1])
        )
        if uuid is not None:
            committed["HEALTHCHECKS_PING_URL uuid"] = uuid
    recipient = _configured(secrets.digest_to_address)
    logs_only = {"DIGEST_TO_ADDRESS": recipient} if recipient is not None else {}
    return committed, logs_only


def _first_hit(
    paths: Iterable[Path], values: Mapping[str, SecretStr]
) -> tuple[str, int, str] | None:
    """The first ``(file, 1-based line, env name)`` whose bytes hold a value, or None.

    Each value is searched as UTF-8 and as UTF-16-LE bytes (``deployment/*.xml`` is UTF-16).
    """
    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for env_name, secret in values.items():
            for encoding in ("utf-8", "utf-16-le"):
                at = data.find(secret.get_secret_value().encode(encoding))
                if at >= 0:
                    where = (
                        path.relative_to(REPO_ROOT).as_posix()
                        if path.is_relative_to(REPO_ROOT)
                        else path.as_posix()
                    )
                    line = data.count("\n".encode(encoding), 0, at) + 1
                    return where, line, env_name
    return None


def test_a_utf16_file_holding_a_value_is_found(tmp_path: Path) -> None:
    dummy = "dummy-utf16-value-789"
    xml = tmp_path / "task.xml"
    xml.write_bytes(f"<Task>\n  <Arg>{dummy}</Arg>\n</Task>\n".encode("utf-16-le"))

    hit = _first_hit([xml], {"VERCEL_TOKEN": SecretStr(dummy)})

    assert hit == (xml.as_posix(), 2, "VERCEL_TOKEN")


def test_no_configured_secret_is_committed_or_logged() -> None:
    committed, logs_only = _configured_values()
    if not committed and not logs_only:
        pytest.skip("no notification secret configured in .env yet")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.split("\0")
    logs_dir = REPO_ROOT / "logs"
    scanned = [
        *(REPO_ROOT / name for name in tracked if name),
        *sorted(logs_dir.glob("*.jsonl")),
        *sorted(logs_dir.glob("*.log")),
    ]
    hit = _first_hit(scanned, committed) or _first_hit(
        (REPO_ROOT / name for name in RECIPIENT_LOGS), logs_only
    )
    if hit is not None:
        where, line, env_name = hit
        pytest.fail(f"{where}:{line} contains the value of {env_name}", pytrace=False)
