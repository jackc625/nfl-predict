"""The one website-push routine, driven end to end against fakes (Phase 36, D-05, SPEC R5/R6).

WHAT THESE TESTS PROVE
----------------------
``pipeline.site_push.push_site`` is run for real -- the switch, the secrets, the publish lock, the
refusals, the stamp on a copy of the cache, the digest, one deploy and the read-back -- with only
its two outside edges faked:

* the Vercel CLI is :class:`FakeCli`, injected through ``site_push.RUN_CLI``;
* the hosted site is :class:`ReadbackHost`, an ``httpx.MockTransport`` handler that answers
  ``/health`` by reading the tmp cache file AT REQUEST TIME, so a success can only come from the
  values the push really stamped and the bytes it really left on disk; it answers each page the
  push then checks with just enough to carry that route's marker (:func:`page_answer`).

``tests/conftest.py`` refuses every real Vercel request and every un-injected CLI spawn, and points
the push log, the publish lock, the switch and the CLI's config directory at per-test paths outside
the repository (the switch absent). Plan 36-07's tests reuse :class:`FakeCli`,
:class:`ReadbackHost`, :func:`push_harness` and the ``switched_on`` fixture.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import duckdb
import httpx
import pytest
from pydantic import SecretStr

from api.cache import populate_cache
from api.cache_reader import open_cache_readonly
from api.cache_schema import (
    DATA_FORMAT_VERSION,
    DATA_FORMAT_VERSION_KEY,
    PUBLISH_ID_KEY,
    PUSH_TIME_KEY,
)
from pipeline import cache_lock, site_push, steps
from scripts import fetch_hosted_routes
from scripts import populate_cache as populate_cache_script
from utils.file_lock import FileLockHeldError

if TYPE_CHECKING:
    from conf.settings import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DUMMY_TOKEN = "dummy-vercel-token-0000"
DUMMY_BYPASS = "dummy-bypass-secret-1111"
DEPLOYMENT_ID = "dpl_Abc123"
DEPLOYMENT_URL = "https://nfl-predict-abc123-team.vercel.app"
DEPLOY_STDOUT = (
    '{"status": "ok", "deployment": {"id": "dpl_Abc123", '
    '"url": "https://nfl-predict-abc123-team.vercel.app"}}'
)
PUSH_NOW = datetime(2026, 10, 18, 21, 5, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared helpers (Plan 36-07 reuses these)
# ---------------------------------------------------------------------------


@dataclass
class FakeCli:
    """Stands in for ``site_push.RUN_CLI``: records every call and answers like the CLI.

    Attributes:
        returncode: The exit code to answer with.
        stdout: The standard output to answer with.
        raises: An exception to raise instead of answering (a timeout, a missing binary).
        calls: One ``{"argv", "env", "cwd", "timeout"}`` entry per call.
    """

    returncode: int = 0
    stdout: str = DEPLOY_STDOUT
    raises: BaseException | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(
            {
                "argv": [str(item) for item in argv],
                "env": dict(kwargs.get("env") or {}),
                "cwd": kwargs.get("cwd"),
                "timeout": kwargs.get("timeout"),
            }
        )
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=self.stdout, stderr=""
        )


def file_sha256(path: Path) -> str:
    """The SHA-256 of *path*'s bytes, as lowercase hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_meta(path: Path) -> dict[str, str]:
    """Every ``cache_meta`` key and value of the cache at *path* (the connection is closed)."""
    conn = open_cache_readonly(path)
    try:
        return dict(conn.execute("SELECT key, value FROM cache_meta").fetchall())
    finally:
        conn.close()


def page_answer(path: str) -> httpx.Response:
    """A page route answered as the live check expects: its title, a fragment, or JSON."""
    if path.startswith("/fragments/"):
        return httpx.Response(200, html="<div>fragment</div>")
    if path.startswith("/api/export/"):
        return httpx.Response(200, json=[])
    (route,) = [route for route in fetch_hosted_routes.routes("") if route.path == path]
    return httpx.Response(200, html=f"<html><title>{route.marker}</title></html>")


@dataclass
class ReadbackHost:
    """The hosted site: ``/health`` served from the tmp cache file at request time, and its pages.

    By default it answers what a correct deployment of the cache would serve: the cache's own
    ``publish_id`` and the SHA-256 of the file at ``/health``, and :func:`page_answer` for every
    other path. *answer* replaces that with a response of its own, for the failure cases.

    Attributes:
        cache_path: The tmp cache the host "serves".
        answer: When set, called with the request and returning the response to send.
        requests: Every request the host received.
    """

    cache_path: Path
    answer: Any = None
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.answer is not None:
            return self.answer(request)
        return self.default(request)

    def default(self, request: httpx.Request) -> httpx.Response:
        """What a correct deployment of the tmp cache answers."""
        if request.url.path != "/health":
            return page_answer(request.url.path)
        return httpx.Response(
            200,
            json={
                "publish_id": cache_meta(self.cache_path).get(PUBLISH_ID_KEY),
                "cache_sha256": file_sha256(self.cache_path),
            },
        )


@dataclass
class PushHarness:
    """What one test's push runs against: the cache, the fake CLI, the host and the sleeps."""

    cache_path: Path
    cli: FakeCli
    host: ReadbackHost
    sleeps: list[float]

    def records(self) -> list[dict[str, Any]]:
        """The push-log records written so far."""
        return site_push.read_site_pushes()


def build_cache(tmp_path: Path) -> Path:
    """A real cache, built by the writer over empty inputs, under tmp_path only."""
    empty = tmp_path / "empty_inputs"
    empty.mkdir(exist_ok=True)
    db_path = tmp_path / "web_cache.duckdb"
    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
    )
    return db_path


def dummy_secrets(
    *, token: str | None = DUMMY_TOKEN, bypass: str | None = DUMMY_BYPASS
) -> site_push.PushSecrets:
    """Dummy push secrets; a None stands for a value missing from ``.env``."""
    return site_push.PushSecrets(
        vercel_token=None if token is None else SecretStr(token),
        bypass_secret=None if bypass is None else SecretStr(bypass),
    )


def push_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PushHarness:
    """Wire a real tmp cache, the fake CLI, the request-time host, dummy secrets and a sleep log.

    The read-back client is the PRODUCTION client (``site_push._default_client``) with only its
    transport swapped, so its redirect and timeout settings are the ones under test.
    """
    cache_path = build_cache(tmp_path)
    cli = FakeCli()
    host = ReadbackHost(cache_path)
    sleeps: list[float] = []
    monkeypatch.setattr(site_push, "RUN_CLI", cli)
    monkeypatch.setattr(
        site_push,
        "CLIENT_FACTORY",
        lambda: site_push._default_client(transport=httpx.MockTransport(host)),
    )
    monkeypatch.setattr(site_push, "SLEEP", sleeps.append)
    monkeypatch.setattr(site_push, "LOAD_PUSH_SECRETS", dummy_secrets)
    return PushHarness(cache_path=cache_path, cli=cli, host=host, sleeps=sleeps)


def no_stylesheet_problem() -> str | None:
    """An injected stylesheet check that finds the sheet in step."""
    return None


def rebuild_must_not_run() -> None:
    """An injected rebuild for a tree whose format already matches: calling it is a failure."""
    pytest.fail("the push rebuilt a cache whose data format already matched")


def git(repo: Path, *args: str) -> str:
    """Run git in *repo* (no signing, a fixed identity) and return its stripped output."""
    completed = subprocess.run(
        [
            "git",
            *("-C", str(repo)),
            *("-c", "user.name=test", "-c", "user.email=test@example.invalid"),
            *("-c", "commit.gpgsign=false"),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def assert_no_secret_in(*texts: str) -> None:
    """Neither dummy secret appears in any of *texts*."""
    for text in texts:
        assert DUMMY_TOKEN not in text
        assert DUMMY_BYPASS not in text


@pytest.fixture
def switched_on() -> Path:
    """The (quarantined) D-23 switch file, written so the push is switched on."""
    marker = site_push.PUSH_ENABLED_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"switched_on": "test"}\n', encoding="utf-8")
    return marker


# ---------------------------------------------------------------------------
# Task 1 (tracer): one push, end to end
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("switched_on")
def test_one_push_succeeds_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = push_harness(monkeypatch, tmp_path)

    outcome = site_push.push_site(
        source="manual",
        cache_path=harness.cache_path,
        rebuild=rebuild_must_not_run,
        check_stylesheet=no_stylesheet_problem,
        now=PUSH_NOW,
    )

    assert (outcome.outcome, outcome.readback) == ("succeeded", "matched")
    assert outcome.reason is None
    meta = cache_meta(harness.cache_path)
    assert uuid.UUID(outcome.publish_id).version == 4
    assert meta[PUBLISH_ID_KEY] == outcome.publish_id
    assert meta[PUSH_TIME_KEY] == PUSH_NOW.isoformat()

    records = harness.records()
    assert len(records) == 1
    record = records[0]
    assert set(record) == set(site_push.SITE_PUSH_FIELDS)
    assert record["outcome"] == "succeeded"
    assert record["source"] == "manual"
    assert record["publish_id"] == outcome.publish_id
    assert record["deployment_id"] == DEPLOYMENT_ID
    assert record["deployment_url"] == DEPLOYMENT_URL
    assert record["local_sha256"] == file_sha256(harness.cache_path)
    assert record["readback"] == "matched"
    # The deployed code is this repository's HEAD (dirty or not, depending on the working tree).
    head = git(REPOSITORY_ROOT, "rev-parse", "HEAD")
    assert record["code_commit"] in (head, head + site_push.DIRTY_SUFFIX)

    assert len(harness.cli.calls) == 1
    call = harness.cli.calls[0]
    argv = call["argv"]
    assert argv[:2] == [str(site_push.NODE_EXE), str(site_push.VERCEL_CLI_JS)]
    for flag in ("deploy", "--prod", "--yes", "--json"):
        assert flag in argv
    config_at = argv.index("--global-config")
    assert argv[config_at + 1] == str(site_push.VERCEL_GLOBAL_CONFIG)
    assert DUMMY_TOKEN not in argv
    assert "--token" not in argv
    assert "-t" not in argv
    assert call["env"]["VERCEL_TOKEN"] == DUMMY_TOKEN

    # One matching /health, then one GET of each page the push checks, in order.
    assert [request.url.path for request in harness.host.requests] == [
        "/health",
        *site_push.SMOKE_PATHS,
    ]
    for request in harness.host.requests:
        assert str(request.url) == f"{site_push.PRODUCTION_URL}{request.url.path}"
        assert request.url.query == b""
        assert request.headers[site_push.BYPASS_HEADER] == DUMMY_BYPASS
        carried = [
            name for name, value in request.headers.items() if value == DUMMY_BYPASS
        ]
        assert carried == [site_push.BYPASS_HEADER]

    captured = capsys.readouterr()
    assert any(
        line.startswith("SITE_PUSH= succeeded") for line in captured.out.splitlines()
    )
    assert_no_secret_in(
        site_push.SITE_PUSH_LOG.read_text(encoding="utf-8"),
        captured.out,
        captured.err,
    )


@pytest.mark.usefixtures("switched_on")
def test_two_pushes_mint_two_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = push_harness(monkeypatch, tmp_path)
    push = {
        "source": "manual",
        "cache_path": harness.cache_path,
        "rebuild": rebuild_must_not_run,
        "check_stylesheet": no_stylesheet_problem,
    }

    first = site_push.push_site(**push)
    second = site_push.push_site(**push)

    assert first.outcome == second.outcome == "succeeded"
    assert first.publish_id != second.publish_id
    records = harness.records()
    assert [record["publish_id"] for record in records] == [
        first.publish_id,
        second.publish_id,
    ]
    assert cache_meta(harness.cache_path)[PUBLISH_ID_KEY] == records[1]["publish_id"]


def test_the_code_commit_says_when_the_uploaded_code_is_in_no_commit(
    tmp_path: Path,
) -> None:
    """review WR-01: the deploy uploads the working tree, so the record marks an unsaved edit."""
    repo = tmp_path / "repo"
    (repo / "api").mkdir(parents=True)
    (repo / "api" / "main.py").write_text("one\n", encoding="utf-8")
    (repo / "notes.md").write_text("one\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "api/main.py", "notes.md")
    git(repo, "commit", "-q", "-m", "first")
    head = git(repo, "rev-parse", "HEAD")

    assert site_push._code_commit(repo) == head
    (repo / "notes.md").write_text("two\n", encoding="utf-8")
    assert site_push._code_commit(repo) == head, "an edit that is not uploaded is clean"
    (repo / "api" / "main.py").write_text("two\n", encoding="utf-8")
    assert site_push._code_commit(repo) == head + site_push.DIRTY_SUFFIX
    git(repo, "checkout", "--", "api/main.py")
    (repo / "web").mkdir()
    (repo / "web" / "new.html").write_text("<p>new</p>\n", encoding="utf-8")
    assert site_push._code_commit(repo) == head + site_push.DIRTY_SUFFIX
    assert site_push._code_commit(tmp_path / "not_a_repo") is None


def test_the_suite_cannot_reach_vercel() -> None:
    for url in ("https://nfl-predict.vercel.app/health", "https://api.vercel.com/"):
        host = httpx.URL(url).host
        with (
            httpx.Client() as client,
            pytest.raises(pytest.fail.Exception) as refusal,
        ):
            client.get(url)
        assert host in str(refusal.value)
        assert "/health" not in str(refusal.value)

    with pytest.raises(pytest.fail.Exception):
        site_push.RUN_CLI(["node", "vc.js", "deploy"])

    for path in (
        site_push.SITE_PUSH_LOG,
        cache_lock.CACHE_PUBLISH_LOCK,
        site_push.PUSH_ENABLED_MARKER,
        site_push.VERCEL_GLOBAL_CONFIG,
    ):
        assert not Path(path).resolve().is_relative_to(REPOSITORY_ROOT), path
    assert not site_push.PUSH_ENABLED_MARKER.exists()


# ---------------------------------------------------------------------------
# Task 2: the switch, the simple refusals and the secrets
# ---------------------------------------------------------------------------


def _must_not_be_called(what: str):
    def _refuse(*_args, **_kwargs):
        pytest.fail(f"the push called {what}")

    return _refuse


@pytest.mark.parametrize("source", ["daily", "manual"])
def test_a_push_that_is_not_switched_on_only_records_it(
    source: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_path = build_cache(tmp_path)
    before = cache_path.read_bytes()
    monkeypatch.setattr(
        site_push, "LOAD_PUSH_SECRETS", _must_not_be_called("the secrets")
    )

    outcome = site_push.push_site(
        source=source,
        cache_path=cache_path,
        rebuild=_must_not_be_called("the rebuild"),
        check_stylesheet=_must_not_be_called("the stylesheet check"),
    )

    assert (outcome.outcome, outcome.reason) == ("not_pushed", "push_not_enabled")
    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        pass
    assert cache_path.read_bytes() == before
    records = site_push.read_site_pushes()
    assert len(records) == 1
    assert (records[0]["outcome"], records[0]["source"]) == ("not_pushed", source)
    assert "SITE_PUSH= not_pushed push_not_enabled" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reason", "arguments", "stylesheet_problem"),
    [
        (
            "rebuild_incomplete",
            {"rebuild_attempted": True, "rebuild_completed": False},
            None,
        ),
        ("cache_missing", {"cache_path": Path("absent") / "web_cache.duckdb"}, None),
        ("stylesheet_out_of_step", {}, "stylesheet_out_of_step"),
        ("stylesheet_unverifiable", {}, "stylesheet_unverifiable"),
    ],
)
@pytest.mark.usefixtures("switched_on")
def test_each_simple_refusal_deploys_nothing(
    reason: str,
    arguments: dict[str, Any],
    stylesheet_problem: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = push_harness(monkeypatch, tmp_path)
    before = harness.cache_path.read_bytes()
    push = {"cache_path": harness.cache_path, **arguments}
    if reason == "cache_missing":
        push["cache_path"] = tmp_path / push["cache_path"]

    outcome = site_push.push_site(
        source="daily",
        rebuild=rebuild_must_not_run,
        check_stylesheet=lambda: stylesheet_problem,
        **push,
    )

    assert (outcome.outcome, outcome.reason) == ("refused", reason)
    assert harness.cli.calls == []
    assert harness.host.requests == []
    assert [record["reason"] for record in harness.records()] == [reason]
    assert harness.cache_path.read_bytes() == before


@pytest.mark.parametrize(
    ("missing", "reason"),
    [("token", "missing_token"), ("bypass", "missing_bypass_secret")],
)
@pytest.mark.usefixtures("switched_on")
def test_a_missing_secret_fails_before_the_lock(
    missing: str, reason: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = push_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(
        site_push, "LOAD_PUSH_SECRETS", lambda: dummy_secrets(**{missing: None})
    )

    # The lock is held throughout: a push that reached the lock would refuse lock_held instead.
    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        outcome = site_push.push_site(
            source="manual",
            cache_path=harness.cache_path,
            lock_wait_seconds=0.0,
            rebuild=rebuild_must_not_run,
            check_stylesheet=no_stylesheet_problem,
        )

    assert (outcome.outcome, outcome.reason) == ("failed", reason)
    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        pass
    assert harness.cli.calls == []
    assert len(harness.records()) == 1

    # A blank value in .env counts as missing.
    blank = site_push.load_push_secrets(
        cast(
            "Settings",
            SimpleNamespace(vercel_token=SecretStr("  "), vercel_bypass_secret=None),
        )
    )
    assert blank == site_push.PushSecrets(vercel_token=None, bypass_secret=None)


def test_record_not_pushed_writes_one_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Held throughout: recording a run that did not push takes no lock.
    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        outcome = site_push.record_not_pushed(
            source="daily", reason="run_failed", run_id="r1"
        )

    assert (outcome.outcome, outcome.reason) == ("not_pushed", "run_failed")
    records = site_push.read_site_pushes()
    assert len(records) == 1
    assert set(records[0]) == set(site_push.SITE_PUSH_FIELDS)
    assert (records[0]["outcome"], records[0]["reason"], records[0]["run_id"]) == (
        "not_pushed",
        "run_failed",
        "r1",
    )
    assert "SITE_PUSH= not_pushed run_failed" in capsys.readouterr().out


def stamp_data_format(path: Path, value: str) -> None:
    """Rewrite the cache's stored data-format version (a planted mismatch or its repair)."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "UPDATE cache_meta SET value = ? WHERE key = ?",
            [value, DATA_FORMAT_VERSION_KEY],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Task 3: the lock at every rebuild, the self-heal, the read-back and deploy failures, a raise
# ---------------------------------------------------------------------------


def _lock_probe(seen: list[str]) -> None:
    """Try the publish lock without waiting; record whether something already held it."""
    try:
        with cache_lock.cache_publish_lock(wait_seconds=0.0):
            seen.append("free")
    except FileLockHeldError:
        seen.append("held")


def test_every_rebuild_entry_point_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(steps, "populate_web_cache_unlocked", lambda: _lock_probe(seen))

    steps.step_populate_web_cache()

    assert seen == ["held"]

    rebuilt: list[dict[str, Any]] = []
    monkeypatch.setattr(
        populate_cache_script, "populate_cache", lambda **kw: rebuilt.append(kw)
    )
    monkeypatch.setattr(populate_cache_script, "REBUILD_LOCK_WAIT_SECONDS", 0.0)
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "populate_cache.py",
            *("--db-path", str(elsewhere / "web_cache.duckdb")),
            *("--artifacts-dir", str(elsewhere)),
            *("--outputs-dir", str(elsewhere)),
            *("--gold-dir", str(elsewhere)),
            *("--silver-dir", str(elsewhere)),
            *("--predictions-dir", str(elsewhere)),
            *("--bet-list-dir", str(elsewhere)),
        ],
    )
    with (
        cache_lock.cache_publish_lock(wait_seconds=0.0),
        pytest.raises(SystemExit) as refused,
    ):
        populate_cache_script.main()

    assert refused.value.code == 1
    assert rebuilt == []
    assert "POPULATE_CACHE= refused" in capsys.readouterr().out


@pytest.mark.parametrize("rebuild_does", ["heals", "raises", "leaves_the_mismatch"])
@pytest.mark.usefixtures("switched_on")
def test_a_format_mismatch_self_heals_or_refuses(
    rebuild_does: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = push_harness(monkeypatch, tmp_path)
    stamp_data_format(harness.cache_path, str(DATA_FORMAT_VERSION + 1))
    lock_seen: list[str] = []

    def _rebuild() -> None:
        _lock_probe(lock_seen)
        if rebuild_does == "raises":
            raise RuntimeError("rebuild-detail")
        if rebuild_does == "heals":
            stamp_data_format(harness.cache_path, str(DATA_FORMAT_VERSION))

    # The DEFAULT rebuild is used, so the push really reaches the unlocked populate step.
    monkeypatch.setattr(steps, "populate_web_cache_unlocked", _rebuild)

    outcome = site_push.push_site(
        source="daily",
        cache_path=harness.cache_path,
        check_stylesheet=no_stylesheet_problem,
    )

    assert lock_seen == ["held"], "the self-heal did not run under the push's lock"
    printed = capsys.readouterr().out
    if rebuild_does == "heals":
        assert outcome.outcome == "succeeded"
        assert len(harness.cli.calls) == 1
        return
    expected_reason = {
        "raises": "format_mismatch_rebuild_failed",
        "leaves_the_mismatch": "format_mismatch_after_rebuild",
    }[rebuild_does]
    assert (outcome.outcome, outcome.reason) == ("refused", expected_reason)
    assert outcome.data_format_found == str(DATA_FORMAT_VERSION + 1)
    assert outcome.data_format_expected == DATA_FORMAT_VERSION
    assert harness.cli.calls == []
    record = harness.records()[0]
    assert (record["data_format_found"], record["data_format_expected"]) == (
        str(DATA_FORMAT_VERSION + 1),
        DATA_FORMAT_VERSION,
    )
    if rebuild_does == "raises":
        assert f"SITE_PUSH= refused {expected_reason} RuntimeError" in printed
        assert "rebuild-detail" not in printed


@pytest.mark.usefixtures("switched_on")
def test_a_held_lock_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = push_harness(monkeypatch, tmp_path)

    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        outcome = site_push.push_site(
            source="manual",
            cache_path=harness.cache_path,
            lock_wait_seconds=0.0,
            rebuild=rebuild_must_not_run,
            check_stylesheet=no_stylesheet_problem,
        )

    assert (outcome.outcome, outcome.reason) == ("refused", "lock_held")
    assert harness.cli.calls == []
    assert len(harness.records()) == 1


def _served_publish_id(cache_path: Path) -> str | None:
    return cache_meta(cache_path).get(PUBLISH_ID_KEY)


@pytest.mark.parametrize(
    ("readback", "answer"),
    [
        (
            "old_id",
            lambda _cache: httpx.Response(
                200, json={"publish_id": "an-older-publish", "cache_sha256": "0" * 64}
            ),
        ),
        (
            "digest_mismatch",
            lambda cache: httpx.Response(
                200,
                json={
                    "publish_id": _served_publish_id(cache),
                    "cache_sha256": "0" * 64,
                },
            ),
        ),
        ("no_id", lambda _cache: httpx.Response(200, json={"status": "ok"})),
        (
            "http_302",
            lambda _cache: httpx.Response(
                302, headers={"location": "https://vercel.com/login"}
            ),
        ),
        ("http_401", lambda _cache: httpx.Response(401)),
    ],
)
@pytest.mark.usefixtures("switched_on")
def test_a_read_back_that_never_matches_fails_after_a_fixed_number_of_attempts(
    readback: str,
    answer: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = push_harness(monkeypatch, tmp_path)
    harness.host.answer = lambda _request: answer(harness.cache_path)

    outcome = site_push.push_site(
        source="daily",
        cache_path=harness.cache_path,
        rebuild=rebuild_must_not_run,
        check_stylesheet=no_stylesheet_problem,
    )

    assert (outcome.outcome, outcome.reason, outcome.readback) == (
        "failed",
        "readback_failed",
        readback,
    )
    assert len(harness.host.requests) == site_push.READBACK_ATTEMPTS
    assert harness.sleeps == [site_push.READBACK_INTERVAL_SECONDS] * (
        site_push.READBACK_ATTEMPTS - 1
    )
    production_host = httpx.URL(site_push.PRODUCTION_URL).host
    assert {request.url.host for request in harness.host.requests} == {production_host}
    assert len(harness.cli.calls) == 1, "a failed read-back must not deploy again"
    records = harness.records()
    assert len(records) == 1
    captured = capsys.readouterr()
    assert_no_secret_in(
        site_push.SITE_PUSH_LOG.read_text(encoding="utf-8"),
        captured.out,
        captured.err,
    )


@pytest.mark.parametrize(
    ("broken_path", "broken_answer", "readback"),
    [
        (
            "/bets",
            httpx.Response(500, text="Internal Server Error"),
            "page_bets_http_500",
        ),
        (
            "/",
            httpx.Response(200, html="<html><title>Login - Vercel</title></html>"),
            "page_root_no_marker",
        ),
    ],
    ids=["a_page_answers_500", "a_page_lacks_its_marker"],
)
@pytest.mark.usefixtures("switched_on")
def test_a_broken_page_fails_a_push_whose_health_matched(
    broken_path: str,
    broken_answer: httpx.Response,
    readback: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The 36-10 incident: /health read back while the pages answered 500 (review CR-01)."""
    assert set(site_push.SMOKE_PATHS) <= {
        route.path for route in fetch_hosted_routes.routes("")
    }, "a checked page is not in the live check's route list, so it would be skipped"
    harness = push_harness(monkeypatch, tmp_path)
    harness.host.answer = lambda request: (
        broken_answer
        if request.url.path == broken_path
        else harness.host.default(request)
    )

    outcome = site_push.push_site(
        source="daily",
        cache_path=harness.cache_path,
        rebuild=rebuild_must_not_run,
        check_stylesheet=no_stylesheet_problem,
    )

    assert (outcome.outcome, outcome.reason, outcome.readback) == (
        "failed",
        "pages_failed",
        readback,
    )
    assert len(harness.cli.calls) == 1, "a failed page check must not deploy again"
    (record,) = harness.records()
    assert (record["reason"], record["readback"]) == ("pages_failed", readback)


@pytest.mark.parametrize(
    ("reason", "cli_does"),
    [
        ("deploy_exit_1", {"returncode": 1}),
        ("deploy_output_unparseable", {"stdout": "Deploying nfl-predict... done"}),
        (
            "deploy_timeout",
            {"raises": subprocess.TimeoutExpired(cmd="vc.js", timeout=900)},
        ),
        ("cli_missing", {"raises": FileNotFoundError("node.exe")}),
    ],
)
@pytest.mark.usefixtures("switched_on")
def test_deploy_failures_are_named(
    reason: str,
    cli_does: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = push_harness(monkeypatch, tmp_path)
    for name, value in cli_does.items():
        setattr(harness.cli, name, value)

    outcome = site_push.push_site(
        source="daily",
        cache_path=harness.cache_path,
        rebuild=rebuild_must_not_run,
        check_stylesheet=no_stylesheet_problem,
    )

    assert (outcome.outcome, outcome.reason) == ("failed", reason)
    assert len(harness.cli.calls) == 1
    assert harness.host.requests == []
    assert [record["reason"] for record in harness.records()] == [reason]


@pytest.mark.usefixtures("switched_on")
def test_an_unexpected_raise_still_writes_one_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = push_harness(monkeypatch, tmp_path)

    def _stylesheet_check_raises() -> str | None:
        raise RuntimeError("boom-detail")

    outcome = site_push.push_site(
        source="daily",
        cache_path=harness.cache_path,
        rebuild=rebuild_must_not_run,
        check_stylesheet=_stylesheet_check_raises,
    )

    assert (outcome.outcome, outcome.reason) == ("failed", "RuntimeError")
    records = harness.records()
    assert len(records) == 1
    assert set(records[0]) == set(site_push.SITE_PUSH_FIELDS)
    assert harness.cli.calls == []
    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        pass
    captured = capsys.readouterr()
    assert "SITE_PUSH= failed RuntimeError" in captured.out
    assert "boom-detail" not in site_push.SITE_PUSH_LOG.read_text(encoding="utf-8")
    assert "boom-detail" not in captured.out + captured.err
