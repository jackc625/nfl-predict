"""The manual "push now" command (Phase 36, D-05, SPEC R11).

``scripts.push_site.main`` is driven for real through ``pipeline.site_push.push_site`` -- the same
routine, switch, refusals, read-back and push-log record as the daily push -- with only the outside
edges faked by Plan 36-05's helpers: the Vercel CLI is ``FakeCli`` and the hosted site is
``ReadbackHost``, which answers from the tmp cache at request time. The push's default cache path
points at that tmp cache, and the stylesheet check is the in-step stand-in, so no test touches the
production cache or runs the Tailwind binary. ``tests/conftest.py`` keeps the switch absent unless a
test writes it, quarantines the push log and lock, and refuses every real Vercel call.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from api.cache_schema import PUBLISH_ID_KEY
from pipeline import cache_lock, site_push
from scripts import push_site as command
from tests.unit.test_site_push import (
    FakeCli,
    PushHarness,
    cache_meta,
    no_stylesheet_problem,
    push_harness,
    rebuild_must_not_run,
    switched_on,  # noqa: F401 - the quarantined D-23 switch fixture
)


def manual_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PushHarness:
    """Plan 36-05's harness, with the command's default cache and stylesheet check redirected."""
    harness = push_harness(monkeypatch, tmp_path)
    monkeypatch.setattr(site_push, "DEFAULT_CACHE_PATH", harness.cache_path)
    monkeypatch.setattr(site_push, "_default_stylesheet_check", no_stylesheet_problem)
    return harness


def _newest_record() -> dict[str, Any]:
    return site_push.read_site_pushes()[-1]


@pytest.mark.usefixtures("switched_on")
def test_a_manual_push_after_a_failed_daily_push_succeeds_and_reads_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = manual_harness(monkeypatch, tmp_path)
    harness.host.answer = lambda _request: httpx.Response(401)
    daily = site_push.push_site(
        source="daily",
        cache_path=harness.cache_path,
        rebuild=rebuild_must_not_run,
        check_stylesheet=no_stylesheet_problem,
    )
    assert (daily.outcome, daily.reason) == ("failed", "readback_failed")
    harness.host.answer = None  # the host now serves what was really deployed

    assert command.main([]) == 0

    records = site_push.read_site_pushes()
    assert [(r["source"], r["outcome"]) for r in records] == [
        ("daily", "failed"),
        ("manual", "succeeded"),
    ]
    newest = records[-1]
    assert newest["readback"] == "matched"
    served = cache_meta(harness.cache_path)[PUBLISH_ID_KEY]
    assert newest["publish_id"] == served != daily.publish_id
    assert f"PUSH_SITE= succeeded {served}" in capsys.readouterr().out


@pytest.mark.usefixtures("switched_on")
def test_the_command_refuses_while_the_lock_is_held(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = manual_harness(monkeypatch, tmp_path)

    # Held here, standing in for a running cache rebuild or another push.
    with cache_lock.cache_publish_lock(wait_seconds=0.0):
        assert command.main([]) == 1

    newest = _newest_record()
    assert (newest["outcome"], newest["reason"], newest["source"]) == (
        "refused",
        "lock_held",
        "manual",
    )
    assert harness.cli.calls == []
    assert "PUSH_SITE= refused lock_held" in capsys.readouterr().out


def test_the_command_does_nothing_before_setup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = FakeCli()
    monkeypatch.setattr(site_push, "RUN_CLI", cli)

    def _no_secret_read() -> None:
        pytest.fail("the command read a secret before setup")

    monkeypatch.setattr(site_push, "LOAD_PUSH_SECRETS", _no_secret_read)

    assert command.main([]) == 1

    assert not site_push.PUSH_ENABLED_MARKER.exists()
    (record,) = site_push.read_site_pushes()
    assert (record["outcome"], record["reason"], record["source"]) == (
        "not_pushed",
        "push_not_enabled",
        "manual",
    )
    assert cli.calls == []
    assert "PUSH_SITE= not_pushed push_not_enabled" in capsys.readouterr().out


@pytest.mark.usefixtures("switched_on")
def test_an_unreadable_settings_file_reports_the_class_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = manual_harness(monkeypatch, tmp_path)

    def _unreadable() -> None:
        raise ValueError("VERCEL_TOKEN=secret-value-xyz is not valid")

    monkeypatch.setattr(site_push, "LOAD_PUSH_SECRETS", _unreadable)

    assert command.main([]) == 1

    captured = capsys.readouterr()
    assert "PUSH_SITE= failed ValueError" in captured.out
    assert "secret-value-xyz" not in captured.out + captured.err
    newest = _newest_record()
    assert (newest["outcome"], newest["reason"], newest["source"]) == (
        "failed",
        "ValueError",
        "manual",
    )
    assert harness.cli.calls == []
