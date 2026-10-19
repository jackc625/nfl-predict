"""Host-aware /health (Phase 36, SPEC R10, R9 acceptance 2, D-21).

On the hosted copy (``VERCEL == "1"``, a fact Vercel itself sets) /health reports only what the host
can know: the publish stamp, the season and week of the pushed data, both data-format versions,
staleness, and the SHA-256 of the cache file it serves -- the evidence the push's read-back
compares (``pipeline/site_push.py``). The laptop-only sections read "not available on the hosted
copy" and their absent files never make the verdict "unhealthy". The laptop copy keeps every field
and verdict it had and gains the new fields.

Every test enters ``TestClient`` as a context manager, so the real lifespan runs against a small
cache built under tmp_path; ``DB_PATH`` is patched in both modules that read it, and the laptop-only
paths point at files that do not exist. ``api.presentation.NOW`` is pinned, so staleness is exact.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

import api.dependencies as deps
import api.routes.health as health_module
from api import presentation
from api.cache_schema import (
    CURRENT_SLATE_KEY,
    DATA_FORMAT_VERSION,
    DATA_FORMAT_VERSION_KEY,
    PUBLISH_ID_KEY,
    PUSH_TIME_KEY,
    STALE_AFTER,
)
from api.main import app
from api.schemas import HOSTED_UNAVAILABLE
from api.services import clear_cache

PINNED_NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
PUBLISH_ID = "0123456789abcdef0123456789abcdef"
FRESH_PUSH = PINNED_NOW - timedelta(hours=1)
_STATE_FIELDS = ("db_conn", "db_identity", "data_format")
_LAPTOP_SECTIONS = ("pipeline", "data_freshness", "model_status")


def _write_cache(
    path: Path,
    *,
    version: str | None = str(DATA_FORMAT_VERSION),
    publish: bool = True,
    push_time: datetime = FRESH_PUSH,
) -> Path:
    """A minimal cache: only the cache_meta rows /health reads."""
    rows: list[tuple[str, str]] = [
        ("last_updated", (push_time - timedelta(days=1)).isoformat()),
        (CURRENT_SLATE_KEY, "2026:7"),
    ]
    if version is not None:
        rows.append((DATA_FORMAT_VERSION_KEY, version))
    if publish:
        rows += [(PUBLISH_ID_KEY, PUBLISH_ID), (PUSH_TIME_KEY, push_time.isoformat())]
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "CREATE TABLE cache_meta (key VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMP)"
        )
        conn.executemany(
            "INSERT INTO cache_meta VALUES (?, ?, ?)",
            [(key, value, PINNED_NOW) for key, value in rows],
        )
    return path


def _reset_app_state() -> None:
    app.dependency_overrides.clear()
    for field in _STATE_FIELDS:
        if hasattr(app.state, field):
            delattr(app.state, field)
    clear_cache()


@pytest.fixture()
def served_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The path the app serves its cache from, with every laptop-only file absent."""
    db_path = tmp_path / "web_cache.duckdb"
    absent = tmp_path / "absent"
    monkeypatch.setattr(deps, "DB_PATH", db_path)
    monkeypatch.setattr(health_module, "DB_PATH", db_path)
    monkeypatch.setattr(health_module, "PIPELINE_LOG_PATH", absent / "pipeline.json")
    monkeypatch.setattr(health_module, "DAILY_RUN_RECORDS_PATH", absent / "runs.jsonl")
    monkeypatch.setattr(health_module, "SILVER_GAMES_PATH", absent / "games.parquet")
    monkeypatch.setattr(health_module, "SILVER_ODDS_PATH", absent / "odds.parquet")
    monkeypatch.setattr(health_module, "SILVER_WEATHER_DIR", absent / "weather")
    monkeypatch.setattr(
        health_module,
        "MODEL_FILES",
        {target: absent / f"{target}.pkl" for target in ("wp", "ats", "ou")},
    )
    monkeypatch.setattr(presentation, "NOW", lambda: PINNED_NOW)
    _reset_app_state()
    app.state.db_lock = threading.RLock()
    try:
        yield db_path
    finally:
        _reset_app_state()


def _health() -> dict[str, object]:
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")
    assert response.status_code == 200
    return response.json()


def test_hosted_health_reports_the_stamp_and_the_digest(
    served_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERCEL", "1")
    _write_cache(served_cache)

    data = _health()

    assert data["status"] == "ok"
    assert data["cache_ready"] is True
    assert data["publish_id"] == PUBLISH_ID
    # The push stamp, never the cache build time (a day earlier here).
    assert datetime.fromisoformat(str(data["push_time"])) == FRESH_PUSH
    assert data["cache_sha256"] == hashlib.sha256(served_cache.read_bytes()).hexdigest()
    assert (data["data_season"], data["data_week"]) == (2026, 7)
    assert data["code_data_format"] == DATA_FORMAT_VERSION
    assert data["cache_data_format"] == str(DATA_FORMAT_VERSION)
    assert data["data_format_matches"] is True
    assert data["stale"] is False
    for section in _LAPTOP_SECTIONS:
        assert data[section] == HOSTED_UNAVAILABLE


@pytest.mark.parametrize(
    ("case", "status"),
    [
        ("stale_push", "degraded"),
        ("no_publish_keys", "degraded"),
        ("version_ahead", "unhealthy"),
        ("version_missing", "unhealthy"),
        ("unreadable_cache", "unhealthy"),
    ],
)
def test_the_hosted_verdict(
    served_cache: Path, monkeypatch: pytest.MonkeyPatch, case: str, status: str
) -> None:
    monkeypatch.setenv("VERCEL", "1")
    if case == "stale_push":
        _write_cache(
            served_cache, push_time=PINNED_NOW - STALE_AFTER - timedelta(seconds=1)
        )
    elif case == "no_publish_keys":
        _write_cache(served_cache, publish=False)
    elif case == "version_ahead":
        _write_cache(served_cache, version=str(DATA_FORMAT_VERSION + 1))
    elif case == "version_missing":
        _write_cache(served_cache, version=None)
    else:
        served_cache.write_bytes(bytes(range(256)) * 64)

    data = _health()

    assert data["status"] == status
    for section in _LAPTOP_SECTIONS:
        assert data[section] == HOSTED_UNAVAILABLE
    if case in ("stale_push", "no_publish_keys"):
        assert data["stale"] is True
    if case == "no_publish_keys":
        assert data["publish_id"] is None
    if case in ("version_ahead", "version_missing"):
        assert data["data_format_matches"] is False
        assert data["code_data_format"] == DATA_FORMAT_VERSION
        expected_raw = (
            None if case == "version_missing" else str(DATA_FORMAT_VERSION + 1)
        )
        assert data["cache_data_format"] == expected_raw


def test_local_health_gains_the_fields_and_reports_a_mismatch(
    served_cache: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    log_path = tmp_path / "friday_pipeline.json"
    log_path.write_text(
        json.dumps({"status": "success", "start_time": PINNED_NOW.isoformat()}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health_module, "PIPELINE_LOG_PATH", log_path)
    _write_cache(served_cache)

    data = _health()

    assert data["publish_id"] == PUBLISH_ID
    assert data["cache_sha256"] == hashlib.sha256(served_cache.read_bytes()).hexdigest()
    assert data["data_format_matches"] is True
    for section in _LAPTOP_SECTIONS:
        assert isinstance(data[section], dict), (
            f"local {section} is not its usual object"
        )

    _reset_app_state()
    app.state.db_lock = threading.RLock()
    served_cache.unlink()
    _write_cache(served_cache, version=str(DATA_FORMAT_VERSION + 1))

    mismatched = _health()

    assert mismatched["status"] == "unhealthy"
    assert mismatched["data_format_matches"] is False


def test_health_never_echoes_the_environment(
    served_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "dummy-bypass-value-7Qx2ZpLw9"
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_AUTOMATION_BYPASS_SECRET", secret)
    _write_cache(served_cache)

    with TestClient(app, raise_server_exceptions=False) as client:
        body = client.get("/health").text

    assert secret not in body
