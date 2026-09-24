"""End-to-end integration test for Phase 15 (Performance and Tech Debt Cleanup).

Exercises the full reviewed flow across plans 15-01 through 15-04 in a
single test file so a regression in any Phase 15 guardrail (DI, cache,
deletions, timezone) surfaces here:

- App startup -> lifespan creates shared DuckDB connection in app.state
- get_db() dependency injection returns the shared connection
- DataService (cached) returns predictions via the injected connection
- clear_cache() evicts cached entries; next call re-queries
- Simulated missing DB causes ModelUnavailableError
- tz-aware datetime ingest via data/storage.py succeeds
- tz-naive datetime ingest via data/storage.py raises ValueError with "naive"
- ensure_utc_aware() lets a producer fix-forward a previously naive datetime
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.dependencies as deps
import api.routes.health as health_module
from api.cache import CACHE_SCHEMA
from api.exceptions import ModelUnavailableError
from api.services import _cache, clear_cache
from data.storage import DuckDBConnection
from utils.date_utils import ensure_utc_aware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_duckdb(db_path: Path) -> None:
    """Create a minimal DuckDB cache with one prediction row.

    Mirrors the structure of ``tests/api/conftest.py::test_db`` so the
    integration test does not need to share that fixture across packages.
    """
    conn = duckdb.connect(str(db_path))
    try:
        for stmt in CACHE_SCHEMA.strip().split(";"):
            if stmt.strip():
                conn.execute(stmt)

        conn.execute(
            """
            INSERT INTO predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                "2024_W01_BUF@KC",
                2024,
                1,
                datetime(2024, 9, 5, 20, 15, tzinfo=UTC),
                "KC",
                "BUF",
                "completed",
                27,
                20,
                0.62,
                "medium",
                -3.5,
                "high",
                48.5,
                "medium",
                -3.0,
                47.5,
                -155,
                135,
                0.05,
                0.02,
                0.01,
                0.60,
                -3.2,
                48.0,
                None,  # market_wp
            ],
        )

        # Cache_meta keeps the homepage banner happy.
        now = datetime.now(tz=UTC)
        conn.executemany(
            "INSERT INTO cache_meta VALUES (?, ?, ?)",
            [
                ("last_updated", now.isoformat(), now),
                ("prediction_count", "1", now),
                ("season_range", "2024-2024", now),
            ],
        )
    finally:
        conn.close()


def _set_db_path(new_path: Path) -> tuple[Path, Path]:
    """Point both ``api.dependencies.DB_PATH`` and ``api.routes.health.DB_PATH``
    at *new_path*. Returns the originals so the caller can restore them.
    """
    original_deps = deps.DB_PATH
    original_health = health_module.DB_PATH
    deps.DB_PATH = new_path
    health_module.DB_PATH = new_path
    return original_deps, original_health


def _restore_db_path(original_deps: Path, original_health: Path) -> None:
    """Restore the originals captured by :func:`_set_db_path`."""
    deps.DB_PATH = original_deps
    health_module.DB_PATH = original_health


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db = tmp_path / "phase15_integration.duckdb"
    _seed_duckdb(db)
    return db


# ---------------------------------------------------------------------------
# 15-01 + 15-02: startup -> DI -> cache hit -> clear_cache -> re-query
# ---------------------------------------------------------------------------


def test_startup_injects_shared_connection_and_caches_queries(
    seeded_db: Path,
) -> None:
    """Full startup -> DI -> cached service call -> clear_cache -> re-query flow."""
    from api.main import app

    clear_cache()
    original_deps, original_health = _set_db_path(seeded_db)
    # Drop any leftover dependency overrides from other tests so the real
    # lifespan path runs and the real get_db dependency is exercised.
    app.dependency_overrides.clear()
    try:
        with TestClient(app) as client:
            # Lifespan must have created the shared connection.
            # Reference the imported ``app`` directly (not ``client.app``)
            # so the type checker sees the FastAPI ``state`` attribute.
            assert app.state.db_conn is not None
            assert hasattr(app.state, "db_lock")
            assert hasattr(app.state.db_lock, "acquire")

            # First request populates the cache via DataService.get_predictions
            response1 = client.get("/")
            assert response1.status_code == 200, response1.text
            assert any(k[0] == "predictions" for k in _cache)

            # Second call is a cache hit -- no new entries
            cached_before = len(_cache)
            response2 = client.get("/")
            assert response2.status_code == 200, response2.text
            assert len(_cache) == cached_before

            # clear_cache() evicts everything
            clear_cache()
            assert len(_cache) == 0

            # Next call re-populates
            response3 = client.get("/")
            assert response3.status_code == 200, response3.text
            assert len(_cache) > 0
    finally:
        _restore_db_path(original_deps, original_health)
        clear_cache()


# ---------------------------------------------------------------------------
# 15-01 + 15-03: missing DB raises ModelUnavailableError via get_db
# ---------------------------------------------------------------------------


def test_missing_db_raises_model_unavailable(tmp_path: Path) -> None:
    """Missing DuckDB file triggers ModelUnavailableError via get_db."""
    from api.dependencies import get_db

    missing = tmp_path / "no_such.duckdb"
    original_deps, original_health = _set_db_path(missing)
    try:
        req = MagicMock()
        req.app.state.db_conn = None
        req.app.state.db_lock = threading.RLock()

        with pytest.raises(ModelUnavailableError):
            get_db(req)
    finally:
        _restore_db_path(original_deps, original_health)


# ---------------------------------------------------------------------------
# 15-04: tz-aware ingest succeeds, tz-naive ingest is rejected
# ---------------------------------------------------------------------------


def test_tz_aware_ingest_roundtrip_through_storage(tmp_path: Path) -> None:
    """A DataFrame with tz-aware datetimes survives the storage normalization."""
    conn = DuckDBConnection(str(tmp_path / "tz_aware.duckdb"))
    try:
        df = pd.DataFrame(
            {
                "game_date": pd.to_datetime(["2024-09-05T00:15:00Z"]),
                "home": ["KC"],
                "away": ["BUF"],
            }
        )
        normalized = conn._normalize_datetime_columns(df)
        assert normalized["game_date"].dt.tz is not None
        assert normalized["home"].iloc[0] == "KC"
    finally:
        conn.close()


def test_tz_naive_ingest_raises_value_error_with_naive_in_message(
    tmp_path: Path,
) -> None:
    """A DataFrame with naive datetimes raises ValueError during normalization."""
    conn = DuckDBConnection(str(tmp_path / "tz_naive.duckdb"))
    try:
        df = pd.DataFrame(
            {
                "game_date": pd.to_datetime(["2024-09-05T00:15:00"]),  # naive
                "home": ["KC"],
            }
        )
        assert df["game_date"].dt.tz is None
        with pytest.raises(ValueError) as excinfo:
            conn._normalize_datetime_columns(df)
        # Substring check -- "naive" is the stable word
        assert "naive" in str(excinfo.value)
    finally:
        conn.close()


def test_ensure_utc_aware_roundtrip_fix_forward(tmp_path: Path) -> None:
    """Caller can fix-forward a naive datetime using ensure_utc_aware()."""
    naive = datetime(2024, 9, 5, 0, 15)  # naive
    aware = ensure_utc_aware(naive)
    assert aware.tzinfo is UTC

    # Now it survives storage normalization
    conn = DuckDBConnection(str(tmp_path / "fix_forward.duckdb"))
    try:
        df = pd.DataFrame({"game_date": pd.to_datetime([aware])})
        # The column should be tz-aware after pd.to_datetime sees an aware element
        assert df["game_date"].dt.tz is not None
        result = conn._normalize_datetime_columns(df)
        assert result["game_date"].dt.tz is not None
    finally:
        conn.close()
