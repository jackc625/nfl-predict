"""Shared pytest fixtures for API tests.

Provides:
- test_db: Temporary DuckDB with CACHE_SCHEMA and sample data
- test_client: FastAPI TestClient with overridden DB_PATH
- sample_game_data: List of sample game dicts
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.cache import CACHE_SCHEMA


def _sample_game_data() -> list[dict]:
    """Return a list of 3 sample game dicts with all prediction fields."""
    return [
        {
            "game_id": "2024_W01_BUF@KC",
            "season": 2024,
            "week": 1,
            "game_date": datetime(2024, 9, 5, 20, 15, tzinfo=UTC),
            "home_team": "KC",
            "away_team": "BUF",
            "status": "completed",
            "home_score": 27,
            "away_score": 20,
            "wp_prob": 0.62,
            "wp_confidence": "medium",
            "ats_prediction": -3.5,
            "ats_confidence": "high",
            "ou_prediction": 48.5,
            "ou_confidence": "medium",
            "market_spread": -3.0,
            "market_total": 47.5,
            "market_ml_home": -155,
            "market_ml_away": 135,
            "wp_edge": 0.05,
            "ats_edge": 0.02,
            "ou_edge": 0.01,
            "blended_wp": 0.60,
            "blended_ats": -3.2,
            "blended_ou": 48.0,
        },
        {
            "game_id": "2024_W01_PHI@GB",
            "season": 2024,
            "week": 1,
            "game_date": datetime(2024, 9, 6, 20, 15, tzinfo=UTC),
            "home_team": "GB",
            "away_team": "PHI",
            "status": "completed",
            "home_score": 34,
            "away_score": 29,
            "wp_prob": 0.45,
            "wp_confidence": "low",
            "ats_prediction": 1.5,
            "ats_confidence": "medium",
            "ou_prediction": 50.0,
            "ou_confidence": "high",
            "market_spread": 2.5,
            "market_total": 49.0,
            "market_ml_home": 110,
            "market_ml_away": -130,
            "wp_edge": -0.03,
            "ats_edge": 0.04,
            "ou_edge": 0.03,
            "blended_wp": 0.47,
            "blended_ats": 1.8,
            "blended_ou": 49.5,
        },
        {
            "game_id": "2023_W18_SF@SEA",
            "season": 2023,
            "week": 18,
            "game_date": datetime(2024, 1, 7, 16, 25, tzinfo=UTC),
            "home_team": "SEA",
            "away_team": "SF",
            "status": "completed",
            "home_score": 20,
            "away_score": 24,
            "wp_prob": 0.38,
            "wp_confidence": "medium",
            "ats_prediction": 3.0,
            "ats_confidence": "low",
            "ou_prediction": 45.0,
            "ou_confidence": "medium",
            "market_spread": 3.5,
            "market_total": 44.5,
            "market_ml_home": 150,
            "market_ml_away": -175,
            "wp_edge": 0.01,
            "ats_edge": -0.01,
            "ou_edge": 0.02,
            "blended_wp": 0.40,
            "blended_ats": 3.2,
            "blended_ou": 44.8,
        },
    ]


@pytest.fixture()
def sample_game_data() -> list[dict]:
    """Return sample game data for testing."""
    return _sample_game_data()


@pytest.fixture()
def test_db(tmp_path: Path) -> Path:
    """Create a temporary DuckDB with CACHE_SCHEMA and sample data.

    Returns the path to the database file.
    """
    db_path = tmp_path / "test_cache.duckdb"
    conn = duckdb.connect(str(db_path))

    # Create schema
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)

    # Insert sample predictions
    games = _sample_game_data()
    for game in games:
        conn.execute(
            """
            INSERT INTO predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            [
                game["game_id"], game["season"], game["week"],
                game["game_date"], game["home_team"], game["away_team"],
                game["status"], game["home_score"], game["away_score"],
                game["wp_prob"], game["wp_confidence"],
                game["ats_prediction"], game["ats_confidence"],
                game["ou_prediction"], game["ou_confidence"],
                game["market_spread"], game["market_total"],
                game["market_ml_home"], game["market_ml_away"],
                game["wp_edge"], game["ats_edge"], game["ou_edge"],
                game["blended_wp"], game["blended_ats"], game["blended_ou"],
            ],
        )

    # Insert sample feature importances
    importances = [
        ("_model_", "wp", "elo_diff", 0.25),
        ("_model_", "wp", "rolling_off_epa", 0.18),
        ("_model_", "wp", "snapshot_spread", 0.15),
        ("_model_", "ats", "elo_diff", 0.20),
        ("_model_", "ats", "rolling_off_epa", 0.22),
        ("_model_", "ou", "rolling_total_epa", 0.30),
    ]
    conn.executemany(
        "INSERT INTO feature_importances VALUES (?, ?, ?, ?)",
        importances,
    )

    # Insert cache metadata
    now = datetime.now(tz=UTC)
    conn.executemany(
        "INSERT INTO cache_meta VALUES (?, ?, ?)",
        [
            ("last_updated", now.isoformat(), now),
            ("prediction_count", "3", now),
            ("season_range", "2023-2024", now),
        ],
    )

    conn.close()
    return db_path


@pytest.fixture()
def test_client(test_db: Path) -> TestClient:
    """Create a FastAPI TestClient with the test database.

    Overrides the DB_PATH dependency so the app reads from the test DB.
    """
    import api.dependencies as deps
    from api.main import app

    # Override the DB_PATH module-level variable
    original_db_path = deps.DB_PATH
    deps.DB_PATH = test_db

    client = TestClient(app, raise_server_exceptions=False)
    yield client  # type: ignore[misc]

    # Restore
    deps.DB_PATH = original_db_path
