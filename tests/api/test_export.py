"""Tests for CSV and JSON export endpoints.

Validates UIAP-06: CSV and JSON export with Content-Disposition headers
and valid file content.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_csv_export(test_client: TestClient):
    """UIAP-06: CSV export returns valid file with Content-Disposition."""
    response = test_client.get("/api/export/csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "Content-Disposition" in response.headers
    assert "attachment" in response.headers["Content-Disposition"]
    # Verify CSV structure
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2  # header + at least 1 row
    assert "game_id" in lines[0]


def test_json_export(test_client: TestClient):
    """UIAP-06: JSON export returns valid JSON array."""
    response = test_client.get("/api/export/json")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "game_id" in data[0]


def test_csv_export_with_season_filter(test_client: TestClient):
    """CSV export with season filter returns matching data."""
    response = test_client.get("/api/export/csv?season=2024")
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2  # header + at least 1 row
    # All data rows should be from 2024
    assert "2024" in response.text


def test_csv_export_with_week_filter(test_client: TestClient):
    """CSV export with week filter returns matching data."""
    response = test_client.get("/api/export/csv?season=2024&week=1")
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2


def test_csv_export_no_data(test_client: TestClient):
    """CSV export returns 404 when no matching data exists."""
    response = test_client.get("/api/export/csv?season=1900")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_json_export_no_data(test_client: TestClient):
    """JSON export returns 404 when no matching data exists."""
    response = test_client.get("/api/export/json?season=1900")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_csv_export_single_game(test_client: TestClient):
    """CSV export for a single game_id returns one data row."""
    response = test_client.get("/api/export/csv?game_id=2024_W01_BUF@KC")
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 2  # header + 1 row
    assert "2024_W01_BUF@KC" in response.text


def test_json_export_single_game(test_client: TestClient):
    """JSON export for a single game_id returns one-element array."""
    response = test_client.get("/api/export/json?game_id=2024_W01_BUF@KC")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert len(data) == 1
    assert data[0]["game_id"] == "2024_W01_BUF@KC"


def test_backtest_export(test_client: TestClient):
    """Backtest CSV export returns backtest predictions."""
    response = test_client.get("/api/export/csv?type=backtest")
    assert response.status_code == 200
    assert "game_id" in response.text
    assert "model_prob" in response.text


def test_backtest_json_export(test_client: TestClient):
    """Backtest JSON export returns backtest predictions."""
    response = test_client.get("/api/export/json?type=backtest")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_csv_content_disposition_filename(test_client: TestClient):
    """Content-Disposition header has appropriate filename."""
    response = test_client.get("/api/export/csv")
    disposition = response.headers["Content-Disposition"]
    assert "nfl_predictions" in disposition
    assert ".csv" in disposition


def test_json_content_disposition_filename(test_client: TestClient):
    """JSON Content-Disposition header has appropriate filename."""
    response = test_client.get("/api/export/json")
    disposition = response.headers["Content-Disposition"]
    assert "nfl_export.json" in disposition
