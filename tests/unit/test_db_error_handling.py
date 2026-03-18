"""Tests for DuckDB error handling -- FOUN-04."""

import pytest

from utils.exceptions import StorageError


class TestDuckDBFailLoud:
    """Verify DuckDB failures raise StorageError, never return empty results."""

    def test_missing_db_raises_storage_error(self, tmp_path):
        """FOUN-04: Missing database file raises StorageError."""
        from api.services import DataService

        service = DataService()
        service.db_path = tmp_path / "nonexistent.duckdb"

        with pytest.raises(StorageError, match="Database file not found"):
            service._get_db_connection()

    def test_corrupt_db_raises_storage_error(self, tmp_path):
        """FOUN-04: Corrupt database raises StorageError."""
        from api.services import DataService

        corrupt_db = tmp_path / "corrupt.duckdb"
        corrupt_db.write_text("not a database")

        service = DataService()
        service.db_path = corrupt_db

        with pytest.raises(StorageError):
            service._get_db_connection()

    def test_no_in_memory_fallback(self, tmp_path):
        """FOUN-04: No silent fallback to in-memory database."""
        from api.services import DataService

        service = DataService()
        service.db_path = tmp_path / "missing.duckdb"

        # Must raise, not return an in-memory connection
        with pytest.raises(StorageError):
            service._get_db_connection()
