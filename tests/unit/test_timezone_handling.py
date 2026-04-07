"""Tests for timezone enforcement in data/storage.py (OPS-08, Phase 15-04).

Covers:
- _normalize_datetime_columns strict enforcement (aware vs naive)
- _normalize_parquet_datetime_columns strict enforcement
- ensure_utc_aware() helper behavior

The strict enforcement replaces the previous "assume naive == UTC" pattern,
which silently corrupted data when sources actually used a non-UTC timezone.
After Phase 15-04, all datetime columns reaching storage MUST be tz-aware;
producers fix-forward via utils.date_utils.ensure_utc_aware() or by using
datetime.now(UTC) at the source.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.storage import DuckDBConnection, ParquetManager
from utils.date_utils import ensure_utc_aware

ET = ZoneInfo("America/New_York")


@pytest.fixture
def db_conn(tmp_path: Path) -> Iterator[DuckDBConnection]:
    """Minimal DuckDBConnection for testing the normalization methods.

    The normalization methods do not require an open connection -- they
    operate purely on DataFrames. The fixture still passes a real file
    path so the test stays aligned with Phase 15-03's removal of the
    silent in-memory fallback.
    """
    conn = DuckDBConnection(str(tmp_path / "tz_test.duckdb"))
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def parquet_manager(tmp_path: Path) -> ParquetManager:
    """Minimal ParquetManager for exercising _normalize_parquet_datetime_columns.

    The Phase 15-04 strict-naive rejection lives on ParquetManager because
    that is the class whose ``save()`` path is the only production caller
    of the parquet normalization (via ``upsert_silver`` in data/storage.py).
    Tests must therefore call the ParquetManager copy directly so the
    contract is exercised on the real production code path.
    """
    return ParquetManager(str(tmp_path))


# ---------------------------------------------------------------------
# _normalize_datetime_columns
# ---------------------------------------------------------------------


def test_normalize_datetime_columns_accepts_utc_aware(
    db_conn: DuckDBConnection,
) -> None:
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"])}
    )
    result = db_conn._normalize_datetime_columns(df)
    assert result["ts"].dt.tz is not None


def test_normalize_datetime_columns_converts_non_utc_aware_to_utc(
    db_conn: DuckDBConnection,
) -> None:
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01T14:00:00"])})
    df["ts"] = df["ts"].dt.tz_localize(ET)  # 14:00 ET = 19:00 UTC in winter
    result = db_conn._normalize_datetime_columns(df)
    assert result["ts"].dt.tz is not None
    # 14:00 ET in January = 19:00 UTC
    assert result["ts"].iloc[0].hour == 19


def test_normalize_datetime_columns_rejects_naive(
    db_conn: DuckDBConnection,
) -> None:
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01T00:00:00"])})
    # Ensure column is truly naive
    assert df["ts"].dt.tz is None
    with pytest.raises(ValueError) as excinfo:
        db_conn._normalize_datetime_columns(df)
    # Substring check per review item #2 -- "naive" is stable, full message is not
    assert "naive" in str(excinfo.value)
    assert "ts" in str(excinfo.value)


def test_normalize_datetime_columns_handles_mixed(
    db_conn: DuckDBConnection,
) -> None:
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01T00:00:00Z"]),
            "name": ["buf"],
            "count": [1],
        }
    )
    result = db_conn._normalize_datetime_columns(df)
    assert result["name"].iloc[0] == "buf"
    assert result["count"].iloc[0] == 1


def test_normalize_datetime_columns_no_datetime_columns(
    db_conn: DuckDBConnection,
) -> None:
    df = pd.DataFrame({"name": ["a", "b"], "count": [1, 2]})
    result = db_conn._normalize_datetime_columns(df)
    assert list(result.columns) == ["name", "count"]


# ---------------------------------------------------------------------
# _normalize_parquet_datetime_columns (lives on ParquetManager so tests
# exercise the same code path that production hits via upsert_silver)
# ---------------------------------------------------------------------


def test_normalize_parquet_datetime_columns_rejects_naive(
    parquet_manager: ParquetManager,
) -> None:
    df = pd.DataFrame({"game_date": pd.to_datetime(["2024-01-01T00:00:00"])})
    with pytest.raises(ValueError) as excinfo:
        parquet_manager._normalize_parquet_datetime_columns(df)
    assert "naive" in str(excinfo.value)


def test_normalize_parquet_datetime_columns_accepts_aware(
    parquet_manager: ParquetManager,
) -> None:
    df = pd.DataFrame({"game_date": pd.to_datetime(["2024-01-01T00:00:00Z"])})
    result = parquet_manager._normalize_parquet_datetime_columns(df)
    assert result["game_date"].dt.tz is not None


# ---------------------------------------------------------------------
# ensure_utc_aware helper
# ---------------------------------------------------------------------


def test_ensure_utc_aware_utc_aware_passthrough() -> None:
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    result = ensure_utc_aware(dt)
    assert result == dt
    assert result.tzinfo is UTC


def test_ensure_utc_aware_converts_non_utc_aware() -> None:
    dt = datetime(2024, 1, 1, 14, 0, tzinfo=ET)
    result = ensure_utc_aware(dt)
    assert result.tzinfo is UTC
    # 14:00 ET in January = 19:00 UTC (winter, no DST)
    assert result.hour == 19


def test_ensure_utc_aware_reinterprets_naive_as_utc() -> None:
    dt = datetime(2024, 1, 1, 12, 0)  # naive
    result = ensure_utc_aware(dt)
    assert result.tzinfo is UTC
    # Wall time preserved (no clock shift)
    assert result.hour == 12


def test_ensure_utc_aware_raises_on_none() -> None:
    with pytest.raises(ValueError):
        ensure_utc_aware(None)  # type: ignore[arg-type]


def test_ensure_utc_aware_raises_on_non_datetime() -> None:
    with pytest.raises(ValueError):
        ensure_utc_aware("2024-01-01")  # type: ignore[arg-type]
