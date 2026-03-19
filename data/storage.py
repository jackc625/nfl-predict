"""Data storage utilities using DuckDB and Parquet."""

import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from conf.settings import get_settings
from utils import DataIngestionError, get_logger

logger = get_logger(__name__)


class DuckDBConnection:
    """DuckDB connection manager with utilities."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize DuckDB connection.

        Args:
            db_path: Path to DuckDB file (None for in-memory)
        """
        self.db_path = db_path
        self._connection = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._connection is None:
            try:
                if self.db_path:
                    # Ensure directory exists
                    Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                    self._connection = duckdb.connect(self.db_path)
                    logger.info("Connected to DuckDB file", db_path=self.db_path)
                else:
                    self._connection = duckdb.connect()
                    logger.info("Connected to DuckDB in-memory")

                # Configure DuckDB settings
                self._connection.execute("SET memory_limit='4GB'")
                self._connection.execute("SET threads=4")

            except Exception as e:
                logger.error("Failed to connect to DuckDB", error=str(e))
                raise DataIngestionError(f"DuckDB connection failed: {e}")

        return self._connection

    def close(self) -> None:
        """Close DuckDB connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("DuckDB connection closed")

    def execute(self, query: str, parameters: dict | None = None):
        """Execute SQL query."""
        conn = self.connect()
        try:
            if parameters:
                result = conn.execute(query, parameters)
            else:
                result = conn.execute(query)
            logger.debug("Executed query", query=query[:100] + "...")
            return result
        except Exception as e:
            logger.error("Query execution failed", query=query[:100], error=str(e))
            raise DataIngestionError(f"Query failed: {e}")

    def fetchall(self, query: str, parameters: dict | None = None) -> list[tuple]:
        """Execute query and fetch all results."""
        result = self.execute(query, parameters)
        return result.fetchall()

    def fetch_df(self, query: str, parameters: dict | None = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        result = self.execute(query, parameters)
        return result.df()

    def create_table_from_df(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"
    ) -> None:
        """Create table from DataFrame."""
        conn = self.connect()
        sanitized_name = self._sanitize_table_name(table_name)

        try:
            if if_exists == "replace":
                conn.execute(f"DROP TABLE IF EXISTS {sanitized_name}")
            elif if_exists == "fail" and self.table_exists(sanitized_name):
                raise DataIngestionError(f"Table {sanitized_name} already exists")

            # Normalize datetime columns to UTC consistently
            df_copy = self._normalize_datetime_columns(df)

            temp_table = f"temp_{sanitized_name}"
            conn.register(temp_table, df_copy)
            conn.execute(f"CREATE TABLE {sanitized_name} AS SELECT * FROM {temp_table}")
            conn.unregister(temp_table)

            logger.info(
                "Created table from DataFrame",
                table=sanitized_name,
                rows=len(df_copy),
                columns=len(df_copy.columns),
            )
        except Exception as e:
            logger.error("Failed to create table", table=sanitized_name, error=str(e))
            raise DataIngestionError(f"Table creation failed: {e}")

    def _normalize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime columns to UTC naive for consistent storage."""
        df_copy = df.copy()

        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                if df_copy[col].dt.tz is not None:
                    # Convert timezone-aware to UTC naive datetime
                    df_copy[col] = df_copy[col].dt.tz_convert("UTC")
                else:
                    # Convert timezone-naive to UTC timezone-aware for DuckDB compatibility
                    # (assumes input is already UTC if timezone-naive)
                    df_copy[col] = df_copy[col].dt.tz_localize("UTC")
                # Ensure all datetime columns are consistently stored as UTC
                # (assumes input is already UTC if timezone-naive)

        return df_copy

    def _sanitize_table_name(self, table_name: str) -> str:
        """Sanitize table name to prevent SQL injection."""
        if not table_name:
            raise ValueError("Table name cannot be empty")

        # Allow only alphanumeric characters and underscores
        if not re.match(r"^[A-Za-z0-9_]+$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. Only alphanumeric characters and underscores allowed."
            )

        # Prevent reserved words and ensure reasonable length
        reserved_words = {
            "select",
            "insert",
            "update",
            "delete",
            "drop",
            "create",
            "alter",
            "truncate",
        }
        if table_name.lower() in reserved_words:
            raise ValueError(f"Table name '{table_name}' is a reserved word")

        if len(table_name) > 64:
            raise ValueError(f"Table name too long: {len(table_name)} chars (max 64)")

        return table_name

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists using DuckDB information_schema."""
        try:
            sanitized_name = self._sanitize_table_name(table_name)
            result = self.execute(
                "SELECT 1 FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
                [sanitized_name],
            )
            return len(result.fetchall()) > 0
        except Exception as e:
            logger.debug(f"Table existence check failed for {table_name}: {e}")
            return False

    def fetchone(self, query: str, parameters: list | None = None):
        """Execute query and fetch one result."""
        result = self.execute(query, parameters)
        return result.fetchone()

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get table schema information."""
        sanitized_name = self._sanitize_table_name(table_name)
        return self.fetch_df(f"DESCRIBE {sanitized_name}")

    def _normalize_parquet_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime columns for consistent Parquet storage.

        Converts all datetime columns to UTC timezone-aware timestamps with consistent string format.
        This prevents timezone/format issues when round-tripping through Parquet.
        """
        df_copy = df.copy()

        for col in df_copy.columns:
            if col == "snapshot_ts" and df_copy[col].dtype == "object":
                # Handle mixed timestamp objects in snapshot_ts column
                df_copy[col] = df_copy[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                # Handle datetime columns consistently
                if df_copy[col].dt.tz is not None:
                    # Convert timezone-aware to UTC, then store as UTC timestamp with Arrow
                    df_copy[col] = df_copy[col].dt.tz_convert("UTC")
                else:
                    # Assume naive datetimes are already UTC, make them timezone-aware
                    df_copy[col] = df_copy[col].dt.tz_localize("UTC")
            elif df_copy[col].dtype == "object":
                # Handle mixed timestamp objects in other columns
                def normalize_timestamp(x):
                    if pd.isna(x):
                        return x

                    # Convert various timestamp formats to consistent UTC string
                    try:
                        if hasattr(x, "tz_convert"):  # pandas Timestamp with timezone
                            return x.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S.%fZ")
                        if hasattr(x, "strftime"):  # pandas Timestamp or datetime
                            # Assume naive timestamps are UTC
                            return x.strftime("%Y-%m-%d %H:%M:%S.%fZ")
                        if hasattr(x, "isoformat"):  # datetime object
                            if hasattr(x, "tzinfo") and x.tzinfo is not None:
                                # Convert timezone-aware to UTC
                                utc_dt = x.astimezone(UTC)
                                return utc_dt.strftime("%Y-%m-%d %H:%M:%S.%fZ")
                            # Assume naive datetime is UTC
                            return x.strftime("%Y-%m-%d %H:%M:%S.%fZ")
                        return str(x)
                    except Exception:
                        return str(x)

                # Only apply to columns that might contain timestamps
                if (
                    "time" in col.lower()
                    or "date" in col.lower()
                    or "_ts" in col.lower()
                ):
                    try:
                        df_copy[col] = df_copy[col].apply(normalize_timestamp)
                    except Exception as e:
                        logger.warning(
                            f"Failed to normalize timestamp column {col}: {e}"
                        )
                        df_copy[col] = df_copy[col].astype(str)

        return df_copy


class ParquetManager:
    """Parquet file management utilities."""

    def __init__(self, base_path: str | None = None):
        """
        Initialize Parquet manager.

        Args:
            base_path: Base directory for Parquet files
        """
        settings = get_settings()
        self.base_path = (
            Path(base_path) if base_path else Path(settings.config.data.root_path)
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _normalize_parquet_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime columns for consistent Parquet storage.

        Converts all datetime columns to UTC timezone-aware timestamps with consistent string format.
        This prevents timezone/format issues when round-tripping through Parquet.
        """
        df_copy = df.copy()

        for col in df_copy.columns:
            if col == "snapshot_ts" and df_copy[col].dtype == "object":
                # Handle mixed timestamp objects in snapshot_ts column
                df_copy[col] = df_copy[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                # Handle datetime columns consistently
                if df_copy[col].dt.tz is not None:
                    # Convert timezone-aware to UTC, then store as UTC timestamp with Arrow
                    df_copy[col] = df_copy[col].dt.tz_convert("UTC")
                else:
                    # Assume naive datetimes are already UTC, make them timezone-aware
                    df_copy[col] = df_copy[col].dt.tz_localize("UTC")
            elif df_copy[col].dtype == "object":
                # Handle mixed timestamp objects in other columns
                def normalize_timestamp(x):
                    if pd.isna(x):
                        return x

                    # Convert various timestamp formats to consistent UTC string
                    try:
                        if hasattr(x, "tz_convert"):  # pandas Timestamp with timezone
                            return x.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S.%fZ")
                        if hasattr(x, "strftime"):  # pandas Timestamp or datetime
                            # Assume naive timestamps are UTC
                            return x.strftime("%Y-%m-%d %H:%M:%S.%fZ")
                        if hasattr(x, "isoformat"):  # datetime object
                            if hasattr(x, "tzinfo") and x.tzinfo is not None:
                                # Convert timezone-aware to UTC
                                utc_dt = x.astimezone(UTC)
                                return utc_dt.strftime("%Y-%m-%d %H:%M:%S.%fZ")
                            # Assume naive datetime is UTC
                            return x.strftime("%Y-%m-%d %H:%M:%S.%fZ")
                        return str(x)
                    except Exception:
                        return str(x)

                # Only apply to columns that might contain timestamps
                if (
                    "time" in col.lower()
                    or "date" in col.lower()
                    or "_ts" in col.lower()
                ):
                    try:
                        df_copy[col] = df_copy[col].apply(normalize_timestamp)
                    except Exception as e:
                        logger.warning(
                            f"Failed to normalize timestamp column {col}: {e}"
                        )
                        df_copy[col] = df_copy[col].astype(str)

        return df_copy

    def save(
        self,
        df: pd.DataFrame,
        path: str,
        partition_cols: list[str] | None = None,
        compression: str = "snappy",
    ) -> None:
        """
        Save DataFrame as Parquet file.

        Args:
            df: DataFrame to save
            path: Relative path from base_path
            partition_cols: Columns to partition by
            compression: Compression algorithm
        """
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Normalize datetime columns for consistent Parquet storage
            df_copy = self._normalize_parquet_datetime_columns(df)

            table = pa.Table.from_pandas(df_copy)

            if partition_cols:
                # Partitioned dataset
                pq.write_to_dataset(
                    table,
                    root_path=full_path.parent,
                    partition_cols=partition_cols,
                    compression=compression,
                )
                logger.info(
                    "Saved partitioned Parquet dataset",
                    path=str(full_path),
                    rows=len(df_copy),
                    partitions=partition_cols,
                )
            else:
                # Single file
                pq.write_table(table, full_path, compression=compression)
                logger.info(
                    "Saved Parquet file",
                    path=str(full_path),
                    rows=len(df_copy),
                    columns=len(df_copy.columns),
                )

        except Exception as e:
            logger.error(
                "Failed to save Parquet file", path=str(full_path), error=str(e)
            )
            raise DataIngestionError(f"Parquet save failed: {e}")

    def load(
        self,
        path: str,
        columns: list[str] | None = None,
        filters: list[tuple] | None = None,
    ) -> pd.DataFrame:
        """
        Load DataFrame from Parquet file or partitioned dataset.

        Args:
            path: Relative path from base_path
            columns: Columns to load (None for all)
            filters: PyArrow filters

        Returns:
            Loaded DataFrame
        """
        full_path = self.base_path / path

        try:
            # Try to load as single file first
            if full_path.exists() and full_path.is_file():
                df = pd.read_parquet(
                    full_path, columns=columns, filters=filters, engine="pyarrow"
                )
                logger.info(
                    "Loaded Parquet file",
                    path=str(full_path),
                    rows=len(df),
                    columns=len(df.columns),
                )
                return df

            # Try to load as partitioned dataset
            parent_dir = full_path.parent
            if parent_dir.exists():
                # Look for partition directories that follow the expected naming pattern
                table_name = full_path.stem  # e.g., "games" from "games.parquet"

                # Handle different partition types
                if table_name == "odds_snapshot":
                    # Look specifically for snapshot_ts partition directories
                    partition_dirs = [
                        d
                        for d in parent_dir.iterdir()
                        if d.is_dir()
                        and d.name.startswith("snapshot_ts=")
                        and any(d.glob("*.parquet"))
                    ]
                    partition_col = "snapshot_ts"
                elif table_name == "games":
                    # Look specifically for season partition directories
                    partition_dirs = [
                        d
                        for d in parent_dir.iterdir()
                        if d.is_dir()
                        and d.name.startswith("season=")
                        and any(d.glob("*.parquet"))
                    ]
                    partition_col = "season"
                else:
                    # Generic partition detection
                    partition_dirs = [
                        d
                        for d in parent_dir.iterdir()
                        if d.is_dir() and "=" in d.name and any(d.glob("*.parquet"))
                    ]
                    partition_col = None

                if partition_dirs:
                    # Read from individual partition directories and combine
                    dfs = []
                    for partition_dir in partition_dirs:
                        try:
                            partition_df = pd.read_parquet(
                                partition_dir,
                                columns=columns,
                                filters=filters,
                                engine="pyarrow",
                            )
                        except Exception as e:
                            if "timezone" in str(e) or "zone offset" in str(e):
                                # Fallback to fastparquet which handles timezones more gracefully
                                logger.warning(
                                    "PyArrow failed with timezone issue, trying fastparquet",
                                    partition_dir=str(partition_dir),
                                    error=str(e),
                                )
                                try:
                                    partition_df = pd.read_parquet(
                                        partition_dir,
                                        columns=columns,
                                        engine="fastparquet",
                                    )
                                except ImportError:
                                    # If fastparquet not available, try forcing string conversion
                                    logger.warning(
                                        "fastparquet not available, skipping problematic partition",
                                        partition_dir=str(partition_dir),
                                    )
                                    continue
                            else:
                                raise

                        # Extract partition value from directory name and add as column if missing
                        if partition_col and partition_col not in partition_df.columns:
                            if partition_col == "snapshot_ts":
                                # Handle snapshot_ts encoding
                                partition_value = (
                                    partition_dir.name.replace("snapshot_ts=", "")
                                    .replace("%20", " ")
                                    .replace("%3A", ":")
                                )
                            elif partition_col == "season":
                                # Handle season partitions
                                partition_value = int(
                                    partition_dir.name.replace("season=", "")
                                )
                            else:
                                # Generic partition value
                                partition_value = partition_dir.name.split("=", 1)[1]

                            partition_df[partition_col] = partition_value

                        dfs.append(partition_df)

                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                        logger.info(
                            "Loaded partitioned Parquet dataset",
                            path=f"{parent_dir}/{table_name} partitions",
                            rows=len(df),
                            columns=len(df.columns),
                        )
                        return df

            raise DataIngestionError(f"Parquet file or dataset not found: {full_path}")

        except Exception as e:
            logger.error(
                "Failed to load Parquet file/dataset", path=str(full_path), error=str(e)
            )
            raise DataIngestionError(f"Parquet load failed: {e}")

    def exists(self, path: str) -> bool:
        """Check if Parquet file or partitioned dataset exists."""
        full_path = self.base_path / path

        # Check for single file
        if full_path.exists() and full_path.is_file():
            return True

        # Check for partitioned dataset (directory structure)
        # Look for any partition directories in the parent directory
        parent_dir = full_path.parent
        if parent_dir.exists():
            # Look for directories that could be partitions
            for item in parent_dir.iterdir():
                if item.is_dir() and any(item.glob("*.parquet")):
                    return True

        return False

    def delete(self, path: str) -> None:
        """Delete Parquet file."""
        full_path = self.base_path / path
        if full_path.exists():
            if full_path.is_file():
                full_path.unlink()
            else:
                # Remove directory recursively
                import shutil

                shutil.rmtree(full_path)
            logger.info("Deleted Parquet file/directory", path=str(full_path))

    def list_files(self, pattern: str = "*.parquet") -> list[Path]:
        """List Parquet files matching pattern."""
        return list(self.base_path.rglob(pattern))

    def get_file_info(self, path: str) -> dict[str, Any]:
        """Get Parquet file metadata."""
        full_path = self.base_path / path

        if not full_path.exists():
            raise DataIngestionError(f"Parquet file not found: {full_path}")

        try:
            parquet_file = pq.ParquetFile(full_path)
            metadata = parquet_file.metadata

            return {
                "num_rows": metadata.num_rows,
                "num_columns": metadata.num_columns,
                "file_size": full_path.stat().st_size,
                "created_by": metadata.created_by,
                "schema": parquet_file.schema_arrow,
                "column_names": parquet_file.schema_arrow.names,
            }
        except Exception as e:
            logger.error(
                "Failed to get Parquet file info", path=str(full_path), error=str(e)
            )
            raise DataIngestionError(f"Parquet info failed: {e}")


# Global instances
_db_connection = None
_parquet_manager = None


def get_db_connection(db_path: str | None = None) -> DuckDBConnection:
    """Get global DuckDB connection instance."""
    global _db_connection

    if _db_connection is None:
        settings = get_settings()
        if db_path is None:
            db_path = getattr(settings, "duckdb_path", None)
        _db_connection = DuckDBConnection(db_path)

    return _db_connection


def get_parquet_manager(base_path: str | None = None) -> ParquetManager:
    """Get global Parquet manager instance."""
    global _parquet_manager

    if _parquet_manager is None:
        _parquet_manager = ParquetManager(base_path)

    return _parquet_manager


@contextmanager
def db_transaction():
    """Context manager for database transactions."""
    conn = get_db_connection()
    try:
        conn.execute("BEGIN TRANSACTION")
        yield conn
        conn.execute("COMMIT")
        logger.debug("Database transaction committed")
    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error("Database transaction rolled back", error=str(e))
        raise


def execute_query(query: str, parameters: dict | None = None):
    """Execute SQL query using global connection."""
    conn = get_db_connection()
    return conn.execute(query, parameters)


def _migrate_schema_for_append(
    existing_df: pd.DataFrame, new_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Migrate schema of existing DataFrame to match new DataFrame format.

    Handles common schema evolution issues:
    - Missing created_at column
    - Different created_at data types (int64 timestamps vs datetime)
    - Missing columns in general

    Args:
        existing_df: Existing DataFrame loaded from storage
        new_df: New DataFrame to be appended

    Returns:
        Migrated existing DataFrame with consistent schema
    """
    migrated_df = existing_df.copy()

    # Handle created_at column migration
    if "created_at" in new_df.columns:
        if "created_at" not in migrated_df.columns:
            # Missing created_at column - add with default timestamp
            logger.info("Adding missing created_at column to existing data")
            migrated_df["created_at"] = datetime.now(UTC)

        elif migrated_df["created_at"].dtype in ["int64", "float64"]:
            # Convert legacy int64/float64 timestamps to datetime
            logger.info(
                "Converting legacy timestamp format in created_at column",
                existing_dtype=str(migrated_df["created_at"].dtype),
            )
            try:
                # Assume timestamps are Unix timestamps (seconds since epoch)
                migrated_df["created_at"] = pd.to_datetime(
                    migrated_df["created_at"], unit="s", utc=True
                )
            except Exception as e:
                logger.warning(
                    "Failed to convert legacy timestamps, using current time",
                    error=str(e),
                )
                migrated_df["created_at"] = datetime.now(UTC)

        elif migrated_df["created_at"].dtype == "object":
            # Handle mixed object types in created_at
            logger.info("Converting object type created_at column to datetime")
            try:
                migrated_df["created_at"] = pd.to_datetime(
                    migrated_df["created_at"], utc=True
                )
            except Exception as e:
                logger.warning(
                    "Failed to convert object timestamps, using current time",
                    error=str(e),
                )
                migrated_df["created_at"] = datetime.now(UTC)

    # Handle other missing columns by adding them with NaN/None values
    for col in new_df.columns:
        if col not in migrated_df.columns:
            logger.info("Adding missing column to existing data", column=col)
            migrated_df[col] = None

    # Ensure column order matches (for consistency)
    migrated_df = migrated_df.reindex(columns=new_df.columns, fill_value=None)

    # Normalize datetime columns to be timezone-aware for consistency with new data
    for col in migrated_df.columns:
        if pd.api.types.is_datetime64_any_dtype(migrated_df[col]):
            if migrated_df[col].dt.tz is None:
                # Convert timezone-naive to UTC timezone-aware for DuckDB compatibility
                migrated_df[col] = migrated_df[col].dt.tz_localize("UTC")

    return migrated_df


def save_dataframe(
    df: pd.DataFrame,
    table_name: str,
    layer: str = "silver",
    partition_cols: list[str] | None = None,
    save_to_db: bool = True,
    save_to_parquet: bool = True,
    append_mode: bool = True,
) -> None:
    """
    Save DataFrame to both DuckDB and Parquet.

    Args:
        df: DataFrame to save
        table_name: Name of the table
        layer: Data layer (bronze, silver, gold)
        partition_cols: Parquet partitioning columns
        save_to_db: Whether to save to DuckDB
        save_to_parquet: Whether to save to Parquet
        append_mode: Whether to append to existing data (default: True)
    """
    combined_df = df

    if append_mode and save_to_parquet:
        # Check if existing data exists and merge
        try:
            pm = get_parquet_manager()
            parquet_path = f"{layer}/{table_name}.parquet"
            existing_df = pm.load(parquet_path)

            if not existing_df.empty:
                # Handle schema migration for created_at column
                existing_df = _migrate_schema_for_append(existing_df, df)

                # Identify unique key for deduplication (assume game_id if exists)
                if "game_id" in df.columns:
                    # Remove any existing rows with same game_id to avoid duplicates
                    existing_df = existing_df[
                        ~existing_df["game_id"].isin(df["game_id"])
                    ]

                # Combine existing and new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                logger.info(
                    "Appended to existing data",
                    existing_rows=len(existing_df),
                    new_rows=len(df),
                    total_rows=len(combined_df),
                )
            else:
                logger.info(
                    "No existing data found, creating new dataset", rows=len(df)
                )

        except Exception as e:
            logger.info(
                "No existing data to append to, creating new dataset",
                error=str(e),
                rows=len(df),
            )

    if save_to_db:
        db = get_db_connection()
        db.create_table_from_df(combined_df, table_name, if_exists="replace")

    if save_to_parquet:
        pm = get_parquet_manager()
        parquet_path = f"{layer}/{table_name}.parquet"
        pm.save(combined_df, parquet_path, partition_cols=partition_cols)

    logger.info(
        "Saved DataFrame",
        table=table_name,
        layer=layer,
        rows=len(combined_df),
        saved_to_db=save_to_db,
        saved_to_parquet=save_to_parquet,
    )


def load_dataframe(
    table_name: str,
    layer: str = "silver",
    source: str = "auto",
    columns: list[str] | None = None,
    filters: list[tuple] | None = None,
) -> pd.DataFrame:
    """
    Load DataFrame from DuckDB or Parquet.

    Args:
        table_name: Name of the table
        layer: Data layer (bronze, silver, gold)
        source: Source type ("auto", "db", "parquet")
        columns: Columns to load
        filters: Filters to apply

    Returns:
        Loaded DataFrame
    """
    pm = get_parquet_manager()
    db = get_db_connection()

    parquet_path = f"{layer}/{table_name}.parquet"

    if source == "auto":
        # Try DuckDB first, fall back to Parquet
        if db.table_exists(table_name):
            source = "db"
        elif pm.exists(parquet_path):
            source = "parquet"
        else:
            raise DataIngestionError(f"Table {table_name} not found in DB or Parquet")

    if source == "db":
        # Sanitize table name for security
        sanitized_name = db._sanitize_table_name(table_name)

        if columns:
            # Sanitize column names too
            sanitized_columns = []
            for col in columns:
                if not re.match(r"^[A-Za-z0-9_]+$", col):
                    raise ValueError(f"Invalid column name: {col}")
                sanitized_columns.append(col)
            query = f"SELECT {', '.join(sanitized_columns)} FROM {sanitized_name}"
        else:
            query = f"SELECT * FROM {sanitized_name}"

        df = db.fetch_df(query)

    elif source == "parquet":
        df = pm.load(parquet_path, columns=columns, filters=filters)

    else:
        raise ValueError(f"Invalid source: {source}")

    logger.info(
        "Loaded DataFrame",
        table=table_name,
        layer=layer,
        source=source,
        rows=len(df),
        columns=len(df.columns),
    )

    return df


def save_bronze_snapshot(
    df: pd.DataFrame,
    table_name: str,
    season: int,
    week: int,
    base_path: Path | None = None,
) -> Path:
    """Save DataFrame as timestamped Bronze Parquet file (append-only).

    Each call creates a NEW file. Old files are never modified.

    Args:
        df: Raw data to snapshot
        table_name: e.g. "games", "odds", "weather"
        season: NFL season year
        week: NFL week number
        base_path: Base data directory (default from settings)

    Returns:
        Path to the created Bronze file
    """
    if base_path is None:
        settings = get_settings()
        base_path = Path(settings.config.data.root_path)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    filename = f"{table_name}_raw_bronze_{season}_W{week:02d}_{ts}.parquet"
    filepath = base_path / "bronze" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath, compression="snappy")

    logger.info("Saved Bronze snapshot", path=str(filepath), rows=len(df))
    return filepath


def upsert_silver(
    new_df: pd.DataFrame,
    table_name: str,
    key_column: str = "game_id",
    base_path: Path | None = None,
) -> Path:
    """Upsert new data into Silver layer (latest wins by key_column).

    If Silver file exists, removes rows with matching keys, then appends new data.
    If Silver file does not exist, creates it from new data.

    Args:
        new_df: Validated data to upsert
        table_name: e.g. "games", "odds_snapshot", "weather"
        key_column: Column to match on for upsert (default: game_id)
        base_path: Base data directory (default from settings)

    Returns:
        Path to the Silver file
    """
    if base_path is None:
        settings = get_settings()
        base_path = Path(settings.config.data.root_path)

    silver_path = base_path / "silver" / f"{table_name}.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    if silver_path.exists():
        existing = pd.read_parquet(silver_path, engine="pyarrow")
        # Remove rows that match any key in new data (latest wins)
        existing = existing[~existing[key_column].isin(new_df[key_column])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    # Normalize datetime columns before writing
    pm = ParquetManager(str(base_path))
    combined_normalized = pm._normalize_parquet_datetime_columns(combined)
    table = pa.Table.from_pandas(combined_normalized)
    pq.write_table(table, silver_path, compression="snappy")

    logger.info(
        "Upserted Silver table",
        table=table_name,
        path=str(silver_path),
        rows=len(combined),
    )
    return silver_path


def get_latest_bronze_file(
    table_name: str,
    season: int,
    week: int,
    base_path: Path | None = None,
) -> Path | None:
    """Find the latest Bronze snapshot file for a given table/season/week.

    Returns:
        Path to most recent file, or None if no files exist
    """
    if base_path is None:
        settings = get_settings()
        base_path = Path(settings.config.data.root_path)

    bronze_dir = base_path / "bronze"
    if not bronze_dir.exists():
        return None

    pattern = f"{table_name}_raw_bronze_{season}_W{week:02d}_*.parquet"
    files = sorted(bronze_dir.glob(pattern))
    return files[-1] if files else None


def create_data_directories() -> None:
    """Create all required data directories."""
    settings = get_settings()

    directories = [
        settings.get_data_path("bronze"),
        settings.get_data_path("silver"),
        settings.get_data_path("gold"),
        settings.get_data_path("outputs"),
        settings.get_data_path("artifacts"),
        Path("logs"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

        # Create .gitkeep file to track empty directories
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    logger.info("Created data directories", directories=[str(d) for d in directories])


def optimize_database() -> None:
    """Optimize DuckDB database."""
    db = get_db_connection()

    try:
        # Analyze tables for query optimization
        db.execute("ANALYZE")

        # Vacuum if using file-based database
        if db.db_path:
            db.execute("VACUUM")

        logger.info("Database optimization completed")

    except Exception as e:
        logger.warning("Database optimization failed", error=str(e))


def get_database_stats() -> dict[str, Any]:
    """Get database statistics."""
    db = get_db_connection()

    try:
        # Get table list
        tables_df = db.fetch_df("SHOW TABLES")
        table_stats = {}

        for table_name in tables_df["name"].tolist():
            try:
                # Sanitize table name before using in query
                sanitized_name = db._sanitize_table_name(table_name)
                count_result = db.fetch_df(
                    f"SELECT COUNT(*) as row_count FROM {sanitized_name}"
                )
                row_count = count_result.iloc[0]["row_count"]

                table_info = db.get_table_info(sanitized_name)
                column_count = len(table_info)

                table_stats[table_name] = {
                    "row_count": row_count,
                    "columns": column_count,
                }
            except Exception as e:
                logger.warning(f"Failed to get stats for table {table_name}: {e}")
                table_stats[table_name] = {"row_count": "unknown", "columns": "unknown"}

        return {
            "num_tables": len(tables_df),
            "table_stats": table_stats,
            "database_path": db.db_path or "in-memory",
        }

    except Exception as e:
        logger.error("Failed to get database stats", error=str(e))
        return {"error": str(e)}
