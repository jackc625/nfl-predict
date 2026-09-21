"""Data storage utilities using DuckDB and Parquet."""

import re
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from conf.settings import get_settings
from utils import DataIngestionError, get_logger

logger = get_logger(__name__)

# pyarrow is a hard dependency of this module (pa.Table.from_pandas,
# pq.write_table, etc.) so ``pyarrow.lib`` should always be importable.
# The guard exists so narrowing the parquet read/write catches never crashes
# on an unexpectedly slim pyarrow build -- if the symbols cannot be resolved,
# we fall back to an empty tuple so ``except (..., *_PYARROW_EXCEPTIONS)``
# becomes a no-op for pyarrow-specific errors (D-12 review item #11).
try:
    import pyarrow.lib as _pa_lib  # type: ignore[import-untyped]

    _PYARROW_EXCEPTIONS: tuple[type[BaseException], ...] = (
        _pa_lib.ArrowInvalid,
        _pa_lib.ArrowIOError,
    )
except (ImportError, AttributeError):  # pragma: no cover -- defensive
    _PYARROW_EXCEPTIONS = ()


def _atomic_write_parquet(
    table: pa.Table, path: Path, compression: str = "snappy"
) -> None:
    """Write *table* to *path* via a temp file plus ``os.replace``.

    Every full-table writer in this module reads the whole table, rebuilds it in
    memory and then rewrites it straight over the live path. A crash, interrupt,
    power loss or full disk partway through that rewrite leaves a truncated or
    zero-length file where a complete one used to be, and there is no second copy
    to recover from (CR-02 of the Phase-29 code review).

    The temp file is deliberately a SIBLING of the target -- ``path`` with a
    ``.tmp`` suffix -- and never a system temp directory. ``os.replace`` is atomic
    only WITHIN a filesystem; across filesystems it degrades to a copy plus
    delete, which reintroduces exactly the partial-write window this function
    exists to close.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path, compression=compression)
    tmp_path.replace(path)


class DuckDBConnection:
    """DuckDB connection manager with utilities.

    OPENS READ-ONLY UNLESS A CALLER ASKS, IN WORDS, TO WRITE (QT-W8X-01).

    MEASURED on duckdb 1.5.0 / Windows 11, two processes over one database file:

    ===============  ===============  =========================================
    holder           second opener    result
    ===============  ===============  =========================================
    read-only        read-only        both OPEN, concurrently
    read-write       read-only        ``IOException: ... by another process``
    read-write       read-write       ``IOException: ... by another process``
    ===============  ===============  =========================================

    This connection is held as a module global (``_db_connection``) for the whole of
    any process that touched ``load_dataframe``, so before this change a purely
    READ-ONLY workload -- a pytest session, a report script -- locked
    ``data/nfl_predictions.duckdb`` against everything else on the machine,
    including a plain ``open(path, "rb")``. Nothing that only reads should hold a
    write lock.

    The fix is a DEFAULT, not a heuristic. ``connect()`` with no argument is a READ
    connection; a caller that needs to write says ``connect(write=True)`` at the
    call site. Write intent is deliberately NEVER inferred from the text of a query:
    a regex over SQL that decides whether to take a write lock is exactly the
    implicit default this change removes, and an AST scan in
    ``tests/unit/test_storage_connection_mode.py`` asserts against it.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize DuckDB connection.

        Args:
            db_path: Path to DuckDB file (None for in-memory)
        """
        self.db_path = db_path
        self._connection = None
        # None when no handle is live. Tracks the mode of the handle that IS live,
        # so `connect(write=True)` can tell "already sufficient" from "must upgrade"
        # and so `close()` can de-escalate back to the safe default.
        self._write_mode: bool | None = None

    @property
    def connection_mode(self) -> str | None:
        """``"read"``, ``"write"``, or None when no handle is live."""
        if self._connection is None or self._write_mode is None:
            return None
        return "write" if self._write_mode else "read"

    def connect(self, *, write: bool = False) -> duckdb.DuckDBPyConnection:
        """Get or open the DuckDB handle; read-only unless *write* is True.

        Mode changes, and why each is what it is:

        * ``write=True`` over a live READ handle is an UPGRADE: the read handle is
          CLOSED and a read-write one opened. duckdb caches the database instance
          per process and refuses a second connection onto one file under a
          different configuration (MEASURED: ``ConnectionException: Can't open a
          connection to same database file with a different configuration``), so
          reusing the read handle is not possible and neither is opening alongside
          it. Close-then-reopen is required, not stylistic.
        * ``write=False`` over a live WRITE handle returns the existing handle. It is
          NOT auto-downgraded: a live write handle belongs to a caller that is
          mid-work. De-escalation happens at ``close()``, which resets the recorded
          mode so the next lazy open is read-only again -- which is what makes
          ``close_db_connection`` (and through it the test suite's
          ``close_probable_holders``) a real seam rather than a no-op.
        """
        if self._connection is not None:
            if write and not self._write_mode:
                logger.info(
                    "Upgrading the DuckDB handle to read-write",
                    db_path=self.db_path,
                )
                self.close()
            else:
                return self._connection

        try:
            if not self.db_path:
                # In-memory. Locks nothing, so there is nothing to protect.
                self._connection = duckdb.connect()
                self._write_mode = True
                logger.info("Connected to DuckDB in-memory")
            elif write:
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self._connection = duckdb.connect(self.db_path)
                self._write_mode = True
                logger.info(
                    "Connected to DuckDB file", db_path=self.db_path, mode="read-write"
                )
            elif Path(self.db_path).exists():
                # THE BRANCH THE PRODUCTION DATABASE TAKES on every ordinary session.
                try:
                    self._connection = duckdb.connect(self.db_path, read_only=True)
                    self._write_mode = False
                    logger.info(
                        "Connected to DuckDB file",
                        db_path=self.db_path,
                        mode="read-only",
                    )
                except (duckdb.Error, OSError) as read_only_failure:
                    # A `.duckdb.wal` sibling awaiting replay is the realistic case:
                    # duckdb cannot replay a write-ahead log through a read-only
                    # handle. Availability wins here, but LOUDLY -- this fallback
                    # must not be reachable on a healthy database, and the warning
                    # is how a reader finds out if it ever becomes the normal path.
                    logger.warning(
                        "Read-only DuckDB open failed; taking a read-write handle",
                        db_path=self.db_path,
                        error=str(read_only_failure),
                        exception_type=type(read_only_failure).__name__,
                    )
                    Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                    self._connection = duckdb.connect(self.db_path)
                    self._write_mode = True
            else:
                # Read mode over a file that does not exist. Read-only cannot open
                # one, and an absent file has nothing to protect, so today's
                # behaviour is kept EXACTLY: make the parent and open read-write,
                # which brings the database into being. This is what keeps every
                # sandbox and temp-root test working unchanged -- a `load_dataframe`
                # against a fresh root still falls through to parquet.
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self._connection = duckdb.connect(self.db_path)
                self._write_mode = True
                logger.info(
                    "Connected to DuckDB file",
                    db_path=self.db_path,
                    mode="read-write",
                    absent_before=True,
                )

            # Session configuration, issued on every handle. PROBED on duckdb 1.5.0:
            # both statements are ACCEPTED on a read-only handle -- neither is a
            # write to the database -- so the read path issues them unchanged.
            self._connection.execute("SET memory_limit='4GB'")
            self._connection.execute("SET threads=4")

        except (duckdb.Error, OSError) as e:
            logger.error(
                "Failed to connect to DuckDB",
                db_path=self.db_path,
                error=str(e),
                exception_type=type(e).__name__,
            )
            raise DataIngestionError(f"DuckDB connection failed: {e}") from e

        return self._connection

    def close(self) -> None:
        """Close DuckDB connection, resetting the recorded mode to the safe default."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("DuckDB connection closed")
        self._write_mode = None

    def execute(
        self, query: str, parameters: dict | None = None, *, write: bool = False
    ):
        """Execute SQL query.

        *write* is the caller's DECLARATION that this statement mutates the
        database. It is keyword-only so a call site reads as ``execute(sql,
        write=True)`` rather than as an anonymous boolean, and it defaults to False
        so a caller that says nothing gets a read-only handle.
        """
        conn = self.connect(write=write)
        try:
            if parameters:
                result = conn.execute(query, parameters)
            else:
                result = conn.execute(query)
            logger.debug("Executed query", query=query[:100] + "...")
            return result
        except (duckdb.Error, OSError) as e:
            logger.error(
                "Query execution failed",
                query=query[:100],
                error=str(e),
                exception_type=type(e).__name__,
            )
            raise DataIngestionError(f"Query failed: {e}") from e

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
        """Create table from DataFrame.

        A NAMED WRITE: this DROPs and re-creates a table, so it asks for the
        read-write handle explicitly rather than inheriting one by accident.
        """
        conn = self.connect(write=True)
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
        except (duckdb.Error, ValueError, TypeError) as e:
            logger.error(
                "Failed to create table",
                table=sanitized_name,
                error=str(e),
                exception_type=type(e).__name__,
            )
            raise DataIngestionError(f"Table creation failed: {e}") from e

    def _normalize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize datetime columns to UTC for consistent storage.

        All datetime columns reaching storage MUST be timezone-aware.
        Timezone-aware columns are converted to UTC. Naive datetime
        columns raise ValueError -- the previous "assume naive == UTC"
        pattern silently corrupted data when sources actually used a
        non-UTC timezone (Phase 15-04 D-15).

        Raises:
            ValueError: If any datetime column is timezone-naive. Use
                ``utils.date_utils.ensure_utc_aware()`` at the producer
                site to fix-forward, or pass tz-aware datetimes
                (e.g. ``datetime.now(UTC)``) at construction.
        """
        df_copy = df.copy()

        for col in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                if df_copy[col].dt.tz is not None:
                    # Convert timezone-aware to UTC
                    df_copy[col] = df_copy[col].dt.tz_convert("UTC")
                else:
                    raise ValueError(
                        f"Column '{col}' contains naive (timezone-unaware) "
                        f"datetimes. All datetime columns must be timezone-aware. "
                        f"Use utils.date_utils.ensure_utc_aware() at the producer "
                        f"site or pass tz-aware datetimes (e.g. datetime.now(UTC))."
                    )

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
        except (duckdb.Error, DataIngestionError, ValueError) as e:
            # DataIngestionError is raised by self.execute on query failure.
            # ValueError comes from _sanitize_table_name on invalid names.
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

        Converts timezone-aware datetime columns to UTC and preserves a
        consistent string format for object-typed timestamp columns. This
        prevents timezone/format issues when round-tripping through
        Parquet.

        All datetime64-typed columns reaching storage MUST be tz-aware.
        Naive datetime columns raise ValueError -- the previous
        "assume naive == UTC" pattern silently corrupted data when sources
        actually used a non-UTC timezone (Phase 15-04 D-15).

        Object-typed columns containing strings, naive Python datetimes,
        or pandas Timestamps follow the legacy normalization path below
        (string-formatting), which is not affected by the strict tz-check.

        Raises:
            ValueError: If any datetime64-typed column is timezone-naive.
                Use ``utils.date_utils.ensure_utc_aware()`` to fix-forward.
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
                    raise ValueError(
                        f"Column '{col}' contains naive (timezone-unaware) "
                        f"datetimes. All datetime columns must be timezone-aware "
                        f"before writing to Parquet. Use "
                        f"utils.date_utils.ensure_utc_aware() to fix-forward."
                    )
            elif df_copy[col].dtype == "object":
                # Handle mixed timestamp objects in other columns
                def normalize_timestamp(x):
                    if pd.isna(x):
                        return x

                    # Convert various timestamp formats to consistent UTC string.
                    # The try/except covers the narrow set of errors raised by
                    # pandas Timestamp / datetime strftime/tz_convert/astimezone
                    # on unexpected object types.
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
                    except (ValueError, TypeError, AttributeError, OverflowError):
                        return str(x)

                # Only apply to columns that might contain timestamps
                if (
                    "time" in col.lower()
                    or "date" in col.lower()
                    or "_ts" in col.lower()
                ):
                    try:
                        df_copy[col] = df_copy[col].apply(normalize_timestamp)
                    except (ValueError, TypeError) as e:
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
                # Single file, written atomically -- this is the gold-matrix
                # writer, so an interrupted gold rebuild would otherwise
                # truncate gold (CR-02 / N-04).
                _atomic_write_parquet(table, full_path, compression=compression)
                logger.info(
                    "Saved Parquet file",
                    path=str(full_path),
                    rows=len(df_copy),
                    columns=len(df_copy.columns),
                )

        except (OSError, ValueError, TypeError, *_PYARROW_EXCEPTIONS) as e:
            logger.error(
                "Failed to save Parquet file",
                path=str(full_path),
                error=str(e),
                exception_type=type(e).__name__,
            )
            raise DataIngestionError(f"Parquet save failed: {e}") from e

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
                    # Generic partition detection -- table-scoped ONLY.
                    #
                    # CORRECTNESS FIX (Phase 20-06 FIX-01, D-13): the previous
                    # implementation scanned ``parent_dir`` (the SHARED
                    # ``{layer}/`` root) for ANY ``name=value`` directory. Because
                    # several silver tables historically wrote
                    # ``partition_cols=["season"]`` / ``["target_season"]`` into
                    # that same shared root, a table with no single-file parquet
                    # would silently read EVERY sibling table's partition files --
                    # returning another table's rows mislabeled as its own (e.g.
                    # ``load_dataframe("weather_features")`` returning the
                    # ``snapshot_ts=`` odds partitions). That is cross-table data
                    # contamination, not a partitioned read.
                    #
                    # Scope the generic fallback to partition directories nested
                    # UNDER a table-specific subdirectory ``{layer}/{table}/``;
                    # never read ``=`` siblings in the shared layer root. If the
                    # table has neither a single file nor its own subdirectory of
                    # partitions, fall through to the explicit "not found" error
                    # below -- a loud miss is correct, a silent wrong-table read
                    # is not.
                    table_partition_root = parent_dir / table_name
                    if table_partition_root.is_dir():
                        partition_dirs = [
                            d
                            for d in table_partition_root.iterdir()
                            if d.is_dir() and "=" in d.name and any(d.glob("*.parquet"))
                        ]
                    else:
                        partition_dirs = []
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
                        except (
                            FileNotFoundError,
                            OSError,
                            ValueError,
                            *_PYARROW_EXCEPTIONS,
                        ) as e:
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

        except (
            FileNotFoundError,
            OSError,
            ValueError,
            TypeError,
            DataIngestionError,
            *_PYARROW_EXCEPTIONS,
        ) as e:
            if isinstance(e, DataIngestionError):
                # Re-raise without wrapping so the original "not found" message
                # is preserved for callers.
                raise
            logger.error(
                "Failed to load Parquet file/dataset",
                path=str(full_path),
                error=str(e),
                exception_type=type(e).__name__,
            )
            raise DataIngestionError(f"Parquet load failed: {e}") from e

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
        except (OSError, ValueError, *_PYARROW_EXCEPTIONS) as e:
            logger.error(
                "Failed to get Parquet file info",
                path=str(full_path),
                error=str(e),
                exception_type=type(e).__name__,
            )
            raise DataIngestionError(f"Parquet info failed: {e}") from e


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


def close_db_connection() -> None:
    """Close the global DuckDB connection, releasing its file handle.

    A NARROW, NAMED seam added for D33-32. On Windows DuckDB holds an exclusive
    lock on an open database, so ``data/nfl_predictions.duckdb`` is unreadable for
    the whole of any process that has called ``load_dataframe``. The test suite's
    content-digest boundary guard has to be able to read those bytes to render a
    verdict, and the alternative to this seam was for the guard to reach into
    ``_db_connection`` -- a private module global -- from outside the module.

    The connection object itself is KEPT, so ``db_path`` survives; only the
    underlying handle is dropped. ``DuckDBConnection.connect`` reopens lazily on
    the next call, so this is safe to invoke at any point: callers that need the
    database simply get a fresh connection.
    """
    if _db_connection is not None:
        _db_connection.close()


def get_parquet_manager(base_path: str | None = None) -> ParquetManager:
    """Get global Parquet manager instance."""
    global _parquet_manager

    if _parquet_manager is None:
        _parquet_manager = ParquetManager(base_path)

    return _parquet_manager


@contextmanager
def db_transaction():
    """Context manager for database transactions.

    Catches the broad ``Exception`` on purpose: this is a top-level rollback
    boundary that must fire on ANY failure inside the ``with`` block, not just
    DuckDB errors. Callers will see the original exception re-raised via the
    bare ``raise`` statement, so no information is lost.
    """
    conn = get_db_connection()
    # A transaction boundary IS a mutation boundary -- there is no read-only caller
    # of this context manager -- so the write handle is acquired before BEGIN rather
    # than being inferred from whatever the body turns out to run.
    conn.connect(write=True)
    try:
        conn.execute("BEGIN TRANSACTION")
        yield conn
        conn.execute("COMMIT")
        logger.debug("Database transaction committed")
    except Exception as e:
        conn.execute("ROLLBACK")
        logger.error(
            "Database transaction rolled back",
            error=str(e),
            exception_type=type(e).__name__,
        )
        raise


def execute_query(query: str, parameters: dict | None = None, *, write: bool = False):
    """Execute SQL query using global connection.

    *write* exists so a caller OUTSIDE this module that must issue DDL has a named
    way to say so. The default is unchanged and is read-only.
    """
    conn = get_db_connection()
    return conn.execute(query, parameters, write=write)


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
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(
                    "Failed to convert legacy timestamps, using current time",
                    error=str(e),
                    exception_type=type(e).__name__,
                )
                migrated_df["created_at"] = datetime.now(UTC)

        elif migrated_df["created_at"].dtype == "object":
            # Handle mixed object types in created_at
            logger.info("Converting object type created_at column to datetime")
            try:
                migrated_df["created_at"] = pd.to_datetime(
                    migrated_df["created_at"], utc=True
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to convert object timestamps, using current time",
                    error=str(e),
                    exception_type=type(e).__name__,
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
    replace_mode: bool = False,
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
        replace_mode: If True, write the table as a single self-contained
            Parquet file that fully replaces any prior on-disk state for this
            table -- the DataFrame passed in IS the table. This makes a full
            rebuild idempotent and avoids the directory-partitioned-append
            failure mode (Phase 20-06 FIX-01, D-13): ``pq.write_to_dataset``
            writes partition directories into the SHARED ``{layer}/`` root, so
            multiple tables collide in the same ``season=YYYY/`` directories and
            every rebuild appends a NEW hash-named file instead of overwriting,
            silently multiplying row counts on each run. When ``replace_mode``
            is True, ``partition_cols`` and ``append_mode`` are ignored (a single
            file is written) and any pre-existing partition directories for the
            table are left untouched on disk but no longer participate (the
            single file takes read precedence in ``ParquetManager.load``).
    """
    combined_df = df

    if replace_mode:
        # Idempotent single-file replace: the passed DataFrame is the whole
        # table. Skip the append/dedup merge AND directory partitioning so a
        # rebuild always produces byte-stable cardinality regardless of how
        # many times it runs (FIX-01, D-13).
        partition_cols = None
        append_mode = False

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

        except (DataIngestionError, FileNotFoundError, OSError, ValueError) as e:
            # These are expected "no existing data" signals -- the ParquetManager
            # raises DataIngestionError when the file is missing, and OSError /
            # FileNotFoundError / ValueError can come from early filesystem or
            # schema probes. Any other exception should propagate.
            logger.info(
                "No existing data to append to, creating new dataset",
                error=str(e),
                exception_type=type(e).__name__,
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
    *,
    exclusive: bool = False,
) -> Path:
    """Save DataFrame as timestamped Bronze Parquet file (append-only).

    Each call creates a NEW file. Old files are never modified.

    THE KNOWN DEFECT IN THAT SENTENCE, and what ``exclusive`` does about it (WR-04 of the
    32-REVIEW). The filename carries a SECOND-resolution UTC stamp, so two calls for the
    same ``(table_name, season, week)`` inside one second produce the SAME path and the
    second SILENTLY OVERWRITES the first. The contract above is therefore aspirational by
    default, not enforced.

    ``exclusive=True`` creates the file with ``"xb"``, making the CREATE itself the
    exclusive step -- the only place the race can actually be decided -- so a collision
    raises ``FileExistsError`` BEFORE any byte of the previous snapshot is touched. That
    holds ACROSS PROCESSES, which a check-then-write cannot: two processes each complete
    their check before either writes.

    IT IS OPT-IN, AND THAT IS DELIBERATE RATHER THAN TIMID. ``scripts/ingest_odds_timeline``
    writes one bronze snapshot per trajectory timestamp for a single ``(season, week)``, so
    a fast backfill legitimately produces several calls inside one second and RELIES on the
    overwrite today. That reliance is itself a latent data-loss bug in that ingester -- it
    destroys the earlier snapshots' bytes -- but it is PRE-EXISTING, it belongs to the
    line-movement work rather than to the upstream pin, and flipping it to a hard failure
    for every caller at once would convert a quiet defect in one ingester into a broken
    production path. So the guarantee is given to the callers whose append-only contract is
    load-bearing -- the live zone, whose bronze bytes ARE the evidence a published verdict
    was measured against -- and the defect is named here rather than left implicit.

    Args:
        df: Raw data to snapshot
        table_name: e.g. "games", "odds", "weather"
        season: NFL season year
        week: NFL week number
        base_path: Base data directory (default from settings)
        exclusive: Refuse rather than overwrite when the timestamped path already exists.

    Returns:
        Path to the created Bronze file

    Raises:
        FileExistsError: If ``exclusive`` is set and the path is already taken.
    """
    if base_path is None:
        settings = get_settings()
        base_path = Path(settings.config.data.root_path)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    filename = f"{table_name}_raw_bronze_{season}_W{week:02d}_{ts}.parquet"
    filepath = base_path / "bronze" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Deliberately NOT _atomic_write_parquet: every call writes a NEW timestamped
    # file and never overwrites an existing one, so there is no complete previous
    # file for a partial write to destroy.
    #
    # The "xb" branch is the WR-04 collision guard; see this function's docstring for why
    # it is OPT-IN rather than the default. The live zone's pre-write reservation poll
    # (scripts/capture_live_season._reserve_distinct_bronze_second) narrows the window but
    # cannot close it -- it checks and then writes non-atomically, and across two processes
    # each one's check completes before either writes. Making the CREATE exclusive is the
    # only place the race can actually be decided.
    table = pa.Table.from_pandas(df)
    if exclusive:
        with open(filepath, "xb") as handle:
            pq.write_table(table, handle, compression="snappy")
    else:
        pq.write_table(table, filepath, compression="snappy")

    logger.info("Saved Bronze snapshot", path=str(filepath), rows=len(df))
    return filepath


class SilverMirrorSyncError(DataIngestionError):
    """A silver upsert wrote its parquet but could not bring the reader's DuckDB copy along.

    A subclass of :class:`DataIngestionError`, so every caller that already treats a failed
    ingest as a failed ingest keeps doing so. It has its own name because the state it
    reports is specific and dangerous: the parquet is AHEAD, and the message says whether the
    stale DuckDB copy could at least be dropped (readers then fall back to the parquet) or is
    still there (readers keep serving old rows).
    """


def _reader_parquet_root() -> Path:
    """The parquet root ``load_dataframe`` reads -- WITHOUT creating the global manager.

    ``get_parquet_manager()`` would instantiate the module global (and ``mkdir`` its root)
    as a side effect of merely asking. Reading the existing global, or the configured root
    when there is none yet, answers the same question with no side effect.
    """
    if _parquet_manager is not None:
        return Path(_parquet_manager.base_path)
    return Path(get_settings().config.data.root_path)


def _duckdb_copy_exists(db: DuckDBConnection, table_name: str) -> bool:
    """Whether *table_name* has a DuckDB copy, RAISING when that cannot be established.

    Deliberately NOT ``DuckDBConnection.table_exists``, which answers ``False`` when the
    database cannot be opened at all. Here that answer would be read as "no copy to keep in
    step", and a locked database would silently skip the sync -- the exact split this seam
    exists to close. So a connection or query failure propagates as ``DataIngestionError``.
    """
    sanitized = db._sanitize_table_name(table_name)
    rows = db.execute(
        "SELECT 1 FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
        [sanitized],
    ).fetchall()
    return len(rows) > 0


def _keep_duckdb_copy_in_step(
    table_name: str, silver_path: Path, base_path: Path
) -> None:
    """Replace the reader's DuckDB copy of *table_name* from the parquet just written.

    THE DEFECT (N-01, recurring). ``load_dataframe(source="auto")`` reads DuckDB FIRST
    whenever the table exists there, and the silver upserts used to write the parquet ONLY
    -- so every upsert into a table that also had a DuckDB copy was invisible to the whole
    pipeline. Phase 30 measured it on ``games`` (6,499 parquet against 6,292 DuckDB) and
    healed it by a one-off re-sync; Plan 33-18's live acceptance run reproduced it on
    2026-09-15 (6,771 against 6,499, zero 2026 rows readable) and halted at ``data_qa``.
    Fixed here, at the writer, on owner ruling R1 of 2026-09-15. The pattern is
    ``scripts/build_elo.py::_upsert_row_table``'s: read the written parquet back and replace
    the DuckDB copy from it, so the two stores cannot disagree.

    THREE RULES, each pinned by ``tests/unit/test_upsert_silver_duckdb_sync.py``:

    * ONLY THE READER'S ROOT. If *base_path* is not the root ``load_dataframe`` reads, the
      parquet just written is one no reader resolves, so there is no reader copy to keep in
      step -- and writing the process DuckDB from a foreign root would corrupt it. This is
      also what keeps every sandboxed ``upsert_silver(base_path=tmp_path)`` off the
      production database.
    * ONLY AN EXISTING COPY. A table with no DuckDB copy is already read from its parquet,
      so it is consistent as it stands; creating a mirror nobody reads would only widen what
      a write can break.
    * LOUD ON FAILURE. The parquet is already written, so a DuckDB failure leaves it ahead.
      That is never swallowed: the stale copy is dropped if possible (readers then fall back
      to the parquet), and :class:`SilverMirrorSyncError` is raised either way, saying which.

    ORDERING. Parquet first (atomic, via ``_atomic_write_parquet``), DuckDB second. The
    reverse would let a parquet failure leave DuckDB ahead of the file every direct parquet
    reader uses; this order means the only possible half-state is a raised one.
    """
    if Path(base_path).resolve() != _reader_parquet_root().resolve():
        return

    db = get_db_connection()
    try:
        has_copy = _duckdb_copy_exists(db, table_name)
    except DataIngestionError as unknown:
        raise SilverMirrorSyncError(
            f"silver.{table_name} was written to {silver_path.as_posix()}, but whether a "
            f"DuckDB copy exists could not be established ({unknown}). If one exists it is "
            "now STALE and load_dataframe(source='auto') serves it instead of the parquet. "
            "Close whatever holds the database and re-run the ingest."
        ) from unknown
    if not has_copy:
        return

    written = pd.read_parquet(silver_path, engine="pyarrow")
    try:
        db.create_table_from_df(written, table_name, if_exists="replace")
    except DataIngestionError as sync_failure:
        try:
            db.execute(
                f"DROP TABLE IF EXISTS {db._sanitize_table_name(table_name)}",
                write=True,
            )
            outcome = (
                "The stale DuckDB copy was DROPPED, so readers now fall back to the "
                "parquet and see the written rows."
            )
        except (DataIngestionError, ValueError) as drop_failure:
            outcome = (
                f"The stale DuckDB copy could NOT be dropped either ({drop_failure}), so "
                "load_dataframe(source='auto') is STILL SERVING THE OLD ROWS."
            )
        raise SilverMirrorSyncError(
            f"silver.{table_name} was written to {silver_path.as_posix()} "
            f"({len(written)} rows), but its DuckDB copy could not be replaced: "
            f"{sync_failure}. {outcome} Close whatever holds the database and re-run the "
            "ingest."
        ) from sync_failure

    logger.info(
        "Kept the DuckDB copy in step with the silver parquet",
        table=table_name,
        rows=len(written),
    )


def upsert_silver(
    new_df: pd.DataFrame,
    table_name: str,
    key_column: str = "game_id",
    base_path: Path | None = None,
    *,
    order_rows: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    """Upsert new data into Silver layer (latest wins by key_column).

    If Silver file exists, removes rows with matching keys, then appends new data.
    If Silver file does not exist, creates it from new data.

    If the table ALSO has a DuckDB copy under the root ``load_dataframe`` reads, that copy
    is replaced from the written parquet, and a failure to do so RAISES
    :class:`SilverMirrorSyncError` -- see :func:`_keep_duckdb_copy_in_step`. Until Plan
    33-18 this wrote the parquet only, and ``load_dataframe(source="auto")``, which reads
    DuckDB first, never saw the upsert.

    Args:
        new_df: Validated data to upsert
        table_name: e.g. "games", "odds_snapshot", "weather"
        key_column: Column to match on for upsert (default: game_id)
        base_path: Base data directory (default from settings)
        order_rows: Optional function returning the COMBINED table in the physical row
            order it must be stored in. Latest-wins removes a replaced row and appends
            its successor at the END, so without this a table whose readers depend on
            row order drifts out of it on every upsert that rewrites an earlier row --
            measured on ``elo_game_snapshots`` in 2026, where a real week-1 result landed
            after the provisional week-2 row it preceded. The order is applied before
            the one atomic parquet write, so the DuckDB copy (replaced from that parquet)
            carries the same order and the two stores cannot disagree about it.

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

    if order_rows is not None:
        combined = order_rows(cast("pd.DataFrame", combined)).reset_index(drop=True)

    # Normalize datetime columns before writing
    pm = ParquetManager(str(base_path))
    combined_normalized = pm._normalize_parquet_datetime_columns(combined)
    table = pa.Table.from_pandas(combined_normalized)
    _atomic_write_parquet(table, silver_path)
    _keep_duckdb_copy_in_step(table_name, silver_path, base_path)

    logger.info(
        "Upserted Silver table",
        table=table_name,
        path=str(silver_path),
        rows=len(combined),
    )
    return silver_path


def _canonicalize_snapshot_ts_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``snapshot_ts`` as tz-aware UTC datetime64.

    FAILS FAST on any naive (timezone-unaware) ``snapshot_ts`` value rather
    than silently assuming UTC (review 29-02 HIGH). This closes the
    object-dtype / assume-naive==UTC bypass in
    ``ParquetManager._normalize_parquet_datetime_columns``
    (storage.py:277-279 stringifies an object-typed ``snapshot_ts`` verbatim;
    storage.py:306/:313 assume a naive datetime is UTC). For the trajectory
    grain, a naive timestamp must be a hard error, never coerced.

    Coercing the column to ``datetime64[ns, UTC]`` here (instead of leaving it
    object-typed) also makes the parquet round-trip lossless, which is what the
    composite-key dedupe relies on for idempotency across re-runs.
    """
    if "snapshot_ts" not in df.columns:
        return df

    df = df.copy()
    col = df["snapshot_ts"]

    if pd.api.types.is_datetime64_any_dtype(col):
        if getattr(col.dt, "tz", None) is None:
            raise ValueError(
                "upsert_silver_composite: 'snapshot_ts' is timezone-naive. "
                "Trajectory timestamps must be tz-aware UTC; naive values are "
                "rejected and never silently coerced. Use "
                "utils.date_utils.ensure_utc_aware() to fix-forward."
            )
        df["snapshot_ts"] = col.dt.tz_convert("UTC")
        return df

    # Object dtype (strings / Python datetimes / pandas Timestamps / mixed):
    # reject any naive element BEFORE coercing the whole column to tz-aware UTC.
    def _require_aware(x):
        if pd.isna(x):
            return x
        ts = pd.Timestamp(x)
        if ts.tzinfo is None:
            raise ValueError(
                "upsert_silver_composite: 'snapshot_ts' contains a "
                f"timezone-naive value ({x!r}). Trajectory timestamps must be "
                "tz-aware UTC; naive values are rejected and never silently "
                "coerced to UTC."
            )
        return ts

    col.map(_require_aware)  # raises on the first naive value
    df["snapshot_ts"] = pd.to_datetime(col, utc=True)
    return df


def upsert_silver_composite(
    new_df: pd.DataFrame,
    table_name: str,
    key_columns: list[str] | None = None,
    base_path: Path | None = None,
) -> Path:
    """Upsert into a Silver table deduping on a COMPOSITE key (keep='last').

    Derived from :func:`upsert_silver`, but diverges in two ways for the
    ``odds_timeline`` trajectory table (D-11):

    1. ``snapshot_ts`` is canonicalized to tz-aware UTC and a naive/mixed
       timestamp FAILS FAST *before* the dedupe (review 29-02 HIGH). The
       existing ``_normalize_parquet_datetime_columns`` guard only rejects
       *datetime64*-typed naive columns; an object-typed ``snapshot_ts`` is
       otherwise stringified (storage.py:277-279) and the object-datetime
       normalization assumes naive==UTC (storage.py:306/:313). This function
       closes that bypass so a naive trajectory timestamp can never be
       silently stored as UTC.
    2. Dedup is on the COMPOSITE ``key_columns`` (default
       ``["game_id", "snapshot_ts"]``) with ``keep="last"`` -- so distinct
       ``(game_id, snapshot_ts)`` pairs coexist (never the ``game_id``
       latest-wins clobber of :func:`upsert_silver`), while a re-run of the
       same pairs is idempotent.

    :func:`upsert_silver` and the ``odds_snapshot`` write path are untouched
    (D-11). Both upserts now keep an existing DuckDB copy in step through
    :func:`_keep_duckdb_copy_in_step` (Plan 33-18, owner ruling R1).

    Args:
        new_df: Validated trajectory rows to upsert.
        table_name: e.g. "odds_timeline".
        key_columns: Composite dedup key (default ``["game_id", "snapshot_ts"]``).
        base_path: Base data directory (default from settings).

    Returns:
        Path to the Silver file.
    """
    if key_columns is None:
        key_columns = ["game_id", "snapshot_ts"]

    if base_path is None:
        settings = get_settings()
        base_path = Path(settings.config.data.root_path)

    silver_path = base_path / "silver" / f"{table_name}.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail fast on naive snapshot_ts BEFORE any read/concat/dedupe.
    new_df = _canonicalize_snapshot_ts_utc(new_df)

    if silver_path.exists():
        existing = pd.read_parquet(silver_path, engine="pyarrow")
        existing = _canonicalize_snapshot_ts_utc(existing)
        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=key_columns, keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = new_df.drop_duplicates(subset=key_columns, keep="last").reset_index(
            drop=True
        )

    # Normalize datetime columns before writing (snapshot_ts is now a tz-aware
    # datetime64 column, so it round-trips as a real timestamp -- not a string).
    pm = ParquetManager(str(base_path))
    combined_normalized = pm._normalize_parquet_datetime_columns(combined)
    table = pa.Table.from_pandas(combined_normalized)
    _atomic_write_parquet(table, silver_path)
    _keep_duckdb_copy_in_step(table_name, silver_path, base_path)

    logger.info(
        "Upserted Silver table (composite key)",
        table=table_name,
        path=str(silver_path),
        key_columns=key_columns,
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
    """Optimize DuckDB database.

    ANALYZE and VACUUM both rewrite the database, so the write handle is acquired
    by name before either runs.
    """
    db = get_db_connection()

    try:
        db.connect(write=True)

        # Analyze tables for query optimization
        db.execute("ANALYZE", write=True)

        # Vacuum if using file-based database
        if db.db_path:
            db.execute("VACUUM", write=True)

        logger.info("Database optimization completed")

    except (duckdb.Error, DataIngestionError, OSError) as e:
        logger.warning(
            "Database optimization failed",
            error=str(e),
            exception_type=type(e).__name__,
        )


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
            except (duckdb.Error, DataIngestionError, ValueError, KeyError) as e:
                logger.warning(
                    f"Failed to get stats for table {table_name}: {e}",
                    exception_type=type(e).__name__,
                )
                table_stats[table_name] = {"row_count": "unknown", "columns": "unknown"}

        return {
            "num_tables": len(tables_df),
            "table_stats": table_stats,
            "database_path": db.db_path or "in-memory",
        }

    except (duckdb.Error, DataIngestionError, OSError) as e:
        logger.error(
            "Failed to get database stats",
            error=str(e),
            exception_type=type(e).__name__,
        )
        return {"error": str(e)}
