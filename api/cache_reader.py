"""The ONE way the request path opens the web cache, and how it reads the cache's data format.

WHY ONE OPENER (Phase 36, D-22; research PITFALLS E3, Pitfall 4). The hosted copy runs on a
filesystem that is read-only everywhere except the temp directory. Left to its defaults, DuckDB
touches HOME (``~/.duckdb``) and may download and install extensions over the network. And two
connections to one file opened with DIFFERENT configs raise ``ConnectionException`` -- so every
request-path open of the cache must use exactly the same config, which is why there is one opener.

``tempfile.gettempdir()`` is ``/tmp`` on Linux (Vercel) and ``%TEMP%`` on this laptop, so one code
path serves both copies; behaviour differs only through that environment fact.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from api.cache_schema import DATA_FORMAT_VERSION_KEY, DataFormatState, data_format_state


def cache_connect_config() -> dict[str, Any]:
    """The one DuckDB config every request-path connection to the cache uses."""
    scratch = tempfile.gettempdir()
    return {
        "home_directory": scratch,
        "temp_directory": scratch,
        "autoload_known_extensions": False,
        "autoinstall_known_extensions": False,
    }


def open_cache_readonly(
    path: Path,
    *,
    connect: Callable[..., duckdb.DuckDBPyConnection] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open the cache at *path* read-only with the one config.

    Args:
        path: The cache file.
        connect: The connect function to call. A call site passes ITS OWN module's
            ``duckdb.connect``, so a test that replaces that module's ``duckdb`` name still reaches
            the connect call. When None, the ``duckdb.connect`` module attribute is looked up at
            call time (never bound at import), so a test that replaces it is honoured too.

    Returns:
        The open read-only connection.
    """
    connect_fn = duckdb.connect if connect is None else connect
    return connect_fn(str(path), read_only=True, config=cache_connect_config())


def read_data_format(conn: duckdb.DuckDBPyConnection) -> DataFormatState:
    """The cache's stamped data-format version, judged against the one the code expects (D-20).

    A cache with no ``cache_meta`` table, or no version row, reads as a mismatch with no raw
    value -- never as a match and never as a crash.
    """
    try:
        row = conn.execute(
            "SELECT value FROM cache_meta WHERE key = ?", [DATA_FORMAT_VERSION_KEY]
        ).fetchone()
    except duckdb.Error:
        return data_format_state(None)
    return data_format_state(None if row is None else row[0])
