"""The ONE read-only opener for the web cache (Phase 36, D-22; SPEC R4; research PITFALLS E3).

On Vercel the filesystem is read-only except the temp directory. DuckDB would otherwise create
``~/.duckdb`` under HOME and may auto-install extensions over the network, and a second connection
to the same file with a different config raises. ``api.cache_reader.open_cache_readonly`` opens
with home and temp pointed at ``tempfile.gettempdir()`` and extension autoload/autoinstall off.

What this file proves, on this machine, with HOME pointed somewhere that cannot be created:

* the cache opens and answers a query, and the four settings read back as configured;
* a read-only open leaves the file's bytes and its directory untouched, and a second open of the
  same file in the same process works;
* a caller's own ``connect`` (the module attribute a test may have replaced) receives the path
  text, ``read_only=True`` and the one config.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import pytest

from api.cache import CACHE_META_SCHEMA
from api.cache_reader import cache_connect_config, open_cache_readonly


@pytest.fixture
def small_cache(tmp_path: Path) -> Path:
    """A one-table DuckDB file carrying a cache_meta row, built under tmp_path only."""
    db_path = tmp_path / "cache" / "web_cache.duckdb"
    db_path.parent.mkdir()
    writer = duckdb.connect(str(db_path))
    try:
        writer.execute(CACHE_META_SCHEMA)
        writer.execute(
            "INSERT INTO cache_meta VALUES ('last_updated', '2026-10-18T21:00:00+00:00', NULL)"
        )
    finally:
        writer.close()
    return db_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_cache_opens_with_a_non_writable_home(
    small_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOME under a regular FILE cannot be created; the opener never needs it (T-36-02, T-36-03)."""
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("a file, so nothing can be created beneath it", encoding="utf-8")
    unwritable_home = blocker / "home"
    monkeypatch.setenv("HOME", str(unwritable_home))
    monkeypatch.setenv("USERPROFILE", str(unwritable_home))

    conn = open_cache_readonly(small_cache)
    try:
        row = conn.execute(
            "SELECT value FROM cache_meta WHERE key = 'last_updated'"
        ).fetchone()
        settings = {
            name: conn.execute(f"SELECT current_setting('{name}')").fetchone()
            for name in (
                "home_directory",
                "temp_directory",
                "autoload_known_extensions",
                "autoinstall_known_extensions",
            )
        }
    finally:
        conn.close()

    assert row == ("2026-10-18T21:00:00+00:00",)
    scratch = tempfile.gettempdir()
    assert settings["home_directory"] == (scratch,)
    assert settings["temp_directory"] == (scratch,)
    assert settings["autoload_known_extensions"] == (False,)
    assert settings["autoinstall_known_extensions"] == (False,)
    assert not unwritable_home.exists()


def test_a_read_only_open_changes_nothing_and_a_second_open_works(
    small_cache: Path,
) -> None:
    """Same bytes, same directory listing; two opener connections to one file coexist."""
    digest_before = _sha256(small_cache)
    listing_before = sorted(p.name for p in small_cache.parent.iterdir())

    first = open_cache_readonly(small_cache)
    try:
        second = open_cache_readonly(small_cache)
        try:
            assert second.execute("SELECT COUNT(*) FROM cache_meta").fetchone() == (1,)
        finally:
            second.close()
        assert first.execute("SELECT COUNT(*) FROM cache_meta").fetchone() == (1,)
    finally:
        first.close()

    assert _sha256(small_cache) == digest_before
    assert sorted(p.name for p in small_cache.parent.iterdir()) == listing_before


def test_an_injected_connect_receives_the_one_config(small_cache: Path) -> None:
    """A caller's own connect gets the path text, read_only=True and the one config, once."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def recorder(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return sentinel

    returned = open_cache_readonly(small_cache, connect=recorder)

    assert returned is sentinel
    assert calls == [
        ((str(small_cache),), {"read_only": True, "config": cache_connect_config()})
    ]
