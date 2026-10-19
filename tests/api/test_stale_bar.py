"""The out-of-date bar (Phase 36, SPEC R8, D-06, D-07).

Every check here reads a RENDERED page, never a template grep: the bar is only real if the stored
push stamp, the one 36-hour rule (``api.cache_schema.push_is_stale``), the Eastern label, the Jinja
global and ``base.html`` all line up. The clock is ``api.presentation.NOW``, pinned per test, so
the exact boundary (36 h is fresh, 36 h + 1 s is stale) is measured rather than raced.

Each cache is the conftest ``test_db`` with its ``push_time`` row rewritten (or removed) under
tmp_path; production stores are never touched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from api import presentation
from api.cache_schema import PUSH_TIME_KEY, STALE_AFTER

#: The pinned "now" every bar test renders against (a UTC instant after the 2026-11-01 DST change).
PINNED_NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

ClientFactory = Callable[[datetime | None], TestClient]


@pytest.fixture()
def client_with_push(
    test_db: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ClientFactory]:
    """A factory: stamp *push_time* (or none) into a fixture cache and serve it.

    Call it once per test: the cache is rewritten read-write, then opened read-only for the app,
    and DuckDB refuses two configurations of one file in one process.
    """
    from api.dependencies import get_db
    from api.main import app

    monkeypatch.setattr(presentation, "NOW", lambda: PINNED_NOW)
    readers: list[duckdb.DuckDBPyConnection] = []

    def make(push_time: datetime | None) -> TestClient:
        with duckdb.connect(str(test_db)) as conn:
            conn.execute("DELETE FROM cache_meta WHERE key = ?", [PUSH_TIME_KEY])
            if push_time is not None:
                conn.execute(
                    "INSERT INTO cache_meta VALUES (?, ?, ?)",
                    [PUSH_TIME_KEY, push_time.isoformat(), push_time],
                )
        reader = duckdb.connect(str(test_db), read_only=True)
        readers.append(reader)
        app.state.db_lock = threading.RLock()
        app.dependency_overrides[get_db] = lambda: reader
        return TestClient(app, raise_server_exceptions=False)

    try:
        yield make
    finally:
        app.dependency_overrides.clear()
        for reader in readers:
            reader.close()


@pytest.mark.parametrize(
    ("age", "shows_bar"),
    [
        (STALE_AFTER + timedelta(seconds=1), True),
        (STALE_AFTER, False),
    ],
    ids=["36h_plus_1s_is_stale", "exactly_36h_is_fresh"],
)
def test_the_bar_follows_the_36_hour_boundary(
    client_with_push: ClientFactory, age: timedelta, shows_bar: bool
) -> None:
    response = client_with_push(PINNED_NOW - age).get("/")

    assert response.status_code == 200
    html = response.text
    if shows_bar:
        assert 'class="stale-bar"' in html
        # 2026-11-01T23:59:59Z is 6:59 PM on Sunday in Eastern Standard Time.
        assert (
            "Out of date: last update Sun, Nov 1, 6:59 PM ET. "
            "The laptop has not reported since." in html
        )
    else:
        assert "stale-bar" not in html
        assert "Out of date" not in html


def test_no_push_record_shows_the_bar(client_with_push: ClientFactory) -> None:
    response = client_with_push(None).get("/")

    assert response.status_code == 200
    assert 'class="stale-bar"' in response.text
    assert "Out of date: no push recorded." in response.text


@pytest.mark.parametrize(
    ("stamp", "label"),
    [
        ("2026-10-31T21:12:00+00:00", "Sat, Oct 31, 5:12 PM ET"),
        ("2026-11-02T22:12:00+00:00", "Mon, Nov 2, 5:12 PM ET"),
        ("2026-11-02T22:12:00", "Mon, Nov 2, 5:12 PM ET"),
    ],
    ids=["before_dst_change", "after_dst_change", "naive_read_as_utc"],
)
def test_the_weekday_label_is_right_either_side_of_the_dst_change(
    stamp: str, label: str
) -> None:
    assert presentation.et_weekday_timestamp_label(stamp) == label
    # The existing label keeps its exact output (no weekday, with the year).
    assert presentation.et_timestamp_label("2025-09-06T22:00:00+00:00") == (
        "Sep 6, 2025, 6:00 PM ET"
    )
    assert presentation.et_timestamp_label(None) == ""
    assert presentation.et_timestamp_label("not a time") == "not a time"
