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

import re
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from api import presentation
from api.cache_schema import PUSH_TIME_KEY, STALE_AFTER
from api.services import clear_cache

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INPUT_CSS = _REPO_ROOT / "web" / "static" / "input.css"
_ERROR_STATE = _REPO_ROOT / "web" / "templates" / "components" / "_error_state.html"
#: The honesty-note grey (input.css .honesty-note), reserved for permanent statements.
_HONESTY_GREY = "#C4CAD8"

#: The pinned "now" every bar test renders against (a UTC instant after the 2026-11-01 DST change).
PINNED_NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)

ClientFactory = Callable[[datetime | None], TestClient]


@pytest.fixture()
def client_with_push(
    test_db: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ClientFactory]:
    """A factory: stamp *push_time* (or none) into a fixture cache and serve it.

    Each call closes the previous read-only handle before rewriting the cache, because DuckDB
    refuses two configurations of one file in one process.
    """
    from api.dependencies import get_db
    from api.main import app

    monkeypatch.setattr(presentation, "NOW", lambda: PINNED_NOW)
    readers: list[duckdb.DuckDBPyConnection] = []

    def make(push_time: datetime | None) -> TestClient:
        while readers:
            readers.pop().close()
        with duckdb.connect(str(test_db)) as conn:
            conn.execute("DELETE FROM cache_meta WHERE key = ?", [PUSH_TIME_KEY])
            if push_time is not None:
                conn.execute(
                    "INSERT INTO cache_meta VALUES (?, ?, ?)",
                    [PUSH_TIME_KEY, push_time.isoformat(), push_time],
                )
        reader = duckdb.connect(str(test_db), read_only=True)
        readers.append(reader)
        # A real swap reconnects and clears the memoized reads; a re-stamp here must do the same.
        clear_cache()
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


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/bets",
        "/season",
        "/track-record",
        "/how-it-works",
        "/games/2024_W01_BUF@KC",
    ],
)
def test_every_html_page_shows_the_bar_once_before_the_nav_when_stale(
    client_with_push: ClientFactory, path: str
) -> None:
    stale = client_with_push(PINNED_NOW - STALE_AFTER - timedelta(seconds=1)).get(path)

    assert stale.status_code == 200
    assert stale.text.count("stale-bar") == 1
    assert stale.text.index('class="stale-bar"') < stale.text.index("<nav")

    fresh = client_with_push(PINNED_NOW - timedelta(hours=1)).get(path)

    assert fresh.status_code == 200
    assert "stale-bar" not in fresh.text


@pytest.mark.parametrize(
    "path",
    [
        "/fragments/games?week=1&season=2024",
        "/fragments/performance",
        "/fragments/betting",
        "/fragments/season?season=2024",
    ],
)
def test_fragments_never_carry_the_bar(
    client_with_push: ClientFactory, path: str
) -> None:
    response = client_with_push(None).get(path)

    assert response.status_code == 200
    assert "stale-bar" not in response.text


def _token(css: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}:\s*(#[0-9A-Fa-f]{{6}})\s*;", css)
    assert match is not None, f"{name} is not a #RRGGBB token in web/static/input.css"
    return match.group(1).upper()


def test_the_bar_colours_meet_wcag_aa() -> None:
    css = _INPUT_CSS.read_text(encoding="utf-8")
    stale = _token(css, "--color-stale")
    ink = _token(css, "--color-stale-ink")

    assert presentation.contrast_ratio(stale, ink) >= 4.5
    # D-07: none of the brand yellow, the honesty grey or the failure state's colours.
    reserved = {_token(css, "--color-accent"), _HONESTY_GREY}
    reserved |= {
        colour.upper()
        for colour in re.findall(
            r"#[0-9A-Fa-f]{6}", _ERROR_STATE.read_text(encoding="utf-8")
        )
    }
    assert stale not in reserved
    assert ink not in reserved
