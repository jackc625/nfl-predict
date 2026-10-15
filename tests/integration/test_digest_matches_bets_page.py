"""The weekly email and /bets agree figure by figure from one cache (Phase 35 NOTIF-05, SPEC R6).

One ``tmp_path`` web cache holds week 6's live bets of all three targets, one SHADOW-arm live row,
a suppressed candidate, week 5's stored actual-vs-expected row and the forward tracker blocks.
``/bets`` is rendered through the FastAPI TestClient and the digest through
``notify.digest.collect_digest_inputs`` + ``render_digest`` over the same file, and every figure is
extracted from each surface's own ``data-*`` hooks, in document order:

* per bet row: the pick, the line, the EV %, the stake and the EV band (on ``/bets`` the band is the
  word the ``_ev_band_badge.html`` span starts with, found by its ``title="EV band <band>:``);
* the last-week block's record, actual, expected, ungraded and caveat strings;
* the forward tracker's ``(label, value)`` pairs, verdict record first.

The lists must be EQUAL, and the shadow-arm row's game must be in neither ranked list nor count.

A second test proves the email's read-only handle is closed: the cache file can be rebuilt in place
by ``api.cache.populate_cache`` right after the digest read (an open handle blocks the rename on
Windows).

Nothing under ``data/`` is read and no provider is called: the cache is a ``tmp_path`` file and the
digest is rendered, never sent.

Run this module:  uv run python -m pytest tests/integration/test_digest_matches_bets_page.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from fastapi.testclient import TestClient

from api.cache import (
    bet_list_populated_at_key,
    materialize_available_bet_weeks,
    populate_cache,
)
from api.services import DataService, clear_cache
from notify import digest
from tests.unit.test_notify_digest import (
    WEEK6,
    build_cache,
    saturday_report,
    schedule_frame,
    week6_bets,
)
from tests.unit.test_notify_messages import live_bet

SHADOW_GAME = "2026_06_KC@BUF"
SHADOW_MATCHUP = "KC @ BUF"

# The five last-week strings and the two tracker hooks, read the same way on both surfaces.
_LAST_WEEK = re.compile(
    r"data-last-week-(record|actual|expected|ungraded|caveat)\b[^>]*>([^<]*)<"
)
_FIGURE = re.compile(r"data-figure-(label|value)\b[^>]*>([^<]*)<")


def _shadow_row() -> dict[str, Any]:
    """A shadow-arm row with ``status`` live and the highest EV: ranked first if it leaked."""
    return live_bet(
        SHADOW_GAME,
        arm="shadow",
        target="wp",
        bet_side="home",
        line=None,
        per_bet_ev=0.09,
        stake_units=2.0,
        ev_tier="high",
    )


def _fixture_cache(path: Path, *, stamped_minutes: int = 12) -> Path:
    """Week 6's bets (three live, one suppressed, one shadow), week 5's row, the tracker blocks.

    The bet list is stamped populated after the Saturday run's start, so the email reads it as
    current, and the week is navigable on /bets through the schedule-derived week table.
    """
    report = saturday_report()
    build_cache(
        path,
        bets=[*week6_bets(), _shadow_row()],
        stamped_at=report.start + timedelta(minutes=stamped_minutes),
    )
    conn = duckdb.connect(str(path))
    try:
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": game_id, "season": 2026, "week": week}
                    for game_id, week, _kickoff in WEEK6
                ]
            ),
        )
    finally:
        conn.close()
    return path


@contextmanager
def _client(path: Path) -> Iterator[TestClient]:
    """A TestClient serving the app over *path*, opened read-only like production."""
    from api.dependencies import get_db
    from api.main import app

    conn = duckdb.connect(str(path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()
        conn.close()


def _page_bet_rows(body: str) -> list[tuple[str, ...]]:
    """(pick, line, EV %, stake, band) per ranked slip on /bets, in rank order."""
    ranked = body.split('aria-label="Live bets', 1)[1].split("</ol>", 1)[0]
    slips = ranked.split('<li class="bet-slip')[1:]
    rows: list[tuple[str, ...]] = []
    for slip in slips:
        rows.append(
            (
                re.search(r"data-slip-pick>([^<]*)<", slip).group(1),  # type: ignore[union-attr]
                re.search(r"data-slip-line>([^<]*)<", slip).group(1),  # type: ignore[union-attr]
                re.search(r"data-slip-ev>([^<]*)<", slip).group(1),  # type: ignore[union-attr]
                re.search(r"data-slip-stake>([^<]*)<", slip).group(1),  # type: ignore[union-attr]
                re.search(r'title="EV band (\w+):', slip).group(1),  # type: ignore[union-attr]
            )
        )
    return rows


def _email_bet_rows(html: str) -> list[tuple[str, ...]]:
    """(pick, line, EV %, stake, band) per bet row of the email, in rank order."""
    table = html.split("data-digest-bets>", 1)[1].split("</table>", 1)[0]
    rows: list[tuple[str, ...]] = []
    for row in table.split("<tr data-digest-row>")[1:]:
        rows.append(
            tuple(
                re.search(rf"data-digest-{hook}>([^<]*)<", row).group(1)  # type: ignore[union-attr]
                for hook in ("pick", "line", "ev", "stake", "band")
            )
        )
    return rows


def _last_week(markup: str) -> list[tuple[str, str]]:
    return [(hook, value.strip()) for hook, value in _LAST_WEEK.findall(markup)]


def _figure_pairs(markup: str) -> list[tuple[str, str]]:
    """The (label, value) pairs, in document order."""
    found = _FIGURE.findall(markup)
    labels = [value.strip() for hook, value in found if hook == "label"]
    values = [value.strip() for hook, value in found if hook == "value"]
    assert len(labels) == len(values)
    return list(zip(labels, values, strict=True))


def _render_email(path: Path) -> tuple[str, digest.DigestInputs]:
    """The digest's HTML part and its inputs, built from *path* exactly as the Saturday run does."""
    report = saturday_report()
    with digest.open_cache_service(path) as service:
        inputs = digest.collect_digest_inputs(
            report,
            schedule=schedule_frame(WEEK6),
            service=service,
            notifications=[],
            failed_titles=[],
            now=report.start + timedelta(minutes=20),
        )
    return digest.render_digest(inputs).html, inputs


def test_the_email_and_bets_page_agree_figure_by_figure(tmp_path: Path) -> None:
    path = _fixture_cache(tmp_path / "web_cache.duckdb")
    clear_cache()
    with _client(path) as client:
        response = client.get("/bets?season=2026&week=6")
    assert response.status_code == 200
    page = response.text
    clear_cache()
    html, inputs = _render_email(path)

    page_rows = _page_bet_rows(page)
    email_rows = _email_bet_rows(html)
    assert len(page_rows) == 3, page_rows
    assert email_rows == page_rows
    # All three targets are compared: an ats spread line, an ou total and a wp row with no line.
    assert [row[1] for row in page_rows] == ["--", "47.5", "+3.5"]

    page_last_week = _last_week(page)
    assert [hook for hook, _value in page_last_week] == [
        "record",
        "actual",
        "expected",
        "ungraded",
        "caveat",
    ]
    assert _last_week(html) == page_last_week

    # /bets renders the replay sections first; the email carries the forward record only.
    forward_page = page.split('data-tracker-block="forward:forward_realized"', 1)[1]
    page_figures = _figure_pairs(forward_page)
    assert len(page_figures) == 12  # two forward classes, six figures each
    assert _figure_pairs(html) == page_figures

    # The shadow-arm row is ranked on neither surface and counted on neither.
    ranked = page.split('aria-label="Live bets', 1)[1].split("</ol>", 1)[0]
    assert SHADOW_MATCHUP not in ranked
    assert "ranked by expected value &middot; 3 bets" in page
    assert (
        SHADOW_MATCHUP
        not in html.split("data-digest-bets>", 1)[1].split("</table>", 1)[0]
    )
    assert [row["game_id"] for row in inputs.bets] == [
        "2026_06_DAL@PHI",
        "2026_06_CIN@CLE",
        "2026_06_NYJ@NE",
    ]
    assert digest.digest_subject(inputs) == (
        "NFL Predict - 2026 week 6 - 3 bets - run clean"
    )


def test_rendering_the_email_leaves_the_cache_rebuildable(tmp_path: Path) -> None:
    path = _fixture_cache(tmp_path / "web_cache.duckdb")
    _render_email(path)

    # The read-only handle is closed: the population's unlink-and-rename succeeds in place.
    populate_cache(
        db_path=path,
        artifacts_dir=tmp_path / "artifacts",
        outputs_dir=tmp_path / "outputs",
        gold_dir=tmp_path / "gold",
        silver_dir=tmp_path / "silver",
    )

    probe = duckdb.connect(str(path), read_only=True)
    try:
        rebuilt = probe.execute(
            "SELECT count(*) FROM cache_meta WHERE key = ?",
            [bet_list_populated_at_key(2026, 6)],
        ).fetchone()
    finally:
        probe.close()
    assert rebuilt == (0,), (
        "the rebuilt cache still carries the fixture's week-6 marker"
    )
    clear_cache()
    probe = duckdb.connect(str(path), read_only=True)
    try:
        assert DataService(probe).get_bet_list(2026, 6) == []
    finally:
        probe.close()
