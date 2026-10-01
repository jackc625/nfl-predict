"""Route-level proof of the dated old-rule label on the website (Phase 33.2, R16 / D33.2-07).

``tests/unit/test_page_labels.py`` renders each TEMPLATE with a context the test builds. That proves
the templates, not the pages: a template test fed a synthetic season passes whether or not
production ever supplies one. Before Plan 33.2-07, three of the eight handlers -- ``/backtest``,
``/insights`` and ``/betting`` -- built contexts with no season at all, so a template test would
have passed green over three pages that rendered unlabelled in production.

So this module drives the REAL routes through ``TestClient`` and counts the label in the response
body. No case injects a context: every context is the one the handler assembles from the cache.
It runs over two seeded corpora:

  - ``pre_2026``: the shared ``test_db`` fixture (2021-2024, the realistic state), plus a replay
    bet week and replay tracker block, with the population-time season spans stamped by the same
    ``api.cache.stamp_old_rule_season_ranges`` that ``populate_cache`` calls. Every page must
    render exactly its declared ``EXPECTED_PREFIX_BLOCKS`` count.
  - ``only_2026``: the same fixture cut down to one season and moved to 2026, with a forward bet
    week and a forward tracker block. Every page must render ZERO labels.

``status_code == 200`` is asserted on every request as the non-vacuity control: a 500 returns no
label and would otherwise satisfy every zero-count assertion.

This is a PERMANENT committed test, not a throwaway ``scripts/check_*.py``.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.cache import (
    BET_LIST_COLUMNS,
    classify_row_provenance,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_tracker_blocks,
    stamp_old_rule_season_ranges,
)
from tests.unit.test_page_labels import EXPECTED_PREFIX_BLOCKS, LABEL_MARKER

PRE_2026 = "pre_2026"
ONLY_2026 = "only_2026"

# The season the pre-2026 corpus's bet week and game detail use; the 2026 corpus moves it forward.
_FIXTURE_SEASON = 2024
_NEW_RULE_SEASON = 2026
_BET_WEEK = 1
_GAME_ID = "2024_W01_BUF@KC"

# Each full-page route, the page template it renders, and a string that proves the page itself
# rendered (the non-vacuity control beside the status code).
PAGE_ROUTES: dict[str, tuple[str, str]] = {
    "/": ("this_week.html", "This Week's Predictions"),
    "/bets": ("bets.html", "Weekly Bet List"),
    "/how-it-works": ("how_it_works.html", "What the models rely on"),
    "/season": ("season.html", "Season Tracking"),
    "/track-record": ("track_record.html", "Season by season"),
    f"/games/{_GAME_ID}": ("game_detail.html", "BUF @ KC"),
}

# HTMX fragment requests and the labelled blocks each swaps in. The retired /performance and
# /betting URLs answer an HX request with their Track Record block instead of a redirect; the
# /fragments/* routes are the dedicated swap paths.
FRAGMENT_REQUESTS: dict[str, tuple[dict[str, str], int]] = {
    "/performance": ({"HX-Request": "true"}, 1),
    "/betting": ({"HX-Request": "true"}, 1),
    "/fragments/performance": ({"HX-Request": "true"}, 1),
    "/fragments/betting": ({"HX-Request": "true"}, 1),
    "/fragments/games": ({"HX-Request": "true"}, 1),
}


# ---------------------------------------------------------------------------
# Corpus seeding
# ---------------------------------------------------------------------------


def _bet_row(season: int) -> dict[str, Any]:
    """One live bet-list row for the seeded week, labelled the way population labels it."""
    run_mode = "forward" if season >= _NEW_RULE_SEASON else "replay"
    provenance, validation_type = classify_row_provenance(season, run_mode)
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": f"{season}_W01_BUF@KC",
            "season": season,
            "week": _BET_WEEK,
            "target": "ou",
            "bet_side": "under",
            "model_value": 45.0,
            "market_value": 47.5,
            "line": 47.5,
            "slipped_line": 48.0,
            "calibrated_p_side": 0.56,
            "per_bet_ev": 0.07,
            "stake_units": 1.25,
            "ev_tier": "high",
            "status": "live",
            "snapshot_ts": f"{season}-09-09T22:00:00+00:00",
            "freeze_ts": f"{season}-09-09T22:00:00+00:00",
            "selected_odds": -110.0,
            "flat_stake": 1.0,
            "provenance": provenance,
            "validation_type": validation_type,
            "grading_status": "pending",
        }
    )
    return row


def _tracker_block(season: int) -> dict[str, Any]:
    """The graded tracker block a corpus of *season* carries: replay before 2026, forward after."""
    provenance, validation_type = classify_row_provenance(
        season, "forward" if season >= _NEW_RULE_SEASON else "replay"
    )
    return {
        "provenance": provenance,
        "validation_type": validation_type,
        "bets_graded": 12,
        "wins": 7,
        "losses": 5,
        "pushes": 0,
        "hit_rate": 7 / 12,
        "flat_return_units": 0.41,
    }


def _move_corpus_to_2026(conn: duckdb.DuckDBPyConnection) -> None:
    """Cut the shared fixture down to its 2024 rows and relabel them as 2026.

    Every table a page reads keeps real rows -- only the season moves -- so the 2026 corpus renders
    the same blocks with the same numbers, and a zero label count is a statement about the scope
    rather than about an empty page.
    """
    for table in ("predictions", "backtest_predictions", "betting_bets"):
        conn.execute(f"DELETE FROM {table} WHERE season <> ?", [_FIXTURE_SEASON])
        conn.execute(f"UPDATE {table} SET season = ?", [_NEW_RULE_SEASON])
    conn.execute(
        "DELETE FROM backtest_metrics WHERE season NOT IN (0, ?)", [_FIXTURE_SEASON]
    )
    conn.execute(
        "UPDATE backtest_metrics SET season = ? WHERE season = ?",
        [_NEW_RULE_SEASON, _FIXTURE_SEASON],
    )
    # The per-season chart blobs are keyed season_<chart>_<year>: keep the 2024 set, renamed.
    conn.execute(
        "DELETE FROM chart_cache WHERE starts_with(chart_id, 'season_') "
        "AND NOT ends_with(chart_id, ?)",
        [f"_{_FIXTURE_SEASON}"],
    )
    conn.execute(
        "UPDATE chart_cache SET chart_id = replace(chart_id, ?, ?) "
        "WHERE starts_with(chart_id, 'season_')",
        [f"_{_FIXTURE_SEASON}", f"_{_NEW_RULE_SEASON}"],
    )


def _seed(db_path: Path, corpus: str) -> None:
    """Seed *corpus* into the fixture database, then stamp its season spans as population does."""
    season = _NEW_RULE_SEASON if corpus == ONLY_2026 else _FIXTURE_SEASON
    conn = duckdb.connect(str(db_path))
    try:
        if corpus == ONLY_2026:
            _move_corpus_to_2026(conn)
        materialize_bet_list(conn, pd.DataFrame([_bet_row(season)]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": f"{season}_W01_BUF@KC",
                        "season": season,
                        "week": _BET_WEEK,
                    }
                ]
            ),
        )
        materialize_bet_tracker_blocks(conn, pd.DataFrame([_tracker_block(season)]))
        stamp_old_rule_season_ranges(conn, datetime.now(tz=UTC))
    finally:
        conn.close()


@pytest.fixture(params=[PRE_2026, ONLY_2026])
def corpus(request: pytest.FixtureRequest, test_db: Path) -> str:
    """Seed the shared ``test_db`` as one of the two corpora and name it."""
    _seed(test_db, request.param)
    return request.param


@pytest.fixture()
def client(request: pytest.FixtureRequest, corpus: str) -> TestClient:
    """The shared ``test_client``, opened AFTER the corpus is seeded so it serves the seeded data."""
    test_client: TestClient = request.getfixturevalue("test_client")
    return test_client


@pytest.fixture()
def pre_2026_client(request: pytest.FixtureRequest, test_db: Path) -> TestClient:
    """The shared ``test_client`` over the pre-2026 corpus only, for the named regressions."""
    _seed(test_db, PRE_2026)
    test_client: TestClient = request.getfixturevalue("test_client")
    return test_client


def _labels(body: str) -> int:
    return body.count(LABEL_MARKER)


# ---------------------------------------------------------------------------
# Every page and every fragment, both corpora
# ---------------------------------------------------------------------------


class TestEveryRouteOverBothCorpora:
    """Each route's label count is decided by the corpus the handler reads, nothing else."""

    @pytest.mark.parametrize("url", sorted(PAGE_ROUTES))
    def test_page_label_count(self, client: TestClient, corpus: str, url: str) -> None:
        page, marker = PAGE_ROUTES[url]
        response = client.get(url)
        assert response.status_code == 200, f"{url} returned {response.status_code}"
        assert marker in response.text, f"{url} did not render its page"
        expected = EXPECTED_PREFIX_BLOCKS[page] if corpus == PRE_2026 else 0
        assert _labels(response.text) == expected, (
            f"{url} on the {corpus} corpus rendered {_labels(response.text)} labels, "
            f"expected {expected}"
        )

    @pytest.mark.parametrize("url", sorted(FRAGMENT_REQUESTS))
    def test_fragment_label_count(
        self, client: TestClient, corpus: str, url: str
    ) -> None:
        headers, pre_count = FRAGMENT_REQUESTS[url]
        response = client.get(url, headers=headers)
        assert response.status_code == 200, f"{url} returned {response.status_code}"
        assert response.text.strip(), f"{url} returned an empty fragment"
        assert "<html" not in response.text, (
            f"{url} returned a full page, not a fragment"
        )
        expected = pre_count if corpus == PRE_2026 else 0
        assert _labels(response.text) == expected, (
            f"{url} fragment on the {corpus} corpus rendered {_labels(response.text)} labels, "
            f"expected {expected}"
        )

    def test_every_page_template_has_a_route_here(self) -> None:
        """A page added later needs a route case, or it is certified only as a template."""
        covered = sorted(page for page, _ in PAGE_ROUTES.values())
        assert covered == sorted(EXPECTED_PREFIX_BLOCKS)


# ---------------------------------------------------------------------------
# The route-shape regressions the review named
# ---------------------------------------------------------------------------


class TestTheSeasonlessPagesLabel:
    """The merged pages pass no season, yet label the pre-fix corpus each block draws from."""

    @pytest.mark.parametrize(
        ("url", "expected"), [("/how-it-works", 1), ("/track-record", 4)]
    )
    def test_a_seasonless_page_labels_its_pre_fix_corpus(
        self, pre_2026_client: TestClient, url: str, expected: int
    ) -> None:
        response = pre_2026_client.get(url)
        assert response.status_code == 200
        assert _labels(response.text) == expected

    def test_the_betting_fragment_carries_the_scope_its_section_does(
        self, pre_2026_client: TestClient
    ) -> None:
        """_build_betting_context feeds both the page's betting section and the fragment."""
        page = pre_2026_client.get("/track-record")
        fragment = pre_2026_client.get("/fragments/betting")
        assert page.status_code == 200
        assert fragment.status_code == 200
        betting_section = page.text.split('id="betting-content"', 1)[1]
        assert _labels(betting_section) == _labels(fragment.text) == 1


class TestTheSummaryIgnoresTheSelectedSeason:
    """The all-history summary aggregates every season, so selecting 2026 does not unlabel it."""

    def test_selecting_2026_still_labels_the_all_history_summary(
        self, pre_2026_client: TestClient
    ) -> None:
        response = pre_2026_client.get("/track-record?season=2026")
        assert response.status_code == 200
        body = response.text
        # The season-scoped table has no 2026 rows, so it shows its empty state and no label; the
        # summary, the model-vs-market section and the betting simulation keep theirs.
        assert _labels(body) == 3
        assert "No backtest data" in body
        assert body.index(LABEL_MARKER) < body.index("Total Games"), (
            "the first label is not the one above the all-history summary"
        )

    def test_a_past_season_labels_every_block(
        self, pre_2026_client: TestClient
    ) -> None:
        response = pre_2026_client.get("/track-record?season=2023")
        assert response.status_code == 200
        assert _labels(response.text) == 4
