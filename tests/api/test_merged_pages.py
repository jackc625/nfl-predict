"""Track Record, How It Works and the retired-page redirects (redesign Task 15).

The four retired pages -- /performance, /backtest, /insights, /betting -- were merged into two:
Track Record (how the model has done) and How It Works (why it predicts what it does). These
tests pin where every chart landed and that each appears once, that every retired URL still works
(a 301 carrying its query string to the right section, or the block itself for an HTMX request),
and that both new pages keep the read-only cache contract. Their Cache-Control (pages and 301s) is
pinned once, in tests/api/test_cache_headers.py, and the nav in tests/api/test_pages.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.charts import INSIGHTS_CHART_IDS

PAGES_DIR = Path(__file__).resolve().parents[2] / "web" / "templates" / "pages"

_VS_MARKET = tuple(
    c for c in INSIGHTS_CHART_IDS if c.startswith("insights_model_vs_market_")
)
_HOW_IT_WORKS_INSIGHTS = tuple(c for c in INSIGHTS_CHART_IDS if c not in _VS_MARKET)


class TestTrackRecord:
    def test_renders_every_section_with_its_anchor(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get("/track-record")
        assert response.status_code == 200
        html = response.text
        for anchor in ("summary", "seasons", "vs-market", "clv", "betting-sim"):
            assert f'id="{anchor}"' in html, f"missing section anchor #{anchor}"
        # The page's own <h1>: "Track Record" alone would also be found in the nav and <title>.
        assert re.search(r"<h1[^>]*>Track Record</h1>", html), (
            "the page heading is missing"
        )
        for heading in (
            "All-time summary",
            "Total Games",
            "Season by season",
            "Model vs Market",
            "Cumulative CLV",
            "Betting Simulation",
            "Equity Curve",
            "ROI Breakdown",
            "Edge Distribution",
        ):
            assert heading in html, heading

    def test_places_each_backtest_chart_once(self, test_client: TestClient) -> None:
        html = test_client.get("/track-record").text
        assert html.count("test heatmap") == 1
        assert html.count("test clv") == 1
        for chart_id in _VS_MARKET:
            assert html.count(f'data-chart-id="{chart_id}"') == 1, chart_id
        # The duplicates the merge removed: the backtest equity curve repeats the betting equity
        # curve, and WP calibration (with the rest of /insights' model half) lives on How It Works.
        assert "test equity" not in html
        assert 'data-chart-id="calibration"' not in html
        for chart_id in _HOW_IT_WORKS_INSIGHTS:
            assert f'data-chart-id="{chart_id}"' not in html, chart_id

    def test_renders_the_recommended_betting_charts_by_default(
        self, test_client: TestClient
    ) -> None:
        from tests.api.conftest import _BETTING_CHART_BASES

        html = test_client.get("/track-record").text
        for base in _BETTING_CHART_BASES:
            if base in ("betting_kpis", "betting_roi_table"):
                continue
            assert f'data-chart-id="{base}_recommended"' in html, base
            assert f'data-chart-id="{base}_all"' not in html, base

    def test_the_scope_parameter_selects_the_betting_scope(
        self, test_client: TestClient
    ) -> None:
        html = test_client.get("/track-record?scope=all").text
        assert 'data-chart-id="betting_equity_all"' in html
        assert 'data-chart-id="betting_equity_recommended"' not in html

    def test_an_unknown_scope_falls_back_to_recommended(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get("/track-record?scope=bogus")
        assert response.status_code == 200
        assert 'data-chart-id="betting_equity_recommended"' in response.text

    def test_the_season_parameter_selects_the_season(
        self, test_client: TestClient
    ) -> None:
        html = test_client.get("/track-record?season=2023").text
        assert re.search(r'<option value="2023"\s+selected', html)

    @pytest.mark.parametrize("raw", ["abc", "--5", "", "1.5"])
    def test_a_malformed_season_degrades_to_all_seasons(
        self, test_client: TestClient, raw: str
    ) -> None:
        response = test_client.get(f"/track-record?season={raw}")
        assert response.status_code == 200
        assert re.search(r'<option value=""\s+selected', response.text)

    def test_an_empty_cache_renders_empty_states_not_a_500(
        self, empty_test_client: TestClient
    ) -> None:
        response = empty_test_client.get("/track-record")
        assert response.status_code == 200
        assert "Chart unavailable" in response.text
        assert "No backtest data" in response.text


class TestHowItWorks:
    def test_renders_its_sections(self, test_client: TestClient) -> None:
        response = test_client.get("/how-it-works")
        assert response.status_code == 200
        html = response.text
        # The page's own <h1>: "How It Works" alone would also be found in the nav and <title>.
        assert re.search(r"<h1[^>]*>How It Works</h1>", html), (
            "the page heading is missing"
        )
        for heading in ("Calibration", "What the models rely on", "Accuracy over time"):
            assert heading in html, heading
        for anchor in ("calibration", "features", "accuracy"):
            assert f'id="{anchor}"' in html, anchor

    def test_renders_the_model_charts_once_and_not_the_market_ones(
        self, test_client: TestClient
    ) -> None:
        html = test_client.get("/how-it-works").text
        assert html.count('data-chart-id="calibration"') == 1
        for chart_id in _HOW_IT_WORKS_INSIGHTS:
            assert html.count(f'data-chart-id="{chart_id}"') == 1, chart_id
        for chart_id in _VS_MARKET:
            assert f'data-chart-id="{chart_id}"' not in html, chart_id

    def test_an_empty_cache_renders_empty_states_not_a_500(
        self, empty_test_client: TestClient
    ) -> None:
        response = empty_test_client.get("/how-it-works")
        assert response.status_code == 200
        assert "Chart unavailable" in response.text


class TestRetiredUrls:
    @pytest.mark.parametrize(
        ("url", "location"),
        [
            ("/performance", "/track-record#seasons"),
            ("/performance?season=2023", "/track-record?season=2023#seasons"),
            ("/backtest", "/track-record"),
            ("/betting", "/track-record#betting-sim"),
            ("/betting?scope=all", "/track-record?scope=all#betting-sim"),
            ("/insights", "/how-it-works"),
        ],
    )
    def test_a_retired_url_redirects_permanently_with_its_query(
        self, test_client: TestClient, url: str, location: str
    ) -> None:
        response = test_client.get(url, follow_redirects=False)
        assert response.status_code == 301
        assert response.headers["location"] == location

    def test_an_old_season_bookmark_opens_that_season(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get("/performance?season=2023")
        assert response.status_code == 200
        assert response.url.path == "/track-record"
        assert re.search(r'<option value="2023"\s+selected', response.text)

    def test_an_old_scope_bookmark_opens_that_scope(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get("/betting?scope=all")
        assert response.status_code == 200
        assert response.url.path == "/track-record"
        assert 'data-chart-id="betting_equity_all"' in response.text

    def test_an_htmx_request_to_the_old_performance_url_gets_the_season_block(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get(
            "/performance?season=2023", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        html = response.text
        assert "<html" not in html
        assert "<nav" not in html
        assert "Season Metrics" in html

    def test_an_htmx_request_to_the_old_betting_url_gets_the_betting_block(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get("/betting?scope=all", headers={"HX-Request": "true"})
        assert response.status_code == 200
        html = response.text
        assert "<html" not in html
        assert "<nav" not in html
        assert 'data-chart-id="betting_equity_all"' in html

    @pytest.mark.parametrize(
        "name", ["performance.html", "backtest.html", "betting.html", "insights.html"]
    )
    def test_the_retired_templates_are_gone(self, name: str) -> None:
        assert not (PAGES_DIR / name).exists()


def _methodology_section(html: str) -> str:
    """The methodology section's own markup, ending at its closing tag -- so the old-rule label
    that follows it can never satisfy an assertion about the methodology text."""
    assert 'id="methodology"' in html, "How It Works has no methodology section"
    return html.split('id="methodology"', 1)[1].split("</section>", 1)[0]


class TestTheMethodologySection:
    def test_sits_above_the_charts(self, test_client: TestClient) -> None:
        html = test_client.get("/how-it-works").text
        assert html.index('id="methodology"') < html.index('id="calibration"')

    def test_states_the_lock_the_evidence_rule_and_the_advice_disclaimer(
        self, test_client: TestClient
    ) -> None:
        section = _methodology_section(test_client.get("/how-it-works").text)
        assert "6 PM Eastern on the day before kickoff" in section
        assert "not evidence" in section
        assert "not betting advice" in section
        assert "walk-forward" in section

    def test_makes_no_over_claim(self, test_client: TestClient) -> None:
        from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS

        section = _methodology_section(test_client.get("/how-it-works").text).lower()
        assert [word for word in READOUT_FORBIDDEN_WORDS if word in section] == []

    def test_is_ascii(self, test_client: TestClient) -> None:
        section = _methodology_section(test_client.get("/how-it-works").text)
        assert section.isascii()
