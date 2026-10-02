"""The Broadcast game detail page (Broadcast redesign, Task 9).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

import duckdb
from fastapi.testclient import TestClient

from api.cache import CACHE_SCHEMA, PREDICTIONS_TABLE_COLUMNS
from api.dependencies import get_db, templates
from api.presentation import decorate_game
from api.services import DataService, clear_cache

_GAME_ID = "2026_W03_KC@MIA"
_HUE = re.compile(r"\b(?:text|bg|border)-(green|red)-\d{2,3}")
# One rendered band: the data-band hook, the monochrome label inside it, the band that label
# declares, and its visible text. Whitespace between the tags is the include's own newlines.
_BAND = re.compile(
    r'<span data-band="([a-z]+)">\s*'
    r'<span class="band band-([a-z]+)" data-confidence-band="([a-z]+)">([^<]+)</span>\s*'
    r"</span>"
)


class _StubRequest:
    """The one attribute the base template needs from a request: ``url_for`` for static files."""

    def url_for(self, name: str, **path_params: Any) -> str:
        return f"/{name}/{path_params.get('path', '')}"


def _main(html: str) -> str:
    """The page's <main> region, so the site chrome cannot satisfy or spoil an assertion."""
    match = re.search(r"<main[^>]*>(.*)</main>", html, re.DOTALL)
    assert match, "the page has no <main> region"
    return match.group(1)


def _hues(html: str) -> list[str]:
    return [
        match.group(1)
        for classes in re.findall(r'class="([^"]*)"', html)
        for match in _HUE.finditer(classes)
    ]


def _detail_html(game_patch: dict[str, Any] | None = None, **overrides: Any) -> str:
    """Render the detail page for KC at MIA stored in an in-memory cache, decorated as the route does."""
    clear_cache()
    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    row = dict.fromkeys(PREDICTIONS_TABLE_COLUMNS)
    row.update(
        game_id=_GAME_ID,
        season=2026,
        week=3,
        game_date=datetime(2026, 9, 27, 13, 0),
        home_team="MIA",
        away_team="KC",
        status="scheduled",
        wp_prob=0.412,
        ats_prediction=-2.5,
        ou_prediction=41.3,
    )
    row.update(overrides)
    columns = ", ".join(PREDICTIONS_TABLE_COLUMNS)
    placeholders = ", ".join("?" for _ in PREDICTIONS_TABLE_COLUMNS)
    conn.execute(
        f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
        [row[c] for c in PREDICTIONS_TABLE_COLUMNS],
    )
    game = DataService(conn).get_game_detail(_GAME_ID)
    conn.close()
    assert game is not None
    return templates.env.get_template("pages/game_detail.html").render(
        request=_StubRequest(),
        game={**decorate_game(game), **(game_patch or {})},
        current_path="",
        cache_meta={},
        old_rule_scope=DataService.old_rule_scope([]),
    )


def test_the_detail_page_renders_the_band_it_was_served(
    test_client: TestClient,
) -> None:
    """Moved from / (31-17 / D31-23): the page renders the STORED band, never a re-derived one.

    Every stored band must appear once, for its own target, as a monochrome label -- a page that
    turned a served "low" into a "high" keeps the vocabulary intact and changes these counts. The
    check reads the LABEL the reader sees, not only the data-band hook that echoes the stored
    value: the hook, the label's class, its data-confidence-band and its text must all agree.
    """
    from api.main import app
    from utils.edge_tier import EDGE_TIER_LABELS

    html = test_client.get("/games/2024_W01_BUF@KC").text
    # The app object, not test_client.app: TestClient types .app as a bare ASGI callable.
    game = DataService(app.dependency_overrides[get_db]()).get_game_detail(
        "2024_W01_BUF@KC"
    )
    assert game is not None
    served = [
        game[f"{target}_confidence"]
        for target in ("wp", "ats", "ou")
        if game.get(f"{target}_confidence") is not None
    ]
    assert served, "the fixture served no band; the check would pass vacuously"
    assert set(served) <= set(EDGE_TIER_LABELS)

    hooks = re.findall(r'data-band="([a-z]+)"', html)
    labels = _BAND.findall(html)
    assert len(labels) == len(hooks), (
        "a data-band hook does not wrap exactly one monochrome label"
    )
    assert Counter(hook for hook, _css, _attr, _text in labels) == Counter(served)
    for hook, css_band, attr_band, text in labels:
        # The class is exactly "band band-<level>", so no hue can ride on it.
        assert hook == css_band == attr_band == text.strip().lower(), (
            f"the {hook!r} slot renders the label {text!r} (class band-{css_band}, "
            f"data-confidence-band={attr_band!r}); all four must name the stored band"
        )


def test_a_detail_page_with_no_market_lines_reads_cleanly() -> None:
    """Review Focus 1 on the detail page: every market slot says No line, and nothing else leaks."""
    main = _main(_detail_html())

    assert main.count("No line") == 3, "each of the three market cells must say No line"
    assert "data-edge" not in main
    assert "data-band" not in main
    assert "None" not in main
    assert "nan" not in main.lower()
    for pick_word in ("covers", "Over ", "Under "):
        assert pick_word not in main
    assert "KC 58.8%" in main and "KC by 2.5" in main and "41.3" in main


def test_only_a_realised_result_is_green_or_red() -> None:
    before = _main(
        _detail_html(
            market_wp=0.449,
            market_spread=-1.5,
            market_total=44.5,
            wp_edge=-0.037,
            ats_edge=-1.0,
            ou_edge=-0.0719,
            wp_confidence="medium",
            ats_confidence="low",
            ou_confidence="high",
        )
    )
    assert _hues(before) == [], "a pre-game edge or band was drawn in green or red"
    assert re.search(r'data-edge="ats".*?-1\.0 pts', before, re.DOTALL)
    assert "KC covers" in before and "Under 44.5" in before

    after = _main(
        _detail_html(status="completed", away_score=20, home_score=27, wp_prob=0.70)
    )
    assert "20 - 27" in after and "Correct" in after
    assert set(_hues(after)) == {"green"}


def test_the_page_is_decorated_and_links_back_to_its_week(
    test_client: TestClient,
) -> None:
    html = test_client.get("/games/2024_W01_BUF@KC").text

    assert "Chiefs" in html, "the team nickname from api.presentation is missing"
    assert "#E31837" in html, "the KC team colour from utils/team_data.py is missing"
    assert 'href="/?season=2024&amp;week=1"' in html
    assert "Published" in html and "Blended" not in html
    # Spec 5: the win-probability bar grows in once (Task 1's .grow-in, off under reduced motion).
    assert re.search(
        r'<div class="[^"]*\bgrow-in\b[^"]*" role="img" aria-label="Win probability:',
        html,
    ), "the win-probability bar does not carry the grow-in animation"


def test_the_detail_page_marks_this_week_active_in_the_nav(
    test_client: TestClient,
) -> None:
    """A /games/<id> page keeps the reader's place: only the This Week link is active."""
    html = test_client.get("/games/2024_W01_BUF@KC").text
    nav = re.search(r"<nav[^>]*>.*?</nav>", html, re.DOTALL)
    assert nav, "the page has no <nav>"
    links = re.findall(r"<a\b[^>]*>", nav.group(0))
    # The desktop and the mobile menu each render the nav, so This Week appears once in each.
    active = [link for link in links if "skew-control-active" in link]
    assert active, "no nav link is active"
    assert all(
        'href="/"' in link and 'aria-current="page"' in link for link in active
    ), active
    assert sum('aria-current="page"' in link for link in links) == len(active)


def test_the_header_shows_elo_and_form_under_each_team(test_client: TestClient) -> None:
    lines = re.findall(
        r"<p[^>]*data-elo-form>([^<]*)</p>",
        test_client.get("/games/2024_W01_BUF@KC").text,
    )
    assert len(lines) == 2, lines
    for line in lines:
        assert re.fullmatch(
            r"(Elo \d+)?( &middot; )?(last \d: [WLT]( [WLT])*)?", line
        ), line
        assert (
            line.strip()
            and not line.startswith(" &middot;")
            and not line.endswith("&middot; ")
        )
    assert any(line.startswith("Elo ") for line in lines)


def test_the_header_form_line_escapes_text_read_from_the_cache() -> None:
    """The form strings come from the cache, so they are escaped like any other cached text."""
    context = {
        "away_elo": 1550.0,
        "home_elo": None,
        "away_last5_list": ["W", "<b>L</b>"],
        "home_last5_list": [],
    }
    lines = re.findall(
        r"<p[^>]*data-elo-form>(.*?)</p>", _main(_detail_html({"context": context}))
    )
    assert lines == ["Elo 1550 &middot; last 2: W &lt;b&gt;L&lt;/b&gt;"]


def test_the_header_form_line_is_absent_without_context() -> None:
    main = _main(_detail_html())
    assert "data-elo-form" not in main
    assert "&middot;" not in main


def test_a_realised_clv_of_zero_is_neutral() -> None:
    def clv_class(clv: float) -> str:
        html = _main(
            _detail_html(
                {"wp_clv": clv}, status="completed", away_score=20, home_score=27
            )
        )
        match = re.search(r'<span class="([^"]*)">CLV:', html)
        assert match, "no CLV line"
        return match.group(1)

    assert "text-green-400" in clv_class(0.5)
    assert "text-red-400" in clv_class(-0.5)
    zero = clv_class(0.0)
    assert "text-muted" in zero and "green" not in zero and "red" not in zero
