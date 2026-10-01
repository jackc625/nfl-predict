"""Responsive and accessibility audit for the Broadcast redesign (redesign Task 19).

Rendered-HTML rules run against every page the test client serves, AND against /bets and / built
from caches that hold live, suppressed and graded bets -- the shared test_client cache has no bet
tables, so without those the bet slips, tracker, result strip, suppressed disclosure and headliner
cards would never be audited. Source rules read the templates and stylesheets directly. Each rule
pins one line of spec section 10.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Page -> the nav href that must carry aria-current="page" while it is open.
PAGES: dict[str, str] = {
    "/": "/",
    "/bets": "/bets",
    "/season": "/season",
    "/track-record": "/track-record",
    "/how-it-works": "/how-it-works",
    "/games/2024_W01_BUF@KC": "/",
}

# Classes whose element is itself transformed with skewX. Text inside them must sit in a
# descendant .unskew so it reads upright (spec 10). A component that paints its skew on a
# ::before layer instead (its text is never transformed) does not belong in this set.
SKEWED_CLASSES = frozenset(
    {
        "skew",
        "tag",
        "tag-ghost",
        "team-block",
        "team-block-lg",
        "edge-chip",
        "edge-chip-soft",
        "skew-control",
        "skew-control-active",
    }
)

_TEXTLESS_TAGS = frozenset({"script", "style", "option", "title"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }
)
# A proportion bar is drawn with an inline percentage width; 100% is a layout width, not a bar.
_BAR_WIDTH = re.compile(r"(?:^|;)\s*width:\s*(?!100%)\d+(?:\.\d+)?%")
_DIGIT = re.compile(r"\d")
_TOUCH_CLASSES = frozenset(
    {"skew-control", "skew-control-active", "min-h-[44px]", "min-h-11"}
)


class _Element:
    def __init__(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag = tag
        self.attrs = {name: value or "" for name, value in attrs}
        self.classes = frozenset(self.attrs.get("class", "").split())

    def is_labelled_image(self) -> bool:
        return self.attrs.get("role") == "img" and bool(
            _DIGIT.search(self.attrs.get("aria-label", ""))
        )


class PageAudit(HTMLParser):
    """Walks one rendered page and records every accessibility-rule violation."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Element] = []
        self.skewed_text: list[str] = []
        self.unlabelled_bars: list[str] = []
        self.unlabelled_images: list[str] = []
        self.unscrollable_tables = 0
        self.current_nav_hrefs: list[str] = []
        self.mobile_links_without_target: list[str] = []
        self.menu_buttons: list[dict[str, str]] = []
        self.class_sets: list[frozenset[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = _Element(tag, attrs)
        self._check(element)
        if tag not in _VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._check(_Element(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or any(el.tag in _TEXTLESS_TAGS for el in self.stack):
            return
        for element in reversed(self.stack):
            if "unskew" in element.classes:
                return
            skewed = element.classes & SKEWED_CLASSES
            if skewed:
                self.skewed_text.append(
                    f"{sorted(skewed)} <{element.tag}>: {text[:40]!r}"
                )
                return

    def _check(self, element: _Element) -> None:
        self.class_sets.append(element.classes)
        attrs = element.attrs
        labelled = element.is_labelled_image() or any(
            el.is_labelled_image() for el in self.stack
        )
        if _BAR_WIDTH.search(attrs.get("style", "")) and not labelled:
            self.unlabelled_bars.append(f"<{element.tag} style={attrs['style']!r}>")
        if attrs.get("role") == "img" and not element.is_labelled_image():
            self.unlabelled_images.append(
                f"<{element.tag} class={attrs.get('class', '')!r}>"
            )
        if element.tag == "table" and not any(
            "overflow-x-auto" in el.classes for el in self.stack
        ):
            self.unscrollable_tables += 1
        in_nav = any(el.tag == "nav" for el in self.stack)
        if element.tag == "a" and in_nav and attrs.get("aria-current") == "page":
            self.current_nav_hrefs.append(attrs.get("href", ""))
        in_mobile_menu = any(el.attrs.get("id") == "mobile-menu" for el in self.stack)
        if (
            element.tag == "a"
            and in_mobile_menu
            and not element.classes & _TOUCH_CLASSES
        ):
            self.mobile_links_without_target.append(attrs.get("href", ""))
        if element.tag == "button" and attrs.get("aria-controls") == "mobile-menu":
            self.menu_buttons.append(attrs)


def _audit_html(html: str) -> PageAudit:
    audit = PageAudit()
    audit.feed(html)
    audit.close()
    return audit


def _get(client: TestClient, path: str) -> str:
    response = client.get(path)
    assert response.status_code == 200, path
    return response.text


def _audit(client: TestClient, path: str) -> PageAudit:
    return _audit_html(_get(client, path))


# ---------------------------------------------------------------------------
# Rendered-page rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", list(PAGES))
def test_skewed_boxes_never_skew_their_text(test_client: TestClient, path: str) -> None:
    assert not _audit(test_client, path).skewed_text


@pytest.mark.parametrize("path", list(PAGES))
def test_bars_and_strips_carry_their_numbers_for_screen_readers(
    test_client: TestClient, path: str
) -> None:
    audit = _audit(test_client, path)
    assert not audit.unlabelled_bars
    assert not audit.unlabelled_images


@pytest.mark.parametrize("path", list(PAGES))
def test_wide_tables_scroll_instead_of_clipping(
    test_client: TestClient, path: str
) -> None:
    assert _audit(test_client, path).unscrollable_tables == 0


@pytest.mark.parametrize("path", list(PAGES))
def test_the_active_nav_item_is_announced(test_client: TestClient, path: str) -> None:
    assert set(_audit(test_client, path).current_nav_hrefs) == {PAGES[path]}


@pytest.mark.parametrize("path", list(PAGES))
def test_the_mobile_menu_is_wired_for_assistive_tech(
    test_client: TestClient, path: str
) -> None:
    audit = _audit(test_client, path)
    assert len(audit.menu_buttons) == 1
    button = audit.menu_buttons[0]
    assert button.get("aria-expanded") in {"true", "false"}
    assert button.get("aria-label")
    assert not audit.mobile_links_without_target


def test_this_week_slate_steps_4_3_2_1(test_client: TestClient) -> None:
    audit = _audit(test_client, "/")
    assert any(
        {"grid-cols-1", "lg:grid-cols-3", "xl:grid-cols-4"} <= classes
        and bool({"sm:grid-cols-2", "md:grid-cols-2"} & classes)
        for classes in audit.class_sets
    )


# ---------------------------------------------------------------------------
# The same rendered-page rules on the markup only a cache WITH bets produces
# ---------------------------------------------------------------------------

# Audited page -> the nav href that must carry aria-current="page" on it.
_BET_PAGES: dict[str, str] = {"/bets": "/bets", "/": "/"}
_BETS_SEASON = 2023
_BETS_WEEK = 1


@pytest.fixture(scope="module")
def bet_page_audits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, PageAudit]:
    """Audits of /bets and / rendered from caches holding live, suppressed and graded bets.

    Built with the bets and headliner suites' own builders (the production materializers), so
    the audit reads the same markup those suites pin. Each page is checked for the markup that
    makes the audit meaningful before it is audited, so the rules can never pass vacuously.
    """
    from tests.api.test_bets_page import (
        _CONTAMINATED,
        _FORWARD_CLASS,
        _block,
        _client_with_tracker,
        _graded_row,
        _live_row,
        _suppressed_row,
    )
    from tests.api.test_this_week_headliner import (
        _WEEK_GAMES,
        _build_state_cache,
        _insert_predictions,
        _serving,
    )

    tmp_path = tmp_path_factory.mktemp("a11y_bets")
    # Five live slips -- one ungraded, three graded replay bets and one graded forward bet --
    # and two suppressed candidates, one with no recorded reason, so the disclosure renders
    # its tables.
    rows = [
        _live_row("2023_W01_BUF@MIA", "ou"),
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "loss"),
        _graded_row("2023_W01_CIN@CLE", "push"),
        _graded_row("2023_W01_DEN@LVR", "win", pair=_FORWARD_CLASS),
        _suppressed_row("2023_W01_SEA@SFO", "ats", "ev_below_floor"),
        _suppressed_row("2023_W01_NYJ@NE", "wp", None),
    ]
    # Stored blocks that AGREE with the graded rows, so each result strip renders.
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=3,
            wins=1,
            losses=1,
            pushes=1,
            hit_rate=0.5,
            flat_return_units=-0.091,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=1,
            wins=1,
            losses=0,
            pushes=0,
            hit_rate=1.0,
            flat_return_units=0.909,
        ),
    ]
    audits: dict[str, PageAudit] = {}
    with _client_with_tracker(tmp_path, blocks, "a11y_bets", rows=rows) as client:
        bets_html = _get(client, f"/bets?season={_BETS_SEASON}&week={_BETS_WEEK}")
    for marker in (
        '<li class="bet-slip',
        "data-tracker-block=",
        "data-result-strip",
        '<details id="suppressed-candidates"',
    ):
        assert marker in bets_html, (
            f"/bets rendered no {marker}; its audit would be vacuous"
        )
    audits["/bets"] = _audit_html(bets_html)

    home_db = tmp_path / "a11y_home.duckdb"
    _build_state_cache(home_db, "bets")
    _insert_predictions(home_db, _WEEK_GAMES)
    with _serving(home_db) as (client, _conn):
        home_html = _get(client, f"/?season={_BETS_SEASON}&week={_BETS_WEEK}")
    assert "data-headliner-bet=" in home_html, (
        "/ rendered no headliner card; vacuous audit"
    )
    audits["/"] = _audit_html(home_html)
    return audits


@pytest.mark.parametrize("page", list(_BET_PAGES))
def test_bet_carrying_pages_pass_every_rendered_rule(
    bet_page_audits: dict[str, PageAudit], page: str
) -> None:
    """Slips and their team blocks, tracker tiles and result strips, the suppressed tables and
    the headliner cards: upright text, labelled bars and strips, scrolling tables, the
    announced nav item and the wired mobile menu."""
    audit = bet_page_audits[page]
    assert not audit.skewed_text
    assert not audit.unlabelled_bars
    assert not audit.unlabelled_images
    assert audit.unscrollable_tables == 0
    assert set(audit.current_nav_hrefs) == {_BET_PAGES[page]}
    assert len(audit.menu_buttons) == 1
    assert not audit.mobile_links_without_target


# ---------------------------------------------------------------------------
# Source rules (templates and stylesheets)
# ---------------------------------------------------------------------------

_TEMPLATES = Path("web/templates")
_INPUT_CSS = Path("web/static/input.css")
_CUSTOM_CSS = Path("web/static/css/custom.css")
_CLASS_ATTR = re.compile(r'class="([^"]*)"')
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_DISPLAY_TOKENS = frozenset(
    {
        "font-display",
        "display",
        "label",
        "tag",
        "tag-ghost",
        "edge-chip",
        "edge-chip-soft",
        "skew-control",
        "panel-title",
    }
)
_ARBITRARY_SIZE = re.compile(r"^text-\[(\d+(?:\.\d+)?)(px|rem)\]$")
_MIN_44 = re.compile(r"min-height:\s*(?:44px|2\.75rem)|min-h-(?:\[44px\]|11)\b")


def _class_lists(path: Path) -> list[list[str]]:
    return [
        m.group(1).split()
        for m in _CLASS_ATTR.finditer(path.read_text(encoding="utf-8"))
    ]


def _rules(css: str) -> list[tuple[str, str]]:
    return [(selector.strip(), body) for selector, body in _RULE.findall(css)]


def _px(value: str, unit: str) -> float:
    return float(value) * (16 if unit == "rem" else 1)


def _at_rule(css: str, header: str) -> str:
    """Return the full text of the first ``header { ... }`` block, nested braces included."""
    start = css.index(header)
    depth = 0
    for index in range(css.index("{", start), len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[start : index + 1]
    msg = f"unterminated {header}"
    raise AssertionError(msg)


def _stylesheets() -> str:
    return _CUSTOM_CSS.read_text(encoding="utf-8") + _INPUT_CSS.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "template", ["pages/track_record.html", "pages/how_it_works.html"]
)
def test_merged_page_grids_collapse_to_one_column(template: str) -> None:
    for classes in _class_lists(_TEMPLATES / template):
        assert "grid-cols-3" not in classes, classes
        assert "grid-cols-4" not in classes, classes
        if "lg:grid-cols-3" in classes:
            assert "grid-cols-1" in classes, classes


def test_headliner_cards_stack_on_phones() -> None:
    lists = _class_lists(_TEMPLATES / "components/_bet_headliners.html")
    assert any("grid-cols-1" in c and "lg:grid-cols-3" in c for c in lists)


@pytest.mark.parametrize(
    "partial", ["components/_result_strip.html", "components/_week_strip.html"]
)
def test_strip_partials_label_themselves(partial: str) -> None:
    source = (_TEMPLATES / partial).read_text(encoding="utf-8")
    markup = re.sub(r"\{#.*?#\}|\{%.*?%\}", "", source, flags=re.DOTALL)
    first_tag = re.search(r"<([a-z][a-z0-9]*)\b([^>]*)>", markup)
    assert first_tag, partial
    # A labelled list (the week strip: one <li> per week, each with its own record as text) is
    # better than role="img", which would hide those items from assistive tech; a strip with no
    # per-item text (the result strip's tick marks) must be a labelled image.
    is_labelled_list = first_tag.group(1) in ("ol", "ul")
    assert is_labelled_list or 'role="img"' in first_tag.group(2), partial
    assert "aria-label=" in first_tag.group(2), partial


def test_condensed_display_type_is_never_below_12px_in_templates() -> None:
    offenders: list[str] = []
    for path in sorted(_TEMPLATES.rglob("*.html")):
        for classes in _class_lists(path):
            if not _DISPLAY_TOKENS & set(classes):
                continue
            for token in classes:
                match = _ARBITRARY_SIZE.match(token)
                if match and _px(match.group(1), match.group(2)) < 12:
                    offenders.append(f"{path}: {token}")
    assert not offenders


def test_condensed_display_components_are_never_below_12px() -> None:
    for selector, body in _rules(_INPUT_CSS.read_text(encoding="utf-8")):
        if not set(re.findall(r"\.([\w-]+)", selector)) & _DISPLAY_TOKENS:
            continue
        sizes = [
            _px(v, u)
            for v, u in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)(px|rem)", body)
        ] + [
            _px(v, u) for v, u in re.findall(r"text-\[(\d+(?:\.\d+)?)(px|rem)\]", body)
        ]
        assert all(size >= 12 for size in sizes), f"{selector}: {sizes}"


def test_controls_and_disclosures_declare_44px_touch_targets() -> None:
    rules = _rules(_INPUT_CSS.read_text(encoding="utf-8"))
    control = [b for s, b in rules if ".skew-control" in s and "-active" not in s]
    disclosure = [b for s, b in rules if ".honesty-note" in s and "summary" in s]
    assert any(_MIN_44.search(body) for body in control), ".skew-control needs 44px"
    assert any(_MIN_44.search(body) for body in disclosure), (
        ".honesty-note summary needs 44px"
    )


def test_reduced_motion_stops_lifts_and_animations() -> None:
    block = _at_rule(_stylesheets(), "@media (prefers-reduced-motion: reduce)")
    assert "transform: none" in block
    assert "animation" in block


def test_focus_ring_is_the_accent() -> None:
    """One visible ring everywhere: the universal :focus-visible rule draws an accent OUTLINE.

    Keyed on the universal rule, not on any :focus-visible rule: select.skew-control's rule
    colours its text accent and sets outline:none (its clip-path would cut a ring off), so a
    check that accepted any rule mentioning the accent would pass with no ring at all.
    """
    ring_bodies = [
        body
        for selector, body in _rules(_stylesheets())
        if selector.strip().endswith("*:focus-visible")
    ]
    assert ring_bodies, "no universal *:focus-visible rule"
    assert any(
        re.search(
            r"outline:\s*\d+px\s+solid\s+(?:#ffd400|var\(--color-accent\))",
            body,
            re.IGNORECASE,
        )
        for body in ring_bodies
    ), "the universal :focus-visible rule does not draw an accent outline"


def test_print_is_black_on_white() -> None:
    block = _at_rule(_stylesheets(), "@media print")
    assert re.search(
        r"background(?:-color)?:\s*(?:#fff\b|#ffffff|white)", block, re.IGNORECASE
    )
    assert re.search(
        r"(?<![-\w])color:\s*(?:#000\b|#000000|black)", block, re.IGNORECASE
    )
