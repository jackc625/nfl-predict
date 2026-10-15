"""The one Jinja environment the notify messages render through (D-14, D-19).

WHY THE WEB TEMPLATES ARE ON THE PATH
-------------------------------------
The loader searches ``notify/templates`` first and ``web/templates`` second, so a notify template
imports ``/bets``' own components -- ``components/_bet_pick.html`` above all -- instead of copying
them. The spread-sign rule (a pick's line from the picked side, as a sportsbook quotes it) then
lives in exactly one place, and the push and the email cannot word a bet differently from
``/bets``.

STRICT ON PURPOSE
-----------------
``StrictUndefined`` makes a template typo fail loudly instead of rendering a silent blank; the
caller escalates the failure (Plan 35-11 records an email that could not be built and pushes it).
Autoescaping is on only for HTML templates, so a plain-text push is never HTML-escaped.

THE EMAIL'S SHARED WORDING
--------------------------
The environment is built once per process, with the globals ``/bets`` renders from
``api.presentation`` registered on it -- ``last_week_display``, ``not_advice_key`` and
``not_advice_line`` -- so the weekly email cannot word the last-week block or the not-advice line
differently from the page (SPEC R6). The import direction is notify -> api, never the reverse
(D-19, UIAP-01).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import functools
from pathlib import Path

import jinja2

from api.presentation import NOT_ADVICE_KEY, NOT_ADVICE_LINE, last_week_display

__all__ = ["NOTIFY_TEMPLATES_DIR", "WEB_TEMPLATES_DIR", "template_environment"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
NOTIFY_TEMPLATES_DIR = _REPO_ROOT / "notify" / "templates"
WEB_TEMPLATES_DIR = _REPO_ROOT / "web" / "templates"


def _autoescape(template_name: str | None) -> bool:
    """HTML templates are escaped; plain-text ones are not."""
    return template_name is not None and template_name.endswith((".html", ".html.j2"))


@functools.cache
def template_environment() -> jinja2.Environment:
    """The process-wide environment over ``notify/templates`` then ``web/templates``."""
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader([NOTIFY_TEMPLATES_DIR, WEB_TEMPLATES_DIR]),
        undefined=jinja2.StrictUndefined,
        autoescape=_autoescape,
    )
    environment.globals.update(
        last_week_display=last_week_display,
        not_advice_key=NOT_ADVICE_KEY,
        not_advice_line=NOT_ADVICE_LINE,
    )
    return environment
