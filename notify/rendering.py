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

The environment is built once per process; Plan 35-11 registers the email's globals on it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import functools
from pathlib import Path

import jinja2

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
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader([NOTIFY_TEMPLATES_DIR, WEB_TEMPLATES_DIR]),
        undefined=jinja2.StrictUndefined,
        autoescape=_autoescape,
    )
