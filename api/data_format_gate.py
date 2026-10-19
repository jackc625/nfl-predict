"""The data-format gate: a cache built for different code answers a notice, never wrong pages.

Phase 36, SPEC R9, D-08, D-19. The server records the cache's data-format verdict on
``app.state.data_format`` at startup and on every reconnect (``api.main.lifespan``,
``api.dependencies._reconnect_under_lock``). Every page, fragment and export router carries
:func:`data_format_gate` as a router-level dependency, wired in ``api/main.py`` where the routers
are included, and :func:`data_format_mismatch_handler` turns its refusal into HTTP 503. ``/static``
and ``/health`` are not gated.

WHY A DEPENDENCY PLUS A HANDLER AND NOT MIDDLEWARE. Vercel keeps the static mount inside the
function, instead of promoting it to its CDN, when the app has top-level middleware (research
Pattern 3). And the gate computes nothing: it compares one recorded verdict, so it needs no hook
around every request, static files included.

WHY THE GATE LETS THE SWAP DETECTOR RECONNECT FIRST. The recorded verdict is only re-read when the
connection is replaced. A self-heal rebuild (D-03) swaps a matching cache in; judging the stale
verdict first would keep answering 503 until some ungated request reconnected. So when the detector
says the file was replaced, the gate reconnects through ``get_db`` before it judges, with the same
guarded delegation ``api.routes.health._read_last_updated`` uses: any failure there is left to the
route, which reaches ``get_db`` itself.

WHY AN ABSENT VERDICT MAKES NO CLAIM (research Pitfall 2). Without a lifespan -- every test client
built without ``with`` -- there is no recorded verdict, and a cache with no file is no evidence of a
format either. Only a recorded verdict that does not match refuses.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress

import duckdb
from fastapi import Request
from starlette.responses import PlainTextResponse, Response

from api.cache_schema import DataFormatState
from api.dependencies import cache_file_changed, get_db, templates
from api.exceptions import DataFormatMismatchError, NFLPredictionAPIException

# The three kinds of gated route. A page answers with the notice page; a fragment (HTMX does not
# swap an error response) and an export answer with the same sentence as plain text.
PAGE = "page"
FRAGMENT = "fragment"
EXPORT = "export"

_NOTICE_TEMPLATE = "data_format_notice.html"
_NOTICE_OPENING = "This copy of the site can't show its data yet."
_NOTICE_CLOSING = "The next 5 PM run will try to rebuild it."
# A non-integer stored value is echoed back; this many characters of it at most.
_RAW_SHOWN_CHARS = 40
# A refusal must never be cached: the next request after a rebuild has to be judged afresh.
_NO_STORE = {"Cache-Control": "no-store"}


def data_format_gate(kind: str) -> Callable[[Request], None]:
    """The router-level dependency that refuses every route of *kind* under a format clash."""

    def gate(request: Request) -> None:
        if cache_file_changed(request):
            with suppress(NFLPredictionAPIException, duckdb.Error, OSError):
                get_db(request)
        state = getattr(request.app.state, "data_format", None)
        if state is not None and not state.matches:
            raise DataFormatMismatchError(state, kind)

    return gate


def notice_sentence(state: DataFormatState) -> str:
    """The one notice sentence, naming the format the pages expect and what the file holds."""
    label = state.found_label
    if label == "missing":
        found = "the data file has no format number"
    elif label == "non_integer":
        shown = (state.raw or "")[:_RAW_SHOWN_CHARS]
        found = f'the data file\'s format reads "{shown}", which is not a number'
    else:
        found = f"the data file is format {label}"
    return (
        f"{_NOTICE_OPENING} The pages expect data format {state.expected}; {found}. "
        f"{_NOTICE_CLOSING}"
    )


async def data_format_mismatch_handler(request: Request, exc: Exception) -> Response:
    """Answer a :class:`DataFormatMismatchError` with HTTP 503: the notice page or plain text."""
    if not isinstance(exc, DataFormatMismatchError):
        raise exc
    sentence = notice_sentence(exc.state)
    if exc.kind == PAGE:
        return templates.TemplateResponse(
            request,
            _NOTICE_TEMPLATE,
            {"notice": sentence},
            status_code=503,
            headers=_NO_STORE,
        )
    return PlainTextResponse(sentence, status_code=503, headers=_NO_STORE)
