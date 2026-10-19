#!/usr/bin/env python3
"""Check the hosted website from the laptop: every route, or Vercel's refusals (Phase 36, SPEC R3/R4/R12).

WHAT IT PROVES
--------------
Normal mode (SPEC R4, R12(a)). It GETs every route of a fixed list from the production URL, sending
the protection bypass ONLY as the ``x-vercel-protection-bypass`` request header (never in a URL),
never following a redirect. A route passes only when its status is the listed one, a redirect's
Location is the listed target, and the response carries the route's marker:

* a page: its ``<title>`` text, exactly as ``base.html`` renders it;
* a retired URL: the redirect target, the same one the local app sends;
* a fragment: HTML that is not a whole page (no ``<html``);
* an export, the stylesheet and a font: the right content type (``font/woff2`` for the font);
* ``/health``: the publish id AND the cache SHA-256 of the newest SUCCEEDED push-log record (or of
  ``--publish-id``'s record).

So a Vercel sign-in page or an error page that answers 200 fails. A committed test runs the same
list against the local app over a fixture cache, so "passes on the host" means "answers exactly as
the laptop does".

Refusal mode (``--refusals``, SPEC R3). It sends NO credential -- or, with ``--wrong-bypass``, a
freshly generated random bypass value -- to the production URL, a per-deployment URL, ``/health``
and one ``/static`` file, each through a fresh client so no cookie carries over. It records each
real status line and an allowlist of headers (location, content-type, cache-control, server,
x-vercel-id), never ``Set-Cookie``. A URL passes only on 401/403, or a redirect to vercel.com, with
none of the site's content in the body.

OUTPUT CONTRACT
---------------
Normal mode: one ``ROUTE= <path> status=<code> marker=<yes|no>`` line per route.
Refusal mode: one ``REFUSAL= <url> status=<code> content=<none|LEAKED>`` line per URL, followed by
its status line and allowlisted headers, each indented.
Both end with ``FETCH_HOSTED= pass <n>`` or ``FETCH_HOSTED= fail <k> of <n>``. A setup problem prints
``FETCH_HOSTED= refused: <reason>``. A failure is named by a status or an exception CLASS only; the
bypass value is never printed or recorded.

EXIT CODES
----------
0 every route (or every refusal) passed; 1 anything else.

Usage:
    uv run python -m scripts.fetch_hosted_routes [--game-id ID] [--publish-id ID] [--record FILE]
    uv run python -m scripts.fetch_hosted_routes --refusals --deployment-url URL [--wrong-bypass]
                                                 [--record FILE]

Options:
    --game-id ID         the game page to fetch (default: the first game id in the local cache)
    --publish-id ID      the publish id /health must serve (default: the newest succeeded push)
    --record FILE        also write every printed line to FILE
    --refusals           check Vercel's refusals instead of the routes
    --wrong-bypass       with --refusals: send a random bypass value instead of none
    --deployment-url URL with --refusals: the per-deployment URL to check (required there)

Do not run it within two minutes of 18:00 ET: a game locking mid-run changes a page between two
requests. No module under ``api/`` may import this CLI (UIAP-01): a page request never fetches.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import re
import secrets
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

import httpx

from pipeline import site_push


class Route(NamedTuple):
    """One route of the fixed list: what the local app answers, so what the host must answer.

    Attributes:
        path: The request path, with no query string.
        status: The expected HTTP status.
        location: The expected redirect Location, or None for a route that does not redirect.
        marker: What the response must carry: a page's title text, a redirect's target, a content
            type, or a description for ``/health`` and the fragments (checked by :func:`check_route`).
    """

    path: str
    status: int
    location: str | None
    marker: str


#: The headers refusal mode records. Everything else -- Set-Cookie above all -- is left out.
RECORDED_HEADERS = (
    "location",
    "content-type",
    "cache-control",
    "server",
    "x-vercel-id",
)
#: Text that only the site itself serves: the wordmark, /health's JSON and the font rules.
SITE_MARKERS = ("NFL<span", '"publish_id"', "@font-face")
REFUSAL_STATUSES = frozenset({401, 403})

_TITLE_SUFFIX = " | NFL Predict"
_TITLE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_FRAGMENT_MARKER = "HTML without <html"
_HEALTH_MARKER = "the expected publish_id and cache_sha256"
_GAME_TITLE_MARKER = " @ "


def routes(game_id: str) -> list[Route]:
    """The fixed route list (SPEC R4): every route the local app serves, as it answers locally."""

    def page(path: str, title: str) -> Route:
        return Route(path, 200, None, f"{title}{_TITLE_SUFFIX}")

    def moved(path: str, target: str) -> Route:
        return Route(path, 301, target, target)

    def typed(path: str, content_type: str) -> Route:
        return Route(path, 200, None, content_type)

    return [
        page("/", "This Week's Predictions"),
        page("/bets", "Weekly Bet List"),
        page("/season", "Season Tracking"),
        page("/track-record", "Track Record"),
        page("/how-it-works", "How It Works"),
        Route(f"/games/{game_id}", 200, None, _GAME_TITLE_MARKER),
        page("/games/NO_SUCH_GAME", "Game Not Found"),
        moved("/performance", "/track-record#seasons"),
        moved("/backtest", "/track-record"),
        moved("/insights", "/how-it-works"),
        moved("/betting", "/track-record#betting-sim"),
        Route("/fragments/games", 200, None, _FRAGMENT_MARKER),
        Route("/fragments/performance", 200, None, _FRAGMENT_MARKER),
        Route("/fragments/betting", 200, None, _FRAGMENT_MARKER),
        Route("/fragments/season", 200, None, _FRAGMENT_MARKER),
        typed("/api/export/csv", "text/csv"),
        typed("/api/export/json", "application/json"),
        Route("/health", 200, None, _HEALTH_MARKER),
        typed("/static/css/tailwind-compiled.css", "text/css"),
        typed("/static/fonts/inter-latin-wght-normal.woff2", "font/woff2"),
    ]


def _content_type(response: httpx.Response) -> str:
    """The media type alone, lower case (no charset)."""
    return response.headers.get("content-type", "").split(";")[0].strip().lower()


def _has_marker(
    route: Route, response: httpx.Response, expected: tuple[str, str] | None
) -> bool:
    """Whether *response*'s headers or body carry *route*'s marker."""
    if route.location is not None:
        return response.headers.get("location") == route.marker
    if route.path == "/health":
        if expected is None:
            return False
        try:
            body = response.json()
        except ValueError:
            return False
        return (
            isinstance(body, dict)
            and body.get("publish_id") == expected[0]
            and body.get("cache_sha256") == expected[1]
        )
    if route.marker == _FRAGMENT_MARKER:
        return (
            _content_type(response) == "text/html"
            and bool(response.text.strip())
            and "<html" not in response.text.lower()
        )
    if route.path.startswith(("/api/export/", "/static/")):
        if _content_type(response) != route.marker:
            return False
        if route.marker == "application/json":
            try:
                response.json()
            except ValueError:
                return False
        return True
    title = _TITLE.search(response.text)
    if title is None:
        return False
    if route.marker == _GAME_TITLE_MARKER:
        return _GAME_TITLE_MARKER in title.group(1) and title.group(1).endswith(
            _TITLE_SUFFIX
        )
    return title.group(1).strip() == route.marker


def check_route(
    route: Route, response: httpx.Response, expected: tuple[str, str] | None = None
) -> bool:
    """Whether *response* answers *route* as the local app does: status, Location and marker.

    Args:
        route: The route that was fetched.
        response: Its response (redirects not followed).
        expected: For ``/health`` only: the ``(publish_id, cache_sha256)`` it must serve.
    """
    if response.status_code != route.status:
        return False
    if response.headers.get("location") != route.location:
        return False
    return _has_marker(route, response, expected)


class _Output:
    """Print lines as they come and, with ``--record``, write the same lines to a file at the end."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, line: str) -> None:
        print(line)
        self.lines.append(line)

    def save(self, record: Path | None) -> None:
        if record is not None:
            record.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _first_game_id() -> str:
    """The first game id in the local cache's predictions table, read through the read-only opener."""
    from api.cache_reader import open_cache_readonly

    conn = open_cache_readonly(site_push.DEFAULT_CACHE_PATH)
    try:
        row = conn.execute(
            "SELECT game_id FROM predictions ORDER BY game_id LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        msg = "the local cache holds no prediction"
        raise LookupError(msg)
    return str(row[0])


def _expected_health(publish_id: str | None) -> tuple[str, str] | None:
    """The ``(publish_id, cache_sha256)`` /health must serve, from the push log; None if unknown.

    Default: the newest succeeded record. With *publish_id*: the newest record carrying that id.
    """
    records = site_push.read_site_pushes()
    if publish_id is None:
        candidates = [
            r for r in records if r.get("outcome") == site_push.OUTCOME_SUCCEEDED
        ]
    else:
        candidates = [r for r in records if r.get("publish_id") == publish_id]
    if not candidates:
        return None
    newest = candidates[-1]
    found_id, digest = newest.get("publish_id"), newest.get("local_sha256")
    if not (isinstance(found_id, str) and isinstance(digest, str)):
        return None
    return found_id, digest


def _summary(output: _Output, failed: int, total: int) -> int:
    if failed:
        output.say(f"FETCH_HOSTED= fail {failed} of {total}")
        return 1
    output.say(f"FETCH_HOSTED= pass {total}")
    return 0


def _fetch_routes(args: argparse.Namespace, output: _Output) -> int:
    """Normal mode: every route from the production URL, the bypass in the header only."""
    try:
        push_secrets = site_push.LOAD_PUSH_SECRETS()
    except Exception as error:  # noqa: BLE001 - the class name only: a message can carry a value
        output.say(
            f"FETCH_HOSTED= refused: the push secrets could not be read ({type(error).__name__})"
        )
        return 1
    if push_secrets.bypass_secret is None:
        output.say(f"FETCH_HOSTED= refused: {site_push.REASON_MISSING_BYPASS_SECRET}")
        return 1
    try:
        game_id = args.game_id or _first_game_id()
        expected = _expected_health(args.publish_id)
    except Exception as error:  # noqa: BLE001 - name what could not be read, by class
        output.say(
            f"FETCH_HOSTED= refused: the local cache or push log could not be read "
            f"({type(error).__name__})"
        )
        return 1

    headers = {site_push.BYPASS_HEADER: push_secrets.bypass_secret.get_secret_value()}
    route_list = routes(game_id)
    failed = 0
    with site_push.CLIENT_FACTORY() as client:
        for route in route_list:
            try:
                response = client.get(
                    f"{site_push.PRODUCTION_URL}{route.path}", headers=headers
                )
            except httpx.TransportError as error:
                output.say(
                    f"ROUTE= {route.path} status=transport_{type(error).__name__} marker=no"
                )
                failed += 1
                continue
            passed = check_route(route, response, expected)
            failed += not passed
            output.say(
                f"ROUTE= {route.path} status={response.status_code} "
                f"marker={'yes' if passed else 'no'}"
            )
    return _summary(output, failed, len(route_list))


def _is_refusal(response: httpx.Response) -> bool:
    """A 401/403, or a redirect whose Location host is vercel.com or a subdomain of it."""
    if response.status_code in REFUSAL_STATUSES:
        return True
    if 300 <= response.status_code < 400:
        host = (urlsplit(response.headers.get("location", "")).hostname or "").lower()
        return host == "vercel.com" or host.endswith(".vercel.com")
    return False


def _check_refusals(args: argparse.Namespace, output: _Output) -> int:
    """Refusal mode: no credential (or a random wrong bypass), and nothing of the site comes back."""
    if not args.deployment_url:
        output.say("FETCH_HOSTED= refused: --refusals needs --deployment-url")
        return 1
    headers = (
        {site_push.BYPASS_HEADER: secrets.token_urlsafe(32)}
        if args.wrong_bypass
        else {}
    )
    production = site_push.PRODUCTION_URL
    urls = [
        f"{production}/",
        f"{args.deployment_url.rstrip('/')}/",
        f"{production}/health",
        f"{production}/static/css/tailwind-compiled.css",
    ]
    failed = 0
    for url in urls:
        # A fresh client per URL: a cookie one refusal sets can never ride on the next request.
        with site_push.CLIENT_FACTORY() as client:
            try:
                response = client.get(url, headers=headers)
            except httpx.TransportError as error:
                output.say(
                    f"REFUSAL= {url} status=transport_{type(error).__name__} content=none"
                )
                failed += 1
                continue
        leaked = any(marker in response.text for marker in SITE_MARKERS)
        failed += leaked or not _is_refusal(response)
        output.say(
            f"REFUSAL= {url} status={response.status_code} "
            f"content={'LEAKED' if leaked else 'none'}"
        )
        output.say(
            f"  status-line: {response.http_version} {response.status_code} "
            f"{response.reason_phrase}"
        )
        for name in RECORDED_HEADERS:
            if name in response.headers:
                output.say(f"  {name}: {response.headers[name]}")
    return _summary(output, failed, len(urls))


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.fetch_hosted_routes",
        description=(
            "Fetch every route of the hosted website through the protection bypass header and "
            "check each answers as the laptop does; or, with --refusals, record that Vercel "
            "refuses a visitor with no credential. Exits 0 only when every check passes. Do not "
            "run it within two minutes of 18:00 ET."
        ),
    )
    parser.add_argument(
        "--game-id",
        help="the game page to fetch (default: the first in the local cache)",
    )
    parser.add_argument(
        "--publish-id",
        help="the publish id /health must serve (default: the newest succeeded push)",
    )
    parser.add_argument(
        "--record", type=Path, help="also write every printed line to this file"
    )
    parser.add_argument(
        "--refusals",
        action="store_true",
        help="check Vercel's refusals (no credential) instead of the routes",
    )
    parser.add_argument(
        "--wrong-bypass",
        action="store_true",
        help="with --refusals: send a freshly generated random bypass value",
    )
    parser.add_argument(
        "--deployment-url",
        help="with --refusals: the per-deployment URL to check (required there)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the route check (or the refusal check); 0 only when every check passed."""
    args = build_parser().parse_args(argv)
    output = _Output()
    try:
        if args.refusals:
            return _check_refusals(args, output)
        return _fetch_routes(args, output)
    finally:
        output.save(args.record)


if __name__ == "__main__":
    raise SystemExit(main())
