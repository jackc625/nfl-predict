"""The hosted-site live check (Phase 36, SPEC R3 / R4 / R12(a)).

WHAT THESE TESTS PROVE
----------------------
``scripts.fetch_hosted_routes`` is the tool Plans 36-09 and 36-10 run against the real site. Here it
never reaches Vercel: every "host" is an ``httpx.MockTransport`` handed in through
``pipeline.site_push.CLIENT_FACTORY`` (the production client with only its transport swapped), and
``tests/conftest.py`` refuses any real Vercel request and quarantines the push log.

1. The script's fixed route list IS the local app's behaviour: each route's expected status, redirect
   Location and marker are checked against the real app answering over the conftest fixture cache.
   So a hosted pass means the same status (and for a redirect the same target) as locally.
2. A host that answers every request with a 200 sign-in page fails every route, and /health passes
   only with the newest SUCCEEDED push's publish id and digest. The bypass value travels only in
   the ``x-vercel-protection-bypass`` header, never in a URL, the output or the record file.
3. Refusal mode passes only on refusals, records an allowlist of headers and never ``Set-Cookie``,
   and with ``--wrong-bypass`` sends a value that is not the configured one.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pipeline import site_push
from scripts import fetch_hosted_routes as fetch
from tests.api.conftest import (  # noqa: F401 - fixtures used by name below
    _isolate_app_connection_state,
    _isolate_data_service_cache,
    test_client,
    test_db,
)
from tests.unit.test_site_push import DUMMY_BYPASS, dummy_secrets

FIXTURE_GAME_ID = "2024_W01_BUF@KC"
DEPLOYMENT_URL = "https://nfl-predict-abc123-team.vercel.app"
SIGN_IN_PAGE = (
    "<!DOCTYPE html><html><head><title>Login - Vercel</title></head>"
    "<body>Log in to Vercel to continue</body></html>"
)
SITE_PAGE = (
    "<!DOCTYPE html><html><head><title>This Week's Predictions | NFL Predict</title></head>"
    '<body><a href="/" class="wordmark">NFL<span class="text-accent">/</span>Predict</a>'
    "</body></html>"
)
COOKIE_VALUE = "_vercel_sso_nonce=abcdef0123456789"

Handler = Callable[[httpx.Request], httpx.Response]


def _serve(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> list[httpx.Request]:
    """Route the script's client to *handler*; return the list of requests it receives."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        site_push,
        "CLIENT_FACTORY",
        lambda: site_push._default_client(transport=httpx.MockTransport(recording)),
    )
    return seen


def _write_push_log(records: list[dict[str, str]]) -> None:
    """Write push-log records (the conftest points the log at a per-test path)."""
    log = site_push.SITE_PUSH_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({name: record.get(name) for name in site_push.SITE_PUSH_FIELDS})
        for record in records
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "route",
    [route for route in fetch.routes(FIXTURE_GAME_ID) if route.path != "/health"],
    ids=lambda route: route.path,
)
def test_the_route_list_is_the_local_app_s_behavior(
    test_client: TestClient,  # noqa: F811 - the imported conftest fixture
    route: fetch.Route,
) -> None:
    response = test_client.get(route.path, follow_redirects=False)

    assert response.status_code == route.status
    assert response.headers.get("location") == route.location
    assert fetch.check_route(route, response), (
        f"{route.path}: the local app's answer does not carry the marker {route.marker!r}"
    )


def test_a_sign_in_page_answering_200_fails_and_the_bypass_stays_in_the_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(site_push, "LOAD_PUSH_SECRETS", dummy_secrets)
    older, newest, failed = (
        ("a" * 32, "sha-a"),
        ("b" * 32, "sha-b"),
        ("c" * 32, "sha-c"),
    )
    _write_push_log(
        [
            {"publish_id": older[0], "local_sha256": older[1], "outcome": "succeeded"},
            {
                "publish_id": newest[0],
                "local_sha256": newest[1],
                "outcome": "succeeded",
            },
            {"publish_id": failed[0], "local_sha256": failed[1], "outcome": "failed"},
        ]
    )
    health_answer = older

    def sign_in_host(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            publish_id, digest = health_answer
            return httpx.Response(
                200, json={"publish_id": publish_id, "cache_sha256": digest}
            )
        return httpx.Response(
            200, text=SIGN_IN_PAGE, headers={"content-type": "text/html"}
        )

    seen = _serve(monkeypatch, sign_in_host)
    record = tmp_path / "fetch.txt"

    code = fetch.main(["--game-id", FIXTURE_GAME_ID, "--record", str(record)])
    out = capsys.readouterr().out

    routes = fetch.routes(FIXTURE_GAME_ID)
    assert code == 1
    for route in routes:
        assert f"ROUTE= {route.path} status=200 marker=no" in out
    assert f"FETCH_HOSTED= fail {len(routes)} of {len(routes)}" in out
    assert record.read_text(encoding="utf-8").splitlines() == out.splitlines()
    assert len(seen) == len(routes)
    for request in seen:
        assert str(request.url).startswith(site_push.PRODUCTION_URL)
        assert request.url.query == b""
        assert request.headers[site_push.BYPASS_HEADER] == DUMMY_BYPASS
        assert [v for v in request.headers.values() if v == DUMMY_BYPASS] == [
            DUMMY_BYPASS
        ]
        assert DUMMY_BYPASS not in str(request.url)
    assert DUMMY_BYPASS not in out
    assert DUMMY_BYPASS not in record.read_text(encoding="utf-8")

    # The newest SUCCEEDED push (not the newest record) is the one /health must serve.
    health_answer = newest
    fetch.main(["--game-id", FIXTURE_GAME_ID])
    assert "ROUTE= /health status=200 marker=yes" in capsys.readouterr().out


def test_refusal_mode_passes_only_on_refusals_and_records_no_cookie(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(site_push, "LOAD_PUSH_SECRETS", dummy_secrets)
    leaking: set[str] = set()

    def refusing_host(request: httpx.Request) -> httpx.Response:
        headers = {
            "set-cookie": COOKIE_VALUE,
            "server": "Vercel",
            "x-vercel-id": "iad1::abcde-123",
            "cache-control": "no-store",
        }
        if str(request.url) in leaking:
            return httpx.Response(200, text=SITE_PAGE, headers=headers)
        if request.url.host == httpx.URL(DEPLOYMENT_URL).host:
            headers["location"] = "https://vercel.com/sso-api?url=x&nonce=y"
            return httpx.Response(302, headers=headers)
        return httpx.Response(401, text=SIGN_IN_PAGE, headers=headers)

    seen = _serve(monkeypatch, refusing_host)
    record = tmp_path / "refusals.txt"

    code = fetch.main(
        ["--refusals", "--deployment-url", DEPLOYMENT_URL, "--record", str(record)]
    )
    out = capsys.readouterr().out
    text = out + record.read_text(encoding="utf-8")

    assert code == 0
    refusals = [line for line in out.splitlines() if line.startswith("REFUSAL= ")]
    assert len(refusals) == 4
    assert all(line.endswith("content=none") for line in refusals)
    assert f"REFUSAL= {DEPLOYMENT_URL}/ status=302 content=none" in out
    assert f"REFUSAL= {site_push.PRODUCTION_URL}/health status=401 content=none" in out
    assert "x-vercel-id: iad1::abcde-123" in out
    assert "location: https://vercel.com/sso-api" in out
    assert "FETCH_HOSTED= pass 4" in out
    assert "set-cookie" not in text.lower()
    assert COOKIE_VALUE not in text
    for request in seen:
        assert site_push.BYPASS_HEADER not in request.headers
        assert "cookie" not in request.headers

    # One URL answering 200 with the site's own page is a leak, and the run fails.
    leaking.add(f"{site_push.PRODUCTION_URL}/")
    code = fetch.main(["--refusals", "--deployment-url", DEPLOYMENT_URL])
    out = capsys.readouterr().out
    assert code == 1
    assert f"REFUSAL= {site_push.PRODUCTION_URL}/ status=200 content=LEAKED" in out
    assert "FETCH_HOSTED= fail 1 of 4" in out

    # A wrong bypass value: every request carries one, and never the configured value.
    leaking.clear()
    seen.clear()
    code = fetch.main(
        ["--refusals", "--wrong-bypass", "--deployment-url", DEPLOYMENT_URL]
    )
    assert code == 0
    assert len(seen) == 4
    for request in seen:
        sent = request.headers.get(site_push.BYPASS_HEADER)
        assert sent and sent != DUMMY_BYPASS
    assert DUMMY_BYPASS not in capsys.readouterr().out
