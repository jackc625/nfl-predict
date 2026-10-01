"""The self-hosted fonts are served with a font MIME type (redesign Task 1).

Windows' MIME registry has no entry for .woff2, so without the registration in api/main.py
Starlette answers text/plain and a browser honouring nosniff refuses the font.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_the_self_hosted_fonts_are_served_as_woff2(test_client: TestClient) -> None:
    response = test_client.get("/static/fonts/barlow-condensed-latin-800-italic.woff2")
    assert response.status_code == 200
    assert response.headers["content-type"] == "font/woff2"
    assert response.content[:4] == b"wOF2"
