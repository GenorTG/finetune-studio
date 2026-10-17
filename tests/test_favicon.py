"""/favicon.ico (asked for by browsers on their own) redirects to the SVG icon instead of a 404 (walkthrough 2026-10-17)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_favicon_ico_redirects_to_the_svg(client: TestClient) -> None:
    r = client.get("/favicon.ico", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/static/favicon.svg"
