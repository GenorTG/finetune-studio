"""Pages, styles, scripts and JSON go out gzipped; streams, downloads and small bodies do not; versioned static assets
are cached for a year, unversioned ones are revalidated."""
from __future__ import annotations

import gzip

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.testclient import TestClient

from finetune_studio.webui.compression import CompressionMiddleware

GZ = {"Accept-Encoding": "gzip"}
BIG = "<p>" + "the quick brown fox jumps over the lazy dog " * 200 + "</p>"


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CompressionMiddleware)

    @app.get("/page")
    def page():
        return HTMLResponse(BIG)

    @app.get("/tiny")
    def tiny():
        return HTMLResponse("<p>hi</p>")

    @app.get("/stream")
    def stream():
        def gen():
            for i in range(50):
                yield f"data: {BIG[:200]} {i}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/ndjson")
    def ndjson():
        return StreamingResponse((f'{{"n": {i}, "pad": "{BIG[:300]}"}}\n' for i in range(20)), media_type="application/json")

    @app.get("/download")
    def download():
        return Response(BIG.encode(), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="x.json"'})

    @app.get("/archive")
    def archive():
        return Response(gzip.compress(BIG.encode()), media_type="application/gzip")

    return app


def test_whole_text_responses_are_gzipped() -> None:
    c = TestClient(_app())
    r = c.get("/page", headers=GZ)
    assert r.headers.get("content-encoding") == "gzip" and r.text == BIG
    assert "accept-encoding" in r.headers.get("vary", "").lower()
    assert c.get("/page", headers={"Accept-Encoding": "identity"}).headers.get("content-encoding") is None
    assert c.get("/tiny", headers=GZ).headers.get("content-encoding") is None


def test_streams_downloads_and_archives_pass_through() -> None:
    c = TestClient(_app())
    for path in ("/stream", "/ndjson", "/download", "/archive"):
        r = c.get(path, headers=GZ)
        assert r.headers.get("content-encoding") is None, path
    assert c.get("/stream", headers=GZ).text.count("data:") == 50


def test_the_live_app_compresses_its_assets_and_caches_versioned_ones(client) -> None:
    css = client.get("/static/css/app.css?v=1", headers=GZ)
    assert css.status_code == 200 and css.headers.get("content-encoding") == "gzip"
    assert css.headers["cache-control"] == "public, max-age=31536000, immutable"
    bare = client.get("/static/css/app.css", headers=GZ)
    assert bare.headers["cache-control"] == "no-cache, must-revalidate"
    js = client.get("/static/js/app.js?v=1", headers=GZ)
    assert js.headers.get("content-encoding") == "gzip" and "immutable" in js.headers["cache-control"]
    page = client.get("/projects", headers=GZ)
    assert page.status_code == 200 and page.headers.get("content-encoding") == "gzip"


def test_every_page_links_its_css_and_js_with_a_version() -> None:
    """A year-long cache is only safe when a change changes the URL: every css/js the templates load carries ?v=."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src/finetune_studio/webui"
    refs = set()
    for f in [*(root / "templates").rglob("*.html"), *(root / "static/js").rglob("*.js")]:
        refs |= set(re.findall(r"/static/[\w./-]+\.(?:css|js)(?:\?[^\"' )]*)?", f.read_text(encoding="utf-8")))
    unversioned = sorted(r for r in refs if "?v=" not in r)
    assert not unversioned, unversioned
