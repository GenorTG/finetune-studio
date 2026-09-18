"""Header session-bar overflow affordances + reachable project nav links."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient
from jinja2 import Environment, FileSystemLoader, select_autoescape

_ROOT = Path(__file__).resolve().parents[1]
_CSS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "css" / "app.css"
_BASE = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "base.html"
_NAV_JS = (
    _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "nav_overflow.js"
)
_SPA_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "spa.js"
_TEMPLATES = _ROOT / "src" / "finetune_studio" / "webui" / "templates"

# Project-scoped tabs that were clipped behind overflow:hidden with no affordance.
_PROJECT_TABS = (
    "overview",
    "data",
    "data-prep",
    "rag",
    "training",
    "testing",
    "benchmarks",
    "project-chat",
    "export",
)
_GLOBAL_TABS = ("dashboard", "projects", "hf", "inference", "settings")


def _render_base(*, pid: str | None = None) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    child = env.from_string(
        '{% extends "base.html" %}'
        "{% block nav_testing %}active{% endblock %}"
        "{% block content %}nav-probe{% endblock %}"
    )
    kwargs: dict = {"app_version": "0.0-test", "request": None}
    if pid:
        kwargs["pid"] = pid
        kwargs["project"] = {"name": "nav-probe", "id": pid}
    return child.render(**kwargs)


def _href_for_tab(html: str, tab: str) -> str:
    marker = f'data-tab="{tab}"'
    idx = html.find(marker)
    assert idx != -1, f"missing data-tab={tab}"
    open_a = html.rfind("<a ", 0, idx)
    assert open_a != -1
    snippet = html[open_a : idx + len(marker) + 40]
    m = re.search(r'href="([^"]+)"', snippet)
    assert m, f"no href for data-tab={tab} in {snippet!r}"
    return m.group(1)


def test_header_nav_overflow_affordance_markup() -> None:
    """Narrow/desktop markup must expose scroll + [nav] menu affordances."""
    base = _BASE.read_text(encoding="utf-8")
    assert 'id="sb-nav-shell"' in base
    assert 'id="sb-tabs"' in base
    assert "sb-nav-scroll-prev" in base
    assert "sb-nav-scroll-next" in base
    assert 'id="sb-nav-more"' in base
    assert "[nav]" in base
    assert 'id="sb-nav-more-list"' in base
    assert "nav_overflow.js" in base
    assert "☰" in base
    assert "app.css?v=29" in base
    assert "{{ release_channel }}" in base
    assert "· {{ release_channel }}</title>" in base
    assert "spa.js?v=15" in base


def test_header_nav_css_overflow_contract() -> None:
    """CSS must keep tabs scrollable inside the header without page blowout."""
    css = _CSS.read_text(encoding="utf-8")
    assert ".sb-nav-shell {" in css
    assert ".sb-nav-more {" in css
    assert ".sb-nav-scroll {" in css
    # Scrollport (not the page) owns horizontal overflow.
    assert "overflow-x: auto" in css
    assert "scrollbar-width: auto" in css
    assert ".sb-tabs::-webkit-scrollbar { height: 8px; }" in css
    # Session bar still contains width; document clip stays.
    assert "overflow-x: clip" in css
    assert ".session-bar" in css
    bar = css.split(".session-bar {", 1)[1].split("}", 1)[0]
    assert "overflow-x: hidden" in bar
    assert "max-width: 100%" in bar
    # Gutters moved to shell (padding, not margin+width:100%).
    assert "padding: 10px 490px 0 140px" in css
    assert "margin-right: min(490px, 42vw)" not in css
    # Narrow: shell stacks right chrome; tabs keep a full-width scrollport.
    parts = css.split("@media (max-width: 700px)")
    assert len(parts) >= 2
    narrow = parts[-1].split("@media", 1)[0]
    assert ".sb-nav-shell" in narrow
    assert "padding-right: 8px" in narrow
    assert "position: static" in narrow
    # Mid-width gutter still shrinks via shell.
    assert "padding-right: min(490px, 42vw)" in css
    assert "padding-right: min(160px, 28vw)" in css
    desktop = css.split("@media (min-width: 701px)", 1)[1].split("@media", 1)[0]
    assert ".sb-tabs" in desktop and "overflow: visible" in desktop
    assert ".sb-nav-more { display: none !important; }" in desktop
    mobile_start = css.index("@media (max-width: 700px) {\n  .sb-nav-shell {\n    min-height: 48px;")
    mobile = css[mobile_start : mobile_start + 1200]
    assert ".sb-tabs { display: none; }" in mobile
    assert ".sb-nav-more-btn" in mobile


def test_header_nav_js_module_present() -> None:
    js = _NAV_JS.read_text(encoding="utf-8")
    assert "ftsNavOverflow" in js
    assert "scrollActiveIntoView" in js
    assert "updateScrollAffordance" in js
    assert "syncMoreMenu" in js
    spa = _SPA_JS.read_text(encoding="utf-8")
    assert "ftsNavOverflow" in spa
    assert "syncMoreActive" in spa


def test_project_nav_links_present_and_reachable() -> None:
    pid = "deadbeefdeadbeefdeadbeefdeadbeef"
    html = _render_base(pid=pid)
    # Strip + [nav] menu both expose every important route.
    for tab in _PROJECT_TABS + _GLOBAL_TABS:
        assert f'data-tab="{tab}"' in html
    # Critical clipped routes resolve to real project paths (not 404 stubs).
    assert _href_for_tab(html, "testing") == f"/projects/{pid}/testing"
    assert _href_for_tab(html, "benchmarks") == f"/projects/{pid}/benchmarks"
    assert _href_for_tab(html, "project-chat") == f"/projects/{pid}/chat"
    assert _href_for_tab(html, "export") == f"/projects/{pid}/export"
    assert _href_for_tab(html, "inference") == f"/projects/{pid}/chat"
    assert f"/projects/{pid}/inference" not in html
    # data-link preserved for SPA.
    assert html.count("data-link") >= 10
    # [nav] menu lists the same destinations.
    more = html.split('id="sb-nav-more-list"', 1)[1].split("</div>", 1)[0]
    for label in ("testing", "benchmarks", "chat", "export", "inference", "settings"):
        assert label in more


def test_global_nav_hides_project_tabs() -> None:
    html = _render_base(pid=None)
    assert 'data-tab="dashboard"' in html
    assert 'data-tab="projects"' in html
    assert 'data-tab="inference"' in html
    assert 'data-tab="testing"' not in html
    assert 'data-tab="benchmarks"' not in html
    assert _href_for_tab(html, "inference") == "/inference"


def test_active_route_state_on_testing_tab() -> None:
    pid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    html = _render_base(pid=pid)
    # Server-rendered active class on the testing tab.
    m = re.search(
        r'<a[^>]*class="sb-tab[^"]*active[^"]*"[^>]*data-tab="testing"',
        html,
    )
    if not m:
        m = re.search(
            r'<a[^>]*data-tab="testing"[^>]*class="sb-tab[^"]*active',
            html,
        )
    assert m, "testing tab must carry active class from nav_testing block"
    # Only one .sb-tab.active in the strip (not every project child).
    strip = html.split('id="sb-tabs"', 1)[1].split("</nav>", 1)[0]
    actives = re.findall(r'class="sb-tab[^"]*active[^"]*"', strip)
    assert len(actives) == 1


def test_project_pages_serve_nav_affordance(client: TestClient) -> None:
    r = client.post("/api/projects", json={"name": "HdrNav", "base_model": "x/t"})
    assert r.status_code in (200, 201), r.text
    pid = r.json()["id"]
    page = client.get(f"/projects/{pid}/testing")
    assert page.status_code == 200, page.text
    body = page.text
    assert 'id="sb-nav-shell"' in body
    assert 'id="sb-nav-more"' in body
    assert f"/projects/{pid}/testing" in body
    assert f"/projects/{pid}/benchmarks" in body
    assert f"/projects/{pid}/chat" in body
    assert f"/projects/{pid}/export" in body
    assert "[tools]" in body
    assert 'data-tab="testing"' in body
    # Active route preserved on live page.
    assert re.search(
        r'sb-tab[^"]*active[^"]*"[^>]*data-tab="testing"', body
    ) or re.search(r'data-tab="testing"[^>]*class="[^"]*active', body)


def test_css_cache_bust_header_nav() -> None:
    base = _BASE.read_text(encoding="utf-8")
    assert "app.css?v=29" in base
    assert "nav_overflow.js?v=1" in base
