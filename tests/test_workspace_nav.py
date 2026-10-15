"""Model vs RAG workspace subnav labels and links."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPA_JS = _ROOT / "src" / "finetune_studio" / "webui" / "static" / "js" / "spa.js"
_BASE = _ROOT / "src" / "finetune_studio" / "webui" / "templates" / "base.html"


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Workspace Nav"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_spa_syncs_workspace_subnav_outside_content() -> None:
    """SPA must create/update #workspace-subnav (base shell), not only #content.

    Regression: /projects → /projects/{pid} left Model/RAG subnav missing
    because spa.js only swapped #content + breadcrumb innerHTML.
    """
    src = _SPA_JS.read_text(encoding="utf-8")
    assert "workspace-subnav" in src
    assert "workspacePresent" in src
    assert "workspaceOuter" in src
    assert "syncShellNav" in src
    assert "breadcrumbOuter" in src
    # Subnav lives in base.html outside #content (same shell as breadcrumb).
    base = _BASE.read_text(encoding="utf-8")
    content_pos = base.find('id="content"')
    ws_pos = base.find('id="workspace-subnav"')
    assert content_pos != -1 and ws_pos != -1
    assert ws_pos < content_pos


def test_projects_list_has_no_workspace_subnav(client) -> None:
    """Non-project pages must omit subnav so SPA can detect enter/leave."""
    r = client.get("/projects")
    assert r.status_code == 200, r.text
    assert 'id="workspace-subnav"' not in r.text
    assert 'id="project-breadcrumb"' not in r.text


def _bar(body: str) -> str:
    return body.split('id="workspace-subnav"', 1)[1].split("</nav>", 1)[0]


def test_every_project_page_renders_the_one_flow_bar(client) -> None:
    """Genor 2026-10-14: one flow for every project page — 1 Files & Knowledge, 2 train or 2 Smart Brain, 3 test,
    4 use it — with the current page marked (no model/RAG workspace switch)."""
    pid = _project(client)
    for page, tab in (("", "overview"), ("/brain", "brain"), ("/training", "training"), ("/data-prep", "data-prep"),
                      ("/data", "data"), ("/testing", "testing"), ("/export", "export")):
        r = client.get(f"/projects/{pid}{page}")
        assert r.status_code == 200, (page, r.text[:200])
        assert 'id="project-breadcrumb"' in r.text
        bar = _bar(r.text)
        for label in ("1 Files &amp; Knowledge", ">files<", ">pairs<", ">2 train<", ">2 Smart Brain<", ">3 test<",
                      ">4 use it<", ">home<"):
            assert label in bar, (page, label)
        assert "ws-switch" not in bar
        assert "here" in bar.split(f'data-tab="{tab}"', 1)[1].split(">", 1)[0], page
