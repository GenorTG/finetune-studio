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


def test_project_overview_shows_model_workspace_subnav(client) -> None:
    """Direct URL render includes Model workspace subnav (full-page load)."""
    pid = _project(client)
    r = client.get(f"/projects/{pid}")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="workspace-subnav"' in body
    assert 'id="project-breadcrumb"' in body
    assert "Train a model" in body
    assert "Search my files" in body
    assert f'href="/projects/{pid}/rag"' in body


def test_rag_page_shows_rag_workspace_subnav(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/rag")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="workspace-subnav"' in body
    assert "Train a model" in body
    assert "Search my files" in body
    assert f'href="/projects/{pid}/rag"' in body
    assert f'href="/projects/{pid}/rag#rag-ingest"' in body
    assert f'href="/projects/{pid}/rag#rag-search"' in body
    assert f'href="/projects/{pid}/chat"' in body
    assert ">home<" in body
    assert ">2 build<" in body
    assert ">3 search<" in body
    assert ">4 chat<" in body
    assert 'id="rag-ingest"' in body
    assert 'id="rag-search"' in body
    assert 'id="rag-dashboard"' in body


def test_training_page_shows_model_workspace_subnav(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/training")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="workspace-subnav"' in body
    assert "Train a model" in body
    assert "Search my files" in body
    assert f'href="/projects/{pid}/training"' in body
    assert f'href="/projects/{pid}/testing"' in body
    assert ">home<" in body
    assert ">3 train<" in body
    assert ">4 test<" in body
    # RAG section anchors should not appear as the active workspace links set
    assert f'href="/projects/{pid}/rag#rag-ingest"' not in body


def test_data_prep_page_has_workspace_subnav(client) -> None:
    """Flow-scoped nav contract (2026-09-21): the pairs page declares
    workspace=model, so the subnav MUST render with the model flow active."""
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    assert 'id="workspace-subnav"' in r.text
    assert "ws-switch active" in r.text
