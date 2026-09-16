"""Model vs RAG workspace subnav labels and links."""
from __future__ import annotations


def _project(client) -> str:
    r = client.post("/api/projects", json={"name": "Workspace Nav"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def test_rag_page_shows_rag_workspace_subnav(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/rag")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="workspace-subnav"' in body
    assert "Model workspace" in body
    assert "RAG workspace" in body
    assert f'href="/projects/{pid}/rag"' in body
    assert f'href="/projects/{pid}/rag#rag-ingest"' in body
    assert f'href="/projects/{pid}/rag#rag-search"' in body
    assert f'href="/projects/{pid}/rag#rag-tests"' in body
    assert ">dashboard<" in body
    assert ">ingest<" in body
    assert ">search<" in body
    assert ">tests<" in body
    assert 'id="rag-ingest"' in body
    assert 'id="rag-search"' in body
    assert 'id="rag-tests"' in body


def test_training_page_shows_model_workspace_subnav(client) -> None:
    pid = _project(client)
    r = client.get(f"/projects/{pid}/training")
    assert r.status_code == 200, r.text
    body = r.text
    assert 'id="workspace-subnav"' in body
    assert "Model workspace" in body
    assert "RAG workspace" in body
    assert f'href="/projects/{pid}/training"' in body
    assert f'href="/projects/{pid}/testing"' in body
    assert ">dashboard<" in body
    assert ">training<" in body
    assert ">testing<" in body
    # RAG section anchors should not appear as the active workspace links set
    assert f'href="/projects/{pid}/rag#rag-ingest"' not in body


def test_data_prep_page_has_no_workspace_subnav(client) -> None:
    """Unrelated tabs keep the existing breadcrumb only."""
    pid = _project(client)
    r = client.get(f"/projects/{pid}/data-prep")
    assert r.status_code == 200, r.text
    assert 'id="workspace-subnav"' not in r.text
