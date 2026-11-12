"""First-run flow without dead ends (beta walkthrough 2026-11-12): New project opens the project it made, the Package
page's "Build the package" link lands ON the export panel, and an unbuilt Smart Brain page logs no 404."""
from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "static" / "js"


def test_new_project_form_opens_the_new_project(client) -> None:
    page = client.get("/projects").text
    assert 'id="np-form" data-api="/api/projects" data-redirect="/projects/{id}"' in page
    app_js = (STATIC / "app.js").read_text()
    assert "form.dataset.redirect && d && d.id" in app_js and 'replace("{id}", encodeURIComponent(d.id))' in app_js
    made = client.post("/api/projects", json={"name": "agent-test-dead-ends"})
    assert made.status_code == 200 and made.json()["id"]      # what the redirect fills in


def test_spa_navigation_keeps_the_links_hash_target() -> None:
    spa = (STATIC / "spa.js").read_text()
    top = spa.index('window.scrollTo({ top: 0, behavior: "instant" });')
    hashed = spa.index("target.scrollIntoView", top)
    assert top < hashed and 'target.tagName === "DETAILS"' in spa[top:hashed]


def test_package_link_points_at_the_export_panel(client) -> None:
    pid = client.post("/api/projects", json={"name": "agent-test-dead-ends-2"}).json()["id"]
    assert f'href="/projects/{pid}/brain#brain-export-panel"' in client.get(f"/projects/{pid}/export").text
    assert 'id="brain-export-panel"' in client.get(f"/projects/{pid}/brain").text


def test_package_documents_before_the_first_build_is_empty_not_404(client) -> None:
    pid = client.post("/api/projects", json={"name": "agent-test-dead-ends-3"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/brain/package-documents")
    assert r.status_code == 200
    assert r.json() == {"total": 0, "folders": [], "tags": [], "documents": [], "indexed": False}
    assert client.get("/api/projects/nope/brain/package-documents").status_code == 404
