"""Routes that had no UI now have one (Genor 2026-10-27: every API ships with its UI).

Each page carries the control and its script calls the real route; the route answers what the script reads.
"""
from __future__ import annotations

from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"


@pytest.mark.parametrize(("template", "control", "call"), [
    ("models_index.html", 'id="models-rescan"', "api.post('/api/models/refresh')"),
    ("brain.html", 'id="reembed-btn"', "'/brain/rebuild-vectors', { embedder: embedder }"),
    ("data_prep.html", 'id="dp-dedupe"', "rqBase() + '/dedupe'"),
    ("data_prep.html", 'id="dp-audit-run"', "rqBase() + '/audit'"),
])
def test_page_has_the_control_and_calls_the_route(template: str, control: str, call: str) -> None:
    html = (TEMPLATES / template).read_text(encoding="utf-8")
    assert control in html and call in html


def test_the_routes_exist_with_the_methods_the_pages_use() -> None:
    from finetune_studio.webui.app import app

    paths = app.openapi()["paths"]
    assert "post" in paths["/api/models/refresh"]
    assert "post" in paths["/api/projects/{pid}/brain/rebuild-vectors"]
    assert "post" in paths["/api/projects/{pid}/data-prep/dedupe"]
    assert "get" in paths["/api/projects/{pid}/data-prep/audit"]


def test_re_embed_is_disabled_with_the_other_build_buttons() -> None:
    html = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    assert "['quick-index-btn', 'build-btn', 'rebuild-btn', 'reembed-btn']" in html


def test_dedupe_counts_first_and_rejects_only_after_a_yes() -> None:
    html = (TEMPLATES / "data_prep.html").read_text(encoding="utf-8")
    body = html.split("async function dpDedupePending()", 1)[1].split("\n}\n", 1)[0]
    assert body.index("dry_run: true") < body.index("fts.confirm(") < body.index("dry_run: false")


def test_models_rescan_answers_the_count_the_page_shows(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio.webui.app import app
    from finetune_studio.webui.routes import models

    monkeypatch.setattr(models, "refresh_model_registry", lambda: 3)
    assert TestClient(app).post("/api/models/refresh").json() == {"count": 3}


def test_dedupe_dry_run_reports_and_apply_rejects(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio import db
    from finetune_studio.data.fs import qa as qa_fs
    from finetune_studio.webui.app import app
    from tests.knowledge_helpers import add_source

    pid = db.create_project(name="dedupe-ui", description="")["id"]
    sid = add_source(pid, "policy.txt", "Mira owns the travel policy and reviews it every Friday.").id
    for i, q in enumerate(("Who owns the travel policy?", "Who owns the travel policy ?")):
        qa_fs.write_qa_pair(pid, {"id": f"q{i}", "source_id": sid, "chunk_idx": 1, "question": q,
                                  "answer": "Mira owns the travel policy.", "status": "pending"})
    client = TestClient(app)
    url = f"/api/projects/{pid}/data-prep/dedupe"
    probe = client.post(url, json={"dry_run": True}).json()
    assert probe["rejected"] == 1 and probe["of_pending"] == 2
    assert {p["status"] for p in qa_fs.list_qa_pairs(pid)} == {"pending"}          # a dry run changes nothing
    assert client.post(url, json={"dry_run": False}).json()["rejected"] == 1
    assert sorted(p["status"] for p in qa_fs.list_qa_pairs(pid)) == ["pending", "rejected"]


def test_audit_answers_the_fields_the_card_reads() -> None:
    from fastapi.testclient import TestClient

    from finetune_studio import db
    from finetune_studio.webui.app import app

    pid = db.create_project(name="audit-ui", description="")["id"]
    d = TestClient(app).get(f"/api/projects/{pid}/data-prep/audit").json()
    for key in ("source_count", "passed", "status", "sources"):
        assert key in d["sources"]
    for key in ("approved_pair_count", "expected_export_count", "errors", "source_coverage", "status"):
        assert key in d["dataset"]
