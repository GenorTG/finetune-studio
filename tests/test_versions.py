"""Tests for project versioning: db layer, subset dataset builds, RAG coverage gate."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from finetune_studio import db


@pytest.fixture()
def project(temp_db):
    p = db.create_project("Versioning Test", description="t")
    return p["id"]


@pytest.fixture()
def client(temp_db):
    from finetune_studio.webui.app import app
    return TestClient(app)


def test_version_crud_and_monotonic_numbers(project):
    v1 = db.create_version(project, label="base", notes="first",
                           manifest={"base_model": "Qwen/Qwen3-4B"})
    v2 = db.create_version(project, label="v2", parent_version_id=v1["id"])
    assert v1["version_number"] == 1
    assert v2["version_number"] == 2
    assert v2["parent_version_id"] == v1["id"]
    vs = db.list_versions(project)
    assert [v["version_number"] for v in vs] == [2, 1]  # newest first
    latest = db.latest_version(project)
    assert latest["id"] == v2["id"]
    assert db.delete_version(v2["id"]) is True
    assert db.get_version(v2["id"]) is None


def test_version_manifest_key_filtering(project):
    v = db.create_version(project, manifest={
        "base_model": "x", "evil_key": "drop me", "datasets": [{"id": "d1"}]})
    m = json.loads(v["manifest_json"])
    assert m["base_model"] == "x"
    assert "evil_key" not in m


def test_version_lineage_walk(project):
    root = db.create_version(project, label="root")
    mid = db.create_version(project, label="mid", parent_version_id=root["id"])
    leaf = db.create_version(project, label="leaf", parent_version_id=mid["id"])
    chain = db.version_lineage(project, leaf["id"])
    assert [c["label"] for c in chain] == ["root", "mid", "leaf"]


def test_get_version_by_number(project):
    db.create_version(project, label="only")
    v = db.get_version_by_number(project, 1)
    assert v is not None and v["label"] == "only"
    assert db.get_version_by_number(project, 99) is None


def test_versions_routesListing_and_lineage(client, project):
    r = client.get(f"/api/projects/{project}/versions")
    assert r.status_code == 200
    assert r.json()["versions"] == []
    v = client.post(f"/api/projects/{project}/versions",
                    json={"label": "snap", "notes": ""}).json()
    assert v["version_number"] == 1
    r2 = client.get(f"/api/projects/{project}/versions/{v['id']}")
    body = r2.json()
    # manifest decoded + auto-filled defaults present (empty lists are fine)
    assert "manifest" in body
    lin = client.get(f"/api/projects/{project}/versions/{v['id']}/lineage")
    assert lin.status_code == 200
    assert lin.json()["lineage"][0]["id"] == v["id"]
    d = client.delete(f"/api/projects/{project}/versions/{v['id']}")
    assert d.json() == {"ok": True}


def test_save_version_404_unknown_project(client):
    r = client.post("/api/projects/nonexistent/versions", json={"label": "x"})
    assert r.status_code == 404


def test_save_version_rejects_foreign_parent(client, project):
    r = client.post(f"/api/projects/{project}/versions",
                    json={"label": "x", "parent_version_id": "deadbeef"})
    assert r.status_code == 404


def test_subset_dataset_build_route(client, project, temp_db, monkeypatch):
    """Subset export: select known sources -> registered dataset with rows."""
    import tempfile
    from pathlib import Path

    import finetune_studio.config as cfg
    import finetune_studio.data.fs.qa as qafs
    import finetune_studio.data.fs.files as fl_mod
    import finetune_studio.webui.routes.versions as vroutes
    # Point qa fs at a temp project dir
    tmp = Path(tempfile.mkdtemp(prefix="vers-subset-"))
    monkeypatch.setattr(qafs, "project_dir", lambda pid: tmp / pid)

    pid = project
    (tmp / pid / "qa" / "pairs").mkdir(parents=True)
    (tmp / pid / "qa" / "sources").mkdir(parents=True)
    for i, (sid, fn, chunks) in enumerate([
        ("srcAAA", "alpha.txt", 2), ("srcBBB", "beta.txt", 1), ("srcCCC", "gamma.txt", 1),
    ]):
        src = {"id": sid, "filename": fn, "sha256": f"sha{i}", "chunk_count": chunks,
               "uploaded_at": 1.0}
        (tmp / pid / "qa" / "sources" / f"{sid}.json").write_text(json.dumps(src))
    questions = [
        ("srcAAA", "Who rules Vaelindrath?", "Queen Isolde."),
        ("srcAAA", "What is the capital?", "Spirefen."),
        ("srcBBB", "How many wives does the Turnwarden keep?", "Three."),
    ]
    from finetune_studio.data.prep.qa_validate import Provenance, build_qa_record as bqr
    class _P:
        def __init__(self, q, a): self.question, self.answer = q, a
    for j, (sid, q, a) in enumerate(questions):
        qa = bqr(qa_id=f"qa{j:04d}", pair=_P(q, a),
                 provenance=Provenance(source_id=sid, sha256="sha" + sid[-1],
                                       filename="", chunk_idx=1),
                 chunk_text="ctx", difficulty="medium", style="extr con",
                 score=1.0, created_at=1.0, status="approved")
        (tmp / pid / "qa" / "pairs" / f"qa{j:04d}.json").write_text(json.dumps(qa))

    # Subset fill: only the picked sources; unknown id -> 400
    r = client.post(f"/api/projects/{pid}/datasets/subset",
                    json={"source_ids": ["nosuch"]})
    assert r.status_code == 400
    r = client.post(f"/api/projects/{pid}/datasets/subset",
                    json={"source_ids": ["srcAAA", "srcBBB"], "fmt": "sharegpt",
                          "name": "special"})
    body = r.json() if r.status_code == 200 else {"status": r.status_code, "t": r.text[:200]}
    assert r.status_code == 200, body
    assert body["rows"] == 3
    assert body["per_source"] == {"srcAAA": 2, "srcBBB": 1}
    assert body["dataset"]["source"] == "subset-picked"
    # dataset registered and greppable through normal listing
    listing = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert any(d["name"] == body["dataset"]["name"] for d in listing)
    # coverage_fill summary present (may fill 0 since all chunks covered above)
    assert "coverage_fill" in body


def test_subset_empty_payload_rejected(client, project, temp_db, monkeypatch):
    import tempfile
    from pathlib import Path
    import finetune_studio.data.fs.qa as qafs
    tmp = Path(tempfile.mkdtemp(prefix="vers-empty-"))
    monkeypatch.setattr(qafs, "project_dir", lambda pid: tmp / pid)
    (tmp / project / "qa" / "sources").mkdir(parents=True)
    (tmp / project / "qa" / "sources" / "srcX.json").write_text(json.dumps(
        {"id": "srcX", "filename": "x.txt", "sha256": "s", "chunk_count": 1,
         "uploaded_at": 1.0}))
    r = client.post(f"/api/projects/{project}/datasets/subset",
                    json={"source_ids": ["srcX"]})
    assert r.status_code == 400  # no approved pairs -> rejected


def test_rag_coverage_gate(client, project, temp_db, monkeypatch):
    import tempfile
    from pathlib import Path
    import finetune_studio.config as cfg
    import finetune_studio.data.fs.qa as qafs

    tmp = Path(tempfile.mkdtemp(prefix="vers-ragcov-"))
    monkeypatch.setattr(qafs, "project_dir", lambda pid: tmp / pid)
    (tmp / project / "qa" / "sources").mkdir(parents=True)
    for sid, fn in [("s1", "alpha.txt"), ("s2", "beta.txt"), ("s3", "gamma.txt")]:
        (tmp / project / "qa" / "sources" / f"{sid}.json").write_text(json.dumps(
            {"id": sid, "filename": fn, "sha256": "h-" + sid, "chunk_count": 3,
             "uploaded_at": 1.0}))
    # Point rag corpora dir at temp too
    import finetune_studio.webui.routes.versions as vr
    monkeypatch.setattr(Path, "parent", Path.parent, raising=False)  # noop sanity
    corpus = tmp / "rag_corpora" / project
    corpus.mkdir(parents=True)
    manifest = {"extra": {"documents_meta": [
        {"document_id": "d1", "filename": "alpha.txt"},
        {"document_id": "d2", "filename": "beta.txt"},
    ]}}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    # settings.db_path determines corpus root: patch settings used inside route
    class _Fake:
        pass
    _Fake.db_path = str(tmp / "fake.db")
    import finetune_studio.config as cfg
    real_settings = vr.settings if hasattr(vr, "settings") else None
    # The route reads settings at call time via finetune_studio.config; patch there
    from finetune_studio import config as cfgmod
    class _S: pass
    _S.db_path = str(tmp / "fake.db")
    monkeypatch.setattr(cfgmod, "settings", _S, raising=False)

    # No corpus for other pid -> 400 handled; our project: 2/3 covered
    r = client.get(f"/api/projects/{project}/rag/coverage")
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["parsed_sources"] == 3
    assert body["covered"] == 2
    assert body["coverage_pct"] == 66.7
    missing_names = {m["filename"] for m in body["missing_sources"]}
    assert missing_names == {"gamma.txt"}
