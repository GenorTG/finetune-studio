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
    from finetune_studio.data.fs import qa as qa_fs
    from tests.knowledge_helpers import add_source

    # selection/registration under test; the completeness gate has its own test
    monkeypatch.setattr("finetune_studio.data.mining.project_coverage", lambda pid, source_ids=None: [])
    pid = project
    ids = [add_source(pid, fn, f"The {fn} file: Queen Isolde rules from Spirefen.").id
           for fn in ("alpha.txt", "beta.txt", "gamma.txt")]
    a_id, b_id = ids[0], ids[1]
    questions = [
        (a_id, "Who rules Vaelindrath?", "Queen Isolde."),
        (a_id, "What is the capital?", "Spirefen."),
        (b_id, "How many wives does the Turnwarden keep?", "Three."),
    ]
    from finetune_studio.data.prep.qa_validate import Provenance
    from finetune_studio.data.prep.qa_validate import build_qa_record as bqr
    class _P:
        def __init__(self, q, a): self.question, self.answer = q, a
    for j, (sid, q, a) in enumerate(questions):
        qa_fs.write_qa_pair(pid, bqr(qa_id=f"qa{j:04d}", pair=_P(q, a),
                                     provenance=Provenance(source_id=sid, sha256="sha", filename="", chunk_idx=1),
                                     chunk_text="ctx", score=1.0, created_at=1.0, origin="mine", status="approved"))

    # Subset fill: only the picked sources; unknown id -> 400
    r = client.post(f"/api/projects/{pid}/datasets/subset",
                    json={"source_ids": ["nosuch"]})
    assert r.status_code == 400
    r = client.post(f"/api/projects/{pid}/datasets/subset",
                    json={"source_ids": [a_id, b_id], "fmt": "sharegpt",
                          "name": "special"})
    body = r.json() if r.status_code == 200 else {"status": r.status_code, "t": r.text[:200]}
    assert r.status_code == 200, body
    assert body["rows"] == 3
    assert body["per_source"] == {a_id: 2, b_id: 1}
    assert body["dataset"]["source"] == "subset-picked"
    # dataset registered and greppable through normal listing
    listing = client.get(f"/api/projects/{pid}/datasets").json()["datasets"]
    assert any(d["name"] == body["dataset"]["name"] for d in listing)
    assert body["incomplete_files"] == []


def test_subset_empty_payload_rejected(client, project, temp_db, monkeypatch):
    from tests.knowledge_helpers import add_source

    # selection/registration under test; the completeness gate has its own test
    monkeypatch.setattr("finetune_studio.data.mining.project_coverage", lambda pid, source_ids=None: [])
    sid = add_source(project, "x.txt", "A file with no approved pairs at all.").id
    r = client.post(f"/api/projects/{project}/datasets/subset",
                    json={"source_ids": [sid]})
    assert r.status_code == 400  # no approved pairs -> rejected


def test_subset_build_blocks_incomplete_files_unless_forced(client, project, monkeypatch):
    from finetune_studio.data.mining.coverage import FileCoverage, ItemState

    monkeypatch.setattr("finetune_studio.knowledge.sources.names", lambda pid: {"src-gap": "gap.txt"})
    monkeypatch.setattr("finetune_studio.data.mining.project_coverage", lambda pid, source_ids=None: [
        FileCoverage("src-gap", [ItemState("i1", "open")], "gaps", {"total": 0})])
    r = client.post(f"/api/projects/{project}/datasets/subset", json={"source_ids": ["src-gap"]})
    assert r.status_code == 409
    body = r.json()
    assert "blocked" in body["error"] and body["incomplete_files"][0]["status"] == "gaps"
    assert body["incomplete_files"][0]["counts"]["open"] == 1
    r = client.post(f"/api/projects/{project}/datasets/subset", json={"source_ids": ["src-gap"], "force": True})
    assert r.status_code == 400  # forced past the gate; then no approved pairs to build from


def test_brain_coverage_gate(client, project, temp_db):
    from finetune_studio.data.fs.paths import brain_index_dir
    from tests.knowledge_helpers import add_source

    srcs = {fn: add_source(project, fn, f"The {fn} file holds one sentence of facts.")
            for fn in ("alpha.txt", "beta.txt", "gamma.txt")}
    corpus = brain_index_dir(project)
    corpus.mkdir(parents=True)
    manifest = {"extra": {"built_from": {s.id: s.text_sha for fn, s in srcs.items() if fn != "gamma.txt"}}}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    r = client.get(f"/api/projects/{project}/brain/coverage")
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert body["parsed_sources"] == 3
    assert body["covered"] == 2
    assert body["coverage_pct"] == 66.7
    missing_names = {m["filename"] for m in body["missing_sources"]}
    assert missing_names == {"gamma.txt"}


def test_brain_coverage_empty_states_are_200_not_errors(client):
    """Quick work polls coverage on every load; a fresh project is a normal state,
    not a client error (it used to 400 and log a console error each visit)."""
    from tests.knowledge_helpers import add_source

    pid = client.post("/api/projects", json={"name": "cov-empty"}).json()["id"]
    r = client.get(f"/api/projects/{pid}/brain/coverage")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "no_sources" and r.json()["parsed_sources"] == 0

    add_source(pid, "a.txt", "One parsed file with a sentence in it.")
    r = client.get(f"/api/projects/{pid}/brain/coverage")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "not_built" and body["parsed_sources"] == 1 and body["covered"] == 0

    assert client.get("/api/projects/nope/brain/coverage").status_code == 404
