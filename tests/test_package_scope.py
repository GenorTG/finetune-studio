"""Package audience scope: a package carries only the chosen documents (folder/path prefix, tags, single ones), its
corpus rebuilt from them (passages, sliced vectors, recomputed BM25, originals); nothing outside can be found."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from finetune_studio.smart_brain import scope as sc
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package, verify_package
from tests.test_brain_encrypted_package import _load_server
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)
from tests.test_package_originals import files  # noqa: F401  (fixture)

PASS = "correct horse battery staple"
FAST = BrainExportConfig(kdf_log_n=10)
DOCS = [sc.DocRef("c1", "ledger.txt", "public/ledger.txt", ("faq",)),
        sc.DocRef("c2", "highmere.txt", "public/prices/highmere.txt"),
        sc.DocRef("c3", "emberfall.txt", "internal/emberfall.txt", ("faq", "hr"))]


def _keep(**kw) -> set[str] | None:
    return sc.resolve(sc.DocumentScope(**kw), DOCS)[0]


def test_scope_rules() -> None:
    assert _keep() is None
    assert _keep(include=["public/"]) == {"c1", "c2"}
    assert _keep(include=["/PUBLIC/prices"]) == {"c2"}                       # leading slash and case do not matter
    assert _keep(tags=["hr"]) == {"c3"}
    assert _keep(include=["public/"], tags=["hr"]) == {"c1", "c2", "c3"}     # any choice
    assert _keep(include=["public/"], exclude=["public/prices/"]) == {"c1"}
    assert _keep(exclude=["internal/"]) == {"c1", "c2"}                      # exclude alone: everything else
    assert _keep(document_ids=["c3"]) == {"c3"}
    with pytest.raises(ValueError, match="no document matches"):
        _keep(include=["nothing/"])
    _keep_ids, record = sc.resolve(sc.DocumentScope(include=["public/"], exclude=["public/prices/"]), DOCS)
    assert record["documents"] == 1 and record["of"] == 3 and record["label"] == "paths public/ except public/prices/"
    ov = sc.overview(DOCS)
    assert {f["prefix"]: f["count"] for f in ov["folders"]} == {"internal/": 1, "public/": 1, "public/prices/": 1}
    assert {t["tag"]: t["count"] for t in ov["tags"]} == {"faq": 2, "hr": 1} and ov["total"] == 3


def _open(archive: Path, dest: Path):
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    root = next(dest.rglob("server.py")).parent
    srv = _load_server()
    return srv, srv.Corpus(root / "corpus", srv.resolve_config({}, {}, {})[0], PASS), root


def test_scoped_package_holds_only_the_chosen_documents(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    keep, record = sc.resolve(sc.DocumentScope(include=["public/"]), DOCS)
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=FAST.model_copy(update={"include_originals": True}),
                          passphrase=PASS, originals=files + [files[0].__class__("c3", "x.pdf", files[0].path,
                                                                                 "application/pdf", 1, "0", 0.0)],
                          documents=keep, scope=record)
    _srv, corpus, root = _open(built.path, tmp_path / "x")
    assert {c["document_id"] for c in corpus.chunks} == {"c1", "c2"} and corpus.chunk_ids == ["c1", "c2"]
    assert corpus.vectors.shape == (2, 3)
    assert np.array_equal(corpus.vectors, np.eye(3, dtype=np.float32)[:2])        # each passage keeps its own vector
    assert corpus.idx_map == {"c1": 0, "c2": 1}
    assert corpus.bm25["doc_count"] == 2 and "emberfall" not in corpus.bm25["terms"]   # recomputed over the subset
    assert corpus.bm25["avgdl"] == pytest.approx(np.mean(corpus.bm25["doc_lens"]))
    assert corpus.search("Emberfall meteoric shower") == []
    assert corpus.search("crowns per season")[0]["document_id"] == "c2"
    assert set(corpus.originals) == {"c1", "c2"}
    assert sorted(n for n in corpus._container.names() if n.startswith("sources/")) == ["sources/c1.txt", "sources/c2.txt"]
    info = corpus.info()
    assert info["documents"] == 2 and info["chunks"] == 2 and info["scope"]["label"] == "paths public/"
    assert "2 of 3 documents" in (root / "README.md").read_text()


def test_unscoped_package_states_all_documents(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=FAST, passphrase=PASS)
    _srv, corpus, root = _open(built.path, tmp_path / "x")
    assert corpus.info()["scope"] == {"label": "all documents", "documents": 3, "of": 3}
    assert "All 3 documents of the index" in (root / "README.md").read_text()


def test_self_test_proves_nothing_outside_the_scope_is_searchable(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    keep, record = sc.resolve(sc.DocumentScope(document_ids=["c1", "c2"]), DOCS)
    probes = sc.probe_queries(corpus_dir, {"c3"})
    assert probes == {"c3": "A meteoric salt shower over Emberfall is remembered as the Emberfall."}
    cfg = FAST.model_copy(update={"semantic_search": False})
    scoped = build_package(corpus_dir, tmp_path / "s.tar.gz", config=cfg, passphrase=PASS, documents=keep, scope=record)
    good = verify_package(scoped.path, PASS, out_of_scope=probes)
    assert good["ok"] is True and good["scope"]["ok"] is True, good
    leaky = build_package(corpus_dir, tmp_path / "all.tar.gz", config=cfg, passphrase=PASS)   # everything inside
    bad = verify_package(leaky.path, PASS, out_of_scope=probes)
    assert bad["ok"] is False and "outside the scope can be found" in bad["detail"]


def test_project_documents_carry_folder_path_and_tags(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(json.dumps({"extra": {"documents_meta": [
        {"document_id": "aaa", "filename": "a.pdf"}, {"document_id": "aaa~facts", "filename": "a.pdf · facts"},
        {"document_id": "bbb", "filename": "b.txt"}]}}))
    files_ = [{"id": "f1", "original_name": "a.pdf", "tags": "public, faq"}, {"id": "f2", "original_name": "b.txt", "tags": ""}]
    monkeypatch.setattr("finetune_studio.data.fs.file_library.list_files",
                        lambda pid, folder_id=None: [f for f in files_ if folder_id in (None, "fu1") and
                                                     (folder_id is None or f["id"] == "f1")])
    monkeypatch.setattr("finetune_studio.data.fs.file_library.list_folders",
                        lambda pid, include_auto=True: [{"id": "fu1", "name": "public"}])
    monkeypatch.setattr("finetune_studio.knowledge.sources.list_sources", lambda pid: [
        SimpleNamespace(id="aaa", file_id="f1"), SimpleNamespace(id="bbb", file_id="f2")])
    docs = sc.documents(corpus, "p1")
    assert docs == [sc.DocRef("aaa", "a.pdf", "public/a.pdf", ("public", "faq")), sc.DocRef("bbb", "b.txt", "b.txt", ())]


@pytest.fixture()
def export_env(client, tmp_path: Path, monkeypatch, corpus_dir: Path):  # noqa: F811
    from finetune_studio.webui.routes import settings as settings_mod

    pid = client.post("/api/projects", json={"name": "agent-test-scope"}).json()["id"]
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr("finetune_studio.webui.routes.brain_export._corpus_dir", lambda p: corpus_dir)
    monkeypatch.setattr("finetune_studio.smart_brain.scope.documents", lambda corpus, p: DOCS)
    monkeypatch.chdir(tmp_path)
    client.put(f"/api/projects/{pid}/brain/export-config", json={"scope": "studio", "values": {"kdf_log_n": 10}})
    return pid


def test_route_and_ui_drive_the_scope(client, export_env) -> None:
    pid = export_env
    ov = client.get(f"/api/projects/{pid}/brain/package-documents").json()
    assert ov["total"] == 3 and {f["prefix"] for f in ov["folders"]} == {"public/", "public/prices/", "internal/"}
    r = client.post(f"/api/projects/{pid}/brain/standalone", json={
        "semantic_search": False, "passphrase": PASS, "scope": {"include": ["public/"]}})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["scope"]["documents"] == 2 and j["scope"]["of"] == 3
    assert j["verified"]["ok"] is True and j["verified"]["scope"]["ok"] is True, j["verified"]
    none = client.post(f"/api/projects/{pid}/brain/standalone", json={"passphrase": PASS, "scope": {"tags": ["zzz"]}})
    assert none.status_code == 422 and "no document matches" in none.json()["detail"]
    page = client.get(f"/projects/{pid}/brain").text
    for marker in ('id="mcp-scope"', 'id="mcp-scope-summary"', 'id="mcp-scope-folders"',
                   "/brain/package-documents", "scope: mcpScope()", "' of ' + scopeDocs.length + ' documents"):
        assert marker in page
    assert client.get("/api/projects/nope/brain/package-documents").status_code == 404
