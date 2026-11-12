"""Include original files: the uploaded files ship in the package (encrypted with the corpus) and come back over MCP
(``rag_get_document``, ``rag_list_documents``), HTTP (``GET /files/<id>``) and the chat page's download links."""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from finetune_studio.smart_brain import originals as brain_originals
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package, verify_package
from tests.test_brain_encrypted_package import _load_server
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)

PASS = "correct horse battery staple"
ON = BrainExportConfig(kdf_log_n=10, include_originals=True)
PDF = b"%PDF-1.4\n% ledger of Vaelindrath\n" + bytes(range(256)) * 4
CSV = "crowns,season\n9,winter\n".encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def files(tmp_path: Path) -> list[brain_originals.OriginalFile]:
    d = tmp_path / "uploads"
    d.mkdir()
    (d / "ledger.pdf").write_bytes(PDF)
    (d / "highmere prices.csv").write_bytes(CSV)
    return [brain_originals.OriginalFile("c1", "ledger.pdf", d / "ledger.pdf", "application/pdf", len(PDF), _sha(PDF), 1.0),
            brain_originals.OriginalFile("c2", "highmere prices.csv", d / "highmere prices.csv", "text/csv", len(CSV),
                                         _sha(CSV), 2.0)]


def _open(archive: Path, dest: Path, passphrase: str | None = PASS, cfg: dict | None = None):
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    root = next(dest.rglob("server.py")).parent
    srv = _load_server()
    config = srv.resolve_config(cfg or {}, {}, {})[0]
    return srv, srv.Corpus(root / "corpus", config, passphrase), root


def test_originals_ship_encrypted_and_come_back_unchanged(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=ON, passphrase=PASS, originals=files)
    raw = built.path.read_bytes()
    with tarfile.open(built.path) as tf:
        names = tf.getnames()
        blob = b"".join(tf.extractfile(m).read() for m in tf.getmembers() if m.isfile())
    assert not any("originals" in n or "ledger" in n for n in names)          # nothing beside the container
    assert b"ledger of Vaelindrath" not in blob and b"crowns,season" not in blob and b"ledger" not in raw
    srv, corpus, root = _open(built.path, tmp_path / "x")
    assert corpus.encrypted and set(corpus.originals) == {"c1", "c2"}
    assert corpus.read_original("c1") == PDF and corpus.read_original("c2~facts") == CSV
    hits = corpus.search("salt seal oath")
    assert hits and hits[0]["document_id"] == "c1" and hits[0]["original"] is True
    assert corpus.info()["original_files"] == 2
    assert "Original files" in (root / "README.md").read_text() and "rag_get_document" in (root / "README.md").read_text()


def test_mcp_tools_list_and_fetch_a_document(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=ON, passphrase=PASS, originals=files)
    srv, corpus, _root = _open(built.path, tmp_path / "x")
    respond = srv.mcp_responder(corpus, base_url="http://127.0.0.1:9")
    tools = {t["name"]: t for t in respond({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert {"rag_list_documents", "rag_get_document"} <= set(tools)
    assert all(tools[n]["annotations"]["readOnlyHint"] is True for n in ("rag_list_documents", "rag_get_document"))

    def call(name, args):
        return respond({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": args}})

    listed = json.loads(call("rag_list_documents", {})["result"]["content"][0]["text"])
    assert {d["filename"]: d["original"] for d in listed} == {
        "ledger.txt": True, "highmere.txt": True, "emberfall.txt": False}
    assert {d["filename"]: d.get("original_filename") for d in listed}["highmere.txt"] == "highmere prices.csv"
    assert [d["filename"] for d in json.loads(call("rag_list_documents", {"name_contains": "HIGH"})
                                              ["result"]["content"][0]["text"])] == ["highmere.txt"]
    about = json.loads(call("rag_list_documents", {"about": "warehousing crowns"})["result"]["content"][0]["text"])
    assert about[0]["document_id"] == "c2"

    got = call("rag_get_document", {"document_id": "c1"})["result"]
    meta = json.loads(got["content"][0]["text"])
    res = got["content"][1]
    assert res["type"] == "resource" and res["resource"]["mimeType"] == "application/pdf"
    assert base64.b64decode(res["resource"]["blob"]) == PDF and meta["sha256"] == _sha(PDF)
    assert meta["url"] == "http://127.0.0.1:9/files/c1"
    by_name = call("rag_get_document", {"filename": "Highmere Prices.CSV"})["result"]
    assert base64.b64decode(by_name["content"][1]["resource"]["blob"]) == CSV
    missing = call("rag_get_document", {"document_id": "nope"})["result"]
    assert missing["isError"] is True and "rag_list_documents" in missing["content"][0]["text"]


def test_a_file_over_the_inline_cap_comes_back_as_a_link(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    big = tmp_path / "uploads" / "big.bin"
    big.write_bytes(b"\0" * (1024 * 1024 + 1))
    rows = [*files[:1], brain_originals.OriginalFile("c3", "big.bin", big, "application/octet-stream",
                                                     big.stat().st_size, _sha(big.read_bytes()), 3.0)]
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=ON, passphrase=PASS, originals=rows)
    srv, corpus, _root = _open(built.path, tmp_path / "x", cfg={"file_inline_max_mb": 1})
    out = srv.mcp_responder(corpus)({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "rag_get_document", "arguments": {"document_id": "c3"}}})["result"]
    assert len(out["content"]) == 1
    meta = json.loads(out["content"][0]["text"])
    assert meta["inline"] is False and meta["url"] == "/files/c3" and "1 MB" in meta["note"]


def test_option_off_ships_no_originals(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=BrainExportConfig(kdf_log_n=10),
                          passphrase=PASS, originals=files)
    srv, corpus, root = _open(built.path, tmp_path / "x")
    assert corpus.originals == {} and "originals.json" not in corpus._container.names()
    names = [t["name"] for t in srv.mcp_responder(corpus)({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
             ["result"]["tools"]]
    assert "rag_get_document" not in names and "rag_list_documents" in names
    assert "original" not in corpus.search("salt")[0]
    assert "Original files" not in (root / "README.md").read_text()


def test_plaintext_package_keeps_originals_in_corpus_dir(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=ON, encrypt=False, originals=files)
    with tarfile.open(built.path) as tf:
        names = tf.getnames()
    assert any(n.endswith("corpus/originals/c1") for n in names) and any(n.endswith("corpus/originals.json") for n in names)
    _srv, corpus, _root = _open(built.path, tmp_path / "x", passphrase=None)
    assert corpus.read_original("c2") == CSV


def _serve(root: Path, tmp_path: Path, env: dict | None = None) -> tuple[subprocess.Popen, str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = (tmp_path / "srv.log").open("w")
    proc = subprocess.Popen([sys.executable, "-I", str(root / "server.py"), "--http", str(port)], cwd=root,
                            env={"RAG_PASSPHRASE": PASS, **(env or {})}, stdout=subprocess.DEVNULL, stderr=log)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(base + "/", timeout=2).close()
            break
        except OSError:
            time.sleep(0.25)
    return proc, base


def _status(url: str, headers: dict | None = None) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_http_files_route_downloads_with_auth_and_refuses_the_rest(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=ON, passphrase=PASS, originals=files)
    _srv, _corpus, root = _open(built.path, tmp_path / "x")
    proc, base = _serve(root, tmp_path, {"RAG_AUTH_TOKEN": "tok-123"})
    try:
        auth = {"Authorization": "Bearer tok-123"}
        with urllib.request.urlopen(urllib.request.Request(base + "/files/c1", headers=auth), timeout=10) as r:
            assert r.read() == PDF and r.headers["Content-Type"] == "application/pdf"
            assert r.headers["Content-Disposition"].startswith('attachment; filename="ledger.pdf"')
        with urllib.request.urlopen(urllib.request.Request(base + "/files/c2", headers=auth), timeout=10) as r:
            assert "filename*=UTF-8''highmere%20prices.csv" in r.headers["Content-Disposition"]
        assert _status(base + "/files/c1") == 401                              # same token as every API route
        assert _status(base + "/files/c3", auth) == 404                         # a document without its original
        assert _status(base + "/files/unknown", auth) == 404
        assert _status(base + "/files/..%2Fserver.py", auth) == 404
        assert _status(base + "/files/../server.py", auth) == 404
        assert _status(base + "/files/", auth) == 404
        js = urllib.request.urlopen(base + "/app.js", timeout=5).read().decode()
        assert "'/files/' + encodeURIComponent(id)" in js and "createObjectURL" in js
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_self_test_fetches_an_original_over_mcp_and_http(corpus_dir: Path, tmp_path: Path, files) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", config=ON.model_copy(update={"semantic_search": False}),
                          passphrase=PASS, originals=files)
    good = verify_package(built.path, PASS, expected_originals={f.document_id: f.sha256 for f in files})
    assert good["ok"] is True, good["detail"]
    assert good["original"]["ok"] and good["original"]["document_id"] == "c2"   # the smallest
    assert "original-file download" in good["http"]["detail"]
    bad = verify_package(built.path, PASS, expected_originals={"c2": "0" * 64})
    assert bad["ok"] is False and "differs from the source" in bad["detail"]


def test_collect_finds_project_uploads_and_corpus_sources(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    up = tmp_path / "raw.pdf"
    up.write_bytes(PDF)
    loose = tmp_path / "notes.md"
    loose.write_text("notes")
    meta = [{"document_id": "aaa111bbb222", "filename": "Report.pdf", "source": str(tmp_path / "parsed.txt")},
            {"document_id": "aaa111bbb222~facts", "filename": "Report.pdf · reviewed facts", "source": ""},
            {"document_id": "ccc", "filename": "gone.pdf", "source": str(tmp_path / "parsed.txt")}]
    (corpus / "manifest.json").write_text(json.dumps({"extra": {"documents_meta": meta}}))
    src = SimpleNamespace(id="aaa111bbb222", filename="Report.pdf", raw_path=str(up), mime_type="application/pdf",
                          sha256=_sha(PDF), uploaded_at=5.0)
    monkeypatch.setattr("finetune_studio.knowledge.sources.list_sources", lambda pid: [src])
    rows = brain_originals.collect(corpus, "p1")
    assert [(r.document_id, r.filename, r.sha256, r.size) for r in rows] == [("aaa111bbb222", "Report.pdf", _sha(PDF), len(PDF))]

    (corpus / "manifest.json").write_text(json.dumps({"extra": {"documents_meta": [
        {"document_id": "d1", "filename": "notes.md", "source": str(loose)}]}}))
    assert [(r.document_id, r.mime_type) for r in brain_originals.collect(corpus)] == [("d1", "text/markdown")]


def test_export_config_and_ui_drive_the_option(client, tmp_path: Path, monkeypatch, corpus_dir: Path, files) -> None:  # noqa: F811
    from finetune_studio.webui.routes import settings as settings_mod

    assert BrainExportConfig().include_originals is False
    assert BrainExportConfig().runtime_config()["file_inline_max_mb"] == 20
    pid = client.post("/api/projects", json={"name": "agent-test-originals"}).json()["id"]
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr("finetune_studio.webui.routes.brain_export._corpus_dir", lambda p: corpus_dir)
    monkeypatch.setattr("finetune_studio.smart_brain.originals.collect", lambda corpus, p: files)
    monkeypatch.chdir(tmp_path)
    client.put(f"/api/projects/{pid}/brain/export-config", json={"scope": "studio", "values": {"kdf_log_n": 10}})
    r = client.post(f"/api/projects/{pid}/brain/standalone",
                    json={"include_originals": True, "semantic_search": False, "passphrase": PASS})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["original_files"] == 2 and j["config"]["include_originals"] is True
    assert j["verified"]["ok"] is True and j["verified"]["original"]["ok"] is True, j["verified"]
    off = client.post(f"/api/projects/{pid}/brain/standalone", json={"semantic_search": False, "passphrase": PASS})
    assert off.json()["original_files"] == 0 and "original" not in off.json()["verified"]
    page = client.get(f"/projects/{pid}/brain").text
    assert 'id="mcp-originals"' in page and "Leave this off for a public bot" in page
    assert "include_originals: g('mcp-originals').checked" in page
