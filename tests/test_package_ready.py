"""The package opens itself before it is handed over, and the Package page says what is left to do."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_studio.knowledge import readiness as rd
from finetune_studio.smart_brain import identity as bi
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package, verify_package
from tests.test_brain_server_package import corpus_dir  # noqa: F401

FAST = BrainExportConfig(kdf_log_n=10)  # cheap scrypt for tests
PASS = "correct horse battery staple"


def _states(result: dict) -> dict[str, str]:
    return {s["key"]: s["state"] for s in result["steps"]}


@pytest.fixture()
def embed_url():
    """An OpenAI-compatible /embeddings endpoint answering 3-dimensional vectors, like the fixture corpus's."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = json.dumps({"data": [{"index": i, "embedding": [1.0, 0.0, 0.0]} for i in range(len(texts))]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_a) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_a_good_encrypted_package_passes_every_search_layer(corpus_dir: Path, tmp_path: Path, embed_url: str) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS,
                          identity={"name": "Ada", "system_prompt": "You are Ada."})
    out = verify_package(built.path, PASS, embed_base_url=embed_url)
    assert out["ok"] is True, out["detail"]
    assert out["hits"] > 0 and out["identity"] is True and out["http"]["ok"] is True
    assert out["exercised"] == {"fusion": True, "keyword": True, "semantic": True}
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".verify-")] == []     # nothing left behind


def test_meaning_search_that_cannot_run_fails_the_check(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS)
    out = verify_package(built.path, PASS)       # vectors shipped, no embedder inside and none lent
    assert out["ok"] is False and "semantic" in out["detail"]
    assert out["layers"]["semantic"]["available"] is False and out["exercised"]["keyword"] is True


def test_the_wrong_passphrase_fails_and_says_so(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS)
    out = verify_package(built.path, "wrong passphrase here")
    assert out["ok"] is False and out["detail"] and out["hits"] == 0


def test_a_damaged_archive_fails_instead_of_raising(tmp_path: Path) -> None:
    bad = tmp_path / "p.tar.gz"
    bad.write_bytes(b"not an archive")
    out = verify_package(bad, PASS)
    assert out["ok"] is False and "could not open" in out["detail"]


def test_a_zip_package_passes(corpus_dir: Path, tmp_path: Path, embed_url: str) -> None:  # noqa: F811
    built = build_package(corpus_dir, tmp_path / "p.zip", name="T", fmt="zip",
                          config=BrainExportConfig(kdf_log_n=10, archive_format="zip"), passphrase=PASS)
    assert verify_package(built.path, PASS, embed_base_url=embed_url)["ok"] is True


def test_package_self_test_uses_the_bundled_search_settings(corpus_dir: Path, tmp_path: Path, monkeypatch,  # noqa: F811
                                                             embed_url: str) -> None:
    from finetune_studio.smart_brain import server_package

    calls: list[list[str]] = []
    original = server_package.subprocess.run

    def record(args, *a, **kw):
        calls.append(list(args))
        return original(args, *a, **kw)

    monkeypatch.setattr(server_package.subprocess, "run", record)
    built = build_package(corpus_dir, tmp_path / "p.tar.gz", name="T", config=FAST, passphrase=PASS)
    assert verify_package(built.path, PASS, embed_base_url=embed_url)["ok"] is True
    mcp = next(args for args in calls if "--mcp" in args)
    assert "--no-semantic" not in mcp and "--no-reranker" not in mcp


def test_package_self_test_fails_when_a_search_layer_is_not_exercised(tmp_path: Path, monkeypatch) -> None:
    from finetune_studio.smart_brain import server_package

    archive = tmp_path / "package.tar.gz"
    archive.touch()
    monkeypatch.setattr(server_package, "_unpack_runtime", lambda archive, dest: tmp_path)
    monkeypatch.setattr(server_package, "_sample_query", lambda root, env: "sample query")

    def run_mcp(root, env, requests):
        if requests[1]["id"] == 2:
            return {
                2: {"result": {"content": [{"text": json.dumps({"layers": {
                    "semantic": "on (bundled embedder)", "keyword": "on", "fusion": "on (RRF)",
                    "reranker": "off",
                }})}]}},
                3: {"result": {"content": [{"text": json.dumps({"instructions": "grounded"})}]}},
            }
        return {4: {"result": {"content": [{"text": json.dumps([
            {"chunk_id": "c1", "text": "sample", "bm25_score": 1.0},
        ])}]}}}

    monkeypatch.setattr(server_package, "_run_mcp", run_mcp)
    result = verify_package(archive, None)

    assert result["ok"] is False
    assert result["exercised"] == {"fusion": False, "keyword": True, "semantic": False}
    assert "fusion, semantic" in result["detail"]


@pytest.fixture
def project(client) -> str:
    return client.post("/api/projects", json={"name": "ready-test"}).json()["id"]


def test_an_empty_project_is_not_ready_and_every_step_is_listed(client, project: str) -> None:
    r = client.get(f"/api/projects/{project}/brain/readiness").json()
    assert r["ready"] is False
    assert list(_states(r)) == ["files", "check", "identity", "index", "test", "package"]
    assert set(_states(r).values()) == {"todo"}


def test_identity_and_a_failed_self_test_show_up(client, project: str) -> None:
    bi.save(project, {"name": "Ada"})
    rd.record_package(project, filename="p.tar.gz", size=1, encrypted=True, identity=True,
                      verified={"ok": False, "detail": "the server would not start"})
    r = client.get(f"/api/projects/{project}/brain/readiness").json()
    steps = {s["key"]: s for s in r["steps"]}
    assert steps["identity"]["state"] == "done" and steps["identity"]["detail"] == "Ada"
    assert steps["package"]["state"] == "problem" and "would not start" in steps["package"]["detail"]


def test_a_package_older_than_the_index_needs_review(monkeypatch, client, project: str) -> None:
    rd.record_package(project, filename="p.tar.gz", size=1, encrypted=True, identity=False, verified={"ok": True, "detail": "x"})
    monkeypatch.setattr(rd, "_index_time", lambda pid: 9e12)
    assert _states(rd.readiness(project))["package"] == "review"


def test_unreviewed_superseded_documents_block_package_build(monkeypatch, client, project: str,
                                                             tmp_path: Path) -> None:
    from finetune_studio.knowledge import lint as knowledge_lint
    from finetune_studio.webui.routes import brain_export

    finding = {"rule": "old_version", "ignored": False,
               "files": [{"name": "staff-handbook-superseded.pdf"}]}
    report = {"errors": 0, "warnings": 1, "ignored": 0, "findings": [finding]}
    monkeypatch.setattr(rd, "list_sources", lambda pid: [SimpleNamespace(ready=True)])
    monkeypatch.setattr(rd, "lint", lambda pid: report)
    monkeypatch.setattr(knowledge_lint, "lint", lambda pid: report)
    monkeypatch.setattr(rd.knowledge_index, "state", lambda pid: SimpleNamespace(fresh=True, built=True))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(brain_export, "_corpus_dir", lambda pid: corpus)

    status = client.get(f"/api/projects/{project}/brain/readiness").json()
    assert status["ready"] is False
    check = next(step for step in status["steps"] if step["key"] == "check")
    assert check["state"] == "problem" and "removed or explicitly accepted" in check["detail"]

    response = client.post(f"/api/projects/{project}/brain/standalone", json={})
    assert response.status_code == 409
    assert "staff-handbook-superseded.pdf" in response.json()["detail"]

    accepted = {**report, "warnings": 0, "ignored": 1,
                "findings": [{**finding, "ignored": True}]}
    monkeypatch.setattr(rd, "lint", lambda pid: accepted)
    assert rd.readiness(project)["ready"] is True


def test_the_package_page_shows_the_list(client, project: str) -> None:
    body = client.get(f"/projects/{project}/export").text
    assert 'id="package-readiness"' in body and "<h1>Package</h1>" in body and "Not ready yet" in body
    assert 'id="package-build-link"' in body and "Trained models" in body
    assert ">4 package<" in body
