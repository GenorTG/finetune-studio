"""Smart Brain export: every search layer is a separate switch (Genor 2026-10-28).

Meaning search (vectors), word search (BM25; both = rank fusion), the reranker and answer generation (the "G" in
RAG) can each be turned off at export; the package ships only the files the kept layers need, the server runs only
those layers (a flag can switch a shipped layer off, never on one that is not in the package) and reports which
ones run. Answer generation calls an OpenAI-compatible chat endpoint with the studio's grounded prompt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError

from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package, pipeline_text
from tests.test_brain_server_package import corpus_dir  # noqa: F401  (fixture)

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "templates"


def _plain(corpus: Path, tmp: Path, **cfg) -> Path:
    """Build a plaintext package with ``cfg`` and return its extracted root."""
    out = build_package(corpus, tmp / "p.tar.gz", name="Test Corpus", encrypt=False,
                        config=BrainExportConfig(**cfg)).path
    with tarfile.open(out, "r:gz") as tf:
        tf.extractall(tmp / "x", filter="data")
    (root,) = [p for p in (tmp / "x").iterdir() if p.is_dir()]
    return root


def _run(root: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    full["PYTHONPATH"] = ""
    return subprocess.run([sys.executable, str(root / "server.py"), "--corpus", str(root / "corpus"), *args],
                          capture_output=True, text=True, timeout=120, check=False, env={**full, **(env or {})})


def _cfg(**over) -> dict:
    return srv.resolve_config(over, {}, {})[0]


# ── export settings ──────────────────────────────────────────────────────


def test_a_package_needs_at_least_one_way_to_find_passages() -> None:
    with pytest.raises(ValidationError, match="meaning search, word search or both"):
        BrainExportConfig(semantic_search=False, keyword_search=False)
    with pytest.raises(ValidationError, match="http"):
        BrainExportConfig(llm_base_url="localhost:11434")
    cfg = BrainExportConfig(semantic_search=False, answer_enabled=True,
                            llm_base_url="http://127.0.0.1:11434/v1/", llm_model="qwen3:8b")
    rt = cfg.runtime_config()
    assert (rt["semantic"], rt["keyword"], rt["answer"]) == (False, True, True)
    assert (rt["llm_base_url"], rt["llm_model"]) == ("http://127.0.0.1:11434/v1", "qwen3:8b")
    assert "api_key" not in json.dumps(rt)                  # the key is a run-time env var only


def test_pipeline_line_names_each_layer() -> None:
    assert pipeline_text(BrainExportConfig(), bundled_reranker=True) == (
        "question → meaning search (vectors) + word search (BM25) → rank fusion (RRF) → reranker (cross-encoder)"
        " → passages with sources")
    assert pipeline_text(BrainExportConfig(semantic_search=False, answer_enabled=True), bundled_reranker=False) == (
        "question → word search (BM25) → LLM answer with sources")


# ── package contents ─────────────────────────────────────────────────────


def test_package_ships_only_the_files_its_layers_use(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    kw = _plain(corpus_dir, tmp_path / "kw", semantic_search=False)
    assert (kw / "corpus" / "bm25.json").is_file()
    assert not (kw / "corpus" / "vectors.npy").exists() and not (kw / "corpus" / "vectors.idx.json").exists()
    manifest = json.loads((kw / "corpus" / "manifest.json").read_text())
    assert manifest["search_layers"] == {"semantic": False, "keyword": True, "fusion": False,
                                         "reranker": False, "answer": False}   # no reranker model inside
    readme = (kw / "README.md").read_text()
    assert "**Pipeline:** question → word search (BM25) → passages with sources" in readme
    assert "## Search layers" in readme and "off (no reranker model in this package)" in readme

    sem = _plain(corpus_dir, tmp_path / "sem", keyword_search=False)
    assert (sem / "corpus" / "vectors.npy").is_file() and not (sem / "corpus" / "bm25.json").exists()
    assert json.loads((sem / "rag.config.json").read_text())["keyword"] is False


# ── the standalone server runs exactly the layers it has ────────────────


def test_word_search_only_package_searches_and_reports_its_layers(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root = _plain(corpus_dir, tmp_path, semantic_search=False)
    r = _run(root, "--query", "oath of salt ledger", "--top-k", "1")
    assert r.returncode == 0, r.stderr
    (hit,) = json.loads(r.stdout)
    assert hit["chunk_id"] == "c1" and hit["bm25_score"] > 0 and "dense_score" not in hit
    corpus = srv.Corpus(root / "corpus", _cfg())
    lay = corpus.info()["layers"]
    assert lay["semantic"] == "not in this package" and lay["keyword"] == "on" and lay["fusion"] == "off"
    assert corpus.info()["mode"] == "keyword"


def test_a_flag_switches_a_layer_off_but_never_on_one_not_shipped(corpus_dir: Path, tmp_path: Path) -> None:  # noqa: F811
    root = _plain(corpus_dir, tmp_path, semantic_search=False)
    r = _run(root, "--query", "salt", "--no-keyword")              # the package itself turned meaning search off
    assert r.returncode == 2 and "both off" in r.stderr
    r = _run(root, "--query", "salt", "--semantic", "--no-keyword")  # asks for a layer this package does not ship
    assert r.returncode == 2 and "no search layer can run" in r.stderr
    r = _run(root, "--query", "salt", "--semantic")
    assert r.returncode == 0 and "meaning search (no vectors) is not in this package" in r.stderr


def test_meaning_search_alone_ranks_by_vectors_and_skips_bm25(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    root = _plain(corpus_dir, tmp_path, keyword_search=False)
    with pytest.raises(srv.ConfigError, match="no bundled embedder and no RAG_EMBED_BASE_URL"):
        srv.Corpus(root / "corpus", _cfg())
    corpus = srv.Corpus(root / "corpus", _cfg(embed_base_url="http://embed.invalid/v1"))
    monkeypatch.setattr(corpus, "embed_query", lambda q: np.array([0, 0, 1], dtype=np.float32))
    monkeypatch.setattr(corpus, "bm25_scores", lambda q: pytest.fail("word search ran with the layer off"))
    hits = corpus.search("anything at all", 2)
    assert hits[0]["chunk_id"] == "c3"                      # the vectors are one-hot: c3 matches the question
    assert "bm25_score" not in hits[0] and hits[0]["dense_score"] == 1.0
    assert corpus.layers()["semantic"] == "on (embeddings API)" and corpus.layers()["keyword"] == "not in this package"


def test_hybrid_flag_off_now_really_turns_word_search_off(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Before: the server read the hybrid setting and searched with BM25 anyway."""
    root = _plain(corpus_dir, tmp_path)
    corpus = srv.Corpus(root / "corpus", _cfg(keyword=False, embed_base_url="http://embed.invalid/v1"))
    monkeypatch.setattr(corpus, "embed_query", lambda q: np.array([1, 0, 0], dtype=np.float32))
    monkeypatch.setattr(corpus, "bm25_scores", lambda q: pytest.fail("word search ran with the layer off"))
    assert corpus.search("highmere crowns", 1)[0]["chunk_id"] == "c1"
    assert corpus.layers()["keyword"] == "off"


# ── answer generation (the G in RAG) ─────────────────────────────────────


class _FakeChat(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeChat.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        reply = json.dumps({"choices": [{"message": {"content": "<think>x</think>Nine crowns [highmere.txt]."}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply.encode())

    def log_message(self, *a):
        return


@pytest.fixture()
def fake_chat():
    _FakeChat.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_answer_sends_the_grounded_prompt_and_returns_sources(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                             fake_chat: str) -> None:
    root = _plain(corpus_dir, tmp_path, semantic_search=False)
    with pytest.raises(srv.ConfigError, match="needs an LLM endpoint"):
        _cfg(answer=True)
    corpus = srv.Corpus(root / "corpus", _cfg(answer=True, llm_base_url=fake_chat + "/", llm_model="m1",
                                              llm_api_key="sk-test"))
    out = corpus.answer("How much does warehousing at the Highmere Stair cost?")
    assert out["answer"] == "Nine crowns [highmere.txt]." and out["model"] == "m1"
    assert out["sources"][0]["filename"] == "highmere.txt" and out["weak"] is False
    (call,) = _FakeChat.seen
    assert call["path"] == "/v1/chat/completions" and call["auth"] == "Bearer sk-test"
    system = call["body"]["messages"][0]["content"]
    assert system.startswith(srv.DEFAULT_SYSTEM_PROMPT) and "CONTEXT:\n[1] (source: highmere.txt" in system
    assert call["body"]["model"] == "m1" and call["body"]["messages"][1]["role"] == "user"


def test_answer_is_reachable_over_http_and_mcp_only_when_on(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                                            fake_chat: str) -> None:
    root = _plain(corpus_dir, tmp_path, semantic_search=False)
    off = srv.Corpus(root / "corpus", _cfg())
    tools = srv.mcp_responder(off)({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert "rag_answer" not in {t["name"] for t in tools}
    with pytest.raises(ValueError, match="answer generation is off"):
        off.answer("anything")
    on = srv.Corpus(root / "corpus", _cfg(answer=True, llm_base_url=fake_chat, llm_model="m1"))
    respond = srv.mcp_responder(on)
    tools = respond({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert "rag_answer" in {t["name"] for t in tools}
    res = respond({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "rag_answer", "arguments": {"question": "oath of salt ledger"}}})["result"]
    assert not res["isError"] and json.loads(res["content"][0]["text"])["answer"].startswith("Nine crowns")
    assert on.info()["layers"]["answer"] == "on (m1)"


def test_server_prompt_and_gate_match_the_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server ships on its own, so it carries copies; they must stay identical to the studio's."""
    from finetune_studio.smart_brain import prompt
    from finetune_studio.webui.routes.brain import weak_match

    assert srv.DEFAULT_SYSTEM_PROMPT == prompt.DEFAULT_SYSTEM_PROMPT
    hits = [{"rank": 1, "filename": "a.txt", "text": "alpha", "rrf_score": 0.03},
            {"rank": 2, "filename": "b.txt", "text": "beta", "ce_score": 1.5}]
    assert srv.format_context_blocks(hits) == prompt.format_context_blocks(hits, max_chars=srv.ANSWER_CONTEXT_CHARS)
    assert srv.build_system_prompt("C") == prompt.build_system_prompt("C")
    for q in ("Who owns the travel policy?", "nothing here at all", "alpha beta gamma delta"):
        assert srv.weak_match(q, hits) == weak_match(q, hits)


# ── CLI and UI ───────────────────────────────────────────────────────────


def test_fts_brain_export_takes_the_layer_flags(corpus_dir: Path, tmp_path: Path,  # noqa: F811
                                               monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from finetune_studio.cli import _registry

    monkeypatch.setattr(sys, "argv", ["fts", "brain", "export", "--corpus", str(corpus_dir), "--no-semantic",
                                      "--no-encrypt", "--out", str(tmp_path)])
    _registry.main()
    out = json.loads(capsys.readouterr().out)
    assert out["layers"]["semantic"] is False and out["pipeline"] == "question → word search (BM25) → passages with sources"
    with tarfile.open(out["path"]) as tf:
        names = tf.getnames()
    assert any(n.endswith("corpus/bm25.json") for n in names) and not any(n.endswith("vectors.npy") for n in names)
    monkeypatch.setattr(sys, "argv", ["fts", "brain", "export", "--corpus", str(corpus_dir), "--no-semantic",
                                      "--no-keyword", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        _registry.main()
    assert "meaning search, word search or both" in capsys.readouterr().err


def test_export_card_has_one_switch_per_layer_and_sends_them() -> None:
    html = (TEMPLATES / "brain.html").read_text(encoding="utf-8")
    for el in ('id="mcp-sem"', 'id="mcp-kw"', 'id="mcp-fusion"', 'id="mcp-rerank"', 'id="mcp-answer"',
               'id="mcp-llm-url"', 'id="mcp-llm-model"', 'id="mcp-pipeline"'):
        assert el in html, el
    for key in ("semantic_search: g('mcp-sem').checked", "keyword_search: g('mcp-kw').checked",
                "answer_enabled: g('mcp-answer').checked", "llm_base_url:", "llm_model:"):
        assert key in html, key
    assert "rr.disabled = !models;" in html                  # the reranker needs its model inside the package


def test_export_settings_route_saves_the_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from finetune_studio import db
    from finetune_studio.webui.app import app
    from finetune_studio.webui.routes import settings as s

    monkeypatch.setattr(s, "SETTINGS_PATH", tmp_path / "settings.json")
    pid = db.create_project(name="layers", description="")["id"]
    client = TestClient(app)
    r = client.put(f"/api/projects/{pid}/brain/export-config",
                   json={"scope": "project", "values": {"semantic_search": False, "answer_enabled": True,
                                                        "llm_base_url": "http://h:1/v1", "llm_model": "m"}})
    assert r.status_code == 200, r.text
    eff = r.json()["effective"]
    assert (eff["semantic_search"], eff["keyword_search"], eff["answer_enabled"]) == (False, True, True)
    bad = client.put(f"/api/projects/{pid}/brain/export-config",
                     json={"scope": "project", "values": {"semantic_search": False, "keyword_search": False}})
    assert bad.status_code == 422
