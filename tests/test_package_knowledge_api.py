"""R2 contract against real exported servers; deterministic upstream is not model certification."""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from finetune_studio.smart_brain import standalone_server as srv
from finetune_studio.smart_brain.export_config import BrainExportConfig
from finetune_studio.smart_brain.server_package import build_package
from tests import test_brain_server_package as packages
from tests.test_connect import recipe

CHAT = {"model": "smart-brain", "messages": [{"role": "user", "content": "Highmere Stair warehousing cost?"}]}


def request(base: str, path: str, body: object = None, token: str = "api-test-token",
            origin: str = "", extra_headers: dict | None = None) -> tuple[int, str, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if origin:
        headers["Origin"] = origin
    headers.update(extra_headers or {})
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=15)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.code, response.read().decode(), dict(response.headers)


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    return packages.corpus_dir.__wrapped__(tmp_path)


@pytest.fixture()
def upstream() -> Iterator[tuple[str, list[dict]]]:
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
            if body["model"] == "broken":
                status, out = 401, {"error": "private-upstream-key PRIVATE_DOC_TEXT"}
            elif body["model"] == "malformed":
                status, out = 200, {"choices": [{"message": {"content": {"private": "PRIVATE_DOC_TEXT"}}}]}
            elif body["model"] == "empty":
                status, out = 200, {"choices": [{"message": {"content": ""}}]}
            elif self.path.endswith("/messages"):
                status, out = 200, {"content": [{"type": "text", "text": "Nine crowns [highmere.txt]."}]}
            else:
                status, out = 200, {"choices": [{"message": {"content": "Nine crowns [highmere.txt]."}}]}
            data = json.dumps(out).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def host(corpus_dir: Path, tmp_path: Path, *, upstream_url: str = "", provider: str = "openai",
         model: str = "test-model", encrypted: bool = False, hidden: bool = False,
         llm_body: dict | None = None, qc_cases: list[dict] | None = None) -> Iterator[tuple[str, Path]]:
    config = BrainExportConfig(encrypt=encrypted, kdf_log_n=10, hide_file_names=hidden,
                               answer_enabled=bool(upstream_url), llm_base_url=upstream_url, llm_model=model,
                               llm_provider=provider, semantic_search=False, reranker_enabled=False,
                               llm_body=llm_body or {})
    archive = build_package(corpus_dir, tmp_path / "api.tar.gz", config=config,
                            passphrase="synthetic-r2-passphrase" if encrypted else None,
                            documents={"c2"}, scope={"label": "Highmere only"}, qc_cases=qc_cases)
    root = packages._extract(archive.path, tmp_path / "export")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # Clear inherited operator overrides: the test owns this exported package and only this process.
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAG_")}
    env.update(RAG_AUTH_TOKEN="api-test-token", RAG_LLM_API_KEY="private-upstream-key", RAG_DEVICE="cpu",
               RAG_FOLLOW_UP="0")
    if encrypted:
        env["RAG_PASSPHRASE"] = "synthetic-r2-passphrase"
    with (root / "http.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-I", str(root / "server.py"), "--http", str(port)],
                                   env=env, stdout=subprocess.DEVNULL, stderr=log)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(80):
                try:
                    if request(base, "/v1/models")[0] == 200:
                        break
                except OSError:
                    if process.poll() is not None:
                        pytest.fail((root / "http.log").read_text())
                    time.sleep(0.1)
            else:
                pytest.fail("exported API did not start")
            yield base, root
        finally:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.parametrize("encrypted", [False, True])
def test_real_export_auth_scoped_retrieval_and_honest_no_provider(corpus_dir: Path, tmp_path: Path,
                                                               encrypted: bool) -> None:
    with host(corpus_dir, tmp_path, encrypted=encrypted) as (base, root):
        for route, body in (("/v1/models", None), ("/v1/retrieve", {"query": "Highmere"}),
                            ("/v1/chat/completions", CHAT)):
            assert request(base, route, body, token="")[0] == 401
            assert request(base, route, body, token="wrong")[0] == 401
        models = json.loads(request(base, "/v1/models")[1])
        assert [m["id"] for m in models["data"]] == ["smart-brain"]
        status, text, _ = request(base, "/v1/retrieve", {"query": "Highmere Stair salt", "images": True})
        assert status == 200
        result = json.loads(text)
        assert [h["document_id"] for h in result["hits"]] == ["c2"]
        assert result["hits"][0]["score"] > 0
        assert "nine crowns" in result["hits"][0]["text"]
        assert "ledger.txt" not in text and "emberfall.txt" not in text
        assert result["images"] == []
        for stream in (False, True):
            status, text, _ = request(base, "/v1/chat/completions", {**CHAT, "stream": stream})
            assert status == 503 and json.loads(text)["error"]["code"] == "llm_not_configured"
        assert request(base, "/v1/retrieve", {"query": "Highmere"}, origin="https://evil.example")[0] == 403
        assert request(base, "/v1/retrieve", {"query": "Highmere"}, origin="http://localhost:6666")[0] == 403
        assert request(base, "/v1/retrieve", {"query": "Highmere"}, origin=base)[0] == 200
        assert request(base, "/v1/nope", {"query": "Highmere"})[0] == 404
        assert "private-upstream-key" not in (root / "README.md").read_text()
        assert "/v1/retrieve" in (root / "README.md").read_text()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_exported_completion_reuses_operator_adapter_and_sse(corpus_dir: Path, tmp_path: Path,
                                                            upstream: tuple[str, list[dict]], provider: str) -> None:
    url, calls = upstream
    with host(corpus_dir, tmp_path, upstream_url=url, provider=provider) as (base, _):
        body = {**CHAT, "messages": [{"role": "system", "content": "Be concise."},
                                    {"role": "user", "content": "We discussed storage."},
                                    {"role": "assistant", "content": "Understood."}, *CHAT["messages"]],
                "max_completion_tokens": 64, "temperature": 0.3, "top_p": 0.9}
        status, text, _ = request(base, "/v1/chat/completions", body)
        assert status == 200, text
        result = json.loads(text)
        assert result["object"] == "chat.completion" and result["model"] == "smart-brain"
        assert "Nine crowns" in result["choices"][0]["message"]["content"]
        assert result["knowledge"]["generated"] is True
        assert result["knowledge"]["sources"][0]["filename"] == "highmere.txt"
        assert "usage" not in result
        sent = calls[-1]["body"]
        assert sent["model"] == "test-model" and sent["max_tokens"] == 64 and sent["temperature"] == 0.3
        assert sent["top_p"] == 0.9
        assert "nine crowns" in json.dumps(sent) and "Be concise." in json.dumps(sent)
        assert "We discussed storage." in json.dumps(sent)
        assert calls[-1]["path"] == ("/v1/messages" if provider == "anthropic" else "/v1/chat/completions")
        status, text, headers = request(base, "/v1/chat/completions", {**CHAT, "stream": True})
        assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
        events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert len({c["id"] for c in chunks}) == 1
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["knowledge"]["sources"]
        # A retrieval miss must be labelled as such, with no upstream call or fake model answer.
        count = len(calls)
        status, text, _ = request(base, "/v1/chat/completions", {**CHAT, "messages": [{
            "role": "user", "content": "unmatchablexyz"}]})
        assert status == 200 and json.loads(text)["knowledge"]["generated"] is False
        assert len(calls) == count


@pytest.mark.parametrize("model", ["broken", "malformed", "empty"])
def test_upstream_error_does_not_leak_private_material(corpus_dir: Path, tmp_path: Path,
                                                      upstream: tuple[str, list[dict]], model: str) -> None:
    with host(corpus_dir, tmp_path, upstream_url=upstream[0], model=model) as (base, _):
        status, text, _ = request(base, "/v1/chat/completions", CHAT)
        assert status == 502 and json.loads(text)["error"]["code"] == "upstream_error"
        assert "PRIVATE_DOC_TEXT" not in text and "private-upstream-key" not in text
        assert upstream[0] not in text


def test_hidden_filenames_are_not_a_client_override(corpus_dir: Path, tmp_path: Path,
                                                   upstream: tuple[str, list[dict]]) -> None:
    with host(corpus_dir, tmp_path, upstream_url=upstream[0], hidden=True) as (base, _):
        status, text, _ = request(base, "/v1/retrieve", {"query": "Highmere"})
        assert status == 200 and "filename" not in text and "document_id" not in text
        assert json.loads(text)["sources_hidden"] is True
        status, text, _ = request(base, "/v1/chat/completions", CHAT)
        assert status == 200 and "highmere.txt" not in text
        assert json.loads(text)["knowledge"]["sources"] == []
        assert request(base, "/v1/chat/completions", {**CHAT, "include_sources": True})[0] == 400


@pytest.mark.parametrize("patch", [
    {"model": "arbitrary-upstream"}, {"messages": []}, {"messages": "wrong"}, {"stream": "true"}, {"n": 2},
    {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    {"messages": [{"role": "tool", "content": "result"}]},
    {"messages": [{"role": [], "content": "invalid"}]},
    {"messages": [{"role": "assistant", "content": "wrong final turn"}]},
    {"messages": [{"role": "user", "content": "x" * 16001}]},
    {"messages": CHAT["messages"] * 33}, {"tools": []}, {"llm_base_url": "http://evil.example"},
    {"max_tokens": 513}, {"max_tokens": 0}, {"max_tokens": True}, {"temperature": float("nan")},
    {"top_p": 2}, {"max_tokens": 8, "max_completion_tokens": 8}, {"stream_options": {"include_usage": True}},
])
def test_completion_validation(patch: dict) -> None:
    with pytest.raises(ValueError):
        srv.CompletionRequest.parse({**CHAT, **patch}, srv.DEFAULT_CONFIG)


def test_retrieval_validation(corpus_dir: Path, tmp_path: Path) -> None:
    with host(corpus_dir, tmp_path) as (base, _):
        for body in ([], {"query": []}, {"query": ""}, {"query": "x", "top_k": True},
                     {"query": "x", "top_k": 101}, {"query": "x", "images": "yes"},
                     {"query": "x", "documents": ["c1"]}, {"query": "x", "source_prefix": "ledger"}):
            status, text, _ = request(base, "/v1/retrieve", body)
            assert status == 400 and isinstance(json.loads(text)["error"], dict)


def test_ui_recipe_drives_models_retrieve_and_completions() -> None:
    result = recipe("knowledge-api", address="https://brain.example/team", listener_port="9010")
    assert result["availableSetup"] is True
    assert "https://brain.example/team/v1" in result["config"]
    assert "RAG_LLM_BASE_URL" in result["config"] and "--answer" in result["start"]
    assert "--port 9010" in result["start"]
    for route in ("models", "retrieve", "chat/completions"):
        assert "https://brain.example/team/v1/" + route in result["test"]
    root = Path(__file__).resolve().parents[1] / "src/finetune_studio"
    for path in ("webui/templates/connect.html", "smart_brain/package_connect.html"):
        assert 'value="knowledge-api"' in (root / path).read_text()
    script = (root / "smart_brain/package_connect.js").read_text()
    assert "package-retrieve" in script and "package-answer" in script
    assert "v1/chat/completions" in script and "v1/retrieve" in script


@pytest.mark.parametrize("token_key", ["max_tokens", "max_completion_tokens"])
def test_client_budget_overrides_static_provider_body(corpus_dir: Path, tmp_path: Path,
                                                     upstream: tuple[str, list[dict]], token_key: str) -> None:
    corpus = packages._in_process_corpus(corpus_dir, tmp_path)
    corpus.cfg.update(answer=True, llm_base_url=upstream[0], llm_model="test-model", llm_body={token_key: 400})
    corpus._chat(CHAT["messages"], {"answer_max_tokens": 32})
    sent = upstream[1][-1]["body"]
    assert sent[token_key] == 32
    assert ("max_tokens" in sent) is (token_key == "max_tokens")


def test_sampling_does_not_replace_the_operator_static_budget(corpus_dir: Path, tmp_path: Path,
                                                            upstream: tuple[str, list[dict]]) -> None:
    with host(corpus_dir, tmp_path, upstream_url=upstream[0], llm_body={"max_tokens": 128}) as (base, _):
        assert request(base, "/v1/chat/completions", {**CHAT, "temperature": 0.5})[0] == 200
        assert upstream[1][-1]["body"]["max_tokens"] == 128
