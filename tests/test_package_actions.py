"""Actions contracts against actual isolated exported packages and real loopback relay HTTP.

No tunnel, cloud account, external request, LLM or GPU is used. Vendor import is not certified.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from finetune_studio.smart_brain import package_actions as actions
from tests import test_brain_server_package as packages
from tests.test_package_knowledge_api import host, request

PUBLIC = "https://knowledge.example.com/brain"
RELAY_TOKEN = "synthetic-actions-relay-token"


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    return packages.corpus_dir.__wrapped__(tmp_path)


@contextlib.contextmanager
def relay(root: Path, base: str, *, package_token: str = "api-test-token") -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith("RAG_")}
    env.update(RAG_ACTIONS_TOKEN=RELAY_TOKEN, RAG_AUTH_TOKEN=package_token,
               HTTP_PROXY="http://127.0.0.1:1", HTTPS_PROXY="http://127.0.0.1:1")
    with (root / "relay.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-I", str(root / "package_actions.py"),
                                 "--package-url", base, "--public-url", PUBLIC, "--port", str(port)],
                                env=env, stdout=log, stderr=log)
        address = f"http://127.0.0.1:{port}"
        try:
            for _ in range(60):
                try:
                    if request(address, "/openapi.json", token="")[0] == 200:
                        break
                except OSError:
                    if proc.poll() is not None:
                        pytest.fail((root / "relay.log").read_text())
                    time.sleep(.1)
            else:
                pytest.fail("Relay did not start")
            yield address
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def test_schema_contract_and_plan_never_runs_or_contains_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: pytest.fail("No DNS in configuration generation"))
    for tunnel in ("cloudflare", "tailscale"):
        plan = actions.setup_plan(PUBLIC, "http://127.0.0.1:8899", 8900, tunnel)
        assert plan["network_checked"] is False and plan["public_exposure_started"] is False
        schema = plan["schema"]
        assert schema["openapi"] == "3.1.0" and schema["servers"] == [{"url": PUBLIC}]
        assert set(schema["paths"]) == {"/v1/retrieve"}
        op = schema["paths"]["/v1/retrieve"]["post"]
        assert op["operationId"] == "retrieveKnowledge"
        assert op["x-openai-isConsequential"] is False
        assert schema["components"]["securitySchemes"]["packageBearer"]["scheme"] == "bearer"
        assert "RAG_" not in json.dumps(schema)
        assert "127.0.0.1:8900" in plan["tunnel_command"]
        assert "--bg" not in plan["tunnel_command"]
        assert plan["gemini_tool"]["functionDeclarations"][0]["name"] == "retrieveKnowledge"
        assert "additionalProperties" not in plan["gemini_tool"]["functionDeclarations"][0]["parameters"]


@pytest.mark.parametrize("url", [
    "http://example.com", "https://localhost", "https://127.0.0.1", "https://[::1]",
    "https://10.0.0.1", "https://user:secret@example.com", "https://example.com:8443",
    "https://example.com/?token=secret", "https://example.com/#token",
    "https://example.com/a/../qc", "https://example.com/%2fqc", "https://home.local",
    "https://machine", "https://example.com/\n", "https://example.com/';touch",
])
def test_reject_unsafe_public_configuration(url: str) -> None:
    with pytest.raises(ValueError):
        actions.openapi_schema(url)


@pytest.mark.parametrize("url", [
    "http://localhost:8899", "http://example.com:8899", "https://127.0.0.1:8899",
    "http://127.0.0.1:8899/private", "http://u:p@127.0.0.1:8899",
    "http://127.0.0.1:8899?url=elsewhere", "http://127.0.0.1", "http://127.0.0.1:99999",
])
def test_upstream_is_a_literal_loopback_origin(url: str) -> None:
    with pytest.raises(ValueError):
        actions.package_address(url)


def test_query_and_text_budget_preserves_scope_privacy() -> None:
    query = "💡" * 4000
    result = actions.text_response({"hits": [
        {"text": "💡" * 4000, "filename": "private.txt", "document_id": "private-id", "id": "private-chunk",
         "original": "/files/private", "locator": "private", "image_base64": "SECRET", "score": .9}
        for _ in range(8)], "sources_hidden": True}, query)
    assert result["truncated"] and result["sources_hidden"]
    assert len(result["hits"]) == 5 and len(result["hits"][0]["text"]) == 3000
    raw = json.dumps(result, ensure_ascii=False).encode()
    assert len(raw) < actions.MAX_RESPONSE and b"private" not in raw and b"SECRET" not in raw
    for body in ({}, {"query": " "}, {"query": "x", "top_k": True}, {"query": "x", "top_k": 6},
                 {"query": "x", "images": True}, {"query": "x", "source_prefix": "other"},
                 {"query": "x", "llm_base_url": "https://evil.example.com"}, {"query": "x" * 4001}):
        with pytest.raises(ValueError):
            actions.validate_request(body)
    with pytest.raises(ValueError):
        actions.setup_plan(PUBLIC, "http://127.0.0.1:8899", 8899)


@pytest.mark.parametrize(("encrypted", "hidden"), [(False, False), (True, False), (True, True)])
def test_real_export_schema_and_narrow_relay(corpus_dir: Path, tmp_path: Path,
                                           encrypted: bool, hidden: bool) -> None:
    with host(corpus_dir, tmp_path, encrypted=encrypted, hidden=hidden) as (base, root):
        assert (root / "package_actions.py").read_bytes() == Path(actions.__file__).read_bytes()
        assert request(base, "/actions.js", token="")[0] == 200
        query = urllib.parse.urlencode({"public_url": PUBLIC, "package_url": base})
        status, body, _ = request(base, "/connect/actions-plan?" + query, token="")
        assert status == 200 and json.loads(body)["schema"] == actions.openapi_schema(PUBLIC)
        assert request(base, "/connect/actions-plan?public_url=http://localhost", token="")[0] == 400
        with relay(root, base) as address:
            assert request(address, "/openapi.json", token="")[0] == 200
            assert request(address, "/brain/openapi.json", token="")[0] == 200
            for path in ("/", "/health", "/usage/report", "/qc/report", "/mcp", "/files/c2",
                         "/v1/chat/completions", "/v1/retrieve?url=other", "/v1/../qc"):
                assert request(address, path, {}, token=RELAY_TOKEN)[0] == 404
            assert request(address, "/v1/retrieve", {"query": "Highmere"}, token="")[0] == 401
            assert request(address, "/v1/retrieve", {"query": "Highmere"}, token="api-test-token")[0] == 401
            assert request(address, "/v1/retrieve", {"query": "Highmere"}, token=RELAY_TOKEN,
                           origin="https://foreign.example.com")[0] == 403
            status, text, _ = request(address, "/v1/retrieve",
                                     {"query": "Highmere Stair warehousing", "top_k": 3}, token=RELAY_TOKEN,
                                     extra_headers={"X-Usage-Owner": "ignored", "X-Forwarded-For": "ignored"})
            assert status == 200, text
            result = json.loads(text)
            assert "nine crowns" in result["hits"][0]["text"]
            assert "ledger" not in text and "emberfall" not in text
            assert "images" not in result
            assert result["sources_hidden"] is hidden
            if hidden:
                assert all("filename" not in hit and "document_id" not in hit for hit in result["hits"])
            else:
                assert {hit["document_id"] for hit in result["hits"]} == {"c2"}
            status, text, _ = request(address, "/brain/v1/retrieve", {"query": "zzzzzz"}, token=RELAY_TOKEN)
            assert status == 200 and json.loads(text)["hits"] == []
            assert request(address, "/v1/retrieve", {"query": "Highmere", "images": True}, token=RELAY_TOKEN)[0] == 400
            # No owner/QC access or model call was performed; R8 counts ordinary forwarded retrieval.
            assert RELAY_TOKEN not in (root / "relay.log").read_text()


def test_relay_errors_never_follow_redirects_and_rate_limit(tmp_path: Path) -> None:
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization"),
                          "owner": self.headers.get("X-Usage-Owner"),
                          "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/PRIVATE-UPSTREAM-SECRET")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    local = actions.RelayServer(("127.0.0.1", 0), actions.make_handler(
        f"http://127.0.0.1:{backend.server_port}", PUBLIC, RELAY_TOKEN, "private-package-token"))
    runner = threading.Thread(target=local.serve_forever, daemon=True)
    runner.start()
    address = f"http://127.0.0.1:{local.server_port}"
    try:
        status, text, _ = request(address, "/v1/retrieve", {"query": "Highmere"}, token=RELAY_TOKEN,
                                  extra_headers={"X-Usage-Owner": "do-not-forward"})
        assert status == 502 and "PRIVATE" not in text and "private-package-token" not in text
        assert len(calls) == 1 and calls[0]["path"] == "/v1/retrieve"
        assert calls[0]["authorization"] == "Bearer private-package-token"
        assert calls[0]["owner"] is None and calls[0]["body"]["images"] is False
        for _ in range(29):
            assert request(address, "/v1/retrieve", {}, token=RELAY_TOKEN)[0] == 400
        status, _, headers = request(address, "/v1/retrieve", {}, token=RELAY_TOKEN)
        assert status == 429 and headers["Retry-After"] == "60"
    finally:
        local.shutdown()
        local.server_close()
        backend.shutdown()
        backend.server_close()
        runner.join(timeout=5)
        thread.join(timeout=5)


def test_credentials_required_and_studio_ui_contract(client) -> None:
    for relay_token, package_token in (("", "package"), ("short", "package"),
                                       (RELAY_TOKEN, ""), (RELAY_TOKEN, RELAY_TOKEN), (RELAY_TOKEN, "key\nheader")):
        with pytest.raises(ValueError, match="RAG_ACTIONS_TOKEN"):
            actions.make_handler("http://127.0.0.1:8899", PUBLIC, relay_token, package_token)
    response = client.get("/connect/actions-plan", params={"public_url": PUBLIC})
    assert response.status_code == 200 and response.json()["status"] == "configuration_only"
    assert client.get("/connect/actions-plan", params={"public_url": "http://localhost"}).status_code == 400
    page = client.get("/connect").text
    assert 'id="actions-generate"' in page and 'id="actions-download"' in page
    assert 'data-endpoint="/connect/actions-plan"' in page and "/static/js/actions.js?v=1" in page
    from finetune_studio.guide.registry import CONTROLS
    from finetune_studio.guide.settings_help import SETTING_HELP
    assert "connect.actions_generate" in CONTROLS
    assert SETTING_HELP["actions_public_url"]["kb"] == "package-actions"
