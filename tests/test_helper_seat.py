"""The helper seat: local GGUF or an API provider, key write-only, remote calls hardened.

A fake OpenAI-compatible gateway (stdlib server) stands in for OpenCode Go: it rejects requests that lack
``x-opencode-session`` (the real gateway does), can fail once with 503, and can answer with inline
``<think>`` blocks or a reasoning-only ``length`` stop.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from finetune_studio.models import helper
from finetune_studio.models.providers import OpenAICompatProvider, ProviderConfig, ProviderError

KEY = "sk-test-secret-123456"


class _Gateway:
    def __init__(self) -> None:
        self.mode = "ok"
        self.fail_once = False
        self.seen: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:  # silence
                pass

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self._send(401, {"error": "bad key"})
                self._send(200, {"data": [{"id": "model-b"}, {"id": "model-a"}]})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append({"headers": dict(self.headers), "body": body})
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self._send(401, {"error": "bad key"})
                if not self.headers.get("x-opencode-session"):
                    return self._send(400, {"error": {"type": "MissingSessionID"}})
                if outer.fail_once:
                    outer.fail_once = False
                    return self._send(503, {"error": "busy"})
                if outer.mode == "think":
                    msg, finish = "<think>hmm</think>\npong", "stop"
                elif outer.mode == "reasoning_only":
                    msg, finish = None, "length"
                else:
                    msg, finish = "pong", "stop"
                self._send(200, {"choices": [{"message": {"content": msg}, "finish_reason": finish}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def gateway() -> Iterator[_Gateway]:
    gw = _Gateway()
    yield gw
    gw.server.shutdown()


def _provider(gw: _Gateway, **extra: Any) -> OpenAICompatProvider:
    cfg = ProviderConfig(id="api-helper", name="x", kind="openai_compat", model_id="m", base_url=gw.url,
                         api_key=KEY, extra={"headers": {"x-opencode-session": "{session}"}, "retries": 2, **extra})
    return OpenAICompatProvider(cfg)


def test_session_header_is_expanded_and_one_per_provider(gateway: _Gateway) -> None:
    p = _provider(gateway)
    assert p.chat([{"role": "user", "content": "hi"}]) == "pong"
    p.chat([{"role": "user", "content": "again"}])
    sessions = {s["headers"]["x-opencode-session"] for s in gateway.seen}
    assert len(sessions) == 1 and "{session}" not in sessions.pop()


def test_503_is_retried_then_succeeds(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("finetune_studio.models.providers.time.sleep", lambda s: None)
    gateway.fail_once = True
    assert _provider(gateway).chat([{"role": "user", "content": "hi"}]) == "pong"
    assert len(gateway.seen) == 2


def test_error_message_names_status_and_never_the_key(gateway: _Gateway) -> None:
    cfg = ProviderConfig(id="x", name="x", kind="openai_compat", model_id="m", base_url=gateway.url,
                         api_key="wrong-key-999", extra={"retries": 0})
    with pytest.raises(ProviderError) as exc:
        OpenAICompatProvider(cfg).chat([{"role": "user", "content": "hi"}])
    assert "HTTP 401" in str(exc.value) and "wrong-key-999" not in str(exc.value)


def test_inline_think_block_is_stripped(gateway: _Gateway) -> None:
    gateway.mode = "think"
    assert _provider(gateway).chat([{"role": "user", "content": "hi"}]) == "pong"


def test_reasoning_only_length_stop_raises_actionable_error(gateway: _Gateway) -> None:
    gateway.mode = "reasoning_only"
    with pytest.raises(ProviderError, match="reasoning effort"):
        _provider(gateway).chat([{"role": "user", "content": "hi"}])


def test_extra_body_is_merged_into_requests(gateway: _Gateway) -> None:
    _provider(gateway, body={"reasoning_effort": "low"}).chat([{"role": "user", "content": "hi"}])
    assert gateway.seen[0]["body"]["reasoning_effort"] == "low"


# ── seat + settings routes ─────────────────────────────────────────────────────

def _put(client: Any, gw: _Gateway, **over: Any) -> Any:
    body = {"seat": "api", "preset": "opencode-go", "base_url": gw.url, "model": "deepseek-v4-flash",
            "api_key": KEY, **over}
    return client.put("/api/settings/helper", json=body)


def test_default_seat_is_the_local_gguf(client: Any) -> None:
    st = client.get("/api/settings/helper").json()
    assert st["seat"] == "local" and st["seat_provider_id"] == helper.DEFAULT_HELPER_PROVIDER_ID
    assert not st["api"]["configured"]


def test_seating_the_api_helper_and_key_is_never_returned(client: Any, gateway: _Gateway) -> None:
    r = _put(client, gateway)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["seat"] == "api" and st["api"]["key_set"] and st["api"]["model"] == "deepseek-v4-flash"
    assert st["loaded_provider_id"] == helper.API_HELPER_PROVIDER_ID  # loads instantly, no VRAM
    for url in ("/api/settings/helper", "/api/providers"):
        assert KEY not in client.get(url).text
    assert helper.get_helper_provider_id() == helper.API_HELPER_PROVIDER_ID
    assert "Helper · API · deepseek-v4-flash" in (helper.get_configured_helper_provider() or {})["label"]


def test_seated_api_helper_really_chats_with_its_stored_headers(client: Any, gateway: _Gateway) -> None:
    """Regression: a freshly inserted provider row lost ``extra`` (headers/body) — the gateway then refused it."""
    from finetune_studio.models.manager import get_manager
    _put(client, gateway, reasoning_effort="low")
    assert get_manager().chat([{"role": "user", "content": "hi"}]) == "pong"
    sent = gateway.seen[-1]
    assert sent["headers"]["x-opencode-session"] and sent["body"]["reasoning_effort"] == "low"
    assert client.get("/api/settings/helper").json()["api"]["preset"] == "opencode-go"


def test_seat_follows_into_helper_consumers(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.data.prep.generator import resolve_helper_backend
    _put(client, gateway)
    backend = resolve_helper_backend()
    assert backend and backend["provider_id"] == helper.API_HELPER_PROVIDER_ID
    assert client.get("/api/providers").json()["helper_provider_id"] == helper.API_HELPER_PROVIDER_ID


def test_api_seat_requires_a_key(client: Any, gateway: _Gateway) -> None:
    r = _put(client, gateway, api_key="")
    assert r.status_code == 400 and "API key" in r.json()["detail"]
    assert client.get("/api/settings/helper").json()["seat"] == "local"


def test_blank_key_keeps_the_stored_key_and_clear_removes_it(client: Any, gateway: _Gateway) -> None:
    _put(client, gateway)
    assert _put(client, gateway, api_key="", model="deepseek-v4.1-flash").json()["api"]["key_set"]
    cleared = client.put("/api/settings/helper", json={"seat": "local", "clear_key": True, "base_url": gateway.url,
                                                       "model": "x"}).json()
    assert not cleared["api"]["key_set"] and cleared["seat"] == "local"


def test_back_to_local_restores_the_gguf_seat(client: Any, gateway: _Gateway) -> None:
    _put(client, gateway)
    st = client.put("/api/settings/helper", json={"seat": "local"}).json()
    assert st["seat_provider_id"] == helper.DEFAULT_HELPER_PROVIDER_ID and st["loaded_provider_id"] == ""


def test_deleted_seat_row_falls_back_to_local(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models.manager import get_manager
    _put(client, gateway)
    get_manager().delete_provider(helper.API_HELPER_PROVIDER_ID)
    assert helper.get_helper_provider_id() == helper.DEFAULT_HELPER_PROVIDER_ID


@pytest.mark.parametrize("over,msg", [({"base_url": "ftp://x"}, "http"), ({"model": ""}, "model"),
                                      ({"reasoning_effort": "extreme"}, "reasoning_effort")])
def test_validation(client: Any, gateway: _Gateway, over: dict[str, Any], msg: str) -> None:
    r = _put(client, gateway, **over)
    assert r.status_code == 400 and msg in r.json()["detail"]


def test_connection_test_reports_latency_and_models_list(client: Any, gateway: _Gateway) -> None:
    body = {"preset": "opencode-go", "base_url": gateway.url, "model": "m", "api_key": KEY}
    ok = client.post("/api/settings/helper/test", json=body)
    assert ok.status_code == 200 and ok.json()["reply"] == "pong" and ok.json()["latency_ms"] >= 0
    assert client.post("/api/settings/helper/models", json=body).json()["models"] == ["model-a", "model-b"]
    bad = client.post("/api/settings/helper/test", json={**body, "api_key": "nope"})
    assert bad.status_code == 502 and "HTTP 401" in bad.json()["error"] and "nope" not in bad.text
