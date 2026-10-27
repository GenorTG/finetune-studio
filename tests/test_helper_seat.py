"""The AI assistant: connections (local GGUF + API, keys write-only), jobs, and hardened remote calls.

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
from finetune_studio.models.providers import (
    OpenAICompatProvider,
    ProviderConfig,
    ProviderError,
)

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

            def do_GET(self) -> None:
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    return self._send(401, {"error": "bad key"})
                self._send(200, {"data": [{"id": "model-b"}, {"id": "model-a"}]})

            def do_POST(self) -> None:
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
                elif outer.mode == "dsml":
                    if body.get("tools"):
                        return self._send(200, {"choices": [{"finish_reason": "tool_calls", "message": {
                            "content": None, "tool_calls": [{"function": {
                                "name": "project_overview", "arguments": "{}"}}]}}]})
                    msg, finish = "<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name=\"project_overview\">", "stop"
                elif outer.mode == "reasoning_then_answer":
                    big = body.get("max_tokens", 0) >= 4000
                    msg, finish = ("pong", "stop") if big else (None, "length")
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


def test_reasoning_budget_retry_succeeds_with_a_bigger_budget(gateway: _Gateway) -> None:
    gateway.mode = "reasoning_then_answer"
    assert _provider(gateway).chat([{"role": "user", "content": "hi"}], max_tokens=1200) == "pong"
    assert [s["body"]["max_tokens"] for s in gateway.seen] == [1200, 4800]


def test_extra_body_is_merged_into_requests(gateway: _Gateway) -> None:
    _provider(gateway, body={"reasoning_effort": "low"}).chat([{"role": "user", "content": "hi"}])
    assert gateway.seen[0]["body"]["reasoning_effort"] == "low"


# ── AI assistant: connections, main assistant, jobs (Settings -> AI assistant) ──

def _create(client: Any, gw: _Gateway, **over: Any) -> Any:
    body = {"name": "OpenCode", "preset": "opencode-go", "base_url": gw.url, "model": "deepseek-v4-flash",
            "api_key": KEY, **over}
    return client.post("/api/settings/connections", json=body)


def _make_assistant(client: Any, gw: _Gateway, **over: Any) -> str:
    r = _create(client, gw, **over)
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert client.put("/api/settings/assistant", json={"provider_id": pid}).status_code == 200
    return pid


def test_default_assistant_is_the_local_gguf(client: Any) -> None:
    st = client.get("/api/settings/assistant").json()
    assert st["assistant_provider_id"] == helper.DEFAULT_HELPER_PROVIDER_ID
    assert [c["id"] for c in st["connections"]] == [helper.DEFAULT_HELPER_PROVIDER_ID]
    jobs = {j["id"]: j for j in st["jobs"]}
    assert jobs["mining"]["setting"] == "" and jobs["mining"]["provider_id"] == helper.DEFAULT_HELPER_PROVIDER_ID
    assert jobs["vision"]["setting"] == "off"                      # pictures leave the machine only on purpose


def test_api_assistant_key_is_never_returned_and_nothing_is_loaded(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models.manager import get_manager
    pid = _make_assistant(client, gateway)
    st = client.get("/api/settings/assistant").json()
    conn = next(c for c in st["connections"] if c["id"] == pid)
    assert conn["key_set"] and conn["key_state"] != "unset" and conn["is_assistant"] and conn["model"] == "deepseek-v4-flash"
    for url in ("/api/settings/assistant", "/api/providers"):
        assert KEY not in client.get(url).text
    assert helper.get_helper_provider_id() == pid
    assert (helper.get_configured_helper_provider() or {})["label"] == "OpenCode · deepseek-v4-flash"
    assert get_manager().active() is None                          # an API connection takes no load slot


def test_api_assistant_really_chats_with_its_stored_headers(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.data.prep.generator import resolve_helper_backend
    _make_assistant(client, gateway, reasoning_effort="low")
    backend = resolve_helper_backend()
    assert backend and backend["manager"].chat([{"role": "user", "content": "hi"}]) == "pong"
    sent = gateway.seen[-1]
    assert sent["headers"]["x-opencode-session"] and sent["body"]["reasoning_effort"] == "low"


def test_an_update_without_the_effort_keeps_it_and_an_empty_one_clears_it(client: Any, gateway: _Gateway) -> None:
    """Regression (2026-10-12): an update that omitted reasoning_effort dropped it (mining 7x slower)."""
    pid = _make_assistant(client, gateway, reasoning_effort="none")
    st = client.put(f"/api/settings/connections/{pid}", json={"model": "deepseek-v4.1-flash"}).json()
    conn = next(c for c in st["connections"] if c["id"] == pid)
    assert conn["reasoning_effort"] == "none" and conn["model"] == "deepseek-v4.1-flash" and conn["key_set"]
    st = client.put(f"/api/settings/connections/{pid}", json={"reasoning_effort": ""}).json()
    assert next(c for c in st["connections"] if c["id"] == pid)["reasoning_effort"] == ""


def test_the_assistant_follows_into_helper_consumers(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.data.prep.generator import resolve_helper_backend
    pid = _make_assistant(client, gateway)
    assert resolve_helper_backend()["provider_id"] == pid
    providers = client.get("/api/providers").json()
    assert providers["helper_provider_id"] == pid and providers["jobs"]["mining"]["provider_id"] == pid


def test_a_new_connection_needs_a_key_and_a_keyless_one_cannot_be_the_assistant(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models.manager import get_manager
    r = _create(client, gateway, api_key="")
    assert r.status_code == 400 and "API key" in r.json()["detail"]
    get_manager().upsert_provider(id="api-nokey", name="No key", kind="openai_compat", model_id="m",
                                  base_url=gateway.url)
    r = client.put("/api/settings/assistant", json={"provider_id": "api-nokey"})
    assert r.status_code == 400 and "no saved API key" in r.json()["detail"]
    assert client.get("/api/settings/assistant").json()["assistant_provider_id"] == helper.DEFAULT_HELPER_PROVIDER_ID


def test_blank_key_keeps_the_stored_key_and_clear_removes_it(client: Any, gateway: _Gateway) -> None:
    pid = _create(client, gateway).json()["id"]
    st = client.put(f"/api/settings/connections/{pid}", json={"api_key": "", "model": "x"}).json()
    assert next(c for c in st["connections"] if c["id"] == pid)["key_set"]
    st = client.put(f"/api/settings/connections/{pid}", json={"clear_key": True}).json()
    assert not next(c for c in st["connections"] if c["id"] == pid)["key_set"]


def test_the_assistants_key_and_row_cannot_be_removed_until_another_takes_over(client: Any, gateway: _Gateway) -> None:
    pid = _make_assistant(client, gateway)
    assert client.put(f"/api/settings/connections/{pid}", json={"clear_key": True}).status_code == 409
    assert client.delete(f"/api/settings/connections/{pid}").status_code == 409
    client.put("/api/settings/assistant", json={"provider_id": helper.DEFAULT_HELPER_PROVIDER_ID})
    assert client.delete(f"/api/settings/connections/{pid}").status_code == 200
    assert client.delete(f"/api/settings/connections/{helper.DEFAULT_HELPER_PROVIDER_ID}").status_code == 400


def test_a_saved_key_only_goes_to_its_own_url(client: Any, gateway: _Gateway) -> None:
    pid = _create(client, gateway).json()["id"]
    r = client.put(f"/api/settings/connections/{pid}", json={"base_url": "https://elsewhere.invalid/v1"})
    assert r.status_code == 400 and "type the API key again" in r.json()["detail"]
    r = client.post("/api/settings/connections/test", json={"id": pid, "base_url": "https://elsewhere.invalid/v1"})
    assert r.status_code == 400 and "only sent to its own URL" in r.json()["detail"]


def test_deleted_assistant_row_falls_back_to_local(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models.manager import get_manager
    pid = _make_assistant(client, gateway)
    get_manager().delete_provider(pid)
    assert helper.get_helper_provider_id() == helper.DEFAULT_HELPER_PROVIDER_ID


@pytest.mark.parametrize("over,msg", [({"base_url": "ftp://x"}, "http"), ({"model": ""}, "model"),
                                      ({"reasoning_effort": "extreme"}, "reasoning_effort"),
                                      ({"name": ""}, "name"), ({"concurrency": 99}, "concurrency")])
def test_validation(client: Any, gateway: _Gateway, over: dict[str, Any], msg: str) -> None:
    r = _create(client, gateway, **over)
    assert r.status_code == 400 and msg in r.json()["detail"]


def test_connection_test_reports_latency_and_models_list(client: Any, gateway: _Gateway) -> None:
    body = {"preset": "opencode-go", "base_url": gateway.url, "model": "m", "api_key": KEY}
    ok = client.post("/api/settings/connections/test", json=body)
    assert ok.status_code == 200 and ok.json()["reply"] == "pong" and ok.json()["latency_ms"] >= 0
    assert client.post("/api/settings/connections/models", json=body).json()["models"] == ["model-a", "model-b"]
    bad = client.post("/api/settings/connections/test", json={**body, "api_key": "nope"})
    assert bad.status_code == 502 and "HTTP 401" in bad.json()["detail"] and "nope" not in bad.text
    pid = _create(client, gateway).json()["id"]            # a saved connection tests with its stored key
    assert client.post("/api/settings/connections/test", json={"id": pid}).json()["reply"] == "pong"


def test_local_seat_is_never_loaded_implicitly(client: Any) -> None:
    from finetune_studio.data.prep.generator import resolve_helper_backend
    assert resolve_helper_backend() is None


def test_a_job_can_run_on_its_own_connection_and_falls_back_when_it_is_removed(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models import assistant
    pid = _create(client, gateway).json()["id"]
    st = client.put("/api/settings/assistant/jobs/judge", json={"provider_id": pid}).json()
    judge = next(j for j in st["jobs"] if j["id"] == "judge")
    assert judge["setting"] == pid and judge["provider_id"] == pid
    assert assistant.job_provider_id("mining") == helper.DEFAULT_HELPER_PROVIDER_ID   # others stay on the assistant
    assert next(c for c in st["connections"] if c["id"] == pid)["used_by"] == ["Test judge"]
    client.delete(f"/api/settings/connections/{pid}")
    assert assistant.job_setting("judge") == "" and assistant.job_provider_id("judge") == helper.DEFAULT_HELPER_PROVIDER_ID


def test_job_settings_are_validated(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models.manager import get_manager
    assert client.put("/api/settings/assistant/jobs/nope", json={}).status_code == 404
    assert client.put("/api/settings/assistant/jobs/mining", json={"provider_id": "off"}).status_code == 400
    assert client.put("/api/settings/assistant/jobs/mining", json={"provider_id": "gone"}).status_code == 400
    assert client.put("/api/settings/assistant/jobs/mining", json={"instructions": "x" * 4001}).status_code == 400
    get_manager().upsert_provider(id="local-two", name="Two", kind="local_gguf", model_id="/m/two.gguf")
    r = client.put("/api/settings/assistant/jobs/teacher", json={"provider_id": "local-two"})
    assert r.status_code == 400 and "API connection" in r.json()["detail"]


def test_job_instructions_reach_the_model_as_a_system_message(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.data.prep.generator import resolve_generator
    from finetune_studio.models import assistant
    _make_assistant(client, gateway)
    client.put("/api/settings/assistant/jobs/quiz", json={"instructions": "Write the questions in Polish."})
    assert assistant.job_instructions("quiz") == "Write the questions in Polish."
    assert resolve_generator("quiz")([{"role": "user", "content": "reword this"}]) == "pong"
    sent = gateway.seen[-1]["body"]["messages"]
    assert sent[0]["role"] == "system" and "Write the questions in Polish." in sent[0]["content"]
    resolve_generator("study")([{"role": "user", "content": "x"}])       # another job: no extra system message
    assert gateway.seen[-1]["body"]["messages"][0]["role"] == "user"
    assert assistant.instructions_tag("quiz") and not assistant.instructions_tag("study")


def test_a_judge_saved_by_an_older_version_moves_into_the_judge_job(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.models import assistant
    from finetune_studio.webui.routes import settings as settings_routes
    pid = _create(client, gateway).json()["id"]
    settings_routes._save({**settings_routes._load(), settings_routes._TEST_JUDGE_PROVIDER: pid})
    assert settings_routes.get_test_settings()["judge_provider_id"] == pid
    assert assistant.job_setting("judge") == pid
    assert settings_routes._TEST_JUDGE_PROVIDER not in settings_routes._load()


def test_declared_tools_come_back_as_the_loops_text_protocol(gateway: _Gateway) -> None:
    gateway.mode = "dsml"
    p = _provider(gateway)
    messages = [{"role": "user", "content": "hi"}]
    assert "DSML" in p.chat(messages)  # undeclared: the model's own dialect leaks as plain text
    out = p.chat(messages, tools=[{"name": "project_overview", "description": "d", "parameters": {"type": "object"}}])
    assert out == '<tool_call>{"name": "project_overview", "arguments": {}}</tool_call>'
    assert gateway.seen[-1]["body"]["tool_choice"] == "auto"


def test_guide_chat_declares_tools_only_for_api_connections(client: Any, gateway: _Gateway) -> None:
    from finetune_studio.data.prep.generator import resolve_helper_backend
    from finetune_studio.webui.chat_backend import _chat_local
    gateway.mode = "dsml"
    _make_assistant(client, gateway)
    reply = _chat_local({"manager": resolve_helper_backend()["manager"]}, [{"role": "user", "content": "hi"}],
                        {"max_tokens": 64})
    assert "<tool_call>" in reply and "DSML" not in reply


def _local_seat_at(path: str) -> None:
    from finetune_studio.models.manager import get_manager
    get_manager().upsert_provider(id=helper.DEFAULT_HELPER_PROVIDER_ID, name="x", kind="local_gguf", model_id=path)


def test_local_seat_test_loads_the_gguf_not_the_api(client: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test connection on the local seat loads the GGUF, asks it, and unloads it again; the API row is not used."""
    from finetune_studio.models.manager import get_manager
    gguf = tmp_path / "helper.gguf"
    gguf.write_bytes(b"GGUF")
    _local_seat_at(str(gguf))
    mgr, calls = get_manager(), []
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine.model", None, raising=False)
    monkeypatch.setattr(mgr, "load", lambda pid, extra=None: calls.append(("load", pid)) or {"id": pid})
    monkeypatch.setattr(mgr, "chat", lambda messages, **gen: calls.append(("chat",)) or " pong\n")
    monkeypatch.setattr(mgr, "unload", lambda: calls.append(("unload",)))
    r = client.post("/api/settings/helper/test")
    assert r.status_code == 200 and r.json()["reply"] == "pong" and "load_ms" in r.json()
    assert calls == [("load", helper.DEFAULT_HELPER_PROVIDER_ID), ("chat",), ("unload",)]


def test_local_seat_test_reports_a_missing_file_and_a_busy_gpu(client: Any, tmp_path: Any,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _local_seat_at(str(tmp_path / "gone.gguf"))
    r = client.post("/api/settings/helper/test")
    assert r.status_code == 400 and r.json()["code"] == "helper_missing"
    gguf = tmp_path / "helper.gguf"
    gguf.write_bytes(b"GGUF")
    _local_seat_at(str(gguf))
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine.model", object(), raising=False)
    monkeypatch.setattr("finetune_studio.webui.app.inference_engine.model_path", "/m/chat-model.gguf", raising=False)
    r = client.post("/api/settings/helper/test")
    assert r.status_code == 409 and "chat-model.gguf" in r.json()["detail"]
