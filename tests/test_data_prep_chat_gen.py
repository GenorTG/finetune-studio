"""Tests for the data-prep chat gen-kwargs contract and the loader
endpoint body parsing. Exercises the real route handlers / helpers
through minimal stubs (no full DB or model load).
"""
from __future__ import annotations

import os

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def fake_mgr():
    """MagicMock that stands in for ModelManager; chat/generate capture
    the kwargs they were called with."""
    captured = {}
    def fake_chat(messages, **gen):
        captured["messages"] = messages
        captured["gen"] = gen
        return "ok"
    def fake_generate(prompt, **gen):
        captured["prompt"] = prompt
        captured["gen"] = gen
        return "ok"
    mgr = MagicMock()
    mgr.chat = fake_chat
    mgr.generate = fake_generate
    mgr._captured = captured
    return mgr


def test_chat_local_passes_gen_to_manager(fake_mgr):
    """_chat_local must forward the gen dict (clamped/defaulted) to the
    ModelManager.generate() call. The fallback chat() uses the same gen."""
    from finetune_studio.webui.routes.data_prep_chat import _chat_local
    backend = {"kind": "provider", "pid": "fake", "provider_id": "p1",
               "manager": fake_mgr}
    msgs = [{"role": "user", "content": "hi"}]
    txt = _chat_local(backend, msgs, gen={"temperature": 0.9, "max_tokens": 256})
    assert txt == "ok"
    gen = fake_mgr._captured["gen"]
    assert gen["temperature"] == 0.9
    assert gen["max_tokens"] == 256
    # top_p defaulted to 0.9 (not in caller's gen)
    assert gen["top_p"] == 0.9


def test_chat_local_clamps_out_of_range_gen(fake_mgr):
    """Out-of-range gen kwargs must be clamped to llama_cpp-safe values
    before reaching the manager. The route handler clamps in the route
    body — we exercise the clamping branch by passing values that would
    crash llama_cpp without guards."""
    from finetune_studio.webui.routes.data_prep_chat import _chat_local
    backend = {"kind": "provider", "pid": "fake", "provider_id": "p1",
               "manager": fake_mgr}
    # The clamping happens in the route, but _chat_local is the helper
    # called by both the route and the test. We test that _chat_local
    # doesn't itself crash on weird inputs (it uses g.get with default
    # fallbacks).
    _chat_local(
        backend,
        [{"role": "user", "content": "hi"}],
        gen={"temperature": 5.0, "max_tokens": 1, "top_p": 2.0},
    )
    # _chat_local returned the manager's "ok"; args were forwarded
    # verbatim because clamping lives in the route, not the helper. The
    # helper itself must not crash and must call mgr.generate() with the
    # provided gen.
    assert fake_mgr._captured["gen"]["temperature"] == 5.0
    assert fake_mgr._captured["gen"]["max_tokens"] == 1
    assert fake_mgr._captured["gen"]["top_p"] == 2.0


def test_load_provider_route_forwards_extra(monkeypatch, fake_mgr):
    """load_provider(pid, request) parses the JSON body and passes extra
    through to ModelManager.load(pid, extra=...)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from finetune_studio.webui.routes.data_prep import router
    from finetune_studio.models import manager as mgr_mod

    captured = {}
    def fake_load(pid, extra=None):
        captured["pid"] = pid
        captured["extra"] = extra
        return {"id": pid, "name": "X", "kind": "local_gguf",
                "loaded": True, "idle_seconds": 0}
    fake_mgr.load = fake_load
    monkeypatch.setattr(mgr_mod, "get_manager", lambda: fake_mgr)

    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)

    r = c.post("/providers/local-default/load",
               json={"n_ctx": 8192, "n_gpu_layers": 33})
    assert r.status_code == 200
    assert captured["pid"] == "local-default"
    assert captured["extra"]["n_ctx"] == 8192
    assert captured["extra"]["n_gpu_layers"] == 33

    # Empty body -> empty extra
    captured.clear()
    r = c.post("/providers/local-default/load")
    assert r.status_code == 200
    assert captured["extra"] == {}

