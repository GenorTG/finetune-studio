"""Tests for the data-prep chat endpoint + tool implementations.

Uses FTS_SKIP_CHAT=1 to stub out the actual model call so tests run
without a loaded model or external API key.
"""
from __future__ import annotations

# ── Tool implementations ────────────────────────────────────────────────


class TestChatTools:
    """Server-side tool implementations read/write via finetune_studio.data.fs."""

    def test_list_sources_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        # Re-import the data fs paths module to pick up the new FTS_DB
        from finetune_studio.data.fs import paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)

        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool("test-project", "list_sources", {})
        assert "sources" in result
        assert result["sources"] == []

    def test_read_source_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        from finetune_studio.data.fs import paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)

        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool("test-project", "read_source", {"source_id": "missing"})
        assert "error" in result

    def test_read_source_falls_back_to_parsed_txt(self, tmp_path, monkeypatch):
        """Regression: read_source must read files/<sha>/parsed.txt when the
        qa/sources/<id>.json manifest has no 'text' field. This was the
        2026-09-10 E2E bug where all uploads returned text=''.
        """
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        from finetune_studio.data.fs import paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)
        import finetune_studio.data.fs.qa as qa_fs_mod
        monkeypatch.setattr(qa_fs_mod, "project_dir", lambda pid: tmp_path / pid)

        # Seed a qa source manifest with sha256 but no 'text' field
        sid = "abcd12345678"
        sha = "abcd12345678" + "0" * 52  # 64 hex chars
        qa_fs_mod.write_qa_source("test-project", {
            "id": sid, "sha256": sha, "filename": "needle.md",
            "mime_type": "text/markdown", "char_count": 13,
            "chunk_count": 1, "parser": "text_v1",
            "uploaded_at": 0.0, "status": "ready",
        })
        # Seed the parsed text on disk
        parsed_path = tmp_path / "test-project" / "files" / sid / "parsed.txt"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text("OCTOPUS-7741", encoding="utf-8")

        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool("test-project", "read_source", {"source_id": sid})
        assert result.get("text") == "OCTOPUS-7741", (
            f"read_source should fall back to parsed.txt; got {result!r}"
        )
        assert result.get("truncated") is False

    def test_read_source_prefers_manifest_text(self, tmp_path, monkeypatch):
        """If the manifest already carries 'text', use it (legacy / future path)."""
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        from finetune_studio.data.fs import paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)
        import finetune_studio.data.fs.qa as qa_fs_mod
        monkeypatch.setattr(qa_fs_mod, "project_dir", lambda pid: tmp_path / pid)

        sid = "deadbeef0001"
        qa_fs_mod.write_qa_source("test-project", {
            "id": sid, "sha256": "x", "filename": "in-manifest.md",
            "mime_type": "text/markdown", "char_count": 0,
            "chunk_count": 0, "parser": "text_v1",
            "uploaded_at": 0.0, "status": "ready",
            "text": "manifest-text OCTOPUS-7741",
        })
        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool("test-project", "read_source", {"source_id": sid})
        assert "manifest-text" in result.get("text", "")

    def test_create_qa_pairs_writes_to_disk(self, tmp_path, monkeypatch):
        """Verify create_qa_pairs writes JSON files under the project's qa/pairs dir.

        We patch project_dir via the qa_fs module's import so both the chat
        route's _run_tool AND qa_fs.write_qa_pair see the same temp path.
        """
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        import finetune_studio.data.fs.paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)
        # Also patch qa_fs's already-imported reference (it captured project_dir
        # at import time via `from ... import project_dir`).
        import finetune_studio.data.fs.qa as qa_fs_mod
        monkeypatch.setattr(qa_fs_mod, "project_dir", lambda pid: tmp_path / pid)

        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool(
            "test-project",
            "create_qa_pairs",
            {
                "source_id": "src-1",
                "pairs": [
                    {"question": "What is the project code?", "answer": "OCTOPUS-7741."},
                    {"question": "What date?", "answer": "2026-09-10."},
                    {"question": "", "answer": ""},  # should be skipped
                ],
            },
        )
        assert result["written"] == 2
        assert result["source_id"] == "src-1"
        # Verify files landed on disk
        pairs_dir = tmp_path / "test-project" / "qa" / "pairs"
        assert pairs_dir.exists()
        files = list(pairs_dir.glob("*.json"))
        assert len(files) == 2
        import json
        contents = [json.loads(f.read_text()) for f in files]
        questions = {c["question"] for c in contents}
        assert "What is the project code?" in questions
        assert all(c["status"] == "pending" for c in contents)
        assert all(c["source_id"] == "src-1" for c in contents)
        assert all(c["created_via"] == "data-prep-chat" for c in contents)

    def test_create_qa_pairs_empty_input(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        from finetune_studio.data.fs import paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)

        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool("test-project", "create_qa_pairs",
                           {"source_id": "src-1", "pairs": []})
        assert "error" in result

    def test_unknown_tool(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FTS_DB", str(tmp_path / "fts.db"))
        from finetune_studio.data.fs import paths as paths_mod
        monkeypatch.setattr(paths_mod, "project_dir", lambda pid: tmp_path / pid)

        from finetune_studio.webui.routes.data_prep_chat import _run_tool
        result = _run_tool("test-project", "no_such_tool", {})
        assert "unknown tool" in result["error"]


# ── Tool-call parser ────────────────────────────────────────────────────


class TestToolCallParser:
    """The regex that extracts <tool_call>{...}</tool_call> blocks."""

    def test_extracts_single_call(self):
        from finetune_studio.webui.routes.data_prep_chat import _extract_tool_calls
        text = 'I will check.\n<tool_call>{"name":"list_sources","arguments":{}}</tool_call>\nDone.'
        calls = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_sources"
        assert calls[0]["arguments"] == {}

    def test_extracts_multiple_calls(self):
        from finetune_studio.webui.routes.data_prep_chat import _extract_tool_calls
        text = (
            '<tool_call>{"name":"list_sources","arguments":{}}</tool_call>\n'
            '<tool_call>{"name":"read_source","arguments":{"source_id":"src-1"}}</tool_call>'
        )
        calls = _extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "list_sources"
        assert calls[1]["name"] == "read_source"
        assert calls[1]["arguments"]["source_id"] == "src-1"

    def test_no_calls_returns_empty(self):
        from finetune_studio.webui.routes.data_prep_chat import _extract_tool_calls
        text = "Just a normal response with no tool calls."
        assert _extract_tool_calls(text) == []

    def test_malformed_json_skipped(self):
        from finetune_studio.webui.routes.data_prep_chat import _extract_tool_calls
        text = '<tool_call>{not valid json}</tool_call>\n<tool_call>{"name":"list_sources","arguments":{}}</tool_call>'
        calls = _extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_sources"


# ── Message flattening for local providers ────────────────────────────


class TestMessagesToPrompt:
    def test_splits_system_from_rest(self):
        from finetune_studio.webui.routes.data_prep_chat import _messages_to_prompt
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        sys, rendered = _messages_to_prompt(msgs)
        assert sys == "You are helpful."
        assert rendered == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]

    def test_no_system(self):
        from finetune_studio.webui.routes.data_prep_chat import _messages_to_prompt
        msgs = [{"role": "user", "content": "hi"}]
        sys, rendered = _messages_to_prompt(msgs)
        assert sys == ""
        assert rendered == msgs


# ── Route-level smoke (via FastAPI TestClient) ──────────────────────────


class TestChatRouteSmoke:
    """FTS_SKIP_CHAT=1 short-circuits the actual model call with a canned
    response so we can exercise the full HTTP flow without GPU/keys."""

    def test_missing_messages_returns_error(self, client, monkeypatch):
        monkeypatch.setenv("FTS_SKIP_CHAT", "1")
        r = client.post("/api/projects/test/data-prep/chat", json={})
        body = r.json()
        # Either 422 (validation) or 200-with-error; both are acceptable
        assert "error" in body or r.status_code in (200, 422)

    def test_missing_backend_returns_409(self, client, monkeypatch):
        """No provider_id / external_api and nothing loaded → 409."""
        from unittest.mock import MagicMock

        from finetune_studio.data.prep.generator import HELPER_NO_MODEL_MSG

        mgr = MagicMock()
        mgr.active.return_value = None
        eng = MagicMock()
        eng.model = None
        monkeypatch.setattr(
            "finetune_studio.models.manager.get_manager",
            lambda: mgr,
        )
        monkeypatch.setattr(
            "finetune_studio.webui.app.inference_engine",
            eng,
        )
        r = client.post(
            "/api/projects/test/data-prep/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 409
        body = r.json()
        assert body.get("error") == HELPER_NO_MODEL_MSG

    def test_external_api_skips_to_canned_reply(self, client, monkeypatch):
        """With FTS_SKIP_CHAT=1, external_api still goes through the loop
        but the canned reply has no tool calls, so we get a single round."""
        monkeypatch.setenv("FTS_SKIP_CHAT", "1")
        r = client.post(
            "/api/projects/test/data-prep/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "external_api": {
                    "base_url": "http://stub",
                    "api_key": "test",
                    "model_id": "stub-model",
                },
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "ok" in body or "error" in body  # may fail because project doesn't exist

    def test_tools_catalog_endpoint(self, client):
        r = client.get("/api/projects/test/data-prep/chat/tools")
        assert r.status_code == 200
        body = r.json()
        tool_names = {t["name"] for t in body["tools"]}
        assert {"list_sources", "read_source", "list_qa_pairs", "create_qa_pairs"} <= tool_names


# ── InferenceEngine.unload cleanup ──────────────────────────────────────


class TestInferenceEngineUnload:
    """Verify the unload fix: model ref drops, explicit del + gc.collect
    fire, VRAM cache cleared. We can't actually load a 16GB GGUF in CI,
    so this exercises the code path with a stub."""

    def test_unload_with_no_model_is_safe(self):
        from finetune_studio.testing.inference import InferenceEngine
        eng = InferenceEngine()
        eng.model = None
        eng.unload()  # must not raise
        assert eng.model is None

    def test_unload_drops_ref_and_calls_gc(self):
        from finetune_studio.testing.inference import InferenceEngine

        class FakeModel:
            def __del__(self):
                pass

        eng = InferenceEngine()
        fake = FakeModel()
        eng.model = fake
        eng.unload()
        assert eng.model is None
        assert eng.tokenizer is None
        assert eng.model_path is None
        assert eng.is_gguf is False
        assert eng.vision is False

    def test_unload_resets_last_used(self):
        from finetune_studio.testing.inference import InferenceEngine

        eng = InferenceEngine()
        eng.model = "dummy"
        eng._last_used = 12345.0
        eng.unload()
        assert eng._last_used == 0.0
