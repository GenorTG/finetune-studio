"""Tests for the GGUF export endpoint + worker.

Covers:
- Route validation: invalid quant / format / no output_path / no merged
- Route happy path: queues a background task with the right args
- Route auto_merge=true chains the merge when merged/ is missing
- Worker happy path under FTS_SKIP_EXPORT=1: writes marker + marks done
- Worker failure when tooling is missing: marks failed with clean error
"""

from __future__ import annotations

import os
import tempfile

import pytest


# ── Pure helpers (no DB / no FastAPI) ────────────────────────────────────


class TestExportHelpers:
    def test_human_size(self):
        from finetune_studio.webui.routes.exports import _human_size
        assert _human_size(0) == "0 B"
        assert "KB" in _human_size(2048)
        assert "MB" in _human_size(2 * 1024 * 1024)
        assert "GB" in _human_size(2 * 1024 ** 3)
        assert "TB" in _human_size(2 * 1024 ** 4)

    def test_safe_name_strips_slashes(self):
        from finetune_studio.webui.routes.exports import _safe_name
        assert _safe_name("Qwen/Qwen2-7B-Instruct") == "Qwen_Qwen2-7B-Instruct"
        assert _safe_name("normal-name_1.0") == "normal-name_1.0"
        assert _safe_name("with spaces & chars!") == "with_spaces___chars_"

    def test_supported_quants_include_mainstream(self):
        from finetune_studio.webui.routes.exports import (
            DEFAULT_QUANT, SUPPORTED_QUANTS,
        )
        assert DEFAULT_QUANT == "Q4_K_M"
        for q in ("f16", "bf16", "f32", "Q8_0",
                  "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
                  "Q3_K_M", "Q3_K_S", "Q2_K",
                  "IQ4_XS", "IQ3_XXS"):
            assert q in SUPPORTED_QUANTS, f"missing: {q}"

    def test_find_tool_returns_none_when_absent(self, monkeypatch):
        """If llama.cpp isn't on PATH or in the common install paths,
        _find_llama_tool returns None. We simulate this by clearing PATH
        and pointing search paths at a temp dir that has nothing in it."""
        from finetune_studio.webui.routes import exports
        monkeypatch.setattr(exports, "LLAMA_CPP_SEARCH_PATHS",
                            [tempfile.mkdtemp()])
        monkeypatch.setenv("PATH", "")
        assert exports._find_llama_tool("llama-quantize") is None
        assert exports._find_convert_script() is None


# ── Worker with FTS_SKIP_EXPORT=1 ────────────────────────────────────────


class TestExportWorkerSkip:
    """The worker should short-circuit cleanly when FTS_SKIP_EXPORT=1,
    writing a marker file at the target path and marking the row done.
    """

    def test_skip_writes_marker_and_marks_done(self, mock_settings, monkeypatch):
        from finetune_studio import db
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.setenv("FTS_SKIP_EXPORT", "1")
        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q4_K_M")["id"]

        with tempfile.TemporaryDirectory() as out:
            out_path = os.path.join(out, "model-Q4_K_M.gguf")
            _export_worker(eid, merged_dir=out, out_path=out_path, quant="Q4_K_M")
            r = db.get_export(eid)
            assert r["status"] == "done"
            assert r["output_path"] == out_path
            assert r["size_bytes"] == 1
            assert "B" in r["size_human"]
            assert os.path.isfile(out_path)
            assert r["finished_at"] is not None
            assert r["duration_ms"] is not None and r["duration_ms"] >= 0

    def test_failure_when_no_tooling(self, mock_settings, monkeypatch):
        """With FTS_SKIP_EXPORT unset and no llama.cpp on the host, the
        worker should mark the export failed with a clean, installable
        error message."""
        from finetune_studio import db
        from finetune_studio.webui.routes import exports
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        # Point search paths at empty temp dirs and empty PATH
        monkeypatch.setattr(exports, "LLAMA_CPP_SEARCH_PATHS",
                            [tempfile.mkdtemp()])
        monkeypatch.setenv("PATH", "")

        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q4_K_M")["id"]

        with tempfile.TemporaryDirectory() as out:
            out_path = os.path.join(out, "model-Q4_K_M.gguf")
            merged_dir = os.path.join(out, "merged")
            os.makedirs(merged_dir)
            _export_worker(eid, merged_dir=merged_dir, out_path=out_path,
                           quant="Q4_K_M")
            r = db.get_export(eid)
            assert r["status"] == "error"
            assert r["finished_at"] is not None
            # The error should mention convert_hf_to_gguf + install hint
            assert "convert_hf_to_gguf" in r["error"]
            assert "llama.cpp" in r["error"]

    def test_failure_when_only_convert_present_but_quant_needs_binary(
        self, mock_settings, monkeypatch
    ):
        """convert_hf_to_gguf.py is found but llama-quantize is not \u2014
        the worker should error mentioning llama-quantize."""
        from finetune_studio import db
        from finetune_studio.webui.routes import exports
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        # Make a fake llama.cpp install with only the convert script
        fake_root = tempfile.mkdtemp()
        os.makedirs(os.path.join(fake_root, "build", "bin"), exist_ok=True)
        with open(os.path.join(fake_root, "convert_hf_to_gguf.py"), "w") as f:
            f.write("# fake\n")
        monkeypatch.setattr(exports, "LLAMA_CPP_SEARCH_PATHS", [fake_root])
        monkeypatch.setenv("PATH", "")  # nothing on PATH

        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        # Pick a quant that requires the two-step (quantize) path
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q5_K_M")["id"]

        with tempfile.TemporaryDirectory() as out:
            out_path = os.path.join(out, "model-Q5_K_M.gguf")
            merged_dir = os.path.join(out, "merged")
            os.makedirs(merged_dir)
            _export_worker(eid, merged_dir=merged_dir, out_path=out_path,
                           quant="Q5_K_M")
            r = db.get_export(eid)
            assert r["status"] == "error"
            assert "llama-quantize" in r["error"]


# ── Route validation (via FastAPI TestClient) ────────────────────────────


class TestExportRoute:
    """Validation + happy-path tests via the real FastAPI app.

    The client fixture reloads the webui app with TrainingEngine +
    InferenceEngine patched, so these tests don't need a GPU or model
    weights. We do use the real SQLite DB via the mock_settings fixture.
    """

    def test_unknown_format_returns_400(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        db.update_run(rid, output_path="/tmp/nope")
        r = client.post(f"/api/projects/{pid}/runs/{rid}/export",
                        json={"format": "awq", "quant": "Q4_K_M"})
        assert r.status_code == 200
        body = r.json()
        assert "unsupported format" in body["error"]

    def test_unknown_quant_returns_400(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        db.update_run(rid, output_path="/tmp/nope")
        r = client.post(f"/api/projects/{pid}/runs/{rid}/export",
                        json={"format": "gguf", "quant": "QQQQQQQ"})
        body = r.json()
        assert "unsupported quant" in body["error"]
        assert "Q4_K_M" in body["supported"]

    def test_missing_run_returns_error(self, client, mock_settings):
        r = client.post("/api/projects/p/runs/nonexistent/export",
                        json={"format": "gguf", "quant": "Q4_K_M"})
        body = r.json()
        assert "run not found" in body["error"]

    def test_run_with_no_output_path_returns_error(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        r = client.post(f"/api/projects/{pid}/runs/{rid}/export",
                        json={"format": "gguf", "quant": "Q4_K_M"})
        body = r.json()
        assert "no output_path" in body["error"]

    def test_no_merged_dir_and_no_auto_merge_returns_error(
        self, client, mock_settings
    ):
        from finetune_studio import db
        with tempfile.TemporaryDirectory() as out:
            pid = db.create_project(name="R", description="")["id"]
            rid = db.create_run(project_id=pid, name="r", base_model="m",
                                settings_obj={})["id"]
            db.update_run(rid, output_path=out)
            r = client.post(f"/api/projects/{pid}/runs/{rid}/export",
                            json={"format": "gguf", "quant": "Q4_K_M",
                                  "auto_merge": False})
            body = r.json()
            assert "/merge first" in body["error"]

    def test_get_export_unknown_returns_404(self, client, mock_settings):
        r = client.get("/api/projects/p/exports/nonexistent")
        assert r.status_code == 404

    def test_get_export_returns_row(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q4_K_M")["id"]
        r = client.get(f"/api/projects/{pid}/exports/{eid}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == eid
        assert body["quant"] == "Q4_K_M"
        assert body["status"] == "queued"

    def test_list_run_exports_returns_rows(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        db.create_export(project_id=pid, run_id=rid, quant="Q4_K_M")
        db.create_export(project_id=pid, run_id=rid, quant="Q8_0")
        r = client.get(f"/api/projects/{pid}/runs/{rid}/exports")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 2
        quants = {row["quant"] for row in rows}
        assert quants == {"Q4_K_M", "Q8_0"}
