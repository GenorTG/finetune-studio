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
from pathlib import Path

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
            DEFAULT_QUANT,
            SUPPORTED_QUANTS,
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

    def test_single_step_quant_uses_outfile_flag(self, mock_settings, monkeypatch, tmp_path):
        """Regression for the fan-dragon 2026-09-10 GGUF Q8_0 export:
        convert_hf_to_gguf.py takes [model] as a single positional arg
        and uses --outfile OUTFILE + --outtype QUANT for output.
        """
        from unittest.mock import patch

        from finetune_studio import db
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q8_0")["id"]

        fake_convert = tmp_path / "convert_hf_to_gguf.py"
        fake_convert.write_text("# fake\n")
        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_bytes(b"x")
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        out_path = gguf_dir / "model-Q8_0.gguf"

        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(cmd)
            if "convert_hf_to_gguf.py" in " ".join(str(c) for c in cmd):
                outfile = cmd[cmd.index("--outfile") + 1]
                Path(outfile).write_bytes(b"\x00")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch(
            "finetune_studio.training.gguf_convert.find_gguf_convert_script",
            return_value=str(fake_convert),
        ), patch(
            "finetune_studio.training.gguf_convert.subprocess.run",
            side_effect=fake_run,
        ):
            _export_worker(
                eid,
                merged_dir=str(merged_dir),
                out_path=str(out_path),
                quant="Q8_0",
            )

        assert captured, "subprocess.run was never called"
        cmd = captured[0]
        assert "--outfile" in cmd, f"missing --outfile flag in argv: {cmd}"
        assert "--outtype" in cmd
        assert cmd[cmd.index("--outtype") + 1] == "q8_0"
        script_idx = next(
            i for i, c in enumerate(cmd)
            if str(c).endswith("convert_hf_to_gguf.py")
        )
        tail = cmd[script_idx + 1:]
        assert str(tail[0]) == str(merged_dir)
        assert tail[1] == "--outfile"
        assert str(tail[2]).endswith("model-q8_0.gguf"), tail[2]
        assert tail[3] == "--outtype"
        assert tail[4] == "q8_0"
        assert len(tail) == 5, f"unexpected extra args: {tail[5:]}"
        row = db.get_export(eid)
        assert row["status"] == "done"
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 0

    def test_subprocess_uses_venv_python_not_path_python3(
        self, mock_settings, monkeypatch, tmp_path,
    ):
        """Worker must use sys.executable (venv), not bare python3 on PATH."""
        import sys
        from unittest.mock import patch

        from finetune_studio import db
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q8_0")["id"]

        fake_convert = tmp_path / "convert_hf_to_gguf.py"
        fake_convert.write_text("# fake\n")
        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_bytes(b"x")
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        out_path = gguf_dir / "model-Q8_0.gguf"

        captured = []

        def fake_run(cmd, *a, **kw):
            captured.append(cmd)
            if "convert_hf_to_gguf.py" in " ".join(str(c) for c in cmd):
                outfile = cmd[cmd.index("--outfile") + 1]
                Path(outfile).write_bytes(b"\x00")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch(
            "finetune_studio.training.gguf_convert.find_gguf_convert_script",
            return_value=str(fake_convert),
        ), patch(
            "finetune_studio.training.gguf_convert.subprocess.run",
            side_effect=fake_run,
        ):
            _export_worker(
                eid,
                merged_dir=str(merged_dir),
                out_path=str(out_path),
                quant="Q8_0",
            )
        cmd = captured[0]
        assert cmd[0] == sys.executable, (
            f"worker must use sys.executable, got cmd[0]={cmd[0]!r}"
        )
        assert cmd[0] != "python3"

    def test_failure_when_no_tooling(self, mock_settings, monkeypatch):
        """With FTS_SKIP_EXPORT unset and no llama.cpp on the host, the
        worker should mark the export failed with a clean, installable
        error message."""
        from finetune_studio import db
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        empty = tempfile.mkdtemp()
        monkeypatch.setattr(
            "finetune_studio.training.gguf_convert.llama_cpp_search_paths",
            lambda: [empty],
        )
        monkeypatch.setenv("PATH", "")

        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q4_K_M")["id"]

        with tempfile.TemporaryDirectory() as out:
            out_path = os.path.join(out, "model-Q4_K_M.gguf")
            merged_dir = os.path.join(out, "merged")
            os.makedirs(merged_dir)
            weight = os.path.join(merged_dir, "model.safetensors")
            with open(weight, "wb") as fh:
                fh.write(b"x")
            _export_worker(eid, merged_dir=merged_dir, out_path=out_path,
                           quant="Q4_K_M")
            r = db.get_export(eid)
            assert r["status"] == "error"
            assert r["finished_at"] is not None
            assert "convert_hf_to_gguf" in r["error"]
            assert "llama.cpp" in r["error"]

    def test_failure_when_converter_writes_empty_gguf(
        self, mock_settings, monkeypatch, tmp_path
    ):
        """Refuse to mark done when the .gguf exists but is empty."""
        from unittest.mock import patch

        from finetune_studio import db
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q8_0")["id"]

        fake_convert = tmp_path / "convert_hf_to_gguf.py"
        fake_convert.write_text("# fake\n", encoding="utf-8")
        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_bytes(b"x")
        out_path = tmp_path / "model-Q8_0.gguf"

        def fake_run(cmd, **_kw):
            outfile = cmd[cmd.index("--outfile") + 1]
            Path(outfile).write_bytes(b"")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch(
            "finetune_studio.training.gguf_convert.find_gguf_convert_script",
            return_value=str(fake_convert),
        ), patch(
            "finetune_studio.training.gguf_convert.subprocess.run",
            side_effect=fake_run,
        ):
            _export_worker(eid, merged_dir=str(merged_dir),
                           out_path=str(out_path), quant="Q8_0")
        r = db.get_export(eid)
        assert r["status"] == "error"
        err = (r.get("error") or "").lower()
        assert "empty" in err or "missing" in err or "artifact" in err

    def test_failure_when_only_convert_present_but_quant_needs_binary(
        self, mock_settings, monkeypatch
    ):
        """convert_hf_to_gguf.py is found but llama-quantize is not —
        the worker should error mentioning llama-quantize."""
        from finetune_studio import db
        from finetune_studio.webui.routes.exports import _export_worker

        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        fake_root = tempfile.mkdtemp()
        os.makedirs(os.path.join(fake_root, "build", "bin"), exist_ok=True)
        with open(os.path.join(fake_root, "convert_hf_to_gguf.py"), "w") as f:
            f.write("# fake\n")
        monkeypatch.setattr(
            "finetune_studio.training.gguf_convert.llama_cpp_search_paths",
            lambda: [fake_root],
        )
        monkeypatch.setenv("PATH", "")

        pid = db.create_project(name="E", description="")["id"]
        rid = db.create_run(project_id=pid, name="r",
                            base_model="m", settings_obj={})["id"]
        eid = db.create_export(project_id=pid, run_id=rid, quant="Q5_K_M")["id"]

        with tempfile.TemporaryDirectory() as out:
            out_path = os.path.join(out, "model-Q5_K_M.gguf")
            merged_dir = os.path.join(out, "merged")
            os.makedirs(merged_dir)
            weight = os.path.join(merged_dir, "model.safetensors")
            with open(weight, "wb") as fh:
                fh.write(b"x")
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
        assert r.status_code == 400
        body = r.json()
        assert body.get("ok") is False
        assert body.get("status") == "failed"
        # AWQ is explicitly removed; message must steer to gptq/gguf/merged.
        assert "AWQ" in body["error"]
        assert "gptq" in body["error"].lower()

    def test_unknown_quant_returns_400(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        db.update_run(rid, output_path="/tmp/nope")
        r = client.post(f"/api/projects/{pid}/runs/{rid}/export",
                        json={"format": "gguf", "quant": "QQQQQQQ"})
        assert r.status_code == 400
        body = r.json()
        assert body.get("ok") is False
        assert "unsupported quant" in body["error"]
        assert "Q4_K_M" in body["supported"]

    def test_missing_run_returns_error(self, client, mock_settings):
        r = client.post("/api/projects/p/runs/nonexistent/export",
                        json={"format": "gguf", "quant": "Q4_K_M"})
        assert r.status_code == 404
        body = r.json()
        assert "run not found" in body["error"]
        assert body.get("ok") is False

    def test_run_with_no_output_path_returns_error(self, client, mock_settings):
        from finetune_studio import db
        pid = db.create_project(name="R", description="")["id"]
        rid = db.create_run(project_id=pid, name="r", base_model="m",
                            settings_obj={})["id"]
        r = client.post(f"/api/projects/{pid}/runs/{rid}/export",
                        json={"format": "gguf", "quant": "Q4_K_M"})
        assert r.status_code == 400
        body = r.json()
        assert "no output_path" in body["error"]
        assert body.get("status") == "failed"

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
            assert r.status_code == 400
            body = r.json()
            assert "/merge first" in body["error"]
            assert body.get("ok") is False

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
