"""Focused tests for merge-at-export + base_model validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _adapter_run(tmp_path: Path, *, base_model: str = "Qwen/Qwen3-4B") -> dict:
    """Create an adapter-only (no merged/) run directory + dict."""
    out = tmp_path / "run-out"
    adapter = out / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"x" * 32)
    return {
        "id": "run-adapter",
        "output_path": str(out),
        "base_model": base_model,
        "status": "done",
    }


def _merged_run(tmp_path: Path) -> dict:
    out = tmp_path / "run-merged"
    merged = out / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"x" * 64)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    return {
        "id": "run-merged",
        "output_path": str(out),
        "base_model": "Qwen/Qwen3-4B",
        "status": "done",
    }


class TestValidateBaseModel:
    def test_hub_id_accepted(self) -> None:
        from finetune_studio.training.run_export import validate_base_model

        assert validate_base_model("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"

    def test_empty_rejected(self) -> None:
        from finetune_studio.training.run_export import validate_base_model

        with pytest.raises(ValueError, match="empty"):
            validate_base_model("  ")

    def test_local_dir_requires_config(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import validate_base_model

        bare = tmp_path / "model"
        bare.mkdir()
        with pytest.raises(ValueError, match="config.json"):
            validate_base_model(str(bare))

        (bare / "config.json").write_text("{}", encoding="utf-8")
        assert validate_base_model(str(bare)) == str(bare.resolve())

    def test_missing_absolute_path_rejected(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import validate_base_model

        missing = tmp_path / "nope"
        with pytest.raises(ValueError, match="does not exist"):
            validate_base_model(str(missing))


class TestEnsureMergedForExport:
    def test_skips_when_merged_ready(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import ensure_merged_for_export

        run = _merged_run(tmp_path)
        result = ensure_merged_for_export(run)
        assert result["skipped"] is True
        assert result["merged_path"].endswith("merged")

    def test_adapter_only_calls_merge_with_override(
        self, tmp_path: Path
    ) -> None:
        from finetune_studio.training.run_export import ensure_merged_for_export

        run = _adapter_run(tmp_path, base_model="org/quantized-bnb-4bit")
        base = tmp_path / "fp16"
        base.mkdir()
        (base / "config.json").write_text("{}", encoding="utf-8")

        with patch(
            "finetune_studio.training.engine.merge_adapter_for_run"
        ) as merge_fn:
            merge_fn.return_value = {
                "merged_path": os.path.join(run["output_path"], "merged"),
                "skipped": False,
                "size_bytes": 1,
                "size_human": "1 B",
            }
            result = ensure_merged_for_export(
                run, base_model=str(base), force=False,
            )
        assert result["merged"] is True
        assert result["skipped"] is False
        called_run = merge_fn.call_args[0][0]
        assert called_run["base_model"] == str(base.resolve())

    def test_no_adapter_no_merged_errors(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import ensure_merged_for_export

        out = tmp_path / "empty"
        out.mkdir()
        with pytest.raises(ValueError, match="no adapter"):
            ensure_merged_for_export(
                {"output_path": str(out), "base_model": "x/y"},
            )


class TestExportTrainedRun:
    def test_awq_rejected_with_clear_message(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import export_trained_run

        run = _merged_run(tmp_path)
        result = export_trained_run(run, fmt="awq")
        assert result.get("ok") is False
        assert result.get("status") == "failed"
        assert "AWQ" in result["error"]
        assert "gptq" in result["error"].lower()

    def test_gptq_missing_backend_is_structured_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import run_export as re

        monkeypatch.setattr(
            "finetune_studio.training.advanced_quant.is_gptq_available",
            lambda: False,
        )
        run = _merged_run(tmp_path)
        result = re.export_trained_run(run, fmt="gptq")
        assert result.get("ok") is False
        assert result.get("status") == "failed"
        assert result.get("format") == "gptq"
        err = result["error"].lower()
        assert "gptqmodel" in err or "auto_gptq" in err

    def test_merged_format_returns_path(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import export_trained_run

        run = _merged_run(tmp_path)
        result = export_trained_run(run, fmt="merged")
        assert result.get("ok") is True
        assert result["format"] == "merged"
        assert os.path.isdir(result["merged_path"])

    def test_unknown_format(self, tmp_path: Path) -> None:
        from finetune_studio.training.run_export import export_trained_run

        run = _merged_run(tmp_path)
        result = export_trained_run(run, fmt="mlx")
        assert "unknown format" in result["error"]
        assert "gguf" in result["supported"]


class TestGgufArtifactVerification:
    """GGUF must not report success without non-empty artifacts."""

    def test_verify_requires_nonempty_matching_files(
        self, tmp_path: Path
    ) -> None:
        from finetune_studio.training.run_export import verify_gguf_artifacts

        gguf = tmp_path / "gguf"
        gguf.mkdir()
        (gguf / "model-q8_0.gguf").write_bytes(b"")  # empty — ignore
        (gguf / "notes.txt").write_text("nope", encoding="utf-8")
        bad = verify_gguf_artifacts(str(gguf), ["q8_0"])
        assert bad["ok"] is False
        assert "q8_0" in bad["missing"]

        (gguf / "model-q8_0.gguf").write_bytes(b"gguf-bytes")
        ok = verify_gguf_artifacts(str(gguf), ["q8_0"])
        assert ok["ok"] is True
        assert len(ok["files"]) == 1
        assert ok["files"][0].endswith("model-q8_0.gguf")

    def test_missing_converter_is_structured_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import run_export as re

        monkeypatch.setattr(re, "find_gguf_convert_script", lambda: None)
        run = _merged_run(tmp_path)
        result = re.export_trained_run(run, fmt="gguf", quants=["q8_0"])
        assert result.get("ok") is False
        assert result.get("status") == "failed"
        assert result.get("format") == "gguf"
        assert "convert_hf_to_gguf" in result["error"]
        assert "llama.cpp" in result["error"].lower()
        # Must not leave a false "success" footprint for the UI.
        assert result.get("ok") is not True

    def test_engine_false_success_without_artifacts_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: browser saw HTTP 200 with no GGUF on disk."""
        from finetune_studio.training import run_export as re
        from finetune_studio.training.engine import TrainingEngine

        monkeypatch.setattr(
            re, "find_gguf_convert_script", lambda: "/fake/convert_hf_to_gguf.py"
        )

        def _fake_export(self: TrainingEngine, output_dir: str) -> dict:
            gguf_dir = os.path.join(output_dir, "gguf")
            os.makedirs(gguf_dir, exist_ok=True)
            # Historical bug shape: skipped/reason, no error, empty dir.
            return {
                "gguf_path": gguf_dir,
                "skipped": True,
                "reason": "llama.cpp not found",
            }

        monkeypatch.setattr(TrainingEngine, "_do_export_gguf", _fake_export)
        run = _merged_run(tmp_path)
        result = re.export_trained_run(run, fmt="gguf", quants=["q8_0"])
        assert result.get("ok") is False
        assert result.get("status") == "failed"
        assert "q8_0" in (result.get("missing") or [])
        err = (result.get("error") or "").lower()
        assert "gguf" in err or "llama.cpp" in err or "artifact" in err

    def test_existing_artifacts_skip_as_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import run_export as re

        # Converter absent must not matter when verified artifacts already exist.
        monkeypatch.setattr(re, "find_gguf_convert_script", lambda: None)
        run = _merged_run(tmp_path)
        gguf = Path(run["output_path"]) / "gguf"
        gguf.mkdir()
        (gguf / "model-q8_0.gguf").write_bytes(b"real-gguf")
        result = re.export_trained_run(
            run, fmt="gguf", quants=["q8_0"], force=False
        )
        assert result.get("ok") is True
        assert result.get("status") == "skipped"
        assert result.get("format") == "gguf"
        assert any(p.endswith("model-q8_0.gguf") for p in result["files"])

    def test_post_gguf_missing_converter_returns_400(
        self, client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio import db
        from finetune_studio.training import run_export as re

        monkeypatch.setattr(re, "find_gguf_convert_script", lambda: None)
        pid = client.post(
            "/api/projects", json={"name": "No GGUF Converter"}
        ).json()["id"]
        run = _merged_run(tmp_path)
        created = db.create_run(
            project_id=pid,
            name="m",
            base_model=run["base_model"],
            data_path="/d",
        )
        db.update_run(
            created["id"], status="done", output_path=run["output_path"]
        )
        r = client.post(
            f"/api/projects/{pid}/runs/{created['id']}/export",
            json={"format": "gguf", "quants": ["q8_0"]},
        )
        assert r.status_code == 400, r.text
        body = r.json()
        assert body.get("ok") is False
        assert body.get("status") == "failed"
        assert "convert_hf_to_gguf" in body["error"]


class TestExportApiAdapterOnly:
    """HTTP: adapter-only run is exportable via projects export + base_model."""

    def test_page_lists_adapter_only_run(
        self, client, tmp_path: Path
    ) -> None:
        from finetune_studio import db

        pid = client.post(
            "/api/projects", json={"name": "Export Adapter"}
        ).json()["id"]
        out = tmp_path / "adapter-run"
        adapter = out / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        run = db.create_run(
            project_id=pid,
            name="raw-4b",
            base_model="Qwen/Qwen3-4B",
            data_path="/d",
            settings_obj={"merge_on_save": False},
        )
        db.update_run(run["id"], status="done", output_path=str(out))

        body = client.get(f"/projects/{pid}/export").text
        assert "adapter only — merge at export" in body
        assert "Compatible base model" in body
        assert 'value="awq"' not in body
        assert "AWQ is not available" not in body
        assert run["id"] in body

    def test_post_export_merged_with_base_override(
        self, client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio import db

        monkeypatch.setenv("FTS_SKIP_MERGE", "1")
        pid = client.post(
            "/api/projects", json={"name": "Export Merge"}
        ).json()["id"]
        out = tmp_path / "adapter-run2"
        adapter = out / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        base = tmp_path / "base16"
        base.mkdir()
        (base / "config.json").write_text("{}", encoding="utf-8")

        run = db.create_run(
            project_id=pid,
            name="raw",
            base_model="org/quant-bnb-4bit",
            data_path="/d",
            settings_obj={"merge_on_save": False},
        )
        db.update_run(run["id"], status="done", output_path=str(out))

        r = client.post(
            f"/api/projects/{pid}/runs/{run['id']}/export",
            json={
                "format": "merged",
                "force": False,
                "base_model": str(base),
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("ok") is True
        assert data.get("format") == "merged"
        assert os.path.isdir(data["merged_path"])
        assert os.path.isfile(
            os.path.join(data["merged_path"], "SKIPPED_BY_TEST")
        )

    def test_post_rejects_awq(
        self, client, tmp_path: Path
    ) -> None:
        from finetune_studio import db

        pid = client.post(
            "/api/projects", json={"name": "No AWQ"}
        ).json()["id"]
        run = _merged_run(tmp_path)
        created = db.create_run(
            project_id=pid,
            name="m",
            base_model=run["base_model"],
            data_path="/d",
        )
        db.update_run(
            created["id"], status="done", output_path=run["output_path"]
        )
        r = client.post(
            f"/api/projects/{pid}/runs/{created['id']}/export",
            json={"format": "awq", "quants": ["q8_0"]},
        )
        assert r.status_code == 400, r.text
        body = r.json()
        assert body.get("ok") is False
        assert body.get("status") == "failed"
        assert "AWQ" in body["error"]

    def test_post_gptq_missing_module_returns_400(
        self, client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio import db

        monkeypatch.setattr(
            "finetune_studio.training.advanced_quant.is_gptq_available",
            lambda: False,
        )
        pid = client.post(
            "/api/projects", json={"name": "No GPTQ"}
        ).json()["id"]
        run = _merged_run(tmp_path)
        created = db.create_run(
            project_id=pid,
            name="m",
            base_model=run["base_model"],
            data_path="/d",
        )
        db.update_run(
            created["id"], status="done", output_path=run["output_path"]
        )
        r = client.post(
            f"/api/projects/{pid}/runs/{created['id']}/export",
            json={"format": "gptq", "force": True},
        )
        assert r.status_code == 400, r.text
        body = r.json()
        assert body.get("ok") is False
        assert body.get("status") == "failed"
        assert body.get("format") == "gptq"
        err = body["error"].lower()
        assert "gptqmodel" in err or "auto_gptq" in err
        assert body.get("ok") is not True
