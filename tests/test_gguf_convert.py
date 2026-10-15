"""Focused tests: GGUF converter discovery and artifact verification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestGgufDiscovery:
    def test_find_script_none_when_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import gguf_convert as gc

        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setattr(gc, "llama_cpp_search_paths", lambda: [str(tmp_path)])
        monkeypatch.setenv("PATH", "")
        assert gc.find_gguf_convert_script() is None
        assert gc.find_llama_quantize() is None

    def test_find_script_and_quantize_in_search_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import gguf_convert as gc

        root = tmp_path / "llama.cpp"
        root.mkdir()
        script = root / "convert_hf_to_gguf.py"
        script.write_text("# fake\n", encoding="utf-8")
        bin_dir = root / "build" / "bin"
        bin_dir.mkdir(parents=True)
        quant = bin_dir / "llama-quantize"
        quant.write_text("#!/bin/sh\n", encoding="utf-8")
        quant.chmod(0o755)

        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setattr(gc, "llama_cpp_search_paths", lambda: [str(root)])
        monkeypatch.setenv("PATH", "")
        assert gc.find_gguf_convert_script() == str(script)
        assert gc.find_llama_quantize() == str(quant)

    def test_llama_cpp_dir_env_preferred(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import gguf_convert as gc

        env_root = tmp_path / "env-llama"
        env_root.mkdir()
        (env_root / "convert_hf_to_gguf.py").write_text("x", encoding="utf-8")
        other = tmp_path / "other"
        other.mkdir()
        (other / "convert_hf_to_gguf.py").write_text("y", encoding="utf-8")
        monkeypatch.setenv("LLAMA_CPP_DIR", str(env_root))
        paths = gc.llama_cpp_search_paths()
        assert paths[0] == str(env_root)


class TestGgufArtifacts:
    def test_verify_rejects_empty_and_accepts_nonempty(
        self, tmp_path: Path
    ) -> None:
        from finetune_studio.training.gguf_convert import verify_gguf_artifacts

        gguf = tmp_path / "gguf"
        gguf.mkdir()
        (gguf / "model-q8_0.gguf").write_bytes(b"")
        bad = verify_gguf_artifacts(str(gguf), ["q8_0"])
        assert bad["ok"] is False
        assert "q8_0" in bad["missing"]

        (gguf / "model-q8_0.gguf").write_bytes(b"gguf-bytes")
        ok = verify_gguf_artifacts(str(gguf), ["q8_0"])
        assert ok["ok"] is True
        assert ok["files"][0].endswith("model-q8_0.gguf")

    def test_convert_skip_writes_nonempty_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training.gguf_convert import (
            convert_merged_to_gguf,
            verify_gguf_artifacts,
        )

        merged = tmp_path / "merged"
        merged.mkdir()
        (merged / "config.json").write_text("{}", encoding="utf-8")
        (merged / "model.safetensors").write_bytes(b"x" * 8)
        gguf = tmp_path / "gguf"
        monkeypatch.setenv("FTS_SKIP_EXPORT", "1")
        result = convert_merged_to_gguf(
            str(merged), str(gguf), ["q8_0"], force=True
        )
        assert result["ok"] is True
        assert result["status"] == "exported"
        verified = verify_gguf_artifacts(str(gguf), ["q8_0"])
        assert verified["ok"] is True
        assert os.path.getsize(verified["files"][0]) > 0

    def test_convert_fails_without_converter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training import gguf_convert as gc

        merged = tmp_path / "merged"
        merged.mkdir()
        (merged / "x.bin").write_bytes(b"x")
        monkeypatch.delenv("FTS_SKIP_EXPORT", raising=False)
        monkeypatch.setattr(gc, "find_gguf_convert_script", lambda: None)
        result = gc.convert_merged_to_gguf(
            str(merged), str(tmp_path / "gguf"), ["q8_0"], force=True
        )
        assert result["ok"] is False
        assert "convert_hf_to_gguf" in (result.get("error") or "")


class TestSyncGgufExportUsesConverter:
    def test_export_q8_0_via_skip_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from finetune_studio.training.run_export import export_trained_run

        out = tmp_path / "run"
        merged = out / "merged"
        merged.mkdir(parents=True)
        (merged / "config.json").write_text("{}", encoding="utf-8")
        (merged / "model.safetensors").write_bytes(b"x" * 32)
        run = {
            "id": "r",
            "output_path": str(out),
            "base_model": "Qwen/Qwen3-0.6B",
            "status": "done",
        }
        monkeypatch.setenv("FTS_SKIP_EXPORT", "1")
        monkeypatch.setattr(
            "finetune_studio.training.run_export.find_gguf_convert_script",
            lambda: "/fake/convert_hf_to_gguf.py",
        )
        result = export_trained_run(run, fmt="gguf", quants=["q8_0"], force=True)
        assert result.get("ok") is True, result
        assert result.get("format") == "gguf"
        assert any(p.endswith("model-q8_0.gguf") for p in result.get("files", []))
        assert os.path.getsize(result["files"][0]) > 0
