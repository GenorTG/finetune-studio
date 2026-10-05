"""Tests for the single canonical GGUF loader (models/llama_loader.py).

Before this module existed, LocalGGUFProvider.load() (models/providers.py)
and InferenceEngine._load_gguf() (testing/inference.py) each independently
built a llama_cpp.Llama(**kwargs) call and drifted: only one had OOM-retry
(shrink context instead of falling back to mixed CPU/GPU offload), only the
other had mmproj/vision detection and KV-cache-type support. Both now call
load_llama_gguf() — these tests pin its behavior directly so a future edit
can't silently reintroduce either gap.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _llama_cpp_stub(monkeypatch):
    """llama-cpp-python is an optional GPU wheel; stub it so these tests pin the
    loader's behavior on any machine (tests patch ``llama_cpp.Llama`` anyway)."""
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("llama_cpp") is None:
        stub = types.ModuleType("llama_cpp")
        stub.Llama = object  # replaced per-test via patch("llama_cpp.Llama", ...)
        fmt = types.ModuleType("llama_cpp.llama_chat_format")
        fmt.Qwen25VLChatHandler = object
        stub.llama_chat_format = fmt
        monkeypatch.setitem(sys.modules, "llama_cpp", stub)
        monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", fmt)


class FakeLlama:
    """Records the exact kwargs llama_cpp.Llama() was constructed with."""
    calls: list[dict] = []

    def __init__(self, **kwargs):
        FakeLlama.calls.append(kwargs)


def _reset():
    FakeLlama.calls = []


class TestLoadLlamaGguf:
    def setup_method(self):
        _reset()

    def test_basic_load_passes_core_kwargs(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(
                str(gguf), n_ctx=8192, n_gpu_layers=-1, n_batch=256, detect_mmproj=False,
            )
        assert result.llama is not None
        assert result.final_n_ctx == 8192
        kwargs = FakeLlama.calls[0]
        assert kwargs["model_path"] == str(gguf)
        assert kwargs["n_ctx"] == 8192
        assert kwargs["n_gpu_layers"] == -1
        assert kwargs["n_batch"] == 256
        assert kwargs["verbose"] is False

    def test_oom_retry_halves_context_never_reduces_gpu_layers(self, tmp_path):
        """The GH-AAA contract: on OOM, shrink n_ctx — never fall back to
        mixed CPU/GPU offload by reducing n_gpu_layers."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        attempts = {"n": 0}

        class OomThenOk:
            def __init__(self, **kwargs):
                attempts["n"] += 1
                FakeLlama.calls.append(kwargs)
                if attempts["n"] < 3:
                    raise RuntimeError("CUDA out of memory")

        with patch("llama_cpp.Llama", OomThenOk):
            result = mod.load_llama_gguf(
                str(gguf), n_ctx=8192, n_gpu_layers=-1, detect_mmproj=False,
            )
        assert attempts["n"] == 3
        # 8192 -> 4096 -> 2048 (the third attempt succeeds)
        assert [c["n_gpu_layers"] for c in FakeLlama.calls] == [-1, -1, -1]
        assert FakeLlama.calls[0]["n_ctx"] == 8192
        assert FakeLlama.calls[1]["n_ctx"] == 4096
        assert FakeLlama.calls[2]["n_ctx"] == 2048
        assert result.final_n_ctx == 2048

    def test_oom_retry_floors_at_512_then_raises(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        class AlwaysOom:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                raise RuntimeError("CUDA out of memory")

        with patch("llama_cpp.Llama", AlwaysOom):
            with pytest.raises(RuntimeError, match="out of memory"):
                mod.load_llama_gguf(str(gguf), n_ctx=1000, detect_mmproj=False)
        # 1000 -> 512 -> floor reached, stop (no infinite loop, no crazy tiny ctx)
        assert min(c["n_ctx"] for c in FakeLlama.calls) == 512

    def test_non_oom_error_raises_immediately_no_retry(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        class BadModel:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                raise ValueError("corrupt gguf header")

        with patch("llama_cpp.Llama", BadModel):
            with pytest.raises(ValueError, match="corrupt gguf header"):
                mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert len(FakeLlama.calls) == 1  # no retry for a non-OOM failure

    def test_error_mentioning_bloom_or_room_is_not_retried_as_oom(self, tmp_path):
        """D9: a bare "oom" substring made BloomForCausalLM/"room" errors halve n_ctx up to 6 times."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        class BloomBad:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                raise RuntimeError("unsupported architecture BloomForCausalLM; no room for tensor")

        with patch("llama_cpp.Llama", BloomBad):
            with pytest.raises(RuntimeError, match="Bloom"):
                mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert len(FakeLlama.calls) == 1

    def test_mmproj_autodetected_in_same_directory(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "my-model-Q4_K_M.gguf"
        gguf.write_bytes(b"fake")
        mmproj = tmp_path / "mmproj-my-model-f16.gguf"
        mmproj.write_bytes(b"fake")

        fake_handler = MagicMock()
        with patch("llama_cpp.Llama", FakeLlama), \
             patch("llama_cpp.llama_chat_format.Qwen25VLChatHandler", return_value=fake_handler):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=True)
        assert result.vision is True
        assert result.mmproj_path == str(mmproj)
        assert FakeLlama.calls[0]["chat_handler"] is fake_handler

    def test_no_mmproj_present_loads_text_only(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "text-only-model.gguf"
        gguf.write_bytes(b"fake")

        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=True)
        assert result.vision is False
        assert result.mmproj_path is None
        assert "chat_handler" not in FakeLlama.calls[0]

    def test_kv_cache_type_passthrough(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        with patch("llama_cpp.Llama", FakeLlama):
            mod.load_llama_gguf(str(gguf), type_k=8, type_v=8, detect_mmproj=False)
        assert FakeLlama.calls[0]["type_k"] == 8
        assert FakeLlama.calls[0]["type_v"] == 8

    def test_zero_kv_type_omitted_not_sent_as_zero(self, tmp_path):
        """type_k/type_v=0 means 'not set' — llama_cpp treats an explicit 0
        as a real enum value, so it must be omitted, not passed as 0."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        with patch("llama_cpp.Llama", FakeLlama):
            mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert "type_k" not in FakeLlama.calls[0]
        assert "type_v" not in FakeLlama.calls[0]

    def test_n_threads_defaults_to_cpu_count_when_unset(self, tmp_path):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")

        with patch("llama_cpp.Llama", FakeLlama):
            mod.load_llama_gguf(str(gguf), n_threads=None, detect_mmproj=False)
        assert FakeLlama.calls[0]["n_threads"] > 0


class TestUnloadAllModels:
    """There are two independent 'currently loaded model' trackers: the
    global inference_engine and ModelManager's active provider. Nothing
    enforced that only one is ever resident — a data-prep helper loaded via
    ModelManager stayed loaded through a whole training run or benchmark
    judge pass that only unloaded inference_engine, silently doubling VRAM
    use. unload_all_models() must always free both, unconditionally."""

    def test_unloads_both_engines_when_both_have_models(self):
        from finetune_studio.models import llama_loader as mod

        fake_engine = MagicMock()
        fake_engine.model = object()
        fake_manager = MagicMock()

        with patch("finetune_studio.webui.app.inference_engine", fake_engine), \
             patch("finetune_studio.models.manager.get_manager", return_value=fake_manager):
            mod.unload_all_models()

        fake_engine.unload.assert_called_once()
        fake_manager.unload.assert_called_once()

    def test_skips_inference_engine_unload_when_no_model_loaded(self):
        """Don't call unload() on an engine that has nothing loaded —
        matches the existing guard pattern at every other unload call site."""
        from finetune_studio.models import llama_loader as mod

        fake_engine = MagicMock()
        fake_engine.model = None
        fake_manager = MagicMock()

        with patch("finetune_studio.webui.app.inference_engine", fake_engine), \
             patch("finetune_studio.models.manager.get_manager", return_value=fake_manager):
            mod.unload_all_models()

        fake_engine.unload.assert_not_called()
        # ModelManager.unload() is cheap/idempotent even with nothing
        # active, so it's always called rather than probed first.
        fake_manager.unload.assert_called_once()

    def test_one_engine_failing_does_not_block_the_other(self):
        """A broken/import-failing engine must not prevent freeing the
        other one — this runs before every load, so it can't be allowed
        to raise and abort the load that's about to happen."""
        from finetune_studio.models import llama_loader as mod

        fake_engine = MagicMock()
        fake_engine.model = object()
        fake_engine.unload.side_effect = RuntimeError("boom")
        fake_manager = MagicMock()

        with patch("finetune_studio.webui.app.inference_engine", fake_engine), \
             patch("finetune_studio.models.manager.get_manager", return_value=fake_manager):
            mod.unload_all_models()  # must not raise

        fake_manager.unload.assert_called_once()
