"""Tests for the single canonical GGUF loader (models/llama_loader.py).

Before this module existed, LocalGGUFProvider.load() (models/providers.py)
and InferenceEngine._load_gguf() (testing/inference.py) each independently
built a llama_cpp.Llama(**kwargs) call and drifted: only one had OOM-retry,
only the other had mmproj/vision detection and KV-cache-type support. Both now
call load_llama_gguf() — these tests pin its behavior directly so a future edit
can't silently reintroduce either gap. Load policy: n_ctx is never shrunk, GPU
layers step down (layers that do not fit run on the CPU).
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


def _gpu_host(monkeypatch, mod, *, free_gb: float, layers: int, file_gb: float, native_ctx: int = 0,
              swa_window: int = 0) -> None:
    """Pretend: a GPU with ``free_gb`` free, and a GGUF with ``layers`` layers of ``file_gb`` total."""
    from finetune_studio.models.gguf_fit import ModelShape

    shape = ModelShape(total_layers=layers, kv_heads=8, head_dim_k=128, head_dim_v=128, file_gb=file_gb, known=True,
                       native_ctx=native_ctx, swa_window=swa_window)
    monkeypatch.setattr(mod, "_gpu_capable", lambda: True)
    monkeypatch.setattr(mod, "_free_vram_gb", lambda: free_gb)
    monkeypatch.setattr(mod, "read_shape", lambda path: shape)
    monkeypatch.setattr("finetune_studio.models.gguf_fit.read_shape", lambda path: shape)


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

    def test_oom_steps_gpu_layers_down_and_keeps_the_context(self, tmp_path, monkeypatch):
        """Policy (Genor 2026-10-06): n_ctx is never shrunk; layers that do not fit go to the CPU."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=100.0, layers=40, file_gb=10.0)

        attempts = {"n": 0}

        class OomThenOk:
            def __init__(self, **kwargs):
                attempts["n"] += 1
                FakeLlama.calls.append(kwargs)
                if attempts["n"] < 3:
                    raise RuntimeError("CUDA out of memory")

        with patch("llama_cpp.Llama", OomThenOk):
            result = mod.load_llama_gguf(str(gguf), n_ctx=8192, n_gpu_layers=-1, detect_mmproj=False)
        assert [c["n_gpu_layers"] for c in FakeLlama.calls] == [-1, 28, 19]
        assert {c["n_ctx"] for c in FakeLlama.calls} == {8192}        # never shrunk
        assert result.final_n_ctx == 8192 and result.n_gpu_layers == 19
        assert result.offload == "partial" and result.total_layers == 40
        assert any("19/40 layers" in w for w in result.warnings)

    def test_native_oom_is_detected_when_the_exception_text_says_nothing(self, tmp_path, monkeypatch):
        """Real llama-cpp-python raises a bare ValueError('Failed to load model from file'); the OOM is
        only in the native log, so the retry must be driven by that."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=100.0, layers=40, file_gb=10.0)
        monkeypatch.setattr(mod.llama_native_log, "oom_since", lambda pos: len(FakeLlama.calls) < 2)

        class GenericFailure:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                if len(FakeLlama.calls) < 2:
                    raise ValueError("Failed to load model from file: /x/model.gguf")

        with patch("llama_cpp.Llama", GenericFailure):
            result = mod.load_llama_gguf(str(gguf), n_ctx=4096, detect_mmproj=False)
        assert [c["n_gpu_layers"] for c in FakeLlama.calls] == [-1, 28]
        assert result.n_gpu_layers == 28

    def test_plan_limits_layers_before_the_first_attempt(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        # 10 GB of weights, 1 GB KV per 8k per layer-ish: only a slice fits in 6 GB free.
        _gpu_host(monkeypatch, mod, free_gb=6.0, layers=40, file_gb=10.0)
        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(str(gguf), n_ctx=8192, detect_mmproj=False)
        assert len(FakeLlama.calls) == 1                      # planned, so no wasted failed load
        assert 0 < FakeLlama.calls[0]["n_gpu_layers"] < 40
        assert FakeLlama.calls[0]["n_ctx"] == 8192
        assert result.offload == "partial"

    def test_explicit_layer_count_is_an_upper_bound(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=100.0, layers=40, file_gb=10.0)
        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(str(gguf), n_gpu_layers=12, detect_mmproj=False)
        assert FakeLlama.calls[0]["n_gpu_layers"] == 12 and result.n_gpu_layers == 12
        assert any("as requested" in w and "12/40" in w for w in result.warnings)   # not blamed on VRAM

    def test_legacy_99_means_all_layers(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=100.0, layers=40, file_gb=10.0)
        with patch("llama_cpp.Llama", FakeLlama):
            mod.load_llama_gguf(str(gguf), n_gpu_layers=99, detect_mmproj=False)
        assert FakeLlama.calls[0]["n_gpu_layers"] == -1

    def test_oom_even_on_cpu_raises_an_honest_error_without_touching_ctx(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=100.0, layers=40, file_gb=10.0)

        class AlwaysOom:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                raise RuntimeError("CUDA out of memory")

        with patch("llama_cpp.Llama", AlwaysOom), \
             pytest.raises(mod.LlamaLoadError, match="context length was left unchanged"):
            mod.load_llama_gguf(str(gguf), n_ctx=32768, detect_mmproj=False)
        assert {c["n_ctx"] for c in FakeLlama.calls} == {32768}
        assert FakeLlama.calls[-1]["n_gpu_layers"] == 0       # the last attempt was CPU-only
        # ...and truly CPU-only: no KV cache or host-op offload to the (nearly full) device either
        assert FakeLlama.calls[-1]["offload_kqv"] is False and FakeLlama.calls[-1]["op_offload"] is False
        assert FakeLlama.calls[0]["offload_kqv"] is True and FakeLlama.calls[0]["op_offload"] is True
        assert len(FakeLlama.calls) == mod.MAX_LOAD_ATTEMPTS   # bounded, and the last attempt is the CPU-only one

    def test_auto_context_loads_the_native_window_when_it_fits(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=1000.0, layers=40, file_gb=10.0, native_ctx=131072)
        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=False)    # default n_ctx = AUTO_N_CTX
        assert FakeLlama.calls[0]["n_ctx"] == 131072 and result.final_n_ctx == 131072
        assert not any("Context lowered" in w for w in result.warnings)

    def test_auto_context_is_lowered_to_fit_but_an_explicit_one_is_not(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        # 10 GB weights; at 131072 ctx the KV alone is 40 layers * 4 GiB > 20 GB free -> the plan lowers the context
        _gpu_host(monkeypatch, mod, free_gb=20.0, layers=40, file_gb=10.0, native_ctx=131072)
        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert mod.MIN_AUTO_CTX <= FakeLlama.calls[0]["n_ctx"] < 131072 and FakeLlama.calls[0]["n_ctx"] % 1024 == 0
        assert FakeLlama.calls[0]["n_gpu_layers"] == -1                      # every layer stays on the GPU
        assert any("Context lowered" in w and "131072" in w for w in result.warnings)
        _reset()
        with patch("llama_cpp.Llama", FakeLlama):
            mod.load_llama_gguf(str(gguf), n_ctx=131072, detect_mmproj=False)
        assert FakeLlama.calls[0]["n_ctx"] == 131072                         # explicit: kept, layers are shed instead
        assert FakeLlama.calls[0]["n_gpu_layers"] != -1

    def test_auto_context_halves_on_oom_before_shedding_layers(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=1000.0, layers=40, file_gb=10.0, native_ctx=131072)

        class OomAbove40k:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                if kwargs["n_ctx"] > 40000:
                    raise RuntimeError("CUDA out of memory")

        with patch("llama_cpp.Llama", OomAbove40k):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert [c["n_ctx"] for c in FakeLlama.calls] == [131072, 65536, 32768]
        assert {c["n_gpu_layers"] for c in FakeLlama.calls} == {-1}
        assert result.final_n_ctx == 32768 and result.offload == "gpu"

    def test_auto_context_never_goes_below_the_floor_then_sheds_layers(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=1000.0, layers=40, file_gb=10.0, native_ctx=131072)

        class OomUnlessFewLayers:
            def __init__(self, **kwargs):
                FakeLlama.calls.append(kwargs)
                if kwargs["n_gpu_layers"] != 0:
                    raise RuntimeError("CUDA out of memory")

        with patch("llama_cpp.Llama", OomUnlessFewLayers):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert min(c["n_ctx"] for c in FakeLlama.calls) == mod.MIN_AUTO_CTX
        assert result.final_n_ctx == mod.MIN_AUTO_CTX and result.n_gpu_layers == 0

    def test_window_attention_models_get_a_window_sized_cache(self, tmp_path, monkeypatch):
        """Gemma 4 12B: 18.3 GB at 32k with the full-size window cache, 10.0 GB at its native 131k without it."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        _gpu_host(monkeypatch, mod, free_gb=1000.0, layers=48, file_gb=7.0, native_ctx=131072, swa_window=1024)

        class SwaLlama:
            def __init__(self, swa_full=True, **kwargs):
                FakeLlama.calls.append({"swa_full": swa_full, **kwargs})

        with patch("llama_cpp.Llama", SwaLlama):
            mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert FakeLlama.calls[0]["swa_full"] is False
        _reset()
        _gpu_host(monkeypatch, mod, free_gb=1000.0, layers=40, file_gb=10.0, native_ctx=131072)   # no window
        with patch("llama_cpp.Llama", SwaLlama):
            mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert FakeLlama.calls[0]["swa_full"] is True                       # untouched default

    def test_micro_batch_is_capped_for_the_cuda_abort(self, tmp_path, monkeypatch):
        """llama.cpp aborts the process on a full 512-token micro-batch on some Q8_0 models; cap it."""
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        monkeypatch.delenv("FTS_LLAMA_UBATCH", raising=False)
        with patch("llama_cpp.Llama", FakeLlama):
            mod.load_llama_gguf(str(gguf), n_batch=512, detect_mmproj=False)
            mod.load_llama_gguf(str(gguf), n_batch=128, detect_mmproj=False)
            monkeypatch.setenv("FTS_LLAMA_UBATCH", "384")
            mod.load_llama_gguf(str(gguf), n_batch=512, detect_mmproj=False)
        assert [c["n_ubatch"] for c in FakeLlama.calls] == [256, 128, 384]
        assert [c["n_batch"] for c in FakeLlama.calls] == [512, 128, 512]

    def test_cpu_only_host_reports_cpu_without_a_degradation_warning(self, tmp_path, monkeypatch):
        from finetune_studio.models import llama_loader as mod

        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"fake")
        monkeypatch.setattr(mod, "_gpu_capable", lambda: False)
        with patch("llama_cpp.Llama", FakeLlama):
            result = mod.load_llama_gguf(str(gguf), detect_mmproj=False)
        assert result.offload == "cpu" and not any("layers" in w for w in result.warnings)

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
        """D9: a bare "oom" substring made BloomForCausalLM/"room" errors trigger up to 6 retries."""
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
