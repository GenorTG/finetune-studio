"""GPU-layer planning for GGUF loads (models/gguf_fit.py): fixed context, adaptive offload.

Policy (Genor 2026-10-06): n_ctx is never traded away to make a model fit; layers that do not fit in
VRAM run on the CPU instead.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finetune_studio.models import gguf_fit
from finetune_studio.models.gguf_fit import GIB, ModelShape, plan_gpu_layers, step_down

SHAPE = ModelShape(total_layers=40, kv_heads=8, head_dim_k=128, head_dim_v=128, file_gb=10.0, known=True)


@pytest.fixture(autouse=True)
def _shape(monkeypatch):
    monkeypatch.setattr(gguf_fit, "read_shape", lambda path: SHAPE)


def test_kv_per_layer_matches_the_formula() -> None:
    # 32768 tokens * 8 kv heads * (128 K + 128 V) * 2 bytes (f16)
    assert gguf_fit.kv_gb_per_layer(SHAPE, 32768) == pytest.approx(32768 * 8 * 256 * 2 / GIB)
    # q8_0 KV (type 8) is ~half of f16
    assert gguf_fit.kv_gb_per_layer(SHAPE, 32768, 8, 8) == pytest.approx(
        gguf_fit.kv_gb_per_layer(SHAPE, 32768) * 1.0625 / 2)


def test_everything_fits_means_all_layers() -> None:
    plan = plan_gpu_layers("m.gguf", n_ctx=8192, free_gb=80.0)
    assert plan.n_gpu_layers == -1 and plan.reason == "fits" and not plan.partial


def test_tight_vram_places_only_the_layers_that_fit_and_stays_within_free() -> None:
    plan = plan_gpu_layers("m.gguf", n_ctx=32768, free_gb=12.0)
    assert 0 < plan.n_gpu_layers < SHAPE.total_layers and plan.partial and plan.reason == "partial"
    assert gguf_fit.estimate_gpu_gb(SHAPE, 32768, 0, 0, plan.n_gpu_layers) <= 12.0
    # one more layer would not fit
    assert gguf_fit.estimate_gpu_gb(SHAPE, 32768, 0, 0, plan.n_gpu_layers + 1) > 12.0


def test_longer_context_costs_layers_not_context() -> None:
    """The same free VRAM at a longer context offloads fewer layers; the plan has no n_ctx output at all."""
    short = plan_gpu_layers("m.gguf", n_ctx=4096, free_gb=14.0)
    long = plan_gpu_layers("m.gguf", n_ctx=32768, free_gb=14.0)
    assert (short.n_gpu_layers == -1) or short.n_gpu_layers > long.n_gpu_layers
    assert not hasattr(long, "n_ctx")


def test_nothing_fits_is_cpu_only() -> None:
    plan = plan_gpu_layers("m.gguf", n_ctx=32768, free_gb=1.0)
    assert plan.n_gpu_layers == 0 and plan.reason == "cpu"


def test_explicit_request_is_an_upper_bound() -> None:
    plan = plan_gpu_layers("m.gguf", n_ctx=4096, requested=12, free_gb=80.0)
    assert plan.n_gpu_layers == 12 and plan.reason == "requested"
    # asking for more than fits is trimmed to what fits
    trimmed = plan_gpu_layers("m.gguf", n_ctx=32768, requested=39, free_gb=12.0)
    assert 0 < trimmed.n_gpu_layers < 39 and trimmed.reason == "partial"
    # asking for >= every layer is the same as automatic
    assert plan_gpu_layers("m.gguf", n_ctx=4096, requested=40, free_gb=80.0).n_gpu_layers == -1


def test_unknown_topology_trusts_the_request_and_leaves_retry_as_the_safety_net(monkeypatch) -> None:
    monkeypatch.setattr(gguf_fit, "read_shape", lambda path: ModelShape(0, 0, 0, 0, 5.0, known=False))
    plan = plan_gpu_layers("m.gguf", n_ctx=4096, requested=-1, free_gb=2.0)
    assert plan.n_gpu_layers == -1 and plan.reason == "unknown"


def test_step_down_is_strictly_decreasing_and_ends_on_cpu() -> None:
    seq, cur = [], -1
    while (cur := step_down(cur, 40)) is not None:
        seq.append(cur)
    assert seq == sorted(seq, reverse=True) and len(set(seq)) == len(seq)
    assert seq[0] == 28 and seq[-1] == 0
    assert step_down(0, 40) is None
    # unknown depth: straight to the CPU, then nothing
    assert step_down(-1, 0) == 0 and step_down(0, 0) is None


def test_read_shape_reads_a_real_gguf_header(monkeypatch, tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    monkeypatch.undo()   # use the real read_shape, not the autouse stub
    path = tmp_path / "tiny.gguf"
    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_block_count(7)
    w.add_head_count(16)
    w.add_head_count_kv(8)
    w.add_embedding_length(1024)
    w.add_key_length(128)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    shape = gguf_fit.read_shape(str(path))
    assert (shape.total_layers, shape.kv_heads, shape.head_dim_k, shape.known) == (7, 8, 128, True)
    assert shape.file_gb > 0


# ── auto context (Genor 2026-10-08): native by default, lowered only to fit ──────────────────────────────

QWEN35 = {"qwen35.block_count": 32, "qwen35.context_length": 262144, "qwen35.embedding_length": 4096,
          "qwen35.attention.head_count": 16, "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
          "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4}
GEMMA4 = {"gemma4.block_count": 48, "gemma4.context_length": 131072, "gemma4.embedding_length": 3840,
          "gemma4.attention.head_count": 16, "gemma4.attention.head_count_kv": 1, "gemma4.attention.key_length": 512,
          "gemma4.attention.value_length": 512, "gemma4.attention.sliding_window": 1024}


@pytest.fixture
def real_shape(monkeypatch, tmp_path):
    """Undo the module-wide read_shape stub and feed read_shape() header values + a 5 GB file size."""
    import importlib

    importlib.reload(gguf_fit)   # drops the autouse stub
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    size = {"gb": 5.24}
    monkeypatch.setattr(Path, "stat", lambda self, **kw: type("S", (), {"st_size": int(size["gb"] * GIB)})())
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    def use(vals: dict, file_gb: float = 5.24):
        size["gb"] = file_gb
        monkeypatch.setattr(gguf_fit, "gguf_header_values", lambda path: vals)
        return gguf_fit.read_shape("m.gguf")

    return use


def test_hybrid_and_window_models_hold_kv_in_few_layers(real_shape) -> None:
    qwen = real_shape(QWEN35)
    assert (qwen.native_ctx, qwen.kv_layers, qwen.uses_swa) == (262144, 8, False)      # 1 attention layer in 4
    gemma = real_shape(GEMMA4, 6.87)
    assert (gemma.native_ctx, gemma.kv_layers, gemma.swa_window) == (131072, 8, 1024)  # 8 global of 48 (measured)


def test_estimate_tracks_the_measured_vram_of_the_native_loads(real_shape) -> None:
    """Measured on the 3090 (2026-10-08): Qwen3.5-9B Q4_K_M 262144 ctx = 13.3 GB, Gemma 4 12B 131072 = 10.0 GB."""
    qwen = real_shape(QWEN35)
    est = gguf_fit.estimate_gpu_gb(qwen, 262144, 0, 0, -1)
    assert 13.3 <= est <= 13.3 * 1.25            # slightly pessimistic is the safe side
    # the old all-layers formula would have said ~37 GB and refused the load
    assert est < 20
    gemma = real_shape(GEMMA4, 6.87)
    est = gguf_fit.estimate_gpu_gb(gemma, 131072, 0, 0, -1)
    assert 10.0 <= est <= 10.0 * 1.25


def test_auto_ctx_is_native_when_it_fits_and_fitted_when_not(real_shape) -> None:
    real_shape(QWEN35)
    assert gguf_fit.plan_auto_ctx("m.gguf", free_gb=22.0) == (262144, "native")
    ctx, reason = gguf_fit.plan_auto_ctx("m.gguf", free_gb=10.0)
    shape = gguf_fit.read_shape("m.gguf")
    assert reason == "fitted" and 32768 < ctx < 262144 and ctx % 1024 == 0
    assert gguf_fit.estimate_gpu_gb(shape, ctx, 0, 0, -1) <= 10.0 < gguf_fit.estimate_gpu_gb(shape, ctx + 1024, 0, 0, -1)


def test_auto_ctx_floor_and_no_gpu(real_shape) -> None:
    real_shape(QWEN35)
    assert gguf_fit.plan_auto_ctx("m.gguf", free_gb=5.0) == (32768, "floor")       # weights alone nearly fill it
    assert gguf_fit.plan_auto_ctx("m.gguf", free_gb=None) == (32768, "floor")      # no GPU: KV would live in RAM
    assert gguf_fit.plan_auto_ctx("m.gguf", free_gb=22.0, requested=0) == (32768, "floor")   # CPU requested
    real_shape({k: v for k, v in QWEN35.items() if not k.endswith("context_length")})
    assert gguf_fit.plan_auto_ctx("m.gguf", free_gb=22.0)[0] == 32768              # header without a context length
    real_shape({**QWEN35, "qwen35.context_length": 8192})
    assert gguf_fit.plan_auto_ctx("m.gguf", free_gb=22.0) == (8192, "native")      # small models keep their own limit
