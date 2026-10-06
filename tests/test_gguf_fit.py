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
