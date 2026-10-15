"""Decide how a GGUF load fits in VRAM: which context length, then how many layers go on the GPU.

Policy (Genor 2026-10-06, extended 2026-10-08): an explicit n_ctx from the caller is never changed; when
weights + KV cache do not fit in free VRAM, the layers that do not fit run on the CPU instead of the load
failing. With no n_ctx ("auto") the model's NATIVE context is the default and is lowered only as far as
needed for every layer to fit on the GPU, never below ``MIN_AUTO_CTX`` (agentic/tool use needs it); below
that floor the layers are shed as before.
Pure functions over a header read and two numbers (free VRAM, requested layers) so the decision is
unit-testable without a GPU; the loader owns the retry loop that corrects an optimistic estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from finetune_studio.models.gguf_layers import gguf_header_values

GIB = 1024 ** 3

# Bytes per element of a ggml KV-cache type id (0 means "unset" in the loader = f16).
_KV_BYTES = {0: 2.0, 1: 2.0, 2: 0.5625, 3: 0.625, 6: 0.6875, 7: 0.75, 8: 1.0625, 30: 2.0}

# CUDA context + compute buffers + scratch the estimate does not model. Measured on the 3090:
# Qwen3-0.6B Q8_0 at 32k ctx = 0.63 GB weights + 3.76 GB KV, 5.66 GB resident after load.
DEFAULT_MARGIN_GB = 1.5

# Auto-context never goes below this (or the model's native context when that is smaller).
MIN_AUTO_CTX = 32768
# Sliding-window models cache only the window (+ the micro-batch) per window layer when the cache is not full-size.
SWA_EXTRA_CELLS = 512
# Window layers per global layer, for architectures whose pattern llama.cpp hard-codes (not in the file).
# gemma4 verified on the 12B: 8 global + 40 window layers of 48 (llama_kv_cache log, 2026-10-08).
_GLOBAL_EVERY = {"gemma3": 6, "gemma4": 6}


@dataclass(frozen=True)
class ModelShape:
    total_layers: int
    kv_heads: int
    head_dim_k: int
    head_dim_v: int
    file_gb: float
    known: bool          # False when the header gave no layer count: estimates are meaningless
    native_ctx: int = 0  # the model's trained context length (0 = not in the header)
    kv_layers: int = 0   # layers that hold a full-length KV cache (0 = all of them)
    swa_window: int = 0  # sliding-window size; > 0 means a window-attention model (see ``uses_swa``)

    @property
    def uses_swa(self) -> bool:
        return self.swa_window > 0


@dataclass(frozen=True)
class FitPlan:
    n_gpu_layers: int    # -1 = every layer on the GPU, otherwise how many (0 = CPU only)
    total_layers: int
    free_gb: float
    need_all_gb: float   # weights + KV for every layer + margin
    gpu_gb: float        # estimate for what is placed on the GPU
    requested: int       # what the caller asked for (-1 = automatic)
    reason: str          # "fits" | "capped" | "partial" | "cpu" | "unknown" | "requested"

    @property
    def partial(self) -> bool:
        return self.n_gpu_layers != -1 and self.total_layers > 0 and self.n_gpu_layers < self.total_layers


def read_shape(path: str) -> ModelShape:
    p = Path(path)
    file_gb = (p.stat().st_size / GIB) if p.is_file() else 0.0
    vals = gguf_header_values(str(p))

    def pick(suffix: str) -> int:
        for key, val in vals.items():
            if key.endswith(suffix):
                return int(val)
        return 0

    arch = str(next((k.split(".")[0] for k in vals if k.endswith(".block_count")), ""))
    layers = pick(".block_count")
    heads = pick(".attention.head_count")
    kv_heads = pick(".attention.head_count_kv") or heads or 8      # per-layer arrays are not scalars: assume MHA-ish
    embd = pick(".embedding_length")
    key_len = pick(".attention.key_length") or (embd // heads if heads and embd else 128)
    val_len = pick(".attention.value_length") or key_len
    interval = pick(".full_attention_interval")        # hybrid recurrent models: 1 attention layer in N holds KV
    window = pick(".attention.sliding_window")
    every = _GLOBAL_EVERY.get(arch, 0) if window else 0
    kv_layers = -(-layers // interval) if interval > 1 and layers else (-(-layers // every) if every and layers else 0)
    return ModelShape(layers, kv_heads, key_len, val_len, file_gb, known=layers > 0,
                      native_ctx=pick(".context_length"), kv_layers=kv_layers, swa_window=window)


def kv_gb_per_layer(shape: ModelShape, n_ctx: int, type_k: int = 0, type_v: int = 0) -> float:
    per_token = shape.kv_heads * (shape.head_dim_k * _KV_BYTES.get(type_k, 2.0)
                                  + shape.head_dim_v * _KV_BYTES.get(type_v, 2.0))
    return n_ctx * per_token / GIB


def kv_layer_share(shape: ModelShape) -> float:
    """Fraction of the layers that hold a full-length KV cache (hybrid / window-attention models: < 1)."""
    return (shape.kv_layers / shape.total_layers) if shape.kv_layers and shape.total_layers else 1.0


def kv_gb_per_layer_avg(shape: ModelShape, n_ctx: int, type_k: int = 0, type_v: int = 0) -> float:
    """KV per layer averaged over all layers (full-length layers only; window layers are small, see ``SWA_EXTRA_CELLS``)."""
    kv = kv_gb_per_layer(shape, n_ctx, type_k, type_v) * kv_layer_share(shape)
    if shape.uses_swa and shape.kv_layers:
        window_layers = (shape.total_layers - shape.kv_layers) / shape.total_layers
        kv += kv_gb_per_layer(shape, shape.swa_window + SWA_EXTRA_CELLS, type_k, type_v) * window_layers
    return kv


def estimate_gpu_gb(shape: ModelShape, n_ctx: int, type_k: int, type_v: int, gpu_layers: int,
                    margin_gb: float = DEFAULT_MARGIN_GB) -> float:
    """VRAM for ``gpu_layers`` layers (weights share + their KV) plus the fixed margin."""
    layers = shape.total_layers if gpu_layers < 0 else min(gpu_layers, shape.total_layers)
    share = layers / shape.total_layers
    return shape.file_gb * share + kv_gb_per_layer_avg(shape, n_ctx, type_k, type_v) * layers + margin_gb


def plan_gpu_layers(path: str, *, n_ctx: int, type_k: int = 0, type_v: int = 0, requested: int = -1,
                    free_gb: float, margin_gb: float = DEFAULT_MARGIN_GB) -> FitPlan:
    """Pick ``n_gpu_layers`` for ``path`` given ``free_gb`` of VRAM. ``requested`` is an upper bound
    (-1 = no preference); the plan never exceeds what fits and never touches ``n_ctx``."""
    shape = read_shape(path)
    if not shape.known:
        # No topology to reason with: trust the request; the loader's retry loop is the safety net.
        return FitPlan(requested, 0, free_gb, 0.0, 0.0, requested, "unknown")
    total = shape.total_layers
    cap = total if requested < 0 or requested >= total else requested
    need_all = estimate_gpu_gb(shape, n_ctx, type_k, type_v, -1, margin_gb)
    if cap == total and need_all <= free_gb:
        return FitPlan(-1, total, free_gb, need_all, need_all, requested, "fits")
    per_layer = shape.file_gb / total + kv_gb_per_layer_avg(shape, n_ctx, type_k, type_v)
    fit = max(0, math.floor((free_gb - margin_gb) / per_layer))
    chosen = min(cap, fit)
    gpu_gb = estimate_gpu_gb(shape, n_ctx, type_k, type_v, chosen, margin_gb) if chosen else 0.0
    if chosen >= total:
        return FitPlan(-1, total, free_gb, need_all, gpu_gb, requested, "fits")
    if chosen == 0:
        reason = "cpu"
    elif requested >= 0 and chosen == requested and fit >= requested:
        reason = "requested"      # the caller asked for fewer layers than would fit
    else:
        reason = "partial"        # VRAM-limited: the rest runs on the CPU
    return FitPlan(chosen, total, free_gb, need_all, gpu_gb, requested, reason)


def step_down(current: int, total: int) -> int | None:
    """Next, smaller layer count after a failed attempt at ``current`` (-1 = all); None when already CPU-only.

    With an unknown layer count (``total`` 0) the only step is straight to the CPU."""
    if total <= 0:
        return 0 if current != 0 else None
    layers = total if current < 0 else current
    if layers <= 0:
        return None
    nxt = int(layers * 0.7)
    return nxt if nxt < layers else layers - 1


def plan_auto_ctx(path: str, *, type_k: int = 0, type_v: int = 0, requested: int = -1, free_gb: float | None,
                  margin_gb: float = DEFAULT_MARGIN_GB) -> tuple[int, str]:
    """Context for an "auto" load: the model's native length, lowered only until every layer fits on the GPU.

    Returns ``(n_ctx, reason)``; reason is "native" (it fits), "fitted" (lowered to the largest multiple of 1024 that
    fits) or "floor" (even the floor does not fit all layers, or there is no GPU to plan against / the caller wants the
    CPU: the KV cache would live in RAM, so the floor is kept and layers are shed). Never below
    ``min(native, MIN_AUTO_CTX)``; a header without a context length gives ``MIN_AUTO_CTX``.
    """
    shape = read_shape(path)
    native = shape.native_ctx
    if native <= 0 or not shape.known:
        return MIN_AUTO_CTX, "native"
    floor = min(native, MIN_AUTO_CTX)
    if free_gb is None or requested == 0:
        return floor, "floor"

    def fits(ctx: int) -> bool:
        return estimate_gpu_gb(shape, ctx, type_k, type_v, -1, margin_gb) <= free_gb

    if fits(native):
        return native, "native"
    if not fits(floor):
        return floor, "floor"
    lo, hi = floor // 1024, native // 1024          # largest multiple of 1024 in [floor, native) that fits
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid * 1024) else (lo, mid)
    return lo * 1024, "fitted"
