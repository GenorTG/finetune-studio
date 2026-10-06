"""Decide how many GGUF layers go on the GPU so a load fits — the context length never changes.

Policy (Genor 2026-10-06): n_ctx is fixed by the caller (agentic/tool use needs it). When weights + KV
cache do not fit in free VRAM, the layers that do not fit run on the CPU instead of the load failing.
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


@dataclass(frozen=True)
class ModelShape:
    total_layers: int
    kv_heads: int
    head_dim_k: int
    head_dim_v: int
    file_gb: float
    known: bool          # False when the header gave no layer count: estimates are meaningless


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

    layers = pick(".block_count")
    heads = pick(".attention.head_count")
    kv_heads = pick(".attention.head_count_kv") or heads or 8      # per-layer arrays are not scalars: assume MHA-ish
    embd = pick(".embedding_length")
    key_len = pick(".attention.key_length") or (embd // heads if heads and embd else 128)
    val_len = pick(".attention.value_length") or key_len
    return ModelShape(layers, kv_heads, key_len, val_len, file_gb, known=layers > 0)


def kv_gb_per_layer(shape: ModelShape, n_ctx: int, type_k: int = 0, type_v: int = 0) -> float:
    per_token = shape.kv_heads * (shape.head_dim_k * _KV_BYTES.get(type_k, 2.0)
                                  + shape.head_dim_v * _KV_BYTES.get(type_v, 2.0))
    return n_ctx * per_token / GIB


def estimate_gpu_gb(shape: ModelShape, n_ctx: int, type_k: int, type_v: int, gpu_layers: int,
                    margin_gb: float = DEFAULT_MARGIN_GB) -> float:
    """VRAM for ``gpu_layers`` layers (weights share + their KV) plus the fixed margin."""
    layers = shape.total_layers if gpu_layers < 0 else min(gpu_layers, shape.total_layers)
    share = layers / shape.total_layers
    return shape.file_gb * share + kv_gb_per_layer(shape, n_ctx, type_k, type_v) * layers + margin_gb


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
    per_layer = shape.file_gb / total + kv_gb_per_layer(shape, n_ctx, type_k, type_v)
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
