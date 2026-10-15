"""Device resolution for RAG embedders / rerankers (Studio side).

Policy: ``"auto"`` (the default everywhere) = GPU first through the
vendor-neutral accelerator layer (CUDA / ROCm / XPU / MPS), CPU only when the
host has no usable GPU. An explicit ``cpu`` / ``cuda:N`` / ``xpu`` / ``mps``
request is honoured as-is. The shipped standalone package carries its own
inline copy of this logic (``standalone_server._resolve_device``) because it
runs without finetune_studio installed.
"""
from __future__ import annotations

AUTO = ("", "auto")
# Encode batch size per device class: a GPU amortises launch cost over bigger
# batches; CPU stays small to bound RAM and latency.
GPU_BATCH = 64
CPU_BATCH = 16


def resolve_device(requested: str | None = "auto") -> str:
    """Map a user/config device string to a concrete torch device string."""
    req = (requested or "auto").strip().lower()
    if req not in AUTO:
        return req
    from finetune_studio.accel import get_accelerator
    return get_accelerator().torch_device  # "cpu" when no GPU is usable


def encode_batch_size(device: str) -> int:
    return CPU_BATCH if device.split(":")[0] == "cpu" else GPU_BATCH
