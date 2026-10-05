"""Backend-neutral torch device operations.

Single responsibility: the handful of ``torch.cuda.*`` calls scattered through
loading / training / profiling, expressed once for cuda, rocm, xpu, mps and cpu.
Every function is safe to call on a CPU-only host (no-op / sensible default).
"""
from __future__ import annotations

import contextlib
import gc
import re
from typing import Any

from finetune_studio.accel.device import Accelerator, get_accelerator

GIB = 1024 ** 3


def _ns(acc: Accelerator) -> Any | None:
    """The ``torch.<cuda|xpu|mps>`` namespace for this accelerator, or None on CPU."""
    name = acc.torch_namespace
    if name is None:
        return None
    import torch
    return getattr(torch, name, None)


def device_map(acc: Accelerator | None = None) -> dict[str, Any] | str:
    """``device_map`` for ``from_pretrained``: full offload to the chosen GPU, else ``"cpu"``."""
    acc = acc or get_accelerator()
    if acc.kind in ("cuda", "rocm", "xpu"):
        return {"": acc.index}
    if acc.kind == "mps":
        return {"": "mps"}
    return "cpu"


def auto_device_map(acc: Accelerator | None = None) -> str:
    """Spill-to-RAM placement used after an OOM; plain ``cpu`` when there is no GPU."""
    return "auto" if (acc or get_accelerator()).is_gpu else "cpu"


def torch_dtype(acc: Accelerator | None = None) -> Any:
    """Best compute dtype: bf16 where supported, fp16 on other GPUs, fp32 on CPU."""
    import torch
    acc = acc or get_accelerator()
    if not acc.is_gpu:
        return torch.float32
    return torch.bfloat16 if acc.supports_bf16 else torch.float16


def oom_errors() -> tuple[type[BaseException], ...]:
    """Exception types that mean "out of device memory" for the active backend."""
    import torch
    found: list[type[BaseException]] = []
    for ns in ("cuda", "xpu"):
        exc = getattr(getattr(torch, ns, None), "OutOfMemoryError", None)
        if isinstance(exc, type):
            found.append(exc)
    return (*found, RuntimeError)


def _dev(acc: Accelerator) -> tuple[int, ...]:
    """Explicit device argument for ``torch.<ns>`` calls: torch's *current* device is not necessarily
    the accelerator we picked (FTS_DEVICE=cuda:1, or the best card is not index 0); mps has none."""
    return (acc.index,) if acc.kind in ("cuda", "rocm", "xpu") else ()


def _on_device(acc: Accelerator, ns: Any) -> Any:
    """Context making ``acc.index`` current for calls that take no device argument; it never
    initialises a driver context that does not exist yet (cache release must stay a no-op then)."""
    initialised = getattr(ns, "is_initialized", None)
    if _dev(acc) and hasattr(ns, "device") and (initialised is None or initialised()):
        return ns.device(acc.index)
    return contextlib.nullcontext()


def empty_cache() -> None:
    """Release cached allocator blocks (after ``gc``) on the accelerator in use; never raises."""
    gc.collect()
    acc = get_accelerator()
    try:
        ns = _ns(acc)
        if ns is not None and hasattr(ns, "empty_cache"):
            with _on_device(acc, ns):
                ns.empty_cache()
    except Exception:  # noqa: BLE001, S110 - cache release is best effort
        pass


def synchronize() -> None:
    acc = get_accelerator()
    ns = _ns(acc)
    if ns is not None and hasattr(ns, "synchronize"):
        ns.synchronize(*_dev(acc))


def reset_peak_memory() -> None:
    acc = get_accelerator()
    ns = _ns(acc)
    if ns is not None and hasattr(ns, "reset_peak_memory_stats"):
        ns.reset_peak_memory_stats(*_dev(acc))


def peak_memory_gb() -> float:
    """Peak allocated memory of the accelerator in use, GiB since the last reset (0.0 without support)."""
    acc = get_accelerator()
    ns = _ns(acc)
    if ns is not None and hasattr(ns, "max_memory_allocated"):
        return float(ns.max_memory_allocated(*_dev(acc))) / GIB
    return 0.0


def mem_info_gb(acc: Accelerator | None = None) -> tuple[float, float]:
    """``(free_gb, total_gb)`` of the accelerator right now; ``(0, 0)`` on CPU."""
    acc = acc or get_accelerator()
    ns = _ns(acc)
    if ns is not None and hasattr(ns, "mem_get_info"):
        free, total = ns.mem_get_info(acc.index) if acc.kind != "mps" else ns.mem_get_info()
        return free / GIB, total / GIB
    return 0.0, 0.0


def enable_fast_matmul() -> None:
    """TF32 matmul on NVIDIA Ampere+ — free throughput for fp32 paths; no-op elsewhere."""
    acc = get_accelerator()
    if acc.kind == "cuda" and acc.compute_capability[0] >= 8:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


# Substrings that mean "the device ran out of memory" in torch / llama.cpp / ggml
# error text across CUDA, HIP, XPU/SYCL, Metal, Vulkan. "OOM" is matched as a whole word
# only: a bare substring hits BloomForCausalLM / "room" / "zoom" and turns real load errors
# into silent retries.
_OOM_MARKERS = (
    "out of memory", "failed to allocate", "vram", "insufficient memory",
    "outofdevicememory", "out_of_device_memory", "out of device memory", "cannot allocate",
)
_OOM_WORD = re.compile(r"\boom\b")


def is_oom_message(text: str) -> bool:
    """True when ``text`` (an exception message) looks like a device-memory failure."""
    low = text.lower()
    return bool(_OOM_WORD.search(low)) or any(m in low for m in _OOM_MARKERS)


def is_oom_error(exc: BaseException) -> bool:
    """True when ``exc`` is a device OOM (typed torch OOM or a matching RuntimeError)."""
    typed = tuple(t for t in oom_errors() if t is not RuntimeError)
    if typed and isinstance(exc, typed):
        return True
    return isinstance(exc, RuntimeError) and is_oom_message(str(exc))
