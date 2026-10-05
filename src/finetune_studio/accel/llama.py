"""llama.cpp backend capability — does the installed build actually offload?

Single responsibility: report what ``llama-cpp-python`` was compiled with
(CUDA / HIP / Vulkan / SYCL / Metal) and derive the extra ``Llama()`` kwargs
that route work to the accelerator ``device.get_accelerator()`` chose. A GPU
host whose llama.cpp is a CPU wheel is the classic silent slowdown, so the
mismatch is returned as an explicit warning for the loader to surface.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any

from finetune_studio.accel.device import Accelerator, get_accelerator

log = logging.getLogger(__name__)

# ggml backend labels as printed by llama_print_system_info ("CUDA : ARCHS = …").
_GPU_BACKENDS = {"CUDA", "HIP", "ROCM", "VULKAN", "SYCL", "METAL", "OPENCL", "CANN", "MUSA"}
_CUDA_FAMILY = {"CUDA", "HIP", "ROCM", "MUSA"}


@dataclass(frozen=True)
class LlamaSupport:
    installed: bool
    version: str = ""
    gpu_offload: bool = False
    backends: tuple[str, ...] = ()   # GPU backends compiled in, upper-case
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backends"] = list(self.backends)
        return d


def parse_backends(system_info: str) -> tuple[str, ...]:
    """Pull GPU backend names out of ``llama_print_system_info()`` output."""
    found: list[str] = []
    for seg in system_info.split("|"):
        m = re.match(r"\s*([A-Za-z]+)\s*:", seg)
        if m and m.group(1).upper() in _GPU_BACKENDS and m.group(1).upper() not in found:
            found.append(m.group(1).upper())
    return tuple(found)


_lock = threading.Lock()
_cached: LlamaSupport | None = None


def llama_support(refresh: bool = False) -> LlamaSupport:
    """Probe the installed llama-cpp-python once per process."""
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        try:
            import llama_cpp
            info = llama_cpp.llama_print_system_info()
            text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
            _cached = LlamaSupport(
                installed=True, version=getattr(llama_cpp, "__version__", ""),
                gpu_offload=bool(llama_cpp.llama_supports_gpu_offload()),
                backends=parse_backends(text),
            )
        except Exception as exc:  # noqa: BLE001 - missing wheel / ABI skew / missing libcuda
            _cached = LlamaSupport(installed=False, error=f"{type(exc).__name__}: {exc}")
        return _cached


def reset_cache() -> None:
    global _cached
    with _lock:
        _cached = None


def llama_gpu_kwargs(
    acc: Accelerator | None = None, support: LlamaSupport | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Extra ``Llama()`` kwargs for the chosen device, plus user-facing warnings.

    ``n_gpu_layers`` stays the caller's business (-1 = all layers is already
    the right value on a GPU and is ignored on a CPU-only build); this adds
    ``main_gpu`` when several CUDA/HIP devices are visible and reports a
    GPU-present/GPU-unusable mismatch instead of degrading silently.
    """
    acc = acc or get_accelerator()
    support = support or llama_support()
    kwargs: dict[str, Any] = {}
    warnings: list[str] = []
    if not support.installed:
        msg = f"llama-cpp-python is not importable ({support.error}) — run `bash install.sh --repair`."
        return kwargs, [msg]
    if not acc.is_gpu:
        return kwargs, warnings
    if not support.gpu_offload:
        warnings.append(
            f"{acc.name} is available but llama-cpp-python {support.version} was built without "
            "GPU offload — GGUF inference is running on the CPU. "
            "Run `bash install.sh --repair` to install the GPU build."
        )
        return kwargs, warnings
    if acc.kind in ("cuda", "rocm") and acc.device_count > 1 and _CUDA_FAMILY & set(support.backends):
        kwargs["main_gpu"] = acc.index
    return kwargs, warnings
