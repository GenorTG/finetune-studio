"""Optional fast kernels that transformers picks up by import alone.

Qwen3.5 / Qwen3-Next linear-attention layers call ``fla`` (flash-linear-attention, Triton) and ``causal_conv1d``
(CUDA extension) whenever they import, else a PyTorch reference. ``scripts/accel_plan.py`` installs them on NVIDIA
sm_75+ hosts, but the device a process uses is chosen at run time (Settings -> Compute device): on a Pascal card,
the CPU or another backend those kernels would be called and fail. ``guard_linear_attention`` hides them from the
importer for such a device, so transformers takes its PyTorch path instead.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

log = logging.getLogger(__name__)

LINEAR_ATTN_MODULES = ("fla", "causal_conv1d")
LINEAR_ATTN_MIN_CC = (7, 5)
_KINDS_WITHOUT_KERNELS = ("cpu", "mps", "xpu")   # ROCm is left alone: Triton runs there and nothing installs them


def linear_attention_supported(acc: Any) -> bool:
    if acc.kind in _KINDS_WITHOUT_KERNELS:
        return False
    if acc.kind == "cuda":
        cc = tuple(acc.compute_capability or (0, 0))
        return cc >= LINEAR_ATTN_MIN_CC
    return True


def guard_linear_attention(acc: Any) -> list[str]:
    """Block the kernel packages for this process when ``acc`` cannot run them; returns the names blocked."""
    if linear_attention_supported(acc):
        return []
    blocked = []
    for name in LINEAR_ATTN_MODULES:
        if sys.modules.get(name) is not None:
            log.warning("accel: %s is already imported; it cannot be hidden from transformers on %s", name, acc.name)
            continue
        sys.modules[name] = None   # type: ignore[assignment]  # import now raises ImportError -> PyTorch path
        blocked.append(name)
    if blocked:
        log.info("accel: linear-attention kernels %s disabled on %s (%s): Qwen3.5 uses the PyTorch path",
                 ", ".join(blocked), acc.name, acc.kind)
    return blocked


_DISTRIBUTIONS = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d"}


def linear_attention_status(acc: Any) -> dict[str, Any]:
    """Installed versions (package metadata, nothing imported) and whether this device uses them."""
    from importlib import metadata

    versions: dict[str, str] = {}
    for dist in _DISTRIBUTIONS.values():
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return {"installed": versions, "active": bool(versions) and linear_attention_supported(acc)}
