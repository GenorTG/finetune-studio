"""Accelerator detection — one answer to "what do we compute on?".

Single responsibility: turn whatever torch can see (NVIDIA CUDA, AMD ROCm,
Intel XPU, Apple MPS) into one frozen ``Accelerator``. GPU first, always: the
CPU is returned only when no GPU of any vendor is usable, and in that case
``degraded_reason`` says *why* when GPU hardware is physically present (CPU
torch wheel, missing driver) so a silent slow path is impossible.

ROCm builds of torch expose themselves through the ``torch.cuda`` namespace
(``torch.version.hip`` set), so ``cuda`` and ``rocm`` share every code path
and differ only in ``kind`` / labels.
"""
from __future__ import annotations

import logging
import os
import platform
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GPU_KINDS = ("cuda", "rocm", "xpu", "mps")
# PCI vendor ids as exposed by /sys/class/drm/card*/device/vendor.
_PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}


@dataclass(frozen=True)
class Accelerator:
    kind: str                       # cuda | rocm | xpu | mps | cpu
    index: int                      # torch device index (0 for mps/cpu)
    name: str
    total_gb: float
    free_gb: float
    compute_capability: tuple[int, int]
    supports_bf16: bool
    supports_flash_attention: bool  # SDPA flash kernels (Ampere+ NVIDIA)
    supports_4bit: bool             # bitsandbytes 4-bit usable on this backend
    runtime: str                    # "CUDA 13.0", "ROCm 7.0", "XPU", "Metal", "CPU"
    device_count: int = 1
    degraded_reason: str = ""       # set only when GPU hardware exists but is unusable

    @property
    def is_gpu(self) -> bool:
        return self.kind in GPU_KINDS

    @property
    def torch_namespace(self) -> str | None:
        """Name of the ``torch.<ns>`` module holding device APIs, or None on CPU."""
        return {"cuda": "cuda", "rocm": "cuda", "xpu": "xpu", "mps": "mps"}.get(self.kind)

    @property
    def torch_device(self) -> str:
        if self.kind in ("cuda", "rocm"):
            return f"cuda:{self.index}"
        if self.kind == "xpu":
            return f"xpu:{self.index}"
        return self.kind  # "mps" | "cpu"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["torch_device"] = self.torch_device
        d["is_gpu"] = self.is_gpu
        d["compute_capability"] = list(self.compute_capability)
        return d


def _import_torch() -> Any | None:
    """Import torch; None when absent or broken (version skew raises non-ImportError)."""
    try:
        import torch
        return torch
    except Exception as exc:  # noqa: BLE001 - probes must never take the app down
        log.warning("accel: torch unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def hardware_vendors() -> list[str]:
    """GPU vendors physically present (sysfs), independent of any driver/torch."""
    found: list[str] = []
    drm = Path("/sys/class/drm")
    try:
        cards = sorted(drm.glob("card[0-9]*/device/vendor"))
    except OSError:
        return found
    for vendor_file in cards:
        try:
            vendor = _PCI_VENDORS.get(vendor_file.read_text().strip().lower())
        except OSError:
            continue
        if vendor and vendor not in found:
            found.append(vendor)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        found.append("Apple")
    return found


def _degraded_reason(torch: Any | None) -> str:
    """Explain a CPU fallback when GPU hardware is present; empty if truly GPU-less."""
    # Intel is only a target when torch is an XPU build: desktops with an iGPU and a
    # CPU wheel have lost nothing, so they must not be nagged.
    present = [v for v in hardware_vendors() if v != "Intel"]
    if torch is not None and getattr(torch.version, "xpu", None) and "Intel" in hardware_vendors():
        present.append("Intel")
    if not present:
        return ""
    hw = ", ".join(present)
    if torch is None:
        return f"GPU hardware detected ({hw}) but PyTorch is not importable — run `bash install.sh --repair`."
    gpu_build = (getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None)
                 or getattr(torch.version, "xpu", None))
    if not gpu_build and "Apple" not in present:
        return (f"GPU hardware detected ({hw}) but this PyTorch ({torch.__version__}) is a "
                "CPU-only build — run `bash install.sh --repair` to install the matching GPU build.")
    return (f"GPU hardware detected ({hw}) but PyTorch found no usable device — check the driver "
            "(nvidia-smi / rocm-smi / xpu-smi) and visibility env vars "
            "(CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, FTS_DEVICE).")


def _cpu(torch: Any | None) -> Accelerator:
    return Accelerator(
        kind="cpu", index=0, name=f"CPU ({platform.processor() or platform.machine()})",
        total_gb=0.0, free_gb=0.0, compute_capability=(0, 0),
        supports_bf16=False, supports_flash_attention=False, supports_4bit=False,
        runtime="CPU", degraded_reason=_degraded_reason(torch),
    )


def _gb(n: float) -> float:
    return round(float(n) / (1024 ** 3), 2)


def _best_index(torch: Any, ns: Any, count: int) -> int:
    """Highest total VRAM (then compute capability) wins; reads props only (no context)."""
    def key(i: int) -> tuple[int, int, int, int]:
        p = ns.get_device_properties(i)
        return (getattr(p, "total_memory", 0), getattr(p, "major", 0), getattr(p, "minor", 0), -i)
    return max(range(count), key=key)


def _parse_forced(value: str) -> tuple[str, int | None]:
    """FTS_DEVICE: ``cpu`` | ``cuda`` | ``cuda:1`` | ``xpu:0`` | ``mps`` | ``auto``."""
    v = value.strip().lower()
    if ":" in v:
        kind, _, idx = v.partition(":")
        return kind, int(idx) if idx.isdigit() else None
    return v, None


def _cuda_like(torch: Any, forced_index: int | None) -> Accelerator | None:
    if not torch.cuda.is_available():
        return None
    count = torch.cuda.device_count()
    if count < 1:
        return None
    idx = forced_index if forced_index is not None and 0 <= forced_index < count \
        else _best_index(torch, torch.cuda, count)
    props = torch.cuda.get_device_properties(idx)
    total = getattr(props, "total_memory", getattr(props, "total_mem", 0))
    try:
        free = torch.cuda.mem_get_info(idx)[0]
    except Exception:  # noqa: BLE001 - free memory is advisory
        free = 0
    hip = getattr(torch.version, "hip", None)
    cc = (int(getattr(props, "major", 0)), int(getattr(props, "minor", 0)))
    try:
        # torch reads the *current* device, so make the selected one current; without emulation Pascal
        # is correctly "no bf16" (torch's default would call it supported via a slow emulated path).
        with torch.cuda.device(idx):
            bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except Exception:  # noqa: BLE001
        bf16 = bool(hip) or cc[0] >= 8
    return Accelerator(
        kind="rocm" if hip else "cuda", index=idx, name=props.name,
        total_gb=_gb(total), free_gb=_gb(free), compute_capability=cc,
        supports_bf16=bf16, supports_flash_attention=(not hip and cc[0] >= 8),
        supports_4bit=True,
        runtime=f"ROCm {hip}" if hip else f"CUDA {torch.version.cuda}",
        device_count=count,
    )


def _xpu(torch: Any, forced_index: int | None) -> Accelerator | None:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        return None
    count = xpu.device_count()
    if count < 1:
        return None
    idx = forced_index if forced_index is not None and 0 <= forced_index < count \
        else _best_index(torch, xpu, count)
    props = xpu.get_device_properties(idx)
    total = getattr(props, "total_memory", 0)
    try:
        free = xpu.mem_get_info(idx)[0]
    except Exception:  # noqa: BLE001
        free = 0
    try:
        bf16 = bool(xpu.is_bf16_supported())
    except Exception:  # noqa: BLE001
        bf16 = True
    return Accelerator(
        kind="xpu", index=idx, name=props.name, total_gb=_gb(total), free_gb=_gb(free),
        compute_capability=(0, 0), supports_bf16=bf16, supports_flash_attention=False,
        supports_4bit=True, runtime="XPU (oneAPI)", device_count=count,
    )


def _mps(torch: Any) -> Accelerator | None:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or not mps.is_available():
        return None
    try:
        import psutil
        total = psutil.virtual_memory().total
        free = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001
        total = free = 0
    return Accelerator(
        kind="mps", index=0, name=f"Apple GPU ({platform.machine()})",
        total_gb=_gb(total), free_gb=_gb(free), compute_capability=(0, 0),
        supports_bf16=False,  # fp16 is the safe default across M1..M4 / macOS versions
        supports_flash_attention=False, supports_4bit=False, runtime="Metal (MPS)",
    )


def _detect() -> Accelerator:
    torch = _import_torch()
    forced = os.environ.get("FTS_DEVICE", "auto").strip().lower() or "auto"
    kind_req, idx_req = _parse_forced(forced)
    if kind_req == "cpu":
        acc = _cpu(torch)
        return Accelerator(**{**asdict(acc), "degraded_reason": ""})
    if torch is None:
        return _cpu(None)
    probes = {
        "cuda": lambda: _cuda_like(torch, idx_req),
        "rocm": lambda: _cuda_like(torch, idx_req),
        "xpu": lambda: _xpu(torch, idx_req),
        "mps": lambda: _mps(torch),
    }
    order = [kind_req] if kind_req in probes else ["cuda", "xpu", "mps"]
    for kind in order:
        try:
            acc = probes[kind]()
        except Exception as exc:  # noqa: BLE001 - a broken backend must fall through to the next
            log.warning("accel: %s probe failed (%s: %s)", kind, type(exc).__name__, exc)
            continue
        if acc is not None:
            return acc
    return _cpu(torch)


_lock = threading.Lock()
_cached: Accelerator | None = None


def get_accelerator(refresh: bool = False) -> Accelerator:
    """Return the process-wide accelerator (cached; ``refresh=True`` re-probes free VRAM)."""
    global _cached
    with _lock:
        if _cached is None or refresh:
            _cached = _detect()
            if _cached.is_gpu:
                log.info("accel: %s on %s (%.1f GiB, %s)", _cached.torch_device,
                         _cached.name, _cached.total_gb, _cached.runtime)
            elif _cached.degraded_reason:
                log.warning("accel: running on CPU — %s", _cached.degraded_reason)
            else:
                log.info("accel: no GPU detected — running on CPU")
        return _cached


def reset_cache() -> None:
    """Forget the cached accelerator (tests, or after the env policy changed)."""
    global _cached
    with _lock:
        _cached = None
