"""How much memory a model about to load may use — one answer for discrete GPUs, unified memory and the CPU.

Single responsibility: turn the chosen ``Accelerator`` into a ``MemoryBudget``:

- ``device``: a discrete GPU with its own memory (NVIDIA, AMD, Intel Arc cards) — the device's free memory.
- ``shared``: a GPU that shares system RAM with the CPU (``Accelerator.unified_memory``: Apple MPS, an integrated
  GPU / APU, Jetson) — the system's available RAM, because the device's own "VRAM" number is a small carve-out or
  0 and would refuse every load (or a reported number that is RAM the CPU is using too).
- ``system``: no usable GPU — the system's available RAM.

Best-effort and never raises: an unreadable figure is 0.0 with ``known`` False, and callers then let the loader's
own fit logic decide instead of refusing on a guess.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from finetune_studio.accel.device import Accelerator, get_accelerator

log = logging.getLogger(__name__)
GIB = 1024 ** 3

POOL_DEVICE, POOL_SHARED, POOL_SYSTEM = "device", "shared", "system"


@dataclass(frozen=True)
class MemoryBudget:
    pool: str            # device | shared | system
    free_gb: float
    total_gb: float
    device: str          # the torch device the budget is for ("cuda:0", "mps", "cpu")
    known: bool = True   # False = the figure could not be read

    @property
    def label(self) -> str:
        return {POOL_DEVICE: "free GPU memory", POOL_SHARED: "free shared memory (GPU and CPU share the RAM)",
                POOL_SYSTEM: "free RAM"}[self.pool]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "label": self.label}


def _system_ram() -> tuple[float, float] | None:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available / GIB, vm.total / GIB
    except Exception as exc:  # noqa: BLE001 - no psutil / no /proc: unknown
        log.debug("accel.memory: system RAM unreadable (%s)", exc)
        return None


def _device_mem(acc: Accelerator) -> tuple[float, float] | None:
    from finetune_studio.accel.ops import mem_info_gb
    try:
        free, total = mem_info_gb(acc)
    except Exception as exc:  # noqa: BLE001 - a busy or broken driver: unknown, not 0
        log.debug("accel.memory: device memory unreadable (%s)", exc)
        return None
    return (free, total) if total > 0 else None


def memory_budget(acc: Accelerator | None = None) -> MemoryBudget:
    """The budget a model loading on ``acc`` (default: the chosen accelerator) has right now."""
    acc = acc or get_accelerator()
    if acc.is_gpu and not acc.unified_memory:
        got = _device_mem(acc)
        if got is None:
            return MemoryBudget(POOL_DEVICE, 0.0, acc.total_gb, acc.torch_device, known=False)
        return MemoryBudget(POOL_DEVICE, round(got[0], 2), round(got[1], 2), acc.torch_device)
    pool = POOL_SHARED if acc.is_gpu else POOL_SYSTEM
    ram = _system_ram()
    if ram is None:
        return MemoryBudget(pool, 0.0, 0.0, acc.torch_device, known=False)
    return MemoryBudget(pool, round(ram[0], 2), round(ram[1], 2), acc.torch_device)
