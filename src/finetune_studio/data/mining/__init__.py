"""Per-file mining: build a file's data-point inventory, mine pairs against it, check, fix, repeat until covered.

One file = one job with fixed stages (``job``), every model round and every inventory item stored (``db.mining``), coverage
recomputed from the stored items and the file's pairs whenever it is asked for (``coverage``), files run in parallel with the
helper calls bounded globally (``scheduler``).
"""
from __future__ import annotations

from finetune_studio.data.mining.coverage import (
    FileCoverage,
    file_coverage,
    item_states,
    project_coverage,
)
from finetune_studio.data.mining.scheduler import (
    MiningError,
    get_scheduler,
    helper_info,
    interrupt_after_restart,
)

__all__ = [
    "FileCoverage",
    "MiningError",
    "file_coverage",
    "get_scheduler",
    "helper_info",
    "interrupt_after_restart",
    "item_states",
    "project_coverage",
]
