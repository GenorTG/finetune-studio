"""A Chat / Inference model load while a GPU job holds the lease: refused at once, naming the job.

Training, merge/export, Smart Brain builds and picture reading wait for each other in the GPU lease queue
(``JobSpec.queue_gpu``). An interactive model load does not: someone waiting to chat must not sit silently behind a
20-minute build, and the supervisor unloads the chat model before every GPU job anyway. So the load answers 409
``gpu_busy`` with what holds the GPU and how long it has been running (``jobs.launch.gpu_busy_info``).

Plain ``def``: the accelerator probe must stay off the event loop (callers use ``asyncio.to_thread``).
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs import launch


def gpu_busy_refusal(what: str = "A chat model cannot load") -> dict[str, Any] | None:
    """The 409 body when a GPU job holds the lease and the model would load on a GPU, else None.

    On a CPU-only machine nothing is refused: the model loads into RAM, it takes nothing from the job."""
    info = launch.gpu_busy_info()
    if info is None:
        return None
    from finetune_studio.accel import get_accelerator
    if not get_accelerator().is_gpu:
        return None
    return {"status": "error", "loaded": False, "model": None, "code": "gpu_busy",
            "detail": launch.gpu_busy_message(info, what), "gpu_holder": info}
