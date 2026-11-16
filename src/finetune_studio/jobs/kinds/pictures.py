"""The pictures job: the Vision reader reads the figures of a project's parsed files (``knowledge.pictures``).

Params: ``project_id``, ``file_ids`` ([] = every file with OCR-only figures), ``auto`` (started by an upload). A
local reader on the GPU takes the GPU lease (``spec.gpu``, queued behind a build or training instead of refused): the
one inference worker loads it (and unloads it again when it loaded it). An API reader, or a CPU-only host, takes none.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.knowledge import pictures

    result = pictures.run_in_job(str(ctx.params["project_id"]), list(ctx.params.get("file_ids") or []) or None,
                                 stop=ctx.stop, progress=lambda s: ctx.publish(s), holds_gpu=ctx.spec.gpu,
                                 reader_device=ctx.params.get("reader_device") or None)
    return {"status": "done", **result}
