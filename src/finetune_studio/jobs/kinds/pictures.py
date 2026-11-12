"""The pictures job: the Vision reader reads the figures of a project's parsed files (``knowledge.pictures``).

Params: ``project_id``, ``file_ids`` ([] = every file with OCR-only figures), ``auto`` (started by an upload). No GPU
lease: a local reader is a model the one inference worker loads (and unloads again when it loaded it), like mining.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.knowledge import pictures

    result = pictures.run_in_job(str(ctx.params["project_id"]), list(ctx.params.get("file_ids") or []) or None,
                                 stop=ctx.stop, progress=lambda s: ctx.publish(s))
    return {"status": "done", **result}
