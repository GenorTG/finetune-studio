"""The mining worker job: runs the queued ``mining_jobs`` rows of every project (``data.mining.worker``).

No params: the rows are the queue and each carries its helper and Vision reader seat. Started by the web app when it
queues files (``data.mining.scheduler.ensure_worker``); ends by itself once the queue is empty. No GPU lease: a local
helper is the model the inference worker already holds.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.data.mining import worker

    return worker.run(ctx.id, ctx.stop, lambda state, force: ctx.publish(state, force=force))
