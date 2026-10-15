"""The Smart Brain build job: embed one project's passages into its index, in its own process.

Params: ``build_id`` (a ``brain_builds`` row from ``knowledge.index.prepare``), ``project_id``, ``embedder`` (""
= the default), ``reset``, ``fact_cards``. The row records the outcome (``knowledge.index.run``); the lease choice
is ``webui.brain_build_jobs``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.jobs import store
    from finetune_studio.webui import brain_build_jobs

    result = brain_build_jobs.run_in_job(ctx.params, release_gpu=lambda: store.release_gpu(ctx.id))
    return {"status": "done", "build_id": ctx.params["build_id"], **result}
