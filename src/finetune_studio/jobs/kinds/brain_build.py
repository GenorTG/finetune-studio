"""The Smart Brain build job: embed one project's passages into its index, in its own process.

Params: ``build_id`` (a ``brain_builds`` row from ``knowledge.index.prepare`` / ``prepare_rebuild_vectors``),
``project_id``, ``embedder`` ("" = the default), ``reset``, ``fact_cards``, ``mode`` (``build`` | ``quick`` =
parse the pending library files first | ``re-embed`` = replace the vectors only), ``file_ids`` (the quick
index's parse step, [] = every pending file). The row records the outcome (``knowledge.index``); the lease
choice is ``webui.brain_build_jobs``. The quick index's parse step is published as the job state
(``{"phase": "parsing", "done", "total"}``) for the pages.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.jobs import store
    from finetune_studio.webui import brain_build_jobs

    def _progress(phase: str, done: int, total: int) -> None:
        ctx.publish({"phase": phase, "done": done, "total": total}, force=done == total)

    result = brain_build_jobs.run_in_job(ctx.params, release_gpu=lambda: store.release_gpu(ctx.id),
                                         progress=_progress)
    return {"status": "done", "build_id": ctx.params["build_id"], **result}
