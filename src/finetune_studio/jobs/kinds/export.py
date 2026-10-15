"""The export job: one ``model_exports`` row's merge / GGUF convert / quantize, in its own process.

Params: ``export_id``, ``factory`` (a ``webui.export_work`` factory name) and its ``kwargs`` (JSON-safe: the run
row, paths, quants). The lifecycle (row states, cancel, children) is ``webui.export_jobs.run_in_job``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.jobs import store
    from finetune_studio.webui import export_jobs, export_work

    p = ctx.params
    if p["factory"] not in export_jobs.FACTORIES:
        raise ValueError(f"unknown export work {p['factory']!r}")
    work = getattr(export_work, p["factory"])(p["export_id"], **p["kwargs"])
    outcome = export_jobs.run_in_job(p["export_id"], work, ctx.stop, release_gpu=lambda: store.release_gpu(ctx.id))
    status = {"done": "done", "cancelled": "stopped"}.get(outcome, "failed")
    return {"status": status, "export_id": p["export_id"], "outcome": outcome}
