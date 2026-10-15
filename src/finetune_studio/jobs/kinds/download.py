"""The download job: one ``hf_downloads`` row's Hugging Face Hub download, in its own process (no GPU).

Params: ``download_id``, ``repo_id``, ``filename`` ("" = the whole repo), ``revision``, ``dest`` (absolute folder).
The lifecycle (row states, progress, cancel) is ``webui.download_jobs.run_in_job``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.webui import download_jobs

    outcome = download_jobs.run_in_job(ctx.params, ctx.stop)
    status = {"done": "done", "cancelled": "stopped"}.get(outcome, "failed")
    return {"status": status, "download_id": ctx.params["download_id"], "outcome": outcome}
