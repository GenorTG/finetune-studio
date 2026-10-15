"""The compare job: several models answer the same suite, one ``benchmark_runs`` row (kind ``compare``) each, in
order; then the optional auto-judge job for the group.

Params: ``spec`` (``webui.compare_jobs.CompareSpec`` as a dict), ``group_id``, ``rows`` ([{id, owner_run_id}] in
model order), ``auto_judge`` + ``judge_provider_id``, and the registry keys (``role``, ``bids``, ``uses_gpu``).
No GPU lease (the models load into the inference worker). The lifecycle is ``webui.compare_jobs.run_in_job``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.webui import compare_jobs

    return compare_jobs.run_in_job(ctx)
