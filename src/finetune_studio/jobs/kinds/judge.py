"""The judge job: a judge provider gives a verdict on each saved case of one or more ``benchmark_runs`` rows
(one test run, or every row of a compare group, judged with ONE judge load).

Params: ``provider_id``, ``only_unjudged``, ``bids``, ``role`` ("judge"), ``uses_gpu`` (a local GGUF judge, which
also takes the GPU lease: its model is loaded into the inference worker and training/exports must not start and
unload it mid-judge). The lifecycle is ``webui.testing_jobs.judge_in_job``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.webui import testing_jobs

    return testing_jobs.judge_in_job(ctx)
