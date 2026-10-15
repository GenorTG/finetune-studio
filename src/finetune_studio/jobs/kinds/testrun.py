"""The test-run job: ask a model every question of a suite and save each answer on its ``benchmark_runs`` row.

Params: ``spec`` (``webui.testing_jobs.RunSpec`` as a dict, cases included), ``bid`` (the row), ``owner_run_id``,
``auto_judge`` + ``judge_provider_id`` (resolved when the run started), and the registry keys every testing job has
(``role``, ``bids``, ``uses_gpu``). No GPU lease: the model is the inference worker's (``inference.get_engine``).
The lifecycle (row states, cancel, auto-judge) is ``webui.testing_jobs.run_in_job``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.webui import testing_jobs

    return testing_jobs.run_in_job(ctx)
