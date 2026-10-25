"""The style A/B job: a judge provider compares the base and the trained answers of a use-case evaluation, blind and in
both orders (``usecases.style_ab``).

Params: ``pid``, ``uid``, ``group_id``, ``provider_id``, ``style_guide``, and the testing registry keys (``role`` =
"judge", ``bids`` = the base and trained rows, ``uses_gpu`` = a local GGUF judge, which also takes the GPU lease).
The lifecycle is ``usecases.evaluate.ab_in_job``; the result is written to ``usecases/<id>/ab-<group>.json``.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.jobs.runner import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    from finetune_studio.usecases import evaluate

    return evaluate.ab_in_job(ctx)
