"""Judge the saved cases of a test run (the second, separate step after the raw run).

Reads the stored question / answer key / model answer of every case, asks the judge about each, and appends the
result to `case_judgements`. Safe to repeat with another judge — earlier opinions (and any human verdict, which
always wins) are kept.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from finetune_studio import db
from finetune_studio.db import judgements as jdb
from finetune_studio.testing.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeCase,
    JudgeResult,
    LoadedJudge,
    judge_many,
)
from finetune_studio.testing.scoring import rescore_benchmark

log = logging.getLogger(__name__)

# A case already decided by one of these is not re-judged by "judge only what is not judged yet".
_DECIDED_BY = frozenset({"human", "ai", "exact"})


@dataclass
class JudgeSummary:
    judged: int = 0           # judge returned a verdict
    failed: int = 0           # judge call failed / unreadable; the case stays as it was
    skipped_run_error: int = 0  # the model run itself failed — there is no answer to judge
    skipped_decided: int = 0  # already judged (only_unjudged)
    stopped: bool = False


def judge_benchmark(
    bid: str,
    judge: LoadedJudge,
    *,
    only_unjudged: bool = True,
    should_stop: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int], None] | None = None,
) -> JudgeSummary:
    """Judge a run's cases with ``judge`` and refresh its scores. Blocking."""
    summary = JudgeSummary()
    todo: list[tuple[str, JudgeCase]] = []
    for case in db.list_cases(bid):
        if case.get("error") and not (case.get("model_answer") or "").strip():
            summary.skipped_run_error += 1
            continue
        if only_unjudged and case.get("judge") in _DECIDED_BY and case.get("verdict"):
            summary.skipped_decided += 1
            continue
        todo.append((case["id"], JudgeCase(
            question=case.get("question") or "",
            correct_answer=case.get("correct_answer") or "",
            model_answer=case.get("model_answer") or "",
            keywords=list(case.get("keywords") or []),
            expect_abstain=bool(case.get("expect_abstain")),
        )))

    total = len(todo)
    done = 0
    last_push = 0.0

    def record(case_id: str, result: JudgeResult) -> None:
        nonlocal done, last_push
        jdb.add_judgement(
            case_id, kind="ai", verdict=result.verdict, reasoning=result.reasoning,
            confidence=result.confidence, provider_id=judge.provider_id, judge_model=judge.model,
            error=result.error, prompt_version=JUDGE_PROMPT_VERSION,
        )
        if result.verdict:
            summary.judged += 1
        else:
            summary.failed += 1
        done += 1
        now = time.time()
        if on_progress is not None and (now - last_push > 0.5 or done == total):
            last_push = now
            on_progress(done, total)

    if on_progress is not None:
        on_progress(0, total)
    attempted = judge_many(judge, todo, record, should_stop=should_stop)
    summary.stopped = attempted < total
    rescore_benchmark(bid)
    return summary
