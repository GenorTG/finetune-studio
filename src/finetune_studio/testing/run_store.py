"""Persisting a test run: raw transcripts are saved case by case, judging comes later.

A test run is a ``benchmark_runs`` row (status ``running`` while the model answers) whose cases are appended as
they finish, so a crash or a stop keeps everything answered so far and the Testing page can show live progress.
No verdict is written here — except by :func:`save_exact_case`, used only by the official public benchmarks.
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio import db
from finetune_studio.testing.rag_suite import RagCaseResult
from finetune_studio.testing.scoring import rescore_benchmark
from finetune_studio.testing.suite import CaseResult


def resolve_owner_run(project_id: str, model_path: str, *, requested_run_id: str = "") -> str:
    """The training run a test run hangs off.

    A supplied run is used only when it belongs to the project; else the finished run whose output directory
    contains ``model_path``; else the project's hidden ``__evaluation__`` placeholder (created once).
    """
    if not project_id:
        # Every benchmark hangs off a run and every run off a project: fail closed without one.
        raise ValueError("project_id is required to save a test run")
    if requested_run_id:
        requested = db.get_run(requested_run_id)
        if requested and requested.get("project_id") == project_id:
            return requested_run_id
    if model_path:
        for run in db.list_runs(project_id):
            output_path = str(run.get("output_path") or "")
            if run.get("status") == "done" and output_path and model_path.startswith(output_path):
                return str(run["id"])
    from finetune_studio.db.runs import EVAL_RUN_NAME

    for run in db.list_runs(project_id, include_base_probe=True):
        if run.get("name") == EVAL_RUN_NAME:
            return str(run["id"])
    placeholder = db.create_run(project_id=project_id, name=EVAL_RUN_NAME)
    db.update_run(placeholder["id"], status="done", notes="Owns test runs of models that are not a training run's output")
    return str(placeholder["id"])


def start_run(
    run_id: str, suite_name: str, *, model_path: str, kind: str, config: dict[str, Any], total: int,
    scoring: str = "judge",
) -> dict[str, Any]:
    """Create the (still empty) test-run row in status ``running``."""
    return db.create_benchmark(
        run_id, suite_name, {"model_path": model_path}, 0, model_path=model_path, status="running", kind=kind,
        scoring=scoring, config=config, progress_total=total,
    )


def is_grounded(result: CaseResult) -> bool:
    """True when the case was asked WITH retrieved context in a system turn (a grounded training row).

    Such a row teaches "answer from the context", plain rows are asked bare: two different skills, reported apart.
    """
    return bool(result.transcript and (result.transcript[0] or {}).get("role") == "system")


def save_case(bid: str, run_id: str, result: CaseResult, *, judge_input: dict[str, Any] | None = None) -> str:
    """Append one answered case — transcript, answer key, no verdict."""
    if judge_input is None and is_grounded(result):  # a RAG run passes its own retrieval trace; every RAG case is grounded
        judge_input = {"grounded": True}
    return db.create_case(
        bid, run_id, result.case_name, result.category, result.question, result.correct_answer,
        result.model_answer, result.transcript, error=result.error, judge_input=judge_input,
        source_id=result.source_id, chunk_idx=result.chunk_idx, keywords=result.keywords,
        expect_abstain=result.expect_abstain, time_ms=result.time_ms,
    )


def rag_judge_input(rag: RagCaseResult) -> dict[str, Any]:
    """Retrieval provenance stored next to a RAG case so a reviewer can see what the model was given."""
    return {
        "retrieval_hit": rag.retrieval_hit,
        "retrieval_hits": rag.retrieval_hits,
        "context_text": rag.context_text,
        "gold_in_retrieved": rag.gold_in_retrieved,
        "gold_in_context": rag.gold_in_context,
        "chunks_retrieved": rag.chunks_retrieved,
        "chunks_in_context": rag.chunks_in_context,
    }


def save_exact_case(bid: str, run_id: str, result: CaseResult, *, judge_input: dict[str, Any] | None = None) -> str:
    """Append a case of an official public benchmark with its exact-match verdict (the published scoring method)."""
    return db.create_case(
        bid, run_id, result.case_name, result.category, result.question, result.correct_answer,
        result.model_answer, result.transcript, judge="exact", judge_model="exact", verdict=result.verdict,
        judge_reasoning=result.judge_reasoning, scored_at=time.time() if result.verdict else None,
        scoring_method=result.scoring_method, validity=result.validity, error=result.error,
        judge_input=judge_input, source_id=result.source_id, chunk_idx=result.chunk_idx,
        keywords=result.keywords, expect_abstain=result.expect_abstain, record_judgement=bool(result.verdict),
        time_ms=result.time_ms,
    )


def finish_run(
    bid: str, *, status: str, time_ms: int, error: str = "", extra_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close the run: final status, duration and scores recomputed from the saved cases."""
    scores = rescore_benchmark(bid)
    if extra_scores:
        scores = {**scores, **extra_scores}
        db.update_benchmark_scores(bid, scores)
    done = len(db.list_cases(bid))
    db.update_benchmark(bid, status=status, error=error, time_ms=time_ms, progress_done=done, heartbeat_at=time.time())
    return db.get_benchmark(bid) or {}
