"""Test runs and judge jobs: background tasks that persist their own progress.

A **test run** asks a model every question of a suite and saves the raw transcripts case by case; it never scores.
A **judge job** is the second, separate step: it reads the saved cases and asks a judge (any provider row) about
each one. Both write their state to the ``benchmark_runs`` row — status, progress, errors — so a page reload, a
second tab or the SSE stream all read the same truth, and a restart marks an interrupted job failed instead of
leaving it "running" forever.

Concurrency: the one GPU is shared with every other WebUI feature through ``ENGINE_LOCK``. A test run holds it for
its whole duration (nothing may swap the model between two questions); a local-GGUF judge holds it too. An API
judge uses no GPU and only excludes a second judge job on the same run. A second GPU job is refused (409), not
queued behind a many-minute run.

Auto-judge: when the saved setting is on (default off), a finished run starts its judge job by itself.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from finetune_studio import db
from finetune_studio.data.rag_portable.constants import DEFAULT_TOP_K
from finetune_studio.testing import run_store
from finetune_studio.testing.judge import (
    JudgeUnavailable,
    default_judge_provider_id,
    open_judge,
)
from finetune_studio.testing.judging import judge_benchmark
from finetune_studio.testing.rag_suite import (
    RagCaseResult,
    compute_retrieval_metrics,
    open_rag_query,
    run_rag_suite,
)
from finetune_studio.testing.suite import BenchmarkCase, CaseResult, run_suite
from finetune_studio.webui.engine_guard import ENGINE_LOCK

log = logging.getLogger(__name__)


class TestingBusy(RuntimeError):
    """A GPU job (test run or local-judge job) is already running."""

    __test__ = False  # not a pytest class

    def __init__(self, active: dict[str, Any]) -> None:
        if active.get("benchmark_id"):
            super().__init__(f"another {active['kind']} job is already running (run {active['benchmark_id']})")
        else:  # a feature outside testing holds the GPU
            super().__init__(f"the GPU is busy: {active['kind']}")
        self.active = active


@dataclass
class _Job:
    benchmark_id: str
    kind: str              # "run" | "judge"
    gpu: bool
    stop: threading.Event = field(default_factory=threading.Event)
    task: asyncio.Task | None = None


_ACTIVE: dict[tuple[str, str], _Job] = {}
_TASKS: set[asyncio.Task] = set()  # strong refs: a bare create_task result can be garbage-collected mid-run


def active_jobs() -> list[dict[str, Any]]:
    return [{"benchmark_id": j.benchmark_id, "kind": j.kind, "gpu": j.gpu, "stopping": j.stop.is_set()}
            for j in _ACTIVE.values()]


def is_active(benchmark_id: str, kind: str | None = None) -> bool:
    return any(j.benchmark_id == benchmark_id and (kind is None or j.kind == kind) for j in _ACTIVE.values())


def _foreign_gpu_user() -> str:
    """What outside the Testing page holds the GPU right now ('' = nothing): a test run or a local judge would
    unload the model that feature is using."""
    from finetune_studio.webui.app import training_engine

    if training_engine.state.status in ("training", "loading", "saving"):
        return "a training run is using it"
    if db.list_stale_data_prep_runs():  # queued/running prep runs hold the helper
        return "a data-prep run is using the helper model"
    return ""


def _assert_free(benchmark_id: str, kind: str, gpu: bool) -> None:
    """Raise :class:`TestingBusy` when a conflicting job is active (GPU jobs exclude each other, and a test
    never takes the card from training or data-prep)."""
    for job in _ACTIVE.values():
        if (gpu and job.gpu) or (job.benchmark_id == benchmark_id and job.kind == kind):
            raise TestingBusy({"kind": job.kind, "benchmark_id": job.benchmark_id})
    if gpu and (reason := _foreign_gpu_user()):
        raise TestingBusy({"kind": reason, "benchmark_id": ""})


def _claim(benchmark_id: str, kind: str, gpu: bool) -> _Job:
    """Register a job or raise :class:`TestingBusy`. No await between check and set, so it cannot race."""
    _assert_free(benchmark_id, kind, gpu)
    job = _Job(benchmark_id, kind, gpu)
    _ACTIVE[(benchmark_id, kind)] = job
    return job


def _spawn(coro: Any, job: _Job) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    job.task = task
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


def cancel(benchmark_id: str) -> bool:
    """Ask every active job on this run to stop at the next case boundary."""
    hit = False
    for job in _ACTIVE.values():
        if job.benchmark_id == benchmark_id:
            job.stop.set()
            hit = True
    return hit


# ── Test run ──────────────────────────────────────────────────────────────────


@dataclass
class RunSpec:
    """Everything a test run needs; ``model_path`` is already resolved by the route."""

    project_id: str
    kind: str                       # suite | rag | heldout | training_leakage
    suite_name: str
    cases: list[BenchmarkCase]
    model_path: str
    run_id: str = ""                # requested owner run ('' = resolve)
    max_tokens: int = 512
    temperature: float = 0.3
    rag: dict[str, Any] = field(default_factory=dict)   # top_k, corpus_path, max_context_chars
    config: dict[str, Any] = field(default_factory=dict)  # persisted next to the run (suite path, eval meta ...)
    extra_scores: dict[str, Any] = field(default_factory=dict)
    auto_judge: bool | None = None  # None = the saved setting
    judge_provider_id: str = ""     # '' = the saved default judge


def _load_model(model_path: str) -> None:
    from finetune_studio.models.manager import get_manager
    from finetune_studio.webui.app import inference_engine

    if inference_engine.model is not None and inference_engine.model_path == model_path:
        return
    get_manager().unload()  # free a resident helper before the model under test loads
    inference_engine.load(model_path)


def _execute_run(spec: RunSpec, bid: str, owner_run_id: str, job: _Job) -> dict[str, Any]:
    """Blocking body of a test run (worker thread, ENGINE_LOCK held): load, ask, save each case."""
    from finetune_studio.webui.app import inference_engine

    _load_model(spec.model_path)
    saved = 0
    rag_results: list[RagCaseResult] = []

    def progress() -> None:
        nonlocal saved
        saved += 1
        db.update_benchmark(bid, progress_done=saved, heartbeat_at=time.time())

    if spec.kind == "rag":
        query, corpus = open_rag_query(project_id=spec.project_id, corpus_path=str(spec.rag.get("corpus_path") or ""))
        max_chars = spec.rag.get("max_context_chars")
        if not max_chars:
            from finetune_studio.data.rag_portable.prompt import context_char_budget

            max_chars = context_char_budget(getattr(inference_engine, "n_ctx", None), max_new_tokens=spec.max_tokens)

        def on_rag(r: RagCaseResult) -> None:
            rag_results.append(r)
            run_store.save_case(bid, owner_run_id, r.case_result, judge_input=run_store.rag_judge_input(r))
            progress()

        run_rag_suite(
            inference_engine, query, spec.cases, top_k=int(spec.rag.get("top_k") or DEFAULT_TOP_K), max_tokens=spec.max_tokens,
            temperature=spec.temperature, max_context_chars=int(max_chars), on_result=on_rag,
            should_stop=job.stop.is_set,
        )
        return {"retrieval": compute_retrieval_metrics(rag_results), "corpus_path": corpus,
                "max_context_chars": int(max_chars)}

    def on_case(r: CaseResult) -> None:
        run_store.save_case(bid, owner_run_id, r)
        progress()

    run_suite(inference_engine, spec.cases, max_tokens=spec.max_tokens, temperature=spec.temperature,
              on_result=on_case, should_stop=job.stop.is_set)
    return {}


async def start_run_job(spec: RunSpec) -> dict[str, Any]:
    """Create the run row and start the job; returns the (running) benchmark row immediately."""
    _assert_free("", "run", gpu=True)   # refuse before a row exists
    owner = run_store.resolve_owner_run(spec.project_id, spec.model_path, requested_run_id=spec.run_id)
    config = {"kind": spec.kind, "max_tokens": spec.max_tokens, "temperature": spec.temperature,
              "rag": spec.rag, **spec.config}
    row = run_store.start_run(owner, spec.suite_name, model_path=spec.model_path, kind=spec.kind,
                              config=config, total=len(spec.cases))
    job = _claim(row["id"], "run", gpu=True)
    _spawn(_run_job(spec, row["id"], owner, job), job)
    return row


async def _run_job(spec: RunSpec, bid: str, owner: str, job: _Job) -> None:
    t0 = time.time()
    status, error, extra = "done", "", dict(spec.extra_scores)
    try:
        async with ENGINE_LOCK:
            extra.update(await asyncio.to_thread(_execute_run, spec, bid, owner, job))
        if job.stop.is_set():
            status = "cancelled"
    except Exception as exc:
        log.exception("test run %s failed", bid)
        status, error = "failed", str(exc)
    try:
        run_store.finish_run(bid, status=status, time_ms=int((time.time() - t0) * 1000), error=error,
                             extra_scores=extra)
    except Exception:
        log.exception("could not finish test run %s", bid)
    finally:
        _ACTIVE.pop((bid, "run"), None)
    if status == "done" and _auto_judge_enabled(spec.auto_judge):
        try:
            await start_judge_job(bid, spec.judge_provider_id, only_unjudged=True)
        except Exception as exc:  # noqa: BLE001 - e.g. busy; the run itself is complete and saved
            log.warning("auto-judge of %s not started: %s", bid, exc)
            db.update_benchmark(bid, judge_status="failed", judge_error=f"auto-judge not started: {exc}")


def _auto_judge_enabled(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    from finetune_studio.webui.routes.settings import get_test_settings

    return bool(get_test_settings()["auto_judge"])


# ── Judge job ─────────────────────────────────────────────────────────────────


def _provider_is_local(provider_id: str) -> bool:
    from finetune_studio.models.manager import get_manager

    row = get_manager().get_provider(provider_id)
    if row is None:
        raise JudgeUnavailable(f"unknown provider '{provider_id}'")
    return row.get("kind") == "local_gguf"


async def start_judge_job(bid: str, provider_id: str = "", *, only_unjudged: bool = True) -> dict[str, Any]:
    """Start judging a saved run with ``provider_id`` ('' = the configured default judge)."""
    bench = db.get_benchmark(bid)
    if bench is None:
        raise LookupError("test run not found")
    if bench.get("scoring") == "exact":
        raise ValueError("this is an official public benchmark: it keeps its standard exact-match scoring")
    if bench.get("status") in ("queued", "running"):
        raise ValueError("the test run is still running — judge it when it has finished")
    provider = provider_id or default_judge_provider_id()
    local = _provider_is_local(provider)
    total = len(db.list_cases(bid))
    if not total:
        raise ValueError("this run has no saved cases to judge")
    job = _claim(bid, "judge", gpu=local)
    db.update_benchmark(bid, judge_status="running", judge_provider_id=provider, judge_model="", judge_done=0,
                        judge_total=total, judge_error="")
    _spawn(_judge_job(bid, provider, only_unjudged, local, job), job)
    return db.get_benchmark(bid) or {}


def _judge_blocking(bid: str, provider: str, only_unjudged: bool, job: _Job) -> str:
    with open_judge(provider) as judge:
        db.update_benchmark(bid, judge_model=judge.model)

        def on_progress(done: int, total: int) -> None:
            db.update_benchmark(bid, judge_done=done, judge_total=total, heartbeat_at=time.time())

        summary = judge_benchmark(bid, judge, only_unjudged=only_unjudged, should_stop=job.stop.is_set,
                                  on_progress=on_progress)
    if summary.stopped:
        return "cancelled"
    if summary.failed and not summary.judged:
        raise JudgeUnavailable(f"the judge returned no usable verdict for any of {summary.failed} case(s) — see each case's judge error")
    return "done"


async def _judge_job(bid: str, provider: str, only_unjudged: bool, local: bool, job: _Job) -> None:
    status, error = "done", ""
    lock: AbstractAsyncContextManager[Any] = ENGINE_LOCK if local else nullcontext()
    try:
        async with lock:
            status = await asyncio.to_thread(_judge_blocking, bid, provider, only_unjudged, job)
    except Exception as exc:  # noqa: BLE001
        log.warning("judge job %s failed: %s", bid, exc)
        status, error = "failed", str(exc)
    finally:
        _ACTIVE.pop((bid, "judge"), None)
    try:
        from finetune_studio.testing.scoring import rescore_benchmark

        rescore_benchmark(bid)
    except Exception:
        log.exception("could not rescore %s", bid)
    db.update_benchmark(bid, judge_status=status, judge_error=error, heartbeat_at=time.time())
