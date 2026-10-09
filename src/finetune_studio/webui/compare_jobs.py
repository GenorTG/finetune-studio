"""Compare jobs: ask several models the same questions (one test run per model), then judge the saved answers.

Reuses the Testing machinery instead of copying it: each model gets an ordinary test-run row (kind ``compare``)
filled by ``testing_jobs._execute_run`` (the same ``run_suite`` + ``run_store`` path), and judging goes through
``testing.judging.judge_benchmark`` with the same judge providers. Differences from a single run:

* one background job walks the models one after another and holds ``ENGINE_LOCK`` for the whole group, so nothing
  can swap the resident model between two models' questions;
* every row of a group shares one stop event, so the Testing page's cancel on any row stops the whole group;
* a group judge loads the judge ONCE (a local GGUF judge replaces the tested model only after all runs are saved)
  and judges the rows in turn.

No verdict is written by a run, and nothing here compares strings.
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
from finetune_studio.compare.session import COMPARE_KIND, group_rows
from finetune_studio.db.connection import new_id
from finetune_studio.testing import run_store
from finetune_studio.testing.judge import JudgeUnavailable, default_judge_provider_id
from finetune_studio.testing.judging import judge_benchmark
from finetune_studio.testing.suite import BenchmarkCase
from finetune_studio.webui import testing_jobs

log = logging.getLogger(__name__)


@dataclass
class CompareModel:
    label: str
    model_path: str


@dataclass
class CompareSpec:
    """A compare request with every model already resolved to a path."""

    project_id: str
    suite_name: str
    cases: list[BenchmarkCase]
    models: list[CompareModel]
    max_tokens: int = 512
    temperature: float = 0.3
    config: dict[str, Any] = field(default_factory=dict)   # persisted on every row (suite path ...)
    auto_judge: bool | None = None  # None = the saved setting
    judge_provider_id: str = ""     # '' = the saved default judge


def _register(rows: list[dict[str, Any]], kind: str, *, gpu: bool, stop: threading.Event) -> list[testing_jobs._Job]:
    """Put one job per row into the shared registry. All share ``stop``; callers checked ``_assert_free`` first."""
    jobs = [testing_jobs._Job(r["id"], kind, gpu, stop=stop) for r in rows]
    for job in jobs:
        testing_jobs._ACTIVE[(job.benchmark_id, kind)] = job
    return jobs


# ── Run ───────────────────────────────────────────────────────────────────────


async def start_compare_job(spec: CompareSpec) -> list[dict[str, Any]]:
    """Create one queued test run per model and start the job that answers them in order. Returns the rows."""
    testing_jobs._assert_free("", "run", gpu=True)  # refuse before any row exists
    group_id = new_id()
    labels = [m.label for m in spec.models]
    rows = []
    for i, model in enumerate(spec.models):
        owner = run_store.resolve_owner_run(spec.project_id, model.model_path)
        config = {
            "kind": COMPARE_KIND, "max_tokens": spec.max_tokens, "temperature": spec.temperature, **spec.config,
            "compare": {"group_id": group_id, "index": i, "label": model.label, "models": labels},
        }
        row = run_store.start_run(owner, spec.suite_name, model_path=model.model_path, kind=COMPARE_KIND,
                                  config=config, total=len(spec.cases))
        db.update_benchmark(row["id"], status="queued")
        rows.append({**row, "status": "queued", "owner_run_id": owner})
    stop = threading.Event()
    jobs = _register(rows, "run", gpu=True, stop=stop)
    testing_jobs._spawn(_compare_job(spec, group_id, rows, jobs), jobs[0])
    return [db.get_benchmark(r["id"]) or r for r in rows]


def _run_one(spec: CompareSpec, model: CompareModel, bid: str, owner: str, job: testing_jobs._Job) -> None:
    """Blocking: load ``model`` and save every answer of the suite on run ``bid`` (worker thread, lock held)."""
    testing_jobs._execute_run(
        testing_jobs.RunSpec(project_id=spec.project_id, kind=COMPARE_KIND, suite_name=spec.suite_name,
                             cases=spec.cases, model_path=model.model_path, max_tokens=spec.max_tokens,
                             temperature=spec.temperature),
        bid, owner, job,
    )


async def _compare_job(spec: CompareSpec, group_id: str, rows: list[dict[str, Any]], jobs: list[testing_jobs._Job]) -> None:
    stop = jobs[0].stop
    all_done = True
    async with testing_jobs.ENGINE_LOCK:
        for model, row, job in zip(spec.models, rows, jobs, strict=True):
            bid = row["id"]
            t0 = time.time()
            status, error = "done", ""
            if stop.is_set():
                status = "cancelled"
            else:
                db.update_benchmark(bid, status="running", heartbeat_at=t0)
                try:
                    await asyncio.to_thread(_run_one, spec, model, bid, row["owner_run_id"], job)
                    if stop.is_set():
                        status = "cancelled"
                except Exception as exc:
                    log.exception("compare %s: model %s failed", group_id, model.label)
                    status, error = "failed", str(exc)
            try:
                run_store.finish_run(bid, status=status, time_ms=int((time.time() - t0) * 1000), error=error)
            except Exception:
                log.exception("could not finish compare run %s", bid)
            finally:
                testing_jobs._ACTIVE.pop((bid, "run"), None)
            all_done = all_done and status == "done"
    if all_done and testing_jobs._auto_judge_enabled(spec.auto_judge):
        try:
            await start_group_judge(spec.project_id, group_id, spec.judge_provider_id)
        except Exception as exc:  # noqa: BLE001 - e.g. busy; the answers are saved, judge them from the page
            log.warning("auto-judge of compare group %s not started: %s", group_id, exc)
            for row in rows:
                db.update_benchmark(row["id"], judge_status="failed", judge_error=f"auto-judge not started: {exc}")


# ── Judge ─────────────────────────────────────────────────────────────────────


async def start_group_judge(
    project_id: str, group_id: str, provider_id: str = "", *, only_unjudged: bool = True,
) -> list[dict[str, Any]]:
    """Judge every run of a group with one judge provider ('' = the configured default). Returns the rows.

    Raises ``LookupError`` (no such group), ``ValueError`` (still answering / nothing to judge),
    ``JudgeUnavailable`` (unknown provider) or ``TestingBusy`` (a GPU job holds the card, or a judge already runs).
    """
    rows = group_rows(project_id, group_id)
    if not rows:
        raise LookupError("compare group not found")
    if any(r.get("status") in ("queued", "running") for r in rows):
        raise ValueError("the models are still answering; judge when the run has finished")
    rows = [r for r in rows if db.list_cases(r["id"])]
    if not rows:
        raise ValueError("this comparison has no saved answers to judge")
    provider = provider_id or default_judge_provider_id()
    local = testing_jobs._provider_is_local(provider)
    testing_jobs._assert_free("", "judge", gpu=local)
    for r in rows:
        if testing_jobs.is_active(r["id"], "judge"):
            raise testing_jobs.TestingBusy({"kind": "judge", "benchmark_id": r["id"]})
    for r in rows:
        db.update_benchmark(r["id"], judge_status="running", judge_provider_id=provider, judge_model="",
                            judge_done=0, judge_total=len(db.list_cases(r["id"])), judge_error="")
    jobs = _register(rows, "judge", gpu=local, stop=threading.Event())
    testing_jobs._spawn(_group_judge_job(rows, provider, only_unjudged, local, jobs), jobs[0])
    return [db.get_benchmark(r["id"]) or r for r in rows]


def _judge_rows_blocking(rows: list[dict[str, Any]], provider: str, only_unjudged: bool, stop: threading.Event) -> dict[str, tuple[str, str]]:
    """Open the judge once and judge each row in turn. Returns ``{bid: (judge_status, error)}``."""
    out: dict[str, tuple[str, str]] = {}
    with testing_jobs.open_judge(provider) as judge:
        for row in rows:
            bid = row["id"]
            if stop.is_set():
                out[bid] = ("cancelled", "")
                continue
            db.update_benchmark(bid, judge_model=judge.model)

            def on_progress(done: int, total: int, bid: str = bid) -> None:
                db.update_benchmark(bid, judge_done=done, judge_total=total, heartbeat_at=time.time())

            summary = judge_benchmark(bid, judge, only_unjudged=only_unjudged, should_stop=stop.is_set,
                                      on_progress=on_progress)
            if summary.stopped:
                out[bid] = ("cancelled", "")
            elif summary.failed and not summary.judged:
                out[bid] = (
                    "failed",
                    f"the judge returned no usable verdict for any of {summary.failed} case(s) — see each case's judge error",
                )
            else:
                out[bid] = ("done", "")
    return out


async def _group_judge_job(rows: list[dict[str, Any]], provider: str, only_unjudged: bool, local: bool,
                           jobs: list[testing_jobs._Job]) -> None:
    results: dict[str, tuple[str, str]] = {}
    error = ""
    lock: AbstractAsyncContextManager[Any] = testing_jobs.ENGINE_LOCK if local else nullcontext()
    try:
        async with lock:
            results = await asyncio.to_thread(_judge_rows_blocking, rows, provider, only_unjudged, jobs[0].stop)
    except Exception as exc:  # noqa: BLE001 - JudgeUnavailable and provider errors alike; written to every row
        log.warning("compare judge failed: %s", exc)
        error = str(exc) if isinstance(exc, JudgeUnavailable) else f"{type(exc).__name__}: {exc}"
    finally:
        for job in jobs:
            testing_jobs._ACTIVE.pop((job.benchmark_id, "judge"), None)
    from finetune_studio.testing.scoring import rescore_benchmark

    for row in rows:
        bid = row["id"]
        status, row_error = results.get(bid, ("failed", error or "the judge job ended without a result"))
        try:
            rescore_benchmark(bid)
        except Exception:
            log.exception("could not rescore %s", bid)
        db.update_benchmark(bid, judge_status=status, judge_error=row_error, heartbeat_at=time.time())
