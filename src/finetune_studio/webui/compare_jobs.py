"""Compare jobs: ask several models the same questions (one test run per model), then judge the saved answers.

Reuses the Testing machinery instead of copying it: each model gets an ordinary test-run row (kind ``compare``)
filled by ``testing_jobs._execute_run`` (the same ``run_suite`` + ``run_store`` path), and judging is an ordinary
``testing_jobs`` judge job over the group's rows. Differences from a single run:

* one job process (``jobs.kinds.compare``) walks the models one after another; it owns every row of the group, so
  no other card-using testing job starts until the last model has answered (a web restart does not end it);
* the job owns all rows, so the Testing page's cancel on any row stops the whole group;
* a group judge loads the judge ONCE (a local GGUF judge replaces the tested model only after all runs are saved)
  and judges the rows in turn.

No verdict is written by a run, and nothing here compares strings.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_studio import db
from finetune_studio.compare.session import COMPARE_KIND, group_rows
from finetune_studio.db.connection import new_id
from finetune_studio.testing import run_store
from finetune_studio.testing.judge import default_judge_provider_id
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


def spec_from_params(raw: dict[str, Any]) -> CompareSpec:
    return CompareSpec(**{**raw, "cases": [BenchmarkCase(**c) for c in raw.get("cases") or []],
                          "models": [CompareModel(**m) for m in raw.get("models") or []]})


# ── Run ───────────────────────────────────────────────────────────────────────


async def start_compare_job(spec: CompareSpec) -> list[dict[str, Any]]:
    """Create one queued test run per model and launch the job that answers them in order. Returns the rows."""
    return launch_compare(spec)


def launch_compare(spec: CompareSpec) -> list[dict[str, Any]]:
    """:func:`start_compare_job` without the event loop (no await between the busy check and the launch)."""
    testing_jobs._assert_free([], "run", uses_gpu=True)  # refuse before any row exists
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
    auto = testing_jobs._auto_judge_enabled(spec.auto_judge)
    try:
        testing_jobs._launch(
            testing_jobs.KIND_COMPARE, role="run", bids=[r["id"] for r in rows], uses_gpu=True, lease=False,
            project_id=spec.project_id, title=f"compare {len(rows)} models on {spec.suite_name}",
            params={"spec": asdict(spec), "group_id": group_id, "auto_judge": auto,
                    "judge_provider_id": spec.judge_provider_id or (default_judge_provider_id() if auto else ""),
                    "rows": [{"id": r["id"], "owner_run_id": r["owner_run_id"]} for r in rows]},
        )
    except testing_jobs.TestingBusy as exc:
        for r in rows:
            run_store.finish_run(r["id"], status="failed", time_ms=0, error=f"not started: {exc}")
        raise
    return [db.get_benchmark(r["id"]) or r for r in rows]


def _run_one(spec: CompareSpec, model: CompareModel, bid: str, owner: str, stop: threading.Event) -> None:
    """Blocking: load ``model`` and save every answer of the suite on run ``bid`` (job process)."""
    testing_jobs._execute_run(
        testing_jobs.RunSpec(project_id=spec.project_id, kind=COMPARE_KIND, suite_name=spec.suite_name,
                             cases=spec.cases, model_path=model.model_path, max_tokens=spec.max_tokens,
                             temperature=spec.temperature),
        bid, owner, stop,
    )


def run_in_job(ctx: Any) -> dict[str, Any]:
    """The job process side of a comparison (``jobs.kinds.compare``): each model in turn, then the auto-judge."""
    p = ctx.params
    spec = spec_from_params(p["spec"])
    group_id: str = p["group_id"]
    stop = ctx.stop
    statuses = []
    for model, row in zip(spec.models, p["rows"], strict=True):
        bid = row["id"]
        t0 = time.time()
        status, error = "done", ""
        if stop.is_set():
            status = "cancelled"
        else:
            db.update_benchmark(bid, status="running", heartbeat_at=t0)
            try:
                _run_one(spec, model, bid, row["owner_run_id"], stop)
                if stop.is_set():
                    status = "cancelled"
            except Exception as exc:
                log.exception("compare %s: model %s failed", group_id, model.label)
                status, error = "failed", str(exc)
        try:
            run_store.finish_run(bid, status=status, time_ms=int((time.time() - t0) * 1000), error=error)
        except Exception:
            log.exception("could not finish compare run %s", bid)
        statuses.append(status)
    all_done = all(s == "done" for s in statuses)
    if all_done and p.get("auto_judge"):
        try:
            launch_group_judge(spec.project_id, group_id, str(p.get("judge_provider_id") or ""), exclude_job=ctx.id)
        except Exception as exc:  # noqa: BLE001 - e.g. busy; the answers are saved, judge them from the page
            log.warning("auto-judge of compare group %s not started: %s", group_id, exc)
            for row in p["rows"]:
                db.update_benchmark(row["id"], judge_status="failed", judge_error=f"auto-judge not started: {exc}")
    overall = "done" if all_done else ("cancelled" if "cancelled" in statuses else "failed")
    return {"status": testing_jobs._job_status(overall), "group_id": group_id,
            "row_status": dict(zip([r["id"] for r in p["rows"]], statuses, strict=True))}


# ── Judge ─────────────────────────────────────────────────────────────────────


async def start_group_judge(
    project_id: str, group_id: str, provider_id: str = "", *, only_unjudged: bool = True,
) -> list[dict[str, Any]]:
    """Judge every run of a group with one judge provider ('' = the configured default). Returns the rows.

    Raises ``LookupError`` (no such group), ``ValueError`` (still answering / nothing to judge),
    ``JudgeUnavailable`` (unknown provider) or ``TestingBusy`` (a GPU job holds the card, or a judge already runs).
    """
    return launch_group_judge(project_id, group_id, provider_id, only_unjudged=only_unjudged)


def launch_group_judge(project_id: str, group_id: str, provider_id: str = "", *, only_unjudged: bool = True,
                       exclude_job: str = "") -> list[dict[str, Any]]:
    """:func:`start_group_judge` without the event loop; ``exclude_job`` = the compare job chaining it."""
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
    bids = [r["id"] for r in rows]
    testing_jobs._assert_free(bids, "judge", uses_gpu=local, exclude=exclude_job)
    for r in rows:
        db.update_benchmark(r["id"], judge_status="running", judge_provider_id=provider, judge_model="",
                            judge_done=0, judge_total=len(db.list_cases(r["id"])), judge_error="")
    testing_jobs.launch_judge_rows(bids, provider, only_unjudged=only_unjudged, local=local,
                                   title=f"judge comparison {group_id}", project_id=project_id)
    return [db.get_benchmark(r["id"]) or r for r in rows]
