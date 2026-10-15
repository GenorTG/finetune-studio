"""Test runs and judge jobs: supervised job processes that persist their own progress.

A **test run** asks a model every question of a suite and saves the raw transcripts case by case; it never scores.
A **judge job** is the second, separate step: it reads the saved cases and asks a judge (any provider row) about
each one. Both run as their own job process (``jobs.kinds.testrun`` / ``jobs.kinds.judge``; a comparison is
``jobs.kinds.compare``, see ``compare_jobs``), so a web restart (an update, "Apply now" in Settings,
``fts restart web``) no longer ends them. They write their state to the ``benchmark_runs`` row — status, progress,
errors — so a page reload, a second tab, the SSE stream and a restarted web app all read the same truth; the startup
reconcile leaves rows whose job is still alive (:func:`running_benchmark_ids`) and marks the rest failed.

The job files (``jobs.store``) are the registry: every testing job's params carry ``role`` (run | judge), ``bids``
(the rows it owns) and ``uses_gpu``. Nothing crosses the process boundary but JSON-safe params; the job reaches the
loaded model through the inference worker (:func:`_engine`).

Concurrency: a test run uses the inference worker's model, so it does NOT take the GPU lease (a lease makes the
supervisor unload that model); it still excludes every other card-using testing job (a second run, a compare, a
local judge) and never starts while another GPU job (training, an export) or a mining job holds the card. A
local-GGUF judge loads its model into the inference worker and DOES take the lease, so training or an export cannot
start mid-judge and unload it. An API judge uses no GPU and only excludes a second judge job on the same run. A
conflicting job is refused (409), not queued behind a many-minute run.

Auto-judge: when the saved setting is on (default off), a finished run starts its judge job by itself (resolved
when the run starts; the run's job launches the judge job).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.smart_brain.constants import DEFAULT_TOP_K
from finetune_studio.testing import run_store
from finetune_studio.testing.brain_suite import (
    BrainCaseResult,
    compute_retrieval_metrics,
    open_brain_query,
    run_brain_suite,
)
from finetune_studio.testing.judge import (
    JudgeUnavailable,
    default_judge_provider_id,
    open_judge,
)
from finetune_studio.testing.judging import judge_benchmark
from finetune_studio.testing.suite import BenchmarkCase, CaseResult, run_suite

log = logging.getLogger(__name__)

KIND_RUN, KIND_JUDGE, KIND_COMPARE = "testrun", "judge", "compare"
_KINDS = (KIND_RUN, KIND_JUDGE, KIND_COMPARE)
_PREFIXES = tuple(f"{k}-" for k in _KINDS)
STOP_GRACE_S = 60.0                 # a question or a judge call in flight finishes, then the job stops
_STOP_MARK = "stop_requested"       # in the job folder: the page shows "stopping"


class TestingBusy(RuntimeError):
    """A conflicting job (test run, compare, local-judge job, or another GPU job) is already running."""

    __test__ = False  # not a pytest class

    def __init__(self, active: dict[str, Any], message: str | None = None) -> None:
        if message:
            super().__init__(message)
        elif active.get("benchmark_id"):
            super().__init__(f"another {active['kind']} job is already running (run {active['benchmark_id']})")
        else:  # a feature outside testing holds the GPU
            super().__init__(f"the GPU is busy: {active['kind']}")
        self.active = active


# ── registry: the job files ───────────────────────────────────────────────────


def _live_specs() -> list[store.JobSpec]:
    """Specs of the testing jobs (run, judge, compare) whose process is alive. Finished jobs are skipped by their
    exit record before anything else is read, so this stays cheap with a week of old job folders."""
    d = store.jobs_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir(), key=lambda p: p.name):
        if not p.name.startswith(_PREFIXES) or (p / "exit.json").exists():
            continue
        try:
            if launch.describe(p.name)["running"]:
                out.append(store.read_spec(p.name))
        except (FileNotFoundError, ValueError, TypeError):
            continue
    return out


def active_jobs() -> list[dict[str, Any]]:
    """One entry per (row, job): ``benchmark_id``, ``kind`` (run | judge), ``gpu``, ``stopping``, ``job_id``."""
    out = []
    for spec in _live_specs():
        p = spec.params
        stopping = (store.job_dir(spec.id) / _STOP_MARK).exists()
        for bid in p.get("bids") or []:
            out.append({"benchmark_id": bid, "kind": p.get("role", "run"), "gpu": bool(p.get("uses_gpu")),
                        "stopping": stopping, "job_id": spec.id})
    return out


def is_active(benchmark_id: str, kind: str | None = None) -> bool:
    return any(j["benchmark_id"] == benchmark_id and (kind is None or j["kind"] == kind) for j in active_jobs())


def running_benchmark_ids() -> set[str]:
    """For the startup reconcile: rows whose job outlived the web restart are left alone."""
    return {j["benchmark_id"] for j in active_jobs()}


def _foreign_gpu_user() -> str:
    """What outside the Testing page holds the GPU right now ('' = nothing): a test run or a local judge would
    unload the model that feature is using. Training and exports are job processes (``launch.list_jobs``)."""
    for j in launch.list_jobs(running_only=True):
        if j["kind"] in _KINDS or not j.get("gpu") or store.gpu_released(j["id"]):
            continue
        if j["kind"] == "training":
            return "a training run is using it"
        return f"a {j['kind']} job is using it ({j.get('title') or j['id']})"
    if db.mining.list_jobs(active=True, limit=1):  # mining jobs hold the helper
        return "a mining job is using the helper model"
    return ""


def _assert_free(bids: list[str], role: str, uses_gpu: bool, *, exclude: str = "") -> None:
    """Raise :class:`TestingBusy` when a conflicting job is alive: card-using testing jobs exclude each other, the
    same row is never run or judged twice at once, and a test never takes the card from training or data-prep.
    ``exclude`` is the calling job (a run that chains its own judge job)."""
    for spec in _live_specs():
        if spec.id == exclude:
            continue
        p = spec.params
        theirs = list(p.get("bids") or [])
        same_row = p.get("role") == role and bool(set(bids) & set(theirs))
        if (uses_gpu and p.get("uses_gpu")) or same_row:
            bid = next((b for b in theirs if b in bids), theirs[0] if theirs else "")
            raise TestingBusy({"kind": p.get("role", "run"), "benchmark_id": bid})
    if uses_gpu and (reason := _foreign_gpu_user()):
        raise TestingBusy({"kind": reason, "benchmark_id": ""})


def _launch(kind: str, *, role: str, bids: list[str], uses_gpu: bool, lease: bool, params: dict[str, Any],
            title: str, project_id: str | None) -> str:
    """Start the job; returns its id. Raises :class:`TestingBusy` when the job layer refuses it (GPU lease)."""
    spec = store.JobSpec(
        id=store.new_id(kind), kind=kind, gpu=lease, stop_grace_s=STOP_GRACE_S, title=title,
        project_id=project_id or None, params={**params, "role": role, "bids": list(bids), "uses_gpu": uses_gpu},
    )
    try:
        launch.start(spec)
    except launch.JobRejected as e:
        raise TestingBusy({"kind": str(e), "benchmark_id": ""}, str(e)) from e
    return spec.id


def cancel(benchmark_id: str) -> bool:
    """Ask every live job on this run to stop at the next case boundary (a compare job stops its whole group)."""
    hit = False
    for spec in _live_specs():
        if benchmark_id in (spec.params.get("bids") or []):
            try:
                (store.job_dir(spec.id) / _STOP_MARK).touch()
            except OSError:
                log.debug("could not mark %s as stopping", spec.id, exc_info=True)
            launch.stop(spec.id)
            hit = True
    return hit


def _job_status(status: str) -> str:
    """A row outcome as the job's exit status."""
    return {"done": "done", "cancelled": "stopped"}.get(status, "failed")


# ── Test run ──────────────────────────────────────────────────────────────────


@dataclass
class RunSpec:
    """Everything a test run needs; ``model_path`` is already resolved by the route."""

    project_id: str
    kind: str                       # suite | brain | heldout | training_leakage
    suite_name: str
    cases: list[BenchmarkCase]
    model_path: str
    run_id: str = ""                # requested owner run ('' = resolve)
    max_tokens: int = 512
    temperature: float = 0.3
    brain: dict[str, Any] = field(default_factory=dict)   # Smart Brain cases: top_k, corpus_path, max_context_chars
    config: dict[str, Any] = field(default_factory=dict)  # persisted next to the run (suite path, eval meta ...)
    extra_scores: dict[str, Any] = field(default_factory=dict)
    auto_judge: bool | None = None  # None = the saved setting
    judge_provider_id: str = ""     # '' = the saved default judge


def run_spec_to_params(spec: RunSpec) -> dict[str, Any]:
    return asdict(spec)


def run_spec_from_params(raw: dict[str, Any]) -> RunSpec:
    return RunSpec(**{**raw, "cases": [BenchmarkCase(**c) for c in raw.get("cases") or []]})


def _engine() -> Any:
    """This process's inference engine: the web app's global when the web app is loaded (the web process, the
    tests: they replace it), else the model manager's proxy to the inference worker (a job process never imports
    the web app). Both are the same ``RemoteEngine`` kind; the model itself lives in the inference worker."""
    from finetune_studio.models.manager import process_engine

    return process_engine()


def _load_model(model_path: str) -> None:
    from finetune_studio.models.manager import get_manager

    engine = _engine()
    if engine.model is not None and engine.model_path == model_path:
        return
    get_manager().unload()  # free a resident helper before the model under test loads
    engine.load(model_path)


def _execute_run(spec: RunSpec, bid: str, owner_run_id: str, stop: threading.Event) -> dict[str, Any]:
    """Blocking body of a test run (job process): load, ask, save each case.

    Another page can load a different model into the inference worker mid-run (the worker is shared; the job holds
    no lock the web process sees): the run then stops at the next case and fails saying so, instead of recording a
    different model's answers under this one's name."""
    _load_model(spec.model_path)
    engine = _engine()
    baseline = engine.model_path
    swapped: list[str] = []
    saved = 0
    brain_results: list[BrainCaseResult] = []

    def should_stop() -> bool:
        if stop.is_set():
            return True
        now = engine.model_path
        if now != baseline and not getattr(engine, "last_error", ""):  # a status read that failed is not a swap
            swapped.append(str(now or "nothing"))
            return True
        return False

    def progress() -> None:
        nonlocal saved
        saved += 1
        db.update_benchmark(bid, progress_done=saved, heartbeat_at=time.time())

    def check_swap() -> None:
        if swapped and not stop.is_set():
            raise RuntimeError(f"the model under test was replaced in the inference worker ({swapped[0]} is loaded "
                               f"now) after {saved} answer(s); the answers saved so far are kept")

    if spec.kind == "brain":
        query, corpus = open_brain_query(project_id=spec.project_id, corpus_path=str(spec.brain.get("corpus_path") or ""))
        max_chars = spec.brain.get("max_context_chars")
        if not max_chars:
            from finetune_studio.smart_brain.prompt import context_char_budget

            max_chars = context_char_budget(getattr(engine, "n_ctx", None), max_new_tokens=spec.max_tokens)

        def on_brain(r: BrainCaseResult) -> None:
            brain_results.append(r)
            run_store.save_case(bid, owner_run_id, r.case_result, judge_input=run_store.brain_judge_input(r))
            progress()

        run_brain_suite(
            engine, query, spec.cases, top_k=int(spec.brain.get("top_k") or DEFAULT_TOP_K), max_tokens=spec.max_tokens,
            temperature=spec.temperature, max_context_chars=int(max_chars), on_result=on_brain,
            should_stop=should_stop,
        )
        check_swap()
        return {"retrieval": compute_retrieval_metrics(brain_results), "corpus_path": corpus,
                "max_context_chars": int(max_chars)}

    def on_case(r: CaseResult) -> None:
        run_store.save_case(bid, owner_run_id, r)
        progress()

    run_suite(engine, spec.cases, max_tokens=spec.max_tokens, temperature=spec.temperature,
              on_result=on_case, should_stop=should_stop)
    check_swap()
    return {}


async def start_run_job(spec: RunSpec) -> dict[str, Any]:
    """Create the run row and launch its job; returns the (running) benchmark row immediately."""
    return launch_run(spec)


def launch_run(spec: RunSpec) -> dict[str, Any]:
    """:func:`start_run_job` without the event loop (no await between the busy check and the launch)."""
    _assert_free([], "run", uses_gpu=True)   # refuse before a row exists
    owner = run_store.resolve_owner_run(spec.project_id, spec.model_path, requested_run_id=spec.run_id)
    config = {"kind": spec.kind, "max_tokens": spec.max_tokens, "temperature": spec.temperature,
              "brain": spec.brain, **spec.config}
    row = run_store.start_run(owner, spec.suite_name, model_path=spec.model_path, kind=spec.kind,
                              config=config, total=len(spec.cases))
    auto = _auto_judge_enabled(spec.auto_judge)
    try:
        _launch(KIND_RUN, role="run", bids=[row["id"]], uses_gpu=True, lease=False, project_id=spec.project_id,
                title=f"test run {spec.suite_name}",
                params={"spec": run_spec_to_params(spec), "bid": row["id"], "owner_run_id": owner, "auto_judge": auto,
                        "judge_provider_id": spec.judge_provider_id or (default_judge_provider_id() if auto else "")})
    except TestingBusy as exc:
        run_store.finish_run(row["id"], status="failed", time_ms=0, error=f"not started: {exc}")
        raise
    return row


def run_in_job(ctx: Any) -> dict[str, Any]:
    """The job process side of a test run (``jobs.kinds.testrun``): answer, save, close the row, chain the judge."""
    p = ctx.params
    spec = run_spec_from_params(p["spec"])
    bid, owner = p["bid"], p["owner_run_id"]
    t0 = time.time()
    status, error, extra = "done", "", dict(spec.extra_scores)
    try:
        extra.update(_execute_run(spec, bid, owner, ctx.stop))
        if ctx.stop.is_set():
            status = "cancelled"
    except Exception as exc:
        log.exception("test run %s failed", bid)
        status, error = "failed", str(exc)
    try:
        run_store.finish_run(bid, status=status, time_ms=int((time.time() - t0) * 1000), error=error,
                             extra_scores=extra)
    except Exception:
        log.exception("could not finish test run %s", bid)
    if status == "done" and p.get("auto_judge"):
        try:
            launch_judge(bid, str(p.get("judge_provider_id") or ""), only_unjudged=True, exclude_job=ctx.id)
        except Exception as exc:  # noqa: BLE001 - e.g. busy; the run itself is complete and saved
            log.warning("auto-judge of %s not started: %s", bid, exc)
            db.update_benchmark(bid, judge_status="failed", judge_error=f"auto-judge not started: {exc}")
    return {"status": _job_status(status), "benchmark_id": bid, "row_status": status}


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
    return launch_judge(bid, provider_id, only_unjudged=only_unjudged)


def launch_judge(bid: str, provider_id: str = "", *, only_unjudged: bool = True,
                 exclude_job: str = "") -> dict[str, Any]:
    """:func:`start_judge_job` without the event loop; ``exclude_job`` = the run job chaining it (auto-judge)."""
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
    _assert_free([bid], "judge", uses_gpu=local, exclude=exclude_job)
    db.update_benchmark(bid, judge_status="running", judge_provider_id=provider, judge_model="", judge_done=0,
                        judge_total=total, judge_error="")
    launch_judge_rows([bid], provider, only_unjudged=only_unjudged, local=local,
                      title=f"judge {bench.get('suite_name') or bid}", project_id=None)
    return db.get_benchmark(bid) or {}


def launch_judge_rows(bids: list[str], provider: str, *, only_unjudged: bool, local: bool, title: str,
                      project_id: str | None) -> str:
    """Launch the judge job for rows already marked ``judge_status=running``; a refusal is written to every row."""
    try:
        return _launch(KIND_JUDGE, role="judge", bids=bids, uses_gpu=local, lease=local, project_id=project_id,
                       title=title, params={"provider_id": provider, "only_unjudged": only_unjudged})
    except TestingBusy as exc:
        for bid in bids:
            db.update_benchmark(bid, judge_status="failed", judge_error=f"not started: {exc}")
        raise


def judge_rows_blocking(bids: list[str], provider: str, only_unjudged: bool,
                        stop: threading.Event) -> dict[str, tuple[str, str]]:
    """Open the judge once and judge each row in turn. Returns ``{bid: (judge_status, error)}``.

    A local GGUF judge is loaded into the inference worker (``open_judge`` -> the model manager) and unloaded after."""
    out: dict[str, tuple[str, str]] = {}
    with open_judge(provider) as judge:
        for bid in bids:
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


def judge_in_job(ctx: Any) -> dict[str, Any]:
    """The job process side of a judge job (``jobs.kinds.judge``): one run, or every row of a compare group."""
    p = ctx.params
    bids: list[str] = list(p["bids"])
    results: dict[str, tuple[str, str]] = {}
    error = ""
    try:
        results = judge_rows_blocking(bids, str(p["provider_id"]), bool(p.get("only_unjudged", True)), ctx.stop)
    except Exception as exc:  # noqa: BLE001 - JudgeUnavailable and provider errors alike; written to every row
        log.warning("judge job %s failed: %s", ctx.id, exc)
        error = str(exc) if isinstance(exc, JudgeUnavailable) else f"{type(exc).__name__}: {exc}"
    from finetune_studio.testing.scoring import rescore_benchmark

    statuses = []
    for bid in bids:
        status, row_error = results.get(bid, ("failed", error or "the judge job ended without a result"))
        statuses.append(status)
        try:
            rescore_benchmark(bid)
        except Exception:
            log.exception("could not rescore %s", bid)
        db.update_benchmark(bid, judge_status=status, judge_error=row_error, heartbeat_at=time.time())
    overall = "cancelled" if "cancelled" in statuses else ("done" if all(s == "done" for s in statuses) else "failed")
    return {"status": _job_status(overall), "benchmark_ids": bids, "judge_status": dict(zip(bids, statuses, strict=True))}
