"""Smart Brain index builds as supervised job processes: a web restart no longer ends them.

``start`` checks the build can run and records it (``knowledge.index.prepare``: a ``queued`` ``brain_builds`` row),
then launches job ``brain_build-<build id>`` (``jobs.kinds.brain_build``), which embeds the project's passages
(``knowledge.index.run``) and finishes the row: ``done`` with the counts, or ``error`` with the real error. The
build routes answer 202 with the build id; the pages poll ``GET …/brain/builds/<id>`` (``status``) until it ends.

GPU lease: the job takes it (``gpu=True``) only when the embedder will run on a GPU (``smart_brain.devices``):
then it never runs beside training or an export (they fill the card; the second one would OOM), and the
supervisor unloads the inference model first, like for every GPU job. On a CPU-only box a build takes no lease
and runs beside anything. The embedder's memory leaves with the job process.

One build per project at a time: a second one while one runs is a 409.
"""
from __future__ import annotations

import logging
from typing import Any

from finetune_studio import db
from finetune_studio.db import brain_builds as builds_db
from finetune_studio.jobs import launch, store

log = logging.getLogger(__name__)

KIND = "brain_build"
STOP_GRACE_S = 30.0
TERMINAL_STATUSES = frozenset({"done", "error", "failed"})


class BuildBusy(RuntimeError):
    """A build of this project is already running."""

    def __init__(self, build_id: str) -> None:
        super().__init__(f"an index build of this project is already running (build {build_id}); wait for it")
        self.build_id = build_id


def job_id(build_id: str) -> str:
    return f"{KIND}-{build_id}"


def _embedder_on_gpu() -> bool:
    from finetune_studio.smart_brain.devices import resolve_device
    return resolve_device("auto").split(":")[0] != "cpu"


def _running() -> list[dict[str, Any]]:
    """``{build_id, project_id}`` of the builds whose job process is alive."""
    out = []
    for j in launch.list_jobs(KIND, running_only=True):
        try:
            out.append({"build_id": str(store.read_spec(j["id"]).params["build_id"]), "project_id": j["project_id"]})
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return out


def running_build_ids() -> set[str]:
    """For the startup reconcile: rows whose job outlived the web restart are left alone."""
    return {b["build_id"] for b in _running()}


def start(pid: str, *, embedder: str | None = None, reset: bool = False,
          fact_cards: bool | None = None) -> dict[str, Any]:
    """Record the build and launch its job; returns ``build_id``, ``queued_files``, ``fact_cards``, ``gpu``.

    Raises IndexBuildError (404 / 400), :class:`BuildBusy`, or ``launch.JobRejected`` (409 when the GPU is busy).
    """
    from finetune_studio.knowledge import index

    busy = next((b for b in _running() if b["project_id"] == pid), None)
    if busy:
        raise BuildBusy(busy["build_id"])
    plan = index.prepare(pid, fact_cards=fact_cards)
    bid = plan["build_id"]
    gpu = _embedder_on_gpu()
    spec = store.JobSpec(
        id=job_id(bid), kind=KIND, gpu=gpu, stop_grace_s=STOP_GRACE_S, project_id=pid,
        title="Smart Brain index " + ("rebuild" if reset else "build"),
        params={"build_id": bid, "project_id": pid, "embedder": embedder or "", "reset": bool(reset),
                "fact_cards": bool(plan["fact_cards"])},
    )
    try:
        launch.start(spec)
    except launch.JobRejected as e:
        builds_db.fail_if_active(bid, f"not started: {e}")
        raise
    if gpu:
        # GPU work starts: the web process's cached search models go (AGENTS.md: release_brain_models first).
        from finetune_studio.smart_brain.model_cache import release_brain_models
        release_brain_models("index build job")
    return {**plan, "gpu": gpu}


def _job(build_id: str) -> dict[str, Any] | None:
    try:
        return launch.describe(job_id(build_id))
    except (FileNotFoundError, ValueError):
        return None                       # no job: a CLI build (``fts brain build``) or one from before jobs


def _settle(row: dict[str, Any], job: dict[str, Any] | None) -> dict[str, Any]:
    """A row still active whose job has ended without finishing it (crashed, killed): record that, honestly."""
    if row.get("status") in TERMINAL_STATUSES or job is None or job["running"]:
        return row
    ex = job.get("exit") or {}
    reason = str(ex.get("error") or ex.get("detail") or "the build process ended without a result")
    if builds_db.fail_if_active(row["id"], f"index build job ended: {reason}"):
        log.warning("brain build %s: job ended without finishing the row (%s)", row["id"], reason)
    return db.get_brain_build(row["id"]) or row


def status(pid: str, build_id: str) -> dict[str, Any] | None:
    """What the pages poll: the build row (+ the job's result once done). None = no such build in this project."""
    row = db.get_brain_build(build_id)
    if row is None or row.get("project_id") != pid:
        return None
    job = _job(build_id)
    row = _settle(row, job)
    st = str(row.get("status") or "queued")
    result = ((job or {}).get("exit") or {}).get("result") or {}
    running = bool(job and job["running"])
    # Finished once the row is final AND its job has exited: the job's result (vector_dim, fact cards...) is
    # recorded right after the row.
    finished = st in TERMINAL_STATUSES and not running
    out: dict[str, Any] = {
        "build_id": build_id, "status": st, "finished": finished, "ok": finished and st == "done",
        "running": running,
        "documents": int(row.get("doc_count") or 0), "chunks": int(row.get("chunk_count") or 0),
        "error": row.get("error") or None,
        "created_at": row.get("created_at"), "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"), "duration_ms": row.get("duration_ms"),
    }
    for k in ("vector_dim", "skipped", "fact_cards", "fact_card_count", "corpus_dir"):
        if k in result:
            out[k] = result[k]
    return out


def active_build(pid: str) -> dict[str, Any] | None:
    """The project's latest build while it is queued/running (settled first), else None."""
    row = builds_db.latest_for_project(pid)
    if row is None or row.get("status") in TERMINAL_STATUSES:
        return None
    row = _settle(row, _job(row["id"]))
    return None if row.get("status") in TERMINAL_STATUSES else row


def on_finished(job: dict[str, Any]) -> None:
    """Web process (``jobs.watch``): a build job ended; a row it left active (crash, kill) is failed honestly."""
    try:
        bid = str(store.read_spec(job["id"]).params["build_id"])
    except (FileNotFoundError, ValueError, KeyError):
        return
    row = db.get_brain_build(bid)
    if row is not None:
        _settle(row, _job(bid))


def run_in_job(params: dict[str, Any], release_gpu: Any = lambda: None) -> dict[str, Any]:
    """The job process side: run the build; the row records the outcome (``knowledge.index.run``)."""
    from finetune_studio.knowledge import index

    try:
        result = index.run(str(params["project_id"]), str(params["build_id"]), embedder=params.get("embedder") or None,
                           reset=bool(params.get("reset")), fact_cards=bool(params.get("fact_cards")))
    finally:
        release_gpu()
    keep = ("documents", "chunks", "vector_dim", "skipped", "fact_cards", "fact_card_count", "corpus_dir")
    return {k: result[k] for k in keep if k in result}
