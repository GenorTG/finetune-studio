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


MODE_BUILD, MODE_QUICK, MODE_REEMBED = "build", "quick", "re-embed"
MODES = frozenset({MODE_BUILD, MODE_QUICK, MODE_REEMBED})
# What a finished build reports (the job result -> ``status``): counts, the parse step's outcome (quick), the
# embedder a re-embed used.
RESULT_KEYS = ("documents", "chunks", "vector_dim", "skipped", "fact_cards", "fact_card_count", "corpus_dir",
               "promoted", "promoted_files", "failed", "embedder")


def _title(mode: str, reset: bool) -> str:
    if mode == MODE_QUICK:
        return "Smart Brain quick index (parse, then build)"
    if mode == MODE_REEMBED:
        return "Smart Brain re-embed"
    return "Smart Brain index " + ("rebuild" if reset else "build")


def start(pid: str, *, embedder: str | None = None, reset: bool = False,
          fact_cards: bool | None = None, mode: str = MODE_BUILD,
          file_ids: list[str] | None = None) -> dict[str, Any]:
    """Record the build and launch its job; returns ``build_id``, ``queued_files``, ``pending_parses``,
    ``fact_cards``, ``gpu``, ``mode``.

    ``mode``: ``build`` embeds every parsed file; ``quick`` parses the pending library files first (``file_ids``
    restricts that step), then builds; ``re-embed`` keeps the documents and replaces the vectors (the current
    settings' embedder when none is given). All three are the one job kind: every step that loads the embedder
    runs in the job process, never in the web process.

    Raises IndexBuildError (404 / 400), :class:`BuildBusy`, or ``launch.JobRejected`` (409 when the GPU is busy).
    """
    from finetune_studio.knowledge import index

    if mode not in MODES:
        raise ValueError(f"unknown build mode {mode!r}")
    busy = next((b for b in _running() if b["project_id"] == pid), None)
    if busy:
        raise BuildBusy(busy["build_id"])
    if mode == MODE_REEMBED:
        plan = index.prepare_rebuild_vectors(pid)
    else:
        plan = index.prepare(pid, fact_cards=fact_cards, parse_first=mode == MODE_QUICK, file_ids=file_ids)
    bid = plan["build_id"]
    gpu = _embedder_on_gpu()
    spec = store.JobSpec(
        id=job_id(bid), kind=KIND, gpu=gpu, stop_grace_s=STOP_GRACE_S, project_id=pid,
        title=_title(mode, reset),
        params={"build_id": bid, "project_id": pid, "embedder": embedder or "", "reset": bool(reset),
                "fact_cards": bool(plan["fact_cards"]), "mode": mode, "file_ids": list(file_ids or [])},
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
    return {**plan, "gpu": gpu, "mode": mode}


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
    for k in RESULT_KEYS:
        if k in result:
            out[k] = result[k]
    try:
        out["mode"] = str(store.read_spec(job_id(build_id)).params.get("mode") or MODE_BUILD)
    except (FileNotFoundError, ValueError):
        out["mode"] = MODE_BUILD
    state = (job or {}).get("state") or {}
    if running and state.get("phase") in ("parsing", "embedding"):
        out[str(state["phase"])] = {"done": int(state.get("done") or 0), "total": int(state.get("total") or 0)}
    return out


def active_build(pid: str) -> dict[str, Any] | None:
    """The project's latest build while it is queued/running (settled first), else None."""
    row = builds_db.latest_for_project(pid)
    if row is None or row.get("status") in TERMINAL_STATUSES:
        return None
    row = _settle(row, _job(row["id"]))
    return None if row.get("status") in TERMINAL_STATUSES else row


def stop(pid: str) -> dict[str, Any] | None:
    """Ask the project's running build to stop (it ends at the next embedded batch, the old index intact).
    None = no build is running."""
    row = active_build(pid)
    if row is None:
        return None
    launch.stop(job_id(row["id"]))
    return {"build_id": row["id"], "stopping": True}


def on_finished(job: dict[str, Any]) -> None:
    """Web process (``jobs.watch``): a build job ended; a row it left active (crash, kill) is failed honestly."""
    try:
        bid = str(store.read_spec(job["id"]).params["build_id"])
    except (FileNotFoundError, ValueError, KeyError):
        return
    row = db.get_brain_build(bid)
    if row is not None:
        _settle(row, _job(bid))


def run_in_job(params: dict[str, Any], release_gpu: Any = lambda: None,
               progress: Any = None, on_batch: Any = None) -> dict[str, Any]:
    """The job process side: run the build in its ``mode``; the row records the outcome (``knowledge.index``).

    ``progress(phase, done, total)`` publishes the quick index's parse step for the pages; ``on_batch(done, total)``
    the embedding (it raises to stop the build)."""
    from finetune_studio.knowledge import index

    pid, bid = str(params["project_id"]), str(params["build_id"])
    mode = str(params.get("mode") or MODE_BUILD)
    try:
        if mode == MODE_REEMBED:
            result = index.run_rebuild_vectors(pid, bid, embedder=params.get("embedder") or None, on_batch=on_batch)
        else:
            result = index.run(pid, bid, embedder=params.get("embedder") or None, reset=bool(params.get("reset")),
                               fact_cards=bool(params.get("fact_cards")), parse_first=mode == MODE_QUICK,
                               file_ids=list(params.get("file_ids") or []) or None, progress=progress,
                               on_batch=on_batch)
    finally:
        release_gpu()
    return {k: result[k] for k in RESULT_KEYS if k in result}
