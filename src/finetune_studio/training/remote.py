"""The web app's view of training: the same interface as ``TrainingEngine`` (``state``, ``start``, ``stop``), while
the run itself is a supervised job process (``jobs.kinds.training``).

A web restart or crash no longer ends a run: the job keeps training and keeps writing its ``training_runs`` row;
a restarted web app finds the running job again and reads its live ``state.json``.
"""
from __future__ import annotations

import time
from typing import Any

from finetune_studio.jobs import launch, store
from finetune_studio.training.engine import TrainingState

# ``waiting``: the job is queued behind another GPU job (a Smart Brain build, picture reading, an export) and starts
# by itself when the GPU is free (``JobSpec.queue_gpu``).
ACTIVE = ("waiting", "loading", "training", "saving", "running")
STOPPED_WAITING = "Stopped while it was waiting for the GPU (it never started)."
STOP_GRACE_S = 180.0          # the trainer saves the adapter on stop; a 9B save takes a while


class TrainingBusy(RuntimeError):
    pass


class RemoteTrainingEngine:
    def __init__(self) -> None:
        self._job_id: str | None = None
        self._cache: tuple[float, TrainingState] | None = None

    # ── which job ──────────────────────────────────────────────────────
    def _current_job(self) -> dict[str, Any] | None:
        """The running training job, else the most recent one (its final state stays on the page)."""
        jobs = launch.list_jobs("training")
        if not jobs:
            return None
        running = [j for j in jobs if j["running"]]
        job = running[-1] if running else jobs[-1]
        self._job_id = job["id"]
        return job

    @property
    def state(self) -> TrainingState:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < 0.25:     # SSE polls every 0.5 s per open tab
            return self._cache[1]
        st = self._read_state()
        self._cache = (now, st)
        return st

    def _read_state(self) -> TrainingState:
        job = self._current_job()
        if job is None:
            return TrainingState()
        if job["running"] and job.get("queued"):
            label = launch.holder_label(str(job.get("waiting_for") or ""))
            return TrainingState(status="waiting", message=f"Waiting for the GPU: {label}. The run starts by itself "
                                                           "when the GPU is free.")
        raw = job.get("state") or {}
        if raw.get("phase") == "waiting_for_gpu":         # the queue's own state record, not the trainer's
            raw = {}
        st = TrainingState(**{k: v for k, v in raw.items() if k in TrainingState.__dataclass_fields__})
        if not job["running"]:
            ex = job.get("exit") or {}
            if ex.get("status") == "stopped" and not job.get("pid") and not raw:
                st.status, st.message = "stopped", STOPPED_WAITING     # stopped in the queue: it never ran
                return st
            if st.status in ACTIVE or not raw:          # died before it could say so
                st.status = "stopped" if ex.get("status") == "stopped" else "error"
                st.error = st.error or ex.get("error") or ex.get("detail") or (
                    f"the training process ended unexpectedly ({ex.get('signal') or 'exit ' + str(ex.get('code'))})")
                st.message = st.message if st.status == "stopped" else st.error
        elif not raw:
            st.status, st.message = "loading", "Starting the training process..."
        return st

    def _params(self) -> dict[str, Any]:
        if not self._job_id:
            self._current_job()
        if not self._job_id:
            return {}
        try:
            return store.read_spec(self._job_id).params
        except (FileNotFoundError, ValueError):
            return {}

    @property
    def current_project_id(self) -> str | None:
        return self._params().get("project_id")

    @property
    def current_db_run_id(self) -> str | None:
        return self._params().get("run_id")

    @property
    def current_run_id(self) -> str | None:
        p = self._params()
        if not p:
            return None
        return f"{p['project_id']}-{p['run_id']}" if p.get("project_id") else p.get("run_id")

    # ── control ─────────────────────────────────────────────────────────
    def start(self, config: Any, training_data: list, system_prompt: str = "", *, run_id: str, output_dir: str,
              project_id: str | None) -> str:
        """Launch the training job; returns its job id. Raises TrainingBusy / launch.JobRejected."""
        if self.state.status in ACTIVE:
            raise TrainingBusy("another training run is in progress")
        from finetune_studio.jobs.kinds.training import write_rows
        from finetune_studio.smart_brain.model_cache import release_brain_models
        from finetune_studio.training.worker import config_to_dict

        release_brain_models("training start")          # the embedder/reranker live in THIS process
        # Behind another GPU job (a build, picture reading, an export) the run waits in the lease queue and starts by
        # itself, instead of a 409 the user had to retry by hand; its memory checks run in the job, once it has the GPU.
        spec = store.JobSpec(
            id=store.new_id("training"), kind="training", gpu=True, queue_gpu=True, stop_grace_s=STOP_GRACE_S,
            title=f"training run {run_id}", project_id=project_id,
            params={"config": config_to_dict(config), "system_prompt": system_prompt, "run_id": run_id,
                    "output_dir": output_dir, "project_id": project_id},
        )
        d = store.job_dir(spec.id)
        d.mkdir(parents=True, exist_ok=True)
        write_rows(d, training_data)
        launch.start(spec)
        self._job_id, self._cache = spec.id, None
        return spec.id

    def stop(self) -> None:
        job = self._current_job()
        if job and job["running"]:
            queued = bool(job.get("queued"))
            launch.stop(job["id"])
            self._cache = None
            if queued:
                # It never ran, so no trainer callback will finish its row: say "stopped" here.
                run_id = self._params().get("run_id")
                if run_id:
                    from finetune_studio import db
                    db.update_run(run_id, status="stopped", error=STOPPED_WAITING, finished_at=time.time())

    def waiting_for(self) -> str:
        """Plain words for what the queued run waits behind ("" when it is not waiting)."""
        job = self._current_job()
        if job and job["running"] and job.get("queued"):
            return launch.holder_label(str(job.get("waiting_for") or ""))
        return ""

    def running_run_ids(self) -> set[str]:
        """``training_runs`` ids whose job is alive (the startup reconcile must leave those alone)."""
        out = set()
        for j in launch.list_jobs("training", running_only=True):
            try:
                out.add(store.read_spec(j["id"]).params.get("run_id"))
            except (FileNotFoundError, ValueError):
                continue
        return {r for r in out if r}
