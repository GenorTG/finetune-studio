"""Delete one training run: its row (children cascade) and the run's own output directory.

The run delete routes used to drop only the ``training_runs`` row: ``output/projects/<pid>/runs/<rid>`` (the
adapter, the merged weights, the GGUF exports: 18 GB for a merged 9B) stayed on disk for ever, like the project
delete did before 2026-10-01. Both routes (``/api/projects/{pid}/runs/{rid}`` and the benchmarks mirror) call
:func:`delete_run`.

What is removed on disk: the run's ``output_path`` directory, and only when it is this run's own directory
(``output/.../runs/<rid>`` or any ``<...>/<rid>`` under the ``output`` root). A custom ``output_dir`` shared by
several runs (``POST /runs/{id}/set-output`` accepts any path) is kept and the response says so: deleting one
run must never take another run's weights with it.

A run whose training or export job is still alive is not deleted (:class:`RunBusy`, a 409): stop the job first.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from finetune_studio import db
from finetune_studio.jobs import launch, store

log = logging.getLogger(__name__)

OUTPUT_ROOT = "output"


class RunNotFound(LookupError):
    """No such run in this project."""


class RunBusy(RuntimeError):
    """The run's training or export job is still running."""

    def __init__(self, what: str, job_id: str) -> None:
        super().__init__(f"this run's {what} is still running (job {job_id}); stop it before deleting the run")
        self.what, self.job_id = what, job_id


def _live_job(rid: str) -> tuple[str, str] | None:
    """``(what, job id)`` of a running training or export job of this run, else None."""
    for j in launch.list_jobs("training", running_only=True):
        try:
            if store.read_spec(j["id"]).params.get("run_id") == rid:
                return "training", j["id"]
        except (FileNotFoundError, ValueError):
            continue
    for j in launch.list_jobs("export", running_only=True):
        try:
            eid = str(store.read_spec(j["id"]).params.get("export_id") or "")
        except (FileNotFoundError, ValueError):
            continue
        row = db.get_export(eid) if eid else None
        if row and row.get("run_id") == rid:
            return "export", j["id"]
    return None


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def run_dir_to_remove(rid: str, output_path: str) -> tuple[Path | None, str]:
    """The directory a delete may remove for run ``rid``, or ``(None, why it is kept)``.

    Removable = an existing directory under the ``output`` root (cwd-relative, where training writes) whose last
    segment is the run id. Anything else (a shared custom output dir, a path elsewhere) is kept.
    """
    raw = (output_path or "").strip()
    if not raw:
        return None, "the run has no output path"
    try:
        target = Path(raw).resolve()
        root = Path(OUTPUT_ROOT).resolve()
    except (OSError, RuntimeError) as e:
        return None, f"the output path could not be resolved: {e}"
    if not target.is_dir():
        return None, "the output path does not exist" if not target.exists() else "the output path is not a directory"
    if target == root or not target.is_relative_to(root):
        return None, f"the output path is outside {OUTPUT_ROOT}/ and is kept"
    if target.name != rid:
        return None, "the output path is not this run's own directory (a shared output dir) and is kept"
    return target, ""


def delete_run(pid: str, rid: str) -> dict[str, Any]:
    """Delete run ``rid`` of project ``pid``: DB rows (benchmarks, exports, quant exports, suites cascade), the
    project's ``production_run`` pointer when it named this run, and the run's own output directory.

    Returns ``{"ok": True, "run_id", "removed_dir" (path or None), "freed_bytes", "kept": why files stayed}``.
    Raises :class:`RunNotFound`, :class:`RunBusy`.
    """
    run = db.get_run(rid)
    if not run or run.get("project_id") != pid:
        raise RunNotFound(rid)
    live = _live_job(rid)
    if live is not None:
        raise RunBusy(*live)
    target, kept = run_dir_to_remove(rid, str(run.get("output_path") or ""))

    with db.cursor() as c:
        # model_exports has no FK cascade (the row keeps its own output_path, which lives under the run dir).
        c.execute("DELETE FROM model_exports WHERE run_id = ?", (rid,))
        c.execute("UPDATE projects SET production_run = NULL WHERE id = ? AND production_run = ?", (pid, rid))
    db.delete_run(rid)

    freed = 0
    removed: str | None = None
    if target is not None:
        freed = _dir_size(target)
        try:
            shutil.rmtree(target)
            removed = str(target)
        except OSError as e:
            # The row is gone (the user asked for that); the files are reported, not hidden.
            log.warning("run %s: output dir %s could not be removed: %s", rid, target, e)
            kept, freed = f"the output directory could not be removed: {e}", 0
    return {"ok": True, "run_id": rid, "removed_dir": removed, "freed_bytes": freed, "kept": kept or None}
