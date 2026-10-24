"""Study-set API: ``/api/projects/{pid}/data-prep/study`` (each approved fact written several ways).

Thin HTTP layer over ``data.prep.study_set``. One helper call per approved pair, so the build runs in a worker thread
(``asyncio.to_thread``), never on the event loop; ``GET …/progress`` feeds the progress bar; one build per project.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from finetune_studio import db
from finetune_studio.data.prep import study_set as study
from finetune_studio.data.prep.preference import NoGenerator

router = APIRouter()

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class StudyBody(BaseModel):
    reworded: int = Field(default=study.DEFAULT_REWORDED, ge=0, le=study.MAX_PER_KIND)
    restated: int = Field(default=study.DEFAULT_RESTATED, ge=0, le=study.MAX_PER_KIND)
    passage: bool = True


@router.post("/projects/{pid}/data-prep/study", response_model=None)
async def build_study(pid: str, body: StudyBody):
    """Write the study set from the approved pairs and register it as an SFT dataset."""
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    try:
        opts = study.StudyOptions(reworded=body.reworded, restated=body.restated, passage=body.passage)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    with _JOBS_LOCK:
        if _JOBS.get(pid, {}).get("running"):
            return JSONResponse({"detail": "a study-set build is already running for this project"}, status_code=409)
        _JOBS[pid] = {"running": True, "done": 0, "planned": 0, "kept": dict.fromkeys(study.KINDS, 0)}

    def on_progress(update: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            _JOBS[pid].update(update)

    try:
        built = await asyncio.to_thread(study.build_study_dataset, pid, opts, progress=on_progress)
    except study.NoApprovedPairs as exc:
        return JSONResponse({"detail": str(exc), "code": "no_approved_pairs"}, status_code=409)
    except NoGenerator as exc:
        return JSONResponse({"detail": str(exc), "code": "no_model"}, status_code=409)
    except study.NoUsableRows as exc:
        return JSONResponse({"detail": str(exc), "code": "no_usable_rows", **exc.report.as_dict()}, status_code=422)
    finally:
        with _JOBS_LOCK:
            _JOBS[pid]["running"] = False
    ds = built.dataset
    return {
        "dataset": {"id": ds.get("id"), "name": ds.get("name"), "path": built.path, "rows": built.rows},
        **built.report.as_dict(),
        "train_url": f"/projects/{pid}/training?dataset={ds.get('id')}",
    }


@router.get("/projects/{pid}/data-prep/study/progress", response_model=None)
async def study_progress(pid: str):
    """Live progress of the running build (``running: false`` when none)."""
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        job = _JOBS.get(pid)
        return dict(job, kept=dict(job["kept"])) if job else {"running": False}
