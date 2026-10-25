"""Reworded held-out quiz API: ``/api/projects/{pid}/testing/reworded-quiz`` (held-out facts, new wording).

Thin HTTP layer over ``testing.reworded_quiz``. One helper call per held-out fact, so the build runs in a worker thread
(``asyncio.to_thread``), never on the event loop; ``GET …/progress`` feeds the status line; one build per project.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from finetune_studio import db
from finetune_studio.data.prep.preference import NoGenerator
from finetune_studio.testing import reworded_quiz as rq

router = APIRouter()

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class RewordedQuizBody(BaseModel):
    dataset_id: str | None = None
    max_questions: int = Field(default=rq.DEFAULT_MAX_QUESTIONS, ge=1, le=rq.MAX_QUESTIONS)


@router.post("/projects/{pid}/testing/reworded-quiz", response_model=None)
async def build_reworded_quiz(pid: str, body: RewordedQuizBody):
    """Reword each held-out question of the dataset once and save the quiz as a project suite."""
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        if _JOBS.get(pid, {}).get("running"):
            return JSONResponse({"detail": "a reworded quiz is already being built for this project"}, status_code=409)
        _JOBS[pid] = {"running": True, "done": 0, "planned": 0, "kept": 0}

    def on_progress(update: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            _JOBS[pid].update(update)

    try:
        built = await asyncio.to_thread(rq.build_reworded_quiz, pid, dataset_id=body.dataset_id,
                                        max_questions=body.max_questions, progress=on_progress)
    except LookupError as exc:
        status = 404 if body.dataset_id else 409
        return JSONResponse({"detail": str(exc), "code": "no_dataset"}, status_code=status)
    except NoGenerator as exc:
        return JSONResponse({"detail": str(exc), "code": "no_model"}, status_code=409)
    except rq.NoHeldoutRows as exc:
        return JSONResponse({"detail": str(exc), "code": "no_heldout"}, status_code=409)
    except rq.NoUsableQuestions as exc:
        return JSONResponse({"detail": str(exc), "code": "no_usable_questions", **exc.report.as_dict()},
                            status_code=422)
    finally:
        with _JOBS_LOCK:
            _JOBS[pid]["running"] = False
    return {"name": built.name, "path": built.path, "label": built.label, "case_count": built.case_count,
            "dataset": built.dataset, **built.report.as_dict()}


@router.get("/projects/{pid}/testing/reworded-quiz/progress", response_model=None)
async def reworded_quiz_progress(pid: str):
    """Live progress of the running build (``running: false`` when none)."""
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        return dict(_JOBS.get(pid) or {"running": False})
