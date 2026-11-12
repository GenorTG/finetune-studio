"""Smart Brain quiz API: ``/api/projects/{pid}/testing/brain-quiz`` (questions written from the indexed documents).

Thin HTTP layer over ``testing.brain_quiz``. One helper call per question, so the build runs in a worker thread;
``GET .../progress`` feeds the status line; one build per project.
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
from finetune_studio.testing import brain_quiz as bq

router = APIRouter()

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class BrainQuizBody(BaseModel):
    questions: int = Field(default=bq.DEFAULT_QUESTIONS, ge=5, le=bq.MAX_QUESTIONS)


@router.post("/projects/{pid}/testing/brain-quiz", response_model=None)
async def build_brain_quiz(pid: str, body: BrainQuizBody):
    """Write a quiz from the project's indexed passages and save it as a project suite (replacing the last one)."""
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        if _JOBS.get(pid, {}).get("running"):
            return JSONResponse({"detail": "a quiz is already being written for this project"}, status_code=409)
        _JOBS[pid] = {"running": True, "done": 0, "planned": 0, "kept": 0}

    def on_progress(update: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            _JOBS[pid].update(update)

    try:
        return await asyncio.to_thread(bq.build_brain_quiz, pid, questions=body.questions, progress=on_progress)
    except bq.NoIndex as exc:
        return JSONResponse({"detail": str(exc), "code": "no_index"}, status_code=409)
    except NoGenerator as exc:
        return JSONResponse({"detail": str(exc), "code": "no_model"}, status_code=409)
    except bq.NoUsableQuestions as exc:
        return JSONResponse({"detail": str(exc), "code": "no_usable_questions", **exc.report.as_dict()}, status_code=422)
    finally:
        with _JOBS_LOCK:
            _JOBS[pid]["running"] = False


@router.get("/projects/{pid}/testing/brain-quiz/progress", response_model=None)
async def brain_quiz_progress(pid: str):
    """Live progress of the running build (``running: false`` when none)."""
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        return dict(_JOBS.get(pid) or {"running": False})
