"""Preference-pair (DPO) authoring API: ``/api/projects/{pid}/data-prep/preference``.

Thin HTTP layer over ``data.prep.preference``. The build makes many blocking model calls, so it runs in a
worker thread (``asyncio.to_thread``) — never inline on the event loop. While it runs, ``GET …/progress``
reports attempts so the Pairs page can show a real progress bar; one build per project at a time.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from finetune_studio import db
from finetune_studio.data.prep import preference as pref

router = APIRouter()

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class PreferenceBody(BaseModel):
    kinds: list[Literal["hallucination", "abstain"]] = Field(
        default_factory=lambda: list(pref.KINDS), min_length=1, max_length=len(pref.KINDS))
    max_pairs: int = Field(default=pref.DEFAULT_MAX_PAIRS, ge=2, le=pref.MAX_PAIRS_LIMIT)
    seed: int = 42


def _response_body(pid: str, built: pref.BuiltPreference) -> dict[str, Any]:
    ds = built.persisted.dataset
    return {
        "dataset": {"id": ds.get("id"), "name": ds.get("name"), "path": str(built.persisted.path),
                    "rows": built.persisted.rows},
        **built.report.as_dict(),
        "train_url": f"/projects/{pid}/training?dataset={ds.get('id')}&mode=dpo",
    }


@router.post("/projects/{pid}/data-prep/preference", response_model=None)
async def build_preference(pid: str, body: PreferenceBody):
    """Author preference pairs from the project's approved Q&A, write + register the DPO dataset."""
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        if pid in _JOBS and _JOBS[pid]["running"]:
            return JSONResponse({"error": "a preference build is already running for this project"},
                                status_code=409)
        _JOBS[pid] = {"running": True, "attempted": 0, "planned": 0,
                      "kept": {k: 0 for k in body.kinds}, "kind": body.kinds[0]}

    def on_progress(update: dict[str, Any]) -> None:
        with _JOBS_LOCK:
            _JOBS[pid].update(update)

    try:
        built = await asyncio.to_thread(
            pref.build_preference_dataset, pid, tuple(body.kinds), body.max_pairs, body.seed,
            None, on_progress)
    except pref.NoApprovedPairs as exc:
        return JSONResponse({"error": str(exc), "code": "no_approved_pairs"}, status_code=409)
    except pref.NoGenerator as exc:
        return JSONResponse({"error": str(exc), "code": "no_model"}, status_code=409)
    except pref.NoUsablePairs as exc:
        return JSONResponse({"error": str(exc), "code": "no_usable_pairs", **exc.report.as_dict()},
                            status_code=422)
    except pref.GenerationFailed as exc:
        return JSONResponse({"error": str(exc), "code": "generation_failed"}, status_code=502)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    finally:
        with _JOBS_LOCK:
            _JOBS[pid]["running"] = False
    return _response_body(pid, built)


@router.get("/projects/{pid}/data-prep/preference/progress", response_model=None)
async def preference_progress(pid: str):
    """Live progress of the running build (``running: false`` when none)."""
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    with _JOBS_LOCK:
        job = _JOBS.get(pid)
        return dict(job, kept=dict(job["kept"])) if job else {"running": False}
