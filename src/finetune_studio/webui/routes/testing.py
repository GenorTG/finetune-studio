"""Testing tab — run test suites, view results."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.testing.suite import (
    apply_heuristic_judging,
    load_test_suite,
    run_suite,
    score_results,
)
from finetune_studio.webui.app import inference_engine

router = APIRouter()
_log = logging.getLogger(__name__)


def _resolve_merged_model(pid: str) -> str | None:
    """Return path to the most-recent completed run's merged model, or None."""
    runs = db.list_runs(pid)
    for run in runs:  # already newest-first
        if run.get("status") != "completed":
            continue
        output_path = (run.get("output_path") or "").strip()
        if not output_path:
            continue
        merged = os.path.join(output_path, "merged")
        return merged
    return None


@router.post("/load")
async def load_model(request: Request):
    """Load a model into the global inference engine.

    Accepts ``{"model_path": "..."}`` or ``{"path": "..."}`` (same keys as
    ``/api/models/load`` / chat-v2) so UI callers don't get ``No model_path``.
    """
    body = await request.json()
    model_path = body.get("model_path") or body.get("path") or ""
    if not model_path:
        return {"error": "No model_path"}
    try:
        kwargs: dict = {}
        if "max_seq_length" in body:
            kwargs["max_seq_length"] = int(body["max_seq_length"])
        if "load_in_4bit" in body:
            kwargs["load_in_4bit"] = bool(body["load_in_4bit"])
        inference_engine.load(model_path, **kwargs)
        return {"status": "loaded", "model": model_path}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.post("/unload")
async def unload_model():
    inference_engine.unload()
    return {"status": "unloaded"}


@router.get("/status")
async def model_status():
    return {
        "loaded": inference_engine.model is not None,
        "model_path": inference_engine.model_path,
        "is_gguf": inference_engine.is_gguf,
    }


@router.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    response = inference_engine.generate(
        messages, max_tokens=max_tokens, temperature=temperature
    )
    return {"response": response}


@router.post("/run-suite")
async def run_test_suite(request: Request):
    body = await request.json()
    suite_path = body.get("suite_path", "")
    max_tokens = body.get("max_tokens", 512)
    project_id = body.get("project_id") or body.get("pid") or ""

    # QABUG-007 / QABUG-011: auto-load merged model when none is loaded.
    if inference_engine.model is None:
        if not project_id:
            return JSONResponse(
                {"error": "No model loaded"},
                status_code=400,
            )
        merged = _resolve_merged_model(project_id)
        if not merged:
            return JSONResponse(
                {
                    "error": (
                        "no completed training run found for this project; "
                        "run training + merge first"
                    )
                },
                status_code=400,
            )
        try:
            inference_engine.load(merged)
        except Exception as e:  # noqa: BLE001
            _log.warning("auto-load failed for %s: %s", merged, e)
            return JSONResponse(
                {"error": f"auto-load failed: {e}"},
                status_code=400,
            )

    cases = load_test_suite(suite_path)
    results = run_suite(inference_engine, cases, max_tokens=max_tokens)
    apply_heuristic_judging(results)
    scores = score_results(results)
    # v2 schema (CaseResult): case_name / model_answer / verdict (pass|partial|fail|"") /
    # judge / judge_model / judge_reasoning / time_ms / error.
    return {
        "results": [
            {
                "name": r.case_name,
                "category": r.category,
                "question": r.question,
                "correct_answer": r.correct_answer,
                "response": r.model_answer,
                "model_answer": r.model_answer,
                "passed": r.verdict == "pass",
                "verdict": r.verdict,
                "judge": r.judge,
                "judge_model": r.judge_model,
                "judge_reasoning": r.judge_reasoning,
                "time_ms": r.time_ms,
                "error": r.error,
            }
            for r in results
        ],
        "scores": scores,
    }
