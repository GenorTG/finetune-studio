"""Testing tab — run test suites, view results."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio.testing.suite import (
    apply_heuristic_judging,
    load_test_suite,
    run_suite,
    score_results,
)
from finetune_studio.webui.app import inference_engine
from finetune_studio.webui.engine_guard import ENGINE_LOCK
from finetune_studio.webui.live_sse import sse_comment, sse_data, sse_response
from finetune_studio.webui.testing_models import resolve_latest_merged_model

router = APIRouter()
_log = logging.getLogger(__name__)


def _resolve_merged_model(pid: str) -> str | None:
    """Return path to the most-recent completed run's merged model, or None."""
    return resolve_latest_merged_model(pid)


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
        async with ENGINE_LOCK:
            await asyncio.to_thread(inference_engine.load, model_path, **kwargs)
        return {"status": "loaded", "model": model_path}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.post("/unload")
async def unload_model():
    inference_engine.unload()
    return {"status": "unloaded"}


def _testing_status_payload() -> dict:
    return {
        "loaded": inference_engine.model is not None,
        "model_path": inference_engine.model_path,
        "is_gguf": inference_engine.is_gguf,
    }


@router.get("/status")
async def model_status():
    """One-shot testing/inference load status (fallback for non-SSE clients)."""
    return _testing_status_payload()


@router.get("/events")
async def testing_events():
    """SSE stream of testing model-load status for live suite progress."""
    async def gen():
        last: str | None = None
        while True:
            payload = _testing_status_payload()
            key = f"{payload['loaded']}|{payload.get('model_path') or ''}"
            if key != last:
                last = key
                yield sse_data(payload)
            else:
                yield sse_comment()
            await asyncio.sleep(1.0)

    return sse_response(gen())


@router.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    async with ENGINE_LOCK:
        response = await asyncio.to_thread(
            inference_engine.generate, messages,
            max_tokens=max_tokens, temperature=temperature,
        )
    return {"response": response}


@router.post("/run-suite")
async def run_test_suite(request: Request):
    body = await request.json()
    suite_path = body.get("suite_path", "")
    max_tokens = body.get("max_tokens", 512)
    project_id = body.get("project_id") or body.get("pid") or ""
    override_path = (
        body.get("model_path") or body.get("path") or ""
    ).strip()

    cases = load_test_suite(suite_path)

    def _blocking():
        err = _ensure_model_loaded(str(project_id or ""), override_path)
        if err is not None:
            return err
        return run_suite(inference_engine, cases, max_tokens=max_tokens)

    async with ENGINE_LOCK:
        result = await asyncio.to_thread(_blocking)
    from fastapi.responses import JSONResponse as _JSONResponse
    if isinstance(result, _JSONResponse):
        return result
    results = result
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
                "scoring_method": r.scoring_method,
                "validity": r.validity,
                "source_id": r.source_id,
                "chunk_idx": r.chunk_idx,
                "keywords": list(r.keywords),
                "transcript": list(r.transcript),
                "time_ms": r.time_ms,
                "error": r.error,
            }
            for r in results
        ],
        "scores": scores,
        "model_path": inference_engine.model_path,
    }


@router.post("/run-rag-suite")
async def run_rag_test_suite(request: Request):
    """Run a Q&A suite with PortableRAG retrieval grounding.

    Body keys: ``project_id``/``pid``, ``suite_path``, ``model_path``/``path``,
    ``corpus_path`` (defaults to the project's PortableRAG corpus), ``top_k``,
    ``max_tokens``, ``temperature``.

    Blocking model + RAG work runs in ``asyncio.to_thread`` so the event loop
    stays responsive. Response includes transcripts, retrieval hits, context,
    corpus/model paths, scores, and retrieval recall metrics.
    """
    body = await request.json()
    suite_path = str(body.get("suite_path") or "").strip()
    if not suite_path:
        return JSONResponse({"error": "suite_path required"}, status_code=400)

    project_id = str(body.get("project_id") or body.get("pid") or "").strip()
    requested_run_id = str(body.get("run_id") or "").strip()
    override_path = str(body.get("model_path") or body.get("path") or "").strip()
    corpus_path = str(body.get("corpus_path") or "").strip()
    top_k = int(body.get("top_k", 5))
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.3))

    load_err = _ensure_model_loaded(project_id, override_path)
    if load_err is not None:
        return load_err

    from finetune_studio.testing.rag_suite import run_rag_suite_evaluation

    def _blocking() -> dict[str, object]:
        report = run_rag_suite_evaluation(
            inference_engine,
            suite_path=suite_path,
            corpus_path=corpus_path,
            project_id=project_id,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        payload = report.as_api_dict()
        benchmark = _persist_rag_report(
            project_id=project_id,
            requested_run_id=requested_run_id,
            suite_path=suite_path,
            report=payload,
        )
        payload["benchmark"] = benchmark
        payload["benchmark_id"] = benchmark["id"]
        return payload

    try:
        return await asyncio.to_thread(_blocking)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (TypeError, ValueError, OSError) as e:
        _log.warning("run-rag-suite failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001
        _log.exception("run-rag-suite unexpected failure")
        return JSONResponse({"error": str(e)}, status_code=500)


def _persist_rag_report(
    *,
    project_id: str,
    requested_run_id: str,
    suite_path: str,
    report: dict[str, object],
) -> dict:
    """Persist a grounded report without discarding transcript provenance."""
    from finetune_studio import db

    run_id = requested_run_id if requested_run_id and db.get_run(requested_run_id) else ""
    model_path = str(report.get("model_path") or "")
    if not run_id and project_id:
        for run in db.list_runs(project_id):
            output_path = str(run.get("output_path") or "")
            if run.get("status") == "done" and output_path and model_path.startswith(output_path):
                run_id = str(run["id"])
                break
    if not run_id:
        placeholder = db.create_run(
            project_id=project_id,
            name=f"RAG evaluation · {suite_path.rsplit('/', 1)[-1]}",
            base_model=model_path,
            data_path=suite_path,
        )
        run_id = placeholder["id"]
        db.update_run(
            run_id,
            status="done",
            output_path=model_path,
            notes="Evaluation-only RAG benchmark; no training performed",
        )

    results = list(report.get("results") or [])
    scores = dict(report.get("scores") or {})
    scores["retrieval"] = dict(report.get("retrieval") or {})
    scores["corpus_path"] = report.get("corpus_path", "")
    cases = []
    for result in results:
        result_dict = dict(result)
        cases.append({
            "name": result_dict.get("name", ""),
            "category": result_dict.get("category", "general"),
            "question": result_dict.get("question", ""),
            "correct_answer": result_dict.get("correct_answer", ""),
            "model_answer": result_dict.get("model_answer", ""),
            "transcript": result_dict.get("transcript", []),
            "judge": result_dict.get("judge", "none"),
            "judge_model": result_dict.get("judge_model", ""),
            "verdict": result_dict.get("verdict", ""),
            "judge_reasoning": result_dict.get("judge_reasoning", ""),
            "scored_at": time.time(),
            "scoring_method": result_dict.get("scoring_method", ""),
            "validity": result_dict.get("validity", ""),
            "error": result_dict.get("error", ""),
            "judge_input": {
                "retrieval_hit": result_dict.get("retrieval_hit", False),
                "retrieval_hits": result_dict.get("retrieval_hits", []),
                "context_text": result_dict.get("context_text", ""),
            },
            "source_id": result_dict.get("source_id", ""),
            "chunk_idx": result_dict.get("chunk_idx", 0),
        })
    return db.create_benchmark(
        run_id,
        f"rag · {suite_path.rsplit('/', 1)[-1]}",
        scores,
        time_ms=round(sum(float(dict(r).get("time_ms") or 0) for r in results)),
        cases=cases,
        model_path=model_path,
    )


def _ensure_model_loaded(project_id: str, override_path: str) -> JSONResponse | None:
    """Load override or latest merged model; return error response or None."""
    if override_path:
        try:
            if inference_engine.model_path != override_path:
                inference_engine.load(override_path)
        except Exception as e:  # noqa: BLE001
            _log.warning("override load failed for %s: %s", override_path, e)
            return JSONResponse(
                {"error": f"auto-load failed: {e}"},
                status_code=400,
            )
        return None
    if inference_engine.model is not None:
        return None
    if not project_id:
        return JSONResponse({"error": "No model loaded"}, status_code=400)
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
    return None


@router.get("/projects/{pid}/training-datasets")
async def list_training_datasets_for_eval(pid: str):
    """List project datasets that can be used for training-data evaluation."""
    from finetune_studio.db import datasets as datasets_db

    rows = datasets_db.list_datasets(pid)
    return {
        "datasets": [
            {
                "id": d["id"],
                "name": d.get("name"),
                "source": d.get("source"),
                "qa_count": d.get("qa_count"),
                "data_path": d.get("data_path"),
            }
            for d in rows
        ],
        "leakage_warning": (
            "Evaluating the training set measures memorization / leakage risk, "
            "not held-out generalization."
        ),
    }


@router.post("/evaluate-training")
async def evaluate_training_dataset(request: Request):
    """Run heuristic evaluation against a project's approved training dataset.

    Results are labeled ``eval_kind=training_leakage`` — high scores reflect
    in-distribution recall, not generalization.
    """
    body = await request.json()
    project_id = str(body.get("project_id") or body.get("pid") or "").strip()
    if not project_id:
        return JSONResponse({"error": "project_id required"}, status_code=400)
    dataset_id = (body.get("dataset_id") or "").strip() or None
    max_cases = int(body.get("max_cases", 200))
    max_tokens = int(body.get("max_tokens", 512))
    override_path = (body.get("model_path") or body.get("path") or "").strip()

    from finetune_studio.testing.training_eval import (
        build_heldout_eval,
        build_training_eval,
        suite_label_for_training_eval,
    )

    try:
        eval_kind = str(body.get("eval_kind") or "training_leakage")
        builder = build_heldout_eval if eval_kind == "heldout" else build_training_eval
        cases, meta = builder(project_id, dataset_id=dataset_id, max_cases=max_cases)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    load_err = _ensure_model_loaded(project_id, override_path)
    if load_err is not None:
        return load_err

    results = run_suite(inference_engine, cases, max_tokens=max_tokens)
    apply_heuristic_judging(results)
    scores = score_results(results)
    scores = {
        **scores,
        "eval_kind": meta.eval_kind,
        "leakage_warning": meta.leakage_warning,
    }
    return {
        "suite_name": suite_label_for_training_eval(meta),
        "eval": meta.as_dict(),
        "scores": scores,
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
                "scoring_method": r.scoring_method,
                "validity": r.validity,
                "source_id": r.source_id,
                "chunk_idx": r.chunk_idx,
                "keywords": list(r.keywords),
                "transcript": list(r.transcript),
                "time_ms": r.time_ms,
                "error": r.error,
            }
            for r in results
        ],
        "model_path": inference_engine.model_path,
    }
