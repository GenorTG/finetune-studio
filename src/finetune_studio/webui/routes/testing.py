"""Testing tab — run test suites (raw transcripts), judge them afterwards, review by hand.

The flow is deliberately two separate steps (Genor's design):

1. **Run** (``POST .../run-suite``, ``.../run-rag-suite``, ``.../evaluate-training``) asks the model every question
   and saves the raw transcripts — question, model answer, answer key — case by case. It never scores.
2. **Judge** (``POST .../runs/{bid}/judge``) is a separate job: any provider row reads each saved case and gives
   pass / partial / fail with reasoning. A human can override any verdict (``PUT .../cases/{cid}/verdict``).
   Settings can make step 2 start by itself after step 1 (``auto_judge``, off by default).

Both steps are background jobs that persist their progress on the run row (see ``webui/testing_jobs.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from finetune_studio import db
from finetune_studio.db import judgements as jdb
from finetune_studio.testing.judge import JudgeUnavailable
from finetune_studio.testing.scoring import rescore_benchmark
from finetune_studio.testing.suite import load_test_suite
from finetune_studio.webui import testing_jobs
from finetune_studio.webui.app import inference_engine
from finetune_studio.webui.engine_guard import ENGINE_LOCK
from finetune_studio.webui.live_sse import sse_comment, sse_data, sse_response
from finetune_studio.webui.testing_models import resolve_latest_merged_model

router = APIRouter()
_log = logging.getLogger(__name__)


async def _json_object(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


def _resolve_merged_model(pid: str) -> str | None:
    """Return path to the most-recent completed run's merged model, or None."""
    return resolve_latest_merged_model(pid)


# ── Engine load / status (unchanged) ─────────────────────────────────────────


@router.post("/load")
async def load_model(request: Request):
    """Load a model into the global inference engine.

    Accepts ``{"model_path": "..."}`` or ``{"path": "..."}`` (same keys as
    ``/api/models/load`` / chat-v2) so UI callers don't get ``No model_path``.
    """
    body = await _json_object(request)
    model_path = body.get("model_path") or body.get("path") or ""
    if not model_path:
        return JSONResponse({"error": "No model_path"}, status_code=400)
    try:
        from finetune_studio.models.llama_loader import resolve_loader_overrides
        kwargs = resolve_loader_overrides(body, caller="testing/load", model_path=model_path)
        if "max_seq_length" in body:
            kwargs["max_seq_length"] = int(body["max_seq_length"])
        if "load_in_4bit" in body:
            kwargs["load_in_4bit"] = bool(body["load_in_4bit"])
        # Free whatever ModelManager (data-prep's helper, /api/providers/*)
        # has resident before loading into inference_engine — otherwise both
        # sit in VRAM simultaneously until someone happens to click Unload.
        from finetune_studio.models.manager import get_manager
        await asyncio.to_thread(get_manager().unload)
        async with ENGINE_LOCK:
            await asyncio.to_thread(inference_engine.load, model_path, **kwargs)
        return {"status": "loaded", "model": model_path}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


def _job_progress() -> list[dict[str, Any]]:
    """Live progress of every active job (read from the run rows, which are the single source of truth)."""
    out = []
    for job in testing_jobs.active_jobs():
        row = db.get_benchmark(job["benchmark_id"]) or {}
        out.append({**job, "status": row.get("status", ""), "progress_done": row.get("progress_done", 0),
                    "progress_total": row.get("progress_total", 0), "judge_status": row.get("judge_status", ""),
                    "judge_done": row.get("judge_done", 0), "judge_total": row.get("judge_total", 0)})
    return out


def _testing_status_payload() -> dict:
    return {
        "loaded": inference_engine.model is not None,
        "model_path": inference_engine.model_path,
        "is_gguf": inference_engine.is_gguf,
        "jobs": _job_progress(),
    }


@router.get("/status")
async def model_status():
    """One-shot testing/inference load status (fallback for non-SSE clients)."""
    return _testing_status_payload()


@router.get("/events")
async def testing_events():
    """SSE stream of testing model-load status and active test/judge jobs."""
    async def gen():
        last: str | None = None
        while True:
            payload = _testing_status_payload()
            key = f"{payload['loaded']}|{payload.get('model_path') or ''}|{json.dumps(payload['jobs'], sort_keys=True)}"
            if key != last:
                last = key
                yield sse_data(payload)
            else:
                yield sse_comment()
            await asyncio.sleep(1.0)

    return sse_response(gen())


# ── Step 1: run (raw transcripts, no verdicts) ───────────────────────────────


def _project_or_404(pid: str) -> None:
    if not db.get_project(pid):
        raise HTTPException(status_code=404, detail="project not found")


BASE_MODEL_CHOICE = "__base__"   # the Testing page's "untrained base model" entry


def _target_model(project_id: str, override: str) -> str:
    """The model the run will use: the explicit choice, the project's untrained base, else the latest merged export.

    Never "whatever happens to be loaded": the helper or another page's model may be resident.
    """
    if override == BASE_MODEL_CHOICE:
        from finetune_studio.db.runs import backfill_project_base_model

        base = backfill_project_base_model(project_id) or str((db.get_project(project_id) or {}).get("base_model") or "")
        if not base.strip():
            raise ValueError("this project has no base model recorded yet; train a run or set one under project Settings")
        return base.strip()
    if override:
        return override
    merged = _resolve_merged_model(project_id)
    if merged:
        return merged
    raise ValueError(
        "no model to test: pick one in the Model list, or run training + merge/export first",
    )


def _started(row: dict[str, Any]) -> JSONResponse:
    return JSONResponse({"ok": True, "benchmark_id": row["id"], "benchmark": row}, status_code=202)


def _busy(exc: testing_jobs.TestingBusy) -> JSONResponse:
    return JSONResponse({"error": str(exc), "active": exc.active}, status_code=409)


def _sampling(body: dict) -> tuple[int, float]:
    try:
        return int(body.get("max_tokens", 512)), float(body.get("temperature", 0.3))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_tokens and temperature must be numbers") from None


def _judge_opts(body: dict) -> tuple[bool | None, str]:
    auto = body.get("auto_judge")
    if auto is not None and not isinstance(auto, bool):
        raise HTTPException(status_code=400, detail="auto_judge must be true or false")
    return auto, str(body.get("judge_provider_id") or "").strip()


async def _start(spec: testing_jobs.RunSpec) -> JSONResponse:
    try:
        return _started(await testing_jobs.start_run_job(spec))
    except testing_jobs.TestingBusy as exc:
        return _busy(exc)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@router.post("/run-suite")
async def run_test_suite(request: Request):
    """Start a test run of a suite. Returns 202 + ``benchmark_id``; poll ``GET .../runs/{id}``."""
    body = await _json_object(request)
    suite_path = str(body.get("suite_path") or "").strip()
    project_id = str(body.get("project_id") or body.get("pid") or "").strip()
    if not suite_path:
        return JSONResponse({"error": "suite_path required"}, status_code=400)
    if not project_id:
        return JSONResponse({"error": "project_id required"}, status_code=400)
    _project_or_404(project_id)
    try:
        cases = load_test_suite(suite_path)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (TypeError, ValueError, OSError) as e:
        return JSONResponse({"error": f"invalid suite: {e}"}, status_code=400)
    if not cases:
        return JSONResponse({"error": "suite has no usable cases"}, status_code=400)
    max_tokens, temperature = _sampling(body)
    auto, judge_pid = _judge_opts(body)
    try:
        model = _target_model(project_id, str(body.get("model_path") or body.get("path") or "").strip())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return await _start(testing_jobs.RunSpec(
        project_id=project_id, kind="suite", suite_name=suite_path.rsplit("/", 1)[-1], cases=cases,
        model_path=model, run_id=str(body.get("run_id") or "").strip(), max_tokens=max_tokens,
        temperature=temperature, config={"suite_path": suite_path}, auto_judge=auto, judge_provider_id=judge_pid,
    ))


@router.post("/run-rag-suite")
async def run_rag_test_suite(request: Request):
    """Start a RAG-grounded test run.

    Body keys: ``project_id``/``pid``, ``suite_path``, ``model_path``/``path``, ``corpus_path`` (defaults to the
    project's PortableRAG corpus), ``top_k``, ``max_tokens``, ``temperature``, ``max_context_chars`` (default:
    sized from the loaded model's window). The model answers only from the retrieved chunks; retrieval provenance
    is stored with each case.
    """
    body = await _json_object(request)
    suite_path = str(body.get("suite_path") or "").strip()
    project_id = str(body.get("project_id") or body.get("pid") or "").strip()
    if not suite_path:
        return JSONResponse({"error": "suite_path required"}, status_code=400)
    if not project_id:
        return JSONResponse({"error": "project_id required"}, status_code=400)
    _project_or_404(project_id)
    try:
        top_k = int(body.get("top_k", 5))
        max_chars = int(body["max_context_chars"]) if body.get("max_context_chars") else None
    except (TypeError, ValueError):
        return JSONResponse({"error": "top_k and max_context_chars must be integers"}, status_code=400)
    max_tokens, temperature = _sampling(body)
    corpus_path = str(body.get("corpus_path") or "").strip()
    try:
        cases = load_test_suite(suite_path)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (TypeError, ValueError, OSError) as e:
        return JSONResponse({"error": f"invalid suite: {e}"}, status_code=400)
    if not cases:
        return JSONResponse({"error": "suite has no usable cases"}, status_code=400)
    from finetune_studio.testing.rag_suite import resolve_corpus_path

    try:  # fail now, not minutes later inside the job
        resolve_corpus_path(project_id, corpus_path)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    auto, judge_pid = _judge_opts(body)
    try:
        model = _target_model(project_id, str(body.get("model_path") or body.get("path") or "").strip())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return await _start(testing_jobs.RunSpec(
        project_id=project_id, kind="rag", suite_name=f"rag · {suite_path.rsplit('/', 1)[-1]}", cases=cases,
        model_path=model, run_id=str(body.get("run_id") or "").strip(), max_tokens=max_tokens,
        temperature=temperature, rag={"top_k": top_k, "corpus_path": corpus_path, "max_context_chars": max_chars},
        config={"suite_path": suite_path}, auto_judge=auto, judge_provider_id=judge_pid,
    ))


@router.post("/evaluate-training")
async def evaluate_training_dataset(request: Request):
    """Start a run over questions built from a project's dataset (held-out slice or full training set).

    ``eval_kind=training_leakage`` results measure recall of the training rows, not generalisation.
    """
    body = await _json_object(request)
    project_id = str(body.get("project_id") or body.get("pid") or "").strip()
    if not project_id:
        return JSONResponse({"error": "project_id required"}, status_code=400)
    _project_or_404(project_id)
    dataset_id = (body.get("dataset_id") or "").strip() or None
    try:
        max_cases = int(body.get("max_cases", 200))
    except (TypeError, ValueError):
        return JSONResponse({"error": "max_cases and max_tokens must be integers"}, status_code=400)
    try:
        max_tokens, temperature = _sampling(body)
    except HTTPException:
        return JSONResponse({"error": "max_cases and max_tokens must be integers"}, status_code=400)

    from finetune_studio.testing.training_eval import (
        build_heldout_eval,
        build_training_eval,
        suite_label_for_training_eval,
    )

    try:
        eval_kind = str(body.get("eval_kind") or "training_leakage")
        builder = build_heldout_eval if eval_kind == "heldout" else build_training_eval
        cases, meta = await asyncio.to_thread(builder, project_id, dataset_id=dataset_id, max_cases=max_cases)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    auto, judge_pid = _judge_opts(body)
    try:
        model = _target_model(project_id, str(body.get("model_path") or body.get("path") or "").strip())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return await _start(testing_jobs.RunSpec(
        project_id=project_id, kind="heldout" if meta.eval_kind == "heldout" else "training_leakage",
        suite_name=suite_label_for_training_eval(meta), cases=cases, model_path=model,
        run_id=str(body.get("run_id") or "").strip(), max_tokens=max_tokens, temperature=temperature,
        config={"eval": meta.as_dict()},
        extra_scores={"eval_kind": meta.eval_kind, "leakage_warning": meta.leakage_warning,
                      "dataset_id": getattr(meta, "dataset_id", ""), "dataset_name": getattr(meta, "dataset_name", "")},
        auto_judge=auto, judge_provider_id=judge_pid,
    ))


# ── Runs: list / detail / cases / export ─────────────────────────────────────


def _run_for_project(pid: str, bid: str) -> dict[str, Any]:
    bench = db.get_benchmark(bid)
    run = db.get_run(str((bench or {}).get("run_id") or "")) if bench else None
    if not bench or not run or run.get("project_id") != pid:
        raise HTTPException(status_code=404, detail="test run not found")
    return bench


def _run_summary(bench: dict[str, Any]) -> dict[str, Any]:
    """The run row as the UI wants it: no key material, scores flattened to what the list shows."""
    scores = bench.get("scores") if isinstance(bench.get("scores"), dict) else {}
    return {
        "id": bench["id"], "suite_name": bench.get("suite_name", ""), "kind": bench.get("kind", "suite"),
        "scoring": bench.get("scoring", "judge"), "status": bench.get("status", "done"),
        "model_path": bench.get("model_path", ""), "ran_at": bench.get("ran_at"), "time_ms": bench.get("time_ms", 0),
        "error": bench.get("error", ""), "progress_done": bench.get("progress_done", 0),
        "progress_total": bench.get("progress_total", 0), "judge_status": bench.get("judge_status", ""),
        "judge_provider_id": bench.get("judge_provider_id", ""), "judge_model": bench.get("judge_model", ""),
        "judge_done": bench.get("judge_done", 0), "judge_total": bench.get("judge_total", 0),
        "judge_error": bench.get("judge_error", ""), "scores": scores, "config": bench.get("config", {}),
        "active": [j["kind"] for j in testing_jobs.active_jobs() if j["benchmark_id"] == bench["id"]],
    }


@router.get("/projects/{pid}/runs")
async def list_test_runs(pid: str, limit: int = 50):
    """Saved test runs of a project, newest first (live progress included)."""
    _project_or_404(pid)
    rows = await asyncio.to_thread(db.list_benchmarks_for_project, pid, limit=max(1, min(limit, 200)))
    return [_run_summary(r) for r in rows]


@router.get("/projects/{pid}/runs/{bid}")
async def get_test_run(pid: str, bid: str):
    """One run: status, progress of both jobs, scores, and judge-vs-human agreement."""
    bench = _run_for_project(pid, bid)
    out = _run_summary(bench)
    out["agreement"] = await asyncio.to_thread(jdb.agreement, bid)
    return out


_CASE_LIST_DROP = ("transcript",)


def _case_payload(case: dict[str, Any], judgements: list[dict[str, Any]], *, full: bool) -> dict[str, Any]:
    out = {k: v for k, v in case.items() if full or k not in _CASE_LIST_DROP}
    ji = dict(out.get("judge_input") or {})
    if not full:
        ji.pop("context_text", None)
        ji.pop("retrieval_hits", None)
    out["judge_input"] = ji
    out["judgements"] = judgements
    return out


@router.get("/projects/{pid}/runs/{bid}/cases")
async def list_test_cases(pid: str, bid: str):
    """Every saved case with its judgements (transcripts and retrieved context excluded: fetch one case for those)."""
    _run_for_project(pid, bid)

    def load() -> list[dict[str, Any]]:
        by_case: dict[str, list[dict[str, Any]]] = {}
        for j in jdb.list_judgements(benchmark_id=bid):
            by_case.setdefault(j["case_id"], []).append(j)
        return [_case_payload(c, by_case.get(c["id"], []), full=False) for c in db.list_cases(bid)]

    return await asyncio.to_thread(load)


@router.get("/projects/{pid}/runs/{bid}/cases/{cid}")
async def get_test_case(pid: str, bid: str, cid: str):
    """One case in full: transcript, retrieved context and every judgement."""
    _run_for_project(pid, bid)
    case = db.get_case(cid)
    if not case or case.get("benchmark_id") != bid:
        raise HTTPException(status_code=404, detail="case not found")
    return _case_payload(case, jdb.list_judgements(case_id=cid), full=True)


@router.get("/projects/{pid}/runs/{bid}/export")
async def export_test_run(pid: str, bid: str, fmt: str = "json"):
    """The raw transcripts as a file: question, answer key, model answer, transcript, retrieval and all judgements."""
    bench = _run_for_project(pid, bid)

    def build() -> list[dict[str, Any]]:
        by_case: dict[str, list[dict[str, Any]]] = {}
        for j in jdb.list_judgements(benchmark_id=bid):
            by_case.setdefault(j["case_id"], []).append(j)
        return [_case_payload(c, by_case.get(c["id"], []), full=True) for c in db.list_cases(bid)]

    cases = await asyncio.to_thread(build)
    stem = f"test-run-{bid}"
    if fmt == "jsonl":
        text = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
        return Response(text, media_type="application/x-ndjson",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.jsonl"'})
    payload = {"run": _run_summary(bench), "cases": cases}
    return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})


@router.delete("/projects/{pid}/runs/{bid}")
async def delete_test_run(pid: str, bid: str):
    """Delete a saved run with its cases and judgements."""
    _run_for_project(pid, bid)
    if testing_jobs.is_active(bid):
        return JSONResponse({"error": "stop the running job first"}, status_code=409)

    def drop() -> None:
        with db.cursor() as c:
            c.execute("DELETE FROM case_judgements WHERE benchmark_id = ?", (bid,))
            c.execute("DELETE FROM benchmark_cases WHERE benchmark_id = ?", (bid,))
            c.execute("DELETE FROM benchmark_runs WHERE id = ?", (bid,))

    await asyncio.to_thread(drop)
    return {"ok": True}


# ── Step 2: judge (separate job) and human review ────────────────────────────


@router.post("/projects/{pid}/runs/{bid}/judge")
async def judge_test_run(pid: str, bid: str, request: Request):
    """Start judging a saved run. Body: ``provider_id`` ('' = configured default), ``only_unjudged`` (default true)."""
    _run_for_project(pid, bid)
    body = await _json_object(request)
    only = body.get("only_unjudged", True)
    if not isinstance(only, bool):
        raise HTTPException(status_code=400, detail="only_unjudged must be true or false")
    try:
        row = await testing_jobs.start_judge_job(bid, str(body.get("provider_id") or "").strip(), only_unjudged=only)
    except testing_jobs.TestingBusy as exc:
        return _busy(exc)
    except JudgeUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, "benchmark": _run_summary(row)}, status_code=202)


@router.post("/projects/{pid}/runs/{bid}/cancel")
async def cancel_test_job(pid: str, bid: str):
    """Stop the running test or judge job of this run at the next case boundary (saved cases are kept)."""
    _run_for_project(pid, bid)
    if not testing_jobs.cancel(bid):
        return JSONResponse({"error": "nothing is running for this test run"}, status_code=409)
    return {"ok": True}


@router.put("/projects/{pid}/runs/{bid}/cases/{cid}/verdict")
async def set_human_verdict(pid: str, bid: str, cid: str, request: Request):
    """A person's verdict. ``verdict`` is pass | partial | fail, or '' to retract it (the AI verdict applies again)."""
    _run_for_project(pid, bid)
    case = db.get_case(cid)
    if not case or case.get("benchmark_id") != bid:
        raise HTTPException(status_code=404, detail="case not found")
    body = await _json_object(request)
    verdict = str(body.get("verdict") if body.get("verdict") is not None else "").strip().lower()
    if verdict not in ("", *jdb.VERDICTS):
        raise HTTPException(status_code=400, detail=f"verdict must be one of {', '.join(jdb.VERDICTS)} or empty")
    note = str(body.get("reasoning") or "").strip()

    def write() -> dict[str, Any]:
        jdb.add_judgement(cid, kind="human", verdict=verdict, reasoning=note or ("retracted" if not verdict else "human verdict"),
                          judge_model="human")
        scores = rescore_benchmark(bid)
        return {"ok": True, "scores": scores, "case": _case_payload(db.get_case(cid) or {}, jdb.list_judgements(case_id=cid), full=False),
                "agreement": jdb.agreement(bid)}

    result = await asyncio.to_thread(write)
    result["at"] = time.time()
    return result
