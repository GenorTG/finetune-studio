"""Benchmarks tab — run suites, view scores, compare runs."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.benchmarks.real_benchmarks import (
    DEFAULT_SAMPLE_LIMIT,
    DEFAULT_SEED,
    RealBenchmarkSuite,
    is_real_suite_path,
    parse_real_suite_path,
)
from finetune_studio.benchmarks.suite_defs import (
    discover_suites,
    is_selectable_suite,
)
from finetune_studio.testing.suite import BenchmarkCase, load_test_suite

router = APIRouter()
_log = logging.getLogger(__name__)


def _discover_suites(project_id: str | None = None) -> list[dict[str, Any]]:
    """Return selectable suites (real HF + synthetic + local JSON + auto)."""
    return discover_suites(project_id)


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` rather than raising
    ``HTTPException`` (see the 404s below), so the guard matches that style.
    Called before any DB read, filesystem walk or stream so a bad pid never
    starts work.
    """
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return None


def _benchmark_for_project(bid: str, pid: str) -> dict[str, Any] | JSONResponse:
    """Load a benchmark only when its parent run belongs to ``pid``."""
    benchmark = db.get_benchmark(bid)
    if not benchmark:
        return JSONResponse({"error": "benchmark not found"}, status_code=404)
    run = db.get_run(str(benchmark.get("run_id") or ""))
    if not run or run.get("project_id") != pid:
        return JSONResponse({"error": "benchmark not found"}, status_code=404)
    return benchmark


VERDICTS = ("pass", "partial", "fail")


def _rescore_benchmark(bid: str) -> dict[str, Any]:
    """Recompute a benchmark's aggregate scores from its stored case verdicts.

    Every verdict change (re-judge or human override) must go through this, or
    the headline score shown in "Recent scores" goes stale. Non-score metadata
    already in the scores (``eval_kind``, ``leakage_warning``, dataset fields
    written by training-set evals) is kept.
    """
    from finetune_studio.testing.suite import CaseResult, score_results

    results = [
        CaseResult(
            case_name=c.get("case_name") or c.get("name") or "",
            category=c.get("category") or "",
            question=c.get("question") or "",
            correct_answer=c.get("correct_answer") or "",
            model_answer=c.get("model_answer") or "",
            verdict=c.get("verdict") or "",
            time_ms=c.get("time_ms") or 0,
        )
        for c in db.list_cases(bid)
    ]
    old = (db.get_benchmark(bid) or {}).get("scores") or {}
    scores = {**(old if isinstance(old, dict) else {}), **score_results(results)}
    db.update_benchmark_scores(bid, scores)
    return scores


def _int_field(body: dict[str, Any], key: str, default: int) -> int:
    """Read an integer field from a request body; ValueError names the key."""
    try:
        return int(body.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None


def _parse_sample_knobs(body: dict[str, Any]) -> tuple[int | None, bool, int, str]:
    """Parse num_samples / full_run / seed / order from a run request body.

    Raises ValueError (caller returns 400) on non-integer seed / num_samples.
    """
    # The UI sample picker sends num_samples="full" for "all cases".
    full_run = bool(body.get("full_run", False)) or body.get("num_samples") == "full"
    seed = _int_field(body, "seed", DEFAULT_SEED)
    order_raw = str(body.get("order") or "dataset").strip().lower()
    order = order_raw if order_raw in {"dataset", "seeded_shuffle"} else "dataset"
    if full_run:
        return None, True, seed, order
    if "num_samples" not in body or body.get("num_samples") in (None, ""):
        return DEFAULT_SAMPLE_LIMIT, False, seed, order
    return _int_field(body, "num_samples", DEFAULT_SAMPLE_LIMIT), False, seed, order


def _validate_suite_file(
    suite_path: str,
    *,
    project_id: str | None = None,
    require_selectable: bool = False,
    num_samples: int | None = DEFAULT_SAMPLE_LIMIT,
    full_run: bool = False,
    seed: int = DEFAULT_SEED,
    order: str = "dataset",
) -> tuple[list[BenchmarkCase] | None, dict[str, Any] | None, JSONResponse | None]:
    """Ensure suite_path is usable. Returns (cases, real_meta, error).

    Real ``real://`` suites load HuggingFace rows (injectable in tests via
    RealBenchmarkSuite). File suites load JSON as before.
    """
    if not suite_path or not str(suite_path).strip():
        return None, None, JSONResponse({"error": "suite_path required"}, status_code=400)
    if require_selectable and not is_selectable_suite(suite_path, project_id):
        return None, None, JSONResponse(
            {"error": f"suite not selectable: {suite_path}"},
            status_code=400,
        )

    family = parse_real_suite_path(suite_path)
    if family is not None:
        try:
            suite = RealBenchmarkSuite()
            cases, meta = suite.load_cases(
                family,
                num_samples=num_samples,
                full_run=full_run,
                seed=seed,
                order=order,  # type: ignore[arg-type]
            )
        except Exception as exc:  # noqa: BLE001
            return None, None, JSONResponse(
                {"error": f"real suite load failed: {exc}"},
                status_code=400,
            )
        if len(cases) < 1:
            return None, None, JSONResponse(
                {"error": "suite has no cases"},
                status_code=400,
            )
        return cases, meta.as_dict(), None

    path = Path(suite_path)
    if not path.is_file():
        return None, None, JSONResponse(
            {"error": f"suite not found: {suite_path}"},
            status_code=404,
        )
    try:
        cases = load_test_suite(str(path))
    except Exception as exc:  # noqa: BLE001
        return None, None, JSONResponse(
            {"error": f"suite parse failed: {exc}"},
            status_code=400,
        )
    if len(cases) < 1:
        return None, None, JSONResponse(
            {"error": "suite has no cases"},
            status_code=400,
        )
    return cases, None, None


def _run_is_benchmarkable(run: dict[str, Any]) -> bool:
    """True when the run finished successfully and has a trained artifact path."""
    status = str(run.get("status") or "")
    output = (run.get("output_path") or "").strip()
    return status == "done" and bool(output)


def _resolve_trained_target(run: dict[str, Any]) -> str:
    """Prefer merged/ under output_path, then the LoRA adapter/, else output_path.

    An unmerged run's output dir has no config.json, so loading it directly
    fails with an opaque transformers "Unrecognized model" error.
    """
    target_model = str(run.get("output_path") or "").strip()
    merged_candidate = os.path.join(target_model, "merged")
    if os.path.isdir(merged_candidate) and os.path.isfile(
        os.path.join(merged_candidate, "config.json")
    ):
        return merged_candidate
    adapter_candidate = os.path.join(target_model, "adapter")
    if os.path.isfile(os.path.join(adapter_candidate, "adapter_config.json")):
        return adapter_candidate
    return target_model


def _unload_global_inference() -> None:
    """Free VRAM before loading a judge model — both engines, not just one.

    Used to only unload the global InferenceEngine. A data-prep helper
    loaded via ModelManager stayed resident through the whole judge run,
    competing for VRAM with the judge model that was just asked to load.
    """
    from finetune_studio.models.llama_loader import unload_all_models
    unload_all_models()


def _latest_benchmark(run_id: str) -> dict | None:
    """Return the most recent benchmark for a run, or None."""
    bs = db.list_benchmarks(run_id)
    return bs[0] if bs else None


def _primary_score(scores: dict | None) -> float | None:
    """Pick a comparable numeric score from a benchmark scores dict."""
    if not scores or not isinstance(scores, dict):
        return None
    for key in ("pass_rate", "score", "accuracy", "overall"):
        val = scores.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    for val in scores.values():
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _suite_scores_for_run(run_id: str) -> dict[str, float | None]:
    """Latest primary score per suite_name for a training run."""
    out: dict[str, float | None] = {}
    for bench in db.list_benchmarks(run_id):  # already ran_at DESC
        suite = str(bench.get("suite_name") or "unknown")
        if suite in out:
            continue
        out[suite] = _primary_score(bench.get("scores"))
    return out


def _apply_configured_judge(results: list[Any], judge_mode: str) -> None:
    """Judge ``results`` in place with the AI/local judge from Settings.

    Cases the judge cannot verdict (no key, API error, load failure) are left
    unjudged so the caller's heuristic pass handles them — never silent.
    """
    from finetune_studio.testing.judge import judge_case_ai, judge_case_local
    from finetune_studio.webui.routes.settings import get_judge_config

    cfg = get_judge_config()
    todo = [r for r in results if not r.verdict and (r.model_answer or "").strip()]
    if not todo:
        return
    if judge_mode == "ai":
        if not cfg["api_key"]:
            _log.warning("judge_mode=ai but no judge API key is set; using heuristic")
            return
        for r in todo:
            verdict, reasoning, _conf = judge_case_ai(
                r.question, r.correct_answer, r.model_answer,
                model=cfg["model"], api_url=cfg["api_url"], api_key=cfg["api_key"],
            )
            if verdict:
                r.verdict, r.judge = verdict, "ai"
                r.judge_model, r.judge_reasoning = cfg["model"], reasoning
            else:
                _log.warning("AI judge gave no verdict (%s); using heuristic", reasoning)
        return
    # local: cfg["model"] must be a local model path
    model_path = cfg["model"]
    if not os.path.exists(os.path.expanduser(model_path)):
        _log.warning("judge_mode=local but judge model %r is not a local path; using heuristic", model_path)
        return
    from finetune_studio.testing.inference import InferenceEngine

    judge_engine = InferenceEngine()
    try:
        _unload_global_inference()
        judge_engine.load(os.path.expanduser(model_path))
        for r in todo:
            verdict, reasoning, _conf = judge_case_local(
                judge_engine, r.question, r.correct_answer, r.model_answer,
            )
            if verdict:
                r.verdict, r.judge = verdict, "local"
                r.judge_model, r.judge_reasoning = model_path, reasoning
    except Exception as exc:  # noqa: BLE001
        _log.warning("local judge failed (%s); using heuristic", exc)
    finally:
        judge_engine.unload()


async def _execute_benchmark(
    *,
    rid: str,
    suite_name: str,
    judge_mode: str,
    max_tokens: int,
    target_model: str,
    cases: list[BenchmarkCase],
    real_meta: dict[str, Any] | None = None,
) -> dict[str, Any] | JSONResponse:
    """Load model, run suite, judge, persist. Always unloads the bench engine."""
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import (
        apply_heuristic_judging,
        run_suite,
        score_results,
    )

    # Real MCQ/GSM8K prompts need short generations; keep caller max_tokens
    # but default cooler sampling for strict extraction.
    run_temperature = 0.0 if real_meta else 0.3
    if real_meta and real_meta.get("family") == "gsm8k":
        eff_max_tokens = max(max_tokens, 256)
    elif real_meta:
        eff_max_tokens = min(max_tokens, 32)
    else:
        eff_max_tokens = max_tokens

    def _blocking() -> dict[str, Any]:
        engine = InferenceEngine()
        try:
            _unload_global_inference()
            try:
                engine.load(target_model)
            except Exception as exc:  # noqa: BLE001
                return {"_load_error": str(exc)}

            t0 = time.time()
            results = run_suite(
                engine,
                cases,
                max_tokens=eff_max_tokens,
                temperature=run_temperature,
            )
            dt_ms = int((time.time() - t0) * 1000)

            if judge_mode == "none":
                pass
            elif judge_mode in ("ai", "local") and not real_meta:
                engine.unload()  # free VRAM before a local judge loads
                _apply_configured_judge(results, judge_mode)
                apply_heuristic_judging(results)  # fallback for unjudged cases
            else:
                # heuristic, and real MCQ/GSM8K suites (strict scoring is exact)
                apply_heuristic_judging(results)

            scores = score_results(results)
            # Persist the exact artifact used; merged and quantized exports can
            # produce materially different answers and must not be conflated.
            scores["model_path"] = target_model
            if real_meta:
                scores["is_real_benchmark"] = True
                scores["benchmark_metadata"] = real_meta
                scores["accuracy"] = scores.get("pass_rate")

            case_dicts: list[dict[str, Any]] = []
            for r in results:
                judge_input = {
                    "question": r.question,
                    "correct_answer": r.correct_answer,
                    "model_answer": r.model_answer,
                    "keywords": list(r.keywords),
                    "scoring_method": r.scoring_method,
                    "judge_mode": judge_mode,
                }
                case_dicts.append({
                    "name": r.case_name,
                    "category": r.category,
                    "question": r.question,
                    "correct_answer": r.correct_answer,
                    "model_answer": r.model_answer,
                    "transcript": r.transcript,
                    "judge": r.judge or "none",
                    "judge_model": r.judge_model,
                    "verdict": r.verdict,
                    "judge_reasoning": r.judge_reasoning,
                    "scored_at": time.time() if r.verdict else None,
                    "scoring_method": r.scoring_method,
                    "validity": r.validity,
                    "error": r.error,
                    "judge_input": judge_input,
                    "source_id": getattr(r, "source_id", ""),
                    "chunk_idx": getattr(r, "chunk_idx", 0),
                })

            benchmark = db.create_benchmark(
                rid, suite_name, scores, dt_ms, cases=case_dicts,
                model_path=target_model,
            )

            return {
                "benchmark": benchmark,
                "scores": scores,
                "results": [
                    {
                        "name": r.case_name,
                        "category": r.category,
                        "question": r.question,
                        "correct_answer": r.correct_answer,
                        "model_answer": r.model_answer[:500],
                        "transcript": r.transcript,
                        "time_ms": r.time_ms,
                        "verdict": r.verdict,
                        "judge": r.judge,
                        "judge_model": r.judge_model,
                        "judge_reasoning": r.judge_reasoning,
                        "scoring_method": r.scoring_method,
                        "validity": r.validity,
                    }
                    for r in results
                ],
            }
        finally:
            engine.unload()

    result = await asyncio.to_thread(_blocking)
    if "_load_error" in result:
        return JSONResponse({"error": f"load failed: {result['_load_error']}"}, status_code=500)
    return result

# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/suites")
async def list_suites(project_id: str | None = None) -> list[dict[str, Any]]:
    """List available benchmark suites (files that exist + optional auto-suites)."""
    return _discover_suites(project_id)


@router.get("/projects/{pid}/runs", response_model=None)
async def list_runs_with_benchmarks(pid: str) -> list[dict[str, Any]] | JSONResponse:
    """List training runs for a project, augmented with latest benchmark score."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    runs = db.list_runs(pid)
    out: list[dict[str, Any]] = []
    for run in runs:
        b = _latest_benchmark(run["id"])
        run["latest_benchmark"] = b
        out.append(run)
    return out


@router.post("/projects/{pid}/runs/{rid}/run", response_model=None)
async def run_benchmark(pid: str, rid: str, request: Request) -> dict[str, Any] | JSONResponse:
    """Run a benchmark suite against a run's trained output model."""
    body = await request.json()
    suite_name = body.get("suite_name", "default")
    suite_path = body.get("suite_path", "")
    from finetune_studio.webui.routes.settings import get_judge_config

    judge_mode = str(body.get("judge_mode") or get_judge_config()["mode"])
    try:
        max_tokens = _int_field(body, "max_tokens", 512)
        num_samples, full_run, seed, order = _parse_sample_knobs(body)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    run = db.get_run(rid)
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    if run["project_id"] != pid:
        return JSONResponse(
            {"error": "run does not belong to project"},
            status_code=400,
        )

    if not _run_is_benchmarkable(run):
        status = run.get("status") or "unknown"
        return JSONResponse(
            {
                "error": (
                    f"Run {rid} has no trained model (status: {status})"
                ),
            },
            status_code=409,
        )

    cases, real_meta, suite_err = _validate_suite_file(
        suite_path,
        num_samples=num_samples if is_real_suite_path(str(suite_path)) else DEFAULT_SAMPLE_LIMIT,
        full_run=full_run if is_real_suite_path(str(suite_path)) else False,
        seed=seed,
        order=order,
    )
    if suite_err is not None:
        return suite_err
    assert cases is not None

    requested_model = str(body.get("model_path") or "").strip()
    target_model = requested_model or _resolve_trained_target(run)
    if requested_model and not os.path.isabs(target_model):
        target_model = os.path.abspath(target_model)
    if requested_model and not os.path.exists(target_model):
        return JSONResponse(
            {"error": f"requested benchmark model not found: {target_model}"},
            status_code=404,
        )
    if not target_model:
        return JSONResponse(
            {"error": f"Run {rid} has no trained model (status: {run.get('status')})"},
            status_code=409,
        )

    try:
        return await _execute_benchmark(
            rid=rid,
            suite_name=str(suite_name),
            judge_mode=str(judge_mode),
            max_tokens=max_tokens,
            target_model=target_model,
            cases=cases,
            real_meta=real_meta,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/projects/{pid}/base/run", response_model=None)
async def run_benchmark_base(pid: str, request: Request) -> dict[str, Any] | JSONResponse:
    """Benchmark the project's untrained base model (explicit, never a silent fallback)."""
    body = await request.json()
    suite_name = body.get("suite_name", "default")
    suite_path = body.get("suite_path", "")
    from finetune_studio.webui.routes.settings import get_judge_config

    judge_mode = str(body.get("judge_mode") or get_judge_config()["mode"])
    try:
        max_tokens = _int_field(body, "max_tokens", 512)
        num_samples, full_run, seed, order = _parse_sample_knobs(body)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    from finetune_studio.db.runs import backfill_project_base_model
    backfill_project_base_model(pid)
    project = db.get_project(pid)
    if not project:
        return JSONResponse({"error": "project not found"}, status_code=404)

    target_model = (project.get("base_model") or "").strip()
    if not target_model:
        return JSONResponse(
            {"error": "This project has no base model yet. Train a run first, or set one under project Settings."},
            status_code=400,
        )

    cases, real_meta, suite_err = _validate_suite_file(
        suite_path,
        num_samples=num_samples if is_real_suite_path(str(suite_path)) else DEFAULT_SAMPLE_LIMIT,
        full_run=full_run if is_real_suite_path(str(suite_path)) else False,
        seed=seed,
        order=order,
    )
    if suite_err is not None:
        return suite_err
    assert cases is not None

    # Persist results against a synthetic "base" context: create or reuse a
    # placeholder run so create_benchmark has a run_id FK. Prefer an existing
    # run whose name marks it as the base-model probe.
    runs = db.list_runs(pid, include_base_probe=True)
    base_run = next(
        (r for r in runs if r.get("name") == "__base_model__"),
        None,
    )
    if base_run is None:
        base_run = db.create_run(
            pid,
            "__base_model__",
            base_model=target_model,
        )
        db.update_run(
            base_run["id"],
            status="done",
            output_path="",
            notes="Placeholder for explicit base-model (untrained) benchmarks",
        )

    try:
        return await _execute_benchmark(
            rid=base_run["id"],
            suite_name=str(suite_name),
            judge_mode=str(judge_mode),
            max_tokens=max_tokens,
            target_model=target_model,
            cases=cases,
            real_meta=real_meta,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.get("/projects/{pid}/runs/{rid}/history", response_model=None)
async def run_history(pid: str, rid: str) -> list[dict[str, Any]] | dict[str, str] | JSONResponse:
    """List all benchmarks for a specific run."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    run = db.get_run(rid)
    if not run or run["project_id"] != pid:
        return {"error": "not found"}
    return db.list_benchmarks(rid)


@router.delete("/projects/{pid}/runs/{rid}", response_model=None)
async def delete_run(pid: str, rid: str) -> dict[str, Any] | JSONResponse:
    """Delete a training run and its benchmark results."""
    run = db.get_run(rid)
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    if run["project_id"] != pid:
        return JSONResponse({"error": "project mismatch"}, status_code=403)
    with db.cursor() as c:
        c.execute("DELETE FROM benchmark_cases WHERE run_id = ?", (rid,))
        c.execute("DELETE FROM benchmark_runs WHERE run_id = ?", (rid,))
        c.execute("DELETE FROM training_runs WHERE id = ?", (rid,))
    return {"ok": True}


@router.delete("/projects/{pid}/benchmarks/{bid}", response_model=None)
async def delete_benchmark(pid: str, bid: str) -> dict[str, bool] | JSONResponse:
    """Delete a specific benchmark result and its cases.

    Ownership is checked through the parent run (``_benchmark_for_project``);
    ``benchmark_runs`` has no ``project_id`` column, which the old
    ``WHERE ... AND project_id = ?`` made a guaranteed 500.
    """
    benchmark = _benchmark_for_project(bid, pid)
    if isinstance(benchmark, JSONResponse):
        return benchmark
    with db.cursor() as c:
        c.execute("DELETE FROM benchmark_cases WHERE benchmark_id = ?", (bid,))
        c.execute("DELETE FROM benchmark_runs WHERE id = ?", (bid,))
    return {"ok": True}


@router.post("/projects/{pid}/benchmarks/{bid}/judge", response_model=None)
async def judge_benchmark(pid: str, bid: str, request: Request) -> dict[str, Any] | JSONResponse:
    """Run AI/human judge over all cases in a benchmark."""
    body = await request.json()
    from finetune_studio.webui.routes.settings import get_judge_config

    judge_cfg = get_judge_config()
    judge_mode = body.get("judge_mode") or judge_cfg["mode"]
    judge_model = body.get("judge_model", "")

    benchmark = _benchmark_for_project(bid, pid)
    if isinstance(benchmark, JSONResponse):
        return benchmark

    cases = db.list_cases(bid)
    if not cases:
        return JSONResponse({"error": "no cases in benchmark"}, status_code=400)

    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.judge import (
        judge_case_ai,
        judge_case_heuristic,
        judge_case_local,
    )

    if judge_mode == "heuristic":
        updated = 0
        for case in cases:
            if not case.get("model_answer"):
                continue
            verdict, reasoning, _confidence = judge_case_heuristic(
                question=case["question"],
                correct_answer=case["correct_answer"],
                model_answer=case["model_answer"],
            )
            db.update_case(
                case["id"],
                judge="heuristic",
                judge_model="heuristic",
                verdict=verdict,
                judge_reasoning=reasoning,
                scored_at=time.time(),
            )
            updated += 1
        return {
            "ok": True,
            "judged": updated,
            "judge_mode": "heuristic",
            "scores": _rescore_benchmark(bid),
        }

    if judge_mode == "ai":
        updated = 0
        for case in cases:
            if not case.get("model_answer"):
                continue
            verdict, reasoning, _confidence = judge_case_ai(
                question=case["question"],
                correct_answer=case["correct_answer"],
                model_answer=case["model_answer"],
                model=judge_model or judge_cfg["model"],
                api_url=judge_cfg["api_url"],
                api_key=judge_cfg["api_key"],
            )
            db.update_case(
                case["id"],
                judge="ai",
                judge_model=judge_model or judge_cfg["model"],
                verdict=verdict,
                judge_reasoning=reasoning,
                scored_at=time.time(),
            )
            updated += 1
        return {"ok": True, "judged": updated, "scores": _rescore_benchmark(bid)}

    if judge_mode == "local":
        run = db.get_run(benchmark["run_id"])
        model_path = judge_model or (
                judge_cfg["model"] if os.path.exists(os.path.expanduser(judge_cfg["model"])) else ""
            ) or (run.get("base_model", "") if run else "")
        if not model_path:
            return JSONResponse(
                {"error": "no model path for local judge"},
                status_code=400,
            )

        def _local_judge() -> int:
            judge_engine = InferenceEngine()
            try:
                _unload_global_inference()
                judge_engine.load(model_path)
                updated = 0
                for case in cases:
                    if not case.get("model_answer"):
                        continue
                    verdict, reasoning, _confidence = judge_case_local(
                        judge_engine,
                        question=case["question"],
                        correct_answer=case["correct_answer"],
                        model_answer=case["model_answer"],
                    )
                    db.update_case(
                        case["id"],
                        judge="local",
                        judge_model=model_path,
                        verdict=verdict,
                        judge_reasoning=reasoning,
                        scored_at=time.time(),
                    )
                    updated += 1
                return updated
            finally:
                judge_engine.unload()

        updated = await asyncio.to_thread(_local_judge)
        return {"ok": True, "judged": updated, "scores": _rescore_benchmark(bid)}

    if judge_mode == "secondary_local":
        model_path = str(judge_model or "").strip()
        if not model_path:
            return JSONResponse(
                {"error": "judge_model is required for secondary_local"},
                status_code=400,
            )

        def _secondary_judge() -> int:
            judge_engine = InferenceEngine()
            try:
                _unload_global_inference()
                judge_engine.load(model_path)
                updated = 0
                for case in cases:
                    if not case.get("model_answer"):
                        continue
                    verdict, reasoning, confidence = judge_case_local(
                        judge_engine,
                        question=case["question"],
                        correct_answer=case["correct_answer"],
                        model_answer=case["model_answer"],
                        transcript=case.get("transcript") or [],
                    )
                    judge_input = case.get("judge_input") or {}
                    judge_input["secondary_judge"] = {
                        "verdict": verdict,
                        "reasoning": reasoning,
                        "confidence": confidence,
                        "model": model_path,
                        "judged_at": time.time(),
                    }
                    db.update_case(case["id"], judge_input=judge_input)
                    updated += 1
                return updated
            finally:
                judge_engine.unload()

        updated = await asyncio.to_thread(_secondary_judge)
        return {
            "ok": True,
            "judged": updated,
            "judge_mode": "secondary_local",
            "judge_model": model_path,
            "authoritative_scores_unchanged": True,
        }

    return JSONResponse(
        {"error": f"unknown judge_mode: {judge_mode}"},
        status_code=400,
    )


@router.get("/projects/{pid}/benchmarks/{bid}/cases", response_model=None)
async def list_benchmark_cases(pid: str, bid: str) -> list[dict[str, Any]] | JSONResponse:
    """List all cases + judge verdicts for a benchmark."""
    benchmark = _benchmark_for_project(bid, pid)
    if isinstance(benchmark, JSONResponse):
        return benchmark
    return db.list_cases(bid)


@router.get("/projects/{pid}/benchmarks/{bid}/audit", response_model=None)
async def audit_benchmark(pid: str, bid: str) -> dict[str, Any] | JSONResponse:
    """Return every persisted transcript plus an independent score recomputation."""
    benchmark = _benchmark_for_project(bid, pid)
    if isinstance(benchmark, JSONResponse):
        return benchmark
    from finetune_studio.testing.audit import recompute_cases

    cases = db.list_cases(bid)
    return {"benchmark": benchmark, "case_count": len(cases), "cases": cases,
            "independent_audit": recompute_cases(cases)}


@router.post("/projects/{pid}/benchmarks/{bid}/cases/{cid}/verdict", response_model=None)
async def set_verdict(
    pid: str, bid: str, cid: str, request: Request
) -> dict[str, Any] | JSONResponse:
    """Human overrides/sets a verdict, then rescores the benchmark."""
    benchmark = _benchmark_for_project(bid, pid)
    if isinstance(benchmark, JSONResponse):
        return benchmark
    if not any(case.get("id") == cid for case in db.list_cases(bid)):
        return JSONResponse({"error": "case not found"}, status_code=404)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "request body must be valid JSON"}, status_code=400)
    verdict = body.get("verdict") if isinstance(body, dict) else None
    if verdict not in VERDICTS:
        return JSONResponse({"error": f"verdict must be one of {', '.join(VERDICTS)}"}, status_code=400)
    db.update_case(
        cid,
        verdict=verdict,
        judge="human",
        judge_reasoning=str(body.get("reasoning") or "human override"),
        scored_at=time.time(),
    )
    return {"ok": True, "scores": _rescore_benchmark(bid)}


@router.get("/projects/{pid}/compare", response_model=None)
async def compare_runs(pid: str, run_a: str = "", run_b: str = "") -> dict[str, Any] | JSONResponse:
    """Side-by-side per-suite score comparison of two training runs.

    Baseline is run_a; delta is run_b − run_a. Uses each run's latest
    benchmark per suite_name and the primary numeric score (pass_rate, etc.).
    """
    # The project is validated FIRST, like every other project route
    # (_project_or_404): otherwise a bad pid is reported as the unrelated
    # "run_a and run_b required", which sends the caller hunting for a
    # missing query param instead of the project that does not exist.
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)

    if not run_a or not run_b:
        return JSONResponse(
            {"error": "run_a and run_b required"},
            status_code=400,
        )

    run_a_row = db.get_run(run_a)
    run_b_row = db.get_run(run_b)
    if not run_a_row:
        return JSONResponse({"error": f"run not found: {run_a}"}, status_code=404)
    if not run_b_row:
        return JSONResponse({"error": f"run not found: {run_b}"}, status_code=404)
    if run_a_row.get("project_id") != pid or run_b_row.get("project_id") != pid:
        return JSONResponse(
            {"error": "runs do not belong to this project"},
            status_code=400,
        )

    a_by_suite = _suite_scores_for_run(run_a)
    b_by_suite = _suite_scores_for_run(run_b)
    if not a_by_suite and not b_by_suite:
        return JSONResponse(
            {"error": "no benchmarks for either run"},
            status_code=404,
        )

    suites: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, Any]] = {}
    for suite in sorted(set(a_by_suite) | set(b_by_suite)):
        av = a_by_suite.get(suite)
        bv = b_by_suite.get(suite)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diff = float(bv) - float(av)
            delta_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
            suites.append({
                "suite": suite,
                "run_a": float(av),
                "run_b": float(bv),
                "delta": diff,
            })
            deltas[suite] = {
                "run_a": float(av),
                "run_b": float(bv),
                "delta": delta_str,
            }
        else:
            suites.append({
                "suite": suite,
                "run_a": av,
                "run_b": bv,
                "delta": None,
            })
            deltas[suite] = {"run_a": av, "run_b": bv, "delta": "—"}

    a = _latest_benchmark(run_a)
    b = _latest_benchmark(run_b)
    return {
        "run_a": {"id": run_a, "name": run_a_row.get("name", ""), "benchmark": a},
        "run_b": {"id": run_b, "name": run_b_row.get("name", ""), "benchmark": b},
        "suites": suites,
        "deltas": deltas,
    }


@router.post("/projects/{pid}/runs/{rid}/evaluate-training", response_model=None)
async def evaluate_training_for_run(
    pid: str, rid: str, request: Request
) -> dict[str, Any] | JSONResponse:
    """Benchmark a trained run against the project's training dataset.

    Persists a benchmark row with ``eval_kind=training_leakage`` in scores and
    full per-case results (same table pattern as synthetic suites).
    """
    body = await request.json()
    dataset_id = (body.get("dataset_id") or "").strip() or None
    try:
        max_cases = _int_field(body, "max_cases", 200)
        max_tokens = _int_field(body, "max_tokens", 512)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    judge_mode = str(body.get("judge_mode", "heuristic"))

    run = db.get_run(rid)
    if not run:
        return JSONResponse({"error": "run not found"}, status_code=404)
    if run["project_id"] != pid:
        return JSONResponse(
            {"error": "run does not belong to project"},
            status_code=400,
        )
    if not _run_is_benchmarkable(run):
        status = run.get("status") or "unknown"
        return JSONResponse(
            {"error": f"Run {rid} has no trained model (status: {status})"},
            status_code=409,
        )

    from finetune_studio.testing.training_eval import (
        build_training_eval,
        suite_label_for_training_eval,
    )

    try:
        cases, meta = build_training_eval(
            pid, dataset_id=dataset_id, max_cases=max_cases
        )
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    target_model = _resolve_trained_target(run)
    result = await _execute_benchmark(
        rid=rid,
        suite_name=suite_label_for_training_eval(meta),
        judge_mode=judge_mode,
        max_tokens=max_tokens,
        target_model=target_model,
        cases=cases,
    )
    if isinstance(result, JSONResponse):
        return result

    scores = dict(result.get("scores") or {})
    scores["eval_kind"] = meta.eval_kind
    scores["leakage_warning"] = meta.leakage_warning
    scores["dataset_id"] = meta.dataset_id
    scores["dataset_name"] = meta.dataset_name
    bid = (result.get("benchmark") or {}).get("id")
    if bid:
        db.update_benchmark_scores(bid, scores)
        result["benchmark"] = db.get_benchmark(bid) or result.get("benchmark")
    result["scores"] = scores
    result["eval"] = meta.as_dict()
    return result
