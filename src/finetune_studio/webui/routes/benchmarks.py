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


def _parse_sample_knobs(body: dict[str, Any]) -> tuple[int | None, bool, int, str]:
    """Parse num_samples / full_run / seed / order from a run request body."""
    full_run = bool(body.get("full_run", False))
    seed = int(body.get("seed", DEFAULT_SEED))
    order_raw = str(body.get("order") or "dataset").strip().lower()
    order = order_raw if order_raw in {"dataset", "seeded_shuffle"} else "dataset"
    if full_run:
        return None, True, seed, order
    if "num_samples" not in body or body.get("num_samples") in (None, "", "full"):
        return DEFAULT_SAMPLE_LIMIT, False, seed, order
    num = int(body["num_samples"])
    return num, False, seed, order

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
    """Prefer merged/ under output_path when present."""
    target_model = str(run.get("output_path") or "").strip()
    merged_candidate = os.path.join(target_model, "merged")
    if os.path.isdir(merged_candidate) and os.path.isfile(
        os.path.join(merged_candidate, "config.json")
    ):
        return merged_candidate
    return target_model


def _unload_global_inference() -> None:
    """Unload the UI global InferenceEngine if it holds a model."""
    try:
        from finetune_studio.webui.app import inference_engine as _global_ie

        if _global_ie is not None and getattr(_global_ie, "model", None) is not None:
            _global_ie.unload()
    except Exception:  # noqa: BLE001, S110
        pass


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


async def _execute_benchmark(
    *,
    rid: str,
    suite_name: str,
    suite_path: str,
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
            elif judge_mode == "heuristic":
                apply_heuristic_judging(results)
            elif judge_mode in ("ai", "local"):
                _log.warning(
                    "judge_mode=%s not applied during run; falling back to heuristic",
                    judge_mode,
                )
                apply_heuristic_judging(results)
            else:
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

@router.get("/runs/{rid}")
async def get_benchmark_run(rid: str) -> dict[str, Any]:
    """Get a single benchmark run."""
    return db.get_benchmark(rid) or {"error": "not found"}


@router.get("/suites")
async def list_suites(project_id: str | None = None) -> list[dict[str, Any]]:
    """List available benchmark suites (files that exist + optional auto-suites)."""
    return _discover_suites(project_id)


@router.get("/projects/{pid}/runs")
async def list_runs_with_benchmarks(pid: str) -> list[dict[str, Any]]:
    """List training runs for a project, augmented with latest benchmark score."""
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
    judge_mode = body.get("judge_mode", "heuristic")
    max_tokens = int(body.get("max_tokens", 512))
    num_samples, full_run, seed, order = _parse_sample_knobs(body)

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
            suite_path=str(suite_path),
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
    judge_mode = body.get("judge_mode", "heuristic")
    max_tokens = int(body.get("max_tokens", 512))
    num_samples, full_run, seed, order = _parse_sample_knobs(body)

    project = db.get_project(pid)
    if not project:
        return JSONResponse({"error": "project not found"}, status_code=404)

    target_model = (project.get("base_model") or "").strip()
    if not target_model:
        return JSONResponse(
            {"error": "project has no base_model configured"},
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
    runs = db.list_runs(pid)
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
            suite_path=str(suite_path),
            judge_mode=str(judge_mode),
            max_tokens=max_tokens,
            target_model=target_model,
            cases=cases,
            real_meta=real_meta,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.get("/projects/{pid}/runs/{rid}/history")
async def run_history(pid: str, rid: str) -> list[dict[str, Any]] | dict[str, str]:
    """List all benchmarks for a specific run."""
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
        c.execute("DELETE FROM benchmark_runs WHERE run_id = ?", (rid,))
        c.execute("DELETE FROM training_runs WHERE id = ?", (rid,))
    return {"ok": True}


@router.delete("/projects/{pid}/benchmarks/{bid}")
async def delete_benchmark(pid: str, bid: str) -> dict[str, bool]:
    """Delete a specific benchmark result."""
    with db.cursor() as c:
        c.execute(
            "DELETE FROM benchmark_runs WHERE id = ? AND project_id = ?",
            (bid, pid),
        )
    return {"ok": True}


@router.post("/projects/{pid}/benchmarks/{bid}/judge", response_model=None)
async def judge_benchmark(pid: str, bid: str, request: Request) -> dict[str, Any] | JSONResponse:
    """Run AI/human judge over all cases in a benchmark."""
    body = await request.json()
    from finetune_studio.testing.judge import DEFAULT_JUDGE_KEY

    judge_mode = body.get("judge_mode") or ("ai" if DEFAULT_JUDGE_KEY else "heuristic")
    judge_model = body.get("judge_model", "")

    benchmark = db.get_benchmark(bid)
    if not benchmark:
        return JSONResponse({"error": "benchmark not found"}, status_code=404)

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
        cases_updated = db.list_cases(bid)
        from finetune_studio.testing.suite import CaseResult, score_results

        rebuilt = []
        for c in cases_updated:
            rebuilt.append(CaseResult(
                case_name=c.get("name", ""),
                category=c.get("category", ""),
                question=c.get("question", ""),
                correct_answer=c.get("correct_answer", ""),
                model_answer=c.get("model_answer", ""),
                transcript=c.get("transcript", ""),
                verdict=c.get("verdict", ""),
                time_ms=c.get("time_ms", 0),
            ))
        new_scores = score_results(rebuilt)
        db.update_benchmark_scores(bid, new_scores)
        return {
            "ok": True,
            "judged": updated,
            "judge_mode": "heuristic",
            "scores": new_scores,
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
                model=judge_model or None,
            )
            db.update_case(
                case["id"],
                judge="ai",
                judge_model=judge_model or "gpt-4o-mini",
                verdict=verdict,
                judge_reasoning=reasoning,
                scored_at=time.time(),
            )
            updated += 1
        return {"ok": True, "judged": updated}

    if judge_mode == "local":
        run = db.get_run(benchmark["run_id"])
        model_path = judge_model or (run.get("base_model", "") if run else "")
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
        return {"ok": True, "judged": updated}

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


@router.get("/projects/{pid}/benchmarks/{bid}/cases")
async def list_benchmark_cases(pid: str, bid: str) -> list[dict[str, Any]]:
    """List all cases + judge verdicts for a benchmark."""
    return db.list_cases(bid)


@router.get("/projects/{pid}/benchmarks/{bid}/audit", response_model=None)
async def audit_benchmark(pid: str, bid: str) -> dict[str, Any] | JSONResponse:
    """Return every persisted transcript plus an independent score recomputation."""
    benchmark = db.get_benchmark(bid)
    if not benchmark:
        return JSONResponse({"error": "benchmark not found"}, status_code=404)
    from finetune_studio.testing.audit import recompute_cases

    cases = db.list_cases(bid)
    return {"benchmark": benchmark, "case_count": len(cases), "cases": cases,
            "independent_audit": recompute_cases(cases)}


@router.post("/projects/{pid}/benchmarks/{bid}/cases/{cid}/verdict")
async def set_verdict(
    pid: str, bid: str, cid: str, request: Request
) -> dict[str, bool]:
    """Human overrides/sets a verdict."""
    body = await request.json()
    db.update_case(
        cid,
        verdict=body.get("verdict", ""),
        judge="human",
        judge_reasoning=body.get("reasoning", "human override"),
        scored_at=time.time(),
    )
    return {"ok": True}


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
    max_cases = int(body.get("max_cases", 200))
    max_tokens = int(body.get("max_tokens", 512))
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
        suite_path=meta.dataset_path,
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
