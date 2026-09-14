"""Benchmarks tab — run suites, view scores, compare runs."""

from __future__ import annotations

import time
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio import db

router = APIRouter()

_KNOWN_SUITES = [
    {"name": "default", "path": "data/benchmarks/default.json", "description": "General-purpose benchmark"},
    {"name": "tool_calling", "path": "data/benchmarks/tool_calling.json", "description": "Tool-calling accuracy"},
    {"name": "chris_ai_v21", "path": "data/benchmarks/chris_ai_v21.json", "description": "Chris AI v21 suite"},
]


def _discover_suites() -> list[dict]:
    """Return known suites plus any .json files in data/benchmarks/."""
    found: dict[str, dict] = {s["name"]: s for s in _KNOWN_SUITES}
    bench_dir = Path("data/benchmarks")
    if bench_dir.is_dir():
        for f in bench_dir.glob("*.json"):
            name = f.stem
            if name not in found:
                found[name] = {"name": name, "path": str(f), "description": f"Discovered: {f.name}"}
    return sorted(found.values(), key=lambda s: s["name"])


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


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/runs/{rid}")
async def get_benchmark_run(rid: str):
    """Get a single benchmark run."""
    from finetune_studio import db
    return db.get_benchmark(rid) or {"error": "not found"}


@router.get("/suites")
async def list_suites():
    """List available benchmark suites."""
    return _discover_suites()


@router.get("/projects/{pid}/runs")
async def list_runs_with_benchmarks(pid: str):
    """List training runs for a project, augmented with latest benchmark score."""
    runs = db.list_runs(pid)
    out = []
    for run in runs:
        b = _latest_benchmark(run["id"])
        run["latest_benchmark"] = b
        out.append(run)
    return out


@router.post("/projects/{pid}/runs/{rid}/run")
async def run_benchmark(pid: str, rid: str, request: Request):
    """Run a benchmark suite against a run's output model."""
    body = await request.json()
    suite_name = body.get("suite_name", "default")
    suite_path = body.get("suite_path", "")
    judge_mode = body.get("judge_mode", "heuristic")  # heuristic | ai | local | none
    judge_model = body.get("judge_model", "")
    max_tokens = int(body.get("max_tokens", 512))

    run = db.get_run(rid)
    if not run:
        return {"error": "run not found"}
    if run["project_id"] != pid:
        return {"error": "run does not belong to project"}

    target_model = run.get("output_path") or run.get("base_model")
    if not target_model:
        return {"error": "run has no model to benchmark"}
    merged_candidate = os.path.join(target_model, "merged")
    if os.path.isdir(merged_candidate) and os.path.isfile(os.path.join(merged_candidate, "config.json")):
        target_model = merged_candidate

    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import load_test_suite, run_suite, score_results

    if not suite_path:
        return {"error": "suite_path required"}

    # Unload the global inference engine first — it may hold a model from the UI.
    try:
        from finetune_studio.webui.app import inference_engine as _global_ie
        if _global_ie is not None and getattr(_global_ie, "model", None) is not None:
            _global_ie.unload()
    except Exception:
        pass

    engine = InferenceEngine()
    try:
        engine.load(target_model)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load failed: {e}"}

    try:
        cases = load_test_suite(suite_path)
        t0 = time.time()
        results = run_suite(engine, cases, max_tokens=max_tokens)
        dt_ms = int((time.time() - t0) * 1000)

        # Build case dicts for DB
        case_dicts = []
        for r in results:
            case_dicts.append({
                "name": r.case_name,
                "category": r.category,
                "question": r.question,
                "correct_answer": r.correct_answer,
                "model_answer": r.model_answer,
                "transcript": r.transcript,
                "judge": "none",
            })

        scores = score_results(results)
        benchmark = db.create_benchmark(rid, suite_name, scores, dt_ms, cases=case_dicts)

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
                }
                for r in results
            ],
        }
    finally:
        engine.unload()


@router.get("/projects/{pid}/runs/{rid}/history")
async def run_history(pid: str, rid: str):
    """List all benchmarks for a specific run."""
    run = db.get_run(rid)
    if not run or run["project_id"] != pid:
        return {"error": "not found"}
    return db.list_benchmarks(rid)


@router.delete("/projects/{pid}/runs/{rid}")
async def delete_run(pid: str, rid: str):
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
async def delete_benchmark(pid: str, bid: str):
    """Delete a specific benchmark result."""
    with db.cursor() as c:
        c.execute("DELETE FROM benchmark_runs WHERE id = ? AND project_id = ?", (bid, pid))
    return {"ok": True}


@router.post("/projects/{pid}/benchmarks/{bid}/judge")
async def judge_benchmark(pid: str, bid: str, request: Request):
    """Run AI/human judge over all cases in a benchmark."""
    body = await request.json()
    # Default to the heuristic judge: it needs no API key and no model, so
    # scores never stay stuck at judged: 0 on a fresh install. Only prefer the
    # external AI judge when a key is actually configured.
    from finetune_studio.testing.judge import DEFAULT_JUDGE_KEY

    judge_mode = body.get("judge_mode") or ("ai" if DEFAULT_JUDGE_KEY else "heuristic")
    judge_model = body.get("judge_model", "")

    benchmark = db.get_benchmark(bid)
    if not benchmark:
        return {"error": "benchmark not found"}

    cases = db.list_cases(bid)
    if not cases:
        return {"error": "no cases in benchmark"}

    # Load the judge model
    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.judge import (
        judge_case_ai,
        judge_case_heuristic,
        judge_case_local,
    )

    # Heuristic judge — no API key, no model, always returns a verdict.
    if judge_mode == "heuristic":
        updated = 0
        for case in cases:
            if not case.get("model_answer"):
                continue
            verdict, reasoning, confidence = judge_case_heuristic(
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
        # Recalculate scores after judging
        cases_updated = db.list_cases(bid)
        from finetune_studio.testing.suite import score_results, CaseResult
        rebuilt = []
        for c in cases_updated:
            rebuilt.append(CaseResult(
                case_name=c.get('name', ''),
                category=c.get('category', ''),
                question=c.get('question', ''),
                correct_answer=c.get('correct_answer', ''),
                model_answer=c.get('model_answer', ''),
                transcript=c.get('transcript', ''),
                verdict=c.get('verdict', ''),
                time_ms=c.get('time_ms', 0),
            ))
        new_scores = score_results(rebuilt)
        db.update_benchmark_scores(bid, new_scores)
        return {"ok": True, "judged": updated, "judge_mode": "heuristic", "scores": new_scores}

    if judge_mode == "ai":
        # For AI judge via external API
        updated = 0
        for case in cases:
            if not case.get("model_answer"):
                continue
            verdict, reasoning, confidence = judge_case_ai(
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

    # For local judge — load the specified model
    if judge_mode == "local":
        # Unload the global inference engine first — it may hold a model from the UI.
        try:
            from finetune_studio.webui.app import inference_engine as _global_ie
            if _global_ie is not None and getattr(_global_ie, "model", None) is not None:
                _global_ie.unload()
        except Exception:
            pass

        judge_engine = InferenceEngine()
        try:
            # Use the base model as judge — an unbiased evaluator
            run = db.get_run(benchmark["run_id"])
            model_path = judge_model or (run.get("base_model", "") if run else "")
            if not model_path:
                return {"error": "no model path for local judge"}
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
            return {"ok": True, "judged": updated}
        finally:
            judge_engine.unload()

    return {"error": f"unknown judge_mode: {judge_mode}"}


@router.get("/projects/{pid}/benchmarks/{bid}/cases")
async def list_benchmark_cases(pid: str, bid: str):
    """List all cases + judge verdicts for a benchmark."""
    return db.list_cases(bid)


@router.post("/projects/{pid}/benchmarks/{bid}/cases/{cid}/verdict")
async def set_verdict(pid: str, bid: str, cid: str, request: Request):
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


@router.get("/projects/{pid}/compare")
async def compare_runs(pid: str, run_a: str = "", run_b: str = ""):
    """Side-by-side per-suite score comparison of two training runs.

    Baseline is run_a; delta is run_b − run_a. Uses each run's latest
    benchmark per suite_name and the primary numeric score (pass_rate, etc.).
    """
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

    suites: list[dict] = []
    deltas: dict[str, dict] = {}
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
            deltas[suite] = {"run_a": float(av), "run_b": float(bv), "delta": delta_str}
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
