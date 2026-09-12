"""Benchmarks tab — run suites, view scores, compare runs."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Request

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


# ── Endpoints ─────────────────────────────────────────────────────────────

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

    run = db.get_run(rid)
    if not run:
        return {"error": "run not found"}
    if run["project_id"] != pid:
        return {"error": "run does not belong to project"}

    target_model = run.get("output_path") or run.get("base_model")
    if not target_model:
        return {"error": "run has no model to benchmark"}

    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import load_test_suite, run_suite, score_results

    if not suite_path:
        return {"error": "suite_path required"}

    engine = InferenceEngine()
    try:
        engine.load(target_model)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load failed: {e}"}

    try:
        cases = load_test_suite(suite_path)
        t0 = time.time()
        results = run_suite(engine, cases)
        scores = score_results(results)
        dt_ms = int((time.time() - t0) * 1000)
        benchmark = db.create_benchmark(rid, suite_name, scores, dt_ms)
        return {
            "benchmark": benchmark,
            "results": [
                {"name": r.test_name, "passed": r.passed, "time_ms": r.time_ms,
                 "response": r.response[:300]}
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


@router.get("/projects/{pid}/compare")
async def compare_runs(pid: str, run_a: str = "", run_b: str = ""):
    """Side-by-side comparison of latest benchmarks from two runs."""
    if not run_a or not run_b:
        return {"error": "run_a and run_b required"}

    a = _latest_benchmark(run_a)
    b = _latest_benchmark(run_b)
    if not a:
        return {"error": f"no benchmarks for run {run_a}"}
    if not b:
        return {"error": f"no benchmarks for run {run_b}"}

    a_scores = a.get("scores") or {}
    b_scores = b.get("scores") or {}

    # Build deltas for common keys
    deltas = {}
    all_keys = sorted(set(list(a_scores.keys()) + list(b_scores.keys())))
    for k in all_keys:
        av = a_scores.get(k, 0)
        bv = b_scores.get(k, 0)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diff = bv - av
            pct = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
            deltas[k] = {"run_a": av, "run_b": bv, "delta": pct}
        else:
            deltas[k] = {"run_a": av, "run_b": bv, "delta": "—"}

    return {
        "run_a": {"id": run_a, "benchmark": a},
        "run_b": {"id": run_b, "benchmark": b},
        "deltas": deltas,
    }
