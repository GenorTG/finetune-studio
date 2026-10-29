"""Fine-tune several model families on the Quillmoor training rows through the live app (Training page's own call).

    .venv/bin/python tests/corpus/quillmoor/bench/finetune.py --corpus .tmp/quillmoor/s --project agent-test-quillmoor-s \
        --models Qwen__Qwen3.5-4B:q8_0 unsloth__Llama-3.2-3B-Instruct:q8_0

The training rows are uploaded once as a project dataset (POST .../datasets/upload). Each model is one
POST /api/training/start (SFT LoRA, merged and exported to GGUF by the same job), followed on
GET /api/projects/{pid}/runs/{rid} until it ends and project-scoped /api/training/status for live progress. Same
settings for every family; results/finetune/<model>.json records the run id, time, losses, GGUF path, and which facts
the trainer held out for validation (the app keeps 10 %).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT,
    check,
    client,
    gpu_serialized,
    post_when_free,
    project_by_name,
    write_json,
)

HF = Path.home() / ".finetune-studio" / "hf_models"
SETTINGS = {"lora_rank": 32, "lora_alpha": 64, "learning_rate": 2e-4, "num_epochs": 4, "batch_size": 8,
            "gradient_accumulation_steps": 2, "max_seq_length": 256, "warmup_steps": 20, "logging_steps": 10,
            "save_steps": 100000, "bf16": True, "merge_on_save": True, "export_gguf": True}
ACTIVE_RUN_STATUSES = frozenset({
    "created", "pending", "queued", "starting", "preparing", "loading", "running", "training", "saving",
})


def find_adoptable_run(runs: list[dict], model_path: str, data_path: str) -> dict | None:
    """Only adopt an unfinished matching run; terminal runs are new benchmark attempts, not resumable work."""
    data_name = Path(data_path).name
    for run in reversed(runs):
        if (run.get("base_model") == model_path
                and (run.get("data_path") or "").endswith(data_name)
                and run.get("status") in ACTIVE_RUN_STATUSES):
            return run
    return None


def training_result_ok(status: str, error: str, final_loss: float | None, ggufs: list[Path],
                       expected_quant: str | None = None) -> bool:
    """Reject failed runs, missing requested GGUFs, and the NaN-to-zero loss fixed in app PR #41."""
    if status not in ("done", "completed") or error.strip() or final_loss is None:
        return False
    try:
        loss = float(final_loss)
    except (TypeError, ValueError):
        return False
    artifact_names: set[str] = set()
    for path in ggufs:
        try:
            if path.is_file() and path.stat().st_size > 0:
                artifact_names.add(path.name.casefold())
        except OSError:
            continue
    has_artifact = bool(artifact_names)
    if expected_quant:
        normalized_quant = expected_quant.strip().casefold().replace("-", "_").replace(".", "_")
        has_artifact = f"model-{normalized_quant}.gguf" in artifact_names
    return math.isfinite(loss) and loss > 0 and has_artifact


def training_metric_fields(metrics: dict[str, Any], dataset_rows: int) -> dict[str, int | float | None]:
    """Persist actual trainer progress so the report survives project cleanup."""
    def metric_int(name: str) -> int | None:
        value = metrics.get(name)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def metric_float(name: str) -> float | None:
        value = metrics.get(name)
        try:
            parsed = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        return parsed if parsed is not None and math.isfinite(parsed) and parsed >= 0 else None

    return {
        "dataset_rows": dataset_rows,
        "optimizer_steps": metric_int("current_step"),
        "planned_steps": metric_int("total_steps"),
        "training_runtime_seconds": metric_float("elapsed"),
    }


def live_training_metrics(run_status: str, project_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Use the live status snapshot for progress; the persisted run row stores metrics only at completion."""
    if run_status not in ACTIVE_RUN_STATUSES or snapshot.get("project_id") != project_id:
        return {}
    if snapshot.get("status") not in ACTIVE_RUN_STATUSES:
        return {}
    return {
        "current_step": snapshot.get("current_step", snapshot.get("step")),
        "total_steps": snapshot.get("total_steps"),
        "loss": snapshot.get("loss"),
        "elapsed": snapshot.get("elapsed"),
        "eta": snapshot.get("eta"),
        "message": snapshot.get("message"),
    }


def rehydrate_run_result(existing: dict[str, Any], run: dict[str, Any], *, dataset_rows: int,
                         root: Path = ROOT) -> dict[str, Any]:
    """Rebuild persisted metrics and GGUF details from the live run record, including a previously mis-polled run."""
    output_path = run.get("output_path")
    outdir = root / output_path if output_path else None
    ggufs = sorted(str(path) for path in outdir.rglob("*.gguf")) if outdir and outdir.exists() else []
    expected_quant = str(existing.get("quant") or "")
    if expected_quant:
        normalized_quant = expected_quant.strip().casefold().replace("-", "_").replace(".", "_")
        preferred_name = f"model-{normalized_quant}.gguf"
        ggufs.sort(key=lambda path: (Path(path).name.casefold() != preferred_name, path))
    selected_gguf_size = None
    if ggufs:
        selected_path = Path(ggufs[0])
        if selected_path.is_file():
            selected_gguf_size = selected_path.stat().st_size
    metrics = run.get("metrics") or {}
    final_loss = metrics.get("final_loss")
    if final_loss is None:
        final_loss = run.get("final_loss")
    eval_loss = metrics.get("eval_loss")
    if eval_loss is None:
        eval_loss = run.get("eval_loss")
    seconds = metrics.get("duration")
    if seconds is None and run.get("started_at") and run.get("finished_at"):
        seconds = run["finished_at"] - run["started_at"]
    if seconds is None:
        seconds = existing.get("seconds") or 0
    status = str(run.get("status") or "")
    error = str(run.get("error") or "")
    result = dict(existing)
    result.update({
        "run_id": run.get("id") or existing.get("run_id"),
        "status": status,
        "error": error,
        "ok": training_result_ok(
            status, error, final_loss, [Path(path) for path in ggufs], expected_quant or None
        ),
        "output_path": str(outdir) if outdir else "",
        "gguf": ggufs,
        "gguf_size_bytes": selected_gguf_size,
        "final_loss": final_loss,
        "eval_loss": eval_loss,
        "seconds": round(seconds),
        **training_metric_fields(metrics, dataset_rows),
    })
    return result


def dataset_id(c, pid: str, corpus: Path, out: Path, train_file: str = "train.jsonl") -> tuple[str, str]:
    stem = Path(train_file).stem
    cache = out / ("dataset.json" if stem == "train" else f"dataset-{stem}.json")
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["id"], d["data_path"]
    path = corpus / "train" / train_file
    upload_name = "quillmoor-train.jsonl" if stem == "train" else f"quillmoor-{stem}.jsonl"
    d = check(c.post(f"/api/projects/{pid}/datasets/upload", files={"file": (upload_name, path.read_bytes())}))
    write_json(cache, d)
    return d["id"], d["data_path"]


def held_out_questions(data_path: str) -> list[str]:
    """The user questions of the rows the SFT trainer keeps for validation (same split code, same seed)."""
    from finetune_studio.training.data import sft_validation_rows
    rows = [json.loads(x) for x in Path(data_path).read_text().splitlines() if x.strip()]
    out = []
    for r in sft_validation_rows(rows):
        msgs = r.get("messages") or []
        out += [m["content"] for m in msgs if m.get("role") == "user"][:1]
    return out


def adopt(c, pid: str, model_path: str, data_path: str) -> dict | None:
    """A run of this base model on this dataset already in the project (its driver was stopped): wait for it."""
    runs = check(c.get(f"/api/projects/{pid}/runs"))
    return find_adoptable_run(runs, model_path, data_path)


@gpu_serialized
def run_one(c, pid: str, did: str, name: str, quant: str, data_path: str = "", settings: dict | None = None) -> dict:
    settings = settings or SETTINGS
    model_path = name if name.startswith("/") else str(HF / name)
    hold = ROOT / ".tmp" / "bench" / "gpu-hold"
    existing = adopt(c, pid, model_path, data_path) if data_path else None
    while hold.exists() and existing is None:
        time.sleep(20)
    body = {"project_id": pid, "dataset_id": did, "model_path": model_path, "training_mode": "sft",
            "gguf_quants": [quant], **settings}
    t0 = time.time()
    if existing:
        run_id = existing["id"]
        print(f"[{name}] adopting run {run_id} ({existing.get('status')})", flush=True)
    else:
        try:
            started = post_when_free(c, "/api/training/start", json=body)
        except RuntimeError as e:
            return {"model": name, "ok": False, "error": str(e)}
        run_id = started["run_id"]
        steps = started.get("steps")
        suffix = f" ({steps} steps)" if isinstance(steps, int) and steps > 0 else ""
        print(f"[{name}] run {run_id} started{suffix}", flush=True)
    last: dict = {}
    while True:
        try:
            last = check(c.get(f"/api/projects/{pid}/runs/{run_id}"))
        except Exception as e:  # noqa: BLE001 - the web may restart; keep waiting
            print(f"  status error {e}", flush=True)
            time.sleep(15)
            continue
        st = last.get("status")
        metrics = last.get("metrics") or {}
        if st not in ACTIVE_RUN_STATUSES:
            break
        try:
            snapshot = check(c.get("/api/training/status", params={"project_id": pid}))
            live = live_training_metrics(st, pid, snapshot)
        except Exception:  # noqa: BLE001 - persisted run state remains the completion authority
            live = {}
        progress = {**metrics, **live}
        if int(time.time() - t0) % 120 < 15:
            current = progress.get("current_step")
            total = progress.get("total_steps")
            loss = progress.get("loss")
            detail = f" step {current}/{total}" if current is not None and total is not None else ""
            detail += f" loss {loss:.4f}" if isinstance(loss, (int, float)) else ""
            eta = progress.get("eta")
            detail += f" eta {eta:.0f}s" if isinstance(eta, (int, float)) and eta > 0 else ""
            print(f"  {st}{detail}", flush=True)
        time.sleep(15)
    run = check(c.get(f"/api/projects/{pid}/runs/{run_id}"))
    dataset_rows = sum(1 for line in Path(data_path).read_text().splitlines() if line.strip())
    return rehydrate_run_result(
        {"model": name, "quant": quant, "settings": settings, "adopted": existing is not None,
         "seconds": time.time() - t0},
        run,
        dataset_rows=dataset_rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="hf_models dir name[:gguf quant]")
    ap.add_argument("--train-file", default="train.jsonl", help="file in <corpus>/train/")
    ap.add_argument("--tag", default="", help="suffix of the result files (another recipe for the same model)")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--refresh-metrics", action="store_true",
                    help="refresh saved run-step metadata without starting or retrying training")
    a = ap.parse_args()
    settings = dict(SETTINGS)
    if a.epochs:
        settings["num_epochs"] = a.epochs
    if a.rank:
        settings["lora_rank"], settings["lora_alpha"] = a.rank, 2 * a.rank
    corpus = Path(a.corpus)
    out = corpus / "results" / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    c = client(timeout=600)
    pid = project_by_name(c, a.project)["id"]
    did, data_path = dataset_id(c, pid, corpus, out, a.train_file)
    held = held_out_questions(data_path)
    write_json(out / ("held_out_questions.json" if not a.tag else f"held_out_questions-{a.tag}.json"), held)
    print(f"dataset {did}: {len(held)} rows held out for validation by the trainer", flush=True)
    if a.refresh_metrics:
        dataset_rows = sum(1 for line in Path(data_path).read_text().splitlines() if line.strip())
        for spec in a.models:
            name, _, _quant = spec.partition(":")
            result_path = out / (f"{Path(name).name}.json" if not a.tag else f"{Path(name).name}__{a.tag}.json")
            if not result_path.exists():
                print(f"[{name}] no saved result; skipped", flush=True)
                continue
            result = json.loads(result_path.read_text())
            run_id = result.get("run_id")
            if not run_id:
                print(f"[{name}] no run id; skipped", flush=True)
                continue
            run = check(c.get(f"/api/projects/{pid}/runs/{run_id}"))
            result = rehydrate_run_result(result, run, dataset_rows=dataset_rows)
            result["train_file"] = a.train_file
            write_json(result_path, result)
            print(
                f"[{name}] {result.get('status')} {result.get('optimizer_steps')}/"
                f"{result.get('planned_steps')} steps; ok={result.get('ok')}",
                flush=True,
            )
        return 0
    for spec in a.models:
        name, _, quant = spec.partition(":")
        rf = out / (f"{Path(name).name}.json" if not a.tag else f"{Path(name).name}__{a.tag}.json")
        if rf.exists():
            previous = json.loads(rf.read_text())
            prior_ggufs = [Path(path) for path in previous.get("gguf", []) if Path(path).is_file()]
            if previous.get("ok") and training_result_ok(
                previous.get("status", ""), previous.get("error") or "", previous.get("final_loss"),
                prior_ggufs, previous.get("quant"),
            ):
                continue
        res = run_one(c, pid, did, name, quant or "q8_0", data_path, settings)
        res["tag"] = a.tag
        res["train_file"] = a.train_file
        write_json(rf, res)
        print(json.dumps({k: res.get(k) for k in ("model", "ok", "status", "seconds", "final_loss", "eval_loss", "error")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
