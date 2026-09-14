"""Projects API — list/create/select Project, manage RAGs and Training Runs.

WHY THIS EXISTS
===============
The Studio is organised around Projects. Each Project:
  - has a base model + system prompt
  - owns N RAGs (legal, social, paperwork, ...)
  - owns N Training Runs (each with settings + metrics + benchmark results)
  - has one "production" Run that the Inference Chat loads

This route file covers all of that.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter, Request

from finetune_studio import db
from finetune_studio.rag.manager import RAGManager
from finetune_studio.webui.app import training_engine

log = logging.getLogger(__name__)

router = APIRouter()


# ── Projects ─────────────────────────────────────────────────────────────

@router.get("")
async def list_projects():
    return db.list_projects()


@router.post("")
async def create_project(request: Request):
    body = await request.json()
    p = db.create_project(
        name=body.get("name", "Untitled"),
        description=body.get("description", ""),
        base_model=body.get("base_model", ""),
        system_prompt=body.get("system_prompt", ""),
    )
    return p


@router.get("/{pid}")
async def get_project(pid: str):
    p = db.get_project(pid)
    if not p:
        return {"error": "not found"}
    p["rags"] = db.list_rags(pid)
    p["runs"] = db.list_runs(pid)
    # Attach model exports found on disk for each run.
    p["models"] = _scan_run_models(p["runs"])
    # Attach benchmark summaries to runs.
    for run in p["runs"]:
        run["benchmarks"] = db.list_benchmarks(run["id"])
    return p


@router.patch("/{pid}")
async def update_project(pid: str, request: Request):
    body = await request.json()
    return db.update_project(pid, **body)


@router.delete("/{pid}")
async def delete_project(pid: str):
    db.delete_project(pid)
    return {"ok": True}


@router.get("/{pid}/export")
async def export_project(pid: str, name: str = None, fmt: str = "tar.gz"):
    """Export a project as a self-contained archive."""
    import tarfile, io
    from pathlib import Path
    from fastapi.responses import StreamingResponse

    if not db.get_project(pid):
        return JSONResponse({"error": "not found"}, status_code=404)

    def stream():
        buf = io.BytesIO()
        mode = 'w:gz' if fmt == 'tar.gz' else 'w'
        with tarfile.open(fileobj=buf, mode=mode) as tar:
            data_dir = Path.home() / ".finetune-studio" / "projects" / pid
            if data_dir.exists():
                tar.add(str(data_dir), arcname=f"projects/{pid}")
            rag_dir = Path.home() / ".finetune-studio" / "rag_corpora" / pid
            if rag_dir.exists():
                tar.add(str(rag_dir), arcname=f"rag_corpora/{pid}")
            import json
            manifest = json.dumps({"project_id": pid, "exported_at": time.time(), "version": "1.0"}, indent=2)
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest.encode()))
        buf.seek(0)
        yield buf.read()

    filename = (name or f"project-{pid}") + "." + fmt
    media = "application/gzip" if fmt == "tar.gz" else "application/x-tar"
    return StreamingResponse(stream(), media_type=media,
        headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.post("/import")
async def import_project(request: Request):
    """Import a project from an uploaded archive."""
    import tarfile, io, json
    from pathlib import Path

    form = await request.form()
    file = form.get('file')
    if not file:
        return JSONResponse({"error": "no file"}, status_code=400)

    data = await file.read()
    buf = io.BytesIO(data)

    try:
        with tarfile.open(fileobj=buf, mode='r:*') as tar:
            # Read manifest
            try:
                member = tar.getmember('manifest.json')
                f = tar.extractfile(member)
                manifest = json.loads(f.read().decode())
            except (KeyError, json.JSONDecodeError):
                manifest = {"version": "1.0"}

            # Extract to temp dir first
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tar.extractall(tmpdir)

                # Find the projects dir
                src_projects = Path(tmpdir) / "projects"
                if not src_projects.exists():
                    return JSONResponse({"error": "invalid archive: no projects dir"}, status_code=400)

                # Import each project found
                results = []
                for proj_dir in src_projects.iterdir():
                    if not proj_dir.is_dir():
                        continue
                    old_id = proj_dir.name
                    # Create new project entry
                    proj_name = manifest.get('project_name', f'Imported {old_id[:8]}')
                    new_proj = db.create_project(
                        name=proj_name,
                        description=manifest.get('description', f'Imported from archive'),
                        base_model=manifest.get('base_model', ''),
                        system_prompt=manifest.get('system_prompt', 'You are a helpful assistant.'),
                        tags=manifest.get('tags', 'imported'),
                    )
                    new_id = new_proj['id']
                    # Copy files
                    dest_dir = Path.home() / ".finetune-studio" / "projects" / new_id
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copytree(str(proj_dir), str(dest_dir), dirs_exist_ok=True)
                    results.append({"old_id": old_id, "new_id": new_id, "name": proj_name})

                return {"ok": True, "imported": results, "count": len(results)}
    except tarfile.TarError as e:
        return JSONResponse({"error": f"invalid archive: {e}"}, status_code=400)


@router.post("/{pid}/promote")
async def promote_run(pid: str, request: Request):
    """Set a Training Run as the Project's production model."""
    body = await request.json()
    run_id = body.get("run_id", "")
    if not db.get_run(run_id):
        return {"error": "run not found"}
    db.update_project(pid, production_run=run_id)
    return db.get_project(pid)


# ── RAGs ─────────────────────────────────────────────────────────────────

@router.get("/{pid}/rags")
async def list_rags(pid: str):
    return db.list_rags(pid)


@router.post("/{pid}/rags")
async def create_rag(pid: str, request: Request):
    body = await request.json()
    rag = db.create_rag(
        project_id=pid,
        name=body.get("name", "Untitled RAG"),
        description=body.get("description", ""),
        tags=body.get("tags", ""),
        store_path=body.get("store_path", ""),
    )
    return rag


@router.patch("/{pid}/rags/{rid}")
async def update_rag(pid: str, rid: str, request: Request):
    body = await request.json()
    return db.update_rag(rid, **body)


@router.delete("/{pid}/rags/{rid}")
async def delete_rag(pid: str, rid: str):
    db.delete_rag(rid)
    return {"ok": True}


@router.post("/{pid}/rags/{rid}/ingest")
async def ingest_into_rag(pid: str, rid: str, request: Request):
    """Ingest a file or directory into a project RAG.

    Tracks timing + status on both project_rags (latest summary) and
    rag_corpora (history of every build attempt).
    """
    body = await request.json()
    path = body.get("path", "")
    if not path or not os.path.exists(path):
        return {"error": "path not found"}
    rag = db.get_rag(rid)
    if not rag:
        return {"error": "rag not found"}
    # Mark the rag as building + create a history row.
    build_row = db.create_rag_build(project_id=pid, rag_id=rid)
    db.update_rag(rid, status="building", last_build_at=time.time(),
                  last_build_status="running")
    db.mark_rag_build_running(build_row["id"])
    mgr = RAGManager(rag["store_path"])
    try:
        if os.path.isdir(path):
            result = mgr.ingest_directory(path)
        else:
            result = mgr.ingest_file(path)
        stats = mgr.stats()
        doc_count = stats.get("total_documents", 0)
        chunk_count = stats.get("total_chunks", 0)
        db.mark_rag_build_done(build_row["id"], doc_count=doc_count,
                               chunk_count=chunk_count)
        db.update_rag(rid, doc_count=doc_count, chunk_count=chunk_count,
                      status="ready", last_build_status="ok",
                      error="")
        return {"result": result, "rag": db.get_rag(rid), "build": build_row}
    except Exception as e:  # noqa: BLE001
        try:
            db.mark_rag_build_failed(build_row["id"], str(e))
            db.update_rag(rid, status="error", last_build_status="failed",
                          error=str(e)[:500])
        except Exception:  # noqa: BLE001
            pass
        return {"error": f"ingest failed: {e}", "rag": db.get_rag(rid),
                "build": build_row}


@router.post("/{pid}/rags/{rid}/query")
async def query_rag(pid: str, rid: str, request: Request):
    body = await request.json()
    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))
    if not query:
        return {"error": "no query"}
    rag = db.get_rag(rid)
    if not rag:
        return {"error": "rag not found"}
    mgr = RAGManager(rag["store_path"])
    chunks = mgr.store.search(query, top_k=top_k)
    return {"chunks": [
        {"text": c.text, "score": c.score, "source": c.source}
        for c in chunks
    ]}


@router.get("/{pid}/rags/{rid}/stats")
async def rag_stats(pid: str, rid: str):
    rag = db.get_rag(rid)
    if not rag:
        return {"error": "rag not found"}
    mgr = RAGManager(rag["store_path"])
    return mgr.stats()


# ── Training Runs ────────────────────────────────────────────────────────

@router.get("/{pid}/runs")
async def list_runs(pid: str):
    return db.list_runs(pid)


@router.post("/{pid}/runs")
async def create_run(pid: str, request: Request):
    body = await request.json()
    run = db.create_run(
        project_id=pid,
        name=body.get("name", "Run"),
        base_model=body.get("base_model", ""),
        data_path=body.get("data_path", ""),
        rag_ids=body.get("rag_ids", []),
        settings_obj=body.get("settings", {}),
        system_prompt=body.get("system_prompt", ""),
        parent_run_id=body.get("parent_run_id"),
        notes=body.get("notes", ""),
    )
    return run


@router.get("/{pid}/runs/{rid}")
async def get_run(pid: str, rid: str):
    run = db.get_run(rid)
    if not run:
        return {"error": "not found"}
    run["benchmarks"] = db.list_benchmarks(rid)
    return run


@router.patch("/{pid}/runs/{rid}")
async def update_run(pid: str, rid: str, request: Request):
    body = await request.json()
    return db.update_run(rid, **body)


@router.delete("/{pid}/runs/{rid}")
async def delete_run(pid: str, rid: str):
    db.delete_run(rid)
    return {"ok": True}


@router.post("/{pid}/runs/{rid}/start")
async def start_run(pid: str, rid: str, request: Request):
    """Wire a persisted Run into the training engine and start it.

    This bridges the persistent run record with the live training loop.
    Engine state gets tagged with the run_id so progress events can
    update the DB row.
    """
    run = db.get_run(rid)
    if not run:
        return {"error": "run not found"}
    if run["status"] not in ("created", "idle", "error", "stopped"):
        return {"error": f"cannot start run in status {run['status']}"}
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    # Allow per-start overrides; fall back to persisted settings.
    settings_obj = {**run.get("settings", {}), **body.get("settings", {})}
    from finetune_studio.training.engine import TrainingConfig
    config = TrainingConfig(
        model_path=run.get("base_model", ""),
        output_dir=settings_obj.get("output_dir", "output"),
        lora_rank=int(settings_obj.get("lora_rank", 64)),
        learning_rate=float(settings_obj.get("learning_rate", 8e-5)),
        num_epochs=int(settings_obj.get("num_epochs", 4)),
        batch_size=int(settings_obj.get("batch_size", 2)),
        max_seq_length=int(settings_obj.get("max_seq_length", 2048)),
        merge_on_save=bool(settings_obj.get("merge_on_save", False)),
    )
    if not run.get("base_model"):
        return {"error": "run has no base_model"}
    if not run.get("data_path"):
        return {"error": "run has no data_path"}
    from finetune_studio.training.data import load_jsonl
    training_data = load_jsonl(run["data_path"])
    db.update_run(rid, status="running", started_at=time.time())
    training_engine.run_id = rid  # type: ignore[attr-defined]
    training_engine.start(config, training_data, run.get("system_prompt", ""))
    return {"status": "started", "run_id": rid, "run": db.get_run(rid)}


@router.post("/{pid}/runs/{rid}/stop")
async def stop_run(pid: str, rid: str):
    training_engine.stop()
    db.update_run(rid, status="stopped", finished_at=time.time())
    return {"ok": True}


@router.post("/{pid}/runs/{rid}/benchmark")
async def run_benchmark(pid: str, rid: str, request: Request):
    """Run a benchmark suite against a run's output model.

    The output model path is read from run.output_path. If empty,
    falls back to base_model. Persists results under benchmark_runs.

    Load failures and suite failures are caught and surfaced to the
    caller as 200-with-error JSON so the UI can render them inline;
    the engine is unloaded via a try/finally so a benchmark that
    crashes mid-run doesn't leak the loaded model.
    """
    body = await request.json()
    suite_name = body.get("suite_name", "default")
    suite_path = body.get("suite_path", "")

    from finetune_studio.testing.inference import InferenceEngine
    from finetune_studio.testing.suite import load_test_suite, run_suite, score_results

    run = db.get_run(rid)
    if not run:
        return {"error": "run not found"}
    target_model = run.get("output_path") or run.get("base_model")
    if not target_model:
        return {"error": "run has no model to benchmark"}
    engine = InferenceEngine()
    try:
        engine.load(target_model)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load failed: {e}", "benchmark": None}
    if not suite_path:
        engine.unload()
        return {"error": "suite_path required", "benchmark": None}
    cases = load_test_suite(suite_path)
    t0 = time.time()
    try:
        results = run_suite(engine, cases)
        scores = score_results(results)
        dt_ms = int((time.time() - t0) * 1000)
        bid = db.create_benchmark(rid, suite_name, scores, dt_ms)
        return {"benchmark": db.get_benchmark(bid), "results": [
            {"name": r.test_name, "passed": r.passed, "time_ms": r.time_ms,
             "response": r.response[:300]}
            for r in results
        ]}
    except Exception as e:  # noqa: BLE001
        log.exception("benchmark suite failed")
        return {"error": f"benchmark failed: {e}", "benchmark": None}
    finally:
        try:
            engine.unload()
        except Exception:  # noqa: BLE001
            pass


@router.post("/{pid}/runs/{rid}/merge")
async def merge_run(pid: str, rid: str, request: Request, force: str = "false"):
    """Merge a persisted run's adapter on disk into a standalone model.

    Loads run.base_model + <run.output_path>/adapter/, merges via
    PEFT's merge_and_unload(), saves to <run.output_path>/merged/.

    Idempotent: if <output_path>/merged/ already has files, the
    existing merge is returned unless `?force=true`. Always re-reads
    the run row at the end so the response carries fresh status /
    output_path / metrics.
    """
    force = str(force).lower() in ("1", "true", "yes")
    run = db.get_run(rid)
    if not run:
        return {"error": "run not found"}
    # Lazy import — keeps the training/PEFT stack out of the projects module
    # path until a merge is actually requested.
    from finetune_studio.training.engine import merge_adapter_for_run
    try:
        result = merge_adapter_for_run(run, force=force)
    except ValueError as e:
        return {"error": str(e), "status": "skipped"}
    except Exception as e:  # noqa: BLE001
        log.exception("merge failed")
        return {"error": f"merge failed: {e}", "status": "failed"}
    # Backfill output_path on the run if it was empty before — the merge
    # just wrote to <output_path>/merged/ so the parent must exist.
    if not run.get("output_path"):
        merged_path = result.get("merged_path") or ""
        parent = os.path.dirname(merged_path.rstrip("/"))
        if parent:
            db.update_run(rid, output_path=parent)
    fresh = db.get_run(rid)
    return {
        "ok": True,
        "status": "skipped" if result.get("skipped") else "merged",
        "merged_path": result.get("merged_path"),
        "size_bytes": result.get("size_bytes", 0),
        "size_human": result.get("size_human", "0 B"),
        "skipped": bool(result.get("skipped")),
        "force": force,
        "run": fresh,
    }


@router.get("/{pid}/runs/{rid}/benchmarks")
async def list_run_benchmarks(pid: str, rid: str):
    return db.list_benchmarks(rid)
