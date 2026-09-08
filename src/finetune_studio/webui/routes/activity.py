"""Activity feed — list all running/queued background tasks across the studio.

Aggregates: training runs, data-prep parsers, RAG builds, HF downloads,
and the currently-loaded inference model. Each task includes a deep link
back to the relevant page so users can jump from the activity panel.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from finetune_studio import db

router = APIRouter()


def _now() -> float:
    return time.time()


@router.get("/api/activity")
async def activity() -> dict:
    """Aggregate live tasks across all subsystems."""
    tasks: list[dict[str, Any]] = []

    # ── Training engine state ──────────────────────────────────────
    try:
        from finetune_studio.webui.app import training_engine
        s = training_engine.state
        # s has: running, current_step, total_steps, loss, project_id, started_at
        if s.get("running") or s.get("started_at"):
            pid = s.get("project_id", "")
            proj = db.get_project(pid) if pid else None
            tasks.append({
                "kind": "training",
                "project_id": pid,
                "project_name": proj["name"] if proj else "?",
                "status": "running" if s.get("running") else "queued",
                "progress": (s.get("current_step", 0) / max(1, s.get("total_steps", 1))),
                "message": f"step {s.get('current_step', 0)}/{s.get('total_steps', 0)} · loss {s.get('loss', 0):.4f}",
                "started_at": s.get("started_at", 0),
                "url": f"/projects/{pid}/training" if pid else "/projects",
            })
    except Exception as e:
        tasks.append({"kind": "_error", "message": f"training: {e}"})

    # ── Inference engine loaded model ───────────────────────────────
    try:
        from finetune_studio.webui.app import inference_engine
        if getattr(inference_engine, "model", None) is not None:
            path = getattr(inference_engine, "model_path", "")
            tasks.append({
                "kind": "inference",
                "project_id": "",
                "project_name": Path(path).name if path else "?",
                "status": "loaded",
                "progress": 1.0,
                "message": f"ready · {path[-50:]}",
                "started_at": _now() - 3600,  # approximate
                "url": "/inference",
            })
    except Exception as e:
        tasks.append({"kind": "_error", "message": f"inference: {e}"})

    # ── Data-prep active runs ──────────────────────────────────────
    try:
        from finetune_studio.webui.routes.data_prep import _RUNS
        for (pid, run_id), entry in _RUNS.items():
            log = entry.get("log", [])
            last = log[-1] if log else {}
            proj = db.get_project(pid) if pid else None
            # Latest stage: 'done' / 'error' / active
            stage = last.get("stage", "queued")
            pct = last.get("pct", 0) if isinstance(last.get("pct"), (int, float)) else 0
            tasks.append({
                "kind": "data_prep",
                "project_id": pid,
                "project_name": proj["name"] if proj else "?",
                "status": "done" if stage == "done" else ("error" if stage == "error" else "running"),
                "progress": pct / 100.0,
                "message": last.get("message", entry.get("filename", ""))[:80],
                "started_at": log[0].get("ts", _now()) if log else _now(),
                "url": f"/projects/{pid}/data-prep",
                "run_id": run_id,
            })
    except Exception as e:
        tasks.append({"kind": "_error", "message": f"data_prep: {e}"})

    # ── RAG corpora on disk ────────────────────────────────────────
    try:
        corpora_root = Path.home() / ".finetune-studio" / "rag_corpora"
        if corpora_root.exists():
            for d in corpora_root.iterdir():
                if not d.is_dir():
                    continue
                # Active build: presence of .building lockfile
                lock = d / ".building"
                meta = d / "meta.json"
                pid = d.name
                proj = db.get_project(pid)
                if lock.exists():
                    tasks.append({
                        "kind": "rag_build",
                        "project_id": pid,
                        "project_name": proj["name"] if proj else pid,
                        "status": "running",
                        "progress": 0.5,
                        "message": "embedding corpus…",
                        "started_at": lock.stat().st_mtime,
                        "url": f"/projects/{pid}/rag",
                    })
                elif meta.exists():
                    import json as _json
                    try:
                        m = _json.loads(meta.read_text())
                        chunks = m.get("chunks", 0)
                        embeds = m.get("embeddings", 0)
                        built_at = m.get("built_at", 0)
                        tasks.append({
                            "kind": "rag_ready",
                            "project_id": pid,
                            "project_name": proj["name"] if proj else pid,
                            "status": "ready",
                            "progress": 1.0,
                            "message": f"{chunks} chunks · {embeds} embeddings",
                            "started_at": built_at,
                            "url": f"/projects/{pid}/rag",
                        })
                    except Exception:
                        pass
    except Exception as e:
        tasks.append({"kind": "_error", "message": f"rag: {e}"})

    # Sort: running first, then by started_at desc
    tasks.sort(key=lambda t: (
        0 if t.get("status") in ("running", "queued") else 1,
        -t.get("started_at", 0),
    ))

    # Count by kind for the badge
    counts: dict[str, int] = {}
    for t in tasks:
        if t.get("status") in ("running", "queued"):
            k = t.get("kind", "?")
            counts[k] = counts.get(k, 0) + 1

    return {"tasks": tasks, "active_count": sum(counts.values()), "by_kind": counts}
