"""Activity feed — list all running/queued background tasks across the studio.

Aggregates: training runs, data-prep parsers, RAG builds, HF downloads,
and the currently-loaded inference model. Each task includes a deep link
back to the relevant page so users can jump from the activity panel.

Live updates use SSE at ``GET /api/activity/events`` (JSON snapshots).
``GET /api/activity`` remains the one-shot snapshot for fallback clients.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from finetune_studio import db
from finetune_studio.webui.live_sse import sse_comment, sse_data, sse_response

router = APIRouter()


def _now() -> float:
    return time.time()


def collect_activity() -> dict[str, Any]:
    """Aggregate live tasks across all subsystems (sync snapshot)."""
    tasks: list[dict[str, Any]] = []

    # ── Training engine state ──────────────────────────────────────
    try:
        from finetune_studio.webui.app import training_engine
        s = training_engine.state
        # TrainingState is a dataclass, not a dict
        is_active = s.status not in ("idle", "") or s.current_step > 0
        if is_active:
            run_id = getattr(training_engine, "current_run_id", "") or ""
            pid = run_id.split("-", 1)[0] if "-" in run_id else ""
            proj_name = "(running)"
            if pid:
                proj = db.get_project(pid)
                if proj:
                    proj_name = proj["name"]
            tasks.append({
                "kind": "training",
                "project_id": pid,
                "project_name": proj_name,
                "status": s.status or "running",
                "progress": (s.current_step / max(1, s.total_steps)),
                "message": f"step {s.current_step}/{s.total_steps} · loss {s.loss:.4f} · {s.message or ''}",
                "started_at": _now() - int(s.elapsed or 0),
                "url": f"/projects/{pid}/training" if pid else "/projects",
                "run_id": run_id or None,
            })
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
                    except Exception:  # noqa: BLE001, S110
                        pass
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"rag: {e}"})

    # ── HF downloads ──────────────────────────────────────
    try:
        from finetune_studio.webui.routes.hf_models import _DOWNLOADS
        for jid, entry in _DOWNLOADS.items():
            status = entry.get("status", "queued")
            if status not in ("queued", "downloading", "completed", "error", "cancelled"):
                continue
            # Don't accumulate stale finished entries forever
            if status in ("completed", "error", "cancelled"):
                started = entry.get("started_at") or entry.get("created_at") or 0
                age = _now() - float(started or 0)
                # Missing timestamps → treat as fresh (still show briefly).
                if started and age > 600:
                    continue
            repo = entry.get("repo_id", "?")
            done = entry.get("bytes_done", 0) or 0
            total = entry.get("bytes_total", 0) or 0
            # Normalize to activity status vocabulary (done/running/queued/error).
            if status == "completed":
                ui_status = "done"
            elif status == "downloading":
                ui_status = "running"
            elif status == "cancelled":
                ui_status = "error"
            else:
                ui_status = status
            if ui_status == "done":
                msg = f"completed → {entry.get('path','')}"
                prog = 1.0
            elif ui_status == "error":
                msg = f"failed: {(entry.get('error') or '')[:60]}"
                prog = 0
            else:
                msg = f"{done/1e9:.2f}GB / {total/1e9:.2f}GB" if total else "starting…"
                prog = (done / total) if total > 0 else 0
            tasks.append({
                "kind": "download",
                "project_id": "",
                "project_name": repo.split("/")[-1],
                "status": ui_status,
                "progress": prog,
                "message": msg,
                "started_at": entry.get("started_at", _now()),
                "url": f"/models/explore#repo={repo}",
                "id": jid,
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"downloads: {e}"})

    # Sort: running first, then by started_at desc.
    # Coerce missing/None started_at — `.get(..., 0)` still returns None when the key is present.
    tasks.sort(key=lambda t: (
        0 if t.get("status") in ("running", "queued") else 1,
        -float(t.get("started_at") or 0),
    ))

    # Count by kind for the badge
    counts: dict[str, int] = {}
    for t in tasks:
        if t.get("status") in ("running", "queued"):
            k = t.get("kind", "?")
            counts[k] = counts.get(k, 0) + 1

    return {"tasks": tasks, "active_count": sum(counts.values()), "by_kind": counts}


@router.get("/api/activity")
async def activity() -> dict[str, Any]:
    """One-shot activity snapshot (fallback for non-SSE clients)."""
    return collect_activity()


@router.get("/api/activity/events")
async def activity_events():
    """SSE stream of activity snapshots.

    Emits a JSON payload whenever the aggregated task list changes.
    Keepalive comments are sent otherwise so the connection stays open.
    """
    async def gen():
        last: str | None = None
        while True:
            payload = collect_activity()
            # Fingerprint without volatile started_at drift on training rows.
            stable = []
            for t in payload.get("tasks") or []:
                stable.append({
                    "kind": t.get("kind"),
                    "project_id": t.get("project_id"),
                    "status": t.get("status"),
                    "progress": round(float(t.get("progress") or 0), 3),
                    "message": t.get("message"),
                    "run_id": t.get("run_id"),
                    "id": t.get("id"),
                    "url": t.get("url"),
                })
            fingerprint = json.dumps(
                {"tasks": stable, "active_count": payload.get("active_count")},
                sort_keys=True,
            )
            if fingerprint != last:
                last = fingerprint
                yield sse_data(payload)
            else:
                yield sse_comment()
            await asyncio.sleep(1.0)

    return sse_response(gen())
