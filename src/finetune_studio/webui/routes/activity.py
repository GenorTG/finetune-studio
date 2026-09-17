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


# Status vocabularies used across every subsystem so the drawer + filters
# speak one language (running / queued / done / error).
_DONE_STATES = {"completed", "done", "ready", "succeeded", "success"}
_ERR_STATES = {"failed", "error", "cancelled", "canceled"}
_ACTIVE_STATES = {
    "loading", "training", "saving", "running", "downloading",
    "building", "embedding", "parsing", "in_progress",
}


def _norm_status(raw: str) -> str:
    """Collapse per-subsystem status strings into the activity vocabulary."""
    s = (raw or "").strip().lower()
    if s == "queued":
        return "queued"
    if s in _DONE_STATES:
        return "done"
    if s in _ERR_STATES:
        return "error"
    if s in _ACTIVE_STATES:
        return "running"
    return s or "running"


def _progress_for(status: str) -> float:
    """A coarse progress value for persisted rows (no live step counter)."""
    if status == "done":
        return 1.0
    if status == "error":
        return 0.0
    if status == "queued":
        return 0.0
    return 0.5


def _task_key(t: dict[str, Any]) -> str:
    """Stable identity so a live row and its persisted row collapse to one."""
    kind = str(t.get("kind") or "")
    if t.get("run_id"):
        return f"{kind}:run:{t['run_id']}"
    if t.get("id"):
        return f"{kind}:id:{t['id']}"
    return f"{kind}:{t.get('project_id') or ''}:{t.get('url') or ''}"


class _ProjCache:
    """Memoize db.get_project across a single snapshot build."""

    def __init__(self) -> None:
        self._cache: dict[str, dict | None] = {}

    def get(self, pid: str) -> dict | None:
        if not pid:
            return None
        if pid not in self._cache:
            self._cache[pid] = db.get_project(pid)
        return self._cache[pid]

    def name(self, pid: str) -> str | None:
        proj = self.get(pid)
        return proj["name"] if proj else None


def _persistent_tasks(projs: _ProjCache) -> list[dict[str, Any]]:
    """Recent finished/queued work read from the durable DB tables.

    This is what makes past runs survive a restart and makes whole
    subsystems (benchmarks, exports, data-prep, RAG builds, downloads,
    system updates) visible at all — the live snapshot only ever knew
    about the few things still resident in process memory.
    """
    tasks: list[dict[str, Any]] = []

    # ── Training runs (history) ────────────────────────────────────
    try:
        for r in db.list_runs():
            pid = r.get("project_id") or ""
            if pid and not projs.get(pid):
                continue
            status = _norm_status(r.get("status") or "")
            loss = r.get("final_loss")
            bits = [r.get("name") or "run"]
            if loss is not None:
                try:
                    bits.append(f"loss {float(loss):.4f}")
                except (TypeError, ValueError):
                    pass
            if r.get("error"):
                bits.append(str(r["error"])[:60])
            tasks.append({
                "kind": "training",
                "project_id": pid,
                "project_name": projs.name(pid) or "?",
                "status": status,
                "progress": _progress_for(status),
                "message": " · ".join(bits),
                "started_at": r.get("finished_at") or r.get("started_at") or r.get("created_at") or 0,
                "url": f"/projects/{pid}/training" if pid else "/projects",
                "run_id": r.get("id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"training_history: {e}"})

    # ── Benchmark / testing runs ───────────────────────────────────
    try:
        for b in db.list_benchmarks_recent(60):
            run = db.get_run(b.get("run_id") or "")
            pid = (run or {}).get("project_id") or ""
            if pid and not projs.get(pid):
                continue
            scores = b.get("scores") if isinstance(b.get("scores"), dict) else {}
            summary = ""
            for k in ("weighted", "accuracy", "pass_rate", "passed", "score"):
                if k in scores:
                    summary = f"{k} {scores[k]}"
                    break
            msg = b.get("suite_name") or "benchmark"
            if summary:
                msg = f"{msg} · {summary}"
            tasks.append({
                "kind": "benchmark",
                "project_id": pid,
                "project_name": projs.name(pid) or "?",
                "status": "done",
                "progress": 1.0,
                "message": msg,
                "started_at": b.get("ran_at") or 0,
                "url": f"/projects/{pid}/benchmarks" if pid else "/projects",
                "id": b.get("id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"benchmarks: {e}"})

    # ── Model exports / merges ─────────────────────────────────────
    try:
        for x in db.list_exports_recent(60):
            pid = x.get("project_id") or ""
            if pid and not projs.get(pid):
                continue
            status = _norm_status(x.get("status") or "")
            if x.get("error"):
                msg = f"failed: {str(x['error'])[:60]}"
            else:
                fmt = x.get("format") or "export"
                quant = x.get("quant") or ""
                tail = x.get("size_human") or (x.get("output_path") or "")[-40:]
                msg = f"{fmt}{('/' + quant) if quant else ''} → {tail}".strip()
            tasks.append({
                "kind": "export",
                "project_id": pid,
                "project_name": projs.name(pid) or "?",
                "status": status,
                "progress": _progress_for(status),
                "message": msg,
                "started_at": x.get("finished_at") or x.get("created_at") or 0,
                "url": f"/projects/{pid}/export" if pid else "/projects",
                "id": x.get("id"),
                "run_id": x.get("run_id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"exports: {e}"})

    # ── Data-prep runs (history) ───────────────────────────────────
    try:
        for d in db.list_data_prep_recent(60):
            pid = d.get("project_id") or ""
            if pid and not projs.get(pid):
                continue
            status = _norm_status(d.get("status") or "")
            if d.get("error"):
                msg = f"{d.get('filename', '')} · failed: {str(d['error'])[:50]}"
            else:
                qa = f"{d.get('qa_approved', 0)}/{d.get('qa_total', 0)} QA"
                msg = f"{d.get('filename', '')} · {qa}".strip(" ·")
            tasks.append({
                "kind": "data_prep",
                "project_id": pid,
                "project_name": projs.name(pid) or "?",
                "status": status,
                "progress": _progress_for(status),
                "message": msg,
                "started_at": d.get("finished_at") or d.get("created_at") or 0,
                "url": f"/projects/{pid}/data-prep" if pid else "/projects",
                "run_id": d.get("id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"data_prep_history: {e}"})

    # ── RAG builds (history) ───────────────────────────────────────
    try:
        for g in db.list_rag_builds_recent(60):
            pid = g.get("project_id") or ""
            if pid and not projs.get(pid):
                continue
            status = _norm_status(g.get("status") or "")
            kind = "rag_ready" if status == "done" else "rag_build"
            if g.get("error"):
                msg = f"failed: {str(g['error'])[:60]}"
            else:
                msg = f"{g.get('doc_count', 0)} docs · {g.get('chunk_count', 0)} chunks"
            tasks.append({
                "kind": kind,
                "project_id": pid,
                "project_name": projs.name(pid) or "?",
                "status": "ready" if status == "done" else status,
                "progress": _progress_for(status),
                "message": msg,
                "started_at": g.get("finished_at") or g.get("created_at") or 0,
                "url": f"/projects/{pid}/rag" if pid else "/projects",
                "id": g.get("id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"rag_history: {e}"})

    # ── HF downloads (history) ─────────────────────────────────────
    try:
        for h in db.list_hf_downloads_recent(60):
            raw = (h.get("status") or "").lower()
            status = "done" if raw == "completed" else _norm_status(raw)
            repo = h.get("repo_id") or "?"
            if status == "error":
                msg = f"failed: {str(h.get('error') or '')[:60]}"
            elif status == "done":
                msg = f"completed → {h.get('path', '')}"
            else:
                done = h.get("bytes_done") or 0
                total = h.get("bytes_total") or 0
                msg = f"{done/1e9:.2f}GB / {total/1e9:.2f}GB" if total else "downloading…"
            tasks.append({
                "kind": "download",
                "project_id": "",
                "project_name": repo.split("/")[-1],
                "status": status,
                "progress": _progress_for(status),
                "message": msg,
                "started_at": h.get("started_at") or h.get("created_at") or 0,
                "url": f"/models/explore#repo={repo}",
                "id": h.get("id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"downloads_history: {e}"})

    # ── System updates ─────────────────────────────────────────────
    try:
        for u in db.list_updates_recent(20):
            status = _norm_status(u.get("status") or "")
            mode = u.get("mode") or "update"
            msg = f"{mode} · {u.get('status', '')}"
            if u.get("error"):
                msg = f"{mode} · failed: {str(u['error'])[:50]}"
            tasks.append({
                "kind": "system_update",
                "project_id": "",
                "project_name": "system",
                "status": status,
                "progress": _progress_for(status),
                "message": msg,
                "started_at": u.get("finished_at") or u.get("started_at") or u.get("created_at") or 0,
                "url": "/settings",
                "id": u.get("id"),
            })
    except Exception as e:  # noqa: BLE001
        tasks.append({"kind": "_error", "message": f"system_updates: {e}"})

    return tasks


def collect_activity() -> dict[str, Any]:
    """Aggregate live + persisted tasks across all subsystems (sync snapshot)."""
    tasks: list[dict[str, Any]] = []
    projs = _ProjCache()

    # ── Training engine state ──────────────────────────────────────
    try:
        from finetune_studio.webui.app import training_engine
        s = training_engine.state
        # TrainingState is a dataclass, not a dict
        is_active = s.status not in ("idle", "") or s.current_step > 0
        if is_active:
            run_id = getattr(training_engine, "current_run_id", "") or ""
            # current_run_id is the composite ``{pid}-{db_run_id}``. Split it so
            # this live row shares the bare db run id with its persisted row and
            # dedups cleanly (both use run_id == db_run_id).
            pid = run_id.split("-", 1)[0] if "-" in run_id else ""
            bare_run_id = run_id.split("-", 1)[1] if "-" in run_id else run_id
            proj_name = "(running)"
            if pid:
                proj = db.get_project(pid)
                if not proj:
                    # A deleted project must not leave a ghost run in the
                    # global activity drawer.
                    proj_name = ""
                else:
                    proj_name = proj["name"]
            if not (pid and not proj):
                tasks.append({
                "kind": "training",
                "project_id": pid,
                "project_name": proj_name,
                "status": s.status or "running",
                "progress": (s.current_step / max(1, s.total_steps)),
                "message": f"step {s.current_step}/{s.total_steps} · loss {s.loss:.4f} · {s.message or ''}",
                "started_at": _now() - int(s.elapsed or 0),
                "url": f"/projects/{pid}/training" if pid else "/projects",
                "run_id": bare_run_id or None,
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
            if pid and not proj:
                continue
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

    # RAG build activity is read from the durable ``rag_corpora`` table in
    # ``_persistent_tasks`` (which tracks queued/running/done/error with real
    # doc & chunk counts) rather than scanning disk for lockfiles.

    # ── HF downloads (in-flight, in-memory) ────────────────
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

    # Everything appended so far came from live process memory — the sources
    # of truth for what is genuinely running right now. Tag them so the badge
    # can distinguish real live work from stale persisted running/queued rows.
    for t in tasks:
        t["_live"] = True

    # ── Merge persisted history (live rows collected above win on dedup) ──
    # The live rows carry real-time progress; a persisted row for the same
    # run_id/id is the finished record of the same thing, so it is dropped
    # while the live one is present, then takes over once the run leaves
    # process memory.
    seen: set[str] = {_task_key(t) for t in tasks if t.get("kind") != "_error"}
    for p in _persistent_tasks(projs):
        if p.get("kind") == "_error":
            tasks.append(p)
            continue
        key = _task_key(p)
        if key in seen:
            continue
        seen.add(key)
        tasks.append(p)

    # Sort: running/queued first, then by started_at desc.
    # Coerce missing/None started_at — `.get(..., 0)` still returns None when the key is present.
    tasks.sort(key=lambda t: (
        0 if t.get("status") in ("running", "queued") else 1,
        -float(t.get("started_at") or 0),
    ))

    # Bound the drawer: newest 60 rows after sorting. Live in-memory rows are
    # never evicted — a genuinely-running task must always be visible even when
    # a pile of stale persisted rows would otherwise fill the window.
    top = tasks[:60]
    if len(tasks) > 60:
        top_ids = {id(t) for t in top}
        top.extend(t for t in tasks[60:] if t.get("_live") and id(t) not in top_ids)
    tasks = top

    # Count by kind for the badge (only truly-active work). A persisted
    # running/queued row is counted only when it is recent — an interrupted
    # export/download/build that never got a terminal status must not spin
    # the badge forever. Live in-memory rows always count.
    now = _now()
    _STALE_AFTER = 2 * 3600.0
    counts: dict[str, int] = {}
    for t in tasks:
        if t.get("status") in ("running", "queued"):
            fresh = t.get("_live") or (now - float(t.get("started_at") or 0)) < _STALE_AFTER
            if fresh:
                k = t.get("kind", "?")
                counts[k] = counts.get(k, 0) + 1
        t.pop("_live", None)

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
