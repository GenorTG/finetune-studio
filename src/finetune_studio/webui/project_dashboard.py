"""Build enriched context for the project overview dashboard.

Pure helpers — no FastAPI routes. Call from ``project_detail_page`` after
``_project_ctx`` has attached rags/runs/datasets/models.
"""
from __future__ import annotations

import time
from typing import Any


def format_relative(ts: float | None, *, now: float | None = None) -> str:
    """Human relative time from a Unix timestamp (seconds)."""
    if ts is None:
        return "—"
    try:
        t = float(ts)
    except (TypeError, ValueError):
        return "—"
    if t <= 0:
        return "—"
    now_f = float(now) if now is not None else time.time()
    diff = max(0.0, now_f - t)
    if diff < 60:
        return "just now"
    if diff < 3600:
        mins = int(diff // 60)
        return f"{mins}m ago"
    if diff < 86400:
        hours = int(diff // 3600)
        return f"{hours}h ago"
    if diff < 86400 * 30:
        days = int(diff // 86400)
        return f"{days}d ago"
    if diff < 86400 * 365:
        months = int(diff // (86400 * 30))
        return f"{months}mo ago"
    years = int(diff // (86400 * 365))
    return f"{years}y ago"


def truncate_description(text: str | None, *, limit: int = 120) -> dict[str, Any]:
    """Return short one-liner + whether a 'details' link is needed."""
    raw = (text or "").strip()
    if not raw:
        return {
            "short": "No description yet — add one in Settings.",
            "long": False,
            "full": "",
        }
    if len(raw) <= limit:
        return {"short": raw, "long": False, "full": raw}
    cut = (
        raw[: limit - 1].rsplit(" ", 1)[0]
        if " " in raw[:limit]
        else raw[: limit - 1]
    )
    return {"short": cut + "…", "long": True, "full": raw}


def run_status_counts(runs: list[dict]) -> dict[str, int]:
    """Count runs by status bucket for the Runs stat subtitle."""
    done = running = failed = 0
    for run in runs:
        status = (run.get("status") or "").lower()
        if status == "done":
            done += 1
        elif status == "running":
            running += 1
        elif status == "failed":
            failed += 1
    return {"done": done, "running": running, "failed": failed}


def models_size_gb(models: list[dict]) -> float:
    """Sum of export sizes in GB, rounded to 2 decimals."""
    total = 0.0
    for m in models:
        try:
            total += float(m.get("size_gb") or 0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)


def recent_runs(runs: list[dict], *, limit: int = 5) -> list[dict]:
    """Last N runs sorted by started_at desc (fallback: created_at)."""
    def _key(run: dict) -> float:
        for field in ("started_at", "created_at"):
            val = run.get(field)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return 0.0

    ordered = sorted(runs, key=_key, reverse=True)
    return ordered[:limit]


def recent_models(models: list[dict], *, limit: int = 5) -> list[dict]:
    """Last N exports sorted by mtime desc (fallback: created_at string ignored)."""
    def _key(m: dict) -> float:
        for field in ("mtime",):
            val = m.get(field)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return 0.0

    ordered = sorted(models, key=_key, reverse=True)
    return ordered[:limit]


def build_activity(
    *,
    runs: list[dict],
    models: list[dict],
    files: list[dict],
    pid: str,
    limit: int = 10,
    now: float | None = None,
) -> list[dict]:
    """Merge run / benchmark / model / file events into a timeline."""
    events: list[dict] = []

    for run in runs:
        rid = run.get("id") or ""
        name = run.get("name") or rid[:8] or "run"
        href = (
            f"/projects/{pid}/training?run={rid}"
            if rid
            else f"/projects/{pid}/training"
        )
        status = (run.get("status") or "").lower()
        started = run.get("started_at") or run.get("created_at")
        finished = run.get("finished_at")

        if started:
            events.append({
                "kind": "run_start",
                "icon": "▶",
                "message": f"Training started · {name}",
                "ts": float(started),
                "href": href,
            })
        if status == "done" and finished:
            events.append({
                "kind": "run_done",
                "icon": "✓",
                "message": f"Training finished · {name}",
                "ts": float(finished),
                "href": href,
            })
        elif status == "failed" and finished:
            events.append({
                "kind": "run_failed",
                "icon": "✗",
                "message": f"Training failed · {name}",
                "ts": float(finished),
                "href": href,
            })
        elif status == "failed" and started and not finished:
            events.append({
                "kind": "run_failed",
                "icon": "✗",
                "message": f"Training failed · {name}",
                "ts": float(started),
                "href": href,
            })

        for bench in run.get("benchmarks") or []:
            ran_at = bench.get("ran_at")
            if not ran_at:
                continue
            suite = bench.get("suite_name") or "benchmark"
            events.append({
                "kind": "benchmark",
                "icon": "📊",
                "message": f"Benchmark · {suite} on {name}",
                "ts": float(ran_at),
                "href": f"/projects/{pid}/benchmarks",
            })

    for m in models:
        mtime = m.get("mtime")
        if mtime is None:
            continue
        try:
            ts = float(mtime)
        except (TypeError, ValueError):
            continue
        label = m.get("name") or m.get("format") or "export"
        events.append({
            "kind": "export",
            "icon": "📦",
            "message": f"Model exported · {label}",
            "ts": ts,
            "href": f"/projects/{pid}/models",
        })

    for f in files:
        uploaded = f.get("uploaded_at")
        if uploaded is None:
            continue
        try:
            ts = float(uploaded)
        except (TypeError, ValueError):
            continue
        events.append({
            "kind": "upload",
            "icon": "⬆",
            "message": f"File uploaded · {f.get('original_name') or 'file'}",
            "ts": ts,
            "href": f"/projects/{pid}/data-prep",
        })

    events.sort(key=lambda e: e.get("ts") or 0, reverse=True)
    out: list[dict] = []
    for ev in events[:limit]:
        out.append({
            **ev,
            "when": format_relative(ev.get("ts"), now=now),
        })
    return out


def build_dashboard_ctx(
    project: dict,
    pid: str,
    *,
    files: list[dict] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Assemble template extras for ``project.html``."""
    runs = project.get("runs") or []
    models = project.get("models") or []
    datasets = project.get("datasets") or []
    file_rows = files if files is not None else []
    status = run_status_counts(runs)
    size_gb = models_size_gb(models)
    desc = truncate_description(project.get("description"))

    started_vals: list[float] = []
    for run in runs:
        started = run.get("started_at")
        if started is None:
            continue
        try:
            started_vals.append(float(started))
        except (TypeError, ValueError):
            continue
    runs_started_max: float | None = max(started_vals) if started_vals else None

    return {
        "desc_short": desc["short"],
        "desc_long": desc["long"],
        "desc_full": desc["full"],
        "stats": {
            "files": len(file_rows),
            "datasets": len(datasets),
            "runs": len(runs),
            "runs_done": status["done"],
            "runs_running": status["running"],
            "runs_failed": status["failed"],
            "models": len(models),
            "models_size_gb": size_gb,
        },
        "recent_runs": recent_runs(runs, limit=5),
        "recent_models": recent_models(models, limit=5),
        "activity": build_activity(
            runs=runs,
            models=models,
            files=file_rows,
            pid=pid,
            limit=10,
            now=now,
        ),
        "runs_started_max": runs_started_max,
        "format_relative": format_relative,
    }
