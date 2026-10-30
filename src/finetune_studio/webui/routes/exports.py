"""Routes for exporting trained runs to GGUF (and future formats).

Currently drives llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`.
Requires the run to have been merged first; if `auto_merge=true` (default)
and `<output_path>/merged/` is missing, chains the merge before exporting.

Install on the host (one-time):
    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    cmake -B llama.cpp/build && cmake --build llama.cpp/build --config Release

Tool discovery looks in PATH first, then in ~/llama.cpp, /opt/llama.cpp,
/usr/local/llama.cpp.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.webui import export_jobs, export_service
from finetune_studio.webui.export_work import (
    refresh_registry_quietly,
    run_single_quant_gguf,
)
from finetune_studio.webui.live_sse import sse_comment, sse_data, sse_response

log = logging.getLogger(__name__)
router = APIRouter()

# Preserve the route module's historical helper names for callers and tests;
# implementation and validation now live in the shared Export service.
DEFAULT_QUANT = export_service.DEFAULT_QUANT
SUPPORTED_QUANTS = export_service.SUPPORTED_QUANTS
_safe_name = export_service._safe_name


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see ``get_export``),
    so the guard matches that style instead of raising HTTPException. Called
    before the export row is read and before any stream starts.
    """
    if not db.get_project(pid):
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return None

# Common install locations for llama.cpp. Searched in order.
# Project-local first (per the "boxed in" rule) — install.sh drops a build
# at <project-root>/.llama.cpp/. Then venv-sibling fallback, then legacy
# external paths for users who already had a system-wide install.
def _project_root_llama_cpp() -> str:
    # __file__ is src/finetune_studio/webui/routes/exports.py
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(project_root, ".llama.cpp")

LLAMA_CPP_SEARCH_PATHS = [
    _project_root_llama_cpp(),
    os.path.expanduser("~/llama.cpp"),
    "/opt/llama.cpp",
    "/usr/local/llama.cpp",
]


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def _find_llama_tool(name: str) -> str | None:
    """Find an executable in PATH or common llama.cpp install locations."""
    if name == "llama-quantize":
        from finetune_studio.training.gguf_convert import find_llama_quantize
        return find_llama_quantize()
    p = shutil.which(name)
    if p:
        return p
    for base in LLAMA_CPP_SEARCH_PATHS:
        candidate = os.path.join(base, "build", "bin", name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _find_convert_script() -> str | None:
    """Find llama.cpp's convert_hf_to_gguf.py script."""
    from finetune_studio.training.gguf_convert import find_gguf_convert_script
    return find_gguf_convert_script()


@router.post("/projects/{pid}/runs/{rid}/export")
async def export_run(pid: str, rid: str, request: Request):
    """Start an export of a run (GGUF / abliterated / merged) as a tracked job.

    Returns **immediately** (``202 {ok, export_id, status: "queued"}``); the
    work runs on a worker thread (``webui/export_jobs.py``) and its state —
    phase, heartbeat, error — lives in the ``model_exports`` row, readable via
    ``GET /projects/{pid}/exports/{eid}`` and the SSE ``.../events``. Only one
    export runs at a time: a second request gets ``409`` naming the active one.
    Requests that cannot succeed (unknown format, no converter, bad quant, run
    not merged with ``auto_merge`` off) are refused synchronously with 400.

    Two modes:

    1. **Multi-format** — body includes ``quants`` (list) and/or ``format`` in
       {abliterated, merged}; merges the adapter onto ``base_model`` (optional
       override) first when ``merged/`` is missing. A multi-quant GGUF job
       ends with one row per quant; ``export_id`` is the first one.

    2. **Single quant GGUF** — body uses singular ``quant`` (default Q4_K_M)
       without ``quants``; output ``<gguf>/<base>-<quant>.gguf``.

    Body (common):
      format: gguf | abliterated | merged (default gguf).
      force: overwrite existing outputs (multi-format mode)
      base_model: optional compatible 16-bit base for merge-at-export
      auto_merge: bool (default true; single-quant mode)
      quants: list of GGUF quants (multi-format mode)
      quant: single GGUF quant (single-quant mode)
      imatrix: bool (GGUF, multi-format mode) — measure an importance matrix on the run's training text and
               quantize with it; files go to ``<output>/gguf-imatrix/`` beside the plain ones
    """
    body: object = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except ValueError:
            body = None
    if not isinstance(body, dict):
        return JSONResponse(
            {"ok": False, "status": "failed", "detail": "request body must be a JSON object"}, status_code=400,
        )
    result = await asyncio.to_thread(export_service.submit_export, pid, rid, body)
    return JSONResponse(result.body, status_code=result.status_code)


def _row_view(row: dict) -> dict:
    """Export row + derived fields the UI needs (all read-only)."""
    return export_service.row_view(row)


@router.get("/projects/{pid}/exports/active")
async def active_exports(pid: str):
    """Queued/running exports: this project's, and (``other``) any other project's.

    The Export page calls this on load to re-attach to a job that was started
    before a reload and to lock its buttons while the shared GPU is busy.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    rows = [_row_view(r) for r in db.list_active_exports()]
    return {
        "project": [r for r in rows if r["project_id"] == pid],
        "other": [r for r in rows if r["project_id"] != pid],
    }


@router.get("/projects/{pid}/exports/{eid}")
async def get_export(pid: str, eid: str):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    result = export_service.get_export(pid, eid)
    return JSONResponse(result.body, status_code=result.status_code) if result.status_code != 200 else result.body


@router.post("/projects/{pid}/exports/{eid}/cancel")
async def cancel_export(pid: str, eid: str):
    """Stop a queued/running export (kills its llama.cpp child's process group).

    A GPU adapter merge cannot be interrupted mid-flight: it stops when the
    merge returns, before conversion starts.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    result = export_service.cancel_export(pid, eid)
    return JSONResponse(result.body, status_code=result.status_code) if result.status_code != 200 else result.body


@router.get("/projects/{pid}/exports/{eid}/events")
async def export_events(pid: str, eid: str):
    """SSE stream of a single export row until it reaches a terminal status."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    export = db.get_export(eid)
    if export and export.get("project_id") != pid:
        return JSONResponse({"detail": "not found"}, status_code=404)

    async def gen():
        last: str | None = None
        while True:
            row = db.get_export(eid)
            if not row:
                yield sse_data({"error": "not found", "status": "failed", "id": eid})
                return
            if row.get("project_id") != pid:
                yield sse_data({"error": "not found", "status": "failed", "id": eid})
                return
            status = row.get("status") or ""
            fingerprint = (
                f"{status}|{row.get('phase') or ''}|{row.get('phase_detail') or ''}|"
                f"{row.get('heartbeat_at') or 0}|{row.get('output_path') or ''}|"
                f"{row.get('error') or ''}|{row.get('size_bytes') or 0}"
            )
            if fingerprint != last:
                last = fingerprint
                yield sse_data(_row_view(row))
            else:
                yield sse_comment()
            if status in export_jobs.TERMINAL_STATUSES:
                return
            await asyncio.sleep(0.75)

    return sse_response(gen())


@router.get("/projects/{pid}/exports")
async def list_project_exports(pid: str, limit: int = Query(100, ge=1, le=1000)):
    missing = _project_404(pid)
    if missing is not None:
        return missing
    return export_service.list_exports(pid, limit=limit)


_refresh_registry_quietly = refresh_registry_quietly


def _export_worker(eid: str, merged_dir: str, out_path: str, quant: str) -> None:
    """Run one single-quant GGUF export for an existing row, synchronously.

    Kept as a plain function (the job path calls ``run_single_quant_gguf``
    directly on its worker thread). ``FTS_SKIP_EXPORT=1`` short-circuits with
    a 1-byte marker for tests.
    """
    try:
        db.mark_export_running(eid)
        run_single_quant_gguf(eid, merged_dir, out_path, quant)
    except Exception as e:
        log.exception("export failed")
        try:
            db.mark_export_failed(eid, str(e))
        except Exception:
            log.debug("mark_export_failed secondary failure", exc_info=True)
