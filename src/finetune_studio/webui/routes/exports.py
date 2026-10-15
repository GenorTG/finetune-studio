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
import json
import logging
import os
import shutil
import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.training.export_response import ExportResult
from finetune_studio.training.run_export import validate_export_request
from finetune_studio.webui import export_jobs
from finetune_studio.webui.export_work import (
    multi_export_work,
    refresh_registry_quietly,
    run_single_quant_gguf,
    single_quant_work,
)
from finetune_studio.webui.live_sse import sse_comment, sse_data, sse_response

log = logging.getLogger(__name__)
router = APIRouter()


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see ``get_export``),
    so the guard matches that style instead of raising HTTPException. Called
    before the export row is read and before any stream starts.
    """
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return None

# Quantization levels supported by llama.cpp's llama-quantize + convert's
# built-in --outtype flag. f16/bf16/f32/Q8_0 are single-step via convert;
# everything else needs the two-step HF -> fp16 GGUF -> quantized flow.
SUPPORTED_QUANTS = {
    "f16", "bf16", "f32",
    "Q8_0",
    "Q5_K_M", "Q5_K_S",
    "Q4_K_M", "Q4_K_S",
    "Q3_K_M", "Q3_K_S", "Q3_K_L",
    "Q2_K",
    "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_S", "IQ2_XXS", "IQ2_XS",
}
DEFAULT_QUANT = "Q4_K_M"

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


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


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
    """
    body: object = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except ValueError:
            body = None
    if not isinstance(body, dict):
        return JSONResponse(
            {"ok": False, "status": "failed",
             "error": "request body must be a JSON object"},
            status_code=400,
        )
    fmt = str(body.get("format") or "gguf").lower()
    auto_merge = bool(body.get("auto_merge", True))
    base_model_raw = body.get("base_model")
    base_model = (
        str(base_model_raw).strip() or None
        if base_model_raw is not None else None
    )

    run = db.get_run(rid)
    if not run:
        return JSONResponse(
            {"ok": False, "status": "failed", "error": "run not found"},
            status_code=404,
        )
    if run.get("project_id") and pid and run["project_id"] != pid:
        return JSONResponse(
            {
                "ok": False,
                "status": "failed",
                "error": "run does not belong to this project",
            },
            status_code=400,
        )

    def _err(msg: str, *, status_code: int = 400, **extra: object):
        payload = {"ok": False, "status": "failed", "error": msg, **extra}
        return JSONResponse(payload, status_code=status_code)

    # AWQ and GPTQ are not supported formats; they are refused here with the
    # clear removal message (not a generic "unsupported").
    use_multi = (
        "quants" in body
        or fmt in ("abliterated", "merged", "awq", "gptq")
        or bool(body.get("force")) and "quant" not in body
    )
    if use_multi:
        quants_raw = body.get("quants")
        quants = [str(q) for q in quants_raw] if isinstance(quants_raw, list) else None
        refused = validate_export_request(
            run, fmt=fmt, quants=quants, force=bool(body.get("force", False)),
        )
        if refused is not None:
            payload = ExportResult.from_raw(refused)
            return JSONResponse(payload.model_dump(exclude_none=False), status_code=400)
        label = (quants[0] if quants else None) or (
            DEFAULT_QUANT if fmt == "gguf" else "safetensors"
        )
        return _start(
            pid, rid, fmt=fmt, quant=label, quants=quants,
            make_work=lambda eid: multi_export_work(
                eid, run=run, fmt=fmt, quants=quants,
                force=bool(body.get("force", False)), base_model=base_model,
            ),
        )

    if fmt != "gguf":
        return _err(f"unsupported format: {fmt}", format=fmt)
    quant_raw = str(body.get("quant") or DEFAULT_QUANT)
    # SUPPORTED_QUANTS mixes case (f16 vs Q8_0): match case-insensitively
    # and keep the canonical spelling.
    quant = {q.upper(): q for q in SUPPORTED_QUANTS}.get(quant_raw.upper())
    if quant is None:
        return _err(
            f"unsupported quant: {quant_raw}",
            supported=sorted(SUPPORTED_QUANTS),
        )

    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return _err("run has no output_path; training did not finish")
    merged_dir = os.path.join(output_path, "merged")
    if not auto_merge and (not os.path.isdir(merged_dir) or not os.listdir(merged_dir)):
        return _err(
            "run has not been merged; POST /merge first, "
            "or set auto_merge=true in the request body",
        )
    refused = validate_export_request(run, fmt=fmt)
    if refused is not None:
        return _err(str(refused["error"]), status_code=400)

    base = os.path.basename(base_model or run.get("base_model") or "model")
    out_path = os.path.join(output_path, "gguf", f"{_safe_name(base)}-{quant}.gguf")
    return _start(
        pid, rid, fmt=fmt, quant=quant, quants=[quant],
        extra={"output_path": out_path},
        make_work=lambda eid: single_quant_work(
            eid, run=run, base_model=base_model, auto_merge=auto_merge,
            out_path=out_path, quant=quant,
        ),
    )


def _start(
    pid: str, rid: str, *, fmt: str, quant: str, quants: list[str] | None,
    make_work, extra: dict | None = None,
) -> JSONResponse:
    """Queue the job; 202 with its id, or 409 when another export is running."""
    try:
        row = export_jobs.start_job(
            project_id=pid, run_id=rid, fmt=fmt, quant=quant, quants=quants,
            make_work=make_work,
        )
    except export_jobs.ExportBusy as busy:
        return JSONResponse(
            {
                "ok": False, "status": "busy", "error": str(busy),
                "active_export_id": busy.active.get("id"),
                "active_project_id": busy.active.get("project_id"),
            },
            status_code=409,
        )
    return JSONResponse(
        {"ok": True, "export_id": row["id"], "status": "queued",
         "format": fmt, "quant": quant, "quants": quants or [], **(extra or {})},
        status_code=202,
    )


def _row_view(row: dict) -> dict:
    """Export row + derived fields the UI needs (all read-only)."""
    try:
        quants = json.loads(row.get("quants_json") or "[]")
    except ValueError:
        quants = []
    return {
        **row, "quants": quants, "server_now": time.time(),
        "active": export_jobs.is_active(row["id"]),
    }


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
    row = db.get_export(eid)
    if not row or row.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _row_view(row)


@router.post("/projects/{pid}/exports/{eid}/cancel")
async def cancel_export(pid: str, eid: str):
    """Stop a queued/running export (kills its llama.cpp child's process group).

    A GPU adapter merge cannot be interrupted mid-flight: it stops when the
    merge returns, before conversion starts.
    """
    missing = _project_404(pid)
    if missing is not None:
        return missing
    row = db.get_export(eid)
    if not row or row.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)
    if (row.get("status") or "") in export_jobs.TERMINAL_STATUSES:
        return JSONResponse(
            {"ok": False, "error": f"export already {row['status']}"},
            status_code=409,
        )
    if not export_jobs.cancel_job(eid):
        return JSONResponse(
            {"ok": False, "error": "export is not running in this server process"},
            status_code=409,
        )
    return {"ok": True, "status": "cancelling", "export_id": eid}


@router.get("/projects/{pid}/exports/{eid}/events")
async def export_events(pid: str, eid: str):
    """SSE stream of a single export row until it reaches a terminal status."""
    missing = _project_404(pid)
    if missing is not None:
        return missing
    export = db.get_export(eid)
    if export and export.get("project_id") != pid:
        return JSONResponse({"error": "not found"}, status_code=404)

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
    return db.list_exports_for_project(pid, limit=limit)


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
