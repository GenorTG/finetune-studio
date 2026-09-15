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

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.webui.live_sse import sse_comment, sse_data, sse_response

log = logging.getLogger(__name__)
router = APIRouter()

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
async def export_run(pid: str, rid: str, request: Request,
                     background: BackgroundTasks):
    """Export a run to GGUF / GPTQ / abliterated / merged safetensors.

    Two modes:

    1. **UI / multi-format (sync)** — body includes ``quants`` (list) and/or
       ``format`` in {gptq, abliterated, merged}. Merges the adapter onto
       ``base_model`` (optional override) when ``merged/`` is missing.

    2. **Legacy async GGUF** — body uses singular ``quant`` (default Q4_K_M)
       without ``quants``. Queues a background job and returns an export_id.

    Body (common):
      format: gguf | gptq | abliterated | merged (default gguf).
      force: overwrite existing outputs (sync path)
      base_model: optional compatible 16-bit base for merge-at-export
      auto_merge: bool (default true; legacy async path)
      quants: list of GGUF quants (sync UI path)
      quant: single GGUF quant (legacy async path)
    """
    body = await request.json() if request.headers.get(
        "content-type", "").startswith("application/json") else {}
    fmt = (body.get("format") or "gguf").lower()
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

    # Sync multi-format path used by the Export page (quants list / non-gguf).
    # AWQ is not a supported format; route it through export_trained_run so the
    # response carries the clear removal message (not a generic unsupported).
    use_sync = (
        "quants" in body
        or fmt in ("gptq", "abliterated", "merged", "awq")
        or bool(body.get("force")) and "quant" not in body
    )
    if use_sync:
        from finetune_studio.training.export_response import (
            ExportResult,
            dir_size_bytes,
            human_size,
        )
        from finetune_studio.training.run_export import export_trained_run

        quants = body.get("quants")
        raw = export_trained_run(
            run,
            fmt=fmt,
            quants=list(quants) if isinstance(quants, list) else None,
            force=bool(body.get("force", False)),
            base_model=base_model,
        )
        # Always coerce through ExportResult so numpy/tensors cannot 500 the
        # response encoder after a successful GPU merge/abliteration.
        payload = ExportResult.from_raw(raw)
        if not payload.ok or payload.error or payload.status == "failed":
            # Never report format failures as HTTP 200 success.
            return JSONResponse(
                payload.model_dump(exclude_none=False),
                status_code=400,
            )

        # Register successful sync artifacts (merged / abliterated / gptq /
        # verified GGUF) so Export + Models pages list them after reload.
        if payload.status in ("exported", "skipped") and fmt in (
            "merged", "abliterated", "gptq", "gguf",
        ):
            art = payload.artifact_path()
            if art:
                size = dir_size_bytes(art)
                quant_label = (
                    (payload.quants[0] if payload.quants else None)
                    or payload.quant
                    or ("safetensors" if fmt != "gguf" else DEFAULT_QUANT)
                )
                try:
                    row = db.create_export(
                        project_id=pid,
                        run_id=rid,
                        format=fmt,
                        quant=str(quant_label),
                    )
                    db.mark_export_done(
                        row["id"],
                        output_path=art,
                        size_bytes=size,
                        size_human=human_size(size),
                    )
                    payload = payload.model_copy(
                        update={
                            "export_id": row["id"],
                            "output_path": payload.output_path or art,
                            "size_bytes": size,
                            "size_human": human_size(size),
                        }
                    )
                except Exception:
                    log.exception(
                        "failed to register sync export row for %s/%s",
                        pid, rid,
                    )

        return JSONResponse(payload.model_dump(exclude_none=False))

    if fmt != "gguf":
        return _err(f"unsupported format: {fmt}", format=fmt)
    quant = (body.get("quant") or DEFAULT_QUANT).upper()
    if quant not in SUPPORTED_QUANTS:
        return _err(
            f"unsupported quant: {quant}",
            supported=sorted(SUPPORTED_QUANTS),
        )

    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return _err("run has no output_path; training did not finish")

    merged_dir = os.path.join(output_path, "merged")
    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        if not auto_merge:
            return _err(
                "run has not been merged; POST /merge first, "
                "or set auto_merge=true in the request body",
            )
        try:
            from finetune_studio.training.run_export import (
                ensure_merged_for_export,
            )
            merge_result = ensure_merged_for_export(
                run, base_model=base_model, force=False,
            )
            merged_dir = merge_result.get("merged_path") or merged_dir
            log.info("auto-merge for export: %s", merged_dir)
        except ValueError as e:
            return _err(f"auto-merge failed: {e}")
        except Exception as e:
            log.exception("auto-merge failed")
            return _err(f"auto-merge failed: {e}", status_code=500)

    # Persist the export row up front so the caller can poll it.
    export_row = db.create_export(project_id=pid, run_id=rid,
                                  format=fmt, quant=quant)

    base = os.path.basename(
        base_model or run.get("base_model") or "model"
    ).replace("/", "__")
    gguf_dir = os.path.join(output_path, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    out_filename = f"{_safe_name(base)}-{quant}.gguf"
    out_path = os.path.join(gguf_dir, out_filename)

    background.add_task(_export_worker, export_row["id"], merged_dir,
                        out_path, quant)
    return {"ok": True, "export_id": export_row["id"], "status": "queued",
            "quant": quant, "format": fmt, "output_path": out_path}


@router.get("/projects/{pid}/exports/{eid}")
async def get_export(pid: str, eid: str):
    row = db.get_export(eid)
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    return row


@router.get("/projects/{pid}/exports/{eid}/events")
async def export_events(pid: str, eid: str):
    """SSE stream of a single export row until it reaches a terminal status."""
    async def gen():
        last: str | None = None
        while True:
            row = db.get_export(eid)
            if not row:
                yield sse_data({"error": "not found", "status": "failed", "id": eid})
                return
            # Ignore project mismatches quietly — row still streams.
            status = row.get("status") or ""
            fingerprint = (
                f"{status}|{row.get('output_path') or ''}|"
                f"{row.get('error') or ''}|{row.get('size_bytes') or 0}"
            )
            if fingerprint != last:
                last = fingerprint
                yield sse_data(row)
            else:
                yield sse_comment()
            if status in ("done", "failed", "error", "cancelled"):
                return
            await asyncio.sleep(0.75)

    return sse_response(gen())


@router.get("/projects/{pid}/runs/{rid}/exports")
async def list_run_exports(pid: str, rid: str):
    return db.list_exports_for_run(rid)


@router.get("/projects/{pid}/exports")
async def list_project_exports(pid: str, limit: int = 100):
    return db.list_exports_for_project(pid, limit=limit)


def _export_worker(eid: str, merged_dir: str, out_path: str, quant: str) -> None:
    """Background GGUF export worker.

    Delegates to ``convert_merged_to_gguf`` (llama.cpp convert + quantize).
    ``FTS_SKIP_EXPORT=1`` short-circuits with a 1-byte marker for tests.
    """
    try:
        db.mark_export_running(eid)
        from finetune_studio.training.gguf_convert import (
            convert_merged_to_gguf,
            normalize_gguf_quant,
        )

        gguf_dir = os.path.dirname(out_path)
        nq = normalize_gguf_quant(quant)
        result = convert_merged_to_gguf(
            merged_dir, gguf_dir, [nq], force=True,
        )
        if not result.get("ok"):
            raise RuntimeError(
                result.get("error") or "GGUF conversion failed"
            )
        # Prefer the exact outfile the caller queued; fall back to converter path.
        final_path = out_path
        files = result.get("files") or []
        if files:
            # Converter writes model-{quant}.gguf; rename/copy if needed.
            produced = files[0]
            if os.path.abspath(produced) != os.path.abspath(out_path):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if not os.path.isfile(out_path):
                    shutil.copy2(produced, out_path)
            final_path = out_path if os.path.isfile(out_path) else produced
        if not os.path.isfile(final_path) or os.path.getsize(final_path) <= 0:
            raise RuntimeError(
                f"output GGUF missing or empty after conversion: {final_path}. "
                "Refuse to mark export done without a non-empty artifact."
            )
        size = os.path.getsize(final_path)
        db.mark_export_done(
            eid,
            output_path=final_path,
            size_bytes=size,
            size_human=_human_size(size),
            intermediate_path=result.get("intermediate_path") or "",
        )
    except Exception as e:
        log.exception("export failed")
        try:
            db.mark_export_failed(eid, str(e))
        except Exception:
            log.debug("mark_export_failed secondary failure", exc_info=True)