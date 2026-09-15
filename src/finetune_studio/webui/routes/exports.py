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

import logging
import os
import shutil
import subprocess
import sys

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from finetune_studio import db

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
    for base in LLAMA_CPP_SEARCH_PATHS:
        candidate = os.path.join(base, "convert_hf_to_gguf.py")
        if os.path.isfile(candidate):
            return candidate
    return None


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
      format: gguf | gptq | abliterated | merged (default gguf). AWQ removed.
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
        return {"error": "run not found"}
    if run.get("project_id") and pid and run["project_id"] != pid:
        return {"error": "run does not belong to this project"}

    # Sync multi-format path used by the Export page (quants list / non-gguf).
    use_sync = (
        "quants" in body
        or fmt in ("gptq", "abliterated", "merged", "awq")
        or bool(body.get("force")) and "quant" not in body
    )
    if use_sync:
        from finetune_studio.training.run_export import export_trained_run
        quants = body.get("quants")
        return export_trained_run(
            run,
            fmt=fmt,
            quants=list(quants) if isinstance(quants, list) else None,
            force=bool(body.get("force", False)),
            base_model=base_model,
        )

    if fmt != "gguf":
        return {"error": f"unsupported format: {fmt}"}
    quant = (body.get("quant") or DEFAULT_QUANT).upper()
    if quant not in SUPPORTED_QUANTS:
        return {"error": f"unsupported quant: {quant}",
                "supported": sorted(SUPPORTED_QUANTS)}

    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return {"error": "run has no output_path; training did not finish"}

    merged_dir = os.path.join(output_path, "merged")
    if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
        if not auto_merge:
            return {"error": "run has not been merged; POST /merge first, "
                             "or set auto_merge=true in the request body"}
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
            return {"error": f"auto-merge failed: {e}", "status": "skipped"}
        except Exception as e:
            log.exception("auto-merge failed")
            return {"error": f"auto-merge failed: {e}", "status": "skipped"}

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


@router.get("/projects/{pid}/runs/{rid}/exports")
async def list_run_exports(pid: str, rid: str):
    return db.list_exports_for_run(rid)


@router.get("/projects/{pid}/exports")
async def list_project_exports(pid: str, limit: int = 100):
    return db.list_exports_for_project(pid, limit=limit)


def _export_worker(eid: str, merged_dir: str, out_path: str, quant: str) -> None:
    """Background GGUF export worker.

    Workflow:
    - f16 / bf16 / f32 / Q8_0: one step via convert_hf_to_gguf.py --outtype.
    - everything else: two steps:
        1. convert HF merged dir -> fp16 GGUF
        2. llama-quantize fp16 -> target quant

    FTS_SKIP_EXPORT=1 short-circuits with a 1-byte marker so tests can
    assert the workflow end-to-end without spinning up a real conversion.
    """
    intermediate_path = ""
    try:
        db.mark_export_running(eid)
        # Test / CI short-circuit
        if os.environ.get("FTS_SKIP_EXPORT") == "1":
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\x00")
            size = os.path.getsize(out_path)
            db.mark_export_done(eid, output_path=out_path, size_bytes=size,
                                size_human=_human_size(size),
                                intermediate_path="")
            return

        convert_script = _find_convert_script()
        if not convert_script:
            raise RuntimeError(
                "convert_hf_to_gguf.py not found. Install llama.cpp on this "
                "host: git clone https://github.com/ggerganov/llama.cpp && "
                "pip install -r llama.cpp/requirements/"
                "requirements-convert_hf_to_gguf.txt"
            )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        single_step_quants = {"f16": "f16", "bf16": "bf16",
                              "f32": "f32", "Q8_0": "q8_0"}
        if quant in single_step_quants:
            outtype = single_step_quants[quant]
            # convert_hf_to_gguf.py takes the model dir as a positional
            # `[model]` and the output path via `--outfile OUTFILE`. Earlier
            # code passed outfile as a 2nd positional which the CLI rejected
            # with 'unrecognized arguments'.
            # Use sys.executable (the venv python the worker is running in)
            # — `python3` on PATH might be a different interpreter that's
            # missing sentencepiece / torch / etc. (caught on fan-dragon
            # 2026-09-10 when Qwen export died with ModuleNotFoundError
            # on `from sentencepiece import SentencePieceProcessor`).
            cmd = [sys.executable, convert_script, merged_dir,
                   "--outfile", out_path,
                   "--outtype", outtype]
            log.info("export single-step: %s", " ".join(cmd))
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, check=False,
            )
            if r.returncode != 0:
                err_tail = (r.stderr or r.stdout or "")[-1000:]
                raise RuntimeError(
                    f"convert_hf_to_gguf failed (rc={r.returncode}): {err_tail}"
                )
        else:
            quantize_bin = _find_llama_tool("llama-quantize")
            if not quantize_bin:
                raise RuntimeError(
                    f"llama-quantize binary not found (needed for quant={quant}). "
                    "Build llama.cpp: cd llama.cpp && cmake -B build && "
                    "cmake --build build --config Release"
                )
            fp16_path = out_path.replace(f"-{quant}.gguf", "-fp16.gguf")
            cmd1 = [sys.executable, convert_script, merged_dir,
                    "--outfile", fp16_path,
                    "--outtype", "f16"]
            log.info("export step 1 (HF -> fp16): %s", " ".join(cmd1))
            r1 = subprocess.run(
                cmd1, capture_output=True, text=True, timeout=3600, check=False,
            )
            if r1.returncode != 0:
                err_tail = (r1.stderr or r1.stdout or "")[-1000:]
                raise RuntimeError(
                    f"convert_hf_to_gguf to fp16 failed (rc={r1.returncode}): "
                    f"{err_tail}"
                )
            intermediate_path = fp16_path

            cmd2 = [quantize_bin, fp16_path, out_path, quant]
            log.info("export step 2 (quantize -> %s): %s", quant,
                     " ".join(cmd2))
            r2 = subprocess.run(
                cmd2, capture_output=True, text=True, timeout=3600, check=False,
            )
            if r2.returncode != 0:
                err_tail = (r2.stderr or r2.stdout or "")[-1000:]
                # Best-effort cleanup of the intermediate
                try:
                    os.unlink(fp16_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"llama-quantize failed (rc={r2.returncode}): {err_tail}"
                )

        if not os.path.isfile(out_path):
            raise RuntimeError(f"output GGUF not found after conversion: {out_path}")
        size = os.path.getsize(out_path)
        db.mark_export_done(eid, output_path=out_path, size_bytes=size,
                            size_human=_human_size(size),
                            intermediate_path=intermediate_path)
    except Exception as e:
        log.exception("export failed")
        try:
            db.mark_export_failed(eid, str(e))
        except Exception:
            log.debug("mark_export_failed secondary failure", exc_info=True)