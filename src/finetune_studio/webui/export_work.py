"""The work an export job performs on its worker thread.

``export_jobs`` owns the lifecycle (thread, lock, terminal state); the
functions here own the export itself and the ``model_exports`` rows it
produces. Each ``*_work`` factory returns a callable run inside the job's bound
:class:`~finetune_studio.training.export_job.ExportContext`. A work callable
finishes its rows with ``db.mark_export_done`` and **raises** on failure — the
message is what the user sees, so it must carry the real cause (stderr tail
from llama.cpp, the missing-converter hint, ...).
"""

from __future__ import annotations

import logging
import os
import shutil
from time import time

from finetune_studio import db
from finetune_studio.training import export_job
from finetune_studio.training.export_response import (
    ExportResult,
    dir_size_bytes,
    human_size,
)
from finetune_studio.webui.export_jobs import Work

log = logging.getLogger(__name__)

DEFAULT_QUANT = "Q4_K_M"


def refresh_registry_quietly() -> None:
    """Make new exports visible to Chat/Inference without a restart."""
    try:
        from finetune_studio.webui.routes.models import refresh_model_registry

        refresh_model_registry()
    except Exception:
        log.debug("model registry refresh failed", exc_info=True)


def _release_gpu_caches() -> None:
    """Export merges/quantises on the GPU the RAG model cache may still hold."""
    export_job.report(export_job.PHASE_PREPARING, "freeing GPU memory")
    from finetune_studio.data.rag_portable.model_cache import release_rag_models

    release_rag_models("export")


def _failure_text(payload: ExportResult) -> str:
    text = payload.error or payload.reason or payload.message or "export failed"
    if payload.missing:
        text += f" (missing: {', '.join(payload.missing)})"
    return text


def multi_export_work(
    eid: str,
    *,
    run: dict,
    fmt: str,
    quants: list[str] | None,
    force: bool,
    base_model: str | None,
) -> Work:
    """GGUF (multi-quant) / abliterated / merged export of ``run``."""

    def work(ctx: export_job.ExportContext) -> None:
        from finetune_studio.training.run_export import export_trained_run

        _release_gpu_caches()
        raw = export_trained_run(
            run, fmt=fmt, quants=quants, force=force, base_model=base_model,
        )
        # Always coerce through ExportResult so numpy/tensors cannot break the
        # row writer after a successful GPU merge/abliteration.
        payload = ExportResult.from_raw(raw)
        if not payload.ok or payload.error or payload.status == "failed":
            raise RuntimeError(_failure_text(payload))
        ctx.phase(export_job.PHASE_FINALIZING, "registering the exported files")
        _register_artifacts(eid, run, fmt, payload)
        refresh_registry_quietly()

    return work


def _register_artifacts(eid: str, run: dict, fmt: str, payload: ExportResult) -> None:
    """Finish the job's row; one extra row per additional GGUF quant.

    A multi-quant request yields one file per quant (``payload.files[i]`` <->
    ``payload.quants[i]``, see ``gguf_convert.verify_gguf_artifacts``). The
    job's own row becomes the first artifact's row; the rest get rows of their
    own, each sized from its own file.
    """
    art = payload.artifact_path()
    if not art:
        raise RuntimeError("export reported success but produced no output path")
    pid, rid = run.get("project_id") or "", run.get("id") or ""
    per_quant = (
        fmt == "gguf"
        and payload.quants
        and payload.files
        and len(payload.files) == len(payload.quants)
    )
    if per_quant:
        pairs = list(zip(payload.quants, payload.files, strict=True))
        for i, (label, path) in enumerate(pairs):
            size = os.path.getsize(path) if os.path.isfile(path) else 0
            row_id = eid
            if i == 0:
                db.update_export(eid, quant=str(label))
            else:
                row_id = db.create_export(
                    project_id=pid, run_id=rid, format=fmt, quant=str(label),
                )["id"]
            db.mark_export_done(
                row_id, output_path=path, size_bytes=size,
                size_human=human_size(size),
            )
        return
    quant_label = (
        (payload.quants[0] if payload.quants else None)
        or payload.quant
        or ("safetensors" if fmt != "gguf" else DEFAULT_QUANT)
    )
    size = dir_size_bytes(art)
    db.update_export(eid, quant=str(quant_label))
    db.mark_export_done(
        eid, output_path=art, size_bytes=size, size_human=human_size(size),
    )


def run_single_quant_gguf(
    eid: str,
    merged_dir: str,
    out_path: str,
    quant: str,
) -> None:
    """Convert ``merged_dir`` to one quant at ``out_path``; raise on failure.

    ``FTS_SKIP_EXPORT=1`` short-circuits with a 1-byte marker for tests.
    """
    from finetune_studio.training.gguf_convert import (
        convert_merged_to_gguf,
        normalize_gguf_quant,
    )

    gguf_dir = os.path.dirname(out_path)
    result = convert_merged_to_gguf(
        merged_dir, gguf_dir, [normalize_gguf_quant(quant)], force=True,
    )
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "GGUF conversion failed")
    # Prefer the exact outfile the caller queued; fall back to converter path.
    final_path = out_path
    files = result.get("files") or []
    if files:
        # Converter writes model-{quant}.gguf; copy if the queued name differs.
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
        size_human=human_size(size),
        intermediate_path=result.get("intermediate_path") or "",
    )
    refresh_registry_quietly()


def single_quant_work(
    eid: str,
    *,
    run: dict,
    base_model: str | None,
    auto_merge: bool,
    out_path: str,
    quant: str,
) -> Work:
    """Legacy ``{"quant": ...}`` GGUF export: merge if needed, one quant."""

    def work(_ctx: export_job.ExportContext) -> None:
        output_path = (run.get("output_path") or "").strip()
        merged_dir = os.path.join(output_path, "merged")
        _release_gpu_caches()
        if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
            if not auto_merge:
                raise RuntimeError(
                    "run has not been merged; POST /merge first, "
                    "or set auto_merge=true in the request body"
                )
            from finetune_studio.training.run_export import ensure_merged_for_export

            try:
                merged = ensure_merged_for_export(
                    run, base_model=base_model, force=False,
                )
            except ValueError as e:
                raise RuntimeError(f"auto-merge failed: {e}") from e
            merged_dir = merged.get("merged_path") or merged_dir
            log.info("auto-merge for export: %s", merged_dir)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        run_single_quant_gguf(eid, merged_dir, out_path, quant)

    return work


def imatrix_work(
    eid: str,
    *,
    run: dict,
    merged_dir: str,
    output_dir: str,
    imatrix_path: str,
    quants: list[str],
    bits: int,
    group_size: int,
) -> Work:
    """imatrix-weighted GGUF quantization of the run's merged model."""

    def work(ctx: export_job.ExportContext) -> None:
        from finetune_studio.db.connection import cursor, new_id
        from finetune_studio.training.advanced_quant import quantize_gguf_imatrix

        result = quantize_gguf_imatrix(
            model_path=merged_dir, output_dir=output_dir,
            imatrix_path=imatrix_path, quants=quants,
        )
        if result.get("error"):
            raise RuntimeError(str(result["error"]))
        exported = result.get("exported") or {}
        failed = {q: v["error"] for q, v in exported.items() if v.get("error")}
        if not exported or len(failed) == len(exported):
            detail = "; ".join(f"{q}: {e}" for q, e in failed.items())
            raise RuntimeError(f"imatrix quantization produced no files ({detail})")
        ctx.phase(export_job.PHASE_FINALIZING, "registering the quantized files")
        size = int(result.get("size_bytes") or 0)
        with cursor() as c:
            c.execute(
                "INSERT INTO quant_exports (id, run_id, project_id, model_path, "
                "output_path, method, bits, group_size, size_bytes, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id(), run["id"], run.get("project_id", ""), merged_dir,
                 output_dir, "imatrix", bits, group_size, size, "done", time()),
            )
        note = f"{len(failed)} of {len(exported)} quants failed: {failed}" if failed else ""
        db.update_export(eid, error=note[:1000])
        db.mark_export_done(
            eid, output_path=result.get("output_dir") or output_dir,
            size_bytes=size, size_human=human_size(size),
        )
        if failed:
            # mark_done leaves ``error`` alone, so the partial-failure note stays.
            log.warning("imatrix export %s: %s", eid, note)
        refresh_registry_quietly()

    return work
