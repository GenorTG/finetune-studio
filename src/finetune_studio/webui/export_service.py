"""Shared request and job lifecycle used by the Export page and ``fts export``."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from finetune_studio import db
from finetune_studio.training.export_response import ExportResult
from finetune_studio.training.run_export import validate_export_request
from finetune_studio.webui import export_jobs

SUPPORTED_QUANTS = {
    "f16", "bf16", "f32", "Q8_0", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
    "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_XXS",
    "IQ3_S", "IQ2_XXS", "IQ2_XS",
}
DEFAULT_QUANT = "Q4_K_M"


@dataclass(frozen=True)
class ExportServiceResult:
    status_code: int
    body: dict[str, Any]


def _failure(detail: str, *, status_code: int = 400, **extra: object) -> ExportServiceResult:
    return ExportServiceResult(status_code, {
        "ok": False, "status": "failed", "detail": detail, "code": "export_refused", **extra,
    })


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def submit_export(project_id: str, run_id: str, body: dict[str, Any]) -> ExportServiceResult:
    """Validate and queue one export with the same contract as the WebUI route."""
    fmt = str(body.get("format") or "gguf").lower()
    auto_merge = bool(body.get("auto_merge", True))
    base_model_raw = body.get("base_model")
    base_model = (str(base_model_raw).strip() or None) if base_model_raw is not None else None
    run = db.get_run(run_id)
    if not run:
        return ExportServiceResult(404, {"ok": False, "status": "failed", "detail": "run not found"})
    if run.get("project_id") and project_id and run["project_id"] != project_id:
        return ExportServiceResult(400, {
            "ok": False, "status": "failed", "detail": "run does not belong to this project",
        })

    imatrix = body.get("imatrix", False)
    if not isinstance(imatrix, bool):
        return _failure("imatrix must be true or false")
    force = bool(body.get("force", False))
    use_multi = (
        "quants" in body or imatrix or fmt in ("abliterated", "merged", "awq", "gptq", "heretic")
        or force and "quant" not in body
    )
    if use_multi:
        quants_raw = body.get("quants")
        quants = [str(quant) for quant in quants_raw] if isinstance(quants_raw, list) else None
        refused = validate_export_request(run, fmt=fmt, quants=quants, force=force, imatrix=imatrix)
        if refused is not None:
            payload = ExportResult.from_raw(refused).model_dump(exclude_none=False)
            payload.pop("error", None)
            return _failure(str(refused.get("error") or "export refused"), **payload)
        label = (quants[0] if quants else None) or (DEFAULT_QUANT if fmt == "gguf" else "safetensors")
        return _queue(
            project_id, run_id, fmt=fmt, quant=label, quants=quants,
            work=("multi_export_work", {
                "run": run, "fmt": fmt, "quants": quants, "force": force,
                "base_model": base_model, "imatrix": imatrix,
            }),
        )

    if fmt != "gguf":
        return _failure(f"unsupported format: {fmt}", format=fmt)
    quant_raw = str(body.get("quant") or DEFAULT_QUANT)
    quant = {value.upper(): value for value in SUPPORTED_QUANTS}.get(quant_raw.upper())
    if quant is None:
        return _failure(f"unsupported quant: {quant_raw}", supported=sorted(SUPPORTED_QUANTS))

    output_path = (run.get("output_path") or "").strip()
    if not output_path:
        return _failure("run has no output_path; training did not finish")
    merged_dir = os.path.join(output_path, "merged")
    if not auto_merge and (not os.path.isdir(merged_dir) or not os.listdir(merged_dir)):
        return _failure("run has not been merged; POST /merge first, or set auto_merge=true in the request body")
    refused = validate_export_request(run, fmt=fmt)
    if refused is not None:
        return _failure(str(refused["error"]))

    base = os.path.basename(base_model or run.get("base_model") or "model")
    output_file = os.path.join(output_path, "gguf", f"{_safe_name(base)}-{quant}.gguf")
    return _queue(
        project_id, run_id, fmt=fmt, quant=quant, quants=[quant],
        extra={"output_path": output_file},
        work=("single_quant_work", {
            "run": run, "base_model": base_model, "auto_merge": auto_merge,
            "out_path": output_file, "quant": quant,
        }),
    )


def _queue(
    project_id: str,
    run_id: str,
    *,
    fmt: str,
    quant: str,
    quants: list[str] | None,
    work: export_jobs.WorkSpec,
    extra: dict[str, Any] | None = None,
) -> ExportServiceResult:
    try:
        row = export_jobs.start_job(
            project_id=project_id, run_id=run_id, fmt=fmt, quant=quant, quants=quants, work=work,
        )
    except export_jobs.ExportBusy as busy:
        return ExportServiceResult(409, {
            "ok": False, "status": "busy", "detail": str(busy),
            "active_export_id": busy.active.get("id"), "active_project_id": busy.active.get("project_id"),
        })
    return ExportServiceResult(202, {
        "ok": True, "export_id": row["id"], "status": "queued",
        "format": fmt, "quant": quant, "quants": quants or [], **(extra or {}),
    })


def row_view(row: dict[str, Any]) -> dict[str, Any]:
    """Export record plus derived progress fields shared by UI and CLI."""
    try:
        quants = json.loads(row.get("quants_json") or "[]")
    except (TypeError, ValueError):
        quants = []
    return {**row, "quants": quants, "server_now": time.time(), "active": export_jobs.is_active(row["id"])}


def get_export(project_id: str, export_id: str) -> ExportServiceResult:
    row = db.get_export(export_id)
    if not row or row.get("project_id") != project_id:
        return ExportServiceResult(404, {"detail": "not found"})
    return ExportServiceResult(200, row_view(row))


def list_exports(project_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    return db.list_exports_for_project(project_id, limit=limit)


def cancel_export(project_id: str, export_id: str) -> ExportServiceResult:
    row = db.get_export(export_id)
    if not row or row.get("project_id") != project_id:
        return ExportServiceResult(404, {"detail": "not found"})
    if (row.get("status") or "") in export_jobs.TERMINAL_STATUSES:
        return ExportServiceResult(409, {"ok": False, "detail": f"export already {row['status']}"})
    if not export_jobs.cancel_job(export_id):
        return ExportServiceResult(409, {"ok": False, "detail": "export is not running in this server process"})
    return ExportServiceResult(200, {"ok": True, "status": "cancelling", "export_id": export_id})
