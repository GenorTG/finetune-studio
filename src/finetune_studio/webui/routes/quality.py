"""GUI routes for data quality + training-data CLI commands.

WHAT THIS FILE DOES
===================
Exposes the data-quality and training-quality CLI commands to the WebUI:
  - analyze          → /api/data/analyze
  - augment          → /api/data/augment
  - optimize         → /api/data/optimize
  - validate-hallucination → /api/data/hallucination-check
  - convert          → /api/data/convert

Why a dedicated file?
  The CLI commands existed but had no GUI equivalent. This file closes that
  gap so the WebUI is feature-complete without requiring the user to drop
  into the terminal.
"""

from __future__ import annotations

import json as json_mod
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/data", tags=["data-quality"])


class DataJobRequest(BaseModel):
    path: str = "data/training.jsonl"
    output: str | None = None


class DataJobResponse(BaseModel):
    status: str
    command: str
    path: str
    result: dict[str, Any] | None = None
    error: str | None = None


def _load_jsonl(path: str) -> list[dict]:
    """Read one JSON object per line, skipping malformed lines.

    Matches the loading behavior of the equivalent ``fts`` CLI commands
    (``cli/commands/{optimize,augment,validate_hallucination}.py``) so the
    WebUI and CLI paths see the same data for the same file.
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json_mod.loads(line))
            except json_mod.JSONDecodeError:
                pass
    return data


# ── CLI: fts analyze ─────────────────────────────────────────────────────
@router.post("/analyze", response_model=DataJobResponse)
def data_analyze(req: DataJobRequest) -> DataJobResponse:
    """Analyze training-data quality (length, dup ratio, persona consistency)."""
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(req.path)
        return DataJobResponse(status="ok", command="analyze", path=req.path, result=report)
    except Exception as exc:  # noqa: BLE001 — route boundary must surface errors
        return DataJobResponse(status="error", command="analyze", path=req.path, error=str(exc))


# ── CLI: fts augment ─────────────────────────────────────────────────────
@router.post("/augment", response_model=DataJobResponse)
def data_augment(req: DataJobRequest) -> DataJobResponse:
    """Augment training data to address weaknesses identified by analyze."""
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.training.data_augmentation import DataAugmenter
        from finetune_studio.training.data_quality import DataQualityAnalyzer

        data = _load_jsonl(req.path)
        analysis = DataQualityAnalyzer().analyze(req.path)
        weaknesses = []
        for issue in analysis.get("issues", []):
            issue_type = issue.get("type", "")
            if "language" in issue_type:
                weaknesses.append("language_balance")
            elif "hallucination" in issue_type:
                weaknesses.append("hallucination_guard")
            elif "empty" in issue_type:
                weaknesses.append("refusal")
        weaknesses = list(set(weaknesses)) or ["knowledge", "refusal"]

        augmented = DataAugmenter().augment_dataset(data, weaknesses)

        output_path = req.output or str(Path(req.path).with_suffix(".augmented.jsonl"))
        with open(output_path, "w") as f:
            f.writelines(
                json_mod.dumps(item, ensure_ascii=False) + "\n" for item in augmented
            )

        result = {
            "weaknesses": weaknesses,
            "input_count": len(data),
            "output_count": len(augmented),
            "output_path": output_path,
        }
        return DataJobResponse(status="ok", command="augment", path=req.path, result=result)
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="augment", path=req.path, error=str(exc))


# ── CLI: fts optimize ────────────────────────────────────────────────────
@router.post("/optimize", response_model=DataJobResponse)
def data_optimize(req: DataJobRequest) -> DataJobResponse:
    """Recommend training hyperparameters based on dataset characteristics."""
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer

        data = _load_jsonl(req.path)
        optimizer = TrainingConfigOptimizer()
        recommendations = optimizer.analyze_and_recommend(data, {})
        result = {
            "recommendations": [
                {
                    "parameter": r.parameter,
                    "current": r.current_value,
                    "recommended": r.recommended_value,
                    "reason": r.reason,
                    "priority": r.priority,
                }
                for r in recommendations
            ],
        }
        return DataJobResponse(status="ok", command="optimize", path=req.path, result=result)
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="optimize", path=req.path, error=str(exc))


# ── CLI: fts validate-hallucination ─────────────────────────────────────
@router.post("/hallucination-check", response_model=DataJobResponse)
def data_hallucination_check(req: DataJobRequest) -> DataJobResponse:
    """Scan training data for hallucination risk patterns."""
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.training.hallucination_guard import TrainingDataValidator

        data = _load_jsonl(req.path)
        report = TrainingDataValidator().validate_dataset(data)
        return DataJobResponse(
            status="ok", command="validate-hallucination", path=req.path, result=report,
        )
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="validate-hallucination", path=req.path, error=str(exc))


# ── CLI: fts convert ─────────────────────────────────────────────────────
class ConvertRequest(BaseModel):
    path: str
    target_format: str = "jsonl"
    output: str | None = None
    system_prompt: str = ""


@router.post("/convert", response_model=DataJobResponse)
def data_convert(req: ConvertRequest) -> DataJobResponse:
    """Convert training data between formats (csv/json/jsonl)."""
    src = Path(req.path)
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.data.converter import (
            csv_to_jsonl,
            json_to_jsonl,
            jsonl_to_json,
        )

        output_path = req.output or str(src.with_suffix(f".{req.target_format}"))
        if src.suffix == ".jsonl" and req.target_format == "json":
            jsonl_to_json(str(src), output_path)
        elif src.suffix == ".json" and req.target_format == "jsonl":
            json_to_jsonl(str(src), output_path)
        elif src.suffix == ".csv" and req.target_format == "jsonl":
            csv_to_jsonl(str(src), output_path, system_prompt=req.system_prompt)
        else:
            return DataJobResponse(
                status="error", command="convert", path=req.path,
                error=f"Cannot convert {src.suffix} -> .{req.target_format}",
            )

        result = {"output_path": output_path}
        return DataJobResponse(status="ok", command="convert", path=req.path, result=result)
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="convert", path=req.path, error=str(exc))