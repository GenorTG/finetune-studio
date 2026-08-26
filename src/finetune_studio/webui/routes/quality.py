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
        augmenter = DataAugmenter(req.path, output_path=req.output)
        result = augmenter.run()
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
        from finetune_studio.training.config_optimizer import ConfigOptimizer
        optimizer = ConfigOptimizer(req.path)
        config = optimizer.recommend()
        return DataJobResponse(status="ok", command="optimize", path=req.path, result=config)
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="optimize", path=req.path, error=str(exc))


# ── CLI: fts validate-hallucination ─────────────────────────────────────
@router.post("/hallucination-check", response_model=DataJobResponse)
def data_hallucination_check(req: DataJobRequest) -> DataJobResponse:
    """Scan training data for hallucination risk patterns."""
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.training.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard(req.path)
        report = guard.scan()
        return DataJobResponse(status="ok", command="validate-hallucination", path=req.path, result=report)
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="validate-hallucination", path=req.path, error=str(exc))


# ── CLI: fts convert ─────────────────────────────────────────────────────
class ConvertRequest(BaseModel):
    path: str
    target_format: str = "sharegpt"
    output: str | None = None


@router.post("/convert", response_model=DataJobResponse)
def data_convert(req: ConvertRequest) -> DataJobResponse:
    """Convert training data between formats (chatml, sharegpt, alpaca, etc.)."""
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from finetune_studio.compare.engine import FormatConverter
        converter = FormatConverter(source=req.path, target_format=req.target_format, output=req.output)
        result = converter.convert()
        return DataJobResponse(status="ok", command="convert", path=req.path, result=result)
    except Exception as exc:  # noqa: BLE001
        return DataJobResponse(status="error", command="convert", path=req.path, error=str(exc))