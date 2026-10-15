"""Settings -> Vision reader: the model that reads pictures in documents while mining (``models/vision.py``).

``off`` (default) keeps OCR text; ``local`` is a GGUF with an mmproj next to it; ``api`` is an OpenAI-compatible vision model
with its own row (``api-vision``) and write-only key. Nothing is sent to an API until the user seats an API model here.
"""
from __future__ import annotations

import asyncio
import io
import random
import time
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from finetune_studio.models import vision
from finetune_studio.models.manager import get_manager

router = APIRouter()


def _local_candidates() -> list[dict[str, str]]:
    """Local GGUFs that can see (an mmproj next to them), from the app's model registry."""
    from finetune_studio.webui.app import discovered_models

    out = []
    for m in discovered_models:
        path = str(getattr(m, "path", "") or "")
        if getattr(m, "vision", False) and path.lower().endswith(".gguf") and vision.mmproj_for(path):
            out.append({"path": path, "label": str(getattr(m, "name", "") or path)})
    return sorted(out, key=lambda x: x["label"].lower())


def _state() -> dict[str, Any]:
    return {**vision.describe(), "local_candidates": _local_candidates()}


async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected JSON object")
    return body


def _upsert_api_row(body: dict[str, Any]) -> None:
    mgr = get_manager()
    row = mgr.get_provider(vision.API_VISION_PROVIDER_ID) or {}
    url = str(body.get("base_url") or row.get("base_url") or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="API base URL must start with http:// or https://")
    model = str(body.get("model") or row.get("model_id") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="enter the vision model name")
    fields: dict[str, Any] = {"id": vision.API_VISION_PROVIDER_ID, "name": "Vision · API", "kind": "openai_compat",
                              "model_id": model, "base_url": url, "extra": dict(row.get("extra") or {})}
    key = str(body.get("api_key") or "").strip()
    if key:
        fields["api_key"] = key
    elif body.get("clear_key"):
        fields["api_key"] = ""
    mgr.upsert_provider(**fields)


@router.get("/api/settings/vision")
async def get_vision_settings():
    return await asyncio.to_thread(_state)


@router.put("/api/settings/vision")
async def put_vision_settings(request: Request):
    body = await _json(request)
    mode = str(body.get("mode") or "").strip().lower()
    if mode not in vision.MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {list(vision.MODES)}")

    def _apply() -> dict[str, Any]:
        if body.get("base_url") or body.get("model") or body.get("api_key") or body.get("clear_key"):
            _upsert_api_row(body)
        try:
            vision.set_seat(vision.Seat(mode=mode, local_model=str(body.get("local_model") or "").strip()))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return _state()

    return await asyncio.to_thread(_apply)


def _probe_image(code: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("RGB", (520, 140), "white")
    ImageDraw.Draw(img).text((30, 40), f"Code: {code}", fill="black", font=font)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


@router.post("/api/settings/vision/test")
async def test_vision_settings():
    """One real reading of a generated picture with a random code: proves the seated model sees, and how fast."""
    seat = vision.get_seat()
    if seat.mode == "off":
        return JSONResponse({"ok": False, "error": "the Vision reader is off"}, status_code=400)
    code = str(random.randint(100000, 999999))

    def _read() -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            reader = vision.Reader(seat)
        except vision.VisionError as e:
            return {"ok": False, "error": str(e)}
        try:
            reply = reader.read(_probe_image(code), "Settings test", "test image")
        except vision.VisionError as e:
            return {"ok": False, "error": str(e)}
        finally:
            reader.close()
        ok = code in reply.replace(" ", "")
        return {"ok": ok, "reply": reply[:200], "latency_ms": int((time.perf_counter() - t0) * 1000),
                **({} if ok else {"error": f"the model did not read the code {code} in the test picture"})}

    result = await asyncio.to_thread(_read)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)
