"""Settings -> AI assistant -> Jobs -> "Reading pictures": the test reading and the local GGUFs that can see.

The seat itself (off by default, the main assistant, a connection or a local vision GGUF) is saved through
``PUT /api/settings/assistant/jobs/vision`` (``assistant_settings``). Nothing is sent to an API until the user picks one.
"""
from __future__ import annotations

import asyncio
import io
import random
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from finetune_studio.models import vision

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
        return JSONResponse({"ok": False, "detail": "reading pictures is off"}, status_code=400)
    code = str(random.randint(100000, 999999))

    def _read() -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            reader = vision.Reader(seat)
        except vision.VisionError as e:
            return {"ok": False, "detail": str(e), "code": "vision_failed"}
        try:
            reply = reader.read(_probe_image(code), "Settings test", "test image")
        except vision.VisionError as e:
            return {"ok": False, "detail": str(e), "code": "vision_failed"}
        finally:
            reader.close()
        ok = code in reply.replace(" ", "")
        return {"ok": ok, "reply": reply[:200], "latency_ms": int((time.perf_counter() - t0) * 1000),
                **({} if ok else {"detail": f"the model did not read the code {code} in the test picture",
                                 "code": "vision_misread"})}

    result = await asyncio.to_thread(_read)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)
