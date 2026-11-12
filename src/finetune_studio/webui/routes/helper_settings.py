"""Settings -> AI assistant -> the local model: which installed GGUF it is, the one-click recommended download, and a
real test (load, ask, unload).

API connections, the main assistant and the jobs live in ``assistant_settings``; this module only manages the local
connection (``local-default``).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from finetune_studio.models.helper import (
    DEFAULT_HELPER_LABEL,
    DEFAULT_HELPER_PROVIDER_ID,
    RECOMMENDED_LOCAL_HELPER,
    recommended_helper_paths,
)
from finetune_studio.models.manager import get_manager
from finetune_studio.webui.engine_guard import ENGINE_LOCK

router = APIRouter()
log = logging.getLogger(__name__)


def _state() -> dict[str, Any]:
    from finetune_studio.webui.routes.assistant_settings import (
        _state as assistant_state,
    )

    return assistant_state()


@router.post("/api/settings/helper/test")
async def test_local_helper():
    """Load the local model, ask it for "pong" and unload it again (API connections: /api/settings/connections/test)."""
    return await _test_local_helper()


async def _test_local_helper() -> JSONResponse:
    """Load the local helper GGUF, ask for "pong", and unload it again if this test was what loaded it.

    Testing the local seat with the stored API provider (the old behaviour) reported "Works" for a GGUF that was
    never touched. A model someone else loaded is never displaced: the test answers 409 and names it."""
    from finetune_studio.models.helper import (
        helper_missing_message,
        missing_gguf_for_provider,
    )
    from finetune_studio.smart_brain.model_cache import release_brain_models
    from finetune_studio.webui.app import inference_engine

    missing = missing_gguf_for_provider(DEFAULT_HELPER_PROVIDER_ID)
    if missing:
        return JSONResponse({"ok": False, "code": "helper_missing", "detail": helper_missing_message(missing)},
                            status_code=400)
    async with ENGINE_LOCK:
        mgr = get_manager()
        was_loaded = (mgr.active() or {}).get("id") == DEFAULT_HELPER_PROVIDER_ID
        if not was_loaded and getattr(inference_engine, "model", None) is not None:
            held = Path(str(getattr(inference_engine, "model_path", "") or "another model")).name
            return JSONResponse({"ok": False, "code": "gpu_busy",
                                 "detail": f"{held} is loaded on the GPU; unload it (Inference page) to test the local helper"},
                                status_code=409)

        def _ping() -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                if not was_loaded:
                    release_brain_models("local helper test")
                    mgr.load(DEFAULT_HELPER_PROVIDER_ID)
                t1 = time.perf_counter()
                reply = mgr.chat([{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=64, temperature=0)
            except Exception as exc:  # a failed load/chat is the test's answer, reported as 502
                log.exception("local helper test failed")
                return {"ok": False, "detail": f"the local helper failed: {exc}", "code": "provider_failed"}
            finally:
                if not was_loaded:
                    mgr.unload()
            return {"ok": True, "reply": reply.strip()[:80], "latency_ms": int((time.perf_counter() - t1) * 1000),
                    "load_ms": 0 if was_loaded else int((t1 - t0) * 1000)}

        result = await asyncio.to_thread(_ping)
    return JSONResponse(result, status_code=200 if result["ok"] else 502)


@router.post("/api/settings/helper/local")
async def use_local_helper(request: Request):
    """Point the local model (``local-default``) at an installed GGUF; a resident old one is unloaded. Whether it is the
    main assistant is a separate choice (``PUT /api/settings/assistant``)."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    path = str((body or {}).get("path") or "").strip() if isinstance(body, dict) else ""
    resolved = Path(path).expanduser()
    if not path or resolved.suffix.lower() != ".gguf" or not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"not an installed GGUF file: {path or '(empty)'}")
    if resolved.name.lower().startswith("mmproj"):
        raise HTTPException(status_code=400, detail="that file is a vision projector (mmproj), not a model")
    recommended_gguf, _ = recommended_helper_paths()
    label = DEFAULT_HELPER_LABEL if resolved.resolve() == recommended_gguf.resolve() else f"Helper · {resolved.stem}"
    async with ENGINE_LOCK:
        mgr = get_manager()
        if (mgr.active() or {}).get("id") == DEFAULT_HELPER_PROVIDER_ID:
            from finetune_studio.models.llama_loader import unload_all_models
            await asyncio.to_thread(unload_all_models)
        await asyncio.to_thread(
            mgr.upsert_provider, id=DEFAULT_HELPER_PROVIDER_ID, name=label,
            kind="local_gguf", model_id=str(resolved),
        )
    return await asyncio.to_thread(_state)


@router.post("/api/settings/helper/recommended/download")
async def download_recommended_helper():
    """Start Model library downloads for the recommended helper's files that are missing (GGUF + vision mmproj).

    Returns the download job ids; the page polls ``/api/hf/download/progress`` and then uses the GGUF."""
    from finetune_studio.jobs import launch
    from finetune_studio.webui import download_jobs
    from finetune_studio.webui.errors import ApiError

    rec = RECOMMENDED_LOCAL_HELPER
    gguf, mmproj = recommended_helper_paths()
    jobs = []
    for path in (gguf, mmproj):
        if path.is_file():
            continue
        try:
            row = await asyncio.to_thread(download_jobs.start, repo_id=rec.repo_id, filename=path.name,
                                          revision=rec.revision, dest=str(path.parent))
        except launch.JobRejected as e:
            raise ApiError(e.status if e.status >= 400 else 503, f"could not start the download: {e}",
                           code=e.code or None) from e
        jobs.append({"file": path.name, "job_id": row["id"]})
    return {"ok": True, "jobs": jobs, "path": str(gguf), "already_installed": not jobs}
