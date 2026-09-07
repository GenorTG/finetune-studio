"""Models tab — list, download, configure models."""

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from finetune_studio.models.loader import load_model_info

router = APIRouter()

@router.get("/list")
async def list_models():
    from finetune_studio.webui.app import discovered_models
    return [{"name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture,
            "vision": getattr(m, "vision", False)} for m in discovered_models]

@router.get("/count")
async def count_models():
    from finetune_studio.webui.app import discovered_models
    return PlainTextResponse(str(len(discovered_models)))

@router.get("/info")
async def model_info(path: str):
    return load_model_info(path)

@router.post("/refresh")
async def refresh_models():
    from finetune_studio.config import settings
    from finetune_studio.models.registry import scan_models
    from finetune_studio.webui.app import discovered_models
    discovered_models.clear()
    discovered_models.extend(scan_models(settings.model_dirs))
    return {"count": len(discovered_models)}


@router.post("/load")
async def load_model_endpoint(request: Request):
    """Load a model into the global inference engine.

    Accepts either {"path": "..."} (inference page) or {"model_path": "..."}
    (legacy chat_v2 form). All inference parameters have sensible defaults
    so the inference-page call Just Works.
    """
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    model_path = body.get("path") or body.get("model_path") or ""
    if not model_path:
        return {"error": "No model path provided"}
    try:
        inference_engine.load(
            model_path,
            n_ctx=body.get("n_ctx", 4096),
            n_gpu_layers=body.get("n_gpu_layers", 99),
            n_batch=body.get("n_batch", 512),
            mmap=body.get("mmap", True),
            mlock=body.get("mlock", False),
            n_threads=body.get("n_threads"),
            flash_attn=body.get("flash_attn", True),
            seed=body.get("seed"),
            rope_freq_base=body.get("rope_freq_base", 0.0),
            rope_freq_scale=body.get("rope_freq_scale", 0.0),
        )
        vision = getattr(inference_engine, "vision", False)
        return {"status": "loaded", "model": model_path, "vision": vision}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.post("/unload")
async def unload_model_endpoint():
    """Manually unload the currently loaded model."""
    from finetune_studio.webui.app import inference_engine
    try:
        inference_engine.unload()
        return {"status": "unloaded"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ── /api/inference/* aliases (cleaner URL namespace) ─────────────
# These mirror the routes exposed under /api/chat-v2/* so that the
# inference page can use the natural /api/inference/{status,load,...}
# paths. Single source of truth stays in chat_v2.

inference_router = APIRouter()


@inference_router.get("/status")
async def inference_status():
    from finetune_studio.webui.app import inference_engine
    from finetune_studio.testing.inference import IDLE_TIMEOUT
    loaded = inference_engine.model is not None
    return {
        "loaded": loaded,
        "model": inference_engine.model_path if loaded else None,
        "vision": getattr(inference_engine, "vision", False) if loaded else False,
        "is_gguf": inference_engine.is_gguf if loaded else False,
        "idle_seconds": inference_engine.idle_seconds,
        "idle_timeout": IDLE_TIMEOUT,
        "auto_unload_remaining": (
            max(0, IDLE_TIMEOUT - inference_engine.idle_seconds)
            if loaded and IDLE_TIMEOUT > 0
            else None
        ),
    }


@inference_router.post("/load")
async def inference_load(request: Request):
    """Alias for /api/models/load — same handler."""
    return await load_model_endpoint(request)


@inference_router.post("/unload")
async def inference_unload():
    """Alias for /api/models/unload."""
    return await unload_model_endpoint()


@inference_router.post("/chat")
async def inference_chat(request: Request):
    """Generate a chat completion using the global inference engine."""
    from finetune_studio.webui.app import inference_engine
    from finetune_studio.webui.thinking import split_thinking
    from fastapi.responses import JSONResponse

    if inference_engine.model is None:
        return JSONResponse(
            {"error": "No model loaded. Click 'Load model' first."}, status_code=409
        )

    body = await request.json()
    messages = body.get("messages") or []
    # Prepend system prompt if provided
    sp = body.get("system_prompt", "").strip()
    if sp:
        messages = [{"role": "system", "content": sp}] + messages

    try:
        response = inference_engine.generate(
            messages,
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            top_k=body.get("top_k", 40),
            repeat_penalty=body.get("repeat_penalty", 1.1),
        )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)

    # inference_engine.generate() returns a string — split thinking from response
    if isinstance(response, dict):
        resp_text = response.get("response", "")
        thinking_text = response.get("thinking", "")
    else:
        parts = split_thinking(str(response))
        resp_text = parts["response"]
        thinking_text = parts["thinking"]

    return {
        "response": resp_text,
        "thinking": thinking_text,
        "vision": getattr(inference_engine, "vision", False),
    }


@inference_router.post("/memory-estimate")
async def inference_memory_estimate(request: Request):
    """Estimate VRAM needed for a model with given loader params."""
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    model_path = body.get("model_path") or body.get("path") or ""
    if not model_path:
        return {"error": "No model_path provided"}
    try:
        est = inference_engine.estimate_memory(
            model_path,
            n_ctx=body.get("n_ctx", 4096),
            n_gpu_layers=body.get("n_gpu_layers", 99),
        )
        return est
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
