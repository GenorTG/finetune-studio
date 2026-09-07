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
