"""Models tab — list, download, configure models."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from finetune_studio.models.loader import load_model_info

router = APIRouter()

@router.get("/list")
async def list_models():
    from finetune_studio.webui.app import discovered_models
    return [{"name": m.name, "path": m.path, "format": m.format,
            "size_gb": m.size_gb, "architecture": m.architecture} for m in discovered_models]

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
