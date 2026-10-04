"""Comparison tab (side-by-side model output) routes."""
import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()


async def _json_object(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


@router.post("/compare/load")
async def compare_load(request: Request):
    """Load a model for comparison."""
    body = await _json_object(request)
    name = body.get("name", "model")
    path = body.get("path", "")
    if not path:
        return JSONResponse({"error": "No path provided"}, status_code=400)
    try:
        from finetune_studio.benchmarks.comparison import comparator
        await asyncio.to_thread(comparator.load_model, name, path)
        return {"status": "loaded", "name": name, "models": comparator.model_names()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/compare/run")
async def compare_run(request: Request):
    """Run comparison on test suite."""
    body = await _json_object(request)
    test_suite = body.get("test_suite", [])
    config = body.get("config", {"max_tokens": 512, "temperature": 0.7})

    if not test_suite:
        return JSONResponse({"error": "No test suite provided"}, status_code=400)

    from finetune_studio.benchmarks.comparison import NoModelsLoadedError, comparator
    try:
        result = await asyncio.to_thread(comparator.run_comparison, test_suite, config)
    except NoModelsLoadedError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except (KeyError, TypeError, AttributeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"invalid test_suite case (need name + messages [+ expected]): {e!r}",
        ) from None
    return result


@router.post("/compare/cleanup")
async def compare_cleanup():
    """Unload all comparison models."""
    from finetune_studio.benchmarks.comparison import comparator
    await asyncio.to_thread(comparator.cleanup)
    return {"status": "cleaned", "models": []}
