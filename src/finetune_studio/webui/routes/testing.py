"""Testing tab — run test suites, view results."""

from fastapi import APIRouter, Request

from finetune_studio.testing.suite import load_test_suite, run_suite, score_results
from finetune_studio.webui.app import inference_engine

router = APIRouter()

@router.post("/load")
async def load_model(request: Request):
    body = await request.json()
    model_path = body.get("model_path", "")
    if not model_path:
        return {"error": "No model_path"}
    try:
        inference_engine.load(model_path)
        return {"status": "loaded", "model": model_path}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

@router.post("/unload")
async def unload_model():
    inference_engine.unload()
    return {"status": "unloaded"}

@router.get("/status")
async def model_status():
    return {"loaded": inference_engine.model is not None,
            "model_path": inference_engine.model_path, "is_gguf": inference_engine.is_gguf}

@router.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    response = inference_engine.generate(messages, max_tokens=max_tokens, temperature=temperature)
    return {"response": response}

@router.post("/run-suite")
async def run_test_suite(request: Request):
    body = await request.json()
    suite_path = body.get("suite_path", "")
    max_tokens = body.get("max_tokens", 512)
    cases = load_test_suite(suite_path)
    results = run_suite(inference_engine, cases, max_tokens=max_tokens)
    scores = score_results(results)
    return {"results": [{"name": r.test_name, "response": r.response, "passed": r.passed,
            "keyword_hits": r.keyword_hits, "keyword_misses": r.keyword_misses,
            "forbidden_hits": r.forbidden_hits, "time_ms": r.time_ms, "error": r.error}
            for r in results], "scores": scores}
