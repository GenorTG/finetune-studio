"""Comparison tab (side-by-side model output) and RAG chat routes."""
import asyncio

from fastapi import APIRouter, Request

from finetune_studio.webui.engine_guard import ENGINE_LOCK

router = APIRouter()


@router.post("/compare/load")
async def compare_load(request: Request):
    """Load a model for comparison."""
    body = await request.json()
    name = body.get("name", "model")
    path = body.get("path", "")
    if not path:
        return {"error": "No path provided"}
    try:
        from finetune_studio.benchmarks.comparison import comparator
        await asyncio.to_thread(comparator.load_model, name, path)
        return {"status": "loaded", "name": name, "models": comparator.model_names()}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.post("/compare/run")
async def compare_run(request: Request):
    """Run comparison on test suite."""
    body = await request.json()
    test_suite = body.get("test_suite", [])
    config = body.get("config", {"max_tokens": 512, "temperature": 0.7})

    if not test_suite:
        return {"error": "No test suite provided"}

    from finetune_studio.benchmarks.comparison import NoModelsLoadedError, comparator
    try:
        result = await asyncio.to_thread(comparator.run_comparison, test_suite, config)
    except NoModelsLoadedError as e:
        return {"error": str(e)}
    return result


@router.post("/compare/cleanup")
async def compare_cleanup():
    """Unload all comparison models."""
    from finetune_studio.benchmarks.comparison import comparator
    await asyncio.to_thread(comparator.cleanup)
    return {"status": "cleaned", "models": []}


@router.post("/rag/chat")
async def rag_chat(request: Request):
    """RAG-enhanced chat endpoint."""
    body = await request.json()
    messages = body.get("messages", [])
    top_k = body.get("top_k", 5)
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)

    if not messages:
        return {"error": "No messages provided"}

    # Get the last user message for RAG retrieval
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    if not user_msg or not isinstance(user_msg, str):
        return {"error": "No user message found"}

    from finetune_studio.webui.app import inference_engine
    if not inference_engine or inference_engine.model is None:
        return {"error": "No model loaded"}

    # Retrieve context from RAG
    from finetune_studio.config import settings
    from finetune_studio.rag.store import VectorStore

    rag_store = VectorStore(settings.rag_store_path)
    results = await asyncio.to_thread(
        rag_store.search, user_msg, top_k=top_k,
        embedding_model=settings.rag.embedding_model,
    )

    # Build context
    context_parts = []
    for r in results:
        if r.score >= settings.rag.min_score:
            context_parts.append(f"[Source: {r.source}]\n{r.text}")

    context = "\n\n---\n\n".join(context_parts)

    # Augment messages with context
    augmented_messages = []
    if context:
        augmented_messages.append({
            "role": "system",
            "content": f"Based on the following documents:\n\n{context}\n\nAnswer the question using this information when relevant.",
        })

    augmented_messages.extend(messages)

    # Generate response
    async with ENGINE_LOCK:
        result = await asyncio.to_thread(
            inference_engine.generate,
            augmented_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return {
        "response": result if isinstance(result, str) else result.get("response", str(result)),
        "sources": [
            {"text": r.text[:200], "score": round(r.score, 3), "source": r.source}
            for r in results if r.score >= settings.rag.min_score
        ],
        "chunks_retrieved": len([r for r in results if r.score >= settings.rag.min_score]),
    }
