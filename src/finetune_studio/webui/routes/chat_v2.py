"""Chat v2 — project-scoped inference chat with multi-RAG support."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from finetune_studio.webui.engine_guard import ENGINE_LOCK

router = APIRouter()


def _is_portable_corpus(store_path: str) -> bool:
    """True when ``store_path`` is a PortableRAG directory (manifest + vectors)."""
    root = Path(store_path)
    return (root / "manifest.json").is_file() and (root / "vectors.npy").is_file()


def _search_rag_attachment(
    store_path: str,
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search a project_rags store — PortableRAG when present, else Chroma VectorStore."""
    if _is_portable_corpus(store_path):
        from finetune_studio.data.rag_portable import PortableRAG

        hits = PortableRAG(store_path).load().search(query, top_k=top_k)
        out: list[dict[str, Any]] = []
        for h in hits:
            out.append({
                "text": str(h.get("text") or ""),
                "score": float(h.get("score") or 0.0),
                "chunk_id": str(h.get("chunk_id") or ""),
            })
        return out

    from finetune_studio.rag.store import VectorStore

    store = VectorStore(store_path)
    results = store.search(query, top_k=top_k)
    return [
        {
            "text": r.text,
            "score": float(r.score),
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]


@router.get("/projects")
async def list_chat_projects():
    """List projects with RAG counts for the picker."""
    from finetune_studio import db
    projects = db.list_projects()
    result = []
    for p in projects:
        rags = db.list_rags(p["id"])
        result.append({
            "id": p["id"],
            "name": p["name"],
            "production_run_id": p.get("production_run"),
            "rag_count": len(rags),
        })
    return result


@router.get("/projects/{pid}/context")
async def project_context(pid: str):
    """Return project's production model path, system prompt, RAGs, and discovered models."""
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
    project = db.get_project(pid)
    if not project:
        return {"error": "Project not found"}

    rags = db.list_rags(pid)
    # Resolve production model path from the production run
    production_model_path = ""
    if project.get("production_run"):
        run = db.get_run(project["production_run"])
        if run and run.get("output_path"):
            production_model_path = run["output_path"]

    return {
        "production_model_path": production_model_path,
        "system_prompt": project.get("system_prompt", ""),
        "base_model": project.get("base_model", ""),
        "rags": [{"id": r["id"], "name": r["name"], "doc_count": r.get("doc_count", 0),
                  "chunk_count": r.get("chunk_count", 0)} for r in rags],
        "models": [{"name": m.name, "path": m.path, "format": m.format, "size_gb": m.size_gb,
                     "vision": getattr(m, "vision", False)}
                   for m in discovered_models],
    }


@router.post("/inference/chat")
async def inference_chat(request: Request):
    """Global inference chat — supports text and vision (images)."""
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    messages = body.get("messages", [])
    system_prompt = body.get("system_prompt", "")
    max_tokens = body.get("max_tokens", 1024)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)
    top_k = body.get("top_k", 40)
    repeat_penalty = body.get("repeat_penalty", 1.1)
    thinking = body.get("thinking", False)
    reasoning_effort = body.get("reasoning_effort", 5)
    if not messages:
        return {"error": "No messages"}
    if inference_engine.model is None:
        return {"error": "No model loaded. Load a model first."}
    # System prompt: prepend if provided and not already present
    if system_prompt:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
    # Thinking support: prepend /think instruction for Qwen3 models
    if thinking:
        think_instruction = "/think" if reasoning_effort <= 3 else f"/think\n/think_budget:{reasoning_effort * 100}"
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"] + "\n\n" + think_instruction
        else:
            messages.insert(0, {"role": "system", "content": think_instruction})
    try:
        async with ENGINE_LOCK:
            response = await asyncio.to_thread(
                inference_engine.generate,
                messages, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, repeat_penalty=repeat_penalty,
            )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    from finetune_studio.webui.thinking import split_thinking
    parts = split_thinking(response if isinstance(response, str) else str(response))
    vision_status = getattr(inference_engine, "vision", False)
    return {
        "response": parts["response"],
        "thinking": parts["thinking"],
        "vision": vision_status,
    }


@router.post("/load")
async def load_model(request: Request):
    """Load a model into the inference engine."""
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    # Accept path|model_path (QABUG-014-runtime — UI sometimes sends path only).
    model_path = body.get("model_path") or body.get("path") or ""
    if not model_path:
        return {"error": "No model_path"}
    try:
        async with ENGINE_LOCK:
            await asyncio.to_thread(
                inference_engine.load,
                model_path,
                n_ctx=body.get("n_ctx", 16384),
                n_gpu_layers=body.get("n_gpu_layers", -1),
                n_batch=body.get("n_batch", 512),
                mmap=body.get("mmap", True),
                mlock=body.get("mlock", False),
                n_threads=body.get("n_threads"),
                flash_attn=body.get("flash_attn", True),
                seed=body.get("seed"),
                rope_freq_base=body.get("rope_freq_base", 0.0),
                rope_freq_scale=body.get("rope_freq_scale", 0.0),
                max_seq_length=body.get("max_seq_length"),
                load_in_4bit=body.get("load_in_4bit", True),
            )
        vision = getattr(inference_engine, "vision", False)
        return {"status": "loaded", "model": model_path, "vision": vision}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.get("/status")
async def inference_status():
    """Current inference engine status."""
    from finetune_studio.testing.inference import IDLE_TIMEOUT
    from finetune_studio.webui.app import inference_engine
    loaded = inference_engine.model is not None
    return {
        "loaded": loaded,
        "model": inference_engine.model_path if loaded else None,
        "vision": getattr(inference_engine, "vision", False) if loaded else False,
        "is_gguf": inference_engine.is_gguf if loaded else False,
        "idle_seconds": inference_engine.idle_seconds,
        "idle_timeout": IDLE_TIMEOUT,
        "auto_unload_remaining": max(0, IDLE_TIMEOUT - inference_engine.idle_seconds) if loaded and IDLE_TIMEOUT > 0 else None,
    }


@router.post("/unload")
async def unload_model():
    """Manually unload the current model."""
    from finetune_studio.webui.app import inference_engine
    was = inference_engine.model_path
    inference_engine.unload()
    return {"status": "unloaded", "was": was}


@router.post("/model-info")
async def model_info(request: Request):
    """Read model metadata (layer count, etc.) for UI configuration."""
    from finetune_studio.testing.inference import InferenceEngine
    body = await request.json()
    model_path = body.get("model_path", "")
    if not model_path:
        return {"error": "No model_path"}
    try:
        info = InferenceEngine.read_model_metadata(model_path)
        return info
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.post("/memory-estimate")
async def memory_estimate(request: Request):
    """Estimate VRAM/RAM usage for a model with given settings."""
    from finetune_studio.testing.inference import InferenceEngine
    body = await request.json()
    model_path = body.get("model_path", "")
    n_ctx = body.get("n_ctx", 16384)
    n_gpu_layers = body.get("n_gpu_layers", -1)
    if not model_path:
        return {"error": "No model_path"}
    try:
        est = InferenceEngine.estimate_memory(model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        return est
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@router.post("/inference/benchmark")
async def inference_benchmark(request: Request):
    """Run quick real benchmarks on the loaded model.

    Returns nested per-suite results plus a scalar ``overall`` accuracy.
    Does not treat the nested ``benchmarks`` map as a list of scalars.
    """
    from finetune_studio.benchmarks.real_benchmarks import (
        RealBenchmarkSuite,
        overall_accuracy_from_run_all,
    )
    from finetune_studio.webui.app import inference_engine

    body = await request.json()
    if inference_engine.model is None:
        return {"error": "No model loaded. Load a model first."}

    num_samples = int(body.get("num_samples", 20))
    full_run = bool(body.get("full_run", False))
    benchmarks = body.get("benchmarks")
    if isinstance(benchmarks, str):
        bench_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    elif isinstance(benchmarks, list):
        bench_list = [str(b) for b in benchmarks]
    else:
        bench_list = ["mmlu", "gsm8k", "hellaswag"]

    def _blocking():
        suite = RealBenchmarkSuite()
        results = suite.run_all(
            inference_engine,
            num_samples=None if full_run else num_samples,
            benchmarks=bench_list,
            full_run=full_run,
        )
        return {"results": results, "overall": overall_accuracy_from_run_all(results)}

    try:
        async with ENGINE_LOCK:
            return await asyncio.to_thread(_blocking)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}



@router.post("/projects/{pid}/chat")
async def chat(request: Request, pid: str):
    """RAG-enhanced chat for a project.

    Query each enabled RAG, merge by score, take top-8 context chunks,
    prepend to system prompt, then generate.
    """
    from finetune_studio import db
    from finetune_studio.webui.app import inference_engine

    body = await request.json()
    messages = body.get("messages", [])
    enabled_rag_ids = body.get("enabled_rag_ids", [])
    system_prompt_override = body.get("system_prompt_override")
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)

    if not messages:
        return {"error": "No messages provided"}

    # Check model is loaded
    if inference_engine.model is None:
        return {"error": "No model loaded. Load a model first."}

    # Get project for system prompt default
    project = db.get_project(pid)
    if not project:
        return {"error": "Project not found"}

    # Extract last user message for RAG retrieval
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    # Multi-RAG retrieval
    all_sources = []
    if enabled_rag_ids and user_msg:
        for rag_id in enabled_rag_ids:
            rag = db.get_rag(rag_id)
            if not rag:
                continue
            try:
                results = _search_rag_attachment(
                    rag["store_path"], user_msg, top_k=5
                )
                for r in results:
                    all_sources.append({
                        "text": r["text"],
                        "score": round(float(r["score"]), 4),
                        "rag_id": rag_id,
                        "rag_name": rag["name"],
                        "chunk_id": r["chunk_id"],
                    })
            except Exception:  # noqa: BLE001,S112
                continue

    # Deduplicate by chunk_id, sort by score desc, take top 8
    seen = set()
    unique = []
    for s in sorted(all_sources, key=lambda x: x["score"], reverse=True):
        if s["chunk_id"] not in seen:
            seen.add(s["chunk_id"])
            unique.append(s)
    top_sources = unique[:8]

    # Build augmented system prompt
    base_prompt = system_prompt_override or project.get("system_prompt", "")
    if top_sources:
        context_block = "\n\n---\n\n".join(
            f"[Source: {s['rag_name']} (score {s['score']})]\n{s['text']}" for s in top_sources
        )
        rag_instruction = (
            "Use the following reference documents to help answer the question. "
            "Cite sources when relevant.\n\n"
            f"{context_block}"
        )
        if base_prompt:
            augmented_system = f"{base_prompt}\n\n{rag_instruction}"
        else:
            augmented_system = rag_instruction
    else:
        augmented_system = base_prompt

    # Build messages with augmented system prompt
    gen_messages = []
    if augmented_system:
        gen_messages.append({"role": "system", "content": augmented_system})
    # Add all non-system messages (user handles system prompt separately)
    for m in messages:
        if m.get("role") != "system":
            gen_messages.append(m)

    try:
        async with ENGINE_LOCK:
            response = await asyncio.to_thread(
                inference_engine.generate,
                gen_messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

    from finetune_studio.webui.thinking import split_thinking
    parts = split_thinking(response if isinstance(response, str) else str(response))
    return {
        "response": parts["response"],
        "thinking": parts["thinking"],
        "sources": [
            {"text": s["text"][:300], "score": s["score"],
             "rag_id": s["rag_id"], "rag_name": s["rag_name"]}
            for s in top_sources
        ],
        "chunks_retrieved": len(top_sources),
    }
