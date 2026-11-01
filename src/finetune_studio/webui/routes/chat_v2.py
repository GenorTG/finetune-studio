"""Chat v2 — project-scoped inference chat with multi-RAG support."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from finetune_studio.smart_brain.constants import DEFAULT_TOP_K
from finetune_studio.webui.engine_guard import ENGINE_LOCK

router = APIRouter()
log = logging.getLogger(__name__)


def _is_portable_corpus(store_path: str) -> bool:
    """True when ``store_path`` is a BrainIndex directory (manifest + vectors)."""
    root = Path(store_path)
    return (root / "manifest.json").is_file() and (root / "vectors.npy").is_file()


def _search_brain_index(
    store_path: str,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Search a project's Smart Brain index (a BrainIndex directory)."""
    if not _is_portable_corpus(store_path):
        raise FileNotFoundError(f"no Smart Brain index at {store_path}: build it on the project's Smart Brain page")
    from finetune_studio.smart_brain import BrainIndex

    hits = BrainIndex(store_path).load().search(query, top_k=top_k)
    return [{"text": str(h.get("text") or ""), "score": float(h.get("score") or 0.0),
             "chunk_id": str(h.get("chunk_id") or ""), "filename": str(h.get("filename") or h.get("source") or "")}
            for h in hits]


def loaded_model_trained_prompt(pid: str, model_path: str) -> str | None:
    """The system prompt the loaded model was trained with, when it is one of this project's finished runs.

    ``None`` = the loaded model is not a run output of this project (base model, another project's model): the
    project prompt applies. Testing asks a run's model under this prompt (``testing_jobs.trained_system_prompt``);
    Chat must ask the same model the same way, or the two pages disagree about one model.
    """
    from finetune_studio.testing.run_store import run_owning_model

    run = run_owning_model(pid, model_path)
    if run is None:
        return None
    if (run.get("system_prompt_mode") or "bake") == "none":
        return ""
    return str(run.get("system_prompt") or "")


@router.get("/projects")
async def list_chat_projects():
    """List projects with RAG counts for the picker."""
    from finetune_studio import db
    projects = db.list_projects()
    result = []
    for p in projects:
        indexes = db.list_brain_indexes(p["id"])
        result.append({
            "id": p["id"],
            "name": p["name"],
            "production_run_id": p.get("production_run"),
            "index_count": len(indexes),
        })
    return result


@router.get("/projects/{pid}/context")
async def project_context(pid: str):
    """Return project's production model path, system prompt, RAGs, and discovered models."""
    from finetune_studio import db
    from finetune_studio.webui.app import discovered_models
    project = db.get_project(pid)
    if not project:
        # Raise, don't return {"error": ...} with a 200: a missing project is a
        # 404 like every other project route. Returning 200 made the XHR
        # succeed and the caller had to sniff for an "error" key.
        raise HTTPException(status_code=404, detail="project not found")

    indexes = db.list_brain_indexes(pid)
    # Resolve production model path from the production run
    production_model_path = ""
    if project.get("production_run"):
        run = db.get_run(project["production_run"])
        if run and run.get("output_path"):
            production_model_path = run["output_path"]

    from finetune_studio.webui.testing_models import (
        merged_dir_ready,
        resolve_latest_merged_model,
    )

    tuned_model_path = ""
    if production_model_path and merged_dir_ready(production_model_path):
        tuned_model_path = os.path.join(production_model_path, "merged")
    else:
        tuned_model_path = resolve_latest_merged_model(pid) or ""

    return {
        "production_model_path": production_model_path,
        "tuned_model_path": tuned_model_path,
        "system_prompt": project.get("system_prompt", ""),
        "base_model": project.get("base_model", ""),
        "indexes": [{"id": r["id"], "name": r["name"], "doc_count": r.get("doc_count", 0),
                  "chunk_count": r.get("chunk_count", 0)} for r in indexes],
        "models": [{"name": m.name, "path": m.path, "format": m.format, "size_gb": m.size_gb,
                     "vision": getattr(m, "vision", False)}
                   for m in discovered_models],
    }


@router.post("/load")
async def load_model(request: Request):
    """Load a model into the inference engine."""
    from finetune_studio.webui.app import inference_engine
    body = await request.json()
    # Accept path|model_path (QABUG-014-runtime — UI sometimes sends path only).
    model_path = body.get("model_path") or body.get("path") or ""
    if not model_path:
        return JSONResponse({"detail": "No model_path"}, status_code=400)
    try:
        from finetune_studio.models.llama_loader import resolve_loader_overrides
        kwargs = resolve_loader_overrides(body, caller="chat-v2/load", model_path=model_path)
        kwargs["max_seq_length"] = body.get("max_seq_length")
        kwargs["load_in_4bit"] = body.get("load_in_4bit", True)
        # Free whatever ModelManager (data-prep's helper, /api/providers/*)
        # has resident before loading into inference_engine — otherwise both
        # sit in VRAM simultaneously until someone happens to click Unload.
        from finetune_studio.models.manager import get_manager
        async with ENGINE_LOCK:
            await asyncio.to_thread(get_manager().unload)
            await asyncio.to_thread(inference_engine.load, model_path, **kwargs)
        vision = getattr(inference_engine, "vision", False)
        return {"status": "loaded", "model": model_path, "vision": vision}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": str(e)}, status_code=500)


@router.post("/projects/{pid}/chat")
async def chat(request: Request, pid: str):
    """RAG-enhanced chat for a project.

    Query each enabled RAG, merge by score, take top-8 context chunks,
    prepend to system prompt, then generate.
    """
    from finetune_studio import db
    from finetune_studio.webui.app import inference_engine

    # The project is validated FIRST: it is the path resource, and every other
    # project route 404s on it before touching anything else. Checking the
    # model first made a missing project report "No model loaded" (200) and
    # send the caller to fix a model that was never the problem.
    project = db.get_project(pid)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")

    body = await request.json()
    messages = body.get("messages", [])
    enabled_index_ids = body.get("enabled_index_ids", [])
    system_prompt_override = body.get("system_prompt_override")
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.3)   # the Testing page's default: one model, one answer distribution
    top_p = body.get("top_p", 0.9)
    top_k = body.get("top_k", 40)   # the Top-k control; dropped here before 2026-11-01, so the worker always used 40

    if not messages:
        return JSONResponse({"detail": "No messages provided"}, status_code=400)

    # Check model is loaded
    if inference_engine.model is None:
        return JSONResponse({"detail": "No model loaded. Load a model first."}, status_code=409)

    # `project` (validated at the top) supplies the system prompt default.
    # Extract last user message for RAG retrieval
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    # Multi-RAG retrieval
    all_sources = []
    if enabled_index_ids and user_msg:
        for index_id in enabled_index_ids:
            brain = db.get_brain_index(index_id)
            # A RAG owned by another project is treated exactly like a
            # missing one: the chat path must not read foreign corpora.
            if not brain or brain.get("project_id") != pid:
                continue
            try:
                results = _search_brain_index(
                    brain["store_path"], user_msg, top_k=DEFAULT_TOP_K
                )
                for r in results:
                    all_sources.append({
                        "text": r["text"],
                        "score": round(float(r["score"]), 4),
                        "index_id": index_id,
                        "index_name": brain["name"],
                        "chunk_id": r["chunk_id"],
                        "filename": r.get("filename", ""),
                    })
            except Exception as e:  # any search failure is reported, never answered around
                # The user asked for answers with sources: answering without them would look grounded and is not.
                log.warning("Smart Brain search failed for index %s", index_id, exc_info=True)
                return JSONResponse({"detail": f"Smart Brain search failed ({brain['name']}): {e}. Rebuild it on "
                                              "the Smart Brain page, or untick it to chat without sources."},
                                    status_code=502)

    # Deduplicate by chunk_id, sort by score desc, take top 8
    seen = set()
    unique = []
    for s in sorted(all_sources, key=lambda x: x["score"], reverse=True):
        if s["chunk_id"] not in seen:
            seen.add(s["chunk_id"])
            unique.append(s)
    top_sources = unique[:8]

    # System prompt: an edited box wins; left at the project prompt (or empty), a model trained in this project is
    # asked under the prompt it was trained with, exactly as the Testing page asks it.
    project_prompt = project.get("system_prompt", "") or ""
    trained_prompt = loaded_model_trained_prompt(pid, str(inference_engine.model_path or ""))
    override = (system_prompt_override or "").strip()
    if override and override != project_prompt.strip():
        base_prompt, prompt_source = override, "edited"
    elif trained_prompt is not None:
        base_prompt, prompt_source = trained_prompt, "trained_run"
    else:
        base_prompt, prompt_source = project_prompt, "project"

    history = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"]
    if top_sources:
        # The grounded system turn is smart_brain.prompt's: the same layout and "answer only from the context, say so
        # when it is not there" instructions as the Smart Brain chat and the context-grounded training rows.
        from finetune_studio.smart_brain import prompt as brain_prompt

        hits = [{"rank": i, "text": s["text"], "filename": s.get("filename") or s["index_name"], "rrf_score": s["score"]}
                for i, s in enumerate(top_sources, start=1)]
        history_chars = sum(len(str(m.get("content", ""))) for m in history) + len(base_prompt)
        budget = brain_prompt.context_char_budget(getattr(inference_engine, "n_ctx", None), max_new_tokens=int(max_tokens),
                                                  history_chars=history_chars)
        grounded = brain_prompt.build_system_prompt(brain_prompt.format_context_blocks(hits, max_chars=budget))
        augmented_system = f"{base_prompt}\n\n{grounded}" if base_prompt else grounded
    else:
        augmented_system = base_prompt

    gen_messages = ([{"role": "system", "content": augmented_system}] if augmented_system else []) + history
    request_echo = {"system_prompt_source": prompt_source, "system_prompt": base_prompt, "grounded": bool(top_sources),
                    "temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens,
                    "turns": len(history)}

    try:
        async with ENGINE_LOCK:
            response = await asyncio.to_thread(
                inference_engine.generate,
                gen_messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            )
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": str(e)}, status_code=500)

    from finetune_studio.webui.thinking import split_thinking
    parts = split_thinking(response if isinstance(response, str) else str(response))
    return {
        "response": parts["response"],
        "thinking": parts["thinking"],
        "sources": [
            {"text": s["text"][:300], "score": s["score"],
             "index_id": s["index_id"], "index_name": s["index_name"]}
            for s in top_sources
        ],
        "chunks_retrieved": len(top_sources),
        "request": request_echo,
    }
