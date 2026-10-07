"""Guide API: streaming chat (SSE) plus read-only KB endpoints.

``POST /api/guide/chat`` runs the guide loop (``finetune_studio.guide.loop``) and streams one SSE
frame per event — ``start``, ``thinking``, ``tool_call``, ``tool_result``, ``ui``, ``final`` /
``error``, ``done`` — so the browser shows each tool call the moment it happens and can act on
``ui`` events (navigate / highlight / prefill) immediately. The helper-model rule is the legacy
route's: no silent switch, a clear 409 when no helper is available.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from finetune_studio import db
from finetune_studio.guide.kb import load_kb, table_of_contents
from finetune_studio.guide.loop import run_guide_loop
from finetune_studio.guide.search import search_kb
from finetune_studio.guide.tools import ToolContext
from finetune_studio.webui.live_sse import sse_data, sse_response
from finetune_studio.webui.routes.data_prep_chat import (
    MAX_TOOL_ROUNDS,
    make_chat_fn,
    parse_gen,
    resolve_chat_backend,
)

log = logging.getLogger(__name__)
router = APIRouter()

MAX_HISTORY = 24


@router.post("/guide/chat")
async def guide_chat(request: Request):
    """Stream one guide turn. Body: ``messages``, optional ``project_id``, ``page_path``,
    ``provider_id``, ``external_api``, ``max_rounds`` and generation kwargs."""
    body = await request.json()
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object body required"}, status_code=400)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "messages required"}, status_code=400)
    clean = [
        {"role": m["role"], "content": str(m.get("content") or "")}
        for m in messages
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ][-MAX_HISTORY:]
    if not clean or clean[-1]["role"] != "user":
        return JSONResponse({"error": "the last message must be a user message"}, status_code=400)
    external = body.get("external_api")
    if external is not None and not isinstance(external, dict):
        return JSONResponse({"error": "external_api must be an object"}, status_code=400)
    try:
        max_rounds = max(1, min(int(body.get("max_rounds") or MAX_TOOL_ROUNDS), 12))
    except (TypeError, ValueError):
        return JSONResponse({"error": "max_rounds must be an integer"}, status_code=400)

    ctx = ToolContext.from_path(str(body.get("page_path") or ""), pid=str(body.get("project_id") or "") or None)
    project = db.get_project(ctx.pid) if ctx.pid else None
    if ctx.pid and not project:
        return JSONResponse({"error": f"project {ctx.pid} not found"}, status_code=404)

    gen = parse_gen(body)
    provider_id = body.get("provider_id")
    name = (project or {}).get("name")
    # A named provider may have to LOAD the helper (~50 s): resolve it inside the stream so the
    # user sees "loading helper" at once. Cheap cases (external API, helper already resident)
    # resolve up front so failures keep a real HTTP status.
    deferred = bool(provider_id) and not external
    backend: dict | None = None
    if not deferred:
        backend, error = await resolve_chat_backend(provider_id, external)
        if error is not None or backend is None:
            return error

    async def frames() -> AsyncIterator[str]:
        nonlocal backend
        yield sse_data({"type": "start", "page": ctx.page, "project_id": ctx.pid,
                        "backend": "external" if external else "provider"})
        if backend is None:
            yield sse_data({"type": "status", "message": "Loading the helper model (about a minute the first time)…"})
            backend, error = await resolve_chat_backend(provider_id, external)
            if error is not None or backend is None:
                detail = (json.loads(bytes(error.body)).get("error") if error is not None else None) or "no backend"
                yield sse_data({"type": "error", "status": getattr(error, "status_code", 409), "error": detail})
                yield sse_data({"type": "done"})
                return
        events = run_guide_loop(ctx, clean, make_chat_fn(backend, gen), max_rounds=max_rounds,
                                project_name=name, max_tokens=gen.get("max_tokens", 4096))
        end = object()
        try:
            while (event := await asyncio.to_thread(next, events, end)) is not end:
                yield sse_data(event)
        except Exception as e:
            log.exception("guide stream failed")
            yield sse_data({"type": "error", "status": 500, "error": f"guide failed: {e}"})
        yield sse_data({"type": "done"})

    return sse_response(frames())


@router.get("/guide/kb")
async def guide_kb():
    """Table of contents of the app knowledge base."""
    return {"entries": table_of_contents()}


@router.get("/guide/kb/{entry_id}")
async def guide_kb_entry(entry_id: str):
    entry = load_kb().get(entry_id)
    if entry is None:
        return JSONResponse({"error": f"unknown KB entry {entry_id!r}"}, status_code=404)
    return {"id": entry.id, "title": entry.title, "page": entry.route, "sections": entry.sections}


@router.get("/guide/help")
async def guide_help(q: str, limit: int = 3):
    """Same retrieval as the ``app_help`` tool (debugging / docs links)."""
    if not q.strip():
        return JSONResponse({"error": "q required"}, status_code=400)
    return {"query": q, "results": await asyncio.to_thread(search_kb, q, limit)}
